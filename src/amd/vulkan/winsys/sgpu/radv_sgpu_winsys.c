/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_sgpu_winsys.h"
#include "radv_sgpu_winsys_public.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drm-uapi/amdgpu_drm.h"
#include "util/macros.h"
#include "util/u_sync_provider.h"
#include "vk_drm_syncobj.h"

static uint64_t
radv_sgpu_query_value(struct radeon_winsys *_ws, enum radeon_value_id value)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   uint64_t result = 0;

   switch (value) {
   case RADEON_ALLOCATED_GTT:
      simple_mtx_lock(&ws->lock);
      result = ws->allocated_gtt;
      simple_mtx_unlock(&ws->lock);
      break;
   case RADEON_TIMESTAMP:
      amdgpu_query_info(ws->device, AMDGPU_INFO_TIMESTAMP, sizeof(result), &result);
      break;
   default:
      break;
   }
   return result;
}

static bool
radv_sgpu_read_registers(struct radeon_winsys *ws, unsigned reg_offset, unsigned num_registers, uint32_t *out)
{
   return false;
}

static bool
radv_sgpu_query_gpuvm_fault(struct radeon_winsys *_ws, struct radv_winsys_gpuvm_fault_info *fault_info)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct drm_amdgpu_info_gpuvm_fault fault = {0};

   if (amdgpu_query_info(ws->device, AMDGPU_INFO_GPUVM_FAULT, sizeof(fault), &fault) || !fault.status)
      return false;

   fault_info->addr = fault.addr;
   fault_info->status = fault.status;
   fault_info->vmhub = fault.vmhub;
   return true;
}

static void
radv_sgpu_dump_bo_ranges(struct radeon_winsys *ws, FILE *file)
{
}

static void
radv_sgpu_dump_bo_log(struct radeon_winsys *ws, FILE *file)
{
}

static int
radv_sgpu_get_fd(struct radeon_winsys *ws)
{
   return radv_sgpu_winsys(ws)->fd;
}

static struct util_sync_provider *
radv_sgpu_get_sync_provider(struct radeon_winsys *_ws)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   return ws->sync_provider->clone(ws->sync_provider);
}

static int
radv_sgpu_reserve_vmid(struct radeon_winsys *ws)
{
   return -1;
}

static void
radv_sgpu_unreserve_vmid(struct radeon_winsys *ws)
{
}

/* No ordinary BO may carry the PRT workaround bit: scalar loads and UBO descriptors clear it
 * (has_smem_with_null_prt_bug), so the whole range that has it stays reserved. libdrm gives every
 * winsys of a GPU the same device handle and VA manager, so the reservation is shared by all the
 * winsyses (logical devices) of the process.
 */
struct radv_sgpu_prt_wa_range {
   struct radv_sgpu_prt_wa_range *next;
   amdgpu_device_handle device;
   amdgpu_va_handle va;
   unsigned refcount;
};

static simple_mtx_t prt_wa_ranges_lock = SIMPLE_MTX_INITIALIZER;
static struct radv_sgpu_prt_wa_range *prt_wa_ranges;

static bool
radv_sgpu_reserve_prt_wa_range(struct radv_sgpu_winsys *ws, unsigned control_bit)
{
   struct radv_sgpu_prt_wa_range *range;

   simple_mtx_lock(&prt_wa_ranges_lock);

   for (range = prt_wa_ranges; range; range = range->next) {
      if (range->device == ws->device)
         break;
   }

   if (range) {
      range->refcount++;
   } else {
      range = calloc(1, sizeof(*range));
      if (range) {
         const uint64_t base = 1ull << control_bit;
         uint64_t va = 0;

         if (amdgpu_va_range_alloc(ws->device, amdgpu_gpu_va_range_general, base, 65536, base, &va, &range->va,
                                    0) ||
             va != base) {
            if (range->va)
               amdgpu_va_range_free(range->va);
            free(range);
            range = NULL;
         } else {
            range->device = ws->device;
            range->refcount = 1;
            range->next = prt_wa_ranges;
            prt_wa_ranges = range;
         }
      }
   }

   simple_mtx_unlock(&prt_wa_ranges_lock);

   ws->prt_wa_range = range;
   return range != NULL;
}

static void
radv_sgpu_release_prt_wa_range(struct radv_sgpu_winsys *ws)
{
   struct radv_sgpu_prt_wa_range *range = ws->prt_wa_range;

   simple_mtx_lock(&prt_wa_ranges_lock);

   if (--range->refcount == 0) {
      struct radv_sgpu_prt_wa_range **link = &prt_wa_ranges;
      while (*link != range)
         link = &(*link)->next;
      *link = range->next;

      amdgpu_va_range_free(range->va);
      free(range);
   }

   simple_mtx_unlock(&prt_wa_ranges_lock);
}

static void
radv_sgpu_winsys_destroy(struct radeon_winsys *_ws)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);

   radv_sgpu_ib_pool_finish(ws);
   if (ws->null_prt_bo)
      ws->base.buffer_destroy(&ws->base, ws->null_prt_bo);
   radv_sgpu_release_prt_wa_range(ws);
   simple_mtx_destroy(&ws->null_prt_lock);
   ws->sync_provider->finalize(ws->sync_provider);
   amdgpu_device_deinitialize(ws->device);
   close(ws->fd);
   simple_mtx_destroy(&ws->lock);
   free(ws);
}

VkResult
radv_sgpu_winsys_create(int fd, const struct radeon_info *info, struct radeon_winsys **out)
{
   uint32_t drm_major, drm_minor;

   struct radv_sgpu_winsys *ws = calloc(1, sizeof(*ws));
   if (!ws)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   ws->info = *info;
   ws->fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
   if (ws->fd < 0)
      goto fail;

   if (amdgpu_device_initialize(ws->fd, &drm_major, &drm_minor, &ws->device) || !ws->device) {
      fprintf(stderr, "radv/sgpu: device initialization failed.\n");
      goto fail;
   }

   ws->sync_provider = util_sync_provider_drm(ws->fd);
   if (!ws->sync_provider) {
      amdgpu_device_deinitialize(ws->device);
      goto fail;
   }

   if (!radv_sgpu_reserve_prt_wa_range(ws, info->address_prt_wa_control_bit)) {
      fprintf(stderr, "radv/sgpu: could not reserve the sparse alias VA range.\n");
      ws->sync_provider->finalize(ws->sync_provider);
      amdgpu_device_deinitialize(ws->device);
      goto fail;
   }

   simple_mtx_init(&ws->lock, mtx_plain);
   simple_mtx_init(&ws->null_prt_lock, mtx_plain);
   list_inithead(&ws->resident_bos);

   ws->base.destroy = radv_sgpu_winsys_destroy;
   ws->base.query_value = radv_sgpu_query_value;
   ws->base.read_registers = radv_sgpu_read_registers;
   ws->base.query_gpuvm_fault = radv_sgpu_query_gpuvm_fault;
   ws->base.dump_bo_ranges = radv_sgpu_dump_bo_ranges;
   ws->base.dump_bo_log = radv_sgpu_dump_bo_log;
   ws->base.get_fd = radv_sgpu_get_fd;
   ws->base.get_sync_provider = radv_sgpu_get_sync_provider;
   ws->base.copy_sync_payloads = vk_drm_syncobj_copy_payloads;
   ws->base.reserve_vmid = radv_sgpu_reserve_vmid;
   ws->base.unreserve_vmid = radv_sgpu_unreserve_vmid;
   radv_sgpu_bo_init_functions(ws);
   radv_sgpu_cs_init_functions(ws);

   *out = &ws->base;
   return VK_SUCCESS;

fail:
   if (ws->fd >= 0)
      close(ws->fd);
   free(ws);
   return VK_ERROR_INITIALIZATION_FAILED;
}
