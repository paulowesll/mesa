/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_sgpu_winsys.h"
#include "radv_sgpu_winsys_public.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drm-uapi/amdgpu_drm.h"
#include "util/macros.h"
#include "util/u_sync_provider.h"
#include "vk_drm_syncobj.h"

void *
radv_sgpu_open_vendor_library(void)
{
   char sibling[4096] = "";
   Dl_info dl;

   if (dladdr((void *)radv_sgpu_open_vendor_library, &dl) && dl.dli_fname) {
      const char *slash = strrchr(dl.dli_fname, '/');
      if (slash && (size_t)(slash - dl.dli_fname) + sizeof("/libdrm_sgpu.so") <= sizeof(sibling)) {
         memcpy(sibling, dl.dli_fname, slash - dl.dli_fname);
         strcpy(sibling + (slash - dl.dli_fname), "/libdrm_sgpu.so");
      }
   }

   const char *paths[] = {
      getenv("RADV_SGPU_VENDOR_LIB"),
      sibling,
      "/vendor/lib64/libdrm_sgpu.so",
      "libdrm_sgpu.so",
   };

   for (unsigned i = 0; i < ARRAY_SIZE(paths); i++) {
      if (!paths[i] || !paths[i][0])
         continue;
      void *lib = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
      if (lib)
         return lib;
   }
   return NULL;
}

static bool
radv_sgpu_load_api(void *lib, struct radv_sgpu_api *api)
{
#define LOAD(field, symbol)                                                                        \
   if (!(*(void **)&api->field = dlsym(lib, symbol)))                                              \
      return false;

   LOAD(device_initialize, "amdgpu_device_initialize");
   LOAD(device_deinitialize, "amdgpu_device_deinitialize");
   LOAD(bo_alloc, "amdgpu_bo_alloc");
   LOAD(bo_import, "amdgpu_bo_import");
   LOAD(bo_free, "amdgpu_bo_free");
   LOAD(bo_export, "amdgpu_bo_export");
   LOAD(bo_cpu_map, "amdgpu_bo_cpu_map");
   LOAD(bo_cpu_unmap, "amdgpu_bo_cpu_unmap");
   LOAD(bo_wait_for_idle, "amdgpu_bo_wait_for_idle");
   LOAD(bo_query_info, "amdgpu_bo_query_info");
   LOAD(bo_set_metadata, "amdgpu_bo_set_metadata");
   LOAD(va_range_alloc, "amdgpu_va_range_alloc");
   LOAD(va_range_free, "amdgpu_va_range_free");
   LOAD(bo_va_op, "amdgpu_bo_va_op");
   LOAD(bo_va_op_raw, "amdgpu_bo_va_op_raw");
   LOAD(ctx_create2, "amdgpu_cs_ctx_create2");
   LOAD(ctx_free, "amdgpu_cs_ctx_free");
   LOAD(cs_submit_raw, "amdgpu_cs_submit_raw");
   LOAD(cs_query_fence_status, "amdgpu_cs_query_fence_status");
   LOAD(syncobj_create, "amdgpu_cs_create_syncobj");
   LOAD(syncobj_destroy, "amdgpu_cs_destroy_syncobj");
   LOAD(syncobj_signal, "amdgpu_cs_syncobj_signal");
#undef LOAD

   *(void **)&api->cs_query_reset_state2 = dlsym(lib, "amdgpu_cs_query_reset_state2");
   *(void **)&api->query_info = dlsym(lib, "amdgpu_query_info");
   return true;
}

static uint64_t
radv_sgpu_query_value(struct radeon_winsys *_ws, enum radeon_value_id value)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   uint64_t result = 0;

   if (value == RADEON_ALLOCATED_GTT) {
      simple_mtx_lock(&ws->lock);
      result = ws->allocated_gtt;
      simple_mtx_unlock(&ws->lock);
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

   if (!ws->api.query_info ||
       ws->api.query_info(ws->device, AMDGPU_INFO_GPUVM_FAULT, sizeof(fault), &fault) || !fault.status)
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

         if (ws->api.va_range_alloc(ws->device, amdgpu_gpu_va_range_general, base, 65536, base, &va, &range->va,
                                    0) ||
             va != base) {
            if (range->va)
               ws->api.va_range_free(range->va);
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

      ws->api.va_range_free(range->va);
      free(range);
   }

   simple_mtx_unlock(&prt_wa_ranges_lock);
}

static void
radv_sgpu_winsys_destroy(struct radeon_winsys *_ws)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);

   if (ws->null_prt_bo)
      ws->base.buffer_destroy(&ws->base, ws->null_prt_bo);
   radv_sgpu_release_prt_wa_range(ws);
   simple_mtx_destroy(&ws->null_prt_lock);
   ws->sync_provider->finalize(ws->sync_provider);
   ws->api.device_deinitialize(ws->device);
   dlclose(ws->vendor_lib);
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

   ws->vendor_lib = radv_sgpu_open_vendor_library();
   if (!ws->vendor_lib || !radv_sgpu_load_api(ws->vendor_lib, &ws->api)) {
      fprintf(stderr, "radv/sgpu: could not load libdrm_sgpu.so.\n");
      goto fail;
   }

   if (ws->api.device_initialize(ws->fd, &drm_major, &drm_minor, &ws->device) || !ws->device) {
      fprintf(stderr, "radv/sgpu: vendor device initialization failed.\n");
      goto fail;
   }

   ws->sync_provider = util_sync_provider_drm(ws->fd);
   if (!ws->sync_provider) {
      ws->api.device_deinitialize(ws->device);
      goto fail;
   }

   if (!radv_sgpu_reserve_prt_wa_range(ws, info->address_prt_wa_control_bit)) {
      fprintf(stderr, "radv/sgpu: could not reserve the sparse alias VA range.\n");
      ws->sync_provider->finalize(ws->sync_provider);
      ws->api.device_deinitialize(ws->device);
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
   if (ws->vendor_lib)
      dlclose(ws->vendor_lib);
   if (ws->fd >= 0)
      close(ws->fd);
   free(ws);
   return VK_ERROR_INITIALIZATION_FAILED;
}
