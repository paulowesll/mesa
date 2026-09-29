/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_sgpu_winsys.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "drm-uapi/amdgpu_drm.h"
#include "util/u_math.h"

/* Start of the VA window of capture/replay BOs.  The vendor libdrm ignores
 * AMDGPU_VA_RANGE_REPLAYABLE, so without a separate window a replay's own allocations land on
 * the captured addresses.
 */
#define SGPU_REPLAY_VA_WINDOW 0x4000000000ull

/* Like the vendor driver (GEM flags 0x45) and the amdgpu winsys, non-shareable BOs are
 * per-VM "always valid": the kernel does not have to validate and reserve every resident BO
 * on each submission, whose cost otherwise grows ~1 us per allocation.
 */
static bool
radv_sgpu_use_local_bo(enum radeon_bo_flag flags)
{
   return (flags & RADEON_FLAG_NO_INTERPROCESS_SHARING) && !(flags & RADEON_FLAG_VIRTUAL);
}

static uint64_t
radv_sgpu_create_flags(enum radeon_bo_flag flags)
{
   uint64_t result = 0;

   if (flags & RADEON_FLAG_CPU_ACCESS)
      result |= AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;
   if (flags & RADEON_FLAG_NO_CPU_ACCESS)
      result |= AMDGPU_GEM_CREATE_NO_CPU_ACCESS;
   if (radv_sgpu_use_local_bo(flags))
      result |= AMDGPU_GEM_CREATE_VM_ALWAYS_VALID;
   /* RADEON_FLAG_GTT_WC is deliberately not turned into CPU_GTT_USWC: the command processor
    * and scalar caches need the ordinary cached GTT page attributes on this GPU.
    */
   return result;
}

#define SGPU_NULL_PRT_BO_SIZE (8u * 1024 * 1024)

static VkResult radv_sgpu_virtual_bo_create(struct radv_sgpu_winsys *ws, uint64_t size, unsigned alignment,
                                            enum radeon_bo_flag flags, uint64_t address,
                                            struct radeon_winsys_bo **out_bo);
static VkResult radv_sgpu_buffer_make_resident(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo,
                                               bool resident);

static VkResult
radv_sgpu_buffer_create(struct radeon_winsys *_ws, uint64_t size, unsigned alignment, enum radeon_bo_domain domain,
                        enum radeon_bo_flag flags, unsigned priority, uint64_t address,
                        struct radeon_winsys_bo **out_bo)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct amdgpu_bo_alloc_request request = {0};
   uint64_t va = 0, va_flags = 0;
   int ret;

   *out_bo = NULL;

   if (flags & RADEON_FLAG_VIRTUAL)
      return radv_sgpu_virtual_bo_create(ws, size, alignment, flags, address, out_bo);

   alignment = MAX2(alignment, 4096u);
   const uint64_t alloc_size = align64(MAX2(size, 1), alignment);

   struct radv_sgpu_bo *bo = calloc(1, sizeof(*bo));
   if (!bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* There is no VRAM: everything lives in the unified GTT heap. */
   request.alloc_size = alloc_size;
   request.phys_alignment = alignment;
   request.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;
   request.flags = radv_sgpu_create_flags(flags);

   simple_mtx_lock(&ws->lock);

   ret = ws->api.bo_alloc(ws->device, &request, &bo->bo);
   if (ret) {
      fprintf(stderr, "radv/sgpu: amdgpu_bo_alloc failed (%d), size %" PRIu64 ".\n", ret, alloc_size);
      goto fail;
   }

   /* Low 32-bit window: the device query reports address32_hi = 0. */
   if (flags & RADEON_FLAG_32BIT)
      va_flags |= AMDGPU_VA_RANGE_32_BIT;

   if ((flags & RADEON_FLAG_REPLAYABLE) && !address && !(flags & RADEON_FLAG_32BIT)) {
      const uint64_t base = align64(MAX2(ws->replay_va_cursor, SGPU_REPLAY_VA_WINDOW), alignment);
      if (!ws->api.va_range_alloc(ws->device, amdgpu_gpu_va_range_general, alloc_size, alignment, base, &va, &bo->va,
                                  va_flags) &&
          va == base) {
         ws->replay_va_cursor = base + alloc_size;
      } else {
         if (bo->va)
            ws->api.va_range_free(bo->va);
         bo->va = NULL;
      }
   }

   if (!bo->va) {
      ret = ws->api.va_range_alloc(ws->device, amdgpu_gpu_va_range_general, alloc_size, alignment, address, &va,
                                   &bo->va, va_flags);
      if (ret) {
         fprintf(stderr, "radv/sgpu: amdgpu_va_range_alloc failed (%d), size %" PRIu64 ".\n", ret, alloc_size);
         ws->api.bo_free(bo->bo);
         goto fail;
      }
   }

   /* Command buffers are labelled read-only by the common code, but the vendor driver maps its
    * IBs writable and so do we.
    */
   bo->vm_flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
   if (flags & RADEON_FLAG_GL2_BYPASS)
      bo->vm_flags |= AMDGPU_VM_MTYPE_UC;

   ret = ws->api.bo_va_op(bo->bo, 0, alloc_size, va, bo->vm_flags, AMDGPU_VA_OP_MAP);
   if (ret) {
      fprintf(stderr, "radv/sgpu: amdgpu_bo_va_op(MAP) failed (%d), va 0x%" PRIx64 ".\n", ret, va);
      ws->api.va_range_free(bo->va);
      ws->api.bo_free(bo->bo);
      goto fail;
   }

   bo->ws = ws;
   bo->alloc_size = alloc_size;
   bo->priority = MIN2(priority, UINT8_MAX);
   bo->is_local = radv_sgpu_use_local_bo(flags);
   bo->va_mapped = true;
   list_inithead(&bo->resident_link);

   bo->base.va = va;
   bo->base.size = size;
   bo->base.initial_domain = RADEON_DOMAIN_GTT;
   bo->base.vram_no_cpu_access = !(flags & RADEON_FLAG_CPU_ACCESS);
   bo->base.gfx12_allow_dcc = !!(flags & RADEON_FLAG_GFX12_ALLOW_DCC);
   bo->base.obj_id = va;

   ws->allocated_gtt += alloc_size;
   simple_mtx_unlock(&ws->lock);

   *out_bo = &bo->base;
   return VK_SUCCESS;

fail:
   simple_mtx_unlock(&ws->lock);
   free(bo);
   return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

static void
radv_sgpu_bo_remove_resident_locked(struct radv_sgpu_bo *bo)
{
   if (!bo->base.use_global_list)
      return;

   list_delinit(&bo->resident_link);
   bo->base.use_global_list = false;
}

static void
radv_sgpu_bo_release_locked(struct radv_sgpu_winsys *ws, struct radv_sgpu_bo *bo)
{
   if (bo->base.is_virtual) {
      const uint64_t low_va = bo->base.va & ~(1ull << ws->info.address_prt_wa_control_bit);

      ws->api.bo_va_op_raw(ws->device, NULL, 0, bo->alloc_size, bo->base.va, 0, AMDGPU_VA_OP_CLEAR);
      if (bo->emulate_sparse_residency)
         ws->api.bo_va_op_raw(ws->device, NULL, 0, bo->alloc_size, low_va, 0, AMDGPU_VA_OP_CLEAR);
      ws->api.va_range_free(bo->va);
      free(bo);
      return;
   }

   if (bo->fixed_mapped)
      munmap(bo->fixed_map, bo->base.size);
   if (bo->cpu_mapped)
      ws->api.bo_cpu_unmap(bo->bo);
   if (bo->va_mapped)
      ws->api.bo_va_op(bo->bo, 0, bo->alloc_size, bo->base.va, 0, AMDGPU_VA_OP_UNMAP);
   if (bo->va)
      ws->api.va_range_free(bo->va);
   if (bo->bo)
      ws->api.bo_free(bo->bo);

   ws->allocated_gtt -= MIN2(ws->allocated_gtt, bo->alloc_size);
   free(bo);
}

void
radv_sgpu_bo_release_destroyed_locked(struct radv_sgpu_winsys *ws, struct radv_sgpu_bo *bo)
{
   if (bo->destroyed && !bo->pending_submits)
      radv_sgpu_bo_release_locked(ws, bo);
}

static void
radv_sgpu_buffer_destroy(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct radv_sgpu_bo *bo = radv_sgpu_bo(_bo);

   if (!bo)
      return;

   simple_mtx_lock(&ws->lock);
   radv_sgpu_bo_remove_resident_locked(bo);

   /* An in-flight submission still reads it: the retirement of its last submission frees it. */
   bo->destroyed = true;
   radv_sgpu_bo_release_destroyed_locked(ws, bo);
   simple_mtx_unlock(&ws->lock);
}

bool
radv_sgpu_bo_get_kms_handle_locked(struct radv_sgpu_bo *bo, uint32_t *handle)
{
   if (!bo->kms_handle_valid) {
      if (bo->ws->api.bo_export(bo->bo, amdgpu_bo_handle_type_kms, &bo->kms_handle))
         return false;
      bo->kms_handle_valid = true;
   }

   *handle = bo->kms_handle;
   return true;
}

/* VK_EXT_map_memory_placed: the caller owns a reservation at fixed_addr (the WoW64 path of
 * winevulkan keeps Vulkan memory below 4 GiB), so the BO must be mapped exactly there.  The
 * vendor map cannot place a mapping; use the GEM mmap offset like the amdgpu winsys.
 */
static void *
radv_sgpu_bo_map_fixed_locked(struct radv_sgpu_winsys *ws, struct radv_sgpu_bo *bo, void *fixed_addr)
{
   if (bo->fixed_mapped)
      return bo->fixed_map == fixed_addr ? bo->fixed_map : NULL;

   uint32_t handle;
   if (!radv_sgpu_bo_get_kms_handle_locked(bo, &handle))
      return NULL;

   union drm_amdgpu_gem_mmap args = {.in.handle = handle};
   int ret;
   do {
      ret = ioctl(ws->fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &args);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   if (ret)
      return NULL;

   void *map = mmap(fixed_addr, bo->base.size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, ws->fd,
                    args.out.addr_ptr);
   if (map == MAP_FAILED)
      return NULL;

   bo->fixed_map = map;
   bo->fixed_mapped = true;
   return map;
}

static void *
radv_sgpu_buffer_map(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo, bool use_fixed_addr, void *fixed_addr)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct radv_sgpu_bo *bo = radv_sgpu_bo(_bo);
   void *map = NULL;

   simple_mtx_lock(&ws->lock);
   if (use_fixed_addr) {
      map = radv_sgpu_bo_map_fixed_locked(ws, bo, fixed_addr);
   } else {
      if (!bo->cpu_mapped && !ws->api.bo_cpu_map(bo->bo, &bo->cpu_map))
         bo->cpu_mapped = true;
      map = bo->cpu_mapped ? bo->cpu_map : NULL;
   }
   simple_mtx_unlock(&ws->lock);
   return map;
}

static void
radv_sgpu_buffer_unmap(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo, bool replace)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct radv_sgpu_bo *bo = radv_sgpu_bo(_bo);

   simple_mtx_lock(&ws->lock);
   if (bo->fixed_mapped) {
      /* VK_MEMORY_UNMAP_RESERVE_BIT_EXT hands the range back as an inaccessible reservation. */
      if (replace)
         mmap(bo->fixed_map, bo->base.size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      else
         munmap(bo->fixed_map, bo->base.size);
      bo->fixed_map = NULL;
      bo->fixed_mapped = false;
   } else if (bo->cpu_mapped) {
      ws->api.bo_cpu_unmap(bo->bo);
      bo->cpu_map = NULL;
      bo->cpu_mapped = false;
   }
   simple_mtx_unlock(&ws->lock);
}

static VkResult
radv_sgpu_buffer_from_ptr(struct radeon_winsys *_ws, void *pointer, uint64_t size, unsigned priority,
                          struct radeon_winsys_bo **out_bo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

static VkResult
radv_sgpu_buffer_from_fd(struct radeon_winsys *_ws, int fd, unsigned priority, struct radeon_winsys_bo **out_bo,
                         uint64_t *alloc_size)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct amdgpu_bo_import_result imported = {0};
   struct amdgpu_bo_info info = {0};
   amdgpu_va_handle va_handle = NULL;
   uint64_t va = 0;

   *out_bo = NULL;
   if (fd < 0)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;

   struct radv_sgpu_bo *bo = calloc(1, sizeof(*bo));
   if (!bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   simple_mtx_lock(&ws->lock);

   if (ws->api.bo_import(ws->device, amdgpu_bo_handle_type_dma_buf_fd, (uint32_t)fd, &imported) ||
       !imported.buf_handle || ws->api.bo_query_info(imported.buf_handle, &info)) {
      if (imported.buf_handle)
         ws->api.bo_free(imported.buf_handle);
      simple_mtx_unlock(&ws->lock);
      free(bo);
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   const uint64_t size = imported.alloc_size ? imported.alloc_size : info.alloc_size;
   const uint64_t aligned_size = align64(MAX2(size, 1), 4096);
   const uint64_t vm_flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;

   if (ws->api.va_range_alloc(ws->device, amdgpu_gpu_va_range_general, aligned_size, 4096, 0, &va, &va_handle, 0)) {
      ws->api.bo_free(imported.buf_handle);
      simple_mtx_unlock(&ws->lock);
      free(bo);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   if (ws->api.bo_va_op(imported.buf_handle, 0, aligned_size, va, vm_flags, AMDGPU_VA_OP_MAP)) {
      ws->api.va_range_free(va_handle);
      ws->api.bo_free(imported.buf_handle);
      simple_mtx_unlock(&ws->lock);
      free(bo);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   bo->ws = ws;
   bo->bo = imported.buf_handle;
   bo->va = va_handle;
   bo->alloc_size = aligned_size;
   bo->vm_flags = vm_flags;
   bo->priority = MIN2(priority, UINT8_MAX);
   bo->va_mapped = true;
   list_inithead(&bo->resident_link);

   bo->base.va = va;
   bo->base.size = size;
   bo->base.initial_domain = RADEON_DOMAIN_GTT;
   bo->base.vram_no_cpu_access = !!(info.alloc_flags & AMDGPU_GEM_CREATE_NO_CPU_ACCESS);
   bo->base.obj_id = va;

   ws->allocated_gtt += aligned_size;
   simple_mtx_unlock(&ws->lock);

   if (alloc_size)
      *alloc_size = size;
   *out_bo = &bo->base;
   return VK_SUCCESS;
}

static bool
radv_sgpu_buffer_get_fd(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo, int *fd)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   uint32_t handle = 0;

   if (ws->api.bo_export(radv_sgpu_bo(_bo)->bo, amdgpu_bo_handle_type_dma_buf_fd, &handle) || handle > INT_MAX)
      return false;

   *fd = (int)handle;
   return true;
}

static bool
radv_sgpu_buffer_get_flags_from_fd(struct radeon_winsys *_ws, int fd, enum radeon_bo_domain *domains,
                                   enum radeon_bo_flag *flags)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct amdgpu_bo_import_result imported = {0};
   struct amdgpu_bo_info info = {0};

   *domains = 0;
   *flags = 0;
   if (fd < 0)
      return false;

   int ret = ws->api.bo_import(ws->device, amdgpu_bo_handle_type_dma_buf_fd, (uint32_t)fd, &imported);
   if (!ret && imported.buf_handle) {
      ret = ws->api.bo_query_info(imported.buf_handle, &info);
      ws->api.bo_free(imported.buf_handle);
   }
   if (ret)
      return false;

   *domains = RADEON_DOMAIN_GTT;
   if (info.alloc_flags & AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED)
      *flags |= RADEON_FLAG_CPU_ACCESS;
   if (info.alloc_flags & AMDGPU_GEM_CREATE_NO_CPU_ACCESS)
      *flags |= RADEON_FLAG_NO_CPU_ACCESS;
   if (!(info.alloc_flags & AMDGPU_GEM_CREATE_EXPLICIT_SYNC))
      *flags |= RADEON_FLAG_IMPLICIT_SYNC;
   if (info.alloc_flags & AMDGPU_GEM_CREATE_CPU_GTT_USWC)
      *flags |= RADEON_FLAG_GTT_WC;
   return true;
}

static void
radv_sgpu_buffer_set_metadata(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo,
                              struct radeon_bo_metadata *md)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct radv_sgpu_bo *bo = radv_sgpu_bo(_bo);
   struct amdgpu_bo_metadata metadata = {0};

   bo->metadata = *md;

   /* The vendor stack reads the GFX12 tiling layout of the metadata. */
   metadata.tiling_info = AMDGPU_TILING_SET(GFX12_SWIZZLE_MODE, md->u.gfx12.swizzle_mode) |
                          AMDGPU_TILING_SET(GFX12_DCC_MAX_COMPRESSED_BLOCK, md->u.gfx12.dcc_max_compressed_block) |
                          AMDGPU_TILING_SET(GFX12_DCC_NUMBER_TYPE, md->u.gfx12.dcc_number_type) |
                          AMDGPU_TILING_SET(GFX12_DCC_DATA_FORMAT, md->u.gfx12.dcc_data_format) |
                          AMDGPU_TILING_SET(GFX12_DCC_WRITE_COMPRESS_DISABLE, md->u.gfx12.dcc_write_compress_disable) |
                          AMDGPU_TILING_SET(GFX12_SCANOUT, md->u.gfx12.scanout);
   metadata.size_metadata = MIN2(md->size_metadata, (uint32_t)sizeof(metadata.umd_metadata));
   memcpy(metadata.umd_metadata, md->metadata, metadata.size_metadata);
   ws->api.bo_set_metadata(bo->bo, &metadata);
}

static void
radv_sgpu_buffer_get_metadata(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo, struct radeon_bo_metadata *md)
{
   *md = radv_sgpu_bo(_bo)->metadata;
}

/* Sparse resources, as in the vendor driver and the amdgpu winsys: a VA range mapped PRT, where
 * reads of unbound pages return zero and writes are dropped, and pages bound by replacing
 * parts of it.
 *
 * Scalar loads from unbound PRT pages hang the GPU.  Buffers that shaders may read that way
 * (RADEON_FLAG_EMULATE_SPARSE_RESIDENCY) live at their VA with the PRT workaround control bit
 * set; scalar loads clear the bit (ac_nir_fixup_smem_loads_null_prt), and the low alias maps
 * the same pages, with a zero BO where nothing is bound.
 */
static VkResult
radv_sgpu_init_null_prt_bo(struct radv_sgpu_winsys *ws)
{
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&ws->null_prt_lock);
   if (!ws->null_prt_bo) {
      struct radeon_winsys_bo *bo;

      result = ws->base.buffer_create(&ws->base, SGPU_NULL_PRT_BO_SIZE, 4096, RADEON_DOMAIN_GTT,
                                      RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING,
                                      RADV_BO_PRIORITY_VIRTUAL, 0, &bo);
      if (result == VK_SUCCESS) {
         void *map = ws->base.buffer_map(&ws->base, bo, false, NULL);
         if (map) {
            memset(map, 0, SGPU_NULL_PRT_BO_SIZE);
            ws->base.buffer_unmap(&ws->base, bo, false);
            ws->null_prt_bo = bo;
         } else {
            ws->base.buffer_destroy(&ws->base, bo);
            result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         }
      }
   }
   simple_mtx_unlock(&ws->null_prt_lock);

   return result;
}

/* Maps [va, va + size) to the zero BO, read-only. */
static int
radv_sgpu_bind_null_prt(struct radv_sgpu_winsys *ws, uint64_t va, uint64_t size, uint32_t op)
{
   struct radv_sgpu_bo *null_bo = radv_sgpu_bo(ws->null_prt_bo);

   for (uint64_t offset = 0; offset < size; offset += SGPU_NULL_PRT_BO_SIZE) {
      const uint64_t chunk_size = MIN2(size - offset, SGPU_NULL_PRT_BO_SIZE);
      int ret = ws->api.bo_va_op_raw(ws->device, null_bo->bo, 0, chunk_size, va + offset, AMDGPU_VM_PAGE_READABLE,
                                     op);
      if (ret)
         return ret;
   }

   return 0;
}

static VkResult
radv_sgpu_virtual_bo_create(struct radv_sgpu_winsys *ws, uint64_t size, unsigned alignment,
                            enum radeon_bo_flag flags, uint64_t address, struct radeon_winsys_bo **out_bo)
{
   const uint64_t prt_wa_bit = 1ull << ws->info.address_prt_wa_control_bit;
   const bool emulate_sparse_residency = flags & RADEON_FLAG_EMULATE_SPARSE_RESIDENCY;
   uint64_t va = 0;
   int ret;

   if (emulate_sparse_residency) {
      VkResult result = radv_sgpu_init_null_prt_bo(ws);
      if (result != VK_SUCCESS)
         return result;
   }

   struct radv_sgpu_bo *bo = calloc(1, sizeof(*bo));
   if (!bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   const uint64_t alloc_size = align64(MAX2(size, 1), MAX2(alignment, 65536u));

   /* The low alias is an ordinary VA; the high one is in the range reserved at creation. */
   if (ws->api.va_range_alloc(ws->device, amdgpu_gpu_va_range_general, alloc_size, MAX2(alignment, 65536u),
                              address & ~prt_wa_bit, &va, &bo->va, 0)) {
      free(bo);
      return address ? VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   bo->ws = ws;
   bo->alloc_size = alloc_size;
   bo->emulate_sparse_residency = emulate_sparse_residency;
   /* No kernel BO: never in a CS BO list. */
   bo->is_local = true;
   list_inithead(&bo->resident_link);

   bo->base.va = emulate_sparse_residency ? va | prt_wa_bit : va;
   bo->base.size = size;
   bo->base.initial_domain = RADEON_DOMAIN_GTT;
   bo->base.is_virtual = true;
   bo->base.obj_id = bo->base.va;

   ret = ws->api.bo_va_op_raw(ws->device, NULL, 0, alloc_size, bo->base.va, AMDGPU_VM_PAGE_PRT, AMDGPU_VA_OP_MAP);
   if (!ret && emulate_sparse_residency)
      ret = radv_sgpu_bind_null_prt(ws, va, alloc_size, AMDGPU_VA_OP_MAP);
   if (ret) {
      fprintf(stderr, "radv/sgpu: could not reserve a PRT range (%d).\n", ret);
      simple_mtx_lock(&ws->lock);
      radv_sgpu_bo_release_locked(ws, bo);
      simple_mtx_unlock(&ws->lock);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   *out_bo = &bo->base;
   return VK_SUCCESS;
}

static VkResult
radv_sgpu_buffer_virtual_bind(struct radeon_winsys *_ws, struct radeon_winsys_bo *_parent, uint64_t offset,
                              uint64_t size, struct radeon_winsys_bo *_bo, uint64_t bo_offset)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct radv_sgpu_bo *parent = radv_sgpu_bo(_parent);
   struct radv_sgpu_bo *bo = _bo ? radv_sgpu_bo(_bo) : NULL;
   const uint64_t va = parent->base.va + offset;
   const uint64_t low_va = va & ~(1ull << ws->info.address_prt_wa_control_bit);
   int ret;

   assert(parent->base.is_virtual && (!bo || !bo->base.is_virtual));

   if (bo) {
      ret = ws->api.bo_va_op_raw(ws->device, bo->bo, bo_offset, size, va, bo->vm_flags, AMDGPU_VA_OP_REPLACE);
      if (!ret && parent->emulate_sparse_residency)
         ret = ws->api.bo_va_op_raw(ws->device, bo->bo, bo_offset, size, low_va, bo->vm_flags, AMDGPU_VA_OP_REPLACE);

      /* A submission lists only the BOs its command streams reference. */
      if (!ret && !bo->is_local)
         radv_sgpu_buffer_make_resident(_ws, _bo, true);
   } else {
      ret = ws->api.bo_va_op_raw(ws->device, NULL, 0, size, va, AMDGPU_VM_PAGE_PRT, AMDGPU_VA_OP_REPLACE);
      if (!ret && parent->emulate_sparse_residency)
         ret = radv_sgpu_bind_null_prt(ws, low_va, size, AMDGPU_VA_OP_REPLACE);
   }

   return ret ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}

/* RADV makes resident the allocations it reaches through descriptors only; they must go with
 * every submission.
 */
static VkResult
radv_sgpu_buffer_make_resident(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo, bool resident)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   struct radv_sgpu_bo *bo = radv_sgpu_bo(_bo);

   simple_mtx_lock(&ws->lock);
   if (!resident) {
      radv_sgpu_bo_remove_resident_locked(bo);
   } else if (!bo->base.use_global_list) {
      list_addtail(&bo->resident_link, &ws->resident_bos);
      bo->base.use_global_list = true;
   }
   simple_mtx_unlock(&ws->lock);
   return VK_SUCCESS;
}

static bool
radv_sgpu_bo_wait_for_idle(struct radeon_winsys *_ws, struct radeon_winsys_bo *_bo)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);
   bool busy = false;

   return !ws->api.bo_wait_for_idle(radv_sgpu_bo(_bo)->bo, UINT64_MAX, &busy) && !busy;
}

void
radv_sgpu_bo_init_functions(struct radv_sgpu_winsys *ws)
{
   ws->base.buffer_create = radv_sgpu_buffer_create;
   ws->base.buffer_destroy = radv_sgpu_buffer_destroy;
   ws->base.buffer_map = radv_sgpu_buffer_map;
   ws->base.buffer_unmap = radv_sgpu_buffer_unmap;
   ws->base.buffer_from_ptr = radv_sgpu_buffer_from_ptr;
   ws->base.buffer_from_fd = radv_sgpu_buffer_from_fd;
   ws->base.buffer_get_fd = radv_sgpu_buffer_get_fd;
   ws->base.buffer_get_flags_from_fd = radv_sgpu_buffer_get_flags_from_fd;
   ws->base.buffer_set_metadata = radv_sgpu_buffer_set_metadata;
   ws->base.buffer_get_metadata = radv_sgpu_buffer_get_metadata;
   ws->base.buffer_virtual_bind = radv_sgpu_buffer_virtual_bind;
   ws->base.buffer_make_resident = radv_sgpu_buffer_make_resident;
   ws->base.bo_wait_for_idle = radv_sgpu_bo_wait_for_idle;
}
