/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Winsys for Samsung Xclipse (SGPU).
 *
 * The kernel driver is Samsung's, reached through the vendor libdrm_sgpu.so, which has the
 * libdrm_amdgpu ABI but is not the system libdrm: it is loaded with dlopen and every entry
 * point goes through struct radv_sgpu_api.  Submission builds the AMDGPU CS ioctl chunks
 * directly (amdgpu_cs_submit_raw) with one job per submit.
 */

#ifndef RADV_SGPU_WINSYS_H
#define RADV_SGPU_WINSYS_H

#include <amdgpu.h>

#include "util/list.h"
#include "util/simple_mtx.h"
#include "ac_gpu_info.h"
#include "radv_radeon_winsys.h"

struct util_sync_provider;

struct radv_sgpu_api {
   int (*device_initialize)(int fd, uint32_t *major, uint32_t *minor, amdgpu_device_handle *device);
   int (*device_deinitialize)(amdgpu_device_handle device);
   int (*bo_alloc)(amdgpu_device_handle device, struct amdgpu_bo_alloc_request *request, amdgpu_bo_handle *bo);
   int (*bo_import)(amdgpu_device_handle device, enum amdgpu_bo_handle_type type, uint32_t shared_handle,
                    struct amdgpu_bo_import_result *result);
   int (*bo_free)(amdgpu_bo_handle bo);
   int (*bo_export)(amdgpu_bo_handle bo, enum amdgpu_bo_handle_type type, uint32_t *shared_handle);
   int (*bo_cpu_map)(amdgpu_bo_handle bo, void **cpu);
   int (*bo_cpu_unmap)(amdgpu_bo_handle bo);
   int (*bo_wait_for_idle)(amdgpu_bo_handle bo, uint64_t timeout_ns, bool *busy);
   int (*bo_query_info)(amdgpu_bo_handle bo, struct amdgpu_bo_info *info);
   int (*bo_set_metadata)(amdgpu_bo_handle bo, struct amdgpu_bo_metadata *info);
   int (*va_range_alloc)(amdgpu_device_handle device, enum amdgpu_gpu_va_range type, uint64_t size,
                         uint64_t alignment, uint64_t required_va, uint64_t *allocated_va, amdgpu_va_handle *va,
                         uint64_t flags);
   int (*va_range_free)(amdgpu_va_handle va);
   int (*bo_va_op)(amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr, uint64_t flags, uint32_t op);
   /* Also maps ranges without a BO (sparse PRT). */
   int (*bo_va_op_raw)(amdgpu_device_handle device, amdgpu_bo_handle bo, uint64_t offset, uint64_t size,
                       uint64_t addr, uint64_t flags, uint32_t op);
   int (*ctx_create2)(amdgpu_device_handle device, uint32_t priority, amdgpu_context_handle *context);
   int (*ctx_free)(amdgpu_context_handle context);
   int (*cs_submit_raw)(amdgpu_device_handle device, amdgpu_context_handle context,
                        amdgpu_bo_list_handle bo_list_handle, int num_chunks, struct drm_amdgpu_cs_chunk *chunks,
                        uint64_t *seq_no);
   int (*cs_query_fence_status)(struct amdgpu_cs_fence *fence, uint64_t timeout_ns, uint64_t flags,
                                uint32_t *expired);
   int (*syncobj_create)(amdgpu_device_handle device, uint32_t *syncobj);
   int (*syncobj_destroy)(amdgpu_device_handle device, uint32_t syncobj);
   int (*syncobj_signal)(amdgpu_device_handle device, const uint32_t *syncobjs, uint32_t count);
   /* Optional. */
   int (*cs_query_reset_state2)(amdgpu_context_handle context, uint64_t *flags);
   int (*query_info)(amdgpu_device_handle device, unsigned info_id, unsigned size, void *value);
};

struct radv_sgpu_winsys {
   struct radeon_winsys base;
   struct radeon_info info;
   struct radv_sgpu_api api;
   void *vendor_lib;
   amdgpu_device_handle device;
   struct util_sync_provider *sync_provider;
   int fd;

   /* Protects BO lifetime (pending_submits), the resident list and allocated_gtt. */
   simple_mtx_t lock;
   /* Non-local BOs made resident; they accompany every submission. */
   struct list_head resident_bos;
   uint64_t allocated_gtt;
   /* Bump cursor of the capture/replay VA window. */
   uint64_t replay_va_cursor;

   /* The VA range with the PRT workaround control bit, kept free for the high aliases of sparse
    * buffers (see radv_sgpu_virtual_bo_create).
    */
   amdgpu_va_handle prt_wa_va;
   /* Zero BO behind the low alias of sparse buffers, created on first use. */
   simple_mtx_t null_prt_lock;
   struct radeon_winsys_bo *null_prt_bo;
};

struct radv_sgpu_bo {
   struct radeon_winsys_bo base;
   struct radv_sgpu_winsys *ws;
   amdgpu_bo_handle bo;
   amdgpu_va_handle va;
   uint64_t alloc_size;
   uint64_t vm_flags;
   struct radeon_bo_metadata metadata;
   uint8_t priority;

   /* Created with AMDGPU_GEM_CREATE_VM_ALWAYS_VALID: never listed in a CS BO list. */
   bool is_local;
   bool va_mapped;
   /* Sparse buffer with a low alias that scalar loads read (RADEON_FLAG_EMULATE_SPARSE_RESIDENCY). */
   bool emulate_sparse_residency;

   void *cpu_map;
   bool cpu_mapped;
   /* VK_EXT_map_memory_placed mapping, owned by the winsys. */
   void *fixed_map;
   bool fixed_mapped;

   uint32_t kms_handle;
   bool kms_handle_valid;

   /* Destroyed by the API but still referenced by an in-flight submission. */
   bool destroyed;
   unsigned pending_submits;
   struct list_head resident_link;
   /* Serial of the last submission that listed this BO (dedup under ws->lock). */
   uint64_t submit_stamp;
};

static inline struct radv_sgpu_winsys *
radv_sgpu_winsys(struct radeon_winsys *base)
{
   return (struct radv_sgpu_winsys *)base;
}

static inline struct radv_sgpu_bo *
radv_sgpu_bo(struct radeon_winsys_bo *base)
{
   return (struct radv_sgpu_bo *)base;
}

void radv_sgpu_bo_init_functions(struct radv_sgpu_winsys *ws);
void radv_sgpu_cs_init_functions(struct radv_sgpu_winsys *ws);

/* Physical release of a BO whose submissions retired; ws->lock held. */
void radv_sgpu_bo_release_destroyed_locked(struct radv_sgpu_winsys *ws, struct radv_sgpu_bo *bo);
bool radv_sgpu_bo_get_kms_handle_locked(struct radv_sgpu_bo *bo, uint32_t *handle);

#endif /* RADV_SGPU_WINSYS_H */
