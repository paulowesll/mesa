/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Winsys for Samsung Xclipse (SGPU).
 *
 * The kernel driver is Samsung's, but it takes the amdgpu UAPI, so libdrm_amdgpu drives it.
 * Submission builds the AMDGPU CS ioctl chunks directly (amdgpu_cs_submit_raw) with one job
 * per submit.
 */

#ifndef RADV_SGPU_WINSYS_H
#define RADV_SGPU_WINSYS_H

#include <amdgpu.h>

#include "util/list.h"
#include "util/simple_mtx.h"
#include "ac_gpu_info.h"
#include "radv_radeon_winsys.h"

struct util_sync_provider;

struct radv_sgpu_winsys {
   struct radeon_winsys base;
   struct radeon_info info;
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
   struct radv_sgpu_prt_wa_range *prt_wa_range;
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
