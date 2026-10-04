/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_sgpu_winsys.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drm-uapi/amdgpu_drm.h"
#include "util/libsync.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "util/u_sync_provider.h"
#include "ac_cmdbuf_cp.h"
#include "sid.h"
#include "vk_drm_syncobj.h"
#include "vk_sync_dummy.h"

#define SGPU_MAX_RINGS 8
#define SGPU_NUM_IPS   (AMDGPU_HW_IP_NUM + 1)

/* The INDIRECT_BUFFER packet the kernel emits has a 20-bit IB_SIZE; longer streams are split
 * at packet boundaries into several IBs of the same job, at most this many.
 */
#define SGPU_MAX_IB_DW    ((1u << 20) - 256)
#define SGPU_MAX_SPLIT_IB 8

/*
 * Command streams.
 *
 * Commands are recorded in host memory and copied into a GTT BO at finalize time.  The BO is
 * kept across resets and reused once no submission references it: creating, mapping and
 * destroying one per command buffer costs several ioctls per vkEndCommandBuffer.
 */

struct radv_sgpu_cs {
   struct ac_cmdbuf base;
   struct radv_sgpu_winsys *ws;
   enum amd_ip_type ip_type;
   VkResult status;
   bool finalized;

   /* Finalized stream. */
   struct radeon_winsys_bo *ib_bo;
   uint32_t ib_cdw;
   uint32_t split_count; /* 0 = a single IB of ib_cdw dwords */
   uint32_t split_off_dw[SGPU_MAX_SPLIT_IB];
   uint32_t split_cdw[SGPU_MAX_SPLIT_IB];

   /* Previous ib_bo, kept CPU-mapped for reuse. */
   struct radeon_winsys_bo *spare_bo;

   struct radeon_winsys_bo **bos;
   unsigned num_bos;
   unsigned max_bos;
};

static struct radv_sgpu_cs *
radv_sgpu_cs(struct ac_cmdbuf *base)
{
   return (struct radv_sgpu_cs *)base;
}

static void
radv_sgpu_cs_drop_spare(struct radv_sgpu_cs *cs)
{
   if (cs->spare_bo)
      cs->ws->base.buffer_destroy(&cs->ws->base, cs->spare_bo);
   cs->spare_bo = NULL;
}

static void
radv_sgpu_cs_release_ib(struct radv_sgpu_cs *cs)
{
   if (cs->ib_bo) {
      radv_sgpu_cs_drop_spare(cs);
      cs->spare_bo = cs->ib_bo;
   }
   cs->ib_bo = NULL;
   cs->ib_cdw = 0;
   cs->split_count = 0;
}

static bool
radv_sgpu_cs_track_bo(struct radv_sgpu_cs *cs, struct radeon_winsys_bo *bo)
{
   for (unsigned i = 0; i < cs->num_bos; i++) {
      if (cs->bos[i] == bo)
         return true;
   }

   if (cs->num_bos == cs->max_bos) {
      const unsigned new_max = MAX2(16, cs->max_bos * 2);
      struct radeon_winsys_bo **bos = realloc(cs->bos, new_max * sizeof(*bos));
      if (!bos)
         return false;
      cs->bos = bos;
      cs->max_bos = new_max;
   }

   cs->bos[cs->num_bos++] = bo;
   return true;
}

static struct ac_cmdbuf *
radv_sgpu_cs_create(struct radeon_winsys *_ws, enum amd_ip_type ip_type, bool is_secondary)
{
   const uint32_t initial_dw = 20 * 1024;

   struct radv_sgpu_cs *cs = calloc(1, sizeof(*cs));
   if (!cs)
      return NULL;

   cs->base.buf = malloc(initial_dw * sizeof(uint32_t));
   if (!cs->base.buf) {
      free(cs);
      return NULL;
   }

   cs->ws = radv_sgpu_winsys(_ws);
   cs->ip_type = ip_type;
   cs->status = VK_SUCCESS;
   cs->base.max_dw = initial_dw;
   cs->base.remap_ctx_regs = true;
   return &cs->base;
}

static void
radv_sgpu_cs_destroy(struct ac_cmdbuf *base)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   radv_sgpu_cs_release_ib(cs);
   radv_sgpu_cs_drop_spare(cs);
   free(cs->bos);
   free(cs->base.buf);
   free(cs);
}

static void
radv_sgpu_cs_reset(struct ac_cmdbuf *base)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   radv_sgpu_cs_release_ib(cs);
   cs->num_bos = 0;
   cs->base.cdw = 0;
   cs->base.reserved_dw = 0;
   cs->status = VK_SUCCESS;
   cs->finalized = false;
}

static void
radv_sgpu_cs_grow(struct ac_cmdbuf *base, size_t min_size)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   if (cs->status != VK_SUCCESS)
      return;

   const size_t new_dw = MAX2(min_size + 16, (size_t)base->max_dw * 2);
   uint32_t *buf = new_dw <= UINT32_MAX ? realloc(base->buf, new_dw * sizeof(uint32_t)) : NULL;
   if (!buf) {
      cs->status = VK_ERROR_OUT_OF_HOST_MEMORY;
      base->cdw = 0;
      return;
   }

   base->buf = buf;
   base->max_dw = new_dw;
}

/* One header-only NOP for a single dword, else one NOP packet with an ignored body.  SDMA pads
 * with its own NOP.
 */
static uint32_t
radv_sgpu_emit_nops(enum amd_ip_type ip_type, uint32_t *buf, uint32_t num_dw)
{
   if (!num_dw)
      return 0;

   if (ip_type == AMD_IP_SDMA) {
      for (uint32_t i = 0; i < num_dw; i++)
         buf[i] = SDMA_NOP_PAD;
   } else if (num_dw == 1) {
      buf[0] = PKT3_NOP_PAD;
   } else {
      buf[0] = PKT3(PKT3_NOP, num_dw - 2, 0);
      memset(buf + 1, 0, (num_dw - 1) * sizeof(uint32_t));
   }
   return num_dw;
}

static void
radv_sgpu_cs_pad(struct ac_cmdbuf *base, unsigned leave_dw_space)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);
   const uint32_t pad_mask = cs->ws->info.ip[cs->ip_type].ib_pad_dw_mask;
   const uint32_t unaligned = (base->cdw + leave_dw_space) & pad_mask;

   if (unaligned)
      base->cdw += radv_sgpu_emit_nops(cs->ip_type, base->buf + base->cdw, pad_mask + 1 - unaligned);
}

/* Splits a stream longer than the IB_SIZE limit at packet boundaries.  Returns the padded
 * chunks laid out back to back, or NULL on failure (cs->status set).
 */
static uint32_t *
radv_sgpu_cs_split(struct radv_sgpu_cs *cs, uint32_t *out_dw)
{
   const struct ac_cmdbuf *base = &cs->base;
   const uint32_t pad_dw = cs->ws->info.ip[cs->ip_type].ib_pad_dw_mask + 1;
   const uint32_t limit = SGPU_MAX_IB_DW - pad_dw;

   uint32_t *out = malloc(((size_t)base->cdw + SGPU_MAX_SPLIT_IB * pad_dw) * sizeof(uint32_t));
   if (!out) {
      cs->status = VK_ERROR_OUT_OF_HOST_MEMORY;
      return NULL;
   }

   uint32_t o = 0, chunk_start = 0, n = 0;
   for (uint32_t i = 0; i <= base->cdw;) {
      uint32_t pkt_dw = 1;
      if (i < base->cdw && PKT_TYPE_G(base->buf[i]) == 3)
         pkt_dw = PKT_COUNT_G(base->buf[i]) + 2;

      if (i == base->cdw || i + pkt_dw - chunk_start > limit) {
         if (n == SGPU_MAX_SPLIT_IB) {
            free(out);
            cs->status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            return NULL;
         }

         cs->split_off_dw[n] = o;
         memcpy(out + o, base->buf + chunk_start, (size_t)(i - chunk_start) * sizeof(uint32_t));
         o += i - chunk_start;

         const uint32_t unaligned = (o - cs->split_off_dw[n]) % pad_dw;
         if (unaligned)
            o += radv_sgpu_emit_nops(cs->ip_type, out + o, pad_dw - unaligned);

         cs->split_cdw[n] = o - cs->split_off_dw[n];
         n++;
         chunk_start = i;
         if (i == base->cdw)
            break;
      }
      i += pkt_dw;
   }

   cs->split_count = n;
   *out_dw = o;
   return out;
}

static VkResult
radv_sgpu_cs_finalize(struct ac_cmdbuf *base)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);
   struct radv_sgpu_winsys *ws = cs->ws;
   struct radeon_winsys_bo *bo = NULL;

   if (cs->status != VK_SUCCESS || cs->finalized)
      return cs->status;

   cs->finalized = true;
   if (!base->cdw)
      return VK_SUCCESS;

   radv_sgpu_cs_pad(base, 0);

   const uint32_t *src = base->buf;
   uint32_t src_dw = base->cdw;
   uint32_t *split = NULL;
   cs->split_count = 0;
   if ((cs->ip_type == AMD_IP_GFX || cs->ip_type == AMD_IP_COMPUTE) && base->cdw > SGPU_MAX_IB_DW) {
      split = radv_sgpu_cs_split(cs, &src_dw);
      if (!split)
         return cs->status;
      src = split;
   }

   const uint64_t bytes = (uint64_t)src_dw * 4;

   if (cs->spare_bo) {
      struct radv_sgpu_bo *spare = radv_sgpu_bo(cs->spare_bo);
      simple_mtx_lock(&ws->lock);
      const bool reusable = !spare->pending_submits && spare->base.size >= bytes;
      simple_mtx_unlock(&ws->lock);
      if (reusable) {
         bo = cs->spare_bo;
         cs->spare_bo = NULL;
      } else {
         radv_sgpu_cs_drop_spare(cs);
      }
   }

   if (!bo) {
      /* Round up so a command buffer that grows a little keeps its BO. */
      const uint64_t size = align64(MAX2(bytes + bytes / 4, 64 * 1024), 4096);
      const unsigned alignment = MAX2(ws->info.ip[cs->ip_type].ib_alignment, 4096u);
      const enum radeon_bo_flag flags = RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING |
                                        RADEON_FLAG_READ_ONLY | RADEON_FLAG_GTT_WC | RADEON_FLAG_GL2_BYPASS;

      cs->status = ws->base.buffer_create(&ws->base, size, alignment, RADEON_DOMAIN_GTT, flags, RADV_BO_PRIORITY_CS,
                                          0, &bo);
      if (cs->status != VK_SUCCESS) {
         free(split);
         return cs->status;
      }
   }

   void *map = ws->base.buffer_map(&ws->base, bo, false, NULL);
   if (!map) {
      ws->base.buffer_destroy(&ws->base, bo);
      free(split);
      cs->status = VK_ERROR_MEMORY_MAP_FAILED;
      return cs->status;
   }

   /* The BO stays mapped for reuse. */
   memcpy(map, src, bytes);
   free(split);

   if (!radv_sgpu_cs_track_bo(cs, bo)) {
      ws->base.buffer_destroy(&ws->base, bo);
      cs->status = VK_ERROR_OUT_OF_HOST_MEMORY;
      return cs->status;
   }

   cs->ib_bo = bo;
   cs->ib_cdw = src_dw;
   return VK_SUCCESS;
}

static bool
radv_sgpu_cs_chain(struct ac_cmdbuf *cs, struct ac_cmdbuf *next_cs, bool pre_en)
{
   return false;
}

static void
radv_sgpu_cs_unchain(struct ac_cmdbuf *cs)
{
}

static void
radv_sgpu_cs_add_buffer(struct ac_cmdbuf *base, struct radeon_winsys_bo *bo)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   if (cs->status == VK_SUCCESS && bo && !radv_sgpu_cs_track_bo(cs, bo))
      cs->status = VK_ERROR_OUT_OF_HOST_MEMORY;
}

/* No IB2: secondary command buffers are copied inline. */
static void
radv_sgpu_cs_execute_secondary(struct ac_cmdbuf *parent, struct ac_cmdbuf *child, bool allow_ib2)
{
   struct radv_sgpu_cs *parent_cs = radv_sgpu_cs(parent);
   struct radv_sgpu_cs *child_cs = radv_sgpu_cs(child);

   if (parent_cs->status != VK_SUCCESS)
      return;
   if (child_cs->status != VK_SUCCESS) {
      parent_cs->status = child_cs->status;
      return;
   }

   for (unsigned i = 0; i < child_cs->num_bos; i++)
      radv_sgpu_cs_add_buffer(parent, child_cs->bos[i]);

   if (parent->cdw + child->cdw > parent->max_dw)
      radv_sgpu_cs_grow(parent, parent->cdw + child->cdw);
   if (parent_cs->status != VK_SUCCESS)
      return;

   memcpy(parent->buf + parent->cdw, child->buf, child->cdw * sizeof(uint32_t));
   parent->cdw += child->cdw;
}

static void
radv_sgpu_cs_execute_ib(struct ac_cmdbuf *base, struct radeon_winsys_bo *bo, uint64_t va, uint32_t cdw,
                        bool predicate)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   if (bo) {
      radv_sgpu_cs_add_buffer(base, bo);
      va = bo->va;
   }

   if (cs->status != VK_SUCCESS)
      return;
   if (!va || cdw > 0xfffff) { /* 20-bit IB_SIZE */
      cs->status = VK_ERROR_INITIALIZATION_FAILED;
      return;
   }

   ac_emit_cp_indirect_buffer(base, va, cdw, 0, predicate);
}

static void
radv_sgpu_cs_chain_dgc_ib(struct ac_cmdbuf *cs, uint64_t va, uint32_t cdw, uint64_t trailer_va, bool predicate)
{
   /* Device-generated commands are not exposed on Xclipse. */
   UNREACHABLE("DGC is not supported on SGPU");
}

static void
radv_sgpu_cs_dump(struct ac_cmdbuf *base, FILE *file, const int *trace_ids, int trace_id_count,
                  enum radv_cs_dump_type type)
{
   const struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   fprintf(file, "SGPU command stream: ip %u, %u dwords, %u IB(s) at 0x%" PRIx64 "\n", cs->ip_type, base->cdw,
           MAX2(cs->split_count, 1), cs->ib_bo ? cs->ib_bo->va : 0);
}

static void
radv_sgpu_cs_annotate(struct ac_cmdbuf *cs, const char *marker)
{
}

/*
 * Contexts and submission.
 */

struct radv_sgpu_retention {
   struct radv_sgpu_retention *next;
   struct amdgpu_cs_fence fence;
   struct radv_sgpu_bo **bos;
   unsigned num_bos;
};

struct radv_sgpu_ctx {
   struct radv_sgpu_winsys *ws;
   amdgpu_context_handle context;

   /* Serializes submissions and the per-route state below. */
   simple_mtx_t lock;
   uint32_t queue_syncobj[SGPU_NUM_IPS][SGPU_MAX_RINGS];
   /* Waits of empty submissions were folded into the queue syncobj: the next CS waits on it. */
   bool queue_syncobj_wait[SGPU_NUM_IPS][SGPU_MAX_RINGS];
   struct amdgpu_cs_fence last_fence[SGPU_NUM_IPS][SGPU_MAX_RINGS];
   /* Set once the kernel cancelled a job of this context (GPU hang). */
   uint32_t lost[SGPU_NUM_IPS][SGPU_MAX_RINGS];

   /* In-flight submissions and the BOs they keep alive (ws->lock). */
   struct radv_sgpu_retention *retentions;
};

static bool
radv_sgpu_route_valid(enum amd_ip_type ip, int ring)
{
   return ip < SGPU_NUM_IPS && ring >= 0 && ring < SGPU_MAX_RINGS;
}

/* After a GPU hang the kernel cancels every job of the context.  More submissions would only
 * queue further timeouts behind a dead context and stall the ring the Android compositor
 * shares, so the route reports VK_ERROR_DEVICE_LOST from then on.
 */
static void
radv_sgpu_mark_lost(struct radv_sgpu_ctx *ctx, enum amd_ip_type ip, int ring, uint64_t seq_no)
{
   if (!radv_sgpu_route_valid(ip, ring) || p_atomic_xchg(&ctx->lost[ip][ring], 1))
      return;

   uint64_t flags = 0;
   const int r = amdgpu_cs_query_reset_state2(ctx->context, &flags);
   fprintf(stderr,
           "radv/sgpu: the kernel cancelled the context (GPU hang) at seq_no %" PRIu64 ", ip %u ring %d; "
           "reset query %d: guilty=%u vram_lost=%u.\n",
           seq_no, ip, ring, r, !!(flags & AMDGPU_CTX_QUERY2_FLAGS_GUILTY),
           !!(flags & AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST));
}

static bool
radv_sgpu_is_lost(struct radv_sgpu_ctx *ctx, enum amd_ip_type ip, int ring)
{
   return p_atomic_read(&ctx->lost[ip][ring]) != 0;
}

static int
radv_sgpu_query_fence(struct radv_sgpu_ctx *ctx, struct amdgpu_cs_fence *fence, uint64_t timeout_ns,
                      uint32_t *expired)
{
   const int ret = amdgpu_cs_query_fence_status(fence, timeout_ns, 0, expired);
   if (ret == -ECANCELED)
      radv_sgpu_mark_lost(ctx, fence->ip_type, fence->ring, fence->fence);
   return ret;
}

static void
radv_sgpu_retention_release_locked(struct radv_sgpu_winsys *ws, struct radv_sgpu_retention *r)
{
   for (unsigned i = 0; i < r->num_bos; i++) {
      assert(r->bos[i]->pending_submits > 0);
      r->bos[i]->pending_submits--;
      radv_sgpu_bo_release_destroyed_locked(ws, r->bos[i]);
   }
   free(r->bos);
   free(r);
}

/* A cancelled job's fence has signalled with an error: it no longer runs, so its BOs are
 * released like those of a completed one.
 */
static void
radv_sgpu_retire(struct radv_sgpu_ctx *ctx)
{
   struct radv_sgpu_winsys *ws = ctx->ws;

   simple_mtx_lock(&ws->lock);
   for (struct radv_sgpu_retention **link = &ctx->retentions; *link;) {
      struct radv_sgpu_retention *r = *link;
      uint32_t expired = 0;
      const int ret = radv_sgpu_query_fence(ctx, &r->fence, 0, &expired);

      if ((!ret && expired) || ret == -ECANCELED) {
         *link = r->next;
         radv_sgpu_retention_release_locked(ws, r);
      } else {
         link = &r->next;
      }
   }
   simple_mtx_unlock(&ws->lock);
}

static uint32_t
radv_sgpu_queue_syncobj(struct radv_sgpu_ctx *ctx, enum amd_ip_type ip, int ring)
{
   struct util_sync_provider *provider = ctx->ws->sync_provider;
   uint32_t *syncobj = &ctx->queue_syncobj[ip][ring];

   if (!*syncobj && provider->create(provider, DRM_SYNCOBJ_CREATE_SIGNALED, syncobj))
      *syncobj = 0;
   return *syncobj;
}

static bool
radv_sgpu_use_timeline(const struct radv_sgpu_winsys *ws)
{
   return ws->info.has_timeline_syncobj;
}

/* Returns the syncobj of a vk_sync, or 0 for no-op (absent or dummy) syncs.  Sets *valid to
 * false when the sync cannot be used with the current syncobj mode.
 */
static uint32_t
radv_sgpu_sync_handle(const struct radv_sgpu_winsys *ws, const struct vk_sync *sync, uint64_t value, bool *valid)
{
   if (!sync || vk_sync_type_is_dummy(sync->type))
      return 0;

   const bool timeline = sync->flags & VK_SYNC_IS_TIMELINE;
   struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj((struct vk_sync *)sync);
   if (!syncobj || !syncobj->syncobj || (timeline && !radv_sgpu_use_timeline(ws)) || (!timeline && value)) {
      *valid = false;
      return 0;
   }
   return syncobj->syncobj;
}

/* Merges one application wait into *fd.  A timeline point is first materialized in a
 * temporary binary syncobj, the provider has no point-aware export.
 */
static int
radv_sgpu_accumulate_wait(struct radv_sgpu_ctx *ctx, uint32_t handle, const struct vk_sync_wait *wait, int *fd)
{
   struct util_sync_provider *provider = ctx->ws->sync_provider;
   uint32_t tmp = 0;
   int wait_fd = -1;
   int ret;

   if (wait->sync->flags & VK_SYNC_IS_TIMELINE) {
      ret = provider->create(provider, 0, &tmp);
      if (ret)
         return ret;

      ret = provider->transfer(provider, tmp, 0, handle, wait->wait_value, 0);
      if (ret) {
         /* Like radv_amdgpu_cs_submit_zero: an already signalled point may have no fence left. */
         uint64_t point = 0;
         if (!provider->query(provider, &handle, &point, 1, 0) && point >= wait->wait_value)
            ret = 0;
         provider->destroy(provider, tmp);
         return ret;
      }
   }

   ret = provider->export_sync_file(provider, tmp ? tmp : handle, &wait_fd);
   if (!ret)
      ret = sync_accumulate("radv", fd, wait_fd);

   if (wait_fd >= 0)
      close(wait_fd);
   if (tmp)
      provider->destroy(provider, tmp);
   return ret;
}

/* Copies the payload of the route's queue syncobj into a signal.  Every CS on the route
 * signals the queue syncobj, so the copy is ordered after all prior work on the route.
 */
static int
radv_sgpu_signal_from_queue(struct radv_sgpu_ctx *ctx, uint32_t queue_syncobj, uint32_t handle,
                            const struct vk_sync_signal *signal)
{
   struct util_sync_provider *provider = ctx->ws->sync_provider;

   if (radv_sgpu_use_timeline(ctx->ws)) {
      const uint64_t point = (signal->sync->flags & VK_SYNC_IS_TIMELINE) ? signal->signal_value : 0;
      return provider->transfer(provider, handle, point, queue_syncobj, 0, 0);
   }

   int fd = -1;
   int ret = provider->export_sync_file(provider, queue_syncobj, &fd);
   if (!ret)
      ret = provider->import_sync_file(provider, handle, fd);
   if (fd >= 0)
      close(fd);
   return ret;
}

/* Empty submissions never reach the kernel: waits are folded into the route's queue syncobj,
 * which the next CS waits on, and signals copy it (like radv_amdgpu_cs_submit_zero).
 * ctx->lock held.
 */
static VkResult
radv_sgpu_submit_zero_locked(struct radv_sgpu_ctx *ctx, enum amd_ip_type ip, int ring, uint32_t wait_count,
                             const struct vk_sync_wait *waits, uint32_t signal_count,
                             const struct vk_sync_signal *signals)
{
   struct util_sync_provider *provider = ctx->ws->sync_provider;
   bool valid = true;
   int ret = 0;

   const uint32_t queue_syncobj = radv_sgpu_queue_syncobj(ctx, ip, ring);
   if (!queue_syncobj)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   bool has_waits = false;
   for (uint32_t i = 0; i < wait_count; i++)
      has_waits |= radv_sgpu_sync_handle(ctx->ws, waits[i].sync, waits[i].wait_value, &valid) != 0;
   for (uint32_t i = 0; i < signal_count; i++)
      radv_sgpu_sync_handle(ctx->ws, signals[i].sync, signals[i].signal_value, &valid);
   if (!valid)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   if (has_waits) {
      int fd = -1;
      ret = provider->export_sync_file(provider, queue_syncobj, &fd);
      for (uint32_t i = 0; !ret && i < wait_count; i++) {
         const uint32_t handle = radv_sgpu_sync_handle(ctx->ws, waits[i].sync, waits[i].wait_value, &valid);
         if (handle)
            ret = radv_sgpu_accumulate_wait(ctx, handle, &waits[i], &fd);
      }
      if (!ret)
         ret = provider->import_sync_file(provider, queue_syncobj, fd);
      if (fd >= 0)
         close(fd);
      if (ret)
         goto fail;

      ctx->queue_syncobj_wait[ip][ring] = true;
   }

   for (uint32_t i = 0; i < signal_count; i++) {
      const uint32_t handle = radv_sgpu_sync_handle(ctx->ws, signals[i].sync, signals[i].signal_value, &valid);
      if (handle) {
         ret = radv_sgpu_signal_from_queue(ctx, queue_syncobj, handle, &signals[i]);
         if (ret)
            goto fail;
      }
   }
   return VK_SUCCESS;

fail:
   fprintf(stderr, "radv/sgpu: empty submission failed (%d, errno %d).\n", ret, errno);
   return VK_ERROR_DEVICE_LOST;
}

/* Everything one CS ioctl needs. */
struct radv_sgpu_submission {
   uint64_t stamp;

   struct drm_amdgpu_cs_chunk_ib *ibs;
   unsigned num_ibs;
   unsigned max_ibs;

   struct radv_sgpu_bo **bos;
   unsigned num_bos;
   unsigned max_bos;

   struct drm_amdgpu_cs_chunk_syncobj *waits;
   struct drm_amdgpu_cs_chunk_syncobj *signals;
   struct drm_amdgpu_cs_chunk_sem *wait_sems;
   struct drm_amdgpu_cs_chunk_sem *signal_sems;
   struct drm_amdgpu_bo_list_entry *bo_entries;
   struct drm_amdgpu_bo_list_in bo_list;
   struct drm_amdgpu_cs_chunk chunks[SGPU_MAX_SPLIT_IB * 8 + 3];
   unsigned num_chunks;
};

static void
radv_sgpu_submission_finish(struct radv_sgpu_submission *s)
{
   free(s->ibs);
   free(s->bos);
   free(s->waits);
   free(s->signals);
   free(s->wait_sems);
   free(s->signal_sems);
   free(s->bo_entries);
}

/* Every resident BO is listed on every submission, so duplicates are dropped with a per-BO
 * stamp instead of a list scan.  ws->lock held.
 */
static bool
radv_sgpu_submission_add_bo(struct radv_sgpu_submission *s, struct radeon_winsys_bo *_bo)
{
   struct radv_sgpu_bo *bo = radv_sgpu_bo(_bo);

   if (!bo || bo->submit_stamp == s->stamp)
      return true;
   bo->submit_stamp = s->stamp;

   if (s->num_bos == s->max_bos) {
      const unsigned new_max = MAX2(64, s->max_bos * 2);
      struct radv_sgpu_bo **bos = realloc(s->bos, new_max * sizeof(*bos));
      if (!bos)
         return false;
      s->bos = bos;
      s->max_bos = new_max;
   }

   s->bos[s->num_bos++] = bo;
   return true;
}

static bool
radv_sgpu_submission_add_cs(struct radv_sgpu_submission *s, struct ac_cmdbuf *base, int ring, uint32_t flags)
{
   struct radv_sgpu_cs *cs = radv_sgpu_cs(base);

   if (cs->status != VK_SUCCESS || !cs->finalized)
      return false;
   if (!cs->ib_bo)
      return true; /* empty stream */

   const unsigned parts = MAX2(cs->split_count, 1);
   if (s->num_ibs + parts > s->max_ibs) {
      const unsigned new_max = MAX2(s->num_ibs + parts, s->max_ibs * 2);
      struct drm_amdgpu_cs_chunk_ib *ibs = realloc(s->ibs, new_max * sizeof(*ibs));
      if (!ibs)
         return false;
      s->ibs = ibs;
      s->max_ibs = new_max;
   }

   for (unsigned part = 0; part < parts; part++) {
      const uint32_t off_dw = cs->split_count ? cs->split_off_dw[part] : 0;
      const uint32_t cdw = cs->split_count ? cs->split_cdw[part] : cs->ib_cdw;

      s->ibs[s->num_ibs++] = (struct drm_amdgpu_cs_chunk_ib){
         .ip_type = cs->ip_type,
         .ring = ring,
         .flags = flags,
         .va_start = cs->ib_bo->va + (uint64_t)off_dw * 4,
         .ib_bytes = cdw * 4,
      };
   }

   for (unsigned i = 0; i < cs->num_bos; i++) {
      if (!radv_sgpu_submission_add_bo(s, cs->bos[i]))
         return false;
   }
   return true;
}

static bool
radv_sgpu_submission_add_streams(struct radv_sgpu_submission *s, struct ac_cmdbuf **cs, unsigned count, int ring,
                                 uint32_t flags)
{
   for (unsigned i = 0; i < count; i++) {
      if (!radv_sgpu_submission_add_cs(s, cs[i], ring, flags))
         return false;
   }
   return true;
}

/* Builds the IB, BO list and syncobj chunks.  ws->lock held. */
static bool
radv_sgpu_submission_build(struct radv_sgpu_ctx *ctx, struct radv_sgpu_submission *s,
                           const struct radv_winsys_submit_info *submit, uint32_t wait_count,
                           const struct vk_sync_wait *waits, uint32_t signal_count,
                           const struct vk_sync_signal *signals, uint32_t queue_syncobj, bool queue_syncobj_wait)
{
   static uint64_t stamp;
   struct radv_sgpu_winsys *ws = ctx->ws;
   const int ring = submit->queue_index;
   bool valid = true;

   s->stamp = ++stamp;

   list_for_each_entry (struct radv_sgpu_bo, bo, &ws->resident_bos, resident_link) {
      if (!radv_sgpu_submission_add_bo(s, &bo->base))
         return false;
   }

   /* The vendor driver asks the kernel for a memory sync ahead of every IB except the
    * register-shadowed command IBs.
    */
   const uint32_t flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC | (submit->secure ? AMDGPU_IB_FLAGS_SECURE : 0);
   const uint32_t cmd_flags = (submit->uses_shadow_regs ? 0 : AMDGPU_IB_FLAG_EMIT_MEM_SYNC) |
                              (submit->secure ? AMDGPU_IB_FLAGS_SECURE : 0);

   for (unsigned i = 0; i < submit->cs_count; i++) {
      if (!radv_sgpu_submission_add_streams(s, i ? submit->continue_preamble_cs : submit->initial_preamble_cs,
                                            i ? submit->continue_preamble_count : submit->initial_preamble_count,
                                            ring, flags) ||
          !radv_sgpu_submission_add_cs(s, submit->cs_array[i], ring, cmd_flags))
         return false;
   }
   if (!radv_sgpu_submission_add_streams(s, submit->postamble_cs, submit->postamble_count, ring, 0))
      return false;
   if (!s->num_ibs || s->num_ibs > SGPU_MAX_SPLIT_IB * 8)
      return false;

   for (unsigned i = 0; i < s->num_ibs; i++) {
      s->chunks[s->num_chunks++] = (struct drm_amdgpu_cs_chunk){
         .chunk_id = AMDGPU_CHUNK_ID_IB,
         .length_dw = sizeof(s->ibs[i]) / 4,
         .chunk_data = (uintptr_t)&s->ibs[i],
      };
   }

   /* Syncobjs: the application's, plus the route's queue syncobj (always signalled, waited
    * after an empty submission folded waits into it).
    */
   const bool timeline = radv_sgpu_use_timeline(ws);
   const unsigned max_waits = wait_count + 1, max_signals = signal_count + 1;
   unsigned num_waits = 0, num_signals = 0;

   if (timeline) {
      s->waits = calloc(max_waits, sizeof(*s->waits));
      s->signals = calloc(max_signals, sizeof(*s->signals));
      if (!s->waits || !s->signals)
         return false;

      for (uint32_t i = 0; i < wait_count; i++) {
         const uint32_t handle = radv_sgpu_sync_handle(ws, waits[i].sync, waits[i].wait_value, &valid);
         if (handle) {
            const bool tl = waits[i].sync->flags & VK_SYNC_IS_TIMELINE;
            s->waits[num_waits++] = (struct drm_amdgpu_cs_chunk_syncobj){
               .handle = handle,
               .flags = tl ? DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT : 0,
               .point = tl ? waits[i].wait_value : 0,
            };
         }
      }
      if (queue_syncobj_wait)
         s->waits[num_waits++] = (struct drm_amdgpu_cs_chunk_syncobj){.handle = queue_syncobj};

      for (uint32_t i = 0; i < signal_count; i++) {
         const uint32_t handle = radv_sgpu_sync_handle(ws, signals[i].sync, signals[i].signal_value, &valid);
         if (handle) {
            const bool tl = signals[i].sync->flags & VK_SYNC_IS_TIMELINE;
            s->signals[num_signals++] = (struct drm_amdgpu_cs_chunk_syncobj){
               .handle = handle,
               .flags = tl ? DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT : 0,
               .point = tl ? signals[i].signal_value : 0,
            };
         }
      }
      s->signals[num_signals++] = (struct drm_amdgpu_cs_chunk_syncobj){.handle = queue_syncobj};

      if (num_waits) {
         s->chunks[s->num_chunks++] = (struct drm_amdgpu_cs_chunk){
            .chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT,
            .length_dw = num_waits * sizeof(*s->waits) / 4,
            .chunk_data = (uintptr_t)s->waits,
         };
      }
      s->chunks[s->num_chunks++] = (struct drm_amdgpu_cs_chunk){
         .chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL,
         .length_dw = num_signals * sizeof(*s->signals) / 4,
         .chunk_data = (uintptr_t)s->signals,
      };
   } else {
      s->wait_sems = calloc(max_waits, sizeof(*s->wait_sems));
      s->signal_sems = calloc(max_signals, sizeof(*s->signal_sems));
      if (!s->wait_sems || !s->signal_sems)
         return false;

      for (uint32_t i = 0; i < wait_count; i++) {
         const uint32_t handle = radv_sgpu_sync_handle(ws, waits[i].sync, waits[i].wait_value, &valid);
         if (handle)
            s->wait_sems[num_waits++].handle = handle;
      }
      if (queue_syncobj_wait)
         s->wait_sems[num_waits++].handle = queue_syncobj;

      for (uint32_t i = 0; i < signal_count; i++) {
         const uint32_t handle = radv_sgpu_sync_handle(ws, signals[i].sync, signals[i].signal_value, &valid);
         if (handle)
            s->signal_sems[num_signals++].handle = handle;
      }
      s->signal_sems[num_signals++].handle = queue_syncobj;

      if (num_waits) {
         s->chunks[s->num_chunks++] = (struct drm_amdgpu_cs_chunk){
            .chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_IN,
            .length_dw = num_waits * sizeof(*s->wait_sems) / 4,
            .chunk_data = (uintptr_t)s->wait_sems,
         };
      }
      s->chunks[s->num_chunks++] = (struct drm_amdgpu_cs_chunk){
         .chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_OUT,
         .length_dw = num_signals * sizeof(*s->signal_sems) / 4,
         .chunk_data = (uintptr_t)s->signal_sems,
      };
   }
   if (!valid)
      return false;

   /* Local (VM always valid) BOs are known to the kernel already; the others must be listed. */
   s->bo_entries = calloc(MAX2(s->num_bos, 1), sizeof(*s->bo_entries));
   if (!s->bo_entries)
      return false;

   unsigned num_entries = 0;
   for (unsigned i = 0; i < s->num_bos; i++) {
      struct radv_sgpu_bo *bo = s->bos[i];
      if (bo->is_local)
         continue;
      if (!radv_sgpu_bo_get_kms_handle_locked(bo, &s->bo_entries[num_entries].bo_handle))
         return false;
      s->bo_entries[num_entries++].bo_priority = bo->priority;
   }

   s->bo_list = (struct drm_amdgpu_bo_list_in){
      .operation = ~0u,
      .list_handle = ~0u,
      .bo_number = num_entries,
      .bo_info_size = sizeof(s->bo_entries[0]),
      .bo_info_ptr = (uintptr_t)s->bo_entries,
   };
   s->chunks[s->num_chunks++] = (struct drm_amdgpu_cs_chunk){
      .chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES,
      .length_dw = sizeof(s->bo_list) / 4,
      .chunk_data = (uintptr_t)&s->bo_list,
   };
   return true;
}

/* Takes a reference on every BO of the submission until its fence retires.  ws->lock held. */
static struct radv_sgpu_retention *
radv_sgpu_retain(struct radv_sgpu_submission *s)
{
   struct radv_sgpu_retention *r = calloc(1, sizeof(*r));
   if (!r)
      return NULL;

   r->bos = s->bos;
   r->num_bos = s->num_bos;
   s->bos = NULL;
   for (unsigned i = 0; i < r->num_bos; i++)
      r->bos[i]->pending_submits++;
   return r;
}

static VkResult
radv_sgpu_cs_submit(struct radeon_winsys_ctx *_ctx, const struct radv_winsys_submit_info *submit,
                    uint32_t wait_count, const struct vk_sync_wait *waits, uint32_t signal_count,
                    const struct vk_sync_signal *signals)
{
   struct radv_sgpu_ctx *ctx = (struct radv_sgpu_ctx *)_ctx;
   struct radv_sgpu_winsys *ws = ctx->ws;
   const enum amd_ip_type ip = submit->ip_type;
   const int ring = submit->queue_index;
   VkResult result;

   if (!radv_sgpu_route_valid(ip, ring))
      return VK_ERROR_UNKNOWN;
   if (radv_sgpu_is_lost(ctx, ip, ring))
      return VK_ERROR_DEVICE_LOST;

   simple_mtx_lock(&ctx->lock);

   if (!submit->cs_count) {
      result = (wait_count || signal_count)
                  ? radv_sgpu_submit_zero_locked(ctx, ip, ring, wait_count, waits, signal_count, signals)
                  : VK_SUCCESS;
      simple_mtx_unlock(&ctx->lock);
      return result;
   }

   const uint32_t queue_syncobj = radv_sgpu_queue_syncobj(ctx, ip, ring);
   if (!queue_syncobj) {
      simple_mtx_unlock(&ctx->lock);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   const bool queue_syncobj_wait = ctx->queue_syncobj_wait[ip][ring];

   radv_sgpu_retire(ctx);

   struct radv_sgpu_submission s = {0};
   struct radv_sgpu_retention *retention = NULL;

   simple_mtx_lock(&ws->lock);
   if (radv_sgpu_submission_build(ctx, &s, submit, wait_count, waits, signal_count, signals, queue_syncobj,
                                  queue_syncobj_wait))
      retention = radv_sgpu_retain(&s);
   simple_mtx_unlock(&ws->lock);

   if (!retention) {
      radv_sgpu_submission_finish(&s);
      simple_mtx_unlock(&ctx->lock);
      fprintf(stderr, "radv/sgpu: could not build the submission.\n");
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   uint64_t seq_no = 0;
   const int ret = amdgpu_cs_submit_raw(ws->device, ctx->context, NULL, s.num_chunks, s.chunks, &seq_no);
   radv_sgpu_submission_finish(&s);

   if (ret) {
      if (ret == -ECANCELED)
         radv_sgpu_mark_lost(ctx, ip, ring, seq_no);

      simple_mtx_lock(&ws->lock);
      radv_sgpu_retention_release_locked(ws, retention);
      simple_mtx_unlock(&ws->lock);
      simple_mtx_unlock(&ctx->lock);

      if (ret == -ENOMEM)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      fprintf(stderr, "radv/sgpu: the kernel rejected the submission (%d).\n", ret);
      return (ret == -ECANCELED || ret == -ENODATA || ret == -ETIME) ? VK_ERROR_DEVICE_LOST : VK_ERROR_UNKNOWN;
   }

   const struct amdgpu_cs_fence fence = {
      .context = ctx->context,
      .ip_type = ip,
      .ring = ring,
      .fence = seq_no,
   };
   retention->fence = fence;
   ctx->last_fence[ip][ring] = fence;
   ctx->queue_syncobj_wait[ip][ring] = false;

   simple_mtx_lock(&ws->lock);
   retention->next = ctx->retentions;
   ctx->retentions = retention;
   simple_mtx_unlock(&ws->lock);

   simple_mtx_unlock(&ctx->lock);
   return VK_SUCCESS;
}

static uint32_t
radv_sgpu_ctx_priority(enum radeon_ctx_priority priority)
{
   switch (priority) {
   case RADEON_CTX_PRIORITY_REALTIME:
      return AMDGPU_CTX_PRIORITY_VERY_HIGH;
   case RADEON_CTX_PRIORITY_HIGH:
      return AMDGPU_CTX_PRIORITY_HIGH;
   case RADEON_CTX_PRIORITY_LOW:
      return AMDGPU_CTX_PRIORITY_LOW;
   default:
      return AMDGPU_CTX_PRIORITY_NORMAL;
   }
}

static VkResult
radv_sgpu_ctx_create(struct radeon_winsys *_ws, enum radeon_ctx_priority priority, struct radeon_winsys_ctx **out)
{
   struct radv_sgpu_winsys *ws = radv_sgpu_winsys(_ws);

   struct radv_sgpu_ctx *ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   if (amdgpu_cs_ctx_create2(ws->device, radv_sgpu_ctx_priority(priority), &ctx->context)) {
      free(ctx);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   ctx->ws = ws;
   simple_mtx_init(&ctx->lock, mtx_plain);
   *out = (struct radeon_winsys_ctx *)ctx;
   return VK_SUCCESS;
}

static void
radv_sgpu_ctx_destroy(struct radeon_winsys_ctx *_ctx)
{
   struct radv_sgpu_ctx *ctx = (struct radv_sgpu_ctx *)_ctx;
   struct radv_sgpu_winsys *ws = ctx->ws;
   uint64_t timeout_ns = 2000000000ull;

   /* Fences reference the context, so every submission must retire before it is freed.  One
    * that does not in time keeps its BOs for the rest of the process rather than releasing
    * them under a possibly running IB.
    */
   simple_mtx_lock(&ws->lock);
   while (ctx->retentions) {
      struct radv_sgpu_retention *r = ctx->retentions;
      uint32_t expired = 0;
      const int ret = radv_sgpu_query_fence(ctx, &r->fence, timeout_ns, &expired);

      ctx->retentions = r->next;
      if ((!ret && expired) || ret == -ECANCELED) {
         radv_sgpu_retention_release_locked(ws, r);
      } else {
         free(r->bos);
         free(r);
         timeout_ns = 0;
      }
   }
   simple_mtx_unlock(&ws->lock);

   for (unsigned ip = 0; ip < SGPU_NUM_IPS; ip++) {
      for (unsigned ring = 0; ring < SGPU_MAX_RINGS; ring++) {
         if (ctx->queue_syncobj[ip][ring])
            ws->sync_provider->destroy(ws->sync_provider, ctx->queue_syncobj[ip][ring]);
      }
   }

   amdgpu_cs_ctx_free(ctx->context);
   simple_mtx_destroy(&ctx->lock);
   free(ctx);
}

static bool
radv_sgpu_ctx_wait_idle(struct radeon_winsys_ctx *_ctx, enum amd_ip_type ip, int ring)
{
   struct radv_sgpu_ctx *ctx = (struct radv_sgpu_ctx *)_ctx;

   if (!radv_sgpu_route_valid(ip, ring) || radv_sgpu_is_lost(ctx, ip, ring))
      return false;

   simple_mtx_lock(&ctx->lock);
   struct amdgpu_cs_fence fence = ctx->last_fence[ip][ring];
   simple_mtx_unlock(&ctx->lock);

   if (fence.context) {
      uint32_t expired = 0;
      if (radv_sgpu_query_fence(ctx, &fence, 1000000000ull, &expired) || !expired)
         return false;
   }

   radv_sgpu_retire(ctx);
   return true;
}

static int
radv_sgpu_ctx_set_pstate(struct radeon_winsys_ctx *ctx, uint32_t pstate, uint64_t timeout)
{
   return 0;
}

static enum radeon_bo_domain
radv_sgpu_cs_domain(const struct radeon_winsys *ws)
{
   return RADEON_DOMAIN_GTT;
}

void
radv_sgpu_cs_init_functions(struct radv_sgpu_winsys *ws)
{
   ws->base.ctx_create = radv_sgpu_ctx_create;
   ws->base.ctx_destroy = radv_sgpu_ctx_destroy;
   ws->base.ctx_wait_idle = radv_sgpu_ctx_wait_idle;
   ws->base.ctx_set_pstate = radv_sgpu_ctx_set_pstate;
   ws->base.cs_domain = radv_sgpu_cs_domain;
   ws->base.cs_create = radv_sgpu_cs_create;
   ws->base.cs_destroy = radv_sgpu_cs_destroy;
   ws->base.cs_reset = radv_sgpu_cs_reset;
   ws->base.cs_chain = radv_sgpu_cs_chain;
   ws->base.cs_unchain = radv_sgpu_cs_unchain;
   ws->base.cs_finalize = radv_sgpu_cs_finalize;
   ws->base.cs_grow = radv_sgpu_cs_grow;
   ws->base.cs_submit = radv_sgpu_cs_submit;
   ws->base.cs_add_buffer = radv_sgpu_cs_add_buffer;
   ws->base.cs_execute_secondary = radv_sgpu_cs_execute_secondary;
   ws->base.cs_execute_ib = radv_sgpu_cs_execute_ib;
   ws->base.cs_chain_dgc_ib = radv_sgpu_cs_chain_dgc_ib;
   ws->base.cs_dump = radv_sgpu_cs_dump;
   ws->base.cs_annotate = radv_sgpu_cs_annotate;
   ws->base.cs_pad = radv_sgpu_cs_pad;
}
