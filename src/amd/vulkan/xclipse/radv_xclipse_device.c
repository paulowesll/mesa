/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Xclipse device identity and properties.
 *
 * The kernel driver is Samsung's, but it takes the amdgpu UAPI, so the device is queried
 * through libdrm_amdgpu.  The result is a radeon_info that selects the Phoenix/GFX11 profile
 * plus the Xclipse facts measured on hardware.
 */

#include "radv_xclipse.h"

#include <amdgpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>

#include "drm-uapi/amdgpu_drm.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "winsys/amdgpu/radv_amdgpu_winsys_public.h"
#include "winsys/sgpu/radv_sgpu_winsys_public.h"
#include "ac_gpu_info.h"
#include "radv_radeon_winsys.h"
#include "vk_drm_syncobj.h"
#include "vk_extensions.h"
#include "vk_physical_device_features.h"

/* Topology of Xclipse 960 (fixed; the vendor query does not describe it completely). */
#define XCLIPSE_NUM_SE              2u
#define XCLIPSE_SA_PER_SE           2u
#define XCLIPSE_CU_PER_SA           4u
#define XCLIPSE_RB_PER_SE           4u
#define XCLIPSE_MAX_WAVES_PER_SIMD  16u

/* The marketing name, as the vendor libdrm derives it from the chip revision. */
static void
radv_xclipse_marketing_name(const struct drm_amdgpu_info_device *dev, char *name, size_t size)
{
   static const unsigned series[] = {120, 320, 520, 720, 0, 920};
   const uint32_t rev = dev->chip_rev;
   const uint32_t s = (((rev >> 16) & 0xff) - 0x10) & 0xff;
   const uint32_t idx = (s >> 4) | ((s & 0xf) << 28);

   if (dev->device_id == 0x73a0 && idx < ARRAY_SIZE(series) && series[idx])
      snprintf(name, size, "Samsung Xclipse %u", (rev >> 24) * 10 + series[idx]);
   else
      snprintf(name, size, "Samsung Xclipse");
}

static void
radv_xclipse_init_identity(struct radeon_info *info, const struct drm_amdgpu_info_device *dev)
{
   info->is_xclipse = true;
   info->vendor_id = RADV_XCLIPSE_VENDOR_ID;
   info->pci_id = dev->device_id;
   info->pci_rev_id = dev->pci_rev;
   info->native_family_id = dev->family; /* 0x93 on Xclipse 960 */
   info->chip_rev = dev->chip_rev;
   info->pci.valid = false;

   /* The common code is driven through the Phoenix profile: GFX11 ISA and the GFX11 surface
    * model.  addrlib only knows desktop family/revision ranges, so report Phoenix's.
    */
   info->family = CHIP_PHOENIX;
   info->gfx_level = GFX11;
   info->family_id = AMDGPU_FAMILY_GC_11_0_1;
   info->chip_external_rev = 0x01;
}

static void
radv_xclipse_init_topology(struct radeon_info *info)
{
   STATIC_ASSERT(XCLIPSE_NUM_SE <= AMD_MAX_SE && XCLIPSE_SA_PER_SE <= AMD_MAX_SA_PER_SE);

   info->num_se = XCLIPSE_NUM_SE;
   info->max_se = XCLIPSE_NUM_SE;
   info->max_sa_per_se = XCLIPSE_SA_PER_SE;
   info->num_cu_per_sh = XCLIPSE_CU_PER_SA;
   info->min_good_cu_per_sa = XCLIPSE_CU_PER_SA;
   info->max_good_cu_per_sa = XCLIPSE_CU_PER_SA;
   info->num_cu = XCLIPSE_NUM_SE * XCLIPSE_SA_PER_SE * XCLIPSE_CU_PER_SA;
   info->spi_cu_en = BITFIELD_MASK(XCLIPSE_CU_PER_SA);
   info->spi_cu_en_has_effect = true;

   memset(info->cu_mask, 0, sizeof(info->cu_mask));
   for (unsigned se = 0; se < XCLIPSE_NUM_SE; se++) {
      for (unsigned sa = 0; sa < XCLIPSE_SA_PER_SE; sa++)
         info->cu_mask[se][sa] = BITFIELD_MASK(XCLIPSE_CU_PER_SA);
   }

   /* Device-wide count, disabled backends included. */
   info->max_render_backends = XCLIPSE_NUM_SE * XCLIPSE_RB_PER_SE;
   info->compiler_info.max_waves_per_simd = XCLIPSE_MAX_WAVES_PER_SIMD;
}

VkResult
radv_xclipse_query_info(int fd, struct radeon_winsys_info *out)
{
   struct radeon_info *info = &out->base;
   struct drm_amdgpu_info_device dev = {0};
   struct amdgpu_gpu_info gpu = {0};
   struct drm_amdgpu_memory_info mem = {0};
   amdgpu_device_handle device = NULL;
   uint32_t drm_major = 0, drm_minor = 0;
   VkResult result = VK_SUCCESS;

   memset(out, 0, sizeof(*out));

   if (amdgpu_device_initialize(fd, &drm_major, &drm_minor, &device) || !device) {
      fprintf(stderr, "radv/xclipse: device initialization failed.\n");
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto finish;
   }

   if (amdgpu_query_info(device, AMDGPU_INFO_DEV_INFO, sizeof(dev), &dev) || amdgpu_query_gpu_info(device, &gpu) ||
       amdgpu_query_info(device, AMDGPU_INFO_MEMORY, sizeof(mem), &mem)) {
      fprintf(stderr, "radv/xclipse: device query failed.\n");
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto finish;
   }

   /* The vendor driver reports the topology in amdgpu_gpu_info rather than in the device info. */
   dev.num_shader_engines = gpu.num_shader_engines;
   dev.num_shader_arrays_per_engine = gpu.num_shader_arrays_per_engine;
   dev.gpu_counter_freq = gpu.gpu_counter_freq;
   dev.max_engine_clock = gpu.max_engine_clk;
   dev.max_memory_clock = gpu.max_memory_clk;
   dev.cu_active_number = gpu.cu_active_number;
   dev.enabled_rb_pipes_mask = gpu.enabled_rb_pipes_mask;
   dev.num_rb_pipes = gpu.rb_pipes;
   dev.vram_type = gpu.vram_type;
   dev.vram_bit_width = gpu.vram_bit_width;
   dev.cu_ao_mask = gpu.cu_ao_mask;
   memcpy(dev.cu_bitmap, gpu.cu_bitmap, sizeof(dev.cu_bitmap));

   radv_xclipse_init_identity(info, &dev);

   drmVersionPtr version = drmGetVersion(fd);
   if (version) {
      info->drm_major = version->version_major;
      info->drm_minor = version->version_minor;
      info->drm_patchlevel = version->version_patchlevel;
      drmFreeVersion(version);
   } else {
      info->drm_major = drm_major;
      info->drm_minor = drm_minor;
   }

   /* The vendor driver uses the ordinary VA manager, not the desktop AMDGPU high 32-bit
    * window, so 32-bit addresses live in the low 4 GiB.
    */
   info->address32_hi = 0;

   for (unsigned ip_type = AMD_IP_GFX; ip_type <= AMD_IP_SDMA; ip_type++) {
      struct drm_amdgpu_info_hw_ip ip = {0};
      uint32_t count = 0;

      if (amdgpu_query_hw_ip_info(device, ip_type, 0, &ip) || !ip.available_rings ||
          !ac_fill_hw_ip_info(info, &dev, ip_type, &ip))
         continue;
      if (!amdgpu_query_hw_ip_count(device, ip_type, &count))
         info->ip[ip_type].num_instances = count;
   }

   if (!info->ip[AMD_IP_GFX].num_queues) {
      fprintf(stderr, "radv/xclipse: the device exposes no graphics ring.\n");
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto finish;
   }

   ac_fill_memory_info(info, &dev, &mem);
   ac_fill_hw_info(info, &dev);

   /* One unified GTT heap, no VRAM. */
   info->vram_size_kb = 0;
   info->vram_vis_size_kb = 0;
   info->gart_size_kb = DIV_ROUND_UP(mem.gtt.total_heap_size, 1024);
   info->max_heap_size_kb = info->gart_size_kb;
   info->has_dedicated_vram = false;
   info->all_vram_visible = false;

   radv_xclipse_init_topology(info);

   ac_fill_tiling_info(info, &gpu);
   ac_fill_feature_info(info, &dev);
   ac_fill_bug_info(info);
   ac_fill_tess_info(info);
   /* With distributed tessellation, only the last wave of each HS workgroup produced output.
    * The vendor driver runs it undistributed.
    */
   info->has_distributed_tess = false;
   ac_fill_compiler_info(info, &dev, false);
   info->compiler_info.is_xclipse = true;
   /* Parameters are exported GFX10.3-style: there is no attribute ring, whose registers do not
    * exist on MGFX4.  NGG passthrough without the GS allocation message is not supported either.
    */
   info->compiler_info.has_attr_ring = false;
   info->compiler_info.has_attr_ring_wait_bug = false;
   info->compiler_info.has_ngg_passthru_no_msg = false;

   /* Fields that ac_query_gpu_info derives for amdgpu devices. */
   info->max_scratch_waves = MAX2(32 * info->max_good_cu_per_sa * info->max_sa_per_se * info->num_se, 32);
   info->has_scratch_base_registers = true;
   info->se_tile_repeat = 32 * info->max_se;
   info->instr_prefetch_distance = 3;

   /* TMPRING follows GFX10: WAVESIZE in 1 KiB units and WAVES for the whole device, not per SE.
    * With the GFX11 encoding, waves were placed 4x further apart than the scratch buffer allows.
    */
   info->scratch_wavesize_granularity_shift = 10;
   info->scratch_wavesize_granularity = BITFIELD_BIT(info->scratch_wavesize_granularity_shift);

   /* The kernel rejects jobs with more than one IB per submission. */
   for (unsigned ip_type = 0; ip_type < AMD_NUM_IP_TYPES; ip_type++)
      info->max_submitted_ibs[ip_type] = info->ip[ip_type].num_queues ? 1 : 0;

   /* None of the SET_*_REG_PAIRS packet forms work on Xclipse. */
   info->has_set_context_pairs = false;
   info->has_set_context_pairs_packed = false;
   info->has_set_sh_pairs = false;
   info->has_set_sh_pairs_packed = false;
   info->has_set_uconfig_pairs = false;

   /* The vendor kernel honours AMDGPU_GEM_CREATE_VM_ALWAYS_VALID (its own driver uses it). */
   info->has_vm_always_valid = true;

   /* Sparse resources map PRT ranges like the vendor driver (see the SGPU winsys). */
   info->has_sparse = true;
   info->has_sparse_image_3d = true;
   info->has_sparse_image_standard_3d = true;
   info->has_sparse_unaligned_mip_size = true;

   /* Scalar loads from unmapped PRT pages hang the GPU, as on AMD (has_smem_with_null_prt_bug),
    * and the vendor kernel does not report the address bit of the workaround.  The VA range ends
    * at 1 << 47; the SGPU winsys keeps the upper half for the high aliases of sparse buffers.
    */
   info->address_prt_wa_control_bit = 46;

   /* Not provided by the vendor kernel driver. */
   info->has_userptr = false;
   info->kernel_has_modifiers = false;
   info->has_fence_to_handle = false;

   out->syncobj_sync_type = vk_drm_syncobj_get_type(fd);
   info->has_syncobj = out->syncobj_sync_type.features != 0;
   info->has_timeline_syncobj = !!(out->syncobj_sync_type.features & VK_SYNC_FEATURE_TIMELINE);
   out->global_priority_mask = BITFIELD_BIT(RADEON_CTX_PRIORITY_LOW);

   radv_xclipse_marketing_name(&dev, info->marketing_name, sizeof(info->marketing_name));

finish:
   if (device)
      amdgpu_device_deinitialize(device);
   return result;
}

uint64_t
radv_xclipse_available_memory(void)
{
   static uint64_t available, read_ns;
   const uint64_t now = os_time_get_nano();

   if (!p_atomic_read(&read_ns) || now - p_atomic_read(&read_ns) > 200000000ull) {
      FILE *f = fopen("/proc/meminfo", "r");
      unsigned long long kb = 0;
      char line[128];

      while (f && fgets(line, sizeof(line), f)) {
         if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)
            break;
      }
      if (f)
         fclose(f);
      if (kb) {
         p_atomic_set(&available, kb * 1024ull);
         p_atomic_set(&read_ns, now);
      }
   }

   return p_atomic_read(&available);
}

void
radv_xclipse_fixup_extensions(struct vk_device_extension_table *ext)
{
   /* Generated commands never run. */
   ext->EXT_device_generated_commands = false;

   /* The vendor kernel driver has no format modifiers. */
   ext->EXT_image_drm_format_modifier = false;
}

void
radv_xclipse_fixup_features(struct vk_features *features)
{
   /* fp64 arithmetic is wrong by orders of magnitude. */
   features->shaderFloat64 = false;
   features->shaderFmaFloat64 = false;
   features->shaderBufferFloat64Atomics = false;
   features->shaderBufferFloat64AtomicAdd = false;
   features->shaderBufferFloat64AtomicMinMax = false;
   features->shaderSharedFloat64Atomics = false;
   features->shaderSharedFloat64AtomicAdd = false;
   features->shaderSharedFloat64AtomicMinMax = false;

   features->deviceGeneratedCommands = false;
   features->dynamicGeneratedPipelineLayout = false;

   /* No ray tracing (radv_enable_rt). */
   features->rayQuery = false;

   /* Unmapped sparse texels are reported resident; the vendor driver does not expose it either. */
   features->shaderResourceResidency = false;
}
