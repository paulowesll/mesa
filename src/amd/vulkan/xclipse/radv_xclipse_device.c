/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Xclipse device identity and properties.
 *
 * The kernel driver is Samsung's, so the device is queried through the vendor libdrm
 * (libdrm_sgpu.so, same ABI as libdrm_amdgpu), loaded with dlopen so RADV does not link
 * against it.  The result is a radeon_info that selects the Phoenix/GFX11 profile plus
 * the Xclipse facts measured on hardware.
 */

#include "radv_xclipse.h"

#include <amdgpu.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>

#include "drm-uapi/amdgpu_drm.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "winsys/amdgpu/radv_amdgpu_winsys_public.h"
#include "ac_gpu_info.h"
#include "radv_radeon_winsys.h"
#include "vk_drm_syncobj.h"

/* Topology of Xclipse 960 (fixed; the vendor query does not describe it completely). */
#define XCLIPSE_NUM_SE              2u
#define XCLIPSE_SA_PER_SE           2u
#define XCLIPSE_CU_PER_SA           4u
#define XCLIPSE_RB_PER_SE           4u
#define XCLIPSE_MAX_WAVES_PER_SIMD  16u

struct radv_xclipse_api {
   int (*device_initialize)(int fd, uint32_t *major, uint32_t *minor, amdgpu_device_handle *device);
   int (*device_deinitialize)(amdgpu_device_handle device);
   int (*query_gpu_info)(amdgpu_device_handle device, struct amdgpu_gpu_info *info);
   int (*query_info)(amdgpu_device_handle device, unsigned info_id, unsigned size, void *value);
   int (*query_hw_ip_count)(amdgpu_device_handle device, unsigned type, uint32_t *count);
   int (*query_hw_ip_info)(amdgpu_device_handle device, unsigned type, unsigned ip_instance,
                           struct drm_amdgpu_info_hw_ip *info);
   const char *(*get_marketing_name)(amdgpu_device_handle device);
};

static bool
radv_xclipse_load_api(void *lib, struct radv_xclipse_api *api)
{
#define LOAD(field, symbol)                                                                        \
   if (!(*(void **)&api->field = dlsym(lib, symbol)))                                              \
      return false;

   LOAD(device_initialize, "amdgpu_device_initialize");
   LOAD(device_deinitialize, "amdgpu_device_deinitialize");
   LOAD(query_gpu_info, "amdgpu_query_gpu_info");
   LOAD(query_info, "amdgpu_query_info");
   LOAD(query_hw_ip_count, "amdgpu_query_hw_ip_count");
   LOAD(query_hw_ip_info, "amdgpu_query_hw_ip_info");
   LOAD(get_marketing_name, "amdgpu_get_marketing_name");
#undef LOAD
   return true;
}

/* RADV_SGPU_VENDOR_LIB, then a copy next to this driver (driver packages ship one), then the
 * system copy.
 */
static void *
radv_xclipse_open_vendor_library(void)
{
   char sibling[4096] = "";
   Dl_info dl;

   if (dladdr((void *)radv_xclipse_open_vendor_library, &dl) && dl.dli_fname) {
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
   struct radv_xclipse_api api;
   struct drm_amdgpu_info_device dev = {0};
   struct amdgpu_gpu_info gpu = {0};
   struct drm_amdgpu_memory_info mem = {0};
   amdgpu_device_handle device = NULL;
   uint32_t drm_major = 0, drm_minor = 0;
   VkResult result = VK_SUCCESS;

   memset(out, 0, sizeof(*out));

   void *lib = radv_xclipse_open_vendor_library();
   if (!lib || !radv_xclipse_load_api(lib, &api)) {
      fprintf(stderr, "radv/xclipse: could not load libdrm_sgpu.so.\n");
      if (lib)
         dlclose(lib);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   if (api.device_initialize(fd, &drm_major, &drm_minor, &device) || !device) {
      fprintf(stderr, "radv/xclipse: vendor device initialization failed.\n");
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto finish;
   }

   if (api.query_info(device, AMDGPU_INFO_DEV_INFO, sizeof(dev), &dev) || api.query_gpu_info(device, &gpu) ||
       api.query_info(device, AMDGPU_INFO_MEMORY, sizeof(mem), &mem)) {
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

      if (api.query_hw_ip_info(device, ip_type, 0, &ip) || !ip.available_rings ||
          !ac_fill_hw_ip_info(info, &dev, ip_type, &ip))
         continue;
      if (!api.query_hw_ip_count(device, ip_type, &count))
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
   ac_fill_compiler_info(info, &dev, false);

   /* The kernel rejects jobs with more than one IB per submission. */
   for (unsigned ip_type = 0; ip_type < AMD_NUM_IP_TYPES; ip_type++)
      info->max_submitted_ibs[ip_type] = info->ip[ip_type].num_queues ? 1 : 0;

   /* None of the SET_*_REG_PAIRS packet forms work on Xclipse. */
   info->has_set_context_pairs = false;
   info->has_set_context_pairs_packed = false;
   info->has_set_sh_pairs = false;
   info->has_set_sh_pairs_packed = false;
   info->has_set_uconfig_pairs = false;

   /* Not provided by the vendor kernel driver. */
   info->has_sparse = false;
   info->has_sparse_image_3d = false;
   info->has_sparse_image_standard_3d = false;
   info->has_sparse_unaligned_mip_size = false;
   info->has_userptr = false;
   info->kernel_has_modifiers = false;
   info->has_fence_to_handle = false;

   out->syncobj_sync_type = vk_drm_syncobj_get_type(fd);
   info->has_syncobj = out->syncobj_sync_type.features != 0;
   info->has_timeline_syncobj = !!(out->syncobj_sync_type.features & VK_SYNC_FEATURE_TIMELINE);
   out->global_priority_mask = BITFIELD_BIT(RADEON_CTX_PRIORITY_LOW);

   const char *name = api.get_marketing_name(device);
   snprintf(info->marketing_name, sizeof(info->marketing_name), "%s", name ? name : "Samsung Xclipse");

finish:
   if (device)
      api.device_deinitialize(device);
   dlclose(lib);
   return result;
}
