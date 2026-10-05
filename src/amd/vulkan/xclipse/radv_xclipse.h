/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Samsung Xclipse (SGPU) support.
 *
 * Xclipse 960 is an RDNA-derived GPU driven by Samsung's own kernel driver, which takes the
 * amdgpu UAPI.  It is not a complete match for any AMD
 * generation: RADV drives it through the Phoenix/GFX11 profile (ISA, addressing), while
 * several blocks behave like GFX10.3.  Every deviation from the GFX11 profile is described
 * where it is applied, and none of this code runs for other devices.
 */

#ifndef RADV_XCLIPSE_H
#define RADV_XCLIPSE_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

struct ac_pm4_state;
struct radv_device;
struct radeon_winsys_info;
struct vk_device_extension_table;
struct vk_features;

/* PCI-style vendor ID reported for the device. */
#define RADV_XCLIPSE_VENDOR_ID 0x144d

/* Queries the device through the vendor library; submits no work. */
VkResult radv_xclipse_query_info(int fd, struct radeon_winsys_info *info);

/* Hide what does not work on Xclipse and its vendor driver does not expose either. */
void radv_xclipse_fixup_extensions(struct vk_device_extension_table *ext);
void radv_xclipse_fixup_features(struct vk_features *features);

/* BC4-BC7 emulation state (radv_meta_decode_bc). */
VkResult radv_xclipse_bc_init(struct radv_device *device);
void radv_xclipse_bc_finish(struct radv_device *device);

/* Memory Android can still hand out (MemAvailable), re-read at most every 200 ms; 0 if unknown. */
uint64_t radv_xclipse_available_memory(void);

/* Adds the register state a graphics preamble starts with. */
void radv_xclipse_add_native_state(struct ac_pm4_state *pm4);

/* Brackets a shader compilation (begin true, then false) with the vendor kernel's CPU hint. */
#if RADV_XCLIPSE
void radv_xclipse_compile_hint(struct radv_device *device, bool begin);
#else
static inline void
radv_xclipse_compile_hint(struct radv_device *device, bool begin)
{
}
#endif

#endif /* RADV_XCLIPSE_H */
