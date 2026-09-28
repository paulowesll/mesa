/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Samsung Xclipse (SGPU) support.
 *
 * Xclipse 960 is an RDNA-derived GPU driven by Samsung's own kernel driver and userspace
 * libdrm (libdrm_sgpu.so, loaded at runtime).  It is not a complete match for any AMD
 * generation: RADV drives it through the Phoenix/GFX11 profile (ISA, addressing), while
 * several blocks behave like GFX10.3.  Every deviation from the GFX11 profile is described
 * where it is applied, and none of this code runs for other devices.
 */

#ifndef RADV_XCLIPSE_H
#define RADV_XCLIPSE_H

#include <vulkan/vulkan_core.h>

struct radeon_winsys_info;

/* PCI-style vendor ID reported for the device. */
#define RADV_XCLIPSE_VENDOR_ID 0x144d

/* Queries the device through the vendor library; submits no work. */
VkResult radv_xclipse_query_info(int fd, struct radeon_winsys_info *info);

#endif /* RADV_XCLIPSE_H */
