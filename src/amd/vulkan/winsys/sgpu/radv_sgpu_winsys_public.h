/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_SGPU_WINSYS_PUBLIC_H
#define RADV_SGPU_WINSYS_PUBLIC_H

#include <vulkan/vulkan_core.h>

struct radeon_info;
struct radeon_winsys;

/* dlopen()s libdrm_sgpu.so: RADV_SGPU_VENDOR_LIB, a copy next to this driver (driver packages
 * ship one), then the system copy.  Returns NULL when none loads.
 */
void *radv_sgpu_open_vendor_library(void);

VkResult radv_sgpu_winsys_create(int fd, const struct radeon_info *info, struct radeon_winsys **winsys);

#endif /* RADV_SGPU_WINSYS_PUBLIC_H */
