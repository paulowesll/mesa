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

VkResult radv_sgpu_winsys_create(int fd, const struct radeon_info *info, struct radeon_winsys **winsys);

/* Memory this process holds in SGPU BOs, across all devices. */
uint64_t radv_sgpu_process_allocated(void);

#endif /* RADV_SGPU_WINSYS_PUBLIC_H */
