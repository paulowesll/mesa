/* Copyright 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef AC_CMDBUF_BASE_H
#define AC_CMDBUF_BASE_H

#include <stdint.h>
#include <stdbool.h>

struct ac_cmdbuf {
   uint32_t cdw;         /* Number of used dwords. */
   uint32_t max_dw;      /* Maximum number of dwords. */
   uint32_t reserved_dw; /* Number of dwords reserved. */
   uint32_t *buf;        /* The base pointer of the chunk. */

   bool context_roll;

   /* Samsung Xclipse (MGFX4): context registers are emitted at their MGFX4 offsets, see
    * ac_mgfx4_ctxreg_remap.h.  Set by the SGPU winsys.
    */
   bool remap_ctx_regs;
};

#endif
