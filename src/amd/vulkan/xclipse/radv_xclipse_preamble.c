/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Register state that the graphics preamble establishes on Xclipse.
 *
 * Nothing resets the context registers between processes (there is no CLEAR_STATE), so every
 * register RADV does not program keeps what another context left: a stale pixel-centre offset
 * once shifted every interpolated coordinate.  The preamble therefore starts with the whole
 * register state of the vendor driver, then the MGFX4 front-end setup; the rest of the preamble
 * and the command stream override what RADV programs.
 */

#include "radv_xclipse.h"
#include "radv_xclipse_native_defaults.h"

#include "util/macros.h"
#include "ac_mgfx4_regs.h"
#include "ac_pm4.h"
#include "sid.h"

/* MGFX4 registers without a GFX11 name (UCONFIG and raw context dword offsets). */
#define MGFX4_UCONFIG_GE_VRS_RATE          0x263
#define MGFX4_UCONFIG_SIMD_CONFIG          0x480 /* 8 pairs */
#define MGFX4_UCONFIG_490                  0x490
#define MGFX4_UCONFIG_493                  0x493
#define MGFX4_UCONFIG_494                  0x494 /* 3 + 8 registers */
#define MGFX4_UCONFIG_4C0                  0x4c0
#define MGFX4_CONTEXT_PIXEL_CENTER_OFFSET  0x2af

/* A GFX11 context offset without a name, next to PA_SC_CLIPRECT_*_EXT; remapped like the others. */
#define R_028384_MGFX4_CLIPRECT_EXT_NEXT   0x028384

static void
add_uconfig_seq(struct ac_pm4_state *pm4, unsigned offset, unsigned count, const uint32_t *values)
{
   ac_pm4_cmd_add(pm4, PKT3(PKT3_SET_UCONFIG_REG, count, 0));
   ac_pm4_cmd_add(pm4, offset);
   for (unsigned i = 0; i < count; i++)
      ac_pm4_cmd_add(pm4, values[i]);
}

static void
add_uconfig(struct ac_pm4_state *pm4, unsigned offset, uint32_t value)
{
   add_uconfig_seq(pm4, offset, 1, &value);
}

void
radv_xclipse_add_native_state(struct ac_pm4_state *pm4)
{
   static const uint8_t opcode[] = {PKT3_SET_SH_REG, PKT3_SET_CONTEXT_REG, PKT3_SET_UCONFIG_REG};

   for (unsigned r = 0; r < ARRAY_SIZE(radv_xclipse_native_default_runs); r++) {
      const struct radv_xclipse_native_default_run *run = &radv_xclipse_native_default_runs[r];

      ac_pm4_cmd_add(pm4, PKT3(opcode[run->space], run->count, 0));
      ac_pm4_cmd_add(pm4, run->offset);
      for (unsigned i = 0; i < run->count; i++)
         ac_pm4_cmd_add(pm4, radv_xclipse_native_default_values[run->first + i]);
   }

   /* Front-end setup of the vendor driver: primitive and index state, the SIMD configuration
    * and cache control registers of MGFX4, and the scissor/clip rectangles.
    */
   add_uconfig(pm4, (R_030904_MGFX4_VGT_GS_OUT_PRIM_TYPE - CIK_UCONFIG_REG_OFFSET) >> 2, 0);
   add_uconfig(pm4, (R_030908_VGT_PRIMITIVE_TYPE - CIK_UCONFIG_REG_OFFSET) >> 2, 0);
   add_uconfig(pm4, (R_03092C_GE_MULTI_PRIM_IB_RESET_EN - CIK_UCONFIG_REG_OFFSET) >> 2, 0);
   ac_pm4_cmd_add(pm4, PKT3(PKT3_SET_UCONFIG_REG_INDEX, 1, 0));
   ac_pm4_cmd_add(pm4, ((R_030934_VGT_NUM_INSTANCES - CIK_UCONFIG_REG_OFFSET) >> 2) | (3u << 28));
   ac_pm4_cmd_add(pm4, 0);
   add_uconfig(pm4, (R_03096C_GE_CNTL - CIK_UCONFIG_REG_OFFSET) >> 2, 0);
   add_uconfig_seq(pm4, (R_03097C_GE_STEREO_CNTL - CIK_UCONFIG_REG_OFFSET) >> 2, 3,
                   (const uint32_t[]){0, 0, 0}); /* GE_STEREO_CNTL, GE_PC_ALLOC, VGT_TF_MEMORY_BASE_HI */
   add_uconfig(pm4, MGFX4_UCONFIG_GE_VRS_RATE, 0);
   add_uconfig_seq(pm4, (R_030E00_TA_CS_BC_BASE_ADDR - CIK_UCONFIG_REG_OFFSET) >> 2, 2, (const uint32_t[]){0, 0});

   /* The first pair differs: the vendor rewrites it (and 0x4c0) in the preamble of every submit. */
   uint32_t simd_config[16] = {0x124996db, 0x0001369a, 0, 0};
   for (unsigned i = 2; i < 8; i++) {
      simd_config[i * 2] = 0x1b6db6db;
      simd_config[i * 2 + 1] = 0x0001b6db;
   }
   add_uconfig_seq(pm4, MGFX4_UCONFIG_SIMD_CONFIG, 16, simd_config);
   add_uconfig_seq(pm4, MGFX4_UCONFIG_490, 2, (const uint32_t[]){0, 0});
   add_uconfig(pm4, MGFX4_UCONFIG_493, 0x00000041);
   /* The vendor clears bit 7 of 0x494/0x495, whose meaning is unknown. With it set, the Z pre-pass of
    * the binner was 20% slower on six RGBA16F targets.
    */
   add_uconfig_seq(pm4, MGFX4_UCONFIG_494, 11,
                   (const uint32_t[]){0x0001ff7f, 0x0001ff7f, 0x0001ffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff});
   add_uconfig(pm4, MGFX4_UCONFIG_4C0, 0x00080128);
   add_uconfig(pm4, (R_030964_GE_MAX_VTX_INDX - CIK_UCONFIG_REG_OFFSET) >> 2, 0xffffffff);
   add_uconfig_seq(pm4, (R_030924_GE_MIN_VTX_INDX - CIK_UCONFIG_REG_OFFSET) >> 2, 2,
                   (const uint32_t[]){0, 0}); /* GE_MIN_VTX_INDX, GE_INDX_OFFSET */

   /* A per-rate pixel-centre offset that the GFX11 path never programs; the vendor writes 0. */
   ac_pm4_cmd_add(pm4, PKT3(PKT3_SET_CONTEXT_REG, 1, 0));
   ac_pm4_cmd_add(pm4, MGFX4_CONTEXT_PIXEL_CENTER_OFFSET);
   ac_pm4_cmd_add(pm4, 0);

   /* GFX11 context registers, remapped by ac_pm4_set_reg. */
   ac_pm4_set_reg(pm4, R_028374_PA_SC_CLIPRECT_0_EXT, 0x02000000);
   ac_pm4_set_reg(pm4, R_028384_MGFX4_CLIPRECT_EXT_NEXT, 0);
   ac_pm4_set_reg(pm4, R_028350_PA_SC_RASTER_CONFIG, 0x02000000);
   ac_pm4_set_reg(pm4, R_028C54_PA_SC_BINNER_CNTL_2, 0x00100000);
   ac_pm4_set_reg(pm4, R_028230_PA_SC_EDGERULE, 0xaa99aaaa);
   ac_pm4_set_reg(pm4, R_028034_PA_SC_SCREEN_SCISSOR_BR, 0x40004000);
   ac_pm4_set_reg(pm4, R_028204_PA_SC_WINDOW_SCISSOR_TL, 0x80000000);
   ac_pm4_set_reg(pm4, R_028208_PA_SC_WINDOW_SCISSOR_BR, 0x40004000);
   ac_pm4_set_reg(pm4, R_02820C_PA_SC_CLIPRECT_RULE, 0x0000ffff);
   for (unsigned i = 0; i < 4; i++)
      ac_pm4_set_reg(pm4, R_028214_PA_SC_CLIPRECT_0_BR + i * 8, 0x40004000);
   ac_pm4_set_reg(pm4, R_028BDC_PA_SC_LINE_CNTL, 0x00001000);
   ac_pm4_set_reg(pm4, R_028BE4_PA_SU_VTX_CNTL, 0x00000005);
   ac_pm4_set_reg(pm4, R_028C38_PA_SC_AA_MASK_X0Y0_X1Y0, 0xffffffff);
   ac_pm4_set_reg(pm4, R_028C3C_PA_SC_AA_MASK_X0Y1_X1Y1, 0xffffffff);
   ac_pm4_set_reg(pm4, R_028C44_PA_SC_BINNER_CNTL_0, 0x00000003);
   ac_pm4_set_reg(pm4, R_028830_PA_SU_SMALL_PRIM_FILTER_CNTL, 0x00000041);
   ac_pm4_set_reg(pm4, R_028424_CB_FDCC_CONTROL, 0x00000018);
   ac_pm4_set_reg(pm4, R_028750_SX_PS_DOWNCONVERT_CONTROL, 0x000000ff);
   ac_pm4_set_reg(pm4, R_0286E8_SPI_TMPRING_SIZE, 0x00000200);
   ac_pm4_set_reg(pm4, R_0286EC_SPI_GFX_SCRATCH_BASE_LO, 0);
   ac_pm4_set_reg(pm4, R_0286F0_SPI_GFX_SCRATCH_BASE_HI, 0);
   ac_pm4_set_reg(pm4, R_028240_PA_SC_GENERIC_SCISSOR_TL, 0x80000000);
   ac_pm4_set_reg(pm4, R_028244_PA_SC_GENERIC_SCISSOR_BR, 0x40004000);
   for (unsigned i = 0; i < 16; i++) {
      ac_pm4_set_reg(pm4, R_028250_PA_SC_VPORT_SCISSOR_0_TL + i * 8, 0x80000000);
      ac_pm4_set_reg(pm4, R_028254_PA_SC_VPORT_SCISSOR_0_BR + i * 8, 0x40004000);
   }
}

/* L2 policies of the colour and depth blocks, as the vendor driver programs them: colour and DCC
 * writes stay in L2 (LRU) and HTILE is cached for reads and writes, while depth/stencil data
 * streams. The GFX10+ preamble streams colour, DCC and HTILE instead, which on Xclipse costs about
 * a fifth of the fill and blend rate and of depth-tested overdraw. This has to be emitted after
 * ac_init_graphics_preamble_state, which programs both registers.
 */
void
radv_xclipse_add_cache_policy(struct ac_pm4_state *pm4)
{
   ac_pm4_set_reg(pm4, R_028410_CB_RMI_GL2_CACHE_CONTROL, 0x00800040);
   ac_pm4_set_reg(pm4, R_02807C_DB_RMI_L2_CACHE_CONTROL, 0x000a0045);
}
