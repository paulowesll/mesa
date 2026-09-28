/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* Shader registers that Samsung Xclipse (MGFX4) placed differently from GFX11, and the vendor
 * packet that writes the CU-enable pairs.  Observed in the command streams of Samsung's driver.
 */

#ifndef AC_MGFX4_REGS_H
#define AC_MGFX4_REGS_H

/* NGG (ES/GS) program. */
#define R_00B210_MGFX4_SPI_SHADER_PGM_RSRC1_GS 0x00B210
#define R_00B214_MGFX4_SPI_SHADER_PGM_RSRC2_GS 0x00B214
#define R_00B218_MGFX4_SPI_SHADER_PGM_LO_ES    0x00B218
#define R_00B21C_MGFX4_SPI_SHADER_PGM_HI_ES    0x00B21C
#define R_00B228_MGFX4_SPI_SHADER_PGM_RSRC3_GS 0x00B228
#define R_00B22C_MGFX4_SPI_SHADER_PGM_RSRC4_GS 0x00B22C

/* HS program. */
#define R_00B410_MGFX4_SPI_SHADER_PGM_RSRC1_HS 0x00B410
#define R_00B414_MGFX4_SPI_SHADER_PGM_RSRC2_HS 0x00B414
#define R_00B418_MGFX4_SPI_SHADER_PGM_LO_HS    0x00B418
#define R_00B41C_MGFX4_SPI_SHADER_PGM_HI_HS    0x00B41C
#define R_00B428_MGFX4_SPI_SHADER_PGM_RSRC3_HS 0x00B428
#define R_00B42C_MGFX4_SPI_SHADER_PGM_RSRC4_HS 0x00B42C

/* PS: RSRC3 sits next to RSRC4 (0xB004, as on GFX11). */
#define R_00B000_MGFX4_SPI_SHADER_PGM_RSRC3_PS 0x00B000

/* Viewports use the GFX12 layout from here: XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET,
 * ZMIN and ZMAX for each viewport.
 */
#define R_028444_MGFX4_PA_CL_VPORT_XSCALE      0x028444

/* The vendor writes each RSRC3/RSRC4 CU-enable pair with packet 0x3E and register index 3, which
 * applies the CU mask; with plain SET_SH_REG the waves of that stage run on part of the CUs only.
 * RSRC4 carries bit 31 in that form.
 */
#define PKT3_MGFX4_SET_SH_REG_INDEX       0x3E
#define MGFX4_CU_PAIR_HEADER              (PKT3(PKT3_MGFX4_SET_SH_REG_INDEX, 2, 0) | 0x08)
#define MGFX4_CU_PAIR_REG(rsrc3_reg)      (0x30000000 | (((rsrc3_reg) - SI_SH_REG_OFFSET) >> 2))
#define MGFX4_CU_PAIR_RSRC4_BIT           (1u << 31)

#endif /* AC_MGFX4_REGS_H */
