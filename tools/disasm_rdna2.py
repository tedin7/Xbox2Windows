#!/usr/bin/env python3
"""
AMD RDNA 2 (GFX1013 / Scarlett) Shader Disassembler & Inspector.

Decodes raw binary machine code (.bin) from Xbox Scarlett shaders,
prints human-readable assembly with byte offsets, raw hex words, register
tracking, and explanations mapped to the AMD RDNA 2 ISA manual.
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import struct
import sys
import tempfile

# Register decode helpers

# Source operand codes that name hardware values rather than SGPRs (LLVM spellings).
SPECIAL_SRC_OPERANDS = {
    235: "src_shared_base",
    236: "src_shared_limit",
    237: "src_private_base",
    238: "src_private_limit",
    239: "src_pops_exiting_wave_id",
    251: "src_vccz",
    252: "src_execz",
    253: "src_scc",
    254: "src_lds_direct",
}


def decode_s_reg(reg_id: int) -> str:
    """Decode scalar register or scalar constant code."""
    if reg_id <= 105:
        return f"s{reg_id}"
    if reg_id == 106:
        return "vcc_lo"
    if reg_id == 107:
        return "vcc_hi"
    if 108 <= reg_id <= 123:
        return f"ttmp{reg_id - 108}"
    if reg_id == 124:
        return "m0"
    if reg_id == 125:
        return "null"
    if reg_id == 126:
        return "exec_lo"
    if reg_id == 127:
        return "exec_hi"
    if 128 <= reg_id <= 192:
        return str(reg_id - 128)
    if 193 <= reg_id <= 208:
        return str(-(reg_id - 192))
    if reg_id == 240:
        return "0.5"
    if reg_id == 241:
        return "-0.5"
    if reg_id == 242:
        return "1.0"
    if reg_id == 243:
        return "-1.0"
    if reg_id == 244:
        return "2.0"
    if reg_id == 245:
        return "-2.0"
    if reg_id == 246:
        return "4.0"
    if reg_id == 247:
        return "-4.0"
    if reg_id == 248:
        return "0.15915494"
    if reg_id in SPECIAL_SRC_OPERANDS:
        return SPECIAL_SRC_OPERANDS[reg_id]
    if reg_id == 255:
        return "literal"
    return f"s{reg_id}"


def decode_s_reg64(reg_id: int) -> str:
    """Decode a 64-bit scalar operand as an SGPR pair or a named 64-bit special."""
    if reg_id <= 105 and reg_id % 2 == 0:
        return f"s[{reg_id}:{reg_id + 1}]"
    if reg_id == 106:
        return "vcc"
    if 108 <= reg_id <= 122 and (reg_id - 108) % 2 == 0:
        n = reg_id - 108
        return f"ttmp[{n}:{n + 1}]"
    if reg_id == 125:
        return "null"
    if reg_id == 126:
        return "exec"
    if reg_id == 248:
        return "0.15915494309189532"
    return decode_s_reg(reg_id)


def decode_v_reg(reg_id: int) -> str:
    """Decode a 9-bit VOP SRC0 operand: 0..255 scalar/constant, 256..511 VGPR."""
    if 256 <= reg_id <= 511:
        return f"v{reg_id - 256}"
    return decode_s_reg(reg_id)


def decode_v_reg64(reg_id: int) -> str:
    """Decode a 64-bit VOP operand as a VGPR pair, SGPR pair, or constant."""
    if 256 <= reg_id <= 511:
        vreg = reg_id - 256
        return f"v[{vreg}:{vreg + 1}]" if vreg < 255 else "<invalid VGPR pair>"
    return decode_s_reg64(reg_id)


def decode_v_reg128(reg_id: int) -> str:
    """Decode a 128-bit VOP operand that only a VGPR quad can hold (a scalar source is refused by the caller)."""
    vreg = reg_id - 256
    return f"v[{vreg}:{vreg + 3}]" if vreg <= 252 else "<invalid VGPR quad>"


def _format_operand_modifiers(val: str, is_abs: int, is_neg: int) -> str:
    """Wrap a disassembled operand with inline abs (`|...|`) and neg (`-`) modifiers."""
    if is_abs and is_neg:
        return f"-|{val}|"
    if is_abs:
        return f"|{val}|"
    if is_neg:
        return f"-{val}"
    return val


# Instruction Opcode Tables
SOPP_OPS = {
    0x00: ("s_nop", "No operation (wait cycles)"),
    0x01: ("s_endpgm", "Terminate shader execution"),
    0x02: ("s_branch", "Unconditional branch"),
    0x03: ("s_wakeup", "Wake a sleeping wave"),
    0x04: ("s_cbranch_scc0", "Branch if SCC == 0"),
    0x05: ("s_cbranch_scc1", "Branch if SCC == 1"),
    0x06: ("s_cbranch_vccz", "Branch if VCC is zero"),
    0x07: ("s_cbranch_vccnz", "Branch if VCC is non-zero"),
    0x08: ("s_cbranch_execz", "Branch if EXEC is zero"),
    0x09: ("s_cbranch_execnz", "Branch if EXEC is non-zero"),
    0x0a: ("s_barrier", "Synchronize wavefronts in workgroup"),
    0x0b: ("s_setkill", "Set the wave kill bit"),
    0x0c: ("s_waitcnt", "Wait for memory operations"),
    0x0d: ("s_sethalt", "Set the wave halt bit"),
    0x0e: ("s_sleep", "Suspend the wave temporarily"),
    0x0f: ("s_setprio", "Set wave priority"),
    0x10: ("s_sendmsg", "Send a message"),
    0x11: ("s_sendmsghalt", "Send a message and halt the wave"),
    0x12: ("s_trap", "Raise a trap"),
    0x13: ("s_icache_inv", "Invalidate the instruction cache"),
    0x14: ("s_incperflevel", "Increment the selected performance counter"),
    0x15: ("s_decperflevel", "Decrement the selected performance counter"),
    0x16: ("s_ttracedata", "Emit thread trace data"),
    0x1b: ("s_endpgm_saved", "Terminate the wave and signal that its context was saved"),
    0x1e: ("s_endpgm_ordered_ps_done", "Terminate an ordered pixel shader wave"),
    0x1f: ("s_code_end", "Code end marker"),
    0x20: ("s_inst_prefetch", "Instruction cache prefetch"),
    0x21: ("s_clause", "Begin hard clause"),
    0x22: ("s_wait_idle", "Wait for outstanding operations to finish"),
    0x23: ("s_waitcnt_depctr", "Wait for dependency counters"),
    0x24: ("s_round_mode", "Set floating-point rounding mode"),
    0x25: ("s_denorm_mode", "Set floating-point denormal mode"),
    0x28: ("s_ttracedata_imm", "Emit immediate thread trace data"),
}

SOP1_OPS = {
    0x03: ("s_mov_b32", "Scalar move 32-bit"),
    0x04: ("s_mov_b64", "Scalar move 64-bit"),
    0x05: ("s_cmov_b32", "Scalar conditional move 32-bit"),
    0x06: ("s_cmov_b64", "Scalar conditional move 64-bit"),
    0x07: ("s_not_b32", "Scalar bitwise NOT 32-bit"),
    0x08: ("s_not_b64", "Scalar bitwise NOT 64-bit"),
    0x09: ("s_wqm_b32", "Scalar whole-quad-mode 32-bit"),
    0x0a: ("s_wqm_b64", "Scalar whole-quad-mode 64-bit"),
    0x0b: ("s_brev_b32", "Scalar bit reverse 32-bit"),
    0x0c: ("s_brev_b64", "Scalar bit reverse 64-bit"),
    0x0d: ("s_bcnt0_i32_b32", "Count zero bits in scalar 32-bit"),
    0x0e: ("s_bcnt0_i32_b64", "Count zero bits in scalar 64-bit"),
    0x0f: ("s_bcnt1_i32_b32", "Count one bits in scalar 32-bit"),
    0x10: ("s_bcnt1_i32_b64", "Count one bits in scalar 64-bit"),
    0x11: ("s_ff0_i32_b32", "Find first zero bit in scalar 32-bit"),
    0x12: ("s_ff0_i32_b64", "Find first zero bit in scalar 64-bit"),
    0x13: ("s_ff1_i32_b32", "Find first one bit in scalar 32-bit"),
    0x14: ("s_ff1_i32_b64", "Find first one bit in scalar 64-bit"),
    0x15: ("s_flbit_i32_b32", "Find first bit high in scalar 32-bit unsigned"),
    0x16: ("s_flbit_i32_b64", "Find first bit high in scalar 64-bit unsigned"),
    0x17: ("s_flbit_i32", "Find first bit high in scalar 32-bit signed"),
    0x18: ("s_flbit_i32_i64", "Find first bit high in scalar 64-bit signed"),
    0x19: ("s_sext_i32_i8", "Sign-extend 8-bit to 32-bit"),
    0x1a: ("s_sext_i32_i16", "Sign-extend 16-bit to 32-bit"),
    0x1b: ("s_bitset0_b32", "Clear bit in scalar 32-bit"),
    0x1c: ("s_bitset0_b64", "Clear bit in scalar 64-bit"),
    0x1d: ("s_bitset1_b32", "Set bit in scalar 32-bit"),
    0x1e: ("s_bitset1_b64", "Set bit in scalar 64-bit"),
    0x1f: ("s_getpc_b64", "Get program counter 64-bit"),
    0x20: ("s_setpc_b64", "Set program counter 64-bit"),
    0x21: ("s_swappc_b64", "Swap program counter 64-bit"),
    0x22: ("s_rfe_b64", "Return from exception 64-bit"),
    0x24: ("s_and_saveexec_b64", "Save execution mask and bitwise AND with scalar source 64-bit"),
    0x25: ("s_or_saveexec_b64", "Save execution mask and bitwise OR with scalar source 64-bit"),
    0x26: ("s_xor_saveexec_b64", "Save execution mask and bitwise XOR with scalar source 64-bit"),
    0x27: ("s_andn2_saveexec_b64", "Save execution mask and bitwise AND NOT2 with scalar source 64-bit"),
    0x28: ("s_orn2_saveexec_b64", "Save execution mask and bitwise OR NOT2 with scalar source 64-bit"),
    0x29: ("s_nand_saveexec_b64", "Save execution mask and bitwise NAND with scalar source 64-bit"),
    0x2a: ("s_nor_saveexec_b64", "Save execution mask and bitwise NOR with scalar source 64-bit"),
    0x2b: ("s_xnor_saveexec_b64", "Save execution mask and bitwise XNOR with scalar source 64-bit"),
    0x2c: ("s_quadmask_b32", "Compute quad mask scalar 32-bit"),
    0x2d: ("s_quadmask_b64", "Compute quad mask scalar 64-bit"),
    0x2e: ("s_movrels_b32", "Scalar move 32-bit with relative source index in M0"),
    0x2f: ("s_movrels_b64", "Scalar move 64-bit with relative source index in M0"),
    0x30: ("s_movreld_b32", "Scalar move 32-bit with relative destination index in M0"),
    0x31: ("s_movreld_b64", "Scalar move 64-bit with relative destination index in M0"),
    0x34: ("s_abs_i32", "Scalar absolute value signed 32-bit"),
    0x37: ("s_andn1_saveexec_b64", "Save execution mask and bitwise AND NOT1 with scalar source 64-bit"),
    0x38: ("s_orn1_saveexec_b64", "Save execution mask and bitwise OR NOT1 with scalar source 64-bit"),
    0x39: ("s_andn1_wrexec_b64", "Save execution mask and write bitwise AND NOT1 scalar source 64-bit"),
    0x3a: ("s_andn2_wrexec_b64", "Save execution mask and write bitwise AND NOT2 scalar source 64-bit"),
    0x3b: ("s_bitreplicate_b64_b32", "Replicate each bit of 32-bit source into 64-bit pair"),
    0x3c: ("s_and_saveexec_b32", "Save execution mask and bitwise AND with scalar source"),
    0x3d: ("s_or_saveexec_b32", "Save execution mask and bitwise OR with scalar source 32-bit"),
    0x3e: ("s_xor_saveexec_b32", "Save execution mask and bitwise XOR with scalar source 32-bit"),
    0x3f: ("s_andn2_saveexec_b32", "Save execution mask and bitwise AND NOT2 with scalar source 32-bit"),
    0x40: ("s_orn2_saveexec_b32", "Save execution mask and bitwise OR NOT2 with scalar source 32-bit"),
    0x41: ("s_nand_saveexec_b32", "Save execution mask and bitwise NAND with scalar source 32-bit"),
    0x42: ("s_nor_saveexec_b32", "Save execution mask and bitwise NOR with scalar source 32-bit"),
    0x43: ("s_xnor_saveexec_b32", "Save execution mask and bitwise XNOR with scalar source 32-bit"),
    0x44: ("s_andn1_saveexec_b32", "Save execution mask and bitwise AND NOT1 with scalar source 32-bit"),
    0x45: ("s_orn1_saveexec_b32", "Save execution mask and bitwise OR NOT1 with scalar source 32-bit"),
    0x46: ("s_andn1_wrexec_b32", "Save execution mask and write bitwise AND NOT1 scalar source 32-bit"),
    0x47: ("s_andn2_wrexec_b32", "Save execution mask and write bitwise AND NOT2 scalar source 32-bit"),
    0x49: ("s_movrelsd_2_b32", "Scalar move 32-bit with relative source and destination indices in M0"),
}

SOP2_OPS = {
    0x00: ("s_add_u32", "Scalar add unsigned 32-bit"),
    0x01: ("s_sub_u32", "Scalar sub unsigned 32-bit"),
    0x02: ("s_add_i32", "Scalar add signed 32-bit"),
    0x03: ("s_sub_i32", "Scalar sub signed 32-bit"),
    0x04: ("s_addc_u32", "Scalar add unsigned 32-bit with carry in SCC"),
    0x05: ("s_subb_u32", "Scalar sub unsigned 32-bit with borrow in SCC"),
    0x06: ("s_min_i32", "Scalar minimum signed 32-bit"),
    0x07: ("s_min_u32", "Scalar minimum unsigned 32-bit"),
    0x08: ("s_max_i32", "Scalar maximum signed 32-bit"),
    0x09: ("s_max_u32", "Scalar maximum unsigned 32-bit"),
    0x0a: ("s_cselect_b32", "Scalar conditional select 32-bit"),
    0x0b: ("s_cselect_b64", "Scalar conditional select 64-bit"),
    0x0e: ("s_and_b32", "Scalar bitwise AND 32-bit"),
    0x0f: ("s_and_b64", "Scalar bitwise AND 64-bit"),
    0x10: ("s_or_b32", "Scalar bitwise OR 32-bit"),
    0x11: ("s_or_b64", "Scalar bitwise OR 64-bit"),
    0x12: ("s_xor_b32", "Scalar bitwise XOR 32-bit"),
    0x13: ("s_xor_b64", "Scalar bitwise XOR 64-bit"),
    0x14: ("s_andn2_b32", "Scalar AND with NOT 32-bit"),
    0x15: ("s_andn2_b64", "Scalar AND with NOT 64-bit"),
    0x16: ("s_orn2_b32", "Scalar OR with NOT 32-bit"),
    0x17: ("s_orn2_b64", "Scalar OR with NOT 64-bit"),
    0x18: ("s_nand_b32", "Scalar bitwise NAND 32-bit"),
    0x19: ("s_nand_b64", "Scalar bitwise NAND 64-bit"),
    0x1a: ("s_nor_b32", "Scalar bitwise NOR 32-bit"),
    0x1b: ("s_nor_b64", "Scalar bitwise NOR 64-bit"),
    0x1c: ("s_xnor_b32", "Scalar bitwise XNOR 32-bit"),
    0x1d: ("s_xnor_b64", "Scalar bitwise XNOR 64-bit"),
    0x1e: ("s_lshl_b32", "Scalar logical shift left 32-bit"),
    0x1f: ("s_lshl_b64", "Scalar logical shift left 64-bit"),
    0x20: ("s_lshr_b32", "Scalar logical shift right 32-bit"),
    0x21: ("s_lshr_b64", "Scalar logical shift right 64-bit"),
    0x22: ("s_ashr_i32", "Scalar arithmetic shift right 32-bit"),
    0x23: ("s_ashr_i64", "Scalar arithmetic shift right 64-bit"),
    0x24: ("s_bfm_b32", "Scalar bitfield mask 32-bit"),
    0x25: ("s_bfm_b64", "Scalar bitfield mask 64-bit"),
    0x26: ("s_mul_i32", "Scalar multiply signed 32-bit"),
    0x27: ("s_bfe_u32", "Scalar bitfield extract unsigned 32-bit"),
    0x28: ("s_bfe_i32", "Scalar bitfield extract signed 32-bit"),
    0x29: ("s_bfe_u64", "Scalar bitfield extract unsigned 64-bit"),
    0x2a: ("s_bfe_i64", "Scalar bitfield extract signed 64-bit"),
    0x2c: ("s_absdiff_i32", "Scalar absolute difference signed 32-bit"),
    0x2e: ("s_lshl1_add_u32", "Scalar shift left 1 and add 32-bit"),
    0x2f: ("s_lshl2_add_u32", "Scalar shift left 2 and add 32-bit"),
    0x30: ("s_lshl3_add_u32", "Scalar shift left 3 and add 32-bit"),
    0x31: ("s_lshl4_add_u32", "Scalar shift left 4 and add 32-bit"),
    0x32: ("s_pack_ll_b32_b16", "Scalar pack low halves of two sources into 32-bit"),
    0x33: ("s_pack_lh_b32_b16", "Scalar pack low half and high half into 32-bit"),
    0x34: ("s_pack_hh_b32_b16", "Scalar pack high halves of two sources into 32-bit"),
    0x35: ("s_mul_hi_u32", "Scalar multiply high unsigned 32-bit"),
    0x36: ("s_mul_hi_i32", "Scalar multiply high signed 32-bit"),
}

SOPK_OPS = {
    0x00: ("s_movk_i32", "Scalar move 16-bit sign-extended immediate"),
    0x01: ("s_version", "Version marker with a 16-bit immediate, no destination"),
    0x02: ("s_cmovk_i32", "Scalar conditional move immediate"),
    0x03: ("s_cmpk_eq_i32", "Scalar compare equal immediate"),
    0x04: ("s_cmpk_lg_i32", "Scalar compare not-equal immediate"),
    0x05: ("s_cmpk_gt_i32", "Scalar compare greater immediate"),
    0x06: ("s_cmpk_ge_i32", "Scalar signed compare greater-or-equal immediate"),
    0x07: ("s_cmpk_lt_i32", "Scalar signed compare less immediate"),
    0x08: ("s_cmpk_le_i32", "Scalar signed compare less-or-equal immediate"),
    0x09: ("s_cmpk_eq_u32", "Scalar unsigned compare equal immediate"),
    0x0a: ("s_cmpk_lg_u32", "Scalar unsigned compare not-equal immediate"),
    0x0b: ("s_cmpk_gt_u32", "Scalar unsigned compare greater immediate"),
    0x0c: ("s_cmpk_ge_u32", "Scalar unsigned compare greater-or-equal immediate"),
    0x0d: ("s_cmpk_lt_u32", "Scalar unsigned compare less immediate"),
    0x0e: ("s_cmpk_le_u32", "Scalar unsigned compare less-or-equal immediate"),
    0x0f: ("s_addk_i32", "Scalar add immediate"),
    0x10: ("s_mulk_i32", "Scalar multiply immediate"),
    0x12: ("s_getreg_b32", "Read a hardware register field into an SGPR"),
    0x13: ("s_setreg_b32", "Write a hardware register field from an SGPR"),
    0x15: ("s_setreg_imm32_b32", "Write a hardware register field from a 32-bit literal"),
    0x16: ("s_call_b64", "Call a PC-relative target and save the return address"),
    0x17: ("s_waitcnt_vscnt", "Wait for vector store counter"),
    0x18: ("s_waitcnt_vmcnt", "Wait for vector memory counter"),
    0x19: ("s_waitcnt_expcnt", "Wait for export counter"),
    0x1a: ("s_waitcnt_lgkmcnt", "Wait for LDS/GDS/K/M counter"),
}

# Hardware register IDs that LLVM 15 prints by name for gfx1013 (hwreg ID field, SIMM16[5:0]).
# Every other ID prints as a decimal number, e.g. ID 4 (HW_REG_HW_ID before gfx10) as hwreg(4).
HWREG_NAMES = {
    1: "HW_REG_MODE",
    2: "HW_REG_STATUS",
    3: "HW_REG_TRAPSTS",
    5: "HW_REG_GPR_ALLOC",
    6: "HW_REG_LDS_ALLOC",
    7: "HW_REG_IB_STS",
    15: "HW_REG_SH_MEM_BASES",
    16: "HW_REG_TBA_LO",
    17: "HW_REG_TBA_HI",
    18: "HW_REG_TMA_LO",
    19: "HW_REG_TMA_HI",
    20: "HW_REG_FLAT_SCR_LO",
    21: "HW_REG_FLAT_SCR_HI",
    22: "HW_REG_XNACK_MASK",
    23: "HW_REG_HW_ID1",
    24: "HW_REG_HW_ID2",
    25: "HW_REG_POPS_PACKER",
}

# 32-bit patterns LLVM prints as inline floating-point values when they appear as a literal.
INLINE_FLOAT_LITERALS = {
    0x3f000000: "0.5",
    0xbf000000: "-0.5",
    0x3f800000: "1.0",
    0xbf800000: "-1.0",
    0x40000000: "2.0",
    0xc0000000: "-2.0",
    0x40800000: "4.0",
    0xc0800000: "-4.0",
    0x3e22f983: "0.15915494",
}


def format_hwreg(simm16: int) -> str:
    """Format a SOPK hwreg operand as LLVM 15 prints it.

    SIMM16 holds the register ID in bits 5:0, the bit offset in bits 10:6 and the field size minus
    one in bits 15:11. A whole-register field (offset 0, size 32) prints as hwreg(ID).
    """
    reg = simm16 & 0x3f
    offset = (simm16 >> 6) & 0x1f
    size = ((simm16 >> 11) & 0x1f) + 1
    name = HWREG_NAMES.get(reg, str(reg))
    if offset == 0 and size == 32:
        return f"hwreg({name})"
    return f"hwreg({name}, {offset}, {size})"


def format_literal32(value: int) -> str:
    """Format a 32-bit literal operand as LLVM 15 prints it.

    Values that equal an inline integer (-16 to 64) or an inline float print in that form;
    every other value prints in hex.
    """
    signed = value - (1 << 32) if value & 0x80000000 else value
    if -16 <= signed <= 64:
        return str(signed)
    if value in INLINE_FLOAT_LITERALS:
        return INLINE_FLOAT_LITERALS[value]
    return f"0x{value:x}"


SOPC_OPS = {
    # GFX10 / RDNA 2 opcode numbers, as in LLVM SOPC_Real_gfx10 definitions
    0x00: ("s_cmp_eq_i32", "Scalar compare equal signed 32-bit"),
    0x01: ("s_cmp_lg_i32", "Scalar compare not-equal signed 32-bit"),
    0x02: ("s_cmp_gt_i32", "Scalar compare greater signed 32-bit"),
    0x03: ("s_cmp_ge_i32", "Scalar compare greater-or-equal signed 32-bit"),
    0x04: ("s_cmp_lt_i32", "Scalar compare less signed 32-bit"),
    0x05: ("s_cmp_le_i32", "Scalar compare less-or-equal signed 32-bit"),
    0x06: ("s_cmp_eq_u32", "Scalar compare equal unsigned 32-bit"),
    0x07: ("s_cmp_lg_u32", "Scalar compare not-equal unsigned 32-bit"),
    0x08: ("s_cmp_gt_u32", "Scalar compare greater unsigned 32-bit"),
    0x09: ("s_cmp_ge_u32", "Scalar compare greater-or-equal unsigned 32-bit"),
    0x0a: ("s_cmp_lt_u32", "Scalar compare less unsigned 32-bit"),
    0x0b: ("s_cmp_le_u32", "Scalar compare less-or-equal unsigned 32-bit"),
    0x0c: ("s_bitcmp0_b32", "Scalar test that bit src1 of src0 is zero"),
    0x0d: ("s_bitcmp1_b32", "Scalar test that bit src1 of src0 is one"),
    0x0e: ("s_bitcmp0_b64", "Scalar test that bit src1 of 64-bit src0 is zero"),
    0x0f: ("s_bitcmp1_b64", "Scalar test that bit src1 of 64-bit src0 is one"),
    0x12: ("s_cmp_eq_u64", "Scalar compare equal unsigned 64-bit"),
    0x13: ("s_cmp_lg_u64", "Scalar compare not-equal unsigned 64-bit"),
}

SMEM_OPS = {
    0x00: "s_load_dword",
    0x01: "s_load_dwordx2",
    0x02: "s_load_dwordx4",
    0x03: "s_load_dwordx8",
    0x04: "s_load_dwordx16",
    0x08: "s_buffer_load_dword",
    0x09: "s_buffer_load_dwordx2",
    0x0a: "s_buffer_load_dwordx4",
    0x0b: "s_buffer_load_dwordx8",
    0x0c: "s_buffer_load_dwordx16",
    0x18: "s_buffer_store_dword",
    0x19: "s_buffer_store_dwordx2",
    0x1a: "s_buffer_store_dwordx4",
    0x1f: "s_gl1_inv",
    0x20: "s_dcache_inv",
    0x21: "s_dcache_wb",
    0x24: "s_memtime",
    0x25: "s_memrealtime",
    0x26: "s_atc_probe",
    0x27: "s_atc_probe_buffer",
}

# SMEM ops that take no operands (cache control) and ops whose sdata field is a 7-bit probe immediate.
SMEM_NO_OPERANDS = frozenset({0x1f, 0x20, 0x21})
SMEM_PROBE_OPS = frozenset({0x26, 0x27})

SMEM_WIDTHS = {
    0x00: (1, 2), 0x01: (2, 2), 0x02: (4, 2), 0x03: (8, 2), 0x04: (16, 2),
    0x08: (1, 4), 0x09: (2, 4), 0x0a: (4, 4), 0x0b: (8, 4), 0x0c: (16, 4),
    0x18: (1, 4), 0x19: (2, 4), 0x1a: (4, 4),
    0x24: (2, 0), 0x25: (2, 0),
    0x26: (0, 2), 0x27: (0, 4),
}

MIMG_OPS = {
    0x00: ("image_load", False, 0),
    0x01: ("image_load_mip", False, 1),
    0x02: ("image_load_pck", False, 0),
    0x03: ("image_load_pck_sgn", False, 0),
    0x04: ("image_load_mip_pck", False, 1),
    0x05: ("image_load_mip_pck_sgn", False, 1),
    0x08: ("image_store", False, 0),
    0x09: ("image_store_mip", False, 1),
    0x0a: ("image_store_pck", False, 0),
    0x0b: ("image_store_mip_pck", False, 1),
    0x0e: ("image_get_resinfo", False, 0),
    0x0f: ("image_atomic_swap", False, 0),
    0x10: ("image_atomic_cmpswap", False, 0),
    0x11: ("image_atomic_add", False, 0),
    0x12: ("image_atomic_sub", False, 0),
    0x14: ("image_atomic_smin", False, 0),
    0x15: ("image_atomic_umin", False, 0),
    0x16: ("image_atomic_smax", False, 0),
    0x17: ("image_atomic_umax", False, 0),
    0x18: ("image_atomic_and", False, 0),
    0x19: ("image_atomic_or", False, 0),
    0x1a: ("image_atomic_xor", False, 0),
    0x1b: ("image_atomic_inc", False, 0),
    0x1c: ("image_atomic_dec", False, 0),
    0x1d: ("image_atomic_fcmpswap", False, 0),
    0x1e: ("image_atomic_fmin", False, 0),
    0x1f: ("image_atomic_fmax", False, 0),
    0x20: ("image_sample", True, 0),
    0x21: ("image_sample_cl", True, 1),
    0x22: ("image_sample_d", True, 0),
    0x23: ("image_sample_d_cl", True, 1),
    0x24: ("image_sample_l", True, 1),
    0x25: ("image_sample_b", True, 1),
    0x26: ("image_sample_b_cl", True, 2),
    0x27: ("image_sample_lz", True, 0),
    0x28: ("image_sample_c", True, 1),
    0x29: ("image_sample_c_cl", True, 2),
    0x2a: ("image_sample_c_d", True, 1),
    0x2b: ("image_sample_c_d_cl", True, 2),
    0x2c: ("image_sample_c_l", True, 2),
    0x2d: ("image_sample_c_b", True, 2),
    0x2e: ("image_sample_c_b_cl", True, 3),
    0x2f: ("image_sample_c_lz", True, 1),
    0x30: ("image_sample_o", True, 1),
    0x31: ("image_sample_cl_o", True, 2),
    0x32: ("image_sample_d_o", True, 1),
    0x33: ("image_sample_d_cl_o", True, 2),
    0x34: ("image_sample_l_o", True, 2),
    0x35: ("image_sample_b_o", True, 2),
    0x36: ("image_sample_b_cl_o", True, 3),
    0x37: ("image_sample_lz_o", True, 1),
    0x38: ("image_sample_c_o", True, 2),
    0x39: ("image_sample_c_cl_o", True, 3),
    0x3a: ("image_sample_c_d_o", True, 2),
    0x3b: ("image_sample_c_d_cl_o", True, 3),
    0x3c: ("image_sample_c_l_o", True, 3),
    0x3d: ("image_sample_c_b_o", True, 3),
    0x3e: ("image_sample_c_b_cl_o", True, 4),
    0x3f: ("image_sample_c_lz_o", True, 2),
    0x40: ("image_gather4", True, 0),
    0x41: ("image_gather4_cl", True, 1),
    0x44: ("image_gather4_l", True, 1),
    0x45: ("image_gather4_b", True, 1),
    0x46: ("image_gather4_b_cl", True, 2),
    0x47: ("image_gather4_lz", True, 0),
    0x48: ("image_gather4_c", True, 1),
    0x49: ("image_gather4_c_cl", True, 2),
    0x4c: ("image_gather4_c_l", True, 2),
    0x4d: ("image_gather4_c_b", True, 2),
    0x4e: ("image_gather4_c_b_cl", True, 3),
    0x4f: ("image_gather4_c_lz", True, 1),
    0x50: ("image_gather4_o", True, 1),
    0x51: ("image_gather4_cl_o", True, 2),
    0x54: ("image_gather4_l_o", True, 2),
    0x55: ("image_gather4_b_o", True, 2),
    0x56: ("image_gather4_b_cl_o", True, 3),
    0x57: ("image_gather4_lz_o", True, 1),
    0x58: ("image_gather4_c_o", True, 2),
    0x59: ("image_gather4_c_cl_o", True, 3),
    0x5c: ("image_gather4_c_l_o", True, 3),
    0x5d: ("image_gather4_c_b_o", True, 3),
    0x5e: ("image_gather4_c_b_cl_o", True, 4),
    0x5f: ("image_gather4_c_lz_o", True, 2),
    0x60: ("image_get_lod", True, 0),
    0x68: ("image_sample_cd", True, 0),
    0x69: ("image_sample_cd_cl", True, 1),
    0x6a: ("image_sample_c_cd", True, 1),
    0x6b: ("image_sample_c_cd_cl", True, 2),
    0x6c: ("image_sample_cd_o", True, 1),
    0x6d: ("image_sample_cd_cl_o", True, 2),
    0x6e: ("image_sample_c_cd_o", True, 2),
    0x6f: ("image_sample_c_cd_cl_o", True, 3),
    0x80: ("image_msaa_load", False, 0),
    0xa2: ("image_sample_d_g16", True, 0),
    0xa3: ("image_sample_d_cl_g16", True, 1),
    0xaa: ("image_sample_c_d_g16", True, 1),
    0xab: ("image_sample_c_d_cl_g16", True, 2),
    0xb2: ("image_sample_d_o_g16", True, 1),
    0xb3: ("image_sample_d_cl_o_g16", True, 2),
    0xba: ("image_sample_c_d_o_g16", True, 2),
    0xbb: ("image_sample_c_d_cl_o_g16", True, 3),
    0xe8: ("image_sample_cd_g16", True, 0),
    0xe9: ("image_sample_cd_cl_g16", True, 1),
    0xea: ("image_sample_c_cd_g16", True, 1),
    0xeb: ("image_sample_c_cd_cl_g16", True, 2),
    0xec: ("image_sample_cd_o_g16", True, 1),
    0xed: ("image_sample_cd_cl_o_g16", True, 2),
    0xee: ("image_sample_c_cd_o_g16", True, 2),
    0xef: ("image_sample_c_cd_cl_o_g16", True, 3),
}

# Sampling ops whose address carries explicit derivatives (the _d and _cd families).
MIMG_DERIV_OPS = frozenset({
    0x22, 0x23, 0x2a, 0x2b, 0x32, 0x33, 0x3a, 0x3b,
    0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
})
# The _g16 forms take packed 16-bit derivatives: address dwords per dimension code.
MIMG_G16_DERIV_OPS = frozenset({
    0xa2, 0xa3, 0xaa, 0xab, 0xb2, 0xb3, 0xba, 0xbb,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
})
MIMG_G16_DERIV_ADDRS = (2, 2, 4, 2, 2, 2, 2, 2)

# Ray-intersection ops: opcode -> (name, address dwords, address dwords with a16).
MIMG_BVH_OPS = {
    0xe6: ("image_bvh_intersect_ray", 11, 8),
    0xe7: ("image_bvh64_intersect_ray", 12, 9),
}
# Word 0 of a BVH op with the opcode field masked out (r128, unorm and dmask 0xf are implied).
MIMG_BVH_W0 = 0xf0009f01

MIMG_DIMS = {
    0: ("SQ_RSRC_IMG_1D", 1, 2),
    1: ("SQ_RSRC_IMG_2D", 2, 4),
    2: ("SQ_RSRC_IMG_3D", 3, 6),
    3: ("SQ_RSRC_IMG_CUBE", 3, 4),
    4: ("SQ_RSRC_IMG_1D_ARRAY", 2, 2),
    5: ("SQ_RSRC_IMG_2D_ARRAY", 3, 4),
    6: ("SQ_RSRC_IMG_2D_MSAA", 3, 4),
    7: ("SQ_RSRC_IMG_2D_MSAA_ARRAY", 4, 4),
}

VOP1_OPS = {
    0x00: ("v_nop", "Vector no-op"),
    0x01: ("v_mov_b32_e32", "Vector copy 32-bit"),
    0x02: ("v_readfirstlane_b32", "Read first active lane into SGPR"),
    0x03: ("v_cvt_i32_f64_e32", "Convert float64 to int32"),
    0x04: ("v_cvt_f64_i32_e32", "Convert int32 to float64"),
    0x05: ("v_cvt_f32_i32_e32", "Convert int32 to float32"),
    0x06: ("v_cvt_f32_u32_e32", "Convert uint32 to float32"),
    0x07: ("v_cvt_u32_f32_e32", "Convert float32 to uint32"),
    0x08: ("v_cvt_i32_f32_e32", "Convert float32 to int32"),
    0x0a: ("v_cvt_f16_f32_e32", "Convert float32 to float16"),
    0x0b: ("v_cvt_f32_f16_e32", "Convert float16 to float32"),
    0x0c: ("v_cvt_rpi_i32_f32_e32", "Convert float32 to int32 (round to +inf)"),
    0x0d: ("v_cvt_flr_i32_f32_e32", "Convert float32 to int32 (floor)"),
    0x0e: ("v_cvt_off_f32_i4_e32", "Convert 4-bit signed offset to float32"),
    0x0f: ("v_cvt_f32_f64_e32", "Convert float64 to float32"),
    0x10: ("v_cvt_f64_f32_e32", "Convert float32 to float64"),
    0x11: ("v_cvt_f32_ubyte0_e32", "Convert unsigned byte 0 to float32"),
    0x12: ("v_cvt_f32_ubyte1_e32", "Convert unsigned byte 1 to float32"),
    0x13: ("v_cvt_f32_ubyte2_e32", "Convert unsigned byte 2 to float32"),
    0x14: ("v_cvt_f32_ubyte3_e32", "Convert unsigned byte 3 to float32"),
    0x15: ("v_cvt_u32_f64_e32", "Convert float64 to uint32"),
    0x16: ("v_cvt_f64_u32_e32", "Convert uint32 to float64"),
    0x17: ("v_trunc_f64_e32", "Truncate toward zero float64"),
    0x18: ("v_ceil_f64_e32", "Ceiling float64"),
    0x19: ("v_rndne_f64_e32", "Round to nearest even float64"),
    0x1a: ("v_floor_f64_e32", "Floor float64"),
    0x1b: ("v_pipeflush", "Flush the pipeline (no operands)"),
    0x20: ("v_fract_f32_e32", "Fractional part float32"),
    0x21: ("v_trunc_f32_e32", "Truncate toward zero float32"),
    0x22: ("v_ceil_f32_e32", "Ceiling float32"),
    0x23: ("v_rndne_f32_e32", "Round to nearest even float32"),
    0x24: ("v_floor_f32_e32", "Floor float32"),
    0x25: ("v_exp_f32_e32", "Exponent 2^x float32"),
    0x27: ("v_log_f32_e32", "Logarithm log2(x) float32"),
    0x2a: ("v_rcp_f32_e32", "Reciprocal 1/x float32"),
    0x2b: ("v_rcp_iflag_f32_e32", "Reciprocal integer flag float32"),
    0x2e: ("v_rsq_f32_e32", "Reciprocal square root 1/sqrt(x)"),
    0x2f: ("v_rcp_f64_e32", "Reciprocal 1/x float64"),
    0x31: ("v_rsq_f64_e32", "Reciprocal square root 1/sqrt(x) float64"),
    0x33: ("v_sqrt_f32_e32", "Square root sqrt(x) float32"),
    0x34: ("v_sqrt_f64_e32", "Square root sqrt(x) float64"),
    0x35: ("v_sin_f32_e32", "Sine float32"),
    0x36: ("v_cos_f32_e32", "Cosine float32"),
    0x37: ("v_not_b32_e32", "Vector bitwise NOT"),
    0x38: ("v_bfrev_b32_e32", "Vector bit-reverse"),
    0x39: ("v_ffbh_u32_e32", "Find first bit high uint32"),
    0x3a: ("v_ffbl_b32_e32", "Find first bit low uint32"),
    0x3b: ("v_ffbh_i32_e32", "Find first bit high int32"),
    0x3c: ("v_frexp_exp_i32_f64_e32", "Exponent of float64 as int32"),
    0x3d: ("v_frexp_mant_f64_e32", "Mantissa of float64"),
    0x3e: ("v_fract_f64_e32", "Fractional part float64"),
    0x3f: ("v_frexp_exp_i32_f32_e32", "Exponent of float32 as int32"),
    0x40: ("v_frexp_mant_f32_e32", "Mantissa of float32"),
    0x41: ("v_clrexcp", "Clear the floating-point exception flags (no operands)"),
    0x42: ("v_movreld_b32_e32", "Vector copy with relative destination"),
    0x43: ("v_movrels_b32_e32", "Vector copy with relative source"),
    0x44: ("v_movrelsd_b32_e32", "Vector copy with relative source and destination"),
    0x48: ("v_movrelsd_2_b32_e32", "Vector copy with relative source and destination (extended form)"),
    0x50: ("v_cvt_f16_u16_e32", "Convert uint16 to float16"),
    0x51: ("v_cvt_f16_i16_e32", "Convert int16 to float16"),
    0x52: ("v_cvt_u16_f16_e32", "Convert float16 to uint16"),
    0x53: ("v_cvt_i16_f16_e32", "Convert float16 to int16"),
    0x54: ("v_rcp_f16_e32", "Reciprocal 1/x float16"),
    0x55: ("v_sqrt_f16_e32", "Square root sqrt(x) float16"),
    0x56: ("v_rsq_f16_e32", "Reciprocal square root 1/sqrt(x) float16"),
    0x57: ("v_log_f16_e32", "Logarithm log2(x) float16"),
    0x58: ("v_exp_f16_e32", "Exponent 2^x float16"),
    0x59: ("v_frexp_mant_f16_e32", "Mantissa of float16"),
    0x5a: ("v_frexp_exp_i16_f16_e32", "Exponent of float16 as int16"),
    0x5b: ("v_floor_f16_e32", "Floor float16"),
    0x5c: ("v_ceil_f16_e32", "Ceiling float16"),
    0x5d: ("v_trunc_f16_e32", "Truncate toward zero float16"),
    0x5e: ("v_rndne_f16_e32", "Round to nearest even float16"),
    0x5f: ("v_fract_f16_e32", "Fractional part float16"),
    0x60: ("v_sin_f16_e32", "Sine float16"),
    0x61: ("v_cos_f16_e32", "Cosine float16"),
    0x62: ("v_sat_pk_u8_i16_e32", "Saturate int16 pair to packed uint8"),
    0x63: ("v_cvt_norm_i16_f16_e32", "Convert float16 to normalized int16"),
    0x64: ("v_cvt_norm_u16_f16_e32", "Convert float16 to normalized uint16"),
    0x65: ("v_swap_b32", "Swap the 32-bit values of two VGPRs"),
    0x68: ("v_swaprel_b32", "Swap two VGPRs, relative to M0"),
}

VOPC_OPS = {
    0x00: ("v_cmp_f_f32_e32", "Vector compare v_cmp_f_f32"),
    0x01: ("v_cmp_lt_f32_e32", "Vector compare v_cmp_lt_f32"),
    0x02: ("v_cmp_eq_f32_e32", "Vector compare v_cmp_eq_f32"),
    0x03: ("v_cmp_le_f32_e32", "Vector compare v_cmp_le_f32"),
    0x04: ("v_cmp_gt_f32_e32", "Vector compare v_cmp_gt_f32"),
    0x05: ("v_cmp_lg_f32_e32", "Vector compare v_cmp_lg_f32"),
    0x06: ("v_cmp_ge_f32_e32", "Vector compare v_cmp_ge_f32"),
    0x07: ("v_cmp_o_f32_e32", "Vector compare v_cmp_o_f32"),
    0x08: ("v_cmp_u_f32_e32", "Vector compare v_cmp_u_f32"),
    0x09: ("v_cmp_nge_f32_e32", "Vector compare v_cmp_nge_f32"),
    0x0a: ("v_cmp_nlg_f32_e32", "Vector compare v_cmp_nlg_f32"),
    0x0b: ("v_cmp_ngt_f32_e32", "Vector compare v_cmp_ngt_f32"),
    0x0c: ("v_cmp_nle_f32_e32", "Vector compare v_cmp_nle_f32"),
    0x0d: ("v_cmp_neq_f32_e32", "Vector compare v_cmp_neq_f32"),
    0x0e: ("v_cmp_nlt_f32_e32", "Vector compare v_cmp_nlt_f32"),
    0x0f: ("v_cmp_tru_f32_e32", "Vector compare v_cmp_tru_f32"),
    0x10: ("v_cmpx_f_f32_e32", "Vector compare v_cmpx_f_f32"),
    0x11: ("v_cmpx_lt_f32_e32", "Vector compare v_cmpx_lt_f32"),
    0x12: ("v_cmpx_eq_f32_e32", "Vector compare v_cmpx_eq_f32"),
    0x13: ("v_cmpx_le_f32_e32", "Vector compare v_cmpx_le_f32"),
    0x14: ("v_cmpx_gt_f32_e32", "Vector compare v_cmpx_gt_f32"),
    0x15: ("v_cmpx_lg_f32_e32", "Vector compare v_cmpx_lg_f32"),
    0x16: ("v_cmpx_ge_f32_e32", "Vector compare v_cmpx_ge_f32"),
    0x17: ("v_cmpx_o_f32_e32", "Vector compare v_cmpx_o_f32"),
    0x18: ("v_cmpx_u_f32_e32", "Vector compare v_cmpx_u_f32"),
    0x19: ("v_cmpx_nge_f32_e32", "Vector compare v_cmpx_nge_f32"),
    0x1a: ("v_cmpx_nlg_f32_e32", "Vector compare v_cmpx_nlg_f32"),
    0x1b: ("v_cmpx_ngt_f32_e32", "Vector compare v_cmpx_ngt_f32"),
    0x1c: ("v_cmpx_nle_f32_e32", "Vector compare v_cmpx_nle_f32"),
    0x1d: ("v_cmpx_neq_f32_e32", "Vector compare v_cmpx_neq_f32"),
    0x1e: ("v_cmpx_nlt_f32_e32", "Vector compare v_cmpx_nlt_f32"),
    0x1f: ("v_cmpx_tru_f32_e32", "Vector compare v_cmpx_tru_f32"),
    0x20: ("v_cmp_f_f64_e32", "Vector compare v_cmp_f_f64"),
    0x21: ("v_cmp_lt_f64_e32", "Vector compare v_cmp_lt_f64"),
    0x22: ("v_cmp_eq_f64_e32", "Vector compare v_cmp_eq_f64"),
    0x23: ("v_cmp_le_f64_e32", "Vector compare v_cmp_le_f64"),
    0x24: ("v_cmp_gt_f64_e32", "Vector compare v_cmp_gt_f64"),
    0x25: ("v_cmp_lg_f64_e32", "Vector compare v_cmp_lg_f64"),
    0x26: ("v_cmp_ge_f64_e32", "Vector compare v_cmp_ge_f64"),
    0x27: ("v_cmp_o_f64_e32", "Vector compare v_cmp_o_f64"),
    0x28: ("v_cmp_u_f64_e32", "Vector compare v_cmp_u_f64"),
    0x29: ("v_cmp_nge_f64_e32", "Vector compare v_cmp_nge_f64"),
    0x2a: ("v_cmp_nlg_f64_e32", "Vector compare v_cmp_nlg_f64"),
    0x2b: ("v_cmp_ngt_f64_e32", "Vector compare v_cmp_ngt_f64"),
    0x2c: ("v_cmp_nle_f64_e32", "Vector compare v_cmp_nle_f64"),
    0x2d: ("v_cmp_neq_f64_e32", "Vector compare v_cmp_neq_f64"),
    0x2e: ("v_cmp_nlt_f64_e32", "Vector compare v_cmp_nlt_f64"),
    0x2f: ("v_cmp_tru_f64_e32", "Vector compare v_cmp_tru_f64"),
    0x30: ("v_cmpx_f_f64_e32", "Vector compare v_cmpx_f_f64"),
    0x31: ("v_cmpx_lt_f64_e32", "Vector compare v_cmpx_lt_f64"),
    0x32: ("v_cmpx_eq_f64_e32", "Vector compare v_cmpx_eq_f64"),
    0x33: ("v_cmpx_le_f64_e32", "Vector compare v_cmpx_le_f64"),
    0x34: ("v_cmpx_gt_f64_e32", "Vector compare v_cmpx_gt_f64"),
    0x35: ("v_cmpx_lg_f64_e32", "Vector compare v_cmpx_lg_f64"),
    0x36: ("v_cmpx_ge_f64_e32", "Vector compare v_cmpx_ge_f64"),
    0x37: ("v_cmpx_o_f64_e32", "Vector compare v_cmpx_o_f64"),
    0x38: ("v_cmpx_u_f64_e32", "Vector compare v_cmpx_u_f64"),
    0x39: ("v_cmpx_nge_f64_e32", "Vector compare v_cmpx_nge_f64"),
    0x3a: ("v_cmpx_nlg_f64_e32", "Vector compare v_cmpx_nlg_f64"),
    0x3b: ("v_cmpx_ngt_f64_e32", "Vector compare v_cmpx_ngt_f64"),
    0x3c: ("v_cmpx_nle_f64_e32", "Vector compare v_cmpx_nle_f64"),
    0x3d: ("v_cmpx_neq_f64_e32", "Vector compare v_cmpx_neq_f64"),
    0x3e: ("v_cmpx_nlt_f64_e32", "Vector compare v_cmpx_nlt_f64"),
    0x3f: ("v_cmpx_tru_f64_e32", "Vector compare v_cmpx_tru_f64"),
    0x80: ("v_cmp_f_i32_e32", "Vector compare v_cmp_f_i32"),
    0x81: ("v_cmp_lt_i32_e32", "Vector compare v_cmp_lt_i32"),
    0x82: ("v_cmp_eq_i32_e32", "Vector compare v_cmp_eq_i32"),
    0x83: ("v_cmp_le_i32_e32", "Vector compare v_cmp_le_i32"),
    0x84: ("v_cmp_gt_i32_e32", "Vector compare v_cmp_gt_i32"),
    0x85: ("v_cmp_ne_i32_e32", "Vector compare v_cmp_ne_i32"),
    0x86: ("v_cmp_ge_i32_e32", "Vector compare v_cmp_ge_i32"),
    0x87: ("v_cmp_t_i32_e32", "Vector compare v_cmp_t_i32"),
    0x88: ("v_cmp_class_f32_e32", "Vector compare v_cmp_class_f32"),
    0x89: ("v_cmp_lt_i16_e32", "Vector compare v_cmp_lt_i16"),
    0x8a: ("v_cmp_eq_i16_e32", "Vector compare v_cmp_eq_i16"),
    0x8b: ("v_cmp_le_i16_e32", "Vector compare v_cmp_le_i16"),
    0x8c: ("v_cmp_gt_i16_e32", "Vector compare v_cmp_gt_i16"),
    0x8d: ("v_cmp_ne_i16_e32", "Vector compare v_cmp_ne_i16"),
    0x8e: ("v_cmp_ge_i16_e32", "Vector compare v_cmp_ge_i16"),
    0x8f: ("v_cmp_class_f16_e32", "Vector compare v_cmp_class_f16"),
    0x90: ("v_cmpx_f_i32_e32", "Vector compare v_cmpx_f_i32"),
    0x91: ("v_cmpx_lt_i32_e32", "Vector compare v_cmpx_lt_i32"),
    0x92: ("v_cmpx_eq_i32_e32", "Vector compare v_cmpx_eq_i32"),
    0x93: ("v_cmpx_le_i32_e32", "Vector compare v_cmpx_le_i32"),
    0x94: ("v_cmpx_gt_i32_e32", "Vector compare v_cmpx_gt_i32"),
    0x95: ("v_cmpx_ne_i32_e32", "Vector compare v_cmpx_ne_i32"),
    0x96: ("v_cmpx_ge_i32_e32", "Vector compare v_cmpx_ge_i32"),
    0x97: ("v_cmpx_t_i32_e32", "Vector compare v_cmpx_t_i32"),
    0x98: ("v_cmpx_class_f32_e32", "Vector compare v_cmpx_class_f32"),
    0x99: ("v_cmpx_lt_i16_e32", "Vector compare v_cmpx_lt_i16"),
    0x9a: ("v_cmpx_eq_i16_e32", "Vector compare v_cmpx_eq_i16"),
    0x9b: ("v_cmpx_le_i16_e32", "Vector compare v_cmpx_le_i16"),
    0x9c: ("v_cmpx_gt_i16_e32", "Vector compare v_cmpx_gt_i16"),
    0x9d: ("v_cmpx_ne_i16_e32", "Vector compare v_cmpx_ne_i16"),
    0x9e: ("v_cmpx_ge_i16_e32", "Vector compare v_cmpx_ge_i16"),
    0x9f: ("v_cmpx_class_f16_e32", "Vector compare v_cmpx_class_f16"),
    0xa0: ("v_cmp_f_i64_e32", "Vector compare v_cmp_f_i64"),
    0xa1: ("v_cmp_lt_i64_e32", "Vector compare v_cmp_lt_i64"),
    0xa2: ("v_cmp_eq_i64_e32", "Vector compare v_cmp_eq_i64"),
    0xa3: ("v_cmp_le_i64_e32", "Vector compare v_cmp_le_i64"),
    0xa4: ("v_cmp_gt_i64_e32", "Vector compare v_cmp_gt_i64"),
    0xa5: ("v_cmp_ne_i64_e32", "Vector compare v_cmp_ne_i64"),
    0xa6: ("v_cmp_ge_i64_e32", "Vector compare v_cmp_ge_i64"),
    0xa7: ("v_cmp_t_i64_e32", "Vector compare v_cmp_t_i64"),
    0xa8: ("v_cmp_class_f64_e32", "Vector compare v_cmp_class_f64"),
    0xa9: ("v_cmp_lt_u16_e32", "Vector compare v_cmp_lt_u16"),
    0xaa: ("v_cmp_eq_u16_e32", "Vector compare v_cmp_eq_u16"),
    0xab: ("v_cmp_le_u16_e32", "Vector compare v_cmp_le_u16"),
    0xac: ("v_cmp_gt_u16_e32", "Vector compare v_cmp_gt_u16"),
    0xad: ("v_cmp_ne_u16_e32", "Vector compare v_cmp_ne_u16"),
    0xae: ("v_cmp_ge_u16_e32", "Vector compare v_cmp_ge_u16"),
    0xb0: ("v_cmpx_f_i64_e32", "Vector compare v_cmpx_f_i64"),
    0xb1: ("v_cmpx_lt_i64_e32", "Vector compare v_cmpx_lt_i64"),
    0xb2: ("v_cmpx_eq_i64_e32", "Vector compare v_cmpx_eq_i64"),
    0xb3: ("v_cmpx_le_i64_e32", "Vector compare v_cmpx_le_i64"),
    0xb4: ("v_cmpx_gt_i64_e32", "Vector compare v_cmpx_gt_i64"),
    0xb5: ("v_cmpx_ne_i64_e32", "Vector compare v_cmpx_ne_i64"),
    0xb6: ("v_cmpx_ge_i64_e32", "Vector compare v_cmpx_ge_i64"),
    0xb7: ("v_cmpx_t_i64_e32", "Vector compare v_cmpx_t_i64"),
    0xb8: ("v_cmpx_class_f64_e32", "Vector compare v_cmpx_class_f64"),
    0xb9: ("v_cmpx_lt_u16_e32", "Vector compare v_cmpx_lt_u16"),
    0xba: ("v_cmpx_eq_u16_e32", "Vector compare v_cmpx_eq_u16"),
    0xbb: ("v_cmpx_le_u16_e32", "Vector compare v_cmpx_le_u16"),
    0xbc: ("v_cmpx_gt_u16_e32", "Vector compare v_cmpx_gt_u16"),
    0xbd: ("v_cmpx_ne_u16_e32", "Vector compare v_cmpx_ne_u16"),
    0xbe: ("v_cmpx_ge_u16_e32", "Vector compare v_cmpx_ge_u16"),
    0xc0: ("v_cmp_f_u32_e32", "Vector compare v_cmp_f_u32"),
    0xc1: ("v_cmp_lt_u32_e32", "Vector compare v_cmp_lt_u32"),
    0xc2: ("v_cmp_eq_u32_e32", "Vector compare v_cmp_eq_u32"),
    0xc3: ("v_cmp_le_u32_e32", "Vector compare v_cmp_le_u32"),
    0xc4: ("v_cmp_gt_u32_e32", "Vector compare v_cmp_gt_u32"),
    0xc5: ("v_cmp_ne_u32_e32", "Vector compare v_cmp_ne_u32"),
    0xc6: ("v_cmp_ge_u32_e32", "Vector compare v_cmp_ge_u32"),
    0xc7: ("v_cmp_t_u32_e32", "Vector compare v_cmp_t_u32"),
    0xc8: ("v_cmp_f_f16_e32", "Vector compare v_cmp_f_f16"),
    0xc9: ("v_cmp_lt_f16_e32", "Vector compare v_cmp_lt_f16"),
    0xca: ("v_cmp_eq_f16_e32", "Vector compare v_cmp_eq_f16"),
    0xcb: ("v_cmp_le_f16_e32", "Vector compare v_cmp_le_f16"),
    0xcc: ("v_cmp_gt_f16_e32", "Vector compare v_cmp_gt_f16"),
    0xcd: ("v_cmp_lg_f16_e32", "Vector compare v_cmp_lg_f16"),
    0xce: ("v_cmp_ge_f16_e32", "Vector compare v_cmp_ge_f16"),
    0xcf: ("v_cmp_o_f16_e32", "Vector compare v_cmp_o_f16"),
    0xd0: ("v_cmpx_f_u32_e32", "Vector compare v_cmpx_f_u32"),
    0xd1: ("v_cmpx_lt_u32_e32", "Vector compare v_cmpx_lt_u32"),
    0xd2: ("v_cmpx_eq_u32_e32", "Vector compare v_cmpx_eq_u32"),
    0xd3: ("v_cmpx_le_u32_e32", "Vector compare v_cmpx_le_u32"),
    0xd4: ("v_cmpx_gt_u32_e32", "Vector compare v_cmpx_gt_u32"),
    0xd5: ("v_cmpx_ne_u32_e32", "Vector compare v_cmpx_ne_u32"),
    0xd6: ("v_cmpx_ge_u32_e32", "Vector compare v_cmpx_ge_u32"),
    0xd7: ("v_cmpx_t_u32_e32", "Vector compare v_cmpx_t_u32"),
    0xd8: ("v_cmpx_f_f16_e32", "Vector compare v_cmpx_f_f16"),
    0xd9: ("v_cmpx_lt_f16_e32", "Vector compare v_cmpx_lt_f16"),
    0xda: ("v_cmpx_eq_f16_e32", "Vector compare v_cmpx_eq_f16"),
    0xdb: ("v_cmpx_le_f16_e32", "Vector compare v_cmpx_le_f16"),
    0xdc: ("v_cmpx_gt_f16_e32", "Vector compare v_cmpx_gt_f16"),
    0xdd: ("v_cmpx_lg_f16_e32", "Vector compare v_cmpx_lg_f16"),
    0xde: ("v_cmpx_ge_f16_e32", "Vector compare v_cmpx_ge_f16"),
    0xdf: ("v_cmpx_o_f16_e32", "Vector compare v_cmpx_o_f16"),
    0xe0: ("v_cmp_f_u64_e32", "Vector compare v_cmp_f_u64"),
    0xe1: ("v_cmp_lt_u64_e32", "Vector compare v_cmp_lt_u64"),
    0xe2: ("v_cmp_eq_u64_e32", "Vector compare v_cmp_eq_u64"),
    0xe3: ("v_cmp_le_u64_e32", "Vector compare v_cmp_le_u64"),
    0xe4: ("v_cmp_gt_u64_e32", "Vector compare v_cmp_gt_u64"),
    0xe5: ("v_cmp_ne_u64_e32", "Vector compare v_cmp_ne_u64"),
    0xe6: ("v_cmp_ge_u64_e32", "Vector compare v_cmp_ge_u64"),
    0xe7: ("v_cmp_t_u64_e32", "Vector compare v_cmp_t_u64"),
    0xe8: ("v_cmp_u_f16_e32", "Vector compare v_cmp_u_f16"),
    0xe9: ("v_cmp_nge_f16_e32", "Vector compare v_cmp_nge_f16"),
    0xea: ("v_cmp_nlg_f16_e32", "Vector compare v_cmp_nlg_f16"),
    0xeb: ("v_cmp_ngt_f16_e32", "Vector compare v_cmp_ngt_f16"),
    0xec: ("v_cmp_nle_f16_e32", "Vector compare v_cmp_nle_f16"),
    0xed: ("v_cmp_neq_f16_e32", "Vector compare v_cmp_neq_f16"),
    0xee: ("v_cmp_nlt_f16_e32", "Vector compare v_cmp_nlt_f16"),
    0xef: ("v_cmp_tru_f16_e32", "Vector compare v_cmp_tru_f16"),
    0xf0: ("v_cmpx_f_u64_e32", "Vector compare v_cmpx_f_u64"),
    0xf1: ("v_cmpx_lt_u64_e32", "Vector compare v_cmpx_lt_u64"),
    0xf2: ("v_cmpx_eq_u64_e32", "Vector compare v_cmpx_eq_u64"),
    0xf3: ("v_cmpx_le_u64_e32", "Vector compare v_cmpx_le_u64"),
    0xf4: ("v_cmpx_gt_u64_e32", "Vector compare v_cmpx_gt_u64"),
    0xf5: ("v_cmpx_ne_u64_e32", "Vector compare v_cmpx_ne_u64"),
    0xf6: ("v_cmpx_ge_u64_e32", "Vector compare v_cmpx_ge_u64"),
    0xf7: ("v_cmpx_t_u64_e32", "Vector compare v_cmpx_t_u64"),
    0xf8: ("v_cmpx_u_f16_e32", "Vector compare v_cmpx_u_f16"),
    0xf9: ("v_cmpx_nge_f16_e32", "Vector compare v_cmpx_nge_f16"),
    0xfa: ("v_cmpx_nlg_f16_e32", "Vector compare v_cmpx_nlg_f16"),
    0xfb: ("v_cmpx_ngt_f16_e32", "Vector compare v_cmpx_ngt_f16"),
    0xfc: ("v_cmpx_nle_f16_e32", "Vector compare v_cmpx_nle_f16"),
    0xfd: ("v_cmpx_neq_f16_e32", "Vector compare v_cmpx_neq_f16"),
    0xfe: ("v_cmpx_nlt_f16_e32", "Vector compare v_cmpx_nlt_f16"),
    0xff: ("v_cmpx_tru_f16_e32", "Vector compare v_cmpx_tru_f16"),
}


VOP2_OPS = {
    0x01: ("v_cndmask_b32_e32", "Vector conditional mask select"),
    0x03: ("v_add_f32_e32", "Vector add float32"),
    0x04: ("v_sub_f32_e32", "Vector subtract float32"),
    0x05: ("v_subrev_f32_e32", "Vector reverse subtract float32"),
    0x06: ("v_mac_legacy_f32_e32", "Vector multiply-accumulate zero-preserving float32"),
    0x07: ("v_mul_legacy_f32_e32", "Vector multiply zero-preserving float32"),
    0x08: ("v_mul_f32_e32", "Vector multiply float32"),
    0x09: ("v_mul_i32_i24_e32", "Vector multiply 24-bit signed integer"),
    0x0a: ("v_mul_hi_i32_i24_e32", "Vector multiply high 24-bit signed integer"),
    0x0b: ("v_mul_u32_u24_e32", "Vector multiply 24-bit unsigned integer"),
    0x0c: ("v_mul_hi_u32_u24_e32", "Vector multiply high 24-bit unsigned integer"),
    0x0f: ("v_min_f32_e32", "Vector minimum float32"),
    0x10: ("v_max_f32_e32", "Vector maximum float32"),
    0x11: ("v_min_i32_e32", "Vector minimum int32"),
    0x12: ("v_max_i32_e32", "Vector maximum int32"),
    0x13: ("v_min_u32_e32", "Vector minimum uint32"),
    0x14: ("v_max_u32_e32", "Vector maximum uint32"),
    0x16: ("v_lshrrev_b32_e32", "Vector shift right logical"),
    0x18: ("v_ashrrev_i32_e32", "Vector shift right arithmetic"),
    0x1a: ("v_lshlrev_b32_e32", "Vector shift left logical"),
    0x1b: ("v_and_b32_e32", "Vector bitwise AND"),
    0x1c: ("v_or_b32_e32", "Vector bitwise OR"),
    0x1d: ("v_xor_b32_e32", "Vector bitwise XOR"),
    0x1e: ("v_xnor_b32_e32", "Vector bitwise XNOR"),
    0x1f: ("v_mac_f32_e32", "Vector multiply-accumulate float32"),
    0x20: ("v_madmk_f32", "Vector multiply-add (non-fused) with literal multiplier"),
    0x21: ("v_madak_f32", "Vector multiply-add (non-fused) with literal addend"),
    0x25: ("v_add_nc_u32_e32", "Vector add uint32 without carry"),
    0x26: ("v_sub_nc_u32_e32", "Vector sub uint32 without carry"),
    0x27: ("v_subrev_nc_u32_e32", "Vector reverse sub uint32 without carry"),
    0x28: ("v_add_co_ci_u32_e32", "Vector add with carry-in and carry-out uint32"),
    0x29: ("v_sub_co_ci_u32_e32", "Vector sub with carry-in and carry-out uint32"),
    0x2a: ("v_subrev_co_ci_u32_e32", "Vector reverse sub with carry-in and carry-out uint32"),
    0x2b: ("v_fmac_f32_e32", "Vector fused multiply-accumulate float32"),
    0x2c: ("v_fmamk_f32", "Vector fused multiply-add with literal multiplier"),
    0x2d: ("v_fmaak_f32", "Vector fused multiply-add with literal addend"),
    0x2f: ("v_cvt_pkrtz_f16_f32_e32", "Convert two float32 to packed float16 (round to zero)"),
    0x32: ("v_add_f16_e32", "Vector add float16"),
    0x33: ("v_sub_f16_e32", "Vector subtract float16"),
    0x34: ("v_subrev_f16_e32", "Vector reverse subtract float16"),
    0x35: ("v_mul_f16_e32", "Vector multiply float16"),
    0x36: ("v_fmac_f16_e32", "Vector fused multiply-accumulate float16"),
    0x37: ("v_fmamk_f16", "Vector fused multiply-add float16 with literal multiplier"),
    0x38: ("v_fmaak_f16", "Vector fused multiply-add float16 with literal addend"),
    0x39: ("v_max_f16_e32", "Vector maximum float16"),
    0x3a: ("v_min_f16_e32", "Vector minimum float16"),
    0x3b: ("v_ldexp_f16_e32", "Vector scale float16 by a power of two"),
    0x3c: ("v_pk_fmac_f16", "Vector packed fused multiply-accumulate float16"),
}

VOP3_NATIVE_OPS = {
    0x128: ("v_add_co_ci_u32_e64", "V_ADD_CO_CI_U32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x129: ("v_sub_co_ci_u32_e64", "V_SUB_CO_CI_U32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x12a: ("v_subrev_co_ci_u32_e64", "V_SUBREV_CO_CI_U32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x140: ("v_mad_legacy_f32", "V_MAD_LEGACY_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x141: ("v_mad_f32", "V_MAD_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x142: ("v_mad_i32_i24", "V_MAD_I32_I24 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x143: ("v_mad_u32_u24", "V_MAD_U32_U24 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x144: ("v_cubeid_f32", "V_CUBEID_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x145: ("v_cubesc_f32", "V_CUBESC_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x146: ("v_cubetc_f32", "V_CUBETC_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x147: ("v_cubema_f32", "V_CUBEMA_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x148: ("v_bfe_u32", "V_BFE_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x149: ("v_bfe_i32", "V_BFE_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14a: ("v_bfi_b32", "V_BFI_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14b: ("v_fma_f32", "V_FMA_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14c: ("v_fma_f64", "V_FMA_F64 (VOP3A)", 3, False, 2, (2, 2, 2)),
    0x14d: ("v_lerp_u8", "V_LERP_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14e: ("v_alignbit_b32", "V_ALIGNBIT_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14f: ("v_alignbyte_b32", "V_ALIGNBYTE_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x150: ("v_mullit_f32", "V_MULLIT_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x151: ("v_min3_f32", "V_MIN3_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x152: ("v_min3_i32", "V_MIN3_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x153: ("v_min3_u32", "V_MIN3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x154: ("v_max3_f32", "V_MAX3_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x155: ("v_max3_i32", "V_MAX3_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x156: ("v_max3_u32", "V_MAX3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x157: ("v_med3_f32", "V_MED3_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x158: ("v_med3_i32", "V_MED3_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x159: ("v_med3_u32", "V_MED3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15a: ("v_sad_u8", "V_SAD_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15b: ("v_sad_hi_u8", "V_SAD_HI_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15c: ("v_sad_u16", "V_SAD_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15d: ("v_sad_u32", "V_SAD_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15e: ("v_cvt_pk_u8_f32", "V_CVT_PK_U8_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15f: ("v_div_fixup_f32", "V_DIV_FIXUP_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x160: ("v_div_fixup_f64", "V_DIV_FIXUP_F64 (VOP3A)", 3, False, 2, (2, 2, 2)),
    0x164: ("v_add_f64", "V_ADD_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x165: ("v_mul_f64", "V_MUL_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x166: ("v_min_f64", "V_MIN_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x167: ("v_max_f64", "V_MAX_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x168: ("v_ldexp_f64", "V_LDEXP_F64 (VOP3A)", 2, False, 2, (2, 1)),
    0x169: ("v_mul_lo_u32", "V_MUL_LO_U32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16a: ("v_mul_hi_u32", "V_MUL_HI_U32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16b: ("v_mul_lo_i32", "V_MUL_LO_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16c: ("v_mul_hi_i32", "V_MUL_HI_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16d: ("v_div_scale_f32", "V_DIV_SCALE_F32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x16e: ("v_div_scale_f64", "V_DIV_SCALE_F64 (VOP3B)", 3, True, 2, (2, 2, 2)),
    0x16f: ("v_div_fmas_f32", "V_DIV_FMAS_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x170: ("v_div_fmas_f64", "V_DIV_FMAS_F64 (VOP3A)", 3, False, 2, (2, 2, 2)),
    0x171: ("v_msad_u8", "V_MSAD_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x172: ("v_qsad_pk_u16_u8", "V_QSAD_PK_U16_U8 (VOP3A)", 3, False, 2, (2, 1, 2)),
    0x173: ("v_mqsad_pk_u16_u8", "V_MQSAD_PK_U16_U8 (VOP3A)", 3, False, 2, (2, 1, 2)),
    0x174: ("v_trig_preop_f64", "V_TRIG_PREOP_F64 (VOP3A)", 2, False, 2, (2, 1)),
    0x175: ("v_mqsad_u32_u8", "V_MQSAD_U32_U8 (VOP3A)", 3, False, 4, (2, 1, 4)),
    0x176: ("v_mad_u64_u32", "V_MAD_U64_U32 (VOP3B)", 3, True, 2, (1, 1, 2)),
    0x177: ("v_mad_i64_i32", "V_MAD_I64_I32 (VOP3B)", 3, True, 2, (1, 1, 2)),
    0x178: ("v_xor3_b32", "V_XOR3_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x202: ("v_interp_mov_f32_e64", "V_INTERP_MOV_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x2ff: ("v_lshlrev_b64", "V_LSHLREV_B64 (VOP3A)", 2, False, 2, (1, 2)),
    0x300: ("v_lshrrev_b64", "V_LSHRREV_B64 (VOP3A)", 2, False, 2, (1, 2)),
    0x301: ("v_ashrrev_i64", "V_ASHRREV_I64 (VOP3A)", 2, False, 2, (1, 2)),
    0x303: ("v_add_nc_u16", "V_ADD_NC_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x304: ("v_sub_nc_u16", "V_SUB_NC_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x305: ("v_mul_lo_u16", "V_MUL_LO_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x307: ("v_lshrrev_b16", "V_LSHRREV_B16 (VOP3A)", 2, False, 1, (1, 1)),
    0x308: ("v_ashrrev_i16", "V_ASHRREV_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x309: ("v_max_u16", "V_MAX_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30a: ("v_max_i16", "V_MAX_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30b: ("v_min_u16", "V_MIN_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30c: ("v_min_i16", "V_MIN_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30d: ("v_add_nc_i16", "V_ADD_NC_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30e: ("v_sub_nc_i16", "V_SUB_NC_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30f: ("v_add_co_u32", "V_ADD_CO_U32 (VOP3B)", 2, True, 1, (1, 1)),
    0x310: ("v_sub_co_u32", "V_SUB_CO_U32 (VOP3B)", 2, True, 1, (1, 1)),
    0x311: ("v_pack_b32_f16", "V_PACK_B32_F16 (VOP3A)", 2, False, 1, (1, 1)),
    0x312: ("v_cvt_pknorm_i16_f16", "V_CVT_PKNORM_I16_F16 (VOP3A)", 2, False, 1, (1, 1)),
    0x313: ("v_cvt_pknorm_u16_f16", "V_CVT_PKNORM_U16_F16 (VOP3A)", 2, False, 1, (1, 1)),
    0x314: ("v_lshlrev_b16", "V_LSHLREV_B16 (VOP3A)", 2, False, 1, (1, 1)),
    0x319: ("v_subrev_co_u32", "V_SUBREV_CO_U32 (VOP3B)", 2, True, 1, (1, 1)),
    0x340: ("v_mad_u16", "V_MAD_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x344: ("v_perm_b32", "V_PERM_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x345: ("v_xad_u32", "V_XAD_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x346: ("v_lshl_add_u32", "V_LSHL_ADD_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x347: ("v_add_lshl_u32", "V_ADD_LSHL_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x34b: ("v_fma_f16", "V_FMA_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x351: ("v_min3_f16", "V_MIN3_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x352: ("v_min3_i16", "V_MIN3_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x353: ("v_min3_u16", "V_MIN3_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x354: ("v_max3_f16", "V_MAX3_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x355: ("v_max3_i16", "V_MAX3_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x356: ("v_max3_u16", "V_MAX3_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x357: ("v_med3_f16", "V_MED3_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x358: ("v_med3_i16", "V_MED3_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x359: ("v_med3_u16", "V_MED3_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x35e: ("v_mad_i16", "V_MAD_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x35f: ("v_div_fixup_f16", "V_DIV_FIXUP_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x360: ("v_readlane_b32", "Read lane from vector register into SGPR (VOP3A)", 2, False, 1, (1, 1)),
    0x361: ("v_writelane_b32", "Write lane into vector register from scalar (VOP3A)", 2, False, 1, (1, 1)),
    0x362: ("v_ldexp_f32", "V_LDEXP_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x363: ("v_bfm_b32", "V_BFM_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x364: ("v_bcnt_u32_b32", "V_BCNT_U32_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x365: ("v_mbcnt_lo_u32_b32", "V_MBCNT_LO_U32_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x366: ("v_mbcnt_hi_u32_b32", "V_MBCNT_HI_U32_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x368: ("v_cvt_pknorm_i16_f32", "V_CVT_PKNORM_I16_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x369: ("v_cvt_pknorm_u16_f32", "V_CVT_PKNORM_U16_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x36a: ("v_cvt_pk_u16_u32", "V_CVT_PK_U16_U32 (VOP3A)", 2, False, 1, (1, 1)),
    0x36b: ("v_cvt_pk_i16_i32", "V_CVT_PK_I16_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x36d: ("v_add3_u32", "V_ADD3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x36f: ("v_lshl_or_b32", "V_LSHL_OR_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x371: ("v_and_or_b32", "V_AND_OR_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x372: ("v_or3_b32", "V_OR3_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x373: ("v_mad_u32_u16", "V_MAD_U32_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x375: ("v_mad_i32_i16", "V_MAD_I32_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x376: ("v_sub_nc_i32", "V_SUB_NC_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x37f: ("v_add_nc_i32", "V_ADD_NC_I32 (VOP3A)", 2, False, 1, (1, 1)),
}


# gfx1013 VOP3 forms whose modifier grammar is narrower than the generic VOP3A decode:
# name -> (clamp allowed, omod allowed). llvm-mc rejects abs and neg on all of them, and both the
# assembler and the disassembler ignore the op_sel bits, so a word with those bits set is not decoded.
VOP3_RESTRICTED_MODIFIERS = {
    "v_mul_lo_i32": (False, False),
    "v_qsad_pk_u16_u8": (True, False),
    "v_mqsad_pk_u16_u8": (True, False),
    "v_mqsad_u32_u8": (True, False),
    "v_interp_mov_f32_e64": (True, True),
}

# The v_interp_mov_f32 parameter selector (VINTRP vsrc field, VOP3A src1 field): p10, p20 and p0.
# LLVM prints any other value as invalid_param_<n> and its assembler accepts none of them.
VINTRP_MOV_PARAMS = ("p10", "p20", "p0")


def _format_vop3_vgpr_dst(vdst: int, width: int) -> str:
    """Format a VOP3 VGPR destination of 1, 2 or 4 registers; a group running past v255 is flagged, not printed."""
    if vdst + width > 256:
        return {2: "<invalid VGPR pair>", 4: "<invalid VGPR quad>"}[width]
    return _format_vgpr_range(vdst, width)


def _vop3_output_modifiers(clamp: int, omod: int) -> str:
    """Return the trailing clamp and output-modifier spelling of a VOP3 instruction."""
    text = " clamp" if clamp else ""
    return text + {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")


def _vop3_restricted_encoding_is_valid(op_name: str, arity: int, w0: int, w1: int) -> bool:
    """Check a restricted VOP3 word against the modifier grammar llvm-mc accepts for gfx1013."""
    allow_clamp, allow_omod = VOP3_RESTRICTED_MODIFIERS[op_name]
    if (w0 >> 8) & 0x7 or (w1 >> 29) & 0x7:
        return False
    if ((w0 >> 15) & 1 and not allow_clamp) or ((w1 >> 27) & 0x3 and not allow_omod):
        return False
    if arity < 3 and (w1 >> 18) & 0x1ff:
        return False
    if op_name == "v_mqsad_u32_u8" and not 256 <= (w1 >> 18) & 0x1ff <= 511:
        # The accumulator is a VGPR quad: llvm-mc rejects an SGPR, constant or special-register src2.
        return False
    if op_name == "v_interp_mov_f32_e64":
        # src0 holds attr[5:0] and channel[7:6] only; src1 selects p10, p20 or p0.
        return not (w1 & 0x100) and ((w1 >> 9) & 0x1ff) < len(VINTRP_MOV_PARAMS)
    return True


def _decode_interp_mov_e64(w0: int, w1: int, comment: str) -> tuple[int, str, str, str]:
    """Decode the VOP3 spelling of v_interp_mov_f32 for a word that already passed validation."""
    attr = w1 & 0x3f
    channel = "xyzw"[(w1 >> 6) & 0x3]
    param = VINTRP_MOV_PARAMS[(w1 >> 9) & 0x1ff]
    asm = f"v_interp_mov_f32_e64 v{w0 & 0xff}, {param}, attr{attr}.{channel}"
    asm += _vop3_output_modifiers((w0 >> 15) & 1, (w1 >> 27) & 0x3)
    return (8, "v_interp_mov_f32_e64", asm, comment)


VOP3P_OPS = {
    0x00: ("v_pk_mad_i16", "Multiply-add signed 16-bit packed"),
    0x01: ("v_pk_mul_lo_u16", "Multiply low unsigned 16-bit packed"),
    0x02: ("v_pk_add_i16", "Add signed 16-bit packed"),
    0x03: ("v_pk_sub_i16", "Subtract signed 16-bit packed"),
    0x04: ("v_pk_lshlrev_b16", "Logical shift left reverse 16-bit packed"),
    0x05: ("v_pk_lshrrev_b16", "Logical shift right reverse 16-bit packed"),
    0x06: ("v_pk_ashrrev_i16", "Arithmetic shift right reverse 16-bit packed"),
    0x07: ("v_pk_max_i16", "Maximum signed 16-bit packed"),
    0x08: ("v_pk_min_i16", "Minimum signed 16-bit packed"),
    0x09: ("v_pk_mad_u16", "Multiply-add unsigned 16-bit packed"),
    0x0a: ("v_pk_add_u16", "Add unsigned 16-bit packed"),
    0x0b: ("v_pk_sub_u16", "Subtract unsigned 16-bit packed"),
    0x0c: ("v_pk_max_u16", "Maximum unsigned 16-bit packed"),
    0x0d: ("v_pk_min_u16", "Minimum unsigned 16-bit packed"),
    0x0e: ("v_pk_fma_f16", "Fused multiply-add float16 packed"),
    0x0f: ("v_pk_add_f16", "Add float16 packed"),
    0x10: ("v_pk_mul_f16", "Multiply float16 packed"),
    0x11: ("v_pk_min_f16", "Minimum float16 packed"),
    0x12: ("v_pk_max_f16", "Maximum float16 packed"),
    0x13: ("v_dot2_f32_f16", "Dot product 2-element float16 accumulate to float32"),
    0x14: ("v_dot2_i32_i16", "Dot product 2-element signed 16-bit accumulate to int32"),
    0x15: ("v_dot2_u32_u16", "Dot product 2-element unsigned 16-bit accumulate to uint32"),
    0x16: ("v_dot4_i32_i8", "Dot product 4-element signed 8-bit accumulate to int32"),
    0x17: ("v_dot4_u32_u8", "Dot product 4-element unsigned 8-bit accumulate to uint32"),
    0x18: ("v_dot8_i32_i4", "Dot product 8-element signed 4-bit accumulate to int32"),
    0x19: ("v_dot8_u32_u4", "Dot product 8-element unsigned 4-bit accumulate to uint32"),
    0x20: ("v_fma_mix_f32", "Fused multiply-add mixed precision to float32"),
    0x21: ("v_fma_mixlo_f16", "Fused multiply-add mixed precision to float16 low"),
    0x22: ("v_fma_mixhi_f16", "Fused multiply-add mixed precision to float16 high"),
}

VOP3P_3SRC = {
    0x00, 0x09, 0x0e, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22,
}

EXP_TARGETS = {
    0: "mrt0", 1: "mrt1", 2: "mrt2", 3: "mrt3",
    4: "mrt4", 5: "mrt5", 6: "mrt6", 7: "mrt7",
    8: "mrtz", 9: "null",
    12: "pos0", 13: "pos1", 14: "pos2", 15: "pos3",
    **{32 + i: f"param{i}" for i in range(32)},
}

# GFX1013 / RDNA 2 MUBUF opcodes: op -> (mnemonic, vdata_width, comment)
MUBUF_OPS = {
    0x00: ("buffer_load_format_x", 1, "Untyped/typed buffer format load 1 dword"),
    0x01: ("buffer_load_format_xy", 2, "Untyped/typed buffer format load 2 dwords"),
    0x02: ("buffer_load_format_xyz", 3, "Untyped/typed buffer format load 3 dwords"),
    0x03: ("buffer_load_format_xyzw", 4, "Untyped/typed buffer format load 4 dwords"),
    0x04: ("buffer_store_format_x", 1, "Buffer format store 1 dword"),
    0x05: ("buffer_store_format_xy", 2, "Buffer format store 2 dwords"),
    0x06: ("buffer_store_format_xyz", 3, "Buffer format store 3 dwords"),
    0x07: ("buffer_store_format_xyzw", 4, "Buffer format store 4 dwords"),
    0x08: ("buffer_load_ubyte", 1, "Buffer load unsigned byte"),
    0x09: ("buffer_load_sbyte", 1, "Buffer load signed byte"),
    0x0a: ("buffer_load_ushort", 1, "Buffer load unsigned 16-bit"),
    0x0b: ("buffer_load_sshort", 1, "Buffer load signed 16-bit"),
    0x0c: ("buffer_load_dword", 1, "Buffer load 1 dword"),
    0x0d: ("buffer_load_dwordx2", 2, "Buffer load 2 dwords"),
    0x0e: ("buffer_load_dwordx4", 4, "Buffer load 4 dwords"),
    0x0f: ("buffer_load_dwordx3", 3, "Buffer load 3 dwords"),
    0x18: ("buffer_store_byte", 1, "Buffer store byte"),
    0x19: ("buffer_store_byte_d16_hi", 1, "Buffer store byte from high 16-bit lane"),
    0x1a: ("buffer_store_short", 1, "Buffer store 16-bit short"),
    0x1b: ("buffer_store_short_d16_hi", 1, "Buffer store 16-bit short from high lane"),
    0x1c: ("buffer_store_dword", 1, "Buffer store 1 dword"),
    0x1d: ("buffer_store_dwordx2", 2, "Buffer store 2 dwords"),
    0x1e: ("buffer_store_dwordx4", 4, "Buffer store 4 dwords"),
    0x1f: ("buffer_store_dwordx3", 3, "Buffer store 3 dwords"),
    0x20: ("buffer_load_ubyte_d16", 1, "Buffer load unsigned byte to low 16 bits"),
    0x21: ("buffer_load_ubyte_d16_hi", 1, "Buffer load unsigned byte to high 16 bits"),
    0x22: ("buffer_load_sbyte_d16", 1, "Buffer load signed byte to low 16 bits"),
    0x23: ("buffer_load_sbyte_d16_hi", 1, "Buffer load signed byte to high 16 bits"),
    0x24: ("buffer_load_short_d16", 1, "Buffer load 16-bit short to low 16 bits"),
    0x25: ("buffer_load_short_d16_hi", 1, "Buffer load 16-bit short to high 16 bits"),
    0x26: ("buffer_load_format_d16_hi_x", 1, "Buffer format load of one 16-bit component to the high 16 bits"),
    0x27: ("buffer_store_format_d16_hi_x", 1, "Buffer format store of one 16-bit component from the high 16 bits"),
    0x30: ("buffer_atomic_swap", 1, "Buffer atomic swap 32-bit"),
    0x31: ("buffer_atomic_cmpswap", 2, "Buffer atomic compare-and-swap 32-bit"),
    0x32: ("buffer_atomic_add", 1, "Buffer atomic add 32-bit"),
    0x33: ("buffer_atomic_sub", 1, "Buffer atomic subtract 32-bit"),
    0x35: ("buffer_atomic_smin", 1, "Buffer atomic signed min 32-bit"),
    0x36: ("buffer_atomic_umin", 1, "Buffer atomic unsigned min 32-bit"),
    0x37: ("buffer_atomic_smax", 1, "Buffer atomic signed max 32-bit"),
    0x38: ("buffer_atomic_umax", 1, "Buffer atomic unsigned max 32-bit"),
    0x39: ("buffer_atomic_and", 1, "Buffer atomic bitwise AND 32-bit"),
    0x3a: ("buffer_atomic_or", 1, "Buffer atomic bitwise OR 32-bit"),
    0x3b: ("buffer_atomic_xor", 1, "Buffer atomic bitwise XOR 32-bit"),
    0x3c: ("buffer_atomic_inc", 1, "Buffer atomic increment 32-bit"),
    0x3d: ("buffer_atomic_dec", 1, "Buffer atomic decrement 32-bit"),
    0x3e: ("buffer_atomic_fcmpswap", 2, "Buffer atomic float32 compare-and-swap"),
    0x3f: ("buffer_atomic_fmin", 1, "Buffer atomic float32 min"),
    0x40: ("buffer_atomic_fmax", 1, "Buffer atomic float32 max"),
    0x50: ("buffer_atomic_swap_x2", 2, "Buffer atomic swap 64-bit"),
    0x51: ("buffer_atomic_cmpswap_x2", 4, "Buffer atomic compare-and-swap 64-bit"),
    0x52: ("buffer_atomic_add_x2", 2, "Buffer atomic add 64-bit"),
    0x53: ("buffer_atomic_sub_x2", 2, "Buffer atomic subtract 64-bit"),
    0x55: ("buffer_atomic_smin_x2", 2, "Buffer atomic signed min 64-bit"),
    0x56: ("buffer_atomic_umin_x2", 2, "Buffer atomic unsigned min 64-bit"),
    0x57: ("buffer_atomic_smax_x2", 2, "Buffer atomic signed max 64-bit"),
    0x58: ("buffer_atomic_umax_x2", 2, "Buffer atomic unsigned max 64-bit"),
    0x59: ("buffer_atomic_and_x2", 2, "Buffer atomic bitwise AND 64-bit"),
    0x5a: ("buffer_atomic_or_x2", 2, "Buffer atomic bitwise OR 64-bit"),
    0x5b: ("buffer_atomic_xor_x2", 2, "Buffer atomic bitwise XOR 64-bit"),
    0x5c: ("buffer_atomic_inc_x2", 2, "Buffer atomic increment 64-bit"),
    0x5d: ("buffer_atomic_dec_x2", 2, "Buffer atomic decrement 64-bit"),
    0x5e: ("buffer_atomic_fcmpswap_x2", 4, "Buffer atomic float64 compare-and-swap"),
    0x5f: ("buffer_atomic_fmin_x2", 2, "Buffer atomic float64 min"),
    0x60: ("buffer_atomic_fmax_x2", 2, "Buffer atomic float64 max"),
    0x71: ("buffer_gl0_inv", 0, "Invalidate GL0 vector cache"),
    0x72: ("buffer_gl1_inv", 0, "Invalidate GL1 vector cache"),
    0x80: ("buffer_load_format_d16_x", 1, "Buffer format load 1 packed 16-bit channel"),
    0x81: ("buffer_load_format_d16_xy", 1, "Buffer format load 2 packed 16-bit channels"),
    0x82: ("buffer_load_format_d16_xyz", 2, "Buffer format load 3 packed 16-bit channels"),
    0x83: ("buffer_load_format_d16_xyzw", 2, "Buffer format load 4 packed 16-bit channels"),
    0x84: ("buffer_store_format_d16_x", 1, "Buffer format store 1 packed 16-bit channel"),
    0x85: ("buffer_store_format_d16_xy", 1, "Buffer format store 2 packed 16-bit channels"),
    0x86: ("buffer_store_format_d16_xyz", 2, "Buffer format store 3 packed 16-bit channels"),
    0x87: ("buffer_store_format_d16_xyzw", 2, "Buffer format store 4 packed 16-bit channels"),
}

# The two high-half single-component format ops: spelled by llvm-mc 19 only (llvm-mc 15 rejects them on gfx1013).
MUBUF_D16_HI_FORMAT_OPS = {"buffer_load_format_d16_hi_x", "buffer_store_format_d16_hi_x"}

# GFX1013 / RDNA 2 MTBUF opcodes: op -> (mnemonic, vdata_width, comment)
MTBUF_OPS = {
    0x00: ("tbuffer_load_format_x", 1, "Typed buffer load 1 channel"),
    0x01: ("tbuffer_load_format_xy", 2, "Typed buffer load 2 channels"),
    0x02: ("tbuffer_load_format_xyz", 3, "Typed buffer load 3 channels"),
    0x03: ("tbuffer_load_format_xyzw", 4, "Typed buffer load 4 channels"),
    0x04: ("tbuffer_store_format_x", 1, "Typed buffer store 1 channel"),
    0x05: ("tbuffer_store_format_xy", 2, "Typed buffer store 2 channels"),
    0x06: ("tbuffer_store_format_xyz", 3, "Typed buffer store 3 channels"),
    0x07: ("tbuffer_store_format_xyzw", 4, "Typed buffer store 4 channels"),
    0x08: ("tbuffer_load_format_d16_x", 1, "Typed buffer load 1 packed 16-bit channel"),
    0x09: ("tbuffer_load_format_d16_xy", 1, "Typed buffer load 2 packed 16-bit channels"),
    0x0a: ("tbuffer_load_format_d16_xyz", 2, "Typed buffer load 3 packed 16-bit channels"),
    0x0b: ("tbuffer_load_format_d16_xyzw", 2, "Typed buffer load 4 packed 16-bit channels"),
    0x0c: ("tbuffer_store_format_d16_x", 1, "Typed buffer store 1 packed 16-bit channel"),
    0x0d: ("tbuffer_store_format_d16_xy", 1, "Typed buffer store 2 packed 16-bit channels"),
    0x0e: ("tbuffer_store_format_d16_xyz", 2, "Typed buffer store 3 packed 16-bit channels"),
    0x0f: ("tbuffer_store_format_d16_xyzw", 2, "Typed buffer store 4 packed 16-bit channels"),
}

# GFX1013 MTBUF 7-bit format identifiers; format 1 (BUF_FMT_8_UNORM) is LLVM's omitted default.
MTBUF_FORMATS = {
    0: "BUF_FMT_INVALID",
    2: "BUF_FMT_8_SNORM",
    3: "BUF_FMT_8_USCALED",
    4: "BUF_FMT_8_SSCALED",
    5: "BUF_FMT_8_UINT",
    6: "BUF_FMT_8_SINT",
    7: "BUF_FMT_16_UNORM",
    8: "BUF_FMT_16_SNORM",
    9: "BUF_FMT_16_USCALED",
    10: "BUF_FMT_16_SSCALED",
    11: "BUF_FMT_16_UINT",
    12: "BUF_FMT_16_SINT",
    13: "BUF_FMT_16_FLOAT",
    14: "BUF_FMT_8_8_UNORM",
    15: "BUF_FMT_8_8_SNORM",
    16: "BUF_FMT_8_8_USCALED",
    17: "BUF_FMT_8_8_SSCALED",
    18: "BUF_FMT_8_8_UINT",
    19: "BUF_FMT_8_8_SINT",
    20: "BUF_FMT_32_UINT",
    21: "BUF_FMT_32_SINT",
    22: "BUF_FMT_32_FLOAT",
    23: "BUF_FMT_16_16_UNORM",
    24: "BUF_FMT_16_16_SNORM",
    25: "BUF_FMT_16_16_USCALED",
    26: "BUF_FMT_16_16_SSCALED",
    27: "BUF_FMT_16_16_UINT",
    28: "BUF_FMT_16_16_SINT",
    29: "BUF_FMT_16_16_FLOAT",
    30: "BUF_FMT_10_11_11_UNORM",
    31: "BUF_FMT_10_11_11_SNORM",
    32: "BUF_FMT_10_11_11_USCALED",
    33: "BUF_FMT_10_11_11_SSCALED",
    34: "BUF_FMT_10_11_11_UINT",
    35: "BUF_FMT_10_11_11_SINT",
    36: "BUF_FMT_10_11_11_FLOAT",
    37: "BUF_FMT_11_11_10_UNORM",
    38: "BUF_FMT_11_11_10_SNORM",
    39: "BUF_FMT_11_11_10_USCALED",
    40: "BUF_FMT_11_11_10_SSCALED",
    41: "BUF_FMT_11_11_10_UINT",
    42: "BUF_FMT_11_11_10_SINT",
    43: "BUF_FMT_11_11_10_FLOAT",
    44: "BUF_FMT_10_10_10_2_UNORM",
    45: "BUF_FMT_10_10_10_2_SNORM",
    46: "BUF_FMT_10_10_10_2_USCALED",
    47: "BUF_FMT_10_10_10_2_SSCALED",
    48: "BUF_FMT_10_10_10_2_UINT",
    49: "BUF_FMT_10_10_10_2_SINT",
    50: "BUF_FMT_2_10_10_10_UNORM",
    51: "BUF_FMT_2_10_10_10_SNORM",
    52: "BUF_FMT_2_10_10_10_USCALED",
    53: "BUF_FMT_2_10_10_10_SSCALED",
    54: "BUF_FMT_2_10_10_10_UINT",
    55: "BUF_FMT_2_10_10_10_SINT",
    56: "BUF_FMT_8_8_8_8_UNORM",
    57: "BUF_FMT_8_8_8_8_SNORM",
    58: "BUF_FMT_8_8_8_8_USCALED",
    59: "BUF_FMT_8_8_8_8_SSCALED",
    60: "BUF_FMT_8_8_8_8_UINT",
    61: "BUF_FMT_8_8_8_8_SINT",
    62: "BUF_FMT_32_32_UINT",
    63: "BUF_FMT_32_32_SINT",
    64: "BUF_FMT_32_32_FLOAT",
    65: "BUF_FMT_16_16_16_16_UNORM",
    66: "BUF_FMT_16_16_16_16_SNORM",
    67: "BUF_FMT_16_16_16_16_USCALED",
    68: "BUF_FMT_16_16_16_16_SSCALED",
    69: "BUF_FMT_16_16_16_16_UINT",
    70: "BUF_FMT_16_16_16_16_SINT",
    71: "BUF_FMT_16_16_16_16_FLOAT",
    72: "BUF_FMT_32_32_32_UINT",
    73: "BUF_FMT_32_32_32_SINT",
    74: "BUF_FMT_32_32_32_FLOAT",
    75: "BUF_FMT_32_32_32_32_UINT",
    76: "BUF_FMT_32_32_32_32_SINT",
    77: "BUF_FMT_32_32_32_32_FLOAT",
}


# VOP1 and VOP2 SRC0 codes that announce a second instruction dword instead of naming an operand:
# DPP8 (233, or 234 with FI set), SDWA (249) and DPP16 (250). The dword holds the real SRC0 and
# the lane or sub-dword controls.
VOP_EXTENSION_FORMS = {233: "DPP8", 234: "DPP8", 249: "SDWA", 250: "DPP16"}
VOP1_DPP_UNSUPPORTED = frozenset(("v_readfirstlane_b32", "v_pipeflush", "v_clrexcp", "v_swap_b32", "v_swaprel_b32"))
VOP2_DPP_UNSUPPORTED_OPS = frozenset((0x06, 0x20, 0x21, 0x2c, 0x2d, 0x37, 0x38, 0x3c))
SDWA_SEL_NAMES = ("BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD")
SDWA_DST_UNUSED_NAMES = ("UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE")
SDWA_OMOD_NAMES = ("", " mul:2", " mul:4", " div:2")
VOP1_SDWA_UNSUPPORTED = frozenset(("v_readfirstlane_b32", "v_pipeflush", "v_clrexcp", "v_swap_b32", "v_swaprel_b32"))
VOP1_NO_OPERAND_OPS = frozenset(("v_pipeflush", "v_clrexcp"))
VOP1_SDWA_VGPR_ONLY_SRC0 = frozenset(("v_movrels_b32", "v_movrelsd_b32", "v_movrelsd_2_b32"))
VOP2_SDWA_UNSUPPORTED_OPS = frozenset((0x02, 0x06, 0x0d, 0x1f, 0x20, 0x21, 0x2b, 0x2c, 0x2d, 0x36, 0x37, 0x38, 0x3c))
# Scalar source codes 209-234 are outside the supported LLVM 15 decoder subset for GFX10 SDWA: LLVM 15
# refuses every VOPC SDWA record that names one as src0 or src1. This records LLVM 15's behaviour; it is
# not a claim about the ISA.
SDWA_LLVM15_UNSUPPORTED_SCALAR_CODES = range(209, 235)


def _format_sdwa_float_modifiers(val: str, scalar_code: int | None, is_abs: int, is_neg: int) -> str:
    """Wrap a float SDWA source in abs/neg modifiers as LLVM 15 prints them.

    neg without abs on an inline constant (scalar codes 128-208 and 240-248) is printed `neg(c)`, as LLVM 15
    does: `-1` would name the inline constant -1 (0xffffffff), not 1 with its sign bit flipped (0x80000001).
    """
    inline_constant = scalar_code is not None and (128 <= scalar_code <= 208 or 240 <= scalar_code <= 248)
    if inline_constant and is_neg and not is_abs:
        return f"neg({val})"
    return _format_operand_modifiers(val, is_abs, is_neg)


def _decode_vopc_sdwa(w1: int, vsrc1: int, op_name: str, comment: str) -> tuple[int, str, str, str]:
    """Decode a GFX10 VOP_SDWA9Be (VOPC SDWA) 8-byte instruction."""
    if op_name.startswith("v_vopc_") or op_name.endswith(("_f64_e32", "_i64_e32", "_u64_e32")):
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "VOPC SDWA is not supported for this opcode")
    src0_sel = (w1 >> 16) & 0x7
    src1_sel = (w1 >> 24) & 0x7
    if src0_sel >= len(SDWA_SEL_NAMES) or src1_sel >= len(SDWA_SEL_NAMES):
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOPC SDWA selector")
    sdwa_name = f"{op_name[:-4]}_sdwa" if op_name.endswith("_e32") else f"{op_name}_sdwa"
    src0_raw = w1 & 0xff
    sdst = (w1 >> 8) & 0x7f
    sdst_en = (w1 >> 15) & 1
    src0_sext = (w1 >> 19) & 1
    src0_neg = (w1 >> 20) & 1
    src0_abs = (w1 >> 21) & 1
    src0_sgpr = (w1 >> 23) & 1
    src1_sext = (w1 >> 27) & 1
    src1_neg = (w1 >> 28) & 1
    src1_abs = (w1 >> 29) & 1
    src1_sgpr = (w1 >> 31) & 1

    is_cmpx = op_name.startswith("v_cmpx_")
    if is_cmpx and sdst_en:
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "VOPCX SDWA forbids sdst_en")
    if (src0_sgpr and src0_raw in (249, 250, 255)) or (src1_sgpr and vsrc1 in (249, 250, 255)):
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOPC SDWA scalar operand")
    if ((src0_sgpr and src0_raw in SDWA_LLVM15_UNSUPPORTED_SCALAR_CODES)
            or (src1_sgpr and vsrc1 in SDWA_LLVM15_UNSUPPORTED_SCALAR_CODES)):
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>",
                "VOPC SDWA scalar operand outside the supported LLVM 15 decoder subset")

    is_float = op_name.endswith(("_f32_e32", "_f16_e32"))
    is_class = "_class_" in op_name
    src0_is_float = is_float
    src1_is_float = is_float and not is_class
    # Reject modifier bits disallowed by VOP_SDWA9e for the operand type (LLVM 15 refuses these).
    if (src0_is_float and src0_sext) or (not src0_is_float and (src0_neg or src0_abs)):
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOPC SDWA src0 modifier")
    if (src1_is_float and src1_sext) or (not src1_is_float and (src1_neg or src1_abs)):
        return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOPC SDWA src1 modifier")

    src0_str = decode_s_reg(src0_raw) if src0_sgpr else f"v{src0_raw}"
    src1_str = decode_s_reg(vsrc1) if src1_sgpr else f"v{vsrc1}"

    if src0_is_float:
        src0_str = _format_sdwa_float_modifiers(src0_str, src0_raw if src0_sgpr else None, src0_abs, src0_neg)
    elif src0_sext:
        src0_str = f"sext({src0_str})"
    if src1_is_float:
        src1_str = _format_sdwa_float_modifiers(src1_str, vsrc1 if src1_sgpr else None, src1_abs, src1_neg)
    elif src1_sext:
        src1_str = f"sext({src1_str})"

    sels = f"src0_sel:{SDWA_SEL_NAMES[src0_sel]} src1_sel:{SDWA_SEL_NAMES[src1_sel]}"
    if is_cmpx:
        asm = f"{sdwa_name} {src0_str}, {src1_str} {sels}"
    else:
        dst_str = decode_s_reg(sdst) if sdst_en else "vcc_lo"
        asm = f"{sdwa_name} {dst_str}, {src0_str}, {src1_str} {sels}"
    return (8, sdwa_name, asm, comment)


def _format_sdwa_source(raw: int, is_sgpr: int, is_float: bool, is_sext: int, is_neg: int, is_abs: int) -> str:
    """Format a GFX10 SDWA source operand with type-appropriate modifiers."""
    val = decode_s_reg(raw) if is_sgpr else f"v{raw}"
    if is_float:
        return _format_sdwa_float_modifiers(val, raw if is_sgpr else None, is_abs, is_neg)
    if is_sext:
        return f"sext({val})"
    return val


def _vop1_sdwa_spec(op_name: str) -> tuple[bool, bool, bool] | None:
    """Return (src0_is_float, dst_is_float, src0_vgpr_only) or None if VOP1 SDWA is unsupported."""
    if op_name.startswith("v_vop1_") or op_name in VOP1_SDWA_UNSUPPORTED:
        return None
    if (op_name.startswith(("v_cvt_f64_", "v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64", "v_frexp_exp_i32_f64"))
            or op_name.endswith(("_f64_e32", "_f64", "_b64_e32", "_b64"))):
        return None
    base = op_name[:-4] if op_name.endswith("_e32") else op_name
    src0_is_float = base.endswith(("_f32", "_f16"))
    dst_is_float = (
        "_f32_" in base
        or "_f16_" in base
        or (src0_is_float and not any(tok in base for tok in ("_i32_", "_u32_", "_i16_", "_u16_")))
    )
    return (src0_is_float, dst_is_float, base in VOP1_SDWA_VGPR_ONLY_SRC0)


def _decode_vop1_sdwa(w1: int, vdst: int, op_name: str, comment: str) -> tuple[int, str, str, str]:
    """Decode a GFX10 VOP_SDWA9Ae (VOP1 SDWA) 8-byte instruction."""
    if op_name == "v_nop":
        # VOP_NOP_PROFILE allows V_NOP_sdwa_gfx10 only when vdst and all fixed-zero SDWA control fields are zero.
        if vdst == 0 and (w1 & 0xbfffe0ff) == 0:
            return (8, "v_nop", "v_nop", comment)
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP1 SDWA encoding")
    spec = _vop1_sdwa_spec(op_name)
    if spec is None:
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "VOP1 SDWA is not supported for this opcode")
    if (w1 >> 24) & 0xbf:
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP1 SDWA src1 field")
    src0_is_float, dst_is_float, src0_vgpr_only = spec
    dst_sel = (w1 >> 8) & 0x7
    dst_unused = (w1 >> 11) & 0x3
    clamp = (w1 >> 13) & 1
    omod = (w1 >> 14) & 0x3
    src0_sel = (w1 >> 16) & 0x7
    if dst_sel >= len(SDWA_SEL_NAMES) or dst_unused >= len(SDWA_DST_UNUSED_NAMES) or src0_sel >= len(SDWA_SEL_NAMES):
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP1 SDWA selector")
    if omod and not dst_is_float:
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP1 SDWA omod modifier")

    src0_raw = w1 & 0xff
    src0_sext = (w1 >> 19) & 1
    src0_neg = (w1 >> 20) & 1
    src0_abs = (w1 >> 21) & 1
    src0_sgpr = (w1 >> 23) & 1
    if src0_sgpr and src0_vgpr_only:
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "VOP1 SDWA source operand must be a VGPR")
    if src0_sgpr and src0_raw in (249, 250, 255):
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP1 SDWA scalar operand")
    if src0_sgpr and src0_raw in SDWA_LLVM15_UNSUPPORTED_SCALAR_CODES:
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>",
                "VOP1 SDWA scalar operand outside the supported LLVM 15 decoder subset")
    if (src0_is_float and src0_sext) or (not src0_is_float and (src0_neg or src0_abs)):
        return (8, "vop1_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP1 SDWA src0 modifier")

    sdwa_name = f"{op_name[:-4]}_sdwa" if op_name.endswith("_e32") else f"{op_name}_sdwa"
    src0_str = _format_sdwa_source(src0_raw, src0_sgpr, src0_is_float, src0_sext, src0_neg, src0_abs)
    out_mods = f"{' clamp' if clamp else ''}{SDWA_OMOD_NAMES[omod]}"
    sels = (f"dst_sel:{SDWA_SEL_NAMES[dst_sel]} dst_unused:{SDWA_DST_UNUSED_NAMES[dst_unused]} "
            f"src0_sel:{SDWA_SEL_NAMES[src0_sel]}")
    return (8, sdwa_name, f"{sdwa_name} v{vdst}, {src0_str}{out_mods} {sels}", comment)


def _vop2_sdwa_spec(op: int, op_name: str) -> tuple[bool, bool, bool] | None:
    """Return (src0_is_float, src1_is_float, dst_is_float) or None if VOP2 SDWA is unsupported."""
    if op_name.startswith("v_vop2_") or op in VOP2_SDWA_UNSUPPORTED_OPS:
        return None
    base = op_name[:-4] if op_name.endswith("_e32") else op_name
    if base == "v_ldexp_f16":
        return (True, False, True)
    if base == "v_cvt_pkrtz_f16_f32":
        return (True, True, True)
    is_float = base.endswith(("_f32", "_f16"))
    return (is_float, is_float, is_float)


def _decode_vop2_sdwa(w1: int, vdst: int, vsrc1: int, op: int, op_name: str, comment: str) -> tuple[int, str, str, str]:
    """Decode a GFX10 VOP_SDWA9Ae (VOP2 SDWA) 8-byte instruction."""
    spec = _vop2_sdwa_spec(op, op_name)
    if spec is None:
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>", "VOP2 SDWA is not supported for this opcode")
    src0_is_float, src1_is_float, dst_is_float = spec
    dst_sel = (w1 >> 8) & 0x7
    dst_unused = (w1 >> 11) & 0x3
    clamp = (w1 >> 13) & 1
    omod = (w1 >> 14) & 0x3
    src0_sel = (w1 >> 16) & 0x7
    src1_sel = (w1 >> 24) & 0x7
    if (dst_sel >= len(SDWA_SEL_NAMES) or dst_unused >= len(SDWA_DST_UNUSED_NAMES)
            or src0_sel >= len(SDWA_SEL_NAMES) or src1_sel >= len(SDWA_SEL_NAMES)):
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP2 SDWA selector")
    if omod and not dst_is_float:
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP2 SDWA omod modifier")

    src0_raw = w1 & 0xff
    src0_sext = (w1 >> 19) & 1
    src0_neg = (w1 >> 20) & 1
    src0_abs = (w1 >> 21) & 1
    src0_sgpr = (w1 >> 23) & 1
    src1_sext = (w1 >> 27) & 1
    src1_neg = (w1 >> 28) & 1
    src1_abs = (w1 >> 29) & 1
    src1_sgpr = (w1 >> 31) & 1
    if (src0_sgpr and src0_raw in (249, 250, 255)) or (src1_sgpr and vsrc1 in (249, 250, 255)):
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP2 SDWA scalar operand")
    if ((src0_sgpr and src0_raw in SDWA_LLVM15_UNSUPPORTED_SCALAR_CODES)
            or (src1_sgpr and vsrc1 in SDWA_LLVM15_UNSUPPORTED_SCALAR_CODES)):
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>",
                "VOP2 SDWA scalar operand outside the supported LLVM 15 decoder subset")
    if (src0_is_float and src0_sext) or (not src0_is_float and (src0_neg or src0_abs)):
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP2 SDWA src0 modifier")
    if (src1_is_float and src1_sext) or (not src1_is_float and (src1_neg or src1_abs)):
        return (8, "vop2_sdwa", f"{op_name} <unsupported SDWA>", "Invalid VOP2 SDWA src1 modifier")

    sdwa_name = f"{op_name[:-4]}_sdwa" if op_name.endswith("_e32") else f"{op_name}_sdwa"
    src0_str = _format_sdwa_source(src0_raw, src0_sgpr, src0_is_float, src0_sext, src0_neg, src0_abs)
    src1_str = _format_sdwa_source(vsrc1, src1_sgpr, src1_is_float, src1_sext, src1_neg, src1_abs)
    out_mods = f"{' clamp' if clamp else ''}{SDWA_OMOD_NAMES[omod]}"
    sels = (f"dst_sel:{SDWA_SEL_NAMES[dst_sel]} dst_unused:{SDWA_DST_UNUSED_NAMES[dst_unused]} "
            f"src0_sel:{SDWA_SEL_NAMES[src0_sel]} src1_sel:{SDWA_SEL_NAMES[src1_sel]}")
    if op_name == "v_cndmask_b32_e32":
        asm = f"{sdwa_name} v{vdst}, {src0_str}, {src1_str}, vcc_lo{out_mods} {sels}"
    elif op in (0x28, 0x29, 0x2a):
        asm = f"{sdwa_name} v{vdst}, vcc_lo, {src0_str}, {src1_str}, vcc_lo{out_mods} {sels}"
    else:
        asm = f"{sdwa_name} v{vdst}, {src0_str}, {src1_str}{out_mods} {sels}"
    return (8, sdwa_name, asm, comment)


def _vop1_dpp_has_mods(op_name: str) -> bool | None:
    """Return whether VOP1 DPP16 supports float input modifiers, or None if DPP is unsupported."""
    if op_name.startswith("v_vop1_") or op_name in VOP1_DPP_UNSUPPORTED:
        return None
    if (op_name.startswith(("v_cvt_f64_", "v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64", "v_frexp_exp_i32_f64"))
            or op_name.endswith(("_f64_e32", "_f64", "_b64_e32", "_b64"))):
        return None
    base = op_name[:-4] if op_name.endswith("_e32") else op_name
    return base.endswith(("_f32", "_f16"))


def _vop2_dpp_mods(op: int, op_name: str) -> tuple[bool, bool] | None:
    """Return (src0_has_mods, src1_has_mods) for VOP2 DPP16, or None if DPP is unsupported."""
    if op_name.startswith("v_vop2_") or op in VOP2_DPP_UNSUPPORTED_OPS:
        return None
    base = op_name[:-4] if op_name.endswith("_e32") else op_name
    if base == "v_ldexp_f16":
        return (True, False)
    if base == "v_cvt_pkrtz_f16_f32":
        return (True, True)
    is_float = base.endswith(("_f32", "_f16"))
    return (is_float, is_float)


def _format_dpp16_ctrl(dpp_ctrl: int) -> str | None:
    """Decode a 9-bit GFX10 DPP16 control field, or return None if invalid on GFX10."""
    if dpp_ctrl <= 0xff:
        return (f"quad_perm:[{dpp_ctrl & 0x3},{(dpp_ctrl >> 2) & 0x3},"
                f"{(dpp_ctrl >> 4) & 0x3},{(dpp_ctrl >> 6) & 0x3}]")
    if 0x101 <= dpp_ctrl <= 0x10f:
        return f"row_shl:{dpp_ctrl & 0xf}"
    if 0x111 <= dpp_ctrl <= 0x11f:
        return f"row_shr:{dpp_ctrl & 0xf}"
    if 0x121 <= dpp_ctrl <= 0x12f:
        return f"row_ror:{dpp_ctrl & 0xf}"
    if dpp_ctrl == 0x140:
        return "row_mirror"
    if dpp_ctrl == 0x141:
        return "row_half_mirror"
    if 0x150 <= dpp_ctrl <= 0x15f:
        return f"row_share:{dpp_ctrl & 0xf}"
    if 0x160 <= dpp_ctrl <= 0x16f:
        return f"row_xmask:{dpp_ctrl & 0xf}"
    return None


def _format_dpp8_ctrl(w1: int, fi: int) -> str:
    """Format the 8-lane DPP8 selector list and optional fi:1 flag."""
    lanes = ",".join(str((w1 >> (8 + 3 * k)) & 0x7) for k in range(8))
    return f"dpp8:[{lanes}]{' fi:1' if fi else ''}"


def _decode_vop1_dpp(w1: int, vdst: int, src0: int, op_name: str, comment: str) -> tuple[int, str, str, str]:
    """Decode a GFX10 VOP1 DPP8 (src0=233/234) or DPP16 (src0=250) 8-byte instruction."""
    is_dpp8 = src0 in (233, 234)
    form = "DPP8" if is_dpp8 else "DPP16"
    kind = "vop1_dpp8" if is_dpp8 else "vop1_dpp16"
    src0_mods = _vop1_dpp_has_mods(op_name)
    if src0_mods is None:
        return (8, kind, f"{op_name} <unsupported {form}>", f"VOP1 {form} is not supported for this opcode")

    src0_vgpr = w1 & 0xff
    if is_dpp8:
        fi = 1 if src0 == 234 else 0
        if op_name == "v_nop":
            if vdst == 0 and src0_vgpr == 0:
                return (8, "v_nop", f"v_nop {_format_dpp8_ctrl(w1, fi)}", comment)
            return (8, kind, f"{op_name} <unsupported DPP8>", "Invalid VOP1 DPP8 encoding")
        dpp_name = f"{op_name[:-4]}_dpp" if op_name.endswith("_e32") else f"{op_name}_dpp"
        return (8, dpp_name, f"{dpp_name} v{vdst}, v{src0_vgpr} {_format_dpp8_ctrl(w1, fi)}", comment)

    dpp_ctrl = (w1 >> 8) & 0x1ff
    dpp_ctrl_str = _format_dpp16_ctrl(dpp_ctrl)
    if dpp_ctrl_str is None:
        return (8, kind, f"{op_name} <unsupported DPP16>", "Invalid GFX10 DPP16 control")
    fi = (w1 >> 18) & 1
    bound_ctrl = (w1 >> 19) & 1
    src0_neg = (w1 >> 20) & 1
    src0_abs = (w1 >> 21) & 1
    src1_neg = (w1 >> 22) & 1
    src1_abs = (w1 >> 23) & 1
    bank_mask = (w1 >> 24) & 0xf
    row_mask = (w1 >> 28) & 0xf
    if src1_neg or src1_abs or (not src0_mods and (src0_neg or src0_abs)):
        return (8, kind, f"{op_name} <unsupported DPP16>", "Invalid VOP1 DPP16 modifier")
    suffix = (f"{dpp_ctrl_str} row_mask:0x{row_mask:x} bank_mask:0x{bank_mask:x}"
              f"{' bound_ctrl:1' if bound_ctrl else ''}{' fi:1' if fi else ''}")
    if op_name == "v_nop":
        if vdst == 0 and src0_vgpr == 0:
            return (8, "v_nop", f"v_nop {suffix}", comment)
        return (8, kind, f"{op_name} <unsupported DPP16>", "Invalid VOP1 DPP16 encoding")
    dpp_name = f"{op_name[:-4]}_dpp" if op_name.endswith("_e32") else f"{op_name}_dpp"
    src0_str = _format_operand_modifiers(f"v{src0_vgpr}", src0_abs, src0_neg)
    return (8, dpp_name, f"{dpp_name} v{vdst}, {src0_str} {suffix}", comment)


def _decode_vop2_dpp(w1: int, vdst: int, vsrc1: int, src0: int, op: int, op_name: str,
                     comment: str) -> tuple[int, str, str, str]:
    """Decode a GFX10 VOP2 DPP8 (src0=233/234) or DPP16 (src0=250) 8-byte instruction."""
    is_dpp8 = src0 in (233, 234)
    form = "DPP8" if is_dpp8 else "DPP16"
    kind = "vop2_dpp8" if is_dpp8 else "vop2_dpp16"
    mods = _vop2_dpp_mods(op, op_name)
    if mods is None:
        return (8, kind, f"{op_name} <unsupported {form}>", f"VOP2 {form} is not supported for this opcode")
    src0_has_mods, src1_has_mods = mods
    src0_vgpr = w1 & 0xff
    dpp_name = f"{op_name[:-4]}_dpp" if op_name.endswith("_e32") else f"{op_name}_dpp"

    if is_dpp8:
        fi = 1 if src0 == 234 else 0
        ctrl = _format_dpp8_ctrl(w1, fi)
        src0_str = f"v{src0_vgpr}"
        src1_str = f"v{vsrc1}"
    else:
        dpp_ctrl = (w1 >> 8) & 0x1ff
        dpp_ctrl_str = _format_dpp16_ctrl(dpp_ctrl)
        if dpp_ctrl_str is None:
            return (8, kind, f"{op_name} <unsupported DPP16>", "Invalid GFX10 DPP16 control")
        fi = (w1 >> 18) & 1
        bound_ctrl = (w1 >> 19) & 1
        src0_neg = (w1 >> 20) & 1
        src0_abs = (w1 >> 21) & 1
        src1_neg = (w1 >> 22) & 1
        src1_abs = (w1 >> 23) & 1
        bank_mask = (w1 >> 24) & 0xf
        row_mask = (w1 >> 28) & 0xf
        if (not src0_has_mods and (src0_neg or src0_abs)) or (not src1_has_mods and (src1_neg or src1_abs)):
            return (8, kind, f"{op_name} <unsupported DPP16>", "Invalid VOP2 DPP16 modifier")
        src0_str = _format_operand_modifiers(f"v{src0_vgpr}", src0_abs, src0_neg)
        src1_str = _format_operand_modifiers(f"v{vsrc1}", src1_abs, src1_neg)
        ctrl = (f"{dpp_ctrl_str} row_mask:0x{row_mask:x} bank_mask:0x{bank_mask:x}"
                f"{' bound_ctrl:1' if bound_ctrl else ''}{' fi:1' if fi else ''}")

    if op_name == "v_cndmask_b32_e32":
        asm = f"{dpp_name} v{vdst}, {src0_str}, {src1_str}, vcc_lo {ctrl}"
    elif op in (0x28, 0x29, 0x2a):
        asm = f"{dpp_name} v{vdst}, vcc_lo, {src0_str}, {src1_str}, vcc_lo {ctrl}"
    else:
        asm = f"{dpp_name} v{vdst}, {src0_str}, {src1_str} {ctrl}"
    return (8, dpp_name, asm, comment)


def _vop_extension(data: bytes, offset: int, encoding: str, op_name: str, src0: int) -> tuple[int, str, str, str]:
    """Consume a DPP8, DPP16 or SDWA instruction as one 8-byte record; its operands are not decoded."""
    form = VOP_EXTENSION_FORMS[src0]
    if offset + 8 > len(data):
        return (4, f"{encoding}_truncated", f"{op_name} <truncated>", f"Incomplete {encoding.upper()} {form}")
    base = op_name[:-4] if op_name.endswith("_e32") else op_name
    mnemonic = f"{base}_sdwa" if form == "SDWA" else f"{base}_dpp"
    return (8, f"{encoding}_{form.lower()}", f"{mnemonic} <unsupported {form}>",
            f"{encoding.upper()} {form} is not decoded")


def _format_vgpr_range(reg: int, width: int) -> str:
    """Format a VGPR operand covering `width` consecutive 32-bit registers."""
    return f"v{reg}" if width == 1 else f"v[{reg}:{reg + width - 1}]"


def _format_srsrc4(srsrc: int) -> str:
    """Format a 4-SGPR buffer resource descriptor (s[n:n+3] or ttmp[n:n+3])."""
    if 108 <= srsrc <= 120:
        t = srsrc - 108
        return f"ttmp[{t}:{t + 3}]"
    return f"s[{srsrc}:{srsrc + 3}]"


def _format_buffer_operands_and_mods(
    w0: int,
    w1: int,
    vdata_width: int,
    *,
    format_mod: str | None = None,
    allow_lds: bool = False,
    allow_tfe: bool = True,
) -> tuple[list[str], list[str]]:
    """Decode operands and modifiers shared by MUBUF and MTBUF encodings."""
    imm_offset = w0 & 0xfff
    offen = (w0 >> 12) & 1
    idxen = (w0 >> 13) & 1
    glc = (w0 >> 14) & 1
    dlc = (w0 >> 15) & 1
    lds = ((w0 >> 16) & 1) if allow_lds else 0

    vaddr = w1 & 0xff
    vdata = (w1 >> 8) & 0xff
    srsrc = ((w1 >> 16) & 0x1f) * 4
    slc = (w1 >> 22) & 1
    tfe = (w1 >> 23) & 1
    soffset = (w1 >> 24) & 0xff

    vaddr_str = f"v[{vaddr}:{vaddr + 1}]" if (idxen and offen) else (f"v{vaddr}" if (idxen or offen) else "off")
    srsrc_str = _format_srsrc4(srsrc)
    soffset_str = decode_s_reg(soffset)

    if lds:
        operands = [vaddr_str, srsrc_str, soffset_str]
    else:
        operands = [_format_vgpr_range(vdata, vdata_width), vaddr_str, srsrc_str, soffset_str]

    mods = []
    if format_mod is not None:
        mods.append(format_mod)
    if idxen:
        mods.append("idxen")
    if offen:
        mods.append("offen")
    if imm_offset:
        mods.append(f"offset:{imm_offset}")
    if glc:
        mods.append("glc")
    if slc:
        mods.append("slc")
    if dlc:
        mods.append("dlc")
    if lds:
        mods.append("lds")
    if allow_tfe and tfe:
        mods.append("tfe")
    return operands, mods

# GFX1013 / RDNA 2 FLAT/GLOBAL/SCRATCH base opcodes:
# op -> (suffix, kind, vdst_width, vdata_width, description)
FLAT_BASE_OPS = {
    0x08: ("load_ubyte", "load", 1, 0, "load unsigned byte"),
    0x09: ("load_sbyte", "load", 1, 0, "load signed byte"),
    0x0a: ("load_ushort", "load", 1, 0, "load unsigned 16-bit"),
    0x0b: ("load_sshort", "load", 1, 0, "load signed 16-bit"),
    0x0c: ("load_dword", "load", 1, 0, "load 1 dword"),
    0x0d: ("load_dwordx2", "load", 2, 0, "load 2 dwords"),
    0x0e: ("load_dwordx4", "load", 4, 0, "load 4 dwords"),
    0x0f: ("load_dwordx3", "load", 3, 0, "load 3 dwords"),
    0x18: ("store_byte", "store", 0, 1, "store byte"),
    0x19: ("store_byte_d16_hi", "store", 0, 1, "store byte from high 16-bit lane"),
    0x1a: ("store_short", "store", 0, 1, "store 16-bit short"),
    0x1b: ("store_short_d16_hi", "store", 0, 1, "store 16-bit short from high lane"),
    0x1c: ("store_dword", "store", 0, 1, "store 1 dword"),
    0x1d: ("store_dwordx2", "store", 0, 2, "store 2 dwords"),
    0x1e: ("store_dwordx4", "store", 0, 4, "store 4 dwords"),
    0x1f: ("store_dwordx3", "store", 0, 3, "store 3 dwords"),
    0x20: ("load_ubyte_d16", "load", 1, 0, "load unsigned byte to low 16 bits"),
    0x21: ("load_ubyte_d16_hi", "load", 1, 0, "load unsigned byte to high 16 bits"),
    0x22: ("load_sbyte_d16", "load", 1, 0, "load signed byte to low 16 bits"),
    0x23: ("load_sbyte_d16_hi", "load", 1, 0, "load signed byte to high 16 bits"),
    0x24: ("load_short_d16", "load", 1, 0, "load 16-bit short to low 16 bits"),
    0x25: ("load_short_d16_hi", "load", 1, 0, "load 16-bit short to high 16 bits"),
    0x30: ("atomic_swap", "atomic", 1, 1, "atomic swap 32-bit"),
    0x31: ("atomic_cmpswap", "atomic", 1, 2, "atomic compare-and-swap 32-bit"),
    0x32: ("atomic_add", "atomic", 1, 1, "atomic add 32-bit"),
    0x33: ("atomic_sub", "atomic", 1, 1, "atomic subtract 32-bit"),
    0x35: ("atomic_smin", "atomic", 1, 1, "atomic signed min 32-bit"),
    0x36: ("atomic_umin", "atomic", 1, 1, "atomic unsigned min 32-bit"),
    0x37: ("atomic_smax", "atomic", 1, 1, "atomic signed max 32-bit"),
    0x38: ("atomic_umax", "atomic", 1, 1, "atomic unsigned max 32-bit"),
    0x39: ("atomic_and", "atomic", 1, 1, "atomic bitwise AND 32-bit"),
    0x3a: ("atomic_or", "atomic", 1, 1, "atomic bitwise OR 32-bit"),
    0x3b: ("atomic_xor", "atomic", 1, 1, "atomic bitwise XOR 32-bit"),
    0x3c: ("atomic_inc", "atomic", 1, 1, "atomic increment 32-bit"),
    0x3d: ("atomic_dec", "atomic", 1, 1, "atomic decrement 32-bit"),
    0x3e: ("atomic_fcmpswap", "atomic", 1, 2, "atomic float32 compare-and-swap"),
    0x3f: ("atomic_fmin", "atomic", 1, 1, "atomic float32 min"),
    0x40: ("atomic_fmax", "atomic", 1, 1, "atomic float32 max"),
    0x50: ("atomic_swap_x2", "atomic", 2, 2, "atomic swap 64-bit"),
    0x51: ("atomic_cmpswap_x2", "atomic", 2, 4, "atomic compare-and-swap 64-bit"),
    0x52: ("atomic_add_x2", "atomic", 2, 2, "atomic add 64-bit"),
    0x53: ("atomic_sub_x2", "atomic", 2, 2, "atomic subtract 64-bit"),
    0x55: ("atomic_smin_x2", "atomic", 2, 2, "atomic signed min 64-bit"),
    0x56: ("atomic_umin_x2", "atomic", 2, 2, "atomic unsigned min 64-bit"),
    0x57: ("atomic_smax_x2", "atomic", 2, 2, "atomic signed max 64-bit"),
    0x58: ("atomic_umax_x2", "atomic", 2, 2, "atomic unsigned max 64-bit"),
    0x59: ("atomic_and_x2", "atomic", 2, 2, "atomic bitwise AND 64-bit"),
    0x5a: ("atomic_or_x2", "atomic", 2, 2, "atomic bitwise OR 64-bit"),
    0x5b: ("atomic_xor_x2", "atomic", 2, 2, "atomic bitwise XOR 64-bit"),
    0x5c: ("atomic_inc_x2", "atomic", 2, 2, "atomic increment 64-bit"),
    0x5d: ("atomic_dec_x2", "atomic", 2, 2, "atomic decrement 64-bit"),
    0x5e: ("atomic_fcmpswap_x2", "atomic", 2, 4, "atomic float64 compare-and-swap"),
    0x5f: ("atomic_fmin_x2", "atomic", 2, 2, "atomic float64 min"),
    0x60: ("atomic_fmax_x2", "atomic", 2, 2, "atomic float64 max"),
}

FLAT_OPS = {
    op: (f"flat_{suffix}", kind, vdst_width, vdata_width, f"Flat memory {desc}")
    for op, (suffix, kind, vdst_width, vdata_width, desc) in FLAT_BASE_OPS.items()
}
GLOBAL_OPS = {
    op: (f"global_{suffix}", kind, vdst_width, vdata_width, f"Global memory {desc}")
    for op, (suffix, kind, vdst_width, vdata_width, desc) in FLAT_BASE_OPS.items()
}
SCRATCH_OPS = {
    op: (f"scratch_{suffix}", kind, vdst_width, vdata_width, f"Scratch memory {desc}")
    for op, (suffix, kind, vdst_width, vdata_width, desc) in FLAT_BASE_OPS.items()
    if kind in ("load", "store")
}
FLAT_SEGMENT_TABLES = {
    0: ("flat", FLAT_OPS),
    1: ("scratch", SCRATCH_OPS),
    2: ("global", GLOBAL_OPS),
}


def _decode_flat_family(w0: int, w1: int) -> tuple[int, str, str, str]:
    """Decode one 64-bit FLAT, SCRATCH, or GLOBAL memory instruction."""
    op = (w0 >> 18) & 0x7f
    slc = (w0 >> 17) & 1
    glc = (w0 >> 16) & 1
    seg = (w0 >> 14) & 0x3
    lds = (w0 >> 13) & 1
    dlc = (w0 >> 12) & 1
    raw_off = w0 & 0xfff

    addr = w1 & 0xff
    data_reg = (w1 >> 8) & 0xff
    saddr = (w1 >> 16) & 0x7f
    vdst = (w1 >> 24) & 0xff

    if seg not in FLAT_SEGMENT_TABLES:
        return (8, f"flat_seg{seg}_{op:02x}", f"flat_seg{seg}_{op:02x}", f"Unknown FLAT segment {seg}")

    seg_name, ops_table = FLAT_SEGMENT_TABLES[seg]
    if op not in ops_table:
        return (8, f"{seg_name}_op_{op:02x}", f"{seg_name}_op_{op:02x}", f"Unknown {seg_name.upper()} opcode")

    op_name, kind, vdst_width, vdata_width, comment = ops_table[op]
    if seg == 0:
        # FLAT uses a 12-bit unsigned offset field in GFX10 machine encoding.
        imm_off = raw_off
        vaddr_str = _format_vgpr_range(addr, 2)
        saddr_str = None
    else:
        # GLOBAL and SCRATCH use a 12-bit signed offset.
        imm_off = raw_off - 0x1000 if raw_off >= 0x800 else raw_off
        if seg == 2:
            if saddr == 0x7d:
                vaddr_str = _format_vgpr_range(addr, 2)
                saddr_str = "off"
            else:
                vaddr_str = f"v{addr}"
                saddr_str = decode_s_reg64(saddr)
        else:
            if saddr == 0x7d:
                vaddr_str = f"v{addr}"
                saddr_str = "off"
            else:
                vaddr_str = "off"
                saddr_str = decode_s_reg(saddr)

    operands = []
    if kind == "load":
        if not (lds and seg in (1, 2)):
            operands.append(_format_vgpr_range(vdst, vdst_width))
        operands.append(vaddr_str)
    elif kind == "store":
        operands.append(vaddr_str)
        operands.append(_format_vgpr_range(data_reg, vdata_width))
    else:
        if glc:
            operands.append(_format_vgpr_range(vdst, vdst_width))
        operands.append(vaddr_str)
        operands.append(_format_vgpr_range(data_reg, vdata_width))

    if saddr_str is not None:
        operands.append(saddr_str)

    mods = []
    if imm_off != 0:
        mods.append(f"offset:{imm_off}")
    if glc:
        mods.append("glc")
    if slc:
        mods.append("slc")
    if dlc:
        mods.append("dlc")
    if lds and seg in (1, 2) and kind == "load":
        mods.append("lds")

    asm = f"{op_name} {', '.join(operands)}"
    if mods:
        asm += f" {' '.join(mods)}"
    return (8, op_name, asm, comment)


# GFX1013 DS (LDS / GDS) opcode table:
# opcode -> (mnemonic, vdst_width, has_addr, data0_width, data1_width, offset_mode, gds_mode)
# where offset_mode is 'none' | 'single' | 'dual' | 'swizzle' and gds_mode is 'optional' | 'gds_only' | 'no_gds'.
DS_OPS = {
    0x00: ("ds_add_u32", 0, True, 1, 0, "single", "optional"),
    0x01: ("ds_sub_u32", 0, True, 1, 0, "single", "optional"),
    0x02: ("ds_rsub_u32", 0, True, 1, 0, "single", "optional"),
    0x03: ("ds_inc_u32", 0, True, 1, 0, "single", "optional"),
    0x04: ("ds_dec_u32", 0, True, 1, 0, "single", "optional"),
    0x05: ("ds_min_i32", 0, True, 1, 0, "single", "optional"),
    0x06: ("ds_max_i32", 0, True, 1, 0, "single", "optional"),
    0x07: ("ds_min_u32", 0, True, 1, 0, "single", "optional"),
    0x08: ("ds_max_u32", 0, True, 1, 0, "single", "optional"),
    0x09: ("ds_and_b32", 0, True, 1, 0, "single", "optional"),
    0x0a: ("ds_or_b32", 0, True, 1, 0, "single", "optional"),
    0x0b: ("ds_xor_b32", 0, True, 1, 0, "single", "optional"),
    0x0c: ("ds_mskor_b32", 0, True, 1, 1, "single", "optional"),
    0x0d: ("ds_write_b32", 0, True, 1, 0, "single", "optional"),
    0x0e: ("ds_write2_b32", 0, True, 1, 1, "dual", "optional"),
    0x0f: ("ds_write2st64_b32", 0, True, 1, 1, "dual", "optional"),
    0x10: ("ds_cmpst_b32", 0, True, 1, 1, "single", "optional"),
    0x11: ("ds_cmpst_f32", 0, True, 1, 1, "single", "optional"),
    0x12: ("ds_min_f32", 0, True, 1, 0, "single", "optional"),
    0x13: ("ds_max_f32", 0, True, 1, 0, "single", "optional"),
    0x14: ("ds_nop", 0, False, 0, 0, "none", "no_gds"),
    0x15: ("ds_add_f32", 0, True, 1, 0, "single", "optional"),
    0x18: ("ds_gws_sema_release_all", 0, False, 0, 0, "single", "gds_only"),
    0x19: ("ds_gws_init", 0, True, 0, 0, "single", "gds_only"),
    0x1a: ("ds_gws_sema_v", 0, False, 0, 0, "single", "gds_only"),
    0x1b: ("ds_gws_sema_br", 0, True, 0, 0, "single", "gds_only"),
    0x1c: ("ds_gws_sema_p", 0, False, 0, 0, "single", "gds_only"),
    0x1d: ("ds_gws_barrier", 0, True, 0, 0, "single", "gds_only"),
    0x1e: ("ds_write_b8", 0, True, 1, 0, "single", "optional"),
    0x1f: ("ds_write_b16", 0, True, 1, 0, "single", "optional"),
    0x20: ("ds_add_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x21: ("ds_sub_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x22: ("ds_rsub_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x23: ("ds_inc_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x24: ("ds_dec_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x25: ("ds_min_rtn_i32", 1, True, 1, 0, "single", "optional"),
    0x26: ("ds_max_rtn_i32", 1, True, 1, 0, "single", "optional"),
    0x27: ("ds_min_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x28: ("ds_max_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x29: ("ds_and_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2a: ("ds_or_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2b: ("ds_xor_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2c: ("ds_mskor_rtn_b32", 1, True, 1, 1, "single", "optional"),
    0x2d: ("ds_wrxchg_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2e: ("ds_wrxchg2_rtn_b32", 2, True, 1, 1, "dual", "optional"),
    0x2f: ("ds_wrxchg2st64_rtn_b32", 2, True, 1, 1, "dual", "optional"),
    0x30: ("ds_cmpst_rtn_b32", 1, True, 1, 1, "single", "optional"),
    0x31: ("ds_cmpst_rtn_f32", 1, True, 1, 1, "single", "optional"),
    0x32: ("ds_min_rtn_f32", 1, True, 1, 0, "single", "optional"),
    0x33: ("ds_max_rtn_f32", 1, True, 1, 0, "single", "optional"),
    0x34: ("ds_wrap_rtn_b32", 1, True, 1, 1, "single", "optional"),
    0x35: ("ds_swizzle_b32", 1, True, 0, 0, "swizzle", "optional"),
    0x36: ("ds_read_b32", 1, True, 0, 0, "single", "optional"),
    0x37: ("ds_read2_b32", 2, True, 0, 0, "dual", "optional"),
    0x38: ("ds_read2st64_b32", 2, True, 0, 0, "dual", "optional"),
    0x39: ("ds_read_i8", 1, True, 0, 0, "single", "optional"),
    0x3a: ("ds_read_u8", 1, True, 0, 0, "single", "optional"),
    0x3b: ("ds_read_i16", 1, True, 0, 0, "single", "optional"),
    0x3c: ("ds_read_u16", 1, True, 0, 0, "single", "optional"),
    0x3d: ("ds_consume", 1, False, 0, 0, "single", "optional"),
    0x3e: ("ds_append", 1, False, 0, 0, "single", "optional"),
    0x3f: ("ds_ordered_count", 1, True, 0, 0, "single", "gds_only"),
    0x40: ("ds_add_u64", 0, True, 2, 0, "single", "optional"),
    0x41: ("ds_sub_u64", 0, True, 2, 0, "single", "optional"),
    0x42: ("ds_rsub_u64", 0, True, 2, 0, "single", "optional"),
    0x43: ("ds_inc_u64", 0, True, 2, 0, "single", "optional"),
    0x44: ("ds_dec_u64", 0, True, 2, 0, "single", "optional"),
    0x45: ("ds_min_i64", 0, True, 2, 0, "single", "optional"),
    0x46: ("ds_max_i64", 0, True, 2, 0, "single", "optional"),
    0x47: ("ds_min_u64", 0, True, 2, 0, "single", "optional"),
    0x48: ("ds_max_u64", 0, True, 2, 0, "single", "optional"),
    0x49: ("ds_and_b64", 0, True, 2, 0, "single", "optional"),
    0x4a: ("ds_or_b64", 0, True, 2, 0, "single", "optional"),
    0x4b: ("ds_xor_b64", 0, True, 2, 0, "single", "optional"),
    0x4c: ("ds_mskor_b64", 0, True, 2, 2, "single", "optional"),
    0x4d: ("ds_write_b64", 0, True, 2, 0, "single", "optional"),
    0x4e: ("ds_write2_b64", 0, True, 2, 2, "dual", "optional"),
    0x4f: ("ds_write2st64_b64", 0, True, 2, 2, "dual", "optional"),
    0x50: ("ds_cmpst_b64", 0, True, 2, 2, "single", "optional"),
    0x51: ("ds_cmpst_f64", 0, True, 2, 2, "single", "optional"),
    0x52: ("ds_min_f64", 0, True, 2, 0, "single", "optional"),
    0x53: ("ds_max_f64", 0, True, 2, 0, "single", "optional"),
    0x55: ("ds_add_rtn_f32", 1, True, 1, 0, "single", "optional"),
    0x60: ("ds_add_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x61: ("ds_sub_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x62: ("ds_rsub_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x63: ("ds_inc_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x64: ("ds_dec_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x65: ("ds_min_rtn_i64", 2, True, 2, 0, "single", "optional"),
    0x66: ("ds_max_rtn_i64", 2, True, 2, 0, "single", "optional"),
    0x67: ("ds_min_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x68: ("ds_max_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x69: ("ds_and_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6a: ("ds_or_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6b: ("ds_xor_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6c: ("ds_mskor_rtn_b64", 2, True, 2, 2, "single", "optional"),
    0x6d: ("ds_wrxchg_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6e: ("ds_wrxchg2_rtn_b64", 4, True, 2, 2, "dual", "optional"),
    0x6f: ("ds_wrxchg2st64_rtn_b64", 4, True, 2, 2, "dual", "optional"),
    0x70: ("ds_cmpst_rtn_b64", 2, True, 2, 2, "single", "optional"),
    0x71: ("ds_cmpst_rtn_f64", 2, True, 2, 2, "single", "optional"),
    0x72: ("ds_min_rtn_f64", 2, True, 2, 0, "single", "optional"),
    0x73: ("ds_max_rtn_f64", 2, True, 2, 0, "single", "optional"),
    0x76: ("ds_read_b64", 2, True, 0, 0, "single", "optional"),
    0x77: ("ds_read2_b64", 4, True, 0, 0, "dual", "optional"),
    0x78: ("ds_read2st64_b64", 4, True, 0, 0, "dual", "optional"),
    0x7e: ("ds_condxchg32_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x80: ("ds_add_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x81: ("ds_sub_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x82: ("ds_rsub_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x83: ("ds_inc_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x84: ("ds_dec_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x85: ("ds_min_src2_i32", 0, True, 0, 0, "single", "optional"),
    0x86: ("ds_max_src2_i32", 0, True, 0, 0, "single", "optional"),
    0x87: ("ds_min_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x88: ("ds_max_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x89: ("ds_and_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x8a: ("ds_or_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x8b: ("ds_xor_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x8d: ("ds_write_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x92: ("ds_min_src2_f32", 0, True, 0, 0, "single", "optional"),
    0x93: ("ds_max_src2_f32", 0, True, 0, 0, "single", "optional"),
    0x95: ("ds_add_src2_f32", 0, True, 0, 0, "single", "optional"),
    0xa0: ("ds_write_b8_d16_hi", 0, True, 1, 0, "single", "optional"),
    0xa1: ("ds_write_b16_d16_hi", 0, True, 1, 0, "single", "optional"),
    0xa2: ("ds_read_u8_d16", 1, True, 0, 0, "single", "optional"),
    0xa3: ("ds_read_u8_d16_hi", 1, True, 0, 0, "single", "optional"),
    0xa4: ("ds_read_i8_d16", 1, True, 0, 0, "single", "optional"),
    0xa5: ("ds_read_i8_d16_hi", 1, True, 0, 0, "single", "optional"),
    0xa6: ("ds_read_u16_d16", 1, True, 0, 0, "single", "optional"),
    0xa7: ("ds_read_u16_d16_hi", 1, True, 0, 0, "single", "optional"),
    0xb0: ("ds_write_addtid_b32", 0, False, 1, 0, "single", "optional"),
    0xb1: ("ds_read_addtid_b32", 1, False, 0, 0, "single", "optional"),
    0xb2: ("ds_permute_b32", 1, True, 1, 0, "single", "no_gds"),
    0xb3: ("ds_bpermute_b32", 1, True, 1, 0, "single", "no_gds"),
    0xc0: ("ds_add_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc1: ("ds_sub_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc2: ("ds_rsub_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc3: ("ds_inc_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc4: ("ds_dec_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc5: ("ds_min_src2_i64", 0, True, 0, 0, "single", "optional"),
    0xc6: ("ds_max_src2_i64", 0, True, 0, 0, "single", "optional"),
    0xc7: ("ds_min_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc8: ("ds_max_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc9: ("ds_and_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xca: ("ds_or_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xcb: ("ds_xor_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xcd: ("ds_write_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xd2: ("ds_min_src2_f64", 0, True, 0, 0, "single", "optional"),
    0xd3: ("ds_max_src2_f64", 0, True, 0, 0, "single", "optional"),
    0xde: ("ds_write_b96", 0, True, 3, 0, "single", "optional"),
    0xdf: ("ds_write_b128", 0, True, 4, 0, "single", "optional"),
    0xfe: ("ds_read_b96", 3, True, 0, 0, "single", "optional"),
    0xff: ("ds_read_b128", 4, True, 0, 0, "single", "optional"),
}


def _format_ds_swizzle(imm: int) -> str | None:
    """Format the 16-bit offset field of ds_swizzle_b32 like LLVM AMDGPUInstPrinter."""
    if imm == 0:
        return None
    if imm & 0x8000:
        if (imm & 0xff00) == 0x8000:
            lanes = ",".join(str((imm >> (2 * i)) & 3) for i in range(4))
            return f"offset:swizzle(QUAD_PERM,{lanes})"
        return f"offset:{imm}"
    and_mask = imm & 0x1f
    or_mask = (imm >> 5) & 0x1f
    xor_mask = (imm >> 10) & 0x1f
    if and_mask == 0x1f and or_mask == 0:
        if xor_mask in (1, 2, 4, 8, 16):
            return f"offset:swizzle(SWAP,{xor_mask})"
        if xor_mask in (3, 7, 15, 31):
            return f"offset:swizzle(REVERSE,{xor_mask + 1})"
    broadcast_groups = {0x1e: 2, 0x1c: 4, 0x18: 8, 0x10: 16, 0x00: 32}
    if xor_mask == 0 and and_mask in broadcast_groups and (or_mask & and_mask) == 0:
        return f"offset:swizzle(BROADCAST,{broadcast_groups[and_mask]},{or_mask})"
    chars = []
    for bit in range(4, -1, -1):
        ab = (and_mask >> bit) & 1
        ob = (or_mask >> bit) & 1
        xb = (xor_mask >> bit) & 1
        if ab == 0 or ob == 1:
            chars.append(str(ob ^ xb))
        else:
            chars.append("i" if xb else "p")
    return f'offset:swizzle(BITMASK_PERM,"{"".join(chars)}")'


def _decode_ds(w0: int, w1: int) -> tuple[int, str, str, str]:
    """Decode one 64-bit DS (Local / Global Data Share) instruction."""
    op = (w0 >> 18) & 0xff
    gds = (w0 >> 17) & 1
    offset0 = w0 & 0xff
    offset1 = (w0 >> 8) & 0xff
    offset16 = w0 & 0xffff

    addr = w1 & 0xff
    data0 = (w1 >> 8) & 0xff
    data1 = (w1 >> 16) & 0xff
    vdst = (w1 >> 24) & 0xff

    entry = DS_OPS.get(op)
    if entry is None:
        return (8, f"ds_op_{op:02x}", f"ds_op_{op:02x}", f"Unknown DS opcode 0x{op:02x}")

    op_name, vdst_w, has_addr, data0_w, data1_w, offset_mode, gds_mode = entry
    if (gds_mode == "gds_only" and not gds) or (gds_mode == "no_gds" and gds):
        return (8, f"ds_op_{op:02x}", f"ds_op_{op:02x}", f"Unsupported DS gds={gds} for {op_name}")
    if offset_mode == "none" and offset16 != 0:
        return (8, f"ds_op_{op:02x}", f"ds_op_{op:02x}", f"Unsupported DS offset={offset16} for {op_name}")

    operands = []
    if vdst_w:
        operands.append(_format_vgpr_range(vdst, vdst_w))
    if has_addr:
        operands.append(f"v{addr}")
    if data0_w:
        operands.append(_format_vgpr_range(data0, data0_w))
    if data1_w:
        operands.append(_format_vgpr_range(data1, data1_w))

    mods = []
    if offset_mode == "single":
        if offset16 != 0:
            mods.append(f"offset:{offset16}")
    elif offset_mode == "dual":
        if offset0 != 0:
            mods.append(f"offset0:{offset0}")
        if offset1 != 0:
            mods.append(f"offset1:{offset1}")
    elif offset_mode == "swizzle":
        swizzle = _format_ds_swizzle(offset16)
        if swizzle is not None:
            mods.append(swizzle)

    if gds:
        mods.append("gds")

    asm = op_name
    if operands:
        asm += f" {', '.join(operands)}"
    if mods:
        asm += f" {' '.join(mods)}"
    comment = "GDS data share" if gds else "LDS data share"
    return (8, op_name, asm, comment)


# VOP3 16-bit interpolation (issue #938, step 2). As LLVM 15 encodes them for gfx1013, SRC0 holds the
# attribute number (bits 5:0), the channel (bits 7:6) and the high-half select (bit 8); SRC1 and SRC2 are
# the data sources. LLVM prints `vdst, src1, attrN.c[, src2][ high][ clamp][ mul:2|mul:4|div:2]`.
INTERP_F16_OPS = {
    0x342: ("v_interp_p1ll_f16", False, "Interpolate binary16 P1, low half, two-pass (VOP3)"),
    0x343: ("v_interp_p1lv_f16", True, "Interpolate binary16 P1, low half, with a P0 source (VOP3)"),
    0x35a: ("v_interp_p2_f16", True, "Interpolate binary16 P2 (VOP3)"),
}
INTERP_OMOD = ("", " mul:2", " mul:4", " div:2")
# Source codes LLVM 15 refuses in these forms (no literal, no 209-234/249/250), and the inline constants it
# accepts but prints as `/*invalid immediate*/`. Both were measured against LLVMDisasmInstruction for gfx1013.
INTERP_F16_REFUSED_SOURCES = frozenset([*range(209, 235), 249, 250, 255])
INTERP_F16_IMMEDIATES = frozenset([*range(128, 209), *range(240, 249)])


def _decode_interp_f16(op: int, w0: int, w1: int) -> tuple[int, str, str, str]:
    name, has_src2, comment = INTERP_F16_OPS[op]
    abs_bits = (w0 >> 8) & 0x7
    clamp = (w0 >> 15) & 1
    attribute = w1 & 0x1ff
    src1, src2, omod = (w1 >> 9) & 0x1ff, (w1 >> 18) & 0x1ff, (w1 >> 27) & 0x3
    neg_bits = (w1 >> 29) & 0x7
    # Encodings LLVM 15 refuses keep their eight-byte record but are not reported as the instruction:
    # a modifier on the attribute operand, an unused p1ll third source, an output modifier on p2, or a
    # refused source code. op_sel bits are ignored, as LLVM ignores them.
    if ((abs_bits | neg_bits) & 1 or (not has_src2 and (src2 or (abs_bits | neg_bits) & 4))
            or (op == 0x35a and omod) or src1 in INTERP_F16_REFUSED_SOURCES
            or (has_src2 and src2 in INTERP_F16_REFUSED_SOURCES)):
        return (8, f"{name}_unsupported", f"{name} <unsupported VOP3 encoding>",
                f"LLVM 15 refuses this {name} encoding for gfx1013")

    def source(code: int, index: int) -> str:
        is_abs, is_neg = (abs_bits >> index) & 1, (neg_bits >> index) & 1
        if code in INTERP_F16_IMMEDIATES:
            # LLVM spells a lone negation of a numeric operand as neg(...), so that it is not read as a sign.
            if is_neg and not is_abs:
                return "neg(/*invalid immediate*/)"
            return _format_operand_modifiers("/*invalid immediate*/", is_abs, is_neg)
        return _format_operand_modifiers(decode_v_reg(code), is_abs, is_neg)

    operands = [f"v{w0 & 0xff}", source(src1, 1), f"attr{attribute & 0x3f}.{'xyzw'[(attribute >> 6) & 3]}"]
    if has_src2:
        operands.append(source(src2, 2))
    asm = f"{name} {', '.join(operands)}"
    if attribute >> 8:
        asm += " high"
    if clamp:
        asm += " clamp"
    asm += INTERP_OMOD[omod]
    return (8, name, asm, comment)


# VOP3 32-bit interpolation (issue #938). gfx10/gfx1013 encodes v_interp_p1_f32_e64,
# v_interp_p2_f32_e64 and v_interp_mov_f32_e64 at 0x200, 0x201 and 0x202
# (LLVM 15 llvm/lib/Target/AMDGPU/VOP3Instructions.td:1045-1047), not at the gfx8/gfx9 opcodes
# 0x270-0x272 that the VI branch below still handles. The VOP3_INTERP / VOP3_INTERP_MOV operand
# profiles (VOP3Instructions.td:55-105) print
# "$vdst, $src0_modifiers, $attr$attrchan$clamp$omod" (the mov form prints its InterpSlot
# instead of a register), and VOP3Interp_gfx10 (llvm/lib/Target/AMDGPU/VOPInstructions.td:296-312)
# places attr in SRC0[5:0], attrchan in SRC0[7:6], the data VGPR in SRC1, and fixes SRC0[8] (the
# f16-only high bit), the SRC2 field and every abs/neg bit except index 1 to zero. The vectors
# this layout was derived from are VOP3_INTERP_F32_E64_VECTORS in tests/rdna2_tools_check.py.
INTERP_F32_E64_OPS = {
    0x200: ("v_interp_p1_f32_e64", "Interpolate f32 attribute, P1 stage (VOP3A)"),
    0x201: ("v_interp_p2_f32_e64", "Interpolate f32 attribute, P2 stage (VOP3A)"),
}


def _decode_interp_f32_e64(op: int, w0: int, w1: int) -> tuple[int, str, str, str]:
    """Decode the gfx10/gfx1013 VOP3 spelling of v_interp_p1_f32 or v_interp_p2_f32.

    SRC0 is the attribute operand, so the data source keeps the abs/neg bits of index 1 (w0[9]
    and w1[30]); the modifier bits of index 0 (the attribute) and index 2 (the operand this
    profile does not have), the f16-only high bit and the SRC2 field are fixed zero and are
    reported as an unsupported encoding when set, because LLVM's TableGen decoder only matches
    the zeroes. The data source in SRC1 is a VGPR_32 (VOPInterp_gfx10 fixes it to a VGPR, unlike
    the VRegSrc_32 of _decode_interp_f16), so only source codes 256..511 are legal there.
    """
    name, comment = INTERP_F32_E64_OPS[op]
    attr, attrchan = w1 & 0x3f, (w1 >> 6) & 0x3
    source_code = (w1 >> 9) & 0x1ff
    abs_bits, neg_bits = (w0 >> 8) & 0x7, (w1 >> 29) & 0x7
    # SRC1 is a VGPR_32 field, so codes 0..255 -- SGPRs, inline constants, literals, and every code
    # in INTERP_F16_REFUSED_SOURCES -- are not a legal data operand. LLVM's gfx1013 disassembler
    # marks them as an invalid register instead of printing an operand, so refuse them.
    if abs_bits & 0x5 or neg_bits & 0x5 or w1 & 0x100 or (w1 >> 18) & 0x1ff \
            or not 256 <= source_code <= 511:
        return (8, f"{name}_unsupported", f"{name} <unsupported VOP3 encoding>",
                f"LLVM 15 refuses this {name} encoding for gfx1013")

    is_abs, is_neg = (abs_bits >> 1) & 1, (neg_bits >> 1) & 1
    source = _format_operand_modifiers(decode_v_reg(source_code), is_abs, is_neg)
    asm = f"{name} v{w0 & 0xff}, {source}, attr{attr}.{'xyzw'[attrchan]}"
    asm += _vop3_output_modifiers((w0 >> 15) & 1, (w1 >> 27) & 0x3)
    return (8, name, asm, comment)


# VOP3 lane permutes (issue #1298). LLVM 15 defines v_permlane16_b32 and v_permlanex16_b32 for gfx10 at
# VOP3 0x377 and 0x378 with the operand profile `vdst, src0, src1, src2 op_sel:[fi,bound_ctrl]`: src0 is a
# VGPR, src1 and src2 are scalar lane selects, and op_sel[0] (w0[11]) and op_sel[1] (w0[12]) carry the
# fetch-inactive and bound_ctrl flags. The same-row form also accepts a VOP3 literal selector
# as specified by the RDNA 2 ISA operand rules. Unsupported words keep their record length.
PERMLANE_OPS = {
    0x377: ("v_permlane16_b32", "Permute lanes within each row of 16 (VOP3A)"),
    0x378: ("v_permlanex16_b32", "Permute lanes across the two rows of 16 (VOP3A)"),
}
# Lane-select source codes the LLVM 15 gfx1013 assembler does not accept (measured by assembling the
# disassembler's spelling of every code 0..255): the reserved codes, the SDWA/DPP markers, LDS direct and
# the literal.
PERMLANE_REFUSED_SELECTS = frozenset([*range(209, 235), 249, 250, 254, 255])


def _decode_permlane(op: int, w0: int, w1: int, literal: int | None = None) -> tuple[int, str, str, str]:
    """Decode v_permlane16_b32 / v_permlanex16_b32; length is 12 when a source names the literal."""
    name, comment = PERMLANE_OPS[op]
    length = 12 if literal is not None else 8
    src0, src1, src2 = w1 & 0x1ff, (w1 >> 9) & 0x1ff, (w1 >> 18) & 0x1ff
    # abs (w0[10:8]), op_sel[2:3] (w0[14:13]), clamp (w0[15]), omod (w1[28:27]) and neg (w1[31:29]).
    def valid_selector(src):
        if src == 255:
            return op == 0x377 and literal is not None
        return src < 255 and src not in PERMLANE_REFUSED_SELECTS

    if (w0 & 0xe700 or w1 & 0xf8000000 or not 256 <= src0 <= 511
            or not valid_selector(src1) or not valid_selector(src2) or src1 == src2 == 255):
        return (length, f"{name}_unsupported", f"{name} <unsupported VOP3A encoding>",
                f"{name} word outside the forms the gfx1013 LLVM 15 assembler accepts")
    def selector(src):
        return f"0x{literal:x}" if src == 255 else decode_s_reg(src)

    asm = f"{name} v{w0 & 0xff}, {decode_v_reg(src0)}, {selector(src1)}, {selector(src2)}"
    fi, bound_ctrl = (w0 >> 11) & 1, (w0 >> 12) & 1
    if fi or bound_ctrl:
        asm += f" op_sel:[{fi},{bound_ctrl}]"
    return (length, name, asm, comment)


def _packed_addr_regs(count: int) -> int:
    """Registers LLVM 15 prints for a packed (non-NSA) MIMG address of count dwords.

    LLVM 15 has no tuple class for 9 to 15 registers and prints those as 16, so the printed tuple is wider than the
    number of address dwords the instruction consumes.
    """
    return count if count <= 8 else 16


def _decode_mimg_bvh(op: int, w0: int, w1: int, length: int, data: bytes = b"",
                     offset: int = 0) -> tuple[int, str, str, str]:
    """Decode image_bvh_intersect_ray and image_bvh64_intersect_ray.

    The hardware fixes every modifier field of these ops except the non-sequential-address length in w0[2:1], so
    anything else is reported as unsupported with the record length intact. A non-sequential-address form lists
    one VGPR per address dword, the first from the instruction word and the rest from the appended bytes.
    """
    op_name, addr_dwords, addr_dwords_a16 = MIMG_BVH_OPS[op]
    a16 = (w1 >> 30) & 1
    if (w0 & ~((0x7f << 18) | 0x6)) != MIMG_BVH_W0 or (w1 >> 31) or (w1 >> 21) & 0x1f:
        return (length, f"{op_name}_unsupported", f"{op_name} <unsupported MIMG modifier bits>",
                "Modifier bits the ray-intersection encoding does not allow")
    vaddr = w1 & 0xff
    vdata = (w1 >> 8) & 0xff
    srsrc = ((w1 >> 16) & 0x1f) * 4
    count = addr_dwords_a16 if a16 else addr_dwords
    if (w0 >> 1) & 0x3:
        appended = list(data[offset + 8:offset + length])
        vaddr_str = "[" + ", ".join(f"v{r}" for r in [vaddr] + appended[:count - 1]) + "]"
    else:
        vaddr_str = f"v[{vaddr}:{vaddr + _packed_addr_regs(count) - 1}]"
    asm = (f"{op_name} v[{vdata}:{vdata + 3}], {vaddr_str}, "
           f"s[{srsrc}:{srsrc + 3}]" + (" a16" if a16 else ""))
    return (length, op_name, asm, "Ray-box and ray-triangle intersection against a BVH")


def decode_instruction(data: bytes, offset: int) -> tuple[int, str, str, str]:
    """
    Decode one RDNA 2 instruction at given offset.
    Returns: (instruction_byte_length, opcode_name, assembly_string, comment)
    """
    if offset + 4 > len(data):
        return (len(data) - offset, "data", f".byte {data[offset:].hex()}", "Trailing bytes")

    w0, = struct.unpack_from("<I", data, offset)

    if w0 == 0:
        return (4, "word_0x00000000", ".long 0x00000000", "Unrecognized RDNA 2 instruction encoding")

    # 1. SOPP: [31:23] == 0b101111111 (0xbf800000)
    if (w0 & 0xff800000) == 0xbf800000:
        op = (w0 >> 16) & 0x7f
        simm = w0 & 0xffff
        signed_simm = simm if simm < 0x8000 else simm - 0x10000
        op_info = SOPP_OPS.get(op, (f"s_sopp_{op:02x}", "Unknown SOPP"))
        op_name, comment = op_info

        if op_name in {
            "s_endpgm", "s_code_end", "s_wakeup", "s_icache_inv", "s_ttracedata",
            "s_endpgm_saved", "s_endpgm_ordered_ps_done", "s_wait_idle",
        }:
            asm = op_name
        elif op_name in {
            "s_nop", "s_setkill", "s_sethalt", "s_sleep", "s_setprio", "s_trap",
            "s_incperflevel", "s_decperflevel", "s_denorm_mode",
        }:
            asm = f"{op_name} {simm}"
        elif op_name == "s_waitcnt":
            # GFX10 split: vmcnt is simm16[3:0] plus [15:14], expcnt [6:4], lgkmcnt [13:8].
            # Like LLVM, omit counters at their "no wait" maximum unless all three are.
            counters = (("vmcnt", (simm & 0xf) | ((simm >> 10) & 0x30), 0x3f),
                        ("expcnt", (simm >> 4) & 0x7, 0x7),
                        ("lgkmcnt", (simm >> 8) & 0x3f, 0x3f))
            waits = [f"{name}({value})" for name, value, limit in counters if value != limit]
            asm = "s_waitcnt " + " ".join(waits or (f"{name}({value})" for name, value, _ in counters))
        elif op_name.startswith("s_cbranch") or op_name == "s_branch":
            target = offset + 4 + signed_simm * 4
            asm = f"{op_name} label_0x{target:x}"
        else:
            asm = f"{op_name} 0x{simm:x}"
        return (4, op_name, asm, comment)

    # 2. SOP1: [31:23] == 0b101111101 (0xbe800000)
    if (w0 & 0xff800000) == 0xbe800000:
        sdst = (w0 >> 16) & 0x7f
        op = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOP1_OPS.get(op, (f"s_sop1_{op:02x}", "Unknown SOP1"))
        op_name, comment = op_info
        length = 4
        is_64 = op_name.endswith(("_b64", "_u64", "_i64"))
        dst_64 = (is_64 and not op_name.startswith(("s_bcnt0_i32_", "s_bcnt1_i32_", "s_ff0_i32_", "s_ff1_i32_", "s_flbit_i32_"))) or op_name == "s_bitreplicate_b64_b32"
        src0_64 = is_64 and op_name not in ("s_bitreplicate_b64_b32", "s_bitset0_b64", "s_bitset1_b64")
        if ssrc0 == 255:
            if offset + 8 > len(data):
                return (4, "sop1_truncated", f"{op_name} <truncated>", "Incomplete SOP1 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            src0_str = f"0x{literal:x}"
            length = 8
        else:
            src0_str = decode_s_reg64(ssrc0) if src0_64 else decode_s_reg(ssrc0)
        sdst_str = decode_s_reg64(sdst) if dst_64 else decode_s_reg(sdst)
        if op_name == "s_getpc_b64":
            asm = f"{op_name} {sdst_str}"
        elif op_name in ("s_setpc_b64", "s_rfe_b64"):
            asm = f"{op_name} {src0_str}"
        else:
            asm = f"{op_name} {sdst_str}, {src0_str}"
        return (length, op_name, asm, comment)

    # SOPC: [31:23] == 0b101111110 (0xbf000000)
    # Must run before the SOPK prefix check, which would otherwise consume these words.
    if (w0 & 0xff800000) == 0xbf000000:
        op = (w0 >> 16) & 0x7f
        ssrc1 = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOPC_OPS.get(op, (f"s_sopc_{op:02x}", "Unknown SOPC"))
        op_name, comment = op_info
        length = 4
        literal = None
        if ssrc0 == 255 or ssrc1 == 255:
            if offset + 8 > len(data):
                return (4, "sopc_truncated", f"{op_name} <truncated>", "Incomplete SOPC literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8

        def sopc_operand(reg_id: int, wide: bool) -> str:
            if reg_id == 255 and literal is not None:
                return f"0x{literal:x}"
            if wide:
                return decode_s_reg64(reg_id)
            return decode_s_reg(reg_id)

        src0_wide = op_name in {"s_cmp_eq_u64", "s_cmp_lg_u64", "s_bitcmp0_b64", "s_bitcmp1_b64"}
        src1_wide = op_name in {"s_cmp_eq_u64", "s_cmp_lg_u64"}
        asm = f"{op_name} {sopc_operand(ssrc0, src0_wide)}, {sopc_operand(ssrc1, src1_wide)}"
        return (length, op_name, asm, comment)

    # 3. SOPK: [31:28] == 0b1011 (0xb0000000)
    if (w0 & 0xf0000000) == 0xb0000000 and (w0 & 0x0f800000) not in (0x0f800000, 0x0e800000, 0x0f000000):
        op = (w0 >> 23) & 0x1f
        sdst = (w0 >> 16) & 0x7f
        simm = w0 & 0xffff
        signed_simm = simm if simm < 0x8000 else simm - 0x10000
        op_info = SOPK_OPS.get(op, (f"s_sopk_{op:02x}", "Unknown SOPK"))
        op_name, comment = op_info
        if op_name == "s_setreg_imm32_b32":
            # The only SOPK form followed by a literal dword; SDST is unused.
            if offset + 8 > len(data):
                return (4, "sopk_truncated", f"{op_name} <truncated>", "Incomplete SOPK literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            return (8, op_name, f"{op_name} {format_hwreg(simm)}, {format_literal32(literal)}", comment)
        if op_name == "s_getreg_b32":
            return (4, op_name, f"{op_name} {decode_s_reg(sdst)}, {format_hwreg(simm)}", comment)
        if op_name == "s_setreg_b32":
            # The SDST field names the source SGPR.
            return (4, op_name, f"{op_name} {format_hwreg(simm)}, {decode_s_reg(sdst)}", comment)
        if op_name == "s_version":
            # No destination operand; the SDST field is not printed.
            return (4, op_name, f"{op_name} 0x{simm:x}", comment)
        if op_name == "s_call_b64":
            return (4, op_name, f"{op_name} {decode_s_reg64(sdst)}, {simm}", comment)
        asm = f"{op_name} {decode_s_reg(sdst)}, 0x{simm:x}"
        return (4, op_name, asm, comment)

    # 4. SOP2: [31:30] == 0b10 (0x80000000)
    if (w0 & 0xc0000000) == 0x80000000 and (w0 & 0x30000000) != 0x30000000:
        op = (w0 >> 23) & 0x7f
        sdst = (w0 >> 16) & 0x7f
        ssrc1 = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOP2_OPS.get(op, (f"s_sop2_{op:02x}", "Unknown SOP2"))
        op_name, comment = op_info
        length = 4
        literal = None
        if ssrc0 == 255 or ssrc1 == 255:
            if offset + 8 > len(data):
                return (4, "sop2_truncated", f"{op_name} <truncated>", "Incomplete SOP2 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        is_64 = op_name.endswith(("_b64", "_u64", "_i64"))
        src0_wide = is_64 and not op_name.startswith("s_bfm_")
        src0_str = f"0x{literal:x}" if ssrc0 == 255 and literal is not None else (decode_s_reg64(ssrc0) if src0_wide else decode_s_reg(ssrc0))
        src1_wide = is_64 and not op_name.startswith(("s_lshl_", "s_lshr_", "s_ashr_", "s_bfe_", "s_bfm_"))
        src1_str = f"0x{literal:x}" if ssrc1 == 255 and literal is not None else (decode_s_reg64(ssrc1) if src1_wide else decode_s_reg(ssrc1))
        sdst_str = decode_s_reg64(sdst) if is_64 else decode_s_reg(sdst)
        asm = f"{op_name} {sdst_str}, {src0_str}, {src1_str}"
        return (length, op_name, asm, comment)

    # 5. VOP1: [31:25] == 0b0111111 (0x7e000000)
    if (w0 & 0xfe000000) == 0x7e000000:
        vdst = (w0 >> 17) & 0xff
        op = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOP1_OPS.get(op, (f"v_vop1_{op:02x}", "Unknown VOP1"))
        op_name, comment = op_info
        if src0 == 249:
            if offset + 8 > len(data):
                return (4, "vop1_truncated", f"{op_name} <truncated>", "Incomplete VOP1 SDWA")
            w1, = struct.unpack_from("<I", data, offset + 4)
            return _decode_vop1_sdwa(w1, vdst, op_name, comment)
        if src0 in (233, 234, 250):
            form = VOP_EXTENSION_FORMS[src0]
            if offset + 8 > len(data):
                return (4, "vop1_truncated", f"{op_name} <truncated>", f"Incomplete VOP1 {form}")
            w1, = struct.unpack_from("<I", data, offset + 4)
            return _decode_vop1_dpp(w1, vdst, src0, op_name, comment)
        if src0 in VOP_EXTENSION_FORMS:
            return _vop_extension(data, offset, "vop1", op_name, src0)
        if op_name in VOP1_NO_OPERAND_OPS:
            # LLVM prints the bare mnemonic and takes no literal dword even when SRC0 is 255.
            return (4, op_name, op_name, comment)
        length = 4
        # A 32-bit result from a binary64 source: the name ends in _f64 but the destination is one VGPR.
        narrow_dst = op_name.startswith(("v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64", "v_frexp_exp_i32_f64"))
        wide_dst = op_name.startswith(("v_cvt_f64_i32", "v_cvt_f64_u32", "v_cvt_f64_f32")) or (
            op_name.endswith(("_f64_e32", "_f64")) and not narrow_dst)
        wide_src0 = narrow_dst or op_name.endswith(("_f64_e32", "_f64"))
        if src0 == 255:
            if offset + 8 > len(data):
                return (4, "vop1_truncated", f"{op_name} <truncated>", "Incomplete VOP1 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            src0_str = f"0x{literal:x}"
            length = 8
        elif wide_src0:
            src0_str = decode_v_reg64(src0)
        else:
            src0_str = decode_v_reg(src0)
        dst_str = f"v[{vdst}:{vdst + 1}]" if wide_dst and vdst < 255 else (f"<invalid VGPR pair>" if wide_dst else f"v{vdst}")
        if op_name == "v_readfirstlane_b32":
            # This VOP1 instruction writes an SGPR: its vdst field is an SGPR
            # number even though the mnemonic starts with "v_". LLVM 21 for
            # gfx1013 prints s2 for the encoding 0005047e.
            dst_str = decode_s_reg(vdst)
        if op_name == "v_nop":
            asm = "v_nop"
        else:
            asm = f"{op_name} {dst_str}, {src0_str}"
        return (length, op_name, asm, comment)

    # 6. VOPC: [31:25] == 0b0111110 (0x7c000000)
    if (w0 & 0xfe000000) == 0x7c000000:
        op = (w0 >> 17) & 0xff
        vsrc1 = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOPC_OPS.get(op, (f"v_vopc_{op:02x}", "Unknown VOPC"))
        op_name, comment = op_info
        length = 4
        literal = None
        if src0 in (233, 234, 250):
            return _vop_extension(data, offset, "vopc", op_name, src0)
        if src0 in (249, 255):
            extension = "SDWA" if src0 == 249 else "literal"
            if offset + 8 > len(data):
                return (4, "vopc_truncated", f"{op_name} <truncated>", f"Incomplete VOPC {extension}")
            if src0 == 249:
                w1, = struct.unpack_from("<I", data, offset + 4)
                return _decode_vopc_sdwa(w1, vsrc1, op_name, comment)
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        wide0 = op_name.endswith(("_f64_e32", "_i64_e32", "_u64_e32"))
        wide1 = wide0 and "_class_" not in op_name
        if literal is not None:
            src0_str = f"0x{literal:x}"
        elif wide0:
            src0_str = decode_v_reg64(src0)
        else:
            src0_str = decode_v_reg(src0)
        if wide1:
            src1_str = f"v[{vsrc1}:{vsrc1 + 1}]" if vsrc1 < 255 else "<invalid VGPR pair>"
        else:
            src1_str = f"v{vsrc1}"
        if op_name.startswith("v_cmpx_"):
            asm = f"{op_name} {src0_str}, {src1_str}"
        else:
            asm = f"{op_name} vcc_lo, {src0_str}, {src1_str}"
        return (length, op_name, asm, comment)

    # 7. VOP2: [31] == 0
    if (w0 & 0x80000000) == 0:
        op = (w0 >> 25) & 0x3f
        vdst = (w0 >> 17) & 0xff
        vsrc1 = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOP2_OPS.get(op, (f"v_vop2_{op:02x}", "Unknown VOP2"))
        op_name, comment = op_info
        if src0 == 249:
            if offset + 8 > len(data):
                return (4, "vop2_truncated", f"{op_name} <truncated>", "Incomplete VOP2 SDWA")
            w1, = struct.unpack_from("<I", data, offset + 4)
            return _decode_vop2_sdwa(w1, vdst, vsrc1, op, op_name, comment)
        if src0 in (233, 234, 250):
            form = VOP_EXTENSION_FORMS[src0]
            if offset + 8 > len(data):
                return (4, "vop2_truncated", f"{op_name} <truncated>", f"Incomplete VOP2 {form}")
            w1, = struct.unpack_from("<I", data, offset + 4)
            return _decode_vop2_dpp(w1, vdst, vsrc1, src0, op, op_name, comment)
        if src0 in VOP_EXTENSION_FORMS:
            return _vop_extension(data, offset, "vop2", op_name, src0)
        has_literal = src0 == 255 or op in (0x20, 0x21, 0x2c, 0x2d, 0x37, 0x38)
        length = 4
        literal = None
        if has_literal:
            if offset + 8 > len(data):
                return (4, "vop2_truncated", f"{op_name} <truncated>", "Incomplete VOP2 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        src0_str = f"0x{literal:x}" if src0 == 255 and literal is not None else decode_v_reg(src0)
        if op_name == "v_cndmask_b32_e32":
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}, vcc_lo"
        elif op in (0x28, 0x29, 0x2a):
            asm = f"{op_name} v{vdst}, vcc_lo, {src0_str}, v{vsrc1}, vcc_lo"
        elif op in (0x20, 0x2c, 0x37):
            asm = f"{op_name} v{vdst}, {src0_str}, 0x{literal:x}, v{vsrc1}"
        elif op in (0x21, 0x2d, 0x38):
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}, 0x{literal:x}"
        else:
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}"
        return (length, op_name, asm, comment)

    # 8. VINTRP: [31:26] == 0b110010 (0xc8000000)
    if (w0 & 0xfc000000) == 0xc8000000:
        vdst = (w0 >> 18) & 0xff
        op = (w0 >> 16) & 0x3
        attr = (w0 >> 10) & 0x3f
        chan = "xyzw"[(w0 >> 8) & 0x3]
        vsrc = w0 & 0xff
        op_names = {0: "v_interp_p1_f32_e32", 1: "v_interp_p2_f32_e32", 2: "v_interp_mov_f32_e32"}
        op_name = op_names.get(op, f"v_interp_{op}")
        comment = f"Parameter interpolation stage {op+1} for attribute {attr}.{chan}"
        if op == 2:
            # The mov form's vsrc field is the parameter selector, not a VGPR.
            if vsrc >= len(VINTRP_MOV_PARAMS):
                return (4, f"{op_name}_unsupported", f"{op_name} <unsupported VINTRP encoding>",
                        f"v_interp_mov_f32 parameter selector {vsrc} is not p10, p20 or p0")
            return (4, op_name, f"{op_name} v{vdst}, {VINTRP_MOV_PARAMS[vsrc]}, attr{attr}.{chan}", comment)
        asm = f"{op_name} v{vdst}, v{vsrc}, attr{attr}.{chan}"
        return (4, op_name, asm, comment)

    # 9. EXP: 64-bit [31:26] == 0b111110 (0xf8000000)
    if (w0 & 0xfc000000) == 0xf8000000:
        if offset + 8 > len(data):
            return (4, "exp_truncated", "exp <truncated>", "Incomplete 64-bit export instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        en = w0 & 0xf
        target = (w0 >> 4) & 0x3f
        compr = (w0 >> 10) & 1
        done = (w0 >> 11) & 1
        vm = (w0 >> 12) & 1
        vs = [(w1 >> (8 * k)) & 0xff for k in range(4)]
        if compr:
            lo = f"v{vs[0]}" if en & 0x1 else "off"
            hi = f"v{vs[1]}" if en & 0x4 else "off"
            regs = [lo, lo, hi, hi]
        else:
            regs = [f"v{vs[k]}" if (en & (1 << k)) else "off" for k in range(4)]
        target_name = EXP_TARGETS.get(target, f"target_{target}")
        flags = []
        if done:
            flags.append("done")
        if compr:
            flags.append("compr")
        if vm:
            flags.append("vm")
        # The printed compressed operands use the first bit of each pair.
        # Keep a non-paired raw mask visible so it cannot masquerade as a supported export.
        if compr and en not in (0x0, 0x3, 0xc, 0xf):
            flags.append(f"en:{en:#x}")
        flags_str = " ".join(flags)
        asm = f"exp {target_name} {', '.join(regs)} {flags_str}".strip()
        comment = f"Export output pixels/attributes to {target_name.upper()}"
        return (8, "exp", asm, comment)

    # 10. SMEM: 64-bit [31:26] == 0b111101 (0xf4000000)
    if (w0 & 0xfc000000) == 0xf4000000:
        if offset + 8 > len(data):
            return (4, "smem_truncated", "smem <truncated>", "Incomplete 64-bit SMEM instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0xff
        sdata = (w0 >> 6) & 0x7f
        sbase = (w0 & 0x3f) * 2
        op_name = SMEM_OPS.get(op, f"s_smem_{op:02x}")
        data_width, base_width = SMEM_WIDTHS.get(op, (4, 2))
        if op in SMEM_NO_OPERANDS:
            return (8, op_name, op_name, "Scalar memory cache control")
        if op in SMEM_PROBE_OPS:
            sdata_str = str(sdata)
        else:
            sdata_str = decode_s_reg(sdata) if data_width == 1 else f"s[{sdata}:{sdata + data_width - 1}]"
        if base_width == 0:
            # s_memtime / s_memrealtime read no memory address: only the destination pair.
            # Named 64-bit specials (vcc, ttmp pairs, null) keep their spelling.
            timer_dst = decode_s_reg64(sdata) if sdata >= 106 else sdata_str
            return (8, op_name, f"{op_name} {timer_dst}", "Read a 64-bit hardware timer")
        if base_width == 2 and sbase == 106:
            sbase_str = "vcc"
        else:
            sbase_str = f"s[{sbase}:{sbase + base_width - 1}]"
        soffset = (w1 >> 25) & 0x7f
        imm_offset = w1 & 0x1fffff
        if imm_offset & 0x100000:
            # GFX9+ SMEM OFFSET[20:0] is a signed 21-bit byte offset (LLVM sign-extends it)
            imm_offset -= 0x200000
        imm_str = f"-0x{-imm_offset:x}" if imm_offset < 0 else f"0x{imm_offset:x}"
        if soffset == 125:
            off_str = imm_str
        else:
            off_str = decode_s_reg(soffset)
            if imm_offset:
                off_str += f" offset:{imm_str}"
        mods = []
        if (w0 >> 16) & 1:
            mods.append("glc")
        if (w0 >> 14) & 1:
            mods.append("dlc")
        mod_str = (" " + " ".join(mods)) if mods else ""
        asm = f"{op_name} {sdata_str}, {sbase_str}, {off_str}{mod_str}"
        comment = "Scalar memory load from constant buffer/resource descriptor"
        if op in (0x18, 0x19, 0x1a):
            comment = "Scalar memory store to a buffer resource"
        elif op in SMEM_PROBE_OPS:
            comment = "Probe the address translation cache"
        return (8, op_name, asm, comment)

    # 11. MIMG: 64-bit + optional NSA dwords [31:26] == 0b111100 (0xf0000000)
    if (w0 & 0xfc000000) == 0xf0000000:
        nsize = (w0 >> 1) & 0x3
        length = 8 + 4 * nsize
        if offset + length > len(data):
            return (len(data) - offset, "mimg_truncated", "mimg <truncated>", "Incomplete MIMG instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = ((w0 >> 18) & 0x7f) | ((w0 & 1) << 7)
        if op in MIMG_BVH_OPS:
            return _decode_mimg_bvh(op, w0, w1, length, data, offset)
        op_name, has_sampler, extra_addrs = MIMG_OPS.get(op, (f"image_mimg_{op:02x}", False, 0))
        dmask = (w0 >> 8) & 0xf
        dim_code = (w0 >> 3) & 0x7
        dim_name, base_addrs, deriv_addrs = MIMG_DIMS[dim_code]
        if op in MIMG_G16_DERIV_OPS:
            deriv_addrs = MIMG_G16_DERIV_ADDRS[dim_code]
        elif op not in MIMG_DERIV_OPS:
            deriv_addrs = 0
        addr_count = base_addrs + extra_addrs + deriv_addrs
        a16 = (w1 >> 30) & 1
        packed_2d = (
            a16
            and op_name in ("image_load", "image_store", "image_sample", "image_sample_lz")
            and dim_name == "SQ_RSRC_IMG_2D"
        )
        if packed_2d:
            addr_count = 1  # Two 16-bit 2D coordinates share one VGPR.
        d16 = (w1 >> 31) & 1
        # A gather returns the four texels of the selected component whatever the one-hot mask is.
        n_data = 4 if op_name.startswith("image_gather4") else max(1, bin(dmask).count("1"))
        if d16:
            n_data = (n_data + 1) // 2
        vaddr0 = w1 & 0xff
        vdata = (w1 >> 8) & 0xff
        srsrc = ((w1 >> 16) & 0x1f) * 4
        ssamp = ((w1 >> 21) & 0x1f) * 4
        r128 = (w0 >> 15) & 1
        vdata_str = f"v{vdata}" if n_data == 1 else f"v[{vdata}:{vdata + n_data - 1}]"
        if nsize > 0:
            extra_bytes = list(data[offset + 8:offset + length])
            addrs = [vaddr0] + extra_bytes[:max(0, addr_count - 1)]
            vaddr_str = "[" + ", ".join(f"v{r}" for r in addrs) + "]"
        else:
            addr_regs = _packed_addr_regs(addr_count)
            vaddr_str = f"v{vaddr0}" if addr_regs == 1 else f"v[{vaddr0}:{vaddr0 + addr_regs - 1}]"
        srsrc_width = 4 if r128 else 8
        srsrc_str = f"s[{srsrc}:{srsrc + srsrc_width - 1}]"
        operands = [vdata_str, vaddr_str, srsrc_str]
        if has_sampler:
            operands.append(f"s[{ssamp}:{ssamp + 3}]")
        mods = [f"dmask:0x{dmask:x}", f"dim:{dim_name}"]
        if (w0 >> 12) & 1:
            mods.append("unorm")
        if (w0 >> 13) & 1:
            mods.append("glc")
        if (w0 >> 25) & 1:
            mods.append("slc")
        if r128:
            mods.append("r128")
        if (w0 >> 7) & 1:
            mods.append("dlc")
        if (w0 >> 16) & 1:
            mods.append("tfe")
        if (w0 >> 17) & 1:
            mods.append("lwe")
        if a16:
            mods.append("a16")
        if d16:
            mods.append("d16")
        if packed_2d and nsize:
            mods.append("invalid_nsa")  # A single address VGPR requires no NSA extension.
        # Word 0 bits 6 and 14 and word 1 bits 29:26 are reserved on gfx10: LLVM assembles none of them and
        # ignores them when disassembling, so a set bit is surfaced instead of vanishing from the listing.
        if (w0 & 0x4040) or (w1 >> 26) & 0xf:
            mods.append("reserved")
        asm = f"{op_name} {', '.join(operands)} {' '.join(mods)}"
        comment = "Image memory load/store/sample operation"
        return (length, op_name, asm, comment)

    # 12. VOP3P: 64-bit [31:23] == 0b110011000 (0xcc000000)
    if (w0 & 0xff800000) == 0xcc000000:
        if offset + 8 > len(data):
            return (4, "vop3p_truncated", "vop3p <truncated>", "Incomplete 64-bit VOP3P instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 16) & 0x7f
        clamp = (w0 >> 15) & 1
        vdst = w0 & 0xff
        src0 = w1 & 0x1ff
        src1 = (w1 >> 9) & 0x1ff
        src2 = (w1 >> 18) & 0x1ff

        op_info = VOP3P_OPS.get(op, (f"v_vop3p_{op:02x}", "Unknown VOP3P"))
        op_name, comment = op_info
        is_3src = op in VOP3P_3SRC
        arity = 3 if is_3src else 2

        length = 8
        literal = None
        if src0 == 255 or src1 == 255 or (is_3src and src2 == 255):
            if offset + 12 > len(data):
                return (8, "vop3p_truncated", f"{op_name} <truncated>", "Incomplete VOP3P literal")
            literal, = struct.unpack_from("<I", data, offset + 8)
            length = 12

        # Check for unmodeled / reserved bits in 2-source VOP3P encodings.
        # Note: bit 14 of w0 is op_sel_hi[2], which LLVM's MC assembler defaults to 1 (op_sel_hi = 0b111)
        # for all packed VOP3P instructions regardless of arity, so bit 14 is ignored on 2-source forms.
        if not is_3src and (src2 != 0 or ((w0 >> 10) & 1) or ((w0 >> 13) & 1) or ((w1 >> 31) & 1)):
            return (length, f"{op_name}_unsupported", f"{op_name} <unsupported_reserved_bits>", "Reserved bitfields set in 2-source VOP3P encoding")

        is_fma_mix = op_name.startswith("v_fma_mix")

        def vop3p_operand(reg_id: int, is_abs: int = 0, is_neg: int = 0) -> str:
            if reg_id == 255 and literal is not None:
                val = f"0x{literal:x}"
            else:
                val = decode_v_reg(reg_id)
            return _format_operand_modifiers(val, is_abs, is_neg)

        op_sel = [(w0 >> 11) & 1, (w0 >> 12) & 1, (w0 >> 13) & 1][:arity]
        op_sel_hi = [(w1 >> 27) & 1, (w1 >> 28) & 1, (w0 >> 14) & 1][:arity]
        neg_lo = [(w1 >> 29) & 1, (w1 >> 30) & 1, (w1 >> 31) & 1][:arity]
        neg_hi = [(w0 >> 8) & 1, (w0 >> 9) & 1, (w0 >> 10) & 1][:arity]

        if is_fma_mix:
            # Mixed-precision FMA repurposes VOP3P w0[10:8] as abs and w1[31:29] as neg.
            abs_bits = neg_hi
            neg_bits = neg_lo
            neg_lo = []
            neg_hi = []
        else:
            abs_bits = [0] * arity
            neg_bits = [0] * arity

        raw_srcs = [src0, src1, src2][:arity]
        src_strs = [vop3p_operand(raw_srcs[i], abs_bits[i], neg_bits[i]) for i in range(arity)]
        asm = f"{op_name} v{vdst}, {', '.join(src_strs)}"

        # Architectural defaults:
        # Standard packed instructions (v_pk_*, v_dot*): default op_sel is all 0, op_sel_hi is all 1, neg_lo/neg_hi all 0.
        # Mixed-precision FMA (v_fma_mix*): default op_sel and op_sel_hi are all 0 (f32, low half).
        default_op_sel = [0] * arity
        default_op_sel_hi = [0 if is_fma_mix else 1] * arity
        default_neg_lo = [0] * len(neg_lo)
        default_neg_hi = [0] * len(neg_hi)

        if op_sel != default_op_sel:
            asm += f" op_sel:[{','.join(str(b) for b in op_sel)}]"
        if op_sel_hi != default_op_sel_hi:
            asm += f" op_sel_hi:[{','.join(str(b) for b in op_sel_hi)}]"
        if neg_lo != default_neg_lo:
            asm += f" neg_lo:[{','.join(str(b) for b in neg_lo)}]"
        if neg_hi != default_neg_hi:
            asm += f" neg_hi:[{','.join(str(b) for b in neg_hi)}]"

        if clamp:
            asm += " clamp"

        return (length, op_name, asm, comment)

    # 12. MUBUF: 64-bit [31:26] == 0b111000 (0xe0000000)
    if (w0 & 0xfc000000) == 0xe0000000:
        if offset + 8 > len(data):
            return (4, "mubuf_truncated", "mubuf <truncated>", "Incomplete 64-bit MUBUF instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0xff
        op_name, vdata_width, comment = MUBUF_OPS.get(
            op, (f"buffer_mubuf_{op:02x}", 1, "Unknown MUBUF")
        )
        if vdata_width == 0:
            return (8, op_name, op_name, comment)
        allow_lds = op_name.startswith("buffer_load_") and "_d16" not in op_name
        allow_tfe = not op_name.startswith("buffer_atomic_")
        if op_name in MUBUF_D16_HI_FORMAT_OPS:
            # llvm-mc rejects the lds bit on these. With tfe set it prints the extra status dword in the vdata range.
            vdata_width += (w1 >> 23) & 1
            if (w0 >> 16) & 1 or ((w1 >> 8) & 0xff) + vdata_width > 256:
                return (8, f"{op_name}_unsupported", f"{op_name} <unsupported MUBUF encoding>",
                        f"{op_name} word with the lds bit or a VGPR range past v255 that gfx1013 llvm-mc rejects")
        operands, mods = _format_buffer_operands_and_mods(
            w0, w1, vdata_width, allow_lds=allow_lds, allow_tfe=allow_tfe
        )
        asm = f"{op_name} {', '.join(operands)}"
        if mods:
            asm += f" {' '.join(mods)}"
        return (8, op_name, asm, comment)

    # 13. MTBUF: 64-bit [31:26] == 0b111010 (0xe8000000)
    if (w0 & 0xfc000000) == 0xe8000000:
        if offset + 8 > len(data):
            return (4, "mtbuf_truncated", "mtbuf <truncated>", "Incomplete 64-bit MTBUF instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        fmt = (w0 >> 19) & 0x7f
        op = ((w0 >> 16) & 0x7) | (((w1 >> 21) & 1) << 3)
        op_name, vdata_width, comment = MTBUF_OPS.get(
            op, (f"tbuffer_mtbuf_{op:02x}", 1, "Unknown MTBUF")
        )
        if fmt == 1:
            format_mod = None
        elif fmt in MTBUF_FORMATS:
            format_mod = f"format:[{MTBUF_FORMATS[fmt]}]"
        else:
            format_mod = f"format:{fmt}"
        operands, mods = _format_buffer_operands_and_mods(
            w0, w1, vdata_width, format_mod=format_mod
        )
        asm = f"{op_name} {', '.join(operands)}"
        if mods:
            asm += f" {' '.join(mods)}"
        return (8, op_name, asm, comment)

    # 14. FLAT / SCRATCH / GLOBAL: 64-bit [31:26] == 0b110111 (0xdc000000)
    if (w0 & 0xfc000000) == 0xdc000000:
        if offset + 8 > len(data):
            return (4, "flat_truncated", "flat <truncated>", "Incomplete 64-bit FLAT/GLOBAL/SCRATCH instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        return _decode_flat_family(w0, w1)

    # DS: 64-bit [31:26] == 0b110110 (0xd8000000)
    if (w0 & 0xfc000000) == 0xd8000000:
        if offset + 8 > len(data):
            return (4, "ds_truncated", "ds <truncated>", "Incomplete 64-bit DS instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        return _decode_ds(w0, w1)

    # 15. VOP3A / VOP3B: 64-bit [31:26] == 0b110101 (0xd4000000)
    if (w0 & 0xfc000000) == 0xd4000000:
        if offset + 8 > len(data):
            return (len(data) - offset, "vop3_truncated", "vop3 <truncated>", "Incomplete 64-bit VOP3 instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 16) & 0x3ff
        clamp = (w0 >> 15) & 1
        vdst = w0 & 0xff
        src0 = w1 & 0x1ff
        src1 = (w1 >> 9) & 0x1ff
        src2 = (w1 >> 18) & 0x1ff
        omod = (w1 >> 27) & 0x3
        neg = [(w1 >> 29) & 1, (w1 >> 30) & 1, (w1 >> 31) & 1]

        if op in INTERP_F16_OPS:
            return _decode_interp_f16(op, w0, w1)

        if op in PERMLANE_OPS:
            if 255 in (src0, src1, src2):
                # Keep the twelve-byte record of a literal operand, as the generic VOP3 path does.
                if offset + 12 > len(data):
                    return (8, "vop3_truncated", f"{PERMLANE_OPS[op][0]} <truncated>", "Incomplete VOP3 literal")
                return _decode_permlane(op, w0, w1, struct.unpack_from("<I", data, offset + 8)[0])
            return _decode_permlane(op, w0, w1)

        is_vopc = False
        is_vop3b = False
        if op < 0x100:
            is_vopc = True
            base_info = VOPC_OPS.get(op, (f"v_vopc_{op:02x}", "Unknown VOPC compare"))
            op_name = base_info[0].replace("_e32", "_e64")
            comment = f"{base_info[1]} (VOP3A)"
            arity = 2
            wide0 = op_name.endswith(("_f64_e64", "_i64_e64", "_u64_e64"))
            wide1 = wide0 and "_class_" not in op_name
            dst_width = 1
            src_widths = (2 if wide0 else 1, 2 if wide1 else 1)
        elif 0x100 <= op < 0x140:
            vop2_op = op - 0x100
            dst_width = 1
            if op in (0x128, 0x129, 0x12a):
                is_vop3b = True
                carry_ops = {
                    0x128: ("v_add_co_ci_u32_e64", "Vector add with carry-in and carry-out uint32"),
                    0x129: ("v_sub_co_ci_u32_e64", "Vector sub with carry-in and carry-out uint32"),
                    0x12a: ("v_subrev_co_ci_u32_e64", "Vector reverse sub with carry-in and carry-out uint32"),
                }
                op_name, comment = carry_ops[op]
                comment += " (VOP3B)"
                arity = 3
                src_widths = (1, 1, 1)
            elif vop2_op == 0x01:
                op_name = "v_cndmask_b32_e64"
                comment = "Vector conditional mask select (VOP3A)"
                arity = 3
                src_widths = (1, 1, 1)
            else:
                base_info = VOP2_OPS.get(vop2_op, (f"v_vop2_{vop2_op:02x}", "Unknown VOP2"))
                op_name = base_info[0].replace("_e32", "_e64")
                comment = f"{base_info[1]} (VOP3A)"
                arity = 2
                src_widths = (1, 1)
        elif 0x180 <= op < 0x200:
            vop1_op = op - 0x180
            base_info = VOP1_OPS.get(vop1_op, (f"v_vop1_{vop1_op:02x}", "Unknown VOP1"))
            op_name = base_info[0].replace("_e32", "_e64")
            comment = f"{base_info[1]} (VOP3A)"
            if op_name == "v_nop" or op_name in VOP1_NO_OPERAND_OPS:
                arity = 0
                dst_width = 0
                src_widths = ()
            else:
                arity = 1
                if op_name.startswith(("v_cvt_f64_i32", "v_cvt_f64_u32", "v_cvt_f64_f32")):
                    dst_width = 2
                    src_widths = (1,)
                elif op_name.startswith(("v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64", "v_frexp_exp_i32_f64")):
                    dst_width = 1
                    src_widths = (2,)
                elif op_name.endswith(("_f64_e64", "_f64")):
                    dst_width = 2
                    src_widths = (2,)
                else:
                    dst_width = 1
                    src_widths = (1,)
        elif 0x200 <= op < 0x203:
            # gfx10/gfx1013 VOP3 interpolation (see INTERP_F32_E64_OPS). The p1/p2 forms are
            # decoded here; the mov form keeps its table-driven, restricted-modifier decode
            # below, whose second operand is an InterpSlot (p10/p20/p0), not a register.
            if op == 0x202:
                op_name, comment, arity, is_vop3b, dst_width, src_widths = VOP3_NATIVE_OPS[op]
            else:
                return _decode_interp_f32_e64(op, w0, w1)
        elif 0x270 <= op < 0x274:
            # gfx8/gfx9 (VI) range, not gfx1013: LLVM 15 defines these three as
            # VOP3Interp_Real_vi at 0x270-0x272 under AssemblerPredicate = isGFX8GFX9
            # (VOP3Instructions.td:1365-1367). No gfx10 instruction uses them (gfx1013 puts the
            # same three forms at 0x200-0x202 above), so the branch is kept only so gfx8/gfx9
            # words still render the way LLVM's VI tests show.
            vintrp_names = {0: "v_interp_p1_f32_e64", 1: "v_interp_p2_f32_e64", 2: "v_interp_mov_f32_e64"}
            vop_int = op - 0x270
            op_name = vintrp_names.get(vop_int, f"v_interp_{vop_int}_e64")
            comment = f"Parameter interpolation stage {vop_int + 1} (VOP3A)"
            arity = 2
            dst_width = 1
            src_widths = (1, 1)
        elif op in VOP3_NATIVE_OPS:
            op_name, comment, arity, is_vop3b, dst_width, src_widths = VOP3_NATIVE_OPS[op]
        else:
            op_name = f"v_vop3_{op:03x}"
            comment = f"Unknown VOP3 opcode 0x{op:03x}"
            arity = 3
            dst_width = 1
            src_widths = (1, 1, 1)

        if op_name == "v_readfirstlane_b32":
            # gfx1013 assembles this operation only as the 32-bit VOP1: llvm-mc
            # rejects the e64 spelling and reports this VOP3A word as an invalid
            # instruction encoding.  Report a distinct unsupported-encoding
            # record instead of a lowerable instruction, and keep the eight-byte
            # record boundary so the next instruction starts at +8.
            return (8, "v_readfirstlane_b32_unsupported",
                    "v_readfirstlane_b32 <unsupported VOP3A encoding>",
                    "VOP3A encoding of v_readfirstlane_b32 is not available on gfx1013")

        if op_name in ("v_swap_b32", "v_swaprel_b32"):
            # These two exist only as 32-bit VOP1: llvm-mc rejects the e64 spelling. Keep the eight-byte boundary.
            return (8, f"{op_name}_unsupported", f"{op_name} <unsupported VOP3A encoding>",
                    f"{op_name} has no VOP3A encoding on gfx1013")

        if op_name == "v_nop":
            return (8, "v_nop", "v_nop", "Vector no-op (VOP3A)")
        if op_name in VOP1_NO_OPERAND_OPS:
            return (8, op_name, op_name, comment)

        restricted = op_name in VOP3_RESTRICTED_MODIFIERS
        if restricted:
            if not _vop3_restricted_encoding_is_valid(op_name, arity, w0, w1):
                return (8, f"{op_name}_unsupported", f"{op_name} <unsupported VOP3A encoding>",
                        f"{op_name} word with modifier bits that gfx1013 llvm-mc rejects")
            if op_name == "v_interp_mov_f32_e64":
                return _decode_interp_mov_e64(w0, w1, comment)

        has_literal = (arity >= 1 and src0 == 255) or (arity >= 2 and src1 == 255) or (arity >= 3 and src2 == 255)
        length = 8
        literal = None
        if has_literal:
            if offset + 12 > len(data):
                return (8, "vop3_truncated", f"{op_name} <truncated>", "Incomplete VOP3 literal")
            literal, = struct.unpack_from("<I", data, offset + 8)
            length = 12

        if op == 0x13c:
            # The VOP2-only v_pk_fmac_f16 has no VOP3 form on gfx1013 (llvm-mc:
            # "e64 variant of this instruction is not supported").  Keep the
            # real record length so the stream stays aligned, but do not
            # report a lowerable instruction.
            return (length, "v_pk_fmac_f16_unsupported",
                    "v_pk_fmac_f16 <unsupported VOP3A encoding>",
                    "VOP3A encoding of v_pk_fmac_f16 is not available on gfx1013")

        if is_vop3b:
            sdst = (w0 >> 8) & 0x7f
            abs_bits = [0, 0, 0]
            op_sel = [0, 0, 0, 0]
        else:
            sdst = None
            abs_bits = [(w0 >> 8) & 1, (w0 >> 9) & 1, (w0 >> 10) & 1]
            op_sel = [0, 0, 0, 0] if restricted else [(w0 >> 11) & 1, (w0 >> 12) & 1, (w0 >> 13) & 1, (w0 >> 14) & 1]

        def vop3_operand(reg_id: int, is_abs: int, is_neg: int, width: int = 1) -> str:
            if reg_id == 255 and literal is not None:
                val = f"0x{literal:x}"
            elif width == 2:
                val = decode_v_reg64(reg_id)
            elif width == 4:
                val = decode_v_reg128(reg_id)
            else:
                val = decode_v_reg(reg_id)
            return _format_operand_modifiers(val, is_abs, is_neg)

        raw_srcs = [src0, src1, src2][:arity]
        src_strs = [vop3_operand(raw_srcs[i], abs_bits[i], neg[i], src_widths[i]) for i in range(arity)]

        if op_name == "v_readlane_b32" or is_vopc:
            dst_str = decode_s_reg(vdst)
            asm = f"{op_name} {dst_str}, {', '.join(src_strs)}"
        elif is_vop3b:
            sdst_str = decode_s_reg(sdst)
            dst_str = _format_vop3_vgpr_dst(vdst, dst_width)
            asm = f"{op_name} {dst_str}, {sdst_str}, {', '.join(src_strs)}"
        else:
            dst_str = _format_vop3_vgpr_dst(vdst, dst_width)
            asm = f"{op_name} {dst_str}, {', '.join(src_strs)}"

        if any(op_sel):
            if op_sel[3]:
                asm += f" op_sel:[{','.join(str(b) for b in op_sel[:4])}]"
            else:
                asm += f" op_sel:[{','.join(str(b) for b in op_sel[:3])}]"

        asm += _vop3_output_modifiers(clamp, omod)

        return (length, op_name, asm, comment)

    # Default fallback: unknown 32-bit word
    return (4, f"word_0x{w0:08x}", f".long 0x{w0:08x}", "Unrecognized RDNA 2 instruction encoding")


def disassemble_binary(data: bytes, shader_name: str = "shader") -> list[dict]:
    """Decode the complete span and mark records after the first s_endpgm.

    after_endpgm describes byte order only. A branch can still reach a later record.
    """
    instructions = []
    offset = 0
    after_endpgm = False
    while offset < len(data):
        length, op_name, asm, comment = decode_instruction(data, offset)
        hex_bytes = data[offset:offset + length].hex()
        instructions.append({
            "offset": offset,
            "length": length,
            "hex": hex_bytes,
            "opcode": op_name,
            "assembly": asm,
            "comment": comment,
            "after_endpgm": after_endpgm,
        })
        offset += length
        if op_name == "s_endpgm":
            after_endpgm = True
    return instructions


# Alias for library usage
disasm_rdna2 = disassemble_binary


def print_disassembly(shader_name: str, instructions: list[dict], show_comments: bool = True):
    """Print beautifully formatted disassembly with offsets and annotations."""
    print("=" * 80)
    print(f" RDNA 2 DISASSEMBLY: {shader_name} ")
    print("=" * 80)
    print(f"{'Offset':<8} {'Raw Hex':<18} {'Assembly':<42} {'Notes'}")
    print("-" * 80)

    for item in instructions:
        off_str = f"+0x{item['offset']:04x}"
        hex_str = item["hex"]
        asm_str = item["assembly"]
        comment_str = f"// {item['comment']}" if (show_comments and item["comment"]) else ""
        print(f"{off_str:<8} {hex_str:<18} {asm_str:<42} {comment_str}")

    print("=" * 80)
    total_bytes = sum(i["length"] for i in instructions)
    opcodes = Counter(i["opcode"] for i in instructions)
    print(f"Summary: {len(instructions)} instructions | {total_bytes} bytes | Top opcodes: {', '.join(f'{k}:{v}' for k, v in opcodes.most_common(5))}")
    print()


def _paths_collide(path_a: Path, path_b: Path) -> bool:
    """Return True when two paths resolve to or refer to the same file."""
    if path_a.resolve() == path_b.resolve():
        return True
    try:
        return path_a.exists() and path_b.exists() and path_a.samefile(path_b)
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", type=Path, default=[], help="Path to shader binary .bin file(s) or folder")
    parser.add_argument("--hex", default=None, help="Direct hex string to decode (e.g. '0300a0bf8002047e')")
    parser.add_argument("--output", type=Path, default=None, help="Save disassembly listing to output file")

    args = parser.parse_args(argv)

    if args.output is not None:
        try:
            if args.output.is_dir():
                print(f"error: {args.output}: output path is a directory", file=sys.stderr)
                return 1
        except OSError as exc:
            print(f"error: {args.output}: {exc}", file=sys.stderr)
            return 1

    all_listings = []
    had_error = False

    # Hex decode mode
    if args.hex is not None:
        hex_text = "".join(args.hex.split())
        if not hex_text:
            parser.error("--hex: expected a non-empty hex string")
        try:
            raw_bytes = bytes.fromhex(hex_text)
        except ValueError as exc:
            parser.error(f"--hex: {exc}")
        insns = disassemble_binary(raw_bytes, "hex_input")
        print_disassembly("hex_input", insns)
        all_listings.append(("hex_input", insns))
        args.targets = []
    elif not args.targets:
        parser.error("Specify targets or provide hex via --hex")

    for target in args.targets:
        try:
            if target.is_file():
                if args.output is not None and _paths_collide(args.output, target):
                    print(f"error: --output {args.output} collides with input {target}", file=sys.stderr)
                    return 1
                data = target.read_bytes()
                insns = disassemble_binary(data, target.name)
                print_disassembly(target.name, insns)
                all_listings.append((target.name, insns))
            elif target.is_dir():
                files = sorted(f for f in target.glob("*.bin") if f.is_file())
                preview = files[:20]  # Limit preview to 20 when folder passed
                if args.output is not None:
                    for f in preview:
                        if _paths_collide(args.output, f):
                            print(f"error: --output {args.output} collides with input {f}", file=sys.stderr)
                            return 1
                print(f"[*] Found {len(files)} shader binary files in {target.name}...")
                for f in preview:
                    data = f.read_bytes()
                    insns = disassemble_binary(data, f.name)
                    print_disassembly(f.name, insns)
                    all_listings.append((f.name, insns))
            else:
                print(f"error: {target}: not found", file=sys.stderr)
                had_error = True
        except OSError as exc:
            print(f"error: {target}: {exc}", file=sys.stderr)
            had_error = True

    if args.output is not None and all_listings and not had_error:
        tmp_path = None
        try:
            output_path = args.output
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as out_f:
                tmp_path = Path(out_f.name)
                for name, insns in all_listings:
                    out_f.write(f"; Disassembly: {name}\n")
                    for item in insns:
                        out_f.write(f"    {item['assembly']:<40} ; +0x{item['offset']:04x} // {item['comment']}\n")
                    out_f.write("\n")
            tmp_path.replace(output_path)
            tmp_path = None
            print(f"[+] Disassembly saved to: {args.output}")
        except OSError as exc:
            print(f"error: {args.output}: {exc}", file=sys.stderr)
            return 1
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
