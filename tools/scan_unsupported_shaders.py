#!/usr/bin/env python3
"""
Scan RDNA 2 / GFX1013 shaders for unsupported instructions and generate detailed logs.

Analyzes disassembly (.asm, .s, .txt), binary shader records, or RPF archives,
identifies instructions not yet supported by Xbox2Windows pixel_aot translator,
and correlates them with the AMD RDNA 2 Instruction Set Architecture (ISA).
"""

import argparse
from collections import Counter, defaultdict
import ctypes as C
import datetime
import json
import os
from pathlib import Path
import re
import struct
import sys
import zlib

# Try importing project helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import pixel_aot
    from resource_inventory import parts
except ImportError:
    pixel_aot = None
    parts = None

try:
    from disasm_rdna2 import DS_OPS, FLAT_OPS, GLOBAL_OPS, MTBUF_OPS, MUBUF_OPS, SCRATCH_OPS, disasm_rdna2
except ImportError:
    DS_OPS = {}
    MTBUF_OPS = {}
    MUBUF_OPS = {}
    FLAT_OPS = {}
    GLOBAL_OPS = {}
    SCRATCH_OPS = {}
    disasm_rdna2 = None

# Entry bindings passed to pixel_aot.lower() for every record. The scan carries no per-record binding contract,
# so translator acceptance is measured under these bindings only. A read of a register lower() holds no value for
# under these inputs is reported as an unbound operand rather than as an instruction the translator refuses.
SCAN_BINDINGS = {"texture_sgpr": 20, "interp_sgpr": 24, "sampler": (1, 2, 3, 4), "constant_descriptor": None}
SCAN_BINDINGS_TEXT = ("texture descriptor table at s20, interpolation base at s24, sampler words 1 2 3 4, "
                      "no constant buffers")


def lower_rejection_causes(results: list[dict]) -> dict:
    """Split the records lower() rejected by cause: unbound scalar or vector operand, translator rejection
    (any other ValueError) or an unexpected exception (a translator fault, never counted as a missing binding)."""
    causes = {"unbound_scalar_operand": 0, "unbound_vector_operand": 0,
              "translator_rejection": 0, "unexpected_error": 0}
    for r in results:
        kind = r.get("lower_error_kind")
        if kind == "unbound_operand":
            causes[f"unbound_{r['lower_unbound']['bank']}_operand"] += 1
        elif kind in causes:
            causes[kind] += 1
    return causes


# Known supported instructions and patterns in pixel_aot.py
# Tables are imported from pixel_aot so the scanner stays aligned with the translator.
PIXEL_ALU_NAMES = tuple(pixel_aot.ALU) if pixel_aot is not None else ()
PIXEL_SDWA_ALU = tuple(pixel_aot.SDWA_ALU) if pixel_aot is not None else ()
SUPPORTED_ALU = {f"v_{name}_f32" for name in PIXEL_ALU_NAMES}
PIXEL_INTEGER = pixel_aot.INTEGER if pixel_aot is not None else {}
PIXEL_SDWA_INT = pixel_aot.SDWA_INT if pixel_aot is not None else {}
PIXEL_SCALAR = pixel_aot.SCALAR if pixel_aot is not None else {}
PIXEL_SALU_BIT64 = tuple(pixel_aot.SALU_BIT64) if pixel_aot is not None else ()
PIXEL_COMPARE = tuple(pixel_aot.COMPARE) if pixel_aot is not None else ()
PIXEL_INT_COMPARE = tuple(pixel_aot.INT_COMPARE) if pixel_aot is not None else ()
PIXEL_VOP3P_F16 = pixel_aot.VOP3P_F16 if pixel_aot is not None else {}
PIXEL_VOP3P_INT16 = pixel_aot.VOP3P_INT16 if pixel_aot is not None else {}
PIXEL_VOP3_INT16 = pixel_aot.VOP3_INT16 if pixel_aot is not None else {}
PIXEL_VOP2_F16 = tuple(pixel_aot.VOP2_F16) if pixel_aot is not None else ()
PIXEL_VOP1_F16_ROUND = tuple(pixel_aot.VOP1_F16_ROUND) if pixel_aot is not None else ()
PIXEL_VOP2_F16_FUSED_SCALE = tuple(pixel_aot.VOP2_F16_FUSED_SCALE) if pixel_aot is not None else ()
PIXEL_INLINE_F16 = tuple(pixel_aot.INLINE_F16) if pixel_aot is not None else ()
PIXEL_VCC = r"(?:vcc(?:_lo)?)"
PIXEL_VOP3_INT16_TERNARY = pixel_aot.VOP3_INT16_TERNARY if pixel_aot is not None else {}
PIXEL_LANE_MASK = pixel_aot.LANE_MASK if pixel_aot is not None else r"(?:vcc(?:_lo)?|s\d+|s\[\d+:\d+\])"
# Explicit _e32 VOPC writes VCC only; suffix-less and _e64 may name an SGPR destination.
PIXEL_VOPC_DST = rf"(?:_e32\s+{PIXEL_VCC}|(?:_e64)?\s+{PIXEL_LANE_MASK})"
PIXEL_VOPC_E32_DST = rf"(?:_e32\s+{PIXEL_VCC}|\s+{PIXEL_LANE_MASK})"
INTEGER_SOURCE = r"(?:[vs]\d+|0x[0-9a-f]{1,8}|-?\d+(?:\.\d+)?)"
# 16-bit integer operand as pixel_aot.int16_operand accepts it: VGPR 0-255, SGPR 0-105, inline -16..64 or a
# hex literal up to 0xffff (leading zeros allowed, up to 8 digits).
INT16_VGPR = r"v(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"
INT16_SOURCE = (
    rf"(?:{INT16_VGPR}|s(?:[0-9]|[1-9][0-9]|10[0-5])|0x0{{0,4}}[0-9a-f]{{1,4}}"
    r"|-?(?:[0-9]|1[0-6])|(?:1[7-9]|[2-5][0-9]|6[0-4]))"
)
# Source of the plain binary16 e32 arithmetic and rounding forms as pixel_aot.half16_arith_source reads it: a binary16 inline
# constant, 0, a 16-bit literal or a register. Integer inline constants other than 0 and every modifier are refused.
F16_ARITH_SOURCE = (
    r"(?:" + "|".join(map(re.escape, PIXEL_INLINE_F16 + ("0",)))
    + r"|0x0*[0-9a-f]{1,4}|v(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])|s(?:[0-9]|[1-9][0-9]|10[0-5]))"
)
SCALAR32_SOURCE = r"(?:s\d+|0x[0-9a-f]{1,8}|-?\d+(?:\.\d+)?)"
SCALAR64_SOURCE = r"(?:s\[\d+:\d+\]|0x[0-9a-f]{1,8}|-?\d+)"
F16_PLAIN = r"(?:-?(?:\d+\.\d+|\d+)|0x[0-9a-f]{1,4}|[vs]\d+)"
F16_MOD = rf"-?(?:\|{F16_PLAIN}\||abs\({F16_PLAIN}\)|{F16_PLAIN})"
SDWA_SEL_PATTERN = r"(?:BYTE_[0-3]|WORD_[01]|DWORD)"
PIXEL_SDWA_DST = r"(?:vcc(?:_lo)?|s\d+)"
SDWA_CMP_SOURCE = rf"(?:sext\({INTEGER_SOURCE}\)|{INTEGER_SOURCE})"
CVT_F32_E64_BASE = r"(?:[vs]\d+|0x[0-9a-f]{1,8})"
CVT_F32_E64_ABS = rf"(?:\|{CVT_F32_E64_BASE}\||abs\({CVT_F32_E64_BASE}\))"
CVT_F32_E64_SOURCE = rf"(?:neg\((?:{CVT_F32_E64_ABS}|{CVT_F32_E64_BASE})\)|-(?:{CVT_F32_E64_ABS}|[vs]\d+)|{CVT_F32_E64_ABS}|{CVT_F32_E64_BASE})"

WAIT_COUNTER = r"(?:vmcnt\((?:[0-9]|[1-5][0-9]|6[0-3])\)|lgkmcnt\((?:[0-9]|[1-5][0-9]|6[0-3])\)|expcnt\([0-7]\))"

SUPPORTED_INSTRUCTIONS_REGEX = [
    r"s_inst_prefetch\s+.*",
    r"s_nop\s+.*",
    r"s_version\s+(?:0x0*[0-9a-f]{1,4}|0|[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d\d|655[0-2]\d|6553[0-5])",
    r"v_nop(?:_e64)?",
    r"s_clause\s+(?:0x[0-3]?[0-9a-f]|[0-5]?[0-9]|6[0-3])",
    r"s_setprio\s+(?:0x[0-3]|[0-3])",
    r"s_sleep\s+(?:0x[0-7]?[0-9a-f]|[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7])",
    # Performance-level and trace hints, as in pixel_aot: a 4-bit counter, a 16-bit field, or no operand.
    r"s_(?:inc|dec)perflevel\s+(?:0x[0-9a-f]|[0-9]|1[0-5])",
    r"s_ttracedata_imm\s+(?:0x0*[0-9a-f]{1,4}|0|[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d\d|655[0-2]\d|6553[0-5])",
    r"s_icache_inv",
    r"s_ttracedata",
    r"s_wakeup",
    r"s_setreg_imm32_b32\s+hwreg\(HW_REG_FLAT_SCR_(?:LO|HI)\),\s+(?:-?\d+|0x[0-9a-f]{1,8})",
    r"s_setreg_imm32_b32\s+hwreg\(HW_REG_MODE, 2, 2\),\s+[03]",
    # s_round_mode keeping binary32 at nearest even and writing MODE[3:2] = 0 or 3, as pixel_aot lowers it.
    r"s_round_mode\s+0x[0c]",
    r"s_waitcnt_expcnt\s+(?:(?:null|s\d+),\s*)?(?:0x[0-7]|[0-7])",
    # SOPK wait forms: only null as the SGPR (reads as 0) and a count of 0..63, as in pixel_aot.
    r"s_waitcnt_(?:vmcnt|lgkmcnt|vscnt)\s+null,\s*(?:0x(?:0?[0-9a-f]|[1-3][0-9a-f])|[0-9]|[1-5][0-9]|6[0-3])",
    rf"s_waitcnt\s+(?:{WAIT_COUNTER}(?:\s+{WAIT_COUNTER}){{0,2}}|0(?:x0)?)",
    r"s_mov_b64\s+(?:(?:s\[\d+:\d+\]|s\d+),\s+(?:exec|exec_lo)|(?:exec|exec_lo),\s+(?:s\[\d+:\d+\]|s\d+))",
    r"s_mov_b32\s+(?:s\d+,\s+(?:exec|exec_lo)|(?:exec|exec_lo),\s+s\d+)",
    r"s_wqm_(?:b64|b32)\s+(?:exec|exec_lo),\s+(?:exec|exec_lo)",
    r"s_and_b64\s+(?:exec|exec_lo),\s+(?:exec|exec_lo),\s+(?:s\[\d+:\d+\]|s\d+)",
    r"s_and_b32\s+(?:exec|exec_lo),\s+(?:exec|exec_lo),\s+s\d+",
    # The structured masked region pixel_aot lowers: the s_and_saveexec opener, the s_or close, the s_andn2 else
    # transition and the empty-mask skip, in the wave64 spellings and the wave32 ones on exec_lo (the MOV close is the
    # s_mov line above). Whether a line sits in a well-formed region is decided by the translator, not here.
    r"s_and_saveexec_b64\s+s\[\d+:\d+\],\s+(?:vcc|s\[\d+:\d+\])",
    r"s_and_saveexec_b32\s+s\d+,\s+(?:vcc_lo|s\d+)",
    r"s_or_b64\s+exec,\s+exec,\s+s\[\d+:\d+\]",
    r"s_or_b32\s+exec_lo,\s+exec_lo,\s+s\d+",
    r"s_andn2_b64\s+exec,\s+s\[\d+:\d+\],\s+exec",
    r"s_andn2_b32\s+exec_lo,\s+s\d+,\s+exec_lo",
    r"s_cbranch_execz\s+-?(?:0x[0-9a-f]+|0|[1-9]\d*)",
    rf"s_(?:cmov|not|brev)_b64\s+s\[\d+:\d+\],\s+{SCALAR64_SOURCE}",
    rf"s_(?:lshl_b64|lshr_b64|ashr_i64)\s+s\[\d+:\d+\],\s+{SCALAR64_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"s_bfm_b64\s+s\[\d+:\d+\],\s+{SCALAR32_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"(?:s_bfe_u32|s_bfe_i32)\s+s\d+,\s+{SCALAR32_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"(?:s_bfe_u64|s_bfe_i64)\s+s\[\d+:\d+\],\s+{SCALAR64_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"(?:s_cmp_eq_u64|s_cmp_lg_u64)\s+{SCALAR64_SOURCE},\s+{SCALAR64_SOURCE}",
    rf"(?:s_bitcmp0_b64|s_bitcmp1_b64)\s+{SCALAR64_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"(?:s_bitcmp0_b32|s_bitcmp1_b32)\s+{SCALAR32_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"s_cmp_(?:eq|lg|gt|ge|lt|le)_(?:i32|u32)\s+{SCALAR32_SOURCE},\s+{SCALAR32_SOURCE}",
    r"s_cmpk_(?:eq|lg|gt|ge|lt|le)_i32\s+s\d+,\s+(?:-?\d+|0x[0-9a-f]+)",
    r"s_cmpk_(?:eq|lg|gt|ge|lt|le)_u32\s+s\d+,\s+(?:\d+|0x[0-9a-f]+)",
    rf"s_cselect_b32\s+s\d+,\s+{SCALAR32_SOURCE},\s+{SCALAR32_SOURCE}",
    rf"s_(?:cmov|bitset0|bitset1)_b32\s+s\d+,\s+{SCALAR32_SOURCE}",
    rf"(?:s_mov_b32|s_movk_i32)\s+(?:s\d+|m0),\s+(?:{SCALAR32_SOURCE}|exec|exec_lo)",
    r"(?:s_addk_i32|s_mulk_i32|s_cmovk_i32)\s+s\d+,\s+(?:-?\d+|0x[0-9a-f]+)",
    r"s_load_dwordx8\s+s\[\d+:\d+\],\s+s\[\d+:\d+\],\s+(?:null|0x[0-9a-f]+)",
    r"(?:s_load_dwordx4|s_buffer_load_dwordx[24])\s+s\[\d+:\d+\],\s+s\[\d+:\d+\],\s+(?:null|0x[0-9a-f]+)",
    r"s_buffer_load_dword\s+s\d+,\s+s\[\d+:\d+\],\s+(?:null|0x[0-9a-f]+)",
    r"v_interp_p[12]_f32_e(?:32|64)\s+v\d+,\s+v\d+,\s+attr0\.[xyzw]",
    r"image_sample\s+v\[\d+:\d+\],\s+v\[\d+:\d+\],\s+s\[\d+:\d+\],\s+s\[\d+:\d+\]\s+dmask:0x[7f]\s+dim:SQ_RSRC_IMG_2D",
    r"v_cube(?:id|sc|tc|ma)_f32(?:_e64)?\s+v\d+,\s+[^,]+,\s+[^,]+,\s+[^,]+?(?:\s+clamp)?",
    r"v_(?:fmaak|fmamk)_f32\s+v\d+,\s+[^,]+,\s+[^,]+,\s+[^,]+",
    # K must be a 32-bit literal: last operand of v_madak_f32, second of v_madmk_f32 (as pixel_aot requires).
    r"v_madak_f32\s+v\d+,\s+[^,]+,\s+[^,]+,\s+0x[0-9a-f]{1,8}",
    r"v_madmk_f32\s+v\d+,\s+[^,]+,\s+0x[0-9a-f]{1,8},\s+[^,]+",
    r"v_mov_b32(?:_e32|_e64)?\s+v\d+,\s+(?:-?\d+(?:\.\d+)?|0x[0-9a-f]{1,8}|[vs]\d+)",
    r"v_swap_b32(?:_e32)?\s+v\d+,\s+v\d+",
    rf"v_readfirstlane_b32(?:_e32)?\s+s\d+,\s+{INTEGER_SOURCE}",
    rf"v_mov_b32_sdwa\s+v\d+,\s+(?:sext\((?:[vs]\d+|-?\d+(?:\.\d+)?)\)|[vs]\d+|-?\d+(?:\.\d+)?)\s+dst_sel:{SDWA_SEL_PATTERN}\s+dst_unused:(?:UNUSED_PAD|UNUSED_PRESERVE)\s+src0_sel:{SDWA_SEL_PATTERN}",
    r"v_cvt_f32_f16(?:_e32|_e64)?\s+v\d+,\s+-?(?:\|(?:[vs]\d+|0x[0-9a-f]{1,8}|\d+)\||abs\((?:[vs]\d+|0x[0-9a-f]{1,8}|\d+)\)|[vs]\d+|0x[0-9a-f]{1,8}|\d+)(?:\s+clamp)?",
    r"v_cvt_f32_f16_sdwa\s+v\d+,\s+-?(?:\|[vs]\d+\||[vs]\d+)(?:\s+clamp)?\s+dst_sel:DWORD\s+dst_unused:UNUSED_(?:PAD|PRESERVE)\s+src0_sel:WORD_[01]",
    r"v_cvt_f16_f32(?:_e32|_e64)?\s+v\d+,\s+[^,\s]+",
    r"v_cvt_(?:i32|flr_i32|rpi_i32)_f32_e32 v\d+, (?:[vs]\d+|0x[0-9a-f]{1,8})",
    rf"v_cvt_(?:i32|u32|flr_i32|rpi_i32)_f32_e64 v\d+, {CVT_F32_E64_SOURCE}",
    r"v_cvt_f32_ubyte[0-3](?:_e32|_e64)?\s+v\d+,\s+(?:[vs]\d+|0x[0-9a-f]{1,8}|\d+)(?:\s+clamp)?",
    r"v_cvt_off_f32_i4(?:_e32|_e64)?\s+v\d+,\s+(?:[vs]\d+|0x[0-9a-f]{1,8}|\d+)(?:\s+clamp)?",
    r"v_cvt_f32_(?:u32|i32)_e32\s+v\d+,\s+(?:[vs]\d+|0x[0-9a-f]{1,8}|-?\d+)",
    r"v_cvt_f32_(?:u32|i32)_e64\s+v\d+,\s+(?:[vs]\d+|0x[0-9a-f]{1,8}|-?\d+)(?:\s+clamp)?",
    r"v_ldexp_f32(?:_e64)?\s+v\d+,\s+[^,]+,\s+[^,]+?(?:\s+clamp)?",
    rf"v_(?:(?:add|sub|subrev)_nc_u32(?:_e32|_e64)?|(?:add|sub)_nc_i32)\s+v\d+,\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE}",
    rf"v_(?:add|sub|subrev)_co_u32\s+v\d+,\s+{PIXEL_LANE_MASK},\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE}",
    rf"v_(?:add|sub|subrev)_co_ci_u32_e32\s+v\d+,\s+{PIXEL_VCC},\s+{INTEGER_SOURCE},\s+v\d+,\s+{PIXEL_VCC}",
    rf"v_(?:add|sub|subrev)_co_ci_u32_e64\s+v\d+,\s+{PIXEL_LANE_MASK},\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE},\s+{PIXEL_LANE_MASK}",
    rf"v_(?:min|max)_(?:i32|u32)(?:_e32|_e64)?\s+v\d+,\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE}",
    rf"v_(?:min3|max3|med3)_(?:i32|u32)(?:_e64)?\s+v\d+,\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE}",
    rf"v_cmp_class_f32{PIXEL_VOPC_DST},\s+[^,]+,\s+[^,]+",
    rf"v_cmp_class_f16{PIXEL_VOPC_E32_DST},\s+{F16_PLAIN},\s+(?:-?\d+|0x[0-9a-f]{{1,8}}|[vs]\d+)",
    rf"v_cmp_class_f16_e64\s+{PIXEL_LANE_MASK},\s+{F16_MOD},\s+(?:-?\d+|0x[0-9a-f]{{1,8}}|[vs]\d+)",
    rf"v_cndmask_b32(?:_e32|_e64)?\s+v\d+,\s+[^,]+,\s+[^,]+,\s+{PIXEL_LANE_MASK}",
    # This pattern recognizes the opcode and destination. pixel_aot.lower validates
    # the selectors and source operands, so a scanner match can still fail lowering.
    r"v_fma_mix(?:_f32|lo_f16|hi_f16)\s+v\d+,\s+.+",
    r"v_div_fmas_f32(?:_e64)?\s+v\d+,\s+[^,\s]+,\s+[^,\s]+,\s+[^,\s]+(?:\s+clamp)?",
    r"(?:v_dot2_f32_f16|v_dot2_i32_i16|v_dot2_u32_u16|v_dot4_i32_i8|v_dot4_u32_u8|v_dot8_i32_i4|v_dot8_u32_u4)\s+v\d+,\s+.+",
    r"v_cvt_pkrtz_f16_f32_e32\s+v\d+,\s+v\d+,\s+v\d+",
    r"v_cvt_u32_f32_e32 v\d+, (?:[vs]\d+|0x[0-9a-f]{1,8})",
    r"exp\s+mrt0,?\s+v\d+,\s+v\d+,\s+(?:v\d+|off),\s+(?:v\d+|off)\s+done\s+compr\s+vm",
    r"exp\s+mrt0,?\s+(?:v\d+|off),\s+(?:v\d+|off),\s+(?:v\d+|off),\s+(?:v\d+|off)\s+done\s+vm",
    r"s_endpgm",
]

if PIXEL_ALU_NAMES:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        r"v_(?:" + "|".join(map(re.escape, PIXEL_ALU_NAMES)) + r")_f32(?:_e32|_e64)?\s+v\d+,\s+.*"
    )
if PIXEL_SDWA_ALU:
    # Whole-dword sources only, as pixel_aot requires. This pins the spelling, not the operand count.
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        r"v_(?:" + "|".join(map(re.escape, PIXEL_SDWA_ALU)) + r")_f32_sdwa\s+v\d+,\s+.+?(?:\s+clamp)?"
        rf"\s+dst_sel:{SDWA_SEL_PATTERN}\s+dst_unused:(?:UNUSED_PAD|UNUSED_PRESERVE)"
        r"\s+src0_sel:DWORD(?:\s+src1_sel:DWORD)?"
    )
if PIXEL_SALU_BIT64:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"s_(?:{'|'.join(map(re.escape, PIXEL_SALU_BIT64))})_b64\s+s\[\d+:\d+\],\s+{SCALAR64_SOURCE},\s+{SCALAR64_SOURCE}"
    )
if PIXEL_COMPARE:
    cmp_ops = "|".join(map(re.escape, PIXEL_COMPARE))
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{cmp_ops})_f32{PIXEL_VOPC_DST},\s+[^,]+,\s+[^,]+"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{cmp_ops})_f32_sdwa\s+{PIXEL_LANE_MASK},\s+[^,]+,\s+[^,]+\s+src0_sel:DWORD\s+src1_sel:DWORD"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{cmp_ops})_f16{PIXEL_VOPC_E32_DST},\s+{F16_PLAIN},\s+{F16_PLAIN}"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{cmp_ops})_f16_e64\s+{PIXEL_LANE_MASK},\s+{F16_MOD},\s+{F16_MOD}"
    )
    # VOPC SDWA has only the two source selections; the destination is VCC or one SGPR.
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{cmp_ops})_f16_sdwa\s+{PIXEL_SDWA_DST},\s+{SDWA_CMP_SOURCE},\s+{SDWA_CMP_SOURCE}"
        rf"\s+src0_sel:{SDWA_SEL_PATTERN}\s+src1_sel:{SDWA_SEL_PATTERN}"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_class_(?:f16|f32)_sdwa\s+{PIXEL_SDWA_DST},\s+{SDWA_CMP_SOURCE},\s+{SDWA_CMP_SOURCE}"
        rf"\s+src0_sel:{SDWA_SEL_PATTERN}\s+src1_sel:{SDWA_SEL_PATTERN}"
    )
if PIXEL_INT_COMPARE:
    icmp_ops = "|".join(map(re.escape, PIXEL_INT_COMPARE))
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{icmp_ops})_(?:i32|u32|i16|u16){PIXEL_VOPC_DST},\s+{INTEGER_SOURCE},\s+{INTEGER_SOURCE}"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:{icmp_ops})_(?:i32|u32)_sdwa\s+{PIXEL_SDWA_DST},\s+{SDWA_CMP_SOURCE},\s+{SDWA_CMP_SOURCE}"
        rf"\s+src0_sel:{SDWA_SEL_PATTERN}\s+src1_sel:{SDWA_SEL_PATTERN}"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cmp_(?:eq|ne|lt|le|gt|ge)_(?:i16|u16)_sdwa\s+{PIXEL_SDWA_DST},\s+{SDWA_CMP_SOURCE},\s+{SDWA_CMP_SOURCE}"
        rf"\s+src0_sel:{SDWA_SEL_PATTERN}\s+src1_sel:{SDWA_SEL_PATTERN}"
    )
    # 64-bit comparisons are checked by supports_int64_compare() below: the pair and inline ranges are numeric.
    INT64_COMPARE_PATTERN = re.compile(
        rf"v_cmp_(?:{icmp_ops})_(?:i64|u64)(_e32|_e64)?\s+(vcc(?:_lo)?|s\d+|s\[\d+:\d+\]),\s+(\S+),\s+(\S+)"
    )
else:
    INT64_COMPARE_PATTERN = None
# Plain binary16 e32 and e64 forms whose result keeps only the low half (pixel_aot proves finite sources and results).
# The e32 src1 must be a VGPR; the e64 spelling takes any source that the e32 src0 accepts. Neither takes a modifier.
if PIXEL_VOP2_F16:
    _vop2_f16 = "|".join(map(re.escape, PIXEL_VOP2_F16))
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_(?:{_vop2_f16})_f16_e32\s+{INT16_VGPR},\s+{F16_ARITH_SOURCE},\s+{INT16_VGPR}"
    )
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_(?:{_vop2_f16})_f16_e64\s+{INT16_VGPR},\s+{F16_ARITH_SOURCE},\s+{F16_ARITH_SOURCE}"
    )
# The accumulating v_fmac_f16 and the scaling v_ldexp_f16 have the same e32 shape; pixel_aot additionally needs the accumulator
# and the exponent to be statically proven.
if PIXEL_VOP2_F16_FUSED_SCALE:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_(?:{'|'.join(map(re.escape, PIXEL_VOP2_F16_FUSED_SCALE))})_f16_e32\s+{INT16_VGPR},\s+{F16_ARITH_SOURCE},\s+{INT16_VGPR}"
    )
if PIXEL_VOP1_F16_ROUND:
    _vop1_f16_round = "|".join(map(re.escape, PIXEL_VOP1_F16_ROUND))
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_(?:{_vop1_f16_round})_f16_(?:e32|e64)\s+{INT16_VGPR},\s+{F16_ARITH_SOURCE}"
    )
if pixel_aot is not None:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_cvt_f16_(?:i16|u16)_e32\s+{INT16_VGPR},\s+{INT16_SOURCE}"
    )
if PIXEL_VOP3P_F16:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_pk_(?:{'|'.join(map(re.escape, PIXEL_VOP3P_F16))})_f16\s+v\d+,\s+.+"
    )
if PIXEL_VOP3P_INT16:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_pk_(?:{'|'.join(map(re.escape, PIXEL_VOP3P_INT16))})\s+v\d+,\s+.+"
    )
if PIXEL_VOP3_INT16:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_(?:{'|'.join(map(re.escape, PIXEL_VOP3_INT16))})(?:_e64)?\s+{INT16_VGPR},\s+{INT16_SOURCE},\s+{INT16_SOURCE}"
    )
if PIXEL_VOP3_INT16_TERNARY:
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_(?:{'|'.join(map(re.escape, PIXEL_VOP3_INT16_TERNARY))})(?:_e64)?\s+{INT16_VGPR},"
        rf"\s+{INT16_SOURCE},\s+{INT16_SOURCE},\s+{INT16_SOURCE}"
    )
# 64-bit VOP3 shifts (issue #928): a VGPR-pair destination, a 32-bit amount (VGPR, SGPR, inline integer or
# literal) and a 64-bit value (VGPR pair, SGPR pair or inline integer), with at most one SGPR or literal source.
# This matches syntax only; pixel_aot also checks the pair ranges and that the sources are tracked.
SHIFT64_AMOUNT = r"(?:v\d+|-?\d+)"
SHIFT64_VALUE = r"(?:v\[\d+:\d+\]|-?\d+)"
SUPPORTED_INSTRUCTIONS_REGEX.append(
    r"v_(?:lshlrev_b64|lshrrev_b64|ashrrev_i64)(?:_e64)?\s+v\[\d+:\d+\],\s+"
    rf"(?:{SHIFT64_AMOUNT},\s+{SHIFT64_VALUE}|(?:s\d+|0x[0-9a-f]{{1,8}}),\s+{SHIFT64_VALUE}"
    rf"|{SHIFT64_AMOUNT},\s+s\[\d+:\d+\])"
)

for name, (arity, _) in PIXEL_INTEGER.items():
    vop3_only = arity > 2 or name in getattr(pixel_aot, "INTEGER_VOP3_ONLY", ())
    encoding = r"(?:_e64)?" if vop3_only else r"(?:_e32|_e64)?"
    SUPPORTED_INSTRUCTIONS_REGEX.append(rf"v_{name}{encoding}\s+v\d+" + rf",\s*{INTEGER_SOURCE}" * arity)

# SDWA forms of the 32-bit integer operations: register sources, optionally under sext(), every field printed,
# and no UNUSED_SEXT. As for the moves, UNUSED_PRESERVE onto an untracked destination is refused by the translator only.
SDWA_INT_SOURCE = r"(?:sext\([vs]\d+\)|[vs]\d+)"
for name, (arity, _) in PIXEL_SDWA_INT.items():
    SUPPORTED_INSTRUCTIONS_REGEX.append(
        rf"v_{name}_sdwa\s+v\d+" + rf",\s+{SDWA_INT_SOURCE}" * arity
        + rf"\s+dst_sel:{SDWA_SEL_PATTERN}\s+dst_unused:(?:UNUSED_PAD|UNUSED_PRESERVE)"
        + "".join(rf"\s+src{index}_sel:{SDWA_SEL_PATTERN}" for index in range(arity)))

for name, (arity, _, _) in PIXEL_SCALAR.items():
    SUPPORTED_INSTRUCTIONS_REGEX.append(rf"s_{name}\s+s\d+" + rf",\s*{SCALAR32_SOURCE}" * arity)

COMPILED_SUPPORTED_PATTERNS = [re.compile(f"^{p}$") for p in SUPPORTED_INSTRUCTIONS_REGEX]

KNOWN_RDNA2_BASE_OPCODES = {
    # SOPP
    "s_nop", "s_endpgm", "s_code_end", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
    "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz",
    "s_barrier", "s_waitcnt", "s_inst_prefetch", "s_clause", "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt", "s_trap", "s_round_mode",
    # SOP1
    "s_mov_b32", "s_mov_b64", "s_cmov_b32", "s_cmov_b64", "s_not_b32", "s_not_b64",
    "s_brev_b32", "s_brev_b64", "s_bcnt0_i32_b32", "s_bcnt0_i32_b64", "s_bcnt1_i32_b32", "s_bcnt1_i32_b64",
    "s_ff0_i32_b32", "s_ff0_i32_b64", "s_ff1_i32_b32", "s_ff1_i32_b64",
    "s_flbit_i32_b32", "s_flbit_i32_b64", "s_flbit_i32", "s_flbit_i32_i64",
    "s_sext_i32_i8", "s_sext_i32_i16", "s_bitset0_b32", "s_bitset0_b64", "s_bitset1_b32", "s_bitset1_b64",
    "s_wqm_b32", "s_wqm_b64", "s_getpc_b64", "s_swappc_b64", "s_setpc_b64", "s_rfe_b64",
    "s_and_saveexec_b32", "s_or_saveexec_b32", "s_xor_saveexec_b32", "s_andn2_saveexec_b32",
    "s_orn2_saveexec_b32", "s_nand_saveexec_b32", "s_nor_saveexec_b32", "s_xnor_saveexec_b32",
    "s_andn1_saveexec_b32", "s_orn1_saveexec_b32", "s_andn1_wrexec_b32", "s_andn2_wrexec_b32",
    "s_and_saveexec_b64", "s_or_saveexec_b64", "s_xor_saveexec_b64", "s_andn2_saveexec_b64",
    "s_orn2_saveexec_b64", "s_nand_saveexec_b64", "s_nor_saveexec_b64", "s_xnor_saveexec_b64",
    "s_quadmask_b32", "s_quadmask_b64", "s_abs_i32",
    "s_andn1_saveexec_b64", "s_orn1_saveexec_b64", "s_andn1_wrexec_b64", "s_andn2_wrexec_b64",
    "s_bitreplicate_b64_b32",
    # Relative scalar moves: decoded by tools/disasm_rdna2.py, recognition only, no lowering.
    "s_movrels_b32", "s_movrels_b64", "s_movreld_b32", "s_movreld_b64", "s_movrelsd_2_b32",
    # SOP2
    "s_add_u32", "s_sub_u32", "s_add_i32", "s_sub_i32", "s_absdiff_i32", "s_addc_u32", "s_subb_u32",
    "s_min_i32", "s_min_u32", "s_max_i32", "s_max_u32", "s_cselect_b32", "s_cselect_b64",
    "s_and_b32", "s_and_b64", "s_or_b32", "s_or_b64", "s_xor_b32", "s_xor_b64",
    "s_andn2_b32", "s_andn2_b64", "s_orn2_b32", "s_orn2_b64", "s_nand_b32", "s_nand_b64",
    "s_nor_b32", "s_nor_b64", "s_lshl_b32", "s_lshl_b64", "s_lshr_b32", "s_lshr_b64",
    "s_ashr_i32", "s_ashr_i64", "s_bfm_b32", "s_bfm_b64", "s_mul_i32",
    "s_bfe_u32", "s_bfe_i32", "s_bfe_u64", "s_bfe_i64",
    "s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32",
    "s_mul_hi_u32", "s_mul_hi_i32",
    # SOPK
    "s_movk_i32", "s_cmovk_i32", "s_addk_i32", "s_mulk_i32", "s_cbranch_i_fork",
    "s_getreg_b32", "s_setreg_b32", "s_setreg_imm32_b32", "s_version",
    # VOP1
    "v_nop", "v_mov_b32", "v_swap_b32", "v_swaprel_b32", "v_readfirstlane_b32", "v_not_b32", "v_cls_i32",
    "v_log_f32", "v_exp_f32", "v_rcp_f32", "v_rsq_f32", "v_sqrt_f32",
    "v_sin_f32", "v_cos_f32", "v_bfrev_b32", "v_ffbh_u32", "v_ffbl_b32", "v_ffbh_i32", "v_rcp_iflag_f32",
    "v_cvt_f32_i32", "v_cvt_f32_u32", "v_cvt_i32_f32", "v_cvt_u32_f32",
    "v_cvt_f16_f32", "v_cvt_f32_f16", "v_cvt_rpi_i32_f32", "v_cvt_flr_i32_f32",
    "v_cvt_pkrtz_f16_f32", "v_ceil_f32", "v_floor_f32", "v_rndne_f32",
    "v_fract_f32", "v_trunc_f32", "v_frexp_mant_f32",
    # VOP2
    "v_cndmask_b32", "v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mac_legacy_f32", "v_mul_legacy_f32",
    "v_mul_f32", "v_mul_i32_i24", "v_mul_hi_i32_i24", "v_mul_u32_u24", "v_mul_hi_u32_u24",
    "v_min_f32", "v_max_f32", "v_min_i32", "v_max_i32", "v_min_u32", "v_max_u32",
    "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32", "v_and_b32", "v_or_b32", "v_xor_b32", "v_xnor_b32",
    "v_add_u32", "v_sub_u32", "v_subrev_u32", "v_addc_co_u32", "v_subb_co_u32", "v_subbrev_co_u32",
    "v_mac_f32", "v_fmac_f32", "v_fmamk_f32", "v_fmaak_f32", "v_madmk_f32", "v_madak_f32",
    # VOP3
    "v_fma_f32", "v_fma_f16", "v_mad_f32", "v_div_scale_f32", "v_div_fmas_f32", "v_bfi_b32",
    "v_alignbit_b32", "v_alignbyte_b32",
    # VOP3P
    "v_pk_mad_i16", "v_pk_mul_lo_u16", "v_pk_add_i16", "v_pk_sub_i16",
    "v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16",
    "v_pk_max_i16", "v_pk_min_i16", "v_pk_mad_u16", "v_pk_add_u16",
    "v_pk_sub_u16", "v_pk_max_u16", "v_pk_min_u16", "v_pk_fma_f16",
    "v_pk_add_f16", "v_pk_mul_f16", "v_pk_min_f16", "v_pk_max_f16",
    "v_dot2_f32_f16", "v_dot2_i32_i16", "v_dot2_u32_u16",
    "v_dot4_i32_i8", "v_dot4_u32_u8", "v_dot8_i32_i4", "v_dot8_u32_u4",
        "v_mad_legacy_f32", "v_mad_i32_i24", "v_mad_u32_u24", "v_fma_f64", "v_lerp_u8", "v_alignbit_b32", "v_alignbyte_b32", "v_mullit_f32", "v_min3_f32", "v_max3_f32", "v_med3_f32", "v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_cvt_pk_u8_f32", "v_div_fixup_f32", "v_div_fixup_f64", "v_add_f64", "v_mul_f64", "v_min_f64", "v_max_f64", "v_ldexp_f64", "v_mul_lo_u32", "v_mul_hi_u32", "v_mul_lo_i32", "v_mul_hi_i32", "v_div_scale_f32", "v_div_scale_f64", "v_div_fmas_f32", "v_div_fmas_f64", "v_msad_u8", "v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8", "v_trig_preop_f64", "v_mqsad_u32_u8", "v_mad_u64_u32", "v_mad_i64_i32", "v_xor3_b32", "v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64", "v_add_nc_u16", "v_sub_nc_u16", "v_mul_lo_u16", "v_lshrrev_b16", "v_ashrrev_i16", "v_max_u16", "v_max_i16", "v_min_u16", "v_min_i16", "v_add_nc_i16", "v_sub_nc_i16", "v_add_co_u32", "v_sub_co_u32", "v_pack_b32_f16", "v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16", "v_lshlrev_b16", "v_subrev_co_u32", "v_mad_u16", "v_interp_p1ll_f16", "v_interp_p1lv_f16", "v_perm_b32", "v_xad_u32", "v_lshl_add_u32", "v_add_lshl_u32", "v_fma_f16", "v_min3_f16", "v_min3_i16", "v_min3_u16", "v_max3_f16", "v_max3_i16", "v_max3_u16", "v_med3_f16", "v_med3_i16", "v_med3_u16", "v_interp_p2_f16", "v_mad_i16", "v_div_fixup_f16", "v_readlane_b32", "v_writelane_b32", "v_bfm_b32", "v_bcnt_u32_b32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32", "v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32", "v_add3_u32", "v_lshl_or_b32", "v_and_or_b32", "v_or3_b32", "v_mad_u32_u16", "v_mad_i32_i16", "v_sub_nc_i32", "v_permlane16_b32", "v_permlanex16_b32", "v_add_nc_i32", "v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32",
    "v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16",
    # VINTRP
    "v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32",
    # SMEM
    "s_load_dword", "s_load_dwordx2", "s_load_dwordx4", "s_load_dwordx8", "s_load_dwordx16",
    "s_buffer_load_dword", "s_buffer_load_dwordx2", "s_buffer_load_dwordx4", "s_buffer_load_dwordx8",
    # EXP
    "exp",
    # MIMG
    "image_sample", "image_sample_l", "image_sample_b", "image_sample_c", "image_load", "image_store",
}


# Scalar-memory store, cache-control, timer and probe forms the disassembler decodes: recognition only, no lowering.
KNOWN_RDNA2_BASE_OPCODES |= {
    "s_buffer_store_dword", "s_buffer_store_dwordx2", "s_buffer_store_dwordx4", "s_gl1_inv", "s_dcache_inv",
    "s_dcache_wb", "s_memtime", "s_memrealtime", "s_atc_probe", "s_atc_probe_buffer",
}

# Scalar program-control and compare-immediate forms the disassembler decodes: recognition only, no lowering.
KNOWN_RDNA2_BASE_OPCODES |= {
    "s_wakeup", "s_setkill", "s_sethalt", "s_sendmsg", "s_sendmsghalt", "s_icache_inv", "s_incperflevel",
    "s_decperflevel", "s_ttracedata", "s_ttracedata_imm", "s_endpgm_saved", "s_endpgm_ordered_ps_done",
    "s_wait_idle", "s_waitcnt_depctr", "s_denorm_mode", "s_call_b64",
}

# VOP1 forms the disassembler decodes that no translator lowers: recognition only.
KNOWN_RDNA2_BASE_OPCODES |= {"v_pipeflush", "v_clrexcp", "v_frexp_exp_i32_f32"}

KNOWN_RDNA2_BASE_OPCODES |= (
    SUPPORTED_ALU
    | {f"v_{name}" for name in PIXEL_INTEGER}
    | {f"s_{name}" for name in PIXEL_SCALAR}
    | {f"s_{name}_b64" for name in PIXEL_SALU_BIT64}
    | {f"v_pk_{name}_f16" for name in PIXEL_VOP3P_F16}
    | {f"v_pk_{name}" for name in PIXEL_VOP3P_INT16}
    | {f"v_{name}" for name in PIXEL_VOP3_INT16}
    | {f"v_{name}" for name in PIXEL_VOP3_INT16_TERNARY}
    | {f"v_{name}_f16" for name in PIXEL_VOP2_F16 + PIXEL_VOP2_F16_FUSED_SCALE + PIXEL_VOP1_F16_ROUND}
    | {"v_cvt_f16_i16", "v_cvt_f16_u16"}
    | {
        # Decoding only: no translator lowers these binary16 VOP1/VOP2 forms.
        "v_cvt_u16_f16", "v_cvt_i16_f16", "v_fract_f16",
        "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16",
        "v_rcp_f16", "v_sqrt_f16", "v_rsq_f16", "v_log_f16", "v_exp_f16", "v_sin_f16", "v_cos_f16",
        "v_frexp_mant_f16", "v_frexp_exp_i16_f16",
        "v_fmac_f16", "v_max_f16", "v_min_f16", "v_ldexp_f16", "v_pk_fmac_f16",
    }
    | {name for name, _, _ in MUBUF_OPS.values()}
    | {name for name, _, _ in MTBUF_OPS.values()}
    | {entry[0] for entry in FLAT_OPS.values()}
    | {entry[0] for entry in GLOBAL_OPS.values()}
    | {entry[0] for entry in SCRATCH_OPS.values()}
    | {entry[0] for entry in DS_OPS.values()}
    | (set(pixel_aot.HINT_LIMITS) if pixel_aot is not None else set())
    | {
        "s_brev_b64", "s_bfm_b64", "s_bfe_u32", "s_bfe_i32", "s_bfe_u64", "s_bfe_i64",
        "s_cmp_eq_u64", "s_cmp_lg_u64",
        "s_bitcmp0_b32", "s_bitcmp1_b32", "s_bitcmp0_b64", "s_bitcmp1_b64",
        "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
        "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
        "s_cmpk_eq_i32", "s_cmpk_lg_i32", "s_cmpk_gt_i32", "s_cmpk_ge_i32", "s_cmpk_lt_i32", "s_cmpk_le_i32",
        "s_cmpk_eq_u32", "s_cmpk_lg_u32", "s_cmpk_gt_u32", "s_cmpk_ge_u32", "s_cmpk_lt_u32", "s_cmpk_le_u32",
        "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_add_nc_i32", "v_sub_nc_i32",
        "v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32",
        "v_min3_i32", "v_max3_i32", "v_med3_i32", "v_min3_u32", "v_max3_u32", "v_med3_u32",
        "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32",
        "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3",
        "v_cvt_off_f32_i4", "v_ldexp_f32",
        "v_cvt_f64_i32", "v_cvt_f64_u32", "v_cvt_f64_f32",
        "v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64",
        "v_fract_f64", "v_trunc_f64", "v_ceil_f64", "v_rndne_f64", "v_floor_f64",
        "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64",
        "v_frexp_exp_i32_f64", "v_frexp_mant_f64",
        "v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16",
        "v_dot2_f32_f16", "v_dot2_i32_i16", "v_dot2_u32_u16",
        "v_dot4_i32_i8", "v_dot4_u32_u8", "v_dot8_i32_i4", "v_dot8_u32_u4",
    }
)


def is_opcode_recognized(opcode: str) -> bool:
    """Check if opcode belongs to recognized AMD RDNA 2 ISA instructions."""
    clean = opcode.strip().lower()
    base = re.sub(r"_(?:e32|e64|sdwa|dpp.*)$", "", clean)
    if base in KNOWN_RDNA2_BASE_OPCODES or clean in KNOWN_RDNA2_BASE_OPCODES:
        return True
    if base.startswith(("v_cmp_", "v_cmpx_")) or clean.startswith(("v_cmp_", "v_cmpx_")):
        return True
    if base.startswith("image_") or clean.startswith("image_"):
        return True
    if base.startswith("s_load_") or base.startswith("s_buffer_load_"):
        return True
    return False


def classify_isa_category(opcode: str) -> str:
    """Classify instruction into RDNA 2 ISA category according to the AMD manual."""
    op = opcode.lower()
    if op.startswith("v_interp"):
        return "VINTRP (Parameter Interpolation)"
    if op.startswith("v_"):
        if op.startswith(("v_cmp_", "v_cmpx_")):
            return "VOPC (Vector Compare)"
        if op.startswith("v_pk_") or op.startswith("v_dot") or op.startswith("v_fma_mix"):
            return "VOP3P (Vector Packed / Mixed-Precision)"
        if "_f16" in op or "_f32" in op or "_f64" in op or "_u32" in op or "_i32" in op:
            return "VOP / VALU (Vector ALU - VOP1/VOP2/VOP3)"
        return "VALU (Vector ALU)"
    if op.startswith("s_"):
        if op.startswith("s_cbranch") or op.startswith("s_branch"):
            return "SOPP (Scalar Control Flow / Branch)"
        if op.startswith("s_load_") or op.startswith("s_buffer_load_"):
            return "SMEM (Scalar Memory Load)"
        if op.startswith("s_waitcnt"):
            return "SOPP (Scalar Synchronization)"
        if op.startswith("s_mov") or op.startswith("s_and") or op.startswith("s_or") or op.startswith("s_add"):
            return "SOP1 / SOP2 / SOPK (Scalar ALU)"
        return "SALU (Scalar ALU)"
    if op.startswith("image_"):
        return "MIMG (Image / Texture Resource)"
    if op.startswith(("buffer_", "tbuffer_", "flat_", "global_", "scratch_")):
        return "VMEM (Vector Memory / Buffer)"
    if op.startswith("exp"):
        return "EXP (Export)"
    if op.startswith("ds_"):
        return "LDS (Local Data Share)"
    return "Other / Unknown"


_STAGE_TAGS = (
    ("pixel", "pixel", "ps"),
    ("compute", "compute", "cs"),
    ("vertex", "vertex", "vs"),
)
_STAGE_DELIMITERS = re.compile(r"[_.\-]")


def detect_shader_stage(name_or_path: str) -> str:
    """Classify shader stage from path or filename.

    A stage tag (`ps`, `cs`, `vs`) matches only as a whole token of a path
    component, where tokens are delimited by `_`, `.` or `-` (`ps_draw.bin`,
    `blur_cs.s`, `a/vs/x.s`), so a tag inside another word (`loops_cs`,
    `bumps_vs`, `ps5`) does not match. The long names (`pixel`, `compute`,
    `vertex`) match only as a whole directory component. Matching is case
    insensitive. If one path carries tags of several stages, the first stage
    in the order pixel, compute, vertex wins.
    """
    parts = [p for p in name_or_path.lower().replace("\\", "/").split("/") if p]
    directories = set(parts[:-1])
    tokens = set()
    for part in parts:
        tokens.update(t for t in _STAGE_DELIMITERS.split(part) if t)
    for stage, long_name, short_name in _STAGE_TAGS:
        if long_name in directories or short_name in tokens:
            return stage
    return "other"


def disassemble_binary_shader(data: bytes) -> list[str]:
    """Disassemble raw GFX1013 binary shader using disasm_rdna2."""
    if disasm_rdna2 is None:
        raise RuntimeError("disasm_rdna2 module unavailable")
    if not data:
        raise ValueError("empty binary shader data")
    records = disasm_rdna2(data)
    if not records:
        raise ValueError("no instruction records decoded from binary")
    return [rec["assembly"] for rec in records]


def make_failed_shader_record(shader_name: str, error_message: str) -> dict:
    """Create a failed-input analysis record to preserve denominator count."""
    stage = detect_shader_stage(shader_name)
    return {
        "shader_name": shader_name,
        "stage": stage,
        "total_instructions": 0,
        "supported_instructions": 0,
        "unsupported_count": 0,
        "support_percentage": 0.0,
        "opcodes_recognized": False,
        "translator_accepted": False,
        "compatibility_category": "unsupported_opcodes",
        "is_fully_supported": False,
        "lower_status": "decode_failed",
        "lower_error": error_message,
        "lower_error_kind": None,
        "lower_error_type": None,
        "lower_unbound": None,
        "unsupported_opcodes": {"<decode_error>": 1},
        "unsupported_details": [{
            "line_number": 0,
            "raw": error_message,
            "opcode": "<decode_error>",
            "category": "Other / Unknown",
        }],
        "opcode_distribution": {},
    }


def extract_opcode(instruction_line: str) -> str:
    """Extract primary opcode token from assembly instruction."""
    line = instruction_line.strip()
    # Strip comments
    line = re.split(r"//|[;#]", line)[0].strip()
    if not line:
        return ""
    tokens = line.split()
    return tokens[0] if tokens else ""


def supports_int64_compare(clean: str) -> bool:
    """Mirror pixel_aot.int64_operand and the 64-bit comparison destination rules."""
    if INT64_COMPARE_PATTERN is None or not (m := INT64_COMPARE_PATTERN.fullmatch(clean)):
        return False
    enc, dst, *sources = m.groups()
    if enc is None:  # LLVM 15 picks e32 only when the operands fit it
        enc = "_e32" if dst in ("vcc", "vcc_lo") and re.fullmatch(r"v\[\d+:\d+\]", sources[1].strip()) else "_e64"
    if enc == "_e32" and dst not in ("vcc", "vcc_lo"):
        return False
    if (p := re.fullmatch(r"s\[(\d+):(\d+)\]", dst)) and not _sgpr_pair_ok(p):
        return False
    if re.fullmatch(r"s\d+", dst) and int(dst[1:]) > 105:
        return False
    for position, token in enumerate(sources):
        if (p := re.fullmatch(r"v\[(0|[1-9]\d*):(0|[1-9]\d*)\]", token)):
            if int(p[2]) != int(p[1]) + 1 or int(p[1]) > 254:
                return False
        elif enc == "_e32" and position == 1:
            return False
        elif p := re.fullmatch(r"s\[(\d+):(\d+)\]", token):
            if not _sgpr_pair_ok(p):
                return False
        elif not (re.fullmatch(r"0|-?[1-9]\d*", token) and -16 <= int(token) <= 64):
            return False
    return True


def _sgpr_pair_ok(pair: "re.Match") -> bool:
    first, last = int(pair[1]), int(pair[2])
    return last == first + 1 and first % 2 == 0 and last <= 105


def is_instruction_supported(instruction_line: str) -> bool:
    """Check if instruction is supported by pixel_aot regex patterns."""
    clean = instruction_line.strip()
    clean = re.split(r"//|[;#]", clean)[0].strip()
    if not clean:
        return True
    if clean.startswith("v_cmp_") and re.search(r"_[iu]64(?:_e32|_e64)?\s", clean):
        return supports_int64_compare(clean)
    if re.match(r"v_cmp_\w+_f32_sdwa\s", clean) and re.search(r"\bneg\(-?\d+\)", clean):
        return False
    if not any(pattern.match(clean) for pattern in COMPILED_SUPPORTED_PATTERNS):
        return False
    if wait := re.fullmatch(r"s_waitcnt\s+(.*)", clean):
        counters = [token.partition("(")[0] for token in wait[1].split()]
        return len(counters) == len(set(counters))
    if re.match(r"s_and_saveexec_b(?:32|64)\s|s_(?:or|andn2)_b(?:32|64)\s+exec", clean):
        # Masked-region exec forms: an aligned SGPR pair or an SGPR up to s105, as pixel_aot's sgpr_pair/sgpr_name.
        return (all(_sgpr_pair_ok(pair) for pair in re.finditer(r"s\[(\d+):(\d+)\]", clean))
                and all(int(n) <= 105 for n in re.findall(r"(?<![\w\[:])s(\d+)(?!\d)", clean)))
    if execz := re.fullmatch(r"s_cbranch_execz\s+(-?(?:0x[0-9a-f]+|0|[1-9]\d*))", clean):
        # The 16-bit signed displacement pixel_aot accepts.
        return -0x8000 <= int(execz[1], 0) <= 0x7fff
    return True


def analyze_shader_instructions(shader_name: str, instructions: list[str], *,
                                wave_size: int | None = None) -> dict:
    """
    Analyze a list of RDNA 2 assembly instructions.
    Distinguishes:
      1. Opcode recognition: whether every instruction opcode is known to the RDNA 2 ISA scanner.
      2. Translator acceptance: whether pixel_aot.lower() actually accepts the shader sequence.
    wave_size names the guest execution width (32 or 64). Low-only execution mask operations
    require a validated guest wave32 profile (wave_size=32).
    """
    total = 0
    unrecognized_opcodes = Counter()
    unsupported_translator_items = []
    opcode_distribution = Counter()

    for idx, raw_line in enumerate(instructions, start=1):
        line = raw_line.strip()
        line = re.split(r"//|[;#]", line)[0].strip()
        if not line:
            continue

        total += 1
        opcode = extract_opcode(line)
        opcode_distribution[opcode] += 1

        if not is_opcode_recognized(opcode):
            unrecognized_opcodes[opcode] += 1

        if not is_instruction_supported(line):
            category = classify_isa_category(opcode)
            unsupported_translator_items.append({
                "line_number": idx,
                "raw": line,
                "opcode": opcode,
                "category": category,
            })

    supported_count = total - len(unsupported_translator_items)
    support_ratio = (supported_count / total * 100.0) if total > 0 else 0.0

    # Test pixel_aot.lower() to determine translator acceptance under the scan's fixed bindings
    lower_status = "untested"
    lower_error = None
    lower_error_kind = None
    lower_error_type = None
    lower_unbound = None
    if pixel_aot is not None:
        try:
            pixel_aot.lower(instructions=instructions, wave_size=wave_size, **SCAN_BINDINGS)
            lower_status = "passed"
        except Exception as e:
            lower_status = "rejected"
            lower_error = str(e)
            lower_error_type = type(e).__name__
            if isinstance(e, pixel_aot.UnboundOperand):
                # A register with no tracked value under the scanned inputs, not an instruction the translator refuses.
                lower_error_kind = "unbound_operand"
                lower_unbound = {"bank": e.bank, "register": e.register, "instruction_index": e.index,
                                 "byte_offset": e.offset, "instruction": e.instruction}
            elif isinstance(e, ValueError):
                lower_error_kind = "translator_rejection"
            else:
                # Any other exception (a bare KeyError included) is a translator fault, not a missing binding.
                lower_error_kind = "unexpected_error"

    opcodes_recognized = (len(unrecognized_opcodes) == 0)
    translator_accepted = (lower_status == "passed")

    if opcodes_recognized and translator_accepted:
        compatibility_category = "accepted"
    elif opcodes_recognized and not translator_accepted:
        compatibility_category = "opcode_recognized_translator_rejected"
    else:
        compatibility_category = "unsupported_opcodes"

    stage = detect_shader_stage(shader_name)

    return {
        "shader_name": shader_name,
        "stage": stage,
        "total_instructions": total,
        "supported_instructions": supported_count,
        "unsupported_count": len(unsupported_translator_items),
        "support_percentage": round(support_ratio, 2),
        "opcodes_recognized": opcodes_recognized,
        "translator_accepted": translator_accepted,
        "compatibility_category": compatibility_category,
        "is_fully_supported": opcodes_recognized and translator_accepted,
        "lower_status": lower_status,
        "lower_error": lower_error,
        "lower_error_kind": lower_error_kind,
        "lower_error_type": lower_error_type,
        "lower_unbound": lower_unbound,
        "unsupported_opcodes": dict(unrecognized_opcodes),
        "unsupported_details": unsupported_translator_items,
        "opcode_distribution": dict(opcode_distribution),
    }


def parse_assembly_file(path: Path) -> list[str]:
    """Read assembly instructions from a text file (.s, .asm, .txt)."""
    lines = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", ";", "#", "@")):
                continue
            # Symbol metadata emits no instruction. Keep data and alignment directives visible.
            if stripped == ".text" or stripped.split(None, 1)[0] in (".globl", ".global", ".type", ".size"):
                continue
            code = re.split(r"//|[;#]", stripped)[0].strip()  # same comment rule as extract_opcode
            if code.endswith(":") and " " not in code:
                continue
            lines.append(stripped)
    return lines


def scan_directory_or_files(target_paths: list[Path], llvm_lib: str = None) -> list[dict]:
    """Scan files, directories, or archives and collect analysis for each shader."""
    results = []

    for path in target_paths:
        if not path.exists():
            print(f"[!] Warning: Path does not exist: {path}", file=sys.stderr)
            continue

        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in (".s", ".asm", ".txt", ".isa"):
                instructions = parse_assembly_file(path)
                if instructions:
                    results.append(analyze_shader_instructions(path.name, instructions))
                else:
                    results.append(make_failed_shader_record(path.name, "No assembly instructions found in file"))
            elif suffix == ".bin":
                try:
                    data = path.read_bytes()
                    instructions = disassemble_binary_shader(data)
                    results.append(analyze_shader_instructions(path.name, instructions))
                except Exception as e:
                    results.append(make_failed_shader_record(path.name, str(e)))
            elif suffix == ".rpf" and pixel_aot is not None:
                print(f"[*] Scanning archive: {path.name}...")
                archive_results = scan_rpf_archive(path, llvm_lib)
                results.extend(archive_results)
            else:
                try:
                    instructions = parse_assembly_file(path)
                    if instructions and any(" " in line for line in instructions[:5]):
                        results.append(analyze_shader_instructions(path.name, instructions))
                except Exception:
                    pass

        elif path.is_dir():
            all_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    all_files.append(Path(root) / file)

            for idx, file_path in enumerate(all_files, start=1):
                suffix = file_path.suffix.lower()
                rel_name = str(file_path.relative_to(path))
                if suffix in (".s", ".asm", ".txt", ".isa"):
                    instructions = parse_assembly_file(file_path)
                    if instructions:
                        results.append(analyze_shader_instructions(rel_name, instructions))
                    else:
                        results.append(make_failed_shader_record(rel_name, "No assembly instructions found in file"))
                elif suffix == ".bin":
                    try:
                        data = file_path.read_bytes()
                        instructions = disassemble_binary_shader(data)
                        results.append(analyze_shader_instructions(rel_name, instructions))
                    except Exception as e:
                        results.append(make_failed_shader_record(rel_name, str(e)))

                if idx % 500 == 0 or idx == len(all_files):
                    print(f"[*] Scanned {idx}/{len(all_files)} files ({len(results)} valid shaders)...", flush=True)

    return results


def scan_rpf_archive(archive_path: Path, llvm_lib: str = None) -> list[dict]:
    """Scan named records in an RPF archive if LLVM disassembler is provided."""
    archive_results = []
    if llvm_lib is None or not Path(llvm_lib).exists():
        print("    [!] LLVM library path (--llvm) is required to disassemble binary shaders from RPF archives.", file=sys.stderr)
        return archive_results

    try:
        with archive_path.open("rb") as f:
            data = f.read(16 << 20)
            pos = 0
            while True:
                idx = data.find(b"FSSM", pos)
                if idx < 0:
                    break
                pos = idx + 4
    except Exception as e:
        print(f"    [!] Error scanning archive {archive_path.name}: {e}", file=sys.stderr)

    return archive_results


def make_progress_bar(percentage: float, width: int = 24, unicode_blocks: bool = None) -> str:
    """Generate visual progress bar: [████████░░░░░░░░] or [========........] 50.0%."""
    if unicode_blocks is None:
        try:
            encoding = sys.stdout.encoding or "utf-8"
            "█░".encode(encoding)
            unicode_blocks = True
        except (UnicodeEncodeError, LookupError):
            unicode_blocks = False

    fill_char = "█" if unicode_blocks else "="
    empty_char = "░" if unicode_blocks else "."

    pct = max(0.0, min(100.0, percentage))
    filled = int(round(width * (pct / 100.0)))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{fill_char * filled}{empty_char * empty}] {pct:5.1f}%"


def get_implementation_target(opcode: str, category: str) -> str:
    """Identify target file and subsystem in Xbox2Windows for an opcode."""
    op = opcode.lower()
    if "VALU" in category or "Vector ALU" in category:
        if any(x in op for x in ["_f32", "_f16", "_i32", "_u32"]):
            return "`tools/pixel_aot.py` (ALU dictionary / table)"
        return "`tools/pixel_aot.py` (Vector ALU lowering)"
    if "SOPP" in category:
        if "branch" in op:
            return "`tools/pixel_aot.py` (CFG / control flow lowering)"
        if "waitcnt" in op or "barrier" in op:
            return "`tools/pixel_aot.py` (Synchronization lowering)"
        return "`tools/pixel_aot.py` (Scalar control)"
    if "MIMG" in category:
        return "`tools/pixel_aot.py` (Texture/Resource sampling descriptor)"
    if "SMEM" in category:
        return "`tools/pixel_aot.py` (Constant buffer / Root signature)"
    if "VMEM" in category:
        return "`tools/pixel_aot.py` (UAV / ByteAddressBuffer)"
    if "VINTRP" in category:
        return "`tools/pixel_aot.py` (SV_Position / Input attribute)"
    if "LDS" in category:
        return "`tools/pixel_aot.py` (Threadgroup shared memory)"
    return "`tools/pixel_aot.py` (Instruction lowering)"


def generate_progress_dashboard(results: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Generate AnyPS5-style visual progress dashboard (PROGRESS.md and progress.json)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    total_shaders = len(results)
    total_instructions = sum(r["total_instructions"] for r in results)
    total_supported = sum(r["supported_instructions"] for r in results)
    overall_coverage = (total_supported / total_instructions * 100.0) if total_instructions > 0 else 0.0

    accepted_shaders = sum(1 for r in results if r["translator_accepted"])
    partial_shaders = sum(1 for r in results if not r["translator_accepted"] and r["supported_instructions"] > 0)
    unsupported_shaders = sum(1 for r in results if not r["translator_accepted"] and r["supported_instructions"] == 0)

    # Breakdown by stage
    stages = ["pixel", "compute", "vertex", "other"]
    stage_names = {
        "pixel": "Pixel (`PS`)",
        "compute": "Compute (`CS`)",
        "vertex": "Vertex (`VS`)",
        "other": "Other / Unknown",
    }
    stage_data = {}
    for st in stages:
        st_res = [r for r in results if r["stage"] == st]
        count = len(st_res)
        st_total = sum(r["total_instructions"] for r in st_res)
        st_supp = sum(r["supported_instructions"] for r in st_res)
        st_cov = (st_supp / st_total * 100.0) if st_total > 0 else 0.0
        st_acc = sum(1 for r in st_res if r["translator_accepted"])
        stage_data[st] = {
            "name": stage_names[st],
            "shaders": count,
            "instructions": st_total,
            "supported": st_supp,
            "coverage": round(st_cov, 1),
            "accepted": st_acc,
            "bar": make_progress_bar(st_cov, 16, unicode_blocks=True),
        }

    # Breakdown by ISA category & subsystem
    category_counts = Counter()
    category_opcodes = defaultdict(Counter)
    for r in results:
        for op, count in r["opcode_distribution"].items():
            cat = classify_isa_category(op)
            category_counts[cat] += count
            category_opcodes[cat][op] += count

    # High-impact missing opcode rankings (blockers)
    missing_opcode_occurrences = Counter()
    missing_opcode_shaders = defaultdict(set)
    for idx, r in enumerate(results):
        shader_id = r["shader_name"]
        for item in r.get("unsupported_details", []):
            op = item.get("opcode", "")
            if op:
                missing_opcode_occurrences[op] += 1
                missing_opcode_shaders[op].add(shader_id)
        for op, count in r.get("unsupported_opcodes", {}).items():
            if op not in missing_opcode_shaders or shader_id not in missing_opcode_shaders[op]:
                missing_opcode_occurrences[op] += count
                missing_opcode_shaders[op].add(shader_id)

    # Rank by number of distinct shaders blocked
    ranked_missing = []
    for rank, (op, count) in enumerate(
        sorted(missing_opcode_occurrences.items(), key=lambda x: (len(missing_opcode_shaders[x[0]]), x[1]), reverse=True),
        start=1,
    ):
        cat = classify_isa_category(op)
        blocked = len(missing_opcode_shaders[op])
        impact_pct = round((blocked / total_shaders * 100.0), 1) if total_shaders > 0 else 0.0
        ranked_missing.append({
            "rank": rank,
            "opcode": op,
            "category": cat,
            "occurrences": count,
            "shaders_blocked": blocked,
            "impact_percentage": impact_pct,
            "target": get_implementation_target(op, cat),
        })

    # progress.json
    progress_json = {
        "timestamp": timestamp,
        "overall": {
            "total_shaders": total_shaders,
            "fully_accepted": accepted_shaders,
            "partially_supported": partial_shaders,
            "completely_unsupported": unsupported_shaders,
            "total_instructions": total_instructions,
            "supported_instructions": total_supported,
            "instruction_coverage_percentage": round(overall_coverage, 2),
            "coverage_percentage": round(overall_coverage, 2),
            "progress_bar": make_progress_bar(overall_coverage, 24, unicode_blocks=True),
        },
        "stages": stage_data,
        "subsystems": {
            cat: {
                "total_instructions": count,
                "distinct_opcodes": len(category_opcodes[cat]),
                "top_opcodes": [op for op, _ in category_opcodes[cat].most_common(5)],
            }
            for cat, count in category_counts.most_common()
        },
        "missing_opcodes_ranked": ranked_missing,
    }

    json_file = output_dir / "progress.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(progress_json, f, indent=2)

    # PROGRESS.md
    md_file = output_dir / "PROGRESS.md"
    with md_file.open("w", encoding="utf-8") as f:
        f.write("# Xbox2Windows: RDNA 2 Shader Compatibility Status\n\n")
        f.write(f"Updated: `{timestamp}` | Reference: AMD RDNA 2 ISA (GFX10) | Scope: Xbox Series X|S Shaders\n\n")

        f.write("## Overall Status\n\n")
        f.write("```text\n")
        f.write(f"Instruction Coverage: {make_progress_bar(overall_coverage, 28, unicode_blocks=True)}\n")
        f.write("```\n\n")

        f.write("| Metric | Count | Percentage |\n")
        f.write("|:---|---:|---:|\n")
        f.write(f"| **Total Shaders Analyzed** | {total_shaders:,} | 100.0% |\n")
        f.write(f"| **Fully Supported / Accepted** | {accepted_shaders:,} | {(accepted_shaders/total_shaders*100.0 if total_shaders else 0):.1f}% |\n")
        f.write(f"| **Partially Supported** | {partial_shaders:,} | {(partial_shaders/total_shaders*100.0 if total_shaders else 0):.1f}% |\n")
        f.write(f"| **Unsupported / Blocked** | {unsupported_shaders:,} | {(unsupported_shaders/total_shaders*100.0 if total_shaders else 0):.1f}% |\n")
        f.write(f"| **Total Decoded Instructions** | {total_instructions:,} | 100.0% |\n")
        f.write(f"| **Supported Instruction Instances** | {total_supported:,} | {overall_coverage:.1f}% |\n\n")

        f.write("---\n\n")
        f.write("## Pipeline Stage Breakdown\n\n")
        f.write("| Stage | Shaders | Instructions | Supported Insns | Coverage | Progress Bar |\n")
        f.write("|:---|---:|---:|---:|---:|:---|\n")
        for st in stages:
            d = stage_data[st]
            if d["shaders"] > 0:
                f.write(f"| {d['name']} | {d['shaders']:,} | {d['instructions']:,} | {d['supported']:,} | {d['coverage']:.1f}% | `{d['bar']}` |\n")

        f.write("\n---\n\n")
        f.write("## AMD RDNA 2 ISA Subsystem Breakdown\n\n")
        f.write("| Subsystem / ISA Category | Instructions | Distinct Opcodes | % of Total | Top Opcodes |\n")
        f.write("|:---|---:|---:|---:|:---|---|\n")
        for cat, count in category_counts.most_common():
            pct = (count / total_instructions * 100.0) if total_instructions > 0 else 0.0
            top_ops = ", ".join(f"`{op}`" for op, _ in category_opcodes[cat].most_common(4))
            f.write(f"| **{cat}** | {count:,} | {len(category_opcodes[cat])} | {pct:.1f}% | {top_ops} |\n")

        f.write("\n---\n\n")
        f.write("## High-Impact Missing Opcode Rankings (Roadmap)\n\n")
        f.write("Ranked by number of distinct shaders blocked. Implementing these unlocks the highest number of native shaders:\n\n")
        f.write("| Rank | Opcode | ISA Category | Occurrences | Shaders Blocked | Blocker Impact % | Target Implementation |\n")
        f.write("|---:|:---|:---|---:|---:|---:|:---|---|\n")
        for item in ranked_missing[:35]:
            f.write(
                f"| {item['rank']} | `{item['opcode']}` | {item['category']} | {item['occurrences']:,} | "
                f"{item['shaders_blocked']:,} | {item['impact_percentage']:.1f}% | {item['target']} |\n"
            )

        f.write("\n---\n\n")
        f.write("### Methodology Notes\n\n")
        f.write("- **Shaders Blocked**: Count of distinct shaders that cannot be fully lowered because they contain this opcode.\n")
        f.write("- **Blocker Impact %**: Percentage of total scanned shaders unblocked or partially enabled by implementing this opcode.\n")
        f.write(f"- **Fully Supported / Accepted**: Conditional on the fixed entry bindings the scan supplies to `pixel_aot.lower()` ({SCAN_BINDINGS_TEXT}).\n")
        f.write("- **Pure Python Pipeline**: Instruction disassembly is executed natively via `tools/disasm_rdna2.py` without requiring external toolchains.\n")

    return json_file, md_file


def generate_reports(results: list[dict], output_dir: Path):
    """Generate both Markdown and JSON reports as well as the progress dashboard."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "xbox2windows-shader-scan"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[!] Warning: Cannot write to '{output_dir}'. Falling back to '{fallback}'.", file=sys.stderr)
        output_dir = fallback

    json_path = output_dir / "shader_scan_report.json"
    md_path = output_dir / "shader_scan_report.md"

    total_shaders = len(results)
    translator_accepted = sum(1 for r in results if r["translator_accepted"])
    opcodes_recognized_only = sum(1 for r in results if r["compatibility_category"] == "opcode_recognized_translator_rejected")
    unsupported_opcodes_count = sum(1 for r in results if r["compatibility_category"] == "unsupported_opcodes")
    partially_supported = sum(1 for r in results if not r["is_fully_supported"] and r["supported_instructions"] > 0)
    unsupported_completely = sum(1 for r in results if not r["translator_accepted"] and r["supported_instructions"] == 0)
    rejection_causes = lower_rejection_causes(results)

    # Aggregate all unsupported opcodes across all shaders
    global_unsupported_opcodes = Counter()
    global_categories = Counter()
    for r in results:
        for op, count in r["unsupported_opcodes"].items():
            global_unsupported_opcodes[op] += count
            global_categories[classify_isa_category(op)] += count

    # Save JSON
    summary_data = {
        "summary": {
            "total_shaders_scanned": total_shaders,
            "translator_accepted": translator_accepted,
            "opcodes_recognized_translator_rejected": opcodes_recognized_only,
            "unsupported_opcodes_count": unsupported_opcodes_count,
            "fully_supported": translator_accepted,
            "partially_supported": partially_supported,
            "completely_unsupported": unsupported_completely,
            "unique_unsupported_opcodes_count": len(global_unsupported_opcodes),
            # Every record lower() rejected, by cause; the unbound counts are reads of registers with no tracked value
            # under the scanned inputs, not instructions the translator refuses.
            "translator_rejections_by_cause": rejection_causes,
            # translator_accepted and the rejection counts hold only under these fixed entry bindings.
            "translator_bindings": dict(SCAN_BINDINGS, sampler=list(SCAN_BINDINGS["sampler"])),
        },
        "top_unsupported_opcodes": global_unsupported_opcodes.most_common(),
        "unsupported_by_category": dict(global_categories),
        "shaders": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save Markdown
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Xbox2Windows: RDNA 2 Shader Compatibility Scan Report\n\n")
        f.write("Generated analysis comparing scanned shader instructions against `tools/pixel_aot.py` lowering capabilities.\n")
        f.write("Reference: **AMD RDNA 2 Shader Instruction Set Architecture (ISA)** manual.\n\n")

        f.write("## 1. Summary Overview\n\n")
        f.write(f"- **Total Shaders Scanned:** {total_shaders}\n")
        f.write(f"- **Full Shaders Accepted by Translator (`pixel_aot.py`):** {translator_accepted} ({round(translator_accepted/total_shaders*100, 1) if total_shaders else 0}%)\n")
        f.write(f"- **Opcodes Recognized by Scanner, but Rejected by Translator:** {opcodes_recognized_only}\n")
        f.write("  *(Valid RDNA 2 opcodes whose program structure, preamble, or bindings differ from `pixel_aot` straight-line single-texture template)*\n")
        f.write("- **Translator Rejections by Cause** (every record `lower()` rejected):\n")
        f.write(f"  - Unbound scalar operand under the scanned inputs: {rejection_causes['unbound_scalar_operand']}\n")
        f.write(f"  - Unbound vector operand under the scanned inputs: {rejection_causes['unbound_vector_operand']}\n")
        f.write(f"  - Other translator rejection: {rejection_causes['translator_rejection']}\n")
        f.write(f"  - Unexpected translator error: {rejection_causes['unexpected_error']}\n")
        f.write(f"  *(Acceptance and rejection totals are conditional on the fixed entry bindings this scan supplies to every record: {SCAN_BINDINGS_TEXT}.)*\n")
        f.write(f"- **Contains Unrecognized Opcodes:** {unsupported_opcodes_count}\n")
        f.write(f"- **Unique Missing Opcodes:** {len(global_unsupported_opcodes)}\n\n")

        f.write("## 2. Priority Ranking: Most Needed Instructions to Implement\n\n")
        f.write("Implementing these opcodes in `tools/pixel_aot.py` will unlock the greatest number of shaders:\n\n")
        f.write("| Opcode | Occurrences | ISA Category | Implementation Target |\n")
        f.write("|---|---:|---|---|\n")

        for opcode, count in global_unsupported_opcodes.most_common(25):
            cat = classify_isa_category(opcode)
            target = "ALU dictionary (tools/pixel_aot.py)" if "VALU" in cat else "Lowering pattern (tools/pixel_aot.py)"
            f.write(f"| `{opcode}` | {count} | {cat} | {target} |\n")

        f.write("\n## 3. Unsupported Instructions by ISA Category\n\n")
        f.write("| Category | Total Unsupported Instructions |\n")
        f.write("|---|---:|\n")
        for cat, count in global_categories.most_common():
            f.write(f"| {cat} | {count} |\n")

        f.write("\n## 4. Per-Shader Detailed Breakdown\n\n")
        for r in results:
            status_icon = "PASS" if r["is_fully_supported"] else ("PARTIAL" if r["supported_instructions"] > 0 else "FAIL")
            f.write(f"### `[{status_icon}]` {r['shader_name']}\n\n")
            f.write(f"- **Instructions:** {r['total_instructions']} total | {r['supported_instructions']} supported ({r['support_percentage']}%) | {r['unsupported_count']} unsupported\n")
            f.write(f"- **Lowering Simulation Status:** `{r['lower_status']}`\n")
            if r["lower_error"]:
                f.write(f"- **Rejection Cause:** `{r['lower_error']}`\n")

            if r["unsupported_opcodes"]:
                f.write("\n**Missing opcodes:**\n")
                for op, count in r["unsupported_opcodes"].items():
                    f.write(f"- `{op}` ({count}x) — *{classify_isa_category(op)}*\n")

            if r["unsupported_details"]:
                f.write("\n<details><summary>Click to view unsupported instruction lines</summary>\n\n```asm\n")
                for item in r["unsupported_details"][:50]:
                    f.write(f"L{item['line_number']:04d}: {item['raw']}\n")
                if len(r["unsupported_details"]) > 50:
                    f.write(f"... and {len(r['unsupported_details']) - 50} more lines\n")
                f.write("```\n\n</details>\n\n")

    # Generate AnyPS5-style progress dashboard
    generate_progress_dashboard(results, output_dir)

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path, help="Shader files (.s, .asm, .txt, .bin), directories, or .rpf archives")
    parser.add_argument("--llvm", default=None, help="Path to libLLVM with AMDGPU support for disassembling binary shaders")
    parser.add_argument("--output-dir", type=Path, default=Path("out/shader-scan"), help="Directory to save report and logs (default: out/shader-scan)")

    args = parser.parse_args()

    print(f"[*] Scanning {len(args.targets)} target path(s)...")
    results = scan_directory_or_files(args.targets, args.llvm)

    if not results:
        print("[-] No shader files or instructions were found to analyze.")
        return 0

    json_path, md_path = generate_reports(results, args.output_dir)

    total = len(results)
    accepted = sum(1 for r in results if r["translator_accepted"])
    recognized_only = sum(1 for r in results if r["compatibility_category"] == "opcode_recognized_translator_rejected")
    unsupported_opcodes = sum(1 for r in results if r["compatibility_category"] == "unsupported_opcodes")
    total_insns = sum(r["total_instructions"] for r in results)
    total_supp_insns = sum(r["supported_instructions"] for r in results)
    coverage = (total_supp_insns / total_insns * 100.0) if total_insns > 0 else 0.0

    print("\n" + "=" * 65)
    print(" SCAN RESULTS SUMMARY (AnyPS5 Style Dashboard) ")
    print("=" * 65)
    print(f"Instruction Coverage: {make_progress_bar(coverage, 24)}")
    print(f"Total Shaders Scanned:                     {total:,}")
    print(f"Full Shaders Accepted by Translator:       {accepted:,} ({(accepted/total)*100:.1f}%)")
    print(f"Opcodes Recognized, Translator-Rejected:   {recognized_only:,}")
    causes = lower_rejection_causes(results)
    print(f"Rejections, Unbound Scalar/Vector Operand: {causes['unbound_scalar_operand']:,} / {causes['unbound_vector_operand']:,}")
    print(f"Rejections, Other Translator / Unexpected: {causes['translator_rejection']:,} / {causes['unexpected_error']:,}")
    print(f"  (translator totals hold only under the scan's fixed bindings: {SCAN_BINDINGS_TEXT})")
    print(f"Contains Unrecognized Opcodes:             {unsupported_opcodes:,}")
    print(f"Total Decoded Instructions:                {total_insns:,}")
    print(f"Supported Instructions:                    {total_supp_insns:,}")

    # Stage breakdowns
    print("\n--- PIPELINE STAGE BREAKDOWN ---")
    for st, name in [("pixel", "Pixel   [PS]"), ("compute", "Compute [CS]"), ("vertex", "Vertex  [VS]")]:
        st_res = [r for r in results if r["stage"] == st]
        if st_res:
            st_total = sum(r["total_instructions"] for r in st_res)
            st_supp = sum(r["supported_instructions"] for r in st_res)
            st_cov = (st_supp / st_total * 100.0) if st_total > 0 else 0.0
            print(f"  {name:<14} {len(st_res):>5,} shaders | {make_progress_bar(st_cov, 14)} ({st_cov:5.1f}%)")

    # High-impact blockers
    missing_shaders = defaultdict(set)
    missing_counts = Counter()
    for r in results:
        for item in r.get("unsupported_details", []):
            op = item.get("opcode", "")
            if op:
                missing_counts[op] += 1
                missing_shaders[op].add(r["shader_name"])
        for op, count in r.get("unsupported_opcodes", {}).items():
            if op not in missing_shaders or r["shader_name"] not in missing_shaders[op]:
                missing_counts[op] += count
                missing_shaders[op].add(r["shader_name"])

    if missing_counts:
        print("\n--- TOP HIGH-IMPACT MISSING OPCODES (ROADMAP) ---")
        top_ranked = sorted(missing_counts.items(), key=lambda x: (len(missing_shaders[x[0]]), x[1]), reverse=True)[:10]
        for rank, (op, count) in enumerate(top_ranked, start=1):
            cat = classify_isa_category(op)
            blocked = len(missing_shaders[op])
            pct = (blocked / total * 100.0) if total else 0.0
            print(f"  #{rank:<2} {op:<20} {blocked:>4,} shaders blocked ({pct:4.1f}%) | {count:>5,} insns  [{cat}]")

    print("\n[+] Full reports written to:")
    print(f"    - Dashboard:  {args.output_dir / 'PROGRESS.md'}")
    print(f"    - Raw State:  {args.output_dir / 'progress.json'}")
    print(f"    - Markdown:   {md_path}")
    print(f"    - JSON:       {json_path}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
