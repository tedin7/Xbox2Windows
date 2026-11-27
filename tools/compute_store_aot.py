"""Translate bounded raw-UAV gfx1013 compute programs under explicit bindings."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from pixel_aot import INTEGER, check_gather_sampler, container_code, disassemble
from resource_inventory import parts


CONSTANT_WORDS = 16
AXES = "xyz"
# Integer compares a lane guard may use. Inequality is the same for signed and unsigned words.
GUARD_COMPARES = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "ne": "!="}
INT_COMPARE_PREDICATES = {
    "f": lambda a, b: "false",
    "lt": lambda a, b: f"({a} < {b})",
    "eq": lambda a, b: f"({a} == {b})",
    "le": lambda a, b: f"({a} <= {b})",
    "gt": lambda a, b: f"({a} > {b})",
    "ne": lambda a, b: f"({a} != {b})",
    "ge": lambda a, b: f"({a} >= {b})",
    "t": lambda a, b: "true",
}
FLOAT_COMPARE_PREDICATES = {
    "f": lambda a, b: "false",
    "lt": lambda a, b: f"(!isnan({a}) && !isnan({b}) && ({a} < {b}))",
    "eq": lambda a, b: f"(!isnan({a}) && !isnan({b}) && ({a} == {b}))",
    "le": lambda a, b: f"(!isnan({a}) && !isnan({b}) && ({a} <= {b}))",
    "gt": lambda a, b: f"(!isnan({a}) && !isnan({b}) && ({a} > {b}))",
    "lg": lambda a, b: f"(!isnan({a}) && !isnan({b}) && ({a} != {b}))",
    "ge": lambda a, b: f"(!isnan({a}) && !isnan({b}) && ({a} >= {b}))",
    "o": lambda a, b: f"(!isnan({a}) && !isnan({b}))",
    "u": lambda a, b: f"(isnan({a}) || isnan({b}))",
    "nge": lambda a, b: f"(isnan({a}) || isnan({b}) || ({a} < {b}))",
    "nlg": lambda a, b: f"(isnan({a}) || isnan({b}) || ({a} == {b}))",
    "ngt": lambda a, b: f"(isnan({a}) || isnan({b}) || ({a} <= {b}))",
    "nle": lambda a, b: f"(isnan({a}) || isnan({b}) || ({a} > {b}))",
    "neq": lambda a, b: f"(isnan({a}) || isnan({b}) || ({a} != {b}))",
    "nlt": lambda a, b: f"(isnan({a}) || isnan({b}) || ({a} >= {b}))",
    "tru": lambda a, b: "true",
}
# The data ALU subset: the unsigned 32-bit operations the LLVM 15 code generator emits for the
# read-compute-write fixture, on known words only. v_and_b32, v_or_b32, v_xor_b32 and v_lshrrev_b32 (src1 >> src0[4:0])
# reuse the pixel profile's integer templates, as does v_mul_lo_u32. v_lshlrev_b32 and v_ashrrev_i32 shift src1
# by src0[4:0], the latter arithmetically; v_sub_nc_u32 is src0 - src1 and v_subrev_nc_u32 src1 - src0, both
# modulo 2^32. v_mad_u32_u24 is
# src0[23:0] * src1[23:0] + src2, modulo 2^32. None of them writes vcc, so the guard compare stays valid.
ALU = {
    "v_and_b32_e32": f"({INTEGER['and_b32'][1]})",
    "v_bfm_b32": "(((1u << ({0} & 31u)) - 1u) << ({1} & 31u))",  # VOP3 only: ((1 << src0[4:0]) - 1) << src1[4:0]
    "v_lshrrev_b32_e32": f"({INTEGER['lshrrev_b32'][1]})",
    "v_lshlrev_b32_e32": f"({INTEGER['lshlrev_b32'][1]})",
    # v_max3_u32 and v_min3_u32 are the unsigned maximum and minimum of the three sources, max(max(a, b), c).
    "v_max3_u32": "max(max({0}, {1}), {2})",
    "v_min3_u32": "min(min({0}, {1}), {2})",
    "v_ashrrev_i32_e32": INTEGER["ashrrev_i32"][1],
    "v_sub_nc_u32_e32": "({0} - {1})",
    # v_add_nc_i32 and v_sub_nc_i32 (VOP3, RDNA2 ISA: D.i = S0.i + S1.i; D.i = S0.i - S1.i)
    # wrap modulo 2^32 without carry-out; clamp is refused.
    "v_add_nc_i32": "({0} + {1})",
    "v_sub_nc_i32": "({0} - {1})",
    # v_med3_u32 and v_med3_i32 (VOP3) are the median of src0, src1 and src2, compared unsigned or signed.
    "v_med3_u32": "max(min({0}, {1}), min(max({0}, {1}), {2}))",
    "v_med3_i32": "asuint(max(min(asint({0}), asint({1})), min(max(asint({0}), asint({1})), asint({2}))))",
    "v_subrev_nc_u32_e32": "({1} - {0})",
    "v_min_u32_e32": "min({0}, {1})",
    "v_xor3_b32": "({0} ^ {1} ^ {2})",  # VOP3 src0 ^ src1 ^ src2 (RDNA2 ISA: D.u = S0.u ^ S1.u ^ S2.u)
    # v_bcnt_u32_b32 (VOP3 only on gfx10; RDNA2 ISA: D.u = CountOneBits(S0.u) + S1.u), the addend wrapping modulo 2^32.
    "v_bcnt_u32_b32": "(countbits({0}) + {1})",
    "v_max_u32_e32": "max({0}, {1})",
    "v_add_nc_u32_e32": "({0} + {1})",
    # v_mad_i32_i24 is sext(src0[23:0]) * sext(src1[23:0]) + src2, modulo 2^32, multiplied as uint (same low word).
    "v_mad_i32_i24": "((asuint(asint({0} << 8u) >> 8u) * asuint(asint({1} << 8u) >> 8u)) + {2})",
    "v_or_b32_e32": f"({INTEGER['or_b32'][1]})",
    "v_xor_b32_e32": f"({INTEGER['xor_b32'][1]})",
    "v_mul_lo_u32": INTEGER["mul_lo_u32"][1],
    "v_mad_u32_u24": "((({0} & 0xffffffu) * ({1} & 0xffffffu)) + {2})",
}
# The VOP1 bit subset on one known word (AMD RDNA 2 ISA reference, VOP1 opcodes 56 and 59). v_bfrev_b32 reverses
# the 32 bits. v_ffbh_i32 counts from the sign bit (bit 31 is position 0) to the first bit that differs from it,
# and is 0xffffffff for 0 and 0xffffffff: v_ffbh_u32 of m = x ^ (x >> 31, arithmetic). firstbitlow(reversebits(m))
# is 31 - firstbithigh(m) and 0xffffffff for m = 0, so the source appears only twice.
VOP1_ALU = {
    "v_bfrev_b32_e32": "reversebits({0})",
    "v_ffbh_i32_e32": "firstbitlow(reversebits({0} ^ asuint(asint({0}) >> 31u)))",
}
# The scalar ALU subset, on known words only: s_<op> sD, a, b with two scalar sources (an SGPR holding a known
# word, an integer inline constant or a 32-bit literal). The carry out of s_add_u32 and s_sub_u32 and the
# nonzero flag of the logic forms go to SCC, which no accepted instruction reads. s_lshr_b32 is a logical
# shift of src0 by src1[4:0].
SALU = {
    "s_add_u32": "({0} + {1})",
    # RDNA2 SOP2: temp.i64 = (int64)S0.i32 + (int64)S1.i32; D.i32 = temp.i32[31:0]; SCC = (temp.i64 != (int64)D.i32) ? 1 : 0 (unread).
    "s_add_i32": "({0} + {1})",
    # |S0.i - S1.i| on the 32-bit wrapped difference, so 0x80000000 stays 0x80000000; SCC = result != 0, unread.
    "s_absdiff_i32": "asuint(abs(asint({0} - {1})))",
    # RDNA2 SOP1: D.i32 = (S0.i32 < 0) ? -S0.i32 : S0.i32; SCC = (D.i32 != 0) ? 1 : 0 (unread). Wraps INT_MIN (0x80000000) to 0x80000000.
    "s_abs_i32": "asuint(abs(asint({0})))",
    "s_sub_u32": "({0} - {1})",
    # RDNA2 SOP2: temp.i64 = (int64)S0.i32 - (int64)S1.i32; D.i32 = temp.i32[31:0]; SCC = (temp.i64 != (int64)D.i32) ? 1 : 0 (unread).
    "s_sub_i32": "({0} - {1})",
    "s_and_b32": "({0} & {1})",
    "s_bitset0_b32": "({0} & ~(1u << ({1} & 31u)))",
    "s_bitset1_b32": "({0} | (1u << ({1} & 31u)))",
    "s_or_b32": "({0} | {1})",
    "s_xor_b32": "({0} ^ {1})",
    "s_lshr_b32": "({0} >> ({1} & 31u))",
    # RDNA2 ISA (SOP2 s_bfe_u32 / s_bfe_i32):
    #   shift = S1.u[4:0]; length = S1.u[22:16];
    #   if (length == 0) D = 0;
    #   else if (shift + length <= 32) D = (S0 << (32 - shift - length)) >> (32 - length); // logical u32, arithmetic i32
    #   else D = S0 >> shift; // logical u32, arithmetic i32
    #   SCC = (D != 0); (unread in this subset)
    "s_bfe_u32": "s_bfe_u32({0}, {1})",
    "s_bfe_i32": "s_bfe_i32({0}, {1})",
}
SALU_HELPERS = {
    "s_bfe_u32": (
        "uint s_bfe_u32(uint src, uint ctrl) {\n"
        "    uint offset = ctrl & 31u;\n"
        "    uint width = (ctrl >> 16u) & 127u;\n"
        "    if (width == 0u) return 0u;\n"
        "    if (offset + width <= 32u) return (src << (32u - offset - width)) >> (32u - width);\n"
        "    return src >> offset;\n"
        "}\n"
    ),
    "s_bfe_i32": (
        "uint s_bfe_i32(uint src, uint ctrl) {\n"
        "    uint offset = ctrl & 31u;\n"
        "    uint width = (ctrl >> 16u) & 127u;\n"
        "    if (width == 0u) return 0u;\n"
        "    if (offset + width <= 32u) return asuint((asint(src) << (32u - offset - width)) >> (32u - width));\n"
        "    return asuint(asint(src) >> offset);\n"
        "}\n"
    ),
}
# The binary32 vector ALU subset: VOP2 src0 is any source, src1 a VGPR. v_sub_f32 is src0 - src1 and v_subrev_f32
# is src1 - src0. Each operation rounds once to nearest even on the bit pattern of a uint word, so a result is again
# a word. The helpers keep the one rounding per operation: a multiply followed by an add must not be contracted.
# Fused multiply-add (v_fma_f32, v_fmac_f32) is not in the subset: DXIL has no binary32 fused form to lower it to.
# NaN payloads, infinities and denormals are not modelled. A literal that is one is refused; a word read from memory
# is not known statically, so the caller must keep its operands and results finite and normal.
FALU = {
    "v_add_f32_e32": "f32_add({0}, {1})",
    "v_sub_f32_e32": "f32_sub({0}, {1})",
    "v_subrev_f32_e32": "f32_sub({1}, {0})",
    "v_mul_f32_e32": "f32_mul({0}, {1})",
}
F32_HELPERS = {
    name: f"uint f32_{name}(uint a, uint b) {{ precise float r = asfloat(a) {op} asfloat(b); return asuint(r); }}\n"
    for name, op in (("add", "+"), ("sub", "-"), ("mul", "*"))
}
# v_mac_f32_{e32,e64} D, S0, S1 (VOP2 0x1f): D.f = S0.f * S1.f + D.f (unfused: product rounds to binary32 before add).
F32_HELPERS["mac"] = (
    "uint f32_mac(uint a, uint b, uint c) { precise float p = asfloat(a) * asfloat(b); "
    "precise float r = p + asfloat(c); return asuint(r); }\n"
)
# v_cvt_u32_f32_e32 D, S0 (VOP1 opcode 0x06 of the RDNA 2 ISA guide):
#   if (isnan(S0.f)) D.u = 0;
#   else if (S0.f <= 0.0) D.u = 0;
#   else if (S0.f >= 4294967296.0) D.u = 0xffffffff;
#   else D.u = (uint)trunc(S0.f);
# Clamp finite values to 2^32 - 256 (0x4f7fffff, the largest binary32 below 2^32) before uint()
# so even speculatively evaluated casts stay in range.
CVT_U32_F32_HELPER = (
    "uint cvt_u32_f32(uint bits) { float x = asfloat((bits & 0x7fffffffu) > 0x7f800000u ? 0u : bits); "
    "float finite = clamp(x, 0.0f, 4294967040.0f); return x >= 4294967296.0f ? 0xffffffffu : uint(finite); }\n"
)
# v_dot2_u32_u16 D, S0, S1, S2 (VOP3P 0x15):
#   D = (uint32)(((uint64)S0[15:0] * S1[15:0] + (uint64)S0[31:16] * S1[31:16] + (uint64)S2) & 0xffffffff)
# v_dot4_u32_u8 D, S0, S1, S2 (VOP3P 0x17):
#   D = (uint32)(((uint64)S0[7:0]*S1[7:0] + S0[15:8]*S1[15:8] + S0[23:16]*S1[23:16] + S0[31:24]*S1[31:24] + S2) & 0xffffffff)
# Non-clamp forms wrap modulo 2^32, so unsigned 32-bit HLSL arithmetic matches without 64-bit emulation.
DOT_U_HELPERS = {
    "dot2_u32_u16": (
        "uint dot2_u32_u16(uint a, uint b, uint c) {\n"
        "    return (a & 0xffffu) * (b & 0xffffu) + (a >> 16) * (b >> 16) + c;\n"
        "}\n"
    ),
    "dot4_u32_u8": (
        "uint dot4_u32_u8(uint a, uint b, uint c) {\n"
        "    return (a & 0xffu) * (b & 0xffu) + ((a >> 8) & 0xffu) * ((b >> 8) & 0xffu)\n"
        "        + ((a >> 16) & 0xffu) * ((b >> 16) & 0xffu) + (a >> 24) * (b >> 24) + c;\n"
        "}\n"
    ),
}
# The inline float constants of the encoding, as LLVM prints them, and the bits they hold.
FLOAT_INLINE = {
    "0.5": 0x3f000000, "-0.5": 0xbf000000, "1.0": 0x3f800000, "-1.0": 0xbf800000, "2.0": 0x40000000,
    "-2.0": 0xc0000000, "4.0": 0x40800000, "-4.0": 0xc0800000, "0.15915494": 0x3e22f983,
}
# v_cvt_pk_u8_f32 D, S0, S1, S2 (VOP3A opcode 350 of the RDNA 1.0 and RDNA 2 ISA guides): D = (S2 & ~(0xff << 8 * S1[1:0]))
# | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0]). The guides leave flt32_to_uint8 undefined: its rounding, its bounds and
# its NaN and negative results are not stated. The lowering holds only where every reading agrees, a binary32 S0 that is
# +0.0 or an integer in [1, 255]. A constant S0 outside that set is refused; a word read from memory is not known
# statically, so the caller must keep it inside the set.
CVT_PK_U8 = {
    "v_cvt_pk_u8_f32": "(({2} & ~(0xffu << (({1} & 3u) << 3u))) | ((uint(asfloat({0})) & 0xffu) << (({1} & 3u) << 3u)))",
}
FLOAT_SOURCE = r"(s\d+|v\d+|-?\d+\.\d+|0x[0-9a-f]+|-?\d+)"
VCC_HI = 107  # wave32 vcc_hi, tracked as a scalar word (see lower)
SALU_SOURCE = r"(s\d+|-?\d+|0x[0-9a-f]+)"
# Instruction prefixes that never write SCC, so a compare result survives them.
SCC_PRESERVING = ("s_cselect_b32", "s_mov_b32", "s_movk_i32", "s_load", "s_buffer_load", "s_waitcnt",
                  "s_nop", "s_clause", "v_", "buffer_")
ALU_SOURCE = r"(s\d+|v\d+|-?\d+|0x[0-9a-f]+)"
# Results are inline HLSL expressions; a longer one is rejected instead of growing without bound.
ALU_EXPRESSION_LIMIT = 1024
# Caller guarantees for a MIMG image descriptor: an 8-dword resource descriptor that names a
# linear (non-swizzled) image of the declared type and format. The translator checks where the eight words were
# loaded from, not the 9-bit GFX10 format, type, swizzle, level or size fields inside them; the two
# 16-byte descriptor slots are bound to t0 for reads or u0 for writes.
# Value: (HLSL texel type, channel count).
IMAGE_FORMATS = {"R32G32B32A32_UINT": ("uint4", 4), "R8G8B8A8_UINT": ("uint4", 4)}
RGBA32_FLOAT = "R32G32B32A32_FLOAT"
IMAGE_UAV_TYPES = {"1d": "RWTexture1D", "2d": "RWTexture2D", "2d-array": "RWTexture2DArray", "3d": "RWTexture3D"}
FLOAT_IMAGE_SRV_TYPES = {"1d": "Texture1D<float4>", "2d": "Texture2D<float4>"}
RGBA16_FLOAT = "R16G16B16A16_FLOAT"
# image_gather4_l on the gfx1013 RGBA32 FLOAT SRV, with the sampler profile of pixel_aot.check_gather_sampler
# (normalized coordinates, clamp to the last texel, point mip, no LOD bias). RDNA 2 MIMG gathers return one DMASK
# component of the 2x2 bilinear footprint at the selected mip, in (0,1), (1,1), (1,0), (0,0) order in increasing
# coordinates; Direct3D Gather uses the same order. The mip and both extents are relative to the bound view.
# Clamping the scaled coordinate to [-1, size] before floor() selects the same edge texels and keeps the
# float-to-int conversion in range. The caller keeps coordinates off tap boundaries (see experiments.md).
GATHER_HLSL = """uint4 image_gather_2d(float2 uv, uint mip, uint component) {
    uint width, height, levels;
    imageData.GetDimensions(mip, width, height, levels);
    int2 edge = int2(width, height) - 1;
    precise float2 scaled = uv * float2(width, height) - 0.5f;
    int2 origin = int2(floor(clamp(scaled, -1.0f, float2(width, height))));
    return uint4(asuint(imageData.Load(int3(clamp(origin + int2(0, 1), 0, edge), mip))[component]),
                 asuint(imageData.Load(int3(clamp(origin + int2(1, 1), 0, edge), mip))[component]),
                 asuint(imageData.Load(int3(clamp(origin + int2(1, 0), 0, edge), mip))[component]),
                 asuint(imageData.Load(int3(clamp(origin, 0, edge), mip))[component]));
}
"""
# Template operands are the old LDS word and the source VGPR, both HLSL uints.
LDS_RMW = {
    "ds_add_u32": "({0} + {1})",
    "ds_sub_u32": "({0} - {1})",
    "ds_and_b32": "({0} & {1})",
    "ds_or_b32": "({0} | {1})",
    "ds_xor_b32": "({0} ^ {1})",
    "ds_rsub_u32": "({1} - {0})",
    "ds_min_i32": "asuint(min(asint({0}), asint({1})))",
    "ds_max_i32": "asuint(max(asint({0}), asint({1})))",
    "ds_min_u32": "min({0}, {1})",
    "ds_max_u32": "max({0}, {1})",
}

# Group-shared LDS on gfx1013 (issue #1541): the D3D12 groupshared limit, the barrier after every DS step, and the
# non-returning updates as HLSL Interlocked methods on `lds` (all are order-independent within an instruction).
LDS_SHARED_LIMIT = 32768
LDS_BARRIER = "    GroupMemoryBarrierWithGroupSync();"
LDS_SHARED_ATOMICS = {"add_u32": "Add", "sub_u32": "Add", "and_b32": "And", "or_b32": "Or", "xor_b32": "Xor",
                      "min_u32": "Min", "max_u32": "Max"}
LDS_SHARED_FORMS = (
    r"ds_(?P<op>write|read)_b(?P<bits>32|64|128) (?P<a>v\d+|v\[\d+:\d+\]), (?P<b>v\d+|v\[\d+:\d+\])"
    r"(?: offset:(?P<offset>\d+))?",
    r"ds_(?P<op>write2|write2st64)_b(?P<bits>32|64) (?P<a>v\d+), (?P<b>v\d+|v\[\d+:\d+\]), (?P<c>v\d+|v\[\d+:\d+\])"
    r"(?: offset0:(?P<offset0>\d+))?(?: offset1:(?P<offset1>\d+))?",
    r"ds_(?P<op>read2|read2st64)_b(?P<bits>32|64) (?P<b>v\[\d+:\d+\]), (?P<a>v\d+)"
    r"(?: offset0:(?P<offset0>\d+))?(?: offset1:(?P<offset1>\d+))?",
    rf"ds_(?P<op>{'|'.join(LDS_SHARED_ATOMICS)}) (?P<a>v\d+), (?P<b>v\d+)(?: offset:(?P<offset>\d+))?",
)


def vgpr_range(token, count):
    """The VGPR numbers of `token` (vN or v[a:b]), which must name exactly `count` registers below v256."""
    m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", token)
    first, last = (int(m[1]), int(m[1])) if m[1] else (int(m[2]), int(m[3]))
    if last - first + 1 != count or last > 255 or (m[1] is None and count == 1):
        raise ValueError("Unverified shared LDS register range")
    return list(range(first, last + 1))


def shared_lds_step(line, step, vector, words):
    """HLSL statements for one gfx1013 DS instruction on `groupshared uint lds[]`; updates `vector` for reads.

    The byte address is the address VGPR plus the instruction offset (offset0/offset1 in elements of the access
    size, times 64 for st64), and word i of the access is lds[(address >> 2) + i].

    Input contract, not checked: every access is in bounds and aligned (b32 to 4 bytes, wider forms to their
    size); every word a read or an atomic touches was written earlier in the group; and no two lanes of one
    non-atomic write store to the same word. Outside the contract a word past the allocation reads as zero and
    is not written, and the low two address bits are dropped.
    """
    m = next((m for form in LDS_SHARED_FORMS if (m := re.fullmatch(form, line))), None)
    if m is None:
        raise ValueError(f"Unsupported shared LDS instruction: {line}")
    op = m["op"]
    # ds_read_bNN names its destination first; every other form names the address first.
    addr, data = (m["b"], m["a"]) if op == "read" else (m["a"], m["b"])
    address = vector.get(int(addr[1:])) if re.fullmatch(r"v\d+", addr) else None
    if not is_word(address) or int(addr[1:]) > 255:
        raise ValueError("Unverified shared LDS address")
    base = f"lds_a{step}"
    statements = []

    def place(offset):
        statements.append(f"    uint {base} = ({hlsl_word(address)} + {offset}u) >> 2u;")

    def words_of(token, count):
        values = [vector.get(reg) for reg in vgpr_range(token, count)]
        if not all(map(is_word, values)):
            raise ValueError("Shared LDS data is not a known word")
        return list(map(hlsl_word, values))

    def write(index, word):
        statements.append(f"    if ({index} < {words}u) lds[{index}] = {word};")

    def read(index, reg, name):
        statements.append(f"    uint {name} = {index} < {words}u ? lds[{index}] : 0u;")
        vector[reg] = ("alu", name)

    single = int(m["offset"] or 0) if "offset" in m.groupdict() else 0
    if single > 65535:
        raise ValueError("Unverified shared LDS offset")
    if op in LDS_SHARED_ATOMICS:
        place(single)
        (value,) = words_of(data, 1)
        method = LDS_SHARED_ATOMICS[op]
        if op == "sub_u32":
            # mem - v == mem + ~v + 1 (mod 2^32), as two additions. One InterlockedAdd of (0u - v) on groupshared
            # memory fails compute PSO creation with E_OUTOFMEMORY on NVIDIA GeForce RTX 5080 (driver
            # 32.0.16.2124); DXC folds every spelling of the negation to `sub i32 0, v`. The update has no
            # return value and the step ends with a barrier, so the intermediate word is never read.
            statements.append(f"    if ({base} < {words}u) InterlockedAdd(lds[{base}], ~({value}));")
            value = "1u"
        statements.append(f"    if ({base} < {words}u) Interlocked{method}(lds[{base}], {value});")
        return "\n".join(statements)
    count = int(m["bits"]) // 32
    if op in ("write", "read"):
        place(single)
        indices = [f"{base} + {i}u" if i else base for i in range(count)]
        if op == "write":
            for index, word in zip(indices, words_of(data, count)):
                write(index, word)
        else:
            for i, (index, reg) in enumerate(zip(indices, vgpr_range(data, count))):
                read(index, reg, f"lds_r{step}_{i}")
        return "\n".join(statements)
    # Paired forms: two elements of `count` words at offset0 and offset1, in elements (64 elements for st64).
    offsets = [int(m["offset0"] or 0), int(m["offset1"] or 0)]
    if max(offsets) > 255 or offsets[0] == offsets[1]:
        raise ValueError("Unverified shared LDS pair offsets")
    scale = count * (64 if op.endswith("st64") else 1)
    place(0)
    indices = [f"{base} + {offset * scale + i}u" if offset * scale + i else base
               for offset in offsets for i in range(count)]
    if op.startswith("write"):
        for index, word in zip(indices, words_of(m["b"], count) + words_of(m["c"], count)):
            write(index, word)
    else:
        for i, (index, reg) in enumerate(zip(indices, vgpr_range(m["b"], 2 * count))):
            read(index, reg, f"lds_r{step}_{i}")
    return "\n".join(statements)


MUBUF_ATOMIC_METHODS = {
    "add": "Add", "sub": "Add", "and": "And", "or": "Or", "xor": "Xor",
    "smin": "Min", "umin": "Min", "smax": "Max", "umax": "Max",
    "swap": "Exchange", "cmpswap": "CompareExchange",
    "inc": "CompareExchange", "dec": "CompareExchange", "csub": "CompareExchange",
    "fmin": "CompareExchange", "fmax": "CompareExchange", "fcmpswap": "CompareExchange",
}

IMAGE_FLOAT_ATOMIC_OPS = ("fcmpswap", "fmin", "fmax")

def float_min_max_cas(cas_prefix, value, op, *, wide=False):
    """Raw-word finite-normal binary32/binary64 min/max as an InterlockedCompareExchange loop.

    Shared by the image (texel) and MUBUF (byte-address) atomic profiles; cas_prefix is the
    call up to the opening parenthesis: the free intrinsic `InterlockedCompareExchange(dest)`
    for a texel destination and the raw UAV method `outputData.InterlockedCompareExchange(off)`
    for the byte-address destination. The caller guarantees every stored word and the operand
    are finite normals, never NaN, subnormal or signed zero. The sign-dependent key
    orders those values exactly as unsigned integers, so integer CAS reproduces the fmin/fmax
    result bit-for-bit. The zero seed is outside the caller's domain and can only fetch the
    observed word; it never writes.
    """
    comparison = "<" if op == "fmin" else ">"
    kind, zero, sign, mask = ("uint64_t", "0ull", "0x8000000000000000ull", "0xffffffffffffffffull") if wide else (
        "uint", "0u", "0x80000000u", "0xffffffffu")
    return (f"    {kind} value = {value};",
            f"    {kind} valueKey = value ^ ((value & {sign}) ? {mask} : {sign});",
            f"    {kind} observed = {zero};", f"    {kind} expected;",
            "    [allow_uav_condition] do {", "        expected = observed;",
            f"        {kind} oldKey = expected ^ ((expected & {sign}) ? {mask} : {sign});",
            f"        {kind} next = (valueKey {comparison} oldKey) ? value : expected;",
            f"        {cas_prefix}, expected, next, observed);",
            "    } while (observed != expected);")

IMAGE_ATOMIC_OPS = "add|sub|and|or|xor|smin|smax|umin|umax|swap|cmpswap|inc|dec|" + "|".join(IMAGE_FLOAT_ATOMIC_OPS)
IMAGE_ATOMIC_TYPES = {"R32_UINT": "uint", "R32_SINT": "int", "R32_FLOAT": "uint"}


# LLVM's spellings of a 32-bit immediate: decimal, hex, an inline float (`1.0`, `0.15915494`, `-2.5e-3`) and
# the non-finite names. An s_setreg_imm32_b32 operand is always a literal, never a register.
_IMM32_LITERAL = re.compile(r"-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)",
                             re.IGNORECASE)

def checked_group(group):
    """Return (x, y, z) within the D3D12 cs_6_0 numthreads limits."""
    x, y, z = (group, 1, 1) if isinstance(group, int) else tuple(group)
    if not (all(isinstance(n, int) for n in (x, y, z)) and 1 <= x <= 1024 and 1 <= y <= 1024
            and 1 <= z <= 64 and x * y * z <= 1024):
        raise ValueError("Unsupported thread group")
    return x, y, z


def entry_registers(rsrc2):
    """The entry registers a gfx10 compute wave receives, from COMPUTE_PGM_RSRC2.

    Returns (user SGPR count, {axis: group ID SGPR}, thread-ID VGPR count). The user SGPRs come
    first (USER_SGPR, bits 5:1), then one SGPR for each group ID enabled by TGID_X/Y/Z_EN
    (bits 7-9), in X, Y, Z order with no gap for an omitted axis. TIDIG_COMP_CNT (bits 12:11)
    gives v0 (X), v0-v1 (X, Y) or v0-v2 (X, Y, Z), each local ID in its own VGPR on gfx10.
    TG_SIZE_EN (bit 10) is refused: the gfx1013 TG_SIZE word layout and the guest wave width
    are not established, so its register cannot be given a value.
    """
    if not isinstance(rsrc2, int) or isinstance(rsrc2, bool) or not 0 <= rsrc2 < 1 << 32:
        raise ValueError("Invalid COMPUTE_PGM_RSRC2")
    if rsrc2 >> 10 & 1:
        raise ValueError("COMPUTE_PGM_RSRC2 TG_SIZE_EN is not supported")
    reg = user = (rsrc2 >> 1) & 0x1f
    groups = {}
    for axis in range(3):
        if rsrc2 >> (7 + axis) & 1:
            groups[axis] = reg
            reg += 1
    components = (rsrc2 >> 11) & 3
    if components == 3:
        raise ValueError("Unsupported COMPUTE_PGM_RSRC2 thread-ID component count")
    return user, groups, components + 1


def group_size(psv):
    size = struct.unpack_from("<I", psv)[0] if len(psv) >= 4 else 0
    if not 48 <= size <= len(psv) - 4 or psv[28] != 5:
        raise ValueError("Expected compute PSV runtime info with thread dimensions")
    return checked_group(struct.unpack_from("<3I", psv, 40))


def is_word(value):
    return isinstance(value, int) or (isinstance(value, tuple)
                                      and value[0] in ("constant_word", "thread", "source", "alu"))


def alu_data_word(value):
    """A known word that is not a dispatch index: a shift of one is data, where a shifted index is a byte offset."""
    return is_word(value) and not (isinstance(value, tuple) and value[0] == "thread")


def hlsl_word(value):
    if isinstance(value, int):
        return f"0x{value:08x}u"
    if value[0] == "source":
        # The source binding is read-only, so the read may sit where the word is used.
        return f"inputData.Load({value[1]})"
    if value[0] == "alu":
        return value[1]
    return f"tid.{AXES[value[1]]}" if value[0] == "thread" else f"params[{value[1] // 4}].{'xyzw'[value[1] % 4]}"


def alu_word(token, scalar, vector, pending, sizes=None):
    """The known word an ALU source names, or None: a VGPR, a loaded SGPR, an inline constant or a literal.

    sizes maps a descriptor word tag to the HLSL expression of the size the caller guarantees for it."""
    if token[0] == "v":
        value = vector.get(int(token[1:]))
    elif token[0] == "s":
        n = int(token[1:])
        value = scalar.get(n) if (n <= 105 or n == VCC_HI) and n not in pending else None
        if sizes and isinstance(value, tuple) and value in sizes:
            value = ("alu", sizes[value])
    else:
        value = int(token, 0) & 0xffffffff if -(1 << 31) <= int(token, 0) <= 0xffffffff else None
    return value if is_word(value) else None


def float_word(token, scalar, vector, pending, sizes=None):
    """The known word a binary32 source names, or None: as alu_word, but a constant must be a finite normal value or zero."""
    word = FLOAT_INLINE.get(token) if "." in token else alu_word(token, scalar, vector, pending, sizes)
    if isinstance(word, int) and ((word >> 23) & 0xff == 0xff or ((word >> 23) & 0xff == 0 and word & 0x7fffff)):
        return None
    return word


def float_e64_source(token, scalar, vector, pending):
    """The HLSL word of a VOP3 binary32 source after its source modifiers, or None for an unknown word.

    LLVM prints the VOP3 source modifiers on a register as -v, |v| and -|v|. They act on the sign bit of the word:
    abs clears bit 31, then neg flips it, as pixel_aot does for v_cvt_f64_f32_e64 and binary64 compare sources, so a
    zero or NaN keeps its other bits. Modifiers on a constant (neg(0x...), |1.0|) and the abs() spelling are refused.
    """
    m = re.fullmatch(r"(-?)(\|?)(s\d+|v\d+)(\|?)", token)
    if not m and not re.fullmatch(FLOAT_SOURCE, token):
        raise ValueError(f"Unsupported float source modifier: {token}")
    if m and m[2] != m[4]:
        raise ValueError(f"Unsupported float source modifier: {token}")
    word = float_word(m[3] if m else token, scalar, vector, pending)
    if word is None:
        return None
    expression = hlsl_word(word)
    if m and m[2]:
        expression = f"({expression} & 0x7fffffffu)"
    if m and m[1]:
        expression = f"({expression} ^ 0x80000000u)"
    return expression


def vector_operands(line):
    """VGPR numbers named by an instruction, single or as a range."""
    regs = set()
    for single, first, last in re.findall(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]", line):
        regs.update([int(single)] if single else range(int(first), int(last) + 1))
    return regs


def labelled(decoded):
    """Name each s_cbranch_execz target as a label; decoded is [(byte offset, instruction)].

    Only forward targets at a decoded instruction boundary are named. lower() then checks
    where the label sits.
    """
    starts = {pc for pc, _ in decoded}
    targets, lines = set(), []
    for pc, line in decoded:
        if m := re.fullmatch(r"s_cbranch_execz (-?\d+)", line):
            # A backward target, printed signed or as its 16-bit pattern, is never accepted.
            target = pc + 4 + 4 * int(m[1])
            if target <= pc or target not in starts:
                raise ValueError("Branch outside decoded instruction boundaries")
            targets.add(target)
            line = f"s_cbranch_execz .L{target}"
        lines.append((pc, line))
    return [item for pc, line in lines for item in ([f".L{pc}:"] if pc in targets else []) + [line]]


MUBUF_UINT_FORMATS = tuple(f"BUF_FMT_{'32_' * count}UINT" for count in range(1, 5))
MUBUF_SOURCE_FORMAT = MUBUF_UINT_FORMATS[-1]
MUBUF_D16_OUTPUT_FORMATS = {1: "BUF_FMT_16_UINT", 2: "BUF_FMT_16_16_UINT", 4: "BUF_FMT_16_16_16_16_UINT"}
MUBUF_OUTPUT_FORMATS = (*MUBUF_UINT_FORMATS, *MUBUF_D16_OUTPUT_FORMATS.values())
MUBUF_TEXEL_BYTES = 16
MUBUF_D16_SOURCE_FORMAT = "BUF_FMT_16_16_16_16_UINT"
MUBUF_D16_TEXEL_BYTES = 8
MUBUF_SOURCE_TEXELS = {MUBUF_SOURCE_FORMAT: MUBUF_TEXEL_BYTES, MUBUF_D16_SOURCE_FORMAT: MUBUF_D16_TEXEL_BYTES}

# GFX10 SQ_BUF_RSRC_WORD3. Identity DST_SEL is SQ_SEL_X/Y/Z/W (4, 5, 6, 7). FORMAT is bits
# 18:12. RESOURCE_LEVEL (bit 24) is 1 for that encoding and 0 for the legacy DFMT/NFMT field
# in the same bits: NFMT[14:12], DFMT[18:15] (Mesa sid.h). Every other bit, including
# ADD_TID_ENABLE, index stride, OOB and type, has to match or the format is not one we read.
DESCRIPTOR_FORMAT_SHIFT = 12
DESCRIPTOR_FORMAT_MASK = 0x7F
DESCRIPTOR_FIXED_MASK = 0xFFF80FFF
DESCRIPTOR_IDENTITY_SEL = 0xFAC
DESCRIPTOR_GFX10_FIXED = DESCRIPTOR_IDENTITY_SEL | (1 << 24)
DESCRIPTOR_LEGACY_FIXED = DESCRIPTOR_IDENTITY_SEL
# GFX10 BUF_FMT code -> (packer, channel count). The packer consumes binary32 bits, or the
# raw VGPR bits for raw32 and the 16-bit integer forms. A d16 opcode converts each half to
# binary32 first. 16-bit UINT/SINT with d16 is not a defined conversion, so those stores
# are skipped rather than guessed. Codes absent from this table are refused the same way.
# AMD format names list the fields from the most significant bit: Mesa's gfx10_format_table.py reverses
# them ("Keep the bit sizes in little-endian order"), and radeonsi and radv map R10G10B10A2 (x in bits
# 9:0, w in bits 31:30) to 2_10_10_10. So unorm10 is BUF_FMT_2_10_10_10_UNORM (50, legacy DFMT 9), and
# BUF_FMT_10_10_10_2_UNORM (44, legacy DFMT 8), whose x is the 2-bit field in bits 1:0, is not handled.
DESCRIPTOR_STORE_FORMATS = {
    20: ("raw32", 1), 22: ("raw32", 1),
    62: ("raw32", 2), 64: ("raw32", 2),
    72: ("raw32", 3), 74: ("raw32", 3),
    75: ("raw32", 4), 77: ("raw32", 4),
    56: ("unorm8", 4), 23: ("unorm16x2", 2), 50: ("unorm10", 4),
    29: ("f16x2", 2), 71: ("f16x4", 4),
    7: ("unorm16", 1), 8: ("snorm16", 1),
    11: ("uint16", 1), 12: ("sint16", 1), 13: ("f16", 1),
}
# Legacy (DFMT << 3) | NFMT for the same packers. DFMT: 16=2, 32=4, 16_16=5, 2_10_10_10=9,
# 8_8_8_8=10, 32_32=11, 16_16_16_16=12, 32_32_32=13, 32_32_32_32=14.
# NFMT: UNORM=0, SNORM=1, UINT=4, SINT=5, FLOAT=7.
def _legacy_field(dfmt, nfmt):
    return (dfmt << 3) | nfmt


LEGACY_DESCRIPTOR_FORMATS = {
    _legacy_field(4, 4): 20, _legacy_field(4, 7): 22,
    _legacy_field(11, 4): 62, _legacy_field(11, 7): 64,
    _legacy_field(13, 4): 72, _legacy_field(13, 7): 74,
    _legacy_field(14, 4): 75, _legacy_field(14, 7): 77,
    _legacy_field(10, 0): 56, _legacy_field(5, 0): 23, _legacy_field(9, 0): 50,
    _legacy_field(5, 7): 29, _legacy_field(12, 7): 71,
    _legacy_field(2, 0): 7, _legacy_field(2, 1): 8,
    _legacy_field(2, 4): 11, _legacy_field(2, 5): 12, _legacy_field(2, 7): 13,
}
_DESCRIPTOR_KIND_ORDER = ("raw32", "unorm8", "unorm16x2", "unorm10", "f16x2", "f16x4",
                          "unorm16", "snorm16", "uint16", "sint16", "f16")


def decode_buffer_descriptor_format(word):
    """Return (gfx10 code, packer, channels) for one handled V# word 3.

    Raises ValueError when the fixed bits or the format field are not a handled store.
    """
    if not isinstance(word, int) or isinstance(word, bool) or not 0 <= word <= 0xffffffff:
        raise ValueError("Unsupported buffer descriptor format")
    fixed = word & DESCRIPTOR_FIXED_MASK
    field = (word >> DESCRIPTOR_FORMAT_SHIFT) & DESCRIPTOR_FORMAT_MASK
    if fixed == DESCRIPTOR_GFX10_FIXED:
        code = field
    elif fixed == DESCRIPTOR_LEGACY_FIXED:
        code = LEGACY_DESCRIPTOR_FORMATS.get(field)
    else:
        code = None
    kind = DESCRIPTOR_STORE_FORMATS.get(code)
    if kind is None:
        raise ValueError("Unsupported buffer descriptor format")
    return code, kind[0], kind[1]


def descriptor_format_word(code, *, legacy=False):
    """Identity V# word 3 for one handled GFX10 format code."""
    if code not in DESCRIPTOR_STORE_FORMATS:
        raise ValueError("Unsupported buffer descriptor format")
    if legacy:
        field = next(item for item, gfx in LEGACY_DESCRIPTOR_FORMATS.items() if gfx == code)
        fixed = DESCRIPTOR_LEGACY_FIXED
    else:
        field, fixed = code, DESCRIPTOR_GFX10_FIXED
    return fixed | (field << DESCRIPTOR_FORMAT_SHIFT)


def descriptor_store_hlsl():
    """One store function shared by every descriptor-format opcode.

    An unmatched format, a channel-count mismatch, a misaligned dword store, or d16
    combined with a 16-bit integer format writes nothing.

    UNORM and SNORM use the Direct3D float conversion: scale into the integer range,
    add 0.5 for a non-negative value or subtract 0.5 for a negative one, then drop
    the fraction (half away from zero). HLSL round() is ties-to-even and disagrees
    on an exact tie. The RDNA ISA does not state the MUBUF format unit's tie break.
    LLVM's AMDGPU packed conversions (v_cvt_pk_*) are ties-to-even on a different
    unit, so this choice follows the Direct3D rule and is unverified on gfx1013.
    """
    legacy = []
    for field, code in sorted(LEGACY_DESCRIPTOR_FORMATS.items()):
        keyword = "if" if not legacy else "else if"
        legacy.append(f"        {keyword} (field == {field}u) format = {code}u;")
    grouped = {}
    for code, (kind, channels) in DESCRIPTOR_STORE_FORMATS.items():
        grouped.setdefault((kind, channels), []).append(code)
    bodies = {
        ("raw32", 1): "        outputData.Store(addr, e0);",
        ("raw32", 2): "        outputData.Store2(addr, uint2(e0, e1));",
        ("raw32", 3): "        outputData.Store3(addr, uint3(e0, e1, e2));",
        ("raw32", 4): "        outputData.Store4(addr, uint4(e0, e1, e2, e3));",
        ("unorm8", 4): (
            "        uint packed = descriptor_unorm(e0, 255.0) | (descriptor_unorm(e1, 255.0) << 8u)"
            " | (descriptor_unorm(e2, 255.0) << 16u) | (descriptor_unorm(e3, 255.0) << 24u);\n"
            "        outputData.Store(addr, packed);"),
        ("unorm16x2", 2): (
            "        uint packed = descriptor_unorm(e0, 65535.0) | (descriptor_unorm(e1, 65535.0) << 16u);\n"
            "        outputData.Store(addr, packed);"),
        ("unorm10", 4): (
            "        uint packed = descriptor_unorm(e0, 1023.0) | (descriptor_unorm(e1, 1023.0) << 10u)"
            " | (descriptor_unorm(e2, 1023.0) << 20u) | (descriptor_unorm(e3, 3.0) << 30u);\n"
            "        outputData.Store(addr, packed);"),
        ("f16x2", 2): (
            "        uint packed = descriptor_f16(e0) | (descriptor_f16(e1) << 16u);\n"
            "        outputData.Store(addr, packed);"),
        ("f16x4", 4): (
            "        uint lo = descriptor_f16(e0) | (descriptor_f16(e1) << 16u);\n"
            "        uint hi = descriptor_f16(e2) | (descriptor_f16(e3) << 16u);\n"
            "        outputData.Store2(addr, uint2(lo, hi));"),
        ("unorm16", 1): "        descriptor_rmw16(addr, descriptor_unorm(e0, 65535.0));",
        ("snorm16", 1): "        descriptor_rmw16(addr, descriptor_snorm16(e0));",
        ("uint16", 1): "        descriptor_rmw16(addr, e0);",
        ("sint16", 1): "        descriptor_rmw16(addr, e0);",
        ("f16", 1): "        descriptor_rmw16(addr, descriptor_f16(e0));",
    }
    arms = []
    for kind in _DESCRIPTOR_KIND_ORDER:
        for channels in range(1, 5):
            codes = grouped.get((kind, channels))
            if not codes:
                continue
            test = " || ".join(f"format == {code}u" for code in sorted(codes))
            integer_d16 = " && d16 == 0u" if kind in ("uint16", "sint16") else ""
            aligned = "" if kind in ("unorm16", "snorm16", "uint16", "sint16", "f16") else " && (addr & 3u) == 0u"
            keyword = "if" if not arms else "else if"
            arms.append(f"    {keyword} (({test}) && channels == {channels}u{integer_d16}{aligned}) {{\n"
                        f"{bodies[(kind, channels)]}\n    }}")
    fixed = f"{DESCRIPTOR_FIXED_MASK:#x}u"
    return (
        "uint descriptor_unorm(uint bits, float scale) {\n"
        "    precise float saturated = saturate(asfloat(bits));\n"
        "    precise float scaled = saturated * scale;\n"
        "    return (uint)floor(scaled + 0.5);\n"
        "}\n"
        "uint descriptor_snorm16(uint bits) {\n"
        "    precise float saturated = clamp(asfloat(bits), -1.0, 1.0);\n"
        "    precise float scaled = saturated * 32767.0;\n"
        "    precise float rounded = scaled >= 0.0 ? floor(scaled + 0.5) : ceil(scaled - 0.5);\n"
        "    return (uint)(int)rounded;\n"
        "}\n"
        "uint descriptor_f16(uint bits) { return f32tof16(asfloat(bits)); }\n"
        "uint descriptor_expand(uint packed, uint high) {\n"
        "    uint half = high != 0u ? (packed >> 16u) : (packed & 0xffffu);\n"
        "    return asuint(f16tof32(half));\n"
        "}\n"
        "void descriptor_rmw16(uint addr, uint half) {\n"
        "    if ((addr & 1u) != 0u) return;\n"
        "    uint base = addr & 0xfffffffcu;\n"
        "    uint shift = (addr & 2u) << 3u;\n"
        "    uint clear = ~(0xffffu << shift);\n"
        "    outputData.InterlockedAnd(base, clear);\n"
        "    outputData.InterlockedOr(base, (half & 0xffffu) << shift);\n"
        "}\n"
        "void store_descriptor_format(uint formatWord, uint addr, uint channels, uint d16, "
        "uint c0, uint c1, uint c2, uint c3) {\n"
        f"    uint fixedBits = formatWord & {fixed};\n"
        f"    uint field = (formatWord >> {DESCRIPTOR_FORMAT_SHIFT}u) & {DESCRIPTOR_FORMAT_MASK:#x}u;\n"
        "    uint format = 0u;\n"
        f"    if (fixedBits == {DESCRIPTOR_GFX10_FIXED:#x}u) {{\n"
        "        format = field;\n"
        f"    }} else if (fixedBits == {DESCRIPTOR_LEGACY_FIXED:#x}u) {{\n"
        + "\n".join(legacy) + "\n"
        "    }\n"
        "    uint e0 = d16 != 0u ? descriptor_expand(c0, 0u) : c0;\n"
        "    uint e1 = d16 != 0u ? descriptor_expand(c0, 1u) : c1;\n"
        "    uint e2 = d16 != 0u ? descriptor_expand(c1, 0u) : c2;\n"
        "    uint e3 = d16 != 0u ? descriptor_expand(c1, 1u) : c3;\n"
        + "\n".join(arms) + "\n"
        "}\n"
    )


def linear_uav_address(tag, dims, idxen, offen, stride, offset, size):
    """Address one lane's field within an indexed or shifted dispatch element."""
    if not isinstance(tag, tuple):
        return None
    if idxen and not offen and tag[0] == "thread" and stride is not None and offset + size <= stride:
        axis = tag[1]
        return ("index", axis), axis, f"tid.{AXES[axis]} * {stride}u + {offset}u"
    if offen and not idxen and tag[0] == "thread_bytes":
        _, axis, shift = tag
        if 2 <= shift <= 31 and offset + size <= 1 << shift and dims[axis] << shift <= 1 << 32:
            return ("byte_offset", axis, shift), axis, f"(tid.{AXES[axis]} << {shift}u) + {offset}u"
    return None


def descriptor_load_hlsl():
    """One load function shared by every descriptor-format buffer_load_format opcode.

    It returns the four result words of one load from the 4-byte-aligned t0 address and sets valid only when the
    fixed bits are an identity value, the format is handled, and the opcode reads exactly the format's channels.
    The format unit's results outside that are not established here, so lower() makes an invalid load suppress
    every store of the invocation: no result is published, and zero data from a valid read stays distinguishable.

    Every conversion is integer arithmetic, so the result does not depend on the GPU's float division. UNORM is
    x / (2^n - 1) and SNORM16 is max(x / 32767, -1), rounded to nearest even binary32 by long division. FLOAT16
    widens exactly, keeping NaN payloads. 16-bit UINT and SINT zero- and sign-extend the low halfword.

    With d16 the channels pack two per result word, x in the low half. Float-type formats give binary16: the
    FLOAT16 halves as stored, and UNORM and SNORM16 rounded to nearest even binary16 (denormals kept) by long
    division. 16-bit UINT and SINT give the 16-bit integer, not a float. A 32-bit format has no d16 rule here and
    is invalid.
    """
    legacy = []
    for field, code in sorted(LEGACY_DESCRIPTOR_FORMATS.items()):
        keyword = "if" if not legacy else "else if"
        legacy.append(f"        {keyword} (field == {field}u) format = {code}u;")
    bodies = {
        "raw32": "r = raw;",
        "unorm8": ("r = uint4(descriptor_ratio(raw.x & 0xffu, 255u), descriptor_ratio((raw.x >> 8u) & 0xffu, 255u), "
                   "descriptor_ratio((raw.x >> 16u) & 0xffu, 255u), descriptor_ratio(raw.x >> 24u, 255u));"),
        "unorm16x2": "r.xy = uint2(descriptor_ratio(raw.x & 0xffffu, 65535u), descriptor_ratio(raw.x >> 16u, 65535u));",
        "unorm10": ("r = uint4(descriptor_ratio(raw.x & 0x3ffu, 1023u), descriptor_ratio((raw.x >> 10u) & 0x3ffu, 1023u), "
                    "descriptor_ratio((raw.x >> 20u) & 0x3ffu, 1023u), descriptor_ratio(raw.x >> 30u, 3u));"),
        "f16x2": "r.xy = uint2(descriptor_half(raw.x & 0xffffu), descriptor_half(raw.x >> 16u));",
        "f16x4": ("r = uint4(descriptor_half(raw.x & 0xffffu), descriptor_half(raw.x >> 16u), "
                  "descriptor_half(raw.y & 0xffffu), descriptor_half(raw.y >> 16u));"),
        "unorm16": "r.x = descriptor_ratio(raw.x & 0xffffu, 65535u);",
        "snorm16": "r.x = descriptor_snorm16_load(raw.x & 0xffffu);",
        "uint16": "r.x = raw.x & 0xffffu;",
        "sint16": "r.x = (uint)((int)(raw.x << 16u) >> 16u);",
        "f16": "r.x = descriptor_half(raw.x & 0xffffu);",
    }
    grouped = {}
    for code, (kind, channels) in DESCRIPTOR_STORE_FORMATS.items():
        grouped.setdefault((kind, channels), []).append(code)
    halves = {
        "unorm8": ("h = uint4(descriptor_ratio16(raw.x & 0xffu, 255u), descriptor_ratio16((raw.x >> 8u) & 0xffu, 255u), "
                   "descriptor_ratio16((raw.x >> 16u) & 0xffu, 255u), descriptor_ratio16(raw.x >> 24u, 255u));"),
        "unorm16x2": ("h.xy = uint2(descriptor_ratio16(raw.x & 0xffffu, 65535u), "
                      "descriptor_ratio16(raw.x >> 16u, 65535u));"),
        "unorm10": ("h = uint4(descriptor_ratio16(raw.x & 0x3ffu, 1023u), "
                    "descriptor_ratio16((raw.x >> 10u) & 0x3ffu, 1023u), "
                    "descriptor_ratio16((raw.x >> 20u) & 0x3ffu, 1023u), descriptor_ratio16(raw.x >> 30u, 3u));"),
        "f16x2": "h.xy = uint2(raw.x & 0xffffu, raw.x >> 16u);",
        "f16x4": "h = uint4(raw.x & 0xffffu, raw.x >> 16u, raw.y & 0xffffu, raw.y >> 16u);",
        "unorm16": "h.x = descriptor_ratio16(raw.x & 0xffffu, 65535u);",
        "snorm16": "h.x = descriptor_snorm16_half(raw.x & 0xffffu);",
        "uint16": "h.x = raw.x & 0xffffu;",
        "sint16": "h.x = raw.x & 0xffffu;",
        "f16": "h.x = raw.x & 0xffffu;",
    }
    arms = []
    for d16, table in ((False, bodies), (True, halves)):
        for kind in _DESCRIPTOR_KIND_ORDER:
            for channels in range(1, 5):
                codes = grouped.get((kind, channels))
                if codes and kind in table:
                    test = " || ".join(f"format == {code}u" for code in sorted(codes))
                    keyword = "if" if not arms else "else if"
                    arms.append(f"    {keyword} (d16 {'!=' if d16 else '=='} 0u && ({test}) && channels == {channels}u) "
                                f"{{ {table[kind]} valid = true; }}")
    return (
        "uint descriptor_ratio(uint x, uint d) {\n"
        "    // binary32 bits of x / d for 0 <= x <= d < 2^16, rounded to nearest even by long division.\n"
        "    if (x == 0u) return 0u;\n"
        "    if (x == d) return 0x3f800000u;\n"
        "    uint r = x, bits = 0u, taken = 0u, lead = 0u;\n"
        "    for (uint i = 0u; i < 48u && taken < 25u; ++i) {\n"
        "        r <<= 1u;\n"
        "        uint bit = r >= d ? 1u : 0u;\n"
        "        r -= bit * d;\n"
        "        if (taken == 0u && bit == 0u) { ++lead; continue; }\n"
        "        bits = (bits << 1u) | bit;\n"
        "        ++taken;\n"
        "    }\n"
        "    uint result = ((126u - lead) << 23u) | ((bits >> 1u) & 0x7fffffu);\n"
        "    if ((bits & 1u) != 0u && (r != 0u || (result & 1u) != 0u)) ++result;\n"
        "    return result;\n"
        "}\n"
        "uint descriptor_ratio16(uint x, uint d) {\n"
        "    // binary16 bits of x / d for 0 <= x <= d < 2^16, rounded to nearest even, denormals kept.\n"
        "    if (x == 0u) return 0u;\n"
        "    if (x == d) return 0x3c00u;\n"
        "    uint r = x, f = 0u;\n"
        "    for (uint i = 0u; i < 26u; ++i) {\n"
        "        r <<= 1u;\n"
        "        uint bit = r >= d ? 1u : 0u;\n"
        "        r -= bit * d;\n"
        "        f = (f << 1u) | bit;\n"
        "    }\n"
        "    // x / d = f / 2^26 plus a remainder. A normal result keeps 11 bits; a denormal counts units of 2^-24.\n"
        "    uint lead = firstbithigh(f);\n"
        "    uint shift = lead >= 12u ? lead - 10u : 2u;\n"
        "    uint result = lead >= 12u ? (((lead - 11u) << 10u) | ((f >> shift) & 0x3ffu)) : (f >> shift);\n"
        "    uint roundBit = (f >> (shift - 1u)) & 1u;\n"
        "    bool sticky = (f & ((1u << (shift - 1u)) - 1u)) != 0u || r != 0u;\n"
        "    if (roundBit != 0u && (sticky || (result & 1u) != 0u)) ++result;\n"
        "    return result;\n"
        "}\n"
        "uint descriptor_snorm16_half(uint bits16) {\n"
        "    int value = (int)(bits16 << 16u) >> 16;\n"
        "    if (value <= -32767) return 0xbc00u;\n"
        "    return value < 0 ? 0x8000u | descriptor_ratio16((uint)(-value), 32767u) : descriptor_ratio16((uint)value, 32767u);\n"
        "}\n"
        "uint descriptor_snorm16_load(uint bits16) {\n"
        "    int value = (int)(bits16 << 16u) >> 16;\n"
        "    if (value <= -32767) return 0xbf800000u;\n"
        "    return value < 0 ? 0x80000000u | descriptor_ratio((uint)(-value), 32767u) : descriptor_ratio((uint)value, 32767u);\n"
        "}\n"
        "uint descriptor_half(uint h) {\n"
        "    uint sign = (h & 0x8000u) << 16u, e = (h >> 10u) & 31u, m = h & 0x3ffu;\n"
        "    if (e == 31u) return sign | 0x7f800000u | (m << 13u);\n"
        "    if (e != 0u) return sign | ((e + 112u) << 23u) | (m << 13u);\n"
        "    if (m == 0u) return sign;\n"
        "    uint top = firstbithigh(m);\n"
        "    return sign | ((top + 103u) << 23u) | ((m << (23u - top)) & 0x7fffffu);\n"
        "}\n"
        "uint4 load_descriptor_format(uint formatWord, uint addr, uint channels, uint d16, out bool valid) {\n"
        "    valid = false;\n"
        f"    uint fixedBits = formatWord & {DESCRIPTOR_FIXED_MASK:#x}u;\n"
        f"    uint field = (formatWord >> {DESCRIPTOR_FORMAT_SHIFT}u) & {DESCRIPTOR_FORMAT_MASK:#x}u;\n"
        "    uint format = 0u;\n"
        f"    if (fixedBits == {DESCRIPTOR_GFX10_FIXED:#x}u) {{\n"
        "        format = field;\n"
        f"    }} else if (fixedBits == {DESCRIPTOR_LEGACY_FIXED:#x}u) {{\n"
        + "\n".join(legacy) + "\n"
        "    }\n"
        "    uint4 raw = inputData.Load4(addr);\n"
        "    uint4 r = uint4(0u, 0u, 0u, 0u), h = uint4(0u, 0u, 0u, 0u);\n"
        + "\n".join(arms) + "\n"
        "    if (d16 != 0u) r = uint4(h.x | (h.y << 16u), h.z | (h.w << 16u), 0u, 0u);\n"
        "    return r;\n"
        "}\n"
    )


HEAP_VIEW_RECORD_BYTES = 32
HEAP_VIEW_LIVE = 1
HEAP_VIEW_MAX = 64


def heap_view_hlsl(table_offset, view_count, buffer_bytes):
    """Retrieve the guest V# of one heap view from a side record table and bound the read by it.

    The runtime writes one 32-byte record per heap view: V# words 0 to 3 (base byte offset, stride in
    word 1 bits 29:16, num_records, format word), then a state word that is 1 only while the view is live,
    then three reserved words. Words 1 to 3 follow the GFX10 V# layout of the public LLVM AMDGPU
    documentation and Mesa's ac_build_buffer_descriptor; the record around them is this project's own.

    load_heap_view_format sets valid to false, and returns zeros, for a slot beyond the table, a record that is not live, a zero stride, a base, stride or instruction offset that is not DWORD aligned, or an unhandled format. lower() makes an
    invalid load suppress every store, as for the constant format word. A live view whose index is not below
    num_records, whose opcode needs more than one stride, or whose bytes leave the buffer_bytes of t0, reads as zero
    and stays valid: the buffer-format unit's out-of-range result is zero (RDNA 2 ISA, buffer addressing).
    """
    return (
        "uint4 load_heap_view_format(uint slot, uint index, uint instOffset, uint channels, out bool valid) {\n"
        "    valid = false;\n"
        "    uint4 zero = uint4(0u, 0u, 0u, 0u);\n"
        f"    const uint length = {buffer_bytes}u;\n"
        f"    if (slot >= {view_count}u) return zero;\n"
        f"    uint record = {table_offset}u + slot * {HEAP_VIEW_RECORD_BYTES}u;\n"
        f"    if (inputData.Load(record + 16u) != {HEAP_VIEW_LIVE}u) return zero;\n"
        "    uint4 v = inputData.Load4(record);\n"
        "    uint stride = (v.y >> 16u) & 0x3fffu;\n"
        "    if (stride == 0u) return zero;\n"
        "    if (((v.x | stride | instOffset) & 3u) != 0u) return zero;\n"
        "    bool inRange = index < v.z && instOffset + 4u * channels <= stride && v.x <= length;\n"
        "    uint addr = 0u;\n"
        "    if (inRange) {\n"
        "        uint left = length - v.x;\n"
        "        inRange = index <= left / stride;\n"
        "        uint start = inRange ? index * stride : 0u;\n"
        "        inRange = inRange && instOffset + 4u * channels <= left - start;\n"
        "        addr = v.x + start + instOffset;\n"
        "    }\n"
        "    uint4 r = load_descriptor_format(v.w, inRange ? addr : 0u, channels, 0u, valid);\n"
        "    return inRange ? r : zero;\n"
        "}\n"
    )


def heap_store_view_hlsl(table_offset, view_count, output_bytes, *, components=4):
    """Store UINT4 or expanded binary16 components through a live side-table record."""
    width = 4 * components
    if components in (1, 2):
        sig = "void store_heap_word_d16(uint slot, uint index, uint offset, uint value)"
        format_word = descriptor_format_word(22 if components == 1 else 64)  # BUF_FMT_32_FLOAT or BUF_FMT_32_32_FLOAT
        fmt_check = f"    if (view.w != {format_word:#x}u || (view.y & 0xc000ffffu) != 0u) return;"
        if components == 1:
            write = "    outputData.Store(view.x + index * stride + offset, asuint(f16tof32(value & 0xffffu)));"
        else:
            write = ("    outputData.Store2(view.x + index * stride + offset, "
                     "asuint(f16tof32(uint2(value & 0xffffu, value >> 16u))));")
    elif components == 3:
        sig = "void store_heap_xyz_d16(uint slot, uint index, uint offset, uint2 value)"
        fmt_check = f"    if (view.w != {descriptor_format_word(74):#x}u || (view.y & 0xc000ffffu) != 0u) return;"
        write = ("    uint3 halfWords = uint3(value.x & 0xffffu, value.x >> 16u, value.y & 0xffffu);\n"
                 "    outputData.Store3(view.x + index * stride + offset, asuint(f16tof32(halfWords)));")
    else:
        sig = "void store_heap_words4(uint slot, uint index, uint offset, bool d16, uint4 value)"
        fmt_check = (f"    uint formatWord = d16 ? {descriptor_format_word(77):#x}u : {descriptor_format_word(75):#x}u;\n"
                     "    if (view.w != formatWord || (view.y & 0xc000ffffu) != 0u) return;")
        write = ("    if (d16) value = asuint(f16tof32(uint4(value.x, value.x >> 16u, value.y, value.y >> 16u)));\n"
                 "    outputData.Store4(view.x + index * stride + offset, value);")
    return (
        f"{sig} {{\n"
        f"    if (slot >= {view_count}u) return;\n"
        f"    uint record = {table_offset}u + slot * {HEAP_VIEW_RECORD_BYTES}u;\n"
        f"    if (inputData.Load(record + 16u) != {HEAP_VIEW_LIVE}u) return;\n"
        f"    uint4 view = inputData.Load4(record);\n"
        f"{fmt_check}\n"
        f"    uint stride = view.y >> 16u;\n"
        f"    if (((view.x | stride) & 3u) != 0u || stride < offset + {width}u || index >= view.z) return;\n"
        f"    const uint bytes = {output_bytes}u;\n"
        f"    if (view.x > bytes || offset + {width}u > bytes - view.x) return;\n"
        f"    if (index > (bytes - view.x - offset - {width}u) / stride) return;\n"
        f"{write}\n"
        f"}}\n"
    )


def lower(instructions, group, *, table_reg, descriptor_offset=None, group_reg=None, stride=None,
          image_uav_dimension="2d", image_srv_dimension="2d",
          image_srv_descriptor=None, constant_descriptor=None, group_reg_y=None, group_reg_z=None, source_descriptor=None,
          source_stride=None, source_format=None, output_format=None, image_descriptor=None, image_format=None,
          image_uav_mip=None, image_float_domain=None, buffer_float_domain=None, cpu="gfx1013", lds_bytes=0,
          lds_f32_mode=None, lds_f64_mode=None, mubuf_d16_xyz_mode=None,
          descriptor_format_constant=None, rsrc2=None, wave_size=None, load_format_constant=None, heap_view=None,
          heap_store_view=None, gather_mips=None):
    """v0, v1 and v2 are local X, Y and Z; group_reg, group_reg_y and group_reg_z are the group IDs.

    rsrc2 is the program's COMPUTE_PGM_RSRC2. When given, it places the group IDs
    (entry_registers) instead of group_reg, group_reg_y and group_reg_z, binds only the local IDs
    it enables, and table_reg must lie within its user SGPRs. TG_SIZE_EN is refused.

    gfx10 compute waves receive each local ID in its own VGPR. Y and Z are bound only when that
    group dimension is larger than one. Descriptors name u0 and optionally b0 and a read-only raw
    source t0; source_stride is the t0 byte stride that indexed reads use. source_format declares
    the MUBUF source descriptor format, and output_format declares the u0 descriptor format
    for formatted MUBUF stores. Both declarations require identity channel selectors.
    descriptor_format_constant lists one b0 word index per formatted MUBUF store, in program
    order. Each index must already have been read from the constant descriptor. The word is
    V# word 3. An unknown format or a channel count that does not match the opcode does not store.
    load_format_constant does the same for gfx1013 buffer_load_format_{x,xy,xyz,xyzw} through t0: one b0
    word index per formatted load, in program order, read by load_descriptor_format (see descriptor_load_hlsl).
    heap_view=(slot_word, table_offset, view_count, buffer_bytes) makes the same loads take the V# from a run-time view record
    instead of a constant format word (see heap_view_hlsl). It excludes load_format_constant.
    heap_store_view uses the same t0 record layout for one indexed UINT4, packed D16-to-FLOAT4, or
    packed D16-to-FLOAT1 store to u0.
    Its fifth argument is the writable byte extent of u0. Root UAV descriptors carry no length.
    gfx1013 uses RGBA32 FLOAT for full-mask image_store or image_load_mip. When
    both are used, image_srv_descriptor supplies the separate t0 descriptor.
    One full-mask plain image_load can read the same mip-zero UAV texel its
    lane will write. This path requires a dispatch with one Z group.
    image_srv_dimension declares that view for a full-mask 1D mip load.
    A full-mask 2D D16 image_store requires an RGBA16 FLOAT UAV.
    mubuf_d16_xyz_mode opts into a caller-verified W-preserving RGBA16 store.
    wave_size is the caller's statement that the program runs as wave32 (32). A lane index is only
    derived for a group of more than 32 threads under that statement. With group-shared LDS on gfx1013
    the statement may also be 64 (wave64), which lets a group of up to 64 threads count as one guest wave.
    image_descriptor is the table byte offset of the 8-dword image descriptor. image_format is
    the caller's statement of the image format inside it. The image binds to t0 for loads or u0
    for writes, where image_uav_mip selects the writable subresource and replaces descriptor_offset.
    image_uav_dimension declares the view for a store. The 1D, array and volume
    stores are available for gfx1013 RGBA32 FLOAT.
    R32_UINT selects the 4-sample Texture2DMS profile for an SRV or the
    non-returning 2D integer atomic profile for a UAV. R32_SINT is UAV-only
    and selects signed min/max. R32_FLOAT declares a native atomic image carried as raw words
    in a desktop R32_UINT UAV. The caller guarantees finite-normal binary32 in the accessed texels
    and operand VGPRs by setting image_float_domain="finite-normal".
    A raw fixed-address floating buffer atomic requires buffer_float_domain="finite-normal":
    the caller guarantees finite-normal binary32 or binary64 at the accessed address and in the
    operands, never NaN, subnormal or signed zero. For binary64 or compare-swap, the lowering
    further requires finite-normal literal operands.
    gather_mips declares the visible mip count of matching gfx1013 RGBA32 FLOAT native and desktop 2D views at t0
    for one image_gather4_l (see GATHER_HLSL).
    """
    dims = checked_group(group)
    if cpu not in ("gfx1013", "gfx1030"):
        raise ValueError("Unsupported compute target")
    local_axes = 3
    if rsrc2 is not None:
        if group_reg is not None or group_reg_y is not None or group_reg_z is not None:
            raise ValueError("COMPUTE_PGM_RSRC2 places the group IDs; do not also give group registers")
        user_sgprs, placed, local_axes = entry_registers(rsrc2)
        if table_reg + 1 >= user_sgprs:
            raise ValueError("The table pointer must lie within the user SGPRs")
        group_reg, group_reg_y, group_reg_z = (placed.get(axis) for axis in range(3))
    ds_group = cpu == "gfx1030" and dims == (32, 1, 1)
    if not 0 <= table_reg <= 105:
        raise ValueError("Invalid table or thread group binding")
    image_output = image_uav_mip is not None
    image_float = image_format == "R32_FLOAT"
    if (image_float and (not image_output or image_float_domain != "finite-normal")
            or (not image_float and image_float_domain is not None)):
        raise ValueError("The finite-normal image domain requires an R32_FLOAT atomic UAV and vice versa")
    if buffer_float_domain is not None and buffer_float_domain != "finite-normal":
        raise ValueError("Invalid buffer float domain")
    image_msaa = not image_output and image_format == "R32_UINT"
    if image_msaa and cpu != "gfx1030":
        raise ValueError("The R32_UINT MSAA image binding requires gfx1030")
    image_atomic_type = IMAGE_ATOMIC_TYPES.get(image_format) if image_output else None
    rgba32_float_image = cpu == "gfx1013" and image_format == RGBA32_FLOAT
    rgba16_float_image = cpu == "gfx1013" and image_output and image_format == RGBA16_FLOAT
    float_image = rgba32_float_image or rgba16_float_image
    if image_format == RGBA16_FLOAT and not rgba16_float_image:
        raise ValueError("RGBA16 FLOAT requires a gfx1013 D16 image store UAV")
    if image_uav_dimension not in IMAGE_UAV_TYPES:
        raise ValueError("Unknown image UAV dimension")
    if image_uav_dimension != "2d" and not (image_output and rgba32_float_image):
        if image_uav_dimension == "1d":
            raise ValueError("1D image UAVs require a gfx1013 RGBA32 FLOAT store binding")
        raise ValueError("Array and 3D image UAVs require a gfx1013 RGBA32 FLOAT store binding")
    if image_srv_dimension not in FLOAT_IMAGE_SRV_TYPES:
        raise ValueError("Unknown image SRV dimension")
    if image_srv_dimension == "1d" and not (rgba32_float_image and image_srv_descriptor is not None):
        raise ValueError("1D image loads require a separate gfx1013 RGBA32 FLOAT SRV binding")
    # gfx1013 takes the shared image atomic lowering on an R32_UINT UAV only
    # (add, umin, umax, cmpswap; see lower()).
    gfx1013_image_atomic = cpu == "gfx1013" and image_atomic_type is not None and image_format == "R32_UINT"
    if image_output:
        if ((cpu != "gfx1030" and not float_image and not gfx1013_image_atomic)
                or descriptor_offset is not None or stride is not None
                or image_descriptor is None
                or (image_format not in IMAGE_FORMATS and not image_atomic_type and not float_image)
                or (image_atomic_type and image_uav_mip != 0)
                or not 0 <= image_uav_mip <= 15):
            raise ValueError("Invalid image UAV binding")
        descriptor_bindings = {}
    else:
        if descriptor_offset is None or not 0 <= descriptor_offset < 1 << 20 or descriptor_offset % 4:
            raise ValueError("Invalid table or thread group binding")
        descriptor_bindings = {descriptor_offset: "uav"}
    if constant_descriptor is not None:
        if (not 0 <= constant_descriptor < 1 << 20 or constant_descriptor % 4
                or any(abs(constant_descriptor - offset) < 16 for offset in descriptor_bindings)):
            raise ValueError("Invalid or overlapping constant descriptor binding")
        descriptor_bindings[constant_descriptor] = "cbv"
    if source_descriptor is not None:
        if (not 0 <= source_descriptor < 1 << 20 or source_descriptor % 4
                or any(abs(source_descriptor - offset) < 16 for offset in descriptor_bindings)):
            raise ValueError("Invalid or overlapping source descriptor binding")
        descriptor_bindings[source_descriptor] = "srv"
    if source_stride is not None and (source_descriptor is None or not 4 <= source_stride <= 0x3fff
                                      or source_stride % 4):
        raise ValueError("Indexed reads require a source binding and aligned stride")
    source_texel_bytes = MUBUF_SOURCE_TEXELS.get(source_format)
    if source_format is not None and (source_texel_bytes is None or cpu != "gfx1030"
                                      or source_stride is None or source_stride % source_texel_bytes):
        raise ValueError("Unsupported formatted source binding")
    if image_descriptor is not None or image_format is not None:
        # The 8-dword image descriptor occupies two 16-byte slots.
        if (not (image_format in IMAGE_FORMATS or image_msaa or image_atomic_type or float_image)
                or image_descriptor is None or source_descriptor is not None
                or not 0 <= image_descriptor < 1 << 20 or image_descriptor % 4
                or any(image_descriptor + half in descriptor_bindings for half in (0, 16))
                or any(abs(image_descriptor + half - offset) < 16
                       for half in (0, 16) for offset in descriptor_bindings)):
            raise ValueError("Invalid or overlapping image descriptor binding")
        descriptor_bindings[image_descriptor] = ("image", 0)
        descriptor_bindings[image_descriptor + 16] = ("image", 1)
    if image_srv_descriptor is not None:
        if not (rgba32_float_image and image_output):
            raise ValueError("A separate image SRV requires a gfx1013 RGBA32 FLOAT image UAV")
        if (not 0 <= image_srv_descriptor < 1 << 20 or image_srv_descriptor % 4
                or any(abs(image_srv_descriptor + half - offset) < 16
                       for half in (0, 16) for offset in descriptor_bindings)):
            raise ValueError("Invalid or overlapping image SRV descriptor (the UAV must be distinct)")
        descriptor_bindings[image_srv_descriptor] = ("image_srv", 0)
        descriptor_bindings[image_srv_descriptor + 16] = ("image_srv", 1)
    if gather_mips is not None and (type(gather_mips) is not int or not 1 <= gather_mips <= 16
                                    or not rgba32_float_image or image_output or image_descriptor is None):
        raise ValueError("Gather requires a gfx1013 RGBA32 FLOAT image SRV at t0 with 1 to 16 visible mips")
    gather_steps = []
    group_regs = {axis: reg for axis, reg in enumerate((group_reg, group_reg_y, group_reg_z)) if reg is not None}
    if (any(not 0 <= reg <= 105 or reg == table_reg for reg in group_regs.values())
            or len(set(group_regs.values())) != len(group_regs)):
        raise ValueError("Invalid group register")
    if stride is not None and (not group_regs or not 4 <= stride <= 0x3fff or stride % 4):
        raise ValueError("Indexed stores require a group register and aligned stride")
    if descriptor_format_constant is not None and output_format is not None:
        raise ValueError("Descriptor format constants conflict with an output format")
    if output_format is not None and (output_format not in MUBUF_OUTPUT_FORMATS or cpu != "gfx1030"
                                      or stride is None):
        raise ValueError("Unsupported formatted output binding")
    if mubuf_d16_xyz_mode is not None and (mubuf_d16_xyz_mode != "preserve-w"
                                           or output_format != MUBUF_D16_OUTPUT_FORMATS[4] or cpu != "gfx1030"):
        raise ValueError("D16 XYZ mode requires a gfx1030 RGBA16 UINT output binding")
    format_words = None
    if descriptor_format_constant is not None:
        format_words = tuple(descriptor_format_constant)
        if (constant_descriptor is None or not format_words
                or any(not isinstance(index, int) or isinstance(index, bool)
                       or index not in range(CONSTANT_WORDS) for index in format_words)):
            raise ValueError("Invalid buffer descriptor format constant")
    load_format_words = None
    if load_format_constant is not None:
        load_format_words = tuple(load_format_constant)
        if (cpu != "gfx1013" or constant_descriptor is None or source_stride is None or source_format is not None
                or not load_format_words
                or any(not isinstance(index, int) or isinstance(index, bool)
                       or index not in range(CONSTANT_WORDS) for index in load_format_words)):
            raise ValueError("Invalid buffer load format constant")
    heap_slot_word = heap_table = heap_count = None
    view_binding = heap_view
    if heap_store_view is not None:
        if (heap_view is not None or format_words is not None or stride is not None or output_format is not None
                or image_output or not isinstance(heap_store_view, (tuple, list)) or len(heap_store_view) != 5):
            raise ValueError("Invalid heap store view binding")
        view_binding, output_bytes = heap_store_view[:4], heap_store_view[4]
        if not isinstance(output_bytes, int) or not 4 <= output_bytes < 1 << 32 or output_bytes % 4:
            raise ValueError("Invalid heap store view output size")
    if view_binding is not None:
        if (load_format_words is not None or cpu != "gfx1013" or constant_descriptor is None
                or source_descriptor is None or source_stride is not None or source_format is not None
                or not isinstance(view_binding, (tuple, list)) or len(view_binding) != 4
                or any(not isinstance(v, int) or isinstance(v, bool) for v in view_binding)):
            raise ValueError("Invalid heap view binding")
        heap_slot_word, heap_table, heap_count, heap_bytes = view_binding
        if (heap_slot_word not in range(CONSTANT_WORDS) or heap_table % 4 or not 0 <= heap_table < 1 << 20
                or not 1 <= heap_count <= HEAP_VIEW_MAX or heap_bytes % 4
                or not heap_table + HEAP_VIEW_RECORD_BYTES * heap_count <= heap_bytes < 1 << 30):
            raise ValueError("Invalid heap view binding")
    format_load_index = 0
    scalar = {table_reg: "table"}
    # Descriptor word 2 (num_records) may be read as a number only for a raw buffer, where the caller states
    # that it equals the bound view's byte size: no stride, no image output. Other fields stay opaque.
    sizes = {}
    if descriptor_offset is not None and stride is None and not image_output:
        sizes[("uav", 2)] = "uav_bytes()"
    if source_descriptor is not None and source_stride is None:
        sizes[("srv", 2)] = "srv_bytes()"
    # HLSL for SCC as set by the last s_cmp, valid only until another instruction may write SCC.
    scc = None
    for axis, reg in group_regs.items():
        scalar[reg] = ("group", axis)
    vector = {axis: ("local", axis) for axis in range(local_axes) if axis == 0 or dims[axis] > 1}
    pending = set()
    # Pending VMEM results, with the word they receive (None when unknown);
    # they enter vector only at s_waitcnt vmcnt(0).
    loading = {}
    stores = []
    heap_store_components = 4
    format_store_index = 0
    atomic_update = False
    buffer_float_update = False
    shared_atomic_offset = None
    # DS operations and image loads can consume each other's temporaries. Keep them in instruction order.
    staged_steps = []
    ds_step_count = 0
    image_step_count = 0
    image_uav_address = None
    lds_enabled = ds_group and 128 <= lds_bytes <= 65536
    # gfx1013 has no private LDS profile; an allocation there selects group-shared LDS.
    shared_lds = cpu == "gfx1013" and lds_bytes != 0
    if shared_lds and not (4 <= lds_bytes <= LDS_SHARED_LIMIT and lds_bytes % 4 == 0):
        raise ValueError("Invalid shared LDS allocation")
    # wave64 is stated only for the one-wave shared LDS profile; elsewhere it is not a known lane layout.
    if wave_size not in (None, 32) and not (wave_size == 64 and shared_lds):
        raise ValueError("Unsupported wave size statement")
    # Per-instruction group barriers give the group one wave's DS order. With several guest waves the order
    # before s_barrier is not fixed by the program, so the profile is one wave: at most 32 threads is one wave
    # of either width, and a larger group needs the wave64 statement.
    if shared_lds and dims[0] * dims[1] * dims[2] > (wave_size or 32):
        raise ValueError("Shared LDS requires a group of one guest wave")
    lds_word = None
    lds_strided_words = {}
    lds_strided_width = None
    lds_wide_words = {}
    lds_address_shift = None
    lds_m0_zero = False
    ds_wait = False
    dpp_step_count = 0
    readlane_step_count = 0
    written_spans = []
    layout = None
    # One per-lane guard: v_cmp into vcc_lo, s_and_saveexec_b32, the stores under the mask, then
    # s_or_b32 restoring exec or the final exit. An optional s_cbranch_execz must land exactly
    # where the guarded stores end. The guarded stores become an HLSL if.
    vcc = guard = None
    vcc_carry = None
    vcc_cond = None
    guard_span = None
    if not instructions or instructions[-1] != "s_endpgm":
        raise ValueError("Missing final program exit")

    for line in instructions[:-1]:
        if "vcc_hi" in line and wave_size == 32:
            # A wave32 VCC is vcc_lo alone and every lowered write of it (compares, carries, saveexec) touches only
            # that half, so vcc_hi keeps what s_mov_b32 wrote. It is tracked as the otherwise unused SGPR 107,
            # which only s_mov_b32 may write and only a known-word read may use. Without the wave32 statement the
            # name stays unrecognised (wave64 VCC is the pair).
            line = re.sub(r"\bvcc_hi\b", "s107", line)
        elif re.search(r"\bs107\b", line):
            raise ValueError("Invalid scalar destination or pending load")
        if ds_wait:
            if line != "s_waitcnt lgkmcnt(0)":
                raise ValueError("Expected immediate s_waitcnt lgkmcnt(0) after DS")
            ds_wait = False
        if m := re.fullmatch(r"(\.L\w+):", line):
            if guard is None or guard["label"] != m[1] or guard["at_label"]:
                raise ValueError("Unverified branch target")
            guard["at_label"] = True
            continue
        if (guard is not None and guard["at_label"]
                and not re.fullmatch(r"s_waitcnt_depctr 0x[0-9a-f]+|s_nop \d+|s_or_b32 .*", line)):
            raise ValueError("Lane guard branch does not reach its restore")
        if m := re.fullmatch(r"s_setreg_imm32_b32 hwreg\(HW_REG_FLAT_SCR_(?:LO|HI)\), (.+)", line):
            if not _IMM32_LITERAL.fullmatch(m[1]):
                raise ValueError("Unsupported scratch-base immediate spelling: " + m[1])
            # A whole-register immediate write of the scratch base. Nothing the lowering accepts can read
            # either half: a read through s_getreg_b32 and every scratch or flat memory form are themselves
            # unmodelled and refused, so the write has no observable consumer and lowers to nothing. Its value
            # is not tracked, which is why only this exact whole-register immediate form is accepted.
            continue
        if re.fullmatch(r"s_inst_prefetch (?:0x[0-9a-f]+|\d+)|s_nop \d+|s_waitcnt_depctr 0x[0-9a-f]+", line):
            continue
        if m := re.fullmatch(r"s_clause (0x[0-9a-f]+|\d+)", line):
            if int(m[1], 0) > 0x3f:
                raise ValueError("Invalid s_clause operand")
            continue
        if not line.startswith(SCC_PRESERVING):
            scc = None
        if not line.startswith("s_waitcnt") and vector_operands(line) & loading.keys():
            raise ValueError("Vector load result used or overwritten before s_waitcnt vmcnt(0)")
        if line in ("s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(0) vmcnt(0)", "s_waitcnt vmcnt(0) lgkmcnt(0)",
                    "s_waitcnt vmcnt(0)"):
            if "lgkmcnt" in line:
                pending.clear()
            if "vmcnt" in line:
                for reg, value in loading.items():
                    if value is None:
                        vector.pop(reg, None)
                    else:
                        vector[reg] = value
                loading.clear()
        elif image_output and stores and line == "s_waitcnt_vscnt null, 0x0":
            # The single image write becomes visible when the dispatch completes.
            continue
        elif ((m := re.fullmatch(rf"v_cmp_(f|lt|eq|le|gt|ne|ge|t)_(u32|i32)_(e32) (vcc_lo), {ALU_SOURCE}, (v\d+)", line))
              or (m := re.fullmatch(rf"v_cmp_(f|lt|eq|le|gt|ne|ge|t)_(u32|i32)_(e64) (vcc_lo|s\d+), {ALU_SOURCE}, {ALU_SOURCE}", line))):
            # RDNA2 ISA: VOPC sets SDST[lane] = pred(S0, S1) for each active lane.
            pred, typ, enc, sdst, src0, src1 = m.groups()
            operands = (src0, src1)
            if sdst != "vcc_lo":
                sdst_reg = int(sdst[1:])
                if sdst_reg > 105 or sdst_reg in pending:
                    raise ValueError("Unverified compare destination")
            words = [alu_word(token, scalar, vector, pending, sizes) for token in operands]
            if any(token.startswith("v") and int(token[1:]) > 255 for token in operands) or None in words:
                raise ValueError("Unverified lane guard compare")
            a, b = map(hlsl_word, words)
            if typ == "i32":
                a, b = f"asint({a})", f"asint({b})"
            cond_expr = INT_COMPARE_PREDICATES[pred](a, b)
            if len(cond_expr) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            guard_ok = (
                sdst == "vcc_lo"
                and ((typ == "u32" and enc == "e32" and pred in ("gt", "lt", "ge", "le"))
                     or (typ == "i32" and pred == "ne"))
                and not any(isinstance(word, tuple) and word[0] in ("source", "alu") for word in words)
            )
            if guard is not None and not guard_ok:
                raise ValueError("Unverified lane guard compare")
            if sdst == "vcc_lo":
                vcc = (pred, *words) if guard_ok else ("invalid",)
                vcc_cond = ("cond", cond_expr)
                vcc_carry = None
            else:
                scalar[sdst_reg] = ("cond", cond_expr)
        elif ((m := re.fullmatch(rf"v_cmp_(f|lt|eq|le|gt|lg|ge|o|u|nge|nlg|ngt|nle|neq|nlt|tru)_f32_(e32) (vcc_lo), {FLOAT_SOURCE}, (v\d+)", line))
              or (m := re.fullmatch(rf"v_cmp_(f|lt|eq|le|gt|lg|ge|o|u|nge|nlg|ngt|nle|neq|nlt|tru)_f32_(e64) (vcc_lo|s\d+), {FLOAT_SOURCE}, {FLOAT_SOURCE}", line))):
            # RDNA2 ISA: VOPC f32 compare sets SDST[lane] = pred(S0, S1) with IEEE unordered NaN rules.
            pred, enc, sdst, src0, src1 = m.groups()
            operands = (src0, src1)
            if guard is not None:
                raise ValueError("Unverified lane guard compare")
            if sdst != "vcc_lo":
                sdst_reg = int(sdst[1:])
                if sdst_reg > 105 or sdst_reg in pending:
                    raise ValueError("Unverified compare destination")
            words = [FLOAT_INLINE.get(token) if "." in token else alu_word(token, scalar, vector, pending)
                     for token in operands]
            if any(token.startswith("v") and int(token[1:]) > 255 for token in operands) or None in words:
                raise ValueError("Unverified float operand")
            fa, fb = (f"asfloat({hlsl_word(w)})" for w in words)
            cond_expr = FLOAT_COMPARE_PREDICATES[pred](fa, fb)
            if len(cond_expr) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            if sdst == "vcc_lo":
                vcc = ("invalid",)
                vcc_cond = ("cond", cond_expr)
                vcc_carry = None
            else:
                scalar[sdst_reg] = ("cond", cond_expr)
        elif m := re.fullmatch(r"s_and_saveexec_b32 s(\d+), vcc_lo", line):
            if cpu == "gfx1013" and atomic_update:
                raise ValueError("gfx1013 buffer atomics require an unguarded program")
            saved = int(m[1])
            if vcc == ("invalid",) and guard_span is None and saved <= 105 and saved not in pending:
                raise ValueError("Unverified lane guard compare")
            if guard_span is not None or vcc is None or saved > 105 or saved in pending:
                raise ValueError("Unverified or repeated lane guard")
            guard = dict(label=None, at_label=False, holders=None, vcc_carry=vcc_carry, vcc_cond=vcc_cond,
                         scalar=dict(scalar), vector=dict(vector), pending=set(pending), loading=dict(loading))
            scalar[saved] = ("exec",)
            guard_span = [len(stores), None, vcc]
        elif m := re.fullmatch(r"s_cbranch_execz (\.L\w+)", line):
            # Taken only when no lane passed the guard; it must skip exactly the guarded stores.
            if guard is None:
                raise ValueError("Unverified lane guard branch")
            guard["label"] = m[1]
            # A taken branch skips the block: only an SGPR that holds the mask here holds it on both paths.
            holders = {reg for reg, value in scalar.items() if value == ("exec",)}
            guard["holders"] = holders if guard["holders"] is None else guard["holders"] & holders
        elif m := re.fullmatch(r"s_or_b32 exec_lo, (?:exec_lo, s(\d+)|s(\d+), exec_lo)", line):
            # Any SGPR that still holds the saved mask restores it, on the skipped path too.
            reg = int(m[1] or m[2])
            if (guard is None or scalar.get(reg) != ("exec",)
                    or guard["label"] is not None and (not guard["at_label"] or reg not in guard["holders"])):
                raise ValueError("Unverified lane guard restore")
            # Registers written under the guard keep their old values in the lanes that did not
            # pass it, or in the whole wave when the branch skips the block: unknown after the
            # restore. A wait under the guard may have been skipped, so earlier loads stay pending.
            # A load from before the guard keeps its word only if its register still holds that
            # word on the guarded path, waited for or not. Any other loaded register, including a
            # load issued under the guard (it writes only the lanes that passed), is unknown after
            # its wait.
            loading = {reg: guard["loading"].get(reg) if loading.get(reg, vector.get(reg)) == guard["loading"].get(reg)
                       else None for reg in guard["loading"].keys() | loading.keys()}
            for state, before in ((scalar, guard["scalar"]), (vector, guard["vector"])):
                for reg in [reg for reg, value in state.items() if reg not in before or before[reg] != value]:
                    del state[reg]
            pending |= guard["pending"]
            if vcc_carry != guard["vcc_carry"]:
                vcc_carry = None
            if vcc_cond != guard["vcc_cond"]:
                vcc_cond = None
            guard = None
            guard_span[1] = len(stores)
        elif line in ("s_mov_b32 m0, 0", "s_mov_b32 m0, 0x0"):
            if not lds_enabled or guard_span is not None:
                raise ValueError("Unverified private LDS thread-ID base")
            lds_m0_zero = True
        elif m := re.fullmatch(r"s_mov_b32 s(\d+), (s\d+|-?\d+|0x[0-9a-f]+)", line):
            dst, src = int(m[1]), m[2]
            if dst in pending or (dst > 105 and dst != VCC_HI):
                raise ValueError("Invalid scalar destination or pending load")
            if src.startswith("s"):
                n = int(src[1:])
                if (n > 105 and n != VCC_HI) or n in pending or n not in scalar:
                    raise ValueError("Unknown or pending scalar source")
                scalar[dst] = scalar[n]
            else:
                value = int(src, 0)
                if not -(1 << 31) <= value <= 0xffffffff:
                    raise ValueError("Invalid scalar immediate")
                scalar[dst] = value & 0xffffffff
        elif m := (
            re.fullmatch(rf"(s_(?:add_u32|add_i32|sub_u32|sub_i32|and_b32|or_b32|xor_b32|lshr_b32|absdiff_i32|bfe_u32|bfe_i32)) s(\d+), {SALU_SOURCE}, {SALU_SOURCE}",
                         line)
            or re.fullmatch(rf"(s_abs_i32) s(\d+), {SALU_SOURCE}", line)
        ):
            op, dst = m[1], int(m[2])
            words = [alu_word(token, scalar, vector, pending, sizes) for token in m.groups()[2:]]
            if dst > 105 or dst in pending or None in words:
                raise ValueError("Unverified scalar ALU operand")
            expression = SALU[op].format(*map(hlsl_word, words))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            scalar[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"(s_bitset[01]_b32) s(\d+), {SALU_SOURCE}", line):
            # SOP1 s_bitset0_b32 and s_bitset1_b32 clear or set bit src0[4:0] of sD and keep its other bits, so sD
            # is read as well as written: it must already hold a known word. SCC is not written.
            op, dst = m[1], int(m[2])
            words = [alu_word(f"s{dst}", scalar, vector, pending, sizes), alu_word(m[3], scalar, vector, pending, sizes)]
            if dst > 105 or dst in pending or None in words:
                raise ValueError("Unverified scalar ALU operand")
            expression = SALU[op].format(*map(hlsl_word, words))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            scalar[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"s_cmp_(eq|lg)_([iu])32 {SALU_SOURCE}, {SALU_SOURCE}", line):
            # SCC = S0 == S1 (eq) or S0 != S1 (lg); the signedness does not change a 32-bit equality.
            words = [alu_word(token, scalar, vector, pending, sizes) for token in m.groups()[2:]]
            if None in words:
                raise ValueError("Unverified scalar compare operand")
            scc = f"({hlsl_word(words[0])} {'==' if m[1] == 'eq' else '!='} {hlsl_word(words[1])})"
            if len(scc) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
        elif m := re.fullmatch(rf"s_cselect_b32 s(\d+), {SALU_SOURCE}, {SALU_SOURCE}", line):
            # SOP2: D = SCC ? S0 : S1. SCC must come from an s_cmp with only SCC-preserving code between.
            dst = int(m[1])
            words = [alu_word(token, scalar, vector, pending, sizes) for token in m.groups()[1:]]
            if dst > 105 or dst in pending or None in words:
                raise ValueError("Unverified scalar select operand")
            if scc is None:
                raise ValueError("Unverified SCC for scalar select")
            expression = f"({scc} ? {hlsl_word(words[0])} : {hlsl_word(words[1])})"
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            scalar[dst] = ("alu", expression)
        elif m := re.fullmatch(r"s_movk_i32 s(\d+), (-?\d+|0x[0-9a-f]+)", line):
            dst = int(m[1])
            value = int(m[2], 0)
            if dst in pending or dst > 105 or not -0x8000 <= value <= 0xffff:
                raise ValueError("Invalid scalar destination or movk immediate")
            signed = value - 0x10000 if value >= 0x8000 else value
            scalar[dst] = signed & 0xffffffff
        elif m := re.fullmatch(r"s_bcnt1_i32_b32 s(\d+), (\S+)", line):
            # Active lane count of a full wave: sD = popcount(exec_lo), SCC = sD != 0 (SCC is unread in this
            # subset). Outside a guarded region exec covers the whole wave: at most 32 threads is one wave of
            # either size, larger groups need wave_size=32 and a multiple of 32 threads.
            dst, src = int(m[1]), m[2]
            if dst in pending or dst > 105:
                raise ValueError("Invalid scalar destination or pending load")
            if src != "exec_lo":
                # Any other source is a tracked scalar word (a loaded constant, a move or ALU result, a literal or
                # wave32 vcc_hi): D = CountOneBits(S0) as for the 64-bit form, SCC unread. A source that is not
                # a known word (descriptor word, untracked SGPR, exec or vcc halves) stays refused.
                if not re.fullmatch(r"s\d+|-?\d+|0x[0-9a-f]+", src):
                    raise ValueError("Unsupported active lane count source: " + src)
                word = alu_word(src, scalar, vector, pending)
                if word is None:
                    raise ValueError("Unsupported active lane count source: " + src)
                if isinstance(word, int):
                    scalar[dst] = word.bit_count()
                else:
                    expression = f"countbits({hlsl_word(word)})"
                    if len(expression) > ALU_EXPRESSION_LIMIT:
                        raise ValueError("ALU expression too large")
                    scalar[dst] = ("alu", expression)
                continue
            if guard is not None:
                raise ValueError("Active lane count inside guarded region")
            threads = dims[0] * dims[1] * dims[2]
            if threads <= 32:
                scalar[dst] = threads
            elif wave_size == 32 and threads % 32 == 0:
                scalar[dst] = 32
            else:
                raise ValueError("Unverified active lane count wave size or group size")
        elif m := re.fullmatch(r"s_bcnt1_i32_b64 s(\d+), s\[(\d+):(\d+)\]", line):
            # SOP1 counts both words and sets SCC to D != 0. The profile has no SCC readers.
            dst, lo, hi = map(int, m.groups())
            if dst > 105 or dst in pending or lo % 2 or hi != lo + 1 or hi > 105:
                raise ValueError("Invalid scalar bit count register or pending destination")
            words = [alu_word(f"s{reg}", scalar, vector, pending, sizes) for reg in (lo, hi)]
            if None in words:
                raise ValueError("Unverified scalar bit count source")
            if all(isinstance(word, int) for word in words):
                scalar[dst] = sum(word.bit_count() for word in words)
            else:
                expression = "(countbits({}) + countbits({}))".format(*map(hlsl_word, words))
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("ALU expression too large")
                scalar[dst] = ("alu", expression)
        elif m := re.fullmatch(r"s_ff1_i32_(b32|b64) s(\d+), (\S+)", line):
            # exec/exec_lo: first active lane of a full wave. SCC is preserved. Outside a guarded
            # region lane 0 of every wave is active for any group of at least 1 thread, and for
            # wave32 the high half of exec is 0, so the result folds to 0.
            # Any other source must be a tracked data word (the same words scalar ALU accepts). The
            # result is the index of the lowest set bit, or 0xffffffff when the source is 0. b64
            # scans both halves of an even SGPR pair. SCC stays unmodelled.
            width, dst, src = m[1], int(m[2]), m[3]
            expected_src = "exec_lo" if width == "b32" else "exec"
            expression = None
            if src != expected_src:
                if width == "b32" and re.fullmatch(r"s\d+", src):
                    word = alu_word(src, scalar, vector, pending, sizes)
                    if word is not None:
                        expression = f"firstbitlow({hlsl_word(word)})"
                elif width == "b64" and (pair := re.fullmatch(r"s\[(\d+):(\d+)\]", src)):
                    lo, hi = int(pair[1]), int(pair[2])
                    low = scalar.get(lo) if lo <= 105 and lo not in pending else None
                    high = scalar.get(hi) if hi <= 105 and hi not in pending else None
                    if lo % 2 == 0 and hi == lo + 1 and hi <= 105 and is_word(low) and is_word(high):
                        lo_text, hi_text = hlsl_word(low), hlsl_word(high)
                        expression = (f"({lo_text} != 0u ? firstbitlow({lo_text}) : "
                                      f"({hi_text} != 0u ? 32u + firstbitlow({hi_text}) : 0xffffffffu))")
                if expression is None:
                    raise ValueError("Unsupported first active lane source: " + src)
            if dst in pending or dst > 105:
                raise ValueError("Invalid scalar destination or pending load")
            if src == expected_src:
                if guard is not None:
                    raise ValueError("First active lane inside guarded region")
                scalar[dst] = 0
            else:
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("ALU expression too large")
                scalar[dst] = ("alu", expression)
        elif m := re.fullmatch(r"s_bitreplicate_b64_b32 s\[(\d+):(\d+)\], (\S+)", line):
            # Each source bit fills two adjacent bits of the pair: D[2i] = D[2i + 1] = S0[i]. With an immediate
            # source both halves are known words; the low one replicates bits 0-15 and the high one bits 16-31.
            dst, high, src = int(m[1]), int(m[2]), m[3]
            if dst % 2 or high != dst + 1 or high > 105 or {dst, high} & pending:
                raise ValueError("Invalid lane mask destination")
            if not re.fullmatch(r"-?\d+|0x[0-9a-f]+", src) or not -(1 << 31) <= int(src, 0) <= 0xffffffff:
                raise ValueError("Unverified lane mask source")
            pair = sum(((int(src, 0) >> (bit // 2)) & 1) << bit for bit in range(64))
            scalar[dst], scalar[high] = pair & 0xffffffff, pair >> 32
        elif ((m := re.fullmatch(rf"v_cndmask_b32_e32 v(\d+), {ALU_SOURCE}, (v\d+), (vcc_lo)", line))
              or (m := re.fullmatch(rf"v_cndmask_b32_e64 v(\d+), {ALU_SOURCE}, {ALU_SOURCE}, (vcc_lo|s\d+)", line))):
            # D = mask[lane] ? src1 : src0, either from a per-lane ("cond", ...) or from a statically known
            # wave32 constant mask in an SGPR. For a constant mask, a one-dimensional group of at most 32
            # threads is one wave at either wave size and the lane is local X; past lane 31 the same encoding
            # reads the high mask word in wave64, which the translator cannot tell from the code.
            dst, src0, src1, mask_tok = int(m[1]), m[2], m[3], m[4]
            if dst > 255:
                raise ValueError("Invalid vector destination")
            words = [alu_word(token, scalar, vector, pending, sizes) for token in (src0, src1)]
            if None in words:
                raise ValueError("Unverified ALU operand")
            if mask_tok == "vcc_lo":
                mask = vcc_cond
            else:
                reg = int(mask_tok[1:])
                mask = scalar.get(reg) if reg <= 105 and reg not in pending else None
            if isinstance(mask, tuple) and mask[0] == "cond":
                expression = f"({mask[1]} ? {hlsl_word(words[1])} : {hlsl_word(words[0])})"
            elif isinstance(mask, int):
                if dims[1:] != (1, 1) or dims[0] > 32:
                    raise ValueError("Lane mask select requires a one-dimensional group of at most 32 threads")
                expression = (f"(((0x{mask:08x}u >> (tid.x % {dims[0]}u)) & 1u) != 0u ? {hlsl_word(words[1])}"
                              f" : {hlsl_word(words[0])})")
            else:
                raise ValueError("Unverified lane mask")
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(r"s_lshl_b32 s(\d+), s(\d+), (\d+)", line):
            dst, group, shift = map(int, m.groups())
            tag = scalar.get(group)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group" else None
            # A group's first dispatch index, group * size, only for a power-of-two size.
            if (dst in pending or dst > 105 or axis is None or dims[axis] & (dims[axis] - 1)
                    or shift != dims[axis].bit_length() - 1):
                raise ValueError("Unverified group base")
            scalar[dst] = ("group_base", axis)
        elif m := re.fullmatch(r"s_load_dwordx([48]) s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line):
            # x8 loads two descriptors bound at consecutive 16-byte slots of the table.
            count = int(m[1])
            dest, last, base, high = map(int, m.groups()[1:5])
            offset = 0 if m[6] == "null" else int(m[6], 16)
            regs = range(dest, last + 1)
            slots = range(offset, offset + 4 * count, 16)
            if (dest % 4 or last != dest + count - 1 or last > 105 or base % 2 or high != base + 1
                    or high > 105 or pending.intersection((*regs, base, high))
                    or (scalar.get(base), scalar.get(high)) != ("table", 0)
                    or any(slot not in descriptor_bindings for slot in slots)):
                raise ValueError("Unresolved compute descriptor load")
            for word, reg in enumerate(regs):
                scalar[reg] = (descriptor_bindings[slots[word // 4]], word % 4)
            pending.update(regs)
        elif m := re.fullmatch(r"(?:s_buffer_load_dwordx([24]) s\[(\d+):(\d+)\]|s_buffer_load_dword s(\d+)), "
                               r"s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line):
            count = int(m[1] or 1)
            dest, last = (int(m[4]), int(m[4])) if m[4] else (int(m[2]), int(m[3]))
            base, high = int(m[5]), int(m[6])
            offset = 0 if m[7] == "null" else int(m[7], 16)
            regs = range(dest, last + 1)
            if (dest % count or last != dest + count - 1 or last > 105
                    or base % 4 or high != base + 3 or high > 105
                    or offset % 4 or offset // 4 + count > CONSTANT_WORDS
                    or pending.intersection((*regs, *range(base, high + 1)))
                    or any(scalar.get(base + i) != ("cbv", i) for i in range(4))):
                raise ValueError("Unresolved or out-of-range compute constant read")
            for word, reg in enumerate(regs, offset // 4):
                scalar[reg] = ("constant_word", word)
            pending.update(regs)
        elif m := re.fullmatch(r"v_readlane_b32 s(\d+), v(\d+), (s\d+|-?\d+|0x[0-9a-f]+)", line):
            # A wave-wide SGPR read is emulated from logical 1D wave32 groupshared storage. Dynamic lane
            # selectors and incomplete logical waves are refused so every selected source is defined.
            dst, src = int(m[1]), int(m[2])
            lane = alu_word(m[3], scalar, vector, pending, sizes)
            threads = dims[0] * dims[1] * dims[2]
            if (dst > 105 or dst in pending or src > 255 or guard_span is not None
                    or dims[1:] != (1, 1) or not isinstance(lane, int) or not 0 <= lane < 32
                    or (threads <= 32 and lane >= threads)
                    or (threads > 32 and (wave_size != 32 or threads % 32))):
                raise ValueError("Unverified readlane operand or wave layout")
            value = vector.get(src)
            if not is_word(value):
                raise ValueError("Unverified readlane source")
            step = readlane_step_count
            readlane_step_count += 1
            slot, result = f"readlane_stage_{step}", f"readlane_value_{step}"
            staged_steps.append((slot, f"    {slot}[gindex] = {hlsl_word(value)};\n"
                                 "    GroupMemoryBarrierWithGroupSync();\n"
                                 f"    uint {result} = {slot}[(gindex & ~31u) + {lane}u];"))
            scalar[dst] = ("alu", result)
        elif m := re.fullmatch(rf"v_mbcnt_(lo|hi)_u32_b32 v(\d+), (exec_lo|s\d+|-?\d+|0x[0-9a-f]+), {ALU_SOURCE}", line):
            # D = popcount(S0 & lanes below this one, in the low or high half of the mask) + S1. A wave32 lane
            # has no bit in the high half, so v_mbcnt_hi adds nothing. gfx10 numbers lanes by the flat local
            # thread index (SV_GroupIndex) modulo the wave size, which is fixed only by the program, not by the
            # desktop wave width: a group of at most 32 threads is one wave of either size, a larger group needs
            # the wave32 statement. The mask must be uniform (a scalar word); exec is not read.
            half, dst = m[1], int(m[2])
            # exec_lo as the mask (the per-lane rank among active lanes) while the wave is full: every lane
            # below the current one is active, so the mask is all ones. Inside an open guard exec is the
            # lane compare and is refused; after the restore it is full again and accepted.
            if m[3] == "exec_lo":
                if guard is not None:
                    raise ValueError("Active lane mask inside guarded region")
                mask = 0xffffffff
            else:
                mask = alu_word(m[3], scalar, vector, pending, sizes)
            base = alu_word(m[4], scalar, vector, pending, sizes)
            if (dst > 255 or mask is None or base is None
                    or (dims[0] * dims[1] * dims[2] > 32 and wave_size != 32)):
                raise ValueError("Unverified lane count operand or wave size")
            if half == "hi":
                vector[dst] = base
            else:
                expression = f"(countbits({hlsl_word(mask)} & ((1u << (gindex & 31u)) - 1u)) + {hlsl_word(base)})"
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("ALU expression too large")
                vector[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"v_writelane_b32 v(\d+), {SALU_SOURCE}, {SALU_SOURCE}", line):
            # D[lane & 31] = S, leaving other lanes unchanged; exec is ignored. gfx10 numbers lanes by
            # SV_GroupIndex modulo the wave size: at most 32 threads is one wave of either size, a larger
            # group requires the wave32 statement.
            dst = int(m[1])
            src = alu_word(m[2], scalar, vector, pending, sizes)
            lane = alu_word(m[3], scalar, vector, pending, sizes)
            old = vector.get(dst)
            if dst > 255:
                raise ValueError("Invalid writelane vector destination")
            if guard is not None:
                raise ValueError("v_writelane_b32 inside guarded region")
            if dims[0] * dims[1] * dims[2] > 32 and wave_size != 32:
                raise ValueError("Unverified writelane wave size")
            if src is None:
                raise ValueError("Unverified writelane scalar source")
            if lane is None:
                raise ValueError("Unverified writelane lane")
            if wave_size != 32 and (not isinstance(lane, int) or not 0 <= lane <= 31):
                raise ValueError("Unverified writelane lane")
            if not is_word(old):
                raise ValueError("Unverified writelane old value")
            lane_hlsl = f"({hlsl_word(lane)} & 31u)" if wave_size == 32 else hlsl_word(lane)
            expression = f"(((gindex & 31u) == {lane_hlsl}) ? {hlsl_word(src)} : {hlsl_word(old)})"
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(r"v_mov_b32(?:_e32|_e64)? v(\d+), (s\d+|v\d+|-?\d+|0x[0-9a-f]+|-?\d+\.\d+)", line):
            dst, src = int(m[1]), m[2]
            if dst > 255:
                raise ValueError("Invalid vector destination")
            if src.startswith("s"):
                n = int(src[1:])
                val = scalar.get(n)
                if n > 105 or n in pending or not is_word(val):
                    raise ValueError("Unresolved scalar move source")
                vector[dst] = val
            elif src.startswith("v"):
                n = int(src[1:])
                val = vector.get(n)
                if n > 255 or not is_word(val):
                    raise ValueError("Unresolved vector move source")
                vector[dst] = val
            elif "." in src:
                # LLVM prints an inline float constant of a b32 move as a float (1.0 for 0x3f800000).
                if src not in FLOAT_INLINE:
                    raise ValueError("Invalid literal move")
                vector[dst] = FLOAT_INLINE[src]
            else:
                value = int(src, 0)
                if not -(1 << 31) <= value <= 0xffffffff:
                    raise ValueError("Invalid literal move")
                vector[dst] = value & 0xffffffff
        elif m := re.fullmatch(r"v_lshl_add_u32 v(\d+), s(\d+), (\d+), v(\d+)", line):
            dst, group, shift, local = map(int, m.groups())
            tag = scalar.get(group)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group" else None
            # The dispatch-thread ID is group * size + local only for a power-of-two size.
            if (dst > 255 or group in pending or axis is None or dims[axis] & (dims[axis] - 1)
                    or shift != dims[axis].bit_length() - 1 or vector.get(local) != ("local", axis)):
                raise ValueError("Unverified dispatch index")
            vector[dst] = ("thread", axis)
        elif m := re.fullmatch(r"v_add_lshl_u32 v(\d+), s(\d+), v(\d+), (\d+)", line):
            dst, base, local, shift = map(int, m.groups())
            tag = scalar.get(base)
            # (group base + local ID of the same axis) << shift: the byte offset of a dispatch
            # index, as from v_lshlrev_b32. The store checks bound the shift.
            if (dst > 255 or not (isinstance(tag, tuple) and tag[0] == "group_base")
                    or vector.get(local) != ("local", tag[1])):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("thread_bytes", tag[1], shift)
        elif (ds_group and (m := re.fullmatch(r"v_add_nc_u32_e32 v(\d+), (\d+), v(\d+)", line))
              and vector.get(int(m[3])) == ("local", 0)):
            dst, rotation, _ = map(int, m.groups())
            if dst > 255 or rotation > 31:
                raise ValueError("Unverified DS lane bias")
            vector[dst] = ("ds_biased_lane", rotation)
        elif (ds_group and (m := re.fullmatch(r"v_and_b32_e32 v(\d+), 31, v(\d+)", line))
              and (tag := vector.get(int(m[2]))) and isinstance(tag, tuple) and tag[0] == "ds_biased_lane"):
            dst = int(m[1])
            if dst > 255:
                raise ValueError("Invalid vector destination")
            vector[dst] = ("ds_lane", tag[1])
        elif ((m := re.fullmatch(r"v_lshlrev_b32(?:_e32|_e64)? v(\d+), (\d+), v(\d+)", line))
              and not alu_data_word(vector.get(int(m[3])))):
            dst, shift, src = map(int, m.groups())
            tag = vector.get(src)
            if shared_lds and dst <= 255 and shift <= 31 and isinstance(tag, tuple) and tag[0] == "local":
                # A shared LDS byte address from the local ID; it is data only to the DS instructions below.
                vector[dst] = ("alu", f"(local.{AXES[tag[1]]} << {shift}u)")
                continue
            if ds_group and dst <= 255 and shift == 2 and isinstance(tag, tuple) and tag[0] == "ds_lane":
                vector[dst] = ("ds_address", tag[1])
                continue
            if (lds_enabled and dst <= 255 and tag == ("local", 0)
                    and (shift == 2 or (shift == 4 and lds_bytes >= 512))):
                if lds_address_shift is not None and shift != lds_address_shift:
                    raise ValueError("Mixed private LDS address strides")
                lds_address_shift = shift
                vector[dst] = ("lds_address",) if shift == 2 else ("lds_wide_address",)
                continue
            if lds_enabled and dst <= 255 and shift == 3 and tag == ("local", 0):
                vector[dst] = ("lds_strided_qword_address",)
                continue
            # The element and row checks at the store bound the shift.
            if dst > 255 or not (isinstance(tag, tuple) and tag[0] == "thread"):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("thread_bytes", tag[1], shift)
        elif shared_lds and (line == "s_barrier" or line.startswith("ds_")):
            # Group-shared LDS (issue #1541). Each DS instruction becomes HLSL statements on `lds`, followed by a
            # group barrier, so the group observes DS operations in program order, as one wave does; s_barrier
            # adds nothing beyond that and becomes the same barrier. Barriers need uniform control flow, so no
            # DS instruction may follow a lane guard.
            if guard_span is not None or guard is not None:
                raise ValueError("Shared LDS access under a lane guard")
            if line == "s_barrier":
                if not staged_steps or not staged_steps[-1][1].endswith(LDS_BARRIER):
                    staged_steps.append((None, LDS_BARRIER))
                continue
            text = shared_lds_step(line, ds_step_count, vector, lds_bytes // 4)
            ds_step_count += 1
            staged_steps.append((None, text + "\n" + LDS_BARRIER))
            ds_wait = True
        elif m := re.fullmatch(r"ds_(b?permute)_b32 v(\d+), v(\d+), v(\d+)", line):
            op, dst, addr, src = m[1], *map(int, m.groups()[1:])
            address = vector.get(addr)
            value = vector.get(src)
            if (not ds_group or guard_span is not None or max(dst, addr, src) > 255
                    or not isinstance(address, tuple) or address[0] != "ds_address" or not is_word(value)
                    or ds_step_count >= 64):
                raise ValueError("Unverified DS lane routing")
            # The bijective address is (local X + bias) mod 32. Scatter reads from its inverse.
            bias = address[1] if op == "bpermute" else (-address[1]) % 32
            slot = f"ds_stage_{ds_step_count}"
            result = f"ds_result_{ds_step_count}"
            ds_step_count += 1
            staged_steps.append((slot, f"    {slot}[local.x] = {hlsl_word(value)};\n"
                                   f"    GroupMemoryBarrierWithGroupSync();\n"
                                   f"    uint {result} = {slot}[(local.x + {bias}u) & 31u];"))
            vector[dst] = ("alu", result)
            ds_wait = True
        elif m := re.fullmatch(r"ds_((?:cmpst|mskor)(?:_rtn)?|wrxchg_rtn)_b32 "
                               r"(v\d+(?:, v\d+){1,3})(?: offset:0)?", line):
            op = m[1]
            returns = op.endswith("_rtn")
            exchange = op == "wrxchg_rtn"
            regs = [int(token[1:]) for token in m[2].split(", ")]
            if len(regs) != (2 if exchange else 3) + returns:
                raise ValueError("Unverified private LDS replacement operands")
            addr, first, *second = regs[returns:]
            data0 = vector.get(first)
            data1 = vector.get(second[0]) if second else None
            if (not lds_enabled or guard_span is not None or max(regs) > 255
                    or vector.get(addr) != ("lds_address",) or lds_word is None or not is_word(data0)
                    or (not exchange and not is_word(data1))):
                raise ValueError("Unverified private LDS replacement")
            if exchange:
                new_word = data0
            else:
                old, a, b = hlsl_word(lds_word), hlsl_word(data0), hlsl_word(data1)
                if op.startswith("cmpst"):
                    expression = f"(({old} == {a}) ? {b} : {old})"
                else:
                    expression = f"(({old} & ~{a}) | {b})"
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
                new_word = ("alu", expression)
            if returns:
                vector[regs[0]] = lds_word
            lds_word = new_word
            ds_wait = True
        elif m := re.fullmatch(r"ds_(add|min|max|cmpst)(?P<ret>_rtn)?_f32 (v\d+(?:, v\d+){1,3})(?: offset:0)?", line):
            op = m[1]
            returns = bool(m["ret"])
            regs = [int(reg[1:]) for reg in m[3].split(", ")]
            if lds_f32_mode != "normal-rne":
                raise ValueError("Unverified private LDS floating-point mode")
            expected = 2 + int(returns) + int(op == "cmpst")
            if len(regs) != expected:
                raise ValueError("Unverified private LDS floating-point operands")
            start = int(returns)
            addr, data = regs[start:start + 2]
            value = vector.get(data)
            replacement = vector.get(regs[-1]) if op == "cmpst" else value
            if (guard_span is not None or max(regs) > 255 or vector.get(addr) != ("lds_address",)
                    or not all(map(is_word, (lds_word, value, replacement)))):
                raise ValueError("Unverified private LDS floating-point access")
            old, src = map(hlsl_word, (lds_word, value))
            if op.startswith("add"):
                if ds_step_count >= 64:
                    raise ValueError("Too many private LDS floating-point steps")
                name = f"ds_f32_{ds_step_count}"
                ds_step_count += 1
                staged_steps.append((None, f"    precise float {name} = asfloat({old}) + asfloat({src});"))
                expression = f"asuint({name})"
            else:
                condition = {"min": "<", "max": ">", "cmpst": "=="}[op]
                expression = f"((asfloat({src}) {condition} asfloat({old})) ? {hlsl_word(replacement)} : {old})"
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
            if returns:
                vector[regs[0]] = lds_word
            lds_word = ("alu", expression)
            ds_wait = True
        elif m := re.fullmatch(r"ds_(cmpst|min|max)(?P<ret>_rtn)?_f64 "
                               r"(?:(v\[\d+:\d+\]), )?v(\d+), (v\[\d+:\d+\])(?:, (v\[\d+:\d+\]))?"
                               r"(?: offset:(0|8))?", line):
            op, returns, dest, addr, compare, replacement, offset = m.groups()
            if lds_f64_mode != "normal":
                raise ValueError("Unverified private LDS binary64 mode")
            slot = int(offset or 0) // 4
            prev = tuple(lds_wide_words.get(i) for i in range(slot, slot + 2))
            if (guard_span is not None or vector.get(int(addr)) != ("lds_wide_address",)
                    or bool(returns) != (dest is not None) or (op == "cmpst") != (replacement is not None)
                    or not all(map(is_word, prev))):
                raise ValueError("Unverified private LDS binary64 access")
            ranges = []
            for operand in (operand for operand in (compare, replacement, dest) if operand is not None):
                first, last = map(int, operand[2:-1].split(":"))
                if last > 255 or last != first + 1:
                    raise ValueError("Unverified private LDS binary64 range")
                ranges.append((first, last))
            cmp_word = tuple(vector.get(i) for i in ranges[0])
            new_word = tuple(vector.get(i) for i in ranges[1]) if replacement else cmp_word
            if not all(map(is_word, cmp_word + new_word)):
                raise ValueError("Unverified private LDS binary64 source")
            inputs = tuple(map(hlsl_word, prev + cmp_word + new_word))
            if ds_step_count >= 64 or any(len(word) > ALU_EXPRESSION_LIMIT for word in inputs):
                raise ValueError("LDS expression exceeds the profile limit")
            prefix = f"ds_f64_{ds_step_count}"
            ds_step_count += 1
            old_lo, old_hi, cmp_lo, cmp_hi, new_lo, new_hi = (
                f"{prefix}_{suffix}" for suffix in ("old_lo", "old_hi", "cmp_lo", "cmp_hi", "new_lo", "new_hi")
            )
            if op == "cmpst":
                condition = f"({old_lo} == {cmp_lo} && {old_hi} == {cmp_hi})"
            else:
                # For equal signs, compare the full bit patterns in the
                # opposite direction for negative numbers. Handle signs first.
                below = f"({cmp_hi} < {old_hi} || ({cmp_hi} == {old_hi} && {cmp_lo} < {old_lo}))"
                above = f"({cmp_hi} > {old_hi} || ({cmp_hi} == {old_hi} && {cmp_lo} > {old_lo}))"
                opposite = f"(({cmp_hi} ^ {old_hi}) & 0x80000000u) != 0u"
                negative = f"({cmp_hi} & 0x80000000u) != 0u"
                if op == "min":
                    condition = f"({opposite} ? ({negative}) : (({negative}) ? {above} : {below}))"
                else:
                    condition = f"({opposite} ? !({negative}) : (({negative}) ? {below} : {above}))"
            names = (old_lo, old_hi, cmp_lo, cmp_hi, new_lo, new_hi)
            take = f"{prefix}_take"
            statements = [f"    uint {name} = {word};" for name, word in zip(names, inputs)]
            statements.append(f"    bool {take} = {condition};")
            staged_steps.append((None, "\n".join(statements)))
            update = (f"({take} ? {new_lo} : {old_lo})", f"({take} ? {new_hi} : {old_hi})")
            if dest:
                vector.update(zip(ranges[-1], prev))
            lds_wide_words.update((slot + i, ("alu", word)) for i, word in enumerate(update))
            ds_wait = True
        elif m := re.fullmatch(r"ds_wrxchg2_rtn_b64 v\[(\d+):(\d+)\], v(\d+), "
                               r"v\[(\d+):(\d+)\], v\[(\d+):(\d+)\]"
                               r"(?: offset0:([01]))?(?: offset1:([01]))?", line):
            dst, last, addr, first0, last0, first1, last1 = map(int, m.groups()[:7])
            offsets = tuple(int(offset or 0) for offset in m.groups()[7:])
            if (guard_span is not None or vector.get(addr) != ("lds_wide_address",) or offsets[0] == offsets[1]
                    or max(last, last0, last1) > 255 or last != dst + 3 or last0 != first0 + 1 or last1 != first1 + 1):
                raise ValueError("Unverified private paired qword exchange")
            slots = [2 * offset + i for offset in offsets for i in range(2)]
            previous = [lds_wide_words.get(slot) for slot in slots]
            data = [vector.get(reg) for reg in (first0, last0, first1, last1)]
            if not all(map(is_word, previous + data)):
                raise ValueError("Uninitialized private paired qword exchange")
            vector.update(zip(range(dst, last + 1), previous))
            lds_wide_words.update(zip(slots, data))
            ds_wait = True
        elif m := re.fullmatch(
                r"ds_((?:cmpst|mskor)(?:_rtn)?|wrxchg_rtn)_b64 "
                r"(?:(v\[\d+:\d+\]), )?v(\d+), (v\[\d+:\d+\])(?:, (v\[\d+:\d+\]))?"
                r"(?: offset:(0|8))?", line):
            op, dst, addr, data, data2, offset = m.groups()
            returns = op.endswith("_rtn")
            exchange = op.startswith("wrxchg")
            slot = int(offset or 0) // 4
            old_words = (lds_wide_words.get(slot), lds_wide_words.get(slot + 1))
            if (guard_span is not None or vector.get(int(addr)) != ("lds_wide_address",)
                    or returns != (dst is not None) or exchange == (data2 is not None)
                    or not all(map(is_word, old_words))):
                raise ValueError("Unverified private wide LDS replacement")
            ranges = []
            for operand in (item for item in (data, data2, dst) if item is not None):
                first, last = map(int, operand[2:-1].split(":"))
                if last > 255 or last != first + 1:
                    raise ValueError("Unverified private wide LDS replacement range")
                ranges.append((first, last))
            source = tuple(vector.get(reg) for reg in ranges[0])
            other = tuple(vector.get(reg) for reg in ranges[1]) if not exchange else ()
            if not all(map(is_word, source + other)):
                raise ValueError("Unverified private wide LDS replacement source")
            if not exchange:
                old_low, old_high = map(hlsl_word, old_words)
                data_low, data_high = map(hlsl_word, source)
                new_low, new_high = map(hlsl_word, other)
                if op.startswith("cmpst"):
                    equal = f"({old_low} == {data_low} && {old_high} == {data_high})"
                    low, high = f"({equal} ? {new_low} : {old_low})", f"({equal} ? {new_high} : {old_high})"
                else:
                    low = f"(({old_low} & ~{data_low}) | {new_low})"
                    high = f"(({old_high} & ~{data_high}) | {new_high})"
                if max(len(low), len(high)) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
                source = (("alu", low), ("alu", high))
            if returns:
                vector.update(zip(ranges[-1], old_words))
            lds_wide_words.update(zip(range(slot, slot + 2), source))
            ds_wait = True
        elif m := re.fullmatch(
                r"ds_(read2st64_b(?:32|64)|write2st64_b(?:32|64)|wrxchg2st64_rtn_b(?:32|64)) "
                r"(v\[\d+:\d+\]|v\d+), (v\[\d+:\d+\]|v\d+)"
                r"(?:, (v\[\d+:\d+\]|v\d+))?(?:, (v\[\d+:\d+\]|v\d+))?"
                r" offset0:(\d+) offset1:(\d+)", line):
            op = m[1]
            width = int(op[-2:]) // 8
            reads = not op.startswith("write")
            writes = not op.startswith("read")
            exchange = reads and writes
            addr_text = m[3] if reads else m[2]
            addr = re.fullmatch(r"v(\d+)", addr_text)
            address_tag = ("lds_address",) if width == 4 else ("lds_strided_qword_address",)
            offsets = tuple(map(int, m.groups()[-2:]))
            # Keep strided planes above the contiguous private records, and
            # use a single element width so no two lane-owned planes alias.
            if (guard_span is not None or not addr or vector.get(int(addr[1])) != address_tag
                    or lds_strided_width not in (None, width) or max(offsets) > 255
                    or min(offsets) * 64 * width < 512 or (max(offsets) * 64 + 32) * width > lds_bytes
                    or writes != (m[4] is not None) or writes and offsets[0] == offsets[1]):
                raise ValueError("Unverified private strided LDS access")
            lds_strided_width = width
            count = width // 4
            if writes:
                sources = (m[4], m[5]) if exchange else (m[3], m[4])
                if not all(sources) or not exchange and m[5] is not None:
                    raise ValueError("Unverified private strided LDS operands")
                pairs = []
                for source in sources:
                    regs = re.fullmatch(r"v(\d+)" if count == 1 else r"v\[(\d+):(\d+)\]", source)
                    if not regs or count == 2 and int(regs[2]) != int(regs[1]) + 1:
                        raise ValueError("Unverified private strided LDS source range")
                    first = int(regs[1])
                    words = tuple(vector.get(reg) for reg in range(first, first + count))
                    if first + count > 256 or not all(map(is_word, words)):
                        raise ValueError("Unverified private strided LDS source")
                    pairs.append(words)
            if reads:
                regs = re.fullmatch(r"v\[(\d+):(\d+)\]", m[2])
                if not regs or int(regs[2]) > 255 or int(regs[2]) - int(regs[1]) + 1 != 2 * count:
                    raise ValueError("Unverified private strided LDS destination range")
                if not all(offset in lds_strided_words for offset in offsets):
                    raise ValueError("Uninitialized private strided LDS plane")
                first = int(regs[1])
                words = lds_strided_words[offsets[0]] + lds_strided_words[offsets[1]]
                vector.update(zip(range(first, first + 2 * count), words))
            if writes:
                lds_strided_words.update(zip(offsets, pairs))
            ds_wait = True
        elif m := re.fullmatch(r"ds_(write|read)_b(64|96|128) (v\[\d+:\d+\]|v\d+), "
                               r"(v\[\d+:\d+\]|v\d+)(?: offset:(0|8))?", line):
            write = m[1] == "write"
            count = int(m[2]) // 32
            addr_text, regs_text = (m[3], m[4]) if write else (m[4], m[3])
            addr = re.fullmatch(r"v(\d+)", addr_text)
            regs = re.fullmatch(r"v\[(\d+):(\d+)\]", regs_text)
            first_slot = int(m[5] or 0) // 4
            if (guard_span is not None or not addr or not regs or int(addr[1]) > 255
                    or vector.get(int(addr[1])) != ("lds_wide_address",) or first_slot + count > 4):
                raise ValueError("Unverified private wide LDS access")
            first, last = map(int, regs.groups())
            if last > 255 or last - first + 1 != count:
                raise ValueError("Unverified private wide LDS register range")
            slots = range(first_slot, first_slot + count)
            registers = range(first, last + 1)
            if write:
                words = [vector.get(reg) for reg in registers]
                if not all(map(is_word, words)):
                    raise ValueError("Unverified private wide LDS source")
                lds_wide_words.update(zip(slots, words))
            else:
                if not all(slot in lds_wide_words for slot in slots):
                    raise ValueError("Uninitialized private wide LDS word")
                vector.update(zip(registers, (lds_wide_words[slot] for slot in slots)))
            ds_wait = True
        elif m := re.fullmatch(r"ds_wrxchg2_rtn_b32 v\[(\d+):(\d+)\], v(\d+), v(\d+), v(\d+)"
                               r"(?: offset0:([0-3]))?(?: offset1:([0-3]))?", line):
            first, last, addr, src0, src1 = map(int, m.groups()[:5])
            slots = tuple(int(offset or 0) for offset in m.groups()[5:])
            if (guard_span is not None or max(last, addr, src0, src1) > 255 or last != first + 1
                    or vector.get(addr) != ("lds_wide_address",) or slots[0] == slots[1]):
                raise ValueError("Unverified private paired dword exchange")
            previous = [lds_wide_words.get(slot) for slot in slots]
            replacement = [vector.get(reg) for reg in (src0, src1)]
            if not all(map(is_word, previous + replacement)):
                raise ValueError("Uninitialized private paired dword exchange")
            vector.update(zip(range(first, last + 1), previous))
            lds_wide_words.update(zip(slots, replacement))
            ds_wait = True
        elif m := re.fullmatch(r"ds_(append|consume) v(\d+)(?: offset:0)?", line):
            op, dst = m[1], int(m[2])
            if (not lds_m0_zero or guard_span is not None or dst > 255
                    or lds_address_shift != 2 or not is_word(lds_word) or ds_step_count >= 64):
                raise ValueError("Unverified LDS wave counter")
            old = hlsl_word(lds_word)
            slot = f"ds_stage_{ds_step_count}"
            previous = f"ds_result_{ds_step_count}"
            ds_step_count += 1
            # LDS byte zero is lane zero's word. Snapshot it once per group,
            # then only change that lane's word in the private layout.
            operator = "+" if op == "append" else "-"
            staged_steps.append((slot, f"    if (local.x == 0u) {slot}[0] = {old};\n"
                             "    GroupMemoryBarrierWithGroupSync();\n"
                             f"    uint {previous} = {slot}[0];"))
            vector[dst] = ("alu", previous)
            lds_word = ("alu", f"(local.x == 0u ? ({previous} {operator} 32u) : {old})")
            if len(hlsl_word(lds_word)) > ALU_EXPRESSION_LIMIT:
                raise ValueError("LDS expression exceeds the profile limit")
            ds_wait = True
        elif m := re.fullmatch(
                r"ds_(read|write)2_b(32|64) (v\[\d+:\d+\]|v\d+), (v\[\d+:\d+\]|v\d+)"
                r"(?:, (v\[\d+:\d+\]|v\d+))?(?: offset0:([0-3]))?(?: offset1:([0-3]))?", line):
            write = m[1] == "write"
            width = int(m[2]) // 32
            addr_text = m[3] if write else m[4]
            addr = re.fullmatch(r"v(\d+)", addr_text)
            offsets = tuple(int(offset or 0) * width for offset in m.groups()[5:])
            slots = tuple(slot for offset in offsets for slot in range(offset, offset + width))
            if (guard_span is not None or not addr or vector.get(int(addr[1])) != ("lds_wide_address",)
                    or max(slots) >= 4 or write != (m[5] is not None) or write and offsets[0] == offsets[1]):
                raise ValueError("Unverified private paired LDS access")
            operands = (m[4], m[5]) if write else (m[3],)
            count = width if write else 2 * width
            registers = []
            for operand in operands:
                match = re.fullmatch(r"v(\d+)" if count == 1 else r"v\[(\d+):(\d+)\]", operand)
                if not match:
                    raise ValueError("Unverified private paired LDS register range")
                first = int(match[1])
                last = first if count == 1 else int(match[2])
                if last > 255 or last - first + 1 != count:
                    raise ValueError("Unverified private paired LDS register range")
                registers.extend(range(first, last + 1))
            if write:
                words = [vector.get(reg) for reg in registers]
                if not all(map(is_word, words)):
                    raise ValueError("Unverified private wide LDS source")
                lds_wide_words.update(zip(slots, words))
            else:
                if not all(slot in lds_wide_words for slot in slots):
                    raise ValueError("Uninitialized private wide LDS word")
                vector.update(zip(registers, (lds_wide_words[slot] for slot in slots)))
            ds_wait = True
        elif ((m := re.fullmatch(r"ds_(add|sub|rsub|and|or|xor|min|max|inc|dec|condxchg32)(?P<ret>_rtn)?(_[uib]64) "
                                 r"(?(ret)v\[(?P<dst_first>\d+):(?P<dst_last>\d+)\], )"
                                 r"v(?P<addr>\d+), v\[(?P<src_first>\d+):(?P<src_last>\d+)\]"
                                 r"(?: offset:(?P<offset>0|8))?", line))
              and ((op := m[1] + m[3]) in ("add_u64", "sub_u64", "rsub_u64", "and_b64", "or_b64",
                                        "xor_b64", "min_i64", "max_i64", "min_u64", "max_u64", "inc_u64", "dec_u64")
                   or op == "condxchg32_b64" and m["ret"])):
            addr, first, last = (int(m[key]) for key in ("addr", "src_first", "src_last"))
            destination = tuple(int(m[key]) for key in ("dst_first", "dst_last")) if m["ret"] else None
            slot = int(m["offset"] or 0) // 4
            source = (vector.get(first), vector.get(last))
            previous = (lds_wide_words.get(slot), lds_wide_words.get(slot + 1))
            if (guard_span is not None or vector.get(addr) != ("lds_wide_address",)
                    or last > 255 or last != first + 1 or not all(map(is_word, previous + source))
                    or destination and (destination[1] > 255 or destination[1] != destination[0] + 1)):
                raise ValueError("Unverified private wide LDS update")
            left, right = (source, previous) if op == "rsub_u64" else (previous, source)
            lo, hi = map(hlsl_word, left)
            rhs_lo, rhs_hi = map(hlsl_word, right)
            if op == "inc_u64":
                reset = f"({hi} > {rhs_hi} || ({hi} == {rhs_hi} && {lo} >= {rhs_lo}))"
                low = f"({reset} ? 0u : ({lo} + 1u))"
                high = f"({reset} ? 0u : ({hi} + uint({lo} == 0xffffffffu)))"
            elif op == "dec_u64":
                reset = f"(({hi} | {lo}) == 0u || {hi} > {rhs_hi} || ({hi} == {rhs_hi} && {lo} > {rhs_lo}))"
                low = f"({reset} ? {rhs_lo} : ({lo} - 1u))"
                high = f"({reset} ? {rhs_hi} : ({hi} - uint({lo} == 0u)))"
            elif op == "condxchg32_b64":
                low = f"(({rhs_lo} & 0x80000000u) != 0u ? {rhs_lo} : {lo})"
                high = f"(({rhs_hi} & 0x80000000u) != 0u ? {rhs_hi} : {hi})"
            elif op in ("and_b64", "or_b64", "xor_b64"):
                symbol = {"and_b64": "&", "or_b64": "|", "xor_b64": "^"}[op]
                low = f"({lo} {symbol} {rhs_lo})"
                high = f"({hi} {symbol} {rhs_hi})"
            elif op in ("min_i64", "max_i64", "min_u64", "max_u64"):
                relation = "<" if op.startswith("min") else ">"
                if op.endswith("_i64"):
                    first = f"asint({hi}) {relation} asint({rhs_hi})"
                else:
                    first = f"{hi} {relation} {rhs_hi}"
                condition = f"(({first}) || (({hi} == {rhs_hi}) && ({lo} {relation} {rhs_lo})))"
                low = f"({condition} ? {lo} : {rhs_lo})"
                high = f"({condition} ? {hi} : {rhs_hi})"
            elif op == "add_u64":
                low = f"({lo} + {rhs_lo})"
                high = f"({hi} + {rhs_hi} + uint({low} < {lo}))"
            else:
                low = f"({lo} - {rhs_lo})"
                high = f"({hi} - {rhs_hi} - uint({lo} < {rhs_lo}))"
            if max(len(low), len(high)) > ALU_EXPRESSION_LIMIT:
                raise ValueError("LDS expression exceeds the profile limit")
            lds_wide_words[slot] = ("alu", low)
            lds_wide_words[slot + 1] = ("alu", high)
            if destination is not None:
                vector.update(zip(destination, previous))
            ds_wait = True
        elif m := re.fullmatch(r"ds_(write|read)_addtid_b32 v(\d+)(?: offset:0)?", line):
            op, reg = m[1], int(m[2])
            # A single full wave32 makes TID equal to local.x. Zero M0 and
            # offset make this the same private word as an explicit byte address.
            if not lds_m0_zero or guard_span is not None or reg > 255:
                raise ValueError("Unverified private LDS thread-ID access")
            if lds_address_shift not in (None, 2):
                raise ValueError("Mixed private LDS address strides")
            lds_address_shift = 2
            if op == "write":
                value = vector.get(reg)
                if not is_word(value):
                    raise ValueError("Unverified private LDS thread-ID source")
                lds_word = value
            else:
                if lds_word is None:
                    raise ValueError("Uninitialized private LDS thread-ID word")
                vector[reg] = lds_word
            ds_wait = True
        elif m := re.fullmatch(r"ds_write_b(8|16|32)(?P<high>_d16_hi)? v(\d+), v(\d+)(?: offset:([0-3]))?", line):
            width, addr, src = map(int, (m[1], m[3], m[4]))
            offset = int(m[5] or 0)
            high = m["high"]
            if high and width == 32:
                raise ValueError("Unsupported compute instruction: " + line)
            value = vector.get(src)
            if (not lds_enabled or guard_span is not None or max(addr, src) > 255
                    or vector.get(addr) != ("lds_address",) or not is_word(value)
                    or offset % (width // 8) or (width < 32 and lds_word is None)):
                raise ValueError("Unverified private LDS write")
            # Every lane owns one disjoint word. Capture values even if their VGPRs are overwritten.
            if width == 32:
                lds_word = value
            else:
                shift = offset * 8
                mask = (1 << width) - 1
                keep = 0xffffffff ^ (mask << shift)
                data = hlsl_word(value)
                if high:
                    data = f"({data} >> 16u)"
                expression = (f"(({hlsl_word(lds_word)} & 0x{keep:08x}u) | "
                              f"(({data} & 0x{mask:x}u) << {shift}u))")
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
                lds_word = ("alu", expression)
            ds_wait = True
        elif ((m := re.fullmatch(r"(ds_[a-z]+)(?P<ret>_rtn)?(_[iub]32) "
                                 r"(?(ret)v(\d+), )v(\d+), v(\d+)(?: offset:0)?", line))
              and (template := LDS_RMW.get(m[1] + m[3]))):
            dst = int(m[4]) if m[4] is not None else None
            addr, src = int(m[5]), int(m[6])
            value = vector.get(src)
            if (not lds_enabled or guard_span is not None or max(addr, src) > 255
                    or (dst is not None and dst > 255) or vector.get(addr) != ("lds_address",)
                    or lds_word is None or not is_word(value)):
                raise ValueError("Unverified private LDS update")
            expression = template.format(hlsl_word(lds_word), hlsl_word(value))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("LDS expression exceeds the profile limit")
            # Capture the inputs before assigning the old word to a possibly overlapping VGPR.
            if dst is not None:
                vector[dst] = lds_word
            lds_word = ("alu", expression)
            ds_wait = True
        elif m := re.fullmatch(r"ds_read_(b32|[ui](?:8|16))(?P<d16>_d16(?P<hi>_hi)?)? "
                               r"v(?P<dst>\d+), v(?P<addr>\d+)(?: offset:(?P<offset>[0-3]))?", line):
            op, dst, addr, offset = m[1], int(m["dst"]), int(m["addr"]), int(m["offset"] or 0)
            d16, high = m["d16"], m["hi"]
            if d16 and op not in ("u8", "i8", "u16"):
                raise ValueError("Unsupported compute instruction: " + line)
            width = int(op[1:])
            if (not lds_enabled or guard_span is not None or max(dst, addr) > 255
                    or vector.get(addr) != ("lds_address",) or lds_word is None or offset % (width // 8)):
                raise ValueError("Unverified private LDS read")
            if width == 32:
                vector[dst] = lds_word
            else:
                shifted = f"({hlsl_word(lds_word)} >> {offset * 8}u)"
                if op.startswith("i"):
                    # Shifting to the sign bit discards any higher neighbors before the arithmetic shift.
                    extend = 32 - width
                    expression = f"asuint(asint({shifted} << {extend}u) >> {extend}u)"
                else:
                    expression = f"({shifted} & 0x{(1 << width) - 1:x}u)"
                if d16:
                    previous = vector.get(dst)
                    if not is_word(previous):
                        raise ValueError("Unverified D16 destination word")
                    keep, shift = ("0x0000ffffu", 16) if high else ("0xffff0000u", 0)
                    expression = f"(({hlsl_word(previous)} & {keep}) | (({expression} & 0xffffu) << {shift}u))"
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
                vector[dst] = ("alu", expression)
            ds_wait = True
        elif m := re.fullmatch(r"ds_((?:inc|dec)(?:_rtn)?_u32|wrap_rtn_b32) "
                               r"(v\d+(?:, v\d+){1,3})(?: offset:0)?", line):
            op = m[1]
            returns = "_rtn_" in op
            wraps = op == "wrap_rtn_b32"
            regs = [int(token[1:]) for token in m[2].split(", ")]
            if len(regs) != 2 + returns + wraps:
                raise ValueError("Unverified LDS counter operands")
            addr, src, *step = regs[returns:]
            value = vector.get(src)
            advance = vector.get(step[0]) if wraps else None
            if (not lds_enabled or guard_span is not None or max(regs) > 255
                    or vector.get(addr) != ("lds_address",) or lds_word is None or not is_word(value)
                    or (wraps and not is_word(advance))):
                raise ValueError("Unverified private LDS counter update")
            old, limit = hlsl_word(lds_word), hlsl_word(value)
            if wraps:
                expression = f"(({old} >= {limit}) ? ({old} - {limit}) : ({old} + {hlsl_word(advance)}))"
            elif op.startswith("inc_"):
                expression = f"(({old} >= {limit}) ? 0u : ({old} + 1u))"
            else:
                expression = f"((({old} == 0u) || ({old} > {limit})) ? {limit} : ({old} - 1u))"
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("LDS expression exceeds the profile limit")
            if returns:
                vector[regs[0]] = lds_word
            lds_word = ("alu", expression)
            ds_wait = True
        elif m := re.fullmatch(rf"(v_cvt_pk_u8_f32) v(\d+), {FLOAT_SOURCE}, {ALU_SOURCE}, {ALU_SOURCE}", line):
            # VOP3 without input or output modifiers. S0 is the binary32 value, S1 the byte select, S2 the other bytes.
            op, dst = m[1], int(m[2])
            words = [float_word(m[3], scalar, vector, pending, sizes),
                     *(alu_word(token, scalar, vector, pending, sizes) for token in m.groups()[3:])]
            if (dst > 255 or None in words or isinstance(words[0], int)
                    and not (words[0] == 0 or 0x3f800000 <= words[0] <= 0x437f0000
                             and struct.unpack("<f", struct.pack("<I", words[0]))[0].is_integer())):
                raise ValueError("Unverified ALU operand")
            expression = CVT_PK_U8[op].format(*map(hlsl_word, words))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(r"v_lshl_add_u32 v(\d+), v(\d+), (\d+), v(\d+)", line):
            dst, row, shift, column = map(int, m.groups())
            row_tag, column_tag = vector.get(row), vector.get(column)
            # (row index << shift) + column byte offset, from two different dispatch axes.
            # The plane and element checks at the store bound both shifts.
            if (dst > 255 or not (isinstance(row_tag, tuple) and row_tag[0] == "thread")
                    or not (isinstance(column_tag, tuple) and column_tag[0] == "thread_bytes")
                    or row_tag[1] == column_tag[1]):
                raise ValueError("Unverified plane byte offset")
            vector[dst] = ("plane_bytes", row_tag[1], shift, *column_tag[1:])
        elif ((m := re.fullmatch(r"v_and_b32_e32 v(\d+), (?:0xffff|65535), v(\d+)", line))
              and isinstance(partial := vector.get(int(m[2])), tuple) and partial[0] in ("source_low16", "alu_low16")):
            dst = int(m[1])
            if dst > 255:
                raise ValueError("Invalid vector destination")
            word = f"inputData.Load({partial[1]})" if partial[0] == "source_low16" else partial[1]
            vector[dst] = ("alu", f"({word} & 0xffffu)")
        elif ((m := re.fullmatch(r"(v_(?:add|sub|subrev|mul)_f32)_e64 v(\d+), ([^,\s]+), ([^,\s]+)( .+)?", line))
              and any(token.startswith(("-", "|", "neg(", "abs(")) and not re.fullmatch(FLOAT_SOURCE, token)
                      for token in m.groups()[2:4])):
            # The VOP3 spelling with a source modifier on at least one source; either source is any binary32 source.
            # The unmodified _e64 form is left to the fall-through. clamp and omod (mul:2, mul:4, div:2) are refused.
            op, dst = m[1] + "_e32", int(m[2])
            if m[5]:
                raise ValueError(f"Unsupported float output modifier: {line}")
            sources = [float_e64_source(token, scalar, vector, pending) for token in m.groups()[2:4]]
            if dst > 255 or None in sources:
                raise ValueError("Unverified float operand")
            expression = FALU[op].format(*sources)
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"(v_(?:add|sub|subrev|mul)_f32_e32) v(\d+), {FLOAT_SOURCE}, (v\d+)", line):
            op, dst = m[1], int(m[2])
            words = [float_word(token, scalar, vector, pending, sizes) for token in m.groups()[2:]]
            if dst > 255 or None in words:
                raise ValueError("Unverified float operand")
            expression = FALU[op].format(*map(hlsl_word, words))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif ((m := re.fullmatch(rf"v_mac_f32_e32 v(\d+), {FLOAT_SOURCE}, (v\d+)", line))
              or (m := re.fullmatch(rf"v_mac_f32_e64 v(\d+), {FLOAT_SOURCE}, {FLOAT_SOURCE}", line))):
            dst = int(m[1])
            if dst > 255:
                raise ValueError("Invalid vector destination")
            words = [float_word(token, scalar, vector, pending) for token in (*m.groups()[1:], f"v{dst}")]
            if None in words:
                raise ValueError("Unverified float operand")
            expression = f"f32_mac({', '.join(map(hlsl_word, words))})"
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"v_cvt_u32_f32_e32 v(\d+), {FLOAT_SOURCE}", line):
            dst = int(m[1])
            if dst > 255:
                raise ValueError("Invalid vector destination")
            token = m[2]
            word = FLOAT_INLINE.get(token) if "." in token else alu_word(token, scalar, vector, pending, sizes)
            if word is None:
                raise ValueError("Unverified float operand")
            expression = f"cvt_u32_f32({hlsl_word(word)})"
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"(v_dot2_u32_u16|v_dot4_u32_u8) v(\d+), {ALU_SOURCE}, {ALU_SOURCE}, {ALU_SOURCE}", line):
            op, dst = m[1], int(m[2])
            if dst > 255:
                raise ValueError("Invalid vector destination")
            words = [alu_word(token, scalar, vector, pending) for token in m.groups()[2:]]
            if None in words:
                raise ValueError("Unverified ALU operand")
            helper = "dot2_u32_u16" if op == "v_dot2_u32_u16" else "dot4_u32_u8"
            expression = f"{helper}({', '.join(map(hlsl_word, words))})"
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(r"v_or_b32_dpp v(\d+), v(\d+), v(\d+) quad_perm:\[([0-3]),([0-3]),([0-3]),([0-3])\] row_mask:0xf bank_mask:0xf", line):
            dst, src0, src1 = map(int, m.groups()[:3])
            # This lowering indexes the guest's logical 1D quad through group-shared memory.
            if (cpu != "gfx1013" or max(dst, src0, src1) > 255 or guard_span is not None
                    or dims[1] != 1 or dims[2] != 1 or dims[0] % 4):
                raise ValueError("Unsupported or masked DPP16 quad permutation")
            values = (vector.get(src0), vector.get(src1))
            if not all(is_word(value) for value in values):
                raise ValueError("Unverified DPP16 operand")
            perm = tuple(map(int, m.groups()[3:]))
            step = dpp_step_count
            dpp_step_count += 1
            lane, quad = f"dpp{step}_lane", f"dpp{step}_quad"
            source_thread = f"dpp{step}_source_thread"
            source, value = f"dpp{step}_source", f"dpp{step}_value"
            selectors = " | ".join(
                f"(({quad} == {index}u) ? {perm[index]}u : 0u)" for index in range(4))
            slot = "dpp_stage"
            staged_steps.append((slot, f"    uint {lane} = local.x & 31u;\n"
                                 f"    uint {quad} = {lane} & 3u;\n"
                                 f"    {slot}[local.x] = {hlsl_word(values[0])};\n"
                                 "    GroupMemoryBarrierWithGroupSync();\n"
                                 f"    uint {source_thread} = (local.x & ~3u) | ({selectors});\n"
                                 f"    uint {source} = {slot}[{source_thread}];\n"
                                 "    GroupMemoryBarrierWithGroupSync();\n"
                                 f"    uint {value} = {source} | {hlsl_word(values[1])};"))
            vector[dst] = ("alu", value)
        elif m := re.fullmatch(r"v_(or_b32|add_nc_u32)_dpp v(\d+), v(\d+), v(\d+) row_shr:(\d+) "
                               r"row_mask:0xf bank_mask:0xf( bound_ctrl:[01])?", line):
            op, dst, src0, src1, shift = m[1], *map(int, m.groups()[1:5])
            # LLVM 15 assembles bound_ctrl:0 and bound_ctrl:1 to the same set BOUND_CTRL bit and prints :1.
            bound = m[6] is not None
            # Rows are 16 of the guest's logical 1D lanes, indexed through group-shared memory.
            if (cpu != "gfx1013" or max(dst, src0, src1) > 255 or not 1 <= shift <= 15
                    or guard_span is not None or dims[1] != 1 or dims[2] != 1 or dims[0] % 16):
                raise ValueError("Unsupported or masked DPP16 row shift")
            # Without BOUND_CTRL a lane whose source is before its row start keeps its old destination.
            values = [vector.get(reg) for reg in (src0, src1) + (() if bound else (dst,))]
            if not all(is_word(value) for value in values):
                raise ValueError("Unverified DPP16 operand")
            step = dpp_step_count
            dpp_step_count += 1
            row_lane, inside = f"dpp{step}_row_lane", f"dpp{step}_inside"
            source_thread = f"dpp{step}_source_thread"
            source, value = f"dpp{step}_source", f"dpp{step}_value"
            result = f"({source} {'|' if op == 'or_b32' else '+'} ({hlsl_word(values[1])}))"
            slot = "dpp_stage"
            staged_steps.append((slot, f"    uint {row_lane} = local.x & 15u;\n"
                                 f"    bool {inside} = {row_lane} >= {shift}u;\n"
                                 f"    {slot}[local.x] = {hlsl_word(values[0])};\n"
                                 "    GroupMemoryBarrierWithGroupSync();\n"
                                 f"    uint {source_thread} = {inside} ? local.x - {shift}u : local.x;\n"
                                 f"    uint {source} = {inside} ? {slot}[{source_thread}] : 0u;\n"
                                 "    GroupMemoryBarrierWithGroupSync();\n"
                                 f"    uint {value} = "
                                 + (f"{result};" if bound else f"{inside} ? {result} : ({hlsl_word(values[2])});")))
            vector[dst] = ("alu", value)
        elif ((m := re.fullmatch(rf"(v_(?:and_b32|lshrrev_b32|min_u32|max_u32|add_nc_u32|or_b32|xor_b32|lshlrev_b32|ashrrev_i32|sub_nc_u32|subrev_nc_u32)_e32) v(\d+), {ALU_SOURCE}, (v\d+)", line))
              or (m := re.fullmatch(rf"(v_mul_lo_u32|v_add_nc_i32|v_sub_nc_i32|v_bcnt_u32_b32|v_bfm_b32) v(\d+), {ALU_SOURCE}, {ALU_SOURCE}", line))
              or (m := re.fullmatch(rf"(v_mad_u32_u24|v_xor3_b32|v_max3_u32|v_min3_u32|v_med3_u32|v_med3_i32|v_mad_i32_i24) v(\d+), {ALU_SOURCE}, {ALU_SOURCE}, {ALU_SOURCE}", line))
              and None not in [alu_word(token, scalar, vector, pending, sizes) for token in m.groups()[2:]]):
            # VOP2 takes a VGPR as src1. A v_mad_u32_u24 with an address operand is a row pitch below.
            op, dst = m[1], int(m[2])
            words = [alu_word(token, scalar, vector, pending, sizes) for token in m.groups()[2:]]
            if dst > 255 or None in words:
                raise ValueError("Unverified ALU operand")
            expression = ALU[op].format(*map(hlsl_word, words))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(rf"(v_(?:bfrev_b32|ffbh_i32)_e32) v(\d+), {ALU_SOURCE}", line):
            # VOP1 src0 is any source. The VOP3, SDWA and DPP spellings and source modifiers do not match.
            op, dst = m[1], int(m[2])
            word = alu_word(m[3], scalar, vector, pending)
            if dst > 255 or word is None:
                raise ValueError("Unverified ALU operand")
            expression = VOP1_ALU[op].format(hlsl_word(word))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif ((m := re.fullmatch(rf"(v_sub(?:rev)?_co_u32) v(\d+), (vcc_lo|s\d+), {ALU_SOURCE}, {ALU_SOURCE}", line))
              or (m := re.fullmatch(rf"(v_sub(?:rev)?_co_ci_u32_e32) v(\d+), (vcc_lo), {ALU_SOURCE}, (v\d+), (vcc_lo)", line))
              or (m := re.fullmatch(rf"(v_sub(?:rev)?_co_ci_u32_e64) v(\d+), (vcc_lo|s\d+), {ALU_SOURCE}, {ALU_SOURCE}, (vcc_lo|s\d+)", line))):
            # RDNA2 ISA:
            #   v_sub_co_u32:       temp = (uint64)S0 - (uint64)S1; D = temp[31:0]; SDST[lane] = (S0 < S1) ? 1 : 0
            #   v_subrev_co_u32:    temp = (uint64)S1 - (uint64)S0; D = temp[31:0]; SDST[lane] = (S1 < S0) ? 1 : 0
            #   v_sub_co_ci_u32:    temp = (int64)S0 - (int64)S1 - (int64)borrow_in[lane]; D = temp[31:0]; SDST[lane] = (temp < 0) ? 1 : 0
            #   v_subrev_co_ci_u32: temp = (int64)S1 - (int64)S0 - (int64)borrow_in[lane]; D = temp[31:0]; SDST[lane] = (temp < 0) ? 1 : 0
            op, dst, sdst, src0, src1 = m[1], int(m[2]), m[3], m[4], m[5]
            borrow_in_tok = m[6] if len(m.groups()) == 6 else None
            if dst > 255:
                raise ValueError("Invalid vector destination")
            if dims[0] * dims[1] * dims[2] > 32 and wave_size != 32:
                raise ValueError("Borrow-out subtract requires wave_size=32 for groups larger than 32 threads")
            if sdst != "vcc_lo":
                sdst_reg = int(sdst[1:])
                if sdst_reg > 105 or sdst_reg in pending:
                    raise ValueError("Unverified borrow destination")
            words = [alu_word(token, scalar, vector, pending) for token in (src0, src1)]
            if None in words:
                raise ValueError("Unverified ALU operand")
            a, b = map(hlsl_word, words)
            if op.startswith("v_subrev_"):
                a, b = b, a
            diff_ab = f"({a} - {b})"
            if borrow_in_tok is None:
                expression = diff_ab
                borrow_expression = f"({a} < {b} ? 1u : 0u)"
            else:
                if borrow_in_tok == "vcc_lo":
                    borrow_in = vcc_carry
                else:
                    bin_reg = int(borrow_in_tok[1:])
                    borrow_in = scalar.get(bin_reg) if bin_reg <= 105 and bin_reg not in pending else None
                if not (isinstance(borrow_in, tuple) and borrow_in[0] == "borrow"):
                    raise ValueError("Unverified borrow-in operand")
                c = borrow_in[1]
                expression = f"({diff_ab} - {c})"
                borrow_expression = f"(({a} < {b} || {diff_ab} < {c}) ? 1u : 0u)"
            if len(expression) > ALU_EXPRESSION_LIMIT or len(borrow_expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
            if sdst == "vcc_lo":
                vcc = None
                vcc_cond = None
                vcc_carry = ("borrow", borrow_expression)
            else:
                scalar[sdst_reg] = ("borrow", borrow_expression)
        elif ((m := re.fullmatch(rf"(v_add_co_u32) v(\d+), (vcc_lo|s\d+), {ALU_SOURCE}, {ALU_SOURCE}", line))
              or (m := re.fullmatch(rf"(v_add_co_ci_u32_e32) v(\d+), (vcc_lo), {ALU_SOURCE}, (v\d+), (vcc_lo)", line))
              or (m := re.fullmatch(rf"(v_add_co_ci_u32_e64) v(\d+), (vcc_lo|s\d+), {ALU_SOURCE}, {ALU_SOURCE}, (vcc_lo|s\d+)", line))):
            # RDNA2 ISA:
            #   v_add_co_u32:    temp = (uint64)S0 + (uint64)S1; D = temp[31:0]; SDST[lane] = (temp >> 32) & 1
            #   v_add_co_ci_u32: temp = (uint64)S0 + (uint64)S1 + (uint64)carry_in[lane]; D = temp[31:0]; SDST[lane] = (temp >> 32) & 1
            dst, sdst, src0, src1 = int(m[2]), m[3], m[4], m[5]
            carry_in_tok = m[6] if len(m.groups()) == 6 else None
            if dst > 255:
                raise ValueError("Invalid vector destination")
            if dims[0] * dims[1] * dims[2] > 32 and wave_size != 32:
                raise ValueError("Carry-out add requires wave_size=32 for groups larger than 32 threads")
            if sdst != "vcc_lo":
                sdst_reg = int(sdst[1:])
                if sdst_reg > 105 or sdst_reg in pending:
                    raise ValueError("Unverified carry destination")
            words = [alu_word(token, scalar, vector, pending, sizes) for token in (src0, src1)]
            if None in words:
                raise ValueError("Unverified ALU operand")
            a, b = map(hlsl_word, words)
            sum_ab = f"({a} + {b})"
            if carry_in_tok is None:
                expression = sum_ab
                carry_expression = f"({sum_ab} < {a} ? 1u : 0u)"
            else:
                if carry_in_tok == "vcc_lo":
                    carry_in = vcc_carry
                else:
                    cin_reg = int(carry_in_tok[1:])
                    carry_in = scalar.get(cin_reg) if cin_reg <= 105 and cin_reg not in pending else None
                if not (isinstance(carry_in, tuple) and carry_in[0] == "carry"):
                    raise ValueError("Unverified carry-in operand")
                c = carry_in[1]
                expression = f"({sum_ab} + {c})"
                carry_expression = f"(({sum_ab} < {a} || {expression} < {c}) ? 1u : 0u)"
            if len(expression) > ALU_EXPRESSION_LIMIT or len(carry_expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
            if sdst == "vcc_lo":
                vcc = None
                vcc_cond = None
                vcc_carry = ("carry", carry_expression)
            else:
                scalar[sdst_reg] = ("carry", carry_expression)
        elif m := re.fullmatch(r"v_mad_u32_u24 v(\d+), (v\d+|-?\d+|0x[0-9a-f]+), (v\d+|-?\d+|0x[0-9a-f]+), v(\d+)", line):
            dst, column = int(m[1]), int(m[4])
            # The literal pitch may be either multiplicand; LLVM 15 places it in src0.
            row, pitch = (m[2], m[3]) if m[2].startswith("v") else (m[3], m[2])
            row_tag = vector.get(int(row[1:])) if row.startswith("v") else None
            column_tag = vector.get(column)
            # (bits [23:0] of the row index) * pitch + column byte offset. The row check at the store
            # bounds the column shift against the pitch.
            if (dst > 255 or pitch.startswith("v") or not 4 <= int(pitch, 0) < 1 << 24 or int(pitch, 0) % 4
                    or not (isinstance(row_tag, tuple) and row_tag[0] == "thread")
                    or not (isinstance(column_tag, tuple) and column_tag[0] == "thread_bytes")
                    or row_tag[1] == column_tag[1]):
                raise ValueError("Unverified row pitch byte offset")
            vector[dst] = ("pitch_bytes", row_tag[1], int(pitch, 0), *column_tag[1:])
        elif cpu == "gfx1030" and line in ("buffer_gl0_inv", "buffer_gl1_inv"):
            # The desktop profile has no mapping for these cache levels or their acknowledgments.
            raise ValueError(f"Unsupported MUBUF cache invalidation: {line}")
        elif m := re.fullmatch(r"(?:buffer_load_(?:dword(?:x(?P<raw_count>[234]))?|(?P<subword>[us](?:byte|short))"
                               r"|(?P<d16_kind>[us]byte|short)_d16(?P<hi>_hi)?)"
                               r"|tbuffer_load_format_(?P<channels>(?:d16_)?(?:x|xy|xyz|xyzw))"
                               r"|buffer_load_format_(?P<resource_channels>(?:d16_)?(?:x|xy|xyz|xyzw))) "
                               r"(?P<dst>v\d+|v\[\d+:\d+\]), (?P<addr>off|v\d+), s\[(?P<base>\d+):(?P<high>\d+)\], 0"
                               r"(?(channels) format:\[(?P<format>BUF_FMT_(?:(?:32_){1,4}|(?:16_){1,4})UINT)\])"
                               r"(?P<idxen> idxen)?(?P<offen> offen)?"
                               r"(?: offset:(?P<offset>\d+))?", line):
            channels, resource_channels = m["channels"], m["resource_channels"]
            suffix = channels or resource_channels
            packed = suffix is not None and suffix.startswith("d16_")
            raw_count = m["raw_count"]
            components = len(suffix.removeprefix("d16_")) if suffix else int(raw_count or 1)
            count = (components + 1) // 2 if packed else components
            dst, *last = map(int, re.findall(r"\d+", m["dst"]))
            base, high = int(m["base"]), int(m["high"])
            offset = int(m["offset"] or 0)
            addr, idxen, offen = m["addr"], m["idxen"], m["offen"]
            d16_kind = m["d16_kind"]
            subword = m["subword"] or d16_kind
            width = 4 if subword is None else 1 if subword.endswith("byte") else 2
            if subword and (cpu != "gfx1030" or not idxen or offen):
                raise ValueError("Unsupported subword buffer read target or address mode")
            if raw_count and (cpu != "gfx1030" or not idxen or offen):
                raise ValueError("Unsupported vector buffer read target or address mode")
            if channels and (cpu != "gfx1030" or not idxen or offen):
                raise ValueError("Unsupported typed buffer read target or address mode")
            descriptor_load = None
            if resource_channels and load_format_words is not None:
                # The format is V# word 3 from a constant word, decoded in the shader (descriptor_load_hlsl).
                if not idxen or offen:
                    raise ValueError("Unsupported formatted buffer read target or address mode")
                if format_load_index >= len(load_format_words):
                    raise ValueError("Formatted buffer load is missing a format constant")
                descriptor_load = load_format_words[format_load_index]
                format_load_index += 1
                if not any(tag == ("constant_word", descriptor_load) and reg not in pending
                           for reg, tag in scalar.items()):
                    raise ValueError("Formatted buffer load format constant is not ready")
            elif resource_channels and heap_view is not None:
                # The V# comes from the view record at the slot named by a constant word (heap_view_hlsl).
                if packed:
                    raise ValueError("Descriptor-format d16 loads are not lowered")
                if not idxen or offen:
                    raise ValueError("Unsupported formatted buffer read target or address mode")
                descriptor_load = heap_slot_word
                format_load_index += 1
                if not any(tag == ("constant_word", descriptor_load) and reg not in pending
                           for reg, tag in scalar.items()):
                    raise ValueError("Heap view slot constant is not ready")
            elif resource_channels:
                if cpu != "gfx1030" or not idxen or offen:
                    raise ValueError("Unsupported formatted buffer read target or address mode")
                if source_format is None:
                    raise ValueError("Formatted buffer reads require a source format")
                expected = MUBUF_D16_SOURCE_FORMAT if packed else MUBUF_SOURCE_FORMAT
                if source_format != expected:
                    raise ValueError("Formatted buffer read does not match source format")
            # The three-component D16 form uses the four-component 16-bit format.
            format_components = 4 if packed and components == 3 else components
            if channels and m["format"] != f"BUF_FMT_{('16_' if packed else '32_') * format_components}UINT":
                raise ValueError("Unsupported typed buffer read format")
            if last != ([] if count == 1 else [dst + count - 1]) or dst + count > 256:
                raise ValueError("Unverified buffer read destination")
            old = vector.get(dst) if d16_kind else None
            if d16_kind and not is_word(old):
                raise ValueError("Unverified D16 buffer read destination input")
            if (high != base + 3 or base % 4 or high > 105
                    or pending.intersection(range(base, high + 1))):
                raise ValueError("Unverified buffer read descriptor")
            kinds = tuple(scalar.get(base + i) for i in range(4))
            if kinds == tuple(("uav", i) for i in range(4)):
                # The store target would need ordering against other invocations' stores.
                raise ValueError("Reads from the stored UAV are not supported")
            if kinds != tuple(("srv", i) for i in range(4)):
                raise ValueError("Unverified buffer read descriptor")
            # A descriptor-format load reads whole dwords from a 4-byte-aligned address.
            alignment = 4 if descriptor_load is not None else source_texel_bytes if resource_channels else width
            if offset % alignment or offset > 4095:
                raise ValueError("Unverified read offset")
            tag = vector.get(int(addr[1:])) if addr != "off" else None
            if addr == "off" and not idxen and not offen:
                addresses = [f"{offset}u"]
            elif (offen and not idxen and isinstance(tag, tuple) and tag[0] == "thread_bytes"
                  and 2 <= tag[2] < 32):
                # A shift below 2 gives unaligned byte offsets, which ByteAddressBuffer.Load cannot express.
                addresses = [f"(tid.{AXES[tag[1]]} << {tag[2]}u) + {offset}u"]
            elif (idxen and not offen and isinstance(tag, tuple) and tag[0] == "thread"
                  and heap_view is not None and descriptor_load is not None):
                # Only the formatted heap-view helper obtains a runtime stride and bounds the read.
                aligned = offset & ~3
            elif (idxen and not offen and source_stride is not None
                  and isinstance(tag, tuple) and tag[0] == "thread"):
                # idxen: index * stride + instruction offset, inside one source element.
                # A formatted load selects a prefix of a complete four-component texel.
                # A descriptor-format load uses at most four bytes per channel of the opcode.
                read_bytes = (4 * components if descriptor_load is not None
                              else source_texel_bytes if resource_channels else width * count)
                if offset + read_bytes > source_stride:
                    raise ValueError("Read leaves its source element")
                aligned = offset & ~3
                addresses = [f"tid.{AXES[tag[1]]} * {source_stride}u + {aligned + 4*i}u" for i in range(count)]
            else:
                raise ValueError("Unsupported read address")
            if descriptor_load is not None:
                result = f"format_load_{format_load_index - 1}"
                slot = hlsl_word(('constant_word', descriptor_load))
                if heap_view is not None and packed:
                    raise ValueError("D16 descriptor-format loads are not lowered through a heap view")
                call = (f"load_heap_view_format({slot}, tid.{AXES[tag[1]]}, {aligned}u, {components}u, "
                        f"format_valid_{format_load_index - 1})" if heap_view is not None else
                        f"load_descriptor_format({slot}, {addresses[0]}, {components}u, {int(packed)}u, "
                        f"format_valid_{format_load_index - 1})")
                staged_steps.append((None, f"    bool format_valid_{format_load_index - 1};\n"
                                           f"    uint4 {result} = {call};"))
                for reg in range(dst, dst + count):
                    vector.pop(reg, None)
                    # An odd d16 count leaves the high half of the last word undefined, as for source_low16.
                    low16 = packed and components % 2 and reg == dst + count - 1
                    loading[reg] = ("alu_low16" if low16 else "alu", f"{result}.{'xyzw'[reg - dst]}")
                continue
            # The address has been captured, so the index can alias a destination.
            for reg, address in enumerate(addresses, dst):
                vector.pop(reg, None)
                # Whole-word consumers cannot read the undefined upper half of an odd D16 result.
                kind = "source_low16" if packed and components % 2 and reg == dst + count - 1 else "source"
                word = (kind, address)
                if subword:
                    # The stride and SRV base are DWORD-aligned, so the enclosing word stays in the record.
                    bits, shift = width * 8, (offset % 4) * 8
                    field = f"(({hlsl_word(word)} >> {shift}u) & {((1 << bits) - 1):#x}u)"
                    if subword in ("sbyte", "sshort"):
                        sign = f"{1 << (bits - 1):#x}u"
                        field = f"(({field} ^ {sign}) - {sign})"
                    if d16_kind:
                        hi = bool(m["hi"])
                        keep = "0xffffu" if hi else "0xffff0000u"
                        field = f"(({hlsl_word(old)} & {keep}) | (({field} & 0xffffu) << {16 if hi else 0}u))"
                        if len(field) > ALU_EXPRESSION_LIMIT:
                            raise ValueError("D16 buffer read expression is too large")
                    word = ("alu", field)
                loading[reg] = word
        elif m := re.fullmatch(r"image_msaa_load v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], "
                               r"s\[(\d+):(\d+)\] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm", line):
            first, last, addr, addr_last, base, high = map(int, m.groups())
            if not image_msaa or guard_span is not None:
                raise ValueError("image_msaa_load requires an unguarded R32_UINT MSAA binding")
            if last != first + 3 or addr_last != addr + 2 or max(last, addr_last) > 255:
                raise ValueError("Invalid image_msaa_load VGPR tuple")
            if high != base + 7 or base % 4 or pending.intersection(range(base, high + 1)):
                raise ValueError("Unresolved image descriptor")
            if tuple(scalar.get(base + index) for index in range(8)) != tuple(
                    (("image", index // 4), index % 4) for index in range(8)):
                raise ValueError("Unverified image descriptor")
            x, y, fragment = (vector.get(reg) for reg in range(addr, addr_last + 1))
            if not is_word(x) or not is_word(y):
                raise ValueError("Unverified image_msaa_load coordinates")
            if fragment != 0:
                raise ValueError("image_msaa_load requires a known zero fragment ID")
            # Snapshot both coordinates before publishing any result, including aliased VADDR/VDATA.
            address = f"int2(asint({hlsl_word(x)}), asint({hlsl_word(y)}))"
            for sample in range(4):
                name = f"image_msaa_{image_step_count}_{sample}"
                staged_steps.append((None, f"    uint {name} = imageData.Load({address}, {sample});"))
                reg = first + sample
                vector.pop(reg, None)
                loading[reg] = ("alu", name)
            image_step_count += 1
        elif m := re.fullmatch(rf"image_atomic_(?P<op>{IMAGE_ATOMIC_OPS}) "
                               r"v(?:(?P<single>\d+)|\[(?P<first>\d+):(?P<last>\d+)\]), "
                               r"v\[(?P<x>\d+):(?P<y>\d+)\], s\[(?P<base>\d+):(?P<high>\d+)\] "
                               r"dmask:0x(?P<mask>[0-9a-f]+) dim:SQ_RSRC_IMG_2D(?: unorm)?", line):
            op = m["op"]
            signed = op in ("smin", "smax")
            # gfx1013 is limited to the forms its draws cover; the lowering itself is the gfx1030 one.
            if cpu == "gfx1013" and op not in ("add", "umin", "umax", "cmpswap"):
                raise ValueError("gfx1013 image atomics require add, umin, umax or cmpswap")
            if (image_atomic_type != ("int" if signed else "uint") or image_float != (op in IMAGE_FLOAT_ATOMIC_OPS)
                    or guard_span is not None or stores):
                raise ValueError("Unsupported image atomic target or binding")
            compare_swap = op in ("cmpswap", "fcmpswap")
            data = int(m["single"] or m["first"])
            if (int(m["mask"], 16) != (3 if compare_swap else 1) or data + int(compare_swap) > 255
                    or (compare_swap and (m["last"] is None or int(m["last"]) != data + 1))
                    or (not compare_swap and m["single"] is None)):
                raise ValueError("Unverified image atomic data range or mask")
            base, high = int(m["base"]), int(m["high"])
            if (high != base + 7 or base % 4 or pending.intersection(range(base, high + 1))
                    or tuple(scalar.get(base + index) for index in range(8)) != tuple(
                        (("image", index // 4), index % 4) for index in range(8))):
                raise ValueError("Unverified image atomic descriptor")
            x, y = int(m["x"]), int(m["y"])
            coords = [vector.get(reg) for reg in (x, y)]
            if y != x + 1 or y > 255 or not all(map(is_word, coords)):
                raise ValueError("Unverified image atomic coordinates")
            sources = [vector.get(data + index) for index in range(1 + compare_swap)]
            if not all(map(is_word, sources)):
                raise ValueError("Unverified image atomic data")
            destination = f"outputData[uint2({', '.join(map(hlsl_word, coords))})]"
            value = hlsl_word(sources[0])
            if op in ("inc", "dec"):
                condition = "expected >= limit" if op == "inc" else "expected == 0u || expected > limit"
                wrap, step = ("0u", "+") if op == "inc" else ("limit", "-")
                # A failed CAS supplies the observed value for the next attempt.
                stores.extend((f"    uint limit = {value};", "    uint observed = 0u;", "    uint expected;",
                               "    [allow_uav_condition] do {", "        expected = observed;",
                               f"        uint next = ({condition}) ? {wrap} : expected {step} 1u;",
                               f"        InterlockedCompareExchange({destination}, expected, next, observed);",
                               "    } while (observed != expected);"))
            elif op in ("fmin", "fmax"):
                stores.extend(float_min_max_cas(f"InterlockedCompareExchange({destination}", value, op))
            elif compare_swap:
                # RDNA stores VDATA on a match against VDATA+1. HLSL takes the comparison first.
                # Numeric equality and bit equality agree for finite-normal binary32.
                stores.append(f"    InterlockedCompareStore({destination}, {hlsl_word(sources[1])}, {value});")
            else:
                if op == "swap":
                    stores.append("    uint discardedOldValue;")
                    value += ", discardedOldValue"
                elif op == "sub":
                    value = f"(0u - {value})"
                elif signed:
                    value = f"asint({value})"
                stores.append(f"    Interlocked{MUBUF_ATOMIC_METHODS[op]}({destination}, {value});")
        elif m := re.match(r"image_gather4(\w*) ", line):
            if m[1] != "_l":
                raise ValueError(f"Unsupported gather form image_gather4{m[1]}: only 2D image_gather4_l is lowered")
            m = re.fullmatch(r"image_gather4_l v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], s\[(\d+):(\d+)\], "
                             r"s\[(\d+):(\d+)\] dmask:0x([0-9a-f]+) dim:SQ_RSRC_IMG_2D", line)
            if m is None:
                raise ValueError("Unsupported image_gather4_l form: requires a 2D view, contiguous VGPRs "
                                 "and no other modifier")
            if gather_mips is None or guard_span is not None:
                raise ValueError("image_gather4_l requires an unguarded lane and a declared gather view")
            dst_first, dst_last, addr_first, addr_last, base, high, state_first, state_last = map(int, m.groups()[:8])
            component = {"1": 0, "2": 1, "4": 2, "8": 3}.get(m[9])
            if component is None:
                raise ValueError("image_gather4_l requires a one-hot DMASK")
            # RDNA 2 MIMG: four result VGPRs and U, V, LOD in consecutive address VGPRs.
            if dst_last != dst_first + 3 or dst_last > 255 or addr_last != addr_first + 2 or addr_last > 255:
                raise ValueError("Invalid image_gather4_l register range")
            if high != base + 7 or base % 4 or pending.intersection(range(base, high + 1)):
                raise ValueError("Unresolved image descriptor")
            if tuple(scalar.get(base + index) for index in range(8)) != tuple(
                    (("image", index // 4), index % 4) for index in range(8)):
                raise ValueError("Unverified image descriptor")
            words = [scalar.get(reg) if reg not in pending else None for reg in range(state_first, state_last + 1)]
            if (state_last != state_first + 3 or state_first % 4 or state_last > 105
                    or not all(isinstance(word, int) for word in words)):
                raise ValueError("image_gather4_l requires a constant sampler")
            check_gather_sampler(words, gather_mips, comparison=False)
            u, v, lod = (vector.get(reg) for reg in range(addr_first, addr_last + 1))
            if not (is_word(u) and is_word(v)):
                raise ValueError("Unverified image coordinates")
            # Point mip filtering of an integral LOD selects that view mip on both sides; anything else is refused.
            level = struct.unpack("<f", struct.pack("<I", lod))[0] if isinstance(lod, int) else None
            if level is None or not level.is_integer() or not 0 <= level < gather_mips:
                raise ValueError("image_gather4_l requires a known integer LOD inside the view")
            name = f"image_gather_{image_step_count}"
            image_step_count += 1
            staged_steps.append((None, f"    float2 {name}_uv = asfloat(uint2({hlsl_word(u)}, {hlsl_word(v)}));\n"
                                       f"    uint4 {name} = image_gather_2d({name}_uv, {int(level)}u, {component}u);"))
            gather_steps.append(name)
            for index, reg in enumerate(range(dst_first, dst_last + 1)):
                vector.pop(reg, None)
                loading[reg] = ("alu", f"{name}.{'xyzw'[index]}")
        elif m := re.fullmatch(r"image_(?P<kind>load|store|get_resinfo)(?P<mip>_mip)?(?P<packed>_pck(?:_sgn)?)? "
                               r"(?P<data>v\d+|v\[\d+:\d+\]), (?P<addr>v\d+|v\[\d+:\d+\]), "
                               r"s\[(?P<base>\d+):(?P<high>\d+)\] dmask:0x(?P<mask>[0-9a-f]+) "
                               r"dim:SQ_RSRC_IMG_(?P<dimension>1D|2D_ARRAY|2D|3D)(?: unorm)?(?P<a16> a16)?(?P<d16> d16)?", line):
            # A16 packs x/y into one dword and takes the mip from the low half of the next dword.
            mip, a16 = bool(m["mip"]), bool(m["a16"])
            d16 = bool(m["d16"])
            write = m["kind"] == "store"
            resinfo = m["kind"] == "get_resinfo"
            dimension = m["dimension"].lower().replace("_", "-")
            one_dimensional = dimension == "1d"
            layered = dimension in ("2d-array", "3d")
            if m["d16"] and dimension != "2d":
                # D16 stores are 2D only; refuse other dimensions as an unmatched instruction.
                raise ValueError(f"Unsupported compute instruction: {line}")
            if layered and not write:
                raise ValueError(f"Unsupported compute instruction: {line}")
            view_dimension = image_uav_dimension if write else image_srv_dimension
            if dimension != view_dimension:
                view = "UAV" if write else "SRV"
                raise ValueError(f"Unsupported compute instruction: image_{m['kind']}{'_mip' if mip else ''} "
                                 f"dimension {dimension} does not match the {view_dimension} {view} view")
            if layered and (not rgba32_float_image or a16 or mip or m["packed"]):
                raise ValueError(f"Unsupported compute instruction: {line}")
            vaddr = [int(n) for n in re.findall(r"\d+", m["addr"])]
            # RDNA 2 ISA §8.2.3, Table 42 and LLVM 15 agree: full-width 1D stores take
            # one address VGPR (X), and 1D mip loads take two (X, then mip).
            address_words = 1 if resinfo else (1 if a16 or one_dimensional else 2) + int(mip) + int(layered)
            base, high = int(m["base"]), int(m["high"])
            dmask = int(m["mask"], 16)
            # DMASK selects components in XYZW order and packs them into consecutive VGPRs.
            channels = [component for index, component in enumerate("xyzw") if dmask & (1 << index)]
            data_words = 2 if d16 else len(channels)
            data = [int(n) for n in re.findall(r"\d+", m["data"])]
            first, last = data[0], data[-1]
            uav_load = rgba32_float_image and image_output and m["kind"] == "load" and not mip and dimension == "2d"
            if d16:
                if not write or a16 or mip or m["packed"]:
                    raise ValueError("Unsupported compute instruction: D16 requires a 2D image_store "
                                     "with full-width coordinates and implicit mip zero")
                if not rgba16_float_image:
                    raise ValueError("Unsupported compute instruction: D16 image_store requires a gfx1013 RGBA16 FLOAT UAV")
                if dmask != 0xf:
                    raise ValueError("D16 image_store requires full DMASK 0xf")
                # RDNA 2 ISA §§8.2.1 and 8.2.5 pack two D16 components per VGPR.
                # Full DMASK supplies R/G in the low/high halves of VDATA and
                # B/A in VDATA+1. LLVM 15 requires a two-VGPR data range.
                if len(data) != 2 or last != first + 1 or last > 255:
                    raise ValueError("D16 image_store requires exactly two consecutive data VGPRs")
            elif rgba16_float_image:
                raise ValueError("RGBA16 FLOAT image UAV requires a D16 image_store")
            elif uav_load:
                if a16 or m["packed"] or image_uav_mip != 0:
                    raise ValueError("Unsupported image load target: image UAV load requires full-width mip-zero access")
                if dmask != 0xf:
                    raise ValueError("Unsupported image load target: image UAV load requires full DMASK 0xf")
                if stores:
                    raise ValueError("Image UAV load must precede the image store")
                if image_uav_address is not None:
                    raise ValueError("Only one image UAV load is supported")
            elif cpu == "gfx1013" and (not rgba32_float_image or a16 or m["packed"]
                                     or (m["kind"], mip) not in (("store", False), ("load", True))):
                raise ValueError(f"Unsupported image {m['kind']} target: "
                                 "gfx1013 requires full-width RGBA32 FLOAT image_store or image_load_mip")
            if rgba32_float_image and write and dmask != 0xf:
                raise ValueError("gfx1013 RGBA32 FLOAT image_store requires full DMASK 0xf")
            if one_dimensional and not write and dmask != 0xf:
                raise ValueError("gfx1013 RGBA32 FLOAT 1D image_load_mip requires full DMASK 0xf")
            separate_srv = not write and not uav_load and image_srv_descriptor is not None
            if rgba32_float_image and image_output and m["kind"] == "load" and mip and not separate_srv:
                raise ValueError("Unsupported image load target: image_load_mip requires a separate SRV")
            if ((image_format not in IMAGE_FORMATS and not float_image) or guard_span is not None
                    or (write != image_output and not separate_srv and not uav_load)
                    or len(vaddr) != min(address_words, 2) or vaddr[-1] != vaddr[0] + address_words - 1
                    or vaddr[-1] > 255 or last != first + data_words - 1
                    or not 0 < dmask < (1 << (4 if float_image else IMAGE_FORMATS[image_format][1]))
                    or any(first + index > 255 or first + index in loading for index in range(data_words))
                    or (write and (m["packed"] == "_pck_sgn" or stores))
                    or (resinfo and (mip or m["packed"] or dmask & 0x4))):
                raise ValueError(f"Unsupported image {m['kind']} target")
            if resinfo and image_format != "R32G32B32A32_UINT":
                raise ValueError("Image get_resinfo requires an RGBA32 SRV")
            if bool(m["packed"]) != (image_format == "R8G8B8A8_UINT"):
                raise ValueError(f"Image {m['kind']} opcode does not match the image format")
            if high != base + 7 or base % 4 or pending.intersection(range(base, high + 1)):
                raise ValueError("Unresolved image descriptor")
            binding_kind = "image_srv" if separate_srv else "image"
            if tuple(scalar.get(base + index) for index in range(8)) != tuple(
                    ((binding_kind, index // 4), index % 4) for index in range(8)):
                raise ValueError("Unverified image descriptor")
            coordinates = [vector.get(reg) for reg in range(vaddr[0], vaddr[-1] + 1)]
            if not all(is_word(value) for value in coordinates):
                raise ValueError("Unverified image coordinates")
            if not resinfo:
                if a16:
                    packed = hlsl_word(coordinates[0])
                    x, y = f"{packed} & 0xffffu", f"{packed} >> 16u"
                elif one_dimensional:
                    x = hlsl_word(coordinates[0])
                else:
                    x, y = map(hlsl_word, coordinates[:2])
            if write:
                # A D3D12 image UAV names one mip. Verify native and desktop mip selection agree.
                level = coordinates[-1] if mip else 0
                if not isinstance(level, int) or (level & 0xffff if a16 else level) != image_uav_mip:
                    raise ValueError("Image store mip does not match the UAV view")
                if image_uav_address is not None:
                    if image_uav_address != (vaddr, coordinates):
                        raise ValueError("Image UAV load and store must use the same coordinate VGPRs and values")
                    # Dispatch X/Y are injective in a single Z plane. The caller must
                    # dispatch one Z group, and the shader must have one local Z lane.
                    if coordinates != [("thread", 0), ("thread", 1)] or dims[2] != 1:
                        raise ValueError("Image UAV load/store requires unique dispatch X/Y coordinates and local Z size 1")
                values = [vector.get(first + index) for index in range(data_words)]
                if not all(is_word(value) for value in values):
                    raise ValueError("Unverified image store data")
                words = list(map(hlsl_word, values))
                if m["packed"]:
                    # Raw RGBA8 stores take the low byte of each enabled channel's VGPR.
                    words = [f"({word} & 0xffu)" for word in words]
                if d16:
                    stores.append(f"    uint2 image_store_data = uint2({', '.join(words)});")
                    value = ("float4(f16tof32(image_store_data.x & 0xffffu), f16tof32(image_store_data.x >> 16u), "
                             "f16tof32(image_store_data.y & 0xffffu), f16tof32(image_store_data.y >> 16u))")
                elif rgba32_float_image:
                    value = f"asfloat(uint4({', '.join(words)}))"
                else:
                    selected = dict(zip(channels, words))
                    texel = ", ".join(selected.get(channel, "0u") for channel in "xyzw")
                    value = f"uint4({texel})"
                if one_dimensional:
                    address = x
                elif layered:
                    address = f"uint3({x}, {y}, {hlsl_word(coordinates[2])})"
                else:
                    address = f"uint2({x}, {y})"
                stores.append(f"    outputData[{address}] = {value};")
                continue
            if resinfo:
                name = f"image_resinfo_{image_step_count}"
                level = hlsl_word(coordinates[0])
                if a16:
                    level = f"({level} & 0xffffu)"
                staged_steps.append((None, f"    uint4 {name};\n"
                                     f"    imageData.GetDimensions({level}, {name}.x, {name}.y, {name}.w);"))
            else:
                name = f"image_load_{image_step_count}"
                if uav_load:
                    image_uav_address = (vaddr, coordinates)
                    load = f"outputData[uint2({x}, {y})]"
                else:
                    lod = "0"
                    if mip:
                        mip_word = hlsl_word(coordinates[-1])
                        lod = f"asint({mip_word}{' & 0xffffu' if a16 else ''})"
                    address = (f"int2(asint({x}), {lod})" if one_dimensional
                               else f"int3(asint({x}), asint({y}), {lod})")
                    load = f"imageData.Load({address})"
                if rgba32_float_image:
                    staged_steps.append((None, f"    uint4 {name} = asuint({load});"))
                else:
                    staged_steps.append((None, f"    {IMAGE_FORMATS[image_format][0]} {name} = {load};"))
            image_step_count += 1
            if m["packed"] and m["packed"].endswith("_sgn"):
                # The UINT SRV exposes raw bytes. Extend each byte from bit 7 to a full VGPR.
                staged_steps.append((None, f"    {name} = asuint(asint({name} << 24u) >> 24u);"))
            for index, component in enumerate(channels):
                reg = first + index
                vector.pop(reg, None)
                loading[reg] = ("alu", f"{name}.{component}")
        elif m := re.fullmatch(rf"buffer_atomic_(?P<op>{'|'.join(MUBUF_ATOMIC_METHODS)})(?P<wide>_x2)? "
                               r"(?P<data>v\d+|v\[\d+:\d+\]), (?P<addr>off|v\d+|v\[\d+:\d+\]), "
                               r"s\[(?P<base>\d+):(?P<high>\d+)\], 0"
                               r"(?P<idxen> idxen)?(?P<offen> offen)?(?: offset:(?P<offset>\d+))?(?P<glc> glc)?", line):
            op, wide, glc = m["op"], bool(m["wide"]), bool(m["glc"])
            floating = op in IMAGE_FLOAT_ATOMIC_OPS
            float_extended = floating and (wide or op == "fcmpswap")
            fixed = m["addr"] == "off" and not (m["idxen"] or m["offen"])
            if cpu == "gfx1030" and not fixed:
                raise ValueError(f"Unsupported compute instruction: {line}")
            if output_format is not None or image_output or (cpu == "gfx1030" and stride is not None):
                raise ValueError("Unsupported atomic reduction target or binding")
            # One atomic per program, and no UAV write before it. The atomic itself is appended to
            # stores, so a second atomic is refused here too. That keeps the single atomicOld or
            # ignoredAtomic declaration below unique. On gfx1013, plain DWORD stores of the
            # returned value may follow it.
            if stores:
                raise ValueError("An atomic reduction must be the only UAV operation")
            if cpu == "gfx1013":
                if wide or op not in ("add", "umax", "cmpswap"):
                    raise ValueError("Unsupported atomic reduction target or binding: "
                                     "gfx1013 buffer atomics require 32-bit add, umax or cmpswap")
                if guard_span is not None or image_descriptor is not None:
                    raise ValueError("Unsupported atomic reduction target or binding: "
                                     "gfx1013 requires an unguarded raw-buffer program")
            else:
                if float_extended and (buffer_float_domain != "finite-normal" or glc):
                    raise ValueError("Float buffer compare-swap and _x2 require a finite-normal domain and no glc")
                if floating and not float_extended and (buffer_float_domain != "finite-normal" or glc):
                    raise ValueError("buffer_atomic fmin/fmax require a finite-normal domain, "
                                     "a single 32-bit operand and no glc")
                if op == "csub" and (wide or not glc):
                    raise ValueError("Conditional subtract requires 32-bit glc")
                if op != "csub" and glc:
                    raise ValueError(f"Unsupported compute instruction: {line}")
            base, high = int(m["base"]), int(m["high"])
            offset = int(m["offset"] or 0)
            if (high != base + 3 or base % 4 or pending.intersection(range(base, high + 1))
                    or any(scalar.get(base + i) != ("uav", i) for i in range(4))):
                raise ValueError("Unverified atomic reduction descriptor")
            if offset > 4095 or offset % (8 if wide else 4):
                raise ValueError("Unverified atomic reduction address")
            address = f"{offset}u"
            if cpu == "gfx1013":
                if fixed:
                    shared_atomic_offset = offset
                else:
                    tag = vector.get(int(m["addr"][1:])) if re.fullmatch(r"v\d+", m["addr"]) else None
                    access = linear_uav_address(tag, dims, m["idxen"], m["offen"], stride, offset, 4)
                    if access is None:
                        raise ValueError("Unverified per-thread atomic address")
                    layout, axis, address = access
                    if any(dims[other] > 1 for other in range(3) if other != axis):
                        raise ValueError("Atomic address does not separate every invocation of the group")
                    written_spans.append((offset, offset + 4))
            src, *last = map(int, re.findall(r"\d+", m["data"]))
            width = 2 if wide else 1
            compare_swap = op in ("cmpswap", "fcmpswap")
            count = 2 * width if compare_swap else width
            if last != ([src + count - 1] if count > 1 else []) or src + count > 256:
                raise ValueError("Unverified atomic reduction data range")
            values = [vector.get(reg) for reg in range(src, src + count)]
            if not all(is_word(value) for value in values):
                raise ValueError("Atomic reduction data is not a known word")
            if float_extended:
                shift, exponent = (20, 0x7ff) if wide else (23, 0xff)
                if not all(isinstance(value, int) for value in values) or any(
                        ((high_word >> shift) & exponent) in (0, exponent)
                        for high_word in values[width - 1::width]):
                    raise ValueError("Float buffer atomic operands must be finite-normal constants")
            words = list(map(hlsl_word, values))
            operands = ([f"(uint64_t({lo}) | (uint64_t({hi}) << 32u))" for lo, hi in zip(words[::2], words[1::2])]
                        if wide else words)
            word = operands[0]
            signed = op in ("smin", "smax")
            if signed:
                word = f"int64_t({word})" if wide else f"asint({word})"
            elif op == "sub":
                word = f"({'0ull' if wide else '0u'} - {word})"
            # Raw UAV atomics on pairs require SM 6.6.
            kind = ("int64_t" if signed else "uint64_t") if wide else ("int" if signed else "uint")
            method = MUBUF_ATOMIC_METHODS[op] + ("64" if wide else "")
            if compare_swap:
                # RDNA 2 takes replacement first, then compare, each in one word or a pair.
                # HLSL takes compare first. Finite-normal float equality is bit equality.
                word = f"{operands[1]}, {word}"
            if op in ("fmin", "fmax"):
                stores.extend(float_min_max_cas(f"outputData.Interlocked{method}({offset}u", word, op, wide=wide))
            elif op in ("inc", "dec", "csub"):
                one, zero = ("1ull", "0ull") if wide else ("1u", "0u")
                if op == "inc":
                    desired = f"expected >= limit ? {zero} : expected + {one}"
                elif op == "dec":
                    desired = f"expected == {zero} || expected > limit ? limit : expected - {one}"
                else:
                    desired = "expected < limit ? 0u : expected - limit"
                    # GLC overwrites the data VGPR. Track its wait and reject consumers of the
                    # old value until the profile can emit a value-producing atomic.
                    vector.pop(src, None)
                    loading[src] = None
                # Seed the first compare with zero. A failed CAS returns the word for the next try.
                stores.extend((f"    const {kind} limit = {word};", f"    {kind} observed = {zero};",
                               f"    {kind} expected;", "    [allow_uav_condition] do {",
                               "        expected = observed;", f"        {kind} next = {desired};",
                               f"        outputData.Interlocked{method}({offset}u, expected, next, observed);",
                               "    } while (observed != expected);"))
            else:
                result = "atomicOld" if glc else "ignoredAtomic"
                stores.extend((f"    {kind} {result};",
                               f"    outputData.Interlocked{method}({address}, {word}, {result});"))
                if cpu == "gfx1013":
                    # Without GLC the source is unchanged. This profile refuses consumers of
                    # that register until an explicit rewrite instead of supplying an old value.
                    vector.pop(src, None)
                    if glc:
                        loading[src] = ("alu", result)
            atomic_update = True
            if floating:
                buffer_float_update = True
        elif m := re.fullmatch(r"(?:buffer_store_(?:dword(?:x(?P<raw_count>[234]))?|(?P<subword>byte|short)(?P<store_hi>_d16_hi)?)|"
                               r"(?P<typed>t)?buffer_store_format_(?P<channels>(?:d16_)?(?:x|xy|xyz|xyzw))) "
                               r"(?P<data>v\d+|v\[\d+:\d+\]), (?P<addr>off|v\d+|v\[\d+:\d+\]), "
                               r"s\[(?P<base>\d+):(?P<high>\d+)\], 0"
                               r"(?(typed) format:\[(?P<format>BUF_FMT_(?:(?:32_){1,4}|(?:16_){1,4})UINT)\])"
                               r"(?P<idxen> idxen)?(?P<offen> offen)?(?: offset:(?P<offset>\d+))?", line):
            if atomic_update and (cpu == "gfx1030" or not line.startswith("buffer_store_dword ")):
                raise ValueError("An atomic reduction must be the only UAV operation")
            if image_output:
                raise ValueError("Buffer stores require a raw UAV binding")
            channels = m["channels"]
            packed = channels is not None and channels.startswith("d16_")
            components = len(channels.removeprefix("d16_")) if channels else int(m["raw_count"] or 1)
            partial = packed and components % 2 != 0
            count = (components + 1) // 2 if packed else components
            idxen, offen = m["idxen"], m["offen"]
            subword = m["subword"]
            if subword and (cpu != "gfx1030" or not idxen or offen or stride is None):
                raise ValueError("Unsupported subword buffer store target or address mode")
            width = {"byte": 1, "short": 2}.get(subword, 4)
            format_components = 4 if packed and components == 3 else components
            fmt = f"BUF_FMT_{('16_' if packed else '32_') * format_components}UINT"
            if m["typed"] and (cpu != "gfx1030" or not idxen or offen or m["format"] != fmt):
                raise ValueError("Unsupported typed buffer store format, target or address mode")
            formatted = channels is not None and not m["typed"]
            descriptor_store = False
            format_index = None
            if formatted:
                if not idxen or offen:
                    raise ValueError("Unsupported formatted buffer store target or address mode")
                if heap_store_view is not None:
                    if channels != "xyzw" and not packed:
                        raise ValueError("Heap store view requires buffer_store_format_xyzw or a D16 formatted store")
                    if not any(tag == ("constant_word", heap_slot_word) and reg not in pending
                               for reg, tag in scalar.items()):
                        raise ValueError("Heap store view slot constant is not ready")
                elif format_words is not None:
                    if format_store_index >= len(format_words):
                        raise ValueError("Formatted buffer store is missing a format constant")
                    format_index = format_words[format_store_index]
                    format_store_index += 1
                    if not any(tag == ("constant_word", format_index) and reg not in pending
                               for reg, tag in scalar.items()):
                        raise ValueError("Formatted buffer store format constant is not ready")
                    descriptor_store = True
                else:
                    if cpu != "gfx1030":
                        raise ValueError("Unsupported formatted buffer store target or address mode")
                    if output_format is None:
                        raise ValueError("Formatted buffer stores require an output format")
                    if packed and components == 3 and mubuf_d16_xyz_mode != "preserve-w":
                        raise ValueError("Formatted D16 XYZ store requires an explicit W-preservation contract")
                    expected = (MUBUF_D16_OUTPUT_FORMATS[format_components] if packed
                                else MUBUF_UINT_FORMATS[components - 1])
                    if output_format != expected:
                        raise ValueError("Formatted buffer store width does not match output format")
            first, *last = map(int, re.findall(r"\d+", m["data"]))
            base, high = int(m["base"]), int(m["high"])
            offset = int(m["offset"] or 0)
            # A descriptor-format element may begin on a halfword. The byte offset selects
            # which 16-bit lane a single-channel store updates.
            align = 2 if descriptor_store else width
            if (last != ([] if count == 1 else [first + count - 1])
                    or (formatted and first + count > 256)
                    or high != base + 3 or base % 4 or offset % align or offset > 4095
                    or pending.intersection(range(base, high + 1))
                    or any(scalar.get(base + i) != ("uav", i) for i in range(4))):
                raise ValueError("Unverified raw UAV store")
            byte_shift = (offset % 4) * 8
            # Reserve and address the containing DWORD for a subword store. A descriptor-format
            # store keeps its byte offset so a halfword element is not moved to the low lane.
            if not descriptor_store:
                offset &= ~3
            # The format word is not known here, so a descriptor store reserves four bytes per
            # channel. d16_xyzw is two VGPRs and sixteen bytes. The caller-declared UINT path
            # still reserves one DWORD per data VGPR.
            span = 4 * (components if descriptor_store or heap_store_view is not None else count)
            address = f"{offset}u"
            addr = m["addr"]
            addr_regs = list(map(int, re.findall(r"\d+", addr)))
            tags = [vector.get(reg) for reg in addr_regs]
            linear = (linear_uav_address(tags[0], dims, idxen, offen, stride, offset, span)
                      if heap_store_view is None and len(tags) == 1 and (idxen or atomic_update) else None)
            if heap_store_view is not None:
                if not formatted or len(tags) != 1 or not isinstance(tags[0], tuple) or tags[0][0] != "thread":
                    raise ValueError("Heap store view needs a formatted store indexed by dispatch thread ID")
                axis = tags[0][1]
                store_layout, covered = ("heap_index", axis), {axis}
                address = f"tid.{AXES[axis]}"
            elif addr == "off" and not idxen and not offen and dims == (1, 1, 1) and stride is None:
                store_layout = ("fixed",)
                covered = set()
            elif linear is not None:
                # One index axis: every other group dimension must be one, so no two
                # invocations of a group share an address.
                store_layout, index, address = linear
                covered = {index}
            elif (stride is not None and idxen and offen and len(addr_regs) == 2 and addr_regs[1] == addr_regs[0] + 1
                    and all(isinstance(tag, tuple) for tag in tags)
                    and tags[0][0] == "thread" and tags[1][0] == "thread_bytes" and tags[0][1] != tags[1][1]):
                # idxen offen: address = index * stride + byte offset + instruction offset.
                (_, index), (_, column, shift) = tags
                if (offset + span > 1 << shift or dims[column] << shift > stride):
                    raise ValueError("Store leaves its element or row")
                store_layout = ("index_offset", index, column, shift)
                covered = {index, column}
                address = f"tid.{AXES[index]} * {stride}u + (tid.{AXES[column]} << {shift}u) + {offset}u"
            elif (stride is not None and idxen and offen and len(addr_regs) == 2 and addr_regs[1] == addr_regs[0] + 1
                    and all(isinstance(tag, tuple) for tag in tags)
                    and tags[0][0] == "thread" and tags[1][0] == "plane_bytes"
                    and tags[0][1] not in (tags[1][1], tags[1][3])):
                # Three axes: index * stride + (row << row_shift) + (column << shift) + instruction offset.
                (_, index), (_, row, row_shift, column, shift) = tags
                if (offset + span > 1 << shift or dims[column] << shift > 1 << row_shift
                        or dims[row] << row_shift > stride):
                    raise ValueError("Store leaves its element, row or plane")
                store_layout = ("index_plane", index, row, row_shift, column, shift)
                covered = {index, row, column}
                address = (f"tid.{AXES[index]} * {stride}u + (tid.{AXES[row]} << {row_shift}u)"
                           f" + (tid.{AXES[column]} << {shift}u) + {offset}u")
            elif (offen and not idxen and len(addr_regs) == 1
                    and isinstance(tags[0], tuple) and tags[0][0] == "pitch_bytes"):
                # offen only: (row & 0xffffff) * pitch + (column << shift) + instruction offset.
                # Without idxen the descriptor stride is not used, so the stride binding is ignored.
                _, row, pitch, column, shift = tags[0]
                # A group's rows start on a 24-bit-aligned power-of-two boundary, so masking keeps them
                # consecutive; their pitches must not wrap the 32-bit address onto each other.
                if (offset + span > 1 << shift or dims[column] << shift > pitch
                        or dims[row] * pitch > 1 << 32):
                    raise ValueError("Store leaves its element or row")
                store_layout = ("pitch", row, pitch, column, shift)
                covered = {row, column}
                address = f"(tid.{AXES[row]} & 0xffffffu) * {pitch}u + (tid.{AXES[column]} << {shift}u) + {offset}u"
            else:
                raise ValueError("Unsupported store address")
            if any(dims[axis] > 1 for axis in range(3) if axis not in covered):
                raise ValueError("Store address does not separate every invocation of the group")
            if shared_atomic_offset is not None and (linear is None or offset <= shared_atomic_offset):
                raise ValueError("Return record can overlap the fixed atomic address")
            if layout not in (None, store_layout):
                raise ValueError("Stores use different address layouts")
            layout = store_layout
            span_end = offset + span
            if any(lo < span_end and offset < hi for lo, hi in written_spans):
                raise ValueError("Overlapping raw UAV stores")
            written_spans.append((offset, span_end))
            values = [vector.get(reg) for reg in range(first, first + count)]
            if not all(is_word(value) for value in values):
                raise ValueError("Store data is not a known word")
            words = list(map(hlsl_word, values))
            if subword:
                # The indexed layout and whole-DWORD reservation make the read/modify/write single-writer.
                mask = (1 << (width * 8)) - 1
                keep = 0xffffffff ^ (mask << byte_shift)
                source_shift = 16 if m["store_hi"] else 0
                words[0] = (f"((outputData.Load({address}) & {keep:#x}u) | "
                            f"((({words[0]} >> {source_shift}u) & {mask:#x}u) << {byte_shift}u))")
            elif partial and not descriptor_store and heap_store_view is None:
                # Reserve the entire final DWORD, including the untouched high half. The indexed
                # layout, aligned stride and overlap checks give that word a single writer.
                final_address = address if count == 1 else f"{address} + 4u"
                words[-1] = f"((outputData.Load({final_address}) & 0xffff0000u) | ({words[-1]} & 0xffffu))"
            if heap_store_view is not None:
                slot = hlsl_word(("constant_word", heap_slot_word))
                heap_store_components = components
                if channels in ("d16_x", "d16_xy"):
                    stores.append(f"    store_heap_word_d16({slot}, {address}, {offset}u, {words[0]});")
                elif channels == "d16_xyz":
                    stores.append(f"    store_heap_xyz_d16({slot}, {address}, {offset}u, uint2({', '.join(words)}));")
                else:
                    supplied = words + ["0u"] * (4 - len(words))
                    mode = "true" if packed else "false"
                    stores.append(f"    store_heap_words4({slot}, {address}, {offset}u, {mode}, uint4({', '.join(supplied)}));")
            elif descriptor_store:
                supplied = words + ["0u"] * (4 - len(words))
                stores.append("    store_descriptor_format("
                              f"{hlsl_word(('constant_word', format_index))}, {address}, {components}u, "
                              f"{int(packed)}u, {', '.join(supplied)});")
            else:
                text = ", ".join(words)
                arg = text if count == 1 else f"uint{count}({text})"
                stores.append(f"    outputData.Store{count if count > 1 else ''}({address}, {arg});")
        else:
            raise ValueError(f"Unsupported compute instruction: {line}")
    if ds_wait:
        raise ValueError("Expected immediate s_waitcnt lgkmcnt(0) after DS")
    if format_words is not None and format_store_index != len(format_words):
        raise ValueError("Unused buffer descriptor format constant")
    if load_format_words is not None and format_load_index != len(load_format_words):
        raise ValueError("Unused buffer load format constant")
    if buffer_float_domain is not None and not buffer_float_update:
        raise ValueError("Unused buffer float domain")
    if heap_view is not None and not format_load_index:
        raise ValueError("Unused heap view binding")
    if heap_store_view is not None and (not layout or layout[0] != "heap_index" or len(stores) != 1):
        raise ValueError("Heap store view requires exactly one formatted UAV store")
    if load_format_words is not None or heap_view is not None:
        # Every load is staged before the stores. The format words are uniform constants, so an invalid
        # format leaves the whole dispatch unwritten rather than publishing a placeholder result.
        valid = " && ".join(f"format_valid_{index}" for index in range(format_load_index))
        staged_steps.append((None, f"    if (!({valid})) return;"))
    if gather_mips is not None and len(gather_steps) != 1:
        raise ValueError("Expected one image_gather4_l for the declared gather view")
    if gather_steps:
        # A NaN or infinite coordinate has no established native result: publish nothing for that invocation.
        finite = " && ".join(f"all(isfinite({name}_uv))" for name in gather_steps)
        staged_steps.append((None, f"    if (!({finite})) return;"))
    if not stores:
        raise ValueError("Expected at least one observable store")
    if guard is not None and guard["label"] is not None and not guard["at_label"]:
        raise ValueError("Unverified branch target")
    if guard_span is not None:
        # Without a restore the mask holds until the exit.
        start, end = guard_span[0], len(stores) if guard_span[1] is None else guard_span[1]
        if start == end:
            raise ValueError("Expected a store under the lane guard")
        op, left, right = guard_span[2]
        stores[start:end] = [f"    if ({hlsl_word(left)} {GUARD_COMPARES[op]} {hlsl_word(right)}) {{",
                             *("    " + store for store in stores[start:end]), "    }"]
    body = "\n".join(stores)
    cbuffer = (f"cbuffer Constants : register(b0) {{ uint4 params[{CONSTANT_WORDS // 4}]; }};\n"
               if constant_descriptor is not None else "")
    if source_descriptor is not None:
        cbuffer += "ByteAddressBuffer inputData : register(t0);\n"
    if image_srv_descriptor is not None or (image_descriptor is not None and not image_output):
        texture_type = ("Texture2DMS<uint, 4>" if image_msaa else FLOAT_IMAGE_SRV_TYPES[image_srv_dimension] if rgba32_float_image
                        else f"Texture2D<{IMAGE_FORMATS[image_format][0]}>")
        cbuffer += f"{texture_type} imageData : register(t0);\n"
    local = ""
    if staged_steps:
        body = "\n".join([*(step for _, step in staged_steps), body])
        shared = list(dict.fromkeys(slot for slot, _ in staged_steps if slot is not None))
        cbuffer += "".join(f"groupshared uint {slot}[{dims[0]}];\n" for slot in shared)
        if shared:
            local = ", uint3 local : SV_GroupThreadID"
    if "local." in body:
        local = ", uint3 local : SV_GroupThreadID"
    if shared_lds and ds_step_count:
        cbuffer += f"groupshared uint lds[{lds_bytes // 4}];\n"
    if "gindex" in body:
        local += ", uint gindex : SV_GroupIndex"
    # Helpers are selected from the final body, so a float ALU result used only by a DS step still gets its definition.
    cbuffer += "".join(text for name, text in SALU_HELPERS.items() if f"{name}(" in body)
    cbuffer += "".join(text for name, text in F32_HELPERS.items() if f"f32_{name}(" in body)
    if "cvt_u32_f32(" in body:
        cbuffer += CVT_U32_F32_HELPER
    cbuffer += "".join(text for name, text in DOT_U_HELPERS.items() if f"{name}(" in body)
    uav_texel_type = image_atomic_type or ("float4" if float_image else "uint4")
    uav = f"{IMAGE_UAV_TYPES[image_uav_dimension]}<{uav_texel_type}>" if image_output else "RWByteAddressBuffer"
    # The helper reads outputData, so it follows the UAV declaration.
    descriptor_hlsl = descriptor_store_hlsl() if "store_descriptor_format(" in body else ""
    if "load_descriptor_format(" in body or "load_heap_view_format(" in body:
        descriptor_hlsl += descriptor_load_hlsl()
    if heap_view is not None:
        descriptor_hlsl += heap_view_hlsl(heap_table, heap_count, heap_bytes)
    if heap_store_view is not None:
        descriptor_hlsl += heap_store_view_hlsl(heap_table, heap_count, output_bytes, components=heap_store_components)
    # GetDimensions on a raw buffer returns its size in bytes, the value of descriptor word 2.
    for name, view in (("uav_bytes", "outputData"), ("srv_bytes", "inputData")):
        if f"{name}(" in body:
            descriptor_hlsl += f"uint {name}() {{ uint n; {view}.GetDimensions(n); return n; }}\n"
    if gather_steps:
        descriptor_hlsl += GATHER_HLSL
    return (f"{cbuffer}{uav} outputData : register(u0);\n{descriptor_hlsl}"
            f"[numthreads({dims[0]},{dims[1]},{dims[2]})]\n"
            f"void main(uint3 tid : SV_DispatchThreadID{local}) {{\n{body}\n}}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path, help="Local extracted DXBC container")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm", required=True)
    parser.add_argument("--cpu", choices=("gfx1013", "gfx1030"), default="gfx1013")
    parser.add_argument("--lds-f32-mode", choices=("normal-rne",),
                        help="Private LDS f32 inputs and outputs are normal or +0, with round-to-nearest-even")
    parser.add_argument("--lds-f64-mode", choices=("normal",),
                        help="Private LDS f64 comparisons use normal values or positive zero")
    parser.add_argument("--lds-bytes", type=int, default=0, help="Known native group LDS allocation")
    parser.add_argument("--table-reg", type=int, required=True)
    parser.add_argument("--descriptor-offset", type=lambda n: int(n, 0), help="Table byte offset bound to raw u0")
    parser.add_argument("--constant-descriptor", type=lambda n: int(n, 0), help="Table byte offset bound to b0")
    parser.add_argument("--group-reg", type=int, help="SGPR holding the group X ID")
    parser.add_argument("--group-reg-y", type=int, help="SGPR holding the group Y ID")
    parser.add_argument("--group-reg-z", type=int, help="SGPR holding the group Z ID")
    parser.add_argument("--rsrc2", type=lambda n: int(n, 0),
                        help="COMPUTE_PGM_RSRC2: places the group IDs instead of --group-reg*")
    parser.add_argument("--stride", type=int, help="Indexed raw UAV byte stride")
    parser.add_argument("--source-descriptor", type=lambda n: int(n, 0),
                        help="Table byte offset bound to stable read-only t0, with no overlap with u0")
    parser.add_argument("--source-stride", type=int, help="Indexed t0 byte stride")
    parser.add_argument("--source-format", choices=MUBUF_SOURCE_TEXELS,
                        help="Caller-declared MUBUF t0 descriptor format with identity channel selectors")
    parser.add_argument("--output-format", choices=MUBUF_OUTPUT_FORMATS,
                        help="Caller-declared MUBUF u0 descriptor format with identity channel selectors")
    parser.add_argument("--descriptor-format-constant", action="append", type=lambda n: int(n, 0),
                        help="b0 word index holding V# word 3 for one formatted store, in program order")
    parser.add_argument("--load-format-constant", action="append", type=lambda n: int(n, 0),
                        help="b0 word index holding the t0 V# word 3 for one gfx1013 formatted load, in program order")
    parser.add_argument("--heap-view", nargs=4, type=lambda n: int(n, 0),
                        metavar=("SLOT_WORD", "TABLE_OFFSET", "COUNT", "BUFFER_BYTES"),
                        help="gfx1013 formatted loads take the V# from the run-time view record at the slot in b0 word "
                             "SLOT_WORD; the record table is at byte TABLE_OFFSET of the BUFFER_BYTES-byte t0 and holds COUNT views")
    parser.add_argument("--heap-store-view", nargs=5, type=lambda n: int(n, 0),
                        metavar=("SLOT_WORD", "TABLE_OFFSET", "COUNT", "TABLE_BUFFER_BYTES", "OUTPUT_BYTES"),
                        help="select a UINT4 or expanded D16 FLOAT1/FLOAT2/FLOAT3/FLOAT4 store view from t0 "
                             "and bound writes by OUTPUT_BYTES of u0")
    parser.add_argument("--mubuf-d16-xyz-mode", choices=("preserve-w",),
                        help="Experimental RGBA16 UINT store with caller-verified preservation of W")
    parser.add_argument("--image-descriptor", type=lambda n: int(n, 0),
                        help="Table byte offset of the 8-dword image descriptor bound to t0 or u0")
    parser.add_argument("--image-uav-mip", type=int,
                        help="Bind the image to u0 at this mip instead of binding a raw output descriptor")
    parser.add_argument("--image-uav-dimension", choices=IMAGE_UAV_TYPES, default="2d",
                        help="Declare the u0 view dimension for an image store")
    parser.add_argument("--image-srv-dimension", choices=FLOAT_IMAGE_SRV_TYPES, default="2d",
                        help="Declare the t0 view dimension for a gfx1013 RGBA32 FLOAT mip load")
    parser.add_argument("--image-srv-descriptor", type=lambda n: int(n, 0),
                        help="Separate image descriptor at t0 when u0 is a gfx1013 RGBA32 FLOAT image")
    parser.add_argument("--image-format", choices=sorted(IMAGE_FORMATS.keys() | IMAGE_ATOMIC_TYPES.keys()
                                                         | {RGBA32_FLOAT, RGBA16_FLOAT}),
                        help="Caller-declared image format (R32_UINT for MSAA SRVs or atomic UAVs, "
                             "R32_SINT for signed min/max UAVs, R32_FLOAT for finite-normal atomics, "
                             "R16G16B16A16_FLOAT for gfx1013 D16 image stores)")
    parser.add_argument("--image-float-domain", choices=("finite-normal",),
                        help="Required for R32_FLOAT atomics. All accessed texels and operands are finite normals.")
    parser.add_argument("--buffer-float-domain", choices=("finite-normal",),
                        help="Required for raw fixed-address floating buffer atomics. The stored value and "
                             "operands are finite-normal binary32 or binary64.")
    parser.add_argument("--wave-size", type=int, choices=(32, 64),
                        help="Caller's statement of the wave width: 32 for lane indices in larger groups, "
                             "64 (gfx1013 shared LDS only) for a group of up to 64 threads as one wave")
    parser.add_argument("--gather-mips", type=int,
                        help="Visible mip count of the matching native and desktop RGBA32 FLOAT 2D gather views")
    args = parser.parse_args()
    if (args.descriptor_offset is None) == (args.image_uav_mip is None):
        parser.error("Choose either --descriptor-offset or --image-uav-mip")
    if args.output.suffix.lower() != ".hlsl" or args.output.resolve() == args.container.resolve():
        parser.error("Choose an HLSL destination distinct from the input")
    blob = args.container.read_bytes()
    chunks = parts(blob)
    group = group_size(chunks.get(b"PSV0", b""))
    code = container_code(chunks.get(b"DXIL", b""), stage=5, llvm=args.llvm)
    features = b"+wavefrontsize32,-wavefrontsize64" if args.cpu == "gfx1030" else b""
    decoded = disassemble(code, args.llvm, offsets=True, cpu=args.cpu.encode(), features=features)
    source = lower(labelled(decoded), group, table_reg=args.table_reg,
                   descriptor_offset=args.descriptor_offset, group_reg=args.group_reg, stride=args.stride,
                   constant_descriptor=args.constant_descriptor, group_reg_y=args.group_reg_y,
                   group_reg_z=args.group_reg_z, source_descriptor=args.source_descriptor,
                   source_stride=args.source_stride, source_format=args.source_format,
                   output_format=args.output_format,
                   image_descriptor=args.image_descriptor, image_format=args.image_format,
                   buffer_float_domain=args.buffer_float_domain,
                   image_uav_dimension=args.image_uav_dimension,
                   image_uav_mip=args.image_uav_mip, image_float_domain=args.image_float_domain,
                   cpu=args.cpu, lds_bytes=args.lds_bytes,
                   lds_f32_mode=args.lds_f32_mode, lds_f64_mode=args.lds_f64_mode,
                   mubuf_d16_xyz_mode=args.mubuf_d16_xyz_mode, image_srv_descriptor=args.image_srv_descriptor,
                   image_srv_dimension=args.image_srv_dimension,
                   descriptor_format_constant=args.descriptor_format_constant,
                   rsrc2=args.rsrc2, wave_size=args.wave_size, load_format_constant=args.load_format_constant,
                   heap_view=tuple(args.heap_view) if args.heap_view else None, heap_store_view=args.heap_store_view,
                   gather_mips=args.gather_mips)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="ascii")
    print(json.dumps({"container_sha256": hashlib.sha256(blob).hexdigest(),
                      "native_sha256": hashlib.sha256(code).hexdigest(), "numthreads": list(group)}))


if __name__ == "__main__":
    main()
