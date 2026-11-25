"""SDWA forms of the float-result conversions and their output modifiers in tools/startup/gfx2hlsl.py (#1727, #2023).

  (no arguments)                       translation, bit-model, refusal and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the 16 x 16 pixel fixture (FXC ps_5_1 when given, and DXC ps_6_0),
                                       draw and compare all 1,024 exported words with the RDNA 2 ISA reference,
                                       then require five mutated HLSL shaders to fail against the reference

Moving the float-result conversions into their own table (VEC_U_TO_F) initially left `sdwa()` looking only at VEC_U
(#1727), and the restored path still refused output modifiers (`clamp`, `mul:2`, `mul:4`, `div:2`) in `sdwa()` —
blocking `v_cvt_f32_u32_sdwa ... mul:2` and sibling VEC_U_TO_F SDWA forms (#2023 item E).

In the GFX10 VOP_SDWA encoding (Mesa `src/amd/common/sid.h` S_000504_*, `src/amd/compiler/aco_assembler.cpp`;
LLVM `llvm/lib/Target/AMDGPU/VOPInstructions.td` `VOP_SDWA9Ae`), word 1 carries:
  w0: [31:25] = 0b0111111, [24:17] vdst, [16:9] opcode, [8:0] = 249 (SDWA)
  w1: [7:0] src0, [10:8] dst_sel, [12:11] dst_unused, [13] clamp, [15:14] omod, [18:16] src0_sel,
      [19] src0_sext, [20] src0_neg, [21] src0_abs, [23] src0_sgpr
Hardware order on a float-result SDWA conversion:
  1. Extract `src0_sel` (zero-extended, or sign-extended to 32 bits when `src0_sext` is set) and run the conversion.
  2. Apply `omod` (`* 2.0`, `* 4.0`, `* 0.5`) to the binary32 result.
  3. Apply `clamp` (`saturate` to `[0.0, 1.0]`) after `omod`, even though disassembly prints `clamp` before `omod`.
  4. Reinterpret the resulting binary32 float as 32 bits and place its low bits into `dst_sel` (`UNUSED_PAD` / `UNUSED_PRESERVE`).
"""
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


VOP1_CVT_F32_I32 = 0x05
VOP1_CVT_F32_U32 = 0x06
VOP1_CVT_U32_F32 = 0x07
VOP1_CVT_F16_F32 = 0x0A
VOP1_CVT_F32_F16 = 0x0B
VOP1_CVT_F32_UBYTE0 = 0x11
VOP1_CVT_F32_UBYTE1 = 0x12
VOP1_CVT_F32_UBYTE2 = 0x13
VOP1_CVT_F32_UBYTE3 = 0x14
VOP1_RCP_F32 = 0x2A

FLOAT_CVT_OPS = (
    ('v_cvt_f32_i32', VOP1_CVT_F32_I32),
    ('v_cvt_f32_u32', VOP1_CVT_F32_U32),
    ('v_cvt_f32_f16', VOP1_CVT_F32_F16),
    ('v_cvt_f32_ubyte0', VOP1_CVT_F32_UBYTE0),
    ('v_cvt_f32_ubyte1', VOP1_CVT_F32_UBYTE1),
    ('v_cvt_f32_ubyte2', VOP1_CVT_F32_UBYTE2),
    ('v_cvt_f32_ubyte3', VOP1_CVT_F32_UBYTE3),
)

SEL = {'BYTE_0': 0, 'BYTE_1': 1, 'BYTE_2': 2, 'BYTE_3': 3, 'WORD_0': 4, 'WORD_1': 5, 'DWORD': 6}
SEL_BITS = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
            'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}
UNUSED = {'UNUSED_PAD': 0, 'UNUSED_SEXT': 1, 'UNUSED_PRESERVE': 2}
OMOD = {'': 0, 'mul:2': 1, 'mul:4': 2, 'div:2': 3}
OMOD_SCALE = {'': 1.0, 'mul:2': 2.0, 'mul:4': 4.0, 'div:2': 0.5}
SDWA_SRC0 = 249


def encode_vop1_sdwa(op, vdst, src0, *, dst_sel='DWORD', dst_unused='UNUSED_PAD', src0_sel='WORD_0',
                     omod='', clamp=0, sext=0, neg=0, abs_=0, sgpr=0):
    """Encode a VOP1 instruction in its GFX10 SDWA form."""
    w0 = ((0b0111111 << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | SDWA_SRC0)
    w1 = ((src0 & 0xFF) | (SEL[dst_sel] << 8) | (UNUSED[dst_unused] << 11) | ((clamp & 1) << 13)
          | (OMOD[omod] << 14) | (SEL[src0_sel] << 16) | ((sext & 1) << 19)
          | ((neg & 1) << 20) | ((abs_ & 1) << 21) | ((sgpr & 1) << 23))
    return struct.pack('<II', w0, w1)


class Pipeline:
    """Just enough pipeline state for one conversion."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused(code, substring=''):
    try:
        translate(code)
    except G.Unsupported as error:
        return substring in str(error)
    return False


# ---- Independent HLSL expression evaluator & RDNA 2 ISA reference ------------------------------------------------
class U32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & 0xFFFFFFFF)

    def __lshift__(self, n):
        return U32((int(self) << (int(n) & 31)) & 0xFFFFFFFF)

    def __rshift__(self, n):
        return U32((int(self) & 0xFFFFFFFF) >> (int(n) & 31))

    def __and__(self, other):
        return U32(int(self) & int(other))

    def __rand__(self, other):
        return U32(int(other) & int(self))

    def __or__(self, other):
        return U32(int(self) | int(other))

    def __ror__(self, other):
        return U32(int(other) | int(self))


class I32(int):
    def __new__(cls, value):
        u = int(value) & 0xFFFFFFFF
        return super().__new__(cls, u - 0x100000000 if u & 0x80000000 else u)

    def __rshift__(self, n):
        return I32(int(self) >> (int(n) & 31))


class F32:
    def __init__(self, bits):
        self.bits = int(bits) & 0xFFFFFFFF

    @classmethod
    def from_float(cls, value):
        return cls(struct.unpack('<I', struct.pack('<f', float(value)))[0])

    def to_float(self):
        return struct.unpack('<f', struct.pack('<I', self.bits))[0]

    def __mul__(self, factor):
        return F32.from_float(self.to_float() * float(factor))


def hlsl_saturate(value):
    f = value.to_float()
    if math.isnan(f) or f <= 0.0:
        return F32(0)
    if f >= 1.0:
        return F32(0x3F800000)
    return value


def hlsl_f16tof32(word):
    half = int(word) & 0xFFFF
    val = struct.unpack('<e', struct.pack('<H', half))[0]
    if math.isnan(val):
        return F32(((half & 0x8000) << 16) | 0x7FC00000 | ((half & 0x03FF) << 13))
    return F32.from_float(val)


HLSL_ENV = {
    'U32': U32,
    'asuint': U32,
    'uint': U32,
    'asint': I32,
    'int': I32,
    'float': lambda x: F32.from_float(int(x)),
    'f16tof32': hlsl_f16tof32,
    'saturate': hlsl_saturate,
    'U': lambda f: U32(f.bits),
}


def eval_emitted_v0(statement, v0_in, v1_in):
    assert statement.startswith('v0 = ') and statement.endswith(';'), statement
    expr = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'U32(\1)', statement[len('v0 = '):-1])
    return int(eval(expr, {'__builtins__': {}}, {**HLSL_ENV, 'v0': U32(v0_in), 'v1': U32(v1_in)}))


def isa_reference(op_name, v0_in, v1_in, *, src0_sel, sext, omod, clamp, dst_sel, dst_unused):
    off, width = SEL_BITS[src0_sel]
    selected = (v1_in >> off) & ((1 << width) - 1)
    if sext and width < 32 and (selected & (1 << (width - 1))):
        selected |= (~((1 << width) - 1)) & 0xFFFFFFFF

    if op_name == 'v_cvt_f32_u32':
        res = F32.from_float(selected)
    elif op_name == 'v_cvt_f32_i32':
        signed = selected - 0x100000000 if selected & 0x80000000 else selected
        res = F32.from_float(signed)
    elif op_name == 'v_cvt_f32_f16':
        res = hlsl_f16tof32(selected)
    else:
        byte_idx = int(op_name[-1])
        res = F32.from_float((selected >> (8 * byte_idx)) & 0xFF)

    if omod:
        res = res * OMOD_SCALE[omod]
    if clamp:
        res = hlsl_saturate(res)

    doff, dwidth = SEL_BITS[dst_sel]
    if dwidth == 32:
        return res.bits
    mask = (1 << dwidth) - 1
    placed = (res.bits & mask) << doff
    if dst_unused == 'UNUSED_PAD':
        return placed
    keep = (~(mask << doff)) & 0xFFFFFFFF
    return (v0_in & keep) | placed


def controls():
    # 1. The encoder produces the form the decoder reads; a layout change shows up here first.
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='WORD_0'))
    check('encodes and decodes a conversion SDWA',
          asm == ['v_cvt_f32_u32_sdwa v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0'], repr(asm))

    # 2. Each float-result conversion translates without modifiers as before.
    check('u32 conversion SDWA translates',
          emitted == 'v0 = U(float(((v1 >> 0u) & 0xffffu)));', emitted)
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_I32, 0, 1, src0_sel='BYTE_0'))
    check('i32 conversion SDWA translates',
          emitted == 'v0 = U(float(int(((v1 >> 0u) & 0xffu))));', emitted)
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_F16, 0, 1, src0_sel='WORD_0'))
    check('f16 conversion SDWA translates',
          emitted == 'v0 = U(f16tof32(((v1 >> 0u) & 0xffffu)));', emitted)
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='DWORD'))
    check('DWORD selection reads the whole register', emitted == 'v0 = U(float(v1));', emitted)
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='BYTE_1'))
    check('BYTE_1 selection shifts to the byte', '(v1 >> 8u) & 0xffu' in emitted, emitted)

    # 3. A destination selection still writes the converted value's bit pattern.
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, dst_sel='BYTE_1'))
    check('destination selection masks the result', '& 0xffu) << 8u' in emitted, emitted)
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, dst_sel='BYTE_0',
                                              dst_unused='UNUSED_PRESERVE'))
    check('UNUSED_PRESERVE keeps the other destination bits', 'asuint(v0)' in emitted, emitted)

    # 4. A uint-result conversion takes the uint path and is not wrapped as a float.
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F16_F32, 0, 1, src0_sel='WORD_0'))
    check('uint-result conversion stays on the uint path',
          emitted == 'v0 = f32tof16(F(((v1 >> 0u) & 0xffffu)));', emitted)

    # 5. Output modifiers (clamp, mul:2, mul:4, div:2) on VEC_U_TO_F SDWA conversions (#2023 item E).
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='WORD_0', omod='mul:2'))
    check('u32 conversion SDWA with mul:2 decodes and doubles',
          asm == ['v_cvt_f32_u32_sdwa v0, v1 mul:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0']
          and emitted == 'v0 = U((float(((v1 >> 0u) & 0xffffu))) * 2.0);',
          f'{asm!r} -> {emitted}')
    asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='WORD_1', omod='div:2', clamp=1))
    check('clamp is applied after omod scaling, despite printing before div:2',
          asm == ['v_cvt_f32_u32_sdwa v0, v1 clamp div:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1']
          and emitted == 'v0 = U(saturate((float(((v1 >> 16u) & 0xffffu))) * 0.5));',
          f'{asm!r} -> {emitted}')
    # At v1 WORD_1 = 3, hardware order (3 * 0.5 -> saturate = 1.0) differs from text order (saturate(3) * 0.5 = 0.5).
    check('u=3 under clamp + div:2 evaluates to 1.0 (0x3f800000), not 0.5',
          eval_emitted_v0(emitted, 0, 3 << 16) == 0x3F800000)

    # Exhaustive sweep across all 7 VEC_U_TO_F ops, all 8 (omod x clamp) combinations, sub-dword selects and inputs.
    test_inputs = (0, 1, 3, 0x3C01, 0xBC00, 0x00FF8003, 0x807F3E00, 0x7FFFFFFF, 0xFFFFFFFF)
    old_dst = 0xA55A5AA5
    sweep_failures = []
    sweep_count = 0
    for op_name, op_code in FLOAT_CVT_OPS:
        sext_choices = (0,) if op_name == 'v_cvt_f32_f16' else (0, 1)
        for omod in ('', 'mul:2', 'mul:4', 'div:2'):
            for clamp in (0, 1):
                for src0_sel, dst_sel, dst_unused in (
                    ('WORD_0', 'DWORD', 'UNUSED_PAD'),
                    ('BYTE_1', 'WORD_0', 'UNUSED_PRESERVE'),
                    ('DWORD', 'BYTE_2', 'UNUSED_PAD'),
                ):
                    for sext in sext_choices:
                        if sext and src0_sel == 'DWORD':
                            continue
                        code = encode_vop1_sdwa(op_code, 0, 1, dst_sel=dst_sel, dst_unused=dst_unused,
                                                src0_sel=src0_sel, omod=omod, clamp=clamp, sext=sext)
                        _, stmt = translate(code)
                        for v1_val in test_inputs:
                            sweep_count += 1
                            got = eval_emitted_v0(stmt, old_dst, v1_val)
                            want = isa_reference(op_name, old_dst, v1_val, src0_sel=src0_sel, sext=bool(sext),
                                                 omod=omod, clamp=bool(clamp), dst_sel=dst_sel, dst_unused=dst_unused)
                            if got != want and len(sweep_failures) < 3:
                                sweep_failures.append((op_name, omod, clamp, src0_sel, dst_sel, hex(v1_val),
                                                       hex(got), hex(want)))
    check(f'all {sweep_count} VEC_U_TO_F SDWA (op x omod x clamp x sel) evaluations match ISA reference',
          not sweep_failures, repr(sweep_failures))

    # 6. Refusal boundaries: VEC_U ops, VEC_F ops, float source modifiers and UNUSED_SEXT stay refused.
    check('uint-result conversion v_cvt_f16_f32_sdwa refuses mul:2',
          refused(encode_vop1_sdwa(VOP1_CVT_F16_F32, 0, 1, src0_sel='WORD_0', omod='mul:2'),
                  'v_cvt_f16_f32 with SDWA modifier mul:2'))
    check('uint-result conversion v_cvt_f16_f32_sdwa refuses clamp',
          refused(encode_vop1_sdwa(VOP1_CVT_F16_F32, 0, 1, src0_sel='WORD_0', clamp=1),
                  'v_cvt_f16_f32 with SDWA modifier clamp'))
    check('uint-result conversion v_cvt_u32_f32_sdwa refuses clamp',
          refused(encode_vop1_sdwa(VOP1_CVT_U32_F32, 0, 1, src0_sel='DWORD', clamp=1),
                  'v_cvt_u32_f32 with SDWA modifier clamp'))
    check('non-conversion VEC_F op v_rcp_f32_sdwa stays refused with mul:2',
          refused(encode_vop1_sdwa(VOP1_RCP_F32, 0, 1, src0_sel='DWORD', omod='mul:2')))
    check('v_cvt_f32_f16_sdwa with negated source stays refused',
          refused(encode_vop1_sdwa(VOP1_CVT_F32_F16, 0, 1, src0_sel='WORD_0', omod='mul:2', neg=1)))
    check('v_cvt_f32_f16_sdwa with abs source stays refused',
          refused(encode_vop1_sdwa(VOP1_CVT_F32_F16, 0, 1, src0_sel='WORD_0', omod='mul:2', abs_=1)))
    check('UNUSED_SEXT on v_cvt_f32_u32_sdwa with mul:2 stays refused',
          refused(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, dst_sel='BYTE_0', dst_unused='UNUSED_SEXT',
                                   src0_sel='WORD_0', omod='mul:2'), 'UNUSED_SEXT'))
    # Integer VEC_U op with clamp/omod via Stage.one stays refused.
    for mod in ('clamp', 'mul:2'):
        try:
            Stage().one('v_add_nc_u32_sdwa',
                        f'v_add_nc_u32_sdwa v0, v1, v2 {mod} dst_sel:DWORD dst_unused:UNUSED_PAD '
                        'src0_sel:DWORD src1_sel:DWORD')
        except G.Unsupported as err:
            check(f'v_add_nc_u32_sdwa refuses {mod}', f'v_add_nc_u32 with SDWA modifier {mod}' in str(err), str(err))
        else:
            check(f'v_add_nc_u32_sdwa refuses {mod}', False, 'accepted')

    # 7. Live monkeypatch negative control: dropping out_mods forwarding in f_mods must fail the mul:2 check.
    saved_f_mods = G.Stage.f_mods
    try:
        G.Stage.f_mods = lambda self, expr, mods=(): expr
        _, unscaled = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='WORD_0', omod='mul:2'))
        check('negative control: ignoring out_mods drops * 2.0 and changes evaluation',
              eval_emitted_v0(unscaled, 0, 5) == 0x40A00000 and eval_emitted_v0(unscaled, 0, 5) != 0x41200000)
    finally:
        G.Stage.f_mods = saved_f_mods


# ---- 16 x 16 pixel draw fixture (WARP + hardware, FXC ps_5_1 + DXC ps_6_0) ---------------------------------------
WIDTH = HEIGHT = 16
MULTIPLIER = 0x9B5D
PRESERVE_INIT = 0xA55A0000
PACKED_INPUT = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: packed screen position x | (y << 16) in v0

# Constructed gfx1013 pixel stage:
#   v_mul_u32_u24_e32 v1, 0x9b5d, v0
#   v_mov_b32_e32 v5, 0xa55a0000
#   v_or_b32_e32 v6, 0x3c00, v0
#   v_cvt_f32_u32_sdwa v2, v0 mul:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0
#   v_cvt_f32_u32_sdwa v3, v0 clamp div:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
#   v_cvt_f32_i32_sdwa v4, sext(v1) mul:4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
#   v_cvt_f32_f16_sdwa v5, v6 clamp div:2 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
#   exp mrt0 v2, v3, v4, v5 done vm
#   s_endpgm
PS_WORDS = (
    struct.pack('<II', (0x0B << 25) | (1 << 17) | 255, MULTIPLIER)
    + struct.pack('<II', 0x7E0A02FF, PRESERVE_INIT)
    + struct.pack('<II', (0x1C << 25) | (6 << 17) | 255, 0x3C00)
    + encode_vop1_sdwa(VOP1_CVT_F32_U32, 2, 0, src0_sel='WORD_0', omod='mul:2')
    + encode_vop1_sdwa(VOP1_CVT_F32_U32, 3, 0, src0_sel='WORD_1', omod='div:2', clamp=1)
    + encode_vop1_sdwa(VOP1_CVT_F32_I32, 4, 1, src0_sel='BYTE_1', omod='mul:4', sext=1)
    + encode_vop1_sdwa(VOP1_CVT_F32_F16, 5, 6, dst_sel='WORD_0', dst_unused='UNUSED_PRESERVE',
                       src0_sel='WORD_0', omod='div:2', clamp=1)
    + bytes.fromhex('0f1800f802030405' '000081bf')
)

DRAW_MUTATIONS = {
    'dropped mul:2': (
        '(float(((v0 >> 0u) & 0xffffu))) * 2.0',
        'float(((v0 >> 0u) & 0xffffu))',
    ),
    'early clamp before div:2': (
        'saturate((float(((v0 >> 16u) & 0xffffu))) * 0.5)',
        'saturate(float(((v0 >> 16u) & 0xffffu))) * 0.5',
    ),
    'dropped mul:4': (
        '(float(int(asuint(asint(v1 << 16u) >> 24u)))) * 4.0',
        'float(int(asuint(asint(v1 << 16u) >> 24u)))',
    ),
    'dropped div:2 on f16': (
        'saturate((f16tof32(((v6 >> 0u) & 0xffffu))) * 0.5)',
        'saturate(f16tof32(((v6 >> 0u) & 0xffffu)))',
    ),
    'dropped WORD_0 dst_sel': (
        '((asuint(v5) & 0xffff0000u) | (((U(saturate((f16tof32(((v6 >> 0u) & 0xffffu))) * 0.5))) & 0xffffu) << 0u))',
        'U(saturate((f16tof32(((v6 >> 0u) & 0xffffu))) * 0.5))',
    ),
}


def expected_pixel_words(mode='reference'):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pos = x | (y << 16)
            prod = (pos * MULTIPLIER) & 0xFFFFFFFF
            half_src = 0x3C00 | pos

            ch0 = isa_reference('v_cvt_f32_u32', 0, pos, src0_sel='WORD_0', sext=False,
                                omod='' if mode == 'dropped mul:2' else 'mul:2',
                                clamp=False, dst_sel='DWORD', dst_unused='UNUSED_PAD')
            if mode == 'early clamp before div:2':
                clamped_y = hlsl_saturate(F32.from_float(y)) * 0.5
                ch1 = clamped_y.bits
            else:
                ch1 = isa_reference('v_cvt_f32_u32', 0, pos, src0_sel='WORD_1', sext=False,
                                    omod='div:2', clamp=True, dst_sel='DWORD', dst_unused='UNUSED_PAD')
            ch2 = isa_reference('v_cvt_f32_i32', 0, prod, src0_sel='BYTE_1', sext=True,
                                omod='' if mode == 'dropped mul:4' else 'mul:4',
                                clamp=False, dst_sel='DWORD', dst_unused='UNUSED_PAD')
            ch3 = isa_reference('v_cvt_f32_f16', PRESERVE_INIT, half_src, src0_sel='WORD_0', sext=False,
                                omod='' if mode == 'dropped div:2 on f16' else 'div:2',
                                clamp=True,
                                dst_sel='DWORD' if mode == 'dropped WORD_0 dst_sel' else 'WORD_0',
                                dst_unused='UNUSED_PAD' if mode == 'dropped WORD_0 dst_sel' else 'UNUSED_PRESERVE')
            words.extend((ch0, ch1, ch2, ch3))
    return words


def pixel_hlsl(mode='reference'):
    text = G.Pipeline(VS_WORDS, PS_WORDS, root_blob(), ps_hw_inputs=PACKED_INPUT).hlsl_ps()
    if mode != 'reference':
        old, new = DRAW_MUTATIONS[mode]
        assert text.count(old) == 1, (mode, text)
        text = text.replace(old, new, 1)
    return text


def run_pixel_words(renderer, adapter, vs, ps, expected_path):
    run = subprocess.run(
        [renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT), str(expected_path)],
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        print(run.stdout + run.stderr, end='', flush=True)
        sys.exit(1 if failures else 77)
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    ref_words = expected_pixel_words('reference')
    for mode in DRAW_MUTATIONS:
        assert expected_pixel_words(mode) != ref_words, mode

    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sdwa-cvt-omod-', dir=out_dir) as tmp:
        directory = Path(tmp)
        ref_bin = directory / 'reference.bin'
        ref_bin.write_bytes(struct.pack('<%dI' % len(ref_words), *ref_words))
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, f'vs_{kind}', vs_profile, directory)
            ps = compile_hlsl(compiler, kind, pixel_hlsl('reference'), f'ps_ref_{kind}', ps_profile, directory)
            run = run_pixel_words(renderer, adapter, vs, ps, ref_bin)
            check(f'{kind} ({adapter}): 16x16 pixel draw matches all 1,024 reference words',
                  run.returncode == 0, run.stdout + run.stderr)
            print(run.stdout, end='', flush=True)

            for idx, mode in enumerate(DRAW_MUTATIONS):
                mut_ps = compile_hlsl(compiler, kind, pixel_hlsl(mode), f'ps_mut_{idx}_{kind}', ps_profile, directory)
                fail_run = run_pixel_words(renderer, adapter, vs, mut_ps, ref_bin)
                check(f'{kind} ({adapter}) negative control ({mode}) fails against reference',
                      fail_run.returncode != 0, fail_run.stdout + fail_run.stderr)
                mut_words = expected_pixel_words(mode)
                mut_bin = directory / f'mut_{idx}.bin'
                mut_bin.write_bytes(struct.pack('<%dI' % len(mut_words), *mut_words))
                own_run = run_pixel_words(renderer, adapter, vs, mut_ps, mut_bin)
                check(f'{kind} ({adapter}) negative control ({mode}) matches its own model',
                      own_run.returncode == 0, own_run.stdout + own_run.stderr)


def main():
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)

    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
