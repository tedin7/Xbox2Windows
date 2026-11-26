"""The v_mul_hi_u32 lowering in tools/startup/gfx2hlsl.py (issue #1792).

usage: startup_gfx2hlsl_mul_hi_u32_check.py

`v_mul_hi_u32` (the high 32 bits of an unsigned 32x32 product) had no lowering, so the census form
`v_mul_hi_u32 v1, v0, 0xaaaaaaab` was refused (#870). A plain `*` would be silently wrong: it yields the low half.
The lowering calls a preamble helper built from halfword products, because these stages have no 64-bit multiply.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch, opcode 0x16a): prefix `0b110101`, opcode in `w0[25:16]`, the
destination in `w0[7:0]`, src0 in `w1[8:0]`, src1 in `w1[17:9]`, and a literal dword after the pair.
"""
import os
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


VOP3A_MUL_HI_U32 = 0x16A
VOP3A_MUL_LO_U32 = 0x169
LITERAL = 255
V0, V1, V2 = 0x100, 0x101, 0x102


def encode_vop3a(op, dst, src0, src1, literal=None):
    w0 = (0b110101 << 26) | ((op & 0x3FF) << 16) | (dst & 0xFF)
    w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9)
    code = struct.pack('<II', w0, w1)
    return code + struct.pack('<I', literal) if literal is not None else code


class Pipeline:
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


asm, emitted = translate(encode_vop3a(VOP3A_MUL_HI_U32, 1, V1, V2))
check('register form decodes', asm == ['v_mul_hi_u32 v1, v1, v2'], repr(asm))
check('register form calls the helper', 'v1 = mul_hi_u32(v1, v2);' in emitted, emitted)

asm, emitted = translate(encode_vop3a(VOP3A_MUL_HI_U32, 1, V0, LITERAL, 0xAAAAAAAB))
check('census form decodes', asm == ['v_mul_hi_u32 v1, v0, 0xaaaaaaab'], repr(asm))
check('census form calls the helper', 'v1 = mul_hi_u32(v0, 0xaaaaaaabu);' in emitted, emitted)

asm, emitted = translate(encode_vop3a(VOP3A_MUL_LO_U32, 1, V1, V2))
check('the low multiply is unchanged', 'v1 = v1 * v2;' in emitted and 'mul_hi_u32' not in emitted, emitted)

pipe = object.__new__(G.Pipeline)
pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
common = ' '.join(pipe.common())
check('the helper is in the common preamble', 'uint mul_hi_u32(uint a, uint b)' in common, common[:240])
check('the helper uses halfword products', '& 0xffffu' in common.split('mul_hi_u32(uint a')[1].split('}')[0]
      and '>> 16' in common.split('mul_hi_u32(uint a')[1].split('}')[0])

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('gfx2hlsl mul_hi_u32: all checks passed')
