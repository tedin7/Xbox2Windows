"""Absolute-value source modifier `|vN|` on the vector select in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_abs_mod_check.py
       startup_gfx2hlsl_abs_mod_check.py draw <dxc> <offscreen_check> <warp|hardware> [fxc]

A VOP3 `v_cndmask_b32` can carry the abs source modifier, which clears the source's sign bit before any
negation. The select handler passed the printed `|v30|` to the uint helper and was refused with
`Unsupported: operand '|v30|'`.

Instruction bytes are constructed from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch): w0 [9:8] abs for src1 and src0; w1 [31:29] neg.
"""
import os
import re
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


VOP2_E64_BASE = 0x100
CNDMASK = 0x01
AND_B32 = 0x1b
V30 = 256 + 30  # VGPR operands are encoded from 256 upwards


def encode_vop2_e64(op, vdst, src0, src1, *, neg0=0, neg1=0, abs0=0, abs1=0):
    """Encode a VOP2 instruction in its VOP3A (e64) form."""
    w0 = (0b110101 << 26) | ((VOP2_E64_BASE + op) << 16) | ((abs1 & 1) << 9) | ((abs0 & 1) << 8) | (vdst & 0xff)
    w1 = (src0 & 0x1ff) | ((src1 & 0x1ff) << 9) | ((neg0 & 1) << 29) | ((neg1 & 1) << 30)
    return struct.pack('<II', w0, w1)


class Pipeline:
    """Just enough pipeline state for one vector instruction."""

    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


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


def refused(code):
    try:
        translate(code)
    except G.Unsupported:
        return True
    return False


asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1))
check('abs on the first source decodes as printed', asm == ['v_cndmask_b32_e64 v31, |v30|, v30, s0'], repr(asm))
check('abs on the first source clears the sign bit',
      emitted == 'v31 = (s0 & 1) ? v30 : (v30 & 0x7fffffffu);', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs1=1))
check('abs on the second source clears the sign bit',
      emitted == 'v31 = (s0 & 1) ? (v30 & 0x7fffffffu) : v30;', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1, abs1=1))
check('abs on both sources', emitted.count('0x7fffffffu') == 2, emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1, neg0=1))
check('neg after abs gives a negative value',
      emitted == 'v31 = (s0 & 1) ? v30 : ((v30 & 0x7fffffffu) ^ 0x80000000u);', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30))
check('no modifier is unchanged', emitted == 'v31 = (s0 & 1) ? v30 : v30;', emitted)

check('abs on an inline immediate stays refused', refused(encode_vop2_e64(CNDMASK, 31, 193, V30, abs0=1)))
check('abs on another integer op stays refused', refused(encode_vop2_e64(AND_B32, 1, V30, V30, abs1=1)))

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('PASS')

# ---- WARP / hardware draws: the emitted select against the ISA reference ------------------------------------
sys.path.insert(0, here)
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import PRE_WORDS, POST_WORDS, vs_words, words_text  # noqa: E402

SIGN, MAGNITUDE = 0x80000000, 0x7fffffff
S_MOV_B32 = 0xbe800300  # SOP1 s_mov_b32; sdst in [22:16], ssrc0 in [7:0]
# Four lanes, one select each: mask s11 = 0 takes src0, mask s10 = 1 takes src1; the other source is untouched.
DRAW_SETS = (
    [(0x80000001, 0x00000002, 0), (0x7fffffff, 0xfffffffe, 0), (0xc0490fdb, 0x40490fdb, 0), (0xffc00000, 0x80000000, 0)],
    [(0x00000000, 0x80000000, 0), (0xffffffff, 0x7fffffff, 0), (0x00000000, 0xffffffff, 0), (0x80000000, 0x00000000, 0)],
)


def vop3_cndmask(vdst, src0, src1, mask_sgpr, **mods):
    """v_cndmask_b32_e64 vdst, src0, src1, s<mask_sgpr> with VGPR sources and abs/neg flags."""
    code = encode_vop2_e64(CNDMASK, vdst, V0 + src0, V0 + src1, **mods)
    w0, w1 = struct.unpack('<II', code)
    return struct.pack('<II', w0, w1 | (mask_sgpr << 18))


V0 = 256
PS_BODY = (struct.pack('<2I', S_MOV_B32 | (10 << 16) | 0x81, S_MOV_B32 | (11 << 16) | 0x80)
           + vop3_cndmask(12, 0, 4, 11, abs0=1)
           + vop3_cndmask(13, 1, 5, 10, abs1=1)
           + vop3_cndmask(14, 2, 6, 11, abs0=1, neg0=1)
           + vop3_cndmask(15, 3, 7, 10, abs1=1, neg1=1))
PS_WORDS = PRE_WORDS + PS_BODY + POST_WORDS


def reference(sets):
    """ISA result per lane: |a0|, |b1|, -|a2|, -|b3| (abs clears the sign bit, then neg flips it)."""
    return [sets[0][0] & MAGNITUDE, sets[1][1] & MAGNITUDE, (sets[2][0] & MAGNITUDE) ^ SIGN,
            (sets[3][1] & MAGNITUDE) ^ SIGN]


def without_abs(sets):
    return [sets[0][0], sets[1][1], sets[2][0] ^ SIGN, sets[3][1] ^ SIGN]


def check_draw(dxc, renderer, adapter, fxc=None):
    text = G.Pipeline(vs_words(DRAW_SETS[0]), PS_WORDS, root_blob()).hlsl_ps()
    if ''.join(l for l in text.splitlines() if l.strip().startswith('v1') and ' = (' in l).count('0x7fffffffu') != 4:
        raise AssertionError('expected four abs masks in the emitted select\n' + text)
    mutated = re.sub(r'(v1[2-5] = .*)& 0x7fffffffu', lambda m: m.group(1) + '& 0xffffffffu', text)
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            good_ps = compile_hlsl(compiler, kind, text, 'ps_abs_%s' % kind, ps_profile, directory)
            bad_ps = compile_hlsl(compiler, kind, mutated, 'ps_abs_mut_%s' % kind, ps_profile, directory)
            for index, sets in enumerate(DRAW_SETS):
                vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(sets), PS_WORDS, root_blob()).hlsl_vs(),
                                  'vs_abs_%d_%s' % (index, kind), vs_profile, directory)
                expected = words_text(reference(sets))
                run = draw(renderer, adapter, vs, good_ps, expected)
                if run.returncode:
                    raise AssertionError('%s draw %d\n%s\n%s' % (kind, index, run.stdout, run.stderr))
                print('%s draw %d: %s' % (kind, index, expected), flush=True)
                wrong = words_text(without_abs(sets))
                if wrong == expected:
                    continue
                run = draw(renderer, adapter, vs, bad_ps, expected)
                if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                    raise AssertionError('%s mutation did not read %s\n%s\n%s' % (kind, wrong, run.stdout, run.stderr))
                print('%s mutation: fails as expected, reads %s' % (kind, wrong), flush=True)


if len(sys.argv) > 1:
    if sys.argv[1] != 'draw':
        raise SystemExit(__doc__)
    check_draw(*sys.argv[2:])
