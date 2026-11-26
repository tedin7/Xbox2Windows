"""1D image sampling in tools/startup/gfx2hlsl.py (issue #1721).

usage: startup_gfx2hlsl_image_1d_check.py [draw <dxc> <offscreen_check> <warp|hardware>]

A 1D image has one coordinate, and `image_sample_lz` / `image_sample_l` carry the explicit level in the next
address dword. The translator accepted 1D only for the implicit-level `image_sample`, so the two explicit-level
forms were refused - 2 of the 29 refused shaders in the first loading-screen cache (#870).

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, MIMG branch):
  w0: [31:26] = 0b111100, [24:18] opcode low 7 bits, [0] opcode bit 7, [11:8] dmask, [5:3] dimension,
      [2:1] NSA dword count, [12] unorm, [15] r128
  w1: [7:0] vaddr, [15:8] vdata, [20:16] srsrc/4, [25:21] sampler/4
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

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


IMG_LOAD = 0x00
IMG_SAMPLE = 0x20
IMG_SAMPLE_L = 0x24
IMG_SAMPLE_LZ = 0x27
DIM_1D = 0
DIM_2D = 1


def encode_mimg(op, *, dmask=0xF, dim=DIM_1D, vaddr=0, vdata=4, srsrc=8, ssamp=12, nsa=0,
                extra_addrs=(), unorm=0, r128=0, glc=0, slc=0, dlc=0):
    """Encode one MIMG instruction; returns its bytes (8 + 4 per NSA dword)."""
    w0 = ((0b111100 << 26) | ((op & 0x7F) << 18) | (((op >> 7) & 1) << 0)
          | ((dmask & 0xF) << 8) | ((dim & 0x7) << 3) | ((nsa & 0x3) << 1)
          | ((unorm & 1) << 12) | ((r128 & 1) << 15) | ((dlc & 1) << 7)
          | ((glc & 1) << 13) | ((slc & 1) << 25))
    w1 = ((vaddr & 0xFF) | ((vdata & 0xFF) << 8) | (((srsrc // 4) & 0x1F) << 16)
          | (((ssamp // 4) & 0x1F) << 21))
    return struct.pack('<II', w0, w1) + bytes(extra_addrs)


class Pipeline:
    """Just enough pipeline state for an image sample."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()
        self.lod_unvalidated = set()
        self.uses_sampler_heap = False

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']

    def sampler(self, key, comparison=False, lod_limits=False):
        return 'smp%d' % (1 if comparison else 0)

    def ud(self, source):
        return 0


class Stage(G.Stage):
    """A Stage with a bound image descriptor and a constant sampler, as a pixel shader would have."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()
        self.kind = 'ps'
        self.blocks = []
        self.sym[8] = ('desc', 'image0', None, 0)
        for i in range(4):
            self.sym[12 + i] = ('const', 0)


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused(code):
    try:
        translate(code)
    except G.Unsupported as error:
        return 'Unsupported' in type(error).__name__
    return False


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted = translate(encode_mimg(IMG_SAMPLE_LZ, dmask=0xF, dim=DIM_1D, vdata=4))
check('encodes and decodes a 1D sample_lz',
      asm == ['image_sample_lz v[4:7], v0, s[8:15], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_1D'], repr(asm))

# 2. A 1D sample takes a single float location; the level is zero for the lz form.
check('1D sample_lz samples level zero', 't = r0.SampleLevel(smp0, F(v0), 0);' in emitted, emitted)

# 3. image_sample_l takes the explicit level from the second address dword.
asm, emitted = translate(encode_mimg(IMG_SAMPLE_L, dmask=0x7, dim=DIM_1D, vdata=4))
check('encodes and decodes a 1D sample_l',
      asm == ['image_sample_l v[4:6], v[0:1], s[8:15], s[12:15] dmask:0x7 dim:SQ_RSRC_IMG_1D'], repr(asm))
check('1D sample_l takes the level from the address', 't = r0.SampleLevel(smp0, F(v0), F(v1));' in emitted, emitted)

# 4. The implicit-level form is unchanged.
asm, emitted = translate(encode_mimg(IMG_SAMPLE, dmask=0x3, dim=DIM_1D, vdata=4))
check('1D sample still samples implicitly', 't = r0.Sample(smp0, F(v0));' in emitted, emitted)
check('dmask selects the written channels', emitted.endswith('v4 = U(t.x); v5 = U(t.y);'), emitted)

# 5. A 2D sample is untouched by the 1D rules.
asm, emitted = translate(encode_mimg(IMG_SAMPLE_LZ, dmask=0xF, dim=DIM_2D, vdata=4))
check('2D sample_lz still uses a float2 location', 'SampleLevel(smp0, float2(F(v0), F(v1)), 0)' in emitted, emitted)

# 6. Refusals that must not widen: a unorm 1D load, a 1D sample_lz with
#    two address dwords, and a modifier that would change the address or result layout.
check('1D image_load lowers to Texture1D.Load',
      'Load(int2(' in translate(encode_mimg(IMG_LOAD, dmask=0xF, dim=DIM_1D, vdata=4))[1])
check('a unorm 1D image_load stays refused',
      refused(encode_mimg(IMG_LOAD, dmask=0xF, dim=DIM_1D, vdata=4, unorm=1)))
check('1D sample_lz with two address dwords stays refused',
      refused(encode_mimg(IMG_SAMPLE_LZ, dmask=0xF, dim=DIM_1D, vdata=4, nsa=1, extra_addrs=(1,))))
check('a unorm 1D sample_lz stays refused',
      refused(encode_mimg(IMG_SAMPLE_LZ, dmask=0xF, dim=DIM_1D, vdata=4, unorm=1)))

# 7. The cache-policy bits and the dmask guard are what the 1D rules decide, so each is pinned: the plain
#    implicit-level form keeps its old rule and still refuses a cache bit, an empty dmask is refused, and a
#    cache bit on the lz/l forms lowers to the same text as without it (the policies do not change the value).
check('a glc 1D sample stays refused',
      refused(encode_mimg(IMG_SAMPLE, dmask=0x3, dim=DIM_1D, vdata=4, glc=1)))
check('a 1D sample_lz with an empty dmask is refused',
      refused(encode_mimg(IMG_SAMPLE_LZ, dmask=0x0, dim=DIM_1D, vdata=4)))
_plain_lz = translate(encode_mimg(IMG_SAMPLE_LZ, dmask=0xF, dim=DIM_1D, vdata=4))[1]
_plain_l = translate(encode_mimg(IMG_SAMPLE_L, dmask=0x7, dim=DIM_1D, vdata=4))[1]
for _bit, _value in (('glc', 1), ('slc', 1), ('dlc', 1)):
    _cache_lz = translate(encode_mimg(IMG_SAMPLE_LZ, dmask=0xF, dim=DIM_1D, vdata=4, **{_bit: _value}))[1]
    check('a %s 1D sample_lz lowers like the plain one' % _bit, _cache_lz == _plain_lz, _cache_lz)
    _cache_l = translate(encode_mimg(IMG_SAMPLE_L, dmask=0x7, dim=DIM_1D, vdata=4, **{_bit: _value}))[1]
    check('a %s 1D sample_l lowers like the plain one' % _bit, _cache_l == _plain_l, _cache_l)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')


def draw(dxc, renderer, adapter):
    from startup_image_3d_check import FULLSCREEN_VS, WORDS_1D, compile_hlsl, translate as translate_pair

    # Use the existing Texture1D draw, with v1 reading the provided root constant for the explicit LOD.
    sample = encode_mimg(IMG_SAMPLE, vdata=0, srsrc=16, ssamp=24)
    words = WORDS_1D.replace(bytes.fromhex('ff02027e0000803e'), bytes.fromhex('0902027e'))
    assert sample in words

    def texel(level):
        x = (4 >> level) * 3 // 4
        rgba = ((1 + x + 32 * level) / 128, x, 0, 0.75 if x else 0.5)
        return ','.join(f'{struct.unpack("<I", struct.pack("<f", c))[0]:08x}' for c in rgba)

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)
        for opcode, label in [(IMG_SAMPLE_LZ, 'lz'), (IMG_SAMPLE_L, 'l')]:
            pipe = translate_pair(words.replace(sample, encode_mimg(opcode, vdata=0, srsrc=16, ssamp=24)))
            shader = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), label, 'ps_6_0', directory)

            def read(program, lod, expected):
                args = [renderer, adapter, '--startup-image-1d-mips', str(vs), str(program), str(lod), texel(expected)]
                run = subprocess.run(args, capture_output=True, text=True)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    sys.exit(77)
                return run

            for lod in (1, 2):
                level = lod if label == 'l' else 0
                run = read(shader, lod, level)
                assert run.returncode == 0, run.stdout + run.stderr
                print(run.stdout, end='')
                print(f'{adapter} sample_{label} LOD input {lod}: matched mip {level}')
            if label == 'l':
                source = pipe.hlsl_ps()
                mutant = source.replace('SampleLevel(q0, F(v0), F(v1))', 'SampleLevel(q0, F(v0), 0.0)')
                assert mutant != source, source
                wrong = compile_hlsl(dxc, 'dxc', mutant, 'fixed_zero', 'ps_6_0', directory)
                wrong_zero = read(wrong, 2, 2)
                assert wrong_zero.returncode == 1 and 'actual=' + texel(0) in wrong_zero.stderr, wrong_zero.stderr
                print(f'{adapter} fixed-zero shader read mip zero where mip two was expected')


if len(sys.argv) > 1:
    if len(sys.argv) != 5 or sys.argv[1] != 'draw':
        sys.exit(__doc__)
    draw(*sys.argv[2:])
