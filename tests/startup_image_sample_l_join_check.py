"""Source-built 2D point samplers with differing address modes or maximum LOD.

No arguments: portable translation and independent dispatcher/sample oracle.
binary <llvm-mc>: reassemble every recorded fixture.
draw <dxc> <offscreen_check> <warp|hardware>: actual translated shader and manifest readback.
"""
import math
import re
import struct
import sys
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob
from startup_image_sample_b_flow_check import PS_INPUTS, assemble
from startup_image_get_lod_view_check import read_pixel

CLAMP_XY = 2 | (2 << 3)
PRE = '''s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0x4000000
s_mov_b32 s27, 0
v_mov_b32_e32 v0, 0x3f900000
v_mov_b32_e32 v1, 0x3ec00000
v_mov_b32_e32 v2, 1.0
'''
POST = '''s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
'''


def arm(w0, w1=0x200000):
    return (f's_mov_b32 s24, {w0:#x}\ns_mov_b32 s25, {w1:#x}\n'
            's_mov_b32 s26, 0x4000000\ns_mov_b32 s27, 0\n')


def source(name):
    form = 'lz' if name == 'lz' else 'b' if name == 'implicit bias' else 'l'
    pre = PRE
    cond = 's_cmp_lg_u32 s9, 0\ns_cbranch_scc1 BB1\n'
    if name == 'nonuniform':
        pre = 'v_mov_b32_e32 v8, v0\n' + pre
        cond = 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_cbranch_vccz BB1\n'
    taken = arm(CLAMP_XY | 0x8000 if name == 'invalid mode' else CLAMP_XY)
    if name == 'l_word1':
        taken = arm(0, 0)
    if name == 'high word1':
        # Bit 24 of word1 sits above the translator's whole-word1 mask; a sampler keeping it must be refused.
        taken = arm(CLAMP_XY, 0x200000 | (1 << 24))
    if name == 'partial':
        taken = f's_mov_b32 s24, {CLAMP_XY:#x}\n'
    tail = 's_mov_b32 s24, s9\n' if name == 'overwrite' else ''
    if name == 'extra branch':
        tail = 's_cbranch_scc1 BB3\nBB3:\n'
    addr = 'v[0:1]' if form == 'lz' else 'v[0:2]'
    return (pre + cond + arm(0) + 's_branch BB2\nBB1:\n' + taken + 'BB2:\n' + tail
            + 's_waitcnt lgkmcnt(0)\n'
            + f'image_sample_{form} v[4:7], {addr}, s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D\n'
            + POST)


WORDS = {
    'l': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002000ff039abe0000000480039bbe7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'lz': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002000ff039abe0000000480039bbe7fc08cbf080f9cf00004c400703f8cbf0f1800f804050607000081bf',
    'l_word1': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe050082bf800398be800399beff039abe0000000480039bbe7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'nonuniform': '0003107e04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e8010847d070086bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002000ff039abe0000000480039bbe7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'partial': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe010082bf920398be7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'overwrite': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002000ff039abe0000000480039bbe090398be7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'invalid mode': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe070082bfff0398be12800000ff0399be00002000ff039abe0000000480039bbe7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'high word1': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002001ff039abe0000000480039bbe7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
    'implicit bias': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002000ff039abe0000000480039bbe7fc08cbf080f94f00004c400703f8cbf0f1800f804050607000081bf',
    'extra branch': '04040cf4000000fa800398be800399beff039abe0000000480039bbeff02007e0000903fff02027e0000c03ef202047e098007bf070085bf800398beff0399be00002000ff039abe0000000480039bbe060082bf920398beff0399be00002000ff039abe0000000480039bbe000085bf7fc08cbf080f90f00004c400703f8cbf0f1800f804050607000081bf',
}
REFUSALS = {
    'nonuniform': 'descriptor operand s[24:27]',
    'partial': 'descriptor operand s[24:27]',
    'overwrite': 'descriptor operand s[24:27]',
    'invalid mode': 'unsupported joined sampler words',
    'high word1': 'unsupported joined sampler words',
    'implicit bias': 'image_sample_b with a sampler that is not constant',
    'extra branch': 'descriptor operand s[24:27]',
}


def pipeline(name):
    return G.Pipeline(VS_WORDS, bytes.fromhex(WORDS[name]), root_blob(), ps_hw_inputs=PS_INPUTS)


def texel(mip, x, y):
    return struct.pack('>4f', (1 + x + 4 * y + 32 * mip) / 128, x, y, 2 * mip)


# u=1.125 wraps to x=0 or clamps to the final column; v=0.375 is interior.
EXPECTED = {
    'l': (texel(1, 0, 0), texel(1, 1, 0)),
    'lz': (texel(0, 0, 1), texel(0, 3, 1)),
    'l_word1': (texel(1, 0, 0), texel(0, 0, 1)),
}


def sampler_manifest(pipe, name):
    lines = [line for line in pipe.manifest().splitlines() if line.startswith('smpl ')]
    assert len(lines) == 2, lines
    fields = [line.split() for line in lines]
    assert [int(f[1]) for f in fields] == [0, 1], lines
    samplers = [tuple(int(w, 0) for w in f[2:]) for f in fields]
    expected = ({(0, 0, 1 << 26, 0), (0, 0x200000, 1 << 26, 0)} if name == 'l_word1'
                else {(0, 0x200000, 1 << 26, 0), (CLAMP_XY, 0x200000, 1 << 26, 0)})
    assert set(samplers) == expected, samplers
    return lines, samplers


def selected_texel(text, samplers, selector):
    """Execute this authored dispatcher subset and evaluate its selected point sample independently."""
    regs = {}
    pc, scc = 0, False
    call = re.compile(r't = \(s(\d+) == (0x[0-9a-f]+)u \? r0\.SampleLevel\(q(\d+), '
                      r'float2\(F\(v0\), F\(v1\)\), (F\(v2\)|0)\) : r0\.SampleLevel\(q(\d+), '
                      r'float2\(F\(v0\), F\(v1\)\), \4\)\);')
    for block in re.finditer(r'^\s*if \(pc == (\d+)\) \{\n(.*?)^\s*\}', text, re.M | re.S):
        if int(block[1]) != pc:
            continue
        for line in block[2].splitlines():
            line = line.strip()
            m = re.fullmatch(r'([sv]\d+) = (0x[0-9a-f]+)u;', line)
            if m:
                regs[m[1]] = int(m[2], 16)
                continue
            m = re.fullmatch(r'scc = ud1 != (0x[0-9a-f]+)u;', line)
            if m:
                scc = selector != int(m[1], 16)
                continue
            m = re.fullmatch(r'pc = \(scc\) \? (\d+) : (\d+);', line)
            if m:
                pc = int(m[1 if scc else 2])
                continue
            m = re.fullmatch(r'pc = (\d+);', line)
            if m:
                pc = int(m[1])
                continue
            m = call.fullmatch(line)
            if m:
                q = int(m[3] if regs['s' + m[1]] == int(m[2], 16) else m[5])
                def f32(reg):
                    return struct.unpack('<f', struct.pack('<I', regs[reg]))[0]
                mip = int(f32('v2')) if m[4] != '0' else 0
                w0, w1, _, _ = samplers[q]
                mip = min(max(mip, (w1 & 0xfff) // 256), ((w1 >> 12) & 0xfff) // 256)
                size = 4 >> mip
                def address(coord, mode):
                    index = math.floor(coord * size)
                    return index % size if mode == 0 else min(max(index, 0), size - 1)
                return texel(mip, address(f32('v0'), w0 & 7), address(f32('v1'), (w0 >> 3) & 7))
            raise AssertionError(f'unmodelled dispatcher statement: {line}')
    raise AssertionError('dispatcher never sampled')


def controls():
    for name, expected in EXPECTED.items():
        pipe = pipeline(name)
        text = pipe.hlsl_ps()
        _, samplers = sampler_manifest(pipe, name)
        assert 'udsrc 1 rootconst 1 0' in pipe.manifest()
        assert expected[0] != expected[1]
        for selector, want in zip((0, 0x3f800000), expected):
            assert selected_texel(text, samplers, selector) == want, (name, selector)
            assert selected_texel(text, samplers[::-1], selector) != want, (name, selector, 'swapped')
    for name, reason in REFUSALS.items():
        try:
            pipeline(name).hlsl_ps()
        except G.Unsupported as error:
            assert reason in str(error), (name, reason, str(error))
        else:
            raise AssertionError(f'{name} unexpectedly accepted')
    print(f'uniform sampler diamond: {len(EXPECTED)} forms, both selector outcomes, '
          f'swapped bindings and {len(REFUSALS)} refusal controls passed')


def binary(llvm_mc):
    for name, encoded in WORDS.items():
        assert assemble(llvm_mc, source(name)) == bytes.fromhex(encoded), name
    print(f'all {len(WORDS)} authored fixtures reassemble exactly')


def draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)
        for name, expected in EXPECTED.items():
            pipe = pipeline(name)
            ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), f'joined_{name}', 'ps_6_0', directory)
            lines, _ = sampler_manifest(pipe, name)
            manifest = '\n'.join(lines)
            for selector, want in enumerate(expected):
                got = read_pixel(renderer, adapter, vs, ps, manifest, '0/3', selector=selector)
                assert got == want, (name, selector, got.hex(), want.hex())
            swapped = '\n'.join(f'smpl {i} ' + ' '.join(line.split()[2:]) for i, line in enumerate(reversed(lines)))
            got = read_pixel(renderer, adapter, vs, ps, swapped, '0/3', selector=0)
            assert got == expected[1] and got != expected[0], (name, 'swapped', got.hex())
            print(f'{name}: actual uniform root selector and swapped binding readback matched')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        controls()
    elif args[0] == 'binary' and len(args) == 2:
        binary(args[1])
    elif args[0] == 'draw' and len(args) == 4:
        draw(*args[1:])
    else:
        raise SystemExit(__doc__)
