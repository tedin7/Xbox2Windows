"""Check gfx1013 mip-zero gathers through the startup pixel translator."""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_gather_check import SIDE, texel, write_case
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0x12
s_mov_b32 s25, 0
s_mov_b32 s26, 0x04500000
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0x3e800000
s_waitcnt lgkmcnt(0)
image_gather4_lz v[4:7], v[0:1], s[16:23], s[24:27] dmask:{mask:#x} dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
PREFIX = bytes.fromhex('04040cf4000000fa920398be800399beff039abe0000500480039bbe'
                       'f002007eff02027e0000803e7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf0f1800f804050607000081bf')
CHANNELS = ('Red', 'Green', 'Blue', 'Alpha')


def shader(mask=1, dim=1, opcode=0x47, flags=0, high=0, prefix=PREFIX, nsa=False, dst=4):
    first = 0xf0000000 | (opcode << 18) | (mask << 8) | (dim << 3) | flags
    if nsa:
        first |= 2
    extension = bytes((1, 0, 0, 0)) if nsa else b''
    return prefix + struct.pack('<II', first, 0x00c40000 | (dst << 8) | high) + extension + SUFFIX


def pipeline(code):
    return G.Pipeline(VS_WORDS, code, root_blob(flags=0))


def refused(code, reason):
    try:
        pipeline(code)
    except G.Unsupported as error:
        assert reason in str(error), error
    else:
        raise AssertionError('Gather accepted an unsupported form: ' + reason)


def controls(llvm_mc):
    for component, channel in enumerate(CHANNELS):
        mask = 1 << component
        code = shader(mask)
        if llvm_mc:
            assert code == assemble(llvm_mc, SOURCE.format(mask=mask))
        pipe = pipeline(code)
        text = pipe.hlsl_ps()
        assert f't = r0.Gather{channel}(q0, float2(F(v0), F(v1)));' in text, text
        assert all(f'v{4 + i} = U(t.{lane});' in text for i, lane in enumerate('xyzw')), text
        assert 'srv 0 tex2d table 0 0\n' in pipe.manifest(), pipe.manifest()
        assert 'smp 0 0x12 0x0 0x4500000 0x0\n' in pipe.manifest(), pipe.manifest()
    refused(shader(0), 'one-hot dmask')
    refused(shader(3), 'one-hot dmask')
    refused(shader(dim=5), '2D texture')
    refused(shader(high=1 << 30), 'modifier a16')
    refused(shader(high=1 << 31), 'modifier d16')
    refused(shader(flags=1 << 16), 'modifier tfe')
    refused(shader(nsa=True), 'two contiguous address registers')
    # The destination must be four registers that fit in the file: bases 0xfe and 0xff decode to
    # v[254:257] and v[255:258], which would otherwise write past v255.
    refused(shader(dst=0xfe), 'four destination registers')
    refused(shader(dst=0xff), 'four destination registers')
    # The explicit-LOD and fixed-mip comparison gathers are lowered by their own tests; the implicit,
    # biased and offset gathers stay refused here.
    refused(shader(opcode=0x40), 'image_gather4')
    refused(shader(opcode=0x45), 'image_gather4_b')
    refused(shader(opcode=0x54), 'image_gather4_l_o')
    refused(shader(opcode=0x48), 'image_gather4_c')
    refused(shader(opcode=0x5f), 'image_gather4_c_lz_o')
    bad_wrap = PREFIX.replace(bytes.fromhex('920398be'), bytes.fromhex('930398be'))
    bad_min_lod = PREFIX.replace(bytes.fromhex('800399be'), bytes.fromhex('810399be'))
    missing_sampler = PREFIX.replace(bytes.fromhex('80039bbe'), b'')
    refused(shader(prefix=bad_wrap), 'requires normalized clamp')
    refused(shader(prefix=bad_min_lod), 'requires normalized clamp')
    refused(shader(prefix=missing_sampler), 'constant sampler')
    check_clamp_z()
    check_point_xy()
    check_mip_point_preclamp()
    print('Four fixed-mip gather channels and unsupported encodings checked')


# CLAMP_Z (SQ_IMG_SAMP_WORD0 bits 8:6) addresses a third coordinate a 2D view does not have. Word 0 is moved
# with a literal (s_mov_b32 s24, <word>), so the words below are the 0x12 profile with CLAMP_Z set.
CLAMP_WORD0 = bytes.fromhex('920398be')


def with_word0(word):
    return PREFIX.replace(CLAMP_WORD0, bytes.fromhex('ff0398be') + struct.pack('<I', word))


def check_clamp_z():
    for clamp_z in range(1, 8):
        word = 0x12 | clamp_z << 6
        pipe = pipeline(shader(prefix=with_word0(word)))
        assert 't = r0.GatherRed(q0, float2(F(v0), F(v1)));' in pipe.hlsl_ps()
        # The title's own words are bound, CLAMP_Z included.
        assert f'smp 0 {word:#x} 0x0 0x4500000 0x0\n' in pipe.manifest(), pipe.manifest()
    # The X and Y address modes are still checked: mirror in Y, wrap in X, each with CLAMP_Z set as well.
    refused(shader(prefix=with_word0(0x0a)), 'requires normalized clamp')
    refused(shader(prefix=with_word0(0x10 | 2 << 6)), 'requires normalized clamp')
    refused(shader(prefix=with_word0(0x13 | 2 << 6)), 'requires normalized clamp')
    # Other word 0 fields beside CLAMP_Z still refuse: anisotropy ratio (bits 11:9).
    refused(shader(prefix=with_word0(0x92 | 1 << 9)), 'requires normalized clamp')
    # Negative control: without the 2D profile the CLAMP_Z word is refused, as before this change.
    profile = G.gather_2d_profile_words
    G.gather_2d_profile_words = lambda words: tuple(words)
    try:
        refused(shader(prefix=with_word0(0x92)), 'requires normalized clamp')
    finally:
        G.gather_2d_profile_words = profile


# Word 2 is moved as a literal (s_mov_b32 s26, 0x04500000: bilinear XY, point mip).
FILTER_WORD2 = bytes.fromhex('ff039abe00005004')


def with_word2(word, prefix=PREFIX):
    return prefix.replace(FILTER_WORD2, bytes.fromhex('ff039abe') + struct.pack('<I', word))


def check_point_xy():
    # Gather returns the bilinear footprint whatever the XY filter (D3D11.3 functional spec 22.4.2), so point
    # magnification, minification or both translate. The last word pair is clamp-last-texel in X, Y and Z with
    # point XY and point mip filtering.
    for word0, word2 in ((0x12, 0x04000000), (0x12, 0x04100000), (0x12, 0x04400000), (0x92, 0x04000000)):
        pipe = pipeline(shader(prefix=with_word2(word2, with_word0(word0))))
        assert 't = r0.GatherRed(q0, float2(F(v0), F(v1)));' in pipe.hlsl_ps()
        assert f'smp 0 {word0:#x} 0x0 {word2:#x} 0x0\n' in pipe.manifest(), pipe.manifest()
    # Still refused: anisotropic XY, a linear mip filter, a nonzero LOD bias field.
    refused(shader(prefix=with_word2(0x04a00000)), 'requires normalized clamp')
    refused(shader(prefix=with_word2(0x04800000)), 'requires normalized clamp')
    refused(shader(prefix=with_word2(0x08000000)), 'requires normalized clamp')
    refused(shader(prefix=with_word2(0x04000001)), 'requires normalized clamp')
    # Negative control: a profile that only clears CLAMP_Z refuses the point filter, as before this change.
    profile = G.gather_2d_profile_words
    G.gather_2d_profile_words = lambda words: (words[0] & ~(7 << 6),) + tuple(words[1:])
    try:
        refused(shader(prefix=with_word2(0x04000000, with_word0(0x92))), 'requires normalized clamp')
    finally:
        G.gather_2d_profile_words = profile


def check_mip_point_preclamp():
    # MIP_POINT_PRECLAMP (word 2 bit 28) only acts on mip selection; the _lz gather reads mip zero.
    # The last pair is the full sampler state the refused run pipelines bind.
    for word0, word2 in ((0x12, 0x14500000), (0x92, 0x14000000)):
        pipe = pipeline(shader(prefix=with_word2(word2, with_word0(word0))))
        assert 't = r0.GatherRed(q0, float2(F(v0), F(v1)));' in pipe.hlsl_ps()
        assert f'smp 0 {word0:#x} 0x0 {word2:#x} 0x0\n' in pipe.manifest(), pipe.manifest()
    # Bits 29 to 31 (ANISO_OVERRIDE, BLEND_ZERO_PRT, unnamed) stay refused.
    for bit in (29, 30, 31):
        refused(shader(prefix=with_word2(0x04500000 | 1 << bit)), 'requires normalized clamp')
    # Negative control: a profile that leaves bit 28 refuses it, as before this change.
    profile = G.gather_2d_profile_words
    G.gather_2d_profile_words = lambda words: tuple(w | (1 << 28 if i == 2 and words[2] >> 28 & 1 else 0)
                                                    for i, w in enumerate(profile(words)))
    try:
        refused(shader(prefix=with_word2(0x14000000, with_word0(0x92))), 'requires normalized clamp')
    finally:
        G.gather_2d_profile_words = profile


def expected_taps(mip, component, uv):
    extent = SIDE >> mip
    x, y = (math.floor(coord * extent - 0.5) for coord in uv)
    return tuple(texel(mip, min(max(i, 0), extent - 1), min(max(j, 0), extent - 1), component)
                 for i, j in ((x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y)))


def draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        hlsl = directory / 'gather.hlsl'
        dxil = directory / 'gather.dxil'
        case = directory / 'gather.case'

        def run(mips, first, component, uv, code, *, mutant=False):
            source = pipeline(code).hlsl_ps()
            # The offscreen vertex shader places TEXCOORD0 before SV_Position.
            source = source.replace('struct PSIn {', 'struct PSIn {\n    float4 uv : TEXCOORD0;', 1)
            if mutant:
                source = source.replace('U(t.x)', 'U(t.w)', 1)
            hlsl.write_text(source, encoding='ascii')
            subprocess.run([dxc, '-WX', '-T', 'ps_6_0', '-E', 'main', '-Fo', dxil, hlsl], check=True)
            # The renderer uploads fixture level zero at the first resident mip.
            write_case(case, mips, first, [expected_taps(0, component, uv)] * SIDE**2)
            proc = subprocess.run([renderer, adapter, '--check-pixel-gather', dxil, case],
                                  capture_output=True, text=True)
            output = proc.stdout + proc.stderr
            print(f'{CHANNELS[component]}: mip {first}, {mips} resource mips, uv {uv}, mutant {mutant}\n{output}',
                  end='', flush=True)
            if proc.returncode == 77:
                raise SystemExit(77)
            if mutant:
                assert proc.returncode == 1 and 'pixel_mismatch x=0 y=0' in output, output
            else:
                assert proc.returncode == 0 and 'pixels=64 channels=256 max_absolute_error=0' in output, output

        uv = (0.5, 0.25)
        for channel in range(4):
            run(1, 0, channel, uv, shader(1 << channel))
            run(3, 1, channel, uv, shader(1 << channel))
        clamped = PREFIX.replace(bytes.fromhex('f002007e'), bytes.fromhex('f102007e'))
        clamped = clamped.replace(bytes.fromhex('0000803e'), bytes.fromhex('0000a03f'))
        run(3, 0, 1, (-0.5, 1.25), shader(2, prefix=clamped))
        # The same draw with the title's CLAMP_Z set: a 2D gather cannot observe it.
        run(3, 1, 0, uv, shader(1, prefix=with_word0(0x92)))
        # Point XY filtering with CLAMP_Z set, the sampler state of the refused run pipelines: the same four texels.
        run(3, 1, 3, uv, shader(8, prefix=with_word2(0x04000000, with_word0(0x92))))
        # The full sampler state of those pipelines, MIP_POINT_PRECLAMP included.
        run(3, 1, 1, uv, shader(2, prefix=with_word2(0x14000000, with_word0(0x92))))
        run(3, 1, 2, uv, shader(4), mutant=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--draw', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    options = parser.parse_args()
    if options.draw:
        draw(*options.draw)
    else:
        controls(options.llvm_mc)
