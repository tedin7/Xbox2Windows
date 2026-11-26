"""One-dimensional image loads and explicit-level samples in tools/startup/gfx2hlsl.py.

  (no arguments)     translation controls, pure Python
  binary <llvm-mc>   reassemble every fixture source and compare it with the recorded words

A 1D image (dim:SQ_RSRC_IMG_1D) takes one address component: x. image_load reads Texture1D.Load(int2(x, 0)),
image_load_mip addresses x and then the mip and reads Load(int2(x, mip)), image_sample_lz samples level 0 at x, and
image_sample_l addresses x and then the level and reads SampleLevel(smp, x, level). The mip and the level are the
LAST address register, so the NSA lists [v3, v1] and [v3, v0] read x from v3, not from the first consecutive
register. The implicit-derivative image_sample on a 1D texture is checked by startup_image_3d_check.py.

Each fixture is a source-built gfx1013 pixel stage that sets v0 = 0.5, v1 = 0.25, v2 = 1.0, v3 = 2.0 and runs one image
instruction on a texture from descriptor table 0 (and an all-zero constant sampler for the sampling forms). The
modifiers and shapes the lowering does not model are refused instead of being read at a wrong address; only the
cache-policy bits glc, slc and dlc, which never change the value, are accepted next to dim and dmask.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob  # noqa: E402

PRE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, 1.0
v_mov_b32 v3, 2.0
s_waitcnt lgkmcnt(0)
"""
POST = """\
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
SAMPLER = 's[16:23], s[24:27]'
IMAGE = {
    'load': 'image_load v[4:7], v0, s[16:23] dmask:0xf dim:SQ_RSRC_IMG_1D',
    'load_x': 'image_load v4, v0, s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_1D',
    'load_mip': 'image_load_mip v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_1D',
    'load_mip_nsa': 'image_load_mip v[4:7], [v3, v1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_1D',
    'lz': 'image_sample_lz v[4:7], v0, ' + SAMPLER + ' dmask:0xf dim:SQ_RSRC_IMG_1D',
    'l': 'image_sample_l v[4:7], v[0:1], ' + SAMPLER + ' dmask:0xf dim:SQ_RSRC_IMG_1D',
    'l_nsa': 'image_sample_l v[4:7], [v3, v0], ' + SAMPLER + ' dmask:0xf dim:SQ_RSRC_IMG_1D',
    'l_glc': 'image_sample_l v[4:7], v[0:1], ' + SAMPLER + ' dmask:0xf dim:SQ_RSRC_IMG_1D glc',
    'load_unorm': 'image_load v[4:7], v0, s[16:23] dmask:0xf dim:SQ_RSRC_IMG_1D unorm',
    'load_tfe': 'image_load v[4:8], v0, s[16:23] dmask:0xf dim:SQ_RSRC_IMG_1D tfe',
    'lz_a16': 'image_sample_lz v[4:7], v0, ' + SAMPLER + ' dmask:0xf dim:SQ_RSRC_IMG_1D a16',
    'load_mip_unorm': 'image_load_mip v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_1D unorm',
    'l_dmask0': 'image_sample_l v4, v[0:1], ' + SAMPLER + ' dmask:0x0 dim:SQ_RSRC_IMG_1D',
    'load_dmask0': 'image_load v4, v0, s[16:23] dmask:0x0 dim:SQ_RSRC_IMG_1D',
}
WORDS = {
    'load': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000f00f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_x': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000100f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_mip': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000f04f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_mip_nsa': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf020f04f00304040001000000703f8cbf'
        '0f1800f804050607000081bf'),
    'lz': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000f9cf00004c400703f8cbf0f1800f8'
        '04050607000081bf'),
    'l': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000f90f00004c400703f8cbf0f1800f8'
        '04050607000081bf'),
    'l_nsa': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf020f90f00304c40000000000703f8cbf'
        '0f1800f804050607000081bf'),
    'l_glc': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf002f90f00004c400703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_unorm': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf001f00f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_tfe': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000f01f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
    'lz_a16': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000f9cf00004c440703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_mip_unorm': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf001f04f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
    'l_dmask0': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000090f00004c400703f8cbf0f1800f8'
        '04050607000081bf'),
    'load_dmask0': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf000000f000040400703f8cbf0f1800f8'
        '04050607000081bf'),
}


def source(name):
    return PRE + IMAGE[name] + '\n' + POST


def stage(name):
    return G.Pipeline(VS_WORDS, WORDS[name], root_blob(flags=0))


def call(name):
    """The one texture call of a lowered fixture, and its pipeline."""
    pipe = stage(name)
    text = pipe.hlsl_ps()
    calls = re.findall(r'^\s*t = (r0\.(?:Load|SampleLevel)\(.*\));$', text, re.M)
    assert len(calls) == 1, (name, text)
    assert 'Texture1D<float4> r0 : register(t0);' in text, text
    assert 'srv 0 tex1d table 0 0' in pipe.manifest(), pipe.manifest()
    return calls[0]


def refused(name):
    try:
        stage(name).hlsl_ps()
    except G.Unsupported as error:
        assert re.fullmatch(r'unverified 1D image \w+ address or modifiers', str(error)), (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


def controls():
    assert call('load') == 'r0.Load(int2(v0, 0))', call('load')
    assert call('load_x') == 'r0.Load(int2(v0, 0))', call('load_x')
    assert call('load_mip') == 'r0.Load(int2(v0, v1))', call('load_mip')  # x v0, mip v1
    assert call('load_mip_nsa') == 'r0.Load(int2(v3, v1))', call('load_mip_nsa')  # the list order, not v[n]
    print('load: Load(int2(x, 0)); load_mip: Load(int2(x, mip)), also through an NSA list')

    assert re.fullmatch(r'r0\.SampleLevel\(q0, F\(v0\), 0\)', call('lz')), call('lz')
    assert re.fullmatch(r'r0\.SampleLevel\(q0, F\(v0\), F\(v1\)\)', call('l')), call('l')
    assert re.fullmatch(r'r0\.SampleLevel\(q0, F\(v3\), F\(v0\)\)', call('l_nsa')), call('l_nsa')
    assert call('l_glc') == call('l'), call('l_glc')  # a cache-policy bit does not change the value
    print('sample_lz: level 0 at x; sample_l: level is the last address register (v1, NSA v0), x from the first')

    for name in ('load_unorm', 'load_tfe', 'lz_a16', 'load_mip_unorm', 'l_dmask0', 'load_dmask0'):
        refused(name)
    print('refused: unorm, tfe, a16, empty dmask')


def check_binary(llvm_mc):
    for name in WORDS:
        built = assemble(llvm_mc, source(name))
        if built != WORDS[name]:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), WORDS[name].hex()))
    print('fixture words match llvm-mc')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'binary' and len(sys.argv) == 3:
        check_binary(sys.argv[2])
    else:
        sys.exit(__doc__)
