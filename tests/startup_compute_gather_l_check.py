"""Explicit-LOD 2D image_gather4_l in tools/startup/gfx2hlsl.py (pixel and compute stages).

  (no arguments)                                   translation and refusal controls, pure Python
  --llvm-mc <llvm-mc>                                reassemble every fixture source and compare bytes
  --dispatch <dxc> <offscreen_check> <adapter> [fxc] translate, compile and verify GPU readback
"""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))

from compute_gather4_l_check import HEIGHT, ORDER, SENTINEL, WIDTH, first_mismatch, run_case, texel  # noqa: E402
from pixel_gather_check import SIDE as PS_SIDE, texel as ps_texel, write_case as write_ps_case  # noqa: E402
from startup_gfx2hlsl_compute_check import G, USER7, WORDS, assemble, refused, root_blob  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob as ps_root_blob  # noqa: E402

CHANNELS = ('Red', 'Green', 'Blue', 'Alpha')

COMPUTE_SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[16:23], s[10:11], 0
s_load_dwordx4 s[28:31], s[10:11], 32
s_mov_b32 s24, 0x12
s_mov_b32 s25, {w1}
s_mov_b32 s26, {w2}
s_mov_b32 s27, 0
v_mov_b32 v0, {u}
v_mov_b32 v1, {v}
v_mov_b32 v2, {lod}
s_waitcnt lgkmcnt(0)
image_gather4_l v[4:7], v[0:2], s[16:23], s[24:27] dmask:{mask:#x} dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32 v3, 0
buffer_store_dwordx4 v[4:7], v3, s[28:31], 0 offen
s_endpgm
"""
DYNAMIC_LOD_SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[16:23], s[10:11], 0
s_load_dwordx4 s[28:31], s[10:11], 32
s_mov_b32 s24, 0x12
s_mov_b32 s25, 0xfff000
s_mov_b32 s26, 0x04500000
s_mov_b32 s27, 0
v_cvt_f32_u32 v2, v0
v_lshlrev_b32 v3, 4, v0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0.5
s_waitcnt lgkmcnt(0)
image_gather4_l v[4:7], v[0:2], s[16:23], s[24:27] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[4:7], v3, s[28:31], 0 offen
s_endpgm
"""
PIXEL_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0x12
s_mov_b32 s25, 0xfff000
s_mov_b32 s26, 0x04500000
s_mov_b32 s27, 0
v_mov_b32 v0, {u}
v_mov_b32 v1, {v}
v_mov_b32 v2, {lod}
s_waitcnt lgkmcnt(0)
image_gather4_l v[4:7], v[0:2], s[16:23], s[24:27] dmask:{mask:#x} dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""

COMPUTE_HEAD = bytes.fromhex('06038abe80038bbe05040cf4000000fa050708f4200000fa920398be')
COMPUTE_TAIL = bytes.fromhex('703f8cbf8002067e001078e003040780000081bf')
PIXEL_HEAD = bytes.fromhex('04040cf4000000fa920398be')
PIXEL_TAIL = bytes.fromhex('703f8cbf0f1800f804050607000081bf')

DYNAMIC_LOD_PREFIX = bytes.fromhex(
    '06038abe80038bbe05040cf4000000fa050708f4200000fa'
    '920398beff0399be00f0ff00ff039abe0000500480039bbe'
    '000d047e84000634f002007ef002027e7fc08cbf'
)
DYNAMIC_LOD_SUFFIX = bytes.fromhex('703f8cbf001078e003040780000081bf')

# Pairs of (llvm-mc token, encoded VOP1 v_mov_b32 operand words).
INLINE_VALS = {
    '0': b'\x80',
    '0.5': b'\xf0',
    '-0.5': b'\xf1',
    '1.0': b'\xf2',
    '-1.0': b'\xf3',
    '2.0': b'\xf4',
}


def s_mov_word(reg, value):
    if 0 <= value <= 64:
        return struct.pack('<I', 0xbe800380 | ((reg & 0x7f) << 16) | value)
    return struct.pack('<II', 0xbe8003ff | ((reg & 0x7f) << 16), value & 0xffffffff)


def v_mov_word(reg, token):
    head = 0x7e000200 | ((reg & 0xff) << 17)
    if token in INLINE_VALS:
        return struct.pack('<I', head | INLINE_VALS[token][0])
    return struct.pack('<II', head | 0xff, int(token, 0) & 0xffffffff)


def gather_insn(mask=1, dim=1, opcode=0x44, flags=0, high=0, nsa=False, dst=4, addr=0):
    first = 0xf0000000 | (opcode << 18) | (mask << 8) | (dim << 3) | flags
    if nsa:
        first |= 2
    ext = bytes((1, 2, 0, 0)) if nsa else b''
    return struct.pack('<II', first, 0x00c40000 | (dst << 8) | (addr & 0xff) | high) + ext


def compute_prefix(w1=0xfff000, w2=0x04500000, u='0.5', v='0.5', lod='1.0'):
    return (
        COMPUTE_HEAD
        + s_mov_word(25, w1)
        + s_mov_word(26, w2)
        + s_mov_word(27, 0)
        + v_mov_word(0, u)
        + v_mov_word(1, v)
        + v_mov_word(2, lod)
        + bytes.fromhex('7fc08cbf')
    )


def compute_shader(mask=2, prefix=None, **kw):
    return (compute_prefix(**kw) if prefix is None else prefix) + gather_insn(mask) + COMPUTE_TAIL


def pixel_prefix(w1=0xfff000, w2=0x04500000, u='0.5', v='0.5', lod='1.0'):
    return (
        PIXEL_HEAD
        + s_mov_word(25, w1)
        + s_mov_word(26, w2)
        + s_mov_word(27, 0)
        + v_mov_word(0, u)
        + v_mov_word(1, v)
        + v_mov_word(2, lod)
        + bytes.fromhex('7fc08cbf')
    )


def pixel_shader(mask=1, prefix=None, **kw):
    return (pixel_prefix(**kw) if prefix is None else prefix) + gather_insn(mask) + PIXEL_TAIL


def dynamic_lod_shader():
    return DYNAMIC_LOD_PREFIX + gather_insn(2) + DYNAMIC_LOD_SUFFIX


def cs_pipe(code, group=(1, 1, 1)):
    return G.ComputePipeline(code, root_blob(), USER7, group)


def ps_pipe(code):
    return G.Pipeline(VS_WORDS, code, ps_root_blob(flags=0))


def check_ps_refused(code, reason):
    refused(reason, lambda: ps_pipe(code))


def check_image_provenance(llvm_mc):
    def encoded(source, hex_words):
        words = bytes.fromhex(hex_words)
        if llvm_mc:
            assert assemble(llvm_mc, source + '\n') == words, source
        return words

    base = compute_prefix()
    duplicate = encoded('s_load_dwordx8 s[40:47], s[10:11], 0', '050a0cf4000000fa')
    assert 'srv 0 tex2d table 1 0' in cs_pipe(compute_shader(2, base + duplicate)).manifest()
    for reg, words in ((41, '29ff298800000400'), (46, '2eff2e8800000400')):
        change = encoded(f's_or_b32 s{reg}, s{reg}, 0x40000', words)
        refused('modified image descriptor', lambda: cs_pipe(compute_shader(2, base + duplicate + change)))

    # s_mov_b32 s22, 0 overwrites one word while the first word keeps pointing to table slot 0.
    refused('modified image descriptor', lambda: cs_pipe(compute_shader(2, base + bytes.fromhex('800396be'))))
    # Short four-word load leaves high descriptor words unknown.
    short_image = base.replace(bytes.fromhex('05040cf4'), bytes.fromhex('050408f4'))
    refused('modified image descriptor', lambda: cs_pipe(compute_shader(2, short_image)))
    no_image = base.replace(bytes.fromhex('05040cf4000000fa'), b'')
    refused('fixed table image descriptor', lambda: cs_pipe(compute_shader(2, no_image)))
    # Aliased output and texture table slot.
    aliased = base.replace(bytes.fromhex('050708f4200000fa'), bytes.fromhex('050708f4000000fa'))
    refused('used for a load and a store', lambda: cs_pipe(compute_shader(2, aliased)))

    # Fetch the output first so the x16 load can reuse earlier SGPRs; s[12:27] straddles two descriptors,
    # whereas s[8:23] lands cleanly on table byte offset 32.
    original = bytes.fromhex('05040cf4000000fa050708f4200000fa')
    output = encoded('s_load_dwordx4 s[28:31], s[10:11], 64', '050708f4400000fa')
    middle = encoded('s_load_dwordx16 s[12:27], s[10:11], 0', '050310f4000000fa')
    aligned = encoded('s_load_dwordx16 s[8:23], s[10:11], 0', '050210f4000000fa')
    refused('fixed table image descriptor', lambda: cs_pipe(compute_shader(2, base.replace(original, output + middle))))
    manifest = cs_pipe(compute_shader(2, base.replace(original, output + aligned))).manifest().splitlines()
    assert 'srv 0 tex2d table 1 32' in manifest and 'uav 0 table 1 64' in manifest, manifest


def controls(llvm_mc):
    if llvm_mc:
        assert dynamic_lod_shader() == assemble(llvm_mc, DYNAMIC_LOD_SOURCE)
    for component, name in enumerate(CHANNELS):
        mask = 1 << component
        for u, v, lod in (('0.5', '0.5', '0'), ('0.5', '0.5', '1.0'), ('-0.5', '0x3fa00000', '2.0')):
            code = compute_shader(mask, u=u, v=v, lod=lod)
            if llvm_mc:
                src = COMPUTE_SOURCE.format(w1='0xfff000', w2='0x4500000', u=u, v=v, lod=lod, mask=mask)
                assert code == assemble(llvm_mc, src)
            pipe = cs_pipe(code)
            hlsl = pipe.hlsl_cs()
            assert f't = gather_2d_lod(r0, float2(' in hlsl and f', {component}, 15.99609375f);' in hlsl, hlsl
            assert 'float4 gather_2d_lod(' in hlsl, hlsl
            assert pipe.manifest().splitlines()[-2:] == ['srv 0 tex2d table 1 0', 'uav 0 table 1 32'], pipe.manifest()

        ps_code = pixel_shader(mask, u='0.5', v='0.5', lod='1.0')
        if llvm_mc:
            assert ps_code == assemble(
                llvm_mc, PIXEL_SOURCE.format(u='0.5', v='0.5', lod='1.0', mask=mask)
            )
        ps_p = ps_pipe(ps_code)
        ps_hlsl = ps_p.hlsl_ps()
        assert f't = gather_2d_lod(r0, float2(F(v0), F(v1)), F(v2), {component}, 15.99609375f);' in ps_hlsl, ps_hlsl
        assert 'float4 gather_2d_lod(' in ps_hlsl, ps_hlsl

    # Overlapping destination and address VGPRs read inputs before overwriting v[0:3].
    overlap = pixel_prefix() + gather_insn(mask=1, dst=0, addr=0) + PIXEL_TAIL
    overlap_hlsl = ps_pipe(overlap).hlsl_ps()
    call_pos = overlap_hlsl.index('t = gather_2d_lod(r0, float2(F(v0), F(v1)), F(v2), 0, 15.99609375f);')
    assert call_pos < overlap_hlsl.index('v0 = U(t.x);'), overlap_hlsl

    # CLAMP_Z (bits 8:6 of word 0) and point XY filtering (word 2 bits 21:20, 23:22 = 0) translate cleanly.
    for w0, w2 in ((0x92, 0x04500000), (0x12, 0x04000000), (0x92, 0x04100000)):
        c_code = compute_shader(2, compute_prefix(w2=w2).replace(bytes.fromhex('920398be'), s_mov_word(24, w0)))
        assert 'gather_2d_lod(' in cs_pipe(c_code).hlsl_cs()
    # MAX_LOD values map accurately to the helper limit argument; MIP_POINT_PRECLAMP (word 2 bit 28) is refused for _l.
    assert ', 1, 1.0f);' in cs_pipe(compute_shader(2, w1=0x100000)).hlsl_cs()
    assert ', 1, 0.0f);' in cs_pipe(compute_shader(2, w1=0)).hlsl_cs()
    for w2 in (0x14500000, 0x14000000):
        refused('requires normalized clamp', lambda w2=w2: cs_pipe(compute_shader(2, w2=w2)))
        refused('requires normalized clamp', lambda w2=w2: ps_pipe(pixel_shader(1, w2=w2)))

    # Modifier, dim, dmask, register-range, and sampler refusals on both pixel and compute stages.
    for pipe_fn, mk in ((check_ps_refused, lambda **kw: PIXEL_HEAD + s_mov_word(25, 0xfff000) + s_mov_word(26, 0x04500000)
                         + s_mov_word(27, 0) + v_mov_word(0, '0.5') + v_mov_word(1, '0.5') + v_mov_word(2, '1.0')
                         + bytes.fromhex('7fc08cbf') + gather_insn(**kw) + PIXEL_TAIL),
                        (lambda code, reason: refused(reason, lambda: cs_pipe(code)),
                         lambda **kw: compute_prefix() + gather_insn(**kw) + COMPUTE_TAIL)):
        pipe_fn(mk(mask=0), 'one-hot dmask')
        pipe_fn(mk(mask=3), 'one-hot dmask')
        pipe_fn(mk(mask=0xf), 'one-hot dmask')
        pipe_fn(mk(mask=1, dim=5), '2D texture')
        pipe_fn(mk(mask=1, high=1 << 30), 'modifier a16')
        pipe_fn(mk(mask=1, high=1 << 31), 'modifier d16')
        pipe_fn(mk(mask=1, flags=1 << 16), 'modifier tfe')
        pipe_fn(mk(mask=1, flags=1 << 12), 'modifier unorm')
        pipe_fn(mk(mask=1, flags=1 << 15), 'modifier r128')
        pipe_fn(mk(mask=1, flags=1 << 17), 'modifier lwe')
        pipe_fn(mk(mask=1, nsa=True), 'three contiguous address registers')
        pipe_fn(mk(mask=1, dst=0xfd), 'four destination registers')
        pipe_fn(mk(mask=1, dst=0xff), 'four destination registers')
        pipe_fn(mk(mask=1, addr=0xfe), 'three contiguous address registers')
        pipe_fn(mk(mask=1, opcode=0x45), 'image_gather4_b')
        pipe_fn(mk(mask=1, opcode=0x4c), 'image_gather4_c_l')

    # Cache policy bits (dlc=bit 7, glc=bit 13, slc=bit 25) are accepted and preserve the same gather output.
    base_hlsl = cs_pipe(compute_shader(2)).hlsl_cs()
    for flag in (1 << 7, 1 << 13, 1 << 25):
        assert cs_pipe(compute_prefix() + gather_insn(mask=2, flags=flag) + COMPUTE_TAIL).hlsl_cs() == base_hlsl
        assert ps_pipe(pixel_prefix() + gather_insn(mask=1, flags=flag) + PIXEL_TAIL).hlsl_ps() == ps_pipe(pixel_shader(1)).hlsl_ps()
    refused('requires normalized clamp', lambda: cs_pipe(compute_shader(2, w1=0xfff001)))  # nonzero MIN_LOD
    refused('requires normalized clamp', lambda: cs_pipe(compute_shader(2, w2=0x08500000)))  # linear mip filter
    refused('requires normalized clamp', lambda: cs_pipe(compute_shader(2, w2=0x00500000)))  # mip filter none
    refused('requires normalized clamp', lambda: cs_pipe(compute_shader(2, w2=0x04500001)))  # nonzero LOD bias
    refused('requires normalized clamp', lambda: cs_pipe(compute_shader(2, w2=0x04a00000)))  # anisotropic XY
    missing_sampler = compute_prefix().replace(s_mov_word(27, 0), b'')
    refused('constant sampler', lambda: cs_pipe(compute_shader(2, missing_sampler)))

    check_image_provenance(llvm_mc)
    print('image_gather4_l compute and pixel controls passed')


def expected_taps_at(level, channel, uv):
    width, height = WIDTH >> level, HEIGHT >> level
    origin = []
    for coord, size in ((uv[0], width), (uv[1], height)):
        scaled = coord * size - 0.5
        assert abs(scaled - round(scaled)) > 1 / 64, (uv, level, scaled)
        origin.append(math.floor(scaled))
    return [texel(level, min(max(origin[0] + dx, 0), width - 1), min(max(origin[1] + dy, 0), height - 1), channel)
            for dx, dy in ORDER]


def reference(level, channel, uv=(0.5, 0.5), *, order=ORDER):
    width, height = WIDTH >> level, HEIGHT >> level
    origin = [math.floor(coord * size - 0.5) for coord, size in ((uv[0], width), (uv[1], height))]
    taps = [texel(level, min(max(origin[0] + dx, 0), width - 1), min(max(origin[1] + dy, 0), height - 1), channel)
            for dx, dy in order]
    return taps + [SENTINEL] * (WORDS - len(taps))


def dispatch_ps(directory, compiler, renderer, adapter, fxc=None):
    hlsl = directory / 'ps_gather_l.hlsl'
    cso = directory / 'ps_gather_l.cso'
    fxc_cso = directory / 'ps_gather_l_51.cso'
    case = directory / 'ps_gather_l.case'

    for channel, first, view_mip in ((2, 0, 1), (0, 1, 1)):
        source = ps_pipe(pixel_shader(1 << channel, u='0.5', v='0x3e800000', lod=f'{view_mip}.0')).hlsl_ps()
        source = source.replace('struct PSIn {', 'struct PSIn {\n    float4 uv : TEXCOORD0;', 1)
        hlsl.write_text(source, encoding='ascii')
        extent = PS_SIDE >> view_mip
        ox, oy = (math.floor(c * extent - 0.5) for c in (0.5, 0.25))
        want = tuple(ps_texel(view_mip, min(max(ox + dx, 0), extent - 1), min(max(oy + dy, 0), extent - 1), channel)
                     for dx, dy in ((0, 1), (1, 1), (1, 0), (0, 0)))
        write_ps_case(case, 3, first, [want] * (PS_SIDE ** 2))
        if fxc:
            subprocess.run([fxc, '/nologo', '/WX', '/T', 'ps_5_1', '/E', 'main', '/Fo', str(fxc_cso), str(hlsl)], check=True)
        subprocess.run([compiler, '-WX', '-T', 'ps_6_0', '-E', 'main', '-Fo', str(cso), str(hlsl)], check=True)
        proc = subprocess.run([renderer, adapter, '--check-pixel-gather', str(cso), str(case)],
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        if proc.returncode == 77:
            return 77
        assert proc.returncode == 0 and 'pixels=64 channels=256 max_absolute_error=0' in out, out
    # Pixel sensitivity mutant: replacing F(v2) with 0.0f selects view mip 0 instead of mip 1.
    bad_ps = ps_pipe(pixel_shader(4, u='0.5', v='0x3e800000', lod='1.0')).hlsl_ps().replace(
        'struct PSIn {', 'struct PSIn {\n    float4 uv : TEXCOORD0;', 1).replace('F(v2), 2,', '0.0f, 2,')
    hlsl.write_text(bad_ps, encoding='ascii')
    subprocess.run([compiler, '-WX', '-T', 'ps_6_0', '-E', 'main', '-Fo', str(cso), str(hlsl)], check=True)
    ox, oy = (math.floor(c * (PS_SIDE >> 1) - 0.5) for c in (0.5, 0.25))
    want = tuple(ps_texel(1, min(max(ox + dx, 0), (PS_SIDE >> 1) - 1), min(max(oy + dy, 0), (PS_SIDE >> 1) - 1), 2)
                 for dx, dy in ((0, 1), (1, 1), (1, 0), (0, 0)))
    write_ps_case(case, 3, 0, [want] * (PS_SIDE ** 2))
    proc = subprocess.run([renderer, adapter, '--check-pixel-gather', str(cso), str(case)], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1 and 'pixel_mismatch x=0 y=0' in out, out
    return 0


def dispatch(directory, compiler, renderer, adapter, fxc=None):
    positive = (
        ('mip0_red', 0, 0, ('0.5', '0.5', '0', 0xfff000), (0.5, 0.5), 0),
        ('mip1_green', 1, 1, ('0.5', '0.5', '1.0', 0xfff000), (0.5, 0.5), 0),
        ('mip2_blue', 2, 2, ('0.5', '0.5', '2.0', 0xfff000), (0.5, 0.5), 0),
        ('mip1_alpha', 1, 3, ('0.5', '0.5', '1.0', 0xfff000), (0.5, 0.5), 0),
        ('view1_lod0_alpha', 0, 3, ('0.5', '0.5', '0', 0xfff000), (0.5, 0.5), 1),
        ('view1_lod1_red', 1, 0, ('0.5', '0.5', '1.0', 0xfff000), (0.5, 0.5), 1),
        ('view1_clamped_high_lod', 1, 1, ('0.5', '0.5', '2.0', 0xfff000), (0.5, 0.5), 1),
        ('edge_clamped', 1, 0, ('-0.5', '0x3fa00000', '1.0', 0xfff000), (-0.5, 1.25), 0),
        ('neg_lod_clamped', 0, 1, ('0.5', '0.5', '-1.0', 0xfff000), (0.5, 0.5), 0),
        ('frac_lod_down', 1, 1, ('0.5', '0.5', '0x3fa00000', 0xfff000), (0.5, 0.5), 0),
        ('frac_lod_up', 2, 1, ('0.5', '0.5', '0x3fe00000', 0xfff000), (0.5, 0.5), 0),
        ('sampler_max_lod_1', 1, 1, ('0.5', '0.5', '2.0', 0x100000), (0.5, 0.5), 0),
        ('sampler_max_lod_0', 0, 1, ('0.5', '0.5', '2.0', 0), (0.5, 0.5), 0),
    )
    for name, view_mip, channel, (u, v, lod, w1), uv, first in positive:
        source = cs_pipe(compute_shader(1 << channel, w1=w1, u=u, v=v, lod=lod)).hlsl_cs()
        want = reference(first + view_mip, channel, uv)
        result = run_case(directory, compiler, renderer, adapter, name, source, want, first)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, (name, result.stdout, result.stderr)

    # Non-finite coordinate or LOD returns zero without corrupting surrounding output words.
    for name, u, lod in (('nan_u', '0x7fc00000', '1.0'), ('nan_lod', '0.5', '0x7fc00000')):
        source = cs_pipe(compute_shader(2, u=u, lod=lod)).hlsl_cs()
        want = [0, 0, 0, 0] + [SENTINEL] * (WORDS - 4)
        result = run_case(directory, compiler, renderer, adapter, name, source, want, 0)
        assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, (name, result.stdout, result.stderr)

    # Multi-thread runtime LOD across 4 lanes: tid.x = 0, 1, 2, 3 -> mips 0, 1, 2, 2.
    dyn_source = cs_pipe(dynamic_lod_shader(), group=(4, 1, 1)).hlsl_cs()
    dyn_want = []
    for lane in range(4):
        dyn_want.extend(expected_taps_at(min(lane, 2), 1, (0.5, 0.5)))
    dyn_want.extend([SENTINEL] * (WORDS - len(dyn_want)))
    result = run_case(directory, compiler, renderer, adapter, 'dynamic_lod_4lanes', dyn_source, dyn_want, 0)
    assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, (result.stdout, result.stderr)

    # Negative controls: each mutation must produce the predicted mismatch count and first failing word.
    base_source = cs_pipe(compute_shader(2, u='0.5', v='0.5', lod='1.0', w1=0xfff000)).hlsl_cs()
    want = reference(1, 1)
    swapped = want.copy()
    swapped[0], swapped[3] = want[3], want[0]
    call = 't = gather_2d_lod(r0, float2(F(v0), F(v1)), F(v2), 1, 15.99609375f);'
    assert base_source.count(call) == 1
    max_limited = cs_pipe(compute_shader(2, u='0.5', v='0.5', lod='2.0', w1=0x100000)).hlsl_cs()
    mutants = (
        ('dropped_gather', base_source.replace(call, ''), 0, want, [0] * 4 + want[4:]),
        ('wrong_component', base_source.replace(', 1, 15.99609375f)', ', 0, 15.99609375f)'), 0, want, reference(1, 0)),
        ('forced_mip_zero', base_source.replace('gather_2d_lod(r0, float2(F(v0), F(v1)), F(v2)',
                                                'gather_2d_lod(r0, float2(F(v0), F(v1)), 0.0f'), 0, want, reference(0, 1)),
        ('wrong_first_mip', cs_pipe(compute_shader(8, u='0.5', v='0.5', lod='0', w1=0xfff000)).hlsl_cs(),
         0, reference(1, 3), reference(0, 3)),
        ('wrong_tap_order', base_source.replace('v4, v5, v6, v7', 'v7, v5, v6, v4'), 0, want, swapped),
        ('ignored_max_lod', max_limited.replace(', 1, 1.0f)', ', 1, 15.99609375f)'),
         0, reference(1, 1), reference(2, 1)),
    )
    for name, text, first, expected, actual in mutants:
        res = run_case(directory, compiler, renderer, adapter, name, text, expected, first)
        mismatches = sum(a != b for a, b in zip(expected, actual))
        assert res.returncode == 1 and first_mismatch(expected, actual) in res.stderr, (name, res.stderr)
        assert f'compared_words={WORDS} mismatches={mismatches}' in res.stdout, (name, res.stdout)

    if dispatch_ps(directory, compiler, renderer, adapter, fxc) == 77:
        return 77
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--dispatch', nargs='+', metavar='ARG')
    args = parser.parse_args()
    if args.dispatch:
        if len(args.dispatch) not in (3, 4):
            parser.error('--dispatch expects DXC RENDERER ADAPTER [FXC]')
        out = ROOT / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            raise SystemExit(dispatch(Path(tmp), *args.dispatch))
    controls(args.llvm_mc)
