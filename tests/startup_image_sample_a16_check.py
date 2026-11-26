"""Packed 2D A16 coordinates and modifier validation for image_sample and image_sample_lz in gfx2hlsl.py.

  (no arguments)                       translation, decoder, cache-revision and refusal controls, pure Python
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  decode <libLLVM>                     disassemble the MIMG words with LLVM's C API and compare with disasm_rdna2.py
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw against a
                                       2x2 R32G32B32A32_FLOAT Texture2D: packed binary16 (u, v) halves in one VGPR
                                       must select all four quadrants and wrap negative half-floats, while pre-fix and
                                       bit-shift mutants fail readback

On RDNA 2 (gfx1013), 2D image_sample and image_sample_lz with a16 pack two IEEE binary16 coordinates (u in bits 15:0,
v in bits 31:16) into a single VGPR without an NSA extension. Before this fix, disasm_rdna2.py decoded a two-register
range v[N:N+1] and Stage.image() lowered the sample without checking modifiers, reading the packed word as float u and
the adjacent VGPR v(N+1) as float v. Both the offline and runtime startup shader cache revisions advance together so
cached pre-fix translations are not reused.
"""
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
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob, texel  # noqa: E402

IMG_SAMPLE = 0x20
IMG_SAMPLE_L = 0x24
IMG_SAMPLE_LZ = 0x27
DIM_1D = 0
DIM_2D = 1
DIM_3D = 2
DIM_2D_ARRAY = 5

# Shared pixel-stage preamble:
#   s[16:23] = texture descriptor from table 0, s[24:27] = all-zero inline sampler (point wrap),
#   v0 = s9 (ud1, the packed u | (v << 16) coordinate bits), v1 = 0.75 sentinel (pre-fix read v1 as v),
#   v2 = s9 (non-v0 packed coordinate register), v3 = 0.25 (non-a16 NSA second coordinate).
PRE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, s9
v_mov_b32 v1, 0x3f400000
v_mov_b32 v2, s9
v_mov_b32 v3, 0x3e800000
s_waitcnt lgkmcnt(0)
"""
PRE_WORDS = bytes.fromhex(
    '04040cf4000000fa' '800398be' '800399be' '80039abe' '80039bbe'
    '0902007e' 'ff02027e0000403f' '0902047e' 'ff02067e0000803e' '7fc08cbf')

# Variant preamble where v0 holds 0x3f400000 so only v2 carries s9 (proves cs[0] is not hardcoded to v0).
PRE_V2_ONLY = PRE.replace('v_mov_b32 v0, s9', 'v_mov_b32 v0, 0x3f400000')
PRE_V2_ONLY_WORDS = PRE_WORDS.replace(bytes.fromhex('0902007e'), bytes.fromhex('ff02007e0000403f'), 1)

POST = """\
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
POST_WORDS = bytes.fromhex('703f8cbf' '0f1800f804050607' '000081bf')


def encode_mimg(op, *, dmask=0xf, dim=DIM_2D, vaddr=0, vdata=4, srsrc=16, ssamp=24, nsa=0, extra_addrs=(),
                dlc=0, unorm=0, glc=0, r128=0, tfe=0, lwe=0, slc=0, a16=0, d16=0, w0_reserved=0, w1_reserved=0):
    w0 = (0xf0000000 | ((op & 0x7f) << 18) | ((op >> 7) & 1) | ((nsa & 0x3) << 1)
          | ((dim & 0x7) << 3) | ((dlc & 1) << 7) | ((dmask & 0xf) << 8)
          | ((unorm & 1) << 12) | ((glc & 1) << 13) | ((r128 & 1) << 15)
          | ((tfe & 1) << 16) | ((lwe & 1) << 17) | ((slc & 1) << 25) | w0_reserved)
    w1 = ((vaddr & 0xff) | ((vdata & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16)
          | (((ssamp // 4) & 0x1f) << 21) | ((a16 & 1) << 30) | ((d16 & 1) << 31) | w1_reserved)
    ext = bytes(extra_addrs) + bytes(max(0, 4 * nsa - len(extra_addrs)))
    return struct.pack('<II', w0, w1) + ext


# Source-built MIMG instructions checked against llvm-mc.
MIMG_LLVM_MC = {
    'sample_a16': (
        'image_sample v[4:7], v0, s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D a16',
        encode_mimg(IMG_SAMPLE, a16=1),
    ),
    'lz_a16': (
        'image_sample_lz v[4:7], v0, s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D a16',
        encode_mimg(IMG_SAMPLE_LZ, a16=1),
    ),
    'lz_v2_cache': (
        'image_sample_lz v[4:7], v2, s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D glc slc dlc a16',
        encode_mimg(IMG_SAMPLE_LZ, vaddr=2, glc=1, slc=1, dlc=1, a16=1),
    ),
    'sample_dmask5_a16': (
        'image_sample v[4:5], v0, s[16:23], s[24:27] dmask:0x5 dim:SQ_RSRC_IMG_2D a16',
        encode_mimg(IMG_SAMPLE, dmask=0x5, a16=1),
    ),
    'sample_plain': (
        'image_sample v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D',
        encode_mimg(IMG_SAMPLE),
    ),
    'lz_plain': (
        'image_sample_lz v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D',
        encode_mimg(IMG_SAMPLE_LZ),
    ),
    'sample_nsa': (
        'image_sample v[4:7], [v0, v3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D',
        encode_mimg(IMG_SAMPLE, nsa=1, extra_addrs=(3,)),
    ),
    'lz_nsa': (
        'image_sample_lz v[4:7], [v0, v3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D',
        encode_mimg(IMG_SAMPLE_LZ, nsa=1, extra_addrs=(3,)),
    ),
    'sample_glc': (
        'image_sample v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D glc',
        encode_mimg(IMG_SAMPLE, glc=1),
    ),
    'sample_slc': (
        'image_sample v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D slc',
        encode_mimg(IMG_SAMPLE, slc=1),
    ),
    'sample_dlc': (
        'image_sample v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D dlc',
        encode_mimg(IMG_SAMPLE, dlc=1),
    ),
}
for _prefix, _op in (('sample', IMG_SAMPLE), ('lz', IMG_SAMPLE_LZ)):
    _mnem = 'image_sample' if _op == IMG_SAMPLE else 'image_sample_lz'
    MIMG_LLVM_MC[f'{_prefix}_unorm'] = (
        f'{_mnem} v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D unorm',
        encode_mimg(_op, unorm=1),
    )
    MIMG_LLVM_MC[f'{_prefix}_tfe'] = (
        f'{_mnem} v[4:5], v[0:1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D tfe',
        encode_mimg(_op, dmask=0x1, tfe=1),
    )
    MIMG_LLVM_MC[f'{_prefix}_lwe'] = (
        f'{_mnem} v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D lwe',
        encode_mimg(_op, lwe=1),
    )
    MIMG_LLVM_MC[f'{_prefix}_d16'] = (
        f'{_mnem} v[4:5], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(_op, d16=1),
    )
    MIMG_LLVM_MC[f'{_prefix}_r128'] = (
        f'{_mnem} v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D r128',
        encode_mimg(_op, r128=1),
    )
    MIMG_LLVM_MC[f'{_prefix}_dmask0'] = (
        f'{_mnem} v4, v[0:1], s[16:23], s[24:27] dmask:0x0 dim:SQ_RSRC_IMG_2D',
        encode_mimg(_op, dmask=0x0),
    )
    MIMG_LLVM_MC[f'{_prefix}_dmask0_a16'] = (
        f'{_mnem} v4, v0, s[16:23], s[24:27] dmask:0x0 dim:SQ_RSRC_IMG_2D a16',
        encode_mimg(_op, dmask=0x0, a16=1),
    )

# Valid MIMG instructions whose disasm_rdna2 text matches LLVM 15's disassembler verbatim.
DECODE_KEYS = (
    'sample_a16', 'lz_a16', 'lz_v2_cache', 'sample_dmask5_a16',
    'sample_plain', 'lz_plain', 'sample_nsa', 'lz_nsa',
    'sample_glc', 'sample_slc', 'sample_dlc',
)

RESERVED_BITS = (
    ('w0_6', 1 << 6, 0),
    ('w0_14', 1 << 14, 0),
    *(('w1_%d' % bit, 0, 1 << bit) for bit in range(26, 30)),
)


def pixel_words(mimg_bytes, pre_words=PRE_WORDS):
    return pre_words + mimg_bytes + POST_WORDS


def translate(mimg_bytes, pre_words=PRE_WORDS):
    return G.Pipeline(VS_WORDS, pixel_words(mimg_bytes, pre_words), root_blob())


def expect_refused(fn, expected_substring, label):
    try:
        fn()
    except G.Unsupported as error:
        assert expected_substring in str(error), f'{label}: expected {expected_substring!r}, got {error!r}'
    else:
        raise AssertionError(f'{label}: accepted unexpectedly')


def pack_f16_pair(u_f16, v_f16):
    return (u_f16 & 0xffff) | ((v_f16 & 0xffff) << 16)


def check_decoder():
    for key in DECODE_KEYS:
        text, raw = MIMG_LLVM_MC[key]
        length, _, asm, _ = D.decode_instruction(raw, 0)
        assert length == len(raw) and asm == text, (key, length, asm, text)

    # A single-VGPR packed 2D coordinate requires no NSA extension; fake NSA words surface invalid_nsa.
    for name, op in (('image_sample', IMG_SAMPLE), ('image_sample_lz', IMG_SAMPLE_LZ)):
        fake_nsa = encode_mimg(op, a16=1, nsa=1, extra_addrs=(1,))
        length, op_name, asm, _ = D.decode_instruction(fake_nsa, 0)
        assert length == 12 and op_name == name and 'invalid_nsa' in asm.split(), (name, asm)

    # Explicit-LOD 2D a16 and 3D / 2D-array a16 do not collapse to a single-VGPR v0 operand.
    _, _, asm_l, _ = D.decode_instruction(encode_mimg(IMG_SAMPLE_L, a16=1), 0)
    assert ' v0,' not in asm_l and 'invalid_nsa' not in asm_l, asm_l
    for dim in (DIM_3D, DIM_2D_ARRAY):
        _, _, asm_3, _ = D.decode_instruction(encode_mimg(IMG_SAMPLE, dim=dim, a16=1), 0)
        assert ' v0,' not in asm_3, asm_3
    print('decoder: 2D A16 sample/sample_lz use one VGPR and flag fake NSA; sample_l/3D/array keep multi-VGPR ranges')


def check_cache_revision():
    ps_code = pixel_words(MIMG_LLVM_MC['sample_a16'][1])
    root = root_blob()
    old_key = G.fnv(root, G.fnv(ps_code, G.fnv(VS_WORDS, G.fnv(b'x2w-startup-shader-5'))))
    key = G.pipeline_key(VS_WORDS, ps_code, root)
    assert key != old_key, 'cached pre-fix A16 sample translation would be reused'
    runtime = HERE.parent / 'src' / 'startup_stubs' / 'd3d12_draw.inl'
    revision = re.search(rb'constexpr char revision\[\] = "([^"]+)";', runtime.read_bytes())
    assert revision, 'desktop startup cache revision missing'
    runtime_key = G.fnv(root, G.fnv(ps_code, G.fnv(VS_WORDS, G.fnv(revision[1]))))
    assert key == runtime_key, f'offline ({key:#x}) and desktop ({runtime_key:#x}) cache revisions differ'
    print(f'cache revision: {revision[1].decode("ascii")} matches runtime and invalidates x2w-startup-shader-5')


def check_address_counts():
    """Pin address-register count checks for 2D (plain and A16), 3D and 2D-array sample and sample_lz."""
    dim_codes = {
        'SQ_RSRC_IMG_2D': DIM_2D,
        'SQ_RSRC_IMG_3D': DIM_3D,
        'SQ_RSRC_IMG_2D_ARRAY': DIM_2D_ARRAY,
    }
    cases = (
        ('sample', 'SQ_RSRC_IMG_2D', ('a16',), '[v0, v1]', 2),
        ('lz', 'SQ_RSRC_IMG_2D', ('a16',), '[v0, v1]', 2),
        ('sample', 'SQ_RSRC_IMG_2D', (), 'v0', 1),
        ('sample', 'SQ_RSRC_IMG_2D', (), '[v0, v1, v2]', 3),
        ('lz', 'SQ_RSRC_IMG_2D', (), 'v0', 1),
        ('lz', 'SQ_RSRC_IMG_2D', (), '[v0, v1, v2]', 3),
        ('sample', 'SQ_RSRC_IMG_3D', (), '[v0, v1]', 2),
        ('sample', 'SQ_RSRC_IMG_3D', (), '[v0, v1, v2, v3]', 4),
        ('lz', 'SQ_RSRC_IMG_2D_ARRAY', (), '[v0, v1]', 2),
        ('lz', 'SQ_RSRC_IMG_2D_ARRAY', (), '[v0, v1, v2, v3]', 4),
    )
    for how, dim, extra_mods, coords, count in cases:
        pipe = translate(encode_mimg(IMG_SAMPLE, dim=dim_codes[dim]))
        op_name = 'image_sample' if how == 'sample' else 'image_sample_lz'
        mods = [f'dim:{dim}', 'dmask:0xf', *extra_mods]
        expect_refused(
            lambda p=pipe, h=how, c=coords, m=mods: p.ps.image(['v[4:7]', c, 's[16:23]', 's[24:27]'], m, h),
            f'{op_name} with {count} address registers',
            f'{op_name} {dim} {coords}',
        )


def controls():
    check_decoder()
    check_cache_revision()

    # 2D A16 image_sample: implicit Sample with unpacked binary16 halves from v0; v1 sentinel is untouched.
    sample_pipe = translate(MIMG_LLVM_MC['sample_a16'][1])
    sample_ps = sample_pipe.hlsl_ps()
    assert 't = r0.Sample(q0, float2(f16tof32(v0), f16tof32(v0 >> 16)));' in sample_ps, sample_ps
    assert 'F(v1)' not in sample_ps, sample_ps
    assert 'srv 0 tex2d table 0 0' in sample_pipe.manifest(), sample_pipe.manifest()

    # 2D A16 image_sample_lz: explicit level 0 with unpacked binary16 halves from v0.
    lz_pipe = translate(MIMG_LLVM_MC['lz_a16'][1])
    lz_ps = lz_pipe.hlsl_ps()
    assert 't = r0.SampleLevel(q0, float2(f16tof32(v0), f16tof32(v0 >> 16)), 0);' in lz_ps, lz_ps

    # Non-v0 packed register (v2) + glc/slc/dlc cache-policy flags: uses v2 and matches cache-free output.
    v2_pipe = translate(MIMG_LLVM_MC['lz_v2_cache'][1], PRE_V2_ONLY_WORDS)
    v2_ps = v2_pipe.hlsl_ps()
    assert 't = r0.SampleLevel(q0, float2(f16tof32(v2), f16tof32(v2 >> 16)), 0);' in v2_ps, v2_ps
    v2_plain_ps = translate(encode_mimg(IMG_SAMPLE_LZ, vaddr=2, a16=1), PRE_V2_ONLY_WORDS).hlsl_ps()
    assert v2_ps == v2_plain_ps, 'cache-policy flags changed A16 sample_lz lowering'

    # Partial dmask (0x5 -> x and z into consecutive destination VGPRs v4, v5; v6/v7 not written).
    # Export only v4, v5 so unwritten v6, v7 are not read by the epilogue.
    dmask5_post = bytes.fromhex('703f8cbf' '0f1800f804050405' '000081bf')
    dmask5_ps = G.Pipeline(VS_WORDS, PRE_WORDS + MIMG_LLVM_MC['sample_dmask5_a16'][1] + dmask5_post,
                           root_blob()).hlsl_ps()
    assert 'v4 = U(t.x);' in dmask5_ps and 'v5 = U(t.z);' in dmask5_ps, dmask5_ps
    assert 'U(t.y)' not in dmask5_ps and 'U(t.w)' not in dmask5_ps, dmask5_ps

    # Full-width 2D NSA [v0, v3] and cache-policy flags still lower cleanly.
    nsa_ps = translate(MIMG_LLVM_MC['sample_nsa'][1]).hlsl_ps()
    lz_nsa_ps = translate(MIMG_LLVM_MC['lz_nsa'][1]).hlsl_ps()
    assert 't = r0.Sample(q0, float2(F(v0), F(v3)));' in nsa_ps, nsa_ps
    assert 't = r0.SampleLevel(q0, float2(F(v0), F(v3)), 0);' in lz_nsa_ps, lz_nsa_ps
    plain_ps = translate(MIMG_LLVM_MC['sample_plain'][1]).hlsl_ps()
    for key in ('sample_glc', 'sample_slc', 'sample_dlc'):
        assert translate(MIMG_LLVM_MC[key][1]).hlsl_ps() == plain_ps, key

    # 1D sample keeps its existing 1D guard: plain 1D translates, while 1D glc and 1D a16 stay refused.
    one_d_ps = translate(encode_mimg(IMG_SAMPLE, dim=DIM_1D)).hlsl_ps()
    assert 'r0.Sample(q0, F(v0))' in one_d_ps, one_d_ps
    expect_refused(lambda: translate(encode_mimg(IMG_SAMPLE, dim=DIM_1D, glc=1)),
                   '1D image', '1D glc sample')
    expect_refused(lambda: translate(encode_mimg(IMG_SAMPLE, dim=DIM_1D, a16=1)),
                   '1D image', '1D a16 sample')

    # Unmodeled modifiers, empty dmask, 3D/array A16, fake NSA and reserved bits are refused for both ops.
    for prefix, op, op_name in (('sample', IMG_SAMPLE, 'image_sample'),
                                ('lz', IMG_SAMPLE_LZ, 'image_sample_lz')):
        for mod in ('unorm', 'tfe', 'lwe', 'r128'):
            expect_refused(lambda b=MIMG_LLVM_MC[f'{prefix}_{mod}'][1]: translate(b),
                           f'{op_name} modifier {mod}', f'{op_name} {mod}')
        # d16 is not a layout change of the address: it returns the sampled channels as binary16 pairs (RNE), two per
        # register, so the sample translates (tests/startup_compute_cube_check.py reads the rounding back in compute and
        # check_draw below reads this pixel stage's plain 2D d16 sample back). d16 together with a16 is a second modifier
        # on the same instruction and keeps both halves of the semantics: packed address, packed data.
        d16_text = translate(MIMG_LLVM_MC[f'{prefix}_d16'][1]).hlsl_ps()
        both = translate(encode_mimg(op, a16=1, d16=1)).hlsl_ps()
        assert 'float2(f16tof32(v0), f16tof32(v0 >> 16))' in both and 'pack16_rne(t.x) | (pack16_rne(t.y) << 16)' in both, both
        assert 'pack16_rne(t.x) | (pack16_rne(t.y) << 16)' in d16_text, d16_text
        assert 'pack16_rne(t.z) | (pack16_rne(t.w) << 16)' in d16_text, d16_text
        expect_refused(lambda o=op: translate(encode_mimg(o, a16=1, nsa=1, extra_addrs=(1,))),
                       f'{op_name} modifier invalid_nsa', f'{op_name} invalid_nsa')
        for dim_label, dim in (('3D', DIM_3D), ('2D_ARRAY', DIM_2D_ARRAY)):
            expect_refused(lambda o=op, d=dim: translate(encode_mimg(o, dim=d, a16=1)),
                           f'{op_name} modifier a16', f'{op_name} {dim_label} a16')
        for bit_name, w0_res, w1_res in RESERVED_BITS:
            expect_refused(
                lambda o=op, r0=w0_res, r1=w1_res: translate(encode_mimg(o, w0_reserved=r0, w1_reserved=r1)),
                f'{op_name} modifier reserved', f'{op_name} {bit_name}')
            expect_refused(
                lambda o=op, r0=w0_res, r1=w1_res: translate(encode_mimg(o, a16=1, w0_reserved=r0, w1_reserved=r1)),
                f'{op_name} modifier reserved', f'{op_name} a16 {bit_name}')
        for key in (f'{prefix}_dmask0', f'{prefix}_dmask0_a16'):
            expect_refused(lambda b=MIMG_LLVM_MC[key][1]: translate(b),
                           f'{op_name} with an empty dmask', key)

    check_address_counts()
    print('controls: 2D A16 sample and sample_lz unpacked; modifiers, empty dmask and wrong address counts refused')


def check_binary(llvm_mc):
    assert assemble(llvm_mc, PRE + MIMG_LLVM_MC['sample_a16'][0] + '\n' + POST) == pixel_words(
        MIMG_LLVM_MC['sample_a16'][1])
    assert assemble(llvm_mc, PRE_V2_ONLY + MIMG_LLVM_MC['lz_v2_cache'][0] + '\n' + POST) == pixel_words(
        MIMG_LLVM_MC['lz_v2_cache'][1], PRE_V2_ONLY_WORDS)
    for key, (inst, expected) in sorted(MIMG_LLVM_MC.items()):
        built = assemble(llvm_mc, inst + '\n')
        if built != expected:
            raise AssertionError(f'{key}: assembled {built.hex()}, recorded {expected.hex()}')
    print(f'llvm-mc: {len(MIMG_LLVM_MC)} MIMG encodings and pixel fixtures match recorded words')


def check_decode(library):
    from pixel_aot import disassemble
    blob = b''.join(MIMG_LLVM_MC[k][1] for k in DECODE_KEYS)
    llvm_lines = disassemble(blob, library)
    ours = [D.decode_instruction(MIMG_LLVM_MC[k][1], 0)[2] for k in DECODE_KEYS]
    assert llvm_lines == ours, f'LLVM vs disasm_rdna2 mismatch: {list(zip(llvm_lines, ours))}'
    print(f'libLLVM: {len(DECODE_KEYS)} MIMG instructions match disasm_rdna2.py verbatim')


def run_draw_case(renderer, adapter, vs, ps, packed_word, expected_rgba, expect_fail=False, actual_rgba=None):
    run = subprocess.run(
        [renderer, adapter, '--startup-image-2d', str(vs), str(ps), f'0x{packed_word:08x}', expected_rgba],
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        print(run.stdout + run.stderr, end='', flush=True)
        sys.exit(77)
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
    if expect_fail:
        assert run.returncode == 1, f'mutant unexpectedly passed:\n{run.stdout}{run.stderr}'
        if actual_rgba is not None:
            assert f'expected={expected_rgba} actual={actual_rgba}' in run.stderr, run.stderr
    else:
        assert run.returncode == 0, f'draw failed (0x{packed_word:08x}):\n{run.stdout}{run.stderr}'
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    # IEEE binary16 bit patterns:
    #   0x3400 = +0.25 (col/row 0), 0x3a00 = +0.75 (col/row 1), 0xb400 = -0.25 (wraps under point-wrap to 0.75 -> 1).
    cases = (
        ('quadrant (0, 0)', pack_f16_pair(0x3400, 0x3400), texel(0, 0, 0)),
        ('quadrant (1, 0)', pack_f16_pair(0x3a00, 0x3400), texel(1, 0, 0)),
        ('quadrant (0, 1)', pack_f16_pair(0x3400, 0x3a00), texel(0, 1, 0)),
        ('quadrant (1, 1)', pack_f16_pair(0x3a00, 0x3a00), texel(1, 1, 0)),
        ('negative low-half u=-0.25 wraps to col 1', pack_f16_pair(0xb400, 0x3400), texel(1, 0, 0)),
        ('negative high-half v=-0.25 wraps to row 1', pack_f16_pair(0x3400, 0xb400), texel(0, 1, 0)),
    )
    sample_hlsl = translate(MIMG_LLVM_MC['sample_a16'][1]).hlsl_ps()
    lz_hlsl = translate(MIMG_LLVM_MC['lz_a16'][1]).hlsl_ps()
    v2_hlsl = translate(MIMG_LLVM_MC['lz_v2_cache'][1], PRE_V2_ONLY_WORDS).hlsl_ps()

    # Mutants on the translated HLSL that represent the pre-fix 2-VGPR bug and unpacking defects.
    target_expr = 'float2(f16tof32(v0), f16tof32(v0 >> 16))'
    assert target_expr in sample_hlsl
    mutants_v0 = (
        ('pre-fix 2-VGPR float2(F(v0), F(v1))', 'float2(F(v0), F(v1))',
         pack_f16_pair(0x3a00, 0x3400), texel(1, 0, 0), texel(0, 1, 0)),
        ('swapped halves', 'float2(f16tof32(v0 >> 16), f16tof32(v0))',
         pack_f16_pair(0x3a00, 0x3400), texel(1, 0, 0), texel(0, 1, 0)),
        ('dropped >> 16 shift', 'float2(f16tof32(v0), f16tof32(v0))',
         pack_f16_pair(0x3a00, 0x3400), texel(1, 0, 0), texel(1, 1, 0)),
        ('off-by-one >> 15 shift', 'float2(f16tof32(v0), f16tof32(v0 >> 15))',
         pack_f16_pair(0x3400, 0x3a00), texel(0, 1, 0), texel(0, 0, 0)),
        ('off-by-one >> 17 shift', 'float2(f16tof32(v0), f16tof32(v0 >> 17))',
         pack_f16_pair(0x3400, 0x3a00), texel(0, 1, 0), texel(0, 0, 0)),
    )

    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='image-sample-a16-', dir=out_dir) as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, f'vs_{kind}', vs_profile, directory)
            for fixture_name, source in (('sample_a16', sample_hlsl),
                                         ('lz_a16', lz_hlsl),
                                         ('lz_v2_cache', v2_hlsl)):
                ps = compile_hlsl(compiler, kind, source, f'{fixture_name}_{kind}', ps_profile, directory)
                for label, packed_word, expected_rgba in cases:
                    run = run_draw_case(renderer, adapter, vs, ps, packed_word, expected_rgba)
                    if label == 'quadrant (1, 0)':
                        print(run.stdout, end='', flush=True)
                print(f'{kind} {fixture_name}: all {len(cases)} packed binary16 coordinate cases matched', flush=True)

            for idx, (label, replacement, packed_word, expected_rgba, actual_rgba) in enumerate(mutants_v0):
                mutated = sample_hlsl.replace(target_expr, replacement, 1)
                assert mutated != sample_hlsl, label
                ps_mut = compile_hlsl(compiler, kind, mutated, f'mut_{idx}_{kind}', ps_profile, directory)
                run_draw_case(renderer, adapter, vs, ps_mut, packed_word, expected_rgba,
                              expect_fail=True, actual_rgba=actual_rgba)
                print(f'{kind} negative control ({label}): rejected as expected ({actual_rgba})', flush=True)

            # d16 sample (plain 2D, the case this check used to refuse): v4 holds the red and green halves, v5 the blue and
            # alpha halves; v6 and v7 stay 0. Every texel channel here is exact in binary16, so the words are the halves.
            # The plain d16 fixture has float addresses (v0 = the root constant's bits, v1 = 0.75: one fixed texel), so the
            # four quadrants are read through a16 + d16: packed binary16 address in v0, binary16 pairs in v[4:5].
            d16_hlsl = translate(encode_mimg(IMG_SAMPLE, a16=1, d16=1)).hlsl_ps()
            d16_ps = compile_hlsl(compiler, kind, d16_hlsl, f'sample_d16_{kind}', ps_profile, directory)

            def halves(x, y):
                f = (struct.unpack('<f', struct.pack('<I', int(w, 16)))[0] for w in texel(x, y, 0).split(','))
                h = [struct.unpack('<H', struct.pack('<e', c))[0] for c in f]
                return ','.join('%08x' % w for w in (h[0] | h[1] << 16, h[2] | h[3] << 16, 0, 0))

            for label, packed_word, x, y in (('quadrant (0, 0)', pack_f16_pair(0x3400, 0x3400), 0, 0),
                                             ('quadrant (1, 0)', pack_f16_pair(0x3a00, 0x3400), 1, 0),
                                             ('quadrant (0, 1)', pack_f16_pair(0x3400, 0x3a00), 0, 1),
                                             ('quadrant (1, 1)', pack_f16_pair(0x3a00, 0x3a00), 1, 1)):
                run_draw_case(renderer, adapter, vs, d16_ps, packed_word, halves(x, y))
            print(f'{kind} sample_d16: all 4 texels read back as binary16 pairs', flush=True)
            swapped = d16_hlsl.replace('pack16_rne(t.x) | (pack16_rne(t.y) << 16)', 'pack16_rne(t.y) | (pack16_rne(t.x) << 16)')
            assert swapped != d16_hlsl
            d16_mut = compile_hlsl(compiler, kind, swapped, f'mut_d16_{kind}', ps_profile, directory)
            run_draw_case(renderer, adapter, vs, d16_mut, pack_f16_pair(0x3a00, 0x3400), halves(1, 0),
                          expect_fail=True)
            print(f'{kind} negative control (d16 halves exchanged): rejected as expected', flush=True)

            # Register-selection negative control: reading v0 instead of v2 on lz_v2_cache must read v0=0x3f400000
            # (u = f16(0x0000) = 0.0 -> col 0, v = f16(0x3f40) = 1.8125 -> wraps to 0.8125 -> row 1).
            v2_expr = 'float2(f16tof32(v2), f16tof32(v2 >> 16))'
            assert v2_expr in v2_hlsl
            v2_mut = compile_hlsl(compiler, kind, v2_hlsl.replace(v2_expr, target_expr, 1),
                                  f'mut_v2_{kind}', ps_profile, directory)
            run_draw_case(renderer, adapter, vs, v2_mut, pack_f16_pair(0x3a00, 0x3400),
                          texel(1, 0, 0), expect_fail=True, actual_rgba=texel(0, 1, 0))
            print(f'{kind} negative control (hardcoded v0 instead of v2): rejected as expected', flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary' and len(args) == 1:
        check_binary(args[0])
    elif mode == 'decode' and len(args) == 1:
        check_decode(args[0])
    elif mode == 'draw' and len(args) in (3, 4):
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
