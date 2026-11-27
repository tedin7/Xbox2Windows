"""Check descriptor words moved into vcc, with optional WARP/hardware pixel readback.

usage: startup_gfx2hlsl_desc_vcc_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from startup_gfx2hlsl_sload_s104_check import (
    G, Pipe, TABLE_ROOT, compile_hlsl, draw, translate, vs_words, words_text,
)
from startup_gfx2hlsl_bit_alu_check import POST_WORDS


LOAD = 's_load_dwordx4 s[40:43], s[4:5], 0x20'
LOW = 's_mov_b64 s[104:105], s[40:41]'
COPY = 's_mov_b64 vcc, s[42:43]'
CLOBBER = 's_buffer_load_dwordx4 s[40:43], s[40:43], 0x10'
FETCH = 's_buffer_load_dwordx4 s[12:15], s[104:107], 0'
DIRECT = 's_buffer_load_dwordx4 s[12:15], s[40:43], 0'
END = 's_endpgm'
MASK_READ = 'vcc read after a descriptor load into s106/s107'


def refuse(message, *instructions):
    try:
        translate(*instructions, END)
    except G.Unsupported as error:
        assert message in str(error), (instructions, error)
    else:
        raise AssertionError(('accepted', instructions))


def smem(op, dst, base, offset):
    return (0xf4000000 | op << 18 | dst << 6 | base // 2, 0xfa000000 | offset)


def mov64(dst, src):
    return 0xbe800400 | dst << 16 | src


def pixel_words(alias):
    # s_load x4 into s[40:43] from the root table at s[8:9], then wait for the descriptor.
    words = [*smem(2, 40, 8, 0x20), 0xbf8c0000]
    if alias:
        words += [mov64(104, 40), mov64(106, 42), *smem(10, 40, 40, 0x10), 0xbf8c0000]
    words += [*smem(10, 12, 104 if alias else 40, 0), 0xbf8c0000]
    # v_mov_b32_e32 v12..v15 from s12..s15, then export the four words.
    words += [0x7e000200 | reg << 17 | reg for reg in range(12, 16)]
    return struct.pack('<%dI' % len(words), *words) + POST_WORDS


def odd_image(*, swapped=False, constant_high=False):
    from startup_image_sample_a16_check import DIM_2D, IMG_SAMPLE_LZ, encode_mimg

    # Preserve the six low words from the load into s[100:107]. Copy the last
    # two from a separate load through an odd pair and back into VCC.
    words = [*smem(3, 100, 8, 0x20), *smem(3, 56, 8, 0x20), 0xbf8c0000]
    pair = (63, 62) if swapped else (62, 63)
    high = 128 if constant_high else pair[1]
    words += [0xbe9d0300 | pair[0], 0xbe9e0300 | high, mov64(106, 29)]
    words += [0xbea40380, 0xbea50380, 0xbea60380, 0xbea70380]  # s36:s39 sampler, all zero
    words += [0x7e000200 | 18 << 17 | 255, 0x3f400000, 0x7e000200 | 19 << 17 | 255, 0x3e800000]
    img = encode_mimg(IMG_SAMPLE_LZ, dim=DIM_2D, vaddr=18, vdata=12, srsrc=100, ssamp=36)
    return struct.pack('<%dI' % len(words), *words) + img + struct.pack('<I', 0xbf8c3f70) + POST_WORDS


def sample_pipeline(**options):
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), odd_image(**options), TABLE_ROOT)


def check_odd_sample():
    code = odd_image()
    assert any(asm == 's_mov_b64 vcc, s29' for _, _, _, asm in G.decode(code))
    pipe = sample_pipeline()
    assert list(pipe.resources) == [(('table', 0), 0x20)]
    mixed = sample_pipeline(constant_high=True)
    assert mixed.ps.sym[107] == ('const', 0)
    assert mixed.resources == pipe.resources
    try:
        sample_pipeline(swapped=True)
    except G.Unsupported as error:
        assert 'needs matching descriptor words in vcc' in str(error), error
    else:
        raise AssertionError('swapped VCC descriptor words translated')


def draw_odd_sample(directory, dxc, renderer, adapter):
    from startup_image_sample_c_border_check import f32_hex, manifest_sampler

    pipe = sample_pipeline()
    vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), 'vs_odd_image', 'vs_6_0', directory)
    # Point sampling the 2x2 texture at (0.75, 0.25) selects texel (1, 0).
    expected = f32_hex((1 + 1) / 8, 1, 0, 0)
    for label, sample in (('odd_image', pipe), ('mixed_image', sample_pipeline(constant_high=True))):
        ps = compile_hlsl(dxc, 'dxc', sample.hlsl_ps(), 'ps_' + label, 'ps_6_0', directory)
        command = [renderer, adapter, '--startup-image-2d', vs, ps, '0', expected, manifest_sampler(sample)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, (result.stdout, result.stderr)

    # The swapped descriptor is invalid. Verify that treating it as an unbound
    # (zero) resource in the readback harness cannot satisfy the same reference.
    descriptor = G.Stage.descriptor

    def invalid_as_null(stage, operand, **kwargs):
        try:
            return descriptor(stage, operand, **kwargs)
        except G.Unsupported as error:
            if 'needs matching descriptor words in vcc' not in str(error):
                raise
            return None

    with patch.object(G.Stage, 'descriptor', invalid_as_null):
        mutant = sample_pipeline(swapped=True).hlsl_ps()
    command[4] = compile_hlsl(dxc, 'dxc', mutant, 'ps_swapped_image_words', 'ps_6_0', directory)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and 'actual=00000000,00000000,00000000,00000000' in result.stderr, result.stderr


@patch.object(Pipe, "ud", lambda self, key: key[2] // 4, create=True)
@patch.object(Pipe, "modified", set(), create=True)
def check_descriptor_flags():
    compare = 's_cmp_eq_u32 s42, 0'
    select = 's_cselect_b32 s22, 31, 7'
    stage = translate(LOAD, compare, select, END)
    assert any('s22 = scc ?' in line for _, lines, _ in stage.blocks for line in lines)
    refuse('scc from descriptor data', LOAD, compare, 's_cselect_b32 vcc_lo, 1, 0')
    refuse('scc from descriptor data', LOAD, compare, 's_cselect_b64 exec, -1, 0')
    refuse('without a single wave-uniform producer', LOAD, compare, 's_cbranch_scc0 label_0010',
           's_cmp_eq_u32 s43, 0', select)
    refuse('cannot materialize descriptor word 2 from an advanced descriptor',
           's_add_u32 s20, s4, 64', 's_load_dwordx4 s[40:43], s[20:21], 0', compare, select)
    refuse('numeric use of s42 holding desc', LOAD, 's_xor_b32 s42, s42, 1', compare, select)
    for op in ('s_pack_ll_b32_b16', 's_mul_hi_u32', 's_mul_hi_i32'):
        refuse(op + ' on descriptor data', LOAD, op + ' s22, s42, 1')
    # Simulate the symbols a heap-indexed load supplies, then exercise each newly lifted producer.
    for instruction in ('s_cmp_eq_u32 s42, 0', 's_lshl1_add_u32 s22, s42, 1'):
        with patch.dict(translate.__globals__['TABLE'], {42: ('desc', ('resheap', 0, 0), 0, 2)}):
            refuse('cannot materialize descriptor word 2', instruction, select)


def main():
    check_descriptor_flags()
    stage = translate(LOAD, LOW, COPY, CLOBBER, FETCH, END)
    # Manual mutant: dropping the symbol copy must fail the positive alias or this symbol check.
    assert stage.sym[106] == ('desc', ('table', 0), 0x20, 2)
    assert stage.sym[107] == ('desc', ('table', 0), 0x20, 3)
    assert all(stage.sym[i] == ('fixed_buffer_word',) for i in range(40, 44))
    assert stage.blocks[0][1][-4:] == translate(LOAD, DIRECT, END).blocks[0][1][-4:]
    assert not any('vcc =' in line for _, lines, _ in stage.blocks for line in lines)
    decoded = [asm for _, _, _, asm in G.decode(pixel_words(True))]
    assert COPY in decoded and LOW in decoded, decoded

    # Manual mutant: dropping the vcc_lost marking must fail the straight-line mask refusals.
    for read in (
        'v_cndmask_b32_e32 v0, 0, 1',
        'v_add_co_ci_u32_e32 v0, vcc_lo, 0, 1, vcc_lo',
        'v_sub_co_ci_u32_e64 v0, vcc_lo, 0, 1, vcc_lo',
        'v_subrev_co_ci_u32_e64 v0, vcc_lo, 0, 1, vcc_lo',
        'v_div_fmas_f32_e64 v0, v1, v2, v3',
        'v_cmp_eq_u32_e64 s20, vcc_lo, v1',
        'v_cmp_eq_u32_e32 vcc, v1',
        'v_cmp_eq_u32_sdwa s20, vcc_hi, v1 src0_sel:DWORD src1_sel:DWORD',
        'v_add_u32_e64 v0, vcc, v1',
        's_and_saveexec_b64 s[20:21], vcc',
        's_mov_b32 s20, vcc_lo',
    ):
        refuse(MASK_READ, LOAD, COPY, read)
    for branch in ('s_cbranch_vccz', 's_cbranch_vccnz'):
        refuse(MASK_READ, LOAD, COPY, branch + ' label_000C')
    compare = 'v_cmp_eq_u32_e32 0, v1'
    select = 'v_cndmask_b32_e32 v0, 0, 1'
    translate(LOAD, COPY, compare, select, END)
    refuse(MASK_READ, LOAD, COPY, 's_cbranch_scc0 label_0010', compare, select)
    refuse(MASK_READ, LOAD, COPY, 's_cbranch_scc0 label_000C', 'v_div_fmas_f32_e64 v0, v1, v2, v3')
    refuse('overlaps a redefined vcc', LOAD, LOW, COPY, compare, FETCH)
    refuse('without a matching load on every path',
           LOAD, LOW, 's_cbranch_scc0 label_0010', COPY, FETCH)

    for operation in (
        's_mov_b64 exec, s[42:43]',
        's_mov_b32 vcc_lo, s42',
        's_and_b64 vcc, s[42:43], -1',
    ):
        refuse('mask from descriptor data', LOAD, operation)
    stage = translate(LOAD, 's_mov_b32 s43, 0', COPY, END)
    assert stage.sym[107] == ('const', 0)
    stage = translate(LOAD, 's_mov_b32 s42, 0', COPY, END)
    assert stage.sym[106] == ('const', 0)
    refuse('needs a pair of tracked descriptor words or constants', LOAD, 's_mov_b64 vcc, s[43:44]')
    zeros = ('s_mov_b64 s[104:105], 0', 's_mov_b64 s[38:39], 0', 's_mov_b64 vcc, s[38:39]')
    stage = translate(*zeros, FETCH, END)
    assert stage.sym[106] == stage.sym[107] == ('const', 0)
    assert any('vcc = ' in line for _, lines, _ in stage.blocks for line in lines)
    refuse('without a matching load on every path', *zeros[:2], 's_cbranch_scc0 label_0010', zeros[2], FETCH)
    check_odd_sample()

    # Manual mutant: skipping the consistency check must accept these malformed descriptors.
    message = 'needs four consecutive words of the same descriptor'
    refuse(message, LOAD, 's_load_dwordx4 s[48:51], s[4:5], 0x30',
           's_mov_b64 s[104:105], s[48:49]', COPY, FETCH)
    refuse(message, LOAD, LOW, 's_mov_b32 s105, s40', COPY, FETCH)
    overlapping_load = 's_load_dwordx4 s[104:107], s[4:5], 0x20'
    translate(overlapping_load, FETCH, END)
    refuse(message, overlapping_load, 's_mov_b32 s105, 0', FETCH)
    print('ok')


def check_draw(dxc, renderer, adapter, fxc=None):
    main()
    compilers = ([('fxc', fxc, '5_1')] if fxc else []) + [('dxc', dxc, '6_0')]
    vs_code = vs_words([(0, 0, 0)] * 4)
    payload = [0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        draw_odd_sample(directory, dxc, renderer, adapter)
        for alias in (False, True):
            pipe = G.Pipeline(vs_code, pixel_words(alias), TABLE_ROOT)
            source = pipe.hlsl_ps()
            assert set(pipe.resources) == {(('table', 0), 0x20)}, pipe.resources
            assert source.count('ByteAddressBuffer r0 : register(t0);') == 1
            assert source.count('r0.Load(') == (8 if alias else 4)
            # The flat-param runner only binds b0. Supply the test SRV's bytes in HLSL while
            # keeping the translated load addresses, scalar assignments and export in the draw.
            fixture = ('uint fixture_load(uint address) {\n'
                       '    const uint words[8] = {%s, 0u, 0u, 0u, 0u};\n'
                       '    return words[address >> 2];\n'
                       '}' % ', '.join('0x%08xu' % word for word in payload))
            source = source.replace('ByteAddressBuffer r0 : register(t0);', fixture)
            source = source.replace('r0.Load(', 'fixture_load(')
            for kind, compiler, profile in compilers:
                vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_%s_%d' % (kind, alias),
                                  'vs_' + profile, directory)
                ps = compile_hlsl(compiler, kind, source, 'ps_%s_%d' % (kind, alias),
                                  'ps_' + profile, directory)
                result = draw(renderer, adapter, vs, ps, words_text(payload))
                assert result.returncode == 0, (kind, alias, result.stdout, result.stderr)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
