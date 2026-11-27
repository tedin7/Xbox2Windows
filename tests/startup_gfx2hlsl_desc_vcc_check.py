"""Check descriptor words and wave32 masks moved into vcc, with optional GPU readback.

usage: startup_gfx2hlsl_desc_vcc_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]
"""
import struct
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import gpu_case_runner

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


# Two triangles share the clip-space edge x+y=0.25. Before making the positions,
# classify each vertex's 1.0 (first triangle) or 0.0 (second triangle) into s30.
# s31 still holds word 3 from s_load_dwordx4 s[28:31].
MASK_VS = bytes.fromhex(
    'c103febe 040708f4140000fa 7fc08cbf 8200827d 060001d5f200a901'
    '1e0088d406ff010000010000 1e04eabe 0a0001d580e4a901 0b0001d5f200a901'
    '8002187e f2021a7e 8100847d 010001d580e4a901 8200847d 011b0202'
    '8400847d 011b0202 010041d501edfd030000e0bf 8000847d 020001d580e4a901'
    '8100847d 021b0402 8300847d 021b0402 020041d502edd503'
    'cf0800f801020c0d 0f0200f80a0b0c0d 000081bf'
)
MASK_MOVE = 'vcc = (s30) & 1;'
MASK_REFUSAL = 'mask from descriptor data: vcc needs a pair of tracked descriptor words or constants'


def xor64(dst, a, b):
    return struct.pack('<I', 0x89800000 | dst << 16 | b << 8 | a)


SWAP = xor64(106, 106, 30) + xor64(30, 106, 30) + xor64(106, 106, 30)


def mask_controls():
    pipe = G.Pipeline(MASK_VS, None, TABLE_ROOT)
    source = pipe.hlsl_vs()
    assert hashlib.sha256((source + '\0' + pipe.manifest()).encode()).hexdigest() == (
        '4e4ea0bcf91f6a586e8c46e89798f5721a2febc7cf5253920e669ded3e907897')
    assert source.count(MASK_MOVE) == 1 and 'class32(F(v6)) & 0x00000100u' in source
    # Read vcc_hi as the select mask, or remove the 32-bit EXEC proof.
    high_select = bytes.fromhex('0a0001d580e4ad01')
    high_read = MASK_VS[:44] + high_select + MASK_VS[52:]
    full_write = struct.pack('<I', mov64(106, 128))  # s_mov_b64 vcc, 0
    bypass = struct.pack('<I', 0xbf820001)  # s_branch to the high select
    branched = high_read[:44] + bypass + full_write + high_read[44:]
    for wave in (None, 64):
        pipe.wave_size = wave
        lifted = G.Stage(pipe, 'vs', MASK_VS[4:])
        assert any(MASK_MOVE in line for _, lines, _ in lifted.blocks for line in lines)
    # A later descriptor-word write invalidates the compare's high-word tag.
    write_high = struct.pack('<I', 0xbe9f031d)  # s_mov_b32 s31, s29
    odd_move = struct.pack('<I', mov64(106, 31))
    for code in (MASK_VS[4:40] + write_high + MASK_VS[40:],
                 MASK_VS[4:40] + odd_move + MASK_VS[44:]):
        pipe.wave_size = None
        try:
            G.Stage(pipe, 'vs', code)
        except G.Unsupported as error:
            assert str(error) == MASK_REFUSAL, error
        else:
            raise AssertionError(code.hex())
    swap_code = MASK_VS[4:40] + SWAP * 2 + MASK_VS[40:]
    assert len(G.Stage(pipe, 'vs', swap_code).blocks)
    for a in (0, 1):
        for b in (0, 1):
            x, y = a, b
            for _ in range(2):
                x ^= y
                y ^= x
                x ^= y
            assert (x, y) == (a, b)
    # Model lane bits across both halves of a wave64 and two wave32s.
    bits = [int((lane % 5) < 2) for lane in range(64)]
    mask64 = sum(bit << lane for lane, bit in enumerate(bits))
    for lane in range(64):
        start = (lane // 32) * 32
        low32 = sum(bits[start + i] << i for i in range(32))
        assert (mask64 >> lane) & 1 == (low32 >> (lane % 32)) & 1 == bits[lane]
    numeric_high = struct.pack('<I', 0xbe8a036b)  # s_mov_b32 s10, vcc_hi
    numeric_read = MASK_VS[4:44] + numeric_high + MASK_VS[44:]
    for wave, code in ((None, high_read), (None, branched), (None, numeric_read)):
        pipe.wave_size = wave
        try:
            G.Stage(pipe, 'vs', code)
        except G.Unsupported as error:
            assert str(error) == MASK_REFUSAL, error
        else:
            raise AssertionError((wave, code.hex()))
    # Check liveness alone because a scalar value written to vcc_hi is itself
    # unsupported as a per-lane condition, even after a full overwrite.
    stage = G.Stage.__new__(G.Stage)
    stage.ins = G.decode(high_read[:44] + full_write + high_read[44:])
    stage.spill_cfg = stage.joined_registers(with_cfg=True)[3]
    stage.descriptor_instruction_index = 6
    assert stage.dead_vcc_high()


def check_mask_draw(dxc, renderer, adapter):
    # hlsl_vs() elides parameters if the pipeline has no translated pixel stage.
    # Request a translated PS that copies param0 to the target.
    ps_code = bytes.fromhex(
        '0203fcbe 020052c8 020156c8 02025ac8 02035ec8 0f1800f814151617 000081bf')
    pipe = G.Pipeline(MASK_VS[4:], ps_code, TABLE_ROOT)
    source = pipe.hlsl_vs()
    assert source.count(MASK_MOVE) == 1
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', source, 'vs', 'vs_6_0', directory)
        mutant = compile_hlsl(dxc, 'dxc', source.replace(MASK_MOVE, 'vcc = (~s30) & 1;'),
                              'inverse', 'vs_6_0', directory)
        swapped_pipe = G.Pipeline(MASK_VS[4:40] + SWAP * 2 + MASK_VS[40:], ps_code, TABLE_ROOT)
        swapped = compile_hlsl(dxc, 'dxc', swapped_pipe.hlsl_vs(), 'vs_swapped', 'vs_6_0', directory)
        ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), 'ps', 'ps_6_0', directory)
        expected = directory / 'expected.bin'
        # At the pixel centers, x_ndc+y_ndc = (x-y)/2. No center lies on the edge.
        def reference(wave):
            # Compare writes a lane bit in the appropriate word; in wave32
            # the untouched descriptor high word is never read by a lane.
            pixels = []
            for y in range(4):
                for x in range(4):
                    lane = y * 4 + x
                    bit = int(x > y)
                    words = [0, 0xdeadbeef] if wave == 32 else [0, 0]
                    words[lane // 32] |= bit << (lane % 32)
                    selected = (words[lane // 32] >> (lane % 32)) & 1
                    pixels.append((float(selected), float(1 - selected), 0.0, 1.0))
            return pixels
        pixels = reference(32)
        assert pixels == reference(64)
        assert pixels != [(1 - r, 1 - g, b, a) for r, g, b, a in pixels]
        expected.write_bytes(b''.join(struct.pack('<4f', *pixel) for pixel in pixels))
        for shader, status in ((vs, 0), (swapped, 0), (mutant, 1)):
            result = gpu_case_runner.run(renderer, adapter, (
                '--raster-reference', shader, ps, 4, 4, expected, 'none', 'cw'))
            print(result.stdout, end='')
            if result.returncode == 77:
                raise SystemExit(77)
            assert result.returncode == status, (result.stdout, result.stderr)
            if status:
                assert '16 of 16 pixels differ' in result.stderr, result.stderr


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
    mask_controls()
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
    if adapter == 'warp':
        check_mask_draw(dxc, renderer, adapter)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
