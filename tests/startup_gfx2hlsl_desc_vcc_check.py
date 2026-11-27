"""Check descriptor words and wave32 masks moved into vcc, with optional GPU readback.

usage: startup_gfx2hlsl_desc_vcc_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]
"""
import struct
import sys
import tempfile
from pathlib import Path

import gpu_case_runner

from startup_gfx2hlsl_sload_s104_check import (
    G, TABLE_ROOT, compile_hlsl, draw, translate, vs_words, words_text,
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
MASK_REFUSAL = 'mask from descriptor data: vcc needs an even pair of descriptor words'


def mask_controls():
    pipe = G.Pipeline(MASK_VS, None, TABLE_ROOT)
    source = pipe.hlsl_vs()
    assert source.count(MASK_MOVE) == 1 and 'class32(F(v6)) & 0x00000100u' in source
    # Read vcc_hi as the select mask, or remove the 32-bit EXEC proof.
    high_select = bytes.fromhex('0a0001d580e4ad01')
    high_read = MASK_VS[:44] + high_select + MASK_VS[52:]
    full_write = struct.pack('<I', mov64(106, 128))  # s_mov_b64 vcc, 0
    bypass = struct.pack('<I', 0xbf820001)  # s_branch to the high select
    branched = high_read[:44] + bypass + full_write + high_read[44:]
    for wave, code in ((64, MASK_VS), (None, MASK_VS[4:]), (None, high_read), (None, branched)):
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
        '0203fcbe 025002c8 025106c8 02520ac8 02530ec8 0f1800f814151617 000081bf')
    pipe = G.Pipeline(MASK_VS, ps_code, TABLE_ROOT)
    source = pipe.hlsl_vs()
    assert source.count(MASK_MOVE) == 1
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', source, 'vs', 'vs_6_0', directory)
        mutant = compile_hlsl(dxc, 'dxc', source.replace(MASK_MOVE, 'vcc = (~s30) & 1;'),
                              'inverse', 'vs_6_0', directory)
        ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), 'ps', 'ps_6_0', directory)
        expected = directory / 'expected.bin'
        # At the pixel centers, x_ndc+y_ndc = (x-y)/2. No center lies on the edge.
        expected.write_bytes(b''.join(struct.pack('<4f', float(x > y), float(x <= y), 0.0, 1.0)
                                      for y in range(4) for x in range(4)))
        for shader, status in ((vs, 0), (mutant, 1)):
            result = gpu_case_runner.run(renderer, adapter, (
                '--raster-reference', shader, ps, 4, 4, expected, 'none', 'cw'))
            print(result.stdout, end='')
            if result.returncode == 77:
                raise SystemExit(77)
            assert result.returncode == status, (result.stdout, result.stderr)
            if status:
                assert '16 of 16 pixels differ' in result.stderr, result.stderr


def main():
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
        's_mov_b64 vcc, s[41:42]',
    ):
        refuse('mask from descriptor data', LOAD, operation)
    refuse('even pair of descriptor words', LOAD, 's_mov_b32 s43, 0', COPY)
    refuse('even pair of descriptor words', LOAD, 's_mov_b32 s42, 0', COPY)

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
