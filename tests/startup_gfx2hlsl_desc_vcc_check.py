"""Check descriptor words moved into vcc, with optional WARP/hardware pixel readback.

usage: startup_gfx2hlsl_desc_vcc_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]
"""
import struct
import sys
import tempfile
from pathlib import Path

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
    if alias == 2:
        # The descriptor is in s[39:42]. Copy its low words separately, then
        # use the unaligned s41:s42 source for the high words.
        words = [*smem(2, 39, 8, 0x20), 0xbf8c0000,
                 0xbee80327, 0xbee90328, mov64(106, 41)]
    elif alias:
        words += [mov64(104, 40), mov64(106, 42), *smem(10, 40, 40, 0x10), 0xbf8c0000]
    words += [*smem(10, 12, 104 if alias else 40, 0), 0xbf8c0000]
    # v_mov_b32_e32 v12..v15 from s12..s15, then export the four words.
    words += [0x7e000200 | reg << 17 | reg for reg in range(12, 16)]
    return struct.pack('<%dI' % len(words), *words) + POST_WORDS


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
    odd = G.Pipeline(vs_words([(0, 0, 0)] * 4), pixel_words(2), TABLE_ROOT)
    assert odd.ps.sym[106] == stage.sym[106] and odd.ps.sym[107] == stage.sym[107]

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
    print('ok')


def check_draw(dxc, renderer, adapter, fxc=None):
    main()
    compilers = ([('fxc', fxc, '5_1')] if fxc else []) + [('dxc', dxc, '6_0')]
    vs_code = vs_words([(0, 0, 0)] * 4)
    payload = [0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for alias in (0, 1, 2):
            pipe = G.Pipeline(vs_code, pixel_words(alias), TABLE_ROOT)
            source = pipe.hlsl_ps()
            assert set(pipe.resources) == {(('table', 0), 0x20)}, pipe.resources
            assert source.count('ByteAddressBuffer r0 : register(t0);') == 1
            assert source.count('r0.Load(') == (8 if alias == 1 else 4)
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
                if alias == 2:
                    # Perturb the descriptor-backed read address. The same CPU reference must reject it.
                    mutant = source.replace('return words[address >> 2];', 'return words[(address >> 2) + 1u];')
                    ps = compile_hlsl(compiler, kind, mutant, 'ps_odd_offset_' + kind, 'ps_' + profile, directory)
                    result = draw(renderer, adapter, vs, ps, words_text(payload))
                    assert result.returncode == 1 and 'startup_flat_param mismatch expected=' in result.stderr, (result.stdout, result.stderr)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
