"""Slot 67 vector readlane controls and WARP readback of a scalar lane walk."""
from pathlib import Path
import re
import sys
import tempfile

from startup_compute_global_load_check import (
    G, MASK, SENTINEL, TGID_X, WORDS, assemble, disassemble_words, encode_sop1, encode_sopp, run_case,
)
from startup_compute_translate_check import state_a, write_request


ROOT = Path(__file__).resolve().parents[1]
RSRC2 = (7 << 1) | TGID_X
SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s20, s[12:15], 0x0
s_waitcnt lgkmcnt(0)
v_and_b32_e32 v4, 31, v0
v_mul_lo_u32 v4, v4, 36
buffer_load_dwordx4 v[8:11], v4, s[12:15], 0 offen offset:4
s_waitcnt vmcnt(0)
s_mov_b32 s8, 0
s_mov_b32 s24, 0
s_cmp_eq_u32 s20, 0
s_cbranch_scc1 11
s_xor_b32 s27, s8, 7
v_readlane_b32 s28, v8, s8
v_readlane_b32 s29, v9, s27
s_add_u32 s24, s24, s28
s_xor_b32 s24, s24, s29
v_add_nc_u32_e32 v8, v0, v8
s_add_u32 s8, s8, 1
s_cmp_lt_u32 s8, s20
s_cbranch_scc1 -11
s_lshl_b32 s21, s7, 6
v_add_nc_u32_e32 v4, s21, v0
v_lshlrev_b32_e32 v4, 2, v4
v_mov_b32_e32 v5, s24
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
C67 = bytes.fromhex(
    '06038abe 80038bbe 05030cf4 000000fa 7fc08cbf'
    '060520f4 000000fa 7fc08cbf 9f000836 040069d5'
    '04490100 041038e0 04080380 703f8cbf 800388be'
    '800398be 148006bf 0b0085bf 08871b89 1c0060d7'
    '08110000 1d0060d7 09370000 181c1880 181d1889'
    '0011104a 08810880 08140abf f5ff85bf 0786158f'
    '1500084a 82080834 18020a7e 001070e0 04050480'
    '000081bf'
)
# A full EXEC_LO write proves wave32 for the multi-wave slot 67 request.
WAVE32_C67 = encode_sop1(3, 126, 193) + C67  # s_mov_b32 exec_lo, -1
REFUSAL = 'v_readlane_b32 requires a group-uniform lane select and control flow'


def pipeline(directory, code=C67, threads=32, wave=None):
    group = (threads, 1, 1)
    key = write_request(directory, code, state_a(group, RSRC2))
    if wave is None:
        return G.load_compute_request(str(directory), key)
    request = directory / ('req_%016x' % key)
    return G.ComputePipeline(request.with_suffix('.c67').read_bytes(),
                             request.with_suffix('.root').read_bytes(), RSRC2, group, wave_size=wave)


def controls(directory):
    # Copy a wave-derived word into the loop bound after the comparison.
    # Only the back edge can carry it to the next comparison.
    assert disassemble_words(C67).splitlines()[22] == 's_cbranch_scc1 label_0x48'
    back_edge = (
        C67[:0x44] + encode_sopp(5, 12) + C67[0x48:0x70]
        + encode_sop1(3, 20, 28) + encode_sopp(5, -12) + C67[0x74:]
    )
    bad_select = C67[:0x48] + encode_sop1(3, 27, 126) + C67[0x4c:]
    for code, threads, wave, expected in (
        (back_edge, 32, None, REFUSAL),
        (bad_select, 32, None, REFUSAL),
        (C67, 64, None, 'v_readlane_b32 requires a known guest wave size for groups larger than 32 threads'),
    ):
        try:
            pipeline(directory, code, threads, wave)
        except G.Unsupported as error:
            assert str(error) == expected, (expected, error)
        else:
            raise AssertionError(expected)
    for code, threads, wave in ((C67, 24, None), (C67, 32, None), (WAVE32_C67, 64, None), (C67, 64, 64)):
        pipeline(directory, code, threads, wave).hlsl_cs()
    print('vector readlane slot 67 controls passed')


def reference(source, threads, wave, groups):
    output = [SENTINEL] * WORDS
    for base in range(0, threads, wave):
        total = 0
        for step in range(source[0]):
            a = base + (step % wave)
            b = base + ((step ^ 7) % wave)
            # v8 starts with the first word in each lane's record and adds
            # that lane's local thread ID at the end of each loop iteration.
            total = ((total + source[1 + (a % 32) * 9] + step * a) & MASK) ^ source[2 + (b % 32) * 9]
        for group in range(groups):
            start = group * 64 + base
            active = min(wave, threads - base)
            output[start:start + active] = [total] * active
    return output


def dispatch(directory, compiler, renderer):
    cases = ((24, None, 16), (32, None, 0), (32, None, 32), (64, 32, 39), (64, 64, 64))
    for threads, wave, count in cases:
        code = WAVE32_C67 if wave == 32 else C67
        hlsl = pipeline(directory, code, threads, wave if wave == 64 else None).hlsl_cs()
        source = [count] + [(0x13579bdf + i * 0x10203041) & MASK for i in range(1, WORDS)]
        want = reference(source, threads, wave or 32, 2)
        def run(name, text):
            return run_case(directory, compiler, renderer, 'warp', name, text, (2, 1, 1), want, 0, 0, source)

        name = 'walk_%d_%s_%d' % (threads, wave, count)
        result = run(name, hlsl)
        if result.returncode == 77:
            print(result.stdout + result.stderr)
            return 77
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, result
        assert 'debug_errors=0 debug_warnings=0' in result.stdout or 'debug_layer=unavailable' in result.stdout, result
        print(result.stdout, end='')

        if (threads, wave) == (64, 32):
            mutants = (
                ('wrong_lane', hlsl.replace('(s27 & 31u)', '((s27 + 1u) & 31u)')),
                ('wrong_wave', hlsl.replace('& ~31u', '& 0u')),
            )
            for name, mutant in mutants:
                assert mutant != hlsl, name
                result = run(name, mutant)
                assert result.returncode == 1 and re.search(r'word=\d+ expected=[0-9a-f]+ actual=[0-9a-f]+',
                                                            result.stderr), result
                print(f'{name} rejected by GPU readback')
    return 0


def main(args):
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        if not args:
            controls(directory)
            return 0
        if len(args) == 2 and args[0] == 'binary':
            assert assemble(args[1], SOURCE) == C67
            controls(directory)
            return 0
        if len(args) == 3 and args[0] == 'dispatch':
            return dispatch(directory, *args[1:])
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
