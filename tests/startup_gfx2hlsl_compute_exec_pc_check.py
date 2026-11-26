"""Synthetic compute EXEC and embedded PC data checks, with D3D12 WARP readback."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_compute_check as C

G = C.G
HERE = Path(__file__).resolve().parent
CONSTANT = 0x51A839D2


def sop1(op, dst, src):
    return 0xBE800000 | dst << 16 | op << 8 | src


def mov(dst, src):
    return sop1(3, dst, src)


def compare_tid_lt(limit):
    return 0x7D880080 | limit  # v_cmp_gt_u32 vcc_lo, limit, v0


def store(vdata, offset=0):
    return (0xE0701000 | offset, 0x80040004 | vdata << 8)


def words(*code):
    return struct.pack('<%dI' % len(code), *code)


PROLOGUE = (
    mov(10, 6), mov(11, 128), 0xF4080405, 0xFA000010,  # table UAV descriptor at byte 16
    0xBF8C007F,  # wait for the descriptor
    0x34080083,  # v_lshlrev_b32 v4, 3, v0 (two output words per thread)
)
END = 0xBF810000
WAIT = 0xBF8C007F
SAVE = sop1(0x3C, 26, 106)
RESTORE = mov(126, 26)
GET = sop1(0x1F, 28, 0)
COUNT = sop1(0x0F, 27, 126)
LITERAL = 0x7E0A02FF  # v_mov_b32 v5, literal
NESTED = words(
    *PROLOGUE, mov(25, 126),
    0xBF009104, 0xBF850003,  # if root word s4 equals 17, enter at the inner compare
    compare_tid_lt(5), mov(126, 106),
    0xBF88000A,  # outer execz skips to the outer restore
    compare_tid_lt(2), SAVE, 0xBF880004,
    LITERAL, 0x31, *store(5),
    RESTORE, *store(0, 4), mov(126, 25), END,
)
BALLOT = words(
    *PROLOGUE, mov(25, 126), compare_tid_lt(37), mov(126, 106),
    COUNT, 0x851C8987,  # s_cselect_b32 s28, 7, 9
    mov(126, 25), 0x7E0A021B, 0x7E0C021C, *store(5), *store(6, 4), END,
)
BALLOT64 = BALLOT.replace(words(mov(25, 126)), words(sop1(4, 24, 126))).replace(
    words(mov(126, 106)), words(sop1(4, 126, 106))).replace(
    words(mov(126, 25)), words(sop1(4, 126, 24)))
PC = words(
    *PROLOGUE, sop1(0x1F, 30, 0), GET,
    0x801C9C1C, 0x821D801D,  # add 28 bytes with carry to the GETPC successor
    0xF40006CE, 0xFA000008,  # s_load_dword s27, s[28:29], 8
    WAIT, 0x7E0A021B, *store(5), END,
    CONSTANT,
)


def translate(shader, threads, wave=32):
    return G.ComputePipeline(shader, C.root_blob(), C.USER7, (threads, 1, 1),
                             wave_size=wave).hlsl_cs()


def expected(name, threads, wave, key):
    out = [C.SENTINEL] * C.WORDS
    for lane in range(threads):
        if name == 'nested':
            if lane < 2:
                out[2 * lane] = 0x31
            if lane < 5 or key == 17:
                out[2 * lane + 1] = lane
        elif name == 'ballot':
            first = lane // wave * wave
            limit = 3 if threads == 4 else (27 if wave == 32 else 37)
            count = sum(i < limit for i in range(first, min(first + 32, threads)))
            out[2 * lane:2 * lane + 2] = [count, 7 if count else 9]
        else:
            out[2 * lane] = CONSTANT
    return out


def fixtures():
    # The two ballot cases exercise the low word across guest wave boundaries,
    # including wave64 lanes in the upper half that still consume the result.
    return (
        ('nested', NESTED, 8, 32, 'exec = (s26) & 1;', 'exec = 0u;'),
        ('ballot', BALLOT.replace(words(compare_tid_lt(37)), words(compare_tid_lt(3))), 4, 32,
         'x2w_count_exec_lo(tid, exec != 0u)', '1u'),
        ('ballot', BALLOT.replace(words(compare_tid_lt(37)), words(compare_tid_lt(27))), 40, 32,
         'x2w_count_exec_lo(tid, exec != 0u)', '1u'),
        ('ballot', BALLOT64, 40, 64, 'x2w_count_exec_lo(tid, exec != 0u)', '1u'),
        ('pc', PC, 6, 32, 's27 = 0x51a839d2u;', 's27 = 0x51a839d3u;'),
    )


def sources():
    for name, shader, threads, wave, old, new in fixtures():
        source = translate(shader, threads, wave)
        assert source.count(old) == 1, (name, source.splitlines()[-45:])
        if name == 'nested':
            assert '        if (!(' in source, source.splitlines()[-45:]
        if name == 'ballot':
            assert 'countbits(WaveActiveBallot(true))' in source
            assert f'i / {wave}u * {wave}u' in source
            assert 's28 = scc ? 0x00000007u : 0x00000009u;' in source
        if name == 'pc':
            assert 's28 =' not in source and 's30 =' not in source
        for key in ((0, 17) if name == 'nested' else (0,)):
            reference = expected(name, threads, wave, key)
            yield f'{name}-{threads}-{wave}-{key}', source, source.replace(old, new), reference, key


def rejects(shader, threads, reason):
    try:
        translate(shader, threads)
    except G.Unsupported as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f'expected {reason}')


def controls():
    list(sources())
    rejects(NESTED.replace(words(RESTORE), words(mov(126, 129))), 8, 'exec write')
    rejects(PC[:-4], 6, 'outside trailing data')
    rejects(PC.replace(words(0x7E0A021B), words(0x7E0A021C)), 6, 'PC word escapes')
    dead_before_load = words(sop1(0x1F, 16, 0)) + PC
    assert translate(dead_before_load, 6) == translate(PC, 6)
    # A data word whose encoding is invalid as an instruction is still a
    # valid constant when the address proof puts it beyond the terminator.
    for data in (sop1(0x1F, 30, 1), END):
        assert f's27 = 0x{data:08x}u;' in translate(PC[:-4] + words(data), 6)
    print('PASS: compute EXEC and PC translation controls')


def dispatch(dxc, renderer):
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for name, source, mutant, expected_words, key in sources():
            case = directory / f'{name}.case'
            case.write_bytes(words(0x010101, *expected_words, key, *([0] * 15)))
            for negative, hlsl in enumerate((source, mutant)):
                src = directory / f'{name}-{negative}.hlsl'
                cso = src.with_suffix('.cso')
                src.write_text(hlsl, encoding='ascii')
                subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
                result = subprocess.run([renderer, 'warp', '--compute-store-case', str(cso), str(case)],
                                        capture_output=True, text=True)
                print(result.stdout, end='')
                if result.returncode == 77:
                    print(result.stderr, end='')
                    return 77
                if negative:
                    assert result.returncode == 1 and 'Compute readback differs' in result.stderr, result.stderr
                    print(f'{name} mutant: {result.stderr.strip()}')
                else:
                    assert result.returncode == 0, result.stderr
                    print(f'{name}: all {C.WORDS} readback words match')
    return 0


if __name__ == '__main__':
    controls()
    if len(sys.argv) == 3:
        sys.exit(dispatch(*sys.argv[1:]))
    if len(sys.argv) != 1:
        sys.exit(__doc__)
