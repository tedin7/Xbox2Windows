"""Direct compute 64-bit scalar bit counts and bit scans (#3072)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1, vmove


BCNT0_64, BCNT1_64, FF0_64, FF1_64, FLBIT_B64, FLBIT_I64 = 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18
CSELECT32 = 0x0A
ZERO, ONE, NEG_ONE, VCC, EXEC, FLOAT_ONE, LITERAL = 128, 129, 193, 106, 126, 242, 255
OLD_PAIR = (0xCAFEBABE, 0xDEADBEEF)
LITERAL_VALUE = 0x80000001
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
LANES = 64
PLANE_BYTES = LANES * 8
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969


def set_pair(base, words):
    return [sop1(3, base, LITERAL), words[0], sop1(3, base + 1, LITERAL), words[1]]


def set_scc(value):
    # s_cmp_eq_u32 0, (0 for true, 1 for false)
    return 0xBF060080 | (ZERO if value else ONE) << 8


def compute_program(instructions, dst):
    # Two 32-lane groups, with 8 bytes per lane and 256 bytes per group in each plane.
    program = [sop1(3, 20, 4), sop1(3, 21, ZERO),
               0xF4080000 | 32 << 6 | 10, 0xFA000020,
               scalar(0x1E, 7, 5, 136), vop2(0x1A, 8, 131, 0), vop2(0x25, 8, 7, 8)]
    program += instructions
    program += [vmove(2, dst), vmove(3, dst + 1), 0xE0741000, 0x80080208,
                scalar(CSELECT32, 26, ONE, ZERO), vmove(2, 26), vmove(3, ZERO),
                0xE0741000 | PLANE_BYTES, 0x80080208, 0xBF810000]
    return program


def eval_bitscan(op, lo, hi, src_kind, primed_scc, *, no_hi_offset_ff=False, no_lo_offset_flbit=False,
                 unsigned_flbit=False, zero_ext_lit=False, scan_writes_scc=False, zero_is_64=False):
    if src_kind == 'neg_inline':
        hi_word = MASK32
    elif src_kind == 'literal':
        hi_word = 0 if (zero_ext_lit or op != FLBIT_I64 or not (lo >> 31)) else MASK32
    else:
        hi_word = hi
    val = (hi_word << 32) | (lo & MASK32)

    def ff1(v):
        if not v:
            return 64 if zero_is_64 else MASK32
        if no_hi_offset_ff and not (v & MASK32):
            return ((v >> 32) & -(v >> 32)).bit_length() - 1
        return (v & -v).bit_length() - 1

    def flbit_u(v):
        if not v:
            return 64 if zero_is_64 else MASK32
        if no_lo_offset_flbit and not (v >> 32):
            return 32 - (v & MASK32).bit_length()
        return 64 - v.bit_length()

    if op == BCNT0_64:
        res = 64 - bin(val).count('1')
        scc = int(res != 0)
    elif op == BCNT1_64:
        res = bin(val).count('1')
        scc = int(res != 0)
    elif op == FF0_64:
        res = ff1(val ^ MASK64)
        scc = int(res != 0) if scan_writes_scc else primed_scc
    elif op == FF1_64:
        res = ff1(val)
        scc = int(res != 0) if scan_writes_scc else primed_scc
    elif op == FLBIT_B64 or (op == FLBIT_I64 and unsigned_flbit):
        res = flbit_u(val)
        scc = int(res != 0) if scan_writes_scc else primed_scc
    else:
        res = flbit_u((val ^ MASK64) if (val >> 63) else val)
        scc = int(res != 0) if scan_writes_scc else primed_scc
    return res, scc


def cases():
    pair_inputs = (
        ('zero', (0, 0)),
        ('all_ones', (MASK32, MASK32)),
        ('bit32', (0, 1)),
        ('bit31', (0x80000000, 0)),
        ('neg2', (0xFFFFFFFE, MASK32)),
        ('int64_min', (0, 0x80000000)),
        ('bit0', (1, 0)),
    )
    ops = (
        (BCNT0_64, 'bcnt0'),
        (BCNT1_64, 'bcnt1'),
        (FF0_64, 'ff0'),
        (FF1_64, 'ff1'),
        (FLBIT_B64, 'flbit_b64'),
        (FLBIT_I64, 'flbit_i64'),
    )
    for label, pair in pair_inputs:
        for op, opname in ops:
            primed = 1 if op in (FF0_64, FLBIT_B64) else 0
            # Prime SCC opposite to counts or arbitrary for scans; keep s25 = OLD_PAIR[1].
            if op in (BCNT0_64, BCNT1_64):
                bits = bin((pair[1] << 32) | pair[0]).count('1')
                expected_cnt = (64 - bits) if op == BCNT0_64 else bits
                primed = 1 - int(expected_cnt != 0)
            code = set_pair(24, OLD_PAIR) + set_pair(22, pair) + [set_scc(primed), sop1(op, 24, 22)]
            yield (f'{opname} {label}', code, 24, OLD_PAIR[1], op, pair[0], pair[1], 'pair', primed)

    # Inline -1 (sign-extended to all ones) and 32-bit literal 0x80000001 (zero-ext for _b64, sign-ext for _i64).
    yield ('bcnt1 neg1', set_pair(24, OLD_PAIR) + [set_scc(0), sop1(BCNT1_64, 24, NEG_ONE)],
           24, OLD_PAIR[1], BCNT1_64, MASK32, MASK32, 'neg_inline', 0)
    yield ('flbit_i64 neg1', set_pair(24, OLD_PAIR) + [set_scc(1), sop1(FLBIT_I64, 24, NEG_ONE)],
           24, OLD_PAIR[1], FLBIT_I64, MASK32, MASK32, 'neg_inline', 1)
    yield ('bcnt1 lit', set_pair(24, OLD_PAIR) + [set_scc(0), sop1(BCNT1_64, 24, LITERAL), LITERAL_VALUE],
           24, OLD_PAIR[1], BCNT1_64, LITERAL_VALUE, 0, 'literal', 0)
    yield ('flbit_i64 lit', set_pair(24, OLD_PAIR) + [set_scc(0), sop1(FLBIT_I64, 24, LITERAL), LITERAL_VALUE],
           24, OLD_PAIR[1], FLBIT_I64, LITERAL_VALUE, 0, 'literal', 0)

    # Explicit SCC preservation across count -> scan sequence.
    yield ('scc0 preserved by ff1',
           set_pair(24, OLD_PAIR) + set_pair(22, (0, 1)) + [sop1(BCNT1_64, 24, ZERO), sop1(FF1_64, 24, 22)],
           24, OLD_PAIR[1], FF1_64, 0, 1, 'pair', 0)
    yield ('scc1 preserved by flbit_b64',
           set_pair(24, OLD_PAIR) + set_pair(22, (0, 0)) + [sop1(BCNT1_64, 24, ONE), sop1(FLBIT_B64, 24, 22)],
           24, OLD_PAIR[1], FLBIT_B64, 0, 0, 'pair', 1)

    # Overlapping destination: s22 = ff1(s[22:23]), s23 stays 0x01234567.
    yield ('ff1 overlap s22',
           set_pair(22, (0, 0x01234567)) + [set_scc(0), sop1(FF1_64, 22, 22)],
           22, 0x01234567, FF1_64, 0, 0x01234567, 'pair', 0)


def full_reference(**mutant_flags):
    return [eval_bitscan(op, lo, hi, src_kind, primed, **mutant_flags)
            for _, _, _, _, op, lo, hi, src_kind, primed in cases()]


def controls(cli, sdk, work, root):
    combined = compute_program(
        set_pair(24, OLD_PAIR) + set_pair(22, (0x80000001, 0x00000010)) + [
            sop1(BCNT0_64, 24, 22),
            sop1(BCNT1_64, 24, 22),
            sop1(FF0_64, 24, 22),
            sop1(FF1_64, 24, 22),
            sop1(FLBIT_B64, 24, 22),
            sop1(FLBIT_I64, 24, 22),
        ], 24)
    marker = request(work, root, combined)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    for op_id in (31, 32, 33):
        assert f'@dx.op.unaryBits.i32(i32 {op_id}' in dump, (op_id, dump)
    assert 'shl i64' not in dump, dump

    # Adjacent descriptor preservation regression: s_ff1_i32_b64 writes 32-bit s31 right below s[32:35];
    # transfer() must not clear s32 as a 64-bit destination pair.
    adj_prog = compute_program(set_pair(24, OLD_PAIR) + set_pair(22, (0, 1)) + [sop1(FF1_64, 31, 22)], 24)
    adj_marker = request(work, root, adj_prog)
    adj_out = adj_marker.with_suffix('.adj.cso')
    adj_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', adj_marker, adj_out],
                             capture_output=True, text=True)
    assert adj_res.returncode == 0 and adj_out.is_file(), (adj_res.stdout, adj_res.stderr)

    refused = (
        ([sop1(BCNT1_64, VCC, 22), 0xBF810000], 'invalid s_bcnt1_i32_b64 destination'),
        ([sop1(FF1_64, EXEC, 22), 0xBF810000], 'invalid s_ff1_i32_b64 destination'),
        ([sop1(FF0_64, 24, 23), 0xBF810000], 'invalid s_ff0_i32_b64 source'),
        ([sop1(FLBIT_B64, 24, VCC), 0xBF810000], 'invalid s_flbit_i32_b64 source'),
        ([sop1(BCNT0_64, 24, EXEC), 0xBF810000], 'invalid s_bcnt0_i32_b64 source'),
        ([sop1(FLBIT_I64, 24, FLOAT_ONE), 0xBF810000], 'invalid s_flbit_i32_i64 source'),
    )
    for n, (words, error) in enumerate(refused):
        m = request(work, root, words)
        o = m.with_suffix(f'.refused{n}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o],
                           capture_output=True, text=True)
        assert r.returncode and error in r.stderr, (error, r.stdout, r.stderr)
        assert not o.exists(), o

    baseline = full_reference()
    for label, flags in (
        ('missing +32 in high ff0/ff1', {'no_hi_offset_ff': True}),
        ('missing +32 in low flbit', {'no_lo_offset_flbit': True}),
        ('unsigned flbit_i64', {'unsigned_flbit': True}),
        ('zero-extended negative literal in flbit_i64', {'zero_ext_lit': True}),
        ('scan writes SCC', {'scan_writes_scc': True}),
        ('zero returns 64 instead of 0xffffffff', {'zero_is_64': True}),
    ):
        assert full_reference(**flags) != baseline, label

    print('compiled 64-bit scalar counts and scans (adjacent s32 descriptor preserved); '
          '6 invalid forms refused; 6 reference mutants differ')
    return 0


def hardware(replay, sdk, work, root):
    for name, code, dst, neighbor, op, lo, hi, src_kind, primed in cases():
        res, scc = eval_bitscan(op, lo, hi, src_kind, primed)
        marker = request(work, root, compute_program(code, dst))
        output = work / 'out.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes())
        end = GUARD_WORDS + LANES * 4
        assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS, name
        assert actual[GUARD_WORDS:end] == (res, neighbor) * LANES + (scc, 0) * LANES, (name, res, neighbor, scc)
        assert actual[end:] == (SENTINEL,) * (OUTPUT_WORDS - end), name
        print(f'{name}: result, neighbor, SCC and guards matched')
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG',
                        '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
