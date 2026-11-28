"""Direct compute scalar conditional moves (s_cmov_b32, s_cmov_b64, s_cmovk_i32) (#3070)."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_direct_container_check import words as pack_words
from native_f16_alu_check import _branch, _exec, _i32, compute_container
from native_scalar_lds_check import scalar, sop1, vmove


CMOV32, CMOV64, CMOVK, CSELECT32 = 0x05, 0x06, 0x02, 0x0A
ZERO, ONE, NEG_ONE, VCC, VCC_HI, EXEC, FLOAT_ONE, LITERAL = 128, 129, 193, 106, 107, 126, 242, 255
OLD_PAIR = (0xCAFEBABE, 0xDEADBEEF)
SRC_PAIR = (0x89ABCDEF, 0x01234567)
LITERAL32 = 0x12345678
LITERAL64_LO = 0x80000001
FLOAT_ONE_BITS = 0x3F800000
MASK32 = (1 << 32) - 1
LANES = 64
PLANE_BYTES = LANES * 8
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969


def sopk(op, dst, simm16):
    return 0xB0000000 | (op << 23) | (dst << 16) | (simm16 & 0xFFFF)


def set_pair(base, pair):
    return [sop1(3, base, LITERAL), pair[0], sop1(3, base + 1, LITERAL), pair[1]]


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


def cases():
    # Each tuple: (name, instructions, dst, is_pair, is_cmovk, new_words, expected_scc).
    for scc in (0, 1):
        # s_cmov_b32: SGPR, 32-bit literal, inline float, wave32 vcc_hi dst/src; neighbor word stays OLD_PAIR.
        yield (f'cmov32 sgpr SCC={scc}',
               set_pair(24, OLD_PAIR) + set_pair(22, SRC_PAIR) + [set_scc(scc), sop1(CMOV32, 24, 22)],
               24, False, False, (SRC_PAIR[0], OLD_PAIR[1]), scc)
        yield (f'cmov32 lit SCC={scc}',
               set_pair(24, OLD_PAIR) + [set_scc(scc), sop1(CMOV32, 24, LITERAL), LITERAL32],
               24, False, False, (LITERAL32, OLD_PAIR[1]), scc)
        yield (f'cmov32 float SCC={scc}',
               set_pair(24, OLD_PAIR) + [set_scc(scc), sop1(CMOV32, 24, FLOAT_ONE)],
               24, False, False, (FLOAT_ONE_BITS, OLD_PAIR[1]), scc)
        yield (f'cmov32 vcc_hi dst SCC={scc}',
               set_pair(VCC, OLD_PAIR) + set_pair(22, SRC_PAIR) + [set_scc(scc), sop1(CMOV32, VCC_HI, 22)],
               VCC, False, False, (OLD_PAIR[0], SRC_PAIR[0]), scc)
        yield (f'cmov32 vcc_hi src SCC={scc}',
               set_pair(24, OLD_PAIR) + set_pair(VCC, SRC_PAIR) + [set_scc(scc), sop1(CMOV32, 24, VCC_HI)],
               24, False, False, (SRC_PAIR[1], OLD_PAIR[1]), scc)

        # s_cmovk_i32: negative, positive, and wave32 vcc_hi dst; neighbor word untouched.
        yield (f'cmovk neg SCC={scc}',
               set_pair(24, OLD_PAIR) + [set_scc(scc), sopk(CMOVK, 24, -1234)],
               24, False, True, ((-1234) & MASK32, OLD_PAIR[1]), scc)
        yield (f'cmovk pos SCC={scc}',
               set_pair(24, OLD_PAIR) + [set_scc(scc), sopk(CMOVK, 24, 0x7FFF)],
               24, False, True, (0x7FFF, OLD_PAIR[1]), scc)
        yield (f'cmovk vcc_hi dst SCC={scc}',
               set_pair(VCC, OLD_PAIR) + [set_scc(scc), sopk(CMOVK, VCC_HI, -1234)],
               VCC, False, True, (OLD_PAIR[0], (-1234) & MASK32), scc)

        # s_cmov_b64: SGPR pair, inline -1 (sign-extended), 32-bit literal (zero-extended), and vcc dst.
        yield (f'cmov64 pair SCC={scc}',
               set_pair(24, OLD_PAIR) + set_pair(22, SRC_PAIR) + [set_scc(scc), sop1(CMOV64, 24, 22)],
               24, True, False, SRC_PAIR, scc)
        yield (f'cmov64 neg1 SCC={scc}',
               set_pair(24, OLD_PAIR) + [set_scc(scc), sop1(CMOV64, 24, NEG_ONE)],
               24, True, False, (MASK32, MASK32), scc)
        yield (f'cmov64 lit SCC={scc}',
               set_pair(24, OLD_PAIR) + [set_scc(scc), sop1(CMOV64, 24, LITERAL), LITERAL64_LO],
               24, True, False, (LITERAL64_LO, 0), scc)
        yield (f'cmov64 vcc SCC={scc}',
               set_pair(VCC, OLD_PAIR) + set_pair(22, SRC_PAIR) + [set_scc(scc), sop1(CMOV64, VCC, 22)],
               VCC, True, False, SRC_PAIR, scc)


def eval_case(is_pair, is_cmovk, new_words, scc, *, inverted=False, unconditional=False,
              zero_ext_cmovk=False, zero_ext_b64=False, clobber_scc=False):
    cond = (1 - scc) if inverted else (1 if unconditional else scc)
    if cond:
        lo, hi = new_words
        if zero_ext_cmovk and is_cmovk:
            if lo != OLD_PAIR[0] and (lo >> 15):
                lo &= 0xFFFF
            if hi != OLD_PAIR[1] and (hi >> 15):
                hi &= 0xFFFF
        if zero_ext_b64 and is_pair and new_words == (MASK32, MASK32):
            hi = 0
        result_words = (lo, hi)
    else:
        result_words = OLD_PAIR
    out_scc = (1 - scc) if clobber_scc else scc
    return result_words, out_scc


def full_reference(**mutant_flags):
    return [eval_case(is_pair, is_cmovk, new_words, scc, **mutant_flags)
            for _, _, _, is_pair, is_cmovk, new_words, scc in cases()]


TRANSFER_BASELINE = [sop1(3, 20, 4), sop1(3, 21, ZERO), 0xF4080000 | 32 << 6 | 10, 0xFA000020, 0xBF810000]


def refused_programs():
    programs = [
        ([sop1(CMOV32, EXEC, 22), 0xBF810000], 'invalid s_cmov_b32 destination'),
        ([sopk(CMOVK, EXEC, 1), 0xBF810000], 'invalid s_cmovk_i32 destination'),
        ([sop1(CMOV64, EXEC, 22), 0xBF810000], 'invalid s_cmov_b64 destination'),
        ([sop1(CMOV64, 25, 22), 0xBF810000], 'invalid s_cmov_b64 destination'),
        ([sop1(CMOV32, 24, EXEC), 0xBF810000], 'invalid s_cmov_b32 source'),
        ([sop1(CMOV64, 24, 23), 0xBF810000], 'invalid s_cmov_b64 source'),
        ([sop1(CMOV64, 24, VCC), 0xBF810000], 'invalid s_cmov_b64 source'),
        ([sop1(CMOV64, 24, FLOAT_ONE), 0xBF810000], 'invalid s_cmov_b64 source'),
    ]
    # Omit the store epilogue so only the s_cmov* at +0x20 reads the symbolic descriptor s24.
    sym_prefix = [sop1(3, 20, 4), sop1(3, 21, ZERO), 0xF4040000 | 24 << 6 | 10, 0xFA000020] + set_pair(22, SRC_PAIR)
    for cmov in (sop1(CMOV64, 24, 22), sop1(CMOV32, 24, 22), sopk(CMOVK, 24, 1)):
        programs.append((sym_prefix + [cmov, 0xBF810000], 'native compute +0x20: unbound numeric SGPR s24'))
    # transfer() must clear s21 constant state on s_cmovk_i32 before s_load_dwordx4 at +0x10.
    programs.append((
        [sop1(3, 20, 4), sop1(3, 21, ZERO), set_scc(0), sopk(CMOVK, 21, 1),
         0xF4080000 | 32 << 6 | 10, 0xFA000020, 0xBF810000],
        'native compute +0x10: direct descriptor address outside bound table',
    ))
    return programs


def interpret_pair_stores(ir_text):
    """Lane 0 of a straight-line compute module; returns sorted (byte address, lo, hi) stores."""
    body = ir_text.split('define void @main()', 1)[1].split('\n}', 1)[0]
    blocks, label, buf = {}, None, []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == '{':
            continue
        if line.endswith(':') and ' ' not in line:
            if label is not None:
                blocks[label] = buf
            label, buf = line[:-1], []
            continue
        if label is None:
            raise AssertionError(line)
        buf.append(line)
    if label is not None:
        blocks[label] = buf
    values, memory, raw_stores = {}, {}, []
    label, steps = 'entry', 0
    while steps < 200000:
        steps += 1
        followed = False
        for line in blocks[label]:
            if line.startswith('br '):
                label = _branch(line, values)
                followed = True
                break
            if line == 'ret void':
                return sorted(raw_stores)
            if 'rawBufferStore.i32' in line:
                match = re.search(
                    r'rawBufferStore\.i32\(i32 140, %dx\.types\.Handle \S+, i32 (\S+), i32 \S+, i32 (\S+), i32 (\S+),',
                    line,
                )
                assert match, line
                raw_stores.append((
                    _i32(match.group(1), values),
                    _i32(match.group(2), values),
                    _i32(match.group(3), values),
                ))
            else:
                _exec(line, values, memory, [])
        if not followed:
            raise AssertionError('block fell through: ' + label)
    raise AssertionError('step limit')


def check_mutants():
    baseline = full_reference()
    for label, flags in (
        ('inverted SCC', {'inverted': True}),
        ('unconditional move', {'unconditional': True}),
        ('zero-extended cmovk immediate', {'zero_ext_cmovk': True}),
        ('zero-extended s_cmov_b64 -1', {'zero_ext_b64': True}),
        ('clobbered SCC', {'clobber_scc': True}),
    ):
        assert full_reference(**flags) != baseline, label


def ir(dumper):
    all_cases = list(cases())
    refusals = refused_programs()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for index, (name, code, dst, is_pair, is_cmovk, new_words, scc) in enumerate(all_cases):
            path = work / f'case{index}.cdx'
            path.write_bytes(compute_container(pack_words(*compute_program(code, dst))))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode == 0, (name, run.stderr)
            text = run.stdout
            if name == 'cmovk neg SCC=1':
                assert '4294966062' in text and '64302' not in text, (name, text)
            stores = interpret_pair_stores(text)
            assert len(stores) == 2 and stores[0][0] == 0 and stores[1][0] == PLANE_BYTES, (name, stores)
            expected_words, out_scc = eval_case(is_pair, is_cmovk, new_words, scc)
            assert (stores[0][1], stores[0][2]) == expected_words, (name, stores[0], expected_words)
            assert (stores[1][1], stores[1][2]) == (out_scc, 0), (name, stores[1], out_scc)

        base_path = work / 'transfer_base.cdx'
        base_path.write_bytes(compute_container(pack_words(*TRANSFER_BASELINE)))
        base_run = subprocess.run([dumper, str(base_path)], capture_output=True, text=True)
        assert base_run.returncode == 0, base_run.stderr

        for index, (prog_words, error) in enumerate(refusals):
            path = work / f'refused{index}.cdx'
            path.write_bytes(compute_container(pack_words(*prog_words)))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode != 0 and error in run.stderr, (error, run.returncode, run.stderr)

    check_mutants()
    print(f'evaluated DXIL IR for {len(all_cases)} s_cmov cases, verified {len(refusals)} exact-pc/diagnostic '
          'refusals + transfer() baseline, and confirmed 5 reference mutants differ')
    return 0


def controls(cli, sdk, work, root):
    accepted = [case for case in cases() if case[-1] == 1]
    for index, (name, code, dst, _, _, _, _) in enumerate(accepted):
        marker = request(work, root, compute_program(code, dst))
        output = marker.with_suffix(f'.ok{index}.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        assert result.returncode == 0 and output.is_file(), (name, result.stdout, result.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
        assert 'select i1' in dump, (name, dump)

    base_marker = request(work, root, TRANSFER_BASELINE)
    base_out = base_marker.with_suffix('.transfer.cso')
    base_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', base_marker, base_out],
                              capture_output=True, text=True)
    assert base_res.returncode == 0 and base_out.is_file(), (base_res.stdout, base_res.stderr)

    refusals = refused_programs()
    for n, (prog_words, error) in enumerate(refusals):
        m = request(work, root, prog_words)
        o = m.with_suffix(f'.refused{n}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o],
                           capture_output=True, text=True)
        assert r.returncode and error in r.stderr, (error, r.stdout, r.stderr)
        assert not o.exists(), o

    check_mutants()
    print(f'compiled all {len(accepted)} accepted s_cmov forms; verified {len(refusals)} exact-pc/diagnostic '
          'refusals + transfer() baseline; 5 reference mutants differ')
    return 0


def hardware(replay, sdk, work, root):
    for name, code, dst, is_pair, is_cmovk, new_words, scc in cases():
        result_words, out_scc = eval_case(is_pair, is_cmovk, new_words, scc)
        marker = request(work, root, compute_program(code, dst))
        output = work / 'out.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes())
        end = GUARD_WORDS + LANES * 4
        assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS, name
        assert actual[GUARD_WORDS:end] == result_words * LANES + (out_scc, 0) * LANES, (name, result_words, out_scc)
        assert actual[end:] == (SENTINEL,) * (OUTPUT_WORDS - end), name
        print(f'{name}: result words, preserved SCC and guards matched')
    return 0


def main(mode, tool, sdk=None):
    if mode == 'ir':
        return ir(tool)
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
