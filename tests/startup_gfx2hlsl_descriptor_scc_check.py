"""Synthetic RDNA 2 compute SCC checks, with CPU-reference WARP readback."""

from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_gfx2hlsl_compute_check import (
    G, MASK, SENTINEL, USER7, WORDS, root_blob,
)


def sop1(op, dst, src):
    return struct.pack('<I', 0xbe800000 | dst << 16 | op << 8 | src)


def sop2(op, dst, a, b):
    return struct.pack('<I', 0x80000000 | op << 23 | dst << 16 | b << 8 | a)


def v_mov(dst, src):
    return struct.pack('<I', 0x7e000200 | dst << 17 | src)


SCC, M0, NULL, VCC_LO, EXEC_LO = 253, 124, 125, 106, 126
READ_SCC = sop1(3, 23, SCC)
# Root table s6 (with high word zero), input words s12/s13 at bytes 0/4,
# store descriptor s[16:19] at byte 16. No SRV view is bound for the scalar inputs.
PROLOGUE = bytes.fromhex('06038abe 80038bbe 05030cf4000000fa 7fc08cbf')
SEED = bytes.fromhex('048006bf')  # s_cmp_eq_u32 s4, 0
END = bytes.fromhex('000081bf')


def signed(word):
    return word if word < 0x80000000 else word - (1 << 32)


def unsigned_result(value, underflow=False):
    return value & MASK, int(value < 0 if underflow else value > MASK)


def signed_result(value):
    return value & MASK, int(value != signed(value & MASK))


def mask_result(a, _, incoming):
    result, flag = unsigned_result(a + incoming)
    return result & 1, flag


def nonzero(value):
    word = value & MASK
    return word, int(word != 0)


CASES = {
    'and': (sop2(0x0e, 22, 12, 13), lambda a, b, _: nonzero(a & b)),
    'not': (sop1(7, 22, 12), lambda a, b, _: nonzero(~a)),
    'nand': (sop2(0x18, 22, 12, 13), lambda a, b, _: nonzero(~(a & b))),
    'nor': (sop2(0x1a, 22, 12, 13), lambda a, b, _: nonzero(~(a | b))),
    'xnor': (sop2(0x1c, 22, 12, 13), lambda a, b, _: nonzero(~(a ^ b))),
    'absdiff': (sop2(0x2c, 22, 12, 13), lambda a, b, _: nonzero(abs(signed((a - b) & MASK)))),
    'add': (sop2(0, 22, 12, 13), lambda a, b, _: unsigned_result(a + b)),
    'sub': (sop2(1, 22, 12, 13), lambda a, b, _: unsigned_result(a - b, underflow=True)),
    'signed-add': (sop2(2, 22, 12, 13), lambda a, b, _: signed_result(signed(a) + signed(b))),
    'signed-sub': (sop2(3, 22, 12, 13), lambda a, b, _: signed_result(signed(a) - signed(b))),
    'addc': (sop2(4, 22, 12, 13), lambda a, b, c: unsigned_result(a + b + c)),
    'subb': (sop2(5, 22, 12, 13), lambda a, b, c: unsigned_result(a - b - c, underflow=True)),
    'alias-left': (sop1(3, 22, 5) + sop2(0, 22, 22, SCC), lambda a, b, c: unsigned_result(a + c)),
    'alias-right': (sop1(3, 22, 5) + sop2(0, 22, SCC, 22), lambda a, b, c: unsigned_result(a + c)),
    'm0': (sop1(3, M0, 5) + sop2(0, M0, M0, SCC) + sop1(3, 22, M0),
           lambda a, b, c: unsigned_result(a + c)),
    'null': (sop2(0, NULL, 5, SCC) + sop1(3, 22, 128),
             lambda a, b, c: (0, unsigned_result(a + c)[1])),
    'vcc-lo': (sop2(0, VCC_LO, 5, SCC), mask_result),
    'exec-lo': (sop1(3, 26, EXEC_LO) + sop2(0, EXEC_LO, 5, SCC)
                + sop1(3, VCC_LO, EXEC_LO) + sop1(3, EXEC_LO, 26), mask_result),
}
for shift in range(1, 5):
    CASES[f'shift-add-{shift}'] = (
        sop2(0x2d + shift, 22, 12, 13),
        lambda a, b, _, shift=shift: unsigned_result((a << shift) + b),
    )

INPUTS = ((0, 0, 0), (MASK, MASK, 0), (MASK, 0, 1), (0, MASK, 1),
          (3, 5, 0), (5, 3, 0), (0x80000000, 0, 1), (0x80000000, 1, 1),
          (0x7fffffff, 1, 0), (0x80000000, MASK, 0), (1, MASK, 0),
          (0x08000000, 0, 1))


def program(names):
    code = PROLOGUE + v_mov(0, 128)
    for i, name in enumerate(names):
        # Two dwords at a different immediate offset per case.
        store = struct.pack('<II', 0xe0741000 | 8 * i, 0x80040500)
        # Record the lane's bit for EXEC/VCC, using VCC as the mask source for both.
        result = (v_mov(5, 129) + bytes.fromhex('800a0a02')  # v_cndmask_b32 v5, 0, v5, vcc_lo
                  if name in ('vcc-lo', 'exec-lo') else v_mov(5, 22))
        code += SEED + CASES[name][0] + READ_SCC + result + v_mov(6, 23) + store
    return code + END


def pipeline(code):
    return G.ComputePipeline(code, root_blob(), USER7, (1, 1, 1))


def controls():
    for name, (_, reference) in CASES.items():
        pipe = pipeline(program([name]))
        hlsl = pipe.hlsl_cs()
        assert '(scc ? 1u : 0u)' in hlsl, (name, hlsl)
        assert pipe.retry_scc, name
        assert {reference(*case)[1] for case in INPUTS} == {0, 1}, name

    # A use at entry and a descriptor load through a nonzero high word cannot prove SCC.
    for code, diagnostic in (
        (READ_SCC + END, 'src_scc with unverified SCC'),
        (program(['absdiff']).replace(sop1(3, 11, 128), sop1(3, 11, 129), 1), 'scc from descriptor data'),
    ):
        try:
            pipeline(code)
        except G.Unsupported as error:
            assert diagnostic in str(error), str(error)
        else:
            raise AssertionError('accepted a program without a proven SCC')

    pipe = pipeline(program(CASES))
    for idx, source in enumerate(pipe.uds):
        assert source in (('rootconst', 0, 0), ('rootconst', 0, 1),
                          ('table_word', 1, 0, 0), ('table_word', 1, 0, 1)), source
        assert 'udsrc %d %s\n' % (idx, ' '.join(map(str, source))) in pipe.manifest()
    assert ('table_word', 1, 0, 0) in pipe.uds and ('table_word', 1, 0, 1) in pipe.uds
    assert pipe.manifest().endswith('uav 0 table 1 16\n'), pipe.manifest()
    return pipe


def dispatch(pipe, dxc, renderer, directory):
    hlsl = pipe.hlsl_cs()
    mutant, count = re.subn(r'(s23 = \(scc \? )1u : 0u', r'\g<1>0u : 1u', hlsl)
    assert count == len(CASES), (count, hlsl)
    case_path = directory / 'reference.case'
    for name, source in (('correct', hlsl), ('mutant', mutant)):
        hlsl_path, cso = (directory / (name + suffix) for suffix in ('.hlsl', '.cso'))
        hlsl_path.write_text(source, encoding='ascii')
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, hlsl_path], check=True)

    for a, b, incoming in INPUTS:
        entry = {('rootconst', 0, 0): 1 - incoming, ('rootconst', 0, 1): a,
                 ('table_word', 1, 0, 0): a, ('table_word', 1, 0, 1): b}
        constants = [entry[source] for source in pipe.uds]
        constants += [0] * (16 - len(constants))
        expected = [word for _, reference in CASES.values() for word in reference(a, b, incoming)]

        def run(shader, words):
            padding = [SENTINEL] * (WORDS - len(words))
            case_path.write_bytes(struct.pack('<337I', 0x010101, *words, *padding, *constants))
            result = subprocess.run([renderer, 'warp', '--compute-store-case', directory / (shader + '.cso'),
                                     case_path], capture_output=True, text=True)
            if result.returncode == 77:
                sys.exit(77)
            return result

        good = run('correct', expected)
        assert good.returncode == 0, (hex(a), hex(b), incoming, good.stdout, good.stderr)
        print(good.stdout, end='')
        bad = run('mutant', expected)
        assert bad.returncode == 1 and 'word=1 expected=' in bad.stderr, (bad.stdout, bad.stderr)
        # Check every mutated SCC output, including those after the first mismatch.
        inverted = run('mutant', [word ^ (i % 2) for i, word in enumerate(expected)])
        assert inverted.returncode == 0, (inverted.stdout, inverted.stderr)
    print(f'{len(CASES)} SCC cases passed WARP readback for {len(INPUTS)} operand sets and their mutants')


if __name__ == '__main__':
    pipe = controls()
    if len(sys.argv) == 4 and sys.argv[1] == 'dispatch':
        out = Path(__file__).resolve().parents[1] / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            dispatch(pipe, *sys.argv[2:], Path(tmp))
    elif len(sys.argv) != 1:
        sys.exit('usage: startup_gfx2hlsl_descriptor_scc_check.py [dispatch <dxc> <offscreen_check>]')
    else:
        print(f'{len(CASES)} synthetic SCC translation controls passed')
