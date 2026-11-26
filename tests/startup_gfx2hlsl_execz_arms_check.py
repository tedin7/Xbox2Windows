"""Synthetic scalar selection with a lane-masked UAV store in each arm."""
from pathlib import Path
import struct
import sys
import tempfile

from startup_compute_global_load_check import (
    ROOT, encode_mubuf, encode_smem, encode_sop1, encode_sopp,
    encode_vop1, encode_vop2, encode_vopc, run_case,
)


from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, refused, root_blob


GROUP = (8, 1, 1)
CASES = ((0, 3), (0, 5), (1, 3), (1, 5))


def program(*, extra_entry=None, narrowed=False):
    code = bytearray()
    labels = {}
    branches = []

    def branch(op, target):
        branches.append((len(code), op, target))
        code.extend(bytes(4))

    code += encode_sop1(0x03, 12, 6)  # Descriptor table pointer, low word.
    code += encode_sop1(0x03, 13, 128)
    code += encode_smem(0x03, 20, 12)  # Table load: four SRV words and four UAV words.
    code += encode_sopp(0x0c, 0xc07f)
    code += encode_smem(0x08, 29, 20)   # Read-only buffer, word 0: selector.
    code += encode_sopp(0x0c, 0xc07f)
    code += encode_vop2(0x1a, 6, 130, 0)  # Byte address = lane * 4.
    code += encode_vop1(0x01, 9, 145)      # Arm 1 stores 17.
    code += encode_vop1(0x01, 10, 157)     # Arm 2 stores 29.
    if extra_entry or narrowed:
        code += struct.pack('<I', 0xbf068005)  # s_cmp_eq_u32 s5, 0
    if extra_entry:
        branch(0x04, extra_entry)
    if narrowed:
        branch(0x04, 'compare')
        code += encode_sop1(0x04, 126, 128)
    labels['compare'] = len(code)
    code += encode_vopc(0xc4, 4, 0)  # VCC = lane < bound (root constant s4).
    code += struct.pack('<I', 0xbf06801d)  # s_cmp_eq_u32 s29, 0
    branch(0x04, 'second')
    code += encode_sop1(0x04, 126, 106)
    branch(0x08, 'join')
    labels['first_store'] = len(code)
    code += encode_mubuf(0x1c, 9, 6, 24)
    branch(0x02, 'join')
    labels['second'] = len(code)
    code += encode_sop1(0x04, 126, 106)
    branch(0x08, 'join')
    labels['second_store'] = len(code)
    code += encode_mubuf(0x1c, 10, 6, 24, offset=4 * GROUP[0])
    labels['join'] = len(code)
    code += encode_sopp(0x01)

    for pc, op, target in branches:
        code[pc:pc + 4] = encode_sopp(op, (labels[target] - pc - 4) // 4)
    return bytes(code)


def translate(**variant):
    return G.ComputePipeline(program(**variant), root_blob(), USER7, GROUP).hlsl_cs()


def reference(selector, bound):
    result = [SENTINEL] * WORDS
    for lane in range(GROUP[0]):
        if lane < bound:
            if selector == 0:
                result[lane] = 17
            else:
                result[GROUP[0] + lane] = 29
    return result


def mutations(hlsl):
    start = '        if (!scc) {\n'
    otherwise = '        } else {\n'
    assert hlsl.count(start) == 1
    prefix, arms = hlsl.split(start)
    second, separator, rest = arms.partition(otherwise)
    narrow = 'exec = (vcc) & 1;'
    assert separator and second.count(narrow) == 1 and rest.count(narrow) == 1
    yield ('unmasked_second', prefix + start + second.replace(narrow, 'exec = 1u;') + separator + rest,
           1, 3, GROUP[0] + 3, 29)
    yield ('both_arms', prefix + '        {\n' + second + '        }\n        {\n' + rest,
           0, 3, GROUP[0], 29)
    yield ('wrong_arm', prefix + start.replace('!scc', 'scc') + arms,
           0, 3, 0, SENTINEL)


def check():
    hlsl = translate()
    reason = 'inside a block s_cbranch_execz skips per lane: exec write'
    assert G.lifted_family(reason) == 'execz-arms'
    for variant in ({'extra_entry': 'first_store'}, {'extra_entry': 'second_store'}, {'narrowed': True}):
        refused(reason, lambda: translate(**variant))
    for name, _, selector, bound, word, actual in mutations(hlsl):
        assert reference(selector, bound)[word] != actual, name
    return hlsl


def dispatch(directory, compiler, renderer):
    hlsl = check()
    for selector, bound in CASES:
        result = run_case(directory, compiler, renderer, 'warp', f'case_{selector}_{bound}',
                          hlsl, (1, 1, 1), reference(selector, bound), bound, 0, [selector] + [0] * (WORDS - 1))
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    for name, mutant, selector, bound, word, actual in mutations(hlsl):
        want = reference(selector, bound)
        result = run_case(directory, compiler, renderer, 'warp', name,
                          mutant, (1, 1, 1), want, bound, 0, [selector] + [0] * (WORDS - 1))
        mismatch = f'word={word} expected={want[word]:08x} actual={actual:08x}'
        assert result.returncode == 1 and mismatch in result.stderr, (name, mismatch, result.stdout, result.stderr)
    print('EXEC-zero arms WARP: four readbacks matched and three mutants failed')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'dispatch':
        out = ROOT / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            dispatch(Path(tmp), *sys.argv[2:])
    elif len(sys.argv) == 1:
        check()
        print('EXEC-zero arms translation and refusal controls passed')
    else:
        raise SystemExit('usage: startup_gfx2hlsl_execz_arms_check.py [dispatch DXC OFFSCREEN_CHECK]')
