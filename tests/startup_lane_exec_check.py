"""Per-lane interpreter checks and optional D3D12 WARP readback checks.

The emitted statements are run by a tiny line interpreter for every exec mask of a small wave, once per lowering
(skip_flag, run_restore), and compared with the reference of the model (the lane-exact EXEC model, 3b):
  - a wave with no active lane leaves every scalar the range writes at its pre-range value;
  - a wave with an active lane gets the range's result in every thread, and vector writes only in the active lanes.
The DPP16 rule (section 2a) is checked on all 2^5 combinations of in_range, source exec, FI, BOUND_CTRL and destination
exec against a table written from the rule's text. Mutants (ignore FI, restore per group, write under bound_ctrl 0)
must each be caught. The WARP fixture uses input-selected lanes, nested masks and two HLSL mutants per positive program.
"""
import argparse
import itertools
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
import lane_exec as L  # noqa: E402
from compute_load_format_check import check_references  # noqa: E402
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, assemble, refused, root_blob  # noqa: E402

WAVE = 4
FAILED = []


def expect(name, got, want):
    if got != want:
        FAILED.append(name)
        print('FAIL %s: got %r, want %r' % (name, got, want))


def evaluate(expression, env):
    text = re.sub(r'(\d+)u\b', r'\1', expression)
    text = text.replace('&&', ' and ').replace('||', ' or ')
    text = re.sub(r'!(?!=)', ' not ', text)
    return eval(text, {}, dict(env))


def run(lines, env):
    """Execute the small statement subset the helper emits: `T name = e;`, `name = e;`, `if (c) name = e;`, `if (c) {` ... `}`."""
    stack = [True]
    for raw in lines:
        line = raw.strip()
        if line == '}':
            stack.pop()
            continue
        block = re.fullmatch(r'if \((.*)\) \{', line)
        if block:
            stack.append(stack[-1] and bool(evaluate(block.group(1), env)))
            continue
        if not stack[-1]:
            continue
        guarded = re.fullmatch(r'if \((.*?)\) (\w+) = (.*);', line)
        if guarded:
            if evaluate(guarded.group(1), env):
                env[guarded.group(2)] = evaluate(guarded.group(3), env)
            continue
        declared = re.fullmatch(r'(?:bool|uint) (\w+) = (.*);', line) or re.fullmatch(r'(\w+) = (.*);', line)
        if not declared:
            raise AssertionError('statement not understood: ' + raw)
        env[declared.group(1)] = evaluate(declared.group(2), env)
    return env


def range_body():
    """The guest range: s7 = s7 + 5 (a scalar write), then a vector write under the lane's exec bit."""
    return ['s7 = s7 + 5u;'] + L.masked_write('v3', 'lane + 100u')


def thread(exec_bits, lane, lowering, pre_s7=11):
    """Run one thread's view of the range; returns (s7 after, v3 after)."""
    wave_any = int(any(exec_bits))
    env = {'exec': int(exec_bits[lane]), 'lane': lane, 's7': pre_s7, 'v3': 0, 'wave_any': wave_any}
    lines = L.skip_decision('wave_any')
    if lowering == 'skip_flag':
        lines += L.guarded_range(range_body())
    elif lowering == 'run_restore':
        lines += L.snapshot(['s7']) + range_body() + L.restore(['s7'])
    elif lowering == 'restore_per_group':          # mutant: decision of the whole group instead of the wave
        lines += L.snapshot(['s7']) + range_body() + ['if (group_skip) s7 = pre_s7;']
        env['group_skip'] = 0
    run(lines, env)
    return env['s7'], env['v3']


def reference(exec_bits, lane, pre_s7=11):
    if not any(exec_bits):
        return pre_s7, 0
    return pre_s7 + 5, (lane + 100) if exec_bits[lane] else 0


def range_checks():
    for lowering in ('skip_flag', 'run_restore'):
        for bits in itertools.product((0, 1), repeat=WAVE):
            for lane in range(WAVE):
                expect('%s exec=%s lane %d' % (lowering, ''.join(map(str, bits)), lane),
                       thread(bits, lane, lowering), reference(bits, lane))
    caught = sum(thread(bits, lane, 'restore_per_group') != reference(bits, lane)
                 for bits in itertools.product((0, 1), repeat=WAVE) for lane in range(WAVE))
    expect('mutant: restoring by the group decision instead of the wave is caught', caught > 0, True)


def dpp_reference(in_range, src_exec, fi, bound_ctrl, dst_exec, old, src):
    """From the rule's text: src_valid = in_range and (FI or exec[src]); invalid: BOUND_CTRL reads 0, else no write."""
    if not dst_exec:
        return old
    if in_range and (fi or src_exec):
        return src
    return 0 if bound_ctrl else old


def dpp_checks():
    old, src = 7, 9
    for in_range, src_exec, fi, bound_ctrl, dst_exec in itertools.product((0, 1), repeat=5):
        expect('dpp in_range=%d src_exec=%d fi=%d bc=%d dst_exec=%d' % (in_range, src_exec, fi, bound_ctrl, dst_exec),
               L.dpp_lane(old, src, in_range, src_exec, dst_exec, fi, bound_ctrl),
               dpp_reference(in_range, src_exec, fi, bound_ctrl, dst_exec, old, src))
    for row, bank in ((0, 1), (1, 0), (0, 0)):
        expect('dpp row_mask=%d bank_mask=%d gates the write' % (row, bank),
               L.dpp_lane(old, src, 1, 1, 1, 1, 1, row_enabled=row, bank_enabled=bank), old)
    # the four FI x BOUND_CTRL combinations on an inactive in-range source: the cases the GPU test must tell apart
    combos = {(fi, bc): L.dpp_lane(old, src, 1, 0, 1, fi, bc) for fi in (0, 1) for bc in (0, 1)}
    expect('inactive source, all four FI x BOUND_CTRL', combos, {(0, 0): old, (0, 1): 0, (1, 0): src, (1, 1): src})

    def ignore_fi(in_range, src_exec, fi, bound_ctrl, dst_exec):      # mutant: FI never consulted
        return dpp_reference(in_range, src_exec, 0, bound_ctrl, dst_exec, old, src)
    differs = [c for c in itertools.product((0, 1), repeat=5)
               if ignore_fi(*c) != dpp_reference(c[0], c[1], c[2], c[3], c[4], old, src)]
    expect('mutant: ignoring FI is caught by the table', bool(differs), True)


def join_checks():
    expect('join expression', L.join_value('skip', 'pre_s7', 's7'), '(skip ? pre_s7 : s7)')
    expect('decision text', L.skip_decision('x2w_any_3'), ['bool skip = (x2w_any_3) == 0u;'])
    expect('no barrier or gather in any helper output',
           [line for line in L.skip_decision('a') + L.guarded_range(['x;']) + L.snapshot(['s1']) + L.restore(['s1'])
            if 'Barrier' in line or 'Interlocked' in line], [])


NESTED_PREFIX = """\
s_mov_b32 s30, s6
s_mov_b32 s31, 0
s_load_dwordx8 s[32:39], s[30:31], 0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v11, 2, v0
buffer_load_dword v12, v11, s[32:35], 0 offen
s_waitcnt vmcnt(0)
s_mov_b32 s40, exec_lo
v_cmpx_ne_u32 0, v12
v_lshlrev_b32 v13, 4, v0
"""
NESTED_PREFIX_CODE = "06039ebe80039fbe0f080cf4000000fa7fc08cbf82001634001030e00b0c0880703f8cbf7e03a8be8018aa7d84001a34"

NESTED_BLOCKS = {
    "and": ("""\
s_cbranch_execz 10
s_mov_b32 s41, exec_lo
v_cmpx_lt_u32 5, v12
s_and_saveexec_b32 s42, 0x55555555
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s42
s_mov_b32 exec_lo, s41
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '0a0088bf7e03a9be8518a27dff3caabe55555555001070e00d0c09802a03febe2903febe041070e0'
        '0d0c09802803febe000081bf'),
    "compare": ("""\
s_cbranch_execz 8
s_mov_b32 s41, exec_lo
v_cmp_lt_u32_e64 exec_lo, 5, v12
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s41
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '080088bf7e03a9be7e00c1d485180200001070e00d0c09802903febe041070e00d0c09802803febe'
        '000081bf'),
    "nested": ("""\
s_cbranch_execz 15
s_mov_b32 s41, exec_lo
v_cmpx_lt_u32 5, v12
v_cmp_ne_u32 vcc_lo, 9, v12
s_and_saveexec_b32 s43, vcc_lo
s_and_saveexec_b32 s44, 0xaaaaaaaa
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s44
s_mov_b32 exec_lo, s43
buffer_store_dword v12, v13, s[36:39], 0 offen offset:8
s_mov_b32 exec_lo, s41
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '0f0088bf7e03a9be8518a27d89188a7d6a3cabbeff3cacbeaaaaaaaa001070e00d0c09802c03febe'
        '2b03febe081070e00d0c09802903febe041070e00d0c09802803febe000081bf'),
    "or": ("""\
s_cbranch_execz 12
s_mov_b32 s41, exec_lo
v_cmp_eq_u32 vcc_lo, 3, v12
v_cmpx_lt_u32 5, v12
s_or_saveexec_b32 s45, vcc_lo
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s45
buffer_store_dword v12, v13, s[36:39], 0 offen offset:8
s_mov_b32 exec_lo, s41
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '0c0088bf7e03a9be8318847d8518a27d6a3dadbe001070e00d0c09802d03febe081070e00d0c0980'
        '2903febe041070e00d0c09802803febe000081bf'),
    "join": ("""\
s_mov_b32 s47, exec_lo
s_cbranch_execz 5
v_cmp_lt_u32_e64 exec_lo, 5, v12
s_and_saveexec_b32 s42, -1
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s47
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '7e03afbe050088bf7e00c1d485180200c13caabe001070e00d0c09802f03febe041070e00d0c0980'
        '2803febe000081bf'),
    "or64": ("""\
s_cbranch_execz 12
s_mov_b64 s[48:49], exec
v_cmp_eq_u32 vcc_lo, 3, v12
v_cmpx_lt_u32 5, v12
s_or_saveexec_b64 s[50:51], vcc
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b64 exec, s[50:51]
buffer_store_dword v12, v13, s[36:39], 0 offen offset:8
s_mov_b64 exec, s[48:49]
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_endpgm
""",
        '0c0088bf7e04b0be8318847d8518a27d6a25b2be001070e00d0c09803204febe081070e00d0c0980'
        '3004febe041070e00d0c0980000081bf'),
    "and64": ("""\
s_cbranch_execz 10
s_mov_b64 s[48:49], exec
v_cmp_lt_u32_e64 exec_lo, 5, v12
s_and_saveexec_b64 s[52:53], 0
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b64 exec, s[52:53]
s_mov_b64 exec, s[48:49]
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_endpgm
""",
        '0a0088bf7e04b0be7e00c1d4851802008024b4be001070e00d0c09803404febe3004febe041070e0'
        '0d0c0980000081bf'),
    "or-outside": ("""\
s_cbranch_execz 12
s_mov_b32 s41, exec_lo
v_cmp_eq_u32 vcc_lo, 3, v12
v_cmpx_lt_u32 5, v12
s_or_saveexec_b32 s45, -1
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s45
buffer_store_dword v12, v13, s[36:39], 0 offen offset:8
s_mov_b32 exec_lo, s41
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '0c0088bf7e03a9be8318847d8518a27dc13dadbe001070e00d0c09802d03febe081070e00d0c0980'
        '2903febe041070e00d0c09802803febe000081bf'),
    "count": ("""\
s_cbranch_execz 11
s_mov_b32 s41, exec_lo
v_cmpx_lt_u32 5, v12
s_and_saveexec_b32 s42, 0x55555555
s_bcnt1_i32_b32 s46, exec_lo
buffer_store_dword v12, v13, s[36:39], 0 offen
s_mov_b32 exec_lo, s42
s_mov_b32 exec_lo, s41
buffer_store_dword v12, v13, s[36:39], 0 offen offset:4
s_mov_b32 exec_lo, s40
s_endpgm
""",
        '0b0088bf7e03a9be8518a27dff3caabe555555557e0faebe001070e00d0c09802a03febe2903febe'
        '041070e00d0c09802803febe000081bf'),
    "unrestored": ("""\
s_cbranch_execz 6
s_mov_b32 s41, exec_lo
v_cmpx_lt_u32 5, v12
s_and_saveexec_b32 s42, 0x55555555
buffer_store_dword v12, v13, s[36:39], 0 offen
s_endpgm
""",
        '060088bf7e03a9be8518a27dff3caabe55555555001070e00d0c0980000081bf'),
}


def nested_shaders(llvm_mc):
    errors = {
        'count': 'exec write: wave-wide read of a narrowed mask',
        'or-outside': 's_or_saveexec_b32 inside a block s_cbranch_execz skips per lane: exec write',
        'unrestored': 'exec write: saved mask not restored before the join',
    }
    shaders = {}
    for label, (source, words) in NESTED_BLOCKS.items():
        code = bytes.fromhex(NESTED_PREFIX_CODE + words)
        if llvm_mc:
            assert assemble(llvm_mc, NESTED_PREFIX + source) == code, label

        def translate():
            return G.ComputePipeline(code, root_blob(), USER7, (64, 1, 1)).hlsl_cs()

        if label in errors:
            refused(errors[label], translate)
        else:
            shaders[label] = translate()
    return shaders


def nested_reference(label, source, mutant=None):
    expected = [SENTINEL] * WORDS
    for lane, value in enumerate(source[:64]):
        if not value:
            continue
        passes_compare = value > 5
        primary = {
            'and': passes_compare and lane % 2 == 0,
            'and64': False,
            'compare': passes_compare,
            'join': passes_compare,
            'nested': passes_compare and value != 9 and lane % 2 == 1,
            'or': passes_compare or value == 3,
            'or64': passes_compare or value == 3,
        }[label]
        out = lane * 4
        if primary or mutant == 'unguarded':
            expected[out] = value
        if mutant != 'no-restore' or passes_compare:
            expected[out + 1] = value
        if label in ('nested', 'or', 'or64') and passes_compare:
            expected[out + 2] = value
    return expected


def mutant_shaders(hlsl):
    unguarded, count = re.subn(r'if \(exec != 0u\) \{ (u0\.Store\(a, v12\);) \}', r'\1', hlsl, count=1)
    assert count == 1, 'first store has an EXEC guard'
    no_restore, count = re.subn(r'        exec = (?:x2w_exec_[0-9a-f]+ \? 1u : 0u|\(s47\) & 1);\n'
                               r'(?=        a = \(v13 \+ 4\))', '', hlsl)
    assert count == 1, 'the outer mask is restored before the second store'
    return {'unguarded': unguarded, 'no-restore': no_restore}


def dispatch_nested(directory, shaders, dxc, runner, adapter):
    # Four words per thread. The unused fourth word and the tail must keep their sentinel.
    inputs = {
        'mixed': [(0, 3, 6, 9, 12, 0, 7)[lane % 7] for lane in range(64)],
        'outer-empty': [0] * 64,
        'inner-empty': [1] * 64,
    }
    for label, hlsl in shaders.items():
        cases, mutants = [], {}
        for name, lanes in inputs.items():
            source = lanes + [0] * (WORDS - len(lanes))
            expected = nested_reference(label, source)
            case = directory / f'{label}-{name}.case'
            words = [1, *expected, *([0] * 16), *source]
            case.write_bytes(struct.pack(f'<{len(words)}I', *words))
            cases.append((case, name, None))
            if name == 'mixed':
                for mutant in ('unguarded', 'no-restore'):
                    wrong = nested_reference(label, source, mutant)
                    first = next(i for i in range(WORDS) if wrong[i] != expected[i])
                    mutants[mutant] = [(case, name, f'word={first} expected={expected[first]:08x} actual={wrong[first]:08x}')]
        for name, source, checks in [(label, hlsl, cases)] + [
            (f'{label}-{mutant}', text, mutants[mutant]) for mutant, text in mutant_shaders(hlsl).items()
        ]:
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(source, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, checks, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    range_checks()
    dpp_checks()
    join_checks()
    if FAILED:
        print('FAILED: %d' % len(FAILED))
        return 1
    shaders = nested_shaders(args.llvm_mc)
    for hlsl in shaders.values():
        mutant_shaders(hlsl)
    print('lane_exec: interpreter, nested translations and refusals passed')
    if args.dispatch:
        output = Path(__file__).resolve().parents[1] / 'out'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            return dispatch_nested(Path(temp), shaders, *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
