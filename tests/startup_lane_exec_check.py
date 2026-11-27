"""Unit checks of tools/startup/lane_exec.py against an independent per-lane reference (no GPU, no D3D12).

The emitted statements are run by a tiny line interpreter for every exec mask of a small wave, once per lowering
(skip_flag, run_restore), and compared with the reference of the model (the lane-exact EXEC model, 3b):
  - a wave with no active lane leaves every scalar the range writes at its pre-range value;
  - a wave with an active lane gets the range's result in every thread, and vector writes only in the active lanes.
The DPP16 rule (section 2a) is checked on all 2^5 combinations of in_range, source exec, FI, BOUND_CTRL and destination
exec against a table written from the rule's text. Mutants (ignore FI, restore per group, write under bound_ctrl 0)
must each be caught. The GPU half of the proof (WARP readback of the same cases) is a separate test.
"""
import itertools
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
import lane_exec as L  # noqa: E402

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


def compare_program(encoding):
    from startup_compute_lds_dual_atomics_check import (
        G, PREAMBLE_LINES, USER7, le32, mubuf_store, root_blob, sopp, v_mov_imm, vop2, vopc,
    )

    code = b''.join(raw for _, raw in PREAMBLE_LINES)
    code += vop2(0x1a, 1, 0x9f, 0)             # sign bit from lane parity
    code += vop2(0x1d, 1, 256, 1)              # v1 = lane XOR sign bit
    code += v_mov_imm(2, 16)
    code += vop2(0x1a, 3, 0x82, 0)             # output address
    if encoding == 'e32':
        compare = vopc(0xd1, 257, 2)
    else:
        compare = le32(0xd4d10000) + le32(257 | (258 << 9))
    (_, _, name, asm), = G.decode(compare)
    assert name == 'v_cmpx_lt_u32_' + encoding
    assert G.scalar_effects(name, asm)[1] == {'exec'}
    code += compare + mubuf_store(0x1c, 1, 3) + sopp(0x01)
    return G.ComputePipeline(code, root_blob(), USER7, (32, 1, 1)).hlsl_cs()


def compare_gpu(dxc, runner, adapter):
    from startup_compute_lds_dual_atomics_check import (
        SENTINEL, WORDS, check_debug_layer, renderer_diagnostics, run_shader,
    )

    values = [lane ^ ((lane & 1) << 31) for lane in range(32)]
    expected = [value if value < 16 else SENTINEL for value in values]
    expected += [SENTINEL] * (WORDS - 32)
    hlsl = compare_program('e32')
    assert hlsl.count('v1 < v2') == 1
    variants = {
        'e32': hlsl,
        'e64': compare_program('e64'),
        'signed': hlsl.replace('v1 < v2', 'asint(v1) < asint(v2)'),
    }
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for mode, source in variants.items():
            run = run_shader(mode, dxc, runner, adapter, Path(tmp), source, expected)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            check_debug_layer(mode, run.stdout, run.stderr)
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if mode == 'signed':
                assert run.returncode == 1 and diagnostics, (mode, run.returncode, run.stderr)
                assert diagnostics[0] == 'word=1 expected=c5c5c5c5 actual=80000001', diagnostics
                print('signed control rejected at ' + diagnostics[0])
            else:
                assert not run.returncode and not diagnostics, (mode, run.returncode, run.stderr)
    return 0


def main():
    range_checks()
    dpp_checks()
    join_checks()
    expect('e32 and e64 lower the same unsigned predicate', compare_program('e32'), compare_program('e64'))
    if FAILED:
        print('FAILED: %d' % len(FAILED))
        return 1
    print('lane_exec: all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(compare_gpu(*sys.argv[2:]) if sys.argv[1:2] == ['dispatch'] else main())
