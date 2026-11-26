"""Per-lane EXEC checks and a D3D12 WARP readback of a loop with retiring lanes.

The emitted statements are run by a tiny line interpreter for every exec mask of a small wave, once per lowering
(skip_flag, run_restore), and compared with the reference of the model (the lane-exact EXEC model, 3b):
  - a wave with no active lane leaves every scalar the range writes at its pre-range value;
  - a wave with an active lane gets the range's result in every thread, and vector writes only in the active lanes.
The DPP16 rule (section 2a) is checked on all 2^5 combinations of in_range, source exec, FI, BOUND_CTRL and destination
exec against a table written from the rule's text. Mutants (ignore FI, restore per group, write under bound_ctrl 0)
must each be caught. The GPU half of the proof (WARP readback of the same cases) is a separate test.
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
from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, assemble, refused, root_blob

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



# The input word gives the number of list entries for one lane. Zero bypasses
# the body. Each processed entry adds seven, with at most nine entries visited.
RETIRE_SOURCE = """\
s_mov_b32 s36, s6
s_mov_b32 s37, 0
s_load_dwordx8 s[40:47], s[36:37], 0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v11, 2, v0
buffer_load_dword v12, v11, s[40:43], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v13, 19
v_mov_b32_e32 v14, 0
s_mov_b32 s50, exec_lo
v_cmpx_lt_u32_e32 0, v12
s_mov_b32 s53, 0
s_cbranch_execz 5
v_add_nc_u32_e32 v13, 7, v13
v_add_nc_u32_e32 v14, 1, v14
v_cmpx_gt_u32_e32 v12, v14
s_and_b32 exec_lo, exec_lo, exec_lo
s_cbranch_scc0 3
s_add_u32 s53, s53, 1
s_cmp_ge_u32 s53, 9
s_cbranch_scc0 -9
s_mov_b32 exec_lo, s50
buffer_store_dword v13, v11, s[44:47], 0 offen
s_endpgm
"""
RETIRE_CODE = bytes.fromhex(
    '0603a4be 8003a5be 120a0cf4000000fa 7fc08cbf 82001634 001030e00b0c0a80 '
    '703f8cbf 93021a7e 80021c7e 7e03b2be 8018a27d 8003b5be 050088bf '
    '871a1a4a 811c1c4a 0c1da87d 7e7e7e87 030084bf '
    '35813580 358909bf f7ff84bf 3203febe 001070e00b0d0b80 000081bf')

# The identity fill and permutations OR the active requests across a wave32.
# Putting this at the join makes skipping the tail for one lane unsafe.
REDUCE_SOURCE = """\
s_or_saveexec_b32 s57, -1
v_cndmask_b32_e64 v21, 0, v12, s57
v_or_b32_dpp v21, v21, v21 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_or_b32_dpp v21, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_or_b32_dpp v21, v21, v21 row_mirror row_mask:0xf bank_mask:0xf
v_or_b32_dpp v21, v21, v21 row_half_mirror row_mask:0xf bank_mask:0xf
v_permlanex16_b32 v22, v21, 0, 0
v_or_b32_e32 v21, v22, v21
s_mov_b32 exec_lo, s57
v_readlane_b32 s58, v21, 31
"""
REDUCE_CODE = bytes.fromhex(
    'c13db9be 150001d58018e600 fa2a2a38154e00ff fa2a2a3815b100ff '
    'fa2a2a38154001ff fa2a2a38154101ff 160078d715010102 162b2a38 3903febe 3a0060d7153f0100')
OLD_EXIT = ('s_cbranch_scc0 inside a block s_cbranch_execz skips per lane: '
            'loop exit crosses a backward s_cbranch_scc0')


def retire_hlsl(code):
    return G.ComputePipeline(code, root_blob(), USER7, (64, 1, 1)).hlsl_cs()


def retire_controls(llvm_mc):
    # The reduction sits between the exit branch and the original latch. Adjust
    # both displacements that cross it; the EXEC skip still targets the join.
    extra = len(REDUCE_CODE) // 4
    reduction_source = RETIRE_SOURCE.replace('s_cbranch_scc0 3\n',
                                              f's_cbranch_scc0 {3 + extra}\n' + REDUCE_SOURCE)
    reduction_source = reduction_source.replace('s_cbranch_scc0 -9', f's_cbranch_scc0 {-9 - extra}')
    tail = bytes.fromhex('030084bf 35813580 358909bf f7ff84bf')
    with_reduction = RETIRE_CODE.replace(tail, struct.pack('<H', 3 + extra) + tail[2:4]
                                        + REDUCE_CODE + tail[4:-4]
                                        + struct.pack('<H', (-9 - extra) & 0xffff) + tail[-2:])
    # Reading the scalar loop counter after the common exit would expose the
    # number of wave iterations, which a retiring thread does not observe.
    with_counter = RETIRE_CODE.replace(bytes.fromhex('3203febe'), bytes.fromhex('3203febe 35021a7e'))
    if llvm_mc:
        assert assemble(llvm_mc, RETIRE_SOURCE) == RETIRE_CODE
        assert assemble(llvm_mc, reduction_source) == with_reduction

    source = retire_hlsl(RETIRE_CODE)
    assert 'while (!left)' in source
    refused(OLD_EXIT, lambda: retire_hlsl(with_reduction))
    refused('no uniform or lane-local loop-exit value for s53', lambda: retire_hlsl(with_counter))
    return source


def dispatch_retiring(directory, dxc, runner, adapter, source):
    counts = [(lane * 17) % 64 for lane in range(64)]
    expected = [19 + 7 * min(count, 9) for count in counts]
    padding = [SENTINEL] * (WORDS - len(expected))
    data = [1, *expected, *padding, *([0] * 16), *counts, *padding]
    case = directory / 'retiring.case'
    case.write_bytes(struct.pack('<%dI' % len(data), *data))

    # Vote at the join so bypassing lanes also take the erroneous wave exit.
    local_exit = '                if (left) break;\n            }'
    assert source.count(local_exit) == 1
    mutant = source.replace(local_exit, '            }\n'
                            '            left = WaveActiveAnyTrue(left);\n'
                            '            if (left) break;')
    for name, hlsl, diagnostic in (('retiring', source, None), ('wave-break', mutant, 'word=')):
        src = directory / (name + '.hlsl')
        shader = src.with_suffix('.cso')
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        if check_references(name, runner, adapter, shader, [(case, name, diagnostic)], allow_skip=True) == 77:
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
    source = retire_controls(args.llvm_mc)
    if FAILED:
        print('FAILED: %d' % len(FAILED))
        return 1
    print('lane_exec: all checks passed')
    if args.dispatch:
        output = Path(__file__).resolve().parent.parent / 'out'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            return dispatch_retiring(Path(tmp), *args.dispatch, source)
    return 0


if __name__ == '__main__':
    sys.exit(main())
