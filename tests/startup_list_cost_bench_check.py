"""Speed bench: the cost per call of a translated Dispatch and its neighbours through the stand-in (list slots 35, 39, 14).

usage: startup_list_cost_bench_check.py <startup_stream_consumer> <dxc> <warp|hardware>

Runs the compute fixture of startup_compute_dispatch_check.py in mode dispatch-bench: X2W_TEST_BENCH_CALLS (default 2000 here) rounds of
{SetComputeRootDescriptorTable, SetComputeRoot32BitConstants, Dispatch}, each call timed by the fixture. The output words must still match
the case's expectation (the dispatches really ran), and the run prints `x2w-bench` and the stand-in-shaped slots line that
tools/startup/slot_compare.py reads. The numbers are the scoreboard of the translate-once work: compare runs, not WARP against the tester.
With X2W_BENCH_OUT=<file> the slots line is also written there. Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl as G

CALLS = '2000'


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        env['X2W_TEST_BENCH_CALLS'] = os.environ.get('X2W_TEST_BENCH_CALLS', CALLS)
        got, result = dispatch(directory, consumer, env, 'dispatch')
        published = lines(result, 'x2w-draw compute request')
        key = int(published[0].split()[3], 16)
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out
        env['X2W_TEST_BENCH_THREADS'] = os.environ.get('X2W_TEST_BENCH_THREADS', '1')
        got, result = dispatch(directory, consumer, env, 'dispatch-bench')
        expected = C.expected('A')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, f'{len(wrong)} words differ after the bench dispatches\n{result.stderr[-2000:]}'
        bench = [line for line in result.stdout.splitlines() if line.startswith('x2w-bench ')]
        slots = lines(result, 'command-list slots over 20 ms')
        assert len(bench) == 1 and len(slots) == 1, (result.stdout[-800:], result.stderr[-800:])
        assert 'slot14 ' in slots[0] and 'slot35 ' in slots[0] and 'slot39 ' in slots[0], slots
        assert not lines(result, 'refused'), result.stderr[-2000:]
        print(bench[0])
        print(slots[0])
        target = os.environ.get('X2W_BENCH_OUT')
        if target:
            Path(target).write_text(slots[0] + '\n')
        print(f'startup_list_cost_bench: PASS on {adapter} (all {C.WORDS} words match; numbers above)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
