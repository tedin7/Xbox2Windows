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

CALLS = '2000'  # two recording threads by default: two lists record the same new pipeline at once (a refusal cached while the first read was in flight would show here)


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
        env['X2W_TEST_BENCH_THREADS'] = os.environ.get('X2W_TEST_BENCH_THREADS', '2')
        if os.environ.get('X2W_LOCK_SPLIT'):
            env['X2W_LOCK_SPLIT'] = os.environ['X2W_LOCK_SPLIT']
        got, result = dispatch(directory, consumer, env, 'dispatch-bench')
        expected = C.expected('A')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, f'{len(wrong)} words differ after the bench dispatches\n{result.stderr[-2000:]}'
        bench = [line for line in result.stdout.splitlines() if line.startswith('x2w-bench ')]
        slots = lines(result, 'command-list slots over 20 ms')
        assert len(bench) == 1 and len(slots) == 1, (result.stdout[-800:], result.stderr[-800:])
        assert 'slot14 ' in slots[0] and 'slot35 ' in slots[0] and 'slot39 ' in slots[0], slots
        # A list that asks for the new pipeline while another list's first manifest read is in flight is refused once ("no translation");
        # that is the stand-in's existing behaviour. More than one such refusal per extra list means a refusal outlived the read.
        threads = int(env['X2W_TEST_BENCH_THREADS'])
        rounds = int(env['X2W_TEST_BENCH_CALLS'])
        # The list that reads the manifest first translates all of its Dispatches; a racing list is refused while that read is in flight
        # (the stand-in's existing behaviour), so only this lower bound holds for every thread count.
        census = lines(result, 'x2w-render compute census key ')
        assert len(census) == 1, census
        translated = int(census[0].split(' dispatches, ')[1].split(' translated')[0])
        assert translated >= rounds, f'only {translated} Dispatches were translated, one list alone does {rounds}\n' + census[0]
        if env.get('X2W_LOCK_SPLIT') == '1':
            # A list that races the first manifest read waits for it: nothing is refused, every Dispatch of every list is translated.
            assert translated == threads * rounds and not refused, (
                f'X2W_LOCK_SPLIT=1: {translated} of {threads * rounds} Dispatches translated, {len(refused)} refusals\n' + census[0])
        refused = lines(result, 'refused')
        assert len(refused) <= threads - 1, refused
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
