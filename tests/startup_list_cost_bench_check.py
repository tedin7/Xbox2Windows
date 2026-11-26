"""Speed bench: the cost per call of a translated Dispatch and its neighbours through the stand-in (list slots 35, 39, 14).

usage: startup_list_cost_bench_check.py <startup_stream_consumer> <dxc> <warp|hardware> [startup_indirect_draw_indexed]

Runs the compute fixture of startup_compute_dispatch_check.py in mode dispatch-bench: X2W_TEST_BENCH_CALLS (default 2000 here) rounds of
{SetComputeRootDescriptorTable, SetComputeRoot32BitConstants, Dispatch}, each call timed by the fixture. The output words must still match
the case's expectation (the dispatches really ran), and the run prints `x2w-bench` and the stand-in-shaped slots line that
tools/startup/slot_compare.py reads. The numbers are the scoreboard of the translate-once work: compare runs, not WARP against the tester.
With the indirect fixture executable, also checks mesh refusal counters, request publication and GPU readback
with the switch enabled and with the legacy path as a cost control.
With X2W_BENCH_OUT=<file> the slots line is also written there. Exit 77 when the adapter is not available.
"""
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl as G

CALLS = '2000'  # two recording threads by default: two lists record the same new pipeline at once (a refusal cached while the first read was in flight would show here)


def mesh_result(output, requests):
    stats = re.search(r'^mesh refusal bench: exclusive=(-?\d+) shared=(-?\d+) hashes=(\d+)$', output, re.M)
    assert stats, output[-3000:]
    assert 'indirect-graphics refusals: total 40000; distinct keys 8, distinct pairs 8 (overflow calls 0)' in output
    pairs = dict(re.findall(r'indirect-graphics refusal \d+: key=([0-9a-f]{16}) reason=mesh no translation calls=(\d+)', output))
    uses = dict(re.findall(r'key use mesh ([0-9a-f]{16}) first [^\n]+ uses (\d+)', output))
    files = {path.stem.removeprefix('req_'): path.read_bytes() for path in requests.glob('req_*.m57')}
    assert set(files.values()) == {b'bench' + bytes([index]) for index in range(8)}, files
    assert len(files) == 8
    for key in files:
        assert (requests / f'req_{key}.root').stat().st_size > 0
        assert (requests / f'req_{key}.ps57').stat().st_size == 0
    assert pairs == uses == dict.fromkeys(files, '5000'), (pairs, uses, files)
    lookup = re.search(r'^known pipeline bench: shared=(\d+)$', output, re.M)
    assert lookup, output[-3000:]
    return (*map(int, stats.groups()), int(lookup[1]))


def check_mesh_bench(executable, adapter, directory):
    def run(switch):
        requests = directory / f'mesh-{switch}'
        requests.mkdir()
        env = dict(os.environ, X2W_NOLOCK_REFUSE=switch, X2W_TEST_MESH_REFUSAL_STATS='1',
                   X2W_SHADER_CACHE=str(requests))
        result = subprocess.run([executable, adapter, 'mesh-refusal-bench'], env=env, text=True, capture_output=True)
        if result.returncode == 77:
            print(result.stdout)
            sys.exit(77)
        assert result.returncode == 0, (result.stdout[-3500:], result.stderr[-3500:])
        return mesh_result(result.stdout, requests)

    optimized = run('1')
    legacy = run('0')  # Negative control for the same cost assertions.
    expected = (0, 0, 8, 1)
    assert optimized == expected, f'mesh + resolved-key counters {optimized}, expected {expected}'
    assert legacy == (40000, 40000, 40000, 32), f'legacy cost control did not expose all repeated work: {legacy}'
    print(f'mesh refusal bench: 40000 calls, 8 requests; counters optimized {optimized}, legacy {legacy}')


def main(consumer, dxc, adapter, mesh_fixture=None):
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
        if os.environ.get('X2W_NOLOCK_REFUSE'):
            env['X2W_NOLOCK_REFUSE'] = os.environ['X2W_NOLOCK_REFUSE']
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
        refused = lines(result, 'refused')
        census = lines(result, 'x2w-render compute census key ')
        assert len(census) == 1, census
        translated = int(census[0].split(' dispatches, ')[1].split(' translated')[0])
        assert translated >= rounds, f'only {translated} Dispatches were translated, one list alone does {rounds}\n' + census[0]
        if env.get('X2W_NOLOCK_REFUSE') == '1':
            # A list that races the first manifest read waits for it: nothing is refused, every Dispatch of every list is translated.
            assert translated == threads * rounds and not refused, (
                f'X2W_NOLOCK_REFUSE=1: {translated} of {threads * rounds} Dispatches translated, {len(refused)} refusals\n' + census[0])
        refused = lines(result, 'refused')
        print(bench[0])
        print(slots[0])
        target = os.environ.get('X2W_BENCH_OUT')
        if target:
            Path(target).write_text(slots[0] + '\n')
        if mesh_fixture:
            check_mesh_bench(mesh_fixture, adapter, directory)
        print(f'startup_list_cost_bench: PASS on {adapter} (all {C.WORDS} words match; numbers above)')


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5) or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
