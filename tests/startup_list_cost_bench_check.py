"""Speed bench: the cost per call of a translated Dispatch and its neighbours through the stand-in (list slots 35, 39, 14).

usage: startup_list_cost_bench_check.py <startup_stream_consumer> <dxc> <warp|hardware> [--reset]

Runs the compute fixture of startup_compute_dispatch_check.py in mode dispatch-bench: X2W_TEST_BENCH_CALLS (default 2000 here) rounds of
{SetComputeRootDescriptorTable, SetComputeRoot32BitConstants, Dispatch}, each call timed by the fixture. The output words must still match
the case's expectation (the dispatches really ran), and the run prints `x2w-bench` and the stand-in-shaped slots line that
tools/startup/slot_compare.py reads. The numbers are the scoreboard of the translate-once work: compare runs, not WARP against the tester.
With X2W_BENCH_OUT=<file> the slots line is also written there. Exit 77 when the adapter is not available.
The --reset WARP mode measures Reset lock acquisitions while a translated draw records and checks
the fenced draw readback in both switch states.
"""
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import container, gfx2hlsl as G
from startup_stream_draw_check import vertex, vop1

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
        print(f'startup_list_cost_bench: PASS on {adapter} (all {C.WORDS} words match; numbers above)')


def reset_bench(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache = directory / 'cache'
        cache.mkdir()
        vs, ps, root, output = (directory / name for name in ('vs', 'ps', 'root', 'pixels'))
        vs.write_bytes(vertex())
        # Opaque magenta inside the scissor, blue clear outside.
        ps.write_bytes(container(0, [vop1(1, 2, 242), vop1(1, 3, 128),
                                     0xf800180f, 0x02020302, 0xbf810000]))
        args = [consumer, vs, ps, root]
        env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1',
                   X2W_D3D12_RENDER='1', X2W_SHADER_CACHE=str(cache), X2W_FXC='',
                   X2W_TEST_RESET_BENCH='1', X2W_TEST_VIEWPORT='')
        subprocess.run(args, env=env, check=True, timeout=30)
        subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], env=env, check=True, timeout=60)
        key = G.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
        manifest = cache / f'{key:016x}.txt'
        assert manifest.is_file(), f'No manifest for {key:016x}'
        for stage in ('vs', 'ps'):
            base = cache / f'{key:016x}.{stage}'
            subprocess.run([dxc, '-Gis', '-T', f'{stage}_6_0', '-E', 'main', '-Fo',
                            f'{base}.cso', f'{base}.hlsl'], check=True, timeout=30)
        # The synthesized draw uses the sampler root table even though its pixel color is constant.
        # This exercises recording, submitting and clearing a real shader-visible heap.
        with manifest.open('a', encoding='ascii') as stream:
            stream.write(f'smpheap {G.SAMPLER_HEAP_SIZE} {G.SAMPLER_HEAP_SPACE} fallback {G.SAMPLER_HEAP_FALLBACK}\n')

        counts = {}
        expected = (bytes((255, 0, 255, 255)) * 4 + bytes((0, 0, 255, 255)) * 4) * 4
        for mode in ('1', '0'):
            if output.exists():
                output.unlink()
            result = subprocess.run(args + [adapter, output], env=dict(env, X2W_RESET_NOLOCK=mode),
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
            assert '-> drawn;' in result.stderr and 'sampler_heap=1' in result.stderr, result.stderr
            assert 'x2w-draw sampler heap:' in result.stderr, result.stderr
            actual = output.read_bytes()
            assert actual == expected, f'X2W_RESET_NOLOCK={mode}: {actual.hex()} != {expected.hex()}'
            match = re.search(r'^x2w-reset-bench resets=(\d+) draw_lock_acquires=(\d+)$',
                              result.stdout, re.MULTILINE)
            assert match, result.stdout
            resets, calls = map(int, match.groups())
            assert resets == 4 * 256 + 1, resets
            counts[mode] = calls
            print(f'X2W_RESET_NOLOCK={mode}: {resets} resets, {calls} draw_lock acquisitions, 32 exact pixels')

        def require_no_lock(calls):
            if calls:
                raise AssertionError(f'Reset acquired draw_lock {calls} times')

        require_no_lock(counts['1'])
        assert counts['0'] == 2 * resets, counts
        try:
            require_no_lock(counts['0'])
        except AssertionError as error:
            print(f'Legacy control rejected: {error}')
        else:
            raise AssertionError('Legacy Reset unexpectedly met the zero-lock requirement')


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[3:] == ['warp', '--reset']:
        reset_bench(*sys.argv[1:4])
    elif len(sys.argv) == 4 and sys.argv[3] in ('warp', 'hardware'):
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
