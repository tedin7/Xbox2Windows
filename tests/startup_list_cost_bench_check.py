"""Speed bench: the cost per call of a translated Dispatch and its neighbours through the stand-in (list slots 35, 39, 14).

usage: startup_list_cost_bench_check.py <startup_stream_consumer> <dxc> <warp|hardware>

Runs the compute fixture of startup_compute_dispatch_check.py in mode dispatch-bench: X2W_TEST_BENCH_CALLS (default 2000 here) rounds of
{SetComputeRootDescriptorTable, SetComputeRoot32BitConstants, Dispatch}, each call timed by the fixture. The output words must still match
the case's expectation (the dispatches really ran), and the run prints `x2w-bench` and the stand-in-shaped slots line that
tools/startup/slot_compare.py reads. The numbers are the scoreboard of the translate-once work: compare runs, not WARP against the tester.
With X2W_BENCH_OUT=<file> the slots line is also written there. Exit 77 when the adapter is not available.
"""
import os
import re
import struct
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl as G

CALLS = '2000'  # two recording threads by default: two lists record the same new pipeline at once (a refusal cached while the first read was in flight would show here)


def chunks(path):
    """The capture framing of a replay file: (tag, header bytes, data bytes) per chunk."""
    data = Path(path).read_bytes()
    at, found = 0, []
    while at + 16 <= len(data):
        tag = data[at:at + 4].decode('latin1')
        header, size = struct.unpack_from('<IQ', data, at + 4)
        found.append((tag, header, size))
        at += 16 + header + size
    return found, len(data)


def us_per_call(line):
    return {name: float(value) for name, value in re.findall(r'(slot\d+) ([0-9.]+) us/call', line)}


def measure_recording(directory, consumer, env, plain_line, rounds, threads):
    """The cost of the replay recorder on the same bench: per-call us with X2W_REPLAY_RECORD=all against the plain run above, and the
    recording's bytes per call and per kind of chunk (the number a tester recording's size is extrapolated from)."""
    folder = directory / 'record'
    folder.mkdir()
    recorded_env = dict(env, X2W_REPLAY_RECORD='all', X2W_REPLAY_DIR=str(folder), X2W_REPLAY_MAX_MIB='512')
    got, result = dispatch(directory, consumer, recorded_env, 'dispatch-bench')
    line = [text for text in result.stdout.splitlines() if text.startswith('x2w-bench ')][0]
    file = folder / 'replay.x2wr'
    assert file.exists(), 'no recording was written\n' + result.stderr[-1500:]
    found, size = chunks(file)
    kinds = {}
    for tag, header, data in found:
        total = kinds.setdefault(tag, [0, 0])
        total[0] += 1
        total[1] += 16 + header + data
    assert 'XEND' in kinds, 'the recording has no end chunk: ' + str(sorted(kinds))
    calls = kinds.get('CALL', [0, 0])[0]
    assert calls >= 3 * rounds * threads, f'{calls} recorded calls for {3 * rounds * threads} bench calls'
    plain, recorded = us_per_call(plain_line), us_per_call(line)
    print(plain_line)
    print(line.replace('x2w-bench ', 'x2w-bench RECORDING ', 1))
    print('x2w-bench recorder overhead us/call: ' + ', '.join(f'{slot} +{recorded[slot] - plain[slot]:.2f}' for slot in sorted(plain)))
    print(f'x2w-bench recording: {size / 1048576:.2f} MiB, {calls} CALL chunks, {size / max(calls, 1):.0f} bytes per recorded call; '
          + ', '.join(f'{tag} {count} ({total / 1048576:.2f} MiB)' for tag, (count, total) in sorted(kinds.items(), key=lambda kv: -kv[1][1])[:6]))


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
        threads = int(env['X2W_TEST_BENCH_THREADS'])
        refused = lines(result, 'refused')
        print(bench[0])
        print(slots[0])
        target = os.environ.get('X2W_BENCH_OUT')
        if target:
            Path(target).write_text(slots[0] + '\n')
        if os.environ.get('X2W_BENCH_RECORD'):
            measure_recording(directory, consumer, env, bench[0], rounds, threads)
        print(f'startup_list_cost_bench: PASS on {adapter} (all {C.WORDS} words match; numbers above)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
