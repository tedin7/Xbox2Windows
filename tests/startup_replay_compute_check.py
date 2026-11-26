#!/usr/bin/env python3
"""A translated compute Dispatch is recorded and replayed to the same result (docs/startup-runtime.md#d3d12-replay-record).

Family check "compute dispatches": every scene of startup_compute_dispatch_check that runs one translated Dispatch of case A
is recorded with X2W_REPLAY_RECORD=all, and x2w_replay plays the recording with no title code. The bytes the replay's own
stand-in left in the output buffer (x2w_replay --dump, taken before the pages the recording holds after its last call are
written) must equal the bytes the recorded run read back. Controls:
  - a scene with no Dispatch (dispatch-none) replays to the untouched output, which differs from the dispatched result: the
    comparison sees a missing Dispatch;
  - a recording with the Dispatch's first group count changed replays to a different output (the replay really runs it).

usage: startup_replay_compute_check.py <startup_stream_consumer> <x2w_replay> <dxc> <warp|hardware>
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl as G

SCENES = ('dispatch', 'dispatch-same-root', 'dispatch-reference-2', 'dispatch-reference-5')
CALL = struct.Struct("<QIHHQ7QQdii")
DEVICE_KIND, LIST_KIND, DISPATCH_SLOT = 0, 4, 14


def words_bytes(words):
    return struct.pack('<%dI' % len(words), *words)


def record(directory, consumer, env, mode):
    recording = directory / (mode + '.x2wr')
    got, result = dispatch(directory, consumer, dict(env, X2W_REPLAY_RECORD='all', X2W_REPLAY_RECORD_FILE=str(recording),
                                  X2W_REPLAY_PAGES_AT_SUBMIT='1', X2W_REPLAY_TITLE_RANGES='1'), mode)
    found = re.search(r'x2w-test dispatch output 0x([0-9a-f]+) (\d+)', result.stdout)
    assert found, 'the consumer did not name its output buffer\n' + result.stdout[-1500:]
    return recording, got, result, '%s:%s' % (found.group(1), found.group(2))


def play(player, recording, out, adapter, dump):
    out.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('X2W_')}
    run = subprocess.run([str(player), str(recording), str(out), '--adapter', adapter, '--dump', dump], env=env,
                         capture_output=True, text=True, timeout=300)
    log = (out / 'stderr.txt').read_text(errors='replace') if (out / 'stderr.txt').exists() else ''
    assert run.returncode == 0, 'x2w_replay exit %d\n%s\n%s' % (run.returncode, run.stdout[-1500:], log[-2500:])
    return (out / 'dump.bin').read_bytes(), log


def change_first_dispatch_groups(source, target):
    """Copies the recording with the first Dispatch's x group count (its first argument) changed from 3 to 1."""
    data = bytearray(source.read_bytes())
    at = 0
    while at < len(data):
        tag, header_bytes, data_bytes = struct.unpack_from('<4sIQ', data, at)
        if tag == b'CALL':
            seq, thread, kind, slot, obj, *rest = CALL.unpack_from(data, at + 16)
            args = rest[:7]
            if kind == LIST_KIND and slot == DISPATCH_SLOT:
                where = next((j for j in range(5) if tuple(args[j:j + 3]) == (3, 2, 2)), None)
                assert where is not None, 'the Dispatch arguments are not (3, 2, 2): %s' % (args,)
                struct.pack_into('<Q', data, at + 16 + 24 + 8 * where, 1)  # the x group count
                break
        at += 16 + header_bytes + data_bytes
    else:
        raise AssertionError('no Dispatch in the recording')
    target.write_bytes(bytes(data))


def main(consumer, player, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        _, first = dispatch(directory, consumer, env, 'dispatch')
        published = lines(first, 'x2w-draw compute request')
        assert published and ' published ' in published[0], first.stderr[-2000:]
        key = int(published[0].split()[3], 16)
        done = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                              text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in done, done
        for mode in SCENES:
            recording, got, result, dump = record(directory, consumer, env, mode)
            if mode in ('dispatch', 'dispatch-same-root'):
                assert got == expected, '%s: the recorded run itself differs from the CPU reference' % mode
            dumped, log = play(player, recording, directory / ('replay_' + mode), adapter, dump)
            want = words_bytes(got)
            if dumped != want:
                first = next((i for i, (a, b) in enumerate(zip(dumped, want)) if a != b), min(len(dumped), len(want)))
                raise AssertionError("%s: the replayed output differs from the recorded run (%d bytes against %d, first difference at word %d)\n%s" % (
                    mode, len(dumped), len(want), first // 4, log[-2500:]))
            assert 'Dispatch(3, 2, 2) translated' in log or 'translated' in log, '%s: the replay did not translate the Dispatch\n%s' % (mode, log[-2500:])
            print('replay %s: output equal to the recorded run (%d words)' % (mode, C.WORDS))
        recording, got, _, dump = record(directory, consumer, env, 'dispatch-none')
        assert got == untouched
        dumped, _ = play(player, recording, directory / 'replay_none', adapter, dump)
        assert dumped == words_bytes(untouched) and dumped != words_bytes(expected), 'control: a scene with no Dispatch'
        recording, got, _, dump = record(directory, consumer, env, 'dispatch')
        changed = directory / 'changed.x2wr'
        change_first_dispatch_groups(recording, changed)
        dumped, _ = play(player, changed, directory / 'replay_changed', adapter, dump)
        assert dumped != words_bytes(got), 'control: a Dispatch with fewer groups replayed to the same output'
        print('startup_replay_compute: %d scenes replay equal; no-Dispatch and changed-group controls differ (%s)' % (len(SCENES), adapter))
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
