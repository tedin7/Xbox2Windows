"""Exercise late publication through the same device slot 67 pipeline."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

from startup_compute_dispatch_check import C, G, prepare_case


def words(output):
    return list(struct.unpack(f'<{C.WORDS}I', output.read_bytes()))


def run(directory, consumer, dxc, adapter, publish, nolock=False):
    cache, env = prepare_case(directory, adapter)
    env['X2W_NOLOCK_REFUSE'] = '1' if nolock else '0'
    output = directory / 'out.bin'
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state',
            'dispatch-nolock-retry' if nolock else 'dispatch-retry']
    stdout, stderr = directory / 'stdout', directory / 'stderr'
    sentinel = [C.SENTINEL] * C.WORDS
    with stdout.open('wb') as out, stderr.open('wb') as err, subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=out, stderr=err, env=dict(env, X2W_TEST_DISPATCH_OUT=str(output))) as process:
        try:
            deadline = time.monotonic() + 20
            while 'first slot 67 Dispatch fenced' not in stdout.read_text():
                if process.poll() == 77:
                    print(stdout.read_text(), end='')
                    raise SystemExit(77)
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError(f'first Dispatch did not fence\n{stderr.read_text()[-2500:]}')
                time.sleep(0.025)
            first = time.monotonic()
            if nolock:
                expected = 'manifest lookups: repeated reads 0, draw_lock holds +0, untranslated +10000'
                assert expected in stdout.read_text(), stdout.read_text()
            assert words(output) == sentinel, 'first Dispatch wrote before translation'
            output.unlink()
            request, = cache.glob('req_*.c67')
            key = request.stem.removeprefix('req_')
            assert not (cache / (key + '.txt')).exists()
            if publish:
                result = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                        capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
                assert f'{key}: ok' in result.stdout, result.stdout
            # The no-lock fixture waits for the worker's publication counter.
            if not nolock:
                time.sleep(max(0, first + 1.1 - time.monotonic()))
            process.communicate(input=b'\n', timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    log = stderr.read_text()
    assert process.returncode == 0, log[-2500:]
    if nolock:
        assert log.count('Dispatch(3, 2, 2) translated:') == 2, log[-2500:]
    else:
        assert log.count('Dispatch(3, 2, 2) refused: no translation') == (1 if publish else 2), log[-2500:]
        assert log.count('Dispatch(3, 2, 2) translated:') == int(publish), log[-2500:]
    assert words(output) == (C.expected('A') if publish else sentinel)
    errors = [line for line in log.splitlines() if 'x2w-d3d12-debug ' in line and (' sev=0 ' in line or ' sev=1 ' in line)]
    assert not errors, errors
    print(stdout.read_text(), end='')
    print(f'{key}: both fenced {C.WORDS}-word readbacks match, late publication={publish}')


if __name__ == '__main__':
    nolock = sys.argv[3:] == ['warp', '--nolock']
    if not nolock and (len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware')):
        sys.exit('usage: startup_compute_live_retry_check.py <consumer> <dxc> <warp|hardware> [--nolock for warp]')
    for publish in ((True,) if nolock else (True, False)):
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp), *sys.argv[1:4], publish, nolock=nolock)
