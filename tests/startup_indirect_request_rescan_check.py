"""Run the real watcher scheduler against one late request from a WARP draw.

The worker publishes source-built fixture bytecode in place of lowering the
opaque synthetic guest identifiers. The runtime uses its usual manifest and
PSO path. The control must reach the pixel comparison and fail there.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402


def check(executable, control):
    with tempfile.TemporaryDirectory(prefix='x2w-rescan-') as directory:
        work = Path(directory)
        cache = work / 'requests'
        cache.mkdir()
        with (work / 'fixture.log').open('w+', encoding='utf-8', errors='replace') as log:
            child = None

            def scan(real_scan, *args, **kwargs):
                nonlocal child
                snapshot = real_scan(*args, **kwargs)
                if child is None:
                    if snapshot:
                        raise AssertionError('initial scan must be empty')
                    child = subprocess.Popen([executable, 'warp', 'translated-late'],
                                             env=dict(os.environ, X2W_INDIRECT_RESCAN_DIR=str(work)),
                                             stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 20
                    while not (work / 'first-draw-finished').is_file():
                        if child.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError('fixture did not finish the first untranslated draw')
                        time.sleep(0.01)
                    if len(list(cache.glob('req_*.a'))) != 1:
                        raise AssertionError('runtime did not publish exactly one late request')
                return snapshot

            def publish(out_dir, key):
                stem = f'{key:016x}'
                for ext in ('.vs.cso', '.ps.cso', '.txt'):
                    name = stem + ext
                    (work / 'compiled' / name).replace(Path(out_dir) / name)
                return 'ok'

            try:
                with patch.dict(os.environ, {'X2W_REQUEST_RESCAN': '0' if control else '1'}), \
                     patch.object(gfx2hlsl, 'translate_request', side_effect=publish):
                    # Inject the publication at the boundary of the initial scan without changing its result.
                    real_scan = gfx2hlsl.pending_keys
                    with patch.object(gfx2hlsl, 'pending_keys', side_effect=lambda *a, **kw: scan(real_scan, *a, **kw)):
                        gfx2hlsl.watch_pass(str(cache), True, jobs=1)
                if bool(list(cache.glob('*.txt'))) == control:
                    raise AssertionError('unexpected translation publication')
                (work / 'worker-finished').touch()
                status = child.wait(timeout=20)
            finally:
                if child is not None and child.poll() is None:
                    child.kill()
                    child.wait()
                log.seek(0)
                output = log.read()
                print(output, end='')
            expected = 3 if control else 0
            if status != expected or ('late translation readback: pixel (' in output) != control:
                raise AssertionError(f'expected fixture exit {expected}, got {status}')


if __name__ == '__main__':
    check(str(Path(sys.argv[1]).resolve()), sys.argv[2] == 'control')
