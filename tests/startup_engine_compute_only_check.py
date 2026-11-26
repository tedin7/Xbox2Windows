"""X2W_ENGINE=compute: the engine translates compute requests only.

tools/startup/gfx2hlsl.py hands a slot 67 compute request to tools/startup/gfx2hlsl_engine.py when X2W_ENGINE=compute and to
its own translator otherwise (unset, 0, 1 hands the whole driver over in main() instead, see startup_engine_hook_check.py).
Graphics and mesh requests never reach the engine in this mode. Run with a stand-in engine module so the routing is checked
without a compute fixture; startup_engine_compute_check.py proves the engine's own compute output.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

FAILED = []


def expect(name, got, want):
    if got != want:
        FAILED.append(name)
        print('FAIL %s: got %r, want %r' % (name, got, want))


calls = []
fake = types.ModuleType('gfx2hlsl_engine')
fake.translate_compute = lambda out_dir, key: calls.append(('compute', key)) or 'engine compute'
fake.translate_mesh57 = lambda out_dir, key: calls.append(('mesh', key)) or 'engine mesh'
sys.modules['gfx2hlsl_engine'] = fake

with tempfile.TemporaryDirectory() as tmp:
    # No request files: the default translator refuses an unreadable request by itself (no engine involved).
    for value in (None, '', '0', '1', 'computer', 'Compute'):
        calls.clear()
        if value is None:
            os.environ.pop('X2W_ENGINE', None)
        else:
            os.environ['X2W_ENGINE'] = value
        result = G.translate_compute(tmp, 0x1234)
        expect('X2W_ENGINE=%r keeps the default compute translator' % value, (calls, result.startswith('unsupported')), ([], True))
    os.environ['X2W_ENGINE'] = 'compute'
    calls.clear()
    expect('compute-only hands a compute request to the engine', (G.translate_compute(tmp, 0x1234), calls), ('engine compute', [('compute', 0x1234)]))
    # A graphics or mesh key never reaches the engine in this mode.
    calls.clear()
    result = G.translate_request(tmp, 0x5678)
    expect('graphics request is not handed to the engine', ('engine compute' in str(result), calls), (False, []))
    expect('engine_compute_only is true only for the word compute', G.engine_compute_only(), True)
    os.environ['X2W_ENGINE'] = '1'
    expect('engine_compute_only is false for 1', G.engine_compute_only(), False)

if FAILED:
    print('FAILED: %d' % len(FAILED))
    sys.exit(1)
print('startup_engine_compute_only: PASS (compute goes to the engine only for X2W_ENGINE=compute; graphics and mesh stay on the default)')
