"""gfx2hlsl.py pending scan and --watch on synthetic request files (pure Python, no Windows, no GPU).

A request is consumed only once its ready marker (.a or .rtv) exists and it is not finished: a
published .txt with both .cso files (or a .txt under X2W_FXC='') is done; a .txt without bytecode
is resumed. After the watcher's first pass, a request whose .fail is newer than its marker is not
retried; the first pass and the one-shot --pending scan retry it. The --watch process must retry an
earlier failure once, translate a new request while it runs and not rewrite a failure on later
passes. A --pending scan with four parallel workers must print the same results and write the same
failures as one worker, an unexpected translator error must become <key>.fail, a broken process
pool must end one watcher pass and not the watcher, and an invalid X2W_TRANSLATE_JOBS must fall
back to the processor count.
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402

failures = []


def expect(label, got, want):
    if got != want:
        failures.append('%s: got %s, want %s' % (label, got, want))


def touch(path, data=b'', mtime=None):
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def scan(directory, skip_failed):
    return gfx2hlsl.pending_keys(str(directory), skip_failed)


def pending_run(directory, jobs):
    env = dict(os.environ, X2W_TRANSLATE_JOBS=str(jobs))
    out = subprocess.run([sys.executable, '-B', str(ROOT / 'tools' / 'startup' / 'gfx2hlsl.py'), '--pending', str(directory)],
                         capture_output=True, text=True, timeout=120, env=env)
    def local(text):  # the temporary directory differs between runs; keep the file name only
        return re.sub(r"'[^']*?(req_[0-9a-f]{16}\.\w+)'", r"'\1'", text)
    fails = {p.name: local(p.read_text()) for p in Path(directory).glob('*.fail')}
    return out.returncode, sorted(local(line) for line in out.stdout.splitlines()), fails


with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    now = time.time()
    # 1: complete request; 2: already published with bytecode; 3: no ready marker yet (negative
    # control); 4: failed after its marker was written; 5: marker rewritten after the failure;
    # 6: RTV variant; 9: manifest-only publication that still needs bytecode (#2008 resume).
    for key in (1, 2, 4, 5, 9):
        for ext in ('.b', '.root'):
            touch(d / ('req_%016x%s' % (key, ext)))
    touch(d / ('req_%016x.a' % 1))
    touch(d / ('req_%016x.a' % 2))
    touch(d / ('%016x.txt' % 2))
    touch(d / ('%016x.vs.cso' % 2), b'DXBC')
    touch(d / ('%016x.ps.cso' % 2), b'DXBC')
    touch(d / ('req_%016x.b' % 3))
    touch(d / ('req_%016x.root' % 3))
    touch(d / ('req_%016x.a' % 4), mtime=now - 10)
    touch(d / ('%016x.fail' % 4), mtime=now - 5)
    touch(d / ('%016x.fail' % 5), mtime=now - 10)
    touch(d / ('req_%016x.a' % 5), mtime=now - 5)
    touch(d / ('req_%016x.rtv' % 6))
    touch(d / ('req_%016x.a' % 9))
    touch(d / ('%016x.txt' % 9))
    expect('watch pass', scan(d, True), [1, 5, 6, 9])
    expect('one-shot pass', scan(d, False), [1, 4, 5, 6, 9])

def mesh57_fixture(ms, ps, expected, tamper=False):
    root = (b'DXBC' + bytes(24) + struct.pack('<II', 1, 36) + b'RTS0' +
            struct.pack('<I6I', 24, 1, 0, 24, 0, 24, 0))
    key = gfx2hlsl.mesh57_key(ms, ps, root)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        source = d / ('req_%016x' % key)
        touch(source.with_suffix('.root'), root)
        touch(source.with_suffix('.ps57'), ps)
        expect('mesh request needs a ready marker', scan(d, False), [])
        touch(source.with_suffix('.m57'), ms + (bytes(4) if tamper else b''))
        expect('mesh ready marker is watched', scan(d, False), [key])
        expect('mesh requests stay out of VS/PS enumeration', list(gfx2hlsl.request_keys(d)), [])
        expect('mesh reader: ' + expected, gfx2hlsl.translate_request(d, key), 'unsupported: ' + expected)
        expect('mesh failure is read once per watch pass', scan(d, True), [])
        expect('no vertex or mesh bytecode is published', list(d.glob('*.cso')), [])


# RDNA2 synthetic descriptor programs: allocate geometry, export position and primitive, and terminate.
position = (0xf80008cf, 0x03020100)
primitive = (0xf8000941, 0)
endpgm = 0xbf810000
mesh = struct.pack('<6I', 0xbf900009, *position, *primitive, endpgm)
pixel = struct.pack('<3I', 0xf800080f, 0x03020100, endpgm)
vertex = struct.pack('<3I', *position, endpgm)
mesh57_fixture(mesh, pixel, 'slot 57 mesh code emission is unavailable')
mesh57_fixture(mesh, b'', 'slot 57 mesh code emission is unavailable')
mesh57_fixture(vertex, pixel, 'slot 57 mesh stage has no verified allocation and primitive export')
mesh57_fixture(mesh, vertex, 'slot 57 pixel stage has an incompatible export or termination')
mesh57_fixture(mesh, pixel, 'slot 57 mesh request key mismatch', tamper=True)

# Parallel --pending must give the same results as one worker. Sixteen RTV variants each name a
# different missing base request, so each failure names its own base and a result reported against
# the wrong key is visible. With four workers each batch holds one request.
bases = {}
with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as many:
    for n in range(16):
        request = b'rtv1 %d 28' % (0x7000 + n)
        bases[gfx2hlsl.fnv(request)] = 0x7000 + n
        for d in (one, many):
            touch(Path(d) / ('req_%016x.rtv' % gfx2hlsl.fnv(request)), request)
    serial, parallel = pending_run(one, 1), pending_run(many, 4)
    expect('parallel --pending output and failures', parallel, serial)
    expect('every request answered', len(serial[1]), 16)
    for line in parallel[1]:
        key = int(line[:16], 16)
        expect('result for %016x names its own base' % key, 'req_%016x.a' % bases.get(key, -1) in line, True)

# An unexpected translator error becomes <key>.fail instead of ending the scan.
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    touch(d / ('req_%016x.rtv' % 9))
    real = gfx2hlsl.translate_request
    gfx2hlsl.translate_request = lambda out_dir, key: 1 / 0
    try:
        gfx2hlsl.translate_pending(str(d), jobs=1)
    except ZeroDivisionError:
        failures.append('a translator error escaped translate_pending')
    finally:
        gfx2hlsl.translate_request = real
    fail = d / ('%016x.fail' % 9)
    expect('translator error recorded', fail.read_text().startswith('translator error: ZeroDivisionError') if fail.exists() else None, True)
    # It names the function, file and line that raised it (the lambda above).
    expect('translator error located', bool(re.fullmatch(
        r'translator error: ZeroDivisionError: division by zero \(in <lambda>, %s line \d+\)\n'
        % re.escape(Path(__file__).name), fail.read_text())) if fail.exists() else None, True)

# An error raised inside a standard-library call is located at the code that made the call, not in the library.
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    touch(d / ('req_%016x.rtv' % 9))
    real = gfx2hlsl.translate_request
    gfx2hlsl.translate_request = lambda out_dir, key: json.loads('{')
    try:
        gfx2hlsl.translate_pending(str(d), jobs=1)
    finally:
        gfx2hlsl.translate_request = real
    fail = d / ('%016x.fail' % 9)
    expect('library error located at its caller', bool(re.fullmatch(
        r'translator error: JSONDecodeError: .* \(in <lambda>, %s line \d+\)\n'
        % re.escape(Path(__file__).name), fail.read_text())) if fail.exists() else None, True)

# An operand refusal's .fail names the instruction that refused it; the refusal message itself is unchanged.
sys.path.insert(0, str(ROOT / 'tests'))
import startup_gfx2hlsl_dpp_quad_swap_check as quad  # noqa: E402

AND_NEG = struct.pack('<III', (0b110101 << 26) | ((0x100 + 0x1b) << 16) | 1,  # v_and_b32_e64 v1, v30, -v30
                      (256 + 30) | ((256 + 30) << 9) | (1 << 30), 0xbf810000)  # s_endpgm
with tempfile.TemporaryDirectory() as tmp:
    result = gfx2hlsl.translate_to(tmp, 3, quad.VS_WORDS, AND_NEG, quad.root_blob(), check=False)
    expect('operand refusal result', result, "unsupported: operand '-v30' in v_and_b32_e64")
    expect('operand refusal .fail', Path(tmp, '%016x.fail' % 3).read_text(), "operand '-v30' in v_and_b32_e64\n")
    try:
        gfx2hlsl.Pipeline(quad.VS_WORDS, AND_NEG, quad.root_blob())
    except gfx2hlsl.Unsupported as error:
        expect('the refusal message is unchanged', str(error), "operand '-v30'")
    else:
        failures.append('the negated v_and_b32 operand was not refused')

# A broken process pool (a worker killed from outside) ends one pass, not the watcher.
from concurrent.futures.process import BrokenProcessPool  # noqa: E402


def broken(out_dir, skip_failed=False):
    raise BrokenProcessPool('a worker was terminated abruptly')


real = gfx2hlsl.translate_pending
gfx2hlsl.translate_pending = broken
try:
    expect('broken pool: pass reported as incomplete', gfx2hlsl.watch_pass('.', True), False)
except BrokenProcessPool:
    failures.append('a broken process pool escaped watch_pass')
finally:
    gfx2hlsl.translate_pending = real
with tempfile.TemporaryDirectory() as tmp:
    expect('normal pass reported as complete', gfx2hlsl.watch_pass(tmp, True), True)

# An invalid job count falls back to the processor count instead of raising.
saved = os.environ.get('X2W_TRANSLATE_JOBS')
for value in ('', 'abc', '-3'):
    os.environ['X2W_TRANSLATE_JOBS'] = value
    expect('X2W_TRANSLATE_JOBS=%r' % value, gfx2hlsl.translation_jobs() >= 1, True)
os.environ['X2W_TRANSLATE_JOBS'] = '2'
expect('X2W_TRANSLATE_JOBS=2', gfx2hlsl.translation_jobs(), 2)
if saved is None:
    del os.environ['X2W_TRANSLATE_JOBS']
else:
    os.environ['X2W_TRANSLATE_JOBS'] = saved

with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    # A failure from an earlier run, newer than its marker: the watcher's first pass retries it once.
    now = time.time()
    touch(d / ('req_%016x.rtv' % 8), b'not an rtv request either', mtime=now - 20)
    touch(d / ('%016x.fail' % 8), b'stale reason\n', mtime=now - 10)
    watch = subprocess.Popen([sys.executable, '-B', str(ROOT / 'tools' / 'startup' / 'gfx2hlsl.py'),
                              '--watch', str(d), '0.1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(0.5)
        # Written while the watcher runs; an invalid RTV request fails without a GPU or compiler.
        touch(d / ('req_%016x.rtv' % 7), b'not an rtv request')
        fail = d / ('%016x.fail' % 7)
        deadline = time.time() + 20
        while not fail.exists() and time.time() < deadline and watch.poll() is None:
            time.sleep(0.1)
        expect('watcher wrote a failure for the new request', fail.exists(), True)
        first = fail.stat().st_mtime_ns if fail.exists() else None
        time.sleep(1.0)
        expect('failure left unchanged by later passes', fail.stat().st_mtime_ns if fail.exists() else None, first)
        old = (d / ('%016x.fail' % 8)).read_text().strip()
        expect('earlier failure retried once at start', old, 'invalid RTV variant request')
        expect('watcher still running', watch.poll(), None)
    finally:
        watch.kill()
        out, err = watch.communicate(timeout=20)
    lines = sorted(line for line in out.splitlines() if line.strip())
    expect('watcher output (each request translated once)', [line[:16] for line in lines], ['%016x' % 7, '%016x' % 8])
    if err.strip():
        failures.append('watcher stderr: %r' % err.strip()[-500:])

for item in failures:
    print('FAIL', item)
if failures:
    sys.exit(1)
print('startup_gfx2hlsl_watch_check passed')
