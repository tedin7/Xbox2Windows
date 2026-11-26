"""WARP readback after late publication of compiled code for an existing manifest."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from startup_compute_dispatch_check import C, dispatch, prepare
from startup_compute_live_retry_check import words


HLSL = """RWByteAddressBuffer output_data : register(u0);
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    output_data.Store(id.x * 4, (id.x * 137u + 29u) ^ 0x5317u);
}
"""
MANIFEST = "x2w-manifest 2\nstage cs\ngroup 32 1 1\nud 0\nuav 0 table 1 16\n"
SENTINEL = [C.SENTINEL] * C.WORDS
EXPECTED = [(i * 137 + 29) ^ 0x5317 for i in range(32)] + SENTINEL[32:]


def run(directory, consumer, dxc, enabled):
    cache, env = prepare(directory)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    # The publisher treats these request sections as opaque. The test supplies its own
    # desktop shader and manifest, so the translation step needs no guest instructions.
    (directory / 'program').write_bytes(b'late-compute-fixture')
    (directory / 'data').write_bytes(bytes(range(16)))
    (directory / 'state').write_bytes(bytes(32))
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1', X2W_NOLOCK_REFUSE='0',
               X2W_COMPUTE_PIPE_LATE='1' if enabled else '0',
               X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'))
    env.pop('X2W_TEST_DISPATCH_HARDWARE', None)

    got, _ = dispatch(directory, consumer, env, 'dispatch-none')
    assert got == SENTINEL
    request, = cache.glob('req_*.c67')
    key = request.stem.removeprefix('req_')
    if enabled:
        got, refused = dispatch(directory, consumer, env, 'dispatch-no-pipeline')
        assert got == SENTINEL
        assert 'Dispatch(1, 1, 1) refused: requires a desktop compute pipeline and root' in refused.stderr
        assert 'missing_pipeline=1 ' in refused.stderr, refused.stderr[-2500:]
        failure = cache / f'{key}.fail'
        for reason in ('translation_pending', 'offline_translation_refused'):
            if reason == 'offline_translation_refused':
                failure.write_text('synthetic offline refusal\n', encoding='ascii')
            got, refused = dispatch(directory, consumer, env, 'dispatch')
            assert got == SENTINEL
            assert f'{reason}=1 ' in refused.stderr, refused.stderr[-2500:]
        failure.unlink()
    (cache / f'{key}.txt').write_text(MANIFEST, encoding='ascii')
    code = cache / f'{key}.cs.cso'
    hlsl, staged = directory / 'compute.hlsl', directory / 'staged.cso'
    hlsl.write_text(HLSL, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', staged, hlsl],
                   check=True, timeout=20)

    output = directory / 'late.bin'
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', 'dispatch-pipe-late']
    stdout, stderr = directory / 'stdout', directory / 'stderr'
    with stdout.open('wb') as out, stderr.open('wb') as err, subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=out, stderr=err,
            env=dict(env, X2W_TEST_DISPATCH_OUT=str(output))) as process:
        try:
            deadline = time.monotonic() + 20
            while 'first slot 67 Dispatch fenced' not in stdout.read_text():
                if process.poll() == 77:
                    print(stdout.read_text(), end='')
                    raise SystemExit(77)
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError(f'first Dispatch did not fence\n{stderr.read_text()[-2500:]}')
                time.sleep(0.025)
            assert not code.exists()
            assert words(output) == SENTINEL, 'first Dispatch wrote before code publication'
            output.unlink()
            staged.replace(code)
            # The runtime uses the same bounded retry interval as manifest lookup.
            time.sleep(0.5)
            process.communicate(input=b'\n', timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    log = stderr.read_text()
    assert process.returncode == 0, log[-3000:]
    assert log.count('Dispatch(1, 1, 1) refused: no compiled program') == (1 if enabled else 2), log[-3000:]
    got = words(output)
    assert got == (EXPECTED if enabled else SENTINEL), 'second fenced readback differs'
    assert (got == EXPECTED) == enabled, 'switch-off control must fail the CPU comparison'
    reports = [line for line in log.splitlines() if 'compute dispatch reasons:' in line]
    if enabled:
        assert len(reports) == 2, reports
        assert all('compiled_shader_missing=1 ' in line for line in reports), reports
        assert reports[0].endswith('recovered_pipeline=0'), reports
        assert reports[1].endswith('recovered_pipeline=1'), reports
    else:
        assert not reports, reports
    errors = [line for line in log.splitlines() if 'x2w-d3d12-debug ' in line and (' sev=0 ' in line or ' sev=1 ' in line)]
    assert not errors, errors
    print(f'WARP late compute pipeline, switch={int(enabled)}: both {C.WORDS}-word fenced readbacks matched')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: startup_compute_pipe_late_check.py <consumer> <dxc>')
    for enabled in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp), *sys.argv[1:], enabled)
