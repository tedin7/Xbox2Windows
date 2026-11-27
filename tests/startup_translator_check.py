"""Checks for the startup shader translator (tools/startup/gfx2hlsl.py) that need no game data.

Descriptor offsets: one s_load_dwordx16 from a descriptor table fetches two 8-dword image descriptors.
An image_sample that names the upper one (s[N+8:N+15]) must resolve to byte offset 32 of the table,
not to the start of the load; otherwise both samples bind the same texture.

fxc discovery: a translation asked to compile (check) with no fxc.exe anywhere must write <key>.fail
naming X2W_FXC and no manifest; with the empty X2W_FXC or check=False it publishes the manifest alone.
"""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl  # noqa: E402


def stage_with_table():
    stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)  # no machine code: only the scalar symbol state
    stage.sym = {8: ('ptr', ('table', 0))}
    return stage


def check_wide_load(failures):
    stage = stage_with_table()
    stage.op_s_load_dwordx16(['s[24:39]', 's[8:9]', '0x0'], [])
    got = [stage.descriptor('s[24:31]'), stage.descriptor('s[32:39]')]
    want = [(('table', 0), 0), (('table', 0), 32)]
    if got != want:
        failures.append('s_load_dwordx16 descriptors: got %s, expected %s' % (got, want))


def check_offset_load(failures):
    stage = stage_with_table()
    stage.op_s_load_dwordx8(['s[16:23]', 's[8:9]', '0x40'], [])
    got = [stage.descriptor('s[16:19]'), stage.descriptor('s[20:23]')]
    want = [(('table', 0), 0x40), (('table', 0), 0x50)]
    if got != want:
        failures.append('s_load_dwordx8 at +0x40: got %s, expected %s' % (got, want))


def check_root_descriptor(failures):
    # A descriptor that comes from a root parameter rather than a table has no byte offset; the
    # resource is the parameter itself. Both halves matter: a key that reports 0 instead of None
    # would look like the first entry of a table.
    stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)
    stage.sym = {8: ('ptr', ('root', 2))}
    got = stage.descriptor('s[8:11]')
    want = (('root', 2), None)
    if got != want:
        failures.append('root descriptor: got %s, expected %s' % (got, want))


def check_numeric_symbol_refusal(failures):
    # A register that holds a descriptor or a buffer pointer has no 32-bit value: reading it as a
    # number must be refused rather than silently emitted as s<n>.
    for shape in (('ptr', ('table', 0)), ('desc', ('table', 0), 0, 0)):
        stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)
        stage.sym = {8: shape}
        stage.used_s = set()
        try:
            got = stage.S(8)
        except gfx2hlsl.Unsupported as error:
            if 'numeric use of s8' not in str(error):
                failures.append('numeric symbol %r: unexpected message %s' % (shape, error))
        else:
            failures.append('numeric symbol %r was read as %s' % (shape, got))
        if stage.used_s:
            failures.append('numeric symbol %r marked s8 as used' % (shape,))


def check_negative_control(failures):
    # A key that ignores the register's dword index (the former behaviour) gives the second
    # descriptor offset 0, the same resource as the first: the check above must tell them apart.
    stage = stage_with_table()
    stage.op_s_load_dwordx16(['s[24:39]', 's[8:9]', '0x0'], [])
    first, second = stage.sym[24], stage.sym[32]
    if (first[1], first[2]) != (second[1], second[2]):
        failures.append('negative control: the load no longer records a shared base offset')
    if stage.descriptor('s[32:39]') == (first[1], first[2]):
        failures.append('negative control: the upper descriptor resolves to the load base')


class StubPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def hlsl_vs(self):
        return 'void main() {}\n'

    hlsl_ps = hlsl_vs

    def manifest(self):
        return 'manifest\n'

    def ps_profile(self):
        return 'ps_5_1'


@contextlib.contextmanager
def fxc_environment(work, configured, kits, path=''):
    """X2W_FXC, the Windows Kits bin directory and PATH pointed at temporary locations."""
    saved = (dict(os.environ), gfx2hlsl.DEFAULT_FXC, gfx2hlsl.KITS_BIN, gfx2hlsl.Pipeline)
    os.environ.pop('X2W_FXC', None)
    if configured is not None:
        os.environ['X2W_FXC'] = configured
    os.environ['PATH'] = path
    gfx2hlsl.DEFAULT_FXC = os.path.join(work, 'no-default', 'fxc.exe')
    gfx2hlsl.KITS_BIN = kits
    gfx2hlsl.Pipeline = StubPipeline
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved[0])
        gfx2hlsl.DEFAULT_FXC, gfx2hlsl.KITS_BIN, gfx2hlsl.Pipeline = saved[1:]


def stub_fxc(root, version):
    directory = os.path.join(root, version, 'x64')
    os.makedirs(directory)
    path = os.path.join(directory, 'fxc.exe')
    open(path, 'w').close()
    return path


def translate(work, check=True):
    out = os.path.join(work, 'out')
    os.makedirs(out, exist_ok=True)
    result = gfx2hlsl.translate_to(out, 0x1234, b'', b'', b'', check=check)
    return result, sorted(os.listdir(out))


def check_missing_fxc(failures):
    with tempfile.TemporaryDirectory() as work, fxc_environment(work, os.path.join(work, 'missing.exe'), os.path.join(work, 'none')):
        result, files = translate(work)
        if '0000000000001234.fail' not in files or '0000000000001234.txt' in files or 'X2W_FXC' not in result:
            failures.append('missing fxc: result %r, files %s' % (result, files))
        fail = os.path.join(work, 'out', '0000000000001234.fail')
        if os.path.exists(fail) and 'X2W_FXC' not in open(fail).read():
            failures.append('missing fxc: the .fail does not name X2W_FXC')
    with tempfile.TemporaryDirectory() as work, fxc_environment(work, None, os.path.join(work, 'none')):
        result, files = translate(work, check=False)
        if files != ['0000000000001234.ps.hlsl', '0000000000001234.txt', '0000000000001234.vs.hlsl'] or result != 'ok':
            failures.append('check=False without fxc: result %r, files %s' % (result, files))
    with tempfile.TemporaryDirectory() as work, fxc_environment(work, '', os.path.join(work, 'none')):
        result, files = translate(work)
        if '0000000000001234.txt' not in files or '0000000000001234.fail' in files:
            failures.append('empty X2W_FXC: result %r, files %s' % (result, files))


def check_fxc_discovery(failures):
    with tempfile.TemporaryDirectory() as work:
        kits = os.path.join(work, 'Windows Kits', '10', 'bin')
        stub_fxc(kits, '10.0.22621.0')
        newest = stub_fxc(kits, '10.0.26200.0')
        stub_fxc(kits, '10.0.9.0')  # a string sort would rank 10.0.9.0 above 10.0.26200.0
        os.makedirs(os.path.join(kits, 'x86'))  # not a version directory
        on_path = os.path.join(work, 'path')
        os.makedirs(on_path)
        open(os.path.join(on_path, 'fxc.exe'), 'w').close()
        os.chmod(os.path.join(on_path, 'fxc.exe'), 0o755)  # a PATH search outside Windows needs the execute bit
        explicit = os.path.join(work, 'explicit.exe')
        open(explicit, 'w').close()
        for configured, path, want, label in ((None, '', newest, 'newest Windows Kits version'),
                                              (explicit, '', explicit, 'X2W_FXC'),
                                              (os.path.join(work, 'missing.exe'), '', newest, 'missing X2W_FXC falls back'),
                                              ('', '', '', 'empty X2W_FXC')):
            with fxc_environment(work, configured, kits, path):
                got = gfx2hlsl.find_fxc()
                if got != want:
                    failures.append('find_fxc (%s): got %r, expected %r' % (label, got, want))
        with fxc_environment(work, None, os.path.join(work, 'none'), on_path):
            got = gfx2hlsl.find_fxc()
            if not got or os.path.normcase(got) != os.path.normcase(os.path.join(on_path, 'fxc.exe')):
                failures.append('find_fxc (PATH): got %r' % (got,))


def check_negative_discovery(failures):
    # Control: the Windows Kits tree above holds a stub, so a search that ignores KITS_BIN must be caught
    # by check_fxc_discovery, and one that ignores the missing-fxc branch by check_missing_fxc. Here the
    # stub tree is present and the same translation must no longer fail.
    with tempfile.TemporaryDirectory() as work:
        kits = os.path.join(work, 'bin')
        stub_fxc(kits, '10.0.1.0')
        with fxc_environment(work, None, kits):
            if gfx2hlsl.find_fxc() is None:
                failures.append('negative control: a Windows Kits tree with fxc.exe was not found')


def check_pending_manifest_upgrade(failures):
    with tempfile.TemporaryDirectory() as work:
        out = Path(work)
        key = '0000000000001234'
        (out / ('req_%s.a' % key)).touch()
        manifest = out / (key + '.txt')
        failure = out / (key + '.fail')
        vs, ps = (out / (key + '.' + stage + '.cso') for stage in ('vs', 'ps'))
        runs = []
        fail_pixel = True

        def compile_shader(command, **kwargs):
            profile = command[command.index('/T') + 1]
            runs.append(profile)
            Path(command[command.index('/Fo') + 1]).write_bytes(profile.encode('ascii'))
            if fail_pixel and profile == 'ps_5_1':
                return subprocess.CompletedProcess(command, 1, '', 'synthetic compiler failure')
            return subprocess.CompletedProcess(command, 0, '', '')

        def pending(configured, retry=False):
            request = ({'vs': b'', 'ps': b''}, b'', (), '', None, {}, None, None, None)
            retry_env = {'X2W_RETRY_FAILED': '1'} if retry else {}
            with fxc_environment(work, configured, os.path.join(work, 'none')), \
                 mock.patch.dict(os.environ, retry_env), \
                 mock.patch.object(sys, 'argv', ['gfx2hlsl.py', '--pending', work]), \
                 mock.patch.object(gfx2hlsl, 'load_request', return_value=request), \
                 mock.patch.object(subprocess, 'run', side_effect=compile_shader), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                gfx2hlsl.main()
            return output.getvalue()

        if pending('') != '%s: ok\n' % key or not manifest.is_file() or vs.exists() or ps.exists():
            failures.append('pending manifest-only pass did not publish just the manifest and HLSL')
        if pending('') or runs:
            failures.append('pending manifest-only pass did not skip an existing manifest')

        fxc = out / 'fxc.exe'
        fxc.touch()
        if 'fxc ps failed' not in pending(str(fxc)) or runs != ['vs_5_1', 'ps_5_1']:
            failures.append('pending did not compile an existing manifest when FXC became available')
        if vs.exists() or ps.exists():
            failures.append('a failed compilation published partial shader bytecode')
        if not failure.is_file():
            failures.append('a failed compilation did not write a failure report')

        # A refusal the same translator build wrote is final: --pending leaves it alone unless X2W_RETRY_FAILED=1.
        runs.clear()
        if pending(str(fxc)) or runs or not failure.is_file():
            failures.append('pending retried a refusal the same translator build wrote')
        fail_pixel = False
        if pending(str(fxc), retry=True) != '%s: ok\n' % key or not all(path.is_file() for path in (vs, ps)):
            failures.append('pending did not retry and publish both shader stages')
            return
        if (vs.read_bytes(), ps.read_bytes()) != (b'vs_5_1', b'ps_5_1'):
            failures.append('pending did not publish the expected bytecode for both stages')
        if failure.exists():
            failures.append('pending left a stale failure report after successful compilation')
        runs.clear()
        if pending(str(fxc)) or runs:
            failures.append('pending recompiled an already complete translation')
        ps.write_bytes(b'')
        if pending(str(fxc)) != '%s: ok\n' % key or ps.read_bytes() != b'ps_5_1':
            failures.append('pending did not repair an empty shader bytecode file')

        replace = os.replace

        def reject_manifest_publish(source, destination):
            if Path(destination) == manifest:
                raise PermissionError('synthetic manifest publication failure')
            replace(source, destination)

        ps.unlink()
        with mock.patch.object(StubPipeline, 'manifest', return_value='updated bindings\n'):
            with mock.patch.object(os, 'replace', side_effect=reject_manifest_publish):
                # translate_one records unexpected I/O errors as <key>.fail instead of ending the scan.
                result = pending(str(fxc))
                if 'translator error: PermissionError' not in result:
                    failures.append('pending did not attempt to publish the changed manifest')
            if manifest.exists():
                failures.append('failed publication left an old manifest beside the new bytecode')
            if pending(str(fxc), retry=True) != '%s: ok\n' % key or manifest.read_text() != 'updated bindings\n':
                failures.append('pending did not retry an interrupted manifest publication')


def main():
    failures = []
    for check in (check_wide_load, check_offset_load, check_root_descriptor, check_numeric_symbol_refusal,
                  check_negative_control, check_missing_fxc,
                  check_fxc_discovery, check_negative_discovery, check_pending_manifest_upgrade):
        check(failures)
    for line in failures:
        print('FAIL', line)
    print('startup translator checks: %d failed' % len(failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
