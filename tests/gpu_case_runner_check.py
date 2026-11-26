"""tests/gpu_case_runner.py: the served cases answer like one-shot calls, and every failure of the server falls back to one.

The renderer here is a fake offscreen_check (a Python script behind a one-line launcher) that speaks the `--serve` protocol of
tests/offscreen_check.cpp: a banner closed by the marker pair of case -1, then per case the case text, "X2WEND <i>" on stderr and
"X2WCASE <i> <exit code>" on stdout. Its one-shot mode prints the same text, so the two paths can be compared.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_case_runner as R  # noqa: E402

FAKE = r'''
import os, sys
mode = sys.argv[1].split("=", 1)[1]
adapter, args = sys.argv[2], sys.argv[3:]

def one(args):
    """The text and exit code of one case, as the real tool prints them."""
    op = args[0]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.log"), "a") as log:
        log.write(" ".join(args) + "\n")
    if op == "--pass":
        return "ok %s\ndebug_errors=0 debug_warnings=0\n" % args[1], "", 0
    if op == "--warn":
        return "ok %s\ndebug_errors=0 debug_warnings=1\n" % args[1], "", 0
    if op == "--mismatch":
        return "", "startup_flat_param mismatch %s\n" % args[1], 1
    if op == "--skip":
        return "skip: nothing\n", "", 77
    return "", "Unknown GPU operation or invalid arguments: %s\n" % op, 2

if mode == "grandchild":          # a launcher child that keeps the pipes open after the renderer is gone
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grandchild.pid"), "w") as pid_file:
        pid_file.write(str(child.pid))
banner = "backend=%s debug_layer=enabled\nadapter=Fake\n" % adapter
if args and args[0] == "--serve":
    if mode == "no-serve":
        sys.stderr.write("Unknown GPU operation or invalid arguments: --serve\n")
        sys.exit(2)
    if mode == "no-adapter":
        print("No D3D12-capable hardware adapter visible")
        sys.exit(77)
    sys.stdout.write(banner)
    sys.stderr.write("XBOX2WINDOWS_ADAPTER: skipping Other\n")
    sys.stderr.write("X2WEND -1\n")
    print("X2WCASE -1 0", flush=True)
    for index, line in enumerate(sys.stdin):
        if line == "--end\n":
            sys.exit(0)
        out, err, code = one(line.rstrip("\n").split("\t"))
        if mode == "die-at-%d" % index:
            os._exit(3)
        if mode == "noisy-die-at-%d" % index:
            sys.stdout.write("partial output %d\n" % index)
            sys.stdout.flush()
            sys.stderr.write("renderer diagnostic %d\n" % index)
            sys.stderr.flush()
            os._exit(3)
        if mode == "hang-at-%d" % index:
            sys.stderr.write("hang diagnostic %d\n" % index)
            sys.stderr.flush()
            import time
            time.sleep(60)
        if mode == "skew-at-%d" % index:
            sys.stderr.write("X2WEND %d\n" % (index + 5))
            print("X2WCASE %d 0" % index, flush=True)
            continue
        sys.stdout.write(out)
        sys.stderr.write(err)
        sys.stderr.write("X2WEND %d\n" % index)
        print("X2WCASE %d %d" % (index, code), flush=True)
    sys.exit(0)
out, err, code = one(args)
sys.stdout.write("ONESHOT " + banner + out)
sys.stderr.write(err)
sys.exit(code)
'''


def launcher(directory, name, mode):
    """The renderer as an interpreter plus script, a direct child of the runner: killing it releases its pipes at once (a
    .cmd or shell launcher in between would leave a sleeping grandchild holding them)."""
    fake = directory / 'fake_offscreen.py'
    fake.write_text(FAKE)
    return [sys.executable, str(fake), '--mode=' + mode]


def running(pid):
    """True while the process runs: a killed child nobody reaped (PID 1 of a container) stays a zombie, which is not running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open('/proc/%d/stat' % pid) as stat:
            return stat.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        R._servers.clear()
        self.addCleanup(R._servers.clear)
        self.addCleanup(R._close_all)          # runs first: the servers of the test are closed before the table is emptied
        patch = unittest.mock.patch.object(R, 'SERVER_OFF', False)   # X2W_FLAT_SERVER=0 in the environment must not turn served tests into one-shot calls
        patch.start()
        self.addCleanup(patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_case(self, renderer, *args):
        return R.run(renderer, 'warp', args)

    def test_served_cases_carry_their_own_text_and_exit_code(self):
        renderer = launcher(self.dir, 'fake', 'serve')
        first = self.run_case(renderer, '--pass', 'a')
        second = self.run_case(renderer, '--mismatch', 'b')
        third = self.run_case(renderer, '--pass', 'c')
        self.assertEqual((first.returncode, second.returncode, third.returncode), (0, 1, 0))
        self.assertEqual(first.stdout, 'backend=warp debug_layer=enabled\nadapter=Fake\nok a\ndebug_errors=0 debug_warnings=0\n')
        self.assertEqual(first.stderr, 'XBOX2WINDOWS_ADAPTER: skipping Other\n')
        self.assertEqual(second.stderr, 'XBOX2WINDOWS_ADAPTER: skipping Other\nstartup_flat_param mismatch b\n')
        self.assertEqual(third.stdout, 'backend=warp debug_layer=enabled\nadapter=Fake\nok c\ndebug_errors=0 debug_warnings=0\n')  # banner on every case
        self.assertEqual(third.stderr, 'XBOX2WINDOWS_ADAPTER: skipping Other\n')
        self.assertEqual(self.run_case(renderer, '--bogus').returncode, 2)

    def test_skip_and_missing_adapter_keep_their_exit_code(self):
        self.assertEqual(self.run_case(launcher(self.dir, 'skip', 'serve'), '--skip').returncode, 77)
        none = self.run_case(launcher(self.dir, 'none', 'no-adapter'), '--pass', 'x')
        self.assertEqual(none.returncode, 77)
        self.assertIn('No D3D12-capable hardware adapter visible', none.stdout)

    def test_a_renderer_without_serve_falls_back_to_one_shot_calls(self):
        renderer = launcher(self.dir, 'old', 'no-serve')
        result = self.run_case(renderer, '--mismatch', 'z')
        self.assertEqual(result.returncode, 1)
        self.assertTrue(result.stdout.startswith('ONESHOT '))
        self.assertEqual(self.run_case(renderer, '--pass', 'q').returncode, 0)

    def executed(self):
        log = self.dir / 'cases.log'
        return log.read_text().splitlines() if log.exists() else []

    def test_a_crash_after_the_handshake_is_a_failed_verdict_and_the_case_is_not_run_again(self):
        renderer = launcher(self.dir, 'dies', 'die-at-1')
        self.assertEqual(self.run_case(renderer, '--pass', 'a').returncode, 0)
        lost = self.run_case(renderer, '--pass', 'b')                 # the renderer dies on this case, which would pass if re-run
        self.assertEqual(lost.returncode, 1)
        self.assertIn('exited (code 3)', lost.stderr)
        self.assertEqual(self.executed(), ['--pass a', '--pass b'])  # fake logs the case before dying; never a second time
        self.assertEqual(self.run_case(renderer, '--pass', 'c').returncode, 0)     # the next case gets a new server

    def test_a_silent_renderer_fails_the_case_after_one_deadline(self):
        renderer = launcher(self.dir, 'hangs', 'hang-at-0')
        self.addCleanup(setattr, R, 'CASE_TIMEOUT_S', R.CASE_TIMEOUT_S)
        R.CASE_TIMEOUT_S = 2
        started = time.monotonic()
        lost = self.run_case(renderer, '--pass', 'a')
        self.assertEqual(lost.returncode, 1)
        self.assertIn('timed out after 2 s', lost.stderr)
        self.assertLess(time.monotonic() - started, 4)               # one deadline for both streams, not one each
        self.assertIn('hang diagnostic 0', lost.stderr)              # what it printed before hanging survives
        self.assertIn('debug_layer=enabled', lost.stdout)            # and so does the adapter banner

    def test_a_crash_keeps_the_diagnostics_and_the_banner(self):
        lost = self.run_case(launcher(self.dir, 'noisy', 'noisy-die-at-0'), '--pass', 'a')
        self.assertEqual(lost.returncode, 1)
        self.assertIn('exited (code 3)', lost.stderr)
        self.assertIn('renderer diagnostic 0', lost.stderr)
        self.assertIn('partial output 0', lost.stdout)
        self.assertTrue(lost.stdout.startswith('backend=warp debug_layer=enabled\nadapter=Fake\n'))

    def test_a_reply_with_the_wrong_case_number_fails_the_case(self):
        lost = self.run_case(launcher(self.dir, 'skew', 'skew-at-1'), '--pass', 'a')
        self.assertEqual(lost.returncode, 0)
        lost = self.run_case(launcher(self.dir, 'skew', 'skew-at-1'), '--pass', 'b')
        self.assertEqual(lost.returncode, 1)
        self.assertIn('answered case', lost.stderr)

    def test_a_debug_warning_of_a_later_case_is_visible_to_the_check(self):
        renderer = launcher(self.dir, 'fake', 'serve')
        self.run_case(renderer, '--pass', 'a')
        later = self.run_case(renderer, '--warn', 'b')
        self.assertIn('debug_layer=enabled', later.stdout)           # the check looks for the status in every case
        self.assertIn('debug_errors=0 debug_warnings=1', later.stdout)

    def test_every_exit_leaves_no_process_and_no_open_pipe(self):
        made = []
        real = R._Server

        def record(*args):
            made.append(real(*args))
            return made[-1]

        with unittest.mock.patch.object(R, '_Server', record):
            self.run_case(launcher(self.dir, 'old', 'no-serve'), '--pass', 'a')      # failed handshake, then the one-shot call
            self.run_case(launcher(self.dir, 'none', 'no-adapter'), '--pass', 'a')   # skip 77 at the handshake
            dies = launcher(self.dir, 'dies', 'die-at-1')
            self.run_case(dies, '--pass', 'a')
            self.run_case(dies, '--mismatch', 'b')                                   # dead server: failed verdict
            self.run_case(dies, '--pass', 'c')                                       # next case, new server
            R._close_all()                                                           # that server, at exit
        self.assertEqual(len(made), 4)
        for server in made:
            self.assertIsNotNone(server.proc.poll())
            self.assertTrue(all(s.closed for s in (server.proc.stdin, server.proc.stdout, server.proc.stderr)))

    @unittest.skipUnless(os.name == 'posix', 'process groups')
    def test_a_launcher_child_holding_the_pipes_does_not_block_the_end_of_the_test(self):
        renderer = launcher(self.dir, 'grand', 'grandchild')
        self.assertEqual(self.run_case(renderer, '--pass', 'a').returncode, 0)
        pid = int((self.dir / 'grandchild.pid').read_text())
        started = time.monotonic()
        R._close_all()                                               # the renderer is gone but its child holds stdout and stderr
        self.assertLess(time.monotonic() - started, 20)
        for _ in range(50):                                          # the child was killed with the group
            if not running(pid):
                break
            time.sleep(0.1)
        else:
            self.fail('the launcher child is still alive')

    def test_x2w_flat_server_zero_uses_one_shot_calls(self):
        renderer = launcher(self.dir, 'off', 'serve')
        patch = unittest.mock.patch.object(R, 'SERVER_OFF', True)
        patch.start()
        self.addCleanup(patch.stop)
        self.assertTrue(self.run_case(renderer, '--pass', 'a').stdout.startswith('ONESHOT '))


if __name__ == '__main__':
    unittest.main()
