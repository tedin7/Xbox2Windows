"""Check tools/startup/tester_janitor.py: the tester stops tester_run (Ctrl-C) or closes its window, the report is still packed and
the translator is still ended. Pure Python (fake processes), runs everywhere.

Cases: (1) interrupted run: a fake debugger that ignores the stop file and a fake title and translator are ended, the zip exists
with interrupted=true and an end_cause, janitor.txt tells what happened; (2) the debugger finishes by itself after the stop
file: no hard end; (3) normal end (pack_done present): no second zip, a leftover translator is still ended.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTUP = HERE.parent / "tools" / "startup"
sys.path.insert(0, str(STARTUP))
import stall_watchdog as W  # noqa: E402
import tester_pack as T  # noqa: E402

MANIFEST = {"commit": "abcd1234", "build_commit": "abcd1234", "dirty": False, "seconds": 600, "input_script": "x", "env": {},
            "gpu": "g", "gpu_driver": "d", "windows_build": "1", "wall_seconds": 0, "deep": False, "lazy_textures": True,
            "lazy_memory": True, "memory_gib": 20, "shader_cache": "shader_cache_1", "translations_at_start": 0}


def spawn(code, *args):
    return subprocess.Popen([sys.executable, "-c", code, *args])


class Janitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.logdir = self.root / "run"
        self.logdir.mkdir()
        (self.logdir / "events.txt").write_text("start\n")
        (self.logdir / "stderr.txt").write_text("x2w-d3d12 heartbeat: 90 frames presented in 30.0 s (3.0 fps overall)\n")
        self.cache = self.root / "shader_cache_1"
        self.cache.mkdir()
        self.zip = self.root / "report.zip"
        self.job = self.root / "job.json"
        self.job.write_text(json.dumps({"manifest": MANIFEST, "zip": str(self.zip), "limit_mb": 20, "cache": str(self.cache), "map": ""}))
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        self.tmp.cleanup()

    def start(self, code, *args):
        proc = spawn(code, *args)
        self.procs.append(proc)
        return proc

    def janitor(self, parent_pid, grace="2"):
        command = [sys.executable, "-B", str(STARTUP / "tester_janitor.py"), str(self.logdir), "--parent", str(parent_pid),
                   "--pack-json", str(self.job), "--cache", str(self.cache), "--poll", "0.1", "--grace", grace]
        return subprocess.run(command, capture_output=True, text=True, timeout=60)

    def fake_translator(self):
        # command line: gfx2hlsl.py --watch <cache>; it ignores SIGTERM like a busy pool worker
        return self.start("import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(300)",
                          "gfx2hlsl.py", "--watch", str(self.cache))

    def test_an_interrupted_run_is_ended_and_packed(self):
        title = self.start("import signal,sys,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                           "open(sys.argv[1]+'/title.pid','w').write(str(__import__('os').getpid()));time.sleep(300)", str(self.logdir))
        debugger = self.start("import signal,subprocess,sys,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                              "subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)']);time.sleep(300)",
                              "debug_launch.py", str(self.logdir))
        translator = self.fake_translator()
        for _ in range(50):
            if (self.logdir / "title.pid").exists():
                break
            time.sleep(0.1)
        parent = self.start("import time;time.sleep(1)")      # tester_run, stopped a second in
        result = self.janitor(parent.pid)
        self.assertEqual(result.returncode, 0, result.stderr)
        for proc in (debugger, title, translator):
            proc.wait(timeout=10)
        self.assertTrue(self.zip.exists(), "the report was not packed")
        with zipfile.ZipFile(self.zip) as z:
            manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        self.assertIs(manifest["interrupted"], True)
        self.assertTrue(manifest["end_cause"].startswith("interrupted by the tester (Ctrl-C or the window was closed); "), manifest["end_cause"])
        self.assertIn("last present #90 at 30 s", manifest["end_cause"])
        text = (self.logdir / "janitor.txt").read_text()
        self.assertIn("ended the translator", text)
        self.assertIn("title %d gone" % title.pid, text)
        self.assertTrue((self.logdir / "stop").exists())

    def test_a_debugger_that_obeys_the_stop_file_is_not_killed(self):
        debugger = self.start("import os,sys,time\nwhile not os.path.exists(sys.argv[2]+'/stop'): time.sleep(0.1)\n"
                              "open(sys.argv[2]+'/finished.txt','w').write('timeout after 5.0 s\\n')\ntime.sleep(300)",
                              "debug_launch.py", str(self.logdir))
        parent = self.start("import time;time.sleep(1)")
        self.janitor(parent.pid, grace="20")
        self.assertIsNone(debugger.poll(), "an obedient debugger must be left to end itself")
        text = (self.logdir / "janitor.txt").read_text()
        self.assertIn("the debugger finished by itself", text)
        self.assertTrue(self.zip.exists())

    def test_a_normal_end_is_not_packed_again_but_a_leftover_translator_is_ended(self):
        (self.logdir / "pack_done").write_text("tester_run\n")
        translator = self.fake_translator()
        parent = self.start("import time;time.sleep(1)")
        result = self.janitor(parent.pid)
        self.assertEqual(result.returncode, 0, result.stderr)
        translator.wait(timeout=10)
        self.assertFalse(self.zip.exists())
        self.assertNotIn("packing", (self.logdir / "janitor.txt").read_text())


if __name__ == "__main__":
    unittest.main()
