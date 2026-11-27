"""Check the stall watchdog of the tester run (tools/startup/stall_watchdog.py): the decision on recorded-shape samples and the
process end to end on a fake run folder. Pure Python, runs everywhere.

Fixtures (one sample every 5 s, shaped like the runs they are named after):
- healthy loading (Run K): the picture never changes and no frame is presented for minutes, but storage keeps reading;
- memory climb (Run L): committed memory rises to 95% and stays there;
- starved (Run L): frames, storage, translator and picture all stop after 300 s.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools" / "startup"))
import stall_watchdog  # noqa: E402
from stall_watchdog import SIGNALS, Watch  # noqa: E402

STEP = 5


def healthy_loading(t):
    return {"commit_percent": 40.0 + t / 100, "present": 0, "storage": 1000 * t, "translator": 0, "picture": "black", "cpu": t * 1.5}


def memory_climb(t):
    return {"commit_percent": min(95.0, 50.0 + t / 8), "present": t, "storage": 10 * t, "translator": t, "picture": str(t)}


def starved(t):
    frozen = min(t, 300)
    return {"commit_percent": 60.0, "present": frozen, "storage": 10 * frozen, "translator": frozen, "picture": str(frozen)}


def first_stop(fixture, seconds=900, **options):
    watch = Watch(max_seconds=seconds, **options)
    for t in range(0, seconds, STEP):
        reason = watch.decide(t, fixture(t))
        if reason:
            return t, reason
    return None, None


class Decision(unittest.TestCase):
    def test_healthy_loading_is_never_stopped(self):
        self.assertEqual(first_stop(healthy_loading), (None, None))

    def test_dropping_the_storage_signal_would_stop_healthy_loading(self):
        # The mutant of the proof: without storage the loading run looks frozen, which is why the signal is there.
        def without_storage(t):
            sample = healthy_loading(t)
            sample["storage"] = None
            return sample
        at, reason = first_stop(without_storage)
        self.assertIsNotNone(at)
        self.assertTrue(reason.startswith("frozen:"))

    def test_memory_at_95_percent_in_two_samples_stops_with_the_memory_reason(self):
        at, reason = first_stop(memory_climb)
        # 50 + t/8 reaches 95 at t = 360; the second sample in a row is 5 s later.
        self.assertEqual(at, 365)
        self.assertTrue(reason.startswith("memory: committed memory is 95% of the commit limit (limit 95%)"), reason)

    def test_one_high_sample_is_not_enough(self):
        watch = Watch()
        high = {"commit_percent": 99.0, "present": 1}
        low = {"commit_percent": 50.0, "present": 2}
        self.assertIsNone(watch.decide(200, high))
        self.assertIsNone(watch.decide(205, low))
        self.assertIsNone(watch.decide(210, dict(high, present=3)))

    def test_a_starved_run_stops_after_the_stall_seconds(self):
        at, reason = first_stop(starved)
        self.assertEqual(at, 390)  # last change at 300, then 90 s of nothing
        self.assertTrue(reason.startswith("frozen:") and "for 90 s" in reason, reason)

    def test_each_signal_alone_keeps_the_run_alive(self):
        for kept in SIGNALS:
            def only(t, kept=kept):
                sample = {name: 0 for name in SIGNALS}
                sample["picture"] = "same"
                sample["commit_percent"] = 50.0
                sample[kept] = t
                return sample
            self.assertEqual(first_stop(only), (None, None), kept)

    def test_never_before_the_minimum_seconds(self):
        watch = Watch(stall_seconds=10, min_seconds=150)
        for t in range(0, 150, STEP):
            self.assertIsNone(watch.decide(t, {"commit_percent": 99.0, "present": 0}), t)
        self.assertIsNotNone(watch.decide(150, {"commit_percent": 99.0, "present": 0}))

    def test_never_at_or_after_the_normal_end(self):
        watch = Watch(max_seconds=600)
        watch.decide(0, {"present": 1})
        self.assertIsNotNone(watch.decide(599, {"present": 1}))
        self.assertIsNone(Watch(max_seconds=600).decide(600, {"commit_percent": 99.0, "present": 1}))

    def test_rules_can_be_turned_off(self):
        self.assertEqual(first_stop(starved, stall_seconds=0), (None, None))
        self.assertEqual(first_stop(memory_climb, memory_percent=0), (None, None))

    def test_nothing_known_is_not_a_stop(self):
        watch = Watch()
        self.assertIsNone(watch.decide(400, {}))
        self.assertIsNone(watch.decide(500, {"commit_percent": None}))


    def test_a_deadlock_is_told_from_loading_by_the_cpu(self):
        def deadlocked(t):
            frozen = min(t, 300)
            return {"commit_percent": 60.0, "present": frozen, "storage": 10 * frozen, "translator": frozen, "picture": "same",
                    "cpu": 100.0 + min(t, 300) * 0.002}
        at, reason = first_stop(deadlocked)
        self.assertEqual(at, 390)
        self.assertTrue(reason.startswith("deadlock: no frame presented for 90 s and the title used 0.0 s of CPU"), reason)

        def busy(t):  # nothing presented either, but the title works: loading, not a deadlock
            sample = deadlocked(t)
            sample["cpu"], sample["storage"] = t * 1.5, 1000 * t
            return sample
        self.assertEqual(first_stop(busy), (None, None))


class PresentDump(unittest.TestCase):
    def test_floor_and_stall_threshold(self):
        for first_frame, expected in ((5, [30]), (20, [35])):
            with self.subTest(first_frame=first_frame):
                watch = Watch(stall_seconds=0, memory_percent=0)
                fired = []
                for t in range(0, 55, STEP):
                    self.assertIsNone(watch.decide(t, {"present": 7 if t == first_frame else None, "cpu": t}))
                    if watch.wants_present_dump(t):
                        fired.append(t)
                        self.assertFalse(watch.wants_present_dump(t))
                self.assertEqual(fired, expected)

    def test_rearms_on_frames_and_caps_at_three(self):
        watch = Watch()
        fired = []
        for t in range(0, 140, STEP):
            watch.decide(t, {"present": None if t < 15 else 1 + (t - 15) // 30})
            if watch.wants_present_dump(t):
                fired.append(t)
        self.assertEqual(fired, [30, 60, 90])

    def test_no_dump_without_a_frame_count(self):
        watch = Watch()
        for t in range(0, 100, STEP):
            watch.decide(t, {"storage": t})
            self.assertFalse(watch.wants_present_dump(t))

    def test_off_and_stop_decisions(self):
        enabled, disabled = Watch(), Watch(present_dump_seconds=0)
        dumps, stops = [], []
        for t in range(0, 400, STEP):
            sample = starved(t)
            reason = enabled.decide(t, sample)
            self.assertEqual(reason, disabled.decide(t, sample))
            self.assertFalse(disabled.wants_present_dump(t))
            if enabled.wants_present_dump(t):
                dumps.append(t)
            if reason:
                stops.append(t)
        self.assertEqual(dumps, [315])
        self.assertEqual(stops[0], 390)

    def test_request_names(self):
        parse = stall_watchdog.dump_request_filename
        self.assertEqual(parse("threads_present_stall_1.txt\n"), "threads_present_stall_1.txt")
        self.assertEqual(parse("threads_other_0.txt"), "threads_other_0.txt")
        for invalid in ("", "threads\n", "threads_.txt", "threads_BAD.txt", "threads_a.txt\nthreads_b.txt",
                        "../threads_a.txt", "C:\\threads_a.txt", "sub/threads_a.txt", "threads_a.txt.bak"):
            with self.subTest(invalid=invalid):
                self.assertEqual(parse(invalid), "threads_stall.txt")

    def test_main_writes_request_and_keeps_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            (logdir / "events.txt").touch()
            with (patch.object(stall_watchdog, "parent_alive", side_effect=[True, True, True, False]),
                  patch.object(stall_watchdog.time, "time", side_effect=[100, 100, 130, 135]),
                  patch.object(stall_watchdog.time, "sleep") as sleep,
                  patch.object(stall_watchdog, "take_sample", return_value={"present": 7}) as sample,
                  patch("builtins.print") as say):
                self.assertEqual(stall_watchdog.main([tmp, "--stall-seconds", "0", "--memory-percent", "0"]), 0)
            self.assertEqual(sample.call_count, 3)
            self.assertEqual(sleep.call_count, 3)
            self.assertEqual((logdir / "dump_request").read_text(), "threads_present_stall_1.txt")
            self.assertFalse((logdir / "stop").exists())
            say.assert_called_once_with(
                "tester_run: threads dump at 30 s: no frame presented for 30 s (threads_present_stall_1.txt)", flush=True)


class Process(unittest.TestCase):
    def run_watchdog(self, logdir, *extra, timeout=30):
        command = [sys.executable, "-B", str(HERE.parent / "tools" / "startup" / "stall_watchdog.py"), str(logdir),
                   "--parent", str(os.getpid()), "--every", "0.2", "--hard-grace", "0", *extra]
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    def test_a_frozen_run_gets_the_stop_file_and_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            (logdir / "events.txt").write_text("start\n")
            (logdir / "stderr.txt").write_text("x2w-d3d12 heartbeat: 7 frames presented in 3.0 s (2.3 fps overall)\n"
                                               "x2w-dstorage progress 3 s: pending 0 (oldest none), read 0.0 MB/s, inflated 0.0 MB/s, "
                                               "enqueued 5 retired 5 (written 5, failed 0), total read 64 KiB inflated 64 KiB\n")
            result = self.run_watchdog(logdir, "--min-seconds", "0", "--stall-seconds", "2", "--memory-percent", "0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertRegex(result.stdout, r"^tester_run: STALL STOP at \d+ s: frozen: ")
            self.assertTrue((logdir / "stop").exists())
            data = json.loads((logdir / "stall_stop.json").read_text())
            self.assertTrue(data["reason"].startswith("frozen:") and data["at_seconds"] >= 2, data)

    def test_a_moving_run_is_left_alone_until_the_parent_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            (logdir / "events.txt").write_text("start\n")
            parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            command = [sys.executable, "-B", str(HERE.parent / "tools" / "startup" / "stall_watchdog.py"), str(logdir),
                       "--parent", str(parent.pid), "--every", "0.2", "--hard-grace", "0", "--min-seconds", "0", "--stall-seconds", "2",
                       "--memory-percent", "0"]
            watcher = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
            frames = 0
            while parent.poll() is None:
                frames += 1
                (logdir / "stderr.txt").write_text("x2w-d3d12 heartbeat: %d frames presented in 1.0 s\n" % frames)
                time.sleep(0.1)
            out, _ = watcher.communicate(timeout=20)
            parent.wait()
            self.assertEqual(out, "")
            self.assertFalse((logdir / "stop").exists())

    def test_a_debugger_that_will_not_end_is_ended_with_its_title(self):
        # A fake debugger (debug_launch.py in its command line, the run folder as argument) that ignores the stop file and
        # SIGTERM, with a fake title below it: after --seconds + --hard-grace both are ended, hard_end.json says so.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            logdir = tmp / "run"
            logdir.mkdir()
            (logdir / "events.txt").write_text("start\n")
            fake_title = ("import os, signal, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                          "open(sys.argv[1] + '/title.pid', 'w').write(str(os.getpid()))\ntime.sleep(300)\n")
            (tmp / "title.py").write_text(fake_title)
            (tmp / "debug_launch.py").write_text(
                "import signal, subprocess, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "subprocess.Popen([sys.executable, sys.argv[2], sys.argv[1]])\ntime.sleep(300)\n")
            debugger = subprocess.Popen([sys.executable, str(tmp / "debug_launch.py"), str(logdir), str(tmp / "title.py")])
            for _ in range(50):
                if (logdir / "title.pid").exists():
                    break
                time.sleep(0.1)
            title = int((logdir / "title.pid").read_text())
            started = time.monotonic()
            try:
                result = self.run_watchdog(logdir, "--seconds", "2", "--min-seconds", "1000", "--hard-grace", "2", "--memory-percent", "0")
                debugger.wait(timeout=10)   # reaped: a killed child is a zombie until then
            finally:
                debugger.kill()
            self.assertLess(time.monotonic() - started, 20)
            self.assertIn("tester_run: HARD END at ", result.stdout)
            note = json.loads((logdir / "hard_end.json").read_text())
            self.assertTrue(note["title_gone"], note)
            self.assertFalse(stall_watchdog.pid_alive(title), "the title must be gone")
            self.assertFalse(stall_watchdog.pid_alive(debugger.pid))

    def test_a_run_that_finished_by_itself_is_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            (logdir / "events.txt").write_text("start\n")
            (logdir / "finished.txt").write_text("done\n")
            result = self.run_watchdog(logdir, "--seconds", "1", "--hard-grace", "1", "--min-seconds", "1000")
            self.assertEqual(result.stdout, "")
            self.assertFalse((logdir / "hard_end.json").exists())

    def test_bad_numbers_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (("--stall-seconds", "-1"), ("--memory-percent", "101")):
                self.assertEqual(self.run_watchdog(tmp, *bad).returncode, 2, bad)


if __name__ == "__main__":
    unittest.main()
