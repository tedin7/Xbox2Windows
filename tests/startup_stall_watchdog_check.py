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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools" / "startup"))
import stall_watchdog  # noqa: E402
from stall_watchdog import SIGNALS, Watch  # noqa: E402

STEP = 5


def healthy_loading(t):
    return {"commit_percent": 40.0 + t / 100, "present": 0, "storage": 1000 * t, "translator": 0, "picture": "black"}


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


class Process(unittest.TestCase):
    def run_watchdog(self, logdir, *extra, timeout=30):
        command = [sys.executable, "-B", str(HERE.parent / "tools" / "startup" / "stall_watchdog.py"), str(logdir),
                   "--parent", str(os.getpid()), "--every", "0.2", *extra]
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    def test_a_frozen_run_gets_the_stop_file_and_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            (logdir / "events.txt").write_text("start\n")
            (logdir / "stderr.txt").write_text("x2w-d3d12 heartbeat: 7 frames presented in 3.0 s (2.3 fps overall)\n"
                                               "x2w-dstorage progress 3 s: pending 0 (oldest none), read 0.0 MB/s, inflated 0.0 MB/s, "
                                               "enqueued 5 completed 5, total read 64 KiB inflated 64 KiB\n")
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
                       "--parent", str(parent.pid), "--every", "0.2", "--min-seconds", "0", "--stall-seconds", "2",
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

    def test_bad_numbers_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (("--stall-seconds", "-1"), ("--memory-percent", "101")):
                self.assertEqual(self.run_watchdog(tmp, *bad).returncode, 2, bad)


if __name__ == "__main__":
    unittest.main()
