"""Check the decision of the startup debugger's commit headroom guard (tools/startup/commit_headroom.py).

Pure Python. The Windows side (a real run ended by the guard) is tests/startup_debug_window_check.py.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
from commit_headroom import HeadroomGuard  # noqa: E402


class Reader:
    def __init__(self, values):
        self.values, self.calls = list(values), 0

    def __call__(self):
        self.calls += 1
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class Guard(unittest.TestCase):
    def run_guard(self, headroom, values, step=2.0, warn=0):
        reader = Reader(values)
        guard = HeadroomGuard(headroom, reader, warn)
        for n in range(len(values) + 2):
            line = guard.check(n * step)
            if line:
                return n, line, guard, reader
        return None, None, guard, reader

    def test_a_run_with_room_goes_on(self):
        at, line, guard, _ = self.run_guard(1500, [9000, 8000, 7000, 1600])
        self.assertIsNone(line)
        self.assertEqual(guard.lowest_free_mib, 1600)

    def test_two_samples_under_the_headroom_end_the_run(self):
        at, line, guard, _ = self.run_guard(1500, [9000, 1400, 1300])
        self.assertEqual(at, 2)
        self.assertEqual(line, "system commit has 1300 MiB free, under the headroom of 1500 MiB")
        self.assertEqual(guard.lowest_free_mib, 1300)

    def test_one_dip_is_not_enough(self):
        at, line, _, _ = self.run_guard(1500, [9000, 1400, 4000, 1400, 9000])
        self.assertIsNone(line)

    def test_free_equal_to_the_headroom_is_enough(self):
        self.assertIsNone(self.run_guard(1500, [1500, 1500, 1500])[1])

    def test_off_reads_nothing(self):
        at, line, guard, reader = self.run_guard(0, [0, 0, 0])
        self.assertIsNone(line)
        self.assertEqual(reader.calls, 0)
        self.assertEqual(guard.summary(), "commit headroom: off")

    def test_it_reads_once_per_sample_interval(self):
        reader = Reader([9000])
        guard = HeadroomGuard(1500, reader)
        for now in (0.0, 0.5, 1.0, 1.9, 2.0, 2.5, 3.9, 4.0):
            guard.check(now)
        self.assertEqual(reader.calls, 3)  # at 0.0, 2.0 and 4.0

    def test_an_unreadable_value_counts_as_room(self):
        at, line, _, _ = self.run_guard(1500, [1400, None, 1400, None])
        self.assertIsNone(line)

    def test_summary_names_the_lowest_sample(self):
        guard = HeadroomGuard(1500, Reader([9000, 5200, 7000]))
        self.assertEqual(guard.summary(), "commit headroom: stop below 1500 MiB, no sample")
        for n in range(3):
            guard.check(n * 2.0)
        self.assertEqual(guard.summary(), "commit headroom: stop below 1500 MiB, lowest free commit 5200 MiB")

    def test_a_warning_comes_once_and_again_for_every_further_step(self):
        values = [9000, 2000, 1900, 1800, 1700, 1500, 1500, 9000, 2000]
        reader = Reader(values)
        guard = HeadroomGuard(256, reader, warn_mib=2048)
        lines = []
        for n in range(len(values)):
            self.assertIsNone(guard.check(n * 2.0))
            lines.append(guard.take_warning())
        self.assertIsNone(guard.take_warning(), "a warning is handed out once")
        # 2000 warns; 1900 and 1800 are within one step of it; 1700 is a further step down (below 1744); 1500 is not (above 1444);
        # the recovery to 9000 resets, so 2000 warns again.
        self.assertEqual([i for i, line in enumerate(lines) if line], [1, 4, 8])
        self.assertEqual(lines[1], "system commit has 2000 MiB free (warning below 2048 MiB, the run ends below 256 MiB)")
        self.assertEqual(guard.warnings, 3)

    def test_a_warning_never_ends_the_run(self):
        at, line, guard, _ = self.run_guard(256, [900, 800, 700, 600, 500], warn=2048)
        self.assertIsNone(line)
        self.assertEqual(guard.lowest_free_mib, 500)

    def test_the_stop_is_the_last_resort(self):
        at, line, _, _ = self.run_guard(256, [4000, 600, 300, 255, 200], warn=2048)
        self.assertEqual(line, "system commit has 200 MiB free, under the headroom of 256 MiB")

    def test_warn_only_and_summary(self):
        guard = HeadroomGuard(0, Reader([100]), warn_mib=2048)
        self.assertIsNone(guard.check(0.0))
        self.assertTrue(guard.take_warning().endswith("(warning below 2048 MiB)"))
        self.assertEqual(guard.summary(), "commit headroom: warning below 2048 MiB (1 warning), lowest free commit 100 MiB")
        both = HeadroomGuard(256, Reader([100]), warn_mib=2048)
        both.check(0.0)
        self.assertEqual(both.summary(), "commit headroom: stop below 256 MiB, warning below 2048 MiB (1 warning), lowest free commit 100 MiB")


if __name__ == "__main__":
    unittest.main()
