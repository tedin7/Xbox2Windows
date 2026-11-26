"""tools/startup/slot_compare.py: the per-slot tester/replay comparison (pure Python, runs everywhere).

Fixtures use the real line shapes of the stand-in's log (Run S numbers): a replay with the same per-call cost matches; a replay in which
one slot costs 2x per call (the mutant: a fix that did not take, or a replay that does not reproduce the lock waits) is a MISMATCH;
slots under the threshold never fail; a log without the lines is unusable (exit 2).
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
import slot_compare as S  # noqa: E402

TESTER = """\
x2w-d3d12 heartbeat: 1000 frames presented in 40.0 s (25.0 fps overall)
x2w-d3d12 command-list slots over 20 ms: slot12 15827 ms/1271 slot14 5698 ms/11042 slot61 5498 ms/18856 slot139 3803 ms/669
x2w-d3d12 heartbeat: 1010 frames presented in 45.0 s (22.0 fps overall)
x2w-d3d12 command-list slots over 20 ms: slot12 12316 ms/726 slot14 8596 ms/13560 slot61 4509 ms/15471 slot9 5 ms/615
x2w-d3d12 heartbeat: 1020 frames presented in 50.0 s (20.0 fps overall)
"""


def replay_text(factor12=1.0):
    return TESTER.replace("slot12 15827 ms/1271", f"slot12 {int(15827 * factor12)} ms/1271").replace(
        "slot12 12316 ms/726", f"slot12 {int(12316 * factor12)} ms/726")


class SlotCompare(unittest.TestCase):
    def run_compare(self, tester, replay, *extra):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "tester.txt"), Path(d, "replay.txt")
            a.write_text(tester)
            b.write_text(replay)
            return S.main([str(a), str(b), *extra])

    def test_same_cost_matches(self):
        self.assertEqual(self.run_compare(TESTER, replay_text(1.1)), 0)

    def test_a_slot_twice_as_expensive_per_call_is_a_mismatch(self):
        self.assertEqual(self.run_compare(TESTER, replay_text(2.0)), 1)

    def test_cheap_slots_do_not_fail_the_check(self):
        slow = TESTER.replace("slot9 5 ms/615", "slot9 550 ms/615")
        self.assertEqual(self.run_compare(TESTER, slow), 0)

    def test_report_window_selects_the_same_part(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d, "t.txt")
            a.write_text(TESTER)
            slots, reports, frames = S.read_run(a, 2, 2)
        self.assertEqual((reports, slots[12]), (1, (12316, 726)))
        self.assertEqual(frames, 20)

    def test_no_lines_is_unusable(self):
        self.assertEqual(self.run_compare(TESTER, "nothing here\n"), 2)

    def test_ms_per_call_and_ratio(self):
        rows, failed = S.compare(self.parsed(TESTER),
                                 self.parsed(replay_text(2.0)))
        self.assertEqual(failed, [12])

    def parsed(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "x.txt")
            p.write_text(text)
            return S.read_run(p)


if __name__ == "__main__":
    unittest.main()
