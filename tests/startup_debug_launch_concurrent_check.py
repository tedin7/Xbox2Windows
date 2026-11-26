"""A hooked byte stays patched for every thread except the one stepping over it."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "startup"))
from step_gate import StepGate  # noqa: E402


class Machine:
    """Fake process: threads run unless suspended; the hook byte is patched or restored."""

    def __init__(self, threads):
        self.live = set(threads)
        self.suspended = {}
        self.patched = True
        self.missed = []

    def suspend(self, tid):
        self.suspended[tid] = self.suspended.get(tid, 0) + 1
        return True

    def resume(self, tid):
        self.suspended[tid] -= 1

    def enter(self, tid):
        """A thread calls the hooked function; it is missed when it can run through a restored byte."""
        if self.suspended.get(tid, 0) == 0 and not self.patched:
            self.missed.append(tid)


class StepGateTest(unittest.TestCase):
    def setUp(self):
        self.machine = Machine({1, 2, 3})
        self.gate = StepGate(self.machine.suspend, self.machine.resume, lambda: self.machine.live)

    def hit(self, tid):
        self.machine.patched = False
        self.gate.begin(tid, 0x1000)

    def end(self, tid):
        addr = self.gate.finish(tid)
        if addr is not None:
            self.machine.patched = True

    def test_other_threads_cannot_run_through_a_step(self):
        self.hit(1)
        self.machine.enter(2)
        self.machine.enter(3)
        self.end(1)
        self.assertEqual(self.machine.missed, [])
        self.assertTrue(self.machine.patched)
        self.assertEqual({t: c for t, c in self.machine.suspended.items() if c}, {})

    def test_byte_stays_open_until_the_last_stepper_finishes(self):
        self.hit(1)
        self.assertTrue(self.gate.begin(2, 0x1000))  # a hit already in flight on another thread
        self.end(1)
        self.assertFalse(self.machine.patched)
        self.end(2)
        self.assertTrue(self.machine.patched)
        self.assertEqual(self.gate.dropped, 1)
        self.assertEqual({t: c for t, c in self.machine.suspended.items() if c}, {})

    def test_no_thread_is_resumed_while_another_still_steps(self):
        self.hit(1)
        self.gate.begin(2, 0x1000)
        held = {t: c for t, c in self.machine.suspended.items() if c}
        self.assertTrue(held)
        self.end(1)
        self.assertEqual({t: c for t, c in self.machine.suspended.items() if c}, held)
        self.end(2)
        self.assertEqual({t: c for t, c in self.machine.suspended.items() if c}, {})

    def test_queued_hit_runs_to_step(self):
        self.hit(1)
        self.gate.begin(2, 0x1000)  # thread 2 was gate-suspended, then its queued hit arrives
        self.assertEqual(self.machine.suspended[2], 0)
        self.end(1)
        self.end(2)
        self.assertEqual({t: c for t, c in self.machine.suspended.items() if c}, {})

    def test_failed_suspension_leaves_the_gate_closed_and_rolls_back(self):
        machine = Machine({1, 2, 3, 4})
        refuse = {4}  # peer 4 cannot be suspended
        gate = StepGate(lambda tid: tid not in refuse and machine.suspend(tid), machine.resume, lambda: machine.live)
        self.assertIsNone(gate.begin(1, 0x1000))
        self.assertEqual(gate.stepping, {})  # not stepping: the caller keeps the byte patched
        self.assertEqual(gate.dropped, 0)
        self.assertEqual({t: c for t, c in machine.suspended.items() if c}, {})  # 2 and 3 were released
        refuse.clear()  # the retried hit succeeds
        self.assertFalse(gate.begin(1, 0x1000))
        self.assertEqual({t: c for t, c in machine.suspended.items() if c}, {2: 1, 3: 1, 4: 1})
        self.assertEqual(gate.finish(1), 0x1000)
        self.assertEqual({t: c for t, c in machine.suspended.items() if c}, {})

    def test_last_stepper_exit_releases_held_threads(self):
        self.hit(1)
        self.machine.live.discard(1)
        addr = self.gate.forget(1)
        self.assertEqual(addr, 0x1000)
        self.assertEqual({t: c for t, c in self.machine.suspended.items() if c}, {})

    def test_exited_thread_is_not_resumed(self):
        self.hit(1)
        self.machine.live.discard(3)
        self.gate.forget(3)
        self.end(1)
        self.assertEqual(self.gate.stepping, {})


if __name__ == "__main__":
    unittest.main()
