"""Keeps concurrent threads from running through a hooked byte while it is restored for a single step.

A function-entry hook is one patched byte. The thread that hits it restores the original byte, steps one
instruction, then patches the byte again. Without a gate, every other thread entering the same function in
that window runs the original byte and its call is never traced. The gate suspends the other threads for
the step and patches the byte again only once no thread is still stepping over it.
"""


class StepGate:
    def __init__(self, suspend, resume, live):
        self._suspend = suspend  # tid -> bool, True when the thread is now suspended
        self._resume = resume  # tid -> None
        self._live = live  # callable returning the live thread ids
        self.stepping = {}  # tid -> address being stepped over
        self._held = {}  # tid -> suspend count held by the gate
        self.dropped = 0  # steps begun while another thread was already stepping the same address

    def begin(self, tid, addr):
        """A thread hit the hook at addr and is about to step.

        All-or-nothing: every live peer is suspended before the caller opens the byte. Returns whether addr was
        already open (bool), or None when a peer could not be suspended; the peers taken so far are released,
        the thread is not marked as stepping and the caller must leave the byte patched and retry the hit.
        """
        overlap = addr in self.stepping.values()
        for _ in range(self._held.pop(tid, 0)):  # a gate-held thread that hit the hook must run to step
            self._resume(tid)
        taken = []
        for other in sorted(self._live()):
            if other == tid or other in self.stepping:
                continue
            if not self._suspend(other):
                for done in taken:
                    self._resume(done)
                return None
            taken.append(other)
        for other in taken:
            self._held[other] = self._held.get(other, 0) + 1
        self.stepping[tid] = addr
        if overlap:
            self.dropped += 1
        return overlap

    def finish(self, tid):
        """The thread's step ended. Returns the address to patch again, or None while others still step it."""
        addr = self.stepping.pop(tid, None)
        if addr is None:
            return None
        if not self.stepping:
            for other, count in list(self._held.items()):
                for _ in range(count):
                    self._resume(other)
            self._held.clear()
        return None if addr in self.stepping.values() else addr

    def forget(self, tid):
        """The thread exited. Returns the address to patch again when it was the last stepper, else None."""
        self._held.pop(tid, None)
        return self.finish(tid)
