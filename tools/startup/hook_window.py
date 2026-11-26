"""Step-over window of the startup debugger's one-byte hooks (docs/startup-runtime.md).

A hook is one patched byte per function. To run the original instruction the debugger restores the
byte, single-steps the thread that hit it and patches the byte again. A second thread that enters
the function inside that window would run the original instruction untraced, so the window holds
every other thread for the step: the owner steps alone and the others resume when it is done.

This module is the bookkeeping only (no Win32 calls); debug_launch.py passes the suspend and resume
functions. Every window has a deadline, a thread that cannot be held is counted, and the counters
are always reported so a run says how many calls it may have missed.
"""

HELD, GONE, FAILED = "held", "gone", "failed"
ERROR_INVALID_PARAMETER = 87  # OpenThread: no thread has this id
STILL_ACTIVE = 259  # GetExitCodeThread of a thread that has not exited
STATUS_INVALID_HANDLE = 0xC0000008
STATUS_THREAD_IS_TERMINATING = 0xC000004B  # NtSuspendThread: the thread is exiting and runs no more code


def classify_status(status):
    """GONE for an NtSuspendThread status that says the thread cannot run again, else FAILED."""
    return GONE if status in (STATUS_THREAD_IS_TERMINATING, STATUS_INVALID_HANDLE) else FAILED


def classify_hold(exit_code, error=0, signaled=False):
    """Outcome of a hold that did not take: GONE when the thread has exited, FAILED when it is alive.

    A thread that called ExitThread stays in the debugger's thread list until its EXIT_THREAD event is
    read, and by then OpenThread still works but SuspendThread and GetThreadContext fail (access
    denied). That thread runs no more code, so nothing can race with it: GONE. A live thread that
    cannot be suspended is a real hole in the window and stays FAILED. `exit_code` is
    GetExitCodeThread (None when it could not be read), `signaled` whether the thread handle is
    already signaled, `error` the GetLastError of an OpenThread that returned no handle."""
    if error == ERROR_INVALID_PARAMETER:
        return GONE
    if signaled or (exit_code is not None and exit_code != STILL_ACTIVE):
        return GONE
    return FAILED


class ReturnBook:
    """Pending hooked calls, keyed by return address; each is traced once, when its call returns.

    A call is identified by its thread, the stack slot of its return address, and the hooked entry
    address. The entry hook can trap twice for one call when protection is off: another thread
    re-patches the byte before this one has run the original instruction, so the thread traps again
    at the same entry, thread and slot. That is the same call, so it replaces the first
    registration; two entries made the return write two lines and a run counted 88 opens for 64
    made. A hooked function that tail-calls another hooked function shares (tid, slot) with a
    different entry address, so both registrations are kept."""

    def __init__(self):
        # return address -> list of (tid, stack slot, description, entry, restore)
        self._pending = {}

    def add(self, address, tid, slot, description, entry=None, restore=None):
        """Registers a call. restore, if given, undoes a change made to the call's arguments; it runs
        once, when the call returns. A re-trap of the same call keeps the first restore: it holds the
        caller's original state, the second one would hold the already changed one."""
        items = self._pending.get(address, [])
        for item in items:
            if (item[0], item[1], item[3]) == (tid, slot, entry):
                restore = item[4] or restore
        kept = [item for item in items if (item[0], item[1], item[3]) != (tid, slot, entry)]
        kept.append((tid, slot, description, entry, restore))
        self._pending[address] = kept

    def pending(self, address):
        """Whether any call is still waiting to return to this address (any thread)."""
        return bool(self._pending.get(address))

    def take(self, address, tid, slot):
        """Descriptions of the calls that return here on this thread and slot. Their restore callbacks
        have run when this returns."""
        items = self._pending.get(address, [])
        taken = [item for item in items if (item[0], item[1]) == (tid, slot)]
        if taken:
            self._pending[address] = [item for item in items if (item[0], item[1]) != (tid, slot)]
        for item in taken:
            if item[4]:
                item[4]()
        return [item[2] for item in taken]


class StepWindow:
    def __init__(self, hold, release, protect=True, limit=2.0):
        self._hold, self._release = hold, release  # hold(tid) -> HELD | GONE | FAILED; release(tid)
        self.protect, self.limit = protect, limit
        self.owner = None
        self._held = []
        self._deadline = 0.0
        self._exposed = False
        self.windows = self.unprotected = self.deferred = self.timeouts = 0

    def open(self, owner, others, now):
        """The owner is about to step over its hook: hold every other thread first."""
        self.windows += 1
        if self.owner is not None:
            # A thread that was not held (or reached another hook) steps while a window is open: the
            # window stays with its owner and is reported as exposed.
            self._expose()
            return
        if not self.protect:  # measuring mode: nothing is held, every window with company is exposed
            self.unprotected += 1 if others else 0
            return
        self.owner, self._deadline, self._exposed = owner, now + self.limit, False
        for tid in others:
            self.hold(tid)

    def hold(self, tid):
        """Holds one more thread while a window is open (a thread created during the step)."""
        if self.owner is None or tid == self.owner or tid in self._held:
            return
        result = self._hold(tid)
        if result == HELD:
            self._held.append(tid)
        elif result == FAILED:
            self._expose()

    def defer(self, tid):
        """True when a hook hit comes from a thread held by the open window: it trapped before it was held, so it
        must go back to the hook address and trap again once the window is closed."""
        if tid not in self._held:
            return False
        self.deferred += 1
        return True

    def finish(self, tid):
        """The owner's step is done (or the owner is gone): the held threads resume."""
        if self.owner is None or tid != self.owner:
            return False
        self._close()
        return True

    def expire(self, now):
        """A window that is still open at its deadline is closed and counted as exposed."""
        if self.owner is None or now < self._deadline:
            return False
        self.timeouts += 1
        self._expose()
        self._close()
        return True

    def abandon(self):
        """The run is over: resume whatever is held without counting a timeout."""
        self._close()

    def summary(self, entries):
        return (f"hook windows: entries={entries} windows={self.windows} unprotected={self.unprotected} "
                f"deferred={self.deferred} timeouts={self.timeouts} protection={'on' if self.protect else 'off'}")

    def _expose(self):
        if not self._exposed:
            self._exposed = True
            self.unprotected += 1

    def _close(self):
        held, self._held, self.owner = self._held, [], None
        for tid in held:
            self._release(tid)
