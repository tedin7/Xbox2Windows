"""Commit headroom guard of the startup debugger (debug_launch.py --commit-headroom-mib).

A run that uses all of the system commit (RAM plus page files) ends on error 1455 in the title, and everything else on
the machine suffers too. With a headroom the launcher ends the run first, the way a `stop` file does, so the logs are
still written and packed. This module is the decision only (no Windows calls): debug_launch.py passes a function that
reads the free commit in MiB.
"""

SAMPLE_SECONDS = 2.0   # how often the free commit is read
SAMPLES_BELOW = 2      # consecutive samples under the headroom before the run is ended (one dip is not enough)


class HeadroomGuard:
    def __init__(self, headroom_mib, read_free_mib, sample_seconds=SAMPLE_SECONDS, samples_below=SAMPLES_BELOW):
        self.headroom_mib = headroom_mib
        self._read = read_free_mib
        self._every, self._need = sample_seconds, samples_below
        self._next = 0.0
        self._below = 0
        self.lowest_free_mib = None   # smallest free commit seen, for the summary line

    def check(self, now):
        """None while the run may go on; the line to log when it must end (free commit under the headroom in
        `samples_below` consecutive samples). Reads at most once per sample interval; an unreadable value counts as
        enough room."""
        if not self.headroom_mib or now < self._next:
            return None
        self._next = now + self._every
        free = self._read()
        if free is None:
            self._below = 0
            return None
        self.lowest_free_mib = free if self.lowest_free_mib is None else min(self.lowest_free_mib, free)
        self._below = self._below + 1 if free < self.headroom_mib else 0
        if self._below >= self._need:
            return "system commit has %d MiB free, under the headroom of %d MiB" % (free, self.headroom_mib)
        return None

    def summary(self):
        if not self.headroom_mib:
            return "commit headroom: off"
        low = "no sample" if self.lowest_free_mib is None else "lowest free commit %d MiB" % self.lowest_free_mib
        return "commit headroom: %d MiB, %s" % (self.headroom_mib, low)
