"""Commit headroom guard of the startup debugger (debug_launch.py --commit-headroom-mib, --commit-warn-mib).

A run that uses all of the system commit (RAM plus page files) ends on error 1455 in the title, and everything else on
the machine suffers too. Two thresholds on the free commit:
- warn: below it the launcher logs a warning line (again for every further WARN_STEP_MIB the free commit drops), the run goes on;
- stop: below it, in two consecutive samples, the launcher ends the run the way a `stop` file does, so the logs are
  still written and packed. It is meant as the last resort before error 1455: Windows grows the page file while a run
  is short of commit, so the free commit is allowed to get low before the run is ended.
This module is the decision only (no Windows calls): debug_launch.py passes a function that reads the free commit in MiB.
"""

SAMPLE_SECONDS = 2.0   # how often the free commit is read
SAMPLES_BELOW = 2      # consecutive samples under the stop headroom before the run is ended (one dip is not enough)
WARN_STEP_MIB = 256    # a further warning for every drop of the free commit by this much


class HeadroomGuard:
    def __init__(self, headroom_mib, read_free_mib, warn_mib=0, sample_seconds=SAMPLE_SECONDS, samples_below=SAMPLES_BELOW):
        self.headroom_mib = headroom_mib
        self.warn_mib = warn_mib
        self._read = read_free_mib
        self._every, self._need = sample_seconds, samples_below
        self._next = 0.0
        self._below = 0
        self._warned_at = None
        self._warning = None
        self.lowest_free_mib = None   # smallest free commit seen, for the summary line
        self.warnings = 0

    def check(self, now):
        """None while the run may go on; the line to log when it must end (free commit under the stop headroom in
        `samples_below` consecutive samples). Reads at most once per sample interval; an unreadable value counts as
        enough room. A warning, when due, is left for take_warning()."""
        if not (self.headroom_mib or self.warn_mib) or now < self._next:
            return None
        self._next = now + self._every
        free = self._read()
        if free is None:
            self._below = 0
            return None
        self.lowest_free_mib = free if self.lowest_free_mib is None else min(self.lowest_free_mib, free)
        if self.warn_mib and free < self.warn_mib:
            if self._warned_at is None or free <= self._warned_at - WARN_STEP_MIB:
                self._warned_at = free
                self.warnings += 1
                self._warning = "system commit has %d MiB free (warning below %d MiB%s)" % (
                    free, self.warn_mib, ", the run ends below %d MiB" % self.headroom_mib if self.headroom_mib else "")
        else:
            self._warned_at = None
        if not self.headroom_mib:
            return None
        self._below = self._below + 1 if free < self.headroom_mib else 0
        if self._below >= self._need:
            return "system commit has %d MiB free, under the headroom of %d MiB" % (free, self.headroom_mib)
        return None

    def take_warning(self):
        """The warning line that became due in the last check, once, or None."""
        text, self._warning = self._warning, None
        return text

    def summary(self):
        if not (self.headroom_mib or self.warn_mib):
            return "commit headroom: off"
        low = "no sample" if self.lowest_free_mib is None else "lowest free commit %d MiB" % self.lowest_free_mib
        parts = []
        if self.headroom_mib:
            parts.append("stop below %d MiB" % self.headroom_mib)
        if self.warn_mib:
            parts.append("warning below %d MiB (%d warning%s)" % (self.warn_mib, self.warnings, "" if self.warnings == 1 else "s"))
        return "commit headroom: %s, %s" % (", ".join(parts), low)
