"""Check the launcher's --map-drive option (#1158).

usage: startup_map_drive_check.py <consumer.exe> <debug_launch.py> <scratch dir> [overlap]

A console executable opens its package at a fixed drive letter. The option rewrites an open on a
mapped letter to a real folder, in the caller's own path structure, so the package can live on any
volume.

1. Mapped letter: the consumer opens a file through a letter that is not a real volume and reads it,
   and the trace shows the rewrite.
2. Negative control: without the option the same open fails.
3. Unmapped letter: with the option, an open on a letter that is not mapped is left alone.
4. Reuse: one caller-owned name over 32 serial opens is restored after every return.
5. (overlap argument) Two threads, 32 opens each, with their own caller-owned names in flight at once:
   all 64 opens must see the mapped name and succeed, every caller string must be back as built, and
   the debugger must report no unprotected step window; three times. The control runs the same case
   with --suspend-others 0 (the other thread is not held): it must lose at least one open in 8 runs.
"""
import os
import re
import shutil
import subprocess
import sys

consumer, launcher, scratch = sys.argv[1:4]
python = sys.executable
# The serial and the overlap check run as two CTests that may run at the same time: each owns its folders.
overlap_mode = len(sys.argv) > 4 and sys.argv[4] == "overlap"
tag = "overlap" if overlap_mode else "serial"
failures = []


def free_letter():
    """A letter that is not a mounted volume, so only the rewrite can make the open succeed."""
    for letter in "ZYXWVUTSRQPONMLKJIHG":
        if not os.path.exists(letter + ":\\"):
            return letter
    return None


def run(letter, map_spec, log_name, mode="", extra=()):
    log = os.path.join(scratch, f"map_drive_{tag}_{log_name}")
    shutil.rmtree(log, ignore_errors=True)
    env = dict(os.environ, X2W_PROBE_DRIVE=letter)
    if mode:
        env["X2W_PROBE_MODE"] = mode
    command = [python, "-B", launcher, consumer, scratch, log, "--timeout", "15", "--memory-gib", "2",
               "--trace-files", "1", "--continue-asserts", "0"]
    command += list(extra)
    if map_spec:
        command += ["--map-drive", map_spec]
    launcher_out = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120).stdout
    stdout_path = os.path.join(log, "stdout.txt")
    files_path = os.path.join(log, "files.txt")
    stdout = open(stdout_path, encoding="utf-8", errors="replace").read() if os.path.exists(stdout_path) else ""
    trace = open(files_path, encoding="utf-8", errors="replace").read() if os.path.exists(files_path) else ""
    return stdout, trace, launcher_out


letter = free_letter()
if not letter:
    print("no free drive letter to map; skipping")
    sys.exit(77)

package = os.path.join(scratch, f"map_drive_pkg_{tag}")
shutil.rmtree(package, ignore_errors=True)
os.makedirs(package)
with open(os.path.join(package, "map_drive_probe.txt"), "w", encoding="ascii", newline="") as handle:
    handle.write("mapped ok")
if overlap_mode:
    # The second thread's file has a longer name; the contents tell the files apart in the trace-free case.
    with open(os.path.join(package, "map_drive_probe_second_longer_name.txt"), "w", encoding="ascii", newline="") as handle:
        handle.write("second file")

if not overlap_mode:
    # The launcher exits 0 whether or not the child succeeded, so the child's own output is the signal.
    stdout, trace, _ = run(letter, f"{letter}={package}", "map_on")
    ok = "mapped ok" in stdout and "MAP " in trace and "map_drive_pkg" in trace
    print(f"mapped letter {letter}: {'ok' if ok else 'MISMATCH'} ({stdout.strip()!r})")
    if not ok:
        failures.append(f"mapped letter: stdout {stdout.strip()!r}, trace has MAP: {'MAP ' in trace}")

    stdout, trace, _ = run(letter, "", "map_off")
    ok = "open failed" in stdout and "mapped ok" not in stdout
    print(f"negative control without the option: {'ok' if ok else 'MISMATCH'} ({stdout.strip()!r})")
    if not ok:
        failures.append(f"negative control did not fail as expected: stdout {stdout.strip()!r}")

    stdout, trace, _ = run("C", f"{letter}={package}", "map_unmapped")
    ok = "MAP " not in trace
    print(f"unmapped letter C: {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append("an open on an unmapped letter was rewritten")

    # One caller-owned UNICODE_STRING through many opens: the borrowed name has to be given back after
    # every return, and the backing allocation must be reused rather than grown per open.
    stdout, trace, _ = run(letter, f"{letter}={package}", "map_reuse", mode="reuse")
    name_buffers = trace.count("NAMEBUF alloc")
    ok = "reuse ok" in stdout and name_buffers <= 1 and trace.count("MAP ") >= 32
    print(f"reused caller name over 32 opens: {'ok' if ok else 'MISMATCH'} "
          f"(name allocations={name_buffers}, rewrites={trace.count('MAP ')})")
    if not ok:
        failures.append(f"reuse: stdout {stdout.strip()!r}, name allocations {name_buffers}, "
                        f"rewrites {trace.count('MAP ')}")


OPENS = 64  # two threads, 32 opens each


def overlap(label, extra):
    """One two-thread run: (every open mapped and succeeded and every caller name restored, text, unprotected)."""
    stdout, trace, launcher_out = run(letter, f"{letter}={package}", label, mode="overlap", extra=extra)
    def mapped_opens(name):
        return sum(1 for line in trace.splitlines() if " NtCreateFile " in line and "(mapped from " in line
                   and line.rstrip().endswith("-> 0x00000000") and f"\\{name} from " in line)
    first, second = mapped_opens("map_drive_probe.txt"), mapped_opens("map_drive_probe_second_longer_name.txt")
    mapped_ok = first + second
    rewrites = trace.count("MAP ")
    summary = re.search(r"hook windows: entries=\d+ windows=\d+ unprotected=(\d+)", launcher_out + trace)
    unprotected = int(summary.group(1)) if summary else -1
    good = f"overlap ok: {OPENS} opens across two threads, both caller names intact" in stdout \
        and first == second == OPENS // 2 and rewrites == OPENS
    return good, f"{stdout.strip()!r} (mapped opens traced {first} + {second}, rewrites {rewrites})", unprotected


if overlap_mode:
    # With the other threads held for each step, every one of the 64 opens is rewritten, succeeds and gets
    # its own caller-owned name back; the debugger reports no unprotected window. Three times.
    for attempt in range(3):
        good, text, unprotected = overlap(f"held{attempt}", [])
        print(f"two overlapping threads, attempt {attempt}: {'ok' if good else 'MISMATCH'} {text} unprotected={unprotected}")
        if not good or unprotected != 0:
            failures.append(f"overlap attempt {attempt}: {text}, unprotected={unprotected}")
    # Control (the broken copy): the same case without holding the other thread loses opens, and the
    # debugger says so. If it never does in 8 runs the overlap was not exercised and the case above proves nothing.
    lost = None
    for attempt in range(8):
        good, text, unprotected = overlap(f"free{attempt}", ["--suspend-others", "0"])
        if not good:
            if unprotected <= 0:
                failures.append(f"control attempt {attempt} lost opens but reported no unprotected window: {text}")
            lost = text
            break
    print(f"control without holding the other thread: {'lost opens: ' + lost if lost else 'NEVER LOST'}")
    if lost is None:
        failures.append("the control never lost an open in 8 runs: the overlap was not exercised")

shutil.rmtree(package, ignore_errors=True)
if failures:
    print("FAIL")
    for text in failures:
        print("  " + text)
    sys.exit(1)
print("PASS")
