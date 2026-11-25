"""Main-image TLS slot 0: a /GA EXE under a Windows launch, and under pe_map_run_main with explicit participants.

cpu_exe_tls_slot0 is a CRT-free EXE compiled with /GA, so its thread-local accesses read slot 0 of
ThreadLocalStoragePointer without loading _tls_index (tests/cpu/exe_tls_slot0.cpp). Its TLS callback and
each of its threads (the initial thread and two workers) check whether slot 0 holds the image's block;
only then does a thread check its template copy, write the whole 0x1400-byte array and validate the heap.
The rule and the fixture shape follow MoonYueya's notes on issue #870.

Two modes, registered as two tests, keep the evidence separate:

windows: the EXE launched by Windows, where it owns index 0. The log is the expected guest log.

mapped: cpu_exe_tls_slot0_host, a CRT-free host whose own TLS block owns slot 0, maps the EXE and starts
it in one process per scenario (tests/cpu/exe_tls_slot0_host.cpp). In every scenario a reader thread that
never participates samples its own slot 0 and sentinel from before the map until after release, and a late
host thread created after the start checks its own slot 0 and sentinel; neither may see a change. Under
pe_map_run_main the guest's TLS callback must not run for the late thread (it does not participate), and
its thread-attach/detach totals must count only the guest's own workers; under pe_map_run (plain) the late
thread's attach and detach still reach the guest, as before. A refused release returns 0 with the status
API reporting the refusal; the accepted release is the only one that returns 1.
Process image: the guest's process-attach callback reports whether GetModuleHandleW(nullptr) is its own base
(the third m1 result), as under a Windows launch; under pe_map_run_main it must be, under pe_map_run it must
not. The host reads GetModuleHandleW(nullptr) after the start, after a refused release and after the accepted
release: the mapped base until the accepted release, including across a refusal with a worker paused inside
the image, and its own base afterwards; under pe_map_run the host's base throughout.
- main: pe_map_run_main; the guest log equals the Windows log; a host thread enters and leaves; release
  is accepted once.
- alive: a host participant is still entered at the first release, which returns 0 and changes nothing;
  after it leaves, the second release is accepted.
- exited: a host participant is ended with TerminateThread. Release confirms the exit from its thread
  handle and is accepted.
- untracked: MAIN_TRACKED host threads enter, so the last finds the participant table full and has no
  handle. It cannot be inspected, so the first release is refused until it leaves (the deterministic
  inspection-failure case).
- paused: cpu_exe_tls_slot0_pause's worker c pauses inside the image's code. The first release is refused
  with the image still committed, its unwind data registered, its carrier loaded and slot 0 unchanged; c
  then resumes, finds its own block and values, returns, and the second release is accepted.
- plain (negative control): pe_map_run leaves the host's block in slot 0, so every slot-0 check fails.
Every accepted release must leave the image freed, its unwind data gone and its carrier unloaded, and a
further release must return 0. Under pe_map_run_main the host binds the guest's CreateThread import to
pe_map_main_create_thread, so the guest's workers are enrolled explicitly.

enroll: guest-thread enrollment failures, each in its own child host process with
cpu_exe_tls_slot0_enroll, whose probe worker p touches a global sentinel first in its start routine.
- probe (control): p is enrolled, touches the sentinel and passes every check.
- enroll_refused: p's enrollment is made to fail. The guest's CreateThread returns null with
  ERROR_NOT_ENOUGH_MEMORY, no thread exists, the sentinel stays 0 and the refusal is counted.
- install_failed: the installation on p is made to fail. p ends with 0xE0000201 before its start routine,
  the sentinel stays 0, the guest gets no callback for p and the failure is counted.
- resume_failed: the resume of p inside the bound CreateThread is made to fail. The guest's CreateThread
  returns null with ResumeThread's error (ERROR_ACCESS_DENIED), never a suspended handle; p is ended before
  it runs, the sentinel stays 0, the refusal is counted, and workers a and b are enrolled afterwards.
- cancelled: the host creates and cancels 65 suspended enrolled workers, one at a time. Every worker
  exits without running its start routine, the table keeps admitting workers, and cleanup accounts for
  all 65 exits without changing the participating callback totals.
In every failure case the guest's later workers are enrolled as usual, the host survives and prints its
report, and every other check (reader, late thread, release, cleanup) holds.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

TIMEOUT = 60
EXPECTED = "m1=yyy m:yyyy a2=yy a:yyyy a3=y- b2=yy b:yyyy b3=y- X=0"
PLAIN = "m1=n-n m:n--- a2=n- a:n--- a3=n- b2=n- b:n--- b3=n- X=9"
MAIN_TRACKED = 64
PAUSED = EXPECTED.replace(" X=0", " c2=yy c:yyyy X=0")
PROBED = EXPECTED.replace(" a2=", " p2=yy p:yyyy p3=y- a2=")
ERROR_NOT_ENOUGH_MEMORY = 8
ENROLL_FAILED_EXIT = 0xE0000201
ERROR_ACCESS_DENIED = 5
# stats: installed, restored, exited, refused releases, pending, active, last release (1 accepted), untracked,
# enrollments refused (no thread created), enrolled threads ended without the guest's start routine.
# Participants: the starting thread, the guest's workers (a, b and, paused, c) and the host's participants.
# late: the guest's thread callbacks during the late thread's life; callbacks: the guest's thread-attach and
# thread-detach totals before the accepted release (its own workers only, plus the late thread under plain).
SCENARIOS = {
    "main": {"stats": [4, 4, 0, 0, 0, 0, 1, 0, 0, 0], "refused": None, "participant": True, "resumed": None,
             "late": 0, "callbacks": "2,2"},
    "alive": {"stats": [4, 4, 0, 1, 0, 0, 1, 0, 0, 0], "refused": True, "participant": True, "resumed": None,
              "late": 0, "callbacks": "2,2"},
    "exited": {"stats": [4, 3, 1, 0, 0, 0, 1, 0, 0, 0], "refused": None, "participant": True, "resumed": None,
               "late": 0, "callbacks": "2,2"},
    "untracked": {"stats": [3 + MAIN_TRACKED, 3 + MAIN_TRACKED, 0, 1, 0, 0, 1, 0, 0, 0], "refused": True,
                  "participant": True, "resumed": None, "late": 0, "callbacks": "2,2"},
    "paused": {"stats": [5, 5, 0, 1, 0, 0, 1, 0, 0, 0], "refused": True, "participant": True, "resumed": True,
               "late": 0, "callbacks": "3,3"},
    "plain": {"stats": [0, 0, 0, 0, 0, 0, -1, 0, 0, 0], "refused": None, "participant": None, "resumed": None,
              "late": 2, "callbacks": "3,3"},
}
# probe: sentinel, CreateThread returned a handle, its last error, p's exit code.
ENROLL = {
    "probe": {"stats": [5, 5, 0, 0, 0, 0, 1, 0, 0, 0], "refused": None, "participant": True, "resumed": None,
              "late": 0, "callbacks": "3,3", "log": PROBED, "probe": [1, 1, 0, 0]},
    "enroll_refused": {"stats": [4, 4, 0, 0, 0, 0, 1, 0, 1, 0], "refused": None, "participant": True,
                       "resumed": None, "late": 0, "callbacks": "2,2", "log": EXPECTED,
                       "probe": [0, 0, ERROR_NOT_ENOUGH_MEMORY, 0]},
    "install_failed": {"stats": [4, 4, 0, 0, 0, 0, 1, 0, 0, 1], "refused": None, "participant": True,
                       "resumed": None, "late": 0, "callbacks": "2,2", "log": EXPECTED,
                       "probe": [0, 1, 0, ENROLL_FAILED_EXIT]},
    "resume_failed": {"stats": [4, 4, 0, 0, 0, 0, 1, 0, 1, 0], "refused": None, "participant": True,
                      "resumed": None, "late": 0, "callbacks": "2,2", "log": EXPECTED,
                      "probe": [0, 0, ERROR_ACCESS_DENIED, 0]},
    "cancelled": {"stats": [5, 5, 0, 0, 0, 0, 1, 0, 0, 65], "refused": None, "participant": True,
                  "resumed": None, "late": 0, "callbacks": "3,3", "log": PROBED,
                  "probe": [1, 1, 0, 0]},
}


def run(command):
    done = subprocess.run(command, capture_output=True, timeout=TIMEOUT)
    text = done.stdout.decode(errors="replace")
    result = {"exit": done.returncode & 0xffffffff, "stderr": done.stderr.decode(), "stdout": text,
              "log": next((line[4:].strip() for line in text.splitlines() if line.startswith("LOG:")), None),
              "host": next((line for line in text.splitlines() if line.startswith("HOST:")), None)}
    if result["host"] and "setup failed" not in result["host"]:
        fields = dict(re.findall(r"(\w+)=(\S+)", result["host"]))
        result["fields"] = fields
        result["stats"] = [int(v) for v in fields["stats"].split(",")]
    return result


def flag(fields, name):
    return {"y": True, "n": False, "-": None}[fields[name]]


def windows(exe):
    with TemporaryDirectory() as temp:
        copy = Path(temp) / Path(exe).name
        shutil.copy(exe, copy)
        result = run([str(copy)])
    print(f"windows: LOG:{result['log']} exit {result['exit']:#x}")
    if (result["log"], result["exit"], result["stderr"]) != (EXPECTED, 0, ""):
        raise AssertionError(f"Windows launch: {result}")
    print("windows: the EXE owns index 0, so slot 0 holds its block on every thread and callback; "
          "template intact, whole-array writes read back, heap valid")


def check(mapper, image, host, scenario, want):
    result = run([str(host), str(mapper), str(image), scenario])
    print(f"{scenario}: LOG:{result['log']}")
    print(f"{scenario}: {result['host']}")
    # A host that died (for example on a guest-thread failure) prints no report and exits nonzero.
    if "fields" not in result or result["exit"] or result["stderr"]:
        raise AssertionError(f"{scenario}: host failed: {result}")
    fields = result["fields"]
    problems = []
    log = want["log"]
    if result["log"] != log:
        problems.append(f"guest log {result['log']!r}, expected {log!r}")
    if fields["started"] != "1" or int(fields["exit"], 16) != (9 if scenario == "plain" else 0):
        problems.append(f"started {fields['started']}, guest exit {fields['exit']}")
    if flag(fields, "installed_main") != (scenario != "plain"):
        problems.append(f"installed_main {fields['installed_main']}")
    if not flag(fields, "reader_untouched"):
        problems.append("the non-participating reader saw its slot 0 or sentinel change")
    if flag(fields, "participant") != want["participant"]:
        problems.append(f"participant {fields['participant']}")
    if not flag(fields, "host_tls") or not flag(fields, "heap_valid"):
        problems.append(f"host_tls {fields['host_tls']}, heap_valid {fields['heap_valid']}")
    if flag(fields, "refused") != want["refused"] or flag(fields, "intact_on_refusal") != want["refused"]:
        problems.append(f"refused {fields['refused']}, intact_on_refusal {fields['intact_on_refusal']}: "
                        f"expected {'a refusal with image, unwind data and carrier intact' if want['refused'] else 'none'}")
    if flag(fields, "resumed") != want["resumed"]:
        problems.append(f"resumed {fields['resumed']}")
    if not flag(fields, "late_untouched"):
        problems.append("the late host thread saw its slot 0 or sentinel change")
    if int(fields["late_callbacks"]) != want["late"]:
        problems.append(f"the guest received {fields['late_callbacks']} thread callbacks for the late host "
                        f"thread, expected {want['late']}")
    if fields["callbacks"] != want["callbacks"]:
        problems.append(f"guest thread-attach,thread-detach callbacks {fields['callbacks']}, "
                        f"expected {want['callbacks']}")
    # The process image: the mapped EXE from before its callbacks until the accepted release, including across a
    # refused one; the host's again afterwards. Under plain pe_map_run it never changes.
    identity = ("h" if scenario == "plain" else "i") + ("i" if want["refused"] else "-") + "h"
    if fields["process_image"] != identity:
        problems.append(f"process_image {fields['process_image']} (after start, refusal, release), expected {identity}")
    if not flag(fields, "cleaned") or not flag(fields, "second_release"):
        problems.append(f"cleaned {fields['cleaned']}, second_release {fields['second_release']}: expected one "
                        "final cleanup (image freed, unwind data gone, carrier unloaded) and a further release 0")
    if result["stats"] != want["stats"]:
        problems.append(f"counters {result['stats']} != {want['stats']} (installed, restored, exited, "
                        "refused, pending, active, last release, untracked, enrollments refused, "
                        "enrolled threads failed)")
    if "probe" in want:
        probe = None
        if "probe" in fields:
            probe = [int(v, 16) if i == 3 else int(v) for i, v in enumerate(fields["probe"].split(","))]
        if probe != want["probe"]:
            problems.append(f"probe {fields.get('probe')} (sentinel, created, last error, exit code), "
                            f"expected {want['probe'][:3]} and {want['probe'][3]:#x}")
    if problems:
        raise AssertionError(f"{scenario}: " + "; ".join(problems))


def mapped(mapper, exe, host, paused_exe):
    for scenario, want in SCENARIOS.items():
        want = {**want, "log": {"plain": PLAIN, "paused": PAUSED}.get(scenario, EXPECTED)}
        check(mapper, paused_exe if scenario == "paused" else exe, host, scenario, want)
    print("mapped: the guest log equals the Windows log under pe_map_run_main; the non-participating reader's "
          "and the late thread's slot 0 and sentinel never changed, and the guest's callbacks never ran for the late "
          "thread; a participant still entered, without a handle, or paused inside the image made the first release "
          "return 0 with the refusal reported and the image, unwind data and carrier intact; once it left, one "
          "release returned 1 and freed everything and a further release returned 0; an exited participant was "
          "confirmed from its handle; plain pe_map_run fails every slot-0 check and still notifies the late thread")


def enroll(mapper, exe, host):
    for scenario, want in ENROLL.items():
        check(mapper, exe, host, scenario, want)
    print("enroll: a refused enrollment made the guest's CreateThread return null with no thread created, a failed "
          "resume made it return null with ResumeThread's error after ending the suspended thread, and a failed "
          "installation ended the thread with 0xE0000201 before its start routine; in each the sentinel the start "
          "routine touches first stayed 0, the guest got no callback for the thread, the failure was counted, later "
          "workers were enrolled as usual and the host process survived; the control touched the sentinel; 65 "
          "cancelled suspended workers were all reclaimed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("windows").add_argument("exe", type=Path)
    m = sub.add_parser("mapped")
    m.add_argument("mapper", type=Path)
    m.add_argument("exe", type=Path)
    m.add_argument("host", type=Path)
    m.add_argument("paused_exe", type=Path, help="the fixture built with PAUSED_WORKER")
    e = sub.add_parser("enroll")
    e.add_argument("mapper", type=Path)
    e.add_argument("exe", type=Path, help="the fixture built with ENROLL_PROBE")
    e.add_argument("host", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "windows":
        windows(arguments.exe)
    elif arguments.mode == "enroll":
        enroll(arguments.mapper, arguments.exe, arguments.host)
    else:
        mapped(arguments.mapper, arguments.exe, arguments.host, arguments.paused_exe)
    sys.exit(0)
