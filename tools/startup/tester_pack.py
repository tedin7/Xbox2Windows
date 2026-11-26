#!/usr/bin/env python3
"""Pure helpers of tools/startup/tester_run.ps1: the input preset, the manifest and the zip of one tester run.

  tester_pack.py input-preset <accept|none|off|menus|menus2|raw script> <seconds>      print the X2W_INPUT_SCRIPT value
  tester_pack.py cache-tag <gfx2hlsl.py>                        print the translation revisions, e.g. 28-28-1
  tester_pack.py cache-stats <cache dir>                        print "<translated> <requests> <failed>"
  tester_pack.py first-poll <logdir>                            print when the title first polled the pad
  tester_pack.py seed-cache <cache root> <new cache> [--jobs N]  copy the old caches' requests under the new keys (pre-translation backlog)
  tester_pack.py cache-results <cache dir>                      print the result text of every request key (ok / refused + reason / pending)
  tester_pack.py pack <logdir> <manifest_in.json> <zip> [--limit-mb N] [--cache DIR]   write the zip, print its path and size

The zip holds every file the run wrote in its log folder (stdout, stderr, events, finished, threads, files, trap.log,
translate, translate_err, window*.png, bink_luma.png ...), the debugger summary (`<logdir>.out`) and manifest.json.
Not in it: sub-folders such as a shader cache, memory dumps and game files; with --cache the zip also holds
`shader_results.txt`: for every request key of the cache its kind, ok / refused / pending and, for a refusal, the first line
of its `.fail` text (the translator's own message; a compiler message is cut to its error codes), ranked by reason, and for a key the
run used its first and last use in seconds since the process started (the refused keys used latest are listed first). Never
shader code or binaries. With --map (the linker map of our own d3d12_xs.dll) it also holds
`shader_requests/`: the request files (req_<key>.*) of the 40 worst refused keys, ranked by draws not drawn (graphics) or dispatches
without a translation (compute) from stderr.txt, at most 4 MB in all, listed in the manifest (`request_files`). Private data: mailed, never posted.

`d3d12_xs.symbols.txt`: one `RVA function` line per function of that DLL (never a game file), so a stack frame
d3d12_xs.dll+0xNNNN reads as a name. Nothing is dropped: past the size limit
the zip is split into numbered parts (<name>_part01of03.zip, ...) and a very large file into numbered pieces.
"""
import argparse
import re
import struct
import json
import os
import sys
import zipfile
import zlib

MAX_ENTRIES = 256          # X2W_INPUT_SCRIPT accepts at most 256 timed entries
MAX_SCRIPT_CHARS = 4095
MENUS_START_S = 5          # script time counts from the first pad poll (the default time base), not from process start
MENUS_STEP_S = 1           # one step (DOWN, DOWN, A pattern) every 2 s after the first press (Run K: 6 s was too slow)
MENUS_END_S = 300          # the preset stops here however long the run is
DPAD_UP = "40"             # GameInput button masks, hexadecimal
DPAD_DOWN = "80"
BUTTON_A = "4"
MENUS_HOLD_S = "0.8"       # a press is held this long (0.2 s released before the next one, a press every 1 s): at 28 fps a poll comes every 0.04 s; at 2 to 3 fps every 0.4 s, still two polls
MENUS2_HOLD_S = "1.2"
MENUS2_GAP_S = "2"         # between the presses of one cycle; the first press follows the previous wait by the same gap
MENUS2_SETTLE_S = 2        # the picture must stay unchanged this long after it changed
MENUS2_TIMEOUT_S = 90      # a screen that never changes (or loads slower) lets the script go on after this
EXCLUDED_SUFFIXES = (".dmp", ".mdmp", ".hdmp")
SUMMARY_NAME = "debugger_summary.out"
MANIFEST_KEYS = ("commit", "build_commit", "dirty", "seconds", "input_script", "env", "gpu", "gpu_driver",
                 "windows_build", "wall_seconds", "deep", "lazy_textures", "lazy_memory", "memory_gib", "shader_cache", "translations_at_start")
DEFAULT_LIMIT_MB = 20
SMALL_FILE_BYTES = 512 * 1024   # a file up to this size goes into the zip before the big logs, so part 1 holds all of them
MANIFEST_ROOM = 256 * 1024     # headroom per part for the manifest and zip bookkeeping
CHUNK_BYTES = 48 * 1024 * 1024  # a file larger than this is cut into numbered pieces


ACCEPT_MASK = "4"        # GameInput A: the only button proven accepted by the title (picker, art slides, black window). Never Start.
ACCEPT_FIRST_S = 35      # first press, seconds after the first pad poll
ACCEPT_EVERY_S = 10
ACCEPT_HOLD_S = "0.3"    # a tap
ACCEPT_LONG_HOLD_S = "1.5"   # every ACCEPT_LONG_EVERY-th press is a hold (skip presses for movies that read a held button)
ACCEPT_LONG_EVERY = 3
ACCEPT_END_S = 420       # no press from here on: the prologue is on screen (A in gameplay interacts, in the pause menu it selects settings)


def accept_presses(seconds):
    """[(at, hold)] of the `accept` preset: A at 35 s, then every 10 s while the press starts before min(run seconds, 420 s);
    every 3rd press (the 3rd, 6th, ...) is held 1.5 s instead of tapped 0.3 s."""
    presses, n = [], 0
    while ACCEPT_FIRST_S + n * ACCEPT_EVERY_S < min(seconds, ACCEPT_END_S) and n < MAX_ENTRIES:
        presses.append((ACCEPT_FIRST_S + n * ACCEPT_EVERY_S, ACCEPT_LONG_HOLD_S if (n + 1) % ACCEPT_LONG_EVERY == 0 else ACCEPT_HOLD_S))
        n += 1
    return presses


def accept_script(seconds):
    """`accept`: ONLY A, tap 0.3 s with every 3rd press held 1.5 s, first at 35 s after the first pad poll, then every 10 s, none
    from 420 s on and none after the run (accept_presses). The picker needs an A at 31-43 s; holds cover intro and credits movies."""
    return ",".join("%s@%d+%s" % (ACCEPT_MASK, at, hold) for at, hold in accept_presses(seconds))


def menus2_script():
    """`menus2`: the same presses as `menus` (down, down, A), but after every A the script waits for the presented
    picture to change and settle (docs/startup-runtime.md, "Waiting for the presented picture") before it presses again,
    so nothing is pressed blind while a screen loads. Each press is timed from the previous item (`@+`). A wait that
    times out logs it and the script goes on. Needs render mode, which tester_run.ps1 always uses. The cycle count fills
    the entry limit; the run's own length ends the script."""
    items = ["%s@%d+%s" % (DPAD_DOWN, MENUS_START_S, MENUS2_HOLD_S)]
    press = lambda mask: "%s@+%s+%s" % (mask, MENUS2_GAP_S, MENUS2_HOLD_S)
    cycle = [press(DPAD_DOWN), press(BUTTON_A), "wait=%d/%d/go" % (MENUS2_SETTLE_S, MENUS2_TIMEOUT_S)]
    while len(items) + 1 + len(cycle) <= MAX_ENTRIES:
        items.append(press(DPAD_DOWN))   # first down of the cycle
        items.extend(cycle)
    return ",".join(items)


def privacy_cycles(boxes):
    """The presses for the privacy screen (`boxes` checkboxes, then a Conferma button), as passes that never need to know the
    clock or the state of the boxes, fast: the first pass is the one that is right when nothing is ticked yet (tick every box,
    Conferma); if the screen is still there after it, the later passes toggle ONE box each, Gray-code style, so that over
    the whole round the boxes pass through every state with a press of Conferma at the end of each pass (a press on a box
    toggles it, so repeating a pass cannot repair a wrong state; whatever state the boxes have when the screen opens,
    one pass of the round reaches Conferma with every box ticked, and Conferma then changes the screen). Every pass starts
    with D-pad up `boxes` times: the focus anchors on the first box from any item (the list does not wrap; the
    previous pass ended on Conferma, the last of boxes + 1 items). On the legal-notice screen before it (two list items and
    an Avanti button) every pass only selects items and ends with A on Avanti, which opens the privacy screen.
    Returns the passes, each a list of button masks. Two boxes: 7 + 6 + 6 + 6 presses, a press every second."""
    def one_pass(toggles):
        presses = [DPAD_UP] * boxes
        at = 0
        for box in sorted(toggles):
            presses += [DPAD_DOWN] * (box - at) + [BUTTON_A]
            at = box
        presses += [DPAD_DOWN] * (boxes - at) + [BUTTON_A]
        return presses
    gray = [((i & -i).bit_length() - 1) for i in range(1, 2 ** boxes)]
    # the round starts with "tick all"; the following passes then visit the other states (the XOR of the toggles of the
    # passes so far, starting from all toggled, runs through all 2^boxes values: all, then each Gray step)
    return [one_pass(list(range(boxes)))] + [one_pass([b]) for b in gray]


def privacy_labels(boxes):
    """One label per press of privacy_cycles(boxes), in the same order: what the press is for, so a log can say
    'privacy pass 2: toggle box 1' instead of a button mask."""
    labels = []
    gray = [((i & -i).bit_length() - 1) for i in range(1, 2 ** boxes)]
    for number, toggled in enumerate([list(range(boxes))] + [[b] for b in gray], 1):
        head = "privacy pass %d" % number
        verb = "tick" if number == 1 else "toggle"
        labels += ["%s: up (anchor on box 1)" % head] * boxes
        at = 0
        for box in sorted(toggled):
            labels += ["%s: down to box %d" % (head, k + 2) for k in range(at, box)]
            labels.append("%s: %s box %d" % (head, verb, box + 1))
            at = box
        labels += ["%s: down to %s" % (head, "Conferma" if target == boxes else "box %d" % (target + 1))
                   for target in range(at + 1, boxes + 1)]
        labels.append("%s: Conferma (Avanti on the legal notice)" % head)
    return labels


NO_INPUT = ("none", "off", "")  # input presets that press nothing


def input_labels(spec, seconds, privacy_boxes=2):
    """One intent label per timed press of input_preset(spec, seconds, privacy_boxes), in script order (entry 1 first, as the
    `x2w-input script press item N` log lines count). Only the `menus` preset is labelled; anything else gives []."""
    if spec == "accept":
        return ["press A #%d (accept, %s)" % (n + 1, "hold %s s" % hold if hold == ACCEPT_LONG_HOLD_S else "tap")
                for n, (_, hold) in enumerate(accept_presses(seconds))]
    if spec != "menus":  # none, off, menus2 and raw scripts have no labelled passes
        return []
    count = len(input_preset(spec, seconds, privacy_boxes).split(",")) if input_preset(spec, seconds, privacy_boxes) else 0
    if privacy_boxes > 0:
        cycle = privacy_labels(privacy_boxes)
    else:
        cycle = ["menu: down", "menu: down", "menu: A"]
    return [cycle[i % len(cycle)] for i in range(count)]


def input_preset(spec, seconds, privacy_boxes=2):
    """`none` (or `off`, or an empty value) is no automatic press at all: an empty script, so the title only sees the tester's own
    input. `menus` is D-pad down, D-pad down, A, one press every second from 5 s after the first pad poll; entries run to
    `seconds` of script time, but not past 300 s, at most 256 entries (about 260 s at one press a second). With privacy_boxes > 0 (the default; the privacy
    screen has two boxes) the presses are instead privacy_cycles() repeated, which also carries the title through the
    screens before the privacy screen; privacy_boxes = 0 keeps the plain pattern. Anything else is a raw script."""
    if spec.strip().lower() in NO_INPUT:
        return ""
    if spec == "accept":
        return accept_script(seconds)
    if spec == "menus2":
        return menus2_script()
    if spec != "menus":
        if len(spec) > MAX_SCRIPT_CHARS:
            raise ValueError("input script longer than %d characters" % MAX_SCRIPT_CHARS)
        if len([item for item in spec.split(",") if item.strip()]) > MAX_ENTRIES:
            raise ValueError("input script has more than %d entries" % MAX_ENTRIES)
        return spec
    cycle = ([m for one in privacy_cycles(privacy_boxes) for m in one] if privacy_boxes > 0
             else [DPAD_DOWN, DPAD_DOWN, BUTTON_A])
    items, press, at = [], 0, MENUS_START_S
    while at < min(seconds, MENUS_END_S) and len(items) < MAX_ENTRIES:
        items.append("%s@%d+%s" % (cycle[press % len(cycle)], at, MENUS_HOLD_S))
        press += 1
        at += MENUS_STEP_S
    return ",".join(items)


def validate_manifest(manifest):
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError("manifest is missing: " + ", ".join(missing))
    for key in manifest["env"]:
        if not key.startswith("X2W_"):
            raise ValueError("manifest env holds a non X2W_ variable: " + key)
    return manifest


REQUEST_RE = re.compile(r"^req_([0-9a-f]{16})\.(a|rtv|c67|m57)$")


MEMORY_FRACTION = 0.75     # of physical RAM: the launcher's job memory cap (debug_launch --memory-gib)
MEMORY_MIN_GIB = 14        # run.ps1's own default


def memory_gib(physical_bytes):
    """Job memory cap for the run: 75% of physical RAM, whole GiB, never below run.ps1's default of 14."""
    return max(MEMORY_MIN_GIB, int(physical_bytes / float(1 << 30) * MEMORY_FRACTION))


def cache_tag(translator):
    """Name suffix of the shader cache folder: the translation revisions of gfx2hlsl.py (graphics, compute, mesh 57),
    e.g. `28-28-1` for x2w-startup-shader-28, x2w-startup-shader-28cs and x2w-startup-mesh-57-1. A branch update that
    leaves these alone keeps its warm cache; a bump of any of them starts a new folder."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import shader_cache_revision as R
    found = R.translator_revisions(translator)
    numbers = []
    for kind in R.KINDS:
        match = re.search(r"([0-9]+)(?:cs)?$", found.get(kind, ""))
        numbers.append(match.group(1) if match else "x")
    return "-".join(numbers)


def _non_empty(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _complete_dxbc(path):
    """A DXContainer whose header length matches the file (gfx2hlsl.py has_complete_mesh57_bytecode)."""
    try:
        with open(path, "rb") as shader:
            header = shader.read(32)
            if len(header) != 32 or header[:4] != b"DXBC":
                return False
            size, parts = struct.unpack_from("<II", header, 24)
            return size == os.fstat(shader.fileno()).st_size and 0 < parts <= (size - 32) // 4
    except OSError:
        return False


def _translated(cache_dir, key, kind):
    """The translator's own test for a finished request (gfx2hlsl.py pending_keys): the result manifest
    `<key>.txt` (`<key>.ms57.txt` for a mesh pipeline) is published and the bytecode it describes is on disk, non-empty:
    `.vs.cso` + `.ps.cso` (pixel pipelines), `.cs.cso` (compute), `.ms57.cso` + `.ps57.cso`, both complete DXContainers (mesh; a pixel stage the
    manifest says is `ps none` needs no `.ps57.cso`). The request file itself is `req_<key>.<kind>`."""
    base = os.path.join(cache_dir, key)
    if kind == "m57":
        manifest = base + ".ms57.txt"
        if not (os.path.isfile(manifest) and _complete_dxbc(base + ".ms57.cso")):
            return False
        if _complete_dxbc(base + ".ps57.cso"):
            return True
        with open(manifest, "rb") as handle:
            return b"ps none" in handle.read().splitlines()
    stages = (".cs.cso",) if kind == "c67" else (".vs.cso", ".ps.cso")
    return os.path.isfile(base + ".txt") and all(_non_empty(base + stage) for stage in stages)


def cache_stats(cache_dir):
    """(translations ready, requests, failures) in a shader cache folder, counted per request file req_<key>.<kind>:
    ready = the translator's finished-request test passes (manifest + bytecode, see _translated); failed = not ready and
    `<key>.fail` exists (a refusal or compiler error; the next pass retries it). The rest is pending."""
    ready = requests = failed = 0
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            match = REQUEST_RE.match(name)
            if not match:
                continue
            requests += 1
            if _translated(cache_dir, match.group(1), match.group(2)):
                ready += 1
            elif os.path.isfile(os.path.join(cache_dir, match.group(1) + ".fail")):
                failed += 1
    return ready, requests, failed


def request_count(cache_dir):
    """Request keys (ready markers .a / .rtv / .c67 / .m57) in a shader cache folder."""
    if not os.path.isdir(cache_dir):
        return 0
    return sum(1 for name in os.listdir(cache_dir) if REQUEST_RE.match(name))


def older_caches(cache_root, new_cache):
    """The other shader_cache_* folders of the tester's cache root, most recently changed first."""
    new = os.path.normcase(os.path.abspath(new_cache))
    found = []
    if os.path.isdir(cache_root):
        for name in os.listdir(cache_root):
            path = os.path.join(cache_root, name)
            if name.startswith("shader_cache_") and os.path.isdir(path) and os.path.normcase(os.path.abspath(path)) != new:
                found.append((os.path.getmtime(path), path))
    return [path for _, path in sorted(found, reverse=True)]


def seed_cache(cache_root, new_cache, jobs=0, rekey=None):
    """Pre-translation of an old cache after a shader revision bump. When the new cache folder holds no request yet, the
    request files of every older shader_cache_* folder are copied into it under the keys of the current revision
    (shader_cache_rekey.py: re-hashed with this gfx2hlsl.py, ready marker last, requests only, no result file of the old
    revision). The --watch translator the run starts then finds them as a backlog and translates them in its process pool
    while the title boots; requests the title publishes meanwhile go first. Never raises and never blocks a run: any
    problem is returned as `skipped`/`failed` text.

    Returns {"seeded": new requests, "sources": older caches used, "seconds": time spent, "skipped": reason or None,
    "failed": [(older cache, first line of the error)]}."""
    import subprocess
    import time
    started = time.monotonic()
    result = {"seeded": 0, "sources": 0, "seconds": 0.0, "skipped": None, "failed": []}
    here = os.path.dirname(os.path.abspath(__file__))
    rekey = rekey or os.path.join(here, "shader_cache_rekey.py")
    try:
        olds = older_caches(cache_root, new_cache)
        if request_count(new_cache):
            result["skipped"] = "the new cache already holds requests"
        elif not olds:
            result["skipped"] = "no older shader_cache_* folder"
        elif not os.path.isfile(rekey):
            result["skipped"] = "shader_cache_rekey.py is missing"
        else:
            env = dict(os.environ)
            if jobs > 0:
                env["X2W_TRANSLATE_JOBS"] = str(jobs)
            os.makedirs(new_cache, exist_ok=True)
            for old in olds:
                done = subprocess.run([sys.executable, "-B", rekey, old, new_cache], capture_output=True, text=True, env=env)
                if done.returncode == 0:
                    result["sources"] += 1
                else:
                    text = (done.stderr or done.stdout or "").strip().splitlines()
                    result["failed"].append((os.path.basename(old), text[0][:160] if text else "exit %d" % done.returncode))
            result["seeded"] = request_count(new_cache)
            if not result["sources"]:
                result["skipped"] = "every older cache was refused"
    except Exception as error:  # noqa: BLE001 - seeding must never stop a tester run
        result["skipped"] = "%s: %s" % (type(error).__name__, str(error)[:120])
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def seed_line(result):
    if result["skipped"]:
        return "pre-translate: skipped (%s)" % result["skipped"] + "".join("; %s: %s" % item for item in result["failed"])
    return "pre-translate: %d requests from %d older cache(s) in %g s%s" % (
        result["seeded"], result["sources"], result["seconds"],
        "".join("; skipped %s: %s" % item for item in result["failed"]))


RESULTS_NAME = "shader_results.txt"
KEY_RE = re.compile(r"(?:^|[^0-9a-f])([0-9a-f]{16})(?:[^0-9a-f]|$)")
# request file extension -> kind; result marker: ready file suffix per kind, `.fail` for a refusal
REQUEST_KINDS = {".a": "graphics", ".c67": "compute", ".m57": "mesh57", ".m86": "mesh86"}
READY_SUFFIXES = {"graphics": (".txt",), "compute": (".txt",), "mesh57": (".ms57.txt",), "mesh86": (".mesh86.txt",)}
REASON_CHARS = 200         # per key; the translator's own message, first line only
COMPILER_CODE_RE = re.compile(r"\b(?:error|warning)\s+[A-Z]\d{3,5}\b")
TOP_REASONS = 80


def clean_reason(text):
    """One line of text from a `.fail` file, never shader code: the first line only, at most REASON_CHARS characters.
    A compiler (dxc, fxc) message quotes the HLSL source, so only its stage and error codes are kept."""
    line = text.strip().splitlines()[0].strip() if text.strip() else "(empty .fail)"
    compiler = re.match(r"(dxc|fxc)( \w+)?:", line)
    if compiler:
        codes = sorted(set(COMPILER_CODE_RE.findall(text)))
        return ("%s compile failed" % compiler.group(0).rstrip(":")) + (" (" + ", ".join(codes[:6]) + ")" if codes else "")
    return re.sub(r"\s+", " ", line)[:REASON_CHARS]


def normal_reason(reason):
    """The reason without registers, numbers and addresses, so that the same wall counts as one."""
    reason = re.sub(r"0x[0-9a-fA-F]+", "X", reason)
    reason = re.sub(r"\b[sv]\d+\b", "R", reason)
    return re.sub(r"\d+", "N", reason)[:110]


def cache_results(cache_dir):
    """[(key, kind, status, reason)] for every request key in a shader cache folder, sorted by kind and key. Status is
    `ok` (the ready marker is published), `refused` (a `.fail` file: its first line is the reason) or `pending`."""
    found, by_key = {}, {}
    if not os.path.isdir(cache_dir):
        return []
    for name in sorted(os.listdir(cache_dir)):
        match = KEY_RE.search(name)
        if not match:
            continue
        by_key.setdefault(match.group(1), []).append(name)
        ext = os.path.splitext(name)[1]
        if ext in REQUEST_KINDS:
            found[match.group(1)] = REQUEST_KINDS[ext]
    rows = []
    for key, kind in found.items():
        status, reason = "pending", ""
        for name in by_key[key]:
            if name.endswith(".fail"):
                with open(os.path.join(cache_dir, name), "r", encoding="utf-8", errors="replace") as handle:
                    status, reason = "refused", clean_reason(handle.read(4096))
                break
            if name.endswith(READY_SUFFIXES[kind]) and not name.endswith(".fail"):
                status = "ok"
        rows.append((key, kind, status, reason))
    return sorted(rows, key=lambda row: (row[1], row[0]))


KEY_USE_RE = re.compile(r"^x2w-render key use (graphics|compute) ([0-9a-f]{16}) first ([0-9.]+) s last ([0-9.]+) s uses (\d+)\s*$")
LATEST_USED_KEYS = 40


def key_use(logdir):
    """{key: (first_seconds, last_seconds, uses)} from the runtime's 'x2w-render key use' lines in stderr.txt: the earliest
    first, the latest last and the largest count of each key (a line is printed again only when the count grew)."""
    found = {}
    path = os.path.join(logdir, "stderr.txt") if logdir else None
    if not path or not os.path.isfile(path):
        return found
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("x2w-render key use "):
                continue
            match = KEY_USE_RE.match(line)
            if match:
                first, last, uses = float(match.group(3)), float(match.group(4)), int(match.group(5))
                old = found.get(match.group(2))
                found[match.group(2)] = (min(first, old[0]), max(last, old[1]), max(uses, old[2])) if old else (first, last, uses)
    return found


def format_results(rows, use=None):
    """Text for the zip: totals, the refusals ranked by normalized reason, the refused keys used latest, then one line per
    key. use is key_use(): a key a draw or Dispatch used carries first=, last= (seconds since the process started) and uses=."""
    use = use or {}
    kinds = {}
    for _, kind, _, _ in rows:
        kinds[kind] = kinds.get(kind, 0) + 1
    count = lambda status: sum(1 for row in rows if row[2] == status)
    lines = ["# shader cache results: the translator's own messages, first line of each .fail, no shader code",
             "keys %d (%s): ok %d, refused %d, pending %d" % (
                 len(rows), ", ".join("%s %d" % item for item in sorted(kinds.items())) or "none",
                 count("ok"), count("refused"), count("pending")),
             "", "## refusals by reason (registers and numbers normalized)"]
    ranked = {}
    for _, kind, status, reason in rows:
        if status == "refused":
            entry = ranked.setdefault(normal_reason(reason), {})
            entry[kind] = entry.get(kind, 0) + 1
    for reason, by_kind in sorted(ranked.items(), key=lambda item: (-sum(item[1].values()), item[0]))[:TOP_REASONS]:
        lines.append("%5d  %s   [%s]" % (sum(by_kind.values()), reason, ", ".join("%s %d" % i for i in sorted(by_kind.items()))))
    if use:
        latest = sorted(((use[key][1], key, kind, reason) for key, kind, status, reason in rows if status == "refused" and key in use), reverse=True)
        lines += ["", "## refused keys used latest: last use, first use, uses, key kind reason"]
        lines += ["%8.1f %8.1f %7d %s %s %s" % (last, use[key][0], use[key][2], key, kind, reason) for last, key, kind, reason in latest[:LATEST_USED_KEYS]]
    lines += ["", "## keys: key kind status reason [first= last= uses= seconds since the process started]"]
    lines += ["%s %s %s%s%s" % (key, kind, status, " " + reason if reason else "",
                              " first=%.1fs last=%.1fs uses=%d" % use[key] if key in use else "") for key, kind, status, reason in rows]
    return "\n".join(lines) + "\n"


# The request files of the worst refused keys, so a translator can be fixed from the zip alone (private data: mailed, never posted).
REQUESTS_DIR = "shader_requests"
REQUESTS_TOP = 40                   # keys
REQUESTS_CAP_BYTES = 4 * 1024 * 1024  # all request files together, uncompressed; the zip must stay mailable
REQUEST_PART_EXTS = (".a", ".b", ".root", ".ia", ".psin", ".desc", ".sa", ".sb", ".data", ".c67", ".ps57", ".m57", ".m86")
NOT_DRAWN_KEY_RE = re.compile(r"^x2w-render not drawn key ([0-9a-f]{16}): (\d+)", re.M)
CENSUS_KEY_RE = re.compile(r"^x2w-render compute census key ([0-9a-f]{16}): (\d+) dispatches, (\d+) translated", re.M)


def key_weights(stderr_text):
    """{key: draws not drawn (graphics) or dispatches without a translation (compute)}: each key's newest cumulative line."""
    weights = {}
    for m in NOT_DRAWN_KEY_RE.finditer(stderr_text):
        weights[m.group(1)] = int(m.group(2))
    for m in CENSUS_KEY_RE.finditer(stderr_text):
        weights[m.group(1)] = max(0, int(m.group(2)) - int(m.group(3)))
    return weights


def select_requests(cache_dir, rows, weights, top=REQUESTS_TOP, cap=REQUESTS_CAP_BYTES):
    """[(key, kind, weight, [(file name, bytes)])] for the refused keys ranked by weight (ties by key), at most `top`
    keys, whole keys only, request files of all together within `cap` bytes. Keys without weight are not ranked."""
    names = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
    chosen, used = [], 0
    ranked = sorted((r for r in rows if r[2] == "refused" and weights.get(r[0], 0) > 0), key=lambda r: (-weights[r[0]], r[0]))
    for key, kind, _, _ in ranked:
        if len(chosen) >= top:
            break
        files = []
        for name in sorted(n for n in names if n.startswith("req_" + key + ".") and os.path.splitext(n)[1] in REQUEST_PART_EXTS):
            files.append((name, os.path.getsize(os.path.join(cache_dir, name))))
        size = sum(b for _, b in files)
        if not files or used + size > cap:
            continue
        used += size
        chosen.append((key, kind, weights[key], files))
    return chosen


SYMBOLS_NAME = "d3d12_xs.symbols.txt"
MSVC_SYMBOL = re.compile(r"^\s*[0-9a-fA-F]{4}:[0-9a-fA-F]{8}\s+(\S+)\s+([0-9a-fA-F]{16})\s+(\S+)?")
MSVC_BASE = re.compile(r"Preferred load address is ([0-9a-fA-F]{16})")
NM_SYMBOL = re.compile(r"^([0-9a-fA-F]{8,16})\s+[TtWw]\s+(\S+)\s*$")


def parse_symbols(text):
    """RVA and name of every function in a linker map of OUR OWN DLL: the MSVC /MAP file (the tester's build; only entries
    flagged `f`, RVA = address - preferred load address) or `nm -n` text (address, type T/t/W/w, name; addresses taken as RVAs).
    Sorted by RVA, one name per RVA. Raises ValueError on text of neither kind."""
    base = MSVC_BASE.search(text)
    rows = {}
    if base:
        base = int(base.group(1), 16)
        for line in text.splitlines():
            m = MSVC_SYMBOL.match(line)
            if m and m.group(3) == "f":
                rows.setdefault(int(m.group(2), 16) - base, m.group(1))
    else:
        for line in text.splitlines():
            m = NM_SYMBOL.match(line)
            if m:
                rows.setdefault(int(m.group(1), 16), m.group(2))
    if not rows:
        raise ValueError("no function symbols in the map (expected an MSVC /MAP file or nm -n text)")
    return sorted(rows.items())


def format_symbols(rows):
    """Compact table, one `RVA name` line per function (hex RVA, no 0x): symbols of our DLL, never game files."""
    return "".join("%x %s\n" % row for row in rows)


def load_symbols(text):
    """Reads a table written by format_symbols back into sorted (rva, name) rows."""
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            rows.append((int(parts[0], 16), parts[1].strip()))
    return sorted(rows)


def log_files(logdir):
    """(archive name, path) of every file the run wrote: the run folder's own files and the debugger summary
    `<logdir>.out` beside it. Sub-folders (a shader cache), memory dumps and anything that is not a regular file stay out."""
    out = []
    for name in sorted(os.listdir(logdir)):
        path = os.path.join(logdir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() not in EXCLUDED_SUFFIXES:
            out.append((name, path))
    summary = logdir.rstrip("\\/") + ".out"
    if os.path.isfile(summary):
        out.append((SUMMARY_NAME, summary))
    return out


def chunks(name, path, chunk_bytes):
    """One member per file, or numbered members `name.001`, `name.002` ... when the file is larger than chunk_bytes."""
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        if size <= chunk_bytes:
            yield name, handle.read()
            return
        n = 0
        while True:
            data = handle.read(chunk_bytes)
            if not data:
                return
            n += 1
            yield "%s.%03d" % (name, n), data


POLL_RE = re.compile(r"x2w-input script time base [\w-]+: first poll at ([0-9.]+) s, first entry at ([0-9.]+) s since process start")


def first_poll(logdir):
    """(first poll, first entry) in seconds since process start, from the stand-in's log line, or None."""
    for name in ("stderr.txt", "events.txt", "stdout.txt"):
        path = os.path.join(logdir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = POLL_RE.search(line)
                if match:
                    return float(match.group(1)), float(match.group(2))
    return None


RE_ENDED_TIMEOUT = re.compile(r"^timeout after ([0-9.]+) s", re.M)
RE_ENDED_REQUEST = re.compile(r"^stopped by request after ([0-9.]+) s \(([^)]*)\)", re.M)
RE_ENDED_EXIT = re.compile(r"^process exited with (0x[0-9a-fA-F]+) after ([0-9.]+) s", re.M)
RE_ENDED_STOP = re.compile(r"^stopped at (.+?) \((0x[0-9a-fA-F]+)\) at .* after ([0-9.]+) s", re.M)
RE_HEARTBEAT_FRAMES = re.compile(r"x2w-d3d12 heartbeat: (\d+) frames presented in ([0-9.]+) s")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def last_milestone(logdir):
    """The last frame count the title reported ("last present #N at S s") from its heartbeat lines, or "no present reported"."""
    best = None
    for name in ("stderr.txt", "stdout.txt"):
        path = os.path.join(logdir, name)
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 262144))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for found in RE_HEARTBEAT_FRAMES.finditer(text):
            candidate = (float(found.group(2)), int(found.group(1)))
            best = candidate if best is None or candidate[0] > best[0] else best
    return "last present #%d at %.0f s" % (best[1], best[0]) if best else "no present reported"


def end_cause(logdir):
    """Why the run ended, always one line, from the debugger's summary and the files the watchdog leaves:
    watchdog (stall/deadlock/memory, or a hard end), timeout, stopped by request (stop file or commit headroom), title
    exited (code 0), killed externally (a non-zero exit code and no stop file), crashed (exception code), debugger stop."""
    stall = _read_json(os.path.join(logdir, "stall_stop.json"))
    hard = _read_json(os.path.join(logdir, "hard_end.json"))
    text = ""
    for path in (os.path.join(logdir, "finished.txt"), logdir.rstrip("\\/") + ".out"):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            break
    if stall:
        cause = "watchdog at %s s: %s" % (stall.get("at_seconds"), stall.get("reason"))
    elif hard and not text:
        cause = "debugger did not finish: " + str(hard.get("note"))
    elif RE_ENDED_TIMEOUT.search(text):
        cause = "timeout after %s s (the run was not stopped early)" % RE_ENDED_TIMEOUT.search(text).group(1)
    elif RE_ENDED_REQUEST.search(text):
        found = RE_ENDED_REQUEST.search(text)
        cause = "stopped by request at %s s (%s, no watchdog stop)" % (found.group(1), found.group(2))
    elif RE_ENDED_EXIT.search(text):
        found = RE_ENDED_EXIT.search(text)
        code, at = int(found.group(1), 16), found.group(2)
        if code == 0:
            cause = "title exited by itself (code 0x0) at %s s" % at
        elif code >= 0xC0000000 and code != 0xFFFFFFFF:
            cause = "title crashed (exit code %s) at %s s" % (found.group(1), at)
        else:
            cause = "title killed externally (exit code %s, no stop file) at %s s" % (found.group(1), at)
    elif RE_ENDED_STOP.search(text):
        found = RE_ENDED_STOP.search(text)
        cause = "debugger stop: %s (%s) at %s s" % (found.group(1), found.group(2), found.group(3))
    else:
        cause = "unknown: the debugger wrote no summary (killed, or still running)"
    if hard and text:
        cause += "; hard end: " + str(hard.get("note"))
    return cause + "; " + last_milestone(logdir)


def pack(logdir, manifest_in, zip_path, limit_bytes=DEFAULT_LIMIT_MB * 1024 * 1024, chunk_bytes=CHUNK_BYTES, cache_dir=None, map_file=None):
    """Writes every log of the run into numbered zip parts, each under limit_bytes; nothing is dropped.
    One part keeps the plain name; several become <stem>_partNNofMM.zip. Returns the list of (path, size)."""
    manifest = validate_manifest(dict(manifest_in))
    poll = first_poll(logdir)
    manifest["first_poll_seconds"] = poll[0] if poll else None
    room = limit_bytes - MANIFEST_ROOM
    parts, members, used, listing = [], [[]], 0, {}
    def add(member, data):
        nonlocal used
        packed = len(zlib.compress(data, 6)) + 100 + 2 * len(member)
        if used + packed > room and members[-1]:
            members.append([])
            used = 0
        members[-1].append((member, data))
        listing[member] = {"bytes": len(data), "part": len(members)}
        used += packed

    # Order = what a part 1 that arrives alone must still answer (Run Q: part 2 never came, and with it shader_results.txt, the
    # debugger summary and every small log): the result text and the symbol table first, then every small file
    # (SMALL_FILE_BYTES, the debugger summary included), then the big logs and the pictures, the request files last.
    rows = []
    if cache_dir:  # the result of every request key of the shader cache, text only (cache_results)
        rows = cache_results(cache_dir)
        manifest["cache_keys"] = len(rows)
        add(RESULTS_NAME, format_results(rows, key_use(logdir)).encode("utf-8"))
    if map_file:  # function table of our own d3d12_xs.dll, so analyze_run can name d3d12_xs.dll+0xNNNN
        with open(map_file, encoding="utf-8", errors="replace") as handle:
            symbol_rows = parse_symbols(handle.read())
        manifest["symbols"] = len(symbol_rows)
        add(SYMBOLS_NAME, format_symbols(symbol_rows).encode("utf-8"))
    files = log_files(logdir)
    small = [f for f in files if os.path.getsize(f[1]) <= SMALL_FILE_BYTES]
    for name, path in small + [f for f in files if f not in small]:
        for member, data in chunks(name, path, chunk_bytes):
            add(member, data)
    if cache_dir:  # the request files of the worst refused keys (select_requests)
        stderr_path = os.path.join(logdir, "stderr.txt")
        weights = {}
        if os.path.isfile(stderr_path):
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as handle:
                weights = key_weights(handle.read())
        chosen = select_requests(cache_dir, rows, weights)
        for key, kind, weight, files_of_key in chosen:
            for name, _ in files_of_key:
                with open(os.path.join(cache_dir, name), "rb") as handle:
                    add(REQUESTS_DIR + "/" + name, handle.read())
        manifest["request_files"] = {"folder": REQUESTS_DIR, "keys": len(chosen), "files": sum(len(c[3]) for c in chosen),
                                     "bytes": sum(b for c in chosen for _, b in c[3]),
                                     "ranked": [{"key": k, "kind": kd, "weight": w} for k, kd, w, _ in chosen]}
    stem, ext = os.path.splitext(zip_path)
    count = len(members)
    for index, group in enumerate(members, 1):
        path = zip_path if count == 1 else "%s_part%02dof%02d%s" % (stem, index, count, ext)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for member, data in group:
                z.writestr(member, data)
            if index == 1:
                body = dict(manifest, files=listing, parts=count)
                z.writestr("manifest.json", json.dumps(body, indent=2, sort_keys=True))
        parts.append((path, os.path.getsize(path)))
    return parts


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("input-preset")
    p.add_argument("spec")
    p.add_argument("seconds", type=int)
    p.add_argument("--privacy-boxes", type=int, default=2)
    lab = sub.add_parser("input-labels")
    lab.add_argument("spec")
    lab.add_argument("seconds", type=int)
    lab.add_argument("--privacy-boxes", type=int, default=2)
    m = sub.add_parser("memory-gib")
    m.add_argument("physical_bytes", type=int)
    n = sub.add_parser("cache-tag")
    n.add_argument("translator")
    c = sub.add_parser("cache-stats")
    c.add_argument("cache")
    s = sub.add_parser("seed-cache")
    s.add_argument("cache_root")
    s.add_argument("new_cache")
    s.add_argument("--jobs", type=int, default=0)
    s.add_argument("--json", default=None, help="write the result as JSON (for the manifest)")
    f = sub.add_parser("first-poll")
    f.add_argument("logdir")
    e = sub.add_parser("end-cause")
    e.add_argument("logdir")
    q = sub.add_parser("pack")
    q.add_argument("logdir")
    q.add_argument("manifest")
    q.add_argument("zip")
    q.add_argument("--limit-mb", type=float, default=DEFAULT_LIMIT_MB)
    q.add_argument("--cache", default=None, help="shader cache folder: add shader_results.txt (the result text of every key)")
    q.add_argument("--map", default=None, help="linker map of our d3d12_xs.dll: add d3d12_xs.symbols.txt (RVA and function name only)")
    r = sub.add_parser("cache-results")
    r.add_argument("cache")
    args = parser.parse_args(argv)
    if args.cmd == "input-preset":
        print(input_preset(args.spec, args.seconds, args.privacy_boxes))
        return 0
    if args.cmd == "input-labels":
        print(json.dumps(input_labels(args.spec, args.seconds, args.privacy_boxes)))
        return 0
    if args.cmd == "memory-gib":
        print(memory_gib(args.physical_bytes))
        return 0
    if args.cmd == "cache-tag":
        print(cache_tag(args.translator))
        return 0
    if args.cmd == "cache-stats":
        print("%d %d %d" % cache_stats(args.cache))
        return 0
    if args.cmd == "cache-results":
        sys.stdout.write(format_results(cache_results(args.cache)))
        return 0
    if args.cmd == "seed-cache":
        result = seed_cache(args.cache_root, args.new_cache, args.jobs)
        print(seed_line(result))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
        return 0
    if args.cmd == "end-cause":
        print(end_cause(args.logdir))
        return 0
    if args.cmd == "first-poll":
        poll = first_poll(args.logdir)
        print("first pad poll at %g s after start (first input entry at %g s)" % poll if poll
              else "no pad poll was logged: the input script never started")
        return 0
    with open(args.manifest, encoding="utf-8-sig") as handle:
        manifest_in = json.load(handle)
    for path, size in pack(args.logdir, manifest_in, args.zip, int(args.limit_mb * 1024 * 1024), cache_dir=args.cache, map_file=args.map):
        print("%s %d" % (path, size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
