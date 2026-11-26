#!/usr/bin/env python3
"""Pure helpers of tools/startup/tester_run.ps1: the input preset, the manifest and the zip of one tester run.

  tester_pack.py input-preset <menus|menus2|raw script> <seconds>      print the X2W_INPUT_SCRIPT value
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
of its `.fail` text (the translator's own message; a compiler message is cut to its error codes), ranked by reason. Never
shader code or binaries. With --map (the linker map of our own d3d12_xs.dll) it also holds
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
PRIVACY_START_S = 90       # script second the privacy screen is reached (Run L pictures: legal notice at process 90 s, privacy screen at 105 s, process = script + 8 s)
MENUS_STEP_S = 2           # one step (DOWN, DOWN, A pattern) every 2 s after the first press (Run K: 6 s was too slow)
MENUS_END_S = 300          # the preset stops here however long the run is
DPAD_DOWN = "80"           # GameInput button mask, hexadecimal
BUTTON_A = "4"
MENUS_HOLD_S = "1.2"       # a press is held this long (0.8 s released before the next): at 2 to 3 fps the title polls the pad every 0.4 s, and the 0.3 s default fell between two polls (Run I saw DOWN,A instead of DOWN,DOWN,A)
MENUS2_HOLD_S = "1.2"
MENUS2_GAP_S = "2"         # between the presses of one cycle; the first press follows the previous wait by the same gap
MENUS2_SETTLE_S = 2        # the picture must stay unchanged this long after it changed
MENUS2_TIMEOUT_S = 90      # a screen that never changes (or loads slower) lets the script go on after this
EXCLUDED_SUFFIXES = (".dmp", ".mdmp", ".hdmp")
SUMMARY_NAME = "debugger_summary.out"
MANIFEST_KEYS = ("commit", "build_commit", "dirty", "seconds", "input_script", "env", "gpu", "gpu_driver",
                 "windows_build", "wall_seconds", "deep", "lazy_textures", "lazy_memory", "memory_gib", "shader_cache", "translations_at_start")
DEFAULT_LIMIT_MB = 20
MANIFEST_ROOM = 256 * 1024     # headroom per part for the manifest and zip bookkeeping
CHUNK_BYTES = 48 * 1024 * 1024  # a file larger than this is cut into numbered pieces


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


def privacy_items(boxes, start, skip_down=0):
    """The privacy screen (two checkboxes and a Conferma button): `skip_down` D-pad downs to reach the first checkbox, then A on each box with D-pad down to
    the next one, D-pad down onto the confirm button after the last box and A to confirm. One press every
    MENUS_STEP_S. Returns (items, time of the next free step). The layout (how many boxes, where the focus starts) is
    a parameter until a run's pictures show it."""
    items, at = [], start
    def press(mask):
        nonlocal at
        items.append("%s@%d+%s" % (mask, at, MENUS_HOLD_S))
        at += MENUS_STEP_S
    for _ in range(skip_down):
        press(DPAD_DOWN)
    for _ in range(boxes):
        press(BUTTON_A)
        press(DPAD_DOWN)
    press(BUTTON_A)   # confirm
    return items, at


def input_preset(spec, seconds, privacy_boxes=0, privacy_start=PRIVACY_START_S, privacy_skip_down=0):
    """`menus` is D-pad down, D-pad down, A, one press every 2 s from 5 s after the first pad poll; entries run to
    `seconds` of script time, but not past 300 s, at most 256 entries. With privacy_boxes > 0 the privacy screen comes
    at privacy_start (privacy_items, instead of the pattern presses from that second), then the pattern starts over.
    Anything else is a raw script."""
    if spec == "menus2":
        return menus2_script()
    if spec != "menus":
        if len(spec) > MAX_SCRIPT_CHARS:
            raise ValueError("input script longer than %d characters" % MAX_SCRIPT_CHARS)
        if len([item for item in spec.split(",") if item.strip()]) > MAX_ENTRIES:
            raise ValueError("input script has more than %d entries" % MAX_ENTRIES)
        return spec
    items, press, at = [], 0, MENUS_START_S
    block_done = privacy_boxes <= 0
    while at < min(seconds, MENUS_END_S) and len(items) < MAX_ENTRIES:
        if not block_done and at >= privacy_start:
            # the privacy screen: its own presses, then the pattern starts over with DOWN
            block, at = privacy_items(privacy_boxes, at, privacy_skip_down)
            items.extend(block)
            block_done, press = True, 0
            continue
        items.append("%s@%d+%s" % (BUTTON_A if press % 3 == 2 else DPAD_DOWN, at, MENUS_HOLD_S))
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


def format_results(rows):
    """Text for the zip: totals, the refusals ranked by normalized reason, then one line per key."""
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
    lines += ["", "## keys: key kind status reason"]
    lines += ["%s %s %s%s" % (key, kind, status, " " + reason if reason else "") for key, kind, status, reason in rows]
    return "\n".join(lines) + "\n"


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

    for name, path in log_files(logdir):
        for member, data in chunks(name, path, chunk_bytes):
            add(member, data)
    if cache_dir:  # the result of every request key of the shader cache, text only (cache_results)
        rows = cache_results(cache_dir)
        manifest["cache_keys"] = len(rows)
        add(RESULTS_NAME, format_results(rows).encode("utf-8"))
    if map_file:  # function table of our own d3d12_xs.dll, so analyze_run can name d3d12_xs.dll+0xNNNN
        with open(map_file, encoding="utf-8", errors="replace") as handle:
            rows = parse_symbols(handle.read())
        manifest["symbols"] = len(rows)
        add(SYMBOLS_NAME, format_symbols(rows).encode("utf-8"))
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
    p.add_argument("--privacy-boxes", type=int, default=0)
    p.add_argument("--privacy-start", type=int, default=PRIVACY_START_S)
    p.add_argument("--privacy-skip-down", type=int, default=0)
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
        print(input_preset(args.spec, args.seconds, args.privacy_boxes, args.privacy_start, args.privacy_skip_down))
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
