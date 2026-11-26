#!/usr/bin/env python3
"""Pure helpers of tools/startup/tester_run.ps1: the input preset, the manifest and the zip of one tester run.

  tester_pack.py input-preset <menus|raw script> <seconds>      print the X2W_INPUT_SCRIPT value
  tester_pack.py cache-tag <gfx2hlsl.py>                        print the translation revisions, e.g. 28-28-1
  tester_pack.py cache-stats <cache dir>                        print "<translated> <requests> <failed>"
  tester_pack.py first-poll <logdir>                            print when the title first polled the pad
  tester_pack.py pack <logdir> <manifest_in.json> <zip> [--limit-mb N]   write the zip, print its path and size

The zip holds every file the run wrote in its log folder (stdout, stderr, events, finished, threads, files, trap.log,
translate, translate_err, window*.png, bink_luma.png ...), the debugger summary (`<logdir>.out`) and manifest.json.
Not in it: sub-folders such as a shader cache, memory dumps and game files. Nothing is dropped: past the size limit
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
MENUS_STEP_S = 6
MENUS_END_S = 500          # the preset stops here however long the run is
DPAD_DOWN = "80"           # GameInput button mask, hexadecimal
BUTTON_A = "4"
EXCLUDED_SUFFIXES = (".dmp", ".mdmp", ".hdmp")
SUMMARY_NAME = "debugger_summary.out"
MANIFEST_KEYS = ("commit", "build_commit", "dirty", "seconds", "input_script", "env", "gpu", "gpu_driver",
                 "windows_build", "wall_seconds", "deep", "lazy_textures", "lazy_memory", "memory_gib", "shader_cache", "translations_at_start")
DEFAULT_LIMIT_MB = 20
MANIFEST_ROOM = 256 * 1024     # headroom per part for the manifest and zip bookkeeping
CHUNK_BYTES = 48 * 1024 * 1024  # a file larger than this is cut into numbered pieces


def input_preset(spec, seconds):
    """`menus` is D-pad down, D-pad down, A, one press every 6 s from 5 s after the first pad poll; entries run to
    `seconds` of script time, but not past 500 s, at most 256 entries. Anything else is a raw script."""
    if spec != "menus":
        if len(spec) > MAX_SCRIPT_CHARS:
            raise ValueError("input script longer than %d characters" % MAX_SCRIPT_CHARS)
        if len([item for item in spec.split(",") if item.strip()]) > MAX_ENTRIES:
            raise ValueError("input script has more than %d entries" % MAX_ENTRIES)
        return spec
    items = []
    press, at = 0, MENUS_START_S
    while at < min(seconds, MENUS_END_S) and len(items) < MAX_ENTRIES:
        items.append("%s@%d" % (BUTTON_A if press % 3 == 2 else DPAD_DOWN, at))
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


def pack(logdir, manifest_in, zip_path, limit_bytes=DEFAULT_LIMIT_MB * 1024 * 1024, chunk_bytes=CHUNK_BYTES):
    """Writes every log of the run into numbered zip parts, each under limit_bytes; nothing is dropped.
    One part keeps the plain name; several become <stem>_partNNofMM.zip. Returns the list of (path, size)."""
    manifest = validate_manifest(dict(manifest_in))
    poll = first_poll(logdir)
    manifest["first_poll_seconds"] = poll[0] if poll else None
    room = limit_bytes - MANIFEST_ROOM
    parts, members, used, listing = [], [[]], 0, {}
    for name, path in log_files(logdir):
        for member, data in chunks(name, path, chunk_bytes):
            packed = len(zlib.compress(data, 6)) + 100 + 2 * len(member)
            if used + packed > room and members[-1]:
                members.append([])
                used = 0
            members[-1].append((member, data))
            listing[member] = {"bytes": len(data), "part": len(members)}
            used += packed
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
    m = sub.add_parser("memory-gib")
    m.add_argument("physical_bytes", type=int)
    n = sub.add_parser("cache-tag")
    n.add_argument("translator")
    c = sub.add_parser("cache-stats")
    c.add_argument("cache")
    f = sub.add_parser("first-poll")
    f.add_argument("logdir")
    q = sub.add_parser("pack")
    q.add_argument("logdir")
    q.add_argument("manifest")
    q.add_argument("zip")
    q.add_argument("--limit-mb", type=float, default=DEFAULT_LIMIT_MB)
    args = parser.parse_args(argv)
    if args.cmd == "input-preset":
        print(input_preset(args.spec, args.seconds))
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
    if args.cmd == "first-poll":
        poll = first_poll(args.logdir)
        print("first pad poll at %g s after start (first input entry at %g s)" % poll if poll
              else "no pad poll was logged: the input script never started")
        return 0
    with open(args.manifest, encoding="utf-8-sig") as handle:
        manifest_in = json.load(handle)
    for path, size in pack(args.logdir, manifest_in, args.zip, int(args.limit_mb * 1024 * 1024)):
        print("%s %d" % (path, size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
