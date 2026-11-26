#!/usr/bin/env python3
"""Check tools/startup/tester_run.ps1 in a scratch tree: build check, fresh shader cache, run arguments, zip.

The scratch tree is a throw-away git repository that holds copies of tester_run.ps1 and tester_pack.py, a stub
run.ps1 that records its arguments and writes a small run folder, and a generated x2w_build_rev.cpp. No title, no
debugger and no GPU are involved. Needs git and a PowerShell (powershell or pwsh); without either it prints a
notice and passes, because Linux hosts cannot run the script.

  - a build from another commit is refused with -NoBuild, and accepted when its revision matches HEAD
  - one shader cache folder per shader revision (gfx2hlsl.py): kept across runs and commits, a new empty one on a bump,
    and the summary says how many translations were present at start
  - run.ps1 gets -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000, the menus input script, the capture
    interval and X2W_D3D12_BUDGET=1; -Deep adds -TraceCalls and -TraceApi
  - the zip holds every file of the run folder and a valid manifest, and never the shader cache
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STARTUP = REPO / "tools" / "startup"
STUB_RUN = r"""
param([string]$Exe, [string]$DataDir, [string]$LogDir, [int]$Seconds, [string]$BuildDir, [string]$ShaderCache,
      [int]$ContinueAsserts, [int]$MemoryGiB, [string]$InputScript, [int]$CaptureEvery, [switch]$Render, [switch]$Translate,
      [switch]$FakeUser, [switch]$GpuMirror, [switch]$TraceCalls, [switch]$TraceApi, [switch]$LazyMemory)
$seen = @(Get-ChildItem -LiteralPath $ShaderCache -Force | ForEach-Object { $_.Name })
$record = [ordered]@{
    Exe = $Exe; DataDir = $DataDir; Seconds = $Seconds; ContinueAsserts = $ContinueAsserts; InputScript = $InputScript
    CaptureEvery = $CaptureEvery; Render = [bool]$Render; Translate = [bool]$Translate; FakeUser = [bool]$FakeUser
    GpuMirror = [bool]$GpuMirror; TraceCalls = [bool]$TraceCalls; TraceApi = [bool]$TraceApi
    Budget = $env:X2W_D3D12_BUDGET; Lazy = $env:X2W_D3D12_LAZY_TEXTURES; Sysmem = $env:X2W_D3D12_SYSMEM_FALLBACK; MemoryGiB = $MemoryGiB; LazyMemory = [bool]$LazyMemory; CacheEntries = $seen; CacheFolder = (Split-Path -Leaf $ShaderCache)
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LogDir "args.json")
foreach ($name in "finished.txt", "events.txt", "stdout.txt", "stderr.txt", "files.txt", "trap.log", "threads.txt",
                  "translate.txt", "translate_err.txt") {
    Set-Content -LiteralPath (Join-Path $LogDir $name) -Value "stub $name"
}
[System.IO.File]::WriteAllBytes((Join-Path $LogDir "window.png"), [byte[]](1, 2, 3))
[System.IO.File]::WriteAllBytes((Join-Path $LogDir "window_015s.png"), [byte[]](4, 5, 6))
Add-Content -LiteralPath (Join-Path $LogDir "stderr.txt") -Value "x2w-input script time base first-poll: first poll at 151.9 s, first entry at 156.9 s since process start"
Set-Content -LiteralPath "$LogDir.out" -Value "stub summary"
if ($env:STUB_KILL_SELF) { Stop-Process -Id $PID -Force }   # the tester pressed Ctrl-C / closed the window: nothing after this runs
if ($env:STUB_STALL) {
    # a real debugger writes finished.txt last: the stub does the same, after it was told to stop
    Remove-Item -LiteralPath (Join-Path $LogDir "finished.txt") -Force
    $wait = if ($env:STUB_STALL_WAIT) { [int]$env:STUB_STALL_WAIT } else { 120 }
    $deadline = (Get-Date).AddSeconds($wait)
    while (-not (Test-Path -LiteralPath (Join-Path $LogDir "stop")) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    # what ended the wait: the stop file (the watchdog worked) or the stub's own deadline (it did not)
    $ended = if (Test-Path -LiteralPath (Join-Path $LogDir "stop")) { "stop file" } else { "deadline" }
    Set-Content -LiteralPath (Join-Path $LogDir "wait_ended.txt") -Value $ended
    Set-Content -LiteralPath (Join-Path $LogDir "finished.txt") -Value "stub finished.txt"
}
exit 0
"""
TRANSLATOR = """\
import os, sys, time
def fnv(blob): return 0
key = fnv(b'x2w-startup-shader-%d')
ckey = fnv(b'x2w-startup-shader-%dcs')
mkey = fnv(b'x2w-startup-mesh-57-7')
# one pass (argv: --once <cache>): publish a ready marker for every request that has none; STUB_HANG never ends (the cap)
if len(sys.argv) > 2 and sys.argv[1] == '--once':
    if os.environ.get('STUB_HANG'):
        time.sleep(600)
    for name in sorted(os.listdir(sys.argv[2])):
        if name.startswith('req_') and name.endswith('.a'):
            base = os.path.join(sys.argv[2], name[4:-2])      # req_<key>.a -> <key>.txt + the two bytecode files
            if not os.path.exists(base + '.txt'):
                for suffix in ('.txt', '.vs.cso', '.ps.cso'):
                    open(base + suffix, 'wb').write(b'ok')
"""
DLLS = ["xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput"]
REV_CPP = 'const char x2w_build_rev[] = "%s";\n'


def shells():
    found = [shutil.which(name) for name in ("powershell", "pwsh")]
    return [path for path in found if path]


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, check=True).stdout.strip()


def make_tree(root):
    tree = Path(root) / "tree"
    (tree / "tools" / "startup").mkdir(parents=True)
    for name in ("tester_run.ps1", "tester_pack.py", "shader_cache_revision.py", "runtime_sources.py", "memory_top.py", "stall_watchdog.py", "tester_janitor.py"):
        shutil.copy(STARTUP / name, tree / "tools" / "startup" / name)
    (tree / "src").mkdir()
    (tree / "src" / "device.cpp").write_text("int device;\n", encoding="ascii")
    (tree / "tools" / "startup" / "run.ps1").write_text(STUB_RUN, encoding="utf-8")
    (tree / "tools" / "startup" / "gfx2hlsl.py").write_text((TRANSLATOR % (28, 28)), encoding="utf-8")
    git(tree, "init", "-q")
    git(tree, "config", "user.email", "check@example.invalid")
    git(tree, "config", "user.name", "check")
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", "scratch")
    head = git(tree, "rev-parse", "--short=8", "HEAD")
    gen = tree / "out" / "build" / "release" / "generated"
    gen.mkdir(parents=True)
    for dll in DLLS:
        (gen.parent / (dll + ".dll")).write_bytes(b"MZ")
    return tree, head, gen / "x2w_build_rev.cpp"


def run_script(shell, tree, *args, extra_path=None, extra_env=None, merge_stderr=False):
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if extra_path:
        env["PATH"] = str(extra_path) + os.pathsep + env["PATH"]
    env.update(extra_env or {})
    for key in [k for k in env if k.startswith("X2W_")]:
        del env[key]
    if "-ScanRoots" not in args:
        args = (*args, "-ScanRoots", str(tree / "scan_none"))  # no search of the real AppData during the checks
    if "-FinishTranslateSeconds" not in args:
        args = (*args, "-FinishTranslateSeconds", "0")  # the finish pass has its own checks; the others keep a still cache
    script = str(tree / "tools" / "startup" / "tester_run.ps1")
    if merge_stderr:
        # A caller that merges the script's error stream into its output (`... 2>&1`, as a tester's wrapper or a log pipe
        # does): Windows PowerShell 5.1 then turns every line a native command writes to stderr into an error record.
        words = " ".join(word if word.startswith("-") and not word[1:].isdigit() else "'%s'" % word
                         for word in ("-Exe", "title.exe", "-DataDir", "data", *args))
        command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "& '%s' %s 2>&1; exit $LASTEXITCODE" % (script, words)]
    else:
        command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Exe", "title.exe", "-DataDir", "data", *args]
    return subprocess.run(command, capture_output=True, text=True, errors="replace", env=env, timeout=300, cwd=str(tree))


def check_shell(shell):
    with tempfile.TemporaryDirectory() as root:
        tree, head, rev_file = make_tree(root)
        cache = tree / "out" / "startup" / "shader_cache_28-28-7"
        cache.mkdir(parents=True)
        # A warm cache as the translator leaves it: one finished request (manifest <key>.txt + both bytecode files), one
        # pending, one refused (<key>.fail), and the revision stamp.
        warm = ["req_0123456789abcdef.a", "0123456789abcdef.txt", "0123456789abcdef.vs.cso", "0123456789abcdef.ps.cso",
                "req_1123456789abcdef.a", "req_4123456789abcdef.a", "4123456789abcdef.fail", "revision.txt"]
        for name in warm:
            (cache / name).write_bytes(b"x")

        rev_file.write_text(REV_CPP % "00000000", encoding="ascii")
        stale = run_script(shell, tree, "-NoBuild")
        assert stale.returncode != 0 and "00000000" in stale.stdout + stale.stderr and head in stale.stdout + stale.stderr, \
            (shell, "a build from another commit was not refused", stale.stdout, stale.stderr)
        assert sorted(os.listdir(cache)) == sorted(warm), "the cache was touched"
        assert not list((tree / "out" / "tester").glob("*.zip")) if (tree / "out" / "tester").exists() else True

        rev_file.write_text(REV_CPP % head, encoding="ascii")
        ok = run_script(shell, tree, "-NoBuild", "-Seconds", "200", "-CaptureEvery", "10", "-Input", "menus")
        assert ok.returncode == 0, (shell, ok.stdout, ok.stderr)
        zips = sorted((tree / "out" / "tester").glob("report_%s_*.zip" % head))
        assert len(zips) == 1, zips
        assert "Send this file" in ok.stdout, ok.stdout
        assert "first pad poll at 151.9 s" in ok.stdout, ok.stdout
        with zipfile.ZipFile(zips[0]) as z:
            names = sorted(z.namelist())
            manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
            args = json.loads(z.read("args.json").decode("utf-8-sig"))
            memory = z.read("memory.txt").decode("utf-8").splitlines()
        # The end line after the run: the system commit and the top processes by commit (this shell is one of them).
        assert memory and re.match(r"end commit \d+\.\d/\d+\.\d GiB \| \S+\.exe", memory[-1]), memory
        assert names == sorted(["finished.txt", "events.txt", "stdout.txt", "stderr.txt", "files.txt", "trap.log",
                                "threads.txt", "translate.txt", "translate_err.txt", "window.png", "window_015s.png",
                                "args.json", "debugger_summary.out", "manifest.json", "shader_results.txt", "memory.txt"]), names
        assert args["Render"] and args["Translate"] and args["FakeUser"] and args["GpuMirror"], args
        assert args["ContinueAsserts"] == 1000 and args["Seconds"] == 200 and args["CaptureEvery"] == 10, args
        assert args["LazyMemory"] and manifest["lazy_memory"] is True, args
        assert manifest["build_check"] == "same commit", manifest
        assert manifest["title_input_snapshot"] is True and manifest["env"].get("X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT") == "1" \
            and manifest["env"].get("X2W_TRANSLATED_INDIRECT_COPY_RANGES") == "1", manifest["env"]
        assert args["MemoryGiB"] == manifest["memory_gib"] >= 14 and manifest["physical_ram_gib"] > 0, (args, manifest)
        assert "job memory cap %d GiB" % args["MemoryGiB"] in ok.stdout, ok.stdout
        assert args["Lazy"] == "1" and manifest["lazy_textures"] is True and manifest["env"].get("X2W_D3D12_LAZY_TEXTURES") == "1"
        assert args["Budget"] == "1" and not args["TraceCalls"] and not args["TraceApi"], args
        assert args["Sysmem"] == "1" and manifest["sysmem_fallback"] is True and manifest["env"].get("X2W_D3D12_SYSMEM_FALLBACK") == "1"
        assert manifest["gpu_time"] is True and manifest["env"].get("X2W_GPU_TIME") == "1", manifest  # GPU time per frame on
        assert manifest["hdd_mode"] is True and manifest["env"].get("X2W_STORAGE_HDD") == "1", manifest  # HDD mode on by default
        saves = tree / "out" / "startup" / "saves"  # the title's save folder: one per tree, kept across runs
        assert saves.is_dir() and manifest["save_root"].replace("/", "\\").rstrip("\\").endswith("out\\startup\\saves") and manifest["fresh_save"] is False, manifest
        assert manifest["env"].get("X2W_STORAGE_PATH", "").endswith("saves\\") and manifest["saves_at_start"] == 0 and manifest["saves_at_end"] == 0 and manifest["saves_files"] == [], manifest
        assert manifest["cache_repaired"] == -1, manifest  # no repair tool in the scratch tree: skipped with a warning, the run goes on
        # The default menus script: clock-free privacy passes, a press every second held 0.8 s (tester_pack.privacy_cycles)
        assert args["InputScript"].startswith("40@5+0.8,40@6+0.8,4@7+0.8,80@8+0.8,4@9+0.8,80@10+0.8,4@11+0.8"), args["InputScript"]
        assert manifest["input_labels"][2] == "privacy pass 1: tick box 1" and len(manifest["input_labels"]) == len(args["InputScript"].split(",")), manifest["input_labels"][:3]
        assert args["CacheFolder"] == "shader_cache_28-28-7", args["CacheFolder"]
        assert sorted(args["CacheEntries"]) == sorted(warm), \
            "the run did not see the kept shader cache: %r" % (args["CacheEntries"],)
        assert "1 translations were already present at start (3 requests, 1 failed)" in ok.stdout, ok.stdout
        assert manifest["commit"] == head and manifest["build_commit"] == head and manifest["dirty"] is False
        assert manifest["seconds"] == 200 and manifest["deep"] is False and manifest["input_script"] == args["InputScript"]
        assert manifest["env"].get("X2W_D3D12_BUDGET") == "1" and all(k.startswith("X2W_") for k in manifest["env"])
        assert manifest["first_poll_seconds"] == 151.9
        assert manifest["parts"] == 1 and "gpu" in manifest and "windows_build" in manifest
        assert manifest["shader_cache"] == "shader_cache_28-28-7" and manifest["translations_at_start"] == 1
        assert not any(p.name.startswith("shader_cache.old") for p in cache.parent.iterdir()), "the cache was moved aside"
        assert not any(n.endswith(".bin") or n.startswith("req_") for n in names), "shader cache content travelled"

        deep = run_script(shell, tree, "-NoBuild", "-Deep", "-CaptureEvery", "0", "-Input", "4@60,1@70")
        assert deep.returncode == 0, (shell, deep.stdout, deep.stderr)
        newest = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest) as z:
            deep_args = json.loads(z.read("args.json").decode("utf-8-sig"))
            deep_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert deep_args["TraceCalls"] and deep_args["TraceApi"] and deep_args["InputScript"] == "4@60,1@70", deep_args
        assert deep_manifest["deep"] is True and deep_args["CaptureEvery"] == 0, deep_manifest

        # -NoLazyTextures leaves the variable unset and the manifest says so.
        plain = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-NoLazyTextures")
        assert plain.returncode == 0, (plain.stdout, plain.stderr)
        newest_plain = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_plain) as z:
            plain_args = json.loads(z.read("args.json").decode("utf-8-sig"))
            plain_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert not plain_args["Lazy"] and plain_manifest["lazy_textures"] is False
        nomem = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-NoLazyMemory")
        assert nomem.returncode == 0, (nomem.stdout, nomem.stderr)
        newest_mem = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_mem) as z:
            mem_args = json.loads(z.read("args.json").decode("utf-8-sig"))
            mem_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert not mem_args["LazyMemory"] and mem_manifest["lazy_memory"] is False
        assert mem_manifest["stop_reason"] == "" and mem_manifest["stall_seconds"] == 90, mem_manifest
        assert mem_manifest["end_cause"].startswith("unknown: the debugger wrote no summary"), mem_manifest["end_cause"]  # the stub writes no summary
        # Stall watchdog: a title that stops moving (the stub waits for the `stop` file) ends the run early, the zip is
        # still packed and the manifest says why. Memory rule off here: the host's own commit is not under test.
        stalled = run_script(shell, tree, "-NoBuild", "-Seconds", "200", "-CaptureEvery", "0", "-StallSeconds", "3",
                             "-StallMinSeconds", "0", "-MemoryStopPercent", "0", extra_env={"STUB_STALL": "1"})
        assert stalled.returncode == 0, (shell, stalled.stdout, stalled.stderr)
        assert "tester_run: STALL STOP at " in stalled.stdout and "frozen:" in stalled.stdout, stalled.stdout
        newest_stall = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_stall) as z:
            stall_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
            assert "stall_stop.json" in z.namelist(), z.namelist()
            # Not a wall-clock limit (a loaded VM with a release build running beside took 130 s for the whole test): the stub says
            # whether the stop file or its own deadline ended its wait.
            assert z.read("wait_ended.txt").decode("utf-8-sig").strip() == "stop file", "the run did not end through the stop file"
        assert stall_manifest["end_cause"].startswith("watchdog at ") and "frozen:" in stall_manifest["end_cause"], stall_manifest
        assert stall_manifest["stop_reason"].startswith("stall watchdog at ") and "frozen:" in stall_manifest["stop_reason"], stall_manifest
        # The tester stops the script (the stub ends the whole PowerShell process, as a closed window does) before it packed anything:
        # the janitor, in a window of its own, still packs the report from the early manifest.
        before = set((tree / "out" / "tester").glob("report_*.zip"))
        killed = run_script(shell, tree, "-NoBuild", "-Seconds", "200", "-CaptureEvery", "0", "-StallSeconds", "0",
                            "-MemoryStopPercent", "0", extra_env={"STUB_KILL_SELF": "1"})
        assert killed.returncode != 0, ("the stub did not end the script", killed.stdout, killed.stderr)
        waited = time.monotonic()
        while time.monotonic() - waited < 120 and not (set((tree / "out" / "tester").glob("report_*.zip")) - before):
            time.sleep(1)
        packed = sorted(set((tree / "out" / "tester").glob("report_*.zip")) - before)
        if not packed:
            notes = {str(f.relative_to(tree)): f.read_text(errors="replace")[-1500:] for f in (tree / "out" / "tester").glob("run_*/janitor*.txt")}
            for f in Path(tempfile.gettempdir()).glob("tester_janitor_*.json.*"):   # the janitor's own stdout/stderr
                notes[f.name] = f.read_text(errors="replace")[-1500:]
            raise AssertionError("the janitor did not pack the report of an interrupted run: %r" % notes)
        with zipfile.ZipFile(packed[0]) as z:
            cut = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert cut["interrupted"] is True and cut["end_cause"].startswith("interrupted by the tester"), cut
        off = run_script(shell, tree, "-NoBuild", "-Seconds", "200", "-CaptureEvery", "0", "-StallSeconds", "0",
                         "-MemoryStopPercent", "0", extra_env={"STUB_STALL": "1", "STUB_STALL_WAIT": "5"})
        assert "STALL STOP" not in off.stdout, off.stdout
        nosnap = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-NoTitleInputSnapshot")
        assert nosnap.returncode == 0, (nosnap.stdout, nosnap.stderr)
        newest_snap = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_snap) as z:
            snap_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert snap_manifest["title_input_snapshot"] is False
        # No -Input given: the default is no automatic presses (an empty script, no labels), so the tester's own clicks are not fought
        assert snap_manifest["input_script"] in ("", None) and not snap_manifest["input_labels"], snap_manifest["input_script"]
        assert not {"X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT", "X2W_TRANSLATED_INDIRECT_COPY_RANGES"} & set(snap_manifest["env"]), snap_manifest["env"]
        big = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-MemoryGiB", "33")
        assert big.returncode == 0, (big.stdout, big.stderr)
        newest_big = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_big) as z:
            big_args = json.loads(z.read("args.json").decode("utf-8-sig"))
            big_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert big_args["MemoryGiB"] == 33 and big_manifest["memory_gib"] == 33
        assert "X2W_D3D12_LAZY_TEXTURES" not in plain_manifest["env"]
        assert plain_args["Sysmem"] == "1" and plain_manifest["sysmem_fallback"] is True  # -NoLazyTextures leaves it on

        # -NoSysmemFallback and -NoGpuTime leave their variables unset and the manifest says so.
        nosys = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-NoSysmemFallback", "-NoGpuTime", "-NoHddMode")
        assert nosys.returncode == 0, (nosys.stdout, nosys.stderr)
        newest_nosys = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_nosys) as z:
            nosys_args = json.loads(z.read("args.json").decode("utf-8-sig"))
            nosys_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert not nosys_args["Sysmem"] and nosys_manifest["sysmem_fallback"] is False
        assert nosys_manifest["gpu_time"] is False and "X2W_GPU_TIME" not in nosys_manifest["env"], nosys_manifest
        assert "X2W_D3D12_SYSMEM_FALLBACK" not in nosys_manifest["env"]
        assert nosys_manifest["hdd_mode"] is False and "X2W_STORAGE_HDD" not in nosys_manifest["env"], nosys_manifest

        # The same commit keeps its cache across runs: the second run sees what the first one left.
        for name in ("req_2123456789abcdef.a", "2123456789abcdef.txt", "2123456789abcdef.vs.cso", "2123456789abcdef.ps.cso"):
            (cache / name).write_bytes(b"x")
        again = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0")
        assert again.returncode == 0 and "2 translations were already present at start" in again.stdout, again.stdout

        # A save written by one run is still there for the next; -FreshSave wipes it and the manifest says so.
        (saves / "x2w-gamesave").mkdir(exist_ok=True)
        (saves / "x2w-gamesave" / "profile.bin").write_bytes(b"accepted")
        kept = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0")
        assert kept.returncode == 0 and (saves / "x2w-gamesave" / "profile.bin").read_bytes() == b"accepted", kept.stdout
        assert "1 files kept from earlier runs" in kept.stdout, kept.stdout
        fresh = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-FreshSave")
        assert fresh.returncode == 0 and not (saves / "x2w-gamesave" / "profile.bin").exists(), fresh.stdout
        newest_fresh = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_fresh) as z:
            fresh_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert fresh_manifest["fresh_save"] is True and fresh_manifest["saves_at_end"] == 0 and fresh_manifest["saves_at_start"] == 0, fresh_manifest

        # Files written during the run under a search folder are listed in the manifest (where the title keeps its choices);
        # a file older than the run start is not.
        scan = tree / "scan_here"
        (scan / "Vendor" / "Game").mkdir(parents=True, exist_ok=True)
        written = scan / "Vendor" / "Game" / "privacy.dat"
        written.write_bytes(b"1")
        later = time.time() + 3600  # modified after the run starts
        os.utime(written, (later, later))
        stale = scan / "Vendor" / "Game" / "old.dat"
        stale.write_bytes(b"1")
        os.utime(stale, (1000000000, 1000000000))
        scanned = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-ScanRoots", str(scan))
        assert scanned.returncode == 0, (scanned.stdout, scanned.stderr)
        newest_scan = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest_scan) as z:
            scan_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        listed = scan_manifest["files_written"]
        assert len(listed) == 1 and listed[0].endswith("privacy.dat (1 bytes)"), scan_manifest["files_written"]

        # A tracked change in the runtime sources makes the checkout dirty: the build revision must carry -dirty to match.
        device = tree / "src" / "device.cpp"
        device.write_text(device.read_text() + "// local edit\n")
        dirty = run_script(shell, tree, "-NoBuild")
        assert dirty.returncode != 0 and head + "-dirty" in dirty.stdout + dirty.stderr, (dirty.stdout, dirty.stderr)
        assert "local changes in the runtime sources (src/device.cpp)" in dirty.stdout, dirty.stdout
        git(tree, "checkout", "-q", "--", "src")
        # A local edit of the tools only: the build of HEAD is used as it is, and the summary and manifest say why.
        pack = tree / "tools" / "startup" / "tester_pack.py"
        pack.write_text(pack.read_text() + "\n# local edit\n")
        tools_only = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0")
        assert tools_only.returncode == 0 and "using the build: the build %s has the same runtime sources" % head in tools_only.stdout, \
            (tools_only.stdout, tools_only.stderr)

        # A new commit that leaves the shader revision alone keeps the warm cache. It changes only tools, so the build of
        # the first commit is used without a rebuild (the tester pulled a tools fix).
        git(tree, "commit", "-q", "-am", "second")
        head2 = git(tree, "rev-parse", "--short=8", "HEAD")
        kept = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0")
        assert kept.returncode == 0 and "2 translations were already present at start" in kept.stdout, kept.stdout
        assert "the build %s has the same runtime sources as %s" % (head, head2) in kept.stdout, kept.stdout
        manifest = json.loads(zipfile.ZipFile(sorted((tree / "out" / "tester").glob("report_%s_*.zip" % head2))[-1]).read("manifest.json").decode("utf-8-sig"))
        assert manifest["build_commit"] == head and manifest["build_check"].startswith("the build %s" % head), manifest
        rev_file.write_text(REV_CPP % head2, encoding="ascii")

        # A bumped revision gets its own, empty shader cache folder; the old one stays.
        translator = tree / "tools" / "startup" / "gfx2hlsl.py"
        translator.write_text(TRANSLATOR % (29, 29), encoding="utf-8")
        git(tree, "commit", "-q", "-am", "third")
        rev_file.write_text(REV_CPP % git(tree, "rev-parse", "--short=8", "HEAD"), encoding="ascii")
        fresh = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0")
        assert fresh.returncode == 0 and "0 translations were already present at start" in fresh.stdout, fresh.stdout
        assert (cache.parent / "shader_cache_29-29-7").is_dir() and len(list(cache.glob("????????????????.txt"))) == 2

        # F1: after the zip is packed, one translator pass finishes the pending requests (cap: -FinishTranslateSeconds).
        fresh_cache = cache.parent / "shader_cache_29-29-7"
        (fresh_cache / "req_aaaaaaaaaaaaaaaa.a").write_bytes(b"x")
        (fresh_cache / "req_bbbbbbbbbbbbbbbb.a").write_bytes(b"x")
        for suffix in (".txt", ".vs.cso", ".ps.cso"):
            (fresh_cache / ("bbbbbbbbbbbbbbbb" + suffix)).write_bytes(b"x")
        finished = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-FinishTranslateSeconds", "60")
        assert finished.returncode == 0, (finished.stdout, finished.stderr)
        assert "finishing the translation: 1 request(s) pending" in finished.stdout, finished.stdout
        assert "pending before 1, after 0; ok 2, refused 0 in total (0 new)" in finished.stdout, finished.stdout
        assert (fresh_cache / "aaaaaaaaaaaaaaaa.txt").exists(), "the finish pass did not translate the pending request"
        assert finished.stdout.index("Send this file") < finished.stdout.index("finish translation done"), "zip first"
        # a pass that never ends is stopped at the cap, the exit code and the zip are unaffected
        (fresh_cache / "req_cccccccccccccccc.a").write_bytes(b"x")
        capped = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0", "-FinishTranslateSeconds", "3",
                            extra_env={"STUB_HANG": "1"})
        assert capped.returncode == 0, (capped.stdout, capped.stderr)
        assert "pending before 1, after 1" in capped.stdout and "stopped at the cap" in capped.stdout, capped.stdout
        assert not (fresh_cache / "cccccccccccccccc.txt").exists()
        assert capped.stdout.index("Send this file") < capped.stdout.index("finish translation done"), "the zip comes first"
        assert list((tree / "out" / "tester").glob("report_*.zip"))
        skipped = run_script(shell, tree, "-NoBuild", "-CaptureEvery", "0")
        assert "finishing the translation" not in skipped.stdout, "-FinishTranslateSeconds 0 must skip the pass"
    print("tester_run.ps1 under %s: stale build refused, per-revision shader cache, run arguments, -Deep, zip and manifest pass"
          % Path(shell).name)


def check_targets():
    """tester_run.ps1 builds the startup_runtime target, whose DLLs are the ones run.ps1 copies."""
    script = (STARTUP / "tester_run.ps1").read_text(encoding="utf-8")
    assert "--target startup_runtime" in script
    dlls = re.findall(r'"([A-Za-z0-9_]+)"', re.search(r"\$runDlls = @\((.*?)\)", script).group(1))
    run_dlls = re.findall(r'"(\w+)"', re.search(r'foreach \(\$dll in (.*?)\) \{', (STARTUP / "run.ps1").read_text(encoding="utf-8")).group(1))
    assert dlls == run_dlls, (dlls, run_dlls)
    assert "add_custom_target(startup_runtime)" in (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
    print("tester_run.ps1 builds startup_runtime for the %d DLLs run.ps1 copies" % len(run_dlls))


STUB_CMAKE = """\
cmake_minimum_required(VERSION 3.20)
project(stub NONE)
add_custom_target(x2w_build_rev_generate COMMAND ${CMAKE_COMMAND} "-DSRC=${CMAKE_SOURCE_DIR}"
    "-DOUT=${CMAKE_BINARY_DIR}/generated/x2w_build_rev.cpp" -P "${CMAKE_SOURCE_DIR}/rev.cmake")
foreach(pair IN ITEMS xmem_shim:xmem d3d12_xs_shim:d3d12_xs startup_stub_acphal:AcpHal startup_stub_chad:chad
        startup_stub_xapu:xapu startup_stub_xcrdapi:XCRDAPI startup_stub_xg_xs:xg_xs startup_stub_gameinput:GameInput)
    string(REPLACE ":" ";" parts "${pair}")
    list(GET parts 0 target)
    list(GET parts 1 dll)
    add_custom_target(${target} COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/${dll}.dll")
    list(APPEND runtime_targets ${target})
endforeach()
add_custom_target(startup_runtime)
add_dependencies(startup_runtime x2w_build_rev_generate ${runtime_targets})
# Everything else a full build would compile: it must stay unbuilt for a run.
add_custom_target(all_targets ALL)
add_dependencies(all_targets startup_runtime)
add_custom_target(decoy_test_exe ALL COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/decoy.txt")
"""


REV_CMAKE = r"""
execute_process(COMMAND git -C "${SRC}" rev-parse --short=8 HEAD OUTPUT_VARIABLE rev OUTPUT_STRIP_TRAILING_WHITESPACE)
file(WRITE "${OUT}" "const char x2w_build_rev[] = \"${rev}\";\n")
"""


def check_build(shell):
    """A configured but unbuilt build directory: tester_run.ps1 builds only the run's targets (-FullBuild: all)."""
    if not shutil.which("cmake"):
        print("startup_tester_run_ps1_check: no cmake on this host: build check skipped")
        return
    with tempfile.TemporaryDirectory() as root:
        tree, head, _ = make_tree(root)
        build = tree / "out" / "build" / "release"
        shutil.rmtree(build)
        (tree / "CMakeLists.txt").write_text(STUB_CMAKE, encoding="utf-8")
        (tree / "rev.cmake").write_text(REV_CMAKE, encoding="utf-8")
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []  # as the real presets: single config, no MSBuild
        configured = subprocess.run(["cmake", *generator, "-S", str(tree), "-B", str(build)], capture_output=True, text=True)
        assert configured.returncode == 0, configured.stdout + configured.stderr
        git(tree, "add", "CMakeLists.txt", "rev.cmake")
        git(tree, "commit", "-q", "-m", "cmake")
        head = git(tree, "rev-parse", "--short=8", "HEAD")
        assert not (build / "xmem.dll").exists()
        # cmake writes a warning to stderr (as the real one does) while this script's stderr is captured: under Windows
        # PowerShell 5.1 that is an error record, and with $ErrorActionPreference Stop it used to end the script. The
        # build is judged by its exit code.
        shim = Path(root) / "shim"
        shim.mkdir()
        (shim / "cmake.cmd").write_text('@echo CMake Warning: stub warning on stderr 1>&2\r\n@"%s" %%*\r\n' % shutil.which("cmake"),
                                        encoding="ascii")
        result = run_script(shell, tree, "-CaptureEvery", "0", extra_path=shim, merge_stderr=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "stub warning on stderr" in result.stdout + result.stderr, ("the shim did not run", result.stdout[-1500:], result.stderr[-500:])
        assert "building startup_runtime" in result.stdout, result.stdout
        for dll in DLLS:
            assert (build / (dll + ".dll")).exists(), dll + ".dll was not built"
        assert (build / "generated" / "x2w_build_rev.cpp").read_text().count('"%s"' % head) == 1
        assert not (build / "decoy.txt").exists(), "a target the run does not need was built"
        # A missing DLL is rebuilt even when the revision stamp is current.
        (build / "GameInput.dll").unlink()
        again = run_script(shell, tree, "-CaptureEvery", "0")
        assert again.returncode == 0 and (build / "GameInput.dll").exists() and "GameInput.dll" in again.stdout, again.stdout
        # -FullBuild builds the rest too (the revision is current, so a DLL has to be missing to trigger a build).
        (build / "xmem.dll").unlink()
        full = run_script(shell, tree, "-CaptureEvery", "0", "-FullBuild")
        assert full.returncode == 0 and "-FullBuild" in full.stdout and (build / "decoy.txt").exists(), (full.stdout, full.stderr)
    print("tester_run.ps1 builds only the run's targets (and everything with -FullBuild) under %s" % Path(shell).name)


def main():
    check_targets()
    available = shells()
    if not available or not shutil.which("git"):
        print("startup_tester_run_ps1_check: no PowerShell or no git on this host: skipped")
        return 0
    if os.name != "nt":
        print("startup_tester_run_ps1_check: not a Windows host: skipped")
        return 0
    for shell in available:
        check_shell(shell)
        check_build(shell)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # the pipeline shows only stdout of a failing check
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
