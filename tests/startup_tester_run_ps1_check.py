#!/usr/bin/env python3
"""Check tools/startup/tester_run.ps1 in a scratch tree: build check, fresh shader cache, run arguments, zip.

The scratch tree is a throw-away git repository that holds copies of tester_run.ps1 and tester_pack.py, a stub
run.ps1 that records its arguments and writes a small run folder, and a generated x2w_build_rev.cpp. No title, no
debugger and no GPU are involved. Needs git and a PowerShell (powershell or pwsh); without either it prints a
notice and passes, because Linux hosts cannot run the script.

  - a build from another commit is refused with -NoBuild, and accepted when its revision matches HEAD
  - the old shader cache folder is moved aside and the stub sees an empty one
  - run.ps1 gets -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000, the menus input script, the capture
    interval and X2W_D3D12_BUDGET=1; -Deep adds -TraceCalls and -TraceApi
  - the zip holds every file of the run folder and a valid manifest, and never the shader cache
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STARTUP = REPO / "tools" / "startup"
STUB_RUN = r"""
param([string]$Exe, [string]$DataDir, [string]$LogDir, [int]$Seconds, [string]$BuildDir, [string]$ShaderCache,
      [int]$ContinueAsserts, [string]$InputScript, [int]$CaptureEvery, [switch]$Render, [switch]$Translate,
      [switch]$FakeUser, [switch]$GpuMirror, [switch]$TraceCalls, [switch]$TraceApi)
$seen = @(Get-ChildItem -LiteralPath $ShaderCache -Force | ForEach-Object { $_.Name })
$record = [ordered]@{
    Exe = $Exe; DataDir = $DataDir; Seconds = $Seconds; ContinueAsserts = $ContinueAsserts; InputScript = $InputScript
    CaptureEvery = $CaptureEvery; Render = [bool]$Render; Translate = [bool]$Translate; FakeUser = [bool]$FakeUser
    GpuMirror = [bool]$GpuMirror; TraceCalls = [bool]$TraceCalls; TraceApi = [bool]$TraceApi
    Budget = $env:X2W_D3D12_BUDGET; CacheEntries = $seen
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
exit 0
"""
REV_CPP = 'const char x2w_build_rev[] = "%s";\n'


def shells():
    found = [shutil.which(name) for name in ("powershell", "pwsh")]
    return [path for path in found if path]


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, check=True).stdout.strip()


def make_tree(root):
    tree = Path(root) / "tree"
    (tree / "tools" / "startup").mkdir(parents=True)
    for name in ("tester_run.ps1", "tester_pack.py"):
        shutil.copy(STARTUP / name, tree / "tools" / "startup" / name)
    (tree / "tools" / "startup" / "run.ps1").write_text(STUB_RUN, encoding="utf-8")
    git(tree, "init", "-q")
    git(tree, "config", "user.email", "check@example.invalid")
    git(tree, "config", "user.name", "check")
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", "scratch")
    head = git(tree, "rev-parse", "--short=8", "HEAD")
    gen = tree / "out" / "build" / "release" / "generated"
    gen.mkdir(parents=True)
    return tree, head, gen / "x2w_build_rev.cpp"


def run_script(shell, tree, *args):
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    for key in [k for k in env if k.startswith("X2W_")]:
        del env[key]
    command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tree / "tools" / "startup" / "tester_run.ps1"),
               "-Exe", "title.exe", "-DataDir", "data", *args]
    return subprocess.run(command, capture_output=True, text=True, errors="replace", env=env, timeout=300, cwd=str(tree))


def check_shell(shell):
    with tempfile.TemporaryDirectory() as root:
        tree, head, rev_file = make_tree(root)
        cache = tree / "out" / "startup" / "shader_cache"
        cache.mkdir(parents=True)
        (cache / "old_request.bin").write_bytes(b"x")

        rev_file.write_text(REV_CPP % "00000000", encoding="ascii")
        stale = run_script(shell, tree, "-NoBuild")
        assert stale.returncode != 0 and "00000000" in stale.stdout + stale.stderr and head in stale.stdout + stale.stderr, \
            (shell, "a build from another commit was not refused", stale.stdout, stale.stderr)
        assert (cache / "old_request.bin").exists(), "the cache was touched before the build check passed"
        assert not list((tree / "out" / "tester").glob("*.zip")) if (tree / "out" / "tester").exists() else True

        rev_file.write_text(REV_CPP % head, encoding="ascii")
        ok = run_script(shell, tree, "-NoBuild", "-Seconds", "200", "-CaptureEvery", "10")
        assert ok.returncode == 0, (shell, ok.stdout, ok.stderr)
        zips = sorted((tree / "out" / "tester").glob("report_%s_*.zip" % head))
        assert len(zips) == 1, zips
        assert "Send this file" in ok.stdout, ok.stdout
        assert "first pad poll at 151.9 s" in ok.stdout, ok.stdout
        with zipfile.ZipFile(zips[0]) as z:
            names = sorted(z.namelist())
            manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
            args = json.loads(z.read("args.json").decode("utf-8-sig"))
        assert names == sorted(["finished.txt", "events.txt", "stdout.txt", "stderr.txt", "files.txt", "trap.log",
                                "threads.txt", "translate.txt", "translate_err.txt", "window.png", "window_015s.png",
                                "args.json", "debugger_summary.out", "manifest.json"]), names
        assert args["Render"] and args["Translate"] and args["FakeUser"] and args["GpuMirror"], args
        assert args["ContinueAsserts"] == 1000 and args["Seconds"] == 200 and args["CaptureEvery"] == 10, args
        assert args["Budget"] == "1" and not args["TraceCalls"] and not args["TraceApi"], args
        assert args["InputScript"].startswith("80@5,80@11,4@17,80@23"), args["InputScript"]
        assert not args["CacheEntries"], "the run saw the old shader cache: %r" % (args["CacheEntries"],)
        assert manifest["commit"] == head and manifest["build_commit"] == head and manifest["dirty"] is False
        assert manifest["seconds"] == 200 and manifest["deep"] is False and manifest["input_script"] == args["InputScript"]
        assert manifest["env"].get("X2W_D3D12_BUDGET") == "1" and all(k.startswith("X2W_") for k in manifest["env"])
        assert manifest["first_poll_seconds"] == 151.9
        assert manifest["parts"] == 1 and "gpu" in manifest and "windows_build" in manifest
        moved = [p.name for p in cache.parent.iterdir() if p.name.startswith("shader_cache.old-")]
        assert len(moved) == 1 and (cache.parent / moved[0] / "old_request.bin").exists(), moved
        assert not any(n.endswith(".bin") for n in names), "shader cache content travelled"

        deep = run_script(shell, tree, "-NoBuild", "-Deep", "-CaptureEvery", "0", "-Input", "4@60,1@70")
        assert deep.returncode == 0, (shell, deep.stdout, deep.stderr)
        newest = sorted((tree / "out" / "tester").glob("report_*.zip"), key=os.path.getmtime)[-1]
        with zipfile.ZipFile(newest) as z:
            deep_args = json.loads(z.read("args.json").decode("utf-8-sig"))
            deep_manifest = json.loads(z.read("manifest.json").decode("utf-8-sig"))
        assert deep_args["TraceCalls"] and deep_args["TraceApi"] and deep_args["InputScript"] == "4@60,1@70", deep_args
        assert deep_manifest["deep"] is True and deep_args["CaptureEvery"] == 0, deep_manifest

        # A tracked change makes the checkout dirty: the build revision must carry -dirty to match.
        (tree / "tools" / "startup" / "tester_pack.py").write_text(
            (tree / "tools" / "startup" / "tester_pack.py").read_text() + "\n# local edit\n")
        dirty = run_script(shell, tree, "-NoBuild")
        assert dirty.returncode != 0 and head + "-dirty" in dirty.stdout + dirty.stderr, (dirty.stdout, dirty.stderr)
    print("tester_run.ps1 under %s: stale build refused, fresh cache, run arguments, -Deep, zip and manifest pass"
          % Path(shell).name)


def main():
    available = shells()
    if not available or not shutil.which("git"):
        print("startup_tester_run_ps1_check: no PowerShell or no git on this host: skipped")
        return 0
    if os.name != "nt":
        print("startup_tester_run_ps1_check: not a Windows host: skipped")
        return 0
    for shell in available:
        check_shell(shell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
