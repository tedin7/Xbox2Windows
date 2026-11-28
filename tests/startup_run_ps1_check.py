#!/usr/bin/env python3
"""Check that tools/startup/run.ps1 exits with the startup debugger's exit code.

On Windows the script runs under every PowerShell found (powershell is Windows
PowerShell 5.1, pwsh is PowerShell 7) in a scratch tree whose debug_launch.py
is a stub, so no process is ever started under a debugger. On other hosts,
without a PowerShell, or with --text-only, the launcher statements are checked
as text. Mutated copies of the script must fail the same checks.
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

RUN_PS1 = Path(__file__).resolve().parent.parent / "tools" / "startup" / "run.ps1"
DLLS = ["xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput"]
STUB_LINES = [f"stub debugger line {n}" for n in range(1, 8)]
STUB_ERROR = "synthetic debugger error"
HEAD_LINES = 5  # run.ps1 prints this many lines of "<LogDir>.out"
# The stub stands in for tools/startup/debug_launch.py: it records its arguments,
# prints seven lines and exits with the code under test.
STUB = """\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

record = [sys.argv[1:], os.environ.get("X2W_TRACE_API"), os.environ.get("X2W_NATIVE_SHADERS"),
          os.environ.get("X2W_TITLE_ID"), os.environ.get("X2W_STORAGE_MB")]
Path(__file__).with_name("debugger.json").write_text(json.dumps(record), encoding="utf-8")
# Under -Translate, wait until the watcher stub has started so run.ps1 cannot stop it before then.
watch = Path(__file__).with_name("watch.json")
deadline = time.monotonic() + 45
if Path(__file__).with_name("translate.flag").exists():
    while not watch.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("watcher did not publish watch.json")
        time.sleep(0.05)
if {early_exit}:
    # The watcher has published its PID. Wait for the OS to confirm it is gone before returning.
    pid = json.loads(watch.read_text(encoding="utf-8"))[0]
    pid_column = f'"{{pid}}"'.encode("ascii")
    while pid_column in subprocess.check_output(["tasklist", "/FI", f"PID eq {{pid}}", "/FO", "CSV", "/NH"]):
        if time.monotonic() >= deadline:
            raise RuntimeError("early-exit watcher did not exit")
        time.sleep(0.05)
for n in range(1, 8):
    print("stub debugger line", n)
if {emit_stderr}:
    print("synthetic debugger error", file=sys.stderr)
sys.exit({code})
"""
# Stands in for tools/startup/gfx2hlsl.py under -Translate: starts a child process, as the parallel
# workers are, records the PIDs and arguments, then waits to be stopped. The early-exit case
# publishes just the watcher's PID and arguments and returns without spawning a child.
WATCH_STUB = """\
# The revisions the launcher compares with the cache: fnv(b'x2w-startup-shader-90'), b'x2w-startup-shader-90cs',
# b'x2w-startup-mesh-57-9' (synthetic, as gfx2hlsl.py spells them).
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# run.ps1 lowers the watcher to BELOW_NORMAL right after starting it; the pool workers it creates later inherit that.
# The stub waits for it (as the real watcher does while it imports and scans) and records its class and its child's.
import ctypes
k32 = ctypes.windll.kernel32
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
k32.OpenProcess.restype = ctypes.c_void_p
wait_until = time.monotonic() + 30
while k32.GetPriorityClass(k32.GetCurrentProcess()) != 0x4000 and time.monotonic() < wait_until:
    time.sleep(0.02)
child = None if {early_exit} else subprocess.Popen([sys.executable, "-c", "from threading import Event; Event().wait(180)"])
classes = [k32.GetPriorityClass(k32.GetCurrentProcess())]
if child:
    classes.append(k32.GetPriorityClass(k32.OpenProcess(0x1000, False, child.pid)))
Path(__file__).with_name("priority.json").write_text(json.dumps(classes), encoding="utf-8")
record = Path(__file__).with_name("watch.json")
temp = record.with_suffix(".tmp")
temp.write_text(json.dumps([os.getpid(), child.pid if child else None] + sys.argv[1:]), encoding="utf-8")
temp.replace(record)
if child:
    threading.Event().wait(180)
"""
STOP = "taskkill.exe /T /F /PID $translator.Id"
# Text-only hosts check that the native command is inside both the live-watcher
# guard and the child scope that handles Windows PowerShell's native stderr.
WATCH_CLEANUP = re.compile(
    r"if\s*\(\s*\$translator\s+-and\s+-not\s+\$translator\.HasExited\s*\)\s*\{\s*"
    r"(?:\$treeIds\s*=\s*@\(&\s+\$treeOf\s+\$translator\.Id\)\s*)?&\s*\{\s*"
    r"""\$ErrorActionPreference\s*=\s*['"]Continue['"]\s+&\s+taskkill\.exe\b[^|#\n]*\s/T\b""", re.I)
CAPTURE = re.compile(r"\$(?:script:)?(\w+)\s*=\s*\$LASTEXITCODE", re.I)
EXIT = re.compile(r"exit\s+\$(\w+)", re.I)
HEAD = 'Get-Content "$LogDir.out" -TotalCount 5'
# Without -FakeUser, user APIs hit the host runtime; run.ps1 must warn (docs/startup-runtime.md).
FAKE_USER_WARN = re.compile(
    r"if\s*\(\s*-not\s+\$FakeUser\s+-and\s+\$env:X2W_FAKE_USER\s+-ne\s+\"1\"\s*\)\s*\{\s*"
    r'Write-Warning\s+"Without -FakeUser, user API calls go to the host gaming runtime',
    re.I | re.S,
)
FAKE_USER_WARNING = (
    "Without -FakeUser, user API calls go to the host gaming runtime (docs/startup-runtime.md)."
)
FAKE_USER_WARN_LINE = f'Write-Warning "{FAKE_USER_WARNING}"'
WARNING_PROBE = "x2w warning prefix probe"
WARNING_PREFIXES = {}


def unwrap_first_line(printed, want):
    """Join the leading lines of `printed` that PowerShell wrapped at its host width back into `want`.

    The host wraps a long Write-Warning at the console width it finds (80 columns when it is not
    attached to a wide console, which depends on load and on the parent), so the one warning can
    arrive as two lines. Only a split of exactly `want` is joined; any other text is left as printed.
    """
    for count in range(1, len(printed) + 1):
        if " ".join(printed[:count]) == want:
            return [want, *printed[count:]]
    return printed


def warning_prefix(shell, env):
    """Return the prefix this shell puts before a Write-Warning message, such as "WARNING: ".

    PowerShell localizes the prefix with the host's UI language ("ADVERTENCIA: " on Spanish
    Windows), so it is read from the shell under test with the same environment as run.ps1.
    """
    if shell not in WARNING_PREFIXES:
        out = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-NonInteractive", "-Command", f"Write-Warning '{WARNING_PROBE}'"],
                             env=env, capture_output=True, text=True, errors="replace", timeout=60)
        lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        if out.returncode or len(lines) != 1 or not lines[0].endswith(WARNING_PROBE) or lines[0] == WARNING_PROBE:
            raise ValueError(f"Write-Warning prefix probe exited {out.returncode}\n"
                             f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}")
        WARNING_PREFIXES[shell] = lines[0][:-len(WARNING_PROBE)]
    return WARNING_PREFIXES[shell]


def decode(raw):
    # Windows PowerShell 5.1 writes a redirected file as UTF-16 with a byte order mark.
    wide = raw[:2] in (b"\xff\xfe", b"\xfe\xff")
    return raw.decode("utf-16" if wide else "utf-8-sig", errors="replace")


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, errors="replace")
    return f'"{pid}"' in out.stdout


# A shader cache of an older build (#2815): its revision.txt and one request, against the stub's revisions.
OLDER_CACHE = {"revision.txt": b"graphics x2w-startup-shader-89\ncompute x2w-startup-shader-89cs\n",
               "req_0123456789abcdef.a": b""}
REVISION_PHRASES = ("was written by stand-ins of revision graphics x2w-startup-shader-89",
                    "but the translator", "is revision graphics x2w-startup-shader-90", "Fix: rebuild the stand-ins")


def run_case(shell, script, code, work, translate=False, switches=(), passed=(), fake_user_env=False,
             early_exit=False, trace_api_env=None, absolute_build=False, missing_build=False, cache_files=None,
             emit_stderr=False, missing_python=False, real_log_error=False, native=False, manifest="absent"):
    """Run one copy of the script on a stub that exits with `code`; return what is wrong.

    `switches` are extra run.ps1 switches and `passed` the debugger arguments they must add.
    `translate` exercises -Render -Translate and requires the watcher to stop with its child.
    With `early_exit` the watcher has no child and exits before the debugger finishes.
    `native` passes -Render -NativeShaders <dir>: the stand-in gets the directory and no watcher starts.
    This checks the log head and debugger exit code. The live watcher case checks worker cleanup.
    """
    root = Path(tempfile.mkdtemp(dir=work)).resolve()
    startup = root / "tools" / "startup"
    startup.mkdir(parents=True)
    (startup / "run.ps1").write_bytes(script.encode("utf-8"))
    if real_log_error:
        for name in ("debug_launch.py", "launch.py", "host_cpu.py", "window_capture.py", "commit_headroom.py", "hook_window.py",
                     "log_tail.py", "stall_watchdog.py"):
            shutil.copyfile(RUN_PS1.parent / name, startup / name)
    else:
        (startup / "debug_launch.py").write_text(STUB.format(code=code, early_exit=early_exit, emit_stderr=emit_stderr), encoding="utf-8")
    (startup / "gfx2hlsl.py").write_text(WATCH_STUB.format(early_exit=early_exit), encoding="utf-8")
    shutil.copyfile(RUN_PS1.with_name("shader_cache_revision.py"), startup / "shader_cache_revision.py")
    if cache_files is not None:  # a shader cache left by an earlier run
        (root / "cache").mkdir()
        for name, data in cache_files.items():
            (root / "cache" / name).write_bytes(data)
    build = root / ("build [with spaces]" if absolute_build else "bld")
    build.mkdir()
    for dll in DLLS:  # the stand-ins the script copies next to the executable
        (build / f"{dll}.dll").write_bytes(b"")
    exe, data, log = root / "run" / "t.exe", root / ("data [test]" if manifest.startswith("bracket") else "data"), root / "log"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    data.mkdir()
    if real_log_error:
        (log / "events.txt").mkdir(parents=True)
    if manifest in ("normal", "bracket"):
        (data / "MicrosoftGame.config").write_text("<TitleId>1234</TitleId><SizeMB>42</SizeMB>", encoding="utf-8")
    if manifest.startswith("bracket"):
        decoy = root / "data t"
        decoy.mkdir()
        (decoy / "MicrosoftGame.config").write_text("<TitleId>9999</TitleId><SizeMB>99</SizeMB>", encoding="utf-8")
    env = dict(os.environ)
    env.pop("X2W_TITLE_ID", None)
    env.pop("X2W_STORAGE_MB", None)
    # Compare PowerShell's warning text consistently on hosts that enable ANSI colors.
    env["NO_COLOR"] = "1"
    env.pop("X2W_FAKE_USER", None)
    env.pop("X2W_NATIVE_SHADERS", None)
    if trace_api_env is None:
        env.pop("X2W_TRACE_API", None)
    else:
        env["X2W_TRACE_API"] = trace_api_env
    if fake_user_env:
        env["X2W_FAKE_USER"] = "1"
    # In this tree run.ps1 finds `python` through PATH. This interpreter's directory
    # goes first so the stub runs under the Python that runs this check.
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    if missing_python:
        env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    build_arg = str(build) if absolute_build else "bld"
    if missing_build:
        build_arg = str(root / "missing build")
    command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-NonInteractive", "-File", str(startup / "run.ps1"),
               "-Exe", str(exe), "-DataDir", str(data), "-LogDir", str(log),
               "-BuildDir", build_arg,
               "-Seconds", "9", "-MemoryGiB", "3", "-ContinueAsserts", "4", "-AssertSites", "2", *switches]
    sdk = root / "dxc"
    if native:
        sdk.mkdir()
        command += ["-Render", "-NativeShaders", str(sdk), "-ShaderCache", "cache"]
    if translate:
        command += ["-Render", "-Translate", "-ShaderCache", "cache"]
        (startup / "translate.flag").write_bytes(b"")
    step = f"debugger exit {code}, args {command[command.index('-File') + 2:]!r}"
    for option, enabled in (
        ("early watcher exit", early_exit), ("stale cache", cache_files is not None),
        ("debugger stderr", emit_stderr), ("missing Python", missing_python),
        ("real debugger log error", real_log_error), ("FakeUser environment", fake_user_env),
        (f"TraceApi environment={trace_api_env}", trace_api_env is not None),
    ):
        if enabled:
            step += f", {option}"
    summary = Path(str(log) + ".out")

    def report(problems):
        if not problems:
            return []
        log_output = decode(summary.read_bytes()) if summary.is_file() else "(missing)"
        return [f"{step}\n" + "\n".join(problems) +
                f"\nrun.ps1 stdout:\n{output}\nrun.ps1 stderr:\n{error_text}"
                f"\ndebugger summary:\n{log_output}"]

    # A surviving watcher can keep a capture pipe open after PowerShell exits.
    # Capture to files so we can check the watcher's PID as soon as the shell exits.
    with (tempfile.TemporaryFile(mode="w+", errors="replace", dir=root) as stdout,
          tempfile.TemporaryFile(mode="w+", errors="replace", dir=root) as stderr):
        try:
            out = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            out = None
            launch_failure = str(exc)
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read()
        error_text = stderr.read()
    printed = [line.strip() for line in output.splitlines() if line.strip()]
    if out is None:
        return report([launch_failure])
    if missing_python:
        return report((["missing Python returned success"] if out.returncode == 0 else []) + (
            ["missing Python launched debugger"] if (startup / "debugger.json").exists() else []))
    if missing_build:
        return report((["missing build directory returned success"] if out.returncode == 0 else []) + (
            ["missing build directory launched debugger"] if (startup / "debugger.json").exists() else []))
    if real_log_error:
        text = decode(summary.read_bytes()) if summary.is_file() else ""
        return report(([f"real debugger log error returned {out.returncode}, expected 1"]
                       if out.returncode != 1 else []) +
                      (["real debugger log error was not captured in the summary"]
                       if "PermissionError" not in text else []))
    found = []
    for dll in DLLS:
        if not (exe.parent / f"{dll}.dll").is_file():
            found.append(f"stand-in {dll}.dll was not copied")
    if out.returncode != code:
        found.append(f"stub debugger exited {code}, run.ps1 exited {out.returncode}")
    summary_lines = [line.strip() for line in decode(summary.read_bytes()).splitlines() if line.strip()] if summary.is_file() else []
    want_output = summary_lines[:HEAD_LINES] if emit_stderr else STUB_LINES[:HEAD_LINES]
    if cache_files is not None:
        # One warning with both revisions and the fixes; PowerShell may wrap it, so compare without whitespace.
        flat = re.sub(r"\s", "", " ".join(printed))
        missing = [phrase for phrase in REVISION_PHRASES if re.sub(r"\s", "", phrase) not in flat]
        if missing or flat.count("Fix:rebuildthestand-ins") != 1:
            found.append(f"stale shader cache: no single revision warning with {missing} (printed {printed})")
        printed = [line for line in printed if line in STUB_LINES]  # the case runs with -FakeUser: no other warning
    if not fake_user_env and "-FakeUser" not in switches:
        try:
            prefix = warning_prefix(shell, env)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            found.append(f"warning prefix probe: {exc}")
        else:
            want_output = [prefix + FAKE_USER_WARNING, *want_output]
            printed = unwrap_first_line(printed, want_output[0])
    if printed != want_output:
        found.append(f"stub exit {code}: printed {printed}, expected {want_output} "
                     f"(stderr: {error_text!r})")
    if not summary.is_file():
        found.append(f"stub exit {code}: {summary.name} was not written")
    elif emit_stderr and ([line for line in summary_lines if line in STUB_LINES] != STUB_LINES or
                          not any(STUB_ERROR in line for line in summary_lines)):
        found.append(f"stub exit {code}: {summary.name} does not hold stdout and stderr")
    elif not emit_stderr and summary_lines != STUB_LINES:
        found.append(f"stub exit {code}: {summary.name} does not hold the debugger's output")
    want = [str(exe), str(data), str(log), "--timeout", "9", "--memory-gib", "3",
            "--trace-files", "1", "--continue-asserts", "4", "--assert-sites", "2", *passed]
    record = startup / "debugger.json"
    got = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else None
    expected = [want, "1" if "-TraceApi" in switches else trace_api_env,
                str(sdk) if native else None,
                "1234" if manifest in ("normal", "bracket") else None,
                "42" if manifest in ("normal", "bracket") else None]
    if got != expected:
        found.append(f"stub exit {code}: debugger [argv, trace, native shaders, title ID, storage MB] was {got}, expected {expected}")
    watch = startup / "watch.json"
    if translate:
        started = json.loads(watch.read_text(encoding="utf-8")) if watch.is_file() else None
        if not started or started[2:] != ["--watch", str(root / "cache")]:
            found.append(f"-Translate: the translator received {started}, expected --watch {root / 'cache'}")
        else:
            priority = startup / "priority.json"
            classes = json.loads(priority.read_text(encoding="utf-8")) if priority.is_file() else None
            want_classes = [0x4000] if started[1] is None else [0x4000, 0x4000]  # BELOW_NORMAL_PRIORITY_CLASS
            if classes != want_classes:
                found.append(f"-Translate: the translator and its child ran at priority classes {classes}, expected BELOW_NORMAL {want_classes}")
            for pid, what in ((started[0], "translator"), (started[1], "translator's child process")):
                if pid is not None and alive(pid):
                    subprocess.run(["taskkill", "/T", "/PID", str(pid), "/F"], capture_output=True)
                    found.append(f"-Translate: the {what} was still running after run.ps1 exited")
    elif watch.is_file():
        found.append(f"stub exit {code}: the translator started without -Translate")
    return report(found)


def run_problems(shell, script, work, first_only):
    if shell is None:
        return text_problems(script)
    found = []
    for code in (2, 0, 7):  # 2 is the debugger's "CreateProcess failed"; 7 rules out a fixed code
        found += run_case(shell, script, code, work)
        if found and first_only:
            return found
    found += run_case(shell, script, 7, work, absolute_build=True)
    for manifest in ("normal", "bracket", "bracket_missing"):
        found += run_case(shell, script, 0, work, manifest=manifest)
    if found and first_only:
        return found
    for code in (0, 7):
        found += run_case(shell, script, code, work, emit_stderr=True)
        if found and first_only:
            return found
    found += run_case(shell, script, 7, work, translate=True, emit_stderr=True)
    if found and first_only:
        return found
    found += run_case(shell, script, 0, work, switches=["-FakeUser"])
    found += run_case(shell, script, 0, work, fake_user_env=True)
    if found and first_only:
        return found
    # -TraceCalls arms the debugger's call hooks; without it the arguments above stay as they are.
    found += run_case(shell, script, 0, work, switches=["-TraceCalls"], passed=["--trace-calls", "1"])
    found += run_case(shell, script, 0, work, switches=["-CaptureEvery", "5"], passed=["--capture-every", "5"])
    found += run_case(shell, script, 0, work, switches=["-CommitHeadroomMiB", "256", "-CommitWarnMiB", "2048"], passed=["--commit-headroom-mib", "256", "--commit-warn-mib", "2048"])
    found += run_case(shell, script, 0, work, switches=["-TraceApi"], trace_api_env="0")
    found += run_case(shell, script, 0, work, trace_api_env="1")
    found += run_case(shell, script, 0, work, native=True)
    if found and first_only:
        return found
    found += run_case(shell, script, 7, work, translate=True)
    found += run_case(shell, script, 7, work, translate=True, switches=["-FakeUser"], cache_files=OLDER_CACHE)
    for code in (0, 7):
        found += run_case(shell, script, code, work, translate=True, early_exit=True)
    return found


def text_problems(script):
    code = re.sub(r"<#.*?#>", "", script, flags=re.S)  # the help block names debug_launch.py too
    statements = []
    for line in (line.strip() for line in code.splitlines()):
        if not line or line.startswith("#"):
            continue
        if statements and statements[-1].endswith("`"):  # continuation line
            statements[-1] = statements[-1][:-1] + line
        else:
            statements.append(line)
    # The debugger call is the one statement that names debug_launch.py, however
    # the interpreter is spelled (python, & $python, ...).
    calls = [i for i, statement in enumerate(statements) if "debug_launch.py" in statement]
    if len(calls) != 1:
        return [f"expected one statement that names debug_launch.py, found {len(calls)}"]
    after = statements[calls[0] + 1:]
    capture = CAPTURE.fullmatch(after[0]) if after else None
    if not capture:
        return ["the statement after the debugger call does not store $LASTEXITCODE"]
    found = []
    leave = EXIT.fullmatch(after[-1])
    if not leave or leave.group(1).lower() != capture.group(1).lower():
        found.append(f"the script does not end with 'exit ${capture.group(1)}'")
    if HEAD not in after[1:-1]:
        found.append("the log head is not printed between the capture and the exit")
    if not FAKE_USER_WARN.search(code):
        found.append("missing Write-Warning when -FakeUser is not passed")
    before = "\n".join(statements[:calls[0]])
    if not re.search(r"^\[switch\]\s*\$TraceApi\b", before, re.I | re.M):
        found.append("missing -TraceApi switch")
    if not re.search(r"""^if\s*\(\s*\$TraceApi\s*\)\s*\{\s*\$env:X2W_TRACE_API\s*=\s*["']1["']\s*\}""",
                     before, re.I | re.M):
        found.append("missing -TraceApi environment opt-in before launch")
    if not re.search(r"\$env:X2W_NATIVE_SHADERS\s*=", before):
        found.append("missing -NativeShaders directory hand-over before launch")
    if not WATCH_CLEANUP.search("\n".join(after)):
        found.append("tree cleanup lacks taskkill /T, the live-watcher guard or the local Continue scope")
    return found


def mutants_of(script):
    lines = script.splitlines()
    is_exit = lambda line: bool(EXIT.fullmatch(line.strip()))
    is_capture = lambda line: bool(CAPTURE.fullmatch(line.strip()))
    early = []
    for line in lines:
        if line.strip() == HEAD:
            early += [other for other in lines if is_exit(other)]
        if not is_exit(line):
            early.append(line)
    no_fake_warn = [line for line in lines if FAKE_USER_WARN_LINE not in line]
    mutants = {
        "no capture and no exit": [line for line in lines if not is_exit(line) and not is_capture(line)],
        "fixed exit 2": ["exit 2" if is_exit(line) else line for line in lines],
        "exit before the log head": early,
        "no FakeUser warning": no_fake_warn,
        "no X2W_NATIVE_SHADERS": [line for line in lines if "$env:X2W_NATIVE_SHADERS" not in line],
        # The same text without the warning prefix: the run must still see Write-Warning's prefix.
        "FakeUser warning printed with Write-Host": [
            line.replace(FAKE_USER_WARN_LINE, f'Write-Host "{FAKE_USER_WARNING}"') for line in lines],
    }
    return {name: None if mutant == lines else "\n".join(mutant) + "\n" for name, mutant in mutants.items()}


def shell_failures(shell, script, tmp):
    """Every failure for one shell (None: the text-only checks), in a fixed order."""
    failures = []
    name = Path(shell).stem.lower() if shell else "text"
    bad = run_problems(shell, script, tmp, False)
    failures += [f"{name}: {item}" for item in bad]
    if bad:
        return failures
    for label, mutant in mutants_of(script).items():
        if mutant is None or not run_problems(shell, mutant, tmp, True):
            failures.append(f"{name}: mutant '{label}' was not detected")
    if shell:
        failures += [f"{name}: {item}" for item in run_case(shell, script, 7, tmp, real_log_error=True)]
        failures += [f"{name}: {item}" for item in run_case(shell, script, 7, tmp, missing_python=True)]
        failures += [f"{name}: {item}" for item in run_case(shell, script, 7, tmp, missing_build=True)]
        wrong_arg = script.replace("--continue-asserts $ContinueAsserts", "--continue-asserts 19")
        if wrong_arg == script:
            failures.append(f"{name}: could not inject the wrong debugger argument")
        else:
            problems = run_case(shell, wrong_arg, 7, tmp)
            if not any("debugger [argv" in item and "'19'" in item for item in problems):
                failures.append(f"{name}: wrong debugger argument was not detected: {problems}")
        # Expire the live fixture tree after HasExited has been read. The launcher's
        # own taskkill then hits the missing PID and writes native stderr.
        kill = f"& {STOP} 2>&1 | Out-Null"
        expire = kill + '\nif (-not $translator.WaitForExit(5000)) { throw "fixture watcher still alive" }\n'
        race = script.replace(kill, expire + kill)
        if race == script:
            failures.append(f"{name}: could not exercise the watcher exit during cleanup")
        else:
            failures += [f"{name}: watcher exit during cleanup: {item}"
                         for item in run_case(shell, race, 7, tmp, translate=True)]
        stopless = "\n".join(line for line in script.splitlines() if STOP not in line) + "\n"
        parent_only = script.replace(STOP, "Stop-Process -Id $translator.Id -Force")
        for mutant, who in ((stopless, "translator"), (parent_only, "translator's child process")):
            problems = run_case(shell, mutant, 7, tmp, translate=True)
            if not any(f"-Translate: the {who} was still running after run.ps1 exited" in item
                       for item in problems):
                failures.append(f"{name}: expected a surviving {who}, got {problems}")
    else:
        for removed in (' -and -not $translator.HasExited', '$ErrorActionPreference = "Continue"', '/T '):
            if not text_problems(script.replace(removed, '')):
                failures.append(f"text: removing {removed.strip()} escaped the cleanup check")
    return failures


def unwrap_problems():
    """Controls for unwrap_first_line: a wrapped warning joins, any other text is left alone."""
    want = "WARNING: Without -FakeUser, user API calls go to the host gaming runtime (docs/x.md)."
    head = "WARNING: Without -FakeUser, user API calls go to the host gaming runtime"
    cases = [
        ("one line", [want, "a"], [want, "a"]),
        ("wrapped at 80 columns", [head, "(docs/x.md).", "a"], [want, "a"]),
        ("different text", [head, "(docs/other.md).", "a"], [head, "(docs/other.md).", "a"]),
        ("missing warning", ["a"], ["a"]),
    ]
    return [f"unwrap_first_line {name}: got {got}, expected {expected}"
            for name, printed, expected in cases
            if (got := unwrap_first_line(printed, want)) != expected]


@contextmanager
def scratch_dir(*, attempts=40, delay=0.25, remove=shutil.rmtree):
    path = tempfile.mkdtemp()
    body_failed = False
    try:
        yield path
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            for attempt in range(1, attempts + 1):
                try:
                    remove(path)
                    break
                except OSError as error:
                    winerror = getattr(error, "winerror", None)
                    if not isinstance(error, PermissionError) and winerror not in (32, 5):
                        raise
                    if attempt == attempts:
                        raise
                    print(f"startup run.ps1: temp dir busy ({winerror}) {path}, retry {attempt}/{attempts}")
                    time.sleep(delay)
        except BaseException:
            if not body_failed:
                raise


def scratch_dir_problems(no_retry):
    busy = PermissionError("synthetic sharing violation")
    busy.winerror = 32
    missing = FileNotFoundError("synthetic missing directory")
    body_error = RuntimeError("synthetic body failure")
    cases = [
        ("transient lock", [busy] * 3, 40, None, None, 4),
        ("persistent lock", [busy] * 3, 3, None, busy, 3),
        ("other error", [missing], 3, None, missing, 1),
        ("body failure", [busy] * 3, 3, body_error, body_error, 3),
    ]
    failures = []
    for name, errors, attempts, body, expected_error, expected_calls in cases:
        calls = []

        def remove(path):
            calls.append(path)
            if len(calls) <= len(errors):
                raise errors[len(calls) - 1]
            shutil.rmtree(path)

        output = io.StringIO()
        caught = None
        path = None
        try:
            with redirect_stdout(output), scratch_dir(
                attempts=1 if no_retry else attempts, delay=0, remove=remove,
            ) as path:
                if body is not None:
                    raise body
        except Exception as error:
            caught = error
        finally:
            if path is not None and Path(path).exists():
                shutil.rmtree(path)
        if caught is not expected_error:
            failures.append(f"scratch_dir {name}: got {caught!r}, expected {expected_error!r}")
        if len(calls) != expected_calls:
            failures.append(f"scratch_dir {name}: removal calls {len(calls)}, expected {expected_calls}")
        expected_logs = [
            f"startup run.ps1: temp dir busy (32) {path}, retry {retry}/{attempts}"
            for retry in range(1, expected_calls)
        ]
        if output.getvalue().splitlines() != expected_logs:
            failures.append(f"scratch_dir {name}: unexpected retry output {output.getvalue()!r}")
    return failures


def main():
    args = sys.argv[1:]
    no_retry = args == ["--mutate", "no-cleanup-retry"]
    if args not in ([], ["--text-only"]) and not no_retry:
        sys.exit("usage: startup_run_ps1_check.py [--text-only | --mutate no-cleanup-retry]")
    failures = scratch_dir_problems(no_retry)
    if no_retry:
        for item in failures:
            print("FAIL", item)
        return 1 if failures else 0
    script = RUN_PS1.read_bytes().decode("utf-8")
    shells = []
    if sys.argv[1:]:
        mode = "text only: requested"
    elif os.name != "nt":  # the launcher and its backslash build paths are Windows-only
        mode = "text only: not a Windows host"
    else:
        shells = [found for found in map(shutil.which, ("powershell", "pwsh")) if found]
        mode = "executed with " + ", ".join(Path(shell).stem.lower() for shell in shells)
        if not shells:
            mode = "text only: no PowerShell found"
    failures += unwrap_problems()
    with scratch_dir() as tmp:
        # Each shell works in its own scratch dirs, so PowerShell 5.1 and 7 run side by side.
        with ThreadPoolExecutor(max_workers=len(shells) or 1) as pool:
            jobs = [pool.submit(shell_failures, shell, script, tmp) for shell in shells or [None]]
            for job in jobs:
                failures += job.result()
    for item in failures:
        print("FAIL", item)
    print(f"startup run.ps1 exit code: {len(failures)} failed ({mode})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
