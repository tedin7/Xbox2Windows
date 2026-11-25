#!/usr/bin/env python3
"""Check tools/startup/run.sh, the Linux launcher that runs the startup debugger inside Wine.

The script runs under bash in a scratch tree. WINE names a stub that runs the host
Python instead of a Windows one, and debug_launch.py and gfx2hlsl.py are stubs, so no
process is ever started under a debugger or Wine. The checks cover the exit code, the
debugger's arguments and environment, the DLL copies, the warnings, and that the
shader watcher and its child process run during the debugger call and are gone once
run.sh exits, also after a Ctrl-C. Mutated copies of the script must fail.
Exits 77 (skip) on Windows; on other hosts a missing bash or setsid is a failure.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RUN_SH = Path(__file__).resolve().parent.parent / "tools" / "startup" / "run.sh"
DLLS = ["xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput"]
STUB_LINES = [f"stub debugger line {n}" for n in range(1, 8)]
HEAD_LINES = 5  # run.sh prints this many lines of "<log>.out"
PERMISSIVE = ["X2W_AUDIO_PERMISSIVE", "X2W_D3D12_DEVICE_NOOP", "X2W_D3D12_LIST_NOOP",
              "X2W_D3D12_PERMISSIVE", "X2W_XMEM_PERMISSIVE", "X2W_RUNTIME_PROXY"]
CHECKED = PERMISSIVE + [
    "X2W_FAKE_USER", "X2W_XMEM_GPU_MIRROR", "X2W_HOST_CPU", "X2W_XMEM_STARTUP_BELOW_40", "X2W_HIDE_TITLE_WINDOW",
    "X2W_D3D12_RENDER", "X2W_DRAW_NULL_IDENTITY", "X2W_SHADER_CACHE", "X2W_TITLE_ID", "X2W_STORAGE_MB",
    "X2W_STORAGE_PATH", "X2W_INPUT_SCRIPT", "X2W_XMEM_SHARED_VIEWS", "WINEDLLOVERRIDES", "WINEFSYNC", "WINEDEBUG",
    "WINEPREFIX", "XDG_RUNTIME_DIR", "PULSE_SERVER", "X2W_FXC"]
GAME_OVERRIDES = "d3d12,d3d12core,dxgi,d3d11=n"  # docs/wine.md
FAKE_USER_WARNING = ("WARNING: Without --fake-user, user API calls go to the host gaming runtime "
                     "(docs/startup-runtime.md).")
TRANSLATE_WARNING = "WARNING: --translate needs --render (no shader cache without it); ignored"
HIDE_WARNING = "WARNING: --hide-title-window needs --render (no output window without it); it has no effect"
# Synthetic package values; not any title's. Only the first of each tag counts, as in run.ps1.
CONFIG = ("<Game>\r\n<Identity/>\r\n<TitleId>X2WTEST</TitleId>\r\n<TitleId>SECOND</TitleId>\r\n"
          "<PersistentLocalStorage><SizeMB>123</SizeMB></PersistentLocalStorage>\r\n</Game>\r\n")
# One line, lower-case tags first: run.ps1's -match takes the first tag in any case.
CONFIG_ONE_LINE = "<Game><titleid>X2WTEST</titleid><TitleId>SECOND</TitleId><sizemb>123</sizemb></Game>\n"

# Stands in for wine: "$WINE" python.exe -B script.py args... runs script.py (a Z: path) on the
# host Python and passes the other arguments as they are, so the stubs see Windows spellings.
WINE_STUB = r"""#!/usr/bin/env bash
shift
args=()
for arg in "$@"; do
    case $arg in
        Z:*.py) arg=${arg#Z:}; arg=${arg//\\//} ;;
    esac
    args+=("$arg")
done
exec "$X2W_CHECK_PYTHON" "${args[@]}"
"""
ENV_PREFIXES = ("X2W_", "WINE", "VKD3D", "XDG_RUNTIME_DIR", "PULSE_SERVER")
# A zombie counts as gone: it no longer runs, it only waits for its parent to collect it.
ALIVE = """\
def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True
"""
# Stands in for tools/startup/debug_launch.py: records its arguments and environment, and under
# --translate whether the watcher and its child are running. With hold.flag "int" it then waits for
# a Ctrl-C and exits 3 as a debugger does that ends its run on one; with "term" it finishes 2 s later
# and leaves done.flag. Then it prints seven lines and exits with the code under test.
STUB = """\
import json
import os
import sys
import time
from pathlib import Path

{alive}
here = Path(__file__)
here.with_name("argv.json").write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
here.with_name("env.json").write_text(json.dumps({{k: v for k, v in os.environ.items()
                                                   if k.startswith({prefixes!r})}}), encoding="utf-8")
if here.with_name("translate.flag").exists():
    deadline = time.time() + 30
    while not here.with_name("watch.json").exists() and time.time() < deadline:
        time.sleep(0.05)
    if here.with_name("watch.json").exists():
        time.sleep(0.3)  # a watcher stopped right after its start is gone by now
        pids = json.loads(here.with_name("watch.json").read_text(encoding="utf-8"))[:2]
        here.with_name("alive.json").write_text(json.dumps([alive(pid) for pid in pids]), encoding="utf-8")
hold = here.with_name("hold.flag").read_text() if here.with_name("hold.flag").exists() else ""
code = {code}
if hold:
    here.with_name("ready.flag").write_bytes(b"")
    try:
        time.sleep(60 if hold == "int" else 2)
    except KeyboardInterrupt:
        code = 3
    here.with_name("done.flag").write_bytes(b"")
for n in range(1, 8):
    print("stub debugger line", n)
sys.exit(code)
"""
# Stands in for tools/startup/gfx2hlsl.py under --translate: starts a child process, as the parallel
# workers are, records its own and the child's process ids, its arguments and environment, then
# waits to be stopped. Both ignore SIGTERM, so only run.sh's SIGKILL fallback stops them.
WATCH_STUB = """\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)  # inherited by the child as ignored
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
here = Path(__file__)
here.with_name("watch_env.json").write_text(json.dumps({{k: v for k, v in os.environ.items()
                                                         if k.startswith({prefixes!r})}}), encoding="utf-8")
here.with_name("watch.json").write_text(json.dumps([os.getpid(), child.pid] + sys.argv[1:]), encoding="utf-8")
time.sleep(120)
"""
_namespace = {"os": os, "Path": Path}
exec(ALIVE, _namespace)
alive = _namespace["alive"]


def win(path):
    return "Z:" + str(path).replace("/", "\\")


def gone_within(pid, seconds=3.0):
    deadline = time.time() + seconds  # the kills are asynchronous
    while alive(pid):
        if time.time() > deadline:
            return False
        time.sleep(0.05)
    return True


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_case(script, code, work, switches=(), passed=(), env_extra=None, drive_data=False, config=CONFIG,
             translate=False, stop=None, drive_log=False, log_slash=False, expect_env=None, expect_warnings=None,
             expect_exit=None):
    """Run one copy of the script on a stub that exits with `code`; return what is wrong."""
    root = Path(tempfile.mkdtemp(dir=work)).resolve()
    startup = root / "tools" / "startup"
    startup.mkdir(parents=True)
    watch_spawn = startup / "watch-spawn.flag"
    if not translate and "--translate" in switches:
        # The fast debugger can exit before a spawned watcher reaches Python.
        # Record the launch in the parent shell of this test copy.
        script = replace(script, "    watcher=$!\n",
                         '    watcher=$!\n    : > "$here/watch-spawn.flag"\n')
    (startup / "run.sh").write_text(script, encoding="utf-8")
    (startup / "debug_launch.py").write_text(
        STUB.format(code=code, alive=ALIVE, prefixes=ENV_PREFIXES), encoding="utf-8")
    (startup / "gfx2hlsl.py").write_text(WATCH_STUB.format(prefixes=ENV_PREFIXES), encoding="utf-8")
    wine = root / "wine"
    wine.write_text(WINE_STUB, encoding="utf-8")
    wine.chmod(0o755)
    (root / "bld").mkdir()
    for dll in DLLS:
        (root / "bld" / f"{dll}.dll").write_bytes(b"")
    exe, data, log, prefix = root / "run" / "t.exe", root / "data", root / "log", root / "prefix"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    data.mkdir()
    if config is not None:
        (data / "MicrosoftGame.config").write_bytes(config.encode("utf-8"))
    (prefix / "dosdevices").mkdir(parents=True)
    (prefix / "dosdevices" / "g:").symlink_to(data)
    env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIXES + ("PROTON",))}
    env.update(WINE=str(wine), WINEPREFIX=str(prefix), X2W_WIN_PYTHON="python.exe",
               X2W_CHECK_PYTHON=sys.executable, VKD3D_SHADER_CACHE_PATH=str(root / "vkd3d"),
               XDG_RUNTIME_DIR=str(root / "no-runtime-dir"))  # no pulse socket there: no PULSE_SERVER
    if translate:
        env["X2W_FXC"] = str(root / "sdk" / "fxc.exe")
    env.update(env_extra or {})
    # --log as a drive path is the folder "logs" of the G: drive, that is of the data folder.
    # A trailing separator is dropped (a host path loses it in realpath, a drive path in run.sh).
    log_arg, log_win = (("G:\\logs" + ("\\" if log_slash else ""), "G:\\logs") if drive_log else
                        (str(log) + ("/" if log_slash else ""), win(log)))
    if drive_log:
        log = data / "logs"
    command = ["bash", str(startup / "run.sh"), "--exe", str(exe), "--data", "G:\\" if drive_data else str(data),
               "--log", log_arg, "--build-dir", "bld", "--seconds", "9", "--memory-gib", "3",
               "--continue-asserts", "4", *switches]
    if translate:
        command += ["--render", "--translate", "--shader-cache", "cache"]
        (startup / "translate.flag").write_bytes(b"")
    if stop:
        (startup / "hold.flag").write_text(stop)
    label = f"stub exit {code} {' '.join(switches)}{' --translate' if translate else ''}" \
            f"{f' ({stop})' if stop else ''}{' --log G:' if drive_log else ''}".strip()
    # A surviving watcher can keep a capture pipe open after bash exits: capture to files.
    with (tempfile.TemporaryFile(mode="w+", errors="replace", dir=root) as stdout,
          tempfile.TemporaryFile(mode="w+", errors="replace", dir=root) as stderr):
        # Its own session, so the Ctrl-C below reaches run.sh's group as a terminal's would.
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            if stop:
                deadline = time.time() + 30
                while not (startup / "ready.flag").exists() and time.time() < deadline:
                    time.sleep(0.05)
                if stop == "int":  # a terminal's Ctrl-C reaches the whole foreground group
                    os.killpg(process.pid, signal.SIGINT)
                else:  # TERM to run.sh alone
                    os.kill(process.pid, signal.SIGTERM)
            returncode = process.wait(timeout=120)
            debugger_done = (startup / "done.flag").exists()
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return [f"{label}: run.sh did not finish in 120 s"]
        stdout.seek(0)
        stderr.seek(0)
        printed = [line.strip() for line in stdout if line.strip()]
        warnings = [line.strip() for line in stderr if line.strip()]
    found = []
    # After INT or TERM, run.sh ends by the same signal.
    want_exit = {"int": -signal.SIGINT, "term": -signal.SIGTERM}.get(stop, code if expect_exit is None else expect_exit)
    if returncode != want_exit:
        found.append(f"{label}: expected exit {want_exit}, run.sh exited {returncode} ({warnings[-3:]})")
    if expect_exit is not None:  # a refused run must not reach the debugger
        if (startup / "argv.json").exists():
            found.append(f"{label}: the debugger ran although run.sh refused the run")
        if not any(expect_warnings[0] in w for w in warnings):
            found.append(f"{label}: missing message {expect_warnings[0]!r}, got {warnings}")
        return found
    if expect_warnings is None:
        fake = "--fake-user" in switches or (env_extra or {}).get("X2W_FAKE_USER") == "1"
        expect_warnings = [] if fake else [FAKE_USER_WARNING]
        if "--hide-title-window" in switches and "--render" not in switches and not translate:
            expect_warnings.append(HIDE_WARNING)
    if warnings != expect_warnings:
        found.append(f"{label}: warnings {warnings}, expected {expect_warnings}")
    if stop == "term" and not debugger_done:
        found.append(f"{label}: run.sh ended before the debugger finished")
    if drive_log:
        if "G:\\cache" in switches and not (data / "cache").is_dir():
            found.append(f"{label}: the drive-path shader cache was not created in the G: drive")
        stray = [p.name for p in root.iterdir() if p.name.startswith("G:")]
        if stray:
            found.append(f"{label}: drive-path log written to the working directory as {stray}")
    if not stop:
        if printed != STUB_LINES[:HEAD_LINES]:
            found.append(f"{label}: printed {printed}, expected the first {HEAD_LINES} debugger lines")
        summary = Path(str(log) + ".out")
        lines = [l.strip() for l in summary.read_text().splitlines() if l.strip()] if summary.is_file() else None
        if lines != STUB_LINES:
            found.append(f"{label}: {summary.name} does not hold the debugger's output")
    missing = [dll for dll in DLLS if not (exe.parent / f"{dll}.dll").is_file()]
    if missing:
        found.append(f"{label}: stand-in DLLs not copied next to the executable: {missing}")
    want = [win(exe), "G:\\" if drive_data else win(data), log_win, "--timeout", "9", "--memory-gib", "3",
            "--trace-files", "1", "--continue-asserts", "4", *passed]
    got = read_json(startup / "argv.json")
    if got != want:
        found.append(f"{label}: the debugger received {got}, expected {want}")
    seen = read_json(startup / "env.json") or {}
    expected = {name: "1" for name in PERMISSIVE}
    expected.update(X2W_XMEM_SHARED_VIEWS="1", X2W_TITLE_ID="X2WTEST", X2W_STORAGE_MB="123",
                    WINEDLLOVERRIDES=GAME_OVERRIDES, WINEFSYNC="1", WINEDEBUG="-all", WINEPREFIX=str(prefix))
    if translate:
        expected.update(X2W_D3D12_RENDER="1", X2W_DRAW_NULL_IDENTITY="1", X2W_SHADER_CACHE=win(root / "cache"),
                        X2W_FXC=win(root / "sdk" / "fxc.exe"))
    # A callable expectation is computed from this case's scratch root.
    expected.update({k: v(root) if callable(v) else v for k, v in (expect_env or {}).items()})
    for name in CHECKED:
        if seen.get(name) != expected.get(name):
            found.append(f"{label}: {name}={seen.get(name)!r} in the debugger's environment, "
                         f"expected {expected.get(name)!r}")
    watch = startup / "watch.json"
    if translate:
        started = read_json(watch)
        if not started or started[2:] != ["--watch", win(root / "cache")]:
            found.append(f"--translate: the translator received {started}, expected --watch {win(root / 'cache')}")
            return found
        watch_env = read_json(startup / "watch_env.json") or {}
        for name, value in (("WINEDLLOVERRIDES", GAME_OVERRIDES + ";d3dcompiler_47=n"),
                            ("X2W_FXC", win(root / "sdk" / "fxc.exe"))):
            if watch_env.get(name) != value:
                found.append(f"--translate: the translator's {name} is {watch_env.get(name)!r}, expected {value!r}")
        # Positive control: both ran while the debugger did, so "gone" below means stopped.
        if read_json(startup / "alive.json") != [True, True]:
            found.append(f"--translate: the translator and its child were not both running during the "
                         f"debugger call ({read_json(startup / 'alive.json')})")
        for pid, what in ((started[0], "translator"), (started[1], "translator's child process")):
            if not gone_within(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                found.append(f"--translate: the {what} was still running after run.sh exited")
    else:
        if watch_spawn.is_file() or watch.is_file():
            found.append(f"{label}: the translator started without --render --translate")
    return found


CASES = [
    # 2 is the debugger's "CreateProcess failed"; 7 rules out a fixed code.
    dict(code=2), dict(code=0), dict(code=7),
    dict(code=0, switches=["--fake-user"], expect_env={"X2W_FAKE_USER": "1"}),
    dict(code=0, env_extra={"X2W_FAKE_USER": "1"}, expect_env={"X2W_FAKE_USER": "1"}),
    dict(code=0, switches=["--trace-calls", "--console", "0"], passed=["--trace-calls", "1", "--console", "0"]),
    # --data as a drive path stays as it is; its config is read through the prefix's dosdevices.
    # --storage is relative to the working directory, the scratch root, and gains a backslash.
    dict(code=7, drive_data=True,
         switches=["--fake-user", "--gpu-mirror", "--host-cpu", "--startup-below-40", "--hide-title-window",
                   "--input-script", "4@70", "--storage", "store"],
         expect_env={"X2W_FAKE_USER": "1", "X2W_XMEM_GPU_MIRROR": "1", "X2W_HOST_CPU": "1",
                     "X2W_XMEM_STARTUP_BELOW_40": "1", "X2W_HIDE_TITLE_WINDOW": "1", "X2W_INPUT_SCRIPT": "4@70",
                     "X2W_STORAGE_PATH": lambda root: win(root / "store") + "\\"}),
    # Without a config, a caller's title id stays and no storage size is set.
    dict(code=0, config=None, env_extra={"X2W_TITLE_ID": "CALLER"},
         expect_env={"X2W_TITLE_ID": "CALLER", "X2W_STORAGE_MB": None}),
    # A caller's Wine settings stay.
    dict(code=0, env_extra={"WINEDLLOVERRIDES": "x=b", "WINEFSYNC": "0", "WINEDEBUG": "+seh"},
         expect_env={"WINEDLLOVERRIDES": "x=b", "WINEFSYNC": "0", "WINEDEBUG": "+seh"}),
    # A caller's X2W_SHADER_CACHE lets a watcher started without --render get past `set -u`.
    dict(code=0, switches=["--translate"], env_extra={"X2W_SHADER_CACHE": "C:\\c"},
         expect_env={"X2W_SHADER_CACHE": "C:\\c"}, expect_warnings=[FAKE_USER_WARNING, TRANSLATE_WARNING]),
    dict(code=0, switches=["--hide-title-window"], expect_env={"X2W_HIDE_TITLE_WINDOW": "1"},
         expect_warnings=[FAKE_USER_WARNING, HIDE_WARNING]),
    # Drive paths for --log and --shader-cache are created through the prefix; the debugger gets them as given.
    dict(code=7, drive_log=True, switches=["--render", "--shader-cache", "G:\\cache"],
         expect_env={"X2W_D3D12_RENDER": "1", "X2W_DRAW_NULL_IDENTITY": "1", "X2W_SHADER_CACHE": "G:\\cache"}),
    # A trailing separator on --log still puts the summary beside the folder.
    dict(code=7, log_slash=True),
    dict(code=7, drive_log=True, log_slash=True),
    dict(code=0, config=CONFIG_ONE_LINE),
    dict(code=0, switches=["--render", "--shader-cache", "cache"],
         expect_env={"X2W_D3D12_RENDER": "1", "X2W_DRAW_NULL_IDENTITY": "1",
                     "X2W_SHADER_CACHE": lambda root: win(root / "cache")}),
    dict(code=0, env_extra={"WINEPREFIX": ""}, expect_exit=64, expect_warnings=["run.sh: set WINEPREFIX"]),
    dict(code=0, env_extra={"X2W_WIN_PYTHON": ""}, expect_exit=64, expect_warnings=["run.sh: set X2W_WIN_PYTHON"]),
    dict(code=7, translate=True),
    dict(code=7, translate=True, stop="int"),
    dict(code=7, translate=True, stop="term"),
]


def run_problems(script, work, first_only):
    found = []
    for case in CASES:
        found += run_case(script, work=work, **case)
        if found and first_only:
            break
    return found


def replace(script, old, new, count=1):
    if script.count(old) != count:
        raise SystemExit(f"mutation anchor {old!r} found {script.count(old)} times, expected {count}")
    return script.replace(old, new)


def drop_line(script, text):
    lines = script.splitlines(keepends=True)
    kept = [line for line in lines if text not in line]
    if len(lines) - len(kept) != 1:
        raise SystemExit(f"mutation anchor {text!r} is on {len(lines) - len(kept)} lines, expected 1")
    return "".join(kept)


def mutants_of(script):
    return {
        "exit with head's status": replace(script, "exit $debugger_exit", "exit"),
        "fixed exit 0": replace(script, "exit $debugger_exit", "exit 0"),
        "no FakeUser warning": replace(script, '    echo "WARNING: Without --fake-user', '    : "WARNING: Without --fake-user'),
        "short DLL list": replace(script, " xg_xs GameInput; do", " xg_xs; do"),
        "no shared views": drop_line(script, "export X2W_XMEM_SHARED_VIEWS="),
        "watcher without --render": replace(script, 'if [ -n "$translate" ] && [ -n "$render" ]; then',
                                            'if [ -n "$translate" ]; then'),
        "config not read": replace(script, 'if [ -f "$config" ]; then', "if false; then"),
        "last tag wins": replace(script, "| head -n 1 | cut", "| tail -n 1 | cut", count=2),
        "greedy one-line match": replace(script, "title_id=$(grep -oi '<TitleId>[^<]*' \"$config\" | head -n 1 | cut -d '>' -f 2- | tr -d '\\r')",
                                         "title_id=$(sed -n 's/.*<TitleId>\\([^<]*\\).*/\\1/p' \"$config\" | head -n 1)"),
        "case-sensitive tags": replace(script, "grep -oi '<", "grep -o '<", count=2),
        "no hide warning": replace(script, '    echo "WARNING: --hide-title-window', '    : "WARNING: --hide-title-window'),
        "log through the raw path": replace(script, 'log_host=$(host_path "$log")', 'log_host=$log'),
        "cache through the raw path": replace(script, 'mkdir -p "$(host_path "$shader_cache")"', 'mkdir -p "$shader_cache"'),
        "trailing separator kept": drop_line(script, "do log=${log%?}; done"),
        "no KILL fallback": drop_line(script, 'kill -KILL -- "-$watcher"'),
        "INT ends with exit 130": replace(script, '    kill -s "$1" $$', '    exit 130'),
        "XDG_RUNTIME_DIR kept": drop_line(script, "unset XDG_RUNTIME_DIR"),
        "--console 0 not passed": drop_line(script, 'console_args=(--console "$console")'),
        "drive path through Z:": replace(script, "    if [[ $1 =~ ^[A-Za-z]: ]]; then", "    if false; then"),
        "no storage backslash": drop_line(script, "X2W_STORAGE_PATH+="),
        "no default overrides": drop_line(script, "export WINEDLLOVERRIDES="),
        "compiler override for the game": replace(script, "d3d11=n}", "d3d11=n;d3dcompiler_47=n}"),
        "no compiler override for the watcher": replace(script, 'WINEDLLOVERRIDES="$WINEDLLOVERRIDES;d3dcompiler_47=n"',
                                                        'WINEDLLOVERRIDES="$WINEDLLOVERRIDES"'),
        "FXC host path": drop_line(script, 'X2W_FXC=$(win_path "$X2W_FXC")'),
        # Started, then stopped before the debugger: only the positive control sees it.
        "watcher stopped before the debugger": replace(script, "    watcher=$!\n",
                                                       "    watcher=$!\n    sleep 1; stop_watcher\n"),
        "no INT trap": drop_line(script, "trap 'on_signal INT' INT"),
        "no EXIT trap": drop_line(script, "trap stop_watcher EXIT"),
        "parent-only kill": replace(script, '-- "-$watcher"', '-- "$watcher"', count=3),
        "no setsid": replace(script, '        setsid "$wine"', '        "$wine"'),
    }


def main():
    verbose = sys.argv[1:] == ["-v"]  # also print the first problem each mutant produced
    if sys.argv[1:] and not verbose:
        sys.exit("usage: startup_run_sh_check.py [-v]")
    if os.name == "nt":
        print("startup run.sh: skipped (a POSIX launcher; run it on Linux)")
        return 77
    missing = [tool for tool in ("bash", "setsid") if not shutil.which(tool)]
    if missing:
        print(f"FAIL missing {', '.join(missing)}")
        print("startup run.sh: 1 failed")
        return 1
    script = RUN_SH.read_text(encoding="utf-8")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        failures += run_problems(script, tmp, False)
        if not failures:
            for label, mutant in mutants_of(script).items():
                problems = run_problems(mutant, tmp, True)
                if not problems:
                    failures.append(f"mutant '{label}' was not detected")
                elif verbose:
                    print(f"mutant '{label}': {problems[0]}")
    for item in failures:
        print("FAIL", item)
    print(f"startup run.sh: {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
