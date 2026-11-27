#!/usr/bin/env python3
"""The live debug-log window starts minimized and without the focus (tools/startup/live_log.py).

Everywhere: X2W_LIVE_LOG parsing, what Popen gets in each mode, and the title fallback for hosts that ignore the
start-up flags. On Windows with an interactive desktop: the real window is started while another window has the
foreground, the foreground does not change, and the log window ends up minimized. Without such a desktop that part
is skipped with exit code 77 (the first part has already passed by then).

usage: startup_live_log_check.py [path/to/tools/startup]
"""
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tools" / "startup"
sys.path.insert(0, str(TOOLS))
import live_log as L  # noqa: E402


def check_modes():
    cases = {None: "minimized", "": "minimized", "minimized": "minimized", "MINIMIZED": "minimized",
             "normal": "normal", " Normal ": "normal", "off": "off", "OFF": "off", "garbage": "minimized", "0": "minimized"}
    for value, want in cases.items():
        env = {} if value is None else {"X2W_LIVE_LOG": value}
        assert L.mode(env) == want, (value, L.mode(env))


def check_popen_options():
    assert L.popen_options("off") == {}
    normal = L.popen_options("normal")
    assert normal == {"creationflags": L.CREATE_NEW_CONSOLE}, normal
    minimized = L.popen_options("minimized")
    assert minimized["creationflags"] == L.CREATE_NEW_CONSOLE
    info = minimized["startupinfo"]
    assert info.dwFlags & L.STARTF_USESHOWWINDOW and info.wShowWindow == L.SW_SHOWMINNOACTIVE == 7, vars(info)


def check_start():
    calls = []

    def popen(command, **options):
        calls.append((command, options))
        return "process"

    for value, want_calls in (("off", 0), ("normal", 1), ("minimized", 1), (None, 1)):
        calls.clear()
        env = {} if value is None else {"X2W_LIVE_LOG": value}
        found = []
        result = L.start("tools", "logs/run1", "work", env, popen,
                         lambda: (lambda title: found.append(title) or 0, lambda hwnd, how: None), background=False)
        assert len(calls) == want_calls, (value, calls)
        assert result == (None if value == "off" else "process"), (value, result)
        if want_calls:
            command, options = calls[0]
            assert command[-2:] == ["logs/run1", "work"] and command[-3].endswith("live_view.py"), command
            assert ("startupinfo" in options) == (value != "normal"), (value, options)
        assert bool(found) == (value in ("minimized", None)), (value, found)

    def refuse(command, **options):
        raise OSError("no console")
    assert L.start("tools", "logs/run1", "work", {}, refuse, background=False) is None


def check_title_fallback():
    shown = []
    ticks = iter(range(100))
    seen = []

    def find(title):
        seen.append(title)
        return 0x1234 if len(seen) == 3 else 0

    found = L.minimize_by_title("Xbox2Windows debug log - run1", find, lambda hwnd, how: shown.append((hwnd, how)),
                                wait=5, sleep=lambda s: None, clock=lambda: next(ticks) * 0.1)
    assert found and shown == [(0x1234, 7)] and len(seen) == 3, (found, shown, seen)
    shown.clear()
    assert not L.minimize_by_title("t", lambda title: 0, lambda hwnd, how: shown.append(hwnd), wait=1,
                                   sleep=lambda s: None, clock=lambda: next(ticks) * 0.5) and not shown
    assert L.window_title("a/b/run7") == "Xbox2Windows debug log - run7"


def windows_foreground():
    """0 = passed, 77 = this session cannot show a foreground window."""
    import ctypes
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.IsIconic.argtypes = [ctypes.c_void_p]
    user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    user32.CreateWindowExW.restype = ctypes.c_void_p
    user32.CreateWindowExW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
    user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.DestroyWindow.argtypes = [ctypes.c_void_p]
    user32.OpenInputDesktop.restype = ctypes.c_void_p
    user32.OpenInputDesktop.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    if not user32.OpenInputDesktop(0, 0, 0x100):  # DESKTOP_SWITCHDESKTOP
        print("startup_live_log skipped (window part): no interactive desktop in this session")
        return 77
    game = user32.CreateWindowExW(0, "STATIC", "x2w live log test game", 0x90000000, 100, 100, 200, 100, None, None, None, None)
    user32.ShowWindow(game, 5)
    user32.SetForegroundWindow(game)
    if user32.GetForegroundWindow() != game:
        user32.DestroyWindow(game)
        print("startup_live_log skipped (window part): this session cannot give a window the foreground")
        return 77
    with tempfile.TemporaryDirectory() as temp:
        run = Path(temp) / "live_log_check_run"
        run.mkdir()
        process = L.start(str(TOOLS), str(run), temp, {})
        title = L.window_title(str(run))
        try:
            deadline = time.monotonic() + 20
            minimized = False
            while time.monotonic() < deadline and not minimized:
                hwnd = user32.FindWindowW(None, title)
                minimized = bool(hwnd) and bool(user32.IsIconic(hwnd))
                assert user32.GetForegroundWindow() == game, "the live log window took the foreground"
                time.sleep(0.1)
            assert minimized, "the live log window %r was not found minimized" % title
            assert user32.GetForegroundWindow() == game, "the live log window took the foreground"
        finally:
            if process:
                process.kill()
            user32.DestroyWindow(game)
    return 0


def main():
    check_modes()
    check_popen_options()
    check_start()
    check_title_fallback()
    print("live log window: modes, Popen options and title fallback match")
    if os.name != "nt":
        return 0
    return windows_foreground()


if __name__ == "__main__":
    sys.exit(main())
