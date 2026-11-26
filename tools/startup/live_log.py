"""Starts the live debug-log window of a run (live_view.py) without covering the game.

X2W_LIVE_LOG: unset or "minimized" = a console window that starts minimized and does not take the focus (default),
"normal" = an ordinary new console in front, "off" = no window. Any other value is the default.
Windows Terminal as the default console host ignores the start-up flags, so the window is also found by its title
(live_view.py sets "Xbox2Windows debug log - <run>") and minimized without activating it.
"""
import os
import sys
import subprocess
import threading
import time

CREATE_NEW_CONSOLE = 0x00000010
STARTF_USESHOWWINDOW = 0x00000001
SW_SHOWMINNOACTIVE = 7
MODES = ("minimized", "normal", "off")
TITLE_PREFIX = "Xbox2Windows debug log - "
TITLE_WAIT_SECONDS = 10.0


def mode(environ=None):
    value = (os.environ if environ is None else environ).get("X2W_LIVE_LOG", "").strip().lower()
    return value if value in MODES else "minimized"


def window_title(logdir):
    return TITLE_PREFIX + os.path.basename(os.path.abspath(logdir))


def _startupinfo():
    info = subprocess.STARTUPINFO() if hasattr(subprocess, "STARTUPINFO") else type("StartupInfo", (), {})()
    info.dwFlags = getattr(info, "dwFlags", 0) | STARTF_USESHOWWINDOW
    info.wShowWindow = SW_SHOWMINNOACTIVE
    return info


def popen_options(chosen):
    """The creationflags and startupinfo for Popen in the chosen mode ({} for "off")."""
    if chosen == "off":
        return {}
    options = {"creationflags": CREATE_NEW_CONSOLE}
    if chosen == "minimized":
        options["startupinfo"] = _startupinfo()
    return options


def minimize_by_title(title, find_window, show_window, wait=TITLE_WAIT_SECONDS, sleep=time.sleep, clock=time.monotonic):
    """Finds the window by its title and minimizes it without activating it. True when it was found."""
    deadline = clock() + wait
    while True:
        hwnd = find_window(title)
        if hwnd:
            show_window(hwnd, SW_SHOWMINNOACTIVE)
            return True
        if clock() >= deadline:
            return False
        sleep(0.2)


def _windows_calls():
    import ctypes
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return (lambda title: user32.FindWindowW(None, title)), (lambda hwnd, how: user32.ShowWindow(hwnd, how))


def start(script_dir, logdir, cwd, environ=None, popen=subprocess.Popen, windows_calls=None, background=True):
    """Starts live_view.py for this run. Returns the process, or None when the window is off or could not start."""
    chosen = mode(environ)
    if chosen == "off":
        return None
    command = [sys.executable, "-B", os.path.join(script_dir, "live_view.py"), logdir, cwd]
    try:
        process = popen(command, **popen_options(chosen))
    except OSError as error:
        print(f"live console not started: {error}")
        return None
    if chosen == "minimized":
        def minimize():
            try:
                find_window, show_window = (windows_calls or _windows_calls)()
                minimize_by_title(window_title(logdir), find_window, show_window)
            except (OSError, AttributeError):
                pass  # not Windows, or no user32: the start-up flags are all there is
        if background:
            threading.Thread(target=minimize, daemon=True).start()
        else:
            minimize()
    return process
