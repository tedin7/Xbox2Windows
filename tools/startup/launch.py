"""Run one EXE in a kill-on-close job with a timeout, a commit cap and captured stdout/stderr.

usage: launch.py <exe> <cwd> <logdir> [--timeout S] [--memory-gib N] [--max-processes N] [-- args...]
Prints the exit code (hex), how it ended and the trap log.
"""
import ctypes
import ctypes.wintypes as wt
import math
import os
import sys
import time

k32 = None
if hasattr(ctypes, "WinDLL"):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE),
                ("hStdError", wt.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE), ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD)]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]


# X2W_LOW_ADDRESSES=1: the title starts with bottom-up ASLR always off, so its VirtualAlloc blocks come from the low end of the
# address space instead of a random point above 2^40. The package acceleration-structure builder refuses a destination address
# with bits set above bit 39 (#1758). Source: Microsoft documentation of UpdateProcThreadAttribute,
# PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY and PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_OFF.
PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY = 0x00020007
PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_OFF = 0x2 << 16
EXTENDED_STARTUPINFO_PRESENT = 0x00080000


def low_addresses_requested(env=None):
    return (os.environ if env is None else env).get("X2W_LOW_ADDRESSES") == "1"


def startup_for_create(si, env=None):
    """(structure to pass, extra creation flags, objects to keep alive, note) for CreateProcessW.

    Without X2W_LOW_ADDRESSES=1, or when the attribute list cannot be built, `si` is passed unchanged."""
    if not low_addresses_requested(env) or k32 is None:
        return si, 0, None, None
    size = ctypes.c_size_t()
    k32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not k32.InitializeProcThreadAttributeList(buffer, 1, 0, ctypes.byref(size)):
        return si, 0, None, f"X2W_LOW_ADDRESSES: attribute list unavailable (error {ctypes.get_last_error()})"
    policy = ctypes.c_uint64(PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_OFF)
    if not k32.UpdateProcThreadAttribute(buffer, 0, PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY, ctypes.byref(policy),
                                         ctypes.sizeof(policy), None, None):
        return si, 0, None, f"X2W_LOW_ADDRESSES: mitigation policy refused (error {ctypes.get_last_error()})"
    extended = STARTUPINFOEXW()
    extended.StartupInfo = si
    extended.StartupInfo.cb = ctypes.sizeof(extended)
    extended.lpAttributeList = ctypes.cast(buffer, ctypes.c_void_p)
    return extended, EXTENDED_STARTUPINFO_PRESENT, (buffer, policy), "X2W_LOW_ADDRESSES: bottom-up ASLR off for the title"


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("Basic", BASIC_LIMIT), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", wt.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p), ("bInheritHandle", wt.BOOL)]


if k32 is not None:
    k32.CreateFileW.restype = wt.HANDLE
    k32.CreateJobObjectW.restype = wt.HANDLE


def prepare_storage_path(env, logdir, default_storage="S:\\", exists_fn=os.path.exists, makedirs_fn=os.makedirs):
    """Ensure X2W_STORAGE_PATH has a valid directory when unset and the default drive is absent."""
    current = env.get("X2W_STORAGE_PATH", "").strip()
    if current:
        return None
    if exists_fn(default_storage):
        return None
    fallback = os.path.join(logdir, "storage")
    makedirs_fn(fallback, exist_ok=True)
    env["X2W_STORAGE_PATH"] = fallback
    return f"X2W_STORAGE_PATH unset and {default_storage} missing; created fallback storage: {fallback}"


def check_shader_cache_warning(env, cwd=None):
    """Warn if X2W_SHADER_CACHE points to an existing non-empty directory.

    A relative value is resolved against the child's working directory, as the child sees it.
    Inspection failures are advisory only and never stop the launch.
    """
    shader_cache = env.get("X2W_SHADER_CACHE", "").strip()
    if not shader_cache:
        return None
    inspected = os.path.join(cwd, shader_cache) if cwd and not os.path.isabs(shader_cache) else shader_cache
    try:
        if not os.path.isdir(inspected):
            return None
        with os.scandir(inspected) as entries:
            if any(entries):
                return (f"warning: X2W_SHADER_CACHE ({shader_cache}) is non-empty; "
                        "cached translated shaders from an older build may be stale")
    except OSError:
        return None
    return None


def format_fast_fail_diagnostic(exit_code, elapsed, trap_log, env):
    """Return a concise diagnostic line when the child exits non-zero in under 1.0 second."""
    if elapsed >= 1.0 or exit_code == 0:
        return None
    storage = env.get("X2W_STORAGE_PATH", "").strip() or "S:\\"
    shader_cache = env.get("X2W_SHADER_CACHE", "").strip() or "(unset)"
    return (f"fast-fail: exited with {exit_code:#010x} in {elapsed:.2f} s (< 1.0 s); "
            f"check trap log={trap_log}, X2W_STORAGE_PATH={storage}, X2W_SHADER_CACHE={shader_cache}")


def quote_argument(argument):
    """Quote one argument for CreateProcessW the way the C runtime and CommandLineToArgvW read it.

    Backslashes only matter before a double quote: a run of them that ends the argument, or sits
    before an embedded quote, is doubled (and the quote itself escaped), so a path such as
    D:\\saves\\ does not swallow the closing quote.
    """
    pieces = ['"']
    backslashes = 0
    for char in argument:
        if char == "\\":
            backslashes += 1
            continue
        if char == '"':
            pieces.append("\\" * (2 * backslashes + 1) + '"')
        else:
            pieces.append("\\" * backslashes + char)
        backslashes = 0
    pieces.append("\\" * (2 * backslashes) + '"')
    return "".join(pieces)


def inheritable_file(path):
    sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, True)
    h = k32.CreateFileW(path, 0x40000000, 1 | 2, ctypes.byref(sa), 2, 0x80, None)  # GENERIC_WRITE, CREATE_ALWAYS
    if h in (None, wt.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), path)
    return h


VALID_OPTIONS = ("--timeout", "--memory-gib", "--max-processes")
DWORD_MAX = (1 << 32) - 1
GIB = 1 << 30
SIZE_T_LIMIT = 1 << (ctypes.sizeof(ctypes.c_size_t) * 8)


def _parse_finite_float(raw, flag, allow_zero):
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{flag} expects a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{flag} must be finite, got {raw!r}")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{flag} must be >= 0, got {raw!r}")
    elif value <= 0:
        raise ValueError(f"{flag} must be > 0, got {raw!r}")
    return value


def _parse_non_negative_int(raw, flag):
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"{flag} expects an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{flag} must be >= 0, got {raw!r}")
    return value


def parse_args(argv):
    """Parse and validate CLI arguments for launch.py."""
    args = list(argv)
    if "--" in args:
        split_at = args.index("--")
        head = args[:split_at]
        extra = args[split_at + 1:]
    else:
        head = args
        extra = []
    if len(head) < 3:
        raise ValueError("usage: launch.py <exe> <cwd> <logdir> [--timeout S] [--memory-gib N] [--max-processes N] [-- args...]")
    for label, value in zip(("exe", "cwd", "logdir"), head[:3]):
        if not value or not value.strip():
            raise ValueError(f"<{label}> must not be empty")
    exe, cwd, logdir = (os.path.abspath(a) for a in head[:3])

    opts = {}
    index = 3
    while index < len(head):
        flag = head[index]
        if flag not in VALID_OPTIONS:
            raise ValueError(f"unknown option {flag!r} (expected {', '.join(VALID_OPTIONS)})")
        if index + 1 >= len(head):
            raise ValueError(f"option {flag} requires a value")
        opts[flag] = head[index + 1]
        index += 2

    timeout = _parse_finite_float(opts.get("--timeout", "15"), "--timeout", allow_zero=False)
    memory_gib = _parse_finite_float(opts.get("--memory-gib", "16"), "--memory-gib", allow_zero=True)
    max_processes = _parse_non_negative_int(opts.get("--max-processes", "1"), "--max-processes")
    # INFINITE (DWORD_MAX) is reserved. A positive cap must not round to zero.
    if not 1 <= timeout * 1000 < DWORD_MAX:
        raise ValueError("--timeout must be at least 1 ms and below the Windows INFINITE sentinel")
    if memory_gib and not 1 <= memory_gib * GIB < SIZE_T_LIMIT:
        raise ValueError("--memory-gib must be 0 or represent at least one byte within the Windows size_t limit")
    if max_processes > DWORD_MAX:
        raise ValueError("--max-processes exceeds the Windows DWORD limit")
    return exe, cwd, logdir, timeout, memory_gib, max_processes, extra


def main(argv=None):
    try:
        exe, cwd, logdir, timeout, memory_gib, max_processes, extra = parse_args(
            sys.argv[1:] if argv is None else argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if k32 is None:
        print("error: kernel32 is required to launch processes", file=sys.stderr)
        return 2

    memory = int(memory_gib * GIB)
    job = None
    out = None
    err = None
    process_handle = None
    thread_handle = None
    try:
        os.makedirs(logdir, exist_ok=True)
        trap_log = os.path.join(logdir, "trap.log")
        if os.path.exists(trap_log):
            os.remove(trap_log)

        k32.SetErrorMode(0x0001 | 0x0002 | 0x8000)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX | NOOPENFILEERRORBOX

        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
        info = EXTENDED_LIMIT()
        # KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION | JOB_MEMORY (when > 0) | ACTIVE_PROCESS (when limited)
        info.Basic.LimitFlags = 0x2000 | 0x400 | (0x200 if memory > 0 else 0) | (0x8 if max_processes else 0)
        info.Basic.ActiveProcessLimit = max_processes
        info.JobMemoryLimit = memory
        if not k32.SetInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")

        out = inheritable_file(os.path.join(logdir, "stdout.txt"))
        err = inheritable_file(os.path.join(logdir, "stderr.txt"))
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(si)
        si.dwFlags = 0x100  # STARTF_USESTDHANDLES
        si.hStdOutput, si.hStdError = out, err
        env = dict(os.environ, X2W_TRAP_LOG=trap_log)
        storage_note = prepare_storage_path(env, logdir)
        if storage_note:
            print(storage_note)
        shader_warning = check_shader_cache_warning(env, cwd)
        if shader_warning:
            print(shader_warning)
        block = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
        cmdline = ctypes.create_unicode_buffer(" ".join(quote_argument(a) for a in [exe, *extra]))
        pi = PROCESS_INFORMATION()
        flags = 0x4 | 0x400  # CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT
        startup, extra_flags, startup_keep, startup_note = startup_for_create(si, env)
        if startup_note:
            print(startup_note)
        flags |= extra_flags
        if not k32.CreateProcessW(exe, cmdline, None, None, True, flags, ctypes.create_unicode_buffer(block),
                                  cwd, ctypes.byref(startup), ctypes.byref(pi)):
            print(f"CreateProcess failed: error {ctypes.get_last_error()}")
            return 2
        process_handle = pi.hProcess
        thread_handle = pi.hThread
        if not k32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(process_handle)):
            k32.TerminateProcess(wt.HANDLE(process_handle), 1)
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
        start = time.time()
        k32.ResumeThread(wt.HANDLE(thread_handle))
        result = k32.WaitForSingleObject(wt.HANDLE(process_handle), int(timeout * 1000))
        ended = "exited"
        if result == 0x102:  # WAIT_TIMEOUT
            k32.TerminateJobObject(wt.HANDLE(job), 0xDEAD0001)
            k32.WaitForSingleObject(wt.HANDLE(process_handle), 5000)
            ended = f"killed after {timeout:g} s timeout"
        code = wt.DWORD()
        k32.GetExitCodeProcess(wt.HANDLE(process_handle), ctypes.byref(code))
        k32.QueryInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info), None)
        elapsed = time.time() - start
        for handle in (out, err, thread_handle, process_handle):
            k32.CloseHandle(wt.HANDLE(handle))
        out = err = thread_handle = process_handle = None
        print(f"pid {pi.dwProcessId} {ended} in {elapsed:.2f} s, exit code {code.value:#010x}, "
              f"peak job memory {info.PeakJobMemoryUsed / 2**20:.1f} MiB")
        fast_fail = format_fast_fail_diagnostic(code.value, elapsed, trap_log, env)
        if fast_fail:
            print(fast_fail)
        for name in ("stdout.txt", "stderr.txt", "trap.log"):
            path = os.path.join(logdir, name)
            text = None
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            print(f"--- {name}: " + ("(absent)" if text is None else f"{len(text)} chars"))
            if text:
                print(text[-4000:])
        return 0
    except OSError as exc:
        print(f"error: launch failed: {exc}", file=sys.stderr)
        return 2
    finally:
        for handle in (out, err, thread_handle, process_handle, job):
            if handle is not None:
                k32.CloseHandle(wt.HANDLE(handle))


if __name__ == "__main__":
    sys.exit(main())
