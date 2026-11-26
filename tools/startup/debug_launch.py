"""Run one EXE under a minimal debugger inside a kill-on-close job (startup diagnostics; see docs/startup-runtime.md).

usage: debug_launch.py <exe> <cwd> <logdir> [--timeout S] [--memory-gib N] [--stop-first-chance 1] [--capture-every S]

Records module loads, every exception (code, address as module+offset, first/second chance) and, for
the stopping exception, the integer registers, the stack words and the code bytes around RIP.
Stops at a second-chance exception, an uncontinued main-image breakpoint, or the timeout.
A timeout of zero disables the timer. An assertion continuation limit of -1 removes that limit.
A file named `stop` created in <logdir> during the run ends it as the timeout does (checked twice a
second; a stop file left from an earlier run is removed at start).
"""
import ctypes
import ctypes.wintypes as wt
import os
import re
import struct
import sys
import threading
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenThread.restype = wt.HANDLE
k32.CreateJobObjectW.restype = wt.HANDLE
k32.CreateFileW.restype = wt.HANDLE
k32.SuspendThread.restype = ctypes.c_uint32

from launch import STARTUPINFOW, PROCESS_INFORMATION, EXTENDED_LIMIT, inheritable_file  # noqa: E402
import host_cpu  # noqa: E402
from window_capture import FILE_CAP, Series, png_bytes  # noqa: E402
from step_gate import StepGate  # noqa: E402


class MemoryBasicInformation(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_uint64), ("AllocationBase", ctypes.c_uint64),
                ("AllocationProtect", ctypes.c_uint32), ("PartitionId", ctypes.c_uint16),
                ("RegionSize", ctypes.c_uint64), ("State", ctypes.c_uint32),
                ("Protect", ctypes.c_uint32), ("Type", ctypes.c_uint32)]


k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
k32.VirtualQueryEx.restype = ctypes.c_size_t

EXCEPTION, CREATE_THREAD, CREATE_PROCESS, EXIT_THREAD, EXIT_PROCESS, LOAD_DLL, UNLOAD_DLL, OUTPUT_STRING, RIP = range(1, 10)
DBG_CONTINUE, DBG_EXCEPTION_NOT_HANDLED = 0x00010002, 0x80010001
NAMES = {0x80000003: "breakpoint", 0x80000004: "single step", 0xC0000005: "access violation",
         0xC000001D: "illegal instruction", 0xC0000409: "fast fail", 0xE06D7363: "C++ exception",
         0x406D1388: "thread name", 0xC0000374: "heap corruption", 0x4000001F: "wow breakpoint",
         0xC0000094: "integer divide by zero", 0xC00000FD: "stack overflow", 0x40010006: "debug print"}
# --trace-calls 1: the exports hooked in each module, matched by the module's file name at load.
RUNTIME_MODULE = "xgameruntime.dll"
RUNTIME_CALL_HOOKS = ("QueryApiImpl", "InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2", "XErrorReport")
MOVIE_MODULE = "bink2scarlett.dll"
MOVIE_CALL_HOOKS = ("BinkOpenWithOptions", "BinkClose", "BinkNextFrame", "BinkRegisterFrameBuffers", "BinkPause",
                    "BinkShouldSkip")
REGS = ["Rax", "Rcx", "Rdx", "Rbx", "Rsp", "Rbp", "Rsi", "Rdi", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "Rip"]
LOG_REGS = {reg.lower(): reg for reg in REGS[:-1]}
LOG_HEX = r"(?:0x)?[0-9a-f]+"
LOG_ITEM = re.compile(rf"\[([a-z0-9]+)([+-]{LOG_HEX})\]")


def read_qword(read, address):
    data = read(address, 8)
    return int.from_bytes(data, "little") if len(data) == 8 else None


def parse_log_at(value, log):
    points = {}
    for entry in value.split(",") if value else ():
        site, colon, items = entry.strip().lower().partition(":")
        address, at, limit = site.partition("@")
        try:
            if not re.fullmatch(LOG_HEX, address):
                raise ValueError("expected a hexadecimal RVA")
            rva = int(address, 16)
            if rva > 0xffffffff:
                raise ValueError("RVA exceeds 32 bits")
            max_hits = 16
            if at:
                if not re.fullmatch(r"[0-9]+", limit) or (max_hits := int(limit)) <= 0:
                    raise ValueError("expected a positive decimal hit limit")
            if rva in points:
                raise ValueError("duplicate RVA")
            fields = []
            for item in items.split(";") if colon else LOG_REGS:
                item = item.strip()
                if item in LOG_REGS:
                    field = (item, LOG_REGS[item], None)
                else:
                    memory = LOG_ITEM.fullmatch(item)
                    if not memory or memory[1] not in LOG_REGS:
                        raise ValueError(f"invalid item {item!r}")
                    field = (item, LOG_REGS[memory[1]], int(memory[2], 16))
                # Compare the register and offset, so [rcx+8] and [rcx+0x8] are the same item.
                if any(field[1:] == earlier[1:] for earlier in fields):
                    raise ValueError(f"duplicate item {item!r}")
                fields.append(field)
            points[rva] = (max_hits, fields)
        except ValueError as error:
            log(f"X2W_LOG_AT ignored {entry!r}: {error}")
    return points


def read_stack(read, rsp, words):
    """Return `words` qwords read from `rsp`, zero-padded. The full read fails when it crosses past
    the stack's last committed page; then read only to the end of RSP's page, which is committed."""
    size = 8 * words
    data = read(rsp, size) or read(rsp, min(size, 0x1000 - (rsp & 0xfff))) or b""
    return data.ljust(size, b"\0")


def final_path(handle):
    buf = ctypes.create_unicode_buffer(1024)
    n = k32.GetFinalPathNameByHandleW(wt.HANDLE(handle), buf, 1024, 0)
    return os.path.basename(buf.value) if n else "?"


def main():
    exe, cwd, logdir = (os.path.abspath(a) for a in sys.argv[1:4])
    opts = dict(zip(sys.argv[4::2], sys.argv[5::2]))
    # Everything printed below is also kept for finished.txt, which the live console shows at the end.
    import builtins
    summary = []

    def print(*args, **kwargs):
        summary.append(" ".join(str(a) for a in args))
        builtins.print(*args, **kwargs)
    try:
        timeout = float(opts.get("--timeout", 15))  # zero leaves the session running until exit or a debug stop
        memory_gib = float(opts.get("--memory-gib", 16))
        skip_asserts = int(opts.get("--continue-asserts", "0"))  # -1 continues all diagnostic assertions
    except ValueError:
        print("expected numeric --timeout and --memory-gib values and an integer --continue-asserts value")
        return 2
    stop_first = opts.get("--stop-first-chance") == "1"
    try:
        capture_every = float(opts.get("--capture-every", 0))  # seconds between window pictures; 0 = off
    except ValueError:
        capture_every = -1
    if not (capture_every == 0 or 1 <= capture_every < float("inf")):
        print("expected --capture-every of 0 (off) or a finite number of seconds that is at least 1")
        return 2
    if timeout < 0 or not timeout < float("inf") or skip_asserts < -1:
        print("expected a finite nonnegative timeout and an assertion limit of -1 or greater")
        return 2
    # The job always sets JOB_MEMORY (0x200), so a limit of 0 would deny the process any commit.
    if not 1 <= memory_gib * 2**30 < 2**(8 * ctypes.sizeof(ctypes.c_size_t)):
        print("expected a finite --memory-gib that is at least one byte and fits the Windows size_t limit")
        return 2
    memory = int(memory_gib * 2**30)
    asserts_seen = {}
    # X2W_NOP_SOFT_ASSERTS=1: after the first hit of a main-image int3 that is continued, replace the
    # trap byte with NOP. DBG_CONTINUE already bypasses the title's handler, so the visible behavior is
    # unchanged, but a long mapping that traps once per chunk then runs at native speed.
    nop_soft_asserts = os.environ.get("X2W_NOP_SOFT_ASSERTS") == "1"
    nop_sites = {}
    # X2W_HOST_CPU=1: rewrite main-image instructions the host processor lacks (host_cpu.py).
    host_cpu_on = os.environ.get("X2W_HOST_CPU") == "1"
    host_cpu_sites = {}
    host_cpu_rewritten = set()  # sites replaced in place; a thread may still fault on one (see the handler)
    host_cpu_resumed = set()  # (site, thread) pairs continued once for that reason
    flushopt_missing = [False]  # set when a CLFLUSHOPT faults: CLWB then becomes a NOP instead
    stub_area = [0, 0]  # next free address and end of the block the stubs go to
    os.makedirs(logdir, exist_ok=True)
    trap_log = os.path.join(logdir, "trap.log")
    stop_file = os.path.join(logdir, "stop")  # created from outside: end the run as at the timeout
    for stale in (trap_log, os.path.join(logdir, "finished.txt"), stop_file):  # a reused run directory starts clean
        if os.path.exists(stale):
            os.remove(stale)
    if os.path.exists(trap_log):
        os.remove(trap_log)
    events = open(os.path.join(logdir, "events.txt"), "w", encoding="utf-8")

    def log(text):
        events.write(text + "\n")
        events.flush()

    log_at = parse_log_at(os.environ.get("X2W_LOG_AT", ""), log)
    logged_hits = {}

    k32.SetErrorMode(0x8003)
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED_LIMIT()
    info.Basic.LimitFlags = 0x2000 | 0x200 | 0x8
    info.Basic.ActiveProcessLimit = 1
    info.JobMemoryLimit = memory
    k32.SetInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))

    out = inheritable_file(os.path.join(logdir, "stdout.txt"))
    err = inheritable_file(os.path.join(logdir, "stderr.txt"))
    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(si)
    si.dwFlags = 0x100
    si.hStdOutput, si.hStdError = out, err
    env = dict(os.environ, X2W_TRAP_LOG=trap_log)
    # A process started under a debugger gets the debug heap: no low-fragmentation heap and checks on
    # every allocation and free. That multiplies the cost of each free, and a startup run frees a lot.
    # _NO_DEBUG_HEAP=1 keeps the ordinary heap; a caller who wants the checks sets _NO_DEBUG_HEAP=0.
    env.setdefault("_NO_DEBUG_HEAP", "1")
    block = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
    pi = PROCESS_INFORMATION()
    flags = 0x2 | 0x4 | 0x400  # DEBUG_ONLY_THIS_PROCESS | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT
    if not k32.CreateProcessW(exe, ctypes.create_unicode_buffer(f'"{exe}"'), None, None, True, flags,
                              ctypes.create_unicode_buffer(block), cwd, ctypes.byref(si), ctypes.byref(pi)):
        print("CreateProcess failed", ctypes.get_last_error())
        return 2
    k32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(pi.hProcess))
    k32.DebugSetProcessKillOnExit(True)
    k32.ResumeThread(wt.HANDLE(pi.hThread))
    if opts.get("--console", "1") == "1":
        # Live debug console in its own window: follows this run's logs (live_view.py).
        import subprocess
        try:
            subprocess.Popen([sys.executable, "-B", os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_view.py"),
                              logdir, cwd], creationflags=0x00000010)  # CREATE_NEW_CONSOLE
        except OSError as error:
            builtins.print(f"live console not started: {error}")

    modules = {}  # base -> name
    main_base = None
    event = (ctypes.c_ubyte * 176)()
    start = time.time()
    loader_break_seen = False
    stop = None
    exit_code = None

    def where(addr):
        best = None
        for base, name in modules.items():
            if base <= addr and (best is None or base > best[0]):
                best = (base, name)
        return f"{best[1]}+{addr - best[0]:#x}" if best and addr - best[0] < 0x20000000 else f"{addr:#x}"

    def read(addr, size):
        buf = (ctypes.c_ubyte * size)()
        got = ctypes.c_size_t()
        ok = k32.ReadProcessMemory(wt.HANDLE(pi.hProcess), ctypes.c_void_p(addr), buf, size, ctypes.byref(got))
        return bytes(buf[:got.value]) if ok or got.value else b""

    def query_region(addr):
        mbi = MemoryBasicInformation()
        if k32.VirtualQueryEx(wt.HANDLE(pi.hProcess), addr, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            return mbi

    def write(addr, data):
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        old = wt.DWORD()
        k32.VirtualProtectEx(wt.HANDLE(pi.hProcess), ctypes.c_void_p(addr), len(data), 0x40, ctypes.byref(old))
        ok = k32.WriteProcessMemory(wt.HANDLE(pi.hProcess), ctypes.c_void_p(addr), buf, len(data), None)
        k32.VirtualProtectEx(wt.HANDLE(pi.hProcess), ctypes.c_void_p(addr), len(data), old.value, ctypes.byref(old))
        k32.FlushInstructionCache(wt.HANDLE(pi.hProcess), ctypes.c_void_p(addr), len(data))
        return ok

    def rewrite_for_host(address):
        """Rewrites the instruction at a faulting main-image address; returns its name, or None."""
        code = read(address, 48)
        clwb = host_cpu.clwb_rewrite(code)
        if clwb and not flushopt_missing[0]:
            return "clwb" if write(address + clwb[0], clwb[1]) else None
        nop = host_cpu.flush_nop(code)
        if nop:
            # CLFLUSHOPT itself faulted (a host older than it), or a CLWB after that was seen.
            flushopt_missing[0] = True
            return ("clwb-nop" if clwb else "clflushopt-nop") if write(address, nop) else None
        wait = host_cpu.wait_rewrite(code)
        if wait:
            return ("mwaitx" if code[2] == 0xFB else "monitorx") if write(address, wait) else None
        store = host_cpu.nt_store_rewrite(code)
        if store:
            return ("movntss" if code[0] == 0xF3 else "movntsd") if write(address + store[0], store[1]) else None
        rdpru = host_cpu.rdpru_rewrite(code)
        if rdpru:
            return "rdpru" if write(address, rdpru) else None
        found = host_cpu.stub_for(code)
        if not found:
            return None
        length, stub = found
        if stub_area[0] + len(stub) + 64 > stub_area[1]:
            # A block within rel32 reach of the image: the first free 64 KiB below its base.
            k32.VirtualAllocEx.restype = ctypes.c_void_p
            for step in range(1, 4096):
                block = k32.VirtualAllocEx(wt.HANDLE(pi.hProcess), ctypes.c_void_p(main_base - step * 0x10000),
                                           0x10000, 0x3000, 0x40)
                if block:
                    stub_area[:] = [block, block + 0x10000]
                    break
            else:
                return None
        try:
            plan = host_cpu.plan_replacement(code, address, length, stub, stub_area[0])
        except Exception as error:  # iced-x86 missing, or an instruction it cannot move
            log(f"{time.time() - start:7.3f} host-cpu no replacement at {where(address)}: {error}")
            plan = None
        if not plan or not write(stub_area[0], plan[0]) or not write(address, plan[1]):
            return None
        stub_area[0] += (len(plan[0]) + 15) & ~15
        if code[:3] == host_cpu.CLZERO:
            return "clzero"
        if host_cpu.rdpid_stub(code):
            return "rdpid"
        return "insertq" if code[0] == 0xF2 else "extrq"

    def sha_for_host(tid, address):
        """Performs a SHA instruction on the stopped thread's context; True when it did."""
        thread, ctx = get_ctx(tid)
        if not thread:
            return False

        def gpr(n):
            return struct.unpack_from("<Q", ctx, 0x78 + 8 * n)[0]

        def xmm(n):
            return int.from_bytes(bytes(ctx[0x1A0 + 16 * n:0x1B0 + 16 * n]), "little")

        try:
            done = host_cpu.sha_apply(read(address, 15), address, gpr, xmm, read)
        except ImportError:
            done = None
        if not done:
            k32.CloseHandle(wt.HANDLE(thread))
            return False
        length, dest, value = done
        struct.pack_into("<16s", ctx, 0x1A0 + 16 * dest, value.to_bytes(16, "little"))
        struct.pack_into("<Q", ctx, 0xF8, address + length)
        set_ctx(thread)
        return True

    ctx_raw = (ctypes.c_ubyte * (1232 + 16))()
    ctx_addr = (ctypes.addressof(ctx_raw) + 15) & ~15
    ctx_view = (ctypes.c_ubyte * 1232).from_address(ctx_addr)

    ctx_error = [0]  # GetLastError of the last failed get_ctx

    def get_ctx(tid):
        thread = k32.OpenThread(0x0008 | 0x0010 | 0x0002, False, tid)  # GET | SET_CONTEXT | SUSPEND
        struct.pack_into("<I", ctx_view, 0x30, 0x10000B)
        if not thread or not k32.GetThreadContext(wt.HANDLE(thread), ctypes.c_void_p(ctx_addr)):
            ctx_error[0] = k32.GetLastError()  # before CloseHandle can change it
            if thread:
                k32.CloseHandle(wt.HANDLE(thread))
            return None, None
        return thread, ctx_view

    def set_ctx(thread):
        k32.SetThreadContext(wt.HANDLE(thread), ctypes.c_void_p(ctx_addr))
        k32.CloseHandle(wt.HANDLE(thread))

    def ustr(addr):  # UNICODE_STRING*
        raw = read(addr, 16)
        if len(raw) < 16:
            return "?"
        length, _, buffer = struct.unpack_from("<HH4xQ", raw)
        return read(buffer, length).decode("utf-16-le", errors="replace")

    def oa_name(addr):  # OBJECT_ATTRIBUTES*
        raw = read(addr, 24)
        if len(raw) < 24:
            return "?"
        root, name = struct.unpack_from("<8xQQ", raw)
        text = ustr(name) if name else ""
        return (f"[root {root:#x}]" if root else "") + text

    trace = opts.get("--trace-files") == "1" or opts.get("--offline", "1") == "1"  # offline mode needs the hooks
    # --trace-calls 1: also break on gaming-runtime and movie-player calls. Every hit stops the whole
    # process, so these hooks are off by default; the movie-player report (bink_report) needs them.
    # The call hooks are armed with the file hooks, so without file tracing (--trace-files 1, or offline
    # mode, the default) --trace-calls 1 arms nothing; the summary says so.
    calls_requested = opts.get("--trace-calls") == "1"
    trace_calls = trace and calls_requested
    k32.GetModuleHandleW.restype = wt.HMODULE
    ntdll = k32.GetModuleHandleW("ntdll.dll")
    k32.GetProcAddress.restype = ctypes.c_void_p
    k32.GetProcAddress.argtypes = [wt.HMODULE, ctypes.c_char_p]
    hooks = {}
    if trace:
        for name, arg in (("NtCreateFile", "r8"), ("NtOpenFile", "r8"), ("NtQueryAttributesFile", "rcx"),
                          ("NtQueryFullAttributesFile", "rcx"), ("NtOpenDirectoryObject", "r8"),
                          ("NtOpenSymbolicLinkObject", "r8"), ("NtAlpcConnectPort", "ustr:rdx"),
                          ("NtAlpcConnectPortEx", "rdx")):
            addr = k32.GetProcAddress(ntdll, name.encode())
            if addr:
                hooks[addr] = (name, arg)
    # X2W_VALLOC_MIN=<hex bytes>: log every virtual-memory allocation of at least that size with the
    # main-image and stand-in return addresses found on the caller's stack (files.txt, "VALLOC").
    # The call is logged on entry: "requested" is the address the caller asked for (0 for any), not
    # the address it got.
    valloc_min = int(os.environ.get("X2W_VALLOC_MIN") or "0", 16)
    if valloc_min and trace:
        for name in ("NtAllocateVirtualMemory", "NtAllocateVirtualMemoryEx"):
            addr = k32.GetProcAddress(ntdll, name.encode())
            if addr:
                hooks[addr] = (name, "valloc")
    bink_counts, bink_state = {}, {}
    hit_counts = {}
    redirects = {}
    # Offline by default (--offline 0 turns it off): no sockets, NetBIOS name devices or DNS resolver.
    offline = opts.get("--offline", "1") == "1"
    OFFLINE_DENY = ("\\device\\afd", "\\device\\netbt", "dnsresolver", "\\device\\tcp", "\\device\\udp",
                    "\\device\\rawip", "\\device\\ws2ifsl")
    denied = {}

    def bink_call(name, regs, ret):
        """Counts the call, remembers the movie handle and frame buffers; returns a log line or None."""
        bink_counts[name] = bink_counts.get(name, 0) + 1
        if name == "BinkRegisterFrameBuffers":
            bink_state["bink"], bink_state["buffers"] = regs["Rcx"], regs["Rdx"]
        if name == "BinkNextFrame":
            bink_state["bink"] = regs["Rcx"]
        if name in ("BinkNextFrame", "BinkShouldSkip") and bink_counts[name] > 3:
            return None
        text = ""
        if name == "BinkOpenWithOptions":
            text = repr((read(regs["Rcx"], 200) or b"").split(bytes(1))[0].decode("latin-1", "replace"))
        return f"{name} rcx={regs['Rcx']:#x} rdx={regs['Rdx']:#x} r8={regs['R8']:#x} {text} from {where(ret)}"

    def bink_report():
        """Call counts, the movie header and a PNG of the current frame's luma plane."""
        if not bink_counts:
            return
        print("  bink calls: " + ", ".join(f"{k} x{v}" for k, v in sorted(bink_counts.items())))
        header = read(bink_state["bink"], 32) if bink_state.get("bink") else None
        if header:
            print("  bink header (width, height, frames, frame, ...): " + " ".join(str(v) for v in struct.unpack("<8I", header)))
        buffers = read(bink_state["buffers"], 24 + 2 * 5 * 24) if bink_state.get("buffers") else None
        if not buffers:
            return
        total, ya_w, ya_h, c_w, c_h, current = struct.unpack_from("<iIIIII", buffers, 0)
        print(f"  bink frame buffers: total={total} luma={ya_w}x{ya_h} chroma={c_w}x{c_h} current={current}")
        if not (0 < ya_w <= 8192 and 0 < ya_h <= 8192):
            return
        plane = 24 + (current % 2) * 5 * 24  # header, then two plane sets of five 24-byte planes; luma first
        pointer = struct.unpack_from("<Q", buffers, plane + 8)[0]
        pitch = struct.unpack_from("<I", buffers, plane + 16)[0]
        rows = []
        for y in range(ya_h):
            row = read(pointer + y * pitch, ya_w)
            if not row:
                break
            rows.append(bytes(1) + row)
        if len(rows) != ya_h:
            print(f"  bink luma plane unreadable at {pointer:#x} pitch {pitch}")
            return
        import zlib

        def chunk(kind, payload):
            body = kind + payload
            return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

        signature = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        png = (signature + chunk(b"IHDR", struct.pack(">IIBBBBB", ya_w, ya_h, 8, 0, 0, 0, 0)) +
               chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))
        with open(os.path.join(logdir, "bink_luma.png"), "wb") as handle:
            handle.write(png)
        print(f"  bink luma plane -> bink_luma.png (pointer {pointer:#x}, pitch {pitch})")

    breakpoints = {}  # addr -> original byte
    retired = set()  # X2W_LOG_AT breakpoints removed at their hit limit
    late_hits = {}  # retired addr -> INT3 events that arrived after the removal
    returns = {}  # addr -> list of (tid, rsp after return, description)
    live_threads = set()

    def suspend_thread(thread_id):
        handle = k32.OpenThread(0x0002, False, thread_id)  # SUSPEND_RESUME
        if not handle:
            return False
        suspended = k32.SuspendThread(wt.HANDLE(handle)) != 0xFFFFFFFF
        k32.CloseHandle(wt.HANDLE(handle))
        return suspended

    def resume_thread(thread_id):
        handle = k32.OpenThread(0x0002, False, thread_id)
        if handle:
            k32.ResumeThread(wt.HANDLE(handle))
            k32.CloseHandle(wt.HANDLE(handle))

    gate = StepGate(suspend_thread, resume_thread, lambda: live_threads)
    file_log = open(os.path.join(logdir, "files.txt"), "w", encoding="utf-8") if trace or log_at else None

    def arm(addr, *, skip_int3=False):
        if addr not in breakpoints:
            original = read(addr, 1)
            if original and (not skip_int3 or original != b"\xcc") and write(addr, b"\xcc"):
                breakpoints[addr] = original
        return addr in breakpoints and (not skip_int3 or breakpoints[addr] != b"\xcc")

    def handle_breakpoint(tid, addr):
        thread, ctx = get_ctx(tid)
        if not thread:
            return False
        regs = dict(zip(REGS, struct.unpack_from("<16Q", ctx, 0x78) + struct.unpack_from("<Q", ctx, 0xF8)))
        rva = addr - main_base if log_at else None
        point = log_at.get(rva)
        if point and logged_hits.get(rva, 0) < point[0]:
            hit = logged_hits[rva] = logged_hits.get(rva, 0) + 1
            values = []
            for item, reg, offset in point[1]:
                value = regs[reg]
                if offset is not None:
                    value = read_qword(read, (value + offset) & 0xffffffffffffffff)
                values.append(f"{item}={'unreadable' if value is None else hex(value)}")
            file_log.write(f"{time.time() - start:7.3f} LOG_AT tid={tid} rva={rva:#x} hit={hit} {' '.join(values)}\n")
            file_log.flush()
        rsp = regs["Rsp"]
        if addr in hooks:
            name, arg = hooks[addr]
            ret = read_qword(read, rsp) or 0
            if arg == "valloc":
                # NtAllocateVirtualMemory(process, &base, zero bits, &size, type, protect) and
                # NtAllocateVirtualMemoryEx(process, &base, &size, type, protect, ...).
                extended = name.endswith("Ex")
                size = read_qword(read, regs["R8" if extended else "R9"])
                if size is not None and size >= valloc_min:
                    base = read_qword(read, regs["Rdx"])
                    requested = "unreadable" if base is None else hex(base)
                    stack_args = struct.unpack("<2Q", (read(rsp + 0x28, 16) or b"").ljust(16, b"\0"))
                    kind, protect = (regs["R9"], stack_args[0]) if extended else stack_args
                    words = struct.unpack_from("<200Q", read_stack(read, rsp, 200))
                    chain = [where(w) for w in words if any(b <= w < b + 0x10000000 for b in modules)]
                    chain = [c for c in chain if not c.lower().startswith(("ntdll", "kernelbase", "kernel32"))]
                    file_log.write(f"{time.time() - start:7.3f} VALLOC {name} requested={requested} size={size:#x} "
                                   f"type={kind & 0xffffffff:#x} protect={protect & 0xffffffff:#x} callers: {' '.join(chain[:10])}\n")
                    file_log.flush()
                desc = None
            elif arg == "count":
                # Title function hit counter: first hits are logged with registers and the caller.
                hit_counts[name] = hit_counts.get(name, 0) + 1
                desc = None
                if name in redirects and hit_counts[name] == redirects[name][0]:
                    target = redirects[name][1]
                    file_log.write(f"{time.time() - start:7.3f} REDIRECT {name} hit {hit_counts[name]} -> {where(target)} "
                                   f"(experiment)\n")
                    file_log.flush()
                    struct.pack_into("<Q", ctx, 0xF8, target)
                    set_ctx(thread)
                    return True
                if hit_counts[name] <= 12:
                    file_log.write(f"{time.time() - start:7.3f} HIT {name} rcx={regs['Rcx']:#x} rdx={regs['Rdx']:#x} "
                                   f"r8={regs['R8']:#x} r9={regs['R9']:#x} [rdx]={(read(regs['Rdx'], 32) or b'').hex(' ')} "
                                   f"from {where(ret)}\n")
                    file_log.flush()
            elif arg == "runtime":
                def guid(address):
                    raw = read(address, 16) if address else None
                    if not raw:
                        return f"{address:#x}"
                    a, b, c = struct.unpack_from("<IHH", raw)
                    return f"{a:08x}-{b:04x}-{c:04x}-{raw[8:10].hex()}-{raw[10:].hex()}"
                desc = f"{name} " + (f"provider={guid(regs['Rcx'])} iid={guid(regs['Rdx'])}" if name == "QueryApiImpl"
                                      else f"rcx={regs['Rcx']:#x} rdx={regs['Rdx']:#x}") + f" from {where(ret)}"
            elif arg == "bink":
                desc = bink_call(name, regs, ret)
            else:
                if arg.startswith("ustr:"):
                    target = ustr(regs[arg[5:].capitalize()])
                else:
                    target = oa_name(regs[arg.capitalize()] if arg != 'r8' else regs['R8'])
                desc = f"{name} {target} from {where(ret)}"
                if offline and any(pattern in target.lower() for pattern in OFFLINE_DENY):
                    # Offline mode: the call never runs. Return STATUS_ACCESS_DENIED to the caller.
                    denied[target] = denied.get(target, 0) + 1
                    file_log.write(f"{time.time() - start:7.3f} DENIED (offline) {desc}\n")
                    file_log.flush()
                    struct.pack_into("<Q", ctx, 0x78, 0xC0000022)   # rax
                    struct.pack_into("<Q", ctx, 0x98, rsp + 8)      # rsp: pop the return address
                    struct.pack_into("<Q", ctx, 0xF8, ret)          # rip
                    set_ctx(thread)
                    return True
            if desc and ("memory:" in desc or "CALLERS" in os.environ.get("X2W_TRACE", "")):
                # Title-level callers: module addresses on the stack outside the system DLLs.
                words = struct.unpack_from("<160Q", (read(rsp, 8 * 160) or b"").ljust(8 * 160, b"\0"))
                chain = [where(w) for w in words if any(b <= w < b + 0x10000000 for b in modules)]
                chain = [c for c in chain if not c.lower().startswith(("ntdll", "kernelbase", "kernel32"))]
                desc += " callers: " + " ".join(chain[:8])
            if desc:
                returns.setdefault(ret, []).append((tid, rsp + 8, desc))
                arm(ret)
        for item in list(returns.get(addr, [])):
            if item[0] == tid and item[1] == rsp:
                returns[addr].remove(item)
                file_log.write(f"{time.time() - start:7.3f} {item[2]} -> {regs['Rax'] & 0xFFFFFFFF:#010x}\n")
                file_log.flush()
        struct.pack_into("<Q", ctx, 0xF8, addr)
        # After the final logged hit, let the restored instruction run at full speed unless
        # another hook still owns this breakpoint.
        if (point and logged_hits[rva] == point[0] and addr not in hooks and not returns.get(addr)
                and write(addr, breakpoints[addr])):
            del breakpoints[addr]
            retired.add(addr)
            set_ctx(thread)
            return True
        # Suspend every peer first; only then open the byte. If a peer cannot be suspended the byte stays
        # patched and the thread re-executes the breakpoint, so no call runs through the restored byte.
        if gate.begin(tid, addr) is None:
            set_ctx(thread)
            return True
        # Step over the original instruction, then re-arm.
        eflags = struct.unpack_from("<I", ctx, 0x44)[0]
        struct.pack_into("<I", ctx, 0x44, eflags | 0x100)
        set_ctx(thread)
        write(addr, breakpoints[addr])
        return True

    def dump_threads(path):
        """At a timeout: every thread's rip and the module addresses found on its stack."""
        lines = []
        dbghelp = ctypes.WinDLL("dbghelp", use_last_error=True)
        dbghelp.SymSetOptions(0x00000004 | 0x00000200)  # DEFERRED_LOADS | FAIL_CRITICAL_ERRORS
        dbghelp.SymInitializeW(wt.HANDLE(pi.hProcess), None, True)
        dbghelp.StackWalk64.argtypes = [ctypes.c_uint32, wt.HANDLE, wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        table_access = ctypes.cast(dbghelp.SymFunctionTableAccess64, ctypes.c_void_p)
        module_base = ctypes.cast(dbghelp.SymGetModuleBase64, ctypes.c_void_p)
        for thread_id in sorted(live_threads):
            thread = k32.OpenThread(0x0008 | 0x0002 | 0x0040, False, thread_id)
            if not thread:
                continue
            k32.SuspendThread(wt.HANDLE(thread))
            raw = (ctypes.c_ubyte * (1232 + 16))()
            ctx_addr = (ctypes.addressof(raw) + 15) & ~15
            struct.pack_into("<I", (ctypes.c_ubyte * 1232).from_address(ctx_addr), 0x30, 0x10000B)
            if k32.GetThreadContext(wt.HANDLE(thread), ctypes.c_void_p(ctx_addr)):
                ctx = bytes((ctypes.c_ubyte * 1232).from_address(ctx_addr))
                rsp = struct.unpack_from("<Q", ctx, 0x98)[0]
                rbp = struct.unpack_from("<Q", ctx, 0xA0)[0]
                rip = struct.unpack_from("<Q", ctx, 0xF8)[0]
                # Real unwinding through the modules' unwind tables (dbghelp StackWalk64).
                frame = (ctypes.c_ubyte * 512)()
                for offset, value in ((0, rip), (32, rbp), (48, rsp)):
                    struct.pack_into("<Q", frame, offset, value)
                    struct.pack_into("<I", frame, offset + 12, 3)  # AddrModeFlat
                frames = []
                for _ in range(28):
                    if not dbghelp.StackWalk64(0x8664, wt.HANDLE(pi.hProcess), wt.HANDLE(thread), frame,
                                               ctypes.c_void_p(ctx_addr), None, table_access, module_base, None):
                        break
                    pc = struct.unpack_from("<Q", frame, 0)[0]
                    if not pc:
                        break
                    frames.append(where(pc))
                lines.append(f"thread {thread_id} rip={where(rip)} stack: " + " ".join(frames[1:]))
            k32.ResumeThread(wt.HANDLE(thread))
            k32.CloseHandle(wt.HANDLE(thread))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return lines

    watches = [(int(r, 16), int(n or "8", 16)) for r, _, n in
               (item.partition(":") for item in filter(None, os.environ.get("X2W_WATCH", "").split(",")))]
    watch_values, watch_next = {}, 0.0
    # X2W_DUMP_AT=<seconds>[,...]: write every thread's stack to threads_<seconds>.txt at those times,
    # for a run that stalls before a watchdog or a timeout ends it.
    dump_at = sorted(float(t) for t in filter(None, os.environ.get("X2W_DUMP_AT", "").split(",")))
    # X2W_STOP_AT=<rva>[,...]: stop at a main-image breakpoint at one of these addresses even while
    # --continue-asserts would step over it, and write threads_stop.txt.
    stop_at = {int(r, 16) for r in filter(None, os.environ.get("X2W_STOP_AT", "").split(","))}

    def poll_watches():
        for rva, size in watches:
            blob = read(main_base + rva, size) if main_base else None
            if blob is not None and watch_values.get(rva) != blob:
                watch_values[rva] = blob
                log(f"{time.time() - start:7.3f} watch +{rva:#x} = {blob.hex(' ')}")

    def grab_window():
        """The title's window as (width, height, BGRA bytes, title text length), or the reason there is none.

        Runs on the capture thread as well as at the end of the run; every GDI object is released."""
        user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
        found = []
        proto = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        def visit(hwnd, _):
            owner = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pi.dwProcessId and user32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        user32.EnumWindows(proto(visit), 0)
        if not found:
            return "no visible window"
        hwnd = found[0]
        rect = wt.RECT()
        user32.GetClientRect(wt.HWND(hwnd), ctypes.byref(rect))
        width, height = min(rect.right, 1920), min(rect.bottom, 1080)
        if width <= 0 or height <= 0:
            return "empty client area"
        user32.GetDC.restype = wt.HDC
        gdi32.CreateCompatibleDC.restype = wt.HDC
        gdi32.CreateCompatibleBitmap.restype = wt.HBITMAP
        source = user32.GetDC(wt.HWND(hwnd))
        memory = gdi32.CreateCompatibleDC(wt.HDC(source))
        bitmap = gdi32.CreateCompatibleBitmap(wt.HDC(source), width, height)
        try:
            gdi32.SelectObject(wt.HDC(memory), wt.HBITMAP(bitmap))
            gdi32.BitBlt(wt.HDC(memory), 0, 0, width, height, wt.HDC(source), 0, 0, 0x00CC0020)
            header = struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
            pixels = (ctypes.c_ubyte * (width * height * 4))()
            gdi32.GetDIBits(wt.HDC(memory), wt.HBITMAP(bitmap), 0, height, pixels, header, 0)
            raw = bytes(pixels)
        finally:
            gdi32.DeleteObject(wt.HGDIOBJ(bitmap))
            gdi32.DeleteDC(wt.HDC(memory))
            user32.ReleaseDC(wt.HWND(hwnd), wt.HDC(source))
        title = ctypes.create_unicode_buffer(128)
        user32.GetWindowTextW(wt.HWND(hwnd), title, 128)
        return width, height, raw, len(title.value)

    def capture_window():
        """Saves what the title's window currently shows on screen (client area) as window.png."""
        frame = grab_window()
        if isinstance(frame, str):
            print(f"  window capture: {frame}")
            return
        width, height, raw, title_length = frame
        with open(os.path.join(logdir, "window.png"), "wb") as handle:
            handle.write(png_bytes(width, height, raw))
        print(f"  window capture: {width}x{height} -> window.png (client area as on screen; title text length {title_length})")

    # --capture-every: a thread of its own grabs the window, so a slow capture never delays the debug events.
    series = Series(logdir) if capture_every else None
    capture_done = threading.Event()

    def capture_loop():
        while not capture_done.wait(capture_every):
            try:
                frame = grab_window()
                if isinstance(frame, str):
                    series.miss()
                else:
                    series.offer(time.time() - start, frame[0], frame[1], frame[2])
            except Exception:  # diagnostics only: the run goes on, the tick is counted
                series.miss()

    capture_thread = threading.Thread(target=capture_loop, name="window-capture", daemon=True) if series else None
    if capture_thread:
        capture_thread.start()
    stop_requested = False
    stop_next = 0.0
    while not timeout or time.time() - start < timeout:
        if time.time() >= stop_next:
            stop_next = time.time() + 0.5
            if os.path.exists(stop_file):
                stop_requested = True
                log(f"{time.time() - start:7.3f} stop file found: ending the run")
                break
        if dump_at and time.time() - start >= dump_at[0]:
            dump_threads(os.path.join(logdir, f"threads_{dump_at.pop(0):g}.txt"))
        if watches and time.time() >= watch_next:
            watch_next = time.time() + 1.0
            poll_watches()
        if not k32.WaitForDebugEvent(event, 200):
            continue
        code, pid, tid = struct.unpack_from("<III", event, 0)
        status = DBG_CONTINUE
        if code == CREATE_PROCESS:
            live_threads.add(tid)
            h_file, _, _, base = struct.unpack_from("<QQQQ", event, 16)
            main_base = base
            for item in filter(None, os.environ.get("X2W_COUNT", "").split(",")):
                hooks[base + int(item, 16)] = (f"+{item}", "count")
            # Experiment switch X2W_REDIRECT=<rva>@<n>:<target rva>[,...]: on the n-th hit of a title
            # address, execution continues at another address instead. Used to force one code path.
            for item in filter(None, os.environ.get("X2W_REDIRECT", "").split(",")):
                source, _, target = item.partition(":")
                rva, _, nth = source.partition("@")
                hooks[base + int(rva, 16)] = (f"+{rva}", "count")
                redirects[f"+{rva}"] = (int(nth or "1"), base + int(target, 16))
            modules[base] = os.path.basename(exe)
            log(f"{time.time() - start:7.3f} process {os.path.basename(exe)} at {base:#x}")
            if h_file:
                k32.CloseHandle(wt.HANDLE(h_file))
        elif code == LOAD_DLL:
            h_file, base = struct.unpack_from("<QQ", event, 16)
            name = final_path(h_file) if h_file else "?"
            modules[base] = name
            log(f"{time.time() - start:7.3f} load {name} at {base:#x}")
            if trace_calls and name.lower() == RUNTIME_MODULE:
                # Gaming runtime: log which API interfaces the title asks for and the result.
                k32.LoadLibraryExW.restype = wt.HMODULE
                local = k32.LoadLibraryExW(r"C:\Windows\System32\xgameruntime.dll", None, 1)
                for export in RUNTIME_CALL_HOOKS:
                    address = k32.GetProcAddress(local, export.encode()) if local else 0
                    if address:
                        hooks[base + (address - local)] = (export, "runtime")
                        if loader_break_seen:
                            arm(base + (address - local))
            if trace_calls and name.lower() == MOVIE_MODULE:
                # Movie player: log opens/closes/frames and remember the registered frame buffers.
                k32.LoadLibraryExW.restype = wt.HMODULE
                local = k32.LoadLibraryExW(os.path.join(os.path.dirname(exe), name), None, 1)
                for export in MOVIE_CALL_HOOKS:
                    address = k32.GetProcAddress(local, export.encode()) if local else 0
                    if address:
                        hooks[base + (address - local)] = (export, "bink")
                        if loader_break_seen:
                            arm(base + (address - local))
            if h_file:
                k32.CloseHandle(wt.HANDLE(h_file))
        elif code == CREATE_THREAD:
            log(f"{time.time() - start:7.3f} thread {tid} created")
            live_threads.add(tid)
        elif code == 4:  # EXIT_THREAD
            live_threads.discard(tid)
            addr = gate.forget(tid)
            if addr in breakpoints:
                write(addr, b"\xcc")
        elif code == EXIT_PROCESS:
            exit_code = struct.unpack_from("<I", event, 16)[0]
            log(f"{time.time() - start:7.3f} exit {exit_code:#010x}")
            k32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
            break
        elif code == OUTPUT_STRING:
            addr, unicode, length = struct.unpack_from("<QHH", event, 16)
            raw = read(addr, length * (2 if unicode else 1))
            text = raw.decode("utf-16-le" if unicode else "latin-1", errors="replace").rstrip("\0\r\n")
            log(f"{time.time() - start:7.3f} debug-output {text[:300]}")
        elif code == EXCEPTION:
            ex_code, ex_flags, _, ex_addr, nparams = struct.unpack_from("<IIQQI", event, 16)
            params = struct.unpack_from("<" + "Q" * min(nparams, 15), event, 16 + 32) if nparams else ()
            first = struct.unpack_from("<I", event, 16 + 152)[0]
            label = NAMES.get(ex_code, "")
            if ex_code == 0x80000003 and ex_addr in breakpoints:
                handle_breakpoint(tid, ex_addr)
                k32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
                continue
            if ex_code == 0x80000003 and ex_addr in retired:
                # Another thread executed this INT3 before the hit limit restored the original byte,
                # and its event arrived after that. Rewind and run the restored instruction.
                thread, ctx = get_ctx(tid)
                if thread:
                    struct.pack_into("<Q", ctx, 0xF8, ex_addr)
                    set_ctx(thread)
                    late_hits[ex_addr] = late_hits.get(ex_addr, 0) + 1
                    k32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
                    continue
                log(f"{time.time() - start:7.3f} X2W_LOG_AT late hit at {where(ex_addr)} on thread {tid}: "
                    f"thread context unavailable (error {ctx_error[0]}), handled as a breakpoint")
            if ex_code == 0x80000004 and tid in gate.stepping:
                addr = gate.finish(tid)
                if addr in breakpoints:
                    write(addr, b"\xcc")
                k32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
                continue
            if (ex_code == 0xC000001D and first and host_cpu_on and not stop_first and main_base
                    and main_base <= ex_addr < main_base + 0x10000000):
                name = rewrite_for_host(ex_addr)
                if name:
                    host_cpu_rewritten.add(ex_addr)
                elif sha_for_host(tid, ex_addr):
                    name = "sha (per hit)"
                elif ex_addr in host_cpu_rewritten and (ex_addr, tid) not in host_cpu_resumed:
                    # Another thread ran the old instruction before the rewrite and its fault arrived after:
                    # continued, it runs the replacement. Once per thread, so a fault that stays is reported.
                    host_cpu_resumed.add((ex_addr, tid))
                    k32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
                    continue
                if name:
                    host_cpu_sites[name] = host_cpu_sites.get(name, 0) + 1
                    if sum(host_cpu_sites.values()) <= 40:
                        log(f"{time.time() - start:7.3f} host-cpu {name} at {where(ex_addr)} rewritten")
                    k32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
                    continue
            log(f"{time.time() - start:7.3f} exception {ex_code:#010x} {label} at {where(ex_addr)} "
                f"{'first' if first else 'second'}-chance params={[hex(p) for p in params]}")
            if ex_code == 0x80000003 and not loader_break_seen:
                loader_break_seen = True  # the loader's initial breakpoint
                for addr in hooks:
                    arm(addr)
                for rva in tuple(log_at):
                    addr = main_base + rva
                    region = query_region(addr)
                    # PAGE_EXECUTE variants use bits 4–7. Leave guard pages and other images alone.
                    if (region and region.AllocationBase == main_base and region.Protect & 0xf0
                            and not region.Protect & 0x100 and arm(addr, skip_int3=True)):
                        continue
                    log(f"X2W_LOG_AT ignored {rva:#x}: cannot arm a main-image instruction breakpoint")
                    del log_at[rva]
            elif (ex_code == 0x80000003 and first and main_base and main_base <= ex_addr < main_base + 0x10000000
                  and (skip_asserts == -1 or sum(asserts_seen.values()) < skip_asserts)
                  and (ex_addr - main_base) not in stop_at):
                asserts_seen[ex_addr] = asserts_seen.get(ex_addr, 0) + 1
                if nop_soft_asserts and ex_addr not in nop_sites and read(ex_addr, 1) == b"\xcc" and write(ex_addr, b"\x90"):
                    nop_sites[ex_addr] = 1
                status = DBG_CONTINUE  # rip is already past the int3
            elif (not first) or stop_first or (ex_code == 0x80000003 and main_base and
                                                  main_base <= ex_addr < main_base + 0x10000000):
                stop = (ex_code, ex_addr, tid, first, params)
                log(f"debugger stop {ex_code:#010x} {'first' if first else 'second'}-chance")
                break
            elif ex_code in (0x406D1388, 0x40010006):
                status = DBG_CONTINUE
            else:
                status = DBG_EXCEPTION_NOT_HANDLED
        k32.ContinueDebugEvent(pid, tid, status)

    elapsed = time.time() - start
    if gate.dropped:
        log(f"{elapsed:7.3f} concurrent hook hits held for a step: {gate.dropped}")
    if stop and stop_at:
        dump_threads(os.path.join(logdir, "threads_stop.txt"))
    if stop:
        ex_code, ex_addr, tid, first, params = stop
        print(f"stopped at {NAMES.get(ex_code, hex(ex_code))} ({ex_code:#010x}) at {where(ex_addr)} after {elapsed:.2f} s "
              f"({'first' if first else 'second'} chance) params={[hex(p) for p in params]}")
        thread = k32.OpenThread(0x0008 | 0x0002 | 0x0040, False, tid)  # GET_CONTEXT | SUSPEND | QUERY
        raw = (ctypes.c_ubyte * (1232 + 16))()
        ctx_addr = (ctypes.addressof(raw) + 15) & ~15
        struct.pack_into("<I", (ctypes.c_ubyte * 1232).from_address(ctx_addr), 0x30, 0x10000B)
        if thread and k32.GetThreadContext(wt.HANDLE(thread), ctypes.c_void_p(ctx_addr)):
            ctx = bytes((ctypes.c_ubyte * 1232).from_address(ctx_addr))
            values = struct.unpack_from("<16Q", ctx, 0x78) + struct.unpack_from("<Q", ctx, 0xF8)
            regs = dict(zip(REGS, values))
            print("  " + " ".join(f"{k.lower()}={v:#x}" for k, v in regs.items()))
            stack = read(regs["Rsp"], 8 * 16)
            words = struct.unpack_from(f"<{len(stack) // 8}Q", stack) if stack else ()
            print("  stack: " + " ".join(where(w) if any(b <= w < b + 0x10000000 for b in modules) else f"{w:#x}" for w in words))
            # Likely return addresses: module-range words in the next 8 KiB whose preceding bytes are a call.
            deep = read(regs["Rsp"], 0x2000) or b""
            rets = []
            for i in range(0, len(deep) - 7, 8):
                w = struct.unpack_from("<Q", deep, i)[0]
                if not any(b <= w < b + 0x10000000 for b in modules):
                    continue
                before = read(w - 6, 6) or b""
                if len(before) == 6 and (before[1] == 0xE8 or before[0] == 0xFF and before[1] in (0x15, 0x50, 0x90, 0x10, 0x14)
                                         or before[3] == 0xFF and (before[4] & 0x38) == 0x10 or before[4] == 0xFF and (before[5] & 0xF8) == 0xD0):
                    rets.append(f"+{i:#x}:{where(w)}")
            print("  return addresses: " + " ".join(rets[:40]))
            raw_words = struct.unpack_from(f"<{min(len(deep), 0x300) // 8}Q", deep) if deep else ()
            for row in range(0, len(raw_words), 8):
                print(f"  raw stack +{row * 8:#05x}: " + " ".join(f"{w:#x}" for w in raw_words[row:row + 8]))
            sys.stdout.flush()
            # X2W_TRAP_DUMP="exe+RVA:len,reg+off:len,0xADDR:len": memory dumps at the stop.
            exe_base = next((b for b, n in modules.items() if n.lower().endswith(".exe")), 0)
            for spec in filter(None, os.environ.get("X2W_TRAP_DUMP", "").split(",")):
                try:
                    where_s, length = spec.split(":")
                    parts = where_s.replace("-", "+-").split("+")
                    addr = 0
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        neg = part.startswith("-")
                        part = part.lstrip("-")
                        if part == "exe":
                            value = exe_base
                        elif part.capitalize() in regs:
                            value = regs[part.capitalize()]
                        else:
                            value = int(part, 16)
                        addr += -value if neg else value
                    data = read(addr, int(length, 16)) or b""
                    for o in range(0, len(data), 32):
                        print(f"  dump {spec} {addr + o:#x}: " + data[o:o + 32].hex(" ", 4))
                except Exception as error:
                    print(f"  dump {spec}: {error}")
            sys.stdout.flush()
            if ex_code == 0xC0000005 and len(params) >= 2:
                # Region around the faulting address and the allocations just below it.
                probe = params[1] & ~0xFFFF
                for _ in range(3):
                    mbi = query_region(probe)
                    if not mbi:
                        break
                    print(f"  region base={mbi.BaseAddress:#x} alloc={mbi.AllocationBase:#x} size={mbi.RegionSize:#x} "
                          f"state={mbi.State:#x} protect={mbi.Protect:#x} type={mbi.Type:#x}")
                    if mbi.BaseAddress == 0:
                        break
                    probe = mbi.BaseAddress - 0x1000
            code_bytes = read(regs["Rip"] - 32, 48)
            print(f"  code at rip-32: {code_bytes.hex(' ')}")
        if thread:
            k32.CloseHandle(wt.HANDLE(thread))
    elif exit_code is not None:
        print(f"process exited with {exit_code:#010x} after {elapsed:.2f} s")
    else:
        print(f"stopped by request after {elapsed:.2f} s (stop file)" if stop_requested
              else f"timeout after {elapsed:.2f} s")
        dumped = dump_threads(os.path.join(logdir, "threads.txt"))
        for item in filter(None, os.environ.get("X2W_DUMP", "").split(",")):
            rva, _, size = item.partition(":")
            blob = read(main_base + int(rva, 16), int(size or "64", 16)) or b""
            print(f"  dump +{rva}: " + " ".join(f"{q:#x}" for q in struct.unpack_from(f"<{len(blob) // 8}Q", blob)))
        tops = {}
        for line in dumped:
            top = line.split(" stack:")[0].split("rip=")[1]
            tops[top] = tops.get(top, 0) + 1
        print(f"  {len(dumped)} threads -> threads.txt; rip histogram: " +
              ", ".join(f"{k} x{v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1])[:8]))
        if asserts_seen:
            print("  asserts continued: " + ", ".join(f"{where(a)} x{n}" for a, n in asserts_seen.items()))
    if host_cpu_sites:
        print("  host-cpu sites rewritten: " + ", ".join(f"{name} x{n}" for name, n in sorted(host_cpu_sites.items())))
    if host_cpu_resumed:
        print(f"  host-cpu threads resumed at a site another thread had rewritten: {len(host_cpu_resumed)}")
    if nop_sites:
        print("  soft asserts NOP'd: " + ", ".join(where(addr) for addr in nop_sites))
    bink_report()
    if os.environ.get("X2W_COUNT"):
        print("  hit counts: " + ", ".join(f"+{item} x{hit_counts.get('+' + item, 0)}"
                                           for item in os.environ["X2W_COUNT"].split(",") if item))
    if log_at:
        print("  LOG_AT hit counts: " + ", ".join(f"+{rva:x} x{logged_hits.get(rva, 0)}" for rva in log_at))
        if late_hits:
            print("  LOG_AT late hits resumed after the limit: " +
                  ", ".join(f"+{addr - main_base:x} x{n}" for addr, n in sorted(late_hits.items())))
    if offline:
        print(f"  offline: {sum(denied.values())} network opens refused" +
              ("" if not denied else " (" + ", ".join(f"{k} x{v}" for k, v in list(denied.items())[:6]) + ")"))
    if trace_calls:
        print("  call tracing: on (runtime and movie-player call hooks; --trace-calls 1)")
    elif calls_requested:
        print("  call tracing: off: --trace-calls 1 needs file tracing (--trace-files 1, or offline mode, the default)")
    else:
        print("  call tracing: off (runtime and movie-player call hooks; --trace-calls 1)")

    if capture_thread:
        capture_done.set()
        capture_thread.join(30)
        print("  " + series.summary() + f" (every {capture_every:g} s, window_<sss>s.png, at most {FILE_CAP} files)")
    if exit_code is None:
        try:
            capture_window()
        except Exception as error:  # diagnostics only
            print(f"  window capture failed: {error}")
    k32.TerminateJobObject(wt.HANDLE(job), 0xDEAD0002)
    for h in (out, err, pi.hThread, pi.hProcess):
        k32.CloseHandle(wt.HANDLE(h))
    events.close()
    print(f"modules loaded: {len(modules)}")
    with open(os.path.join(logdir, "finished.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")
    for name in ("stderr.txt", "trap.log"):
        path = os.path.join(logdir, name)
        if os.path.exists(path) and os.path.getsize(path):
            print(f"--- {name}\n" + open(path, encoding="utf-8", errors="replace").read()[-3000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
