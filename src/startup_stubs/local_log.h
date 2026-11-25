// Experimental (docs/startup-runtime.md): opt-in switches and logging shared by the
// permissive startup paths. Lines go to stderr and are appended to %X2W_TRAP_LOG% when set.
#pragma once

#include "startup_stubs/trap.h"

#include <cstdarg>
#include <cstdio>

namespace x2w_local {

// Reads a switch without disturbing the caller's last-error value (GetEnvironmentVariable sets
// ERROR_ENVVAR_NOT_FOUND for an absent variable, and shim callers observe GetLastError).
inline bool env_on(const char* name) noexcept
{
    const DWORD saved = GetLastError();
    char value[4]{};
    const bool on = GetEnvironmentVariableA(name, value, sizeof(value)) && value[0] == '1';
    SetLastError(saved);
    return on;
}

inline void logf(const char* format, ...) noexcept
{
    char line[1536];
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (n < 0) return;
    if (n > static_cast<int>(sizeof(line)) - 3) n = sizeof(line) - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    const DWORD error = GetLastError();
    x2w_trap::write_all(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n));
    x2w_trap::append_log(line, static_cast<DWORD>(n), false);
    SetLastError(error);
}

// True for the first `limit` calls on `counter`. The counter stops at the limit instead of counting
// every call: a gate on a hot path would otherwise pass 2^31 calls, turn negative and pass again.
inline bool first_calls(volatile LONG& counter, LONG limit) noexcept
{
    LONG seen = counter;
    while (seen < limit) {
        const LONG previous = InterlockedCompareExchange(&counter, seen + 1, seen);
        if (previous == seen) return true;
        seen = previous;
    }
    return false;
}

// Number of contiguous readable bytes at start, up to bytes, found by touching one byte per page.
// VirtualQuery is not used for this: on the title's multi-gigabyte regions one query takes about a
// millisecond (it walks the region to size it). A bad pointer costs a first-chance exception, so
// callers on hot paths should pass values that normally are pointers. A guard page that the probe
// trips is re-armed.
inline std::size_t probe_readable(std::uintptr_t start, std::size_t bytes) noexcept
{
    if (start < 0x10000 || start > 0x00007FFFFFFEFFFFull || bytes == 0) return 0;
    std::size_t done = 0;
    DWORD code = 0;
    std::uintptr_t at = start;
    const std::uintptr_t end = start + bytes;
    __try {
        volatile unsigned char sink = 0;
        while (at < end) {
            sink = *reinterpret_cast<const volatile unsigned char*>(at);
            const std::uintptr_t next = (at | 0xFFF) + 1;
            done = static_cast<std::size_t>((next < end ? next : end) - start);
            at = next;
        }
        (void)sink;
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (code == 0x80000001u) {  // STATUS_GUARD_PAGE_VIOLATION: the system dropped the guard bit
            MEMORY_BASIC_INFORMATION info{};
            DWORD old = 0;
            if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) && info.State == MEM_COMMIT)
                VirtualProtect(reinterpret_cast<void*>(at & ~std::uintptr_t{0xFFF}), 0x1000, info.Protect | PAGE_GUARD, &old);
        }
    }
    return done;
}

// With X2W_XMEM_LAZY_COMMIT, physical pages are committed on first touch (xmem_mapped_pages.cpp). A check of the
// commit state, or a write that cannot fault (kernel I/O, GPU), commits the range first. A no-op without xmem.dll
// or without the switch.
inline void commit_lazy(const void* pointer, std::size_t bytes) noexcept
{
    using CommitRange = BOOL(WINAPI*)(const void*, SIZE_T);
    static const CommitRange commit_range = [] {
        const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
        return xmem ? reinterpret_cast<CommitRange>(GetProcAddress(xmem, "x2w_xmem_commit_range")) : nullptr;
    }();
    if (commit_range) commit_range(pointer, bytes);
}

// Commits the whole lazy startup block holding pointer (xmem's x2w_xmem_commit_block, X2W_XMEM_LAZY_STARTUP)
// before the GPU uses memory in it; does nothing when xmem.dll lacks the export.
inline void commit_lazy_block(const void* pointer) noexcept
{
    using CommitBlock = BOOL(WINAPI*)(const void*);
    static const CommitBlock commit_block = [] {
        const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
        return xmem ? reinterpret_cast<CommitBlock>(GetProcAddress(xmem, "x2w_xmem_commit_block")) : nullptr;
    }();
    if (commit_block) commit_block(pointer);
}

// Readability by probing (see probe_readable); writability by one VirtualQuery (rare callers).
inline bool accessible(const void* pointer, std::size_t bytes, bool write) noexcept
{
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    if (start < 0x10000 || start > 0x00007FFFFFFEFFFFull || bytes == 0) return false;
    if (!write) return probe_readable(start, bytes) == bytes;
    commit_lazy(pointer, bytes);
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(pointer, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
    const DWORD protect = info.Protect & 0xFF;
    if ((info.Protect & PAGE_GUARD) || protect == PAGE_NOACCESS || protect == 0) return false;
    const bool writable = protect == PAGE_READWRITE || protect == PAGE_EXECUTE_READWRITE || protect == PAGE_WRITECOPY ||
                          protect == PAGE_EXECUTE_WRITECOPY;
    const auto begin = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
    const auto end = begin + info.RegionSize;
    // start lies inside the queried region, so end - start cannot wrap; start + bytes could.
    if (bytes > end - start || (write && !writable)) return false;
    return true;
}
inline bool readable(const void* pointer, std::size_t bytes) noexcept { return accessible(pointer, bytes, false); }
inline bool writable(const void* pointer, std::size_t bytes) noexcept { return accessible(pointer, bytes, true); }

}  // namespace x2w_local
