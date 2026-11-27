// Readers that do not fault, inside a run break of the shared-view mode (X2W_XMEM_SHARED_VIEWS=1). Unmapping one
// page of a run unmaps the whole run, then maps its other pages again (src/xmem_mapped_pages.cpp, map_pages_locked).
// A thread that touches one of those pages in between faults and is retried; a region query or a kernel copy does
// not fault, so it reads the pages as reserved or fails. Each case below opens that window with the test-controls
// build's X2W_XMEM_TEST_HOLD_WINDOW, proves it is open with a raw VirtualQuery of a neighbour page (not committed),
// starts the reader under test on another thread, checks that it has not answered 300 ms later, releases the window
// and then checks the reader's answer: the neighbour page is committed, mapped and holds its data.
//
// Cases (one window each): x2w_xmem_stable_call, XMemVirtualQuery, this program's VirtualQueryEx import (wrapped by
// xmem.dll when it loads with the switch), this program's ReadFile and WriteFile imports (synchronous, into and out
// of the page; wrapped the same way), a ReadFile from an anonymous pipe into the page (not run again: it fails inside
// the window), x2w_local::writable, x2w_startup::measure_placed_buffer and x2w_startup::snapshot_samplers. Then
// io_outside_runs: a ReadFile into a reserved page in no run fails at once.
//
// With `control` (the build whose local_log.h asks xmem.dll for a missing export, so the header readers run
// directly) the three header readers answer inside the window: the check expects that, and fails if they wait.
//
// Then a page that faults in two breaks in a row (retry_remapped_page): a thread faults on a page of a run taken
// apart by break A, and the test build's X2W_XMEM_TEST_HOLD_RETRY holds the handler until break A has ended, so the
// handler finds the page mapped again and looks up the stamp break A left on the run; the thread is retried and reads
// its page. Break B then takes the same page apart and stamps the run again, and the thread faults on it again: that
// fault must be retried as well (a different stamp). With `next-break-control` the handler is not held, so it sees
// break A in progress and waits for memory.mutex before the lookup: that order passes even when the handler's check
// of a fault after its break has ended is wrong, which shows the hold is what this case depends on.
//
// A fault whose handler looks the page up only after a later break on another run has begun (two_runs: break B on a
// disjoint run; mirror: one permissive XMemMapPhysicalPages call that joins the CPU run and then its GPU mirror run)
// must be retried as well: the page's run keeps the stamp of the break that mapped it again, whatever breaks follow on
// other runs. Joins across calls need X2W_XMEM_JOIN_CALLS=1: main sets it for `mirror` and clears it for every other
// mode.
//
// Many later breaks (later_breaks): a held fault in break A whose handler runs after 63, 64 or 200 later breaks on
// another page is retried each time (the reader's own __except would catch a fault passed on). A fault that stays
// (persistent_fault, a page of a run made no-access): passed on at once when no break mapped the run, and after one
// retry when one did; a first-chance handler of the test counts the faults.
//
// A break that begins while x2w_xmem_stable_call's fn runs (break_during_fn): fn itself starts the unmap, waits until
// the window is open and queries a page of the run (reserved). stable_call must see the generation change and run fn
// again under memory.mutex: 2 calls, the last one committed.
//
// `default` (X2W_XMEM_SHARED_VIEWS unset): every page has a view of its own, and readers that overlap an unmap
// answer inside it, without waiting for memory.mutex, as before x2w_xmem_stable_call.
// `lazy` (X2W_XMEM_SHARED_VIEWS=1, X2W_XMEM_LAZY_COMMIT=1): XMemVirtualQuery and VirtualQueryEx wait for a break
// with their output buffer in a mapped slot that is not committed yet; the answer reaches it after the break. Then a
// ReadFile into another such slot, with no break: the lazy-commit ReadFile wrapper still runs inside the run-break one.
//
// `two-runs` and `mirror` run those two cases alone (an access violation there ends the process); `stamps` runs
// later_breaks with 63, 64 and 200 later breaks, then persistent_fault.
//
// Usage: xmem_run_break_readers_check <xmem.dll built with its test controls>
//        [control|next-break-control|default|lazy|two-runs|mirror|stamps]
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <d3d12.h>

#include "startup_stubs/local_log.h"
#include "startup_stubs/placed_buffer_range.h"
#include "startup_stubs/translated_sampler_bindings.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>

namespace {
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(SIZE_T, SIZE_T*, SIZE_T*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, SIZE_T, SIZE_T*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(SIZE_T, SIZE_T*);
using XMemVirtualQueryFn = BOOL(WINAPI*)(HANDLE, LPCVOID, void*);
using StableCallFn = BOOL(WINAPI*)(void (*)(void*), void*);

constexpr SIZE_T page = SIZE_T{64} << 10;
constexpr SIZE_T pages = 8;
constexpr SIZE_T broken_page = 5;   // the page unmapped; its run [0, 8) is taken apart
constexpr SIZE_T neighbour = 2;     // a surviving page of that run, unmapped while the window is held

int failures = 0;

bool check(bool ok, const char* name, const char* reader)
{
    std::printf("%s: %s %s\n", ok ? "ok" : "check_failed", reader, name);
    if (!ok) ++failures;
    return ok;
}

DWORD raw_state(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) ? info.State : 0;
}

struct Shim {
    XMemVirtualAllocFn virtual_alloc;
    XMemAllocatePhysicalPagesFn allocate;
    XMemMapPhysicalPagesFn map;
    XMemFreePhysicalPagesFn release;
    XMemVirtualQueryFn query;
    StableCallFn stable_call;
};

// One window: maps a fresh run of eight pages at base, marks every page, unmaps broken_page on a thread of its own
// with the hold set, and runs reader on another thread once the window is open. reader returns whether its answer
// says the neighbour page is committed and readable. waits: the reader must not answer before the release.
void window_case(const Shim& shim, unsigned char* base, int index, const char* name, bool waits,
                 const std::function<bool(unsigned char*)>& reader)
{
    SIZE_T count = pages, tokens[pages] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages || shim.map(base, pages, tokens) != base) {
        check(false, "map_run", name);
        return;
    }
    for (SIZE_T i = 0; i < pages; ++i) *reinterpret_cast<std::uint32_t*>(base + i * page) = 0x51000000u + static_cast<std::uint32_t>(i);
    char hold[64], event_name[96];
    std::snprintf(hold, sizeof(hold), "x2w-run-break-readers-%lu-%d", GetCurrentProcessId(), index);
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", hold);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", hold);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold);
    unsigned char* const broken = base + broken_page * page;
    std::thread unmapper([&] { shim.map(broken, 1, nullptr); });
    const bool opened = WaitForSingleObject(unmapped, 10000) == WAIT_OBJECT_0;
    check(opened, "window_reached", name);
    unsigned char* const target = base + neighbour * page;
    if (opened) check(raw_state(target) != MEM_COMMIT, "window_open_raw_query_not_committed", name);
    std::atomic<bool> started{false}, answered{false};
    bool answer = false;
    std::thread reading([&] {
        started = true;
        answer = reader(target);
        answered = true;
    });
    for (int i = 0; i < 10000 && !started.load(); ++i) Sleep(1);  // the 300 ms count from the reader's start
    Sleep(300);
    const bool early = answered.load();
    check(waits ? !early : early, waits ? "reader_waits_for_the_break" : "reader_answers_inside_the_window", name);
    SetEvent(release);
    reading.join();
    unmapper.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(waits ? answer : !answer, waits ? "answer_after_the_break_committed" : "answer_inside_the_window_not_committed", name);
    check(*reinterpret_cast<std::uint32_t*>(target) == 0x51000000u + neighbour, "neighbour_data_kept", name);
    shim.map(base, broken_page, nullptr);
    shim.map(base + (broken_page + 1) * page, pages - broken_page - 1, nullptr);
    shim.release(pages, tokens);
    CloseHandle(unmapped);
    CloseHandle(release);
}
// fn starts the unmap and waits for the window inside x2w_xmem_stable_call.
void break_during_fn(const Shim& shim, unsigned char* base)
{
    const char* const name = "break_during_fn";
    SIZE_T count = pages, tokens[pages] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages || shim.map(base, pages, tokens) != base) {
        check(false, "map_run", name);
        return;
    }
    struct Context {
        const Shim* shim;
        unsigned char* base;
        char hold[64];
        HANDLE unmapped, release;
        std::thread unmapper, releaser;
        int calls = 0;
        DWORD first = 0, last = 0;
        bool opened = false;
    } c{&shim, base, {}, nullptr, nullptr};
    std::snprintf(c.hold, sizeof(c.hold), "x2w-run-break-during-fn-%lu", GetCurrentProcessId());
    char event_name[96];
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", c.hold);
    c.unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", c.hold);
    c.release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    shim.stable_call([](void* context) {
        auto& c = *static_cast<Context*>(context);
        if (++c.calls == 1) {
            SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", c.hold);
            c.unmapper = std::thread([&c] { c.shim->map(c.base + broken_page * page, 1, nullptr); });
            c.opened = WaitForSingleObject(c.unmapped, 10000) == WAIT_OBJECT_0;
            c.first = raw_state(c.base + neighbour * page);
            c.releaser = std::thread([&c] {  // stable_call waits for memory.mutex until the window is released
                Sleep(100);
                SetEvent(c.release);
            });
        }
        c.last = raw_state(c.base + neighbour * page);
    }, &c);
    c.unmapper.join();
    c.releaser.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(c.opened, "window_reached", name);
    check(c.first != MEM_COMMIT, "first_call_inside_the_window_not_committed", name);
    check(c.calls == 2, "fn_ran_again_after_the_break", name);
    check(c.last == MEM_COMMIT, "answer_after_the_break_committed", name);
    shim.map(base, broken_page, nullptr);
    shim.map(base + (broken_page + 1) * page, pages - broken_page - 1, nullptr);
    shim.release(pages, tokens);
    CloseHandle(c.unmapped);
    CloseHandle(c.release);
}

// Kernel I/O into or out of a page of the run: this program's ReadFile and WriteFile imports (wrapped by xmem.dll
// when it loads with the switch). Synchronous calls on a file opened without FILE_FLAG_OVERLAPPED.
constexpr DWORD io_bytes = 4096;

struct TempFile {
    char path[MAX_PATH] = {};
    explicit TempFile(const char* stem)
    {
        char dir[MAX_PATH];
        GetTempPathA(sizeof(dir), dir);
        std::snprintf(path, sizeof(path), "%sx2w-run-break-%s-%lu.bin", dir, stem, GetCurrentProcessId());
    }
    ~TempFile() { DeleteFileA(path); }
    HANDLE open(DWORD access, DWORD disposition) const
    {
        return CreateFileA(path, access, 0, nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
};

// The file's bytes: the neighbour page's mark, then 0xa5.
void io_pattern(unsigned char* out)
{
    std::memset(out, 0xa5, io_bytes);
    const std::uint32_t mark = 0x51000000u + neighbour;
    std::memcpy(out, &mark, sizeof(mark));
}

bool read_file_into(unsigned char* target)
{
    TempFile file("read");
    unsigned char data[io_bytes];
    io_pattern(data);
    HANDLE h = file.open(GENERIC_WRITE, CREATE_ALWAYS);
    DWORD done = 0;
    const bool written = h != INVALID_HANDLE_VALUE && WriteFile(h, data, io_bytes, &done, nullptr) && done == io_bytes;
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    if (!written) return false;
    h = file.open(GENERIC_READ, OPEN_EXISTING);
    if (h == INVALID_HANDLE_VALUE) return false;
    done = 0;
    const BOOL ok = ReadFile(h, target, io_bytes, &done, nullptr);
    const DWORD error = ok ? 0 : GetLastError();
    LARGE_INTEGER position{};
    SetFilePointerEx(h, LARGE_INTEGER{}, &position, FILE_CURRENT);
    CloseHandle(h);
    std::printf("ReadFile into the run: ok=%d bytes=%lu error=%lu position=%lld\n", ok, done, error, position.QuadPart);
    return ok && done == io_bytes && std::memcmp(target, data, io_bytes) == 0;
}

bool write_file_from(unsigned char* target)
{
    TempFile file("write");
    HANDLE h = file.open(GENERIC_WRITE, CREATE_ALWAYS);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD done = 0;
    const BOOL ok = WriteFile(h, target, io_bytes, &done, nullptr);
    const DWORD error = ok ? 0 : GetLastError();
    CloseHandle(h);
    std::printf("WriteFile from the run: ok=%d bytes=%lu error=%lu\n", ok, done, error);
    if (!ok || done != io_bytes) return false;
    unsigned char data[io_bytes];
    h = file.open(GENERIC_READ, OPEN_EXISTING);
    const bool read = h != INVALID_HANDLE_VALUE && ReadFile(h, data, io_bytes, &done, nullptr) && done == io_bytes;
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    return read && std::memcmp(data, target, io_bytes) == 0;
}

// An anonymous pipe with two 4096-byte messages queued, read into the page inside the window. The wrapper runs a
// failed call again only on a disk file, so this one fails inside the window instead of waiting for the break. Its
// error must be one the wrapper retries on a disk file, or answering at once would show nothing. Measured under Wine
// (Proton Experimental, 2026-11): ERROR_NOACCESS inside the window, both messages still queued. A pipe read into a
// mapped page of a run fails there even outside a break (ERROR_INVALID_USER_BUFFER), while one into a VirtualAlloc
// buffer reads the first message, so a retry after the break would not have read it either.
struct PipeRead {
    BOOL ok = FALSE;
    DWORD bytes = 0, error = 0, queued_before = 0, queued_after = 0;
    bool second_message = false;  // the bytes read are the second message's
} pipe_read;

bool read_pipe_into(unsigned char* target)
{
    HANDLE r = nullptr, w = nullptr;
    if (!CreatePipe(&r, &w, nullptr, 16 * io_bytes)) return false;
    unsigned char first[io_bytes], second[io_bytes];
    io_pattern(first);
    std::memset(second, 0x5a, io_bytes);
    DWORD done = 0;
    const bool queued = WriteFile(w, first, io_bytes, &done, nullptr) && done == io_bytes &&
                        WriteFile(w, second, io_bytes, &done, nullptr) && done == io_bytes;
    PipeRead& p = pipe_read;
    PeekNamedPipe(r, nullptr, 0, nullptr, &p.queued_before, nullptr);
    p.ok = queued && ReadFile(r, target, io_bytes, &p.bytes, nullptr);
    p.error = p.ok ? 0 : GetLastError();
    PeekNamedPipe(r, nullptr, 0, nullptr, &p.queued_after, nullptr);
    p.second_message = p.ok && std::memcmp(target, second, io_bytes) == 0;
    CloseHandle(r);
    CloseHandle(w);
    std::printf("ReadFile from a pipe into the run: ok=%d bytes=%lu error=%lu queued before %lu after %lu, second "
                "message %d\n", p.ok, p.bytes, p.error, p.queued_before, p.queued_after, p.second_message);
    return p.ok && p.bytes == io_bytes && std::memcmp(target, first, io_bytes) == 0;
}

// A read into a reserved page that is in no run fails at once, as without the wrapper: nothing waits for it.
void io_outside_runs()
{
    const char* const name = "io_outside_runs";
    auto* const reserved = static_cast<unsigned char*>(VirtualAlloc(nullptr, page, MEM_RESERVE, PAGE_NOACCESS));
    if (!check(reserved != nullptr, "reserve", name)) return;
    TempFile file("outside");
    unsigned char data[io_bytes];
    io_pattern(data);
    HANDLE h = file.open(GENERIC_WRITE | GENERIC_READ, CREATE_ALWAYS);
    DWORD done = 0;
    if (!check(h != INVALID_HANDLE_VALUE && WriteFile(h, data, io_bytes, &done, nullptr) &&
               SetFilePointer(h, 0, nullptr, FILE_BEGIN) == 0, "temp_file", name)) {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        VirtualFree(reserved, 0, MEM_RELEASE);
        return;
    }
    const ULONGLONG from = GetTickCount64();
    const BOOL ok = ReadFile(h, reserved, io_bytes, &done, nullptr);
    const DWORD error = GetLastError();
    const ULONGLONG took = GetTickCount64() - from;
    CloseHandle(h);
    VirtualFree(reserved, 0, MEM_RELEASE);
    std::printf("ReadFile into a reserved page outside the runs: ok=%d error=%lu %llu ms\n", ok, error, took);
    check(!ok && error == ERROR_NOACCESS, "read_into_reserved_page_fails_with_ERROR_NOACCESS", name);
    check(took < 1000, "read_into_reserved_page_fails_at_once", name);
}

// Default mode: eight single-page views; the unmap of page 5 is held open while readers query page 2.
void default_case(const Shim& shim, unsigned char* base)
{
    const char* const name = "default";
    SIZE_T count = pages, tokens[pages] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages || shim.map(base, pages, tokens) != base) {
        check(false, "map", name);
        return;
    }
    unsigned char* const target = base + neighbour * page;
    MEMORY_BASIC_INFORMATION info{};
    check(VirtualQuery(target, &info, sizeof(info)) && info.RegionSize == page, "one_view_per_page", name);
    char hold[64], event_name[96];
    std::snprintf(hold, sizeof(hold), "x2w-run-break-default-%lu", GetCurrentProcessId());
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", hold);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", hold);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold);
    std::thread unmapper([&] { shim.map(base + broken_page * page, 1, nullptr); });
    check(WaitForSingleObject(unmapped, 10000) == WAIT_OBJECT_0, "window_reached", name);
    std::atomic<int> answered{0};
    DWORD stable = 0, xmem = 0;
    bool writable = false;
    std::thread reading([&] {
        struct Query { unsigned char* target; DWORD state; } query{target, 0};
        shim.stable_call([](void* context) {
            auto& q = *static_cast<Query*>(context);
            q.state = raw_state(q.target);
        }, &query);
        stable = query.state;
        unsigned char buffer[64] = {};
        if (shim.query(GetCurrentProcess(), target, buffer)) std::memcpy(&xmem, buffer + 0x20, sizeof(xmem));
        writable = x2w_local::writable(target, 64);
        answered = 1;
    });
    for (int i = 0; i < 300 && !answered.load(); ++i) Sleep(10);
    check(answered.load() == 1, "readers_answer_inside_the_unmap", name);
    check(stable == MEM_COMMIT && xmem == MEM_COMMIT && writable, "answers_committed", name);
    SetEvent(release);
    reading.join();
    unmapper.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    shim.map(base, broken_page, nullptr);
    shim.map(base + (broken_page + 1) * page, pages - broken_page - 1, nullptr);
    shim.release(pages, tokens);
    CloseHandle(unmapped);
    CloseHandle(release);
}

// Lazy commit: the output buffers lie in a mapped page whose slot is not committed until it is touched.
void lazy_case(const Shim& shim, unsigned char* base)
{
    const char* const name = "lazy";
    SIZE_T count = pages, tokens[pages] = {}, out_count = 1, out_token = 0;
    unsigned char* const out = base + 32 * page;
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages || shim.map(base, pages, tokens) != base ||
        !shim.allocate(0x20400000, &out_count, &out_token) || shim.map(out, 1, &out_token) != out) {
        check(false, "map", name);
        return;
    }
    for (SIZE_T i = 0; i < pages; ++i) *reinterpret_cast<std::uint32_t*>(base + i * page) = 0x53000000u + static_cast<std::uint32_t>(i);
    if (!check(raw_state(out) != MEM_COMMIT, "output_slot_not_committed_yet", name)) return;
    char hold[64], event_name[96];
    std::snprintf(hold, sizeof(hold), "x2w-run-break-lazy-%lu", GetCurrentProcessId());
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", hold);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", hold);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold);
    std::thread unmapper([&] { shim.map(base + broken_page * page, 1, nullptr); });
    check(WaitForSingleObject(unmapped, 10000) == WAIT_OBJECT_0, "window_reached", name);
    unsigned char* const target = base + neighbour * page;
    BOOL xmem_ok = FALSE;
    SIZE_T ex_bytes = 0;
    auto* const ex_info = reinterpret_cast<MEMORY_BASIC_INFORMATION*>(out + 0x100);
    std::atomic<int> answered{0};
    std::thread xmem_reader([&] {
        xmem_ok = shim.query(GetCurrentProcess(), target, out);
        ++answered;
    });
    std::thread ex_reader([&] {
        ex_bytes = VirtualQueryEx(GetCurrentProcess(), target, ex_info, sizeof(*ex_info));
        ++answered;
    });
    Sleep(300);
    check(answered.load() == 0, "readers_wait_for_the_break", name);
    SetEvent(release);
    xmem_reader.join();
    ex_reader.join();
    unmapper.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    // The answers are the state after the break, read back from the slot they were written to. Measured under Wine: a
    // slot committed lazily (through the slab) still reads as reserved in the title's view, and so does a page of the
    // view mapped again after the break (its bytes are kept), so compare with a raw query taken now.
    const DWORD after = raw_state(target);
    DWORD xmem_state = 0;
    std::memcpy(&xmem_state, out + 0x20, sizeof(xmem_state));
    std::printf("lazy: target 0x%lx after the break; XMemVirtualQuery %d 0x%lx; VirtualQueryEx %zu 0x%lx\n",
                static_cast<unsigned long>(after), xmem_ok, static_cast<unsigned long>(xmem_state),
                static_cast<std::size_t>(ex_bytes), static_cast<unsigned long>(ex_info->State));
    check(xmem_ok && xmem_state == after, "XMemVirtualQuery_answer_after_the_break_in_the_lazy_slot", name);
    check(ex_bytes == sizeof(*ex_info) && ex_info->State == after, "VirtualQueryEx_answer_after_the_break_in_the_lazy_slot", name);
    check(*reinterpret_cast<std::uint32_t*>(target) == 0x53000000u + neighbour, "neighbour_data_kept", name);
    shim.map(out, 1, nullptr);
    shim.release(1, &out_token);
    shim.map(base, broken_page, nullptr);
    shim.map(base + (broken_page + 1) * page, pages - broken_page - 1, nullptr);
    shim.release(pages, tokens);
    CloseHandle(unmapped);
    CloseHandle(release);
}

// Lazy commit and shared views: the ReadFile import goes through both wrappers. A read into a mapped slot that is not
// committed yet (no break around it) commits the slot first and succeeds, as with lazy commit alone.
void lazy_read_file(const Shim& shim, unsigned char* at)
{
    const char* const name = "lazy_read_file";
    SIZE_T count = 1, token = 0;
    if (!check(shim.allocate(0x20400000, &count, &token) && shim.map(at, 1, &token) == at, "map", name)) return;
    if (check(raw_state(at) != MEM_COMMIT, "slot_not_committed_yet", name)) {
        TempFile file("lazy");
        unsigned char data[io_bytes];
        io_pattern(data);
        HANDLE h = file.open(GENERIC_WRITE | GENERIC_READ, CREATE_ALWAYS);
        DWORD done = 0;
        const bool ready = h != INVALID_HANDLE_VALUE && WriteFile(h, data, io_bytes, &done, nullptr) &&
                           SetFilePointer(h, 0, nullptr, FILE_BEGIN) == 0;
        const BOOL ok = ready && ReadFile(h, at, io_bytes, &done, nullptr);
        const DWORD error = ok ? 0 : GetLastError();
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        std::printf("lazy: ReadFile into an uncommitted mapped slot: ok=%d error=%lu\n", ok, error);
        check(ok && done == io_bytes && std::memcmp(at, data, io_bytes) == 0, "read_into_the_slot", name);
    }
    shim.map(at, 1, nullptr);
    shim.release(1, &token);
}

struct Events {
    char name[64];
    HANDLE first, second;
    Events(const char* stem, const char* a, const char* b)
    {
        std::snprintf(name, sizeof(name), "x2w-run-break-%s-%lu", stem, GetCurrentProcessId());
        char event_name[96];
        std::snprintf(event_name, sizeof(event_name), "%s.%s", name, a);
        first = CreateEventA(nullptr, TRUE, FALSE, event_name);
        std::snprintf(event_name, sizeof(event_name), "%s.%s", name, b);
        second = CreateEventA(nullptr, TRUE, FALSE, event_name);
    }
    ~Events()
    {
        CloseHandle(first);
        CloseHandle(second);
    }
};

// Break A: unmap page 7 of a run of eight; break B: unmap page 6 of what is left, [0, 7). The reader's page 2 is in
// both. hold_retry: the handler for the fault in break A looks the page up only after A has ended.
void next_break_case(const Shim& shim, unsigned char* base, bool hold_retry)
{
    const char* const name = hold_retry ? "next_break" : "next_break_control";
    SIZE_T count = pages, tokens[pages] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages || shim.map(base, pages, tokens) != base) {
        check(false, "map_run", name);
        return;
    }
    for (SIZE_T i = 0; i < pages; ++i) *reinterpret_cast<std::uint32_t*>(base + i * page) = 0x52000000u + static_cast<std::uint32_t>(i);
    unsigned char* const target = base + 2 * page;
    Events window_a("a", "unmapped", "release"), window_b("b", "unmapped", "release"), retry("retry", "faulted", "go");
    HANDLE second_read = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    std::atomic<int> reads{0};
    std::uint32_t first_value = 0, second_value = 0;

    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", window_a.name);
    std::thread unmap_a([&] { shim.map(base + 7 * page, 1, nullptr); });
    const bool a_open = WaitForSingleObject(window_a.first, 10000) == WAIT_OBJECT_0;
    check(a_open, "break_a_held", name);
    if (hold_retry) SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", retry.name);
    std::thread reader([&] {
        first_value = *reinterpret_cast<volatile std::uint32_t*>(target);  // faults in break A
        ++reads;
        WaitForSingleObject(second_read, 10000);
        second_value = *reinterpret_cast<volatile std::uint32_t*>(target);  // faults in break B
        ++reads;
    });
    if (hold_retry) {
        check(WaitForSingleObject(retry.first, 10000) == WAIT_OBJECT_0, "handler_held_before_reading_the_break", name);
        SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", nullptr);
    } else {
        Sleep(100);  // the handler sees break A in progress and waits for memory.mutex
    }
    SetEvent(window_a.second);
    unmap_a.join();
    check(raw_state(target) == MEM_COMMIT, "break_a_ended", name);
    if (hold_retry) SetEvent(retry.second);  // the handler now looks the page up after break A
    for (int i = 0; i < 200 && reads.load() < 1; ++i) Sleep(10);
    check(reads.load() == 1 && first_value == 0x52000002u, "first_fault_retried", name);

    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", window_b.name);
    std::thread unmap_b([&] { shim.map(base + 6 * page, 1, nullptr); });
    const bool b_open = WaitForSingleObject(window_b.first, 10000) == WAIT_OBJECT_0;
    check(b_open && raw_state(target) != MEM_COMMIT, "break_b_held_page_not_committed", name);
    SetEvent(second_read);
    Sleep(300);
    check(reads.load() == 1, "second_fault_waits_for_break_b", name);
    SetEvent(window_b.second);
    unmap_b.join();
    reader.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(reads.load() == 2 && second_value == 0x52000002u, "second_fault_retried", name);
    shim.map(base, 6, nullptr);
    shim.release(pages, tokens);
    CloseHandle(second_read);
    if (!hold_retry) std::printf("next-break control: the handler ran while break A was open\n");
}
// Break A takes apart run 1, break B (begun after A ended) run 2, a disjoint run. The handler for a fault in A is held
// (X2W_XMEM_TEST_HOLD_RETRY) until B is held open, so it looks the page up while B is in progress: it must still find
// the stamp of A on run 1 and retry the fault. A handler that knew only the last break's ranges passes the fault on,
// and the process ends with an access violation.
void two_runs_case(const Shim& shim, unsigned char* base)
{
    const char* const name = "two_runs";
    unsigned char* const run1 = base;
    unsigned char* const run2 = base + 16 * page;
    SIZE_T count = 2 * pages, tokens[2 * pages] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != 2 * pages || shim.map(run1, pages, tokens) != run1 ||
        shim.map(run2, pages, tokens + pages) != run2) {
        check(false, "map_runs", name);
        return;
    }
    for (SIZE_T i = 0; i < pages; ++i) {
        *reinterpret_cast<std::uint32_t*>(run1 + i * page) = 0x54000000u + static_cast<std::uint32_t>(i);
        *reinterpret_cast<std::uint32_t*>(run2 + i * page) = 0x54100000u + static_cast<std::uint32_t>(i);
    }
    unsigned char* const target = run1 + 2 * page;
    Events window_a("two-a", "unmapped", "release"), window_b("two-b", "unmapped", "release"), retry("two-retry", "faulted", "go");
    std::atomic<int> reads{0};
    std::uint32_t value = 0;

    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", window_a.name);
    std::thread unmap_a([&] { shim.map(run1 + 7 * page, 1, nullptr); });
    check(WaitForSingleObject(window_a.first, 10000) == WAIT_OBJECT_0 && raw_state(target) != MEM_COMMIT,
          "break_a_held_on_run_1", name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", retry.name);
    std::thread reader([&] {
        value = *reinterpret_cast<volatile std::uint32_t*>(target);  // faults in break A
        ++reads;
    });
    check(WaitForSingleObject(retry.first, 10000) == WAIT_OBJECT_0, "handler_held_before_reading_the_breaks", name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", nullptr);
    SetEvent(window_a.second);
    unmap_a.join();
    check(raw_state(target) == MEM_COMMIT, "break_a_ended", name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", window_b.name);
    std::thread unmap_b([&] { shim.map(run2 + 7 * page, 1, nullptr); });
    check(WaitForSingleObject(window_b.first, 10000) == WAIT_OBJECT_0 && raw_state(run2 + 2 * page) != MEM_COMMIT,
          "break_b_held_on_run_2", name);
    SetEvent(retry.second);  // the handler looks the page up now, with B in progress
    Sleep(100);
    SetEvent(window_b.second);
    unmap_b.join();
    reader.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(reads.load() == 1 && value == 0x54000002u, "fault_in_break_a_retried_during_break_b", name);
    check(*reinterpret_cast<std::uint32_t*>(run2 + 2 * page) == 0x54100002u, "run_2_data_kept", name);
    shim.map(run1, 7, nullptr);
    shim.map(run2, 7, nullptr);
    shim.release(2 * pages, tokens);
}

// The owner's configuration (X2W_XMEM_PERMISSIVE=1, X2W_XMEM_GPU_MIRROR=1, X2W_XMEM_JOIN_CALLS=1 from main): one
// XMemMapPhysicalPages call joins the CPU run, then the same pages' GPU mirror run, under one memory.mutex hold; the
// two breaks follow each other at once. A fault on the CPU run in the first break, whose handler is held until the GPU
// break is held open, must be retried. The release event is auto-reset, so each of the call's two holds waits for a
// release of its own.
void mirror_case(const Shim& shim, unsigned char* cpu)
{
    const char* const name = "mirror";
    SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_XMEM_GPU_MIRROR", "1");
    auto* const gpu_view = static_cast<unsigned char*>(shim.virtual_alloc(nullptr, SIZE_T{16} << 30, 0x20402000, 3, 0x800 | PAGE_NOACCESS));
    SetEnvironmentVariableA("X2W_XMEM_GPU_MIRROR", nullptr);
    unsigned char* const at = cpu + page;
    SIZE_T count = pages + 1, tokens[pages + 1] = {};
    if (!check(gpu_view != nullptr, "gpu_reservation", name) ||
        !check(shim.allocate(0x280, &count, tokens) && count == pages + 1 && shim.map(at, pages, tokens) == at, "map_run", name)) {
        SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", nullptr);
        if (gpu_view) VirtualFree(gpu_view, 0, MEM_RELEASE);
        return;
    }
    unsigned char* const twin = at - cpu + gpu_view;
    for (SIZE_T i = 0; i < pages; ++i) *reinterpret_cast<std::uint32_t*>(at + i * page) = 0x55000000u + static_cast<std::uint32_t>(i);
    check(*reinterpret_cast<std::uint32_t*>(twin + 2 * page) == 0x55000002u, "gpu_twin_shows_the_cpu_page", name);
    unsigned char* const target = at + 2 * page;
    char hold[64], event_name[96];
    std::snprintf(hold, sizeof(hold), "x2w-run-break-mirror-%lu", GetCurrentProcessId());
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", hold);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", hold);
    HANDLE release = CreateEventA(nullptr, FALSE, FALSE, event_name);
    Events retry("mirror-retry", "faulted", "go");
    std::atomic<int> reads{0};
    std::uint32_t value = 0;
    bool mapped = false;
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold);
    std::thread mapper([&] { mapped = shim.map(at + pages * page, 1, &tokens[pages]) == at + pages * page; });
    check(WaitForSingleObject(unmapped, 10000) == WAIT_OBJECT_0 && raw_state(target) != MEM_COMMIT, "cpu_merge_held", name);
    ResetEvent(unmapped);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", retry.name);
    std::thread reader([&] {
        value = *reinterpret_cast<volatile std::uint32_t*>(target);  // faults in the CPU merge's break
        ++reads;
    });
    check(WaitForSingleObject(retry.first, 10000) == WAIT_OBJECT_0, "handler_held_before_reading_the_breaks", name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", nullptr);
    SetEvent(release);  // the CPU merge ends; the GPU merge begins and holds
    check(WaitForSingleObject(unmapped, 10000) == WAIT_OBJECT_0 && raw_state(twin + 2 * page) != MEM_COMMIT &&
          raw_state(target) == MEM_COMMIT, "gpu_merge_held_after_the_cpu_merge", name);
    SetEvent(retry.second);  // the handler looks the page up now, with the GPU merge in progress
    Sleep(100);
    SetEvent(release);
    mapper.join();
    reader.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(mapped, "mapping_call_succeeded", name);
    check(reads.load() == 1 && value == 0x55000002u, "fault_in_the_cpu_merge_retried_during_the_gpu_merge", name);
    MEMORY_BASIC_INFORMATION info{};
    check(VirtualQuery(at, &info, sizeof(info)) && info.RegionSize == (pages + 1) * page &&
          VirtualQuery(twin, &info, sizeof(info)) && info.RegionSize == (pages + 1) * page, "both_views_joined", name);
    shim.map(at, pages + 1, nullptr);
    shim.release(pages + 1, tokens);
    SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", nullptr);
    VirtualFree(gpu_view, 0, MEM_RELEASE);
    CloseHandle(unmapped);
    CloseHandle(release);
}

// Reads *address into value; false when the access violation reaches this frame (the handler passed it on). No C++
// objects in the __try frame, and not inlined: inlined into the reader's lambda (clang-cl, under Wine), the code after
// the __except wrote through a null pointer.
__declspec(noinline) bool guarded_read(const volatile std::uint32_t* address, std::uint32_t* value)
{
    __try {
        *value = *address;
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// Break A takes apart run 1; a fault in it is held before its handler looks the page up (X2W_XMEM_TEST_HOLD_RETRY).
// A ends, then `later` breaks follow, each the unmap of a lone page mapped by a call of its own (one break per unmap
// call; the lone page joins no run). Run 1's views keep the stamp of break A, which mapped them again, however many
// breaks follow, so the fault is retried and the reader reads its page. A fault passed on would reach the reader's
// own __except.
void later_breaks_case(const Shim& shim, unsigned char* run1, SIZE_T later)
{
    char name[32], stem[32];
    std::snprintf(name, sizeof(name), "later_breaks_%zu", later);
    std::snprintf(stem, sizeof(stem), "later-%zu", later);
    unsigned char* const lone = run1 + 32 * page;
    SIZE_T count = pages + 1, tokens[pages + 1] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages + 1 || shim.map(run1, pages, tokens) != run1) {
        check(false, "map_run", name);
        return;
    }
    for (SIZE_T i = 0; i < pages; ++i) *reinterpret_cast<std::uint32_t*>(run1 + i * page) = 0x56000000u + static_cast<std::uint32_t>(i);
    unsigned char* const target = run1 + 2 * page;
    Events window(stem, "unmapped", "release"), retry(stem, "faulted", "go");
    std::atomic<int> outcome{0};  // 0: the reader has not finished; 1: it read its page; 2: the fault was passed on
    std::uint32_t value = 0;

    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", window.name);
    std::thread unmap_a([&] { shim.map(run1 + 7 * page, 1, nullptr); });
    check(WaitForSingleObject(window.first, 10000) == WAIT_OBJECT_0 && raw_state(target) != MEM_COMMIT, "break_a_held", name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", retry.name);
    std::thread reader([&] { outcome = guarded_read(reinterpret_cast<volatile std::uint32_t*>(target), &value) ? 1 : 2; });
    check(WaitForSingleObject(retry.first, 10000) == WAIT_OBJECT_0, "handler_held_before_reading_the_breaks", name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", nullptr);
    SetEvent(window.second);
    unmap_a.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(raw_state(target) == MEM_COMMIT, "break_a_ended", name);
    // The handler waits at most five seconds for "go" (hold_retry): the later breaks must all end inside that wait.
    const ULONGLONG started = GetTickCount64();
    SIZE_T made = 0;
    for (SIZE_T k = 0; k < later; ++k) {
        if (shim.map(lone, 1, &tokens[pages]) != lone || raw_state(lone) != MEM_COMMIT) break;
        if (shim.map(lone, 1, nullptr) != lone || raw_state(lone) == MEM_COMMIT) break;
        ++made;
    }
    const ULONGLONG elapsed = GetTickCount64() - started;
    std::printf("%s: %zu later breaks in %llu ms\n", name, made, static_cast<unsigned long long>(elapsed));
    check(made == later, "later_breaks_made", name);
    check(outcome.load() == 0 && elapsed < 4000, "handler_still_held_after_the_later_breaks", name);
    SetEvent(retry.second);  // the handler looks the page up now
    reader.join();
    check(outcome.load() == 1 && value == 0x56000002u, "fault_retried_after_the_later_breaks", name);
    shim.map(run1, 7, nullptr);
    shim.release(pages + 1, tokens);
}

std::atomic<int> first_chance_faults{0};
ULONG_PTR counted_page = 0;

// Added after xmem's handler with FirstHandler=1, so it sees each access violation first, also one xmem retries.
LONG CALLBACK count_fault(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* record = e->ExceptionRecord;
    if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2 &&
        (record->ExceptionInformation[1] & ~(ULONG_PTR{page} - 1)) == counted_page)
        ++first_chance_faults;
    return EXCEPTION_CONTINUE_SEARCH;
}

// A fault that stays: page 2 of a run made no-access, read once by a new thread. On a run no break has mapped (stamp
// 0) the handler passes the fault on at once: one fault. After a break maps the run again (the unmap of page 7), it
// retries once and then passes it on: two faults. Both reach the reader's __except.
void persistent_fault_case(const Shim& shim, unsigned char* run1)
{
    const char* const name = "persistent_fault";
    SIZE_T count = pages, tokens[pages] = {};
    if (!shim.allocate(0x20400000, &count, tokens) || count != pages || shim.map(run1, pages, tokens) != run1) {
        check(false, "map_run", name);
        return;
    }
    unsigned char* const target = run1 + 2 * page;
    const auto faults = [&](const char* step) {
        DWORD old = 0;
        if (!check(VirtualProtect(target, page, PAGE_NOACCESS, &old) != FALSE, step, name)) return -1;
        counted_page = reinterpret_cast<ULONG_PTR>(target);
        first_chance_faults = 0;
        void* const counter = AddVectoredExceptionHandler(1, count_fault);
        std::uint32_t value = 0;
        bool read = true;
        std::thread reader([&] { read = guarded_read(reinterpret_cast<volatile std::uint32_t*>(target), &value); });
        reader.join();
        if (counter) RemoveVectoredExceptionHandler(counter);
        VirtualProtect(target, page, PAGE_READWRITE, &old);
        const int counted = first_chance_faults.load();
        std::printf("%s: %s -> read %d, first-chance faults %d\n", name, step, read, counted);
        return counter && !read ? counted : -1;
    };
    check(faults("no_break_mapped_the_run") == 1, "passed_on_at_once_without_a_stamp", name);
    check(shim.map(run1 + 7 * page, 1, nullptr) == run1 + 7 * page, "unmap_page_7", name);
    check(faults("a_break_mapped_the_run") == 2, "retried_once_with_a_stamp", name);
    shim.map(run1, 7, nullptr);
    shim.release(pages, tokens);
}
}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a case that ends the process with an access violation keeps its lines
    const bool control = argc > 2 && std::strcmp(argv[2], "control") == 0;
    const bool next_break_control = argc > 2 && std::strcmp(argv[2], "next-break-control") == 0;
    const bool default_mode = argc > 2 && std::strcmp(argv[2], "default") == 0;
    const bool lazy = argc > 2 && std::strcmp(argv[2], "lazy") == 0;
    const bool two_runs = argc > 2 && std::strcmp(argv[2], "two-runs") == 0;
    const bool mirror = argc > 2 && std::strcmp(argv[2], "mirror") == 0;
    const bool stamps = argc > 2 && std::strcmp(argv[2], "stamps") == 0;
    if (argc < 2 || (argc > 2 && !control && !next_break_control && !default_mode && !lazy && !two_runs && !mirror &&
                     !stamps)) {
        std::printf("usage: xmem_run_break_readers_check <test-controls xmem.dll> "
                    "[control|next-break-control|default|lazy|two-runs|mirror|stamps]\n");
        return 2;
    }
    // The mode is read by the first mapping call; the VirtualQueryEx import is wrapped when xmem.dll loads.
    SetEnvironmentVariableA("X2W_XMEM_SHARED_VIEWS", default_mode ? nullptr : "1");
    SetEnvironmentVariableA("X2W_XMEM_JOIN_CALLS", mirror ? "1" : nullptr);  // only `mirror` joins across calls
    if (lazy) SetEnvironmentVariableA("X2W_XMEM_LAZY_COMMIT", "1");
    const HMODULE xmem = LoadLibraryA(argv[1]);
    if (!xmem) {
        std::printf("check_failed: load (error %lu)\n", GetLastError());
        return 1;
    }
    Shim shim{reinterpret_cast<XMemVirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc")),
              reinterpret_cast<XMemAllocatePhysicalPagesFn>(GetProcAddress(xmem, "XMemAllocatePhysicalPages")),
              reinterpret_cast<XMemMapPhysicalPagesFn>(GetProcAddress(xmem, "XMemMapPhysicalPages")),
              reinterpret_cast<XMemFreePhysicalPagesFn>(GetProcAddress(xmem, "XMemFreePhysicalPages")),
              reinterpret_cast<XMemVirtualQueryFn>(GetProcAddress(xmem, "XMemVirtualQuery")),
              reinterpret_cast<StableCallFn>(GetProcAddress(xmem, "x2w_xmem_stable_call"))};
    if (!shim.virtual_alloc || !shim.allocate || !shim.map || !shim.release || !shim.query || !shim.stable_call) {
        std::printf("check_failed: exports\n");
        return 1;
    }
    auto* const base = static_cast<unsigned char*>(shim.virtual_alloc(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    if (!base) {
        std::printf("check_failed: startup reservation (error %lu)\n", GetLastError());
        return 1;
    }
    unsigned char* const at = base + page;
    if (next_break_control || default_mode || lazy || two_runs || mirror || stamps) {
        if (next_break_control) next_break_case(shim, at, false);
        if (default_mode) default_case(shim, at);
        if (lazy) {
            lazy_case(shim, at);
            lazy_read_file(shim, at + 40 * page);
        }
        if (two_runs) two_runs_case(shim, at);
        if (mirror) mirror_case(shim, base);
        if (stamps) {
            later_breaks_case(shim, at, 63);
            later_breaks_case(shim, at + 64 * page, 64);
            later_breaks_case(shim, at + 128 * page, 200);
            persistent_fault_case(shim, at + 192 * page);
        }
        VirtualFree(base, 0, MEM_RELEASE);
        // The default mode's pass line, at the end and only with no failed check (ctest matches it).
        if (default_mode && !failures) std::printf("default mode: readers answer inside an unmap\n");
        std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
        return failures ? 1 : 0;
    }
    int index = 0;
    if (!control) {
        window_case(shim, at, index++, "x2w_xmem_stable_call", true, [&](unsigned char* target) {
            struct Query { unsigned char* target; DWORD state; } query{target, 0};
            shim.stable_call([](void* context) {
                auto& q = *static_cast<Query*>(context);
                q.state = raw_state(q.target);
            }, &query);
            return query.state == MEM_COMMIT;
        });
        window_case(shim, at, index++, "XMemVirtualQuery", true, [&](unsigned char* target) {
            unsigned char buffer[64] = {};
            DWORD state = 0;
            if (shim.query(GetCurrentProcess(), target, buffer)) std::memcpy(&state, buffer + 0x20, sizeof(state));
            return state == MEM_COMMIT;
        });
        window_case(shim, at, index++, "VirtualQueryEx_import", true, [&](unsigned char* target) {
            MEMORY_BASIC_INFORMATION info{};
            return VirtualQueryEx(GetCurrentProcess(), target, &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT;
        });
        window_case(shim, at, index++, "ReadFile_import", true, read_file_into);
        window_case(shim, at, index++, "WriteFile_import", true, write_file_from);
        window_case(shim, at, index++, "ReadFile_pipe", false, read_pipe_into);
        check(!pipe_read.ok && (pipe_read.error == ERROR_NOACCESS || pipe_read.error == ERROR_INVALID_USER_BUFFER),
              "fails_with_an_error_the_wrapper_retries_for_disk_files", "ReadFile_pipe");
        io_outside_runs();
    }
    // The header readers: waiting with the export, answering inside the window without it (control).
    const bool waits = !control;
    window_case(shim, at, index++, "x2w_local::writable", waits, [&](unsigned char* target) {
        return x2w_local::writable(target, 64);
    });
    window_case(shim, at, index++, "measure_placed_buffer", waits, [&](unsigned char* target) {
        x2w_startup::PlacedBufferRange range;
        return x2w_startup::measure_placed_buffer(target, 4096, range) && range.type == MEM_MAPPED && range.covered;
    });
    window_case(shim, at, index++, "snapshot_samplers", waits, [&](unsigned char* target) {
        const auto snapshot = x2w_startup::snapshot_samplers({reinterpret_cast<std::uintptr_t>(target), 4}, false);
        return snapshot.count == 4 && snapshot.readable.count() == 4 && snapshot.words[0][0] == 0x51000000u + neighbour;
    });
    if (!control) {
        next_break_case(shim, at, true);
        break_during_fn(shim, at);
    }
    VirtualFree(base, 0, MEM_RELEASE);
    std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
