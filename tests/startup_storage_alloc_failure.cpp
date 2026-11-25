// An allocation failure on the DirectStorage stand-in's queue worker fails that request and leaves
// the worker running (#1199, item 3).
//
// Request buffers are sized from request fields. The failure here is a real one, the same kind a
// job-wide commit cap causes in a startup run: this process joins a job object whose process-memory
// limit is a little above its current commit, then submits a zlib + BCPack request whose
// IntermediateSize (+0x34, the inflated size of such a request) is far above the limit. The stand-in's
// buffer allocation throws std::bad_alloc on the worker. A zlib request that carries the same number
// as its CancellationTag (+0x38) is submitted under the same limit and must succeed: the tag is not a
// size (public DSTORAGE_REQUEST layout).
// The stand-in is a DLL with its own C++ runtime, so replacing operator new here would not reach it;
// the commit limit is the one seam that works the same on the real worker.
//
// Checks, on one queue: a normal request succeeds; the oversized request completes with a failure
// status (within a timeout, destination unchanged); a normal request submitted afterwards succeeds
// with correct data. Returns 77 where the system does not enforce the job limit.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr int skipped = 77;
constexpr SIZE_T commit_headroom = SIZE_T{256} << 20;     // job limit above the current commit
constexpr unsigned oversized_inflated = 1536u << 20;       // declared IntermediateSize, above the limit
constexpr DWORD completion_timeout_ms = 15000;
// The submit threads start before commit runs out; a stack committed at creation is all they will have.
constexpr SIZE_T submit_thread_stack = SIZE_T{256} << 10;

void guard_this_thread();

struct Object { void** vtable; };
using GetFactoryFn = long(__stdcall*)(const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);

struct MockFence {
    void** vtable;
    bool signaled = false;
    u64 value = 0;
    HANDLE entered = nullptr;  // when set, signal() reports it was entered and then waits for `release`
    HANDLE release = nullptr;
    static long __stdcall signal(void* self, u64 val)
    {
        auto* f = static_cast<MockFence*>(self);
        if (f->entered) {
            SetEvent(f->entered);
            WaitForSingleObject(f->release, completion_timeout_ms);
        }
        f->signaled = true;
        f->value = val;
        return 0;
    }
};

// Calls one queue function on its own thread, started before commit is exhausted (a thread cannot be created
// afterwards). Waits for `go`, runs, then sets `done`.
struct QueueCall {
    Object* queue = nullptr;
    Object* status = nullptr;
    int enqueue_status_index = -1;  // when >= 0, the call first enqueues a status entry for that index
    HANDLE go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    static DWORD WINAPI run(void* parameter)
    {
        auto* self = static_cast<QueueCall*>(parameter);
        guard_this_thread();
        WaitForSingleObject(self->go, completion_timeout_ms);
        if (self->enqueue_status_index >= 0) {
            reinterpret_cast<EnqueueStatusFn>(self->queue->vtable[4])(
                self->queue, self->status, static_cast<unsigned>(self->enqueue_status_index));
        }
        reinterpret_cast<SubmitFn>(self->queue->vtable[6])(self->queue);
        SetEvent(self->done);
        return 0;
    }
};

std::uint32_t adler32(const unsigned char* data, std::size_t size)
{
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < size; ++i) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

// A zlib stream of one stored block (size at most 65535).
std::vector<unsigned char> stored_zlib(const std::vector<unsigned char>& plain)
{
    const auto size = static_cast<unsigned>(plain.size());
    std::vector<unsigned char> out{0x78, 0x01, 0x01, static_cast<unsigned char>(size), static_cast<unsigned char>(size >> 8),
                                   static_cast<unsigned char>(~size), static_cast<unsigned char>((~size) >> 8)};
    out.insert(out.end(), plain.begin(), plain.end());
    const std::uint32_t sum = adler32(plain.data(), plain.size());
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>(sum >> shift));
    return out;
}

struct Queue {
    Object* queue = nullptr;
    Object* status = nullptr;

    // Enqueues a memory-source zlib request into a linear destination and a status entry, submits,
    // and waits for the status. Returns false on a timeout; hr receives the status. A nonzero
    // intermediate_size also asks for BCPack (format 7) and is written as IntermediateSize.
    bool run(std::vector<unsigned char>& destination, const std::vector<unsigned char>& stream, unsigned index, long& hr,
             unsigned intermediate_size = 0, u64 cancellation_tag = 0) const
    {
        unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = 1 | 0x400000 | (intermediate_size ? 7u << 1 : 0u);
        *reinterpret_cast<unsigned char**>(request + 0x08) = destination.data();
        *reinterpret_cast<unsigned*>(request + 0x18) = static_cast<unsigned>(destination.size());
        *reinterpret_cast<const unsigned char**>(request + 0x20) = stream.data();
        *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(stream.size());
        *reinterpret_cast<unsigned*>(request + 0x34) = intermediate_size;
        *reinterpret_cast<u64*>(request + 0x38) = cancellation_tag;
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, index);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        const DWORD start = GetTickCount();
        while (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, index)) {
            if (GetTickCount() - start > completion_timeout_ms) return false;
            Sleep(5);
        }
        hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, index);
        return true;
    }
};

// Limits this process's commit to a little above what it holds now. False where that cannot be done.
bool cap_commit()
{
    PROCESS_MEMORY_COUNTERS counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) return false;
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) return false;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = counters.PeakPagefileUsage + commit_headroom;
    return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits) &&
           AssignProcessToJobObject(job, GetCurrentProcess());
}


// ---------------------------------------------------------------------------------------------------
// Diagnostics for crashes in this process. A failure in the low-commit path has been seen as an access
// violation or a stack overflow whose exception code and place were not recorded, so the fixture
// records both. Nothing here allocates and the buffers are static: when the process is out of commit a
// handler that allocates, or needs more stack than is already committed, would hide the original fault.
// SetThreadStackGuarantee reserves room on each thread for the filter to run after a stack overflow.
// The filter prints the code, the faulting address, and the unwound frames (module + offset) to stderr
// and ends the process with crash_exit_code. The checks below pass or fail without it.
constexpr DWORD crash_exit_code = 0xE1410;
constexpr ULONG filter_stack_guarantee = 64 * 1024;

char crash_line[256];
char crash_module[MAX_PATH];
CONTEXT crash_context;

std::size_t put_text(std::size_t at, const char* text)
{
    while (*text && at + 1 < sizeof crash_line) crash_line[at++] = *text++;
    return at;
}

std::size_t put_hex(std::size_t at, unsigned long long value)
{
    char digits[16];
    int count = 0;
    do {
        digits[count++] = "0123456789abcdef"[value & 0xF];
        value >>= 4;
    } while (value && count < 16);
    at = put_text(at, "0x");
    while (count && at + 1 < sizeof crash_line) crash_line[at++] = digits[--count];
    return at;
}

void write_crash_line(std::size_t length)
{
    DWORD written = 0;
    crash_line[length < sizeof crash_line ? length : sizeof crash_line - 1] = '\n';
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), crash_line, static_cast<DWORD>(length + 1), &written, nullptr);
}

std::size_t put_place(std::size_t at, unsigned long long pc)
{
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(static_cast<std::uintptr_t>(pc)), &module) &&
        GetModuleFileNameA(module, crash_module, MAX_PATH)) {
        const char* name = crash_module;
        for (const char* p = crash_module; *p; ++p)
            if (*p == '\\' || *p == '/') name = p + 1;
        at = put_text(at, name);
        return put_hex(put_text(at, "+"), pc - reinterpret_cast<std::uintptr_t>(module));
    }
    return put_text(at, "(no module)");
}

LONG WINAPI report_crash(EXCEPTION_POINTERS* info)
{
    const EXCEPTION_RECORD* record = info->ExceptionRecord;
    std::size_t at = put_text(0, "x2w-test crash: code=");
    at = put_hex(at, record->ExceptionCode);
    at = put_hex(put_text(at, " address="), reinterpret_cast<std::uintptr_t>(record->ExceptionAddress));
    if (record->NumberParameters >= 2 && record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        at = put_text(at, record->ExceptionInformation[0] == 1 ? " write-to=" : " read-from=");
        at = put_hex(at, record->ExceptionInformation[1]);
    }
    write_crash_line(at);
#if defined(_M_X64) || defined(__x86_64__)
    crash_context = *info->ContextRecord;
    for (int frame = 0; frame < 8; ++frame) {
        const unsigned long long pc = crash_context.Rip;
        at = put_hex(put_text(put_hex(put_text(0, "  frame "), static_cast<unsigned>(frame)), " rip="), pc);
        at = put_place(put_text(at, " "), pc);
        write_crash_line(at);
        DWORD64 image_base = 0;
        RUNTIME_FUNCTION* function = RtlLookupFunctionEntry(pc, &image_base, nullptr);
        if (!function) break;  // a leaf without unwind data: stop rather than read the stack blindly
        void* handler_data = nullptr;
        DWORD64 frame_pointer = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, pc, function, &crash_context, &handler_data, &frame_pointer,
                         nullptr);
        if (!crash_context.Rip) break;
    }
#endif
    TerminateProcess(GetCurrentProcess(), crash_exit_code);
    return EXCEPTION_EXECUTE_HANDLER;
}

void guard_this_thread()
{
    ULONG guarantee = filter_stack_guarantee;
    SetThreadStackGuarantee(&guarantee);
}

void install_crash_report()
{
    SetUnhandledExceptionFilter(report_crash);
    guard_this_thread();
}

// ---------------------------------------------------------------------------------------------------
// Deterministic worker-creation failure. The stand-in starts its queue worker with CreateThread (first
// with a committed stack, then once more with the default one). The fixture replaces that import in the
// loaded stand-in so that exactly the chosen calls fail with ERROR_COMMITMENT_LIMIT (1455), the code a
// system under commit pressure returns. No memory pressure is needed, so the same sequence runs on every
// machine and every build type.
constexpr DWORD commitment_limit_error = 1455;  // ERROR_COMMITMENT_LIMIT
constexpr long out_of_memory_hr = static_cast<long>(0x8007000E);
constexpr long pending_hr = static_cast<long>(0x8000000A);

using CreateThreadFn = HANDLE(WINAPI*)(LPSECURITY_ATTRIBUTES, SIZE_T, LPTHREAD_START_ROUTINE, LPVOID, DWORD, LPDWORD);
CreateThreadFn real_create_thread = nullptr;
volatile LONG fail_next_calls = 0;  // the next CreateThread calls that fail with 1455
volatile LONG create_thread_calls = 0;
volatile LONG create_thread_failures = 0;

HANDLE WINAPI faulting_create_thread(LPSECURITY_ATTRIBUTES attributes, SIZE_T stack, LPTHREAD_START_ROUTINE start,
                                     LPVOID parameter, DWORD flags, LPDWORD id)
{
    InterlockedIncrement(&create_thread_calls);
    for (LONG left = fail_next_calls; left > 0; left = fail_next_calls) {
        if (InterlockedCompareExchange(&fail_next_calls, left - 1, left) != left) continue;
        InterlockedIncrement(&create_thread_failures);
        SetLastError(commitment_limit_error);
        return nullptr;
    }
    return real_create_thread(attributes, stack, start, parameter, flags, id);
}

// Points the import slots of CreateThread in "module" at the replacement. Returns how many were changed.
int replace_import(HMODULE module, const char* function, void* replacement, void** original)
{
    auto* base = reinterpret_cast<unsigned char*>(module);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    int changed = 0;
    if (!directory.VirtualAddress) return 0;
    for (auto* entry = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress); entry->Name; ++entry) {
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + (entry->OriginalFirstThunk ? entry->OriginalFirstThunk : entry->FirstThunk));
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + entry->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            auto* name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(name->Name), function) != 0) continue;
            DWORD old = 0;
            if (!VirtualProtect(&slots->u1.Function, sizeof(void*), PAGE_READWRITE, &old)) return changed;
            if (!*original) *original = reinterpret_cast<void*>(slots->u1.Function);
            slots->u1.Function = reinterpret_cast<ULONG_PTR>(replacement);
            VirtualProtect(&slots->u1.Function, sizeof(void*), old, &old);
            ++changed;
        }
    }
    return changed;
}

// A fence that notes, when it is signaled, which of the first status slots were already complete. A
// status entry enqueued before a signal must be complete when the signal runs.
struct OrderedFence {
    void** vtable;
    Object* status;
    volatile LONG signaled;
    u64 value;
    bool complete_at_signal[4];
    long hr_at_signal[4];
    static volatile LONG order;
    LONG position;
    static long __stdcall signal(void* self, u64 value)
    {
        auto* fence = static_cast<OrderedFence*>(self);
        for (unsigned i = 0; i < 4; ++i) {
            fence->complete_at_signal[i] =
                reinterpret_cast<IsCompleteFn>(fence->status->vtable[3])(fence->status, i);
            fence->hr_at_signal[i] = reinterpret_cast<GetHResultFn>(fence->status->vtable[4])(fence->status, i);
        }
        fence->value = value;
        fence->position = InterlockedIncrement(&order);
        InterlockedExchange(&fence->signaled, 1);
        return 0;
    }
};
volatile LONG OrderedFence::order = 0;

void* ordered_fence_vtable[11] = {};

struct WorkerQueue {
    Object* queue = nullptr;
    Object* status = nullptr;

    bool create(Object* factory, const void* description, const char* name)
    {
        void* status_ptr = nullptr;
        void* queue_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 4, name, nullptr, &status_ptr) != 0 ||
            reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0)
            return false;
        status = static_cast<Object*>(status_ptr);
        queue = static_cast<Object*>(queue_ptr);
        return true;
    }
    void enqueue_request(std::vector<unsigned char>& destination, const std::vector<unsigned char>& stream)
    {
        unsigned char bytes[0x48] = {};
        *reinterpret_cast<u64*>(bytes + 0x00) = 1 | 0x400000;
        *reinterpret_cast<unsigned char**>(bytes + 0x08) = destination.data();
        *reinterpret_cast<unsigned*>(bytes + 0x18) = static_cast<unsigned>(destination.size());
        *reinterpret_cast<const unsigned char**>(bytes + 0x20) = stream.data();
        *reinterpret_cast<unsigned*>(bytes + 0x30) = static_cast<unsigned>(stream.size());
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, bytes);
    }
    void enqueue_status(unsigned index) { reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, index); }
    void enqueue_signal(OrderedFence& fence, u64 value)
    {
        fence.vtable = ordered_fence_vtable;
        fence.status = status;
        reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &fence, value);
    }
    void submit() { reinterpret_cast<SubmitFn>(queue->vtable[6])(queue); }
    bool complete(unsigned index) const { return reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, index); }
    long hresult(unsigned index) const { return reinterpret_cast<GetHResultFn>(status->vtable[4])(status, index); }
    bool wait_complete(unsigned index) const
    {
        const DWORD start = GetTickCount();
        while (!complete(index)) {
            if (GetTickCount() - start > completion_timeout_ms) return false;
            Sleep(2);
        }
        return true;
    }
};

bool wait_signaled(const OrderedFence& fence)
{
    const DWORD start = GetTickCount();
    while (!fence.signaled) {
        if (GetTickCount() - start > completion_timeout_ms) return false;
        Sleep(2);
    }
    return true;
}

bool injected_check(bool ok, const char* scenario, const char* what)
{
    if (!ok) std::fprintf(stderr, "worker creation failure [%s]: %s\n", scenario, what);
    return ok;
}

struct InjectedInputs {
    Object* factory;
    const void* description;
    const std::vector<unsigned char>* stream;
    const std::vector<unsigned char>* plain;
};

void arm_failures(long count)
{
    InterlockedExchange(&create_thread_calls, 0);
    InterlockedExchange(&create_thread_failures, 0);
    InterlockedExchange(&fail_next_calls, count);
}

// Exactly three Enqueue calls each find the worker missing (each tries both CreateThread variants, so six
// calls fail with 1455); the fourth attempt, at Submit, succeeds. Everything enqueued before runs in order.
bool scenario_three_failures_then_start(const InjectedInputs& in)
{
    const char* const name = "three failures then start";
    WorkerQueue q;
    if (!injected_check(q.create(in.factory, in.description, "Injected1"), name, "cannot create the queue")) return false;
    const auto& plain = *in.plain;
    std::vector<unsigned char> destination(plain.size(), 0xAA);
    OrderedFence fence{};
    arm_failures(6);
    q.enqueue_request(destination, *in.stream);
    q.enqueue_status(0);
    q.enqueue_signal(fence, 0xA1);
    bool ok = injected_check(create_thread_calls == 6 && create_thread_failures == 6, name,
                             "Enqueue did not try (and fail) exactly three times with two CreateThread calls each");
    ok = injected_check(!q.complete(0) && !fence.signaled, name, "an entry completed before Submit") && ok;
    q.submit();
    ok = injected_check(q.wait_complete(0) && wait_signaled(fence), name, "status or fence never completed") && ok;
    ok = injected_check(create_thread_calls == 7 && create_thread_failures == 6, name,
                        "the worker was not started by the first CreateThread call that was allowed to succeed") && ok;
    ok = injected_check(q.hresult(0) == 0 && destination == plain, name, "the request did not run after the worker started") && ok;
    ok = injected_check(fence.value == 0xA1 && fence.complete_at_signal[0], name, "the fence ran before its status entry") && ok;
    arm_failures(0);
    return ok;
}

// The committed-stack CreateThread fails with 1455 and the default-stack one succeeds: the queue runs.
bool scenario_default_stack_fallback(const InjectedInputs& in)
{
    const char* const name = "default stack fallback";
    WorkerQueue q;
    if (!injected_check(q.create(in.factory, in.description, "Injected2"), name, "cannot create the queue")) return false;
    const auto& plain = *in.plain;
    std::vector<unsigned char> destination(plain.size(), 0xAA);
    OrderedFence fence{};
    arm_failures(1);
    q.enqueue_request(destination, *in.stream);
    q.enqueue_status(0);
    q.enqueue_signal(fence, 0xB2);
    q.submit();
    bool ok = injected_check(q.wait_complete(0) && wait_signaled(fence), name, "status or fence never completed");
    ok = injected_check(create_thread_calls == 2 && create_thread_failures == 1, name, "unexpected CreateThread sequence") && ok;
    ok = injected_check(q.hresult(0) == 0 && destination == plain && fence.value == 0xB2, name, "the request did not run") && ok;
    arm_failures(0);
    return ok;
}

std::string read_text_file(const wchar_t* path)
{
    std::string text;
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char chunk[512];
    DWORD got = 0;
    while (ReadFile(file, chunk, sizeof chunk, &got, nullptr) && got) text.append(chunk, got);
    CloseHandle(file);
    return text;
}

bool scenario_worker_never_starts(const InjectedInputs& in)
{
    const char* const name = "worker never starts";
    WorkerQueue q;
    if (!injected_check(q.create(in.factory, in.description, "Injected3"), name, "cannot create the queue")) return false;
    const auto& plain = *in.plain;
    std::vector<unsigned char> first(plain.size(), 0xAA), second(plain.size(), 0xAA), third(plain.size(), 0xAA);
    OrderedFence fence_a{}, fence_b{}, fence_c{};
    arm_failures(100000);

    // Batch one: request, status, signal, request, status, signal.
    q.enqueue_request(first, *in.stream);
    q.enqueue_status(0);
    q.enqueue_signal(fence_a, 1);
    q.enqueue_request(second, *in.stream);
    q.enqueue_status(1);
    q.enqueue_signal(fence_b, 2);
    bool ok = injected_check(!q.complete(0) && !q.complete(1) && !fence_a.signaled && !fence_b.signaled, name,
                             "an entry completed before Submit");
    q.submit();
    ok = injected_check(q.complete(0) && q.complete(1) && fence_a.signaled && fence_b.signaled, name,
                        "Submit returned with a status entry or a fence left pending") && ok;
    ok = injected_check(q.hresult(0) == out_of_memory_hr && q.hresult(1) == out_of_memory_hr, name,
                        "a status entry did not report E_OUTOFMEMORY") && ok;
    ok = injected_check(fence_a.value == 1 && fence_b.value == 2 && fence_a.position < fence_b.position, name,
                        "the fences were not signaled in queue order") && ok;
    ok = injected_check(fence_a.complete_at_signal[0] && !fence_a.complete_at_signal[1] &&
                            fence_a.hr_at_signal[1] == pending_hr && fence_b.complete_at_signal[1],
                        name, "a fence ran before its status entry, or after the next one") && ok;
    ok = injected_check(first == std::vector<unsigned char>(plain.size(), 0xAA) &&
                            second == std::vector<unsigned char>(plain.size(), 0xAA),
                        name, "a failed request changed its destination") && ok;
    ok = injected_check(create_thread_calls > 0 && create_thread_calls == create_thread_failures, name,
                        "a thread was created while every CreateThread call was set to fail") && ok;

    // Batch two on the same queue, which still has no worker; then an empty Submit.
    q.enqueue_request(third, *in.stream);
    q.enqueue_status(2);
    q.enqueue_signal(fence_c, 3);
    q.submit();
    q.submit();
    ok = injected_check(q.complete(2) && q.hresult(2) == out_of_memory_hr && fence_c.signaled && fence_c.value == 3 &&
                            fence_c.complete_at_signal[2] && fence_c.position > fence_b.position,
                        name, "a later batch did not fail in order on the same queue") && ok;

    // Threads can be created again: the same queue starts its worker and serves a request.
    arm_failures(0);
    std::vector<unsigned char> recovered(plain.size(), 0xAA);
    q.enqueue_request(recovered, *in.stream);
    q.enqueue_status(3);
    q.submit();
    ok = injected_check(q.wait_complete(3) && q.hresult(3) == 0 && recovered == plain, name,
                        "the queue did not recover once CreateThread worked again") && ok;
    return ok;
}

// Runs every injected scenario on the loaded stand-in.
bool run_injected_scenarios(const InjectedInputs& in, HMODULE library)
{
    ordered_fence_vtable[10] = reinterpret_cast<void*>(&OrderedFence::signal);
    void* original = nullptr;
    if (replace_import(library, "CreateThread", reinterpret_cast<void*>(&faulting_create_thread), &original) == 0 ||
        !original) {
        std::fprintf(stderr, "cannot intercept CreateThread in the stand-in\n");
        return false;
    }
    real_create_thread = reinterpret_cast<CreateThreadFn>(original);

    // The first scenario also reads the stand-in's own log: each failed Enqueue notes the code it was given.
    wchar_t log_path[MAX_PATH] = {};
    wchar_t temp_dir[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_worker_failure_%lu.log", temp_dir, GetCurrentProcessId());
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);
    bool ok = scenario_three_failures_then_start(in);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", nullptr);
    const std::string log_text = read_text_file(log_path);
    DeleteFileW(log_path);
    std::size_t noted = 0;
    for (std::size_t at = 0; (at = log_text.find("queue worker not started: error 1455", at)) != std::string::npos; ++at) ++noted;
    ok = injected_check(noted == 3, "three failures then start", "the three failures were not logged with error 1455") && ok;

    return ok && scenario_default_stack_fallback(in) && scenario_worker_never_starts(in);
}

}  // namespace

int main(int argc, char** argv)
{
    install_crash_report();
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--injected-only") != 0 &&
                                 std::strcmp(argv[2], "--crash-report-demo") != 0)) {
        std::fprintf(stderr, "usage: startup_storage_alloc_failure XCRDAPI.dll [--injected-only]\n");
        return 2;
    }
    if (argc == 3 && std::strcmp(argv[2], "--crash-report-demo") == 0) {
        *static_cast<volatile int*>(nullptr) = 1;  // shows what the crash report prints (exit code 0xE1410)
    }
    const bool injected_only = argc == 3 && std::strcmp(argv[2], "--injected-only") == 0;
    HMODULE library = LoadLibraryA(argv[1]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    void* factory_ptr = nullptr;
    if (!get_factory || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) {
        std::fprintf(stderr, "cannot load the DirectStorage stand-in\n");
        return 1;
    }
    auto* factory = static_cast<Object*>(factory_ptr);
    Queue queue;
    void* status_ptr = nullptr;
    void* queue_ptr = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 4, "Status", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0) {
        std::fprintf(stderr, "cannot create the queue\n");
        return 1;
    }
    queue.status = static_cast<Object*>(status_ptr);
    queue.queue = static_cast<Object*>(queue_ptr);

    std::vector<unsigned char> plain(60000);
    for (std::size_t i = 0; i < plain.size(); ++i) plain[i] = static_cast<unsigned char>((i % 251) ^ 0x5a);
    const auto stream = stored_zlib(plain);

    long hr = -1;
    std::vector<unsigned char> first(plain.size(), 0xAA);
    if (!queue.run(first, stream, 0, hr) || hr != 0 || first != plain) {
        std::fprintf(stderr, "the baseline request did not succeed (hr 0x%08lx)\n", static_cast<unsigned long>(hr));
        return 1;
    }

    // Worker creation fails with ERROR_COMMITMENT_LIMIT without any real memory pressure.
    const InjectedInputs injected{factory, description, &stream, &plain};
    if (!run_injected_scenarios(injected, library)) return 1;
    if (injected_only) {
        std::puts("worker creation failures completed every status and fence in order");
        return 0;
    }

    if (!cap_commit()) {
        std::puts("cannot place the process under a commit limit");
        return skipped;
    }
    void* probe = VirtualAlloc(nullptr, oversized_inflated, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (probe) {
        VirtualFree(probe, 0, MEM_RELEASE);
        std::puts("the commit limit is not enforced on this system");
        return skipped;
    }

    // The same number as a CancellationTag sizes nothing: the request succeeds under the limit.
    std::vector<unsigned char> tagged(plain.size(), 0xAA);
    if (!queue.run(tagged, stream, 3, hr, 0, oversized_inflated) || hr != 0 || tagged != plain) {
        std::fprintf(stderr, "a zlib request with a large CancellationTag did not succeed (hr 0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return 1;
    }

    std::vector<unsigned char> failing(plain.size(), 0xAA);
    if (!queue.run(failing, stream, 1, hr, oversized_inflated)) {
        std::fprintf(stderr, "the request that hit an allocation failure never completed\n");
        return 1;
    }
    if (hr >= 0) {
        std::fprintf(stderr, "the request that hit an allocation failure reported success (hr 0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return 1;
    }
    for (unsigned char byte : failing)
        if (byte != 0xAA) {
            std::fprintf(stderr, "the failed request changed its destination\n");
            return 1;
        }

    std::vector<unsigned char> after(plain.size(), 0xAA);
    if (!queue.run(after, stream, 2, hr) || hr != 0 || after != plain) {
        std::fprintf(stderr, "the worker did not serve a request after the failure (hr 0x%08lx)\n", static_cast<unsigned long>(hr));
        return 1;
    }

    // Also verify that if worker thread creation fails on a fresh queue under commit exhaustion,
    // enqueued status/signal entries complete with failure instead of hanging at E_PENDING, and
    // the queue recovers once commit is freed.
    Queue recovery_queue;
    void* status2_ptr = nullptr;
    void* queue2_ptr = nullptr;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 3, "Status2", nullptr, &status2_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue2_ptr) != 0) {
        std::fprintf(stderr, "cannot create the recovery queue\n");
        return 1;
    }
    recovery_queue.status = static_cast<Object*>(status2_ptr);
    recovery_queue.queue = static_cast<Object*>(queue2_ptr);

    // Two submitters that run while the fence is held: the first completes the failing entries and is inside the
    // fence, the second must return without waiting for it.
    QueueCall first_submit, second_submit;
    first_submit.queue = second_submit.queue = recovery_queue.queue;
    second_submit.status = recovery_queue.status;
    second_submit.enqueue_status_index = 2;
    HANDLE first_thread = CreateThread(nullptr, submit_thread_stack, QueueCall::run, &first_submit, 0, nullptr);
    HANDLE second_thread = CreateThread(nullptr, submit_thread_stack, QueueCall::run, &second_submit, 0, nullptr);
    if (!first_thread || !second_thread || !first_submit.go || !second_submit.go) {
        std::fprintf(stderr, "cannot start the submit threads\n");
        return 1;
    }

    std::vector<unsigned char> during_exhaustion(plain.size(), 0xAA);
    std::vector<unsigned char> after_recovery(plain.size(), 0xAA);
    std::vector<void*> held;
    held.reserve(4096);
    for (SIZE_T chunk : {SIZE_T{1} << 20, SIZE_T{64} << 10}) {
        while (held.size() < held.capacity()) {
            void* p = VirtualAlloc(nullptr, chunk, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!p) break;
            held.push_back(p);
        }
    }
    if (held.size() < held.capacity()) {
        if (void* region = VirtualAlloc(nullptr, SIZE_T{64} << 10, MEM_RESERVE, PAGE_READWRITE)) {
            held.push_back(region);
            for (SIZE_T off = 0; off < (SIZE_T{64} << 10); off += 4096) {
                if (!VirtualAlloc(static_cast<unsigned char*>(region) + off, 4096, MEM_COMMIT, PAGE_READWRITE)) break;
            }
        }
    }

    void* fence_vtable[11] = {};
    fence_vtable[10] = reinterpret_cast<void*>(&MockFence::signal);
    MockFence fence{fence_vtable};
    fence.entered = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    fence.release = CreateEventW(nullptr, TRUE, FALSE, nullptr);

    unsigned char req[0x48] = {};
    *reinterpret_cast<u64*>(req + 0x00) = 1 | 0x400000;
    *reinterpret_cast<unsigned char**>(req + 0x08) = during_exhaustion.data();
    *reinterpret_cast<unsigned*>(req + 0x18) = static_cast<unsigned>(during_exhaustion.size());
    *reinterpret_cast<const unsigned char**>(req + 0x20) = stream.data();
    *reinterpret_cast<unsigned*>(req + 0x30) = static_cast<unsigned>(stream.size());

    reinterpret_cast<EnqueueRequestFn>(recovery_queue.queue->vtable[3])(recovery_queue.queue, req);
    reinterpret_cast<EnqueueStatusFn>(recovery_queue.queue->vtable[4])(recovery_queue.queue, recovery_queue.status, 0);
    reinterpret_cast<EnqueueSignalFn>(recovery_queue.queue->vtable[5])(recovery_queue.queue, &fence, 0x1234ull);

    // Submit on one thread; its failing entries complete on that thread and block in the fence. A second Submit
    // with a new deferred entry must return while the fence is held (no queue lock across a fence), and its entry
    // must complete after the fence returns.
    SetEvent(first_submit.go);
    const bool fence_entered = WaitForSingleObject(fence.entered, completion_timeout_ms) == WAIT_OBJECT_0;
    bool second_returned = false;
    if (fence_entered) {
        SetEvent(second_submit.go);
        second_returned = WaitForSingleObject(second_submit.done, 3000) == WAIT_OBJECT_0;
    }
    SetEvent(fence.release);
    const bool first_returned = WaitForSingleObject(first_submit.done, completion_timeout_ms) == WAIT_OBJECT_0;
    WaitForSingleObject(second_submit.done, completion_timeout_ms);

    const bool completed_immediately =
        reinterpret_cast<IsCompleteFn>(recovery_queue.status->vtable[3])(recovery_queue.status, 0);
    const long exhausted_hr =
        reinterpret_cast<GetHResultFn>(recovery_queue.status->vtable[4])(recovery_queue.status, 0);
    const DWORD second_start = GetTickCount();
    while (!reinterpret_cast<IsCompleteFn>(recovery_queue.status->vtable[3])(recovery_queue.status, 2) &&
           GetTickCount() - second_start < completion_timeout_ms) {
        Sleep(5);
    }
    const bool second_completed =
        reinterpret_cast<IsCompleteFn>(recovery_queue.status->vtable[3])(recovery_queue.status, 2);
    const long second_hr = reinterpret_cast<GetHResultFn>(recovery_queue.status->vtable[4])(recovery_queue.status, 2);

    for (void* p : held) VirtualFree(p, 0, MEM_RELEASE);
    held.clear();

    if (!fence_entered || !second_returned || !first_returned || !second_completed || second_hr >= 0) {
        std::fprintf(stderr,
                     "fence held across the queue lock (entered=%d second_returned=%d first_returned=%d "
                     "second_complete=%d second_hr=0x%08lx)\n",
                     fence_entered ? 1 : 0, second_returned ? 1 : 0, first_returned ? 1 : 0, second_completed ? 1 : 0,
                     static_cast<unsigned long>(second_hr));
        return 1;
    }

    if (!completed_immediately || exhausted_hr >= 0 || !fence.signaled || fence.value != 0x1234ull) {
        std::fprintf(stderr,
                     "worker-startup failure did not fail cleanly (complete=%d hr=0x%08lx signaled=%d val=0x%llx)\n",
                     completed_immediately ? 1 : 0, static_cast<unsigned long>(exhausted_hr), fence.signaled ? 1 : 0,
                     static_cast<unsigned long long>(fence.value));
        return 1;
    }

    if (!recovery_queue.run(after_recovery, stream, 1, hr) || hr != 0 || after_recovery != plain) {
        std::fprintf(stderr, "the queue did not recover after worker startup failure (hr 0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return 1;
    }

    std::puts("allocation failure failed the request; the worker served the next one");
    return 0;
}