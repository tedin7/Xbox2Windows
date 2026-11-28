// Source-built check of the activity counter in the AcpHal and XApu stand-ins (src/startup_stubs/acphal.cpp and
// xapu.cpp, X2W_AUDIO_ACTIVITY=1, docs/startup-runtime.md). One caller script fills, changes and clears XMA contexts,
// submits HAL commands and enqueues XApu commands. Each case runs in a child process of its own:
//   on    counter on: the exports and the logged lines report exactly what the script did, and a freed and
//         reused context array is no longer counted
//   none  counter on, but the caller writes only next to the contexts and sends nothing: every count is 0 and no
//         activity line is logged
//   off   counter unset, same script as on: the exports report nothing and no line is logged
//   unload counter on: AcpHalAllocateShapeContexts runs again without an ApuFree (the old array's last changes count,
//         later ones do not, the new array is counted), then the caller unloads both DLLs and stays alive for 1.5 s;
//         it must exit 0, so nothing of either DLL may run after FreeLibrary
// The stand-ins log a due activity line only inside a call into them, so the cases call the exports before they read
// the log.
// Every call's result, output and last error, and the final context bytes, must be the same with the counter on and
// off. The 64-byte context stride and the command class (first UINT32) are stand-in conventions, not a console layout.
// Usage: startup_acphal_xma_activity <AcpHal.dll> <xapu.dll>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
using u64 = std::uint64_t;
using CreateHeapFn = long(__cdecl*)(u64, u64);
using AllocFn = long(__cdecl*)(void**, std::uint32_t*, u64, u64, u64);
using FreeFn = long(__cdecl*)(void*);
using HalCreateFn = long(__cdecl*)(void**);
using ShapeAllocFn = long(__cdecl*)(void*);
using HalConnectFn = long(__stdcall*)(void*, unsigned, unsigned);
using HalSubmitFn = long(__stdcall*)(void*, unsigned, u64, unsigned, const unsigned*, u64);
using ConnectFn = long(__cdecl*)(const void*, void*, u64*);
using EnqueueFn = long(__cdecl*)(u64, const void*, u64, u64);
using CountFn = std::uint32_t(__cdecl*)(u64, u64);
using DequeueFn = long(__cdecl*)(u64, void*);
using DisconnectFn = long(__cdecl*)(u64);
using ActivityFn = BOOL(WINAPI*)(ULONGLONG*);

constexpr std::uint32_t contexts = 350;
constexpr int acphal_fields = 15, xapu_fields = 19;
// x2w_acphal_xma_activity: watched, live, filled, changed, cleared, HAL submits, submits of type 0..7, other types.
enum { watched = 0, live = 1, filled = 2, changed = 3, cleared = 4, submits = 5, submit_type0 = 6, submit_other = 14 };
// x2w_xapu_command_counts: classes 0..15, larger classes, unreadable commands, unknown handles.
enum { class_other = 16, unreadable = 17, rejected = 18 };

struct ShapeContexts {
    std::uint32_t counts[6]{};
    unsigned char* arrays[6]{};
    std::uint32_t apu[6]{};
};
static_assert(sizeof(ShapeContexts) == 0x60, "ShapeContexts layout");

struct Api {
    CreateHeapFn create_heap;
    AllocFn alloc;
    FreeFn free;
    HalCreateFn hal_create;
    ShapeAllocFn shape_alloc;
    ConnectFn connect;
    EnqueueFn enqueue;
    CountFn result_count;
    DequeueFn dequeue;
    DisconnectFn disconnect;
    ActivityFn acphal_activity;
    ActivityFn xapu_counts;
};

int failures;
std::string transcript;  // every result, output and last error, compared between the on and off runs

void check(bool condition, const char* what)
{
    if (condition) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

// Calls are made with a sentinel last error, so a stand-in that changes the caller's last error shows up.
void note(const char* what, long hr, unsigned long long value = 0)
{
    char line[160];
    std::snprintf(line, sizeof(line), "%s hr=0x%08lx value=0x%llx last_error=%lu\n", what, static_cast<unsigned long>(hr), value,
                  GetLastError());
    transcript += line;
}
void sentinel() { SetLastError(0x5A5A); }

// XApuConnect refuses a shared buffer size of 0 (the u32 at input +0x0C) with E_INVALIDARG; 4 KiB is enough here.
void set_buffer_size(unsigned char* connect_input)
{
    const std::uint32_t bytes = 0x1000;
    std::memcpy(connect_input + 0x0C, &bytes, sizeof(bytes));
}

// One aligned 8-byte store, so the sampler sees the context before or after it, never half written.
void store(unsigned char* base, std::uint32_t context, unsigned word, u64 value)
{
    InterlockedExchange64(reinterpret_cast<volatile LONG64*>(base + context * 64u + word * 8u), static_cast<LONG64>(value));
}

struct Snapshot {
    BOOL on = FALSE;
    ULONGLONG v[xapu_fields > acphal_fields ? xapu_fields : acphal_fields];
};

Snapshot take(ActivityFn fn)
{
    Snapshot s;
    for (auto& x : s.v) x = 0xA5A5A5A5A5A5A5A5ull;  // the export must overwrite every field, on or off
    s.on = fn(s.v);
    return s;
}

using Fields = std::initializer_list<std::pair<int, u64>>;  // (field, value) pairs

// Every field is zero except the listed ones.
void expect(const char* label, const Snapshot& s, BOOL on, int fields, Fields nonzero)
{
    bool good = s.on == on;
    for (int i = 0; i < fields; ++i) {
        u64 want = 0;
        for (const auto& f : nonzero)
            if (f.first == i) want = f.second;
        good = good && s.v[i] == want;
    }
    if (good) return;
    std::printf("FAIL: %s: returned %d, fields", label, s.on);
    for (int i = 0; i < fields; ++i) std::printf(" %llu", s.v[i]);
    std::printf("\n");
    ++failures;
}

// The fields expected with the counter on; with it off the export returns FALSE and every field must be zero.
void expect_on(const char* label, const Snapshot& s, bool on, int fields, Fields when_on)
{
    expect(label, s, on, fields, on ? when_on : Fields{});
}

std::string read_file(const char* path)
{
    std::string text;
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof(buffer), &got, nullptr) && got) text.append(buffer, got);
    CloseHandle(file);
    return text;
}

// Sums the name=value tokens of the activity lines in the log, keyed "xma.<name>" or "xapu.<name>"; zero sums are
// dropped. A bare number names a submit type (xma) or a command class (xapu). The live=n/m token is not a count.
std::map<std::string, u64> activity_sums(const std::string& log, int* lines)
{
    std::map<std::string, u64> sums;
    *lines = 0;
    std::size_t at = 0;
    while (at < log.size()) {
        std::size_t end = log.find('\n', at);
        if (end == std::string::npos) end = log.size();
        const std::string line = log.substr(at, end - at);
        at = end + 1;
        const char* kind = line.find("x2w-audio xma-activity t=") != std::string::npos    ? "xma."
                           : line.find("x2w-audio xapu-activity t=") != std::string::npos ? "xapu."
                                                                                           : nullptr;
        if (!kind) continue;
        ++*lines;
        std::size_t pos = line.find(" s: ");
        while (pos != std::string::npos && pos < line.size()) {
            const std::size_t start = pos + 1;
            pos = line.find(' ', start);
            std::string token = line.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
            while (!token.empty() && (token.back() == ';' || token.back() == '\r')) token.pop_back();
            const std::size_t eq = token.find('=');
            if (eq == std::string::npos || token.find('/') != std::string::npos) continue;
            sums[kind + token.substr(0, eq)] += std::strtoull(token.c_str() + eq + 1, nullptr, 10);
        }
    }
    for (auto it = sums.begin(); it != sums.end();) it = it->second ? std::next(it) : sums.erase(it);
    return sums;
}

std::string describe(const std::map<std::string, u64>& sums)
{
    std::string text;
    for (const auto& s : sums) text += " " + s.first + "=" + std::to_string(s.second);
    return text;
}

int activity_lines(const char* log_path)
{
    int lines = 0;
    activity_sums(read_file(log_path), &lines);
    return lines;
}

// The caller script of the on and off cases. Snapshots are taken between phases and no other counted call runs while
// the contexts are written, so each context moves once between two samples.
void script(const Api& api, bool on)
{
    sentinel();
    note("ApuCreateHeap", api.create_heap(0x200000, 0));
    void* hal = nullptr;
    sentinel();
    note("AcpHalCreate", api.hal_create(&hal), hal != nullptr);
    if (!hal) return check(false, "AcpHalCreate returns an object");
    auto* const* table = *reinterpret_cast<void* const* const*>(hal);
    auto hal_connect = reinterpret_cast<HalConnectFn>(table[0]);
    auto hal_submit = reinterpret_cast<HalSubmitFn>(table[2]);
    sentinel();
    note("AcpHal connect", hal_connect(hal, 0x1000, 0x1000));
    ShapeContexts ctx{};
    ctx.counts[4] = contexts;
    sentinel();
    note("AcpHalAllocateShapeContexts", api.shape_alloc(&ctx), ctx.apu[4]);
    unsigned char* const base = ctx.arrays[4];
    if (!base) return check(false, "XMA contexts allocated");
    expect_on("before any write", take(api.acphal_activity), on, acphal_fields, {{watched, contexts}});

    // Fill 5 contexts, one word each; context 349 uses its last word.
    store(base, 0, 0, 0x1111);
    store(base, 1, 3, 0x2222);
    store(base, 2, 0, 0x3333);
    store(base, 100, 5, 0x4444);
    store(base, 349, 7, 0x5555);
    expect_on("after 5 fills", take(api.acphal_activity), on, acphal_fields,
              {{watched, contexts}, {live, 5}, {filled, 5}});
    // Change 3 live contexts: the same word, the last word, another word.
    store(base, 0, 0, 0x1112);
    store(base, 1, 7, 0x2223);
    store(base, 100, 2, 0x4445);
    expect_on("after 3 changes", take(api.acphal_activity), on, acphal_fields,
              {{watched, contexts}, {live, 5}, {filled, 5}, {changed, 3}});
    // Clear 2 contexts.
    store(base, 2, 0, 0);
    store(base, 349, 7, 0);
    expect_on("after 2 clears", take(api.acphal_activity), on, acphal_fields,
              {{watched, contexts}, {live, 3}, {filled, 5}, {changed, 3}, {cleared, 2}});

    const unsigned input = 0x12345678u;
    for (unsigned type : {1u, 1u, 3u, 12u}) {
        sentinel();
        note("AcpHal submit", hal_submit(hal, type, 0x1234, 2, &input, 0), type);
    }
    const auto after_submits = take(api.acphal_activity);
    expect_on("after 4 HAL submits", after_submits, on, acphal_fields,
              {{watched, contexts}, {live, 3}, {filled, 5}, {changed, 3}, {cleared, 2}, {submits, 4}, {submit_type0 + 1, 2},
               {submit_type0 + 3, 1}, {submit_other, 1}});

    unsigned char connect_input[0x18]{}, connect_output[0x10]{};
    set_buffer_size(connect_input);
    u64 handle = 0;
    sentinel();
    note("XApuConnect", api.connect(connect_input, connect_output, &handle), handle);
    unsigned char command[0x60]{};
    for (std::uint32_t command_class : {1u, 1u, 2u, 1u, 2u, 0x20u}) {
        std::memcpy(command, &command_class, sizeof(command_class));
        sentinel();
        note("XApuEnqueueCommand", api.enqueue(handle, command, 0, 0), command_class);
    }
    sentinel();
    note("XApuEnqueueCommand null command", api.enqueue(handle, nullptr, 0, 0));
    std::uint32_t one = 1;
    std::memcpy(command, &one, sizeof(one));
    sentinel();
    note("XApuEnqueueCommand unknown handle", api.enqueue(handle + 0x100, command, 0, 0));
    // A type 3 command whose 8-byte header ends where free address space begins (the rest of the page's allocation
    // granule), sent with typed completions on. An X2W_WITH_OPUS build refuses it with E_POINTER before the queue, as
    // too short to decode, and must still count it as class 3; without Opus it is queued like any other command.
    // X2W_PROBE_PRECHECK=1 (every case) refuses the free tail without touching it, so no fault is taken. Typed
    // completions are set for this call only.
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    auto* page = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, system.dwPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    MEMORY_BASIC_INFORMATION after{};
    const bool free_after =
        page && VirtualQuery(page + system.dwPageSize, &after, sizeof(after)) && after.State == MEM_FREE;
    check(free_after, "free address space after the short command");
    if (free_after) {
        unsigned char* short_command = page + system.dwPageSize - 8;
        const std::uint32_t three = 3;
        std::memcpy(short_command, &three, sizeof(three));
        SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", "1");
        sentinel();
        note("XApuEnqueueCommand short type 3 command", api.enqueue(handle, short_command, 0, 0));
        SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", nullptr);
    }
    if (page) VirtualFree(page, 0, MEM_RELEASE);
    sentinel();
    const std::uint32_t pending = api.result_count(handle, 0);
    note("XApuGetResultCount", 0, pending);
    for (int i = 0; i < 8; ++i) {
        unsigned char result[0x58];
        std::memset(result, 0xCC, sizeof(result));
        sentinel();
        const long hr = api.dequeue(handle, result);
        unsigned sum = 0;
        for (unsigned char b : result) sum += b;
        note("XApuDequeueResult", hr, sum);
    }
    sentinel();
    note("XApuDisconnect", api.disconnect(handle));
    expect_on("XApu commands", take(api.xapu_counts), on, xapu_fields,
              {{1, 3}, {2, 2}, {3, 1}, {class_other, 1}, {unreadable, 1}, {rejected, 1}});

    unsigned hash = 2166136261u;  // FNV-1a of the context bytes: the counter never writes caller memory
    for (std::uint32_t i = 0; i < contexts * 64; ++i) hash = (hash ^ base[i]) * 16777619u;
    sentinel();
    note("context bytes", 0, hash);

    // Freed contexts stop being watched; the block that comes back is not counted.
    sentinel();
    note("ApuFree contexts", api.free(base));
    void* again = nullptr;
    std::uint32_t again_apu = 0;
    sentinel();
    note("ApuAlloc same size", api.alloc(&again, &again_apu, contexts * 64, 64, 0), again_apu);
    sentinel();
    note("same block", 0, again == base);
    check(again == base, "ApuAlloc of the same size returns the freed context block");
    if (again) {
        auto* p = static_cast<unsigned char*>(again);
        store(p, 0, 0, 0x6666);
        store(p, 5, 1, 0x7777);
        store(p, 300, 7, 0x8888);
        Sleep(50);  // several 10 ms sample periods
    }
    expect_on("after free and reuse", take(api.acphal_activity), on, acphal_fields,
              {{filled, 5}, {changed, 3}, {cleared, 2}, {submits, 4}, {submit_type0 + 1, 2}, {submit_type0 + 3, 1},
               {submit_other, 1}});
}

// Zero-write control: the counter is on, the caller writes only to the block after the contexts and sends nothing.
void script_none(const Api& api)
{
    api.create_heap(0x200000, 0);
    void* hal = nullptr;
    check(api.hal_create(&hal) == 0 && hal, "AcpHalCreate");
    ShapeContexts ctx{};
    ctx.counts[4] = contexts;
    check(api.shape_alloc(&ctx) == 0 && ctx.arrays[4], "AcpHalAllocateShapeContexts");
    void* next = nullptr;
    std::uint32_t next_apu = 0;
    check(api.alloc(&next, &next_apu, 0x1000, 64, 0) == 0 && next, "ApuAlloc after the contexts");
    // The control writes the 64 bytes right after the last context, so it is only meaningful if they are adjacent.
    check(next_apu == ctx.apu[4] + contexts * 64, "the next block starts right after the last context");
    if (next) {
        store(static_cast<unsigned char*>(next), 0, 0, 0x9999);
        store(static_cast<unsigned char*>(next), 0, 7, 0x9999);
    }
    unsigned char connect_input[0x18]{}, connect_output[0x10]{};
    set_buffer_size(connect_input);
    u64 handle = 0;
    check(api.connect(connect_input, connect_output, &handle) == 0, "XApuConnect");
    check(api.disconnect(handle) == 0, "XApuDisconnect");
    Sleep(50);
    expect("zero-write control", take(api.acphal_activity), TRUE, acphal_fields, {{watched, contexts}});
    expect("zero-write control XApu", take(api.xapu_counts), TRUE, xapu_fields, {});
}

// The unload case. Nothing in it is compared with another run.
void script_unload(const Api& api, HMODULE acphal, HMODULE xapu)
{
    check(api.create_heap(0x200000, 0) == 0, "ApuCreateHeap");
    void* hal = nullptr;
    check(api.hal_create(&hal) == 0 && hal, "AcpHalCreate");
    if (!hal) return;
    auto hal_submit = reinterpret_cast<HalSubmitFn>((*reinterpret_cast<void* const* const*>(hal))[2]);
    const unsigned input = 0;
    check(hal_submit(hal, 1, 0x1234, 2, &input, 0) == 0, "AcpHal submit");
    ShapeContexts first{}, second{};
    first.counts[4] = contexts;
    second.counts[4] = 4;
    check(api.shape_alloc(&first) == 0 && first.arrays[4], "first AcpHalAllocateShapeContexts");
    if (!first.arrays[4]) return;
    store(first.arrays[4], 0, 0, 0x1111);  // not sampled yet: the re-run's last sample of this array counts it
    check(api.shape_alloc(&second) == 0 && second.arrays[4] && second.arrays[4] != first.arrays[4],
          "AcpHalAllocateShapeContexts again, without ApuFree, gives another array");
    if (!second.arrays[4]) return;
    store(first.arrays[4], 1, 0, 0x2222);   // the old array is no longer watched
    store(second.arrays[4], 3, 7, 0x3333);  // the new one is, up to its last word
    expect("re-run without ApuFree", take(api.acphal_activity), TRUE, acphal_fields,
           {{watched, 4}, {live, 1}, {filled, 2}, {submits, 1}, {submit_type0 + 1, 1}});
    unsigned char connect_input[0x18]{}, connect_output[0x10]{}, command[0x60]{};
    set_buffer_size(connect_input);
    u64 handle = 0;
    command[0] = 1;
    check(api.connect(connect_input, connect_output, &handle) == 0, "XApuConnect");
    api.enqueue(handle, command, 0, 0);
    expect("XApu command before unload", take(api.xapu_counts), TRUE, xapu_fields, {{1, 1}});
    // Unloading must really unmap both images, else the wait below proves nothing.
    const void* const code[] = {reinterpret_cast<const void*>(api.create_heap), reinterpret_cast<const void*>(api.enqueue)};
    check(FreeLibrary(acphal) && FreeLibrary(xapu), "FreeLibrary of both stand-ins");
    for (const void* address : code) {
        HMODULE still = nullptr;
        check(!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                  static_cast<const char*>(address), &still),
              "the stand-in is unloaded after FreeLibrary");
    }
    Sleep(1500);  // past a 10 ms sample period and a 1 s line interval: a leftover thread of either DLL would run now
}

int run_case(const char* name, const char* acphal_path, const char* xapu_path, const char* transcript_path)
{
    HMODULE acphal = LoadLibraryA(acphal_path), xapu = LoadLibraryA(xapu_path);
    if (!acphal || !xapu) return std::printf("FAIL: cannot load the stand-ins\n"), 2;
    Api api{};
    api.create_heap = reinterpret_cast<CreateHeapFn>(GetProcAddress(acphal, "ApuCreateHeap"));
    api.alloc = reinterpret_cast<AllocFn>(GetProcAddress(acphal, "ApuAlloc"));
    api.free = reinterpret_cast<FreeFn>(GetProcAddress(acphal, "ApuFree"));
    api.hal_create = reinterpret_cast<HalCreateFn>(GetProcAddress(acphal, "AcpHalCreate"));
    api.shape_alloc = reinterpret_cast<ShapeAllocFn>(GetProcAddress(acphal, "AcpHalAllocateShapeContexts"));
    api.acphal_activity = reinterpret_cast<ActivityFn>(GetProcAddress(acphal, "x2w_acphal_xma_activity"));
    api.connect = reinterpret_cast<ConnectFn>(GetProcAddress(xapu, "XApuConnect"));
    api.enqueue = reinterpret_cast<EnqueueFn>(GetProcAddress(xapu, "XApuEnqueueCommand"));
    api.result_count = reinterpret_cast<CountFn>(GetProcAddress(xapu, "XApuGetResultCount"));
    api.dequeue = reinterpret_cast<DequeueFn>(GetProcAddress(xapu, "XApuDequeueResult"));
    api.disconnect = reinterpret_cast<DisconnectFn>(GetProcAddress(xapu, "XApuDisconnect"));
    api.xapu_counts = reinterpret_cast<ActivityFn>(GetProcAddress(xapu, "x2w_xapu_command_counts"));
    if (!api.create_heap || !api.alloc || !api.free || !api.hal_create || !api.shape_alloc || !api.acphal_activity || !api.connect ||
        !api.enqueue || !api.result_count || !api.dequeue || !api.disconnect || !api.xapu_counts)
        return std::printf("FAIL: missing export\n"), 2;
    char log_path[MAX_PATH] = "";
    GetEnvironmentVariableA("X2W_TRAP_LOG", log_path, MAX_PATH);
    check(api.acphal_activity(nullptr) == FALSE && api.xapu_counts(nullptr) == FALSE, "the exports refuse a null output");

    if (!std::strcmp(name, "unload")) {
        script_unload(api, acphal, xapu);
    } else if (!std::strcmp(name, "none")) {
        script_none(api);
        Sleep(1500);  // longer than the line interval: an unchanged count writes no line
        take(api.acphal_activity);
        take(api.xapu_counts);
        check(activity_lines(log_path) == 0, "zero-write control logs no activity line");
    } else {
        const bool on = !std::strcmp(name, "on");
        script(api, on);
        if (on) {
            const std::map<std::string, u64> want = {
                {"xma.filled", 5},   {"xma.changed", 3},       {"xma.cleared", 2}, {"xma.submits", 4}, {"xma.1", 2},
                {"xma.3", 1},        {"xma.other", 1},         {"xapu.commands", 8}, {"xapu.1", 3},    {"xapu.2", 2},
                {"xapu.3", 1},       {"xapu.other", 1},        {"xapu.unreadable", 1}, {"xapu.rejected", 1}};
            std::map<std::string, u64> got;
            int lines = 0;
            for (int waited = 0; waited <= 5000; waited += 100) {  // a call a second after the start writes the line
                take(api.acphal_activity);
                take(api.xapu_counts);
                got = activity_sums(read_file(log_path), &lines);
                if (got == want) break;
                Sleep(100);
            }
            if (got != want) {
                std::printf("FAIL: logged activity lines (%d) sum to%s\n      expected%s\n", lines, describe(got).c_str(),
                            describe(want).c_str());
                ++failures;
            }
        } else {
            Sleep(1500);
            take(api.acphal_activity);
            take(api.xapu_counts);
            check(activity_lines(log_path) == 0, "counter off logs no activity line");
            check(read_file(log_path).find("-activity") == std::string::npos, "counter off logs nothing about activity");
        }
        HANDLE out = CreateFileA(transcript_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        DWORD wrote = 0;
        check(out != INVALID_HANDLE_VALUE && WriteFile(out, transcript.data(), static_cast<DWORD>(transcript.size()), &wrote, nullptr) &&
                  wrote == transcript.size(),
              "transcript written");
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
    }
    std::printf(failures ? "case %s: %d failure(s)\n" : "case %s: ok\n", name, failures);
    return failures ? 1 : 0;
}

DWORD spawn(const char* self, const char* name, const char* acphal, const char* xapu, const std::string& dir)
{
    const std::string log = dir + "\\" + name + ".log", out = dir + "\\" + name + ".txt";
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_AUDIO_ACTIVITY", std::strcmp(name, "off") ? "1" : nullptr);
    SetEnvironmentVariableA("X2W_PROBE_PRECHECK", "1");  // the short command's free tail is refused, not touched
    SetEnvironmentVariableA("X2W_TRAP_LOG", log.c_str());
    std::string cmd = std::string("\"") + self + "\" --case " + name + " \"" + acphal + "\" \"" + xapu + "\" \"" + out + "\"";
    // The child writes to this process's stdout and stderr, so its lines reach the CTest log.
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    for (HANDLE h : {si.hStdOutput, si.hStdError}) SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    std::fflush(stdout);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) return 0xFFFFFFFFu;
    DWORD code = 0xFFFFFFFEu;
    // 12 s per case keeps all four inside the CTest TIMEOUT of 60 s, so a hung case is named, not killed with the parent.
    if (WaitForSingleObject(pi.hProcess, 12000) == WAIT_OBJECT_0) GetExitCodeProcess(pi.hProcess, &code);
    else TerminateProcess(pi.hProcess, 0xFFFFFFFDu);  // our own child, by handle
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 6 && !std::strcmp(argv[1], "--case")) return run_case(argv[2], argv[3], argv[4], argv[5]);
    if (argc != 3) return std::printf("usage: startup_acphal_xma_activity <AcpHal.dll> <xapu.dll>\n"), 2;
    char temp[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, temp)) return std::printf("FAIL: no temp folder\n"), 2;
    const std::string dir = std::string(temp) + "x2w_xma_activity_" + std::to_string(GetCurrentProcessId());
    if (!CreateDirectoryA(dir.c_str(), nullptr)) return std::printf("FAIL: cannot create %s\n", dir.c_str()), 2;
    char self[MAX_PATH];
    GetModuleFileNameA(nullptr, self, MAX_PATH);
    for (const char* name : {"on", "none", "off", "unload"}) {
        const DWORD code = spawn(self, name, argv[1], argv[2], dir);
        if (code) {
            std::printf("FAIL: case %s exited 0x%08lx\n", name, static_cast<unsigned long>(code));
            ++failures;
        }
    }
    const std::string on = read_file((dir + "\\on.txt").c_str()), off = read_file((dir + "\\off.txt").c_str());
    check(!on.empty(), "the on case wrote a transcript");
    if (on != off) {
        std::size_t at = 0;
        while (at < on.size() && at < off.size() && on[at] == off[at]) ++at;
        const std::size_t line = on.rfind('\n', at) == std::string::npos ? 0 : on.rfind('\n', at) + 1;
        std::printf("FAIL: results differ with the counter on and off, first at: on '%s' off '%s'\n",
                    on.substr(line, on.find('\n', line) - line).c_str(), off.substr(line, off.find('\n', line) - line).c_str());
        ++failures;
    }
    for (const char* file : {"on.log", "none.log", "off.log", "unload.log", "on.txt", "off.txt"}) DeleteFileA((dir + "\\" + file).c_str());
    RemoveDirectoryA(dir.c_str());
    std::printf(failures ? "startup_acphal_xma_activity: %d failure(s)\n" : "startup_acphal_xma_activity: ok\n", failures);
    return failures ? 1 : 0;
}
