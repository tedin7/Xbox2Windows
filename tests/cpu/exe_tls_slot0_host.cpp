// CRT-free host that maps tests/cpu/exe_tls_slot0.cpp with pe_map and starts it as the main image
// (pe_map_run_main) or, as a control, as a plain mapped EXE (pe_map_run).
//
// The host is an EXE with its own TLS directory and one thread-local sentinel, so under Windows its
// block is the one in TLS slot 0. A thread reads its sentinel only while it does not participate in the
// main image; slot 0 itself is read through the TEB.
//
// Usage: cpu_exe_tls_slot0_host <pe_map.dll> <image> <scenario>
//   main       start the image, let a host thread enter and leave, release on the starting thread
//   alive      a host participant is still entered at the first release, which must be refused (0, and
//              the status API reports the refusal); it then leaves and the second release returns 1
//   exited     a host participant is ended with TerminateThread; release confirms the exit
//   untracked  after the start MAIN_TRACKED host threads enter; the last finds the participant table
//              full and has no handle, so the first release must be refused until it leaves
//   paused     <image> is cpu_exe_tls_slot0_pause: its worker c pauses inside the image's code; the first
//              release must be refused with the image, its unwind data and its carrier intact; c then
//              resumes, checks its block, returns, and the second release is accepted
//   plain      start the image with pe_map_run (control): slot 0 keeps the host's block
//   probe            <image> is cpu_exe_tls_slot0_enroll: its probe worker p is enrolled and runs (control)
//   enroll_refused   the same, with p's enrollment made to fail (pe_map_main_tls_inject(1)): the guest's
//                    CreateThread returns null and no thread is created
//   install_failed   the same, with the installation on p made to fail (pe_map_main_tls_inject(2)): p ends
//                    with 0xE0000201 without its start routine
//   resume_failed    the same, with the resume of p made to fail (pe_map_main_tls_inject(3)): the guest's
//                    CreateThread returns null with ERROR_ACCESS_DENIED and p is ended before it runs
// Under pe_map_run_main the guest's CreateThread import is bound to pe_map_main_create_thread, so the guest's
// workers are enrolled; under pe_map_run it is bound to kernel32's CreateThread.
// In every scenario a reader thread that never participates samples its own slot 0 and sentinel from
// before the map until after release, and a late thread created after the start checks its own slot 0
// and sentinel while the guest's TLS callback counts (exe_slot0_callback_counts) are read before and
// after its life.
//
// After the guest's own LOG line, the host prints one line:
//   HOST: scenario=<s> started=<0|1> exit=<hex> installed_main=<y/n> reader_untouched=<y/n>
//         reader_samples=<n> participant=<y/n|-> refused=<y/n|-> intact_on_refusal=<y/n|->
//         resumed=<y/n|-> late_untouched=<y/n> late_callbacks=<n> callbacks=<ta>,<td> host_tls=<y/n>
//         process_image=<start><refusal><release> cleaned=<y/n> second_release=<y/n> heap_valid=<y/n>
//         stats=<installed>,<restored>,<exited>,<refusals>,<pending>,<active>,<last release>,<untracked>,
//               <enrollments refused>,<enrolled threads failed>
//         [probe=<sentinel>,<created 1/0>,<last error>,<exit code hex>]   probe scenarios only
// installed_main: after the start, the starting thread's slot 0 holds the image's block.
// participant: the host participant(s) saw the image's block after entering and their own block and
// sentinel after leaving ('-' when the scenario has none).
// late_untouched: the late thread's slot 0 and sentinel never changed. late_callbacks: how many
// thread-attach and thread-detach callbacks the guest received during the late thread's life (0 when the
// image is the main image, since the late thread does not participate; 2 under pe_map_run). callbacks:
// the guest's thread-attach and thread-detach callback totals just before the accepted release.
// refused / intact_on_refusal: the first pe_map_release returned 0 with the status API reporting a
// refusal (last release 2), and right after it the image was
// still committed, RtlLookupFunctionEntry still found its entry point, the carrier was loaded and the
// starting thread's slot 0 still held the image's block ('-' where no refusal is expected).
// resumed: the paused worker found its own block and values after the refusal.
// host_tls: after the accepted release the starting thread's slot 0 and sentinel are the host's again.
// process_image: which image GetModuleHandleW(nullptr) returns right after the start, right after the
// refused release ('-' without one) and right after the accepted release: i the mapped image, h this
// host, o neither.
// cleaned: after the accepted release the image is no longer committed, its unwind data is gone and
// the carrier is unloaded; it is the only pe_map_release that returned 1. second_release: a further
// pe_map_release returns 0 (one cleanup only).
// probe: the guest's sentinel (touched first by p's start routine), whether its CreateThread returned a
// handle, the last error it saw and p's exit code; the host process reaching this line shows it survived.
// Exit code 0 when the host could run every step, 2 for a usage or setup failure.
#include <windows.h>
#include <intrin.h>
#include <psapi.h>

#include <cstddef>
#include <cstdint>

#pragma runtime_checks("", off)

extern "C" {
ULONG _tls_index = 0;
#pragma data_seg(".tls")
char _tls_start = 0;
#pragma data_seg(".tls$ZZZ")
char _tls_end = 0;
#pragma data_seg()
#pragma section(".CRT$XLA", read)
#pragma section(".CRT$XLZ", read)
__declspec(allocate(".CRT$XLA")) PIMAGE_TLS_CALLBACK __xl_a = nullptr;
__declspec(allocate(".CRT$XLZ")) PIMAGE_TLS_CALLBACK __xl_z = nullptr;
extern const IMAGE_TLS_DIRECTORY64 _tls_used = {
    reinterpret_cast<ULONGLONG>(&_tls_start), reinterpret_cast<ULONGLONG>(&_tls_end),
    reinterpret_cast<ULONGLONG>(&_tls_index), reinterpret_cast<ULONGLONG>(&__xl_a + 1), 0, 0};
}

__declspec(thread) std::uint64_t t_host_sentinel = 0x0123456789abcdefull;

namespace {
constexpr std::uint64_t SENTINEL = 0x0123456789abcdefull;
constexpr int MAIN_TRACKED = 64;  // the participant table size in tools/pe_map.cpp
constexpr int STATS = 10;         // pe_map_main_tls_stats entries

using Resolver = void*(__cdecl*)(const char* dll, const char* name, std::uint32_t ordinal);
using MapImage = void*(__cdecl*)(const wchar_t*, Resolver, std::uint32_t*, char*, std::size_t);
using Run = int(__cdecl*)(void*, std::uint32_t*);
using Release = int(__cdecl*)(void*);
using Join = int(__cdecl*)();
using Stats = void(__cdecl*)(std::int64_t*);
using Export = void*(__cdecl*)(void*, const char*);
using Counts = void(__cdecl*)(LONG*);
using Inject = void(__cdecl*)(int);

HMODULE g_kernel;
void* g_create_thread;  // what the guest's CreateThread import is bound to
Join g_enter, g_leave;
DWORD g_index;  // the image's TLS index (the carrier's)
char g_detail[256];

int ascii_equal(const char* a, const char* b) {
    for (; *a && *b; ++a, ++b) {
        char x = *a, y = *b;
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x + 32);
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y + 32);
        if (x != y) return 0;
    }
    return *a == *b;
}

// Binds kernel32.dll only; any other import stays unresolved and fails the map.
void* __cdecl resolve(const char* dll, const char* name, std::uint32_t) {
    if (!name || !ascii_equal(dll, "kernel32.dll")) return nullptr;
    if (ascii_equal(name, "CreateThread")) return g_create_thread;
    return reinterpret_cast<void*>(GetProcAddress(g_kernel, name));
}

// winnt.h's inline NtCurrentTeb is compiled with the runtime checks of a Debug build.
void** my_slots() { return *reinterpret_cast<void***>(__readgsqword(0x30) + 0x58); }
void* my_slot0() {
    void** slots = my_slots();
    return slots ? slots[0] : nullptr;
}
bool holds_image_block() {
    void** slots = my_slots();
    return slots && slots[g_index] && slots[0] == slots[g_index];
}

// Never participates: samples its own slot 0 and sentinel until told to stop.
volatile LONG g_stop_reader;
volatile LONG g_reader_bad;
volatile LONG g_reader_samples;
HANDLE g_reader_ready;

DWORD WINAPI reader(void*) {
    void* const own = my_slot0();
    SetEvent(g_reader_ready);
    while (!InterlockedCompareExchange(&g_stop_reader, 0, 0)) {
        if (my_slot0() != own || t_host_sentinel != SENTINEL) InterlockedIncrement(&g_reader_bad);
        InterlockedIncrement(&g_reader_samples);
        SwitchToThread();
    }
    return 0;
}

// A host participant: enters, checks, waits for its cue, leaves, checks again.
struct Participant {
    HANDLE entered, cue;
    void* own;
    bool entered_ok, left_ok;
};

DWORD WINAPI participant(void* parameter) {
    auto* self = static_cast<Participant*>(parameter);
    self->own = my_slot0();
    self->entered_ok = g_enter() == 1 && holds_image_block();
    SetEvent(self->entered);
    WaitForSingleObject(self->cue, INFINITE);
    self->left_ok = g_leave() == 1 && my_slot0() == self->own && t_host_sentinel == SENTINEL;
    return 0;
}

// Created after the start; never participates. Checks its slot 0 and sentinel while it lives.
volatile LONG g_late_ok;

DWORD WINAPI late_thread(void*) {
    void* const own = my_slot0();
    bool ok = t_host_sentinel == SENTINEL;
    for (int i = 0; i < 1000 && ok; ++i) {
        ok = my_slot0() == own && t_host_sentinel == SENTINEL;
        SwitchToThread();
    }
    InterlockedExchange(&g_late_ok, ok ? 1 : 0);
    return 0;
}

Participant g_parts[MAIN_TRACKED];
HANDLE g_part_threads[MAIN_TRACKED];

bool start_participant(int i) {
    g_parts[i] = {CreateEventW(nullptr, TRUE, FALSE, nullptr), CreateEventW(nullptr, TRUE, FALSE, nullptr), nullptr,
                  false, false};
    g_part_threads[i] = CreateThread(nullptr, 0, participant, &g_parts[i], 0, nullptr);
    return g_part_threads[i] && WaitForSingleObject(g_parts[i].entered, 10000) == WAIT_OBJECT_0;
}

bool finish_participant(int i) {
    SetEvent(g_parts[i].cue);
    const bool joined = WaitForSingleObject(g_part_threads[i], 10000) == WAIT_OBJECT_0;
    CloseHandle(g_part_threads[i]);
    return joined && g_parts[i].entered_ok && g_parts[i].left_ok;
}

// Whether a module whose file name contains pe_map_tls_ (the carrier) is loaded.
HMODULE g_modules[512];

bool carrier_loaded() {
    HMODULE* modules = g_modules;
    DWORD needed = 0;
    if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(g_modules), &needed)) return false;
    const wchar_t marker[] = L"pe_map_tls_";
    for (DWORD i = 0; i < needed / sizeof(HMODULE) && i < 512; ++i) {
        wchar_t name[MAX_PATH];
        const DWORD length = GetModuleFileNameW(modules[i], name, MAX_PATH);
        for (DWORD at = 0; at + 11 <= length; ++at) {
            DWORD k = 0;
            while (k < 11 && name[at + k] == marker[k]) ++k;
            if (k == 11) return true;
        }
    }
    return false;
}

char g_line[512];
unsigned g_length;

void put(char c) {
    if (g_length < sizeof(g_line)) g_line[g_length++] = c;
}
void put(const char* text) {
    for (; *text; ++text) put(*text);
}
void put_flag(const char* name, int value) {  // 1 y, 0 n, -1 -
    put(' ');
    put(name);
    put('=');
    put(value < 0 ? '-' : value ? 'y' : 'n');
}
void put_number(std::int64_t value, bool hex) {
    if (value < 0) {
        put('-');
        value = -value;
    }
    char digits[20];
    int n = 0;
    const unsigned base = hex ? 16 : 10;
    auto v = static_cast<std::uint64_t>(value);
    do {
        digits[n++] = "0123456789abcdef"[v % base];
        v /= base;
    } while (v);
    while (n) put(digits[--n]);
}

void flush() {
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), g_line, g_length, &written, nullptr);
}

int fail_setup(const char* what) {
    put("HOST: setup failed: ");
    put(what);
    put('\n');
    flush();
    return 2;
}

int split(wchar_t* line, wchar_t** out, int capacity) {
    int count = 0;
    bool quoted = false;
    wchar_t* start = nullptr;
    for (wchar_t* p = line;; ++p) {
        const wchar_t c = *p;
        if (c == L'"') {
            quoted = !quoted;
            *p = 0;
            if (!start) start = p + 1;
            continue;
        }
        if (c == 0 || (!quoted && c == L' ')) {
            if (start && count < capacity) out[count++] = start;
            start = nullptr;
            if (c == 0) break;
            *p = 0;
        } else if (!start) {
            start = p;
        }
    }
    return count;
}

DWORD tls_index_of(void* base) {
    auto* bytes = static_cast<std::uint8_t*>(base);
    const auto* nt =
        reinterpret_cast<const IMAGE_NT_HEADERS64*>(bytes + reinterpret_cast<IMAGE_DOS_HEADER*>(bytes)->e_lfanew);
    const auto& entry = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    const auto* tls = reinterpret_cast<const IMAGE_TLS_DIRECTORY64*>(bytes + entry.VirtualAddress);
    return *reinterpret_cast<const DWORD*>(tls->AddressOfIndex);
}

DWORD entry_rva(void* base) {
    auto* bytes = static_cast<std::uint8_t*>(base);
    return reinterpret_cast<const IMAGE_NT_HEADERS64*>(bytes + reinterpret_cast<IMAGE_DOS_HEADER*>(bytes)->e_lfanew)
        ->OptionalHeader.AddressOfEntryPoint;
}

bool committed(void* base) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(base, &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT;
}

bool has_unwind(DWORD64 address) {
    DWORD64 image_base = 0;
    return RtlLookupFunctionEntry(address, &image_base, nullptr) != nullptr;
}

// Local\xbw_slot0_<pid in hex>_<suffix>: the events the paused worker opens (tests/cpu/exe_tls_slot0.cpp).
HANDLE host_event(const wchar_t* suffix) {
    wchar_t name[64];
    unsigned n = 0;
    for (const wchar_t* p = L"Local\\xbw_slot0_"; *p; ++p) name[n++] = *p;
    const DWORD pid = GetCurrentProcessId();
    for (int shift = 28; shift >= 0; shift -= 4) name[n++] = L"0123456789abcdef"[(pid >> shift) & 15];
    name[n++] = L'_';
    for (const wchar_t* p = suffix; *p; ++p) name[n++] = *p;
    name[n] = 0;
    return CreateEventW(nullptr, TRUE, FALSE, name);
}

// Which image GetModuleHandleW(nullptr) reports as the process image: 'i' the mapped image, 'h' this host.
char process_image(const void* image, HMODULE host) {
    const HMODULE now = GetModuleHandleW(nullptr);
    return now == image ? 'i' : now == host ? 'h' : 'o';
}

bool same(const wchar_t* a, const wchar_t* b) {
    for (; *a && *a == *b; ++a, ++b) {
    }
    return *a == *b;
}
}  // namespace

extern "C" int host_entry() {
    g_kernel = GetModuleHandleW(L"kernel32.dll");
    wchar_t* argv[8];
    const int argc = split(GetCommandLineW(), argv, 8);
    if (argc != 4)
        ExitProcess(fail_setup("usage: host <pe_map.dll> <image> main|alive|exited|untracked|paused|plain|probe|"
                               "enroll_refused|install_failed|resume_failed|cancelled"));
    const wchar_t* scenario = argv[3];
    const bool plain = same(scenario, L"plain");
    const bool alive = same(scenario, L"alive"), exited = same(scenario, L"exited"),
               untracked = same(scenario, L"untracked"), paused = same(scenario, L"paused");
    const bool enroll_refused = same(scenario, L"enroll_refused"), install_failed = same(scenario, L"install_failed");
    const bool cancelled = same(scenario, L"cancelled"), resume_failed = same(scenario, L"resume_failed");
    const bool probe = enroll_refused || install_failed || resume_failed || cancelled || same(scenario, L"probe");
    if (!plain && !alive && !exited && !untracked && !paused && !probe && !same(scenario, L"main"))
        ExitProcess(fail_setup("scenario"));
    HANDLE paused_event = nullptr, resume_event = nullptr, ok_event = nullptr, bad_event = nullptr;
    if (paused) {
        paused_event = host_event(L"paused"), resume_event = host_event(L"resume");
        ok_event = host_event(L"ok"), bad_event = host_event(L"bad");
        if (!paused_event || !resume_event || !ok_event || !bad_event) ExitProcess(fail_setup("events"));
    }

    HMODULE mapper = LoadLibraryW(argv[1]);
    if (!mapper) ExitProcess(fail_setup("cannot load pe_map"));
    auto map = reinterpret_cast<MapImage>(GetProcAddress(mapper, "pe_map_image"));
    auto run = reinterpret_cast<Run>(GetProcAddress(mapper, plain ? "pe_map_run" : "pe_map_run_main"));
    auto release = reinterpret_cast<Release>(GetProcAddress(mapper, "pe_map_release"));
    auto stats = reinterpret_cast<Stats>(GetProcAddress(mapper, "pe_map_main_tls_stats"));
    auto mapped_export = reinterpret_cast<Export>(GetProcAddress(mapper, "pe_map_export"));
    g_enter = reinterpret_cast<Join>(GetProcAddress(mapper, "pe_map_main_thread_enter"));
    g_leave = reinterpret_cast<Join>(GetProcAddress(mapper, "pe_map_main_thread_leave"));
    auto inject = reinterpret_cast<Inject>(GetProcAddress(mapper, "pe_map_main_tls_inject"));
    g_create_thread = plain ? reinterpret_cast<void*>(GetProcAddress(g_kernel, "CreateThread"))
                            : reinterpret_cast<void*>(GetProcAddress(mapper, "pe_map_main_create_thread"));
    if (!map || !run || !release || !stats || !mapped_export || !g_enter || !g_leave || !inject || !g_create_thread)
        ExitProcess(fail_setup("missing pe_map export"));

    const bool sentinel_before = t_host_sentinel == SENTINEL;  // the last host TLS access until release
    void* const my_host_block = my_slot0();
    g_reader_ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE reader_thread = CreateThread(nullptr, 0, reader, nullptr, 0, nullptr);
    if (!reader_thread || WaitForSingleObject(g_reader_ready, 10000) != WAIT_OBJECT_0) ExitProcess(fail_setup("reader"));

    const HMODULE host_image = GetModuleHandleW(nullptr);
    std::uint32_t status = 0;
    void* base = map(argv[2], resolve, &status, g_detail, sizeof(g_detail));
    if (!base) ExitProcess(fail_setup(g_detail[0] ? g_detail : "map failed"));
    g_index = tls_index_of(base);
    const DWORD64 entry_point = reinterpret_cast<DWORD64>(base) + entry_rva(base);

    auto counts = reinterpret_cast<Counts>(mapped_export(base, "exe_slot0_callback_counts"));
    if (!counts) ExitProcess(fail_setup("guest callback counts"));
    auto probe_result = reinterpret_cast<Counts>(probe ? mapped_export(base, "exe_slot0_probe") : nullptr);
    if (probe && !probe_result) ExitProcess(fail_setup("guest probe"));
    if (enroll_refused) inject(1);
    if (install_failed) inject(2);
    if (resume_failed) inject(3);

    std::uint32_t exit_code = 0xdeadbeef;
    const int started = run(base, &exit_code);
    const bool installed_main = holds_image_block();
    // The process image after the start, after a refused release ('-' when none) and after the accepted release.
    char identity[3] = {process_image(base, host_image), '-', '-'};

    // More than one table's worth of workers can end without ever entering their start routine.
    if (cancelled && started) {
        const auto create = reinterpret_cast<decltype(&CreateThread)>(g_create_thread);
        volatile LONG started_workers = 0;
        auto start = +[](void* arg) -> DWORD {
            InterlockedIncrement(static_cast<volatile LONG*>(arg));
            return 0;
        };
        for (int i = 0; i < 65; ++i) {
            HANDLE worker = create(nullptr, 0, start, const_cast<LONG*>(&started_workers), CREATE_SUSPENDED, nullptr);
            if (!worker) {
                put("cancelled: creation failed at iteration ");
                put_number(i + 1, false);
                put("\n");
                ExitProcess(fail_setup("cancelled thread create"));
            }
            if (!TerminateThread(worker, 0) || WaitForSingleObject(worker, 10000) != WAIT_OBJECT_0)
                ExitProcess(fail_setup("cancelled thread exit"));
            CloseHandle(worker);
        }
        if (started_workers) ExitProcess(fail_setup("cancelled thread reached start routine"));
    }

    // A host thread created after the start: never a participant.
    LONG before_late[4], after_late[4];
    counts(before_late);
    HANDLE late = CreateThread(nullptr, 0, late_thread, nullptr, 0, nullptr);
    if (!late || WaitForSingleObject(late, 10000) != WAIT_OBJECT_0) ExitProcess(fail_setup("late thread"));
    CloseHandle(late);
    counts(after_late);
    const bool late_untouched = g_late_ok == 1;
    const LONG late_callbacks = (after_late[2] - before_late[2]) + (after_late[3] - before_late[3]);

    // Host participants.
    int participant_ok = -1;
    int count = 0;
    if (!plain && started) {
        count = untracked ? MAIN_TRACKED : 1;
        for (int i = 0; i < count; ++i)
            if (!start_participant(i)) ExitProcess(fail_setup("participant"));
        if (exited) {
            TerminateThread(g_part_threads[0], 0);  // inside the image, never leaves
            WaitForSingleObject(g_part_threads[0], 10000);
            CloseHandle(g_part_threads[0]);
            participant_ok = g_parts[0].entered_ok;
            count = 0;
        }
    }
    // In alive and untracked the last participant stays entered across the first release; in paused the
    // guest's worker c stays inside the image's code.
    const int stay = (alive || untracked) && count ? count - 1 : -1;
    bool leave_ok = true;
    for (int i = 0; i < count; ++i)
        if (i != stay) leave_ok = finish_participant(i) && leave_ok;

    int refused = -1, intact = -1, resumed = -1;
    if (paused && WaitForSingleObject(paused_event, 10000) != WAIT_OBJECT_0) ExitProcess(fail_setup("paused worker"));
    if (stay >= 0 || paused) {
        const int first = release(base);
        std::int64_t status_now[STATS];
        stats(status_now);
        refused = first == 0 && status_now[6] == 2;
        identity[1] = process_image(base, host_image);
        intact = committed(base) && has_unwind(entry_point) && carrier_loaded() && holds_image_block();
        if (stay >= 0) leave_ok = finish_participant(stay) && leave_ok;
        if (paused) {
            SetEvent(resume_event);
            HANDLE results[2] = {ok_event, bad_event};
            resumed = WaitForMultipleObjects(2, results, FALSE, 10000) == WAIT_OBJECT_0;
            std::int64_t now[STATS];
            for (int i = 0; i < 500; ++i) {  // c's detach removes it from the participants
                stats(now);
                if (now[4] == 1) break;
                Sleep(10);
            }
        }
    }
    if (count) participant_ok = leave_ok;

    LONG final_counts[4];
    counts(final_counts);
    LONG probe_out[4] = {};
    if (probe) probe_result(probe_out);  // the guest's export, read while the image is still mapped
    const int released = release(base);
    identity[2] = process_image(base, host_image);
    const bool cleaned = released == 1 && !committed(base) && !has_unwind(entry_point) && !carrier_loaded();
    const bool second_release = release(base) == 0;
    // From here on the starting thread has the host's block back, so the host may read its thread-locals.
    const bool host_tls =
        released == 1 && my_slot0() == my_host_block && sentinel_before && t_host_sentinel == SENTINEL;
    InterlockedExchange(&g_stop_reader, 1);
    WaitForSingleObject(reader_thread, 10000);
    const bool reader_untouched = g_reader_bad == 0 && g_reader_samples > 0;
    const bool heap_valid = HeapValidate(GetProcessHeap(), 0, nullptr) != FALSE;
    std::int64_t counters[STATS];
    stats(counters);

    put("HOST: scenario=");
    for (const wchar_t* c = scenario; *c; ++c) put(static_cast<char>(*c));
    put(" started=");
    put_number(started, false);
    put(" exit=");
    put_number(exit_code, true);
    put_flag("installed_main", installed_main);
    put_flag("reader_untouched", reader_untouched);
    put(" reader_samples=");
    put_number(g_reader_samples, false);
    put_flag("participant", participant_ok);
    put_flag("refused", refused);
    put_flag("intact_on_refusal", intact);
    put_flag("resumed", resumed);
    put_flag("late_untouched", late_untouched);
    put(" late_callbacks=");
    put_number(late_callbacks, false);
    put(" callbacks=");
    put_number(final_counts[2], false);
    put(',');
    put_number(final_counts[3], false);
    put_flag("host_tls", host_tls);
    put(" process_image=");
    for (char c : identity) put(c);
    put_flag("cleaned", cleaned);
    put_flag("second_release", second_release);
    put_flag("heap_valid", heap_valid);
    put(" stats=");
    for (int i = 0; i < STATS; ++i) {
        if (i) put(',');
        put_number(counters[i], false);
    }
    if (probe) {
        put(" probe=");
        for (int i = 0; i < 3; ++i) {
            put_number(probe_out[i], false);
            put(',');
        }
        put_number(static_cast<std::uint32_t>(probe_out[3]), true);
    }
    put('\n');
    flush();
    ExitProcess(0);
}
