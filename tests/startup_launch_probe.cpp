// Input for the launch tool's containment check (docs/startup-runtime.md). The launcher runs one
// executable in a kill-on-close job with a timeout, a commit cap and captured stdout/stderr, and
// this fixture is the only executable the check needs: it writes known bytes to both streams and
// the trap log, reports a known exit code, or holds memory, spawns a child or stays alive so the
// job limits have to stop it. The "overlap" mode (from X2W_LAUNCH_PROBE_MODE, because the debugger passes no
// arguments) opens one file from four threads that enter CreateFileW together, for tests/startup_hook_check.py. Paths stay UTF-16 throughout: the launcher passes a Unicode trap-log
// path, and its build or temp folder can hold characters the active ANSI code page cannot.
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <iterator>

namespace {
void stream(FILE* out, const char* text)
{
    std::fputs(text, out);
    std::fflush(out);
}

void write_trap_log(const char* text)
{
    const wchar_t* const path = _wgetenv(L"X2W_TRAP_LOG");
    if (!path) return;
    if (FILE* file = _wfopen(path, L"w")) {
        std::fputs(text, file);
        std::fclose(file);
    }
}

void spawn_second_process(const wchar_t* self)
{
    wchar_t command[MAX_PATH + 16]{};
    std::swprintf(command, std::size(command), L"\"%ls\" noop", self);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(nullptr, command, nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &child)) {
        std::printf("spawn failed error=%lu\n", GetLastError());
        return;
    }
    WaitForSingleObject(child.hProcess, 5000);
    CloseHandle(child.hThread);
    CloseHandle(child.hProcess);
    std::printf("spawn ok\n");
}
constexpr int overlap_threads = 4;
constexpr int overlap_rounds = 16;

struct Overlap {
    const wchar_t* path;
    volatile LONG arrived[overlap_rounds];
    volatile LONG opened;
};

// All threads spin for each other before every open, so each round enters the same function at once.
DWORD WINAPI overlap_thread(LPVOID parameter)
{
    auto* shared = static_cast<Overlap*>(parameter);
    for (int round = 0; round < overlap_rounds; ++round) {
        InterlockedIncrement(&shared->arrived[round]);
        const ULONGLONG begin = GetTickCount64();
        while (shared->arrived[round] < overlap_threads && GetTickCount64() - begin < 10000) YieldProcessor();
        const HANDLE file = CreateFileW(shared->path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (file == INVALID_HANDLE_VALUE) continue;
        CloseHandle(file);
        InterlockedIncrement(&shared->opened);
    }
    return 0;
}

int overlapping_opens()
{
    wchar_t path[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableW(L"X2W_LAUNCH_PROBE_FILE", path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return 64;
    static Overlap shared{};
    shared.path = path;
    HANDLE threads[overlap_threads]{};
    for (HANDLE& thread : threads) {
        thread = CreateThread(nullptr, 0, overlap_thread, &shared, 0, nullptr);
        if (!thread) return 65;
    }
    const DWORD waited = WaitForMultipleObjects(overlap_threads, threads, TRUE, 60000);
    constexpr int total = overlap_threads * overlap_rounds;
    std::printf("overlap opens %ld of %d\n", static_cast<long>(shared.opened), total);
    return waited == WAIT_OBJECT_0 && shared.opened == total ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "report";
    char from_env[16]{};
    if (argc == 1 && GetEnvironmentVariableA("X2W_LAUNCH_PROBE_MODE", from_env, sizeof(from_env))) mode = from_env;
    if (std::strcmp(mode, "overlap") == 0) return overlapping_opens();
    if (std::strcmp(mode, "report") == 0) {
        stream(stdout, "launch-probe stdout\n");
        stream(stderr, "launch-probe stderr\n");
        write_trap_log("x2w-trap launch-probe\n");
        return argc > 2 ? std::atoi(argv[2]) : 0;
    }
    if (std::strcmp(mode, "sleep") == 0) {
        stream(stdout, "launch-probe sleeping\n");
        Sleep(static_cast<DWORD>(argc > 2 ? std::atoi(argv[2]) : 30000));
        return 0;
    }
    if (std::strcmp(mode, "allocate") == 0) {
        const SIZE_T mib = static_cast<SIZE_T>(argc > 2 ? std::atoi(argv[2]) : 64);
        void* memory = VirtualAlloc(nullptr, mib << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!memory) {
            std::printf("allocate failed error=%lu\n", GetLastError());
            return 0;
        }
        std::memset(memory, 0x5a, mib << 20);
        std::printf("allocate ok %zu MiB\n", mib);
        VirtualFree(memory, 0, MEM_RELEASE);
        return 0;
    }
    if (std::strcmp(mode, "addresses") == 0) {
        // The highest address VirtualAlloc hands out for blocks of several sizes (the title's blocks come the same way).
        unsigned long long highest = 0;
        const SIZE_T sizes[] = {SIZE_T{64} << 10, SIZE_T{16} << 20, SIZE_T{1} << 30, SIZE_T{4} << 30};
        for (const SIZE_T bytes : sizes) {
            void* memory = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
            if (!memory) {
                std::printf("addresses failed error=%lu\n", GetLastError());
                return 0;
            }
            highest = (std::max)(highest, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(memory)) + bytes - 1);
        }
        std::printf("addresses highest=0x%llx\n", highest);
        return 0;
    }
    if (std::strcmp(mode, "spawn") == 0) {
        wchar_t self[MAX_PATH]{};
        if (!GetModuleFileNameW(nullptr, self, MAX_PATH)) return 65;
        spawn_second_process(self);
        return 0;
    }
    if (std::strcmp(mode, "noop") == 0) return 0;
    return 64;
}