// Checks src/startup_stubs/local_log.h without a GPU device: GetLastError-preserving env_on
// gating, the atomic saturating first_calls gate (also on a counter at the 32-bit wrap point),
// logf formatting/truncation/X2W_TRAP_LOG append,
// one X2W_TRAP_LOG open per module under 8 concurrent writers (in a child process), recovery
// after a first open into a missing directory or of a locked file fails (in child processes),
// probe_readable page truncation and PAGE_GUARD re-arming, readable/writable boundary checks, and that an
// unmapped range is refused by the probe and the guarded copies with zero first-chance exceptions.
#include "startup_stubs/local_log.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;
int cases = 0;

void check(bool cond, const char* name)
{
    ++cases;
    if (!cond) {
        std::printf("FAIL %s (GetLastError=%lu)\n", name, static_cast<unsigned long>(GetLastError()));
        std::fflush(stdout);
        ++failures;
    }
}

std::string read_file_bytes(const wchar_t* path)
{
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    std::string out;
    char buf[512];
    DWORD got = 0;
    while (ReadFile(file, buf, sizeof(buf), &got, nullptr) && got > 0) {
        out.append(buf, buf + got);
    }
    CloseHandle(file);
    return out;
}

void test_env_on()
{
    const char* const var_name = "X2W_TEST_LOCAL_LOG_SWITCH";

    SetEnvironmentVariableA(var_name, nullptr);
    SetLastError(0x12345678u);
    const bool unset_res = x2w_local::env_on(var_name);
    const DWORD unset_err = GetLastError();
    check(!unset_res && unset_err == 0x12345678u, "env_on unset returns false and preserves GetLastError");

    for (const char* val : {"", "0", "2", "01", " 1", "true", "on", "yes"}) {
        SetEnvironmentVariableA(var_name, val);
        SetLastError(0x0BAD0001u);
        const bool res = x2w_local::env_on(var_name);
        const DWORD err = GetLastError();
        check(!res && err == 0x0BAD0001u, "env_on value without a leading 1 returns false and preserves GetLastError");
    }

    for (const char* val : {"1", "10", "1x", "100"}) {
        SetEnvironmentVariableA(var_name, val);
        SetLastError(0x0CAFE002u);
        const bool on_res = x2w_local::env_on(var_name);
        const DWORD on_err = GetLastError();
        check(on_res && on_err == 0x0CAFE002u, "env_on leading 1 returns true and preserves GetLastError");
    }

    SetEnvironmentVariableA(var_name, nullptr);
}

void test_first_calls()
{
    LONG counter = 0;
    check(x2w_local::first_calls(counter, 3) && counter == 1, "first_calls 1/3");
    check(x2w_local::first_calls(counter, 3) && counter == 2, "first_calls 2/3");
    check(x2w_local::first_calls(counter, 3) && counter == 3, "first_calls 3/3");
    // Past the limit the counter stays where it is, so it can never wrap.
    check(!x2w_local::first_calls(counter, 3) && counter == 3, "first_calls 4/3 false, counter stays at 3");
    check(!x2w_local::first_calls(counter, 3) && counter == 3, "first_calls 5/3 false, counter stays at 3");

    LONG zero_limit = 0;
    check(!x2w_local::first_calls(zero_limit, 0) && zero_limit == 0, "first_calls limit=0 false");

    LONG neg_limit = 0;
    check(!x2w_local::first_calls(neg_limit, -1) && neg_limit == 0, "first_calls limit=-1 false");

    LONG shared = 0;
    std::atomic<int> accepted{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; ++i) {
                if (x2w_local::first_calls(shared, 10)) accepted.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();
    check(shared == 10 && accepted.load(std::memory_order_relaxed) == 10,
          "first_calls concurrent exactness, counter saturates at the limit");
}

// The gate the stand-ins used before: it counts every call, so after 2^31 calls the counter turns
// negative and every later call passes again. Used only by the --mutate wrap control.
bool counting_gate(volatile LONG& counter, LONG limit)
{
    return InterlockedIncrement(&counter) <= limit;
}

// A counter that has already reached the wrap point (instead of looping 2^31 times): no call may
// pass a gate whose first calls are long over.
void test_first_calls_wrap(bool counting)
{
    volatile LONG counter = 0x7FFFFFFE;
    int passed = 0;
    for (int i = 0; i < 16; ++i) {
        if (counting ? counting_gate(counter, 4) : x2w_local::first_calls(counter, 4)) ++passed;
    }
    check(passed == 0, "first_calls passes no call at the wrap point");
}

// Child process body: 8 threads x 2,000 lines through logf to the inherited %X2W_TRAP_LOG%. Every
// line must land intact and the module must open the log exactly once. With reopen (the negative
// control) the variable is respelled as the same file after the first line, so append_log takes its
// per-line open path; the lines stay intact and only the open count may fail (exit code 3).
int concurrency_child(bool reopen)
{
    constexpr int kThreads = 8;
    constexpr int kLines = 2000;
    const char* const filler = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    HANDLE null_err = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(null_err != INVALID_HANDLE_VALUE && SetStdHandle(STD_ERROR_HANDLE, null_err) != FALSE,
          "child sends stderr copies to NUL");

    wchar_t path[MAX_PATH]{};
    const DWORD path_len = GetEnvironmentVariableW(L"X2W_TRAP_LOG", path, MAX_PATH);
    check(path_len > 0 && path_len < MAX_PATH, "child inherits X2W_TRAP_LOG");
    if (!path_len || path_len >= MAX_PATH) return 1;
    check(x2w_trap::log_opens() == 0, "child starts with no trap log open");

    x2w_local::logf("main-start");
    if (reopen) {
        std::wstring respelled(path);
        const auto slash = respelled.rfind(L'\\');
        respelled.insert(slash == std::wstring::npos ? 0 : slash + 1, L".\\");
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", respelled.c_str());
    }

    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            ready.fetch_add(1);
            while (ready.load() < kThreads) SwitchToThread();
            for (int i = 0; i < kLines; ++i) x2w_local::logf("t%d-%05d-%s", t, i, filler);
        });
    }
    for (auto& th : threads) th.join();
    const LONG opens = x2w_trap::log_opens();

    const std::string text = read_file_bytes(path);
    std::vector<std::vector<bool>> seen(kThreads, std::vector<bool>(kLines, false));
    std::size_t lines = 0;
    std::size_t bad = 0;
    std::size_t duplicate = 0;
    bool first_ok = false;
    for (std::size_t pos = 0; pos < text.size();) {
        const std::size_t end = text.find("\r\n", pos);
        if (end == std::string::npos) {
            ++bad;
            break;
        }
        const std::string line = text.substr(pos, end - pos);
        pos = end + 2;
        if (lines++ == 0) {
            first_ok = line == "main-start";
            continue;
        }
        int t = -1;
        int i = -1;
        char expected[128];
        if (std::sscanf(line.c_str(), "t%d-%d-", &t, &i) != 2 || t < 0 || t >= kThreads || i < 0 || i >= kLines ||
            std::snprintf(expected, sizeof(expected), "t%d-%05d-%s", t, i, filler) <= 0 || line != expected) {
            ++bad;
            continue;
        }
        if (seen[t][i]) ++duplicate;
        seen[t][i] = true;
    }
    std::size_t missing = 0;
    for (const auto& row : seen)
        for (bool b : row) missing += b ? 0 : 1;
    const std::size_t want = 1 + static_cast<std::size_t>(kThreads) * kLines;
    check(first_ok, "concurrent log starts with the priming line");
    check(lines == want && bad == 0 && duplicate == 0 && missing == 0,
          "concurrent logf lines are all present and intact");

    check(DeleteFileW(path) != FALSE, "the kept trap log can be deleted while this module holds it");

    const int before = failures;
    check(opens == 1, "trap log is opened once per module");
    const bool open_failed = failures != before;
    std::printf("concurrency child: opens=%ld lines=%zu bad=%zu missing=%zu duplicate=%zu\n",
                static_cast<long>(opens), lines, bad, missing, duplicate);
    std::fflush(stdout);
    if (failures - (open_failed ? 1 : 0)) return 1;
    return open_failed ? 3 : 0;
}

// Runs this executable with the given arguments (sharing stdout) and returns its exit code.
DWORD run_self(const std::wstring& args)
{
    wchar_t exe[MAX_PATH]{};
    const DWORD exe_len = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (!exe_len || exe_len >= MAX_PATH) return MAXDWORD;
    std::wstring command = L"\"" + std::wstring(exe) + L"\" " + args;
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    std::fflush(stdout);
    if (!CreateProcessW(exe, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process))
        return MAXDWORD;
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = MAXDWORD;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return code;
}

// Runs concurrency_child in a fresh process (its module log state starts empty) on a temporary
// log, then removes the log. Returns the child's exit code.
DWORD test_concurrency(bool reopen)
{
    wchar_t temp_dir[MAX_PATH]{};
    const DWORD dir_len = GetTempPathW(MAX_PATH, temp_dir);
    check(dir_len > 0 && dir_len < MAX_PATH, "GetTempPathW succeeds for the concurrent log");
    if (!dir_len || dir_len >= MAX_PATH) return MAXDWORD;
    const auto folder = std::wstring(temp_dir) + L"x2w_trap_log_" + std::to_wstring(GetCurrentProcessId());
    const auto log_path = folder + L"\\trap.log";
    if (log_path.size() + 2 >= MAX_PATH || !CreateDirectoryW(folder.c_str(), nullptr)) {
        check(false, "create the concurrent log folder");
        return MAXDWORD;
    }
    wchar_t prev[MAX_PATH]{};
    const DWORD prev_len = GetEnvironmentVariableW(L"X2W_TRAP_LOG", prev, MAX_PATH);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    const DWORD code = run_self(reopen ? L"--phase concurrency-reopen" : L"--phase concurrency");
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", prev_len > 0 && prev_len < MAX_PATH ? prev : nullptr);
    DeleteFileW(log_path.c_str());
    check(RemoveDirectoryW(folder.c_str()) != FALSE, "clean up the concurrent log folder");
    return code;
}

// Child process body for the open-failure recovery cases, on the inherited %X2W_TRAP_LOG% whose
// directory does not exist yet. missing: the first opens fail, the directory is created, and a line
// after the retry interval must land. locked: the file is held without sharing, then released.
// trap: right after a failed open, a flush (trap) line must still try and land. The recovery check
// is the one a module that keeps its failed open forever fails; when it is the child's only failure
// the exit code is 3 (used by startup_local_log_cache_failure_control).
enum class Recovery { missing, locked, trap };
int recovery_child(Recovery kind)
{
    HANDLE null_err = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(null_err != INVALID_HANDLE_VALUE && SetStdHandle(STD_ERROR_HANDLE, null_err) != FALSE,
          "recovery child sends stderr copies to NUL");
    wchar_t path[MAX_PATH]{};
    const DWORD path_len = GetEnvironmentVariableW(L"X2W_TRAP_LOG", path, MAX_PATH);
    check(path_len > 0 && path_len < MAX_PATH, "recovery child inherits X2W_TRAP_LOG");
    if (!path_len || path_len >= MAX_PATH) return 1;
    const std::wstring file(path);
    const std::wstring dir = file.substr(0, file.rfind(L'\\'));
    check(GetFileAttributesW(dir.c_str()) == INVALID_FILE_ATTRIBUTES, "the log directory does not exist yet");
    check(x2w_trap::log_opens() == 0, "recovery child starts with no trap log open");

    HANDLE exclusive = INVALID_HANDLE_VALUE;
    if (kind == Recovery::locked) {
        check(CreateDirectoryW(dir.c_str(), nullptr) != FALSE, "create the log directory");
        exclusive = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        check(exclusive != INVALID_HANDLE_VALUE, "hold the log file without sharing");
    }
    x2w_local::logf("lost-first");
    check(x2w_trap::log_opens() == 1, "the first line attempts one open");
    check(read_file_bytes(path).empty(), "the line of a failed open is dropped");

    if (kind == Recovery::trap) {
        check(CreateDirectoryW(dir.c_str(), nullptr) != FALSE, "create the log directory");
        x2w_trap::append_log("trap-line\r\n", 11, true);  // inside the retry wait: a trap line still tries
        x2w_local::logf("after-trap");
        check(read_file_bytes(path) == "trap-line\r\nafter-trap\r\n", "a trap line opens the log during the retry wait");
        check(x2w_trap::log_opens() == 2, "the trap line's open is kept");
    } else {
        // A path that stays unusable costs at most one open attempt per retry interval.
        const ULONGLONG start = GetTickCount64();
        int lines = 0;
        while (GetTickCount64() - start < 1000) x2w_local::logf("lost-%d", lines++);
        const ULONGLONG elapsed = GetTickCount64() - start;
        const LONG attempts = x2w_trap::log_opens() - 1;
        std::printf("recovery child: %d lines on an unusable path in %llu ms took %ld open attempts\n", lines,
                    static_cast<unsigned long long>(elapsed), static_cast<long>(attempts));
        check(attempts >= 0 && static_cast<ULONGLONG>(attempts) <= elapsed / x2w_trap::log_retry_ms + 1,
              "an unusable path costs at most one open attempt per retry interval");

        if (kind == Recovery::missing) check(CreateDirectoryW(dir.c_str(), nullptr) != FALSE, "create the log directory");
        if (exclusive != INVALID_HANDLE_VALUE) CloseHandle(exclusive);
        Sleep(static_cast<DWORD>(x2w_trap::log_retry_ms) + 100);
        x2w_local::logf("recovered");
        const LONG opens_after_recovery = x2w_trap::log_opens();
        const int before = failures;
        check(read_file_bytes(path) == "recovered\r\n", "a later line reaches the log once the path is usable");
        const bool recovery_failed = failures != before;
        if (!recovery_failed) {
            std::string expected = "recovered\r\n";
            for (int i = 0; i < 100; ++i) {
                x2w_local::logf("after-%d", i);
                expected += "after-" + std::to_string(i) + "\r\n";
            }
            check(read_file_bytes(path) == expected, "lines after recovery land in order");
            check(x2w_trap::log_opens() == opens_after_recovery, "the recovered path is opened once and kept");
        }
        std::fflush(stdout);
        if (failures - (recovery_failed ? 1 : 0)) return 1;
        return recovery_failed ? 3 : 0;
    }
    std::fflush(stdout);
    return failures ? 1 : 0;
}

// Runs recovery_child in a fresh process with %X2W_TRAP_LOG% inside a directory that does not exist,
// then removes what the child created. Returns the child's exit code.
DWORD test_recovery(const wchar_t* phase)
{
    wchar_t temp_dir[MAX_PATH]{};
    const DWORD dir_len = GetTempPathW(MAX_PATH, temp_dir);
    check(dir_len > 0 && dir_len < MAX_PATH, "GetTempPathW succeeds for the recovery log");
    if (!dir_len || dir_len >= MAX_PATH) return MAXDWORD;
    const auto folder = std::wstring(temp_dir) + L"x2w_trap_recovery_" + std::to_wstring(GetCurrentProcessId());
    const auto sub = folder + L"\\later";
    const auto log_path = sub + L"\\trap.log";
    if (log_path.size() + 2 >= MAX_PATH || !CreateDirectoryW(folder.c_str(), nullptr)) {
        check(false, "create the recovery log folder");
        return MAXDWORD;
    }
    wchar_t prev[MAX_PATH]{};
    const DWORD prev_len = GetEnvironmentVariableW(L"X2W_TRAP_LOG", prev, MAX_PATH);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    const DWORD code = run_self(std::wstring(L"--phase ") + phase);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", prev_len > 0 && prev_len < MAX_PATH ? prev : nullptr);
    DeleteFileW(log_path.c_str());
    RemoveDirectoryW(sub.c_str());
    check(RemoveDirectoryW(folder.c_str()) != FALSE, "clean up the recovery log folder");
    return code;
}

void test_logf()
{
    wchar_t prev_trap_log[MAX_PATH]{};
    const DWORD prev_len = GetEnvironmentVariableW(L"X2W_TRAP_LOG", prev_trap_log, MAX_PATH);

    wchar_t temp_dir[MAX_PATH]{};
    const DWORD dir_len = GetTempPathW(MAX_PATH, temp_dir);
    check(dir_len > 0 && dir_len < MAX_PATH, "GetTempPathW succeeds");
    if (!dir_len || dir_len >= MAX_PATH) return;
    const auto folder = std::wstring(temp_dir) + L"x2w_local_log_" + std::to_wstring(GetCurrentProcessId());
    const auto log_path = folder + L"\\trap.log";
    if (log_path.size() >= MAX_PATH) {
        check(false, "temporary folder has space for the log names");
        return;
    }
    if (!CreateDirectoryW(folder.c_str(), nullptr)) {
        check(false, "create temporary log folder");
        return;
    }
    const auto err_path = folder + L"\\stderr";
    const auto max_name = MAX_PATH - 2 - folder.size();
    const auto fit_path = folder + L"\\" + std::wstring(max_name, L'f');
    const auto long_path = folder + L"\\" + std::wstring(max_name + 1, L'g');
    // Keep this module's log handle on NUL: the paths below then take append_log's per-line path for
    // a changed variable (the behaviour before the handle was kept), and their files stay removable.
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", L"NUL");
    const LONG opens_before = x2w_trap::log_opens();
    x2w_trap::append_log("kept\r\n", 6, false);
    x2w_trap::append_log("kept\r\n", 6, true);
    check(x2w_trap::log_opens() - opens_before == 1, "append_log reuses the kept handle");
    const HANDLE saved_stderr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE stderr_file = CreateFileW(err_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                                     CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    const bool captured = stderr_file != INVALID_HANDLE_VALUE &&
                          SetStdHandle(STD_ERROR_HANDLE, stderr_file) != FALSE;
    check(captured, "capture the stderr stream");

    SetEnvironmentVariableW(L"X2W_TRAP_LOG", nullptr);
    SetLastError(0x13572468u);
    x2w_local::logf("unset-path-%d", 1);
    check(GetLastError() == 0x13572468u, "logf preserves GetLastError when X2W_TRAP_LOG is unset");

    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    SetLastError(0x0BADF00Du);
    x2w_local::logf("first-%d", 42);
    check(GetLastError() == 0x0BADF00Du, "logf preserves GetLastError when X2W_TRAP_LOG is set");

    x2w_local::logf("second-%s", "line");

    const std::string two_lines = read_file_bytes(log_path.c_str());
    check(two_lines == "first-42\r\nsecond-line\r\n", "logf appends CRLF lines to X2W_TRAP_LOG");

    std::string expected_log = "first-42\r\nsecond-line\r\n";
    for (std::size_t length : {1532, 1533, 1534, 2000}) {
        const std::string message(length, 'A');
        x2w_local::logf("%s", message.c_str());
        expected_log += message.substr(0, 1533) + "\r\n";
    }
    check(read_file_bytes(log_path.c_str()) == expected_log, "logf clamps only beyond 1533 payload bytes");

    SetEnvironmentVariableW(L"X2W_TRAP_LOG", fit_path.c_str());
    x2w_local::logf("path at limit");
    check(read_file_bytes(fit_path.c_str()) == "path at limit\r\n", "logf accepts a path of MAX_PATH minus one");
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", long_path.c_str());
    x2w_local::logf("path over limit");
    check(read_file_bytes(log_path.c_str()) == expected_log &&
              GetFileAttributesW(long_path.substr(0, MAX_PATH - 1).c_str()) == INVALID_FILE_ATTRIBUTES,
          "logf leaves prior logs alone and declines an overlong path");
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", folder.c_str());
    SetLastError(0x13572468u);
    x2w_local::logf("path is a directory");
    check(GetLastError() == 0x13572468u, "logf preserves the last error on open failure");

    if (prev_len > 0 && prev_len < MAX_PATH) {
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", prev_trap_log);
    } else {
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", nullptr);
    }
    if (captured) {
        SetStdHandle(STD_ERROR_HANDLE, saved_stderr);
    }
    if (stderr_file != INVALID_HANDLE_VALUE) CloseHandle(stderr_file);
    check(read_file_bytes(err_path.c_str()) == "unset-path-1\r\n" + expected_log +
              "path at limit\r\npath over limit\r\npath is a directory\r\n",
          "stderr still receives every formatted line");
    for (const auto* path : {&err_path, &log_path, &fit_path}) DeleteFileW(path->c_str());
    check(RemoveDirectoryW(folder.c_str()) != FALSE, "clean up only the expected files");
}

void test_probe_and_access(bool guard_drop_mutate)
{
    constexpr std::size_t kPage = 0x1000;
    constexpr std::size_t kPages = 4;

    void* const raw = VirtualAlloc(nullptr, kPages * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    check(raw != nullptr, "VirtualAlloc 4 pages succeeds");
    if (!raw) return;

    const auto base = reinterpret_cast<std::uintptr_t>(raw);
    std::memset(raw, 0x5A, kPages * kPage);

    // Invalid addresses and zero-byte lengths.
    check(x2w_local::probe_readable(0, 16) == 0, "probe_readable rejects null");
    check(x2w_local::probe_readable(0xFFFFu, 16) == 0, "probe_readable rejects < 0x10000");
    check(x2w_local::probe_readable(0x00007FFFFFFF0000ull, 16) == 0, "probe_readable rejects > user limit");
    check(x2w_local::probe_readable(base, 0) == 0, "probe_readable rejects 0 bytes");

    check(!x2w_local::readable(nullptr, 1) && !x2w_local::writable(nullptr, 1), "accessible rejects null");
    const auto low_ptr = reinterpret_cast<const void*>(std::uintptr_t{0xFFFF});
    const auto high_ptr = reinterpret_cast<const void*>(std::uintptr_t{0x00007FFFFFFF0000ull});
    check(!x2w_local::readable(low_ptr, 1) && !x2w_local::writable(low_ptr, 1), "accessible rejects < 0x10000");
    check(!x2w_local::readable(high_ptr, 1) && !x2w_local::writable(high_ptr, 1), "accessible rejects > user limit");
    check(!x2w_local::readable(raw, 0) && !x2w_local::writable(raw, 0), "accessible rejects 0 bytes");

    // Multi-page readable range and unaligned slice.
    check(x2w_local::probe_readable(base, kPages * kPage) == kPages * kPage, "probe_readable full 4 pages");
    check(x2w_local::probe_readable(base + 0xF00u, 0x2200u) == 0x2200u, "probe_readable unaligned across pages");
    check(x2w_local::readable(raw, kPages * kPage) && x2w_local::writable(raw, kPages * kPage),
          "4-page RW region is readable and writable");

    // A size so large that start + bytes wraps around the address space is not a range.
    check(!x2w_local::readable(raw, SIZE_MAX - 8) && !x2w_local::writable(raw, SIZE_MAX - 8),
          "accessible rejects a size that wraps start + bytes");
    check(!x2w_local::writable(raw, SIZE_MAX), "writable rejects SIZE_MAX bytes");
    check(!x2w_local::writable(reinterpret_cast<const void*>(base + kPage - 1), SIZE_MAX - kPage),
          "writable rejects a wrapping size from inside the region");

    // PAGE_READONLY on page 1: readable across page 0 -> page 1, writable stops at page 0 end.
    DWORD old_prot = 0;
    check(VirtualProtect(reinterpret_cast<void*>(base + kPage), kPage, PAGE_READONLY, &old_prot) != 0,
          "VirtualProtect page 1 PAGE_READONLY");
    check(x2w_local::readable(reinterpret_cast<const void*>(base + kPage), kPage), "PAGE_READONLY is readable");
    check(!x2w_local::writable(reinterpret_cast<const void*>(base + kPage), 1), "PAGE_READONLY is not writable");
    check(x2w_local::readable(reinterpret_cast<const void*>(base + 0xFF0u), 32),
          "readable crosses RW->RO page boundary");
    check(x2w_local::writable(reinterpret_cast<const void*>(base + 0xFF0u), 16),
          "writable accepts up to RW region end");
    check(!x2w_local::writable(reinterpret_cast<const void*>(base + 0xFF0u), 17),
          "writable rejects crossing RW->RO region boundary");

    // PAGE_WRITECOPY via a section copy-on-write mapping: both readable and writable.
    HANDLE cow_map = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(kPage), nullptr);
    check(cow_map != nullptr, "CreateFileMappingW for PAGE_WRITECOPY succeeds");
    if (cow_map) {
        void* const cow_view = MapViewOfFile(cow_map, FILE_MAP_COPY, 0, 0, kPage);
        check(cow_view != nullptr, "MapViewOfFile FILE_MAP_COPY succeeds");
        if (cow_view) {
            MEMORY_BASIC_INFORMATION cow_mbi{};
            check(VirtualQuery(cow_view, &cow_mbi, sizeof(cow_mbi)) != 0 &&
                  (cow_mbi.Protect & 0xFF) == PAGE_WRITECOPY,
                  "MapViewOfFile FILE_MAP_COPY has PAGE_WRITECOPY protection");
            check(x2w_local::readable(cow_view, kPage), "PAGE_WRITECOPY is readable");
            check(x2w_local::writable(cow_view, kPage), "PAGE_WRITECOPY is writable");
            UnmapViewOfFile(cow_view);
        }
        CloseHandle(cow_map);
    }

    // PAGE_NOACCESS on page 2: truncates probe_readable prefix cleanly.
    check(VirtualProtect(reinterpret_cast<void*>(base + 2 * kPage), kPage, PAGE_NOACCESS, &old_prot) != 0,
          "VirtualProtect page 2 PAGE_NOACCESS");
    check(x2w_local::probe_readable(base, kPages * kPage) == 2 * kPage,
          "probe_readable truncates before PAGE_NOACCESS");
    check(x2w_local::probe_readable(base + 0x123u, kPages * kPage - 0x123u) == 2 * kPage - 0x123u,
          "probe_readable unaligned truncates before PAGE_NOACCESS");
    check(x2w_local::probe_readable(base + 2 * kPage, kPage) == 0,
          "probe_readable on PAGE_NOACCESS returns 0");
    check(!x2w_local::readable(reinterpret_cast<const void*>(base + 2 * kPage), 1) &&
          !x2w_local::writable(reinterpret_cast<const void*>(base + 2 * kPage), 1),
          "PAGE_NOACCESS is neither readable nor writable");
    check(!x2w_local::readable(reinterpret_cast<const void*>(base + 2 * kPage - 16), 17),
          "readable rejects crossing into PAGE_NOACCESS");

    // PAGE_GUARD on page 1: probe stops before guard page and re-arms PAGE_GUARD.
    check(VirtualProtect(reinterpret_cast<void*>(base + kPage), kPage, PAGE_READWRITE | PAGE_GUARD, &old_prot) != 0,
          "VirtualProtect page 1 PAGE_GUARD");
    check(x2w_local::probe_readable(base, 2 * kPage) == kPage,
          "probe_readable stops at PAGE_GUARD page");
    MEMORY_BASIC_INFORMATION mbi{};
    check(VirtualQuery(reinterpret_cast<const void*>(base + kPage), &mbi, sizeof(mbi)) != 0,
          "VirtualQuery after first PAGE_GUARD probe");
    if (guard_drop_mutate) {
        check((mbi.Protect & PAGE_GUARD) == 0, "guard-drop control expects PAGE_GUARD cleared");
    } else {
        check((mbi.Protect & PAGE_GUARD) != 0, "probe_readable re-arms PAGE_GUARD after trip");
        // Second probe must hit the re-armed guard page at the same offset and keep it armed.
        check(x2w_local::probe_readable(base, 2 * kPage) == kPage,
              "second probe_readable still stops at re-armed PAGE_GUARD");
        MEMORY_BASIC_INFORMATION mbi2{};
        check(VirtualQuery(reinterpret_cast<const void*>(base + kPage), &mbi2, sizeof(mbi2)) != 0 &&
              (mbi2.Protect & PAGE_GUARD) != 0,
              "second probe_readable keeps PAGE_GUARD armed");
        check(!x2w_local::readable(reinterpret_cast<const void*>(base + kPage + 0x10u), 16),
              "readable returns false on PAGE_GUARD");
        check(!x2w_local::writable(reinterpret_cast<const void*>(base + kPage), 16),
              "writable returns false on PAGE_GUARD");
        MEMORY_BASIC_INFORMATION mbi3{};
        check(VirtualQuery(reinterpret_cast<const void*>(base + kPage), &mbi3, sizeof(mbi3)) != 0 &&
              (mbi3.Protect & PAGE_GUARD) != 0,
              "readable/writable keeps PAGE_GUARD armed");
    }

    VirtualFree(raw, 0, MEM_RELEASE);

    // MEM_RESERVE (uncommitted) tail boundary.
    void* const reserved = VirtualAlloc(nullptr, 2 * kPage, MEM_RESERVE, PAGE_READWRITE);
    check(reserved != nullptr, "VirtualAlloc MEM_RESERVE 2 pages succeeds");
    if (reserved) {
        void* const first_committed = VirtualAlloc(reserved, kPage, MEM_COMMIT, PAGE_READWRITE);
        check(first_committed == reserved, "commit first page of reserved range");
        const auto rbase = reinterpret_cast<std::uintptr_t>(reserved);
        check(x2w_local::probe_readable(rbase, 2 * kPage) == kPage,
              "probe_readable truncates before MEM_RESERVE");
        check(x2w_local::readable(reserved, kPage) && x2w_local::writable(reserved, kPage),
              "committed head of reserved range is readable and writable");
        check(!x2w_local::readable(reinterpret_cast<const void*>(rbase + kPage), 1) &&
              !x2w_local::writable(reinterpret_cast<const void*>(rbase + kPage), 1),
              "MEM_RESERVE page is neither readable nor writable");
        VirtualFree(reserved, 0, MEM_RELEASE);
    }
}

// Exceptions seen by a vectored handler, for the thread that runs the check only: a debugger stops on every
// first-chance fault, and faults of other threads in the process do not belong to it.
volatile LONG first_chance_faults = 0;
volatile LONG counted_thread = 0;
LONG CALLBACK count_first_chance(PEXCEPTION_POINTERS info)
{
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (GetCurrentThreadId() == static_cast<DWORD>(counted_thread) &&
        (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR || code == 0x80000001u))
        InterlockedIncrement(&first_chance_faults);
    return EXCEPTION_CONTINUE_SEARCH;
}

bool range_is_free(std::uintptr_t base, std::size_t bytes)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(reinterpret_cast<const void*>(base), &info, sizeof(info)) && info.State == MEM_FREE &&
           info.RegionSize >= bytes;
}

// Free address space is refused with zero first-chance exceptions by the probe, readable() and both copy flavours,
// each counted in faults_avoided; a mapped source still copies and counts nothing.
//
// The range is one this check released itself. Nothing else can be trusted to leave it free (another thread, or
// this thread's own heap, may reuse it the moment it is released), so each round is only judged when the range is
// verified MEM_FREE right before and right after the calls, the handler is installed beforehand and the calls
// in between allocate nothing. A round that lost its range is dropped, not counted; the check needs enough stable
// rounds and holds every stable one to zero faults.
void test_unmapped_no_faults()
{
    constexpr std::size_t kPage = 0x1000;
    constexpr int rounds = 40, needed = 10;
    char copy[64] = {};
    {
        void* const mapped = VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        check(mapped != nullptr, "VirtualAlloc a page for the mapped-source check");
        if (!mapped) return;
        std::memset(mapped, 0x5A, kPage);
        x2w_local::forget_mapped_spans();
        const LONG before = x2w_local::faults_avoided;
        const bool copied = x2w_local::guarded_memcpy(copy, mapped, sizeof(copy), false);
        check(copied && copy[0] == 0x5A, "a mapped source is copied");
        check(x2w_local::faults_avoided == before, "a mapped source avoids no fault");
        VirtualFree(mapped, 0, MEM_RELEASE);
    }
    counted_thread = static_cast<LONG>(GetCurrentThreadId());
    void* const handler = AddVectoredExceptionHandler(1, count_first_chance);
    check(handler != nullptr, "vectored handler installed");
    int stable = 0, wrong_results = 0, faulted = 0, miscounted = 0, control_faults = 0, controls = 0;
    for (int round = 0; round < rounds && stable < needed; ++round) {
        void* const block = VirtualAlloc(nullptr, 2 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!block) continue;
        const auto base = reinterpret_cast<std::uintptr_t>(block);
        VirtualFree(block, 0, MEM_RELEASE);
        x2w_local::forget_mapped_spans();
        if (!range_is_free(base, 2 * kPage)) continue;
        InterlockedExchange(&first_chance_faults, 0);
        const LONG before = x2w_local::faults_avoided;
        const bool probe_zero = x2w_local::probe_readable(base, 16) == 0;
        const bool not_readable = !x2w_local::readable(reinterpret_cast<const void*>(base), 16);
        const bool refused_av = !x2w_local::guarded_memcpy(copy, reinterpret_cast<const void*>(base), sizeof(copy), false);
        const bool refused_any = !x2w_local::guarded_memcpy(copy, reinterpret_cast<const void*>(base), sizeof(copy), true);
        const LONG faults = first_chance_faults;
        const LONG counted = x2w_local::faults_avoided - before;
        if (!range_is_free(base, 2 * kPage)) continue;  // the range was taken meanwhile: this round proves nothing
        ++stable;
        if (!(probe_zero && not_readable && refused_av && refused_any)) ++wrong_results;
        if (faults != 0) ++faulted;
        if (counted != 4) ++miscounted;
        if (stable == 1) {
            // Control: the handler does see a real fault, so the zero above means something. Touch a freshly
            // released range directly, again only when it is verified free around the touch.
            void* const second = VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (second) {
                const auto at = reinterpret_cast<std::uintptr_t>(second);
                VirtualFree(second, 0, MEM_RELEASE);
                if (range_is_free(at, kPage)) {
                    InterlockedExchange(&first_chance_faults, 0);
                    __try {
                        volatile unsigned char sink = *reinterpret_cast<const volatile unsigned char*>(at);
                        (void)sink;
                    } __except (EXCEPTION_EXECUTE_HANDLER) {
                    }
                    const LONG seen = first_chance_faults;
                    if (range_is_free(at, kPage)) {
                        ++controls;
                        control_faults = seen;
                    }
                }
            }
        }
    }
    RemoveVectoredExceptionHandler(handler);
    check(stable >= needed, "enough rounds kept their released range free to be judged");
    check(wrong_results == 0, "free address space: probe_readable 0, readable false, both copies refuse");
    check(faulted == 0, "free address space: zero first-chance exceptions in every judged round");
    check(miscounted == 0, "free address space: the four refusals of every judged round are counted");
    check(controls == 1 && control_faults == 1, "control: touching a verified-free range directly is one first-chance fault");
}

}  // namespace

int main(int argc, char** argv)
{
    const bool two_args = argc == 3;
    if (two_args && std::strcmp(argv[1], "--phase") == 0) {
        if (std::strcmp(argv[2], "concurrency") == 0) return concurrency_child(false);
        if (std::strcmp(argv[2], "concurrency-reopen") == 0) return concurrency_child(true);
        if (std::strcmp(argv[2], "recovery-missing") == 0) return recovery_child(Recovery::missing);
        if (std::strcmp(argv[2], "recovery-locked") == 0) return recovery_child(Recovery::locked);
        if (std::strcmp(argv[2], "recovery-trap") == 0) return recovery_child(Recovery::trap);
    }
    // --mutate guard-drop: negative control expecting PAGE_GUARD to be dropped rather than re-armed.
    const bool guard_drop_mutate =
        two_args && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "guard-drop") == 0;
    // --mutate reopen: negative control whose child reopens the log per line. It exits 1 only when
    // the open-count check is the child's sole failure (child exit code 3), so any other fault in
    // that run (lost or torn lines, a crash, a child that never started) makes the control fail.
    const bool reopen_mutate =
        two_args && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "reopen") == 0;
    // --mutate cache-failure: negative control, run by a build whose append_log keeps a failed open
    // forever (a generated copy of trap.h). The whole check runs; it exits 1 only when the two
    // recovery cases fail on their recovery check alone (child exit 3) and nothing else fails, so the
    // unmodified build, or any other fault, makes the control fail.
    const bool cache_failure_mutate =
        two_args && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "cache-failure") == 0;
    // --mutate wrap: negative control running only the wrap-point case with the counting gate.
    const bool wrap_mutate = two_args && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "wrap") == 0;
    if (argc > 1 && !guard_drop_mutate && !reopen_mutate && !cache_failure_mutate && !wrap_mutate) {
        std::printf("unknown mutation\n");
        return 2;
    }
    if (reopen_mutate) {
        const DWORD code = test_concurrency(true);
        std::printf("startup_local_log reopen control: child exit %lu, parent failures %d\n",
                    static_cast<unsigned long>(code), failures);
        return code == 3 && failures == 0 ? 1 : 0;
    }

    if (wrap_mutate) {
        test_first_calls_wrap(true);
        std::printf("startup_local_log: wrap control %d of %d FAILED\n", failures, cases);
        return failures ? 1 : 0;
    }
    test_env_on();
    test_first_calls();
    test_first_calls_wrap(false);
    const DWORD child = test_concurrency(false);
    check(child == 0, "concurrent logging child passes");
    const DWORD want = cache_failure_mutate ? 3 : 0;
    const DWORD missing = test_recovery(L"recovery-missing");
    check(missing == want, "recovery after a missing directory child passes");
    const DWORD locked = test_recovery(L"recovery-locked");
    check(locked == want, "recovery after a locked file child passes");
    check(test_recovery(L"recovery-trap") == 0, "trap line during the retry wait child passes");
    std::printf("recovery children: missing exit %lu, locked exit %lu\n", static_cast<unsigned long>(missing),
                static_cast<unsigned long>(locked));
    test_logf();
    test_probe_and_access(guard_drop_mutate);
    test_unmapped_no_faults();

    if (cache_failure_mutate) {
        std::printf("startup_local_log cache-failure control: parent failures %d\n", failures);
        return failures == 0 ? 1 : 0;
    }
    if (failures) {
        std::printf("startup_local_log: %d of %d FAILED\n", failures, cases);
        return 1;
    }
    std::printf("startup_local_log: PASS (%d cases)\n", cases);
    return 0;
}
