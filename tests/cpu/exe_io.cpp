#include <windows.h>
#include <cstdint>

static DWORD WINAPI hold_worker(void* ready)
{
    HANDLE park = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!park)
        return 1;
    const DWORD result = SetEvent(ready) ? WaitForSingleObject(park, INFINITE) : 1;
    CloseHandle(park);
    return result;
}

static DWORD WINAPI timed_worker(void* ready)
{
    // Signals readiness, outlives the entry body by about two seconds, then ends on its own with code 0.
    // If nothing terminates the process first, that code becomes the process exit status.
    if (!SetEvent(ready))
        return 1;
    Sleep(2000);
    return 0;
}

static bool start_worker(LPTHREAD_START_ROUTINE routine, DWORD timeout_ms)
{
    // Wait for the worker to signal that it started, or report an error if it exits first.
    HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ready)
        return false;
    HANDLE worker = CreateThread(nullptr, 0, routine, ready, 0, nullptr);
    bool started = false;
    if (worker) {
        HANDLE handles[] = {ready, worker};
        started = WaitForMultipleObjects(2, handles, FALSE, timeout_ms) == WAIT_OBJECT_0;
        CloseHandle(worker);
    }
    CloseHandle(ready);
    return started;
}

static bool start_test_worker()
{
    // Used only by the process lifetime regression.
    return start_worker(hold_worker, INFINITE);
}

static bool prepare_test_worker(const wchar_t* flag)
{
    wchar_t value[2];
    return GetEnvironmentVariableW(flag, value, 2) != 1 ||
        value[0] != L'1' || start_test_worker();
}

extern "C" __declspec(dllexport) std::uint32_t start_live_worker()
{
    // Called by the live-worker provider variant. The worker is ready on return and ends by itself later.
    return start_worker(timed_worker, 10000);
}

extern "C" __declspec(dllexport) std::uint32_t read_selector()
{
    if (!prepare_test_worker(L"CPU_EXE_TEST_LIVE_WORKER")) {
        // Distinguish a failed test setup from empty stdin.
        ExitProcess(3);
    }
    BYTE input;
    DWORD count;
    if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), &input, 1, &count, nullptr) || count != 1)
        return UINT32_MAX;
    return input;
}

static bool write_bytes(DWORD stream, const void* data, DWORD size)
{
    DWORD written;
    return WriteFile(GetStdHandle(stream), data, size, &written, nullptr) && written == size;
}

extern "C" __declspec(dllexport) std::uint32_t emit_trace(const std::uint64_t* words)
{
    // The provider wrote its first two arguments and the result into the entry's buffer.
    return write_bytes(STD_ERROR_HANDLE, words, 16) && write_bytes(STD_OUTPUT_HANDLE, words + 2, 8);
}

extern "C" __declspec(dllexport) DWORD finish_exe(const void* image_base, DWORD code)
{
#ifdef FINISH_HANG
    // Only the cpu_exe_io_hang build: finish_exe never terminates, which simulates a stuck termination path.
    Sleep(INFINITE);
#endif
    // The console fixture uses UCRT stdin rather than read_selector.
    if (!prepare_test_worker(L"CPU_CRC_TEST_LIVE_WORKER"))
        code = 3;
    // End the process only when Windows launched this image: it is then both the process image and a
    // module in the loader's list. A pe_map main image is reported as the process image too (pe_map_run_main),
    // but the loader does not know it, so its host gets the code back instead of losing the process.
    HMODULE loaded = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(image_base), &loaded) &&
        loaded == static_cast<HMODULE>(const_cast<void*>(image_base)) && loaded == GetModuleHandleW(nullptr))
        ExitProcess(code);
    return code;
}
