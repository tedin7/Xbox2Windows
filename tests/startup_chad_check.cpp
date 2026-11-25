// Lifecycle, permissive pointer validation, and trap checks for the chad.dll startup stand-in.
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
#include <string>

namespace {

using u64 = std::uint64_t;
using ActivateFn = long(__cdecl*)(u64, u64, u64, u64);
using ConnectFn = long(__cdecl*)(u64, u64, u64, u64, u64, void**);
using DeactivateFn = long(__cdecl*)(u64, u64, u64, u64);
using DisconnectFn = long(__cdecl*)(u64, u64, u64, u64);
using SubmitCommandFn = long(__cdecl*)(u64, u64, u64, u64);

constexpr long hr_ok = 0;
constexpr long hr_pointer = static_cast<long>(0x80004003);  // E_POINTER
constexpr long hr_invalidarg = static_cast<long>(0x80070057);  // E_INVALIDARG
constexpr long hr_no_device_resources = static_cast<long>(0x8AC810A2);  // XDSP_E_NO_DEVICE_RESOURCES
constexpr DWORD trap_activate_exit = 0xE2570200u;
constexpr DWORD trap_connect_exit = 0xE2570201u;
constexpr DWORD trap_deactivate_exit = 0xE2570202u;
constexpr DWORD trap_disconnect_exit = 0xE2570203u;
constexpr DWORD trap_submit_exit = 0xE2570204u;

int failures = 0;

void check(bool condition, const char* label)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL: %s\n", label);
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

int run_trap_control(const char* dll_path, const char* export_name, bool null_handle)
{
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", nullptr);
    SetEnvironmentVariableA("X2W_TRAP_CONTINUE", nullptr);
    SetEnvironmentVariableA("X2W_TRAP_FAIL", nullptr);
    HMODULE module = LoadLibraryA(dll_path);
    if (!module) return 2;
    FARPROC proc = GetProcAddress(module, export_name);
    if (!proc) return 3;
    if (std::strcmp(export_name, "XDspActivate") == 0) {
        reinterpret_cast<ActivateFn>(proc)(0x11, 0x22, 0x33, 0x44);
    } else if (std::strcmp(export_name, "XDspConnectWithMaximumStreamLimit") == 0) {
        void* handle = nullptr;
        reinterpret_cast<ConnectFn>(proc)(0x1000, 0x60000, 30, 0x80, 1, null_handle ? nullptr : &handle);
    } else if (std::strcmp(export_name, "XDspDeactivate") == 0) {
        reinterpret_cast<DeactivateFn>(proc)(0, 0, 0, 0);
    } else if (std::strcmp(export_name, "XDspDisconnect") == 0) {
        reinterpret_cast<DisconnectFn>(proc)(0, 0, 0, 0);
    } else if (std::strcmp(export_name, "XDspSubmitCommand") == 0) {
        reinterpret_cast<SubmitCommandFn>(proc)(0, 0, 0, 0);
    } else {
        return 5;
    }
    return 4;
}

void check_trap_case(const char* self_exe, const char* dll_path, const char* export_name, bool null_handle,
                     DWORD expected_exit)
{
    std::string cmd = std::string("\"") + self_exe + "\" --trap-control \"" + dll_path + "\" " + export_name +
                      (null_handle ? " null" : " valid");
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        check(false, "CreateProcessA for non-permissive trap control");
        return;
    }
    const DWORD wait_rc = WaitForSingleObject(pi.hProcess, 10000);
    DWORD exit_code = 0;
    const BOOL got_code = GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    check(wait_rc == WAIT_OBJECT_0, "non-permissive trap child exited within timeout");
    if (!got_code || exit_code != expected_exit) {
        std::printf("FAIL: %s trap expected 0x%08lx got 0x%08lx\n", export_name,
                    static_cast<unsigned long>(expected_exit), static_cast<unsigned long>(exit_code));
        ++failures;
    }
}

void check_non_permissive_traps(const char* self_exe, const char* dll_path)
{
    check_trap_case(self_exe, dll_path, "XDspActivate", false, trap_activate_exit);
    check_trap_case(self_exe, dll_path, "XDspConnectWithMaximumStreamLimit", false, trap_connect_exit);
    check_trap_case(self_exe, dll_path, "XDspConnectWithMaximumStreamLimit", true, trap_connect_exit);
    check_trap_case(self_exe, dll_path, "XDspDeactivate", false, trap_deactivate_exit);
    check_trap_case(self_exe, dll_path, "XDspDisconnect", false, trap_disconnect_exit);
    check_trap_case(self_exe, dll_path, "XDspSubmitCommand", false, trap_submit_exit);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 5 && std::strcmp(argv[1], "--trap-control") == 0) {
        return run_trap_control(argv[2], argv[3], std::strcmp(argv[4], "null") == 0);
    }
    if (argc != 2) return std::printf("usage: startup_chad_check <chad.dll>\n"), 2;

    wchar_t prev_trap_log[MAX_PATH]{};
    const DWORD prev_len = GetEnvironmentVariableW(L"X2W_TRAP_LOG", prev_trap_log, MAX_PATH);

    wchar_t temp_dir[MAX_PATH]{};
    const DWORD dir_len = GetTempPathW(MAX_PATH, temp_dir);
    check(dir_len > 0 && dir_len < MAX_PATH, "GetTempPathW succeeds");
    std::wstring log_path;
    if (dir_len > 0 && dir_len < MAX_PATH) {
        log_path = std::wstring(temp_dir) + L"x2w_chad_check_" + std::to_wstring(GetCurrentProcessId()) + L".log";
        DeleteFileW(log_path.c_str());
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    }

    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) {
        if (!log_path.empty()) DeleteFileW(log_path.c_str());
        return std::printf("FAIL: cannot load %s\n", argv[1]), 2;
    }

    auto activate = reinterpret_cast<ActivateFn>(GetProcAddress(module, "XDspActivate"));
    auto connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XDspConnectWithMaximumStreamLimit"));
    auto deactivate = reinterpret_cast<DeactivateFn>(GetProcAddress(module, "XDspDeactivate"));
    auto disconnect = reinterpret_cast<DisconnectFn>(GetProcAddress(module, "XDspDisconnect"));
    auto submit = reinterpret_cast<SubmitCommandFn>(GetProcAddress(module, "XDspSubmitCommand"));
    if (!activate || !connect || !deactivate || !disconnect || !submit) {
        FreeLibrary(module);
        if (!log_path.empty()) DeleteFileW(log_path.c_str());
        return std::printf("FAIL: missing export\n"), 2;
    }

    // Negative controls in permissive mode: null and unwritable handle outputs return E_POINTER.
    check(connect(0x1000, 0x60000, 30, 0x80, 1, nullptr) == hr_pointer,
          "XDspConnectWithMaximumStreamLimit(null handle*) -> E_POINTER");

    if (void* ro = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY)) {
        check(connect(0x1000, 0x60000, 30, 0x80, 1, static_cast<void**>(ro)) == hr_pointer,
              "XDspConnectWithMaximumStreamLimit(PAGE_READONLY handle*) -> E_POINTER");
        VirtualFree(ro, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc PAGE_READONLY for negative control");
    }

    if (void* pages = VirtualAlloc(nullptr, 0x2000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) {
        DWORD old_prot = 0;
        if (VirtualProtect(static_cast<unsigned char*>(pages) + 0x1000, 0x1000, PAGE_READONLY, &old_prot)) {
            auto** edge = reinterpret_cast<void**>(static_cast<unsigned char*>(pages) + 0x1000 - 4);
            check(connect(0x1000, 0x60000, 30, 0x80, 1, edge) == hr_pointer,
                  "XDspConnectWithMaximumStreamLimit(straddling RW->RO handle*) -> E_POINTER");
        } else {
            check(false, "VirtualProtect straddling negative control");
        }
        VirtualFree(pages, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc straddling negative control");
    }

    // XDspActivate reports the documented failure and clears both outputs; the caller below gates its
    // first submit on activation success, so no submit happens and the sequence keeps its sentinel.
    check(activate(0x11, 0x22, 0x33, 0x44) == hr_pointer, "XDspActivate(unwritable outputs) -> E_POINTER");
    void* status_out = reinterpret_cast<void*>(0x1);
    void* stream_out = reinterpret_cast<void*>(0x2);
    check(activate(0x11, 0x22, reinterpret_cast<u64>(&status_out), reinterpret_cast<u64>(&stream_out)) ==
              hr_no_device_resources && status_out == nullptr && stream_out == nullptr,
          "XDspActivate -> XDSP_E_NO_DEVICE_RESOURCES with both outputs cleared");
    check(activate(0x11, 0x22, 0, reinterpret_cast<u64>(&stream_out)) == hr_pointer,
          "XDspActivate(null status*) -> E_POINTER");
    check(activate(0x11, 0x22, reinterpret_cast<u64>(&status_out), 0) == hr_pointer,
          "XDspActivate(null stream*) -> E_POINTER");

    // The connect out-handle still succeeds and publishes a handle (#1871).
    void* handle = nullptr;
    check(connect(0x1000, 0x60000, 30, 0x80, 1, &handle) == hr_ok && handle != nullptr,
          "XDspConnectWithMaximumStreamLimit returns S_OK and populates handle");

    // Submit refuses null and unknown handles without returning a sequence, and a known handle still
    // cannot run a command, so the caller value survives every failure path.
    unsigned sequence = 0xdeadbeefu;
    check(submit(0, 0x1234, reinterpret_cast<u64>(&sequence), 0) == hr_invalidarg && sequence == 0xdeadbeefu,
          "XDspSubmitCommand(null handle) -> E_INVALIDARG, sequence preserved");
    check(submit(reinterpret_cast<u64>(handle) + 4, 0x1234, reinterpret_cast<u64>(&sequence), 0) == hr_invalidarg &&
              sequence == 0xdeadbeefu,
          "XDspSubmitCommand(unknown handle) -> E_INVALIDARG, sequence preserved");
    check(submit(reinterpret_cast<u64>(handle), 0x1234, reinterpret_cast<u64>(&sequence), 0) == hr_no_device_resources &&
              sequence == 0xdeadbeefu,
          "XDspSubmitCommand(known handle) -> XDSP_E_NO_DEVICE_RESOURCES, sequence preserved");

    // Synthetic caller: a title that gates its first submit on activation success never reaches submit.
    unsigned gated_sequence = 0;
    if (activate(0x11, 0x22, reinterpret_cast<u64>(&status_out), reinterpret_cast<u64>(&stream_out)) == hr_ok)
        submit(reinterpret_cast<u64>(handle), 0x1234, reinterpret_cast<u64>(&gated_sequence), 0);
    check(gated_sequence == 0, "caller gated on activation success never submits");

    // Deactivate and disconnect write nothing back; they refuse null and unknown handles instead of
    // reporting them as released, and release only the handle connect published.
    check(deactivate(0, 0, 0, 0) == hr_invalidarg, "XDspDeactivate(null handle) -> E_INVALIDARG");
    check(deactivate(reinterpret_cast<u64>(handle) + 4, 0, 0, 0) == hr_invalidarg,
          "XDspDeactivate(unknown handle) -> E_INVALIDARG");
    check(disconnect(0, 0, 0, 0) == hr_invalidarg, "XDspDisconnect(null handle) -> E_INVALIDARG");
    check(disconnect(reinterpret_cast<u64>(handle) + 4, 0, 0, 0) == hr_invalidarg,
          "XDspDisconnect(unknown handle) -> E_INVALIDARG");

    // Synthetic caller: teardown of the stream that the failed activation cleared is not reported as done.
    stream_out = reinterpret_cast<void*>(0x2);
    activate(0x11, 0x22, reinterpret_cast<u64>(&status_out), reinterpret_cast<u64>(&stream_out));
    check(deactivate(reinterpret_cast<u64>(stream_out), 0, 0, 0) != hr_ok,
          "caller deactivating the stream a failed activation cleared is refused");

    check(deactivate(reinterpret_cast<u64>(handle), 0, 0, 0) == hr_ok, "XDspDeactivate(known handle) returns S_OK");
    check(disconnect(reinterpret_cast<u64>(handle), 0, 0, 0) == hr_ok, "XDspDisconnect(known handle) returns S_OK");

    if (!log_path.empty()) {
        const std::string log_text = read_file_bytes(log_path.c_str());
        const char* connect_prefix =
            "x2w-audio XDspConnectWithMaximumStreamLimit memory=0x1000 bytes=0x60000 streams=30 r9=0x80 flag=1 handle_out=";
        std::size_t epointer_hits = 0;
        std::size_t ok_hits = 0;
        std::size_t pos = 0;
        while ((pos = log_text.find(connect_prefix, pos)) != std::string::npos) {
            const std::size_t line_end = log_text.find('\n', pos);
            const std::string line = log_text.substr(pos, line_end == std::string::npos ? std::string::npos : line_end - pos);
            if (line.find(" -> E_POINTER") != std::string::npos) ++epointer_hits;
            if (line.find(" -> 0") != std::string::npos) ++ok_hits;
            pos += std::strlen(connect_prefix);
        }
        check(log_text.find("x2w-trap") == std::string::npos, "permissive lifecycle writes zero x2w-trap entries");
        check(ok_hits == 1, "permissive connect success logged once with -> 0");
        check(epointer_hits == 3, "permissive connect refusals logged three times with -> E_POINTER");
        check(log_text.find("XDspActivate") != std::string::npos &&
                  log_text.find("XDspSubmitCommand") != std::string::npos &&
                  log_text.find("XDspDeactivate") != std::string::npos &&
                  log_text.find("XDspDisconnect") != std::string::npos,
              "permissive lifecycle exports logged");
        check(log_text.find("x2w-audio XDspDeactivate handle=0x0 -> E_INVALIDARG") != std::string::npos &&
                  log_text.find("x2w-audio XDspDisconnect handle=0x0 -> E_INVALIDARG") != std::string::npos,
              "permissive deactivate and disconnect refusals logged");
        DeleteFileW(log_path.c_str());
    }

    if (prev_len > 0 && prev_len < MAX_PATH) {
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", prev_trap_log);
    } else {
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", nullptr);
    }

    FreeLibrary(module);

    // Non-permissive child-process trap checks.
    check_non_permissive_traps(argv[0], argv[1]);

    std::printf(failures ? "startup_chad_check: %d failure(s)\n" : "startup_chad_check: ok\n", failures);
    return failures ? 1 : 0;
}
