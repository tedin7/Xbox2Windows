// Synthetic title-level check for the implemented synchronous XSystem and
// XGame identity subset. It loads the fixture-only DLL by explicit path,
// verifies the fixed console ID, the documented PC device type, and the fixed
// Xbox title ID, repeats each call for stability, and exercises the documented
// short-buffer and null-pointer errors. XUserAddAsync refuses the picker option (None) with a logged E_NOTIMPL.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
namespace fs = std::filesystem;

constexpr HRESULT not_implemented = static_cast<HRESULT>(0x80004001L);
constexpr std::size_t console_id_bytes = 39;
constexpr char expected_console_id[] = "58325731.00000000.00000000.00000000.01";
constexpr std::uint32_t expected_title_id = 0x58325731u;
constexpr std::size_t expected_log_entries = 11;

struct XAsyncBlock;
enum XUserAddOptions : std::uint32_t { XUserAddOptionsNone = 0 };

enum class XSystemDeviceType : std::uint32_t {
    Unknown = 0x00,
    Pc = 0x01,
    XboxOne = 0x02,
    XboxOneS = 0x03,
    XboxOneX = 0x04,
    XboxOneXDevkit = 0x05,
    XboxScarlettLockhart = 0x06,
    XboxScarlettAnaconda = 0x07,
    XboxScarlettDevkit = 0x08
};

struct XGameRuntimeShimLogEntry {
    std::uint32_t sequence;
    HRESULT result;
    const char* component;
    const char* function;
};

using SystemGetConsoleId = HRESULT(WINAPI*)(std::size_t, char*, std::size_t*);
using SystemGetDeviceType = XSystemDeviceType(WINAPI*)();
using GameGetXboxTitleId = HRESULT(WINAPI*)(std::uint32_t*);
using UserAddAsync = HRESULT(WINAPI*)(XUserAddOptions, XAsyncBlock*);
using GetLog = std::size_t(WINAPI*)(XGameRuntimeShimLogEntry*, std::size_t);
using ClearLog = void(WINAPI*)();

struct LoadedShim {
    HMODULE module = nullptr;

    ~LoadedShim()
    {
        if (module) FreeLibrary(module);
    }
};

fs::path normalized_path(const fs::path& path)
{
    std::error_code error;
    const auto canonical = fs::weakly_canonical(path, error);
    return error ? fs::absolute(path).lexically_normal() : canonical;
}

bool same_path(const fs::path& left, const fs::path& right)
{
    return _wcsicmp(normalized_path(left).c_str(), normalized_path(right).c_str()) == 0;
}

bool fixture_path_checks(const wchar_t* dll_path, const wchar_t* fixture_dir,
                         const wchar_t* normal_shim_dir)
{
    const fs::path dll = dll_path;
    const fs::path fixture = fixture_dir;
    const fs::path normal = normal_shim_dir;
    if (dll.filename() != L"xgameruntime.dll" || !same_path(dll.parent_path(), fixture)) {
        std::printf("FAIL: DLL is not at its fixture path %ls\n", fixture.c_str());
        return false;
    }
    if (same_path(dll.parent_path(), normal) || fs::exists(normal / L"xgameruntime.dll")) {
        std::printf("FAIL: fixture DLL is in the normal shim directory\n");
        return false;
    }
    return true;
}

template <typename Proc>
bool find_proc(HMODULE module, const char* name, Proc& proc)
{
    const FARPROC found = GetProcAddress(module, name);
    if (!found) {
        std::printf("FAIL: missing export %s\n", name);
        return false;
    }
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(found), &owner) ||
        owner != module) {
        std::printf("FAIL: export %s is not owned by the loaded shim\n", name);
        return false;
    }
    proc = reinterpret_cast<Proc>(found);
    return true;
}

bool console_id_checks(SystemGetConsoleId get_console_id)
{
    char first[console_id_bytes]{};
    char second[console_id_bytes]{};
    std::size_t first_used = 0;
    std::size_t second_used = 0;
    if (get_console_id(console_id_bytes, first, &first_used) != S_OK ||
        first_used != console_id_bytes) {
        std::printf("FAIL: XSystemGetConsoleId did not report the console ID size\n");
        return false;
    }
    if (std::strlen(first) + 1 != console_id_bytes ||
        std::strcmp(first, expected_console_id) != 0) {
        std::printf("FAIL: XSystemGetConsoleId returned %s\n", first);
        return false;
    }
    if (get_console_id(console_id_bytes, second, &second_used) != S_OK ||
        second_used != first_used ||
        std::memcmp(first, second, console_id_bytes) != 0) {
        std::printf("FAIL: XSystemGetConsoleId changed between calls\n");
        return false;
    }
    if (get_console_id(console_id_bytes, first, nullptr) != S_OK ||
        std::strcmp(first, expected_console_id) != 0) {
        std::printf("FAIL: XSystemGetConsoleId rejected the optional size output\n");
        return false;
    }

    first_used = 0;
    first[0] = '\0';
    if (get_console_id(console_id_bytes - 1, first, &first_used) !=
        HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)) {
        std::printf("FAIL: short console ID buffer did not return ERROR_INSUFFICIENT_BUFFER\n");
        return false;
    }
    if (first_used != 0) {
        std::printf("FAIL: failed XSystemGetConsoleId reported a used size\n");
        return false;
    }
    if (get_console_id(console_id_bytes, nullptr, &first_used) != E_POINTER) {
        std::printf("FAIL: null console ID buffer did not return E_POINTER\n");
        return false;
    }
    return true;
}

bool device_type_checks(SystemGetDeviceType get_device_type)
{
    const XSystemDeviceType first = get_device_type();
    const XSystemDeviceType second = get_device_type();
    if (first != XSystemDeviceType::Pc || second != first) {
        std::printf("FAIL: XSystemGetDeviceType did not return the stable PC type\n");
        return false;
    }
    return true;
}

bool title_id_checks(GameGetXboxTitleId get_title_id)
{
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    if (get_title_id(&first) != S_OK || first != expected_title_id) {
        std::printf("FAIL: XGameGetXboxTitleId returned 0x%08lx\n",
                    static_cast<unsigned long>(first));
        return false;
    }
    if (get_title_id(&second) != S_OK || second != first) {
        std::printf("FAIL: XGameGetXboxTitleId changed between calls\n");
        return false;
    }
    if (get_title_id(nullptr) != E_POINTER) {
        std::printf("FAIL: null title ID output did not return E_POINTER\n");
        return false;
    }
    return true;
}

bool log_checks(HMODULE module)
{
    GetLog get_log = nullptr;
    if (!find_proc(module, "XGameRuntimeShimGetLog", get_log)) return false;
    XGameRuntimeShimLogEntry entries[expected_log_entries]{};
    const std::size_t count = get_log(entries, expected_log_entries);
    if (count != expected_log_entries) {
        std::printf("FAIL: log has %zu entries, expected %zu\n", count, expected_log_entries);
        return false;
    }

    bool saw_console_ok = false;
    bool saw_console_insufficient = false;
    bool saw_console_pointer = false;
    bool saw_device_ok = false;
    bool saw_title_ok = false;
    bool saw_title_pointer = false;
    bool saw_user_stub = false;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& entry = entries[i];
        if (!entry.component || !entry.function || entry.sequence != i + 1) {
            std::printf("FAIL: malformed log entry %zu\n", i);
            return false;
        }
        const bool system = std::strcmp(entry.component, "XSystem") == 0;
        const bool game = std::strcmp(entry.component, "XGame") == 0;
        const bool user = std::strcmp(entry.component, "XUser") == 0;
        if ((system || game) && entry.result == not_implemented) {
            std::printf("FAIL: implemented contract logged E_NOTIMPL: %s\n", entry.function);
            return false;
        }
        if (system && std::strcmp(entry.function, "XSystemGetConsoleId") == 0) {
            if (entry.result == S_OK) saw_console_ok = true;
            if (entry.result == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)) {
                saw_console_insufficient = true;
            }
            if (entry.result == E_POINTER) saw_console_pointer = true;
        } else if (system && std::strcmp(entry.function, "XSystemGetDeviceType") == 0) {
            if (entry.result != S_OK) {
                std::printf("FAIL: XSystemGetDeviceType logged 0x%08lx\n",
                            static_cast<unsigned long>(entry.result));
                return false;
            }
            saw_device_ok = true;
        } else if (game && std::strcmp(entry.function, "XGameGetXboxTitleId") == 0) {
            if (entry.result == S_OK) saw_title_ok = true;
            if (entry.result == E_POINTER) saw_title_pointer = true;
        } else if (user && std::strcmp(entry.function, "XUserAddAsync") == 0) {
            if (entry.result != not_implemented) {
                std::printf("FAIL: XUserAddAsync logged 0x%08lx\n",
                            static_cast<unsigned long>(entry.result));
                return false;
            }
            saw_user_stub = true;
        } else {
            std::printf("FAIL: unexpected log entry %s/%s\n", entry.component, entry.function);
            return false;
        }
    }
    if (!saw_console_ok || !saw_console_insufficient || !saw_console_pointer ||
        !saw_device_ok || !saw_title_ok || !saw_title_pointer || !saw_user_stub) {
        std::printf("FAIL: log is missing positive or negative controls\n");
        return false;
    }
    return true;
}

bool run_checks(const wchar_t* dll_path, const wchar_t* fixture_dir,
                const wchar_t* normal_shim_dir)
{
    if (!fixture_path_checks(dll_path, fixture_dir, normal_shim_dir)) return false;
    LoadedShim shim;
    shim.module = LoadLibraryW(dll_path);
    if (!shim.module) {
        std::printf("FAIL: LoadLibraryW(%ls) error=%lu\n", dll_path, GetLastError());
        return false;
    }

    SystemGetConsoleId get_console_id = nullptr;
    SystemGetDeviceType get_device_type = nullptr;
    GameGetXboxTitleId get_title_id = nullptr;
    UserAddAsync user_add = nullptr;
    ClearLog clear_log = nullptr;
    if (!find_proc(shim.module, "XSystemGetConsoleId", get_console_id) ||
        !find_proc(shim.module, "XSystemGetDeviceType", get_device_type) ||
        !find_proc(shim.module, "XGameGetXboxTitleId", get_title_id) ||
        !find_proc(shim.module, "XUserAddAsync", user_add) ||
        !find_proc(shim.module, "XGameRuntimeShimClearLog", clear_log)) {
        return false;
    }
    clear_log();

    if (!console_id_checks(get_console_id) || !device_type_checks(get_device_type) ||
        !title_id_checks(get_title_id)) {
        return false;
    }
    if (user_add(XUserAddOptionsNone, nullptr) != not_implemented) {
        std::printf("FAIL: XUserAddAsync(None) is not refused with E_NOTIMPL\n");
        return false;
    }
    if (!log_checks(shim.module)) return false;

    std::printf("xgameruntime identity: console=%s device=pc title=0x%08lx "
                "stable=ok short=ERROR_INSUFFICIENT_BUFFER null=E_POINTER "
                "user-async=E_NOTIMPL calls=%zu PASS\n",
                expected_console_id, static_cast<unsigned long>(expected_title_id),
                expected_log_entries);
    return true;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_identity_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    return run_checks(argv[1], argv[2], argv[3]) ? 0 : 1;
}
