// Synthetic title-level check for the implemented synchronous PLS and XPackage
// subset. It loads the fixture-only DLL by explicit path, writes a test-owned
// marker under the PLS directory, verifies the package mount path and unmount
// status, and checks that no implemented call logged E_NOTIMPL. The negative
// controls are XPackageMountWithUiAsync with a null block (E_INVALIDARG) and
// XUserAddAsync(None), which the shim refuses (E_NOTIMPL).
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
constexpr std::uint64_t synthetic_available_free_bytes = 32ull * 1024 * 1024;
constexpr std::uint64_t synthetic_total_free_bytes = 48ull * 1024 * 1024;
constexpr std::uint64_t synthetic_total_bytes = 64ull * 1024 * 1024;
constexpr std::size_t expected_log_entries = 29;

struct XPersistentLocalStorageSpaceInfo {
    std::uint64_t availableFreeBytes;
    std::uint64_t totalFreeBytes;
    std::uint64_t usedBytes;
    std::uint64_t totalBytes;
};

struct XAsyncBlock;
enum XUserAddOptions : std::uint32_t { XUserAddOptionsNone = 0 };
struct XPackageMountStorage;
using XPackageMountHandle = XPackageMountStorage*;

struct XGameRuntimeShimLogEntry {
    std::uint32_t sequence;
    HRESULT result;
    const char* component;
    const char* function;
};

using PlsGetPathSize = HRESULT(WINAPI*)(std::size_t*);
using PlsGetPath = HRESULT(WINAPI*)(std::size_t, char*, std::size_t*);
using PlsGetSpaceInfo = HRESULT(WINAPI*)(XPersistentLocalStorageSpaceInfo*);
using PackageMount = HRESULT(WINAPI*)(const char*, XPackageMountHandle*);
using PackageGetMountPathSize = HRESULT(WINAPI*)(XPackageMountHandle, std::size_t*);
using PackageGetMountPath = HRESULT(WINAPI*)(XPackageMountHandle, std::size_t, char*);
using PackageCloseMountHandle = void(WINAPI*)(XPackageMountHandle);
using PackageMountWithUiAsync = HRESULT(WINAPI*)(const char*, XAsyncBlock*);
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

bool temp_root(std::wstring& root)
{
    wchar_t buffer[MAX_PATH + 1]{};
    constexpr std::size_t capacity = sizeof(buffer) / sizeof(buffer[0]);
    const DWORD length = GetTempPathW(static_cast<DWORD>(capacity), buffer);
    if (length == 0 || length >= capacity) return false;
    root.assign(buffer, length);
    if (root.empty() || root.back() != L'\\') root.push_back(L'\\');
    return true;
}

bool expected_pls_path(fs::path& path)
{
    std::wstring root;
    if (!temp_root(root)) return false;
    path = fs::path(root + L"xbox-recomp-pls\\x2w-synthetic-title-0001");
    return true;
}

bool expected_package_path(const char* identifier, fs::path& path)
{
    std::wstring root;
    if (!temp_root(root)) return false;
    std::wstring name(identifier, identifier + std::strlen(identifier));
    path = fs::path(root + L"xbox-recomp-packages\\" + name);
    return true;
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

bool pls_checks(PlsGetPathSize get_path_size, PlsGetPath get_path,
                PlsGetSpaceInfo get_space_info)
{
    std::size_t first_size = 0;
    std::size_t second_size = 0;
    if (get_path_size(&first_size) != S_OK || first_size < 2) {
        std::printf("FAIL: PLS path size is 0 or failed\n");
        return false;
    }
    if (get_path_size(&second_size) != S_OK || second_size != first_size) {
        std::printf("FAIL: PLS path size changed between calls\n");
        return false;
    }

    std::vector<char> buffer(first_size);
    std::size_t used = 0;
    if (get_path(first_size, buffer.data(), &used) != S_OK || used != first_size) {
        std::printf("FAIL: PLS path call or pathUsed mismatch\n");
        return false;
    }
    if (std::strlen(buffer.data()) + 1 != first_size) {
        std::printf("FAIL: PLS path is not null-terminated within the reported size\n");
        return false;
    }
    fs::path expected;
    if (!expected_pls_path(expected) || !same_path(buffer.data(), expected)) {
        std::printf("FAIL: PLS path %s does not match %ls\n", buffer.data(), expected.c_str());
        return false;
    }
    const std::string pls_path(buffer.data());

    std::vector<char> second_buffer(first_size);
    std::size_t second_used = 0;
    if (get_path(first_size, second_buffer.data(), &second_used) != S_OK ||
        second_used != used || std::memcmp(buffer.data(), second_buffer.data(), first_size) != 0) {
        std::printf("FAIL: PLS path changed between calls\n");
        return false;
    }

    if (get_path(first_size - 1, buffer.data(), &used) != HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)) {
        std::printf("FAIL: PLS short buffer did not return ERROR_INSUFFICIENT_BUFFER\n");
        return false;
    }
    if (get_path(first_size, nullptr, &used) != E_POINTER) {
        std::printf("FAIL: PLS null path did not return E_POINTER\n");
        return false;
    }
    if (get_path_size(nullptr) != E_POINTER) {
        std::printf("FAIL: PLS null path-size output did not return E_POINTER\n");
        return false;
    }

    constexpr std::uint64_t guard = 0x0123456789ABCDEFull;
    struct GuardedSpaceInfo {
        std::uint64_t before = guard;
        XPersistentLocalStorageSpaceInfo space{};
        std::uint64_t after = guard;
    } guarded;
    const auto& first_space = guarded.space;
    XPersistentLocalStorageSpaceInfo second_space{};
    if (get_space_info(&guarded.space) != S_OK || get_space_info(&second_space) != S_OK) {
        std::printf("FAIL: PLS space-info call failed\n");
        return false;
    }
    if (std::memcmp(&first_space, &second_space, sizeof(first_space)) != 0) {
        std::printf("FAIL: PLS space info changed between calls\n");
        return false;
    }
    if (guarded.before != guard || guarded.after != guard) {
        std::printf("FAIL: PLS space info wrote past its documented structure\n");
        return false;
    }
    if (first_space.availableFreeBytes != synthetic_available_free_bytes ||
        first_space.totalFreeBytes != synthetic_total_free_bytes ||
        first_space.usedBytes != 16ull * 1024 * 1024 ||
        first_space.totalBytes != synthetic_total_bytes) {
        std::printf("FAIL: PLS space info is not the documented synthetic value\n");
        return false;
    }
    if (get_space_info(nullptr) != E_POINTER) {
        std::printf("FAIL: PLS null space-info output did not return E_POINTER\n");
        return false;
    }

    const fs::path directory(pls_path);
    if (!fs::is_directory(directory)) {
        std::printf("FAIL: PLS path is not a directory: %s\n", pls_path.c_str());
        return false;
    }
    const fs::path marker = directory / L"x2w-pls-probe.txt";
    constexpr char payload[] = "x2w-pls-persist";
    {
        std::ofstream output(marker, std::ios::binary | std::ios::trunc);
        output.write(payload, sizeof(payload) - 1);
        if (!output) {
            std::printf("FAIL: could not write PLS marker %ls\n", marker.c_str());
            return false;
        }
    }
    {
        std::ifstream input(marker, std::ios::binary);
        std::string read;
        input >> read;
        if (read != payload) {
            std::printf("FAIL: PLS marker did not round-trip\n");
            return false;
        }
    }
    std::error_code error;
    fs::remove(marker, error);
    if (error) {
        std::printf("FAIL: could not remove PLS marker: %s\n", error.message().c_str());
        return false;
    }
    return true;
}

bool package_checks(PackageMount mount, PackageGetMountPathSize get_path_size,
                    PackageGetMountPath get_path, PackageCloseMountHandle close_mount)
{
    constexpr char identifier[] = "x2w-synthetic-package-0001";
    XPackageMountHandle first = nullptr;
    XPackageMountHandle second = nullptr;
    XPackageMountHandle third = nullptr;
    if (mount(identifier, &first) != S_OK || !first) {
        std::printf("FAIL: XPackageMount did not return a handle\n");
        return false;
    }

    std::size_t first_size = 0;
    if (get_path_size(first, &first_size) != S_OK || first_size < 2) {
        std::printf("FAIL: XPackageGetMountPathSize failed\n");
        return false;
    }
    std::vector<char> buffer(first_size);
    if (get_path(first, first_size, buffer.data()) != S_OK) {
        std::printf("FAIL: XPackageGetMountPath failed\n");
        return false;
    }
    fs::path expected;
    if (!expected_package_path(identifier, expected) || !same_path(buffer.data(), expected)) {
        std::printf("FAIL: package mount path %s does not match %ls\n", buffer.data(),
                    expected.c_str());
        return false;
    }
    if (!fs::is_directory(expected)) {
        std::printf("FAIL: package mount path is not a directory %ls\n", expected.c_str());
        return false;
    }

    if (mount(identifier, &second) != S_OK || !second) {
        std::printf("FAIL: second XPackageMount failed\n");
        return false;
    }
    std::size_t second_size = 0;
    if (get_path_size(second, &second_size) != S_OK || second_size != first_size) {
        std::printf("FAIL: second mount path size differs\n");
        return false;
    }
    std::vector<char> second_buffer(second_size);
    if (get_path(second, second_size, second_buffer.data()) != S_OK ||
         std::memcmp(buffer.data(), second_buffer.data(), first_size) != 0) {
        std::printf("FAIL: second mount path differs\n");
        return false;
    }

    close_mount(first);
    if (get_path_size(first, &first_size) != E_HANDLE) {
        std::printf("FAIL: unmounted handle did not return E_HANDLE from path size\n");
        return false;
    }
    if (get_path(first, first_size, buffer.data()) != E_HANDLE) {
        std::printf("FAIL: unmounted handle did not return E_HANDLE from path\n");
        return false;
    }
    close_mount(second);

    if (mount(identifier, &third) != S_OK || !third) {
        std::printf("FAIL: mount after unmount failed\n");
        return false;
    }
    close_mount(third);

    if (mount(nullptr, &third) != E_INVALIDARG) {
        std::printf("FAIL: null package identifier did not return E_INVALIDARG\n");
        return false;
    }
    if (mount(identifier, nullptr) != E_POINTER) {
        std::printf("FAIL: null mount output did not return E_POINTER\n");
        return false;
    }
    if (mount("..", &third) != E_INVALIDARG) {
        std::printf("FAIL: parent-directory package identifier was accepted\n");
        return false;
    }
    if (get_path_size(nullptr, &first_size) != E_HANDLE) {
        std::printf("FAIL: null mount handle did not return E_HANDLE\n");
        return false;
    }
    if (get_path(first, first_size, nullptr) != E_POINTER) {
        std::printf("FAIL: null package path did not return E_POINTER\n");
        return false;
    }
    return true;
}

bool log_checks(HMODULE module)
{
    GetLog get_log = nullptr;
    if (!find_proc(module, "XGameRuntimeShimGetLog", get_log)) return false;
    std::array<XGameRuntimeShimLogEntry, 64> entries{};
    const std::size_t count = get_log(entries.data(), entries.size());
    if (count != expected_log_entries) {
        std::printf("FAIL: log has %zu entries, expected %zu\n", count, expected_log_entries);
        return false;
    }

    bool saw_pls_path_size = false;
    bool saw_pls_path = false;
    bool saw_pls_space = false;
    bool saw_package_mount = false;
    bool saw_package_path_size = false;
    bool saw_package_path = false;
    bool saw_package_close = false;
    bool saw_pls_invalid = false;
    bool saw_pls_pointer = false;
    bool saw_package_invalid = false;
    bool saw_package_pointer = false;
    bool saw_package_handle = false;
    bool saw_async_null_block = false;
    bool saw_user_stub = false;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& entry = entries[i];
        if (!entry.component || !entry.function || entry.sequence != i + 1) {
            std::printf("FAIL: malformed log entry %zu\n", i);
            return false;
        }
        const bool pls = std::strcmp(entry.component, "XPersistentLocalStorage") == 0;
        const bool package = std::strcmp(entry.component, "XPackage") == 0;
        const bool user = std::strcmp(entry.component, "XUser") == 0;
        if ((pls || package) && entry.result == not_implemented) {
            std::printf("FAIL: implemented contract logged E_NOTIMPL: %s\n", entry.function);
            return false;
        }
        if (pls) {
            if (std::strcmp(entry.function, "XPersistentLocalStorageGetPathSize") == 0) {
                saw_pls_path_size = true;
            } else if (std::strcmp(entry.function, "XPersistentLocalStorageGetPath") == 0) {
                saw_pls_path = true;
            } else if (std::strcmp(entry.function, "XPersistentLocalStorageGetSpaceInfo") == 0) {
                saw_pls_space = true;
            } else {
                std::printf("FAIL: unexpected PLS function %s\n", entry.function);
                return false;
            }
            if (entry.result == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)) saw_pls_invalid = true;
            if (entry.result == E_POINTER) saw_pls_pointer = true;
        }
        if (package) {
            if (std::strcmp(entry.function, "XPackageMount") == 0) {
                saw_package_mount = true;
            } else if (std::strcmp(entry.function, "XPackageGetMountPathSize") == 0) {
                saw_package_path_size = true;
            } else if (std::strcmp(entry.function, "XPackageGetMountPath") == 0) {
                saw_package_path = true;
            } else if (std::strcmp(entry.function, "XPackageCloseMountHandle") == 0) {
                saw_package_close = true;
            } else if (std::strcmp(entry.function, "XPackageMountWithUiAsync") == 0) {
                if (entry.result != E_INVALIDARG) {
                    std::printf("FAIL: async package null block returned 0x%08lx\n",
                                static_cast<unsigned long>(entry.result));
                    return false;
                }
                saw_async_null_block = true;
            } else {
                std::printf("FAIL: unexpected XPackage function %s\n", entry.function);
                return false;
            }
            if (entry.result == E_INVALIDARG) saw_package_invalid = true;
            if (entry.result == E_POINTER) saw_package_pointer = true;
            if (entry.result == E_HANDLE) saw_package_handle = true;
        }
        if (user && std::strcmp(entry.function, "XUserAddAsync") == 0) {
            if (entry.result != not_implemented) {
                std::printf("FAIL: XUserAddAsync(None) returned 0x%08lx\n",
                            static_cast<unsigned long>(entry.result));
                return false;
            }
            saw_user_stub = true;
        }
    }
    if (!saw_pls_path_size || !saw_pls_path || !saw_pls_space || !saw_package_mount ||
        !saw_package_path_size || !saw_package_path || !saw_package_close ||
        !saw_pls_invalid || !saw_pls_pointer || !saw_package_invalid ||
        !saw_package_pointer || !saw_package_handle || !saw_async_null_block || !saw_user_stub) {
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

    PlsGetPathSize get_path_size = nullptr;
    PlsGetPath get_path = nullptr;
    PlsGetSpaceInfo get_space_info = nullptr;
    PackageMount mount = nullptr;
    PackageGetMountPathSize get_mount_path_size = nullptr;
    PackageGetMountPath get_mount_path = nullptr;
    PackageCloseMountHandle close_mount = nullptr;
    PackageMountWithUiAsync mount_with_ui = nullptr;
    UserAddAsync user_add = nullptr;
    ClearLog clear_log = nullptr;
    if (!find_proc(shim.module, "XPersistentLocalStorageGetPathSize", get_path_size) ||
        !find_proc(shim.module, "XPersistentLocalStorageGetPath", get_path) ||
        !find_proc(shim.module, "XPersistentLocalStorageGetSpaceInfo", get_space_info) ||
        !find_proc(shim.module, "XPackageMount", mount) ||
        !find_proc(shim.module, "XPackageGetMountPathSize", get_mount_path_size) ||
        !find_proc(shim.module, "XPackageGetMountPath", get_mount_path) ||
        !find_proc(shim.module, "XPackageCloseMountHandle", close_mount) ||
        !find_proc(shim.module, "XPackageMountWithUiAsync", mount_with_ui) ||
        !find_proc(shim.module, "XUserAddAsync", user_add) ||
        !find_proc(shim.module, "XGameRuntimeShimClearLog", clear_log)) {
        return false;
    }
    clear_log();

    if (!pls_checks(get_path_size, get_path, get_space_info) ||
        !package_checks(mount, get_mount_path_size, get_mount_path, close_mount)) {
        return false;
    }
    if (mount_with_ui("x2w-synthetic-package-0001", nullptr) != E_INVALIDARG) {
        std::printf("FAIL: XPackageMountWithUiAsync accepted a null block\n");
        return false;
    }
    if (user_add(XUserAddOptionsNone, nullptr) != not_implemented) {
        std::printf("FAIL: XUserAddAsync(None) is not refused\n");
        return false;
    }
    if (!log_checks(shim.module)) return false;

    std::printf("xgameruntime PLS/XPackage: space=%llu/%llu/%llu async-null-block=E_INVALIDARG "
                "unmount=E_HANDLE user-async=E_NOTIMPL calls=%zu PASS\n",
                static_cast<unsigned long long>(synthetic_available_free_bytes),
                static_cast<unsigned long long>(synthetic_total_free_bytes),
                static_cast<unsigned long long>(synthetic_total_bytes), expected_log_entries);
    return true;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_pls_xpackage_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    return run_checks(argv[1], argv[2], argv[3]) ? 0 : 1;
}
