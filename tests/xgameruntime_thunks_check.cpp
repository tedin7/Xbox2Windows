// Loads the built xgameruntime.dll from its fixture-only directory, checks the
// export table (no explicit stubs remain), and validates the deterministic call log: the
// null-block call to XPackageMountWithUiAsync is refused with E_INVALIDARG and logged. It also rejects a DLL elsewhere or a copy
// in the normal shim directory. The unknown-export lookup is a negative
// control: a missing export returns null.
#include <array>
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

struct XAsyncBlock;
struct XUserHandleStorage;
using XUserHandle = XUserHandleStorage*;

struct ExpectedThunk {
    const char* name;
    const char* component;
    const char* function;
};

constexpr std::array<ExpectedThunk, 1> expected_thunks{{
    {"XPackageMountWithUiAsync", "XPackage", "XPackageMountWithUiAsync"},
}};

struct XGameRuntimeShimLogEntry {
    std::uint32_t sequence;
    HRESULT result;
    const char* component;
    const char* function;
};

static_assert(sizeof(XGameRuntimeShimLogEntry) == 24);
static_assert(offsetof(XGameRuntimeShimLogEntry, sequence) == 0);
static_assert(offsetof(XGameRuntimeShimLogEntry, result) == 4);
static_assert(offsetof(XGameRuntimeShimLogEntry, component) == 8);
static_assert(offsetof(XGameRuntimeShimLogEntry, function) == 16);

using PackageMountWithUiAsync = HRESULT(WINAPI*)(const char*, XAsyncBlock*);
using GetLog = std::size_t(WINAPI*)(XGameRuntimeShimLogEntry*, std::size_t);
using ClearLog = void(WINAPI*)();

struct LoadedShim {
    HMODULE module = nullptr;

    ~LoadedShim()
    {
        if (module) FreeLibrary(module);
    }
};

bool export_names_are_unique() noexcept
{
    for (std::size_t i = 0; i < expected_thunks.size(); ++i) {
        if (!expected_thunks[i].name[0]) {
            std::printf("FAIL: empty export name at index %zu\n", i);
            return false;
        }
        for (std::size_t j = i + 1; j < expected_thunks.size(); ++j) {
            if (std::strcmp(expected_thunks[i].name, expected_thunks[j].name) == 0) {
                std::printf("FAIL: duplicate export name %s\n", expected_thunks[i].name);
                return false;
            }
        }
    }
    return true;
}

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

bool is_within(const fs::path& child, const fs::path& parent)
{
    const auto normalized_child = normalized_path(child);
    const auto normalized_parent = normalized_path(parent);
    auto child_part = normalized_child.begin();
    for (auto parent_part = normalized_parent.begin(); parent_part != normalized_parent.end();
         ++parent_part, ++child_part) {
        if (child_part == normalized_child.end() ||
            _wcsicmp(child_part->c_str(), parent_part->c_str()) != 0) {
            return false;
        }
    }
    return child_part != normalized_child.end();
}

bool fixture_path_checks(const wchar_t* dll_path, const wchar_t* fixture_dir,
                         const wchar_t* normal_shim_dir)
{
    const fs::path dll = dll_path;
    const fs::path fixture = fixture_dir;
    const fs::path normal = normal_shim_dir;
    if (dll.filename() != L"xgameruntime.dll") {
        std::printf("FAIL: unexpected fixture DLL name %ls\n", dll.filename().c_str());
        return false;
    }
    if (!is_within(dll, fixture) || !same_path(dll.parent_path(), fixture)) {
        std::printf("FAIL: %ls is not directly in the fixture directory %ls\n",
                    dll.c_str(), fixture.c_str());
        return false;
    }
    if (same_path(dll.parent_path(), normal)) {
        std::printf("FAIL: fixture DLL is in the normal shim directory %ls\n", normal.c_str());
        return false;
    }
    if (fs::exists(normal / L"xgameruntime.dll")) {
        std::printf("FAIL: normal shim directory contains %ls\n",
                    (normal / L"xgameruntime.dll").c_str());
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

bool calls_are_refused(HMODULE module)
{
    PackageMountWithUiAsync package_mount = nullptr;

    if (!find_proc(module, expected_thunks[0].name, package_mount)) {
        return false;
    }

    struct Call {
        const char* name;
        HRESULT result;
    };
    const std::array<Call, 1> calls{{
        {expected_thunks[0].name, package_mount("x2w-synthetic-package-0001", nullptr)},
    }};
    for (const auto& call : calls) {
        if (call.result != E_INVALIDARG) {
            std::printf("FAIL: %s returned 0x%08lx, expected E_INVALIDARG\n", call.name,
                        static_cast<unsigned long>(call.result));
            return false;
        }
    }
    return true;
}

bool log_matches_calls(HMODULE module)
{
    GetLog get_log = nullptr;
    ClearLog clear_log = nullptr;
    if (!find_proc(module, "XGameRuntimeShimGetLog", get_log) ||
        !find_proc(module, "XGameRuntimeShimClearLog", clear_log)) {
        return false;
    }

    clear_log();
    if (get_log(nullptr, 0) != 0) {
        std::printf("FAIL: log was not empty after clear\n");
        return false;
    }
    if (!calls_are_refused(module)) return false;

    std::array<XGameRuntimeShimLogEntry, expected_thunks.size()> entries{};
    const std::size_t total = get_log(entries.data(), entries.size());
    if (total != entries.size()) {
        std::printf("FAIL: log has %zu entries, expected %zu\n", total, entries.size());
        return false;
    }
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (entry.sequence != i + 1) {
            std::printf("FAIL: log entry %zu has sequence %lu\n", i,
                        static_cast<unsigned long>(entry.sequence));
            return false;
        }
        if (entry.result != E_INVALIDARG) {
            std::printf("FAIL: log entry %zu has result 0x%08lx\n", i,
                        static_cast<unsigned long>(entry.result));
            return false;
        }
        if (!entry.component || std::strcmp(entry.component, expected_thunks[i].component) != 0 ||
            !entry.function || std::strcmp(entry.function, expected_thunks[i].function) != 0) {
            std::printf("FAIL: log entry %zu is %s/%s, expected %s/%s\n", i,
                        entry.component ? entry.component : "(null)",
                        entry.function ? entry.function : "(null)", expected_thunks[i].component,
                        expected_thunks[i].function);
            return false;
        }
    }
    return true;
}

bool unknown_export_is_missing(HMODULE module)
{
    constexpr const char* unknown_name = "XGameRuntimeShimExportThatMustNotExist";
    if (GetProcAddress(module, unknown_name) != nullptr) {
        std::printf("FAIL: unknown export %s unexpectedly resolved\n", unknown_name);
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
    return export_names_are_unique() && unknown_export_is_missing(shim.module) &&
           log_matches_calls(shim.module);
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_shim_check <xgameruntime.dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    const bool passed = run_checks(argv[1], argv[2], argv[3]);
    std::printf("Desktop fixture xgameruntime.dll: exports=%zu calls=%zu unknown_export=null log=%zu %s\n",
                expected_thunks.size(), expected_thunks.size(), expected_thunks.size(),
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
