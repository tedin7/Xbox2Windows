// Title file-call diagnostics. When xmem.dll loads (before the title runs) the main module's file calls are wrapped so
// a run that dies before it opens its first package archive says why: the first 30 FAILING calls (CreateFileW,
// GetFileAttributesW/ExW, FindFirstFileW/ExW, GetFileSizeEx) are logged with GetLastError and the file's attributes
// (reparse point, offline, recall on open / data access: a cloud drive stub), the package drive's free space and
// volume information are logged once, and a null-address write before the first dstorage OpenFile logs one line
// "package archives not opened: ..." naming the first failing call.
//
// Off unless X2W_FILE_DIAG=1 (tester_run.ps1 sets it). Log only: every wrapper returns the real result and last error unchanged. It allocates
// nothing (fixed buffers), so it cannot take address space from the title's arena.
#include <windows.h>

#include <cstdio>
#include <cwchar>

#include "startup_stubs/local_log.h"
#include "xmem_import_patch.h"

namespace {
constexpr LONG kept_calls = 30;

volatile LONG archives_opened = 0;
volatile LONG failing_calls = 0;
volatile LONG fault_reported = 0;

struct FirstFailure {
    char api[24];
    wchar_t path[260];
    DWORD error;
};
FirstFailure first_failure;

struct Real {
    HANDLE(WINAPI* create_file)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
    DWORD(WINAPI* attributes)(LPCWSTR);
    BOOL(WINAPI* attributes_ex)(LPCWSTR, GET_FILEEX_INFO_LEVELS, LPVOID);
    HANDLE(WINAPI* find_first)(LPCWSTR, LPWIN32_FIND_DATAW);
    HANDLE(WINAPI* find_first_ex)(LPCWSTR, FINDEX_INFO_LEVELS, LPVOID, FINDEX_SEARCH_OPS, LPVOID, DWORD);
    BOOL(WINAPI* file_size)(HANDLE, PLARGE_INTEGER);
} real;

bool diag_active() noexcept { return InterlockedCompareExchange(&archives_opened, 0, 0) == 0; }

// "reparse offline recall-open recall-data" for the attribute bits that mark a cloud stub or a mount point.
void attribute_text(DWORD attributes, char* out, std::size_t size) noexcept
{
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        std::snprintf(out, size, "unreadable");
        return;
    }
    std::snprintf(out, size, "0x%lx%s%s%s%s%s", attributes, (attributes & FILE_ATTRIBUTE_REPARSE_POINT) ? " reparse" : "",
                  (attributes & FILE_ATTRIBUTE_OFFLINE) ? " offline" : "",
                  (attributes & 0x00040000) ? " recall-on-open" : "",    // FILE_ATTRIBUTE_RECALL_ON_OPEN
                  (attributes & 0x00400000) ? " recall-on-data-access" : "",  // FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
                  (attributes & FILE_ATTRIBUTE_SPARSE_FILE) ? " sparse" : "");
}

void note_failure(const char* api, const wchar_t* path, DWORD error) noexcept
{
    if (!diag_active()) return;
    const LONG number = InterlockedIncrement(&failing_calls);
    if (number == 1) {
        std::snprintf(first_failure.api, sizeof(first_failure.api), "%s", api);
        if (path) wcsncpy_s(first_failure.path, path, _TRUNCATE);
        first_failure.error = error;
    }
    if (number > kept_calls) return;
    char text[96];
    attribute_text(path && real.attributes ? real.attributes(path) : INVALID_FILE_ATTRIBUTES, text, sizeof(text));
    x2w_local::logf("x2w-filediag failing call %ld: %s(\"%ls\") error %lu, attributes %s", number, api, path ? path : L"(handle)",
                    error, text);
}

HANDLE WINAPI create_file_w(LPCWSTR path, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES security, DWORD disposition,
                            DWORD flags, HANDLE template_file)
{
    const HANDLE result = real.create_file(path, access, share, security, disposition, flags, template_file);
    const DWORD error = GetLastError();
    if (result == INVALID_HANDLE_VALUE) note_failure("CreateFileW", path, error);
    SetLastError(error);
    return result;
}

DWORD WINAPI get_file_attributes_w(LPCWSTR path)
{
    const DWORD result = real.attributes(path);
    const DWORD error = GetLastError();
    if (result == INVALID_FILE_ATTRIBUTES) note_failure("GetFileAttributesW", path, error);
    SetLastError(error);
    return result;
}

BOOL WINAPI get_file_attributes_ex_w(LPCWSTR path, GET_FILEEX_INFO_LEVELS level, LPVOID data)
{
    const BOOL result = real.attributes_ex(path, level, data);
    const DWORD error = GetLastError();
    if (!result) note_failure("GetFileAttributesExW", path, error);
    SetLastError(error);
    return result;
}

HANDLE WINAPI find_first_file_w(LPCWSTR path, LPWIN32_FIND_DATAW data)
{
    const HANDLE result = real.find_first(path, data);
    const DWORD error = GetLastError();
    if (result == INVALID_HANDLE_VALUE) note_failure("FindFirstFileW", path, error);
    SetLastError(error);
    return result;
}

HANDLE WINAPI find_first_file_ex_w(LPCWSTR path, FINDEX_INFO_LEVELS level, LPVOID data, FINDEX_SEARCH_OPS op, LPVOID filter,
                                   DWORD flags)
{
    const HANDLE result = real.find_first_ex(path, level, data, op, filter, flags);
    const DWORD error = GetLastError();
    if (result == INVALID_HANDLE_VALUE) note_failure("FindFirstFileExW", path, error);
    SetLastError(error);
    return result;
}

BOOL WINAPI get_file_size_ex(HANDLE file, PLARGE_INTEGER size)
{
    const BOOL result = real.file_size(file, size);
    const DWORD error = GetLastError();
    if (!result) note_failure("GetFileSizeEx", nullptr, error);
    SetLastError(error);
    return result;
}

// A write or read through an address below 64 KiB before any package archive was opened: say what failed first.
LONG CALLBACK null_fault(EXCEPTION_POINTERS* info) noexcept
{
    const EXCEPTION_RECORD* record = info->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 ||
        record->ExceptionInformation[1] >= 0x10000 || !diag_active() || InterlockedExchange(&fault_reported, 1))
        return EXCEPTION_CONTINUE_SEARCH;
    const LONG failed = InterlockedCompareExchange(&failing_calls, 0, 0);
    if (failed)
        x2w_local::logf("x2w-filediag package archives not opened: the title faulted at address 0x%llx before its first "
                        "dstorage OpenFile; %ld failing file call(s), first %s(\"%ls\") error %lu",
                        static_cast<unsigned long long>(record->ExceptionInformation[1]), failed, first_failure.api,
                        first_failure.path, first_failure.error);
    else
        x2w_local::logf("x2w-filediag package archives not opened: the title faulted at address 0x%llx before its first "
                        "dstorage OpenFile and no file call had failed",
                        static_cast<unsigned long long>(record->ExceptionInformation[1]));
    return EXCEPTION_CONTINUE_SEARCH;
}

// The package drive: the drive the main module was loaded from. A cloud-drive mount reports unusual values here.
void log_package_drive() noexcept
{
    wchar_t module[MAX_PATH]{};
    const DWORD length = GetModuleFileNameW(nullptr, module, MAX_PATH);
    if (!length || length >= MAX_PATH) return;
    wchar_t root[MAX_PATH]{};
    if (!GetVolumePathNameW(module, root, MAX_PATH)) {
        x2w_local::logf("x2w-filediag package drive: GetVolumePathNameW(\"%ls\") failed, error %lu", module, GetLastError());
        return;
    }
    ULARGE_INTEGER caller_free{}, total{}, total_free{};
    const bool space = GetDiskFreeSpaceExW(root, &caller_free, &total, &total_free) != FALSE;
    const DWORD space_error = space ? 0 : GetLastError();
    DWORD sectors = 0, bytes = 0, free_clusters = 0, clusters = 0;
    const bool geometry = GetDiskFreeSpaceW(root, &sectors, &bytes, &free_clusters, &clusters) != FALSE;
    const DWORD geometry_error = geometry ? 0 : GetLastError();
    wchar_t label[MAX_PATH + 1]{}, file_system[MAX_PATH + 1]{};
    DWORD serial = 0, component = 0, flags = 0;
    const bool volume = GetVolumeInformationW(root, label, MAX_PATH, &serial, &component, &flags, file_system, MAX_PATH) != FALSE;
    const DWORD volume_error = volume ? 0 : GetLastError();
    char exe[96];
    attribute_text(GetFileAttributesW(module), exe, sizeof(exe));
    x2w_local::logf("x2w-filediag package drive \"%ls\": type %u (3 fixed, 4 remote), free to caller %llu MiB, total %llu MiB, "
                    "total free %llu MiB (error %lu); sectors per cluster %lu, bytes per sector %lu (error %lu); "
                    "volume \"%ls\" file system \"%ls\" flags 0x%lx serial %08lx max component %lu (error %lu); main module attributes %s",
                    root, GetDriveTypeW(root), static_cast<unsigned long long>(caller_free.QuadPart >> 20),
                    static_cast<unsigned long long>(total.QuadPart >> 20), static_cast<unsigned long long>(total_free.QuadPart >> 20),
                    space_error, sectors, bytes, geometry_error, label, file_system, flags, serial, component, volume_error, exe);
}

bool wrap_title_file_calls() noexcept
{
    if (!x2w_local::env_on("X2W_FILE_DIAG")) return false;
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                            reinterpret_cast<LPCWSTR>(&null_fault), &self))
        return false;  // the wrappers must outlive any unload
    const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    if (!kernel) return false;
    real.create_file = reinterpret_cast<decltype(real.create_file)>(GetProcAddress(kernel, "CreateFileW"));
    real.attributes = reinterpret_cast<decltype(real.attributes)>(GetProcAddress(kernel, "GetFileAttributesW"));
    real.attributes_ex = reinterpret_cast<decltype(real.attributes_ex)>(GetProcAddress(kernel, "GetFileAttributesExW"));
    real.find_first = reinterpret_cast<decltype(real.find_first)>(GetProcAddress(kernel, "FindFirstFileW"));
    real.find_first_ex = reinterpret_cast<decltype(real.find_first_ex)>(GetProcAddress(kernel, "FindFirstFileExW"));
    real.file_size = reinterpret_cast<decltype(real.file_size)>(GetProcAddress(kernel, "GetFileSizeEx"));
    if (!real.create_file || !real.attributes || !real.attributes_ex || !real.find_first || !real.find_first_ex || !real.file_size)
        return false;
    int patched = 0;
    patched += x2w_xmem::patch_main_import("CreateFileW", reinterpret_cast<const void*>(&create_file_w));
    patched += x2w_xmem::patch_main_import("GetFileAttributesW", reinterpret_cast<const void*>(&get_file_attributes_w));
    patched += x2w_xmem::patch_main_import("GetFileAttributesExW", reinterpret_cast<const void*>(&get_file_attributes_ex_w));
    patched += x2w_xmem::patch_main_import("FindFirstFileW", reinterpret_cast<const void*>(&find_first_file_w));
    patched += x2w_xmem::patch_main_import("FindFirstFileExW", reinterpret_cast<const void*>(&find_first_file_ex_w));
    patched += x2w_xmem::patch_main_import("GetFileSizeEx", reinterpret_cast<const void*>(&get_file_size_ex));
    AddVectoredExceptionHandler(0, null_fault);  // last: the lazy-commit handler answers its own faults first
    x2w_local::logf("x2w-filediag title file calls wrapped (X2W_FILE_DIAG=1): %d import(s)", patched);
    log_package_drive();
    return true;
}

[[maybe_unused]] const bool installed = wrap_title_file_calls();
}  // namespace

// Called by the first dstorage Factory::OpenFile: the package archives are being opened, so the diagnostics stop.
extern "C" void WINAPI x2w_file_diag_opened() noexcept
{
    if (InterlockedExchange(&archives_opened, 1)) return;
    x2w_local::logf("x2w-filediag first dstorage OpenFile reached after %ld failing file call(s)",
                    InterlockedCompareExchange(&failing_calls, 0, 0));
}
