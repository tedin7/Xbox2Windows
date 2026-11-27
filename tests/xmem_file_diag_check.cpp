// Title file-call diagnostics (src/xmem_file_diag.cpp). Loading xmem.dll wraps THIS program's own file-call imports,
// as it wraps the title's. WARP-free: a temp folder with files that do not exist makes the calls fail.
//
// on:  failing CreateFileW / GetFileAttributesW / FindFirstFileW calls are logged with their error, the real results and
//      last error come back unchanged, a successful call is not logged, only 30 failing calls are logged, a null-address
//      fault before the first OpenFile logs the "package archives not opened" line naming the first failing call, and
//      x2w_file_diag_opened() ends the logging.
// off: X2W_FILE_DIAG unset (the control): the same failing calls return the same results and nothing is logged.
//
// Usage: xmem_file_diag_check <path to xmem.dll> on|off
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures = 0;

void check(bool ok, const char* name)
{
    std::printf("%s: %s\n", ok ? "ok" : "check_failed", name);
    if (!ok) ++failures;
}

std::string read_log(const std::wstring& path)
{
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    std::string text;
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof(buffer), &got, nullptr) && got) text.append(buffer, got);
    CloseHandle(file);
    return text;
}

bool has(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

// Kept free of C++ objects for __try.
void write_through_null()
{
    __try {
        *reinterpret_cast<volatile int*>(0) = 1;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || (std::strcmp(argv[2], "on") && std::strcmp(argv[2], "off"))) {
        std::fprintf(stderr, "usage: xmem_file_diag_check <path to xmem.dll> on|off\n");
        return 2;
    }
    const bool on = std::strcmp(argv[2], "on") == 0;

    wchar_t temp[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, temp)) return 3;
    const std::wstring folder = std::wstring(temp) + L"x2w_file_diag_" + std::to_wstring(GetCurrentProcessId());
    if (!CreateDirectoryW(folder.c_str(), nullptr)) return 3;
    const std::wstring log_path = folder + L"\\diag.log";
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    SetEnvironmentVariableW(L"X2W_FILE_DIAG", on ? L"1" : nullptr);

    wchar_t library[MAX_PATH];
    if (!MultiByteToWideChar(CP_ACP, 0, argv[1], -1, library, MAX_PATH)) return 3;
    const HMODULE xmem = LoadLibraryExW(library, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!xmem) return 4;

    // A call that succeeds is not logged.
    const std::wstring existing = folder + L"\\existing_ok.bin";
    const HANDLE made = CreateFileW(existing.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(made != INVALID_HANDLE_VALUE, "a new file is created");
    if (made != INVALID_HANDLE_VALUE) CloseHandle(made);

    // Failing calls return the real result and the real last error.
    const std::wstring missing = folder + L"\\missing_archive.rpf";
    SetLastError(0);
    const HANDLE opened = CreateFileW(missing.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(opened == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND, "CreateFileW keeps its result and error 2");
    SetLastError(0);
    const DWORD attributes = GetFileAttributesW(missing.c_str());
    check(attributes == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND, "GetFileAttributesW keeps its result and error 2");
    const std::wstring nowhere = folder + L"\\no_such_folder\\*";
    WIN32_FIND_DATAW found{};
    SetLastError(0);
    const HANDLE search = FindFirstFileW(nowhere.c_str(), &found);
    check(search == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND, "FindFirstFileW keeps its result and error 3");

    // Only 30 failing calls are logged.
    for (int i = 0; i < 40; ++i) GetFileAttributesW((folder + L"\\flood_" + std::to_wstring(i) + L".bin").c_str());

    // A null-address fault before the first OpenFile names the first failing call.
    write_through_null();

    // The first OpenFile ends the diagnostics.
    const auto announce = reinterpret_cast<void(WINAPI*)()>(GetProcAddress(xmem, "x2w_file_diag_opened"));
    check(announce != nullptr, "xmem.dll exports x2w_file_diag_opened");
    if (announce) announce();
    GetFileAttributesW((folder + L"\\after_open.bin").c_str());

    const std::string text = read_log(log_path);
    if (on) {
        check(has(text, "x2w-filediag title file calls wrapped"), "the wrap is announced");
        check(has(text, "x2w-filediag package drive"), "the package drive line is logged");
        check(has(text, "failing call 1: CreateFileW(") && has(text, "missing_archive.rpf\") error 2"), "the failing CreateFileW is logged with error 2");
        check(has(text, "failing call 2: GetFileAttributesW("), "the failing GetFileAttributesW is logged");
        check(has(text, "failing call 3: FindFirstFileW(") && has(text, "no_such_folder") && has(text, "error 3"), "the failing FindFirstFileW is logged with error 3");
        check(!has(text, "existing_ok.bin"), "a successful call is not logged");
        check(has(text, "failing call 30:") && !has(text, "failing call 31:"), "only the first 30 failing calls are logged");
        check(has(text, "package archives not opened: the title faulted at address 0x0 before its first dstorage OpenFile") &&
                  has(text, "first CreateFileW(\"") && has(text, "missing_archive.rpf"),
              "the null fault names the first failing call");
        check(has(text, "first dstorage OpenFile reached after"), "OpenFile ends the diagnostics");
        check(!has(text, "after_open.bin"), "calls after the first OpenFile are not logged");
    } else {
        check(!has(text, "x2w-filediag"), "without X2W_FILE_DIAG nothing is logged");
    }

    DeleteFileW(existing.c_str());  // best effort cleanup; the log stays open in the DLL, delete sharing is allowed
    DeleteFileW(log_path.c_str());
    RemoveDirectoryW(folder.c_str());
    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
