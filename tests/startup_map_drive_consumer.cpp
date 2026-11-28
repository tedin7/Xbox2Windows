// Source-built consumer for the launcher's --map-drive option (#1158).
//
// A console executable opens its package at a fixed drive letter. The letter comes from
// X2W_PROBE_DRIVE (default Z) because the launcher passes no program arguments.
//
// Two modes, chosen with X2W_PROBE_MODE:
//   (default) one open of "<letter>:\map_drive_probe.txt" through CreateFileW, printing what it read.
//             Under the launcher with --map-drive the open is rewritten and succeeds; without it, it
//             fails, which is the negative control.
//   reuse     a caller-owned UNICODE_STRING is passed to NtCreateFile many times. After every return
//             its length, buffer pointer and contents must be the ones the caller built, and every
//             open must still succeed: the rewritten name is only borrowed for the duration of each
//             call, and one backing buffer is reused rather than one allocation per open.
#include <windows.h>
#include <winternl.h>

#include <cstdio>
#include <cwchar>

namespace {
constexpr int reuse_opens = 32;

// ntdll's declarations in <winternl.h> differ across SDK versions, so resolve the two entry points
// at run time: the fixture only needs to make the call, not to link against a signature.
using NtCreateFileFn = NTSTATUS(NTAPI*)(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES, PIO_STATUS_BLOCK,
                                       PLARGE_INTEGER, ULONG, ULONG, ULONG, ULONG, PVOID, ULONG);
using NtCloseFn = NTSTATUS(NTAPI*)(HANDLE);

bool resolve(HMODULE ntdll, NtCreateFileFn& create, NtCloseFn& close)
{
    create = reinterpret_cast<NtCreateFileFn>(GetProcAddress(ntdll, "NtCreateFile"));
    close = reinterpret_cast<NtCloseFn>(GetProcAddress(ntdll, "NtClose"));
    return create && close;
}

wchar_t probe_letter()
{
    wchar_t letter = L'Z';
    wchar_t from_env[8]{};
    if (GetEnvironmentVariableW(L"X2W_PROBE_DRIVE", from_env, 8) == 1) letter = from_env[0];
    return letter;
}

// One open through CreateFileW; the mapped letter has no volume of its own.
int single_open()
{
    wchar_t path[64];
    _snwprintf_s(path, 64, _TRUNCATE, L"%c:\\map_drive_probe.txt", probe_letter());
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        std::printf("open failed: %lu\n", static_cast<unsigned long>(GetLastError()));
        return 2;
    }
    char text[64]{};
    DWORD got = 0;
    const BOOL ok = ReadFile(file, text, sizeof(text) - 1, &got, nullptr);
    CloseHandle(file);
    if (!ok) return 3;
    std::printf("read %lu bytes: %s\n", static_cast<unsigned long>(got), text);
    return 0;
}

int reused_string()
{
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    NtCreateFileFn create = nullptr;
    NtCloseFn close = nullptr;
    if (!ntdll || !resolve(ntdll, create, close)) {
        std::printf("ntdll entry points not resolved\n");
        return 1;
    }
    wchar_t name[64];
    // NtCreateFile takes the NT form of the name; CreateFileW would add this prefix itself.
    _snwprintf_s(name, 64, _TRUNCATE, L"\\??\\%c:\\map_drive_probe.txt", probe_letter());
    wchar_t built[64];
    wcscpy_s(built, name);

    UNICODE_STRING string{};
    string.Buffer = name;
    string.Length = static_cast<USHORT>(std::wcslen(name) * sizeof(wchar_t));
    string.MaximumLength = string.Length;  // exactly the caller's size, so a longer name must borrow
    OBJECT_ATTRIBUTES attributes{};
    InitializeObjectAttributes(&attributes, &string, OBJ_CASE_INSENSITIVE, nullptr, nullptr);

    const USHORT length = string.Length, maximum = string.MaximumLength;
    PWSTR buffer = string.Buffer;
    for (int index = 0; index < reuse_opens; ++index) {
        HANDLE handle = nullptr;
        IO_STATUS_BLOCK status{};
        const NTSTATUS result = create(&handle, GENERIC_READ | SYNCHRONIZE, &attributes, &status, nullptr,
                                       FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_OPEN,
                                       FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, nullptr, 0);
        if (result < 0) {
            std::printf("reuse open %d failed: 0x%08lx\n", index, static_cast<unsigned long>(result));
            return 1;
        }
        close(handle);
        // The launcher borrows the caller's name for the call only; it must be back exactly as built.
        if (string.Length != length || string.MaximumLength != maximum || string.Buffer != buffer ||
            std::wcscmp(string.Buffer, built) != 0) {
            std::printf("caller name changed after open %d: length=%u maximum=%u buffer_moved=%d text=%ls\n",
                        index, string.Length, string.MaximumLength, string.Buffer != buffer ? 1 : 0,
                        string.Buffer ? string.Buffer : L"(null)");
            return 1;
        }
    }
    std::printf("reuse ok: %d opens, caller name intact after every return\n", reuse_opens);
    return 0;
}
// Two threads open through the mapped letter at the same time with their own caller-owned names.
// Each rewritten name must stay valid for the call that is still in flight, so both threads must see
// their own mapped file and get their own UNICODE_STRING back.
struct Overlap {
    HANDLE start;
    wchar_t letter;
    int opens;
    LONG completed;
    LONG broken;
    LONG wrong_file;  // an open that reached a file other than the one its own thread named
};

// The two threads name DIFFERENT files of DIFFERENT lengths, so a name buffer shared between the calls in
// flight would hand one thread the other's name (or a cut-off one) instead of its own file.
struct Worker {
    Overlap* shared;
    int index;
};
const wchar_t* const overlap_file[2] = {L"map_drive_probe.txt", L"map_drive_probe_second_longer_name.txt"};


DWORD WINAPI overlap_thread(LPVOID parameter)
{
    auto* worker = static_cast<Worker*>(parameter);
    Overlap* shared = worker->shared;
    const wchar_t* const file_name = overlap_file[worker->index];
    wchar_t name[96];
    _snwprintf_s(name, 96, _TRUNCATE, L"\\??\\%c:\\%ls", shared->letter, file_name);
    wchar_t built[96];
    wcscpy_s(built, name);

    UNICODE_STRING string{};
    string.Buffer = name;
    string.Length = static_cast<USHORT>(std::wcslen(name) * sizeof(wchar_t));
    string.MaximumLength = string.Length;
    OBJECT_ATTRIBUTES attributes{};
    InitializeObjectAttributes(&attributes, &string, OBJ_CASE_INSENSITIVE, nullptr, nullptr);
    const USHORT length = string.Length, maximum = string.MaximumLength;
    PWSTR buffer = string.Buffer;

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    NtCreateFileFn create = nullptr;
    NtCloseFn close = nullptr;
    if (!ntdll || !resolve(ntdll, create, close)) return 1;
    const size_t file_length = std::wcslen(file_name);

    WaitForSingleObject(shared->start, INFINITE);
    for (int index = 0; index < shared->opens; ++index) {
        HANDLE handle = nullptr;
        IO_STATUS_BLOCK status{};
        const NTSTATUS result = create(&handle, GENERIC_READ | SYNCHRONIZE, &attributes, &status, nullptr,
                                       FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_OPEN,
                                       FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, nullptr, 0);
        if (result < 0) return 1;
        // Which file did the call open? It has to be this thread's own, not the other thread's.
        wchar_t final_path[MAX_PATH * 2]{};
        const DWORD got = GetFinalPathNameByHandleW(handle, final_path, MAX_PATH * 2, FILE_NAME_NORMALIZED);
        const bool own_file = got > file_length && got < MAX_PATH * 2 &&
            _wcsicmp(final_path + got - file_length, file_name) == 0 && final_path[got - file_length - 1] == L'\\';
        close(handle);
        if (!own_file) {
            InterlockedIncrement(&shared->wrong_file);
            return 1;
        }
        if (string.Length != length || string.MaximumLength != maximum || string.Buffer != buffer ||
            std::wcscmp(string.Buffer, built) != 0) {
            InterlockedIncrement(&shared->broken);
            return 1;
        }
        InterlockedIncrement(&shared->completed);
        SwitchToThread();  // give the other thread a chance to be inside its own call
    }
    return 0;
}

int overlapping_names()
{
    Overlap shared{};
    shared.start = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    shared.letter = probe_letter();
    shared.opens = 32;
    if (!shared.start) return 1;
    Worker workers[2] = {{&shared, 0}, {&shared, 1}};
    HANDLE threads[2] = {CreateThread(nullptr, 0, overlap_thread, &workers[0], 0, nullptr),
                         CreateThread(nullptr, 0, overlap_thread, &workers[1], 0, nullptr)};
    if (!threads[0] || !threads[1]) return 1;
    SetEvent(shared.start);
    WaitForMultipleObjects(2, threads, TRUE, 30000);
    const DWORD first = WaitForSingleObject(threads[0], 0), second = WaitForSingleObject(threads[1], 0);
    for (HANDLE thread : threads) CloseHandle(thread);
    CloseHandle(shared.start);
    if (first != WAIT_OBJECT_0 || second != WAIT_OBJECT_0) {
        std::printf("overlap threads did not finish\n");
        return 1;
    }
    if (shared.wrong_file) {
        std::printf("overlap: %ld opens reached another thread's file (a name was shared between calls)\n", shared.wrong_file);
        return 1;
    }
    if (shared.broken) {
        std::printf("overlap: a caller name was changed while its call was in flight\n");
        return 1;
    }
    if (shared.completed != 2 * shared.opens) {
        std::printf("overlap incomplete: %ld of %d opens succeeded\n", shared.completed, 2 * shared.opens);
        return 1;
    }
    std::printf("overlap ok: %ld opens across two threads, both caller names intact\n", shared.completed);
    return 0;
}
}  // namespace

int main()
{
    wchar_t mode[16]{};
    const DWORD len = GetEnvironmentVariableW(L"X2W_PROBE_MODE", mode, 16);
    if (len == 5 && _wcsicmp(mode, L"reuse") == 0) return reused_string();
    if (len == 7 && _wcsicmp(mode, L"overlap") == 0) return overlapping_names();
    return single_open();
}