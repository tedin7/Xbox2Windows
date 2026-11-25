// CRT-free EXE that asks the process for its own image, as an EXE does under a Windows launch.
//
// GetModuleHandleW(NULL) returns the PEB's ImageBaseAddress, and FindResourceW(NULL, ...) looks
// resources up in that module. Under a Windows launch that is this EXE. A loader that maps the EXE
// into another process must report the mapped image there, or the EXE reads the host's headers and
// resources instead of its own.
//
// The log, one line on standard output:
//   self=<y/n>   GetModuleHandleW(NULL) is this image (&__ImageBase)
//   hdr=<y/n>    the PE headers reached through that handle name this entry point
//   res=<y/n>    FindResourceW(NULL, 101, RT_RCDATA) finds this image's resource with its exact bytes
//   miss=<y/n>   FindResourceW(NULL, 102, RT_RCDATA), absent from this image, is not found
//   k32=<y/n>    GetModuleHandleW(L"kernel32.dll") still returns a module (unaffected control)
//   X=<exit code> the number of n results
#include <windows.h>

#pragma runtime_checks("", off)  // CRT-free: no /RTC helpers in Debug builds

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace {
// Must match exe_process_image.rc.
constexpr char resource_bytes[] = "xbox2windows process image resource 101";

void write(const char* text)
{
    DWORD length = 0;
    while (text[length]) ++length;
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text, length, &written, nullptr);
}

bool own_resource()
{
    HRSRC found = FindResourceW(nullptr, MAKEINTRESOURCEW(101), MAKEINTRESOURCEW(10));  // RT_RCDATA
    if (!found) return false;
    const DWORD size = SizeofResource(nullptr, found);
    HGLOBAL loaded = LoadResource(nullptr, found);
    const auto* bytes = static_cast<const char*>(loaded ? LockResource(loaded) : nullptr);
    if (!bytes || size != sizeof(resource_bytes)) return false;
    for (DWORD i = 0; i < size; ++i)
        if (bytes[i] != resource_bytes[i]) return false;
    // The resource lies inside this image.
    const auto* base = reinterpret_cast<const char*>(&__ImageBase);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + __ImageBase.e_lfanew);
    return bytes >= base && bytes + size <= base + nt->OptionalHeader.SizeOfImage;
}
}  // namespace

extern "C" DWORD WINAPI exe_entry();

extern "C" DWORD WINAPI exe_entry()
{
    const auto* process_image = reinterpret_cast<const BYTE*>(GetModuleHandleW(nullptr));
    const bool self = process_image == reinterpret_cast<const BYTE*>(&__ImageBase);
    bool hdr = false;
    if (process_image && reinterpret_cast<const IMAGE_DOS_HEADER*>(process_image)->e_magic == IMAGE_DOS_SIGNATURE) {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(process_image);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(process_image + dos->e_lfanew);
        hdr = nt->Signature == IMAGE_NT_SIGNATURE &&
              process_image + nt->OptionalHeader.AddressOfEntryPoint == reinterpret_cast<const BYTE*>(&exe_entry);
    }
    const bool res = own_resource();
    const bool miss = !FindResourceW(nullptr, MAKEINTRESOURCEW(102), MAKEINTRESOURCEW(10));
    const bool k32 = GetModuleHandleW(L"kernel32.dll") != nullptr;

    const bool results[] = {self, hdr, res, miss, k32};
    DWORD failures = 0;
    for (bool ok : results) failures += ok ? 0 : 1;
    char line[] = "LOG:self=? hdr=? res=? miss=? k32=? X=?\n";
    const int marks[] = {9, 15, 21, 28, 34};
    for (int i = 0; i < 5; ++i) line[marks[i]] = results[i] ? 'y' : 'n';
    line[38] = static_cast<char>('0' + failures);
    write(line);
    return failures;
}
