// Map an x64 PE image without LoadLibrary and bind its imports through a host resolver.
// Scope: DIR64/HIGHLOW relocations, ordinary and delay imports by name or ordinal, section protections,
// .pdata registration, TLS callbacks and the entry point. Delay imports are bound eagerly: the
// image's delay-load helper would call LoadLibrary and bypass the host resolver. An implicit TLS
// data template is served through a generated carrier DLL that Windows loads (see TlsCarrier).
// Header fields are checked against the file or SizeOfImage before use.
#include <windows.h>

#include <bcrypt.h>
#include <intrin.h>
#include <psapi.h>
#include <sddl.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <new>
#include <string>
#include <vector>

// Keep in step with STATUS in tests/cpu/native_map_check.py.
enum MapStatus : std::uint32_t {
    MAP_OK = 0,
    MAP_IO = 1,
    MAP_FORMAT = 2,
    MAP_MEMORY = 3,
    MAP_RELOCATION = 4,
    MAP_UNRESOLVED_IMPORT = 5,
    MAP_UNSUPPORTED = 6,
    MAP_PROTECT = 7,
    MAP_FUNCTION_TABLE = 8,
    MAP_INIT_FAILED = 9,
};

// Returns the address bound to one import, or null to reject it. name is null for an ordinal.
using Resolver = void*(__cdecl*)(const char* dll, const char* name, std::uint32_t ordinal);

namespace {

constexpr std::uint64_t MAX_FILE = 1ull << 30;

struct Failure {
    std::uint32_t* status;
    char* detail;
    std::size_t capacity;

    template <typename... Args>
    void* operator()(MapStatus code, const char* format, Args... args) const {
        if (status) *status = code;
        if (detail && capacity) std::snprintf(detail, capacity, format, args...);
        return nullptr;
    }
};

bool inside(std::uint64_t offset, std::uint64_t size, std::uint64_t limit) {
    return offset <= limit && size <= limit - offset;
}

const IMAGE_DATA_DIRECTORY& directory(const IMAGE_NT_HEADERS64* nt, int index) {
    static const IMAGE_DATA_DIRECTORY empty{};
    return nt->OptionalHeader.NumberOfRvaAndSizes > static_cast<DWORD>(index)
        ? nt->OptionalHeader.DataDirectory[index] : empty;
}

// A view of the mapped image that rejects any range outside SizeOfImage.
struct Image {
    std::uint8_t* base;
    std::uint64_t size;

    template <typename T>
    T* at(std::uint64_t rva, std::uint64_t count = 1) const {
        return inside(rva, sizeof(T) * count, size) ? reinterpret_cast<T*>(base + rva) : nullptr;
    }

    const char* string(std::uint64_t rva) const {
        if (rva >= size) return nullptr;
        const auto* start = base + rva;
        const bool terminated = std::memchr(start, 0, static_cast<std::size_t>(size - rva)) != nullptr;
        return terminated ? reinterpret_cast<const char*>(start) : nullptr;
    }

    const IMAGE_NT_HEADERS64* headers() const {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        return reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    }
};

DWORD protection(DWORD flags) {
    const bool exec = flags & IMAGE_SCN_MEM_EXECUTE, write = flags & IMAGE_SCN_MEM_WRITE;
    const bool read = flags & IMAGE_SCN_MEM_READ;
    if (exec) return write ? PAGE_EXECUTE_READWRITE : read ? PAGE_EXECUTE_READ : PAGE_EXECUTE;
    return write ? PAGE_READWRITE : read ? PAGE_READONLY : PAGE_NOACCESS;
}

// True if rva lies inside a section mapped as executable.
bool executable(const IMAGE_NT_HEADERS64* nt, std::uint64_t rva) {
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if ((s.Characteristics & IMAGE_SCN_MEM_EXECUTE) && rva >= s.VirtualAddress && rva - s.VirtualAddress < span)
            return true;
    }
    return false;
}

// True if [rva, rva + size) lies inside a section mapped as writable. The Windows loader
// writes the TLS index to AddressOfIndex, so that slot must be writable: measured on
// Windows 11, an index in the headers, .text, .rdata or .tlscb fails with ERROR_NOACCESS
// while another DWORD in writable .data loads. The check is by section characteristics,
// like executable() above, so it holds both after mapping and on release.
bool writable_section(const IMAGE_NT_HEADERS64* nt, std::uint64_t rva, std::uint64_t size) {
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if ((s.Characteristics & IMAGE_SCN_MEM_WRITE) && rva >= s.VirtualAddress
            && size <= span && rva - s.VirtualAddress <= span - size)
            return true;
    }
    return false;
}

// Windows x64 seeds the default /GS cookie from a load-config directory whose
// legacy prefix includes SEHandlerCount. A shorter prefix or an unused cookie
// leaves initialization to the image. Call after relocation and before guest code.
const char* seed_security_cookie(const Image& image, const IMAGE_NT_HEADERS64* nt, MapStatus& status) {
    const auto& dir = directory(nt, IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG);
    constexpr auto minimum = offsetof(IMAGE_LOAD_CONFIG_DIRECTORY64, GuardCFCheckFunctionPointer);
    constexpr auto guard_offset = offsetof(IMAGE_LOAD_CONFIG_DIRECTORY64, GuardFlags);
    constexpr std::uint64_t default_cookie = 0x2b992ddfa232ull;
    constexpr std::uint64_t value_mask = (1ull << 48) - 1;
    if (!dir.VirtualAddress || dir.Size < minimum) return nullptr;
    const auto* config = image.at<std::uint8_t>(dir.VirtualAddress, dir.Size);
    status = MAP_FORMAT;
    if (!config) return "load config outside image";
    DWORD declared_size;
    std::memcpy(&declared_size, config, sizeof(declared_size));
    if (declared_size < minimum) return nullptr;
    if (guard_offset + sizeof(DWORD) <= dir.Size && guard_offset + sizeof(DWORD) <= declared_size) {
        DWORD flags;
        std::memcpy(&flags, config + guard_offset, sizeof(flags));
        if (flags & IMAGE_GUARD_SECURITY_COOKIE_UNUSED) return nullptr;
    }
    std::uint64_t address;
    std::memcpy(&address, config + offsetof(IMAGE_LOAD_CONFIG_DIRECTORY64, SecurityCookie), sizeof(address));
    if (!address) return nullptr;
    const auto base = reinterpret_cast<std::uint64_t>(image.base);
    if (address < base || address % alignof(std::uint64_t))
        return "security cookie must be aligned and inside a writable image section";
    const auto rva = address - base;
    auto* target = image.at<std::uint64_t>(rva);
    if (!target || !writable_section(nt, rva, sizeof(*target)))
        return "security cookie must be aligned and inside a writable image section";
    if (*target != default_cookie) return nullptr;
    std::uint64_t random;
    do {
        if (BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(&random), sizeof(random),
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
            status = MAP_INIT_FAILED;
            return "cannot initialize the security cookie";
        }
        random &= value_mask;
    } while (!random || random == default_cookie);
    *target = random;
    return nullptr;
}

// True if every page of [address, address + size) is committed and currently allows reads.
// Image bounds alone do not make bytes readable: section protections apply, and the image may
// change them at run time, so the live protection is queried at each read, including on release.
// The query is not atomic with the read that follows: another thread reprotecting the same pages
// in between is not handled.
bool readable(const void* address, std::size_t size) {
    auto* cursor = static_cast<const std::uint8_t*>(address);
    const auto* end = cursor + size;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(cursor, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
        const DWORD access = info.Protect & 0xff;
        if ((info.Protect & PAGE_GUARD) || access == PAGE_NOACCESS || access == PAGE_EXECUTE) return false;
        cursor = static_cast<const std::uint8_t*>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

// True if the page at address is committed and neither guarded nor PAGE_NOACCESS. Unlike
// readable(), this accepts execute-only pages, which can hold exported code GetProcAddress returns.
bool accessible(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
    const DWORD access = info.Protect & 0xff;
    return !(info.Protect & PAGE_GUARD) && access != PAGE_NOACCESS;
}

// A NUL-terminated string inside the image, scanned only across readable committed pages.
// Returns null if the string would require reading past the image or an unreadable page.
const char* readable_string(const Image& image, std::uint64_t rva) {
    if (rva >= image.size) return nullptr;
    const auto* start = image.base + rva;
    for (std::uint64_t scanned = 0; scanned < image.size - rva;) {
        const auto* cursor = start + scanned;
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(cursor, &info, sizeof(info)) || info.State != MEM_COMMIT) return nullptr;
        const DWORD access = info.Protect & 0xff;
        if ((info.Protect & PAGE_GUARD) || access == PAGE_NOACCESS || access == PAGE_EXECUTE) return nullptr;
        const auto offset = static_cast<std::uint64_t>(cursor - static_cast<const std::uint8_t*>(info.BaseAddress));
        const std::uint64_t run = info.RegionSize - offset;
        const std::uint64_t left = image.size - rva - scanned;
        const std::uint64_t span = run < left ? run : left;
        if (std::memchr(cursor, 0, static_cast<std::size_t>(span))) return reinterpret_cast<const char*>(start);
        scanned += span;
    }
    return nullptr;
}

constexpr std::uint64_t MAX_TLS_CALLBACKS = 256;
constexpr const char* TOO_MANY_TLS_CALLBACKS = "more than 256 TLS callbacks";
// dlattrRva: descriptor fields hold RVAs instead of virtual addresses. Every x64 linker sets it.
constexpr DWORD DELAY_IMPORT_RVA_BASED = 1;

// TLS callbacks of a mapped, relocated image in table order. Returns null or the reason the list is rejected.
const char* tls_callbacks(const Image& image, const IMAGE_NT_HEADERS64* nt, std::vector<PIMAGE_TLS_CALLBACK>& out) {
    const auto& dir = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (!dir.VirtualAddress) return nullptr;
    const auto* tls = image.at<IMAGE_TLS_DIRECTORY64>(dir.VirtualAddress);
    if (!tls) return "TLS directory outside image";
    // Sections are already protected here, both after mapping and on release.
    if (!readable(tls, sizeof(IMAGE_TLS_DIRECTORY64))) return "TLS directory not readable";
    // Each value is read once after its check, so the validated value is the one used.
    // The loader assigns the TLS index even for an empty template, so the slot must be
    // validated whenever the directory is present, independent of the callback list.
    const auto base = reinterpret_cast<std::uint64_t>(image.base);
    const std::uint64_t index = tls->AddressOfIndex;
    if (index < base || index - base >= image.size || !image.at<DWORD>(index - base))
        return "TLS index outside image";
    if (!writable_section(nt, index - base, sizeof(DWORD))) return "TLS index outside writable section";
    const std::uint64_t list = tls->AddressOfCallBacks;
    if (!list) return nullptr;
    for (std::uint64_t i = 0; i <= MAX_TLS_CALLBACKS; ++i) {
        const std::uint64_t va = list + 8 * i;
        const auto* slot = va >= base ? image.at<std::uint64_t>(va - base) : nullptr;
        if (!slot) return "TLS callback list outside image";
        if (!readable(slot, sizeof(std::uint64_t))) return "TLS callback list not readable";
        const std::uint64_t callback = *slot;
        if (!callback) return nullptr;
        if (i == MAX_TLS_CALLBACKS) return TOO_MANY_TLS_CALLBACKS;
        if (callback < base || callback - base >= image.size || !executable(nt, callback - base))
            return "TLS callback outside executable section";
        out.push_back(reinterpret_cast<PIMAGE_TLS_CALLBACK>(callback));
    }
    return nullptr;
}

// Notify the image as the Windows loader does for LoadLibrary/FreeLibrary: TLS callbacks in table
// order, then the entry point, with a null reserved argument. Returns the entry point's result.
// Measured on Windows 11 build 26300: LoadLibrary sends nothing to a non-DLL image, neither TLS
// callbacks nor the entry point (for an EXE the entry point is the program itself).
BOOL notify(const Image& image, const IMAGE_NT_HEADERS64* nt, const std::vector<PIMAGE_TLS_CALLBACK>& callbacks,
            DWORD reason) {
    if (!(nt->FileHeader.Characteristics & IMAGE_FILE_DLL)) return TRUE;
    for (auto callback : callbacks) callback(image.base, reason, nullptr);
    if (!nt->OptionalHeader.AddressOfEntryPoint) return TRUE;
    using Entry = BOOL(WINAPI*)(HINSTANCE, DWORD, LPVOID);
    const auto entry = reinterpret_cast<Entry>(image.base + nt->OptionalHeader.AddressOfEntryPoint);
    return entry(reinterpret_cast<HINSTANCE>(image.base), reason, nullptr);
}

// Implicit TLS data. Compiled code reaches a thread-local through the thread's
// ThreadLocalStoragePointer array (gs:[0x58]), indexed by the DWORD at AddressOfIndex. ntdll owns
// that array: it sizes it, assigns the indexes, gives every thread a block per index and frees the
// blocks. pe_map does not touch it. For an image with a non-empty template it writes a small
// carrier DLL to a private directory under the temporary directory (see load_carrier) and loads
// it with LoadLibraryExW. The carrier has no code
// and imports nothing; it holds a TLS directory whose template is a copy of the guest's relocated template
// and whose SizeOfZeroFill and Characteristics (alignment) are the guest's. Windows then assigns
// the carrier a real index and allocates, aligns, initializes and frees one block per thread,
// for threads that already exist and for threads created later, as it would for the guest.
// pe_map stores the carrier's index at the guest's AddressOfIndex, so the guest's accesses land
// in the carrier's blocks. FreeLibrary on the carrier returns the index and the blocks; the
// carrier's file and directory are then deleted through the handles the map call holds.
//
// A process that ends with an image still mapped leaves its carrier directory behind, since
// deletion is only requested on release. pe_map does not remove such directories: nothing but a
// handle establishes that a carrier belongs to the caller.
//
// Differences from a Windows load of the guest: the template is copied once, when the image is
// mapped, so a later write to the template in the image does not reach threads created afterwards;
// and code that assumes index 0 without reading AddressOfIndex (/GA) is served only when the image is
// started as the main image (pe_map_run_main), on the threads that participate in it.
constexpr std::uint64_t MAX_TLS_TEMPLATE = 16ull << 20;

// One map call's carrier and everything it owns. Nothing else is ever deleted: cleanup goes through
// these handles, never through a name or a pattern.
struct TlsCarrier {
    HMODULE module = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE;       // read and delete access, shared for reading only
    HANDLE directory = INVALID_HANDLE_VALUE;  // the private directory that holds the file
};

// Deletes the object behind handle. POSIX semantics remove the name when the handle closes, which
// the directory needs: measured on Windows 10 build 19045, a carrier deleted with the classic
// disposition after FreeLibrary sometimes stays listed after its last handle closes (1 of 10 runs),
// and its directory then cannot be removed (ERROR_DIR_NOT_EMPTY). With POSIX semantics the name was
// gone and the directory removed in 10 of 10 runs. File systems without them get the classic one.
bool dispose(HANDLE handle) {
    FILE_DISPOSITION_INFO_EX posix{FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS};
    if (SetFileInformationByHandle(handle, FileDispositionInfoEx, &posix, sizeof(posix))) return true;
    const DWORD error = GetLastError();
    if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED && error != ERROR_INVALID_FUNCTION)
        return false;
    FILE_DISPOSITION_INFO classic{TRUE};
    return SetFileInformationByHandle(handle, FileDispositionInfo, &classic, sizeof(classic)) != FALSE;
}

// Unloads the carrier, then deletes its file and its directory through the handles it holds.
// Returns null, or the first cleanup step that failed with error holding GetLastError; whatever
// could not be deleted stays where it is.
const char* unload(TlsCarrier& carrier, DWORD& error) {
    const char* failed = nullptr;
    auto note = [&](const char* step) {
        if (!failed) {
            failed = step;
            error = GetLastError();
        }
    };
    if (carrier.module && !FreeLibrary(carrier.module)) note("cannot unload the TLS carrier");
    if (carrier.file != INVALID_HANDLE_VALUE) {
        if (!dispose(carrier.file)) note("cannot delete the TLS carrier file");
        CloseHandle(carrier.file);
    }
    if (carrier.directory != INVALID_HANDLE_VALUE) {
        if (!dispose(carrier.directory)) note("cannot remove the TLS carrier directory");
        CloseHandle(carrier.directory);
    }
    carrier = TlsCarrier{};
    return failed;
}

void unload(TlsCarrier& carrier) {
    DWORD ignored = 0;
    unload(carrier, ignored);
}

constexpr DWORD CARRIER_DATA_RVA = 0x1000, CARRIER_FILE_ALIGN = 0x200, CARRIER_PAGE = 0x1000;

constexpr DWORD align_up(DWORD value, DWORD alignment) { return (value + alignment - 1) & ~(alignment - 1); }

// The carrier's index slot holds this value until Windows assigns an index.
constexpr DWORD CARRIER_NO_INDEX = 0xffffffff;

// A relocatable PE32+ DLL with one data section (template, TLS directory, index, empty import
// table) and a .reloc section for the directory's three addresses. It has no code and no entry
// point. Measured on Windows 10 build 19045: a DLL without an import directory loads but is given
// no TLS index, with or without an entry point; an import directory that holds only the null
// descriptor is enough. index_rva receives the RVA of the carrier's TLS index.
std::vector<std::uint8_t> carrier_image(const std::uint8_t* tls_template, std::size_t length, DWORD zero_fill,
                                        DWORD characteristics, DWORD& index_rva) {
    constexpr std::uint64_t preferred = 0x180000000ull;
    // The directory is 40 bytes at a multiple of 64, so a page boundary never splits it.
    const DWORD directory_offset = align_up(static_cast<DWORD>(length), 64);
    const DWORD index_offset = directory_offset + sizeof(IMAGE_TLS_DIRECTORY64);
    const DWORD import_offset = index_offset + 8;
    const DWORD data_size = import_offset + align_up(static_cast<DWORD>(sizeof(IMAGE_IMPORT_DESCRIPTOR)), 8);
    const DWORD data_raw = align_up(data_size, CARRIER_FILE_ALIGN);
    const DWORD reloc_rva = align_up(CARRIER_DATA_RVA + data_size, CARRIER_PAGE);
    const DWORD reloc_size = sizeof(IMAGE_BASE_RELOCATION) + 4 * sizeof(WORD);
    const DWORD headers = CARRIER_FILE_ALIGN;
    std::vector<std::uint8_t> file(std::size_t{headers} + data_raw + CARRIER_FILE_ALIGN);

    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(file.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = sizeof(IMAGE_DOS_HEADER);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(file.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 2;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->FileHeader.Characteristics = IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_LARGE_ADDRESS_AWARE | IMAGE_FILE_DLL;
    auto& opt = nt->OptionalHeader;
    opt.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    opt.ImageBase = preferred;
    opt.SectionAlignment = CARRIER_PAGE;
    opt.FileAlignment = CARRIER_FILE_ALIGN;
    opt.MajorOperatingSystemVersion = 6;
    opt.MajorSubsystemVersion = 6;
    opt.SizeOfInitializedData = data_raw + CARRIER_FILE_ALIGN;
    opt.SizeOfImage = reloc_rva + CARRIER_PAGE;
    opt.SizeOfHeaders = headers;
    opt.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_GUI;
    opt.DllCharacteristics = IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE | IMAGE_DLLCHARACTERISTICS_NX_COMPAT
        | IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA;
    opt.SizeOfStackReserve = 0x100000;
    opt.SizeOfStackCommit = 0x1000;
    opt.SizeOfHeapReserve = 0x100000;
    opt.SizeOfHeapCommit = 0x1000;
    opt.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    opt.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS] = {CARRIER_DATA_RVA + directory_offset, sizeof(IMAGE_TLS_DIRECTORY64)};
    opt.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC] = {reloc_rva, reloc_size};
    // The import table is only its null descriptor, which the zero-filled buffer already holds.
    opt.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {CARRIER_DATA_RVA + import_offset, sizeof(IMAGE_IMPORT_DESCRIPTOR)};

    auto* sections = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sections[0].Name, ".tls", 4);
    sections[0].Misc.VirtualSize = data_size;
    sections[0].VirtualAddress = CARRIER_DATA_RVA;
    sections[0].SizeOfRawData = data_raw;
    sections[0].PointerToRawData = headers;
    sections[0].Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE;
    std::memcpy(sections[1].Name, ".reloc", 6);
    sections[1].Misc.VirtualSize = reloc_size;
    sections[1].VirtualAddress = reloc_rva;
    sections[1].SizeOfRawData = CARRIER_FILE_ALIGN;
    sections[1].PointerToRawData = headers + data_raw;
    sections[1].Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE;

    std::uint8_t* data = file.data() + headers;
    if (length) std::memcpy(data, tls_template, length);
    IMAGE_TLS_DIRECTORY64 tls{};
    tls.StartAddressOfRawData = preferred + CARRIER_DATA_RVA;
    tls.EndAddressOfRawData = tls.StartAddressOfRawData + length;
    tls.AddressOfIndex = preferred + CARRIER_DATA_RVA + index_offset;
    tls.SizeOfZeroFill = zero_fill;
    tls.Characteristics = characteristics;
    std::memcpy(data + directory_offset, &tls, sizeof(tls));
    std::memcpy(data + index_offset, &CARRIER_NO_INDEX, sizeof(CARRIER_NO_INDEX));

    // Base relocations for StartAddressOfRawData, EndAddressOfRawData and AddressOfIndex.
    const DWORD directory_rva = CARRIER_DATA_RVA + directory_offset;
    const IMAGE_BASE_RELOCATION block{directory_rva & ~(CARRIER_PAGE - 1), reloc_size};
    WORD entries[4] = {};  // the fourth entry stays IMAGE_REL_BASED_ABSOLUTE padding
    for (DWORD i = 0; i < 3; ++i)
        entries[i] = static_cast<WORD>((IMAGE_REL_BASED_DIR64 << 12) | ((directory_rva + 8 * i) & (CARRIER_PAGE - 1)));
    std::uint8_t* reloc = data + data_raw;
    std::memcpy(reloc, &block, sizeof(block));
    std::memcpy(reloc + sizeof(block), entries, sizeof(entries));

    index_rva = CARRIER_DATA_RVA + index_offset;
    return file;
}

constexpr wchar_t CARRIER_PREFIX[] = L"pe_map_tls_";

// The identity of the file behind handle: volume serial number and 128-bit file ID.
bool file_identity(HANDLE handle, FILE_ID_INFO& out) {
    return GetFileInformationByHandleEx(handle, FileIdInfo, &out, sizeof(out)) != FALSE;
}

bool same_file(const FILE_ID_INFO& a, const FILE_ID_INFO& b) {
    return a.VolumeSerialNumber == b.VolumeSerialNumber && std::memcmp(&a.FileId, &b.FileId, sizeof(a.FileId)) == 0;
}

// A protected DACL that grants full access to the process's user only, inherited by the carrier.
bool private_descriptor(PSECURITY_DESCRIPTOR& descriptor) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    union {
        TOKEN_USER user;
        std::uint8_t bytes[sizeof(TOKEN_USER) + SECURITY_MAX_SID_SIZE];
    } info;
    DWORD size = 0;
    const bool queried = GetTokenInformation(token, TokenUser, &info, sizeof(info), &size) != FALSE;
    CloseHandle(token);
    wchar_t* sid = nullptr;
    if (!queried || !ConvertSidToStringSidW(info.user.User.Sid, &sid)) return false;
    wchar_t sddl[32 + SECURITY_MAX_SID_SIZE * 4];
    std::swprintf(sddl, sizeof(sddl) / sizeof(sddl[0]), L"D:P(A;OICI;FA;;;%ls)", sid);
    LocalFree(sid);
    return ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &descriptor, nullptr)
        != FALSE;
}

// Writes the carrier into a directory of its own and loads it. The path must keep naming the file
// this call wrote until the loader has opened it:
// - The directory is new, under the temporary directory, with a 128-bit random name and a protected
//   DACL for the process's user only. It is held open without delete sharing, so it cannot be
//   renamed or removed. A reparse point in its place fails the map.
// - The file is created there with CREATE_NEW, written, and identified by its file ID. The write
//   handle has to be closed before the load, because the loader's open does not share writing.
// - It is reopened for reading and deletion, shared for reading only, so from then on no one can
//   write, rename or delete it. The reopened file must have the same ID, and its contents, read
//   back through this handle, must be the bytes written.
// - After the load, the file mapped as the carrier must be the file behind this handle.
// Measured on Windows 10 build 19045: while those two handles are held, opening the file for
// writing, renaming it, deleting it and renaming its directory fail with ERROR_SHARING_VIOLATION,
// renaming the directory above fails with ERROR_ACCESS_DENIED, and LoadLibraryExW still loads it.
// index receives the TLS index Windows assigned to the carrier. Returns null on success or the step
// that failed; error then holds GetLastError. On failure this call deletes what it created and
// still holds a handle to; nothing else is touched. If the carrier cannot be reopened, or the
// directory cannot be queried, the file or directory is left where it is, since without a
// verified handle this call cannot tell it is still its own.
const char* load_carrier(const std::vector<std::uint8_t>& file, DWORD index_rva, TlsCarrier& carrier, DWORD& index,
                         DWORD& error) {
    auto failure = [&](const char* step, DWORD code) {
        unload(carrier);
        error = code;
        return step;
    };
    wchar_t temporary[MAX_PATH + 1], directory[MAX_PATH + 1], path[MAX_PATH + 1];
    DWORD length = GetTempPathW(MAX_PATH + 1, temporary);
    if (length && length <= MAX_PATH) length = GetLongPathNameW(temporary, directory, MAX_PATH + 1);
    if (!length || length > MAX_PATH)
        return failure("cannot find the temporary directory for the TLS carrier", GetLastError());
    std::uint8_t random[16];
    const NTSTATUS generated = BCryptGenRandom(nullptr, random, sizeof(random), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (generated != 0)  // error holds the NTSTATUS here
        return failure("cannot generate a name for the TLS carrier", static_cast<DWORD>(generated));
    wchar_t name[sizeof(random) * 2 + 1];
    for (std::size_t i = 0; i < sizeof(random); ++i) std::swprintf(name + 2 * i, 3, L"%02x", random[i]);
    // The directory name, a separator, the file name and ".dll" must fit.
    constexpr std::size_t prefix = sizeof(CARRIER_PREFIX) / sizeof(CARRIER_PREFIX[0]) - 1;
    if (length + 2 * (prefix + sizeof(random) * 2) + 5 > MAX_PATH)
        return failure("the temporary directory path is too long for the TLS carrier", ERROR_FILENAME_EXCED_RANGE);
    std::swprintf(path, MAX_PATH + 1, L"%ls%ls%ls", directory, CARRIER_PREFIX, name);
    std::wcscpy(directory, path);

    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!private_descriptor(descriptor))
        return failure("cannot build the access list of the TLS carrier directory", GetLastError());
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), descriptor, FALSE};
    const BOOL created = CreateDirectoryW(directory, &attributes);
    const DWORD create_error = GetLastError();
    LocalFree(descriptor);
    if (!created) return failure("cannot create the TLS carrier directory", create_error);
    carrier.directory = CreateFileW(directory, DELETE | FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                    nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                                    nullptr);
    if (carrier.directory == INVALID_HANDLE_VALUE)
        return failure("cannot open the TLS carrier directory", GetLastError());
    FILE_BASIC_INFO basic{};
    if (!GetFileInformationByHandleEx(carrier.directory, FileBasicInfo, &basic, sizeof(basic))) {
        const DWORD code = GetLastError();
        CloseHandle(carrier.directory);  // unverified, so not removed
        carrier.directory = INVALID_HANDLE_VALUE;
        return failure("cannot query the TLS carrier directory", code);
    }
    if (!(basic.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) || (basic.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
        CloseHandle(carrier.directory);  // not the directory this call created, so not removed
        carrier.directory = INVALID_HANDLE_VALUE;
        return failure("the TLS carrier directory was replaced", ERROR_SUCCESS);
    }

    std::swprintf(path, MAX_PATH + 1, L"%ls\\%ls%ls.dll", directory, CARRIER_PREFIX, name);
    HANDLE out = CreateFileW(path, GENERIC_WRITE | DELETE, 0, nullptr, CREATE_NEW,
                             FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (out == INVALID_HANDLE_VALUE) return failure("cannot create the TLS carrier file", GetLastError());
    DWORD written = 0;
    FILE_ID_INFO wrote{};
    const bool ok = WriteFile(out, file.data(), static_cast<DWORD>(file.size()), &written, nullptr)
        && written == file.size() && file_identity(out, wrote);
    const DWORD write_error = GetLastError();
    if (!ok) {
        dispose(out);
        CloseHandle(out);
        return failure("cannot write the TLS carrier file", write_error);
    }
    CloseHandle(out);

    // Opened before the load: the loader's own open shares reading and deletion, so this handle can
    // be opened beside it, and it keeps any writer, rename or delete out until the carrier is unloaded.
    HANDLE held = CreateFileW(path, GENERIC_READ | DELETE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (held == INVALID_HANDLE_VALUE) return failure("cannot reopen the TLS carrier file", GetLastError());
    FILE_ID_INFO reopened{};
    if (!file_identity(held, reopened) || !same_file(wrote, reopened)) {
        const DWORD code = GetLastError();
        CloseHandle(held);  // not the file this call wrote, so not deleted
        return failure("the TLS carrier file was replaced before it was loaded", code);
    }
    carrier.file = held;
    std::vector<std::uint8_t> check(file.size() + 1);
    DWORD read = 0;
    if (!ReadFile(carrier.file, check.data(), static_cast<DWORD>(check.size()), &read, nullptr))
        return failure("cannot read back the TLS carrier file", GetLastError());
    if (read != file.size() || std::memcmp(check.data(), file.data(), file.size()) != 0)
        return failure("the TLS carrier file changed before it was loaded", ERROR_SUCCESS);

    carrier.module = LoadLibraryExW(path, nullptr, 0);
    if (!carrier.module) return failure("Windows cannot load the TLS carrier", GetLastError());
    constexpr DWORD NAME = MAX_PATH + 64;
    wchar_t mapped[NAME], held_name[NAME];
    const DWORD mapped_length = GetMappedFileNameW(GetCurrentProcess(), carrier.module, mapped, NAME);
    const DWORD held_length = GetFinalPathNameByHandleW(carrier.file, held_name, NAME, FILE_NAME_OPENED | VOLUME_NAME_NT);
    if (!mapped_length || mapped_length >= NAME || !held_length || held_length >= NAME)
        return failure("cannot name the file mapped as the TLS carrier", GetLastError());
    if (CompareStringOrdinal(mapped, -1, held_name, -1, TRUE) != CSTR_EQUAL)
        return failure("Windows mapped a different file as the TLS carrier", ERROR_SUCCESS);
    std::memcpy(&index, reinterpret_cast<const std::uint8_t*>(carrier.module) + index_rva, sizeof(index));
    if (index == CARRIER_NO_INDEX) return failure("Windows assigned no TLS index to the carrier", ERROR_SUCCESS);
    return nullptr;
}

bool read_file(const wchar_t* path, std::vector<std::uint8_t>& raw) {
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER length{};
    DWORD read = 0;
    bool ok = GetFileSizeEx(file, &length) && length.QuadPart > 0
        && static_cast<std::uint64_t>(length.QuadPart) <= MAX_FILE;
    if (ok) {
        try {
            raw.resize(static_cast<std::size_t>(length.QuadPart));
        } catch (const std::bad_alloc&) {
            ok = false;
        }
    }
    ok = ok && ReadFile(file, raw.data(), static_cast<DWORD>(raw.size()), &read, nullptr) && read == raw.size();
    CloseHandle(file);
    return ok;
}

// Images attached by pe_map_image. pe_map.dll is itself loaded by Windows, so its DllMain receives
// DLL_THREAD_ATTACH/DETACH and forwards them to these images: attach in mapping order, detach in
// reverse mapping order, as measured with two LoadLibraryW copies on Windows 11 build 26300.
//
// Lifetime protocol: each image holds one reference for its registration and one per thread
// notification in flight. DllMain takes those references under mapped_lock and calls the guest
// callbacks after releasing it, so a callback may map or release any image, its own included.
// pe_map_release unregisters the image and drops the registration reference without waiting. The
// image's process detach and unmapping run when the last reference goes, either inside
// pe_map_release or on the thread that finishes the last notification. A notification is admitted
// when its thread takes the reference under the registry lock; none is admitted after the release
// unregisters the image. An admitted notification skips an image already marked released, but one
// whose callback has started finishes: when a TLS callback releases its own image, that image's
// entry point still receives the same thread attach. Addresses are the only identity, so once a
// released address is reused by a new mapping, a stale pointer releases that new mapping.
SRWLOCK mapped_lock = SRWLOCK_INIT;

struct Mapped {
    std::uint8_t* base;
    volatile LONG references;
    volatile LONG released;  // set by pe_map_release; notifiers skip the image from then on
    TlsCarrier carrier;      // holds the image's implicit TLS blocks; empty without a template
    volatile LONG started = 0;  // set by pe_map_run for an EXE; its TLS callbacks then see thread notifications
};

struct SharedLock {
    SharedLock() { AcquireSRWLockShared(&mapped_lock); }
    ~SharedLock() { ReleaseSRWLockShared(&mapped_lock); }
    SharedLock(const SharedLock&) = delete;
    SharedLock& operator=(const SharedLock&) = delete;
};

struct ExclusiveLock {
    ExclusiveLock() { AcquireSRWLockExclusive(&mapped_lock); }
    ~ExclusiveLock() { ReleaseSRWLockExclusive(&mapped_lock); }
    ExclusiveLock(const ExclusiveLock&) = delete;
    ExclusiveLock& operator=(const ExclusiveLock&) = delete;
};

std::vector<Mapped*>& mapped_images() {
    static std::vector<Mapped*> images;
    return images;
}

// Main-image TLS slot 0 (pe_map_run_main). Code compiled with /GA reads an EXE's thread-locals through
// slot 0 of ThreadLocalStoragePointer without loading _tls_index, because Windows gives the process's EXE
// index 0. An EXE started as the main image needs its block, which Windows keeps at the carrier's index,
// in slot 0 of each thread that runs its code. Only participating threads get it, and every thread swaps
// its own slot 0, never another thread's:
// - the thread that calls pe_map_run_main, until pe_map_release on that thread;
// - threads enrolled through pe_map_main_create_thread, from their DLL_THREAD_ATTACH, before any image
//   is notified, to their DLL_THREAD_DETACH, after every image was notified;
// - host threads between their own pe_map_main_thread_enter and pe_map_main_thread_leave.
// A participant keeps its previous slot 0 in a pe_map TLS slot reserved at the start (TlsAlloc below
// TLS_MINIMUM_AVAILABLE, so setting it never allocates) and gets it back on leave, detach or release.
// The detach restore precedes ntdll's freeing of the thread's blocks, which would otherwise free the
// image's block twice (as slot 0's and as the carrier's). Each participant's thread handle goes into a
// fixed table. pe_map_release refuses the main image, returning 0 and changing nothing (the status
// API reports the refusal), while any participant other than the calling thread may still run the
// image's code: its handle is not signaled or cannot be waited on, or it has no handle (the table was
// full or the handle could not be duplicated). The image, its unwind data, its carrier and every participant's slot
// stay as they are, and the caller may retry once those participants have left or exited. A release that
// is accepted gives the calling thread its saved slot 0 back, counts participants whose handle is
// signaled as exited (they ended without a detach; their blocks went with them), ends the mode so that
// no thread joins any more, and then unregisters and frees the image once, as for any image. Threads
// that do not participate are never touched: neither their slot 0 nor their notifications, since the
// main image's TLS callbacks receive DLL_THREAD_ATTACH and DLL_THREAD_DETACH only on participating
// threads (a /GA callback on another thread would read the host's block). Other images, and EXEs
// started with pe_map_run, are notified as before.
// Guest threads are enrolled explicitly; no thread is classified by inspecting it. The host binds the main
// image's CreateThread imports to pe_map_main_create_thread, which reserves an entry in a fixed enrollment
// table, creates the thread suspended at a pe_map start routine and records its thread ID before the thread
// can run. Only a thread found in that table installs the image's block at its DLL_THREAD_ATTACH; every
// other thread, whatever its start address, is left alone and never ends the process. Enrollment fails
// without guest code running: when no entry can be reserved or the thread cannot be watched,
// pe_map_main_create_thread returns null with the last error set and no thread exists; when the block
// cannot be installed on the new thread, the pe_map start routine returns ENROLL_FAILED_EXIT without
// calling the guest's start routine, and the main image's callbacks were not notified because the thread
// does not participate. Both are counted in pe_map_main_tls_stats. An enrolled thread that has not yet
// passed its start routine counts as one that may run the image's code, so release refuses while it lives.
// Not enrolled, and so running with the host's slot 0: threads the image creates by any other route (a
// CreateThread address it looked up itself, CreateRemoteThread(Ex), NtCreateThreadEx, thread-pool work, a
// CRT's _beginthreadex in another DLL), and host threads that call the image without entering.
// On a participating thread the host's own index-0 code reads the image's block, so a host that owns
// slot 0 must not touch its thread-locals there.
// Rule and measurements: MoonYueya on #870.
constexpr std::size_t TEB_TLS_POINTER = 0x58;  // TEB.ThreadLocalStoragePointer on x64
constexpr int MAIN_TRACKED = 64;
constexpr ULONG_PTR UNTRACKED = 0x10000;   // participation value of an installed thread without a handle
constexpr int MAIN_ENROLLING = 64;
constexpr DWORD ENROLL_FAILED_EXIT = 0xE0000201u;  // exit code of an enrolled thread whose block could not be installed

// A thread enrolled by pe_map_main_create_thread, until its pe_map start routine has passed it on.
struct Enrollment {
    // CANCELLED: pe_map_main_create_thread refused the thread after creating it (ResumeThread failed) and is
    // ending it; the entry stays until its handle is signaled.
    enum State { FREE, RESERVED, READY, INSTALLED, FAILED, CANCELLED } state = FREE;
    DWORD thread = 0;         // its thread ID, recorded before it can run
    HANDLE handle = nullptr;  // SYNCHRONIZE, to tell an enrolled thread that ended before its start routine
    LPTHREAD_START_ROUTINE start = nullptr;
    void* parameter = nullptr;
};

struct MainImage {
    std::uint8_t* base = nullptr;  // the image started as the main image, or null
    std::size_t size = 0;
    DWORD index = 0;  // the carrier's TLS index, kept after release for late restores
    HANDLE tracked[MAIN_TRACKED] = {};
    std::int64_t untracked = 0;  // installed participants without a handle
    std::int64_t installed = 0, restored = 0, exited = 0, refusals = 0;
    std::int64_t last_release = -1;  // -1 none yet, 1 accepted, 2 refused (participants may still run the image)
    Enrollment enrolling[MAIN_ENROLLING];
    std::int64_t enroll_refused = 0;  // pe_map_main_create_thread returned null: no thread was created
    std::int64_t enroll_failed = 0;   // an enrolled thread ended without the guest's start routine
    int inject = 0;                   // pe_map_main_tls_inject: the next enrollment (1) or installation (2) fails
};

// Guards main_image(). Taken without mapped_lock held, and mapped_lock is never taken under it.
SRWLOCK slot_lock = SRWLOCK_INIT;

struct SlotLock {
    SlotLock() { AcquireSRWLockExclusive(&slot_lock); }
    ~SlotLock() { ReleaseSRWLockExclusive(&slot_lock); }
    SlotLock(const SlotLock&) = delete;
    SlotLock& operator=(const SlotLock&) = delete;
};

MainImage& main_image() {
    static MainImage image;
    return image;
}

// The main image's base while it is started, read without slot_lock by the notification path.
std::uint8_t* volatile g_main_base = nullptr;

// Process image (pe_map_run_main). Under a Windows launch the EXE is the process image: the PEB's
// ImageBaseAddress holds its base, and GetModuleHandle(NULL), and with it FindResource(NULL, ...), return
// that base. pe_map_run_main makes the mapped EXE the process image in the same field until its accepted
// pe_map_release, which restores the host's value. Guarded by slot_lock. Only one image at a time; the
// PEB's loader list (GetModuleFileName(NULL), GetModuleHandleEx on an image address) is not changed.
struct ProcessImage {
    std::uint8_t* base = nullptr;  // the mapped EXE reported as the process image, or null
    void* saved = nullptr;         // the host's ImageBaseAddress, written back on release
};
ProcessImage g_process_image;

void* volatile* peb_image_base() {
    // x64 PEB: ImageBaseAddress follows the four flag bytes and the Mutant handle.
    return reinterpret_cast<void* volatile*>(__readgsqword(0x60) + 0x10);
}

// Call with slot_lock held.
void report_process_image(std::uint8_t* base) {
    g_process_image.saved = InterlockedExchangePointer(peb_image_base(), base);
    g_process_image.base = base;
}

// Call with slot_lock held. Restores the host's value only if base is the reported process image.
void restore_process_image(std::uint8_t* base) {
    if (!base || g_process_image.base != base) return;
    InterlockedExchangePointer(peb_image_base(), g_process_image.saved);
    g_process_image = {};
}

// Entries in main_image().enrolling that are not FREE, read without slot_lock by DLL_THREAD_ATTACH. An entry
// becomes READY before its thread is resumed, so the thread itself always sees a nonzero count.
volatile LONG g_enrolling = 0;

// pe_map's own TLS slots: the calling thread's participation (0 none, 1..MAIN_TRACKED its table entry,
// UNTRACKED) and its saved slot 0. Allocated once under slot_lock and never freed.
DWORD g_participation = TLS_OUT_OF_INDEXES, g_saved = TLS_OUT_OF_INDEXES;

bool reserve_slots() {
    if (g_participation != TLS_OUT_OF_INDEXES) return true;
    const DWORD participation = TlsAlloc(), saved = TlsAlloc();
    if (participation < TLS_MINIMUM_AVAILABLE && saved < TLS_MINIMUM_AVAILABLE) {
        g_saved = saved;
        g_participation = participation;
        return true;
    }
    if (participation != TLS_OUT_OF_INDEXES) TlsFree(participation);
    if (saved != TLS_OUT_OF_INDEXES) TlsFree(saved);
    return false;
}

void** current_slots() {
    return *reinterpret_cast<void***>(reinterpret_cast<std::uint8_t*>(NtCurrentTeb()) + TEB_TLS_POINTER);
}

std::int64_t pending(const MainImage& main) {
    std::int64_t count = main.untracked;
    for (HANDLE handle : main.tracked) count += handle != nullptr;
    for (const auto& entry : main.enrolling) count += entry.state != Enrollment::FREE;
    return count;
}

// Takes a one-shot injected fault (pe_map_main_tls_inject). Caller holds slot_lock.
bool take_fault(MainImage& main, int fault) {
    if (main.inject != fault) return false;
    main.inject = 0;
    return true;
}

// Caller holds slot_lock.
void free_entry(Enrollment& entry) {
    if (entry.handle) CloseHandle(entry.handle);
    entry = {};
    InterlockedDecrement(&g_enrolling);
}

// A suspended worker can be cancelled before the wrapper gets a chance to consume its entry.
// Caller holds slot_lock. Keep live (including still-suspended) workers registered.
void reap_ended_enrollments(MainImage& main) {
    for (auto& entry : main.enrolling) {
        if (!entry.handle || WaitForSingleObject(entry.handle, 0) != WAIT_OBJECT_0) continue;
        // A cancelled thread was already counted as a refused enrollment when the guest got null.
        if (entry.state != Enrollment::CANCELLED) ++main.enroll_failed;
        free_entry(entry);
    }
}

// Installs the image's block in the calling thread's slot 0. Caller holds slot_lock, main.base is set.
// Nothing here allocates: the table is fixed and the TLS slots are below TLS_MINIMUM_AVAILABLE.
bool install_current(MainImage& main) {
    if (TlsGetValue(g_participation)) return true;
    void** slots = current_slots();
    if (!slots || !slots[main.index]) return false;
    TlsSetValue(g_saved, slots[0]);
    ULONG_PTR participation = UNTRACKED;
    for (int i = 0; i < MAIN_TRACKED; ++i) {
        if (main.tracked[i]) continue;
        HANDLE handle = nullptr;
        if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, SYNCHRONIZE, FALSE,
                            0)) {
            main.tracked[i] = handle;
            participation = static_cast<ULONG_PTR>(i) + 1;
        }
        break;
    }
    if (participation == UNTRACKED) ++main.untracked;
    TlsSetValue(g_participation, reinterpret_cast<void*>(participation));
    slots[0] = slots[main.index];
    ++main.installed;
    return true;
}

// Gives the calling thread its saved slot 0 back if it participates. Caller holds slot_lock. A slot 0
// that no longer holds the image's block is left as it is and the thread still stops participating.
bool restore_current(MainImage& main) {
    if (g_participation == TLS_OUT_OF_INDEXES) return false;
    const auto participation = reinterpret_cast<ULONG_PTR>(TlsGetValue(g_participation));
    if (!participation) return false;
    void** slots = current_slots();
    const bool holds = slots && slots[0] == slots[main.index];
    if (holds) {
        slots[0] = TlsGetValue(g_saved);
        ++main.restored;
    }
    if (participation == UNTRACKED) {
        --main.untracked;
    } else if (HANDLE& handle = main.tracked[participation - 1]) {
        CloseHandle(handle);
        handle = nullptr;
    }
    TlsSetValue(g_participation, nullptr);
    return holds;
}

// Whether the calling thread participates in the main image.
bool participates() {
    return g_participation != TLS_OUT_OF_INDEXES && TlsGetValue(g_participation) != nullptr;
}

// DLL_THREAD_ATTACH: a thread enrolled by pe_map_main_create_thread participates. Any other thread returns
// at once and is never touched. A failed installation only marks the entry; the thread's pe_map start
// routine then ends it without running guest code.
void install_enrolled_thread() {
    if (!InterlockedCompareExchange(&g_enrolling, 0, 0)) return;
    const SlotLock hold;
    auto& main = main_image();
    const DWORD self = GetCurrentThreadId();
    for (auto& entry : main.enrolling) {
        if (entry.state != Enrollment::READY || entry.thread != self ||
            WaitForSingleObject(entry.handle, 0) != WAIT_TIMEOUT) continue;
        const bool installed = main.base && !take_fault(main, 2) && install_current(main);
        entry.state = installed ? Enrollment::INSTALLED : Enrollment::FAILED;
        return;
    }
}

// The start routine of every enrolled thread: runs the guest's start routine only on a thread whose slot 0
// was installed at its attach.
DWORD WINAPI enrolled_thread_start(void* parameter) {
    LPTHREAD_START_ROUTINE start = nullptr;
    void* guest_parameter = nullptr;
    {
        const SlotLock hold;
        auto& main = main_image();
        auto& entry = main.enrolling[reinterpret_cast<ULONG_PTR>(parameter)];
        if (entry.state == Enrollment::INSTALLED) {
            start = entry.start;
            guest_parameter = entry.parameter;
        } else {
            ++main.enroll_failed;  // FAILED, or a thread whose attach pe_map never saw
        }
        free_entry(entry);
    }
    return start ? start(guest_parameter) : ENROLL_FAILED_EXIT;
}

// DLL_THREAD_DETACH, after every image was notified.
void restore_current_thread() {
    if (g_participation == TLS_OUT_OF_INDEXES || !TlsGetValue(g_participation)) return;
    const SlotLock hold;
    restore_current(main_image());
}

// Ends main-image mode for base before pe_map_release unregisters it, or refuses: returns false and
// changes nothing while a participant other than the calling thread may still run the image's code (a
// handle that is not signaled or cannot be waited on, or a participant without a handle). Checking and
// ending under one slot_lock hold means no participant can join in between. Returns true for any other
// pointer.
bool end_main(std::uint8_t* base) {
    const SlotLock hold;
    auto& main = main_image();
    if (main.base != base) return true;
    const auto self = g_participation == TLS_OUT_OF_INDEXES
                          ? ULONG_PTR{0}
                          : reinterpret_cast<ULONG_PTR>(TlsGetValue(g_participation));
    bool live = main.untracked > (self == UNTRACKED ? 1 : 0);
    for (int i = 0; i < MAIN_TRACKED && !live; ++i) {
        if (main.tracked[i] && static_cast<ULONG_PTR>(i) + 1 != self)
            live = WaitForSingleObject(main.tracked[i], 0) != WAIT_OBJECT_0;
    }
    // An enrolled thread that has not passed its start routine may still call into the image.
    for (int i = 0; i < MAIN_ENROLLING && !live; ++i) {
        const auto& entry = main.enrolling[i];
        if (entry.state != Enrollment::FREE)
            live = !entry.handle || WaitForSingleObject(entry.handle, 0) != WAIT_OBJECT_0;
    }
    if (live) {
        ++main.refusals;
        main.last_release = 2;
        return false;
    }
    restore_current(main);
    for (HANDLE& handle : main.tracked) {
        if (!handle) continue;
        ++main.exited;  // ended without a detach; its blocks went with it
        CloseHandle(handle);
        handle = nullptr;
    }
    reap_ended_enrollments(main);
    main.base = nullptr;
    g_main_base = nullptr;
    main.last_release = 1;
    return true;
}

// Notify a mapped image, re-reading its callback list as Windows does; if the list was
// overwritten or made unreadable, the entry point is still notified. An EXE is notified only after
// pe_map_run started it (`started`), and only its TLS callbacks receive thread attach and detach, as
// under a Windows launch where the EXE's entry point is the program and is never called again. An EXE
// never receives process detach here, so its release (finalize) calls none of its code.
void notify_mapped(std::uint8_t* base, DWORD reason, bool started = false) {
    const auto* nt = Image{base, 0}.headers();
    const Image image{base, nt->OptionalHeader.SizeOfImage};
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    try {
        if (tls_callbacks(image, nt, callbacks)) callbacks.clear();
    } catch (const std::bad_alloc&) {
        // Never let an allocation failure escape DllMain or pe_map_release; the entry point is
        // still notified.
        callbacks.clear();
    }
    if (!(nt->FileHeader.Characteristics & IMAGE_FILE_DLL)) {
        if (started && (reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH))
            for (auto callback : callbacks) callback(image.base, reason, nullptr);
        return;
    }
    notify(image, nt, callbacks, reason);
}

// Process detach, function-table removal, TLS release and unmapping, once no reference remains.
// The TLS blocks outlive the detach notifications, as they do under FreeLibrary.
void finalize(Mapped* record) {
    std::uint8_t* base = record->base;
    const auto* nt = Image{base, 0}.headers();
    notify_mapped(base, DLL_PROCESS_DETACH);
    const auto& pdata = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    if (pdata.Size) RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(base + pdata.VirtualAddress));
    unload(record->carrier);
    VirtualFree(base, 0, MEM_RELEASE);
}

void drop_reference(Mapped* record) {
    if (InterlockedDecrement(&record->references) != 0) return;
    finalize(record);
    delete record;
}

}  // namespace

namespace {
void notify_thread(DWORD reason) {
    {
        std::vector<Mapped*> snapshot;
        {
            const SharedLock hold;
            try {
                snapshot.reserve(mapped_images().size());
            } catch (const std::bad_alloc&) {
                return;  // no references were taken, so nothing is left pending
            }
            for (auto* record : mapped_images()) {
                InterlockedIncrement(&record->references);
                snapshot.push_back(record);
            }
        }
        // Guest callbacks run without mapped_lock. An image released meanwhile is still mapped
        // until its last reference is dropped here.
        auto notify_one = [reason](Mapped* record) {
            // The main image is notified only on threads that participate in it.
            const bool main_image_elsewhere = record->base == g_main_base && !participates();
            if (!InterlockedCompareExchange(&record->released, 0, 0) && !main_image_elsewhere)
                notify_mapped(record->base, reason, InterlockedCompareExchange(&record->started, 0, 0) != 0);
            drop_reference(record);
        };
        if (reason == DLL_THREAD_ATTACH) {
            for (auto it = snapshot.begin(); it != snapshot.end(); ++it) notify_one(*it);
        } else {
            for (auto it = snapshot.rbegin(); it != snapshot.rend(); ++it) notify_one(*it);
        }
    }
}
}  // namespace

// A thread enrolled for the main image gets the image's block in slot 0 before any image is notified of
// its attach, and every participant gets its saved slot 0 back after every image was notified of its
// detach (see "Main-image TLS slot 0").
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_THREAD_ATTACH) {
        install_enrolled_thread();
        notify_thread(reason);
    } else if (reason == DLL_THREAD_DETACH) {
        notify_thread(reason);
        restore_current_thread();
    }
    return TRUE;
}

extern "C" __declspec(dllexport) void* pe_map_image(const wchar_t* path, Resolver resolve,
                                                    std::uint32_t* status, char* detail,
                                                    std::size_t capacity) {
    const Failure fail{status, detail, capacity};
    if (status) *status = MAP_OK;
    if (detail && capacity) detail[0] = '\0';

    std::vector<std::uint8_t> raw;
    if (!read_file(path, raw)) return fail(MAP_IO, "cannot read image");
    if (raw.size() < sizeof(IMAGE_DOS_HEADER)) return fail(MAP_FORMAT, "truncated DOS header");
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(raw.data());
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0
        || !inside(dos->e_lfanew, sizeof(IMAGE_NT_HEADERS64), raw.size()))
        return fail(MAP_FORMAT, "bad DOS header");
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(raw.data() + dos->e_lfanew);
    const auto& opt = nt->OptionalHeader;
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64
        || opt.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return fail(MAP_FORMAT, "not an x64 PE32+ image");
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    const std::uint64_t table_end =
        reinterpret_cast<const std::uint8_t*>(sections + nt->FileHeader.NumberOfSections) - raw.data();
    if (nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64)
        || dos->e_lfanew + sizeof(IMAGE_NT_HEADERS64) > opt.SizeOfHeaders || table_end > opt.SizeOfHeaders
        || opt.SizeOfHeaders > opt.SizeOfImage || opt.SizeOfHeaders > raw.size())
        return fail(MAP_FORMAT, "section table outside headers");
    // Section protections are applied per page, so a section that started inside a page shared with
    // the headers or another section would change that page's protection as well. Windows refuses an
    // alignment that is not a power of two and a section that does not start on it. An alignment
    // below the page size, which Windows maps as one view when FileAlignment matches, is not modeled.
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    if (opt.SectionAlignment & (opt.SectionAlignment - 1))
        return fail(MAP_FORMAT, "section alignment %#lx is not a power of two", opt.SectionAlignment);
    if (opt.SectionAlignment < system.dwPageSize)
        return fail(MAP_UNSUPPORTED, "section alignment %#lx below the page size", opt.SectionAlignment);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        if (sections[i].VirtualAddress & (opt.SectionAlignment - 1))
            return fail(MAP_FORMAT, "section %.8s not aligned to %#lx",
                        reinterpret_cast<const char*>(sections[i].Name), opt.SectionAlignment);
    }

    if (opt.AddressOfEntryPoint && !executable(nt, opt.AddressOfEntryPoint))
        return fail(MAP_FORMAT, "entry point outside executable section");

    // Map away from the preferred base so the relocation path always runs.
    auto allocate = [&] {
        return static_cast<std::uint8_t*>(
            VirtualAlloc(nullptr, opt.SizeOfImage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    };
    auto* base = allocate();
    if (base == reinterpret_cast<std::uint8_t*>(opt.ImageBase)) {
        auto* other = allocate();
        VirtualFree(base, 0, MEM_RELEASE);
        base = other;
    }
    if (!base) return fail(MAP_MEMORY, "cannot reserve image");
    const Image image{base, opt.SizeOfImage};
    TlsCarrier carrier;
    // A carrier that cannot be fully cleaned up is reported after the failure that caused the release.
    auto release = [&](void* result) {
        DWORD error = 0;
        const char* cleanup = unload(carrier, error);
        const std::size_t used = cleanup && detail && capacity ? std::strlen(detail) : capacity;
        if (used + 1 < capacity) std::snprintf(detail + used, capacity - used, "; %s (error %lu)", cleanup, error);
        VirtualFree(base, 0, MEM_RELEASE);
        return result;
    };

    std::memcpy(base, raw.data(), opt.SizeOfHeaders);
    const WORD section_count = nt->FileHeader.NumberOfSections;
    for (WORD i = 0; i < section_count; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        const char* name = reinterpret_cast<const char*>(s.Name);
        if (s.VirtualAddress < opt.SizeOfHeaders || !inside(s.VirtualAddress, span, opt.SizeOfImage))
            return release(fail(MAP_FORMAT, "section %.8s outside image", name));
        // Like the Windows loader, check raw data against the file by two rules only: no raw range
        // may reach 4 GiB, and the last section's raw data must end inside the file. Raw data of an
        // earlier section may lie past the end of the file; the missing bytes stay zero.
        const std::uint64_t raw_end = std::uint64_t{s.PointerToRawData} + s.SizeOfRawData;
        if (s.SizeOfRawData && raw_end > 0xffffffffull)
            return release(fail(MAP_FORMAT, "section %.8s raw data reaches 4 GiB", name));
        if (s.SizeOfRawData && i + 1 == section_count && raw_end > raw.size())
            return release(fail(MAP_FORMAT, "last section %.8s raw data runs past the file", name));
        // A section without a file offset stays zero-filled even when SizeOfRawData is set; it does
        // not take the headers at file offset 0.
        if (!s.PointerToRawData || !s.SizeOfRawData) continue;
        // Read whole 0x200-byte file sectors: round the start of the raw data down and its end up.
        const std::uint64_t first = s.PointerToRawData & ~std::uint64_t{0x1ff};
        std::uint64_t end = (raw_end + 0x1ff) & ~std::uint64_t{0x1ff};
        if (end > raw.size()) end = raw.size();
        if (first >= end) continue;
        const std::uint64_t length = end - first < span ? end - first : span;
        std::memcpy(base + s.VirtualAddress, raw.data() + first, static_cast<std::size_t>(length));
    }

    // A TLS data template is placed after the sections are protected, below. Unlike Windows,
    // which assigns an index even for an empty template, pe_map leaves AddressOfIndex as linked
    // when the template is empty and SizeOfZeroFill is zero.
    // Like the Windows loader, the TLS directory is found by its address alone; Size is not used.
    // Measured on Windows 11 build 26200: a TLS directory with Size 0, 1 or 39 still runs its
    // callback, one with address 0 is ignored whatever its Size, and one that does not fit in the
    // image fails the load.
    const auto& tls = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (tls.VirtualAddress) {
        const auto* t = image.at<IMAGE_TLS_DIRECTORY64>(tls.VirtualAddress);
        if (!t) return release(fail(MAP_FORMAT, "TLS directory outside image"));
    }

    const auto& imports = directory(nt, IMAGE_DIRECTORY_ENTRY_IMPORT);
    const auto& delay = directory(nt, IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT);

    // Writes below (relocations, IAT) must land in sections, never in the headers or in the
    // directory that drives them, so validated structures cannot change while they are walked.
    auto overlaps = [&](std::uint64_t rva, const IMAGE_DATA_DIRECTORY& driver, std::uint64_t width = 8) {
        const std::uint64_t start = driver.VirtualAddress, end = start + driver.Size;
        return rva + width > start && rva < end;
    };
    auto writable = [&](std::uint64_t rva, const IMAGE_DATA_DIRECTORY& driver, std::uint64_t width = 8) {
        return rva >= opt.SizeOfHeaders && inside(rva, width, image.size) && !overlaps(rva, driver, width);
    };

    // Only complete relocation blocks and complete WORD entries can be applied.
    const auto delta = reinterpret_cast<std::uint64_t>(base) - opt.ImageBase;
    // Like the Windows loader (RtlImageDirectoryEntryToData), a relocation directory at address 0
    // is absent whatever its Size: the image maps as relocation-free instead of the headers being
    // parsed as relocation blocks.
    const auto& linked_relocs = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
    static const IMAGE_DATA_DIRECTORY no_relocs{};
    const auto& relocs = linked_relocs.VirtualAddress ? linked_relocs : no_relocs;
    if (nt->FileHeader.Characteristics & IMAGE_FILE_RELOCS_STRIPPED)
        return release(fail(MAP_RELOCATION, "relocations stripped"));
    if (!inside(relocs.VirtualAddress, relocs.Size, opt.SizeOfImage))
        return release(fail(MAP_RELOCATION, "relocation directory outside image"));
    if (relocs.VirtualAddress && relocs.Size) {
        std::uint64_t offset = 0;
        while (offset + sizeof(IMAGE_BASE_RELOCATION) <= relocs.Size) {
            const auto* block = image.at<IMAGE_BASE_RELOCATION>(relocs.VirtualAddress + offset);
            const DWORD page = block->VirtualAddress, length = block->SizeOfBlock;
            if (length < sizeof(IMAGE_BASE_RELOCATION) || length > relocs.Size - offset || length % sizeof(WORD))
                return release(fail(MAP_RELOCATION, "bad relocation block"));
            const auto* entry = reinterpret_cast<const WORD*>(block + 1);
            const DWORD count = (length - sizeof(IMAGE_BASE_RELOCATION)) / sizeof(WORD);
            for (DWORD k = 0; k < count; ++k) {
                const WORD type = entry[k] >> 12;
                const std::uint64_t target = page + std::uint64_t{entry[k] & 0xfffu};
                if (type == IMAGE_REL_BASED_ABSOLUTE) continue;
                const std::uint64_t width = type == IMAGE_REL_BASED_DIR64 ? 8 : type == IMAGE_REL_BASED_HIGHLOW ? 4 : 0;
                if (width == 0 || !writable(target, relocs, width))
                    return release(fail(MAP_RELOCATION, "unsupported relocation at RVA %#llx",
                                        static_cast<unsigned long long>(target)));
                // Entries need not be naturally aligned. Copy only their architectural
                // width: HIGHLOW must neither touch nor carry into the next four bytes.
                if (type == IMAGE_REL_BASED_HIGHLOW) {
                    std::uint32_t value;
                    std::memcpy(&value, image.base + target, sizeof(value));
                    value += static_cast<std::uint32_t>(delta);
                    std::memcpy(image.base + target, &value, sizeof(value));
                } else {
                    std::uint64_t value;
                    std::memcpy(&value, image.base + target, sizeof(value));
                    value += delta;
                    std::memcpy(image.base + target, &value, sizeof(value));
                }
            }
            offset += length;
        }
        if (offset != relocs.Size)
            return release(fail(MAP_RELOCATION, "bad relocation block"));
    }

    // The delay-import directory is found by its address, like Windows; Size is not used. Walk both
    // tables to their null descriptors before either is bound, so an IAT write cannot change the
    // table the other walk reads.
    std::uint64_t delay_descriptors = 0;
    IMAGE_DATA_DIRECTORY delay_walked{};
    if (delay.VirtualAddress) {
        for (;; ++delay_descriptors) {
            const auto* d = image.at<IMAGE_DELAYLOAD_DESCRIPTOR>(
                delay.VirtualAddress + delay_descriptors * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR));
            if (!d) return release(fail(MAP_FORMAT, "delay import directory outside image"));
            if (!d->DllNameRVA) break;
        }
        // The walked table, terminator included, is what IAT writes must not overlap.
        delay_walked = {delay.VirtualAddress,
                        static_cast<DWORD>((delay_descriptors + 1) * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR))};
    }

    // Like the Windows loader, find the import table by its address and walk it to the null
    // descriptor; the directory Size is not used. Measured on Windows 11 builds 26200 and 26300:
    // copies with Size 0, 19 or 20 still load and bind every slot. Each descriptor is
    // bounds-checked on its own.
    std::uint64_t import_descriptors = 0;
    IMAGE_DATA_DIRECTORY imports_walked{};
    if (imports.VirtualAddress) {
        for (;; ++import_descriptors) {
            const auto* d = image.at<IMAGE_IMPORT_DESCRIPTOR>(
                imports.VirtualAddress + import_descriptors * sizeof(IMAGE_IMPORT_DESCRIPTOR));
            if (!d) return release(fail(MAP_FORMAT, "import directory outside image"));
            if (!d->Name) break;
        }
        // The walked table, terminator included, is what IAT writes must not overlap.
        imports_walked = {imports.VirtualAddress,
                          static_cast<DWORD>((import_descriptors + 1) * sizeof(IMAGE_IMPORT_DESCRIPTOR))};
    }

    // The PE loader leaves delay imports to the image's helper, which calls LoadLibrary and would
    // bypass the host resolver. pe_map has no LoadLibrary, so it resolves each delay descriptor
    // through the same resolver at map time and writes the bound address into its IAT. A descriptor
    // without dlattrRva stores virtual addresses rather than RVAs and is rejected. The directory is
    // found by its address, like Windows; Size is not used, and the table is walked above.
    if (delay_descriptors) {
        if (!resolve) return release(fail(MAP_UNRESOLVED_IMPORT, "delay imports present without a resolver"));
        const auto* desc = image.at<IMAGE_DELAYLOAD_DESCRIPTOR>(delay.VirtualAddress, delay_descriptors + 1);
        if (!desc) return release(fail(MAP_FORMAT, "delay import directory outside image"));
        for (std::uint64_t d = 0; d < delay_descriptors; ++d) {
            if (!(desc[d].Attributes.AllAttributes & DELAY_IMPORT_RVA_BASED))
                return release(fail(MAP_UNSUPPORTED, "delay imports without RVA attributes"));
            const char* name_in_image = image.string(desc[d].DllNameRVA);
            if (!name_in_image) return release(fail(MAP_FORMAT, "delay import DLL name outside image"));
            // Keep a copy: an IAT write below may legally overlap the string in the image.
            const std::string dll_name(name_in_image);
            const char* dll = dll_name.c_str();
            const DWORD lookup = desc[d].ImportNameTableRVA, iat = desc[d].ImportAddressTableRVA;
            if (!lookup || !iat)
                return release(fail(MAP_FORMAT, "delay import tables for %s outside image", dll));
            for (std::uint64_t i = 0;; ++i) {
                const auto* word = image.at<std::uint64_t>(lookup + 8 * i);
                if (!word) return release(fail(MAP_FORMAT, "delay import thunk for %s outside image", dll));
                const std::uint64_t slot = iat + 8 * i;
                if (!writable(slot, delay_walked) || overlaps(slot, imports_walked))
                    return release(fail(MAP_FORMAT, "delay import IAT for %s outside image", dll));
                if (!*word) break;
                const bool by_ordinal = IMAGE_SNAP_BY_ORDINAL64(*word);
                const auto ordinal = by_ordinal ? static_cast<std::uint32_t>(IMAGE_ORDINAL64(*word)) : 0u;
                // Bits 31..62 of a named thunk must be zero: masking them could alias a valid low RVA.
                const std::uint64_t name_rva = by_ordinal ? 0 : *word;
                const auto* hint = by_ordinal || name_rva >> 31 ? nullptr : image.at<WORD>(name_rva);
                const char* name = hint ? image.string(name_rva + sizeof(WORD)) : nullptr;
                if (!by_ordinal && !name)
                    return release(fail(MAP_FORMAT, "delay import name for %s outside image", dll));
                void* bound = resolve(dll, name, ordinal);
                if (!bound && name)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved delay import %s!%s", dll, name));
                if (!bound)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved delay import %s!#%u", dll,
                                        static_cast<unsigned>(ordinal)));
                *image.at<std::uint64_t>(slot) = reinterpret_cast<std::uint64_t>(bound);
            }
        }
    }

    if (import_descriptors) {
        if (!resolve) return release(fail(MAP_UNRESOLVED_IMPORT, "imports present without a resolver"));
        const auto* desc = image.at<IMAGE_IMPORT_DESCRIPTOR>(imports.VirtualAddress, import_descriptors + 1);
        for (std::uint64_t d = 0; d < import_descriptors; ++d) {
            const char* name_in_image = image.string(desc[d].Name);
            if (!name_in_image) return release(fail(MAP_FORMAT, "import DLL name outside image"));
            // Keep a copy: an IAT write below may legally overlap the string in the image.
            const std::string dll_name(name_in_image);
            const char* dll = dll_name.c_str();
            const DWORD lookup = desc[d].OriginalFirstThunk ? desc[d].OriginalFirstThunk : desc[d].FirstThunk;
            for (std::uint64_t i = 0;; ++i) {
                const auto* word = image.at<std::uint64_t>(lookup + 8 * i);
                const std::uint64_t target = desc[d].FirstThunk + 8 * i;
                // The delay directory was read above, but a normal IAT write must still not reach it.
                if (!word || !writable(target, imports_walked) || overlaps(target, delay_walked))
                    return release(fail(MAP_FORMAT, "import thunk for %s outside image", dll));
                if (!*word) break;
                const bool by_ordinal = IMAGE_SNAP_BY_ORDINAL64(*word);
                const auto ordinal = by_ordinal ? static_cast<std::uint32_t>(IMAGE_ORDINAL64(*word)) : 0u;
                // Bits 31..62 of a named thunk must be zero: masking them could alias a valid low RVA.
                const std::uint64_t name_rva = by_ordinal ? 0 : *word;
                const auto* hint = by_ordinal || name_rva >> 31 ? nullptr : image.at<WORD>(name_rva);
                const char* name = hint ? image.string(name_rva + sizeof(WORD)) : nullptr;
                if (!by_ordinal && !name) return release(fail(MAP_FORMAT, "import name for %s outside image", dll));
                void* bound = resolve(dll, name, ordinal);
                if (!bound && name) return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved import %s!%s", dll, name));
                if (!bound)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved import %s!#%u", dll,
                                        static_cast<unsigned>(ordinal)));
                *image.at<std::uint64_t>(target) = reinterpret_cast<std::uint64_t>(bound);
            }
        }
    }

    MapStatus cookie_status = MAP_OK;
    if (const char* reason = seed_security_cookie(image, nt, cookie_status))
        return release(fail(cookie_status, "%s", reason));

    // Record the actual base in the mapped headers, as the Windows loader does.
    reinterpret_cast<IMAGE_NT_HEADERS64*>(base + dos->e_lfanew)->OptionalHeader.ImageBase =
        reinterpret_cast<std::uint64_t>(base);
    DWORD old = 0;
    if (!VirtualProtect(base, opt.SizeOfHeaders, PAGE_READONLY, &old))
        return release(fail(MAP_PROTECT, "cannot protect headers"));
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if (span && !VirtualProtect(base + s.VirtualAddress, span, protection(s.Characteristics), &old))
            return release(fail(MAP_PROTECT, "cannot protect section %.8s", reinterpret_cast<const char*>(s.Name)));
    }
    FlushInstructionCache(GetCurrentProcess(), base, opt.SizeOfImage);

    // An empty exception directory registers nothing. Require complete entries otherwise.
    const auto& pdata = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    if (pdata.Size) {
        if (!pdata.VirtualAddress || pdata.Size % sizeof(RUNTIME_FUNCTION))
            return release(fail(MAP_FUNCTION_TABLE, "cannot register .pdata"));
        const auto count = pdata.Size / sizeof(RUNTIME_FUNCTION);
        auto* table = image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress, count);
        if (table && !readable(table, pdata.Size))
            return release(fail(MAP_UNSUPPORTED, "exception table is not readable after section protection"));
        if (!table || !RtlAddFunctionTable(table, static_cast<DWORD>(count),
                                           reinterpret_cast<DWORD64>(base)))
            return release(fail(MAP_FUNCTION_TABLE, "cannot register .pdata"));
    }
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    const auto* mapped_nt = image.headers();
    if (const char* reason = tls_callbacks(image, mapped_nt, callbacks)) {
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        const bool limit = reason == TOO_MANY_TLS_CALLBACKS;
        return release(fail(limit ? MAP_UNSUPPORTED : MAP_FORMAT, "%s", reason));
    }
    // Implicit TLS data, before any guest code runs. tls_callbacks() has validated the directory
    // and AddressOfIndex. Measured on Windows 10 build 19045 with tests/cpu/tls_data.cpp: a template
    // that ends past the image fails the load with ERROR_NOACCESS and one that ends before its
    // start with ERROR_BAD_EXE_FORMAT; a non-zero SizeOfZeroFill and every Characteristics value
    // tried still load, so both fields are handed to Windows unchanged in the carrier.
    if (tls.VirtualAddress) {
        auto fail_tls = [&](MapStatus code, const char* reason) {
            if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
            return release(fail(code, "%s", reason));
        };
        const auto* t = image.at<IMAGE_TLS_DIRECTORY64>(tls.VirtualAddress);
        const auto address = reinterpret_cast<std::uint64_t>(base);
        const std::uint64_t start = t->StartAddressOfRawData, end = t->EndAddressOfRawData;
        const DWORD zero_fill = t->SizeOfZeroFill, characteristics = t->Characteristics;
        const std::uint64_t index_va = t->AddressOfIndex;
        if (end != start || zero_fill) {
            if (end < start) return fail_tls(MAP_FORMAT, "TLS template ends before it starts");
            // An empty template names no bytes, so its address is not checked: the linker may leave it 0.
            if (end != start && (start < address || !inside(start - address, end - start, image.size)))
                return fail_tls(MAP_FORMAT, "TLS template outside image");
            if (end - start > MAX_TLS_TEMPLATE || zero_fill > MAX_TLS_TEMPLATE)
                return fail_tls(MAP_UNSUPPORTED, "TLS template larger than 16 MiB");
            const std::uint8_t* bytes = end != start ? base + (start - address) : base;
            const auto length = static_cast<std::size_t>(end - start);
            if (!readable(bytes, length)) return fail_tls(MAP_UNSUPPORTED, "TLS template not readable");
            // The slot lies inside a writable section of the image, see tls_callbacks().
            auto* slot = reinterpret_cast<std::uint8_t*>(index_va);
            DWORD index_rva = 0, index = 0, error = 0;
            const char* failed = nullptr;
            try {
                const auto file = carrier_image(bytes, length, zero_fill, characteristics, index_rva);
                failed = load_carrier(file, index_rva, carrier, index, error);
            } catch (const std::bad_alloc&) {
                return fail_tls(MAP_MEMORY, "cannot allocate the TLS carrier");
            }
            if (failed) {
                if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
                return release(fail(MAP_MEMORY, "%s (error %lu)", failed, error));
            }
            std::memcpy(slot, &index, sizeof(index));
        }
    }
    if (!notify(image, mapped_nt, callbacks, DLL_PROCESS_ATTACH)) {
        // Measured on Windows: a failed attach is followed by a full detach before the image is freed.
        // The callback list collected before attach is reused here; only pe_map_release re-reads it.
        notify(image, mapped_nt, callbacks, DLL_PROCESS_DETACH);
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        return release(fail(MAP_INIT_FAILED, "entry point returned FALSE"));
    }
    Mapped* record = new (std::nothrow) Mapped{base, 1, 0, carrier};
    bool registered = record != nullptr;
    if (record) {
        const ExclusiveLock hold;
        try {
            mapped_images().push_back(record);
        } catch (const std::bad_alloc&) {
            registered = false;
        }
    }
    if (!registered) {
        delete record;
        notify(image, mapped_nt, callbacks, DLL_PROCESS_DETACH);
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        return release(fail(MAP_MEMORY, "cannot register image for thread notifications"));
    }
    carrier = TlsCarrier{};  // owned by the record from here on
    return base;
}

namespace {

struct ExportLookup {
    DWORD rva = 0;
    const char* forwarder = nullptr;
};

// Finds one named export. The export directory, its three tables and the forwarder string are
// checked against the live page protection, as the TLS path does.
bool find_export(const Image& image, const IMAGE_NT_HEADERS64* nt, const char* name, ExportLookup& out) {
    const auto& exports = directory(nt, IMAGE_DIRECTORY_ENTRY_EXPORT);
    if (exports.Size < sizeof(IMAGE_EXPORT_DIRECTORY) || !inside(exports.VirtualAddress, exports.Size, image.size))
        return false;
    const auto* dir = exports.Size ? image.at<IMAGE_EXPORT_DIRECTORY>(exports.VirtualAddress) : nullptr;
    if (!dir || !readable(dir, sizeof(IMAGE_EXPORT_DIRECTORY))) return false;
    const auto* names = image.at<DWORD>(dir->AddressOfNames, dir->NumberOfNames);
    const auto* ordinals = image.at<WORD>(dir->AddressOfNameOrdinals, dir->NumberOfNames);
    const auto* functions = image.at<DWORD>(dir->AddressOfFunctions, dir->NumberOfFunctions);
    if (!names || !ordinals || !functions) return false;
    if (!readable(names, sizeof(DWORD) * dir->NumberOfNames) ||
        !readable(ordinals, sizeof(WORD) * dir->NumberOfNames) ||
        !readable(functions, sizeof(DWORD) * dir->NumberOfFunctions)) return false;
    for (DWORD i = 0; i < dir->NumberOfNames; ++i) {
        const char* exported = readable_string(image, names[i]);
        if (!exported || std::strcmp(exported, name) != 0) continue;
        if (ordinals[i] >= dir->NumberOfFunctions) return false;
        const DWORD rva = functions[ordinals[i]];
        // A forwarded export points back into the export directory, where its target string lives.
        if (rva >= exports.VirtualAddress && rva - exports.VirtualAddress < exports.Size) {
            const Image directory_view{image.base, std::uint64_t{exports.VirtualAddress} + exports.Size};
            out.forwarder = readable_string(directory_view, rva);
            return out.forwarder != nullptr;
        }
        out.rva = rva;
        return true;
    }
    return false;
}

}  // namespace

// Return an address from the export address table by its public ordinal, including
// NONAME entries. An ordinal is Base + address-table index, not a name index.
// Like pe_map_export, this deliberately refuses forwarders and holes; the package
// runner resolves forwarder chains before asking the mapper for their final target.
// The caller keeps the mapping returned by pe_map_image live throughout lookup.
extern "C" __declspec(dllexport) void* pe_map_export_ordinal(void* mapped, std::uint32_t ordinal) {
    if (!mapped || !ordinal || ordinal > 65535) return nullptr;
    const auto* nt = Image{static_cast<std::uint8_t*>(mapped), 0}.headers();
    const Image image{static_cast<std::uint8_t*>(mapped), nt->OptionalHeader.SizeOfImage};
    const auto& exports = directory(nt, IMAGE_DIRECTORY_ENTRY_EXPORT);
    if (exports.Size < sizeof(IMAGE_EXPORT_DIRECTORY)
        || !inside(exports.VirtualAddress, exports.Size, image.size)) return nullptr;
    const auto* dir = image.at<IMAGE_EXPORT_DIRECTORY>(exports.VirtualAddress);
    if (!dir || !readable(dir, sizeof(*dir)) || ordinal < dir->Base) return nullptr;
    const std::uint64_t index = std::uint64_t{ordinal} - dir->Base;
    if (index >= dir->NumberOfFunctions || dir->NumberOfFunctions > image.size / sizeof(DWORD)) return nullptr;
    const auto* functions = image.at<DWORD>(dir->AddressOfFunctions, dir->NumberOfFunctions);
    if (!functions || !readable(functions, sizeof(DWORD) * dir->NumberOfFunctions)) return nullptr;
    const DWORD rva = functions[index];
    if (!rva || rva >= image.size ||
        (rva >= exports.VirtualAddress && rva - exports.VirtualAddress < exports.Size)) return nullptr;
    return accessible(image.base + rva) ? image.base + rva : nullptr;
}

// Only valid for a base returned by pe_map_image, whose headers were validated and are read-only.
extern "C" __declspec(dllexport) void* pe_map_export(void* mapped, const char* name) {
    const auto* nt = Image{static_cast<std::uint8_t*>(mapped), 0}.headers();
    const Image image{static_cast<std::uint8_t*>(mapped), nt->OptionalHeader.SizeOfImage};
    ExportLookup found;
    // A forwarded export is not code in this image, so this entry point keeps returning null.
    if (!find_export(image, nt, name, found) || found.forwarder) return nullptr;
    // An empty slot (RVA 0) names no code; returning the image base would be wrong. An exported
    // address must be accessible, but it may be execute-only, so readable() is not used here.
    return found.rva && found.rva < image.size && accessible(image.base + found.rva)
        ? image.base + found.rva : nullptr;
}

// Like pe_map_export, but a forwarded export is resolved through the host resolver. The forwarder
// string is "Module.Symbol" or "Module.#ordinal", with the module as the linker recorded it.
extern "C" __declspec(dllexport) void* pe_map_export_resolved(void* mapped, const char* name, Resolver resolve) {
    const auto* nt = Image{static_cast<std::uint8_t*>(mapped), 0}.headers();
    const Image image{static_cast<std::uint8_t*>(mapped), nt->OptionalHeader.SizeOfImage};
    ExportLookup found;
    if (!find_export(image, nt, name, found)) return nullptr;
    if (!found.forwarder)
        return found.rva && found.rva < image.size && accessible(image.base + found.rva)
            ? image.base + found.rva : nullptr;
    if (!resolve) return nullptr;
    const std::string target(found.forwarder);
    const auto dot = target.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 == target.size()) return nullptr;
    const std::string module = target.substr(0, dot);
    const std::string symbol = target.substr(dot + 1);
    if (symbol.front() == '#') {
        std::uint32_t ordinal = 0;
        for (std::size_t i = 1; i < symbol.size(); ++i) {
            const char digit = symbol[i];
            if (digit < '0' || digit > '9') return nullptr;
            ordinal = ordinal * 10 + static_cast<unsigned>(digit - '0');
            if (ordinal > 65535) return nullptr;
        }
        return ordinal ? resolve(module.c_str(), nullptr, ordinal) : nullptr;
    }
    return resolve(module.c_str(), symbol.c_str(), 0);
}

// Returns 1 after unregistering an image that pe_map_image returned, and 0 for null, for any
// other pointer and for a second release. Process detach and unmapping follow the protocol above:
// at once if no thread notification for the image is in flight, otherwise when the last one ends.
// Returns 0 without changing anything for the image started with pe_map_run_main while another
// participant may still run its code (see end_main); pe_map_main_tls_stats then reports the refusal
// and the caller may retry later.
extern "C" __declspec(dllexport) int pe_map_release(void* mapped) {
    if (!end_main(static_cast<std::uint8_t*>(mapped))) return 0;
    {
        const SlotLock hold;
        restore_process_image(static_cast<std::uint8_t*>(mapped));
    }
    Mapped* record = nullptr;
    {
        const ExclusiveLock hold;
        auto& images = mapped_images();
        for (auto it = images.begin(); it != images.end(); ++it) {
            if ((*it)->base == mapped) {
                record = *it;
                images.erase(it);
                break;
            }
        }
    }
    if (!record) return 0;
    InterlockedExchange(&record->released, 1);
    drop_reference(record);
    return 1;
}

// Start a mapped EXE on the calling thread the way the loader starts a process: TLS callbacks in table
// order with DLL_PROCESS_ATTACH and a null reserved argument, then the entry point, whose return value is
// the exit code Windows would use. Measured on Windows 11 build 26300 with tests/cpu/exe_start.asm.
//
// Preconditions, none of which is checked:
// - `mapped` is a base returned by pe_map_image, still live and not yet released. Like pe_map_export,
//   this reads the headers of the argument directly, so any other pointer is undefined behavior.
// - The caller trusts the image. The entry point and the callbacks are arbitrary machine code that runs
//   in the host process with the host's privileges. They can crash, hang or terminate it (an entry point
//   that calls ExitProcess ends the host), and they run on the calling thread, so their stack, TLS and
//   loader-lock state are the caller's. Run untrusted or unknown images in a child process.
// pe_map_image maps an EXE without notifying it, like LoadLibrary, so nothing has run before this call.
// Start once per mapping. The caller keeps it live throughout the call and prevents any thread,
// TLS callback or entry point from re-entering run or releasing this mapping during startup.
//
// Returns 0 without running anything for a DLL, which pe_map_image already notified, for an image without
// an entry point, for a null exit_code and for an unreadable TLS callback list. The PEB argument that
// Windows passes to the entry point is not forwarded, and no process-detach notification follows because
// the process goes on.
// Once the process-attach callbacks have returned, and until pe_map_release, threads that attach or detach
// call the EXE's TLS callbacks with DLL_THREAD_ATTACH or DLL_THREAD_DETACH, never its entry point, as under
// a Windows launch. That covers the threads the entry point starts. A thread that attaches during the
// process-attach callbacks is not notified, whoever started it; Windows instead holds such a thread until
// process initialization ends, which pe_map does not reproduce. Release sends the EXE no process detach.
// Measured on Windows 10 build 19045 with tests/cpu/exe_tls.cpp.
namespace {
bool startable(const IMAGE_NT_HEADERS64* nt, const std::uint32_t* exit_code) {
    return exit_code && !(nt->FileHeader.Characteristics & IMAGE_FILE_DLL) && nt->OptionalHeader.AddressOfEntryPoint;
}

int start_exe(std::uint8_t* base, const IMAGE_NT_HEADERS64* nt, std::vector<PIMAGE_TLS_CALLBACK>& callbacks,
              std::uint32_t* exit_code) {
    for (auto callback : callbacks) callback(base, DLL_PROCESS_ATTACH, nullptr);
    {
        // Thread notifications become eligible only now: after every process-attach callback has
        // returned and before the entry point runs (see notify_mapped). A thread that attaches while
        // the process-attach callbacks run, whether a callback or the host started it, is not notified.
        const SharedLock hold;
        for (auto* record : mapped_images())
            if (record->base == base) InterlockedExchange(&record->started, 1);
    }
    using Entry = DWORD(WINAPI*)(void*);
    *exit_code = reinterpret_cast<Entry>(base + nt->OptionalHeader.AddressOfEntryPoint)(nullptr);
    return 1;
}
}  // namespace

extern "C" __declspec(dllexport) int pe_map_run(void* mapped, std::uint32_t* exit_code) {
    auto* base = static_cast<std::uint8_t*>(mapped);
    const auto* nt = Image{base, 0}.headers();
    if (!startable(nt, exit_code)) return 0;
    const Image image{base, nt->OptionalHeader.SizeOfImage};
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    if (tls_callbacks(image, nt, callbacks)) return 0;
    return start_exe(base, nt, callbacks, exit_code);
}

// pe_map_run for the image the host treats as its main executable (see "Main-image TLS slot 0"): the
// image's TLS block goes into the calling thread's slot 0 before the process-attach callbacks, into the
// slot 0 of each thread enrolled through pe_map_main_create_thread (the host binds the image's
// CreateThread imports to it), and into that of host threads that call pe_map_main_thread_enter. Same preconditions and results as pe_map_run, and in addition returns 0
// without running anything when another image is started as the main image, when a participant of an
// earlier one is still pending, or when pe_map's own TLS slots cannot be reserved. An image without a TLS
// template has no block to install and simply starts. Either way the image becomes the process image (see
// "Process image") before its process-attach callbacks, until its accepted pe_map_release.
extern "C" __declspec(dllexport) int pe_map_run_main(void* mapped, std::uint32_t* exit_code) {
    auto* base = static_cast<std::uint8_t*>(mapped);
    const auto* nt = Image{base, 0}.headers();
    if (!startable(nt, exit_code)) return 0;
    const Image image{base, nt->OptionalHeader.SizeOfImage};
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    if (tls_callbacks(image, nt, callbacks)) return 0;
    bool carrier = false;
    {
        const SharedLock hold;
        for (auto* record : mapped_images())
            if (record->base == base) carrier = record->carrier.module != nullptr;
    }
    {
        const SlotLock hold;
        if (g_process_image.base) return 0;  // another image is the process image
        if (carrier) {
            auto& main = main_image();
            if (main.base || pending(main) || !reserve_slots()) return 0;
            const auto& tls = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
            const auto* t = image.at<IMAGE_TLS_DIRECTORY64>(tls.VirtualAddress);
            std::memcpy(&main.index, reinterpret_cast<const void*>(t->AddressOfIndex), sizeof(main.index));
            main.base = base;
            main.size = nt->OptionalHeader.SizeOfImage;
            if (!install_current(main)) {
                main.base = nullptr;
                return 0;
            }
            g_main_base = base;
        }
        report_process_image(base);
    }
    return start_exe(base, nt, callbacks, exit_code);
}

// A host thread joins the main image: its slot 0 gets the image's block until it calls
// pe_map_main_thread_leave. Returns 1 when installed, 0 when no image is started as the main image or the
// thread has no TLS array.
extern "C" __declspec(dllexport) int pe_map_main_thread_enter() {
    const SlotLock hold;
    auto& main = main_image();
    return main.base && install_current(main) ? 1 : 0;
}

// The calling thread leaves the main image. Returns 1 when its saved slot 0 was put back, 0 when it did
// not participate or slot 0 no longer held the image's block (slot left alone; the thread stops participating).
extern "C" __declspec(dllexport) int pe_map_main_thread_leave() {
    const SlotLock hold;
    return restore_current(main_image()) ? 1 : 0;
}

// CreateThread for the main image: the host binds the image's CreateThread imports here. Same parameters,
// results and last error as CreateThread. While an image is started as the main image the new thread is
// enrolled: it is created suspended at a pe_map start routine, its thread ID is recorded before it can run,
// it gets the image's block in slot 0 at its DLL_THREAD_ATTACH, and only then is the guest's start routine
// called; it is resumed here unless flags has CREATE_SUSPENDED. Returns null with ERROR_NOT_ENOUGH_MEMORY and
// creates no thread when the enrollment table is full or the thread cannot be watched. When ResumeThread
// fails, it returns null with ResumeThread's last error, never a suspended handle: the thread is ended before
// it ran anything, and its entry stays reserved until the thread has exited. When the block
// cannot be installed, the thread ends with ENROLL_FAILED_EXIT (0xE0000201) without running the guest's start
// routine. Both failures are counted in pe_map_main_tls_stats. With no main image it is CreateThread.
extern "C" __declspec(dllexport) HANDLE WINAPI pe_map_main_create_thread(LPSECURITY_ATTRIBUTES attributes,
                                                                          SIZE_T stack, LPTHREAD_START_ROUTINE start,
                                                                          LPVOID parameter, DWORD flags,
                                                                          LPDWORD thread_id) {
    int slot = -1;
    {
        const SlotLock hold;
        auto& main = main_image();
        if (main.base) {
            reap_ended_enrollments(main);
            if (!take_fault(main, 1)) {
                for (int i = 0; i < MAIN_ENROLLING && slot < 0; ++i)
                    if (main.enrolling[i].state == Enrollment::FREE) slot = i;
            }
            if (slot < 0) {
                ++main.enroll_refused;
                SetLastError(ERROR_NOT_ENOUGH_MEMORY);
                return nullptr;
            }
            auto& entry = main.enrolling[slot];
            entry.state = Enrollment::RESERVED;
            entry.start = start;
            entry.parameter = parameter;
            InterlockedIncrement(&g_enrolling);
        }
    }
    if (slot < 0) return CreateThread(attributes, stack, start, parameter, flags, thread_id);
    DWORD id = 0;
    HANDLE thread = CreateThread(attributes, stack, enrolled_thread_start,
                                 reinterpret_cast<void*>(static_cast<ULONG_PTR>(slot)), flags | CREATE_SUSPENDED, &id);
    DWORD error = thread ? ERROR_SUCCESS : GetLastError();
    HANDLE watch = nullptr;
    if (thread && !DuplicateHandle(GetCurrentProcess(), thread, GetCurrentProcess(), &watch, SYNCHRONIZE, FALSE, 0)) {
        // Never resumed, so it has run nothing; without a handle release could not tell when it ends.
        error = ERROR_NOT_ENOUGH_MEMORY;
        TerminateThread(thread, ENROLL_FAILED_EXIT);
        CloseHandle(thread);
        thread = nullptr;
    }
    {
        const SlotLock hold;
        auto& main = main_image();
        auto& entry = main.enrolling[slot];
        if (!thread) {
            free_entry(entry);
            ++main.enroll_refused;
        } else {
            entry.thread = id;
            entry.handle = watch;
            entry.state = Enrollment::READY;
        }
    }
    if (!thread) {
        SetLastError(error);
        return nullptr;
    }
    if (thread_id) *thread_id = id;  // before the thread can run, as the entry's thread ID
    if (flags & CREATE_SUSPENDED) return thread;
    DWORD resumed = DWORD(-1);
    {
        const SlotLock hold;
        if (take_fault(main_image(), 3))
            SetLastError(ERROR_ACCESS_DENIED);  // as if ResumeThread had failed
        else
            resumed = 0;
    }
    if (resumed == 0) resumed = ResumeThread(thread);
    if (resumed != DWORD(-1)) return thread;
    // The thread is still suspended at the pe_map start routine and has run nothing, guest code included.
    // Refuse the creation: end the thread and keep its entry and watch handle until the handle is signaled,
    // so release keeps refusing and the slot is not reused while the thread could still exist.
    error = GetLastError();
    {
        const SlotLock hold;
        auto& main = main_image();
        main.enrolling[slot].state = Enrollment::CANCELLED;
        ++main.enroll_refused;
    }
    TerminateThread(thread, ENROLL_FAILED_EXIT);
    // The enrollment reaper can close watch as soon as the worker is signaled.
    WaitForSingleObject(thread, 10000);
    CloseHandle(thread);
    {
        const SlotLock hold;
        reap_ended_enrollments(main_image());  // frees the entry now if the thread has ended
    }
    if (thread_id) *thread_id = 0;
    SetLastError(error);
    return nullptr;
}

// Test-only fault injection for the main-image enrollment: 1 makes the next pe_map_main_create_thread fail
// to enroll, 2 makes the next installation on an enrolled thread fail, 3 makes the next ResumeThread in
// pe_map_main_create_thread fail with ERROR_ACCESS_DENIED, 0 clears. One shot.
extern "C" __declspec(dllexport) void pe_map_main_tls_inject(int fault) {
    const SlotLock hold;
    main_image().inject = fault;
}

// Main-image slot 0 counters since the process started, for tests: out[0] installs, out[1] restores,
// out[2] participants confirmed exited at a release, out[3] refused releases, out[4] participants and
// enrolled threads still pending, out[5] 1 while an image is started as the main image, out[6] the last
// release (-1 none, 1 accepted, 2 refused), out[7] installed participants without a handle, out[8]
// enrollments refused (null returned to the caller), out[9] enrolled threads ended without the guest's
// start routine.
// out must hold 10 entries.
extern "C" __declspec(dllexport) void pe_map_main_tls_stats(std::int64_t* out) {
    if (!out) return;
    const SlotLock hold;
    const auto& main = main_image();
    out[0] = main.installed;
    out[1] = main.restored;
    out[2] = main.exited;
    out[3] = main.refusals;
    out[4] = pending(main);
    out[5] = main.base != nullptr;
    out[6] = main.last_release;
    out[7] = main.untracked;
    out[8] = main.enroll_refused;
    out[9] = main.enroll_failed;
}
