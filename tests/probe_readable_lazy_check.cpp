// probe_readable on lazily committed memory (X2W_XMEM_LAZY_COMMIT=1): probing 1 MiB of mapped, untouched pages
// raises no exception (the old probe faulted once per 64 KiB slot, and every fault is a stop under a debugger),
// returns the whole range, and the bytes read back are the ones written. usage: probe_readable_lazy_check <xmem_shim.dll>
#include "startup_stubs/local_log.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr SIZE_T page = 65536;
constexpr SIZE_T pages = 16;  // 1 MiB
constexpr DWORD physical_64k = 0x20400000, startup_reserve = 0x20402000;

volatile LONG exceptions = 0;

LONG CALLBACK count_exceptions(EXCEPTION_POINTERS* info)
{
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_GUARD_PAGE) InterlockedIncrement(&exceptions);
    return EXCEPTION_CONTINUE_SEARCH;
}

void require(bool ok, const char* what)
{
    if (ok) return;
    std::printf("FAIL: %s (GetLastError=%lu)\n", what, static_cast<unsigned long>(GetLastError()));
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    SetEnvironmentVariableA("X2W_XMEM_LAZY_COMMIT", "1");
    HMODULE shim = LoadLibraryExA(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    require(shim != nullptr, "load xmem shim");
    using Allocate = int(WINAPI*)(SIZE_T, SIZE_T*, SIZE_T*);
    using Map = void*(WINAPI*)(void*, SIZE_T, SIZE_T*);
    using Reserve = void*(WINAPI*)(void*, SIZE_T, DWORD, SIZE_T, DWORD);
    const auto allocate = reinterpret_cast<Allocate>(GetProcAddress(shim, "XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<Map>(GetProcAddress(shim, "XMemMapPhysicalPages"));
    const auto reserve = reinterpret_cast<Reserve>(GetProcAddress(shim, "XMemVirtualAlloc"));
    require(allocate && map && reserve, "shim exports");
    auto* base = static_cast<unsigned char*>(reserve(nullptr, SIZE_T{16} << 30, startup_reserve, 0x21, 4));
    require(base != nullptr, "reserve the startup region");
    SIZE_T got = pages;
    SIZE_T tokens[pages] = {};
    require(allocate(physical_64k, &got, tokens) && got == pages, "allocate 16 physical pages");
    require(map(base, pages, tokens) == base, "map them contiguously");

    // Lazy: nothing is committed until touched.
    MEMORY_BASIC_INFORMATION info{};
    require(VirtualQuery(base + 5 * page, &info, sizeof(info)) && info.State == MEM_RESERVE, "pages are reserved, not committed");

    // One page is written first (this faults, and is not counted); the probe must keep its bytes.
    base[3 * page + 8] = 0x5A;
    require(VirtualQuery(base + 9 * page, &info, sizeof(info)) && info.State == MEM_RESERVE, "a write committed only its own page");

    PVOID handler = AddVectoredExceptionHandler(1, count_exceptions);
    require(handler != nullptr, "install the exception counter");
    exceptions = 0;
    const std::size_t readable = x2w_local::probe_readable(reinterpret_cast<std::uintptr_t>(base), pages * page);
    const LONG raised = exceptions;
    RemoveVectoredExceptionHandler(handler);

    std::printf("probed %zu bytes of %zu, %ld first-chance exceptions\n", readable, static_cast<std::size_t>(pages * page), static_cast<long>(raised));
    require(readable == pages * page, "the whole 1 MiB range is readable");
    require(raised == 0, "no access violation while probing lazily committed memory");
    require(base[3 * page + 8] == 0x5A, "the byte written before the probe is intact");
    require(base[3 * page + 9] == 0 && base[12 * page] == 0 && base[pages * page - 1] == 0, "the other bytes are zero");
    require(VirtualQuery(base + 15 * page, &info, sizeof(info)) && info.State == MEM_COMMIT, "the probe committed the last page");
    std::puts("probe_readable_lazy: ok");
    return 0;
}
