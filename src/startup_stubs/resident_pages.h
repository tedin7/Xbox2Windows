#pragma once
// Shared by XCRDAPI's destination check (src/startup_stubs/xcrdapi.cpp) and xmem's copy into a destination
// (x2w_xmem_copy_to_destination, src/xmem_mapped_pages.cpp).
#include <windows.h>
#include <psapi.h>

#include <cstdint>

namespace x2w_local {
// True when every page of [start, end) is resident and plain read-write: valid in the working set, PAGE_READWRITE
// or PAGE_EXECUTE_READWRITE, no guard bit. One QueryWorkingSetEx per 512 pages, so the cost follows the
// destination's size, not the region's. Reads attributes only; no page is touched.
inline bool resident_read_write(std::uintptr_t start, std::uintptr_t end) noexcept
{
    PSAPI_WORKING_SET_EX_INFORMATION pages[512];
    std::uintptr_t page = start & ~std::uintptr_t{0xFFF};
    while (page < end) {
        DWORD count = 0;
        for (; count < 512 && page < end; ++count, page += 0x1000) pages[count].VirtualAddress = reinterpret_cast<void*>(page);
        if (!QueryWorkingSetEx(GetCurrentProcess(), pages, count * sizeof(pages[0]))) return false;
        for (DWORD i = 0; i < count; ++i) {
            const auto& page_info = pages[i].VirtualAttributes;
            const DWORD protect = static_cast<DWORD>(page_info.Win32Protection);
            if (!page_info.Valid || page_info.Bad || (protect & PAGE_GUARD) ||
                ((protect & 0xFF) != PAGE_READWRITE && (protect & 0xFF) != PAGE_EXECUTE_READWRITE))
                return false;
        }
    }
    return true;
}
}  // namespace x2w_local
