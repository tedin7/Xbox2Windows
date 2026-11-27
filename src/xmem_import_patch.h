// Points the main module's imports of one function name at a replacement. Shared by the xmem.dll wrappers
// (lazy commit in xmem_mapped_pages.cpp, file diagnostics in xmem_file_diag.cpp).
#pragma once

#include <windows.h>

#include <cstring>

namespace x2w_xmem {

// Returns how many import slots were changed. The module name does not matter: the title imports file calls
// both directly and through API-set names.
inline int patch_main_import(const char* name, const void* replacement) noexcept
{
    auto* base = reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
    if (!base) return 0;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    int patched = 0;
    for (auto* imp = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); dir.Size && imp->Name; ++imp) {
        const auto* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            const auto* by_name = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(by_name->Name), name) != 0) continue;
            DWORD old = 0;
            if (!VirtualProtect(&slots->u1.Function, sizeof(slots->u1.Function), PAGE_READWRITE, &old)) continue;
            slots->u1.Function = reinterpret_cast<ULONGLONG>(replacement);
            VirtualProtect(&slots->u1.Function, sizeof(slots->u1.Function), old, &old);
            ++patched;
        }
    }
    return patched;
}

}  // namespace x2w_xmem
