// Experimental (docs/startup-runtime.md): which PE imports the gaming-runtime proxy considers.
//
// The GameInput-hosted proxy used to patch only the main EXE's GetProcAddress IAT. A title that
// resolves QueryApiImpl through a package helper's GetProcAddress (or that imports QueryApiImpl from
// xgameruntime.dll by name) never hits that hook, so FakeUser / X2W_TRACE_API see no providers.
// This header lists the import shapes to notice — regular and delay-import tables. The stand-in
// rewrites only regular IAT slots on the EXE and package-directory helpers: a delay IAT holds a
// delay-load thunk until first use, so treating it as GetProcAddress or QueryApiImpl is wrong.
// Ordinal imports are ignored. Pure walk helpers — no Windows calls — so a source-built control can
// check them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace x2w_runtime_patch {

enum class Kind : unsigned char { get_proc_address = 1, query_api_impl = 2 };

struct Hit {
    Kind kind;
    std::uint32_t first_thunk_rva;  // RVA of the IMAGE_THUNK_DATA64 / delay-IAT slot
    const char* dll_name;           // pointed into the image
    const char* import_name;
    bool delay_import;              // true = delay-load table (do not treat *slot as the real export)
};

inline bool dll_is_xgameruntime(const char* name) noexcept
{
    if (!name) return false;
    static constexpr char expect[] = "xgameruntime.dll";
    for (std::size_t i = 0; expect[i]; ++i) {
        const unsigned char a = static_cast<unsigned char>(name[i]);
        const unsigned char b = static_cast<unsigned char>(expect[i]);
        const unsigned char al = (a >= 'A' && a <= 'Z') ? static_cast<unsigned char>(a - 'A' + 'a') : a;
        if (!al || al != b) return false;
    }
    return name[sizeof(expect) - 1] == 0;
}

inline bool import_is_get_proc_address(const char* name) noexcept
{
    return name && std::strcmp(name, "GetProcAddress") == 0;
}

inline bool import_is_query_api_impl(const char* name) noexcept
{
    return name && std::strcmp(name, "QueryApiImpl") == 0;
}

// Decide whether one named import of `dll_name` should be noticed. Ordinal imports are ignored.
inline bool classify(const char* dll_name, const char* import_name, Kind* kind) noexcept
{
    if (!import_name || !kind) return false;
    if (import_is_get_proc_address(import_name)) {
        *kind = Kind::get_proc_address;
        return true;
    }
    if (dll_is_xgameruntime(dll_name) && import_is_query_api_impl(import_name)) {
        *kind = Kind::query_api_impl;
        return true;
    }
    return false;
}

#pragma pack(push, 1)
struct DosStub {
    std::uint16_t e_magic;
    std::uint16_t pad[29];
    std::int32_t e_lfanew;
};
struct DataDir {
    std::uint32_t virtual_address;
    std::uint32_t size;
};
struct ImportDesc {
    std::uint32_t original_first_thunk;
    std::uint32_t time_date_stamp;
    std::uint32_t forwarder_chain;
    std::uint32_t name;
    std::uint32_t first_thunk;
};
// IMAGE_DELAYLOAD_DESCRIPTOR (PE32+).
struct DelayDesc {
    std::uint32_t attributes;
    std::uint32_t dll_name;
    std::uint32_t module_handle;
    std::uint32_t first_thunk;           // delay IAT
    std::uint32_t original_first_thunk;  // name table
    std::uint32_t bound_iat;
    std::uint32_t unload_iat;
    std::uint32_t time_date_stamp;
};
struct Thunk64 {
    std::uint64_t value;
};
#pragma pack(pop)

inline bool readable_range(const unsigned char* base, std::size_t image_size, std::uint64_t rva, std::size_t bytes) noexcept
{
    return base && bytes && rva <= image_size && bytes <= image_size - static_cast<std::size_t>(rva);
}

inline bool c_string_in_image(const unsigned char* base, std::size_t image_size, std::uint32_t rva) noexcept
{
    if (!readable_range(base, image_size, rva, 1)) return false;
    const char* s = reinterpret_cast<const char*>(base + rva);
    const std::size_t max = image_size - rva;
    for (std::size_t i = 0; i < max; ++i) {
        if (s[i] == 0) return true;
    }
    return false;
}

template <class Fn>
int visit_named_thunks(const unsigned char* base, std::size_t image_size, const char* dll, std::uint32_t name_table_rva,
                       std::uint32_t iat_rva, bool delay_import, Fn on_hit)
{
    int hits = 0;
    for (std::uint32_t ti = 0;; ++ti) {
        const auto oft_rva = name_table_rva + ti * static_cast<std::uint32_t>(sizeof(Thunk64));
        const auto ft_rva = iat_rva + ti * static_cast<std::uint32_t>(sizeof(Thunk64));
        if (!readable_range(base, image_size, oft_rva, sizeof(Thunk64))) break;
        if (!readable_range(base, image_size, ft_rva, sizeof(Thunk64))) break;
        const auto oft = *reinterpret_cast<const Thunk64*>(base + oft_rva);
        if (oft.value == 0) break;
        if (oft.value & 0x8000000000000000ull) continue;  // ordinal: low bits are not a name RVA
        const auto hint_rva = static_cast<std::uint32_t>(oft.value & 0xffffffffu);
        if (!readable_range(base, image_size, hint_rva, 3)) continue;
        if (!c_string_in_image(base, image_size, hint_rva + 2)) continue;
        const char* import_name = reinterpret_cast<const char*>(base + hint_rva + 2);
        Kind kind{};
        if (!classify(dll, import_name, &kind)) continue;
        Hit hit{kind, ft_rva, dll, import_name, delay_import};
        on_hit(hit);
        ++hits;
    }
    return hits;
}

// Walk a PE image's import and delay-import tables. Invokes `on_hit(Hit)` for each classified import.
// Returns the number of hits. Malformed images stop the walk and return what was found.
template <class Fn>
int visit_imports(const unsigned char* base, std::size_t image_size, Fn on_hit)
{
    if (!readable_range(base, image_size, 0, sizeof(DosStub))) return 0;
    const auto* dos = reinterpret_cast<const DosStub*>(base);
    if (dos->e_magic != 0x5A4D) return 0;
    if (dos->e_lfanew < 0) return 0;
    const auto nt_rva = static_cast<std::uint32_t>(dos->e_lfanew);
    // COFF (20) + optional magic through NumberOfRvaAndSizes, then 16 data directories.
    if (!readable_range(base, image_size, nt_rva, 24 + 112 + 16 * sizeof(DataDir))) return 0;
    const auto* nt = base + nt_rva;
    if (*reinterpret_cast<const std::uint32_t*>(nt) != 0x00004550) return 0;
    const auto magic = *reinterpret_cast<const std::uint16_t*>(nt + 24);
    if (magic != 0x20B) return 0;  // PE32+ only
    const auto* dirs = reinterpret_cast<const DataDir*>(nt + 24 + 112);
    int hits = 0;

    const auto& imports_dir = dirs[1];
    if (imports_dir.virtual_address && imports_dir.size) {
        for (std::uint32_t at = imports_dir.virtual_address;; at += sizeof(ImportDesc)) {
            if (!readable_range(base, image_size, at, sizeof(ImportDesc))) break;
            const auto* imp = reinterpret_cast<const ImportDesc*>(base + at);
            if (!imp->name && !imp->original_first_thunk && !imp->first_thunk) break;
            if (!imp->name || !imp->original_first_thunk || !imp->first_thunk) continue;
            if (!c_string_in_image(base, image_size, imp->name)) continue;
            const char* dll = reinterpret_cast<const char*>(base + imp->name);
            hits += visit_named_thunks(base, image_size, dll, imp->original_first_thunk, imp->first_thunk, false, on_hit);
        }
    }

    // DataDirectory[13] = delay imports (reported with delay_import=true; callers must not rewrite the slot as a real export).
    const auto& delay_dir = dirs[13];
    if (delay_dir.virtual_address && delay_dir.size) {
        for (std::uint32_t at = delay_dir.virtual_address;; at += sizeof(DelayDesc)) {
            if (!readable_range(base, image_size, at, sizeof(DelayDesc))) break;
            const auto* delay = reinterpret_cast<const DelayDesc*>(base + at);
            if (!delay->dll_name && !delay->original_first_thunk && !delay->first_thunk) break;
            if (!delay->dll_name || !delay->original_first_thunk || !delay->first_thunk) continue;
            if (!c_string_in_image(base, image_size, delay->dll_name)) continue;
            const char* dll = reinterpret_cast<const char*>(base + delay->dll_name);
            hits += visit_named_thunks(base, image_size, dll, delay->original_first_thunk, delay->first_thunk, true, on_hit);
        }
    }
    return hits;
}

}  // namespace x2w_runtime_patch
