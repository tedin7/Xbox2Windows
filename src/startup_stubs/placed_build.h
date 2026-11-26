#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>

namespace x2w_startup {
// Why a placed acceleration-structure build was refused. The first part of this header is standard C++ only, so
// the rules below are checked on any platform; the resolver that applies them to a device follows it.
enum class PlacedBuildCause : unsigned {
    ok,
    unaligned,                   // destination or scratch is not 256-byte aligned
    not_placed,                  // no live placed buffer covers the range
    wrong_state,                 // the buffer was not created in the state its role needs
    range,                       // the range overflows or its translated address is unusable
    update_without_source,       // PERFORM_UPDATE with no source structure
    source_without_update,       // a source structure without PERFORM_UPDATE
    source_unaligned,
    source_not_placed,
    source_wrong_state,
    source_range,
    source_dest_partial_overlap, // the source and the destination overlap without being the same structure
    source_not_built,            // no build through the resolver produced a structure at the source address
    source_not_updatable,        // the source was built without ALLOW_UPDATE
    count
};

constexpr const char* cause_name(PlacedBuildCause cause)
{
    switch (cause) {
    case PlacedBuildCause::ok: return "ok";
    case PlacedBuildCause::unaligned: return "unaligned";
    case PlacedBuildCause::not_placed: return "not_placed";
    case PlacedBuildCause::wrong_state: return "wrong_state";
    case PlacedBuildCause::range: return "range";
    case PlacedBuildCause::update_without_source: return "update_without_source";
    case PlacedBuildCause::source_without_update: return "source_without_update";
    case PlacedBuildCause::source_unaligned: return "source_unaligned";
    case PlacedBuildCause::source_not_placed: return "source_not_placed";
    case PlacedBuildCause::source_wrong_state: return "source_wrong_state";
    case PlacedBuildCause::source_range: return "source_range";
    case PlacedBuildCause::source_dest_partial_overlap: return "source_dest_partial_overlap";
    case PlacedBuildCause::source_not_built: return "source_not_built";
    case PlacedBuildCause::source_not_updatable: return "source_not_updatable";
    case PlacedBuildCause::count: break;
    }
    return "unknown";
}

// The cause a lookup of the destination or scratch gives, as it reads for the source structure.
constexpr PlacedBuildCause as_source_cause(PlacedBuildCause cause)
{
    switch (cause) {
    case PlacedBuildCause::unaligned: return PlacedBuildCause::source_unaligned;
    case PlacedBuildCause::not_placed: return PlacedBuildCause::source_not_placed;
    case PlacedBuildCause::wrong_state: return PlacedBuildCause::source_wrong_state;
    case PlacedBuildCause::range: return PlacedBuildCause::source_range;
    default: return cause;
    }
}

constexpr std::uint64_t build_alignment = 256;  // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT

// Whether a placed buffer's translated address can be used: it exists, the offset does not wrap it, and the result keeps the
// 256-byte alignment a structure needs. A real device places buffers 64 KB aligned, so only a broken translation reaches this.
constexpr PlacedBuildCause gpu_range_cause(std::uint64_t gpu, std::uint64_t offset)
{
    if (!gpu || offset > UINT64_MAX - gpu || (gpu + offset) % build_alignment) return PlacedBuildCause::range;
    return PlacedBuildCause::ok;
}

// The last byte of [start, start + bytes), clamped to the top of the address space.
constexpr std::uint64_t last_byte(std::uint64_t start, std::uint64_t bytes)
{
    return bytes - 1 > UINT64_MAX - start ? UINT64_MAX : start + (bytes - 1);
}

// Whether [a, a + a_bytes) and [b, b + b_bytes) share a byte. An empty range shares none.
constexpr bool ranges_overlap(std::uint64_t a, std::uint64_t a_bytes, std::uint64_t b, std::uint64_t b_bytes)
{
    if (!a_bytes || !b_bytes) return false;
    return a <= last_byte(b, b_bytes) && b <= last_byte(a, a_bytes);
}

// The rules that need no device: how PERFORM_UPDATE, the source structure and the destination relate. An update
// may run in place (the source is the destination) or write a different structure, but the two may not partly
// overlap. result_bytes is the size of both structures.
constexpr PlacedBuildCause check_update_shape(bool perform_update, std::uint64_t source, std::uint64_t dest,
                                              std::uint64_t result_bytes)
{
    if (perform_update && !source) return PlacedBuildCause::update_without_source;
    if (!perform_update && source) return PlacedBuildCause::source_without_update;
    if (!source) return PlacedBuildCause::ok;
    if (source % build_alignment) return PlacedBuildCause::source_unaligned;
    if (source != dest && ranges_overlap(source, result_bytes, dest, result_bytes))
        return PlacedBuildCause::source_dest_partial_overlap;
    return PlacedBuildCause::ok;
}

// What the resolver published for each placed build, so an update can be checked against the flags its source was built with.
// An update needs a source built with ALLOW_UPDATE (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE). The shim
// cannot read the flags back from the GPU, so it records them when it resolves the build. A source it never saw built is
// refused too: unknown provenance means refuse. Not thread safe; the resolver holds a lock around it.
class PlacedBuildLedger {
public:
    // A build at [start, start + bytes) replaces whatever was recorded over that memory.
    void record(std::uint64_t start, std::uint64_t bytes, bool allow_update)
    {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (ranges_overlap(it->first, it->second.bytes, start, bytes)) it = entries_.erase(it);
            else ++it;
        }
        entries_[start] = {bytes, allow_update};
    }
    // The source of an update names the start of a recorded structure that was built with ALLOW_UPDATE.
    PlacedBuildCause check_source(std::uint64_t source) const
    {
        const auto found = entries_.find(source);
        if (found == entries_.end()) return PlacedBuildCause::source_not_built;
        return found->second.allow_update ? PlacedBuildCause::ok : PlacedBuildCause::source_not_updatable;
    }
    std::size_t size() const { return entries_.size(); }

private:
    struct Entry {
        std::uint64_t bytes{};
        bool allow_update{};
    };
    std::map<std::uint64_t, Entry> entries_;
};
}  // namespace x2w_startup

#if defined(_WIN32)
#include <d3d12.h>

namespace x2w_startup {
// Takes ownership of one reference to the desktop device chosen by the caller. *wrapped is null
// when the wrapper cannot be allocated; the reference is released then.
void attach_device(ID3D12Device* device, void** wrapped);

// Internal desktop build preparation. The caller retains the registered resources
// through execution and supplies prebuild sizes for these exact BLAS/TLAS inputs.
// With PERFORM_UPDATE, build.SourceAccelerationStructureData names the structure to update, in place or not,
// and the scratch size is sizes.UpdateScratchDataSizeInBytes. A successful build is recorded with its ALLOW_UPDATE flag, and an
// update whose source was not recorded as built with ALLOW_UPDATE is refused. A refusal returns E_INVALIDARG, leaves build
// untouched, logs its PlacedBuildCause once per cause and reports it in *cause when that is not null (ok on success).
HRESULT resolve_placed_build(void* guest_device, UINT64 dest, UINT64 scratch,
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& sizes,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build, PlacedBuildCause* cause = nullptr);
}
#endif
