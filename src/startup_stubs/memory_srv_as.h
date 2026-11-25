// Pure validation and address resolution for acceleration-structure memory SRVs (device slot 70).
// Buffer and texture views are handled in memory_srv_buffer.h and d3d12_memory_srv.inl.
//
// An acceleration-structure view names the structure only by an address (its Location). It is
// published when that address starts inside one live placed buffer that was created in the
// acceleration-structure state; the view then carries the desktop GPU virtual address of the same
// byte. Every refusal names the first check that failed, in the order the checks run.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace x2w_startup {

enum class AsSrvRefusal {
    none,
    as_unaligned,   // the address, or the desktop address it maps to, is not 256-byte aligned
    as_not_placed,  // no single live placed buffer holds the address
    as_wrong_state, // the placed buffer was not created in the acceleration-structure state
};

inline const char* as_srv_refusal_name(AsSrvRefusal refusal)
{
    switch (refusal) {
    case AsSrvRefusal::as_unaligned: return "as_unaligned";
    case AsSrvRefusal::as_not_placed: return "as_not_placed";
    case AsSrvRefusal::as_wrong_state: return "as_wrong_state";
    default: return "none";
    }
}

// What the device knows about the placed buffer that holds an address; `found` is false when no live
// placed buffer does, and `ambiguous` when more than one does or the address crosses an allocation.
struct PlacedAsBuffer {
    bool found{};
    bool ambiguous{};
    std::uint64_t title_base{};  // address of the buffer's first byte as the title sees it
    std::uint64_t gpu_base{};    // desktop GPU virtual address of the same byte
    std::uint64_t bytes{};       // allocation size
    D3D12_RESOURCE_STATES initial_state{};
};

inline bool is_as_srv(const D3D12_SHADER_RESOURCE_VIEW_DESC& view)
{
    return view.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
}

inline bool supports_as_memory_srv(const D3D12_SHADER_RESOURCE_VIEW_DESC& view)
{
    return is_as_srv(view) && view.Format == DXGI_FORMAT_UNKNOWN &&
           view.Shader4ComponentMapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
}

// The title address of the structure: the view's Location, or the call's separate address when the
// Location is zero.
inline std::uint64_t as_srv_location(const D3D12_SHADER_RESOURCE_VIEW_DESC& view, std::uint64_t address)
{
    return view.RaytracingAccelerationStructure.Location ? view.RaytracingAccelerationStructure.Location : address;
}

// `view` must be an acceleration-structure view. On success `resolved` is `view` with Location replaced
// by the desktop GPU virtual address; it is left unchanged on a refusal.
inline AsSrvRefusal resolve_as_srv_desc(std::uint64_t location, const PlacedAsBuffer& placed,
                                        const D3D12_SHADER_RESOURCE_VIEW_DESC& view,
                                        D3D12_SHADER_RESOURCE_VIEW_DESC& resolved)
{
    constexpr std::uint64_t alignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
    if (location % alignment != 0) return AsSrvRefusal::as_unaligned;
    if (!placed.found || placed.ambiguous || !placed.title_base || !placed.gpu_base || location < placed.title_base ||
        location - placed.title_base >= placed.bytes)
        return AsSrvRefusal::as_not_placed;
    if (placed.initial_state != D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)
        return AsSrvRefusal::as_wrong_state;
    const std::uint64_t offset = location - placed.title_base;
    if (offset > UINT64_MAX - placed.gpu_base) return AsSrvRefusal::as_not_placed;
    const std::uint64_t gpu = placed.gpu_base + offset;
    if (gpu % alignment != 0) return AsSrvRefusal::as_unaligned;
    resolved = {};
    resolved.Format = view.Format;
    resolved.ViewDimension = view.ViewDimension;
    resolved.Shader4ComponentMapping = view.Shader4ComponentMapping;
    resolved.RaytracingAccelerationStructure.Location = gpu;
    return AsSrvRefusal::none;
}

}  // namespace x2w_startup
