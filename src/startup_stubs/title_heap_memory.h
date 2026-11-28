// Experimental (docs/startup-runtime.md): when a descriptor heap that the executable created
// without memory of its own gets plain memory of ours as its CPU side.
//
// A view heap always does, so that the view ids written there travel with the executable's own
// memory copies. Any other heap does when the desktop heap's CPU start is not pointer-aligned. A
// desktop handle is opaque, and a translation layer may keep tag bits in its low bits (measured:
// sampler heaps start at ...007 through vkd3d-proton). A console executable treats the start as an
// address; with such a start, the source handle it later passed to CopyDescriptorsSimple for the
// shader-visible sampler heap had 0xfd in its top byte, and the copy faulted in the layer.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace x2w_startup {
constexpr unsigned title_view_stride = 32;

inline unsigned view_heap_title_stride(D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned count,
                                       std::uint64_t backing, std::uint64_t bytes, unsigned desktop_stride)
{
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV && backing >= 0x10000 && count && bytes % count == 0) {
        // A depth stencil heap's title memory has records of the console's 8 bytes (or any power of two to 256): the
        // size says so, and handles map to desktop descriptors by index, as for view heaps.
        const std::uint64_t record = bytes / count;
        if (record >= 8 && record <= 256 && (record & (record - 1)) == 0) return static_cast<unsigned>(record);
    }
    return type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && backing >= 0x10000 &&
           bytes == std::uint64_t{count} * title_view_stride ? title_view_stride : desktop_stride;
}

inline std::uint64_t desktop_descriptor_offset(std::uint64_t offset, unsigned title_stride, unsigned desktop_stride)
{
    return (offset / title_stride) * desktop_stride + offset % title_stride;
}

inline bool heap_needs_title_memory(D3D12_DESCRIPTOR_HEAP_TYPE type, std::uint64_t desktop_cpu_start, bool strides_differ = false)
{
    return strides_differ || type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || (desktop_cpu_start & (alignof(void*) - 1)) != 0;
}
// An opaque start cannot be exposed as CPU memory if the required allocation fails.
template <class Allocate>
bool ensure_title_heap_memory(D3D12_DESCRIPTOR_HEAP_TYPE type, std::uint64_t desktop_cpu_start,
                              std::uint64_t bytes, std::uint64_t& title_memory, Allocate allocate, bool strides_differ = false)
{
    if (title_memory || !heap_needs_title_memory(type, desktop_cpu_start, strides_differ)) return true;
    if (bytes) title_memory = allocate(bytes);
    return title_memory != 0;
}
}  // namespace x2w_startup
