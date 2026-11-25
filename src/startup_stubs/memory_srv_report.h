// Experimental (docs/startup-runtime.md): one compact diagnostic line for a refused device slot 70
// call (a shader resource view whose resource is an address).
//
// The line names the first check that refused the call and the fields the checks read, for
// local diagnosis: the resource description, the view description, the destination handle
// and whether it lies in a known CBV/SRV/UAV heap, and the address and whether it is readable. It is
// formatted into a caller-supplied fixed buffer and allocates nothing. A null resource or view pointer
// means the guest memory could not be read; its fields are then shown as "unreadable".
#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace x2w_startup {

// D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE; named here because older headers lack it.
constexpr int srv_dimension_acceleration_structure = 11;

inline int report_resource(char* out, std::size_t size, const D3D12_RESOURCE_DESC* r)
{
    if (!r) return std::snprintf(out, size, "unreadable");
    return std::snprintf(out, size, "dim=%d fmt=%u size=%llux%ux%u mips=%u layout=0x%x flags=0x%x align=%llu samples=%u/%u",
                         static_cast<int>(r->Dimension), static_cast<unsigned>(r->Format),
                         static_cast<unsigned long long>(r->Width), static_cast<unsigned>(r->Height),
                         static_cast<unsigned>(r->DepthOrArraySize), static_cast<unsigned>(r->MipLevels),
                         static_cast<unsigned>(r->Layout), static_cast<unsigned>(r->Flags),
                         static_cast<unsigned long long>(r->Alignment), static_cast<unsigned>(r->SampleDesc.Count),
                         static_cast<unsigned>(r->SampleDesc.Quality));
}

inline int report_view(char* out, std::size_t size, const D3D12_SHADER_RESOURCE_VIEW_DESC* v)
{
    if (!v) return std::snprintf(out, size, "unreadable");
    const int dimension = static_cast<int>(v->ViewDimension);
    const int head = std::snprintf(out, size, "dim=%d fmt=%u map=0x%x ", dimension, static_cast<unsigned>(v->Format),
                                   static_cast<unsigned>(v->Shader4ComponentMapping));
    if (head < 0 || static_cast<std::size_t>(head) >= size) return head;
    char* tail = out + head;
    const std::size_t room = size - static_cast<std::size_t>(head);
    switch (v->ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE2D:
        return std::snprintf(tail, room, "mostmip=%u mips=%u plane=%u clamp=%g", v->Texture2D.MostDetailedMip,
                             v->Texture2D.MipLevels, v->Texture2D.PlaneSlice,
                             static_cast<double>(v->Texture2D.ResourceMinLODClamp));
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        return std::snprintf(tail, room, "mostmip=%u mips=%u first=%u count=%u plane=%u clamp=%g",
                             v->Texture2DArray.MostDetailedMip, v->Texture2DArray.MipLevels,
                             v->Texture2DArray.FirstArraySlice, v->Texture2DArray.ArraySize,
                             v->Texture2DArray.PlaneSlice, static_cast<double>(v->Texture2DArray.ResourceMinLODClamp));
    case D3D12_SRV_DIMENSION_TEXTURE3D:
        return std::snprintf(tail, room, "mostmip=%u mips=%u clamp=%g", v->Texture3D.MostDetailedMip,
                             v->Texture3D.MipLevels, static_cast<double>(v->Texture3D.ResourceMinLODClamp));
    case D3D12_SRV_DIMENSION_BUFFER:
        return std::snprintf(tail, room, "first=%llu count=%u stride=%u flags=0x%x (buffer views are not supported)",
                             static_cast<unsigned long long>(v->Buffer.FirstElement), v->Buffer.NumElements,
                             v->Buffer.StructureByteStride, static_cast<unsigned>(v->Buffer.Flags));
    default: {
        // Any other dimension: the six words after the component mapping, as stored.
        std::uint32_t words[6]{};
        std::memcpy(words, &v->Buffer, sizeof(words));
        return std::snprintf(tail, room, "raw=%08x,%08x,%08x,%08x,%08x,%08x%s", words[0], words[1], words[2], words[3],
                             words[4], words[5],
                             dimension == srv_dimension_acceleration_structure ? " (acceleration structure view)" : "");
    }
    }
}

// `what` is the fixed phrase of the call site, `reason` the first refusing check.
inline int report_refused_memory_srv(char* out, std::size_t size, const char* what, const char* reason,
                                     const D3D12_RESOURCE_DESC* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* view,
                                     std::uint64_t address, bool address_readable, std::uint64_t destination,
                                     bool destination_in_view_heap)
{
    char resource_text[224];
    char view_text[256];
    report_resource(resource_text, sizeof(resource_text), resource);
    report_view(view_text, sizeof(view_text), view);
    return std::snprintf(out, size,
        "x2w-d3d12 Device::slot70 %s: reason=%s res{%s} view{%s} dest=0x%llx dest_in_view_heap=%d address=0x%llx address_readable=%d",
        what, reason, resource_text, view_text, static_cast<unsigned long long>(destination), destination_in_view_heap ? 1 : 0,
        static_cast<unsigned long long>(address), address_readable ? 1 : 0);
}

}  // namespace x2w_startup
