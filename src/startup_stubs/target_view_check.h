// Device-independent checks for UAV, RTV and DSV descriptions.
#pragma once
#include "startup_stubs/view_desc_check.h"
#include "startup_stubs/view_dimension.h"
#include <type_traits>

namespace x2w_startup::target_view {
enum class Refusal { none, null_desc, dimension, mip, array, format, counter, plane };
inline const char* reason(Refusal r) {
    switch (r) {
    case Refusal::null_desc: return "null resource and description";
    case Refusal::dimension: return "ViewDimension/SampleDesc.Count";
    case Refusal::mip: return "MipSlice";
    case Refusal::array: return "FirstArraySlice/ArraySize or FirstWSlice/WSize";
    case Refusal::format: return "Format";
    case Refusal::counter: return "counter resource/offset";
    case Refusal::plane: return "PlaneSlice";
    default: return "none";
    }
}
struct Parts { UINT* mip = nullptr; UINT* first = nullptr; UINT* count = nullptr; UINT* plane = nullptr; bool volume = false; bool ms = false; bool buffer = false; };
inline bool parts(D3D12_UNORDERED_ACCESS_VIEW_DESC& v, Parts& p) {
    switch (v.ViewDimension) {
    case D3D12_UAV_DIMENSION_BUFFER: p.buffer = true; return true;
    case D3D12_UAV_DIMENSION_TEXTURE1D: p.mip = &v.Texture1D.MipSlice; return true;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY: p.mip=&v.Texture1DArray.MipSlice; p.first=&v.Texture1DArray.FirstArraySlice; p.count=&v.Texture1DArray.ArraySize; return true;
    case D3D12_UAV_DIMENSION_TEXTURE2D: p.mip=&v.Texture2D.MipSlice; p.plane=&v.Texture2D.PlaneSlice; return true;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY: p.mip=&v.Texture2DArray.MipSlice; p.first=&v.Texture2DArray.FirstArraySlice; p.count=&v.Texture2DArray.ArraySize; p.plane=&v.Texture2DArray.PlaneSlice; return true;
    case D3D12_UAV_DIMENSION_TEXTURE3D: p.mip=&v.Texture3D.MipSlice; p.first=&v.Texture3D.FirstWSlice; p.count=&v.Texture3D.WSize; p.volume=true; return true;
    default: return false;
    }
}
inline bool parts(D3D12_RENDER_TARGET_VIEW_DESC& v, Parts& p) {
    switch (v.ViewDimension) {
    case D3D12_RTV_DIMENSION_BUFFER: p.buffer=true; return true;
    case D3D12_RTV_DIMENSION_TEXTURE1D: p.mip=&v.Texture1D.MipSlice; return true;
    case D3D12_RTV_DIMENSION_TEXTURE1DARRAY: p.mip=&v.Texture1DArray.MipSlice; p.first=&v.Texture1DArray.FirstArraySlice; p.count=&v.Texture1DArray.ArraySize; return true;
    case D3D12_RTV_DIMENSION_TEXTURE2D: p.mip=&v.Texture2D.MipSlice; p.plane=&v.Texture2D.PlaneSlice; return true;
    case D3D12_RTV_DIMENSION_TEXTURE2DARRAY: p.mip=&v.Texture2DArray.MipSlice; p.first=&v.Texture2DArray.FirstArraySlice; p.count=&v.Texture2DArray.ArraySize; p.plane=&v.Texture2DArray.PlaneSlice; return true;
    case D3D12_RTV_DIMENSION_TEXTURE2DMS: p.ms=true; return true;
    case D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY: p.ms=true; p.first=&v.Texture2DMSArray.FirstArraySlice; p.count=&v.Texture2DMSArray.ArraySize; return true;
    case D3D12_RTV_DIMENSION_TEXTURE3D: p.mip=&v.Texture3D.MipSlice; p.first=&v.Texture3D.FirstWSlice; p.count=&v.Texture3D.WSize; p.volume=true; return true;
    default: return false;
    }
}
inline bool parts(D3D12_DEPTH_STENCIL_VIEW_DESC& v, Parts& p) {
    switch (v.ViewDimension) {
    case D3D12_DSV_DIMENSION_TEXTURE1D: p.mip=&v.Texture1D.MipSlice; return true;
    case D3D12_DSV_DIMENSION_TEXTURE1DARRAY: p.mip=&v.Texture1DArray.MipSlice; p.first=&v.Texture1DArray.FirstArraySlice; p.count=&v.Texture1DArray.ArraySize; return true;
    case D3D12_DSV_DIMENSION_TEXTURE2D: p.mip=&v.Texture2D.MipSlice; return true;
    case D3D12_DSV_DIMENSION_TEXTURE2DARRAY: p.mip=&v.Texture2DArray.MipSlice; p.first=&v.Texture2DArray.FirstArraySlice; p.count=&v.Texture2DArray.ArraySize; return true;
    case D3D12_DSV_DIMENSION_TEXTURE2DMS: p.ms=true; return true;
    case D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY: p.ms=true; p.first=&v.Texture2DMSArray.FirstArraySlice; p.count=&v.Texture2DMSArray.ArraySize; return true;
    default: return false;
    }
}
// D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_DEPTH_STENCIL_VIEW_DESC.
template<class Desc, class Repair>
Refusal check(Desc* v, const D3D12_RESOURCE_DESC* r, Repair repair, bool typed_cast = false, unsigned dropped = 0) {
    constexpr bool dsv = std::is_same_v<Desc, D3D12_DEPTH_STENCIL_VIEW_DESC>;
    constexpr bool uav = std::is_same_v<Desc, D3D12_UNORDERED_ACCESS_VIEW_DESC>;
    // ID3D12Device::CreateRenderTargetView / CreateDepthStencilView: a default view needs a typed format.
    if (!v) {
        if (!r) return Refusal::null_desc;
        if (r->Format == DXGI_FORMAT_UNKNOWN || is_typeless(r->Format) ||
            (dsv ? !view_desc::depth_format(r->Format) : view_desc::depth_format(r->Format))) return Refusal::format;
        if constexpr (uav) if (r->SampleDesc.Count > 1) return Refusal::dimension;
        return Refusal::none;
    }
    Parts p;
    if (!parts(*v, p)) return Refusal::dimension;
    if (r && !x2w_view_dimension::fits(uav ? 19 : dsv ? 21 : 20, static_cast<unsigned>(v->ViewDimension), static_cast<unsigned>(r->Dimension))) return Refusal::dimension;
    if (r && ((uav && r->SampleDesc.Count > 1) || (!uav && p.ms != (r->SampleDesc.Count > 1)))) return Refusal::dimension;
    auto allowed = [&](DXGI_FORMAT f) { return dsv ? view_desc::depth_format(f) : f != DXGI_FORMAT_UNKNOWN && !is_typeless(f) && !view_desc::depth_format(f); };
    // A DSV format that is not a depth format (UNKNOWN included) takes the resource's typed depth format below, else
    // the view is refused; a typeless depth family member (R32_TYPELESS) is cast by format_cast_fits.
    bool special_buffer = false;
    if constexpr (uav) special_buffer = p.buffer && (v->Buffer.StructureByteStride || v->Buffer.Flags == D3D12_BUFFER_UAV_FLAG_RAW);
    if ((!allowed(v->Format) && !special_buffer) || (r && !p.buffer && !view_desc::format_cast_fits(r->Format, v->Format, typed_cast))) {
        if (!r || !allowed(r->Format)) return Refusal::format;
        repair("Format", static_cast<UINT64>(v->Format), static_cast<UINT64>(r->Format)); v->Format = r->Format;
    }
    if constexpr (dsv) {
        if ((v->Flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL) && v->Format != DXGI_FORMAT_D24_UNORM_S8_UINT && v->Format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
            auto old = v->Flags; v->Flags = static_cast<D3D12_DSV_FLAGS>(v->Flags & ~D3D12_DSV_FLAG_READ_ONLY_STENCIL); repair("Flags", old, v->Flags);
        }
    }
    if (!r) return Refusal::none;
    if (p.mip && dropped) *p.mip = *p.mip > dropped ? *p.mip - dropped : 0;
    if (p.mip && *p.mip >= r->MipLevels) return Refusal::mip;
    if (p.count) {
        UINT limit = r->DepthOrArraySize;
        if (p.volume) limit = (std::max)(1u, unsigned(r->DepthOrArraySize) >> (std::min)(*p.mip, 31u));
        if (*p.first >= limit || !*p.count) return Refusal::array;
        const UINT remaining = limit - *p.first;
        if (*p.count > remaining && !(p.volume && *p.count == ~0u)) { repair(p.volume ? "WSize" : "ArraySize", *p.count, remaining); *p.count = remaining; }
    }
    if (p.plane && *p.plane) {
        if (r->Format == DXGI_FORMAT_NV12 || r->Format == DXGI_FORMAT_P010 || r->Format == DXGI_FORMAT_P016 || is_planar_depth_stencil(r->Format)) {
            if (*p.plane > 1) return Refusal::plane;
        } else {
            repair("PlaneSlice", *p.plane, 0); *p.plane = 0;
        }
    }
    return Refusal::none;
}
// ID3D12Device::CreateUnorderedAccessView: counter placement and structured-buffer requirements.
template<class Repair>
Refusal check_uav(D3D12_UNORDERED_ACCESS_VIEW_DESC* v, const D3D12_RESOURCE_DESC* r, const D3D12_RESOURCE_DESC* counter, Repair repair, bool typed_cast = false, unsigned dropped = 0) {
    if (counter) {
        if (!r || !v || r->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || counter->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || v->ViewDimension != D3D12_UAV_DIMENSION_BUFFER || !v->Buffer.StructureByteStride || v->Format != DXGI_FORMAT_UNKNOWN || (v->Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) || v->Buffer.CounterOffsetInBytes % D3D12_UAV_COUNTER_PLACEMENT_ALIGNMENT || counter->Width < 4 || v->Buffer.CounterOffsetInBytes > counter->Width - 4) return Refusal::counter;
    } else if (v && v->ViewDimension == D3D12_UAV_DIMENSION_BUFFER && v->Buffer.CounterOffsetInBytes) {
        repair("CounterOffsetInBytes", v->Buffer.CounterOffsetInBytes, 0); v->Buffer.CounterOffsetInBytes = 0;
    }
    return check(v, r, repair, typed_cast, dropped);
}
}
