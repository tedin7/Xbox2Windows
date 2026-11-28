// The desktop SRV boundary, shared by the device shim and plane compositor.
#pragma once

#include "startup_stubs/view_desc_check.h"
#include "startup_stubs/console_format_log.h"

#include <windows.h>

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace x2w_startup {

using SrvLog = FormatLog;

struct SrvCreation {
    const char* site = nullptr;
    bool described = false;
    unsigned char desc[sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)]{};
    D3D12_RESOURCE_DESC resource{}; // Original backing, including when validation substitutes a null view.
};

inline thread_local SrvCreation last_srv;
using SrvHistory = std::array<SrvCreation, 16>;
inline SrvHistory srv_history{};
inline std::size_t srv_history_next = 0;
inline SRWLOCK srv_history_lock = SRWLOCK_INIT;

inline SrvHistory recent_srvs()
{
    SrvHistory snapshot;
    AcquireSRWLockShared(&srv_history_lock);
    for (std::size_t i = 0; i < snapshot.size(); ++i)
        snapshot[i] = srv_history[(srv_history_next + i) % srv_history.size()];
    ReleaseSRWLockShared(&srv_history_lock);
    return snapshot;
}

inline void srv_diagnostic(const char* site, const char* action, const char* field, UINT from, UINT to, SrvLog log)
{
    static LONG lines;
    if (InterlockedIncrement(&lines) > 64) return;
    char line[256];
    std::snprintf(line, sizeof(line), "x2w-d3d12 SRV %s: site=%s field/reason=%s %u -> %u",
                  action, site, field, from, to);
    if (log) log("%s", line);
    else {
        OutputDebugStringA(line);
        OutputDebugStringA("\n");
    }
}

inline D3D12_SHADER_RESOURCE_VIEW_DESC safe_null_srv(D3D12_SRV_DIMENSION dimension, DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = dimension;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    switch (dimension) {
    case D3D12_SRV_DIMENSION_BUFFER:
        view.Buffer.NumElements = 1;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE1D: view.Texture1D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: view.Texture1DArray = {0, 1, 0, 1, 0}; break;
    case D3D12_SRV_DIMENSION_TEXTURE2D: view.Texture2D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: view.Texture2DArray = {0, 1, 0, 1, 0, 0}; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS: break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY: view.Texture2DMSArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: view.Texture3D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE: view.TextureCube.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: view.TextureCubeArray = {0, 1, 0, 1, 0}; break;
    case D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE: view.Format = DXGI_FORMAT_UNKNOWN; break;
    default:
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        break;
    }
    return view;
}

inline bool srv_format_supported(ID3D12Device* device, DXGI_FORMAT format, D3D12_SRV_DIMENSION dimension)
{
    if (format == DXGI_FORMAT_UNKNOWN || is_typeless(format) || view_desc::depth_format(format) ||
        format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010 || format == DXGI_FORMAT_P016 ||
        format == DXGI_FORMAT_420_OPAQUE)
        return false;
    D3D12_FORMAT_SUPPORT1 required;
    switch (dimension) {
    case D3D12_SRV_DIMENSION_BUFFER: required = D3D12_FORMAT_SUPPORT1_BUFFER; break;
    case D3D12_SRV_DIMENSION_TEXTURE1D:
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: required = D3D12_FORMAT_SUPPORT1_TEXTURE1D; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: required = D3D12_FORMAT_SUPPORT1_TEXTURE3D; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: required = D3D12_FORMAT_SUPPORT1_TEXTURECUBE; break;
    default: required = D3D12_FORMAT_SUPPORT1_TEXTURE2D; break;
    }
    D3D12_FEATURE_DATA_FORMAT_SUPPORT caps{format};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &caps, sizeof(caps))) &&
           (caps.Support1 & required) &&
           (caps.Support1 & (D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE));
}

inline view_desc::Refusal validate_srv(ID3D12Device* device, D3D12_SHADER_RESOURCE_VIEW_DESC* view,
                                      const D3D12_RESOURCE_DESC& resource, unsigned dropped_levels,
                                      const char* site, SrvLog log)
{
    using view_desc::Refusal;
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
    if (view && view->ViewDimension != D3D12_SRV_DIMENSION_BUFFER && view->Format != resource.Format &&
        !is_typeless(resource.Format) && compatible_formats(resource.Format, view->Format))
        device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options, sizeof(options));
    const auto refusal = view ? view_desc::check(*view, resource, dropped_levels,
        [&](view_desc::Field field, UINT from, UINT to) {
            srv_diagnostic(site, "repaired", view_desc::field_name(field), from, to, log);
        }, options.CastingFullyTypedFormatSupported != FALSE) : view_desc::check_default(resource);
    if (refusal != Refusal::none) return refusal;
    const auto dimension = view ? view->ViewDimension : view_desc::default_dimension(resource);
    const auto format = view ? view->Format : resource.Format;
    if (dimension == D3D12_SRV_DIMENSION_BUFFER && view &&
        (view->Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW || view->Buffer.StructureByteStride))
        return Refusal::none;
    return srv_format_supported(device, format, dimension) ? Refusal::none : Refusal::format;
}

inline void srv_census_debug_log(const char* format, ...)
{
    char line[256];
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    OutputDebugStringA(line);
    OutputDebugStringA("\n");
}

inline INT64 srv_plane_slice(const D3D12_SHADER_RESOURCE_VIEW_DESC* view)
{
    if (view && view->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D)
        return static_cast<INT64>(view->Texture2D.PlaneSlice);
    if (view && view->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY)
        return static_cast<INT64>(view->Texture2DArray.PlaneSlice);
    return -1;
}

inline void log_srv_census(const D3D12_RESOURCE_DESC& backing, DXGI_FORMAT format,
                           INT64 requested_plane, INT64 used_plane, SrvLog log)
{
    using Key = std::tuple<decltype(resource_size(backing)), DXGI_FORMAT, DXGI_FORMAT, INT64, INT64>;
    struct Entry {
        Key key{};
        UINT64 count = 0;
    };
    static std::array<Entry, 64> entries{};
    static std::size_t used = 0;
    static UINT64 overflow = 0;
    static SRWLOCK lock = SRWLOCK_INIT;

    const Key key{resource_size(backing), backing.Format, format, requested_plane, used_plane};
    AcquireSRWLockExclusive(&lock);
    std::size_t index = 0;
    while (index < used && entries[index].key != key) ++index;
    if (index == used && used < entries.size()) entries[used++].key = key;
    const bool full = index == entries.size();
    const UINT64 count = full ? ++overflow : ++entries[index].count;
    ReleaseSRWLockExclusive(&lock);

    if (!log_at_power_of_two(count)) return;
    if (full) {
        log("x2w-d3d12 SRV census overflow count=%llu", static_cast<unsigned long long>(count));
        return;
    }
    log("x2w-d3d12 SRV census size=%llux%ux%u resource_format=%u view_format=%u "
        "plane_requested=%lld plane_used=%lld count=%llu",
        static_cast<unsigned long long>(backing.Width), backing.Height, static_cast<UINT>(backing.DepthOrArraySize),
        static_cast<UINT>(backing.Format), static_cast<UINT>(format),
        static_cast<long long>(requested_plane), static_cast<long long>(used_plane),
        static_cast<unsigned long long>(count));
}

// True if the requested view was written (possibly repaired). A refusal always overwrites the destination.
inline bool x2w_create_srv(ID3D12Device* device, ID3D12Resource* resource,
                           const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE dest,
                           const char* site, SrvLog log = nullptr, unsigned dropped_levels = 0)
{
    using view_desc::Refusal;
    const D3D12_RESOURCE_DESC backing = resource ? resource->GetDesc() : D3D12_RESOURCE_DESC{};
    D3D12_SHADER_RESOURCE_VIEW_DESC checked{};
    if (desc) std::memcpy(&checked, desc, sizeof(checked));
    auto dimension = desc ? checked.ViewDimension : view_desc::default_dimension(backing);
    auto refusal = Refusal::none;
    if (resource) {
        refusal = validate_srv(device, desc ? &checked : nullptr, backing, dropped_levels, site, log);
    } else if (!desc || dimension == D3D12_SRV_DIMENSION_UNKNOWN) {
        refusal = Refusal::dimension;
    } else if (dimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) {
        if (checked.Format != DXGI_FORMAT_UNKNOWN ||
            checked.RaytracingAccelerationStructure.Location % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT)
            refusal = Refusal::format;
        if (!view_desc::valid_mapping(checked.Shader4ComponentMapping)) {
            srv_diagnostic(site, "repaired", view_desc::field_name(view_desc::Field::mapping),
                           checked.Shader4ComponentMapping, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, log);
            checked.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }
    } else if (dimension == D3D12_SRV_DIMENSION_BUFFER) {
        D3D12_RESOURCE_DESC bounds{};
        bounds.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bounds.Width = ~UINT64{0};
        refusal = validate_srv(device, &checked, bounds, 0, site, log);
        if (refusal == Refusal::none) {
            checked.Buffer.FirstElement = 0;
            checked.Buffer.NumElements = 1;
        }
    } else if (dimension >= D3D12_SRV_DIMENSION_TEXTURE1D && dimension <= D3D12_SRV_DIMENSION_TEXTURECUBEARRAY) {
        // Null textures have no subresources. Keep the shader-visible type and use canonical coordinates.
        if (!srv_format_supported(device, checked.Format, dimension)) {
            refusal = Refusal::format;
        } else {
            auto canonical = safe_null_srv(dimension, checked.Format);
            if (view_desc::valid_mapping(checked.Shader4ComponentMapping))
                canonical.Shader4ComponentMapping = checked.Shader4ComponentMapping;
            else
                srv_diagnostic(site, "repaired", view_desc::field_name(view_desc::Field::mapping),
                               checked.Shader4ComponentMapping, canonical.Shader4ComponentMapping, log);
            checked = canonical;
        }
    } else {
        refusal = Refusal::dimension;
    }

    if (resource) {
        const auto family = format_family(backing.Format);
        if (family == DXGI_FORMAT_R24G8_TYPELESS || family == DXGI_FORMAT_R32G8X24_TYPELESS ||
            has_console_format_remap(resource)) {
            // A refused view writes a null descriptor, so no resource plane is used.
            const bool accepted = refusal == Refusal::none;
            const auto format = desc ? (accepted ? checked.Format : desc->Format) : backing.Format;
            const INT64 used_plane = accepted ? (desc ? srv_plane_slice(&checked) : 0) : -1;
            log_srv_census(backing, format, srv_plane_slice(desc), used_plane, log ? log : srv_census_debug_log);
        }
    }
    if (refusal != Refusal::none) {
        srv_diagnostic(site, "refused (writing null)", view_desc::refusal_field_name(refusal), 0, 0, log);
        auto format = desc ? desc->Format : backing.Format;
        if (dimension != D3D12_SRV_DIMENSION_BUFFER &&
            !(dimension >= D3D12_SRV_DIMENSION_TEXTURE1D && dimension <= D3D12_SRV_DIMENSION_TEXTURECUBEARRAY) &&
            !(dimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE && !resource)) {
            dimension = view_desc::default_dimension(backing);
        }
        if (dimension == D3D12_SRV_DIMENSION_UNKNOWN) dimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        if (dimension == D3D12_SRV_DIMENSION_BUFFER) {
            // Keep the binding's raw or structured kind when it can be identified.
            checked = safe_null_srv(dimension, format);
            if (desc && desc->ViewDimension == dimension) {
                checked.Buffer.Flags = desc->Buffer.Flags;
                checked.Buffer.StructureByteStride = desc->Buffer.StructureByteStride;
            }
            if (checked.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW) {
                checked.Format = DXGI_FORMAT_R32_TYPELESS;
                checked.Buffer.StructureByteStride = 0;
            } else if (!buffer_srv_element_bytes(checked) ||
                       (!checked.Buffer.StructureByteStride && !srv_format_supported(device, format, dimension))) {
                checked = safe_null_srv(dimension, DXGI_FORMAT_R32_UINT);
            }
        } else {
            if (!srv_format_supported(device, format, dimension))
                format = DXGI_FORMAT_R32_FLOAT;
            checked = safe_null_srv(dimension, format);
        }
        resource = nullptr;
        desc = &checked;
    } else if (desc) {
        desc = &checked;
    }

    last_srv = {};
    last_srv.site = site;
    last_srv.resource = backing;
    last_srv.described = desc != nullptr;
    if (desc) std::memcpy(last_srv.desc, desc, sizeof(last_srv.desc));
    AcquireSRWLockExclusive(&srv_history_lock);
    srv_history[srv_history_next] = last_srv;
    srv_history_next = (srv_history_next + 1) % srv_history.size();
    ReleaseSRWLockExclusive(&srv_history_lock);
    device->CreateShaderResourceView(resource, desc, dest);
    return refusal == Refusal::none;
}

} // namespace x2w_startup
