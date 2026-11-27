#pragma once
#include "gfx1013_compute_native.h"
#include "startup_stubs/native_buffer_stride.h"
#include <d3d12.h>

// Preserve the guest view window. The HLSL resource type determines the desktop component type.
namespace x2w {
inline bool native_buffer_store_view(const ComputeViewBinding &binding, D3D12_UNORDERED_ACCESS_VIEW_DESC &view,
                                     UINT64 resource_bytes, IndexedAtomicView *atomic = nullptr) {
    if (view.ViewDimension != D3D12_UAV_DIMENSION_BUFFER || view.Buffer.CounterOffsetInBytes ||
        !view.Buffer.NumElements)
        return false;
    if (binding.kind == ComputeViewKind::indexed_atomic_uav) {
        if (!atomic) return false;
        // Victory's view_stride: structured indices advance by StructureByteStride,
        // raw indices are byte offsets and typed indices advance by format width.
        const bool raw = view.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_RAW;
        const UINT structured = view.Buffer.StructureByteStride;
        if (raw) {
            if (structured || view.Format != DXGI_FORMAT_R32_TYPELESS) return false;
        } else if (view.Buffer.Flags) {
            return false;
        } else if (structured) {
            if (view.Format != DXGI_FORMAT_UNKNOWN || structured % 4 ||
                structured > D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES) return false;
        } else if (view.Format != DXGI_FORMAT_R32_UINT && view.Format != DXGI_FORMAT_R32_SINT) {
            return false;
        }
        UINT guest_stride = 0;
        if (!x2w_startup::native_view_stride(19, &view, sizeof(view), guest_stride)) return false;
        const UINT bytes_per_element = raw ? 4 : guest_stride;
        const UINT64 bytes = UINT64(view.Buffer.NumElements) * bytes_per_element;
        const UINT64 capacity = resource_bytes / bytes_per_element;
        if (bytes > UINT32_MAX || view.Buffer.FirstElement > capacity ||
            view.Buffer.NumElements > capacity - view.Buffer.FirstElement) return false;
        const UINT64 start = view.Buffer.FirstElement * bytes_per_element;
        if (start % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) return false;
        *atomic = {guest_stride, raw ? UINT(bytes - 3) : view.Buffer.NumElements};
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.Buffer.FirstElement = start / 4;
        view.Buffer.NumElements = UINT(bytes / 4);
        view.Buffer.StructureByteStride = 0;
        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        return true;
    }
    if (view.Buffer.StructureByteStride) return false;
    UINT stride = binding.required_bytes;
    if (binding.kind == ComputeViewKind::raw_buffer_uav) {
        if (view.Format != DXGI_FORMAT_R32_TYPELESS || view.Buffer.Flags != D3D12_BUFFER_UAV_FLAG_RAW)
            return false;
        stride = 4;
    } else if (view.Buffer.Flags) {
        return false;
    } else if (binding.kind == ComputeViewKind::half_buffer_uav) {
        if (!((stride == 2 && view.Format == DXGI_FORMAT_R16_FLOAT) ||
              (stride == 4 && view.Format == DXGI_FORMAT_R16G16_FLOAT) ||
              (stride == 8 && view.Format == DXGI_FORMAT_R16G16B16A16_FLOAT)))
            return false;
    } else if (binding.kind == ComputeViewKind::typed_buffer_uav) {
        const DXGI_FORMAT formats[][3] = {
            {DXGI_FORMAT_R32_UINT,DXGI_FORMAT_R32_SINT,DXGI_FORMAT_R32_FLOAT},
            {DXGI_FORMAT_R32G32_UINT,DXGI_FORMAT_R32G32_SINT,DXGI_FORMAT_R32G32_FLOAT},
            {DXGI_FORMAT_R32G32B32_UINT,DXGI_FORMAT_R32G32B32_SINT,DXGI_FORMAT_R32G32B32_FLOAT},
            {DXGI_FORMAT_R32G32B32A32_UINT,DXGI_FORMAT_R32G32B32A32_SINT,DXGI_FORMAT_R32G32B32A32_FLOAT}
        };
        if (stride < 4 || stride > 16 || stride % 4) return false;
        const auto &shape = formats[stride / 4 - 1];
        if (view.Format != shape[0] && view.Format != shape[1] && view.Format != shape[2]) return false;
        view.Format = shape[0];
    } else {
        return false;
    }
    const UINT64 capacity = resource_bytes / stride;
    return view.Buffer.FirstElement <= capacity && view.Buffer.NumElements <= capacity - view.Buffer.FirstElement;
}
}
