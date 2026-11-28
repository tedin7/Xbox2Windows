#include "startup_stubs/compute_output_view_check.h"
#include <cstdio>
#include <cstring>

int main()
{
    using namespace x2w_startup;
    D3D12_RESOURCE_DESC resource{};
    resource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resource.DepthOrArraySize = 1;
    resource.MipLevels = 4;
    resource.SampleDesc.Count = 1;
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    view.Texture2D.MipSlice = 1;
    const auto original_resource = resource;
    const auto original_view = view;
    const auto shape = [&] { return compute_output_shape_reason(resource, view, true, false, 0); };
    const auto outside = [&] { const char* r = shape(); return r && !std::strcmp(r, "an output texture view is outside the resource"); };
    if (shape() || compute_output_support_reason(kSupport2UavTypedStore | kSupport2UavTypedLoad, true) ||
        std::memcmp(&resource, &original_resource, sizeof(resource)) || std::memcmp(&view, &original_view, sizeof(view))) return 1;
    resource.MipLevels = 1; view.Texture2D.MipSlice = 3;
    if (!outside()) return 2;
    view.Texture2D.MipSlice = 0; resource.SampleDesc.Count = 4;
    if (!outside()) return 3;
    resource.SampleDesc.Count = 1; resource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (!outside()) return 4;
    if (!compute_output_support_reason(kSupport2UavTypedLoad, false) ||
        compute_output_support_reason(kSupport2UavTypedStore, false) ||
        !compute_output_support_reason(kSupport2UavTypedStore, true)) return 5;
    resource = original_resource; view = original_view;
    view.Texture2D.PlaneSlice = 1;
    if (!shape()) return 6;
    view.Texture2D.PlaneSlice = 0;
    if (compute_output_shape_reason(resource, view, false, false, 0)) return 7;
    resource.DepthOrArraySize = 2;
    if (!compute_output_shape_reason(resource, view, false, false, 0)) return 8;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    view.Texture2DArray = {0, 0, 2, 0};
    if (compute_output_shape_reason(resource, view, true, true, 0)) return 9;
    view.Texture2DArray.ArraySize = 3;
    if (!compute_output_shape_reason(resource, view, true, true, 0)) return 10;
    resource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D; view.Texture1D.MipSlice = 0;
    if (compute_output_shape_reason(resource, view, true, false, 1)) return 11;
    view.Texture1D.MipSlice = 4;
    if (!compute_output_shape_reason(resource, view, true, false, 1)) return 12;
    resource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D; view.Texture3D = {0, 0, 2};
    if (compute_output_shape_reason(resource, view, true, false, 3)) return 13;
    view.Texture3D.WSize = 3;
    if (!compute_output_shape_reason(resource, view, true, false, 3)) return 14;
    std::puts("compute output view: PASS");
    return 0;
}
