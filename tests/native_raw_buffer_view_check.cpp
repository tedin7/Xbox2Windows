#include "native_buffer_store_view.h"
#include <cstdio>
#include <limits>

int main()
{
    const x2w::ComputeViewBinding binding{x2w::ComputeViewKind::raw_buffer_uav, 0, 32, 0, 4};
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
    view.Buffer.FirstElement = 4;
    view.Buffer.NumElements = 16;
    view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    const auto valid = [&](D3D12_UNORDERED_ACCESS_VIEW_DESC v, UINT64 bytes) {
        return x2w::native_buffer_store_view(binding, v, bytes);
    };
    if (!valid(view, 128) || valid(view, 79)) return 1;
    auto last = view;
    last.Buffer.FirstElement = 16;
    if (!valid(last, 128) || valid(last, 127)) return 2;
    auto invalid = view;
    invalid.Buffer.FirstElement = 5;
    if (valid(invalid, 128)) return 3;
    invalid = view;
    invalid.Buffer.FirstElement = (std::numeric_limits<UINT64>::max)() - 3;
    if (valid(invalid, 128)) return 4;
    invalid = view;
    invalid.Buffer.NumElements = 0;
    if (valid(invalid, 128)) return 5;
    invalid.Buffer.NumElements = 29;
    if (valid(invalid, 128)) return 6;
    invalid = view;
    invalid.Buffer.CounterOffsetInBytes = 4096;
    if (valid(invalid, 128)) return 7;
    invalid = view;
    invalid.Buffer.StructureByteStride = 4;
    if (valid(invalid, 128)) return 8;
    invalid = view;
    invalid.Format = DXGI_FORMAT_R32_UINT;
    if (valid(invalid, 128)) return 9;
    invalid = view;
    invalid.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    if (valid(invalid, 128)) return 10;
    std::puts("raw buffer view: aligned and exact-end accepted; truncated, unaligned, overflowing, empty, counter, stride, format and flags refused");
    return 0;
}
