#pragma once
#include <d3d12.h>
namespace x2w_startup {
inline D3D12_DESCRIPTOR_HEAP_DESC safe_heap_desc(const D3D12_DESCRIPTOR_HEAP_DESC& input) {
    auto result = input;
    if (input.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
        if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && input.NumDescriptors > 1000000)
            result.NumDescriptors = 1000000;
        else if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && input.NumDescriptors > 2048)
            result.NumDescriptors = 2048;
        else if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV || input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
            result.Flags = static_cast<D3D12_DESCRIPTOR_HEAP_FLAGS>(input.Flags & ~D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    }
    return result;
}
}
