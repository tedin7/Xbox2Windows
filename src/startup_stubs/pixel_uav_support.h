// Experimental (docs/startup-runtime.md): which D3D12_FORMAT_SUPPORT2 bits a pixel-stage image store needs.
//
// tools/startup/gfx2hlsl.py writes 'uav N tex2d table P O' for every RWTexture2D<float4> a pixel program stores to,
// and a 'uavload N' line after it when the program also reads that texture through the UAV: a store whose dmask
// selects only some channels reads the texel back, replaces those channels and writes it again. A typed UAV store
// needs UAV_TYPED_STORE; the read-back also needs UAV_TYPED_LOAD, which many formats lack on many devices, so the
// draw is refused by name instead of reading undefined data.
#pragma once

namespace x2w_startup {

// Values of D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD and D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE (checked against the
// D3D12 header where the runtime uses them).
constexpr unsigned kSupport2UavTypedLoad = 0x40;
constexpr unsigned kSupport2UavTypedStore = 0x80;

// nullptr when a format with these support2 bits can be bound for a store (and a typed read-back when needs_load).
inline const char* pixel_uav_support_reason(unsigned support2, bool needs_load)
{
    if (!(support2 & kSupport2UavTypedStore)) return "format has no typed UAV store on this device";
    if (needs_load && !(support2 & kSupport2UavTypedLoad)) return "format has no typed UAV load on this device";
    return nullptr;
}

}  // namespace x2w_startup
