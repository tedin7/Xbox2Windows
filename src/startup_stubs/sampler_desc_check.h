#pragma once

#include <d3d12.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace x2w_startup {
enum SamplerRepair : unsigned {
    sampler_filter = 1, sampler_anisotropy = 2, sampler_lod = 4, sampler_bias = 8,
    sampler_comparison = 16, sampler_address = 32, sampler_border = 64
};

// D3D12_SAMPLER_DESC and D3D12_COMPARISON_FUNC: preserve fields that are
// ignored by the selected filter. The filter encoding has three one-bit
// basic filters, or the anisotropic encoding, plus two reduction bits.
template<class Desc> inline unsigned sanitize_sampler_common(Desc& s)
{
    unsigned changed = 0;
    unsigned filter = static_cast<unsigned>(s.Filter);
    const unsigned basic = filter & ~0x180u;
    if ((filter & ~0x1d5u) || (basic != 0x55u && (basic & ~0x15u))) {
        filter = (filter & 0x180u) | 0x15u;
        s.Filter = static_cast<D3D12_FILTER>(filter);
        changed |= sampler_filter;
    }
    if ((filter & ~0x180u) == 0x55u && (s.MaxAnisotropy < 1 || s.MaxAnisotropy > 16)) {
        s.MaxAnisotropy = (std::max)(1u, (std::min)(16u, s.MaxAnisotropy));
        changed |= sampler_anisotropy;
    }
    if (std::isnan(s.MinLOD)) { s.MinLOD = 0; changed |= sampler_lod; }
    if (std::isnan(s.MaxLOD)) { s.MaxLOD = D3D12_FLOAT32_MAX; changed |= sampler_lod; }
    // min(max(lod, MinLOD), MaxLOD) is always MaxLOD for a reversed range.
    if (s.MinLOD > s.MaxLOD) { s.MinLOD = s.MaxLOD; changed |= sampler_lod; }
    if (std::isnan(s.MipLODBias)) { s.MipLODBias = 0; changed |= sampler_bias; }
    if (s.MipLODBias < D3D12_MIP_LOD_BIAS_MIN) { s.MipLODBias = D3D12_MIP_LOD_BIAS_MIN; changed |= sampler_bias; }
    if (s.MipLODBias > D3D12_MIP_LOD_BIAS_MAX) { s.MipLODBias = D3D12_MIP_LOD_BIAS_MAX; changed |= sampler_bias; }
    const int cmp = static_cast<int>(s.ComparisonFunc);
    if ((filter & 0x180u) == 0x80u ? (cmp < 1 || cmp > 8) : (cmp < 0 || cmp > 8)) {
        s.ComparisonFunc = (filter & 0x180u) == 0x80u ? D3D12_COMPARISON_FUNC_LESS_EQUAL : D3D12_COMPARISON_FUNC_NEVER;
        changed |= sampler_comparison;
    }
    auto address = [&](D3D12_TEXTURE_ADDRESS_MODE& mode) {
        if (static_cast<unsigned>(mode) < 1 || static_cast<unsigned>(mode) > 5) {
            mode = D3D12_TEXTURE_ADDRESS_MODE_CLAMP; changed |= sampler_address;
        }
    };
    address(s.AddressU); address(s.AddressV); address(s.AddressW);
    return changed;
}
inline unsigned sanitize_sampler(D3D12_SAMPLER_DESC& s)
{
    unsigned changed = sanitize_sampler_common(s);
    for (float& c : s.BorderColor) {
        if (std::isnan(c) || c < 0) { c = 0; changed |= sampler_border; }
        else if (c > 1) { c = 1; changed |= sampler_border; }
    }
    return changed;
}
inline unsigned sanitize_static_sampler(D3D12_STATIC_SAMPLER_DESC& s)
{
    return sanitize_sampler_common(s);
}
}
