#include "startup_stubs/sampler_desc_check.h"
#include <cstring>
#include <limits>
#include <cstdio>
using namespace x2w_startup;

int main()
{
    D3D12_SAMPLER_DESC base{};
    base.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    base.AddressU = base.AddressV = base.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    base.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_SAMPLER_DESC valid[] = {base, base, base, base, base};
    valid[0].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    valid[1].Filter = D3D12_FILTER_ANISOTROPIC; valid[1].MaxAnisotropy = 16;
    valid[2].Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR; valid[2].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    valid[3].AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    for (float& c : valid[3].BorderColor) c = 1;
    for (auto s : valid) { const auto before = s; if (sanitize_sampler(s) || std::memcmp(&s, &before, sizeof(s))) return 1; }
    auto s = base;
    s.Filter = D3D12_FILTER_ANISOTROPIC; s.MaxAnisotropy = 0;
    if (!(sanitize_sampler(s) & sampler_anisotropy) || s.MaxAnisotropy != 1) return 2;
    s.MaxAnisotropy = 64; if (!(sanitize_sampler(s) & sampler_anisotropy) || s.MaxAnisotropy != 16) return 3;
    s = base; s.MinLOD = 4; s.MaxLOD = 2;
    if (!(sanitize_sampler(s) & sampler_lod) || s.MinLOD != 2) return 4;
    s.MinLOD = s.MaxLOD = std::numeric_limits<float>::quiet_NaN();
    if (!(sanitize_sampler(s) & sampler_lod) || s.MinLOD != 0 || s.MaxLOD != D3D12_FLOAT32_MAX) return 5;
    s = base; s.MipLODBias = 20;
    if (!(sanitize_sampler(s) & sampler_bias) || s.MipLODBias != D3D12_MIP_LOD_BIAS_MAX) return 6;
    s.MipLODBias = -20; if (!(sanitize_sampler(s) & sampler_bias) || s.MipLODBias != -16) return 7;
    s.MipLODBias = std::numeric_limits<float>::quiet_NaN(); if (!(sanitize_sampler(s) & sampler_bias) || s.MipLODBias != 0) return 8;
    s = base; s.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
    if (!(sanitize_sampler(s) & sampler_comparison) || s.ComparisonFunc != D3D12_COMPARISON_FUNC_LESS_EQUAL) return 9;
    s = base; s.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(9);
    if (!(sanitize_sampler(s) & sampler_comparison) || s.ComparisonFunc != D3D12_COMPARISON_FUNC_NEVER) return 10;
    s = base; s.AddressU = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(0); s.AddressV = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(6);
    if (!(sanitize_sampler(s) & sampler_address) || s.AddressU != D3D12_TEXTURE_ADDRESS_MODE_CLAMP || s.AddressV != D3D12_TEXTURE_ADDRESS_MODE_CLAMP) return 11;
    s = base; s.BorderColor[0] = std::numeric_limits<float>::quiet_NaN(); s.BorderColor[1] = -1; s.BorderColor[2] = 2;
    if (!(sanitize_sampler(s) & sampler_border) || s.BorderColor[0] != 0 || s.BorderColor[1] != 0 || s.BorderColor[2] != 1) return 12;
    s = base; s.Filter = static_cast<D3D12_FILTER>(0x182);
    if (!(sanitize_sampler(s) & sampler_filter) || static_cast<unsigned>(s.Filter) != 0x195) return 13;
    D3D12_STATIC_SAMPLER_DESC st{};
    st.Filter = D3D12_FILTER_ANISOTROPIC; st.AddressU = st.AddressV = st.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    st.MinLOD = 4; st.MaxLOD = 2; st.MipLODBias = 20;
    const unsigned mask = sanitize_static_sampler(st);
    if ((mask & (sampler_anisotropy | sampler_lod | sampler_bias)) != (sampler_anisotropy | sampler_lod | sampler_bias) || st.MinLOD != 2) return 14;
    const auto before = st; if (sanitize_static_sampler(st) || std::memcmp(&st, &before, sizeof(st))) return 15;
    st.Filter = static_cast<D3D12_FILTER>(0x82); st.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(0);
    st.AddressW = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(0);
    st.MinLOD = st.MaxLOD = std::numeric_limits<float>::quiet_NaN();
    st.MipLODBias = std::numeric_limits<float>::quiet_NaN();
    const unsigned changed = sanitize_static_sampler(st);
    if ((changed & (sampler_filter | sampler_comparison | sampler_address | sampler_lod | sampler_bias)) !=
        (sampler_filter | sampler_comparison | sampler_address | sampler_lod | sampler_bias) ||
        st.Filter != D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR || st.ComparisonFunc != D3D12_COMPARISON_FUNC_LESS_EQUAL ||
        st.AddressW != D3D12_TEXTURE_ADDRESS_MODE_CLAMP || st.MinLOD != 0 || st.MaxLOD != D3D12_FLOAT32_MAX || st.MipLODBias != 0) return 16;

    std::puts("sampler descriptions: passed"); return 0;
}
