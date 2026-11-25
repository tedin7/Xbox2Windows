// Experimental (docs/startup-runtime.md): bound SRV format validation and conversion mode selection
// for translated shaders that execute level-zero MIMG image_load with the d16 modifier (#2023, item D).
//
// gfx2hlsl writes 'd16srv <index> exact' and an entry constant 'udsrc <ud> d16unorm16 <index>' when SRV
// <index> is read by image_load ... d16. Only formats independently verified exact against RDNA 2 D16
// image_load are accepted:
// - D16SrvMode::float_exact (d16unorm16 = 0): 32-bit float views (including R32_FLOAT_X8X24_TYPELESS)
//   and 8-bit UNORM views (256/256 exact under pack16_rne(Texture*<float4>.Load)).
// - D16SrvMode::unorm16 (d16unorm16 = 1): 16-bit UNORM views (65,536/65,536 exact when recovering the
//   16-bit code from Load and rounding u / 65535 directly to binary16, avoiding binary32 double-rounding
//   at 0xffaf and 0xffef).
// Integer (UINT/SINT), typeless, depth-stencil-only, SNORM16, SRGB, block-compressed, packed-target
// stand-in, console extension and unknown/unbound views are refused.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace x2w_startup {

enum class D16SrvMode : std::uint8_t {
    refused = 0,
    float_exact = 1,
    unorm16 = 2,
};

inline constexpr D16SrvMode d16_srv_mode(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
        return D16SrvMode::float_exact;
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
        return D16SrvMode::unorm16;
    default:
        return D16SrvMode::refused;
    }
}

inline constexpr bool d16_exact_srv_format(DXGI_FORMAT format)
{
    return d16_srv_mode(format) != D16SrvMode::refused;
}

struct D16ImageSrvs {
    std::uint32_t exact_mask = 0;

    // Parses 'd16srv <index> exact' for 0 <= index < 32 without leading zeroes or duplicates.
    bool read_manifest_line(std::string_view line)
    {
        constexpr std::string_view prefix = "d16srv ";
        constexpr std::string_view suffix = " exact";
        if (line.size() <= prefix.size() + suffix.size() ||
            line.substr(0, prefix.size()) != prefix ||
            line.substr(line.size() - suffix.size()) != suffix)
            return false;
        const std::string_view digits = line.substr(prefix.size(), line.size() - prefix.size() - suffix.size());
        if (digits.empty() || digits.size() > 2 || (digits.size() == 2 && digits[0] == '0'))
            return false;
        unsigned index = 0;
        for (char c : digits) {
            if (c < '0' || c > '9') return false;
            index = index * 10 + static_cast<unsigned>(c - '0');
        }
        if (index >= 32 || (exact_mask & (std::uint32_t{1} << index))) return false;
        exact_mask |= std::uint32_t{1} << index;
        return true;
    }

    bool any() const { return exact_mask != 0; }

    bool requires_exact(unsigned index) const
    {
        return index < 32 && (exact_mask & (std::uint32_t{1} << index)) != 0;
    }

    template <class SrvVec>
    bool valid_for_srvs(const SrvVec& srvs) const
    {
        if (!exact_mask) return true;
        if (srvs.size() < 32 && (exact_mask >> srvs.size()) != 0) return false;
        for (std::size_t i = 0; i < srvs.size() && i < 32; ++i) {
            if (requires_exact(static_cast<unsigned>(i)) && (srvs[i].kind < 1 || srvs[i].kind > 4))
                return false;
        }
        return true;
    }
};

}  // namespace x2w_startup
