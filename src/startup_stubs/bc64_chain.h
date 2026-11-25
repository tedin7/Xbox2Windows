#pragma once

#include <d3d12.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace x2w_startup::bc64 {
constexpr UINT first_tail = 3, tile_bytes = 65536, tile_side = 256;

inline UINT guest_layout_word(const D3D12_RESOURCE_DESC& d)
{
    UINT word;
    static_assert(sizeof(word) == sizeof(d.Layout));
    std::memcpy(&word, &d.Layout, sizeof(word));
    return word;
}

inline bool supports(const D3D12_RESOURCE_DESC& d)
{
    const bool six = d.MipLevels == 6 && d.Height == 1024 && d.DepthOrArraySize == 1;
    const bool nine = d.MipLevels == 9 && (d.Height == 512 || d.Height == 1024) &&
                      (d.DepthOrArraySize == 1 || d.DepthOrArraySize == 2);
    return (six || nine) && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Alignment == 0 &&
           d.Width == 1024 && d.SampleDesc.Count == 1 && d.SampleDesc.Quality == 0 &&
           guest_layout_word(d) == 0x109 && d.Flags == D3D12_RESOURCE_FLAG_NONE &&
           (d.Format == DXGI_FORMAT_BC5_UNORM || d.Format == DXGI_FORMAT_BC7_UNORM);
}

// Explicit non-XOR 64 KiB standard layout for the shapes accepted above.
// Measurements are in 4x4 BC blocks except for offsets and slice sizes.
struct Chain {
    UINT height, levels, slices;
    explicit Chain(const D3D12_RESOURCE_DESC& d) : height(d.Height), levels(d.MipLevels), slices(d.DepthOrArraySize) {}
    UINT columns(UINT mip) const { return ((1024u >> mip) + 3) / 4; }
    UINT rows(UINT mip) const { return ((height >> mip) + 3) / 4; }
    UINT tiles_wide(UINT mip) const { return (columns(mip) + 63) / 64; }
    UINT tiles_high(UINT mip) const { return (rows(mip) + 63) / 64; }

    UINT start(UINT mip) const {
        constexpr std::array<UINT, 6> tail{0x8000, 0x4000, 0x2000, 0x1000, 0x800, 0x600};
        if (mip >= first_tail) return tail[mip - first_tail];
        UINT tiles = 1;
        for (UINT i = mip + 1; i < first_tail; ++i) tiles += tiles_wide(i) * tiles_high(i);
        return tiles * tile_bytes;
    }
    UINT slice_bytes() const { return start(0) + tiles_wide(0) * tiles_high(0) * tile_bytes; }
    UINT bytes() const { return slice_bytes() * slices; }

    UINT address(UINT mip, UINT x, UINT y, UINT slice) const {
        const UINT tile = (y / 64) * tiles_wide(mip) + x / 64;
        UINT within = ((y & 3) << 4) | ((x & 3) << 6);
        for (UINT bit = 2; bit < 6; ++bit) {
            within |= ((y >> bit) & 1) << (4 + bit * 2);
            within |= ((x >> bit) & 1) << (5 + bit * 2);
        }
        return slice * slice_bytes() + start(mip) + tile * tile_bytes + within;
    }

    template<bool into> void copy(UINT mip, UINT slice, unsigned char* dst, const unsigned char* src, UINT pitch) const {
        for (UINT y = 0; y < rows(mip); ++y)
            for (UINT x = 0; x < columns(mip); ++x) {
                const UINT tiled = address(mip, x, y, slice);
                const std::size_t linear = static_cast<std::size_t>(y) * pitch + 16 * x;
                if constexpr (into) std::memcpy(dst + tiled, src + linear, 16);
                else std::memcpy(dst + linear, src + tiled, 16);
            }
    }
};
} // namespace x2w_startup::bc64
