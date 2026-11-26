#pragma once

#include <d3d12.h>

#include <cstdint>
#include <cstring>

namespace x2w_startup {

inline bool supports_bc7_4k_s(const D3D12_RESOURCE_DESC& desc)
{
    return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
           desc.Format == DXGI_FORMAT_BC7_UNORM && desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN &&
           desc.Width > 0 && desc.Width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
           desc.Height > 0 && desc.Height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
           desc.DepthOrArraySize == 1 && desc.MipLevels == 1 && desc.SampleDesc.Count == 1 &&
           desc.SampleDesc.Quality == 0 && desc.Alignment == 0 && desc.Flags == D3D12_RESOURCE_FLAG_NONE;
}

struct Bc16Surface {
    static constexpr UINT block_texels = 4, block_bytes = 16;
    static constexpr UINT tile_blocks = 16, tile_bytes = tile_blocks * tile_blocks * block_bytes;

    const UINT columns, rows, tile_columns, tile_rows;

    Bc16Surface(UINT width, UINT height)
        : columns((width + block_texels - 1) / block_texels),
          rows((height + block_texels - 1) / block_texels),
          tile_columns((columns + tile_blocks - 1) / tile_blocks),
          tile_rows((rows + tile_blocks - 1) / tile_blocks)
    {
    }

    explicit Bc16Surface(const D3D12_RESOURCE_DESC& desc) : Bc16Surface(static_cast<UINT>(desc.Width), desc.Height) {}

    UINT padded_width() const { return tile_columns * tile_blocks * block_texels; }
    UINT padded_height() const { return tile_rows * tile_blocks * block_texels; }
    UINT padded_row_bytes() const { return tile_columns * tile_blocks * block_bytes; }
    std::uint64_t size() const { return std::uint64_t{tile_columns} * tile_rows * tile_bytes; }

    std::uint64_t block_offset(UINT column, UINT row) const
    {
        const UINT x = column % tile_blocks, y = row % tile_blocks;
        // GFX10 4 KiB standard non-XOR 16-byte block order: y0 y1 x0 x1 y2 x2 y3 x3.
        const UINT block = (y & 3) | ((x & 3) << 2) | ((y & 4) << 2) |
                           ((x & 4) << 3) | ((y & 8) << 3) | ((x & 8) << 4);
        const auto tile = std::uint64_t{row / tile_blocks} * tile_columns + column / tile_blocks;
        return tile * tile_bytes + block * block_bytes;
    }

    // Tile indices increase row-major, and each axis contributes disjoint address bits.
    std::uint64_t used_bytes() const
    {
        return block_offset(columns - 1, rows - 1) + block_bytes;
    }

    void detile(const unsigned char* source, unsigned char* dest, UINT row_pitch) const
    {
        for (UINT y = 0; y < rows; ++y) {
            auto* row = dest + std::uint64_t{y} * row_pitch;
            for (UINT x = 0; x < columns; ++x)
                std::memcpy(row + x * block_bytes, source + block_offset(x, y), block_bytes);
        }
    }
};

}  // namespace x2w_startup
