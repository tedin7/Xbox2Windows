#pragma once

// Console (GFX10) texture memory layouts for the non-XOR standard and display swizzle modes, and
// copies between such a layout and row-major texels.
//
// A console resource description carries its swizzle mode in the Layout field as 0x100 + mode. The
// modes handled here are 256 B, 4 KiB and 64 KiB blocks in the standard (S) and display (D) orders:
// 0x101, 0x102, 0x105, 0x106, 0x109 and 0x10a. The address equations, block dimensions and mip-tail
// rules follow AMD addrlib as Mesa carries it under src/amd/addrlib, at the revision
// tools/fetch_addrlib.py pins: Mesa 25.1.0, commit 5c142e46f3f6e752ed745fd48912ebb8fad67145. The three
// files used carry this notice:
//   Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT
// - src/gfx10/gfx10SwizzlePattern.h: GFX10_SW_{256,4K,64K}_{S,D}_PATINFO, one-pipe rows, and the
//   nibbles they name.
// - src/core/addrlib2.cpp: ComputeThinBlockDimension (a block is 2^ceil(n/2) by 2^floor(n/2) elements)
//   and GetMipTailDim (the tail holds mips up to half a block wide and one block high).
// - src/gfx10/gfx10addrlib.cpp: ComputeSurfaceInfoMacroTiled (the tail block comes first in a slice,
//   then the mips from the smallest to the largest; offsets and origins inside the tail block).
// The rules are restated here; no addrlib code is included.
// Measurements are in elements: one texel, or one 4x4 block of a block-compressed format.

#include <d3d12.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace x2w_startup::gfx10 {
constexpr UINT max_mips = 15;

struct Element {
    UINT bytes = 0;  // 1, 2, 4, 8 or 16; 0 for a format without a layout here
    UINT texels = 1; // side of the texel block one element covers
};

inline Element element_of(DXGI_FORMAT format)
{
    const UINT f = static_cast<UINT>(format);
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) return {8, 4};                          // BC1, BC4
    if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) return {16, 4}; // BC2, BC3, BC5, BC6H, BC7
    if (f >= 1 && f <= 4) return {16, 1};                                                     // R32G32B32A32
    if (f >= 9 && f <= 18) return {8, 1};                                                     // R16G16B16A16, R32G32
    if ((f >= 23 && f <= 47) || f == 67 || f == 87 || f == 88 || (f >= 90 && f <= 93)) return {4, 1};
    if ((f >= 48 && f <= 59) || f == 85 || f == 86 || f == 115) return {2, 1};
    if (f >= 60 && f <= 65) return {1, 1};
    return {};
}

// The swizzle mode of a console description, or ~0u when the Layout field does not hold one.
inline UINT swizzle_mode(const D3D12_RESOURCE_DESC& d)
{
    const UINT layout = static_cast<UINT>(d.Layout);
    return (layout & ~0xffu) == 0x100 ? layout & 0xff : ~0u;
}

inline UINT block_log2(UINT mode)
{
    switch (mode) {
    case 1: case 2: return 8;
    case 5: case 6: return 12;
    case 9: case 10: return 16;
    default: return 0;
    }
}

// A two-dimensional, single-sample texture whose memory this header can address.
inline bool supports(const D3D12_RESOURCE_DESC& d)
{
    const UINT log2 = block_log2(swizzle_mode(d));
    if (!log2 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !element_of(d.Format).bytes) return false;
    if (!d.Width || !d.Height || d.Width > 16384 || d.Height > 16384 || !d.DepthOrArraySize) return false;
    if (!d.MipLevels || d.MipLevels > max_mips || d.SampleDesc.Count != 1) return false;
    return log2 >= 12 || d.MipLevels == 1; // a 256 B block has no mip tail here
}

struct Layout {
    UINT bytes_per_element = 0, texels = 1, width = 0, height = 0, mips = 0, slices = 0;
    UINT block_bytes = 0, block_width = 0, block_height = 0;
    UINT first_tail = 0;  // first mip inside the shared tail block; mips when there is none
    std::uint64_t slice_size = 0;
    std::array<std::uint64_t, max_mips> start{};          // byte offset of a mip in its slice
    std::array<UINT, max_mips> blocks_wide{}, origin_x{}, origin_y{};
    std::array<UINT, 256> x_bits{}, y_bits{};             // in-block byte offset contributed by a coordinate

    explicit Layout(const D3D12_RESOURCE_DESC& d)
        : width(static_cast<UINT>(d.Width)), height(d.Height), mips(d.MipLevels), slices(d.DepthOrArraySize)
    {
        const Element element = element_of(d.Format);
        bytes_per_element = element.bytes;
        texels = element.texels;
        const UINT mode = swizzle_mode(d), log2 = block_log2(mode);
        UINT element_log2 = 0;
        while ((1u << element_log2) < bytes_per_element) ++element_log2;
        const UINT count_log2 = log2 - element_log2, width_log2 = (count_log2 + 1) / 2;
        block_bytes = 1u << log2;
        block_width = 1u << width_log2;
        block_height = 1u << (count_log2 - width_log2);

        // Address bits 0-7 of the 256 B micro block, by element size (1, 2, 4, 8, 16 bytes): 'x' or 'y'
        // and the coordinate bit, '-' for a bit inside the element. Only the 16-byte standard rows have a
        // second source in the tree (bc7_4k_s.h, bc64_chain.h); the display rows that differ from the
        // standard ones (1, 8 and 16 bytes) are checked against the addrlib rows alone
        // (docs/startup-runtime.md, "Textures published by address").
        static constexpr const char* standard[5] = {
            "x0x1x2x3y0y1y2y3", "--x0x1x2y0y1y2x3", "----x0x1y0y1y2x2", "------x0y0y1x1x2", "--------y0y1x0x1"};
        static constexpr const char* display[5] = {
            "x0x1x2y1y0y2x3y3", "--x0x1x2y0y1y2x3", "----x0x1y0y1y2x2", "------x0y0x1x2y1", "--------x0y0x1y1"};
        static constexpr UINT micro[5][2] = {{16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}};
        const bool is_display = mode == 2 || mode == 6 || mode == 10;
        const char* pattern = (is_display ? display : standard)[element_log2];
        std::array<UINT, 8> x_at{}, y_at{}; // address bit of each coordinate bit
        UINT next_x = 0, next_y = 0;
        for (UINT bit = 0; bit < 8; ++bit) {
            const UINT index = static_cast<UINT>(pattern[2 * bit + 1] - '0');
            if (pattern[2 * bit] == 'x') x_at[index] = bit, next_x = (std::max)(next_x, index + 1);
            if (pattern[2 * bit] == 'y') y_at[index] = bit, next_y = (std::max)(next_y, index + 1);
        }
        // Above the micro block the coordinates alternate, y on the even address bits.
        for (UINT bit = 8; bit < log2; ++bit) (bit & 1 ? x_at[next_x++] : y_at[next_y++]) = bit;
        for (UINT v = 0; v < 256; ++v)
            for (UINT bit = 0; bit < 8; ++bit) {
                if (bit < next_x && ((v >> bit) & 1)) x_bits[v] |= 1u << x_at[bit];
                if (bit < next_y && ((v >> bit) & 1)) y_bits[v] |= 1u << y_at[bit];
            }

        // Mip sizes for the layout round up at every level (addrlib GetMipSize).
        const UINT columns0 = (width + texels - 1) / texels, rows0 = (height + texels - 1) / texels;
        const auto shifted = [](UINT v, UINT mip) { return (std::max)(1u, (v + (1u << mip) - 1) >> mip); };
        const UINT tail_mips = log2 - 4; // most mips one tail block holds, for 4 KiB and 64 KiB blocks
        first_tail = mips;
        std::array<std::uint64_t, max_mips> size{};
        for (UINT mip = 0; mips > 1 && mip < mips; ++mip) {
            const UINT w = shifted(columns0, mip), h = shifted(rows0, mip);
            if (w <= block_width / 2 && h <= block_height && mips - mip <= tail_mips) {
                first_tail = mip;
                break;
            }
        }
        for (UINT mip = 0; mip < first_tail; ++mip) {
            const UINT w = shifted(columns0, mip), h = shifted(rows0, mip);
            blocks_wide[mip] = (w + block_width - 1) / block_width;
            size[mip] = std::uint64_t{blocks_wide[mip]} * ((h + block_height - 1) / block_height) * block_bytes;
        }
        std::uint64_t offset = first_tail < mips ? block_bytes : 0;
        for (UINT mip = first_tail; mip-- > 0;) {
            start[mip] = offset;
            offset += size[mip];
        }
        slice_size = offset;
        for (UINT mip = first_tail; mip < mips; ++mip) {
            // The position of a tail mip is its byte offset in the block, read back as micro block coordinates.
            const UINT m = tail_mips - 1 - (mip - first_tail);
            const UINT tail_offset = m > 6 ? 16u << m : m << 8;
            UINT x = 0, y = 0;
            for (UINT bit = 0; bit < 6; ++bit) {
                y |= ((tail_offset >> (8 + 2 * bit)) & 1) << bit;
                x |= ((tail_offset >> (9 + 2 * bit)) & 1) << bit;
            }
            start[mip] = tail_offset;
            origin_x[mip] = x * micro[element_log2][0];
            origin_y[mip] = y * micro[element_log2][1];
        }
    }

    // Elements a mip holds, with the desktop's mip sizes (rounded down, at least one texel).
    UINT columns(UINT mip) const { return ((std::max)(1u, width >> mip) + texels - 1) / texels; }
    UINT rows(UINT mip) const { return ((std::max)(1u, height >> mip) + texels - 1) / texels; }
    std::uint64_t bytes() const { return slice_size * slices; }
    // Bytes from the start of the texture to the end of the last element any mip of the last slice holds. A texture
    // smaller than its block (a 16x4 texture in a 4 KiB block) is read only up to here; the block's padding past it is
    // never addressed, so it does not have to be mapped. Never more than bytes().
    std::uint64_t used_bytes() const
    {
        std::uint64_t extent = 0;
        for (UINT mip = 0; mip < mips; ++mip) {
            const UINT w = columns(mip), h = rows(mip);
            std::uint64_t end = 0;
            if (mip >= first_tail) {
                for (UINT y = 0; y < h; ++y)
                    for (UINT x = 0; x < w; ++x) end = (std::max)(end, address(mip, x, y, 0));
            } else {
                // The last block in row-major order holds the highest address; inside it the x and y bits are disjoint,
                // so the highest address is the highest x contribution plus the highest y contribution.
                const UINT last_x = (w - 1) % block_width, last_y = (h - 1) % block_height;
                std::uint64_t most_x = 0, most_y = 0;
                for (UINT x = 0; x <= last_x; ++x) most_x = (std::max)(most_x, std::uint64_t{x_bits[x]});
                for (UINT y = 0; y <= last_y; ++y) most_y = (std::max)(most_y, std::uint64_t{y_bits[y]});
                const std::uint64_t block = std::uint64_t{(h - 1) / block_height} * blocks_wide[mip] + (w - 1) / block_width;
                end = start[mip] + block * block_bytes + most_x + most_y;
            }
            extent = (std::max)(extent, end + bytes_per_element);
        }
        return (std::min)(slice_size, extent) + slice_size * (slices - 1);
    }

    // Byte offset of element (x, y) of a mip of a slice from the start of the texture.
    std::uint64_t address(UINT mip, UINT x, UINT y, UINT slice) const
    {
        const std::uint64_t base = slice * slice_size;
        if (mip >= first_tail) return base + (x_bits[(origin_x[mip] + x) & 255] | y_bits[(origin_y[mip] + y) & 255]);
        const std::uint64_t block = std::uint64_t{y / block_height} * blocks_wide[mip] + x / block_width;
        return base + start[mip] + block * block_bytes + (x_bits[x % block_width] | y_bits[y % block_height]);
    }

    // Copies one subresource from the console layout at `tiled` to rows `pitch` bytes apart at `linear`.
    void detile(UINT mip, UINT slice, const unsigned char* tiled, unsigned char* linear, std::size_t pitch) const
    {
        const UINT w = columns(mip), h = rows(mip);
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x)
                std::memcpy(linear + y * pitch + std::size_t{x} * bytes_per_element, tiled + address(mip, x, y, slice),
                            bytes_per_element);
    }
    // The opposite copy, from rows into the console layout.
    void tile(UINT mip, UINT slice, unsigned char* tiled, const unsigned char* linear, std::size_t pitch) const
    {
        const UINT w = columns(mip), h = rows(mip);
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x)
                std::memcpy(tiled + address(mip, x, y, slice), linear + y * pitch + std::size_t{x} * bytes_per_element,
                            bytes_per_element);
    }
};
} // namespace x2w_startup::gfx10
