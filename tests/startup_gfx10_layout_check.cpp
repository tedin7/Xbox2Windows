// Checks src/startup_stubs/gfx10_layout.h without a device:
// - the address bit of every coordinate bit against the one-pipe rows of addrlib's GFX10 pattern
//   tables (written out below as the nibbles those rows name), for six swizzle modes and five element sizes;
// - agreement with the three narrower layouts already in the tree (bc7_4k_s.h, bc64_chain.h, tile64_plan.h);
// - the mip offsets docs/experiments.md records for a 1024x1024 six-mip BC7 chain;
// - the linear mode (0x100): mip offsets worked out by hand from addrlib's HwlComputeSurfaceInfoLinear,
//   and, with --addrlib, every element address of several linear textures against addrlib itself;
// - that every element of every subresource gets its own bytes inside the texture, and that a
//   tile followed by a detile returns the rows it started from.
//
// Reference: AMD addrlib as Mesa carries it under src/amd/addrlib, at the revision tools/fetch_addrlib.py
// pins (Mesa 25.1.0, commit 5c142e46f3f6e752ed745fd48912ebb8fad67145): src/gfx10/gfx10SwizzlePattern.h,
// src/gfx10/gfx10addrlib.cpp and src/core/addrlib2.cpp, each with the notice
//   Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT
// The pattern rows are restated below as data; no addrlib code is included.
#include "startup_stubs/address_texture.h"
#include "startup_stubs/gfx10_layout.h"
#include "startup_stubs/bc64_chain.h"
#include "startup_stubs/bc7_4k_s.h"
#include "startup_stubs/tile64_plan.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
using x2w_startup::gfx10::Layout;
int failures = 0;

void fail(const std::string& what)
{
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

D3D12_RESOURCE_DESC texture(UINT width, UINT height, UINT mips, UINT slices, DXGI_FORMAT format, UINT layout)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = static_cast<UINT16>(slices);
    d.MipLevels = static_cast<UINT16>(mips);
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(layout);
    return d;
}

D3D12_RESOURCE_DESC texture_1d(UINT width, UINT mips, UINT slices, DXGI_FORMAT format, UINT layout)
{
    D3D12_RESOURCE_DESC d = texture(width, 1, mips, slices, format, layout);
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    return d;
}

// One format per element size: 1, 2, 4, 8 and 16 bytes.
constexpr DXGI_FORMAT by_size[5] = {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                                    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};

// Address bits from bit 0 up; "0" is a bit inside the element.
struct Equation {
    UINT layout;
    const char* bits[5];
};
const Equation equations[] = {
    {0x101, {"X0 X1 X2 X3 Y0 Y1 Y2 Y3", "0 X0 X1 X2 Y0 Y1 Y2 X3", "0 0 X0 X1 Y0 Y1 Y2 X2", "0 0 0 X0 Y0 Y1 X1 X2",
             "0 0 0 0 Y0 Y1 X0 X1"}},
    {0x102, {"X0 X1 X2 Y1 Y0 Y2 X3 Y3", "0 X0 X1 X2 Y0 Y1 Y2 X3", "0 0 X0 X1 Y0 Y1 Y2 X2", "0 0 0 X0 Y0 X1 X2 Y1",
             "0 0 0 0 X0 Y0 X1 Y1"}},
    {0x105, {"X0 X1 X2 X3 Y0 Y1 Y2 Y3 Y4 X4 Y5 X5", "0 X0 X1 X2 Y0 Y1 Y2 X3 Y3 X4 Y4 X5",
             "0 0 X0 X1 Y0 Y1 Y2 X2 Y3 X3 Y4 X4", "0 0 0 X0 Y0 Y1 X1 X2 Y2 X3 Y3 X4", "0 0 0 0 Y0 Y1 X0 X1 Y2 X2 Y3 X3"}},
    {0x106, {"X0 X1 X2 Y1 Y0 Y2 X3 Y3 Y4 X4 Y5 X5", "0 X0 X1 X2 Y0 Y1 Y2 X3 Y3 X4 Y4 X5",
             "0 0 X0 X1 Y0 Y1 Y2 X2 Y3 X3 Y4 X4", "0 0 0 X0 Y0 X1 X2 Y1 Y2 X3 Y3 X4", "0 0 0 0 X0 Y0 X1 Y1 Y2 X2 Y3 X3"}},
    {0x109, {"X0 X1 X2 X3 Y0 Y1 Y2 Y3 Y4 X4 Y5 X5 Y6 X6 Y7 X7", "0 X0 X1 X2 Y0 Y1 Y2 X3 Y3 X4 Y4 X5 Y5 X6 Y6 X7",
             "0 0 X0 X1 Y0 Y1 Y2 X2 Y3 X3 Y4 X4 Y5 X5 Y6 X6", "0 0 0 X0 Y0 Y1 X1 X2 Y2 X3 Y3 X4 Y4 X5 Y5 X6",
             "0 0 0 0 Y0 Y1 X0 X1 Y2 X2 Y3 X3 Y4 X4 Y5 X5"}},
    {0x10a, {"X0 X1 X2 Y1 Y0 Y2 X3 Y3 Y4 X4 Y5 X5 Y6 X6 Y7 X7", "0 X0 X1 X2 Y0 Y1 Y2 X3 Y3 X4 Y4 X5 Y5 X6 Y6 X7",
             "0 0 X0 X1 Y0 Y1 Y2 X2 Y3 X3 Y4 X4 Y5 X5 Y6 X6", "0 0 0 X0 Y0 X1 X2 Y1 Y2 X3 Y3 X4 Y4 X5 Y5 X6",
             "0 0 0 0 X0 Y0 X1 Y1 Y2 X2 Y3 X3 Y4 X4 Y5 X5"}},
};

void check_equations(bool mutate)
{
    for (const Equation& equation : equations)
        for (UINT size = 0; size < 5; ++size) {
            UINT layout = equation.layout;
            if (mutate && layout == 0x10a) layout = 0x109; // the display order read as the standard one
            const Layout l(texture(4096, 4096, 1, 1, by_size[size], layout));
            UINT bit = 0, xs = 0, ys = 0;
            for (const char* p = equation.bits[size]; *p; ++bit) {
                const char axis = *p++;
                UINT index = 0;
                if (axis != '0') index = static_cast<UINT>(*p++ - '0');
                if (*p == ' ') ++p;
                if (axis == '0') continue;
                const std::uint64_t got = axis == 'X' ? l.address(0, 1u << index, 0, 0) : l.address(0, 0, 1u << index, 0);
                if (got != (1ull << bit))
                    fail("layout " + std::to_string(equation.layout) + " element " + std::to_string(1u << size) + " " + axis +
                         std::to_string(index) + ": " + std::to_string(got) + ", expected bit " + std::to_string(bit));
                (axis == 'X' ? xs : ys) += 1;
            }
            if (l.block_bytes != (1u << bit) || l.block_width != (1u << xs) || l.block_height != (1u << ys))
                fail("layout " + std::to_string(equation.layout) + " element " + std::to_string(1u << size) + ": block size");
        }
}

void check_bc7_4k()
{
    const UINT sizes[][2] = {{64, 64}, {1024, 2112}, {256, 128}, {3840, 2176}, {100, 36}};
    for (const auto& s : sizes) {
        const Layout l(texture(s[0], s[1], 1, 1, DXGI_FORMAT_BC7_UNORM, 0x105));
        const x2w_startup::Bc16Surface surface(s[0], s[1]);
        if (l.bytes() != surface.size()) fail("4 KiB BC7 size");
        for (UINT y = 0; y < surface.rows; ++y)
            for (UINT x = 0; x < surface.columns; ++x)
                if (l.address(0, x, y, 0) != surface.block_offset(x, y)) return fail("4 KiB BC7 block offset");
    }
}

void check_bc64()
{
    const UINT shapes[][3] = {{1024, 6, 1}, {1024, 9, 1}, {512, 9, 2}, {1024, 9, 2}};
    for (const auto& s : shapes)
        for (const DXGI_FORMAT format : {DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC7_UNORM}) {
            const D3D12_RESOURCE_DESC d = texture(1024, s[0], s[1], s[2], format, 0x109);
            if (!x2w_startup::bc64::supports(d)) return fail("64 KiB chain shape is not one bc64_chain.h takes");
            const Layout l(d);
            const x2w_startup::bc64::Chain chain(d);
            if (l.bytes() != chain.bytes() || l.first_tail != x2w_startup::bc64::first_tail) fail("64 KiB chain size or tail");
            for (UINT slice = 0; slice < s[2]; ++slice)
                for (UINT mip = 0; mip < s[1]; ++mip)
                    for (UINT y = 0; y < chain.rows(mip); ++y)
                        for (UINT x = 0; x < chain.columns(mip); ++x)
                            if (l.address(mip, x, y, slice) != chain.address(mip, x, y, slice))
                                return fail("64 KiB chain address at mip " + std::to_string(mip));
        }
    // docs/experiments.md: 1024x1024 BC7, six mips, layout 0x109.
    const Layout l(texture(1024, 1024, 6, 1, DXGI_FORMAT_BC7_UNORM, 0x109));
    const std::uint64_t offsets[6] = {0x60000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000};
    if (l.bytes() != 0x160000) fail("documented chain size");
    for (UINT mip = 0; mip < 6; ++mip)
        if (l.address(mip, 0, 0, 0) != offsets[mip]) fail("documented offset of mip " + std::to_string(mip));
}

void check_tile64()
{
    struct Shape { UINT width, height, mips, slices; DXGI_FORMAT format; };
    const Shape shapes[] = {
        {512, 512, 1, 1, DXGI_FORMAT_BC1_UNORM},   {1024, 512, 1, 1, DXGI_FORMAT_BC7_UNORM}, {1024, 128, 1, 1, DXGI_FORMAT_BC3_UNORM},
        {2048, 2048, 12, 1, DXGI_FORMAT_BC1_UNORM}, {4096, 2048, 13, 1, DXGI_FORMAT_BC7_UNORM}, {1024, 1024, 11, 6, DXGI_FORMAT_BC4_UNORM},
        {1024, 1024, 11, 1, DXGI_FORMAT_R8_UNORM},  {512, 512, 10, 2, DXGI_FORMAT_R16_UNORM},   {2048, 1024, 12, 1, DXGI_FORMAT_R8G8B8A8_UNORM},
        {1000, 600, 10, 1, DXGI_FORMAT_BC3_UNORM},
    };
    for (const Shape& s : shapes) {
        const D3D12_RESOURCE_DESC d = texture(s.width, s.height, s.mips, s.slices, s.format, 0x10a);
        if (!x2w_startup::tile64::Plan::supports(d)) return fail("tile plan refuses a shape of this check");
        const x2w_startup::tile64::Plan plan(d);
        const Layout l(d);
        const std::string name = std::to_string(s.width) + "x" + std::to_string(s.height) + " format " + std::to_string(s.format);
        if (l.first_tail != plan.standard) fail(name + ": first tail mip " + std::to_string(l.first_tail) + " against " + std::to_string(plan.standard));
        if (l.slice_size != std::uint64_t{plan.slice_tiles} * 65536) fail(name + ": slice size");
        for (UINT mip = 0; mip < plan.standard && mip < l.first_tail; ++mip)
            if (l.start[mip] != std::uint64_t{plan.start[mip]} * 65536) fail(name + ": start of mip " + std::to_string(mip));
    }
}

void check_distinct_and_round_trip()
{
    struct Shape { UINT width, height, mips, slices; DXGI_FORMAT format; UINT layout; bool one_d = false; };
    const Shape shapes[] = {
        {1000, 520, 10, 1, DXGI_FORMAT_BC1_UNORM, 0x105}, {300, 200, 9, 2, DXGI_FORMAT_R8_UNORM, 0x109},
        {512, 1056, 1, 1, DXGI_FORMAT_BC1_UNORM, 0x105},   {1024, 2112, 1, 1, DXGI_FORMAT_R16_UNORM, 0x105},
        {640, 360, 10, 1, DXGI_FORMAT_BC7_UNORM, 0x10a},   {257, 129, 9, 3, DXGI_FORMAT_R8G8B8A8_UNORM, 0x106},
        {96, 40, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, 0x102}, {4, 4, 3, 1, DXGI_FORMAT_BC3_UNORM, 0x109},
        {2048, 2048, 12, 1, DXGI_FORMAT_BC5_UNORM, 0x109},
        {16, 4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x105}, {16, 4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x109},
        {16, 4, 3, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x109}, {33, 5, 1, 2, DXGI_FORMAT_R16G16B16A16_FLOAT, 0x10a},
        {100, 10, 3, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, 0x100}, {257, 129, 9, 2, DXGI_FORMAT_R8_UNORM, 0x100},
        {256, 256, 7, 1, DXGI_FORMAT_R32_FLOAT, 0x100},          {64, 64, 4, 3, DXGI_FORMAT_BC1_UNORM, 0x100},
        {300, 1, 9, 1, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0x100, true}, {4096, 1, 13, 1, DXGI_FORMAT_R8_UNORM, 0x100, true},
    };
    for (const Shape& s : shapes) {
        const D3D12_RESOURCE_DESC d = s.one_d ? texture_1d(s.width, s.mips, s.slices, s.format, s.layout)
                                              : texture(s.width, s.height, s.mips, s.slices, s.format, s.layout);
        const std::string name = std::to_string(s.width) + "x" + std::to_string(s.height) + " layout " + std::to_string(s.layout);
        if (!x2w_startup::gfx10::supports(d)) {
            fail(name + ": refused");
            continue;
        }
        const Layout l(d);
        const std::size_t total = static_cast<std::size_t>(l.bytes());
        std::vector<unsigned char> used(total / l.bytes_per_element, 0), tiled(total, 0);
        bool ok = total % l.block_bytes == 0;
        std::uint64_t highest = 0;
        for (UINT slice = 0; ok && slice < s.slices; ++slice)
            for (UINT mip = 0; ok && mip < s.mips; ++mip) {
                const UINT w = l.columns(mip), h = l.rows(mip);
                const std::size_t pitch = std::size_t{w} * l.bytes_per_element + 16;
                std::vector<unsigned char> rows(pitch * h), back(pitch * h, 0);
                for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<unsigned char>(i * 31 + mip * 7 + slice * 3 + 1);
                for (UINT y = 0; ok && y < h; ++y)
                    for (UINT x = 0; ok && x < w; ++x) {
                        const std::uint64_t at = l.address(mip, x, y, slice);
                        ok = at % l.bytes_per_element == 0 && at + l.bytes_per_element <= total && !used[at / l.bytes_per_element];
                        if (ok) used[at / l.bytes_per_element] = 1;
                        highest = (std::max)(highest, at + l.bytes_per_element);
                    }
                if (!ok) break;
                l.tile(mip, slice, tiled.data(), rows.data(), pitch);
                l.detile(mip, slice, tiled.data(), back.data(), pitch);
                for (UINT y = 0; ok && y < h; ++y)
                    ok = std::memcmp(rows.data() + y * pitch, back.data() + y * pitch, std::size_t{w} * l.bytes_per_element) == 0;
            }
        if (!ok) fail(name + ": an element outside the texture, shared by two subresources, or changed by a round trip");
        // The bytes a read must have mapped end at the highest element any texel uses, not at the end of the block.
        else if (l.used_bytes() != highest || l.used_bytes() > l.bytes())
            fail(name + ": used_bytes " + std::to_string(l.used_bytes()) + " is not the end of the highest element " + std::to_string(highest));
    }
}

void check_refusals()
{
    const D3D12_RESOURCE_DESC good = texture(256, 256, 1, 1, DXGI_FORMAT_BC1_UNORM, 0x105);
    if (!x2w_startup::gfx10::supports(good)) fail("plain 4 KiB BC1 texture refused");
    const auto refused = [&](const char* name, D3D12_RESOURCE_DESC d) {
        if (x2w_startup::gfx10::supports(d)) fail(std::string("accepted: ") + name);
    };
    D3D12_RESOURCE_DESC d = good;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; refused("desktop layout", d);
    d = good; d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x100); d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    refused("linear volume", d);
    d = good; d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x115); refused("XOR mode", d);
    d = good; d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x209); refused("layout outside 0x1xx", d);
    d = good; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D; refused("volume", d);
    d = good; d.SampleDesc.Count = 4; refused("multisampled", d);
    d = good; d.Format = DXGI_FORMAT_R32G32B32_FLOAT; refused("12-byte texels", d);
    d = good; d.MipLevels = 16; refused("16 mips", d);
    d = good; d.Width = 16385; refused("too wide", d);
    d = good; d.DepthOrArraySize = 0; refused("no slices", d);
    d = good; d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101); d.MipLevels = 2; refused("256 B chain", d);
    if (!x2w_startup::gfx10::supports(texture_1d(300, 9, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100)))
        fail("linear 1D RGBA8 chain refused");
    refused("1D in a 4 KiB swizzle mode", texture_1d(256, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x105));
    d = texture_1d(256, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100); d.Height = 2; refused("1D two rows high", d);
    refused("1D block-compressed", texture_1d(256, 1, 1, DXGI_FORMAT_BC1_UNORM, 0x100));
    refused("1D array", texture_1d(256, 1, 2, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100));
}

// The linear mode by hand: a 100x10 R16G16B16A16 texture with three mips. Each mip's pitch is its
// rounded-up width aligned to 32 elements (256 bytes), and the mips are stored from the smallest up:
// mip 2 is 32x3 elements at 0, mip 1 64x5 at 768, mip 0 128x10 at 3328, 13568 bytes a slice.
// The mutated run expects the largest mip first and is a negative control.
void check_linear(bool largest_first)
{
    const Layout l(texture(100, 10, 3, 2, DXGI_FORMAT_R16G16B16A16_FLOAT, 0x100));
    const std::uint64_t start[3] = {largest_first ? 0u : 3328u, largest_first ? 10240u : 768u, largest_first ? 12800u : 0u};
    const UINT pitch[3] = {128, 64, 32};
    if (!l.is_linear || l.first_tail != 3 || l.slice_size != 13568 || l.bytes() != 2 * 13568) fail("linear: slice size");
    for (UINT mip = 0; mip < 3; ++mip) {
        if (l.start[mip] != start[mip] || l.row_pitch[mip] != pitch[mip]) fail("linear: mip " + std::to_string(mip));
        const std::uint64_t want = 13568 + start[mip] + (2ull * pitch[mip] + 3) * 8;
        if (l.address(mip, 3, 2, 1) != want) fail("linear: element (3, 2) of slice 1, mip " + std::to_string(mip));
    }
}

// Which slot-70 descriptions are taken, and how their texels are read (address_texture.h).
void check_address_textures()
{
    const auto expect = [](const char* name, const D3D12_RESOURCE_DESC& d, bool want) {
        if (x2w_startup::console_address_texture(d) != want) fail(std::string("address texture: ") + name);
    };
    D3D12_RESOURCE_DESC d = texture(1024, 2112, 1, 1, DXGI_FORMAT_BC1_UNORM, 0x105);
    expect("4 KiB BC1", d, true);
    d.Flags = static_cast<D3D12_RESOURCE_FLAGS>(0x300000);
    expect("console-only flag bits", d, true);
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    expect("render target", d, false);
    d.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    expect("not sampled", d, false);
    expect("64 KiB display R16 chain", texture(512, 512, 10, 1, DXGI_FORMAT_R16_UNORM, 0x10a), true);
    expect("desktop layout RGBA8", texture(1, 1, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0), false);
    expect("linear mode R8", texture(256, 128, 1, 1, DXGI_FORMAT_R8_UNORM, 0x100), true);
    expect("linear 1D RGBA8 sRGB", texture_1d(512, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0x100), true);
    expect("linear 1D array", texture_1d(512, 1, 4, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100), false);
    expect("desktop layout 1D", texture_1d(512, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0), false);
    expect("XOR mode", texture(256, 256, 1, 1, DXGI_FORMAT_BC7_UNORM, 0x119), false);
    expect("row-major desktop layout", texture(256, 256, 1, 1, DXGI_FORMAT_BC7_UNORM, 1), false);
    expect("unknown element size", texture(256, 256, 1, 1, DXGI_FORMAT_R32G32B32_FLOAT, 0), false);
    d = texture(256, 256, 1, 1, DXGI_FORMAT_BC7_UNORM, 0x105);
    d.SampleDesc.Count = 2;
    expect("multisampled", d, false);
    d = texture(256, 256, 1, 1, DXGI_FORMAT_BC7_UNORM, 0x105);
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    expect("volume", d, false);
    expect("BC1 width not whole blocks", texture(126, 128, 1, 1, DXGI_FORMAT_BC1_UNORM, 0x105), false);
    expect("BC1 height not whole blocks", texture(128, 126, 1, 1, DXGI_FORMAT_BC1_UNORM, 0x10a), false);
    expect("RGBA8 odd sides", texture(257, 129, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x106), true);

    expect("combined depth and stencil", texture(64, 64, 1, 1, DXGI_FORMAT_R24G8_TYPELESS, 0x105), false);
    expect("typed depth", texture(64, 64, 1, 1, DXGI_FORMAT_D32_FLOAT, 0x105), false);
    expect("mip count exceeds dimensions", texture(4, 4, 4, 1, DXGI_FORMAT_BC7_UNORM, 0x109), false);
    expect("too many array slices", texture(16, 16, 1, 2049, DXGI_FORMAT_R8_UNORM, 0x105), false);

    D3D12_RESOURCE_DESC chain = texture(512, 512, 10, 1, DXGI_FORMAT_BC7_UNORM, 0x109);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_BC7_UNORM;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 10;
    const auto fits = [&](const char* name, bool want) {
        if (x2w_startup::address_view_fits(chain, v) != want) fail(std::string("address view: ") + name);
    };
    fits("whole chain", true);
    v.Texture2D.MipLevels = ~0u; fits("all mips", true);
    v.Format = DXGI_FORMAT_BC7_UNORM_SRGB; fits("sRGB view of typed UNORM storage", false);
    chain.Format = DXGI_FORMAT_BC7_TYPELESS; fits("sRGB view of typeless storage", true);
    v.Format = DXGI_FORMAT_BC7_TYPELESS; fits("typeless view", false);
    v.Format = DXGI_FORMAT_BC3_UNORM; fits("another family", false);
    chain.Format = v.Format = DXGI_FORMAT_BC7_UNORM;
    v.Shader4ComponentMapping = 0; fits("missing mapping validity bit", false);
    v.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 7);
    fits("reserved component mapping", false);
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 11; fits("too many mips", false);
    v.Texture2D.MipLevels = 0; fits("no mips", false);
    v.Texture2D.MipLevels = 1; v.Texture2D.MostDetailedMip = 10; fits("first mip past the chain", false);
    v.Texture2D.MostDetailedMip = 9; fits("last mip", true);
    v.Texture2D.PlaneSlice = 1; fits("second plane", false);
    v.Texture2D.PlaneSlice = 0;
    v.Texture2D.ResourceMinLODClamp = 2.0f; fits("explicit LOD clamp", false);
    v.Texture2D.ResourceMinLODClamp = 0.0f;
    chain.DepthOrArraySize = 4; fits("non-array view of an array", false);
    v = {};
    v.Format = DXGI_FORMAT_BC7_UNORM;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2DArray.MipLevels = 10;
    v.Texture2DArray.ArraySize = 4; fits("whole array", true);
    v.Texture2DArray.FirstArraySlice = 2; fits("array extends past the last slice", false);
    v.Texture2DArray.ArraySize = 2; fits("last two slices", true);
    v.Texture2DArray.ArraySize = ~0u; fits("unspecified array size", false);
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE; fits("cube view", false);
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    v.Texture3D.MipLevels = 10;
    fits("3D view on 2D chain", false);

    auto cubes = texture(2, 2, 1, 12, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    v = {};
    v.Format = cubes.Format;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
    v.TextureCubeArray = {0, 1, 6, 1, 0};
    if (!x2w_startup::desktop_address_array(cubes) || x2w_startup::address_view_fits(cubes, v) ||
        !x2w_startup::address_view_fits(cubes, v, true))
        fail("opt-in view of the second cube");
    v.TextureCubeArray.NumCubes = ~0u;
    if (x2w_startup::address_view_fits(cubes, v, true)) fail("overflowing cube count");
    v.TextureCubeArray.NumCubes = 1;
    v.TextureCubeArray.First2DArrayFace = 7;
    if (x2w_startup::address_view_fits(cubes, v, true)) fail("cube array extends past the last face");
    v.TextureCubeArray.First2DArrayFace = 6;
    cubes.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    if (!x2w_startup::console_address_texture(cubes) || !x2w_startup::address_view_fits(cubes, v, true))
        fail("console-swizzled view of the second cube");

    D3D12_RESOURCE_DESC vol = texture(32, 32, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    vol.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    vol.DepthOrArraySize = 16;
    v = {};
    v.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Texture2D.MipLevels = 4;
    if (x2w_startup::address_view_fits(vol, v)) fail("2D view on 3D volume");
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    v.Texture3D.MipLevels = 4;
    if (!x2w_startup::address_view_fits(vol, v)) fail("whole 3D volume view");
    v.Texture3D.MostDetailedMip = 2;
    v.Texture3D.MipLevels = 2;
    if (!x2w_startup::address_view_fits(vol, v)) fail("sub-mip 3D volume view");
    v.Texture3D.MipLevels = 3;
    if (x2w_startup::address_view_fits(vol, v)) fail("3D volume view past mip count");
    v.Texture3D.MostDetailedMip = 0;
    v.Texture3D.MipLevels = 4;
    v.Texture3D.ResourceMinLODClamp = 1.0f;
    if (x2w_startup::address_view_fits(vol, v)) fail("3D volume view explicit LOD clamp");

    const D3D12_RESOURCE_DESC row = texture_1d(300, 9, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100);
    v = {};
    v.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
    v.Texture1D.MipLevels = 9;
    if (!x2w_startup::address_view_fits(row, v)) fail("whole 1D chain view");
    v.Texture1D.MostDetailedMip = 8;
    v.Texture1D.MipLevels = 1;
    if (!x2w_startup::address_view_fits(row, v)) fail("last 1D mip view");
    v.Texture1D.MipLevels = 2;
    if (x2w_startup::address_view_fits(row, v)) fail("1D view past the chain");
    v.Texture1D.MostDetailedMip = 0;
    v.Texture1D.MipLevels = 1;
    v.Texture1D.ResourceMinLODClamp = 1.0f;
    if (x2w_startup::address_view_fits(row, v)) fail("1D view explicit LOD clamp");
    v.Texture1D.ResourceMinLODClamp = 0.0f;
    if (x2w_startup::address_view_fits(texture(300, 1, 9, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100), v))
        fail("1D view on a 2D texture");
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Texture2D.MipLevels = 1;
    if (x2w_startup::address_view_fits(row, v)) fail("2D view on a 1D texture");

    const D3D12_RESOURCE_DESC faces = texture(64, 64, 4, 6, DXGI_FORMAT_BC1_UNORM, 0x105);
    v = {};
    v.Format = DXGI_FORMAT_BC1_UNORM;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    v.TextureCube.MipLevels = 4;
    if (!x2w_startup::address_view_fits(faces, v)) fail("whole cube chain view");
    v.TextureCube.MostDetailedMip = 3;
    v.TextureCube.MipLevels = 1;
    if (!x2w_startup::address_view_fits(faces, v)) fail("last cube mip view");
    v.TextureCube.MipLevels = 2;
    if (x2w_startup::address_view_fits(faces, v)) fail("cube view past the chain");
    v.TextureCube.MostDetailedMip = 0;
    v.TextureCube.MipLevels = 1;
    v.TextureCube.ResourceMinLODClamp = 1.0f;
    if (x2w_startup::address_view_fits(faces, v)) fail("cube view explicit LOD clamp");
    v.TextureCube.ResourceMinLODClamp = 0.0f;
    if (x2w_startup::address_view_fits(texture(64, 64, 4, 1, DXGI_FORMAT_BC1_UNORM, 0x105), v))
        fail("cube view on a single slice");
    if (x2w_startup::address_view_fits(texture(64, 64, 4, 12, DXGI_FORMAT_BC1_UNORM, 0x105), v))
        fail("cube view on an array of twelve");
}
} // namespace

#ifndef XBOX2WINDOWS_HAVE_ADDRLIB
int addrlib_linear_check()
{
    std::puts("startup_gfx10_layout --addrlib: SKIP (built without addrlib; run tools/fetch_addrlib.py and configure with "
              "-DXBOX2WINDOWS_ADDRLIB_DIR=<mesa>/src/amd/addrlib)");
    return 77;
}
#else
#include "addrinterface.h"

namespace {
VOID* ADDR_API alloc_sys_mem(const ADDR_ALLOCSYSMEM_INPUT* in) { return std::malloc(in->sizeInBytes); }
ADDR_E_RETURNCODE ADDR_API free_sys_mem(const ADDR_FREESYSMEM_INPUT* in)
{
    std::free(in->pVirtAddr);
    return ADDR_OK;
}
} // namespace

// Every element of every subresource of several linear textures, and their sizes, against
// Addr2ComputeSurfaceAddrFromCoord and Addr2ComputeSurfaceInfo with ADDR_SW_LINEAR on Mesa's "navi21"
// (gfx10.3) configuration, the one xg_swizzle_select_check uses: family 0x8F, revision 0x28,
// GB_ADDR_CONFIG 0x444. Sizes are given to addrlib in elements, so a BC texture is its 4x4 blocks.
int addrlib_linear_check()
{
    ADDR_CREATE_INPUT create{};
    ADDR_CREATE_OUTPUT created{};
    create.size = sizeof create;
    created.size = sizeof created;
    create.chipEngine = 0x0000000D; // CIASICIDGFXENGINE_ARCTICISLAND
    create.chipFamily = 0x8F;
    create.chipRevision = 0x28;
    create.callbacks.allocSysMem = alloc_sys_mem;
    create.callbacks.freeSysMem = free_sys_mem;
    create.regValue.gbAddrConfig = 0x40 | 4u | (4u << 8);
    if (AddrCreate(&create, &created) != ADDR_OK) return 1;
    struct Shape { UINT width, height, mips, slices; DXGI_FORMAT format; bool one_d = false; };
    const Shape shapes[] = {
        {1, 1, 1, 1, DXGI_FORMAT_R8_UNORM},              {300, 7, 1, 1, DXGI_FORMAT_R8_UNORM},
        {257, 129, 9, 2, DXGI_FORMAT_R8_UNORM},          {100, 10, 3, 2, DXGI_FORMAT_R16G16B16A16_FLOAT},
        {1920, 64, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT}, {256, 256, 7, 1, DXGI_FORMAT_R32_FLOAT},
        {33, 17, 6, 3, DXGI_FORMAT_R16_UNORM},           {200, 100, 8, 1, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB},
        {48, 40, 4, 2, DXGI_FORMAT_R32G32B32A32_FLOAT},  {64, 64, 4, 3, DXGI_FORMAT_BC1_UNORM},
        {128, 64, 6, 1, DXGI_FORMAT_BC7_UNORM},
        {300, 1, 9, 1, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true}, {4096, 1, 13, 1, DXGI_FORMAT_R8_UNORM, true},
        {1000, 1, 10, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, true},
    };
    unsigned compared = 0, mismatches = 0;
    for (const Shape& s : shapes) {
        const D3D12_RESOURCE_DESC d = s.one_d ? texture_1d(s.width, s.mips, s.slices, s.format, 0x100)
                                              : texture(s.width, s.height, s.mips, s.slices, s.format, 0x100);
        const AddrResourceType type = s.one_d ? ADDR_RSRC_TEX_1D : ADDR_RSRC_TEX_2D;
        const std::string name = std::string(s.one_d ? "1D " : "") + std::to_string(s.width) + "x" + std::to_string(s.height) + " format " +
                                 std::to_string(s.format) + ", " + std::to_string(s.mips) + " mips";
        if (!x2w_startup::gfx10::supports(d)) {
            fail(name + ": refused");
            ++mismatches;
            continue;
        }
        const Layout l(d);
        ADDR2_COMPUTE_SURFACE_INFO_INPUT info_in{};
        ADDR2_COMPUTE_SURFACE_INFO_OUTPUT info_out{};
        info_in.size = sizeof info_in;
        info_out.size = sizeof info_out;
        info_in.flags.texture = 1;
        info_in.swizzleMode = ADDR_SW_LINEAR;
        info_in.resourceType = type;
        info_in.bpp = l.bytes_per_element * 8;
        info_in.width = (s.width + l.texels - 1) / l.texels;
        info_in.height = (s.height + l.texels - 1) / l.texels;
        info_in.numSlices = s.slices;
        info_in.numMipLevels = s.mips;
        info_in.numSamples = 1;
        info_in.numFrags = 1;
        ++compared;
        if (Addr2ComputeSurfaceInfo(created.hLib, &info_in, &info_out) != ADDR_OK || info_out.surfSize != l.bytes() ||
            info_out.sliceSize != l.slice_size) {
            fail(name + ": size " + std::to_string(l.bytes()) + ", addrlib " + std::to_string(info_out.surfSize));
            ++mismatches;
            continue;
        }
        ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT in{};
        in.size = sizeof in;
        in.flags.texture = 1;
        in.swizzleMode = ADDR_SW_LINEAR;
        in.resourceType = type;
        in.bpp = info_in.bpp;
        in.unalignedWidth = info_in.width;
        in.unalignedHeight = info_in.height;
        in.numSlices = s.slices;
        in.numMipLevels = s.mips;
        in.numSamples = 1;
        in.numFrags = 1;
        unsigned wrong = 0;
        for (in.slice = 0; in.slice < s.slices; ++in.slice)
            for (in.mipId = 0; in.mipId < s.mips; ++in.mipId)
                for (in.y = 0; in.y < l.rows(in.mipId); ++in.y)
                    for (in.x = 0; in.x < l.columns(in.mipId); ++in.x) {
                        ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT out{};
                        out.size = sizeof out;
                        ++compared;
                        const std::uint64_t mine = l.address(in.mipId, in.x, in.y, in.slice);
                        if (Addr2ComputeSurfaceAddrFromCoord(created.hLib, &in, &out) != ADDR_OK || out.addr != mine) {
                            if (!wrong++)
                                fail(name + ": mip " + std::to_string(in.mipId) + " slice " + std::to_string(in.slice) + " (" +
                                     std::to_string(in.x) + ", " + std::to_string(in.y) + ") at " + std::to_string(mine) +
                                     ", addrlib " + std::to_string(out.addr));
                        }
                    }
        mismatches += wrong;
    }
    AddrDestroy(created.hLib);
    std::printf("startup_gfx10_layout --addrlib: %u linear sizes and addresses compared, %u mismatches (Mesa 25.1.0 addrlib, "
                "SW_LINEAR)\n", compared, mismatches);
    return compared && !mismatches ? 0 : 1;
}
#endif

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--addrlib") return addrlib_linear_check();
    const bool mutate = argc == 3 && std::string(argv[1]) == "--mutate" && std::string(argv[2]) == "display";
    const bool mutate_linear = argc == 3 && std::string(argv[1]) == "--mutate" && std::string(argv[2]) == "linear-largest-first";
    if (argc != 1 && !mutate && !mutate_linear) {
        std::printf("usage: startup_gfx10_layout_check [--addrlib | --mutate display | --mutate linear-largest-first]\n");
        return 2;
    }
    check_equations(mutate);
    check_linear(mutate_linear);
    check_bc7_4k();
    check_bc64();
    check_tile64();
    check_distinct_and_round_trip();
    check_refusals();
    check_address_textures();
    if (failures) {
        std::printf("startup_gfx10_layout: %d FAILED\n", failures);
        return 1;
    }
    std::printf("startup_gfx10_layout: PASS (30 equations, 5 4 KiB surfaces, 8 64 KiB chains, 10 tile plans, 15 round trips, 1 linear chain, 15 refusals, 22 address descriptions, 39 views)\n");
    return 0;
}
