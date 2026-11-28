// Checks the volume layout of src/startup_stubs/gfx10_layout.h (a 3D texture in the 4 KiB standard mode,
// 0x105, which addrlib lays out in thick blocks) without a device:
// - the address bit of every x, y and z bit against the one-pipe rows of addrlib's GFX10_SW_4K_S3_PATINFO,
//   restated below, and the block each element size gives;
// - the order of blocks: along x, then y, then a layer of blocks at a time along z, with the sides of
//   the texture padded to whole blocks;
// - a mip chain: each layer of blocks holds the tail block, then the mips from the smallest to the largest,
//   and the tail mips sit at their origins inside the tail block;
// - that every element of a volume gets its own bytes inside the texture, and that a tile followed by a
//   detile (with padded row and layer pitches) returns the texels it started from, for every mip;
// - the descriptions it takes and refuses, and the slot-70 views it allows on them.
// `--addrlib` compares the size and every element address of a set of volumes with
// Addr2ComputeSurfaceInfo and Addr2ComputeSurfaceAddrFromCoord (ADDR_RSRC_TEX_3D, ADDR_SW_4KB_S) and
// skips (77) when the build has no addrlib. `--mutate thin` lays each volume out as a thin 2D array of
// its depth layers, and `--mutate tail` puts every mip of the tail at the start of the tail block; both
// are negative controls and must fail.
//
// Reference: AMD addrlib as Mesa carries it under src/amd/addrlib, at the revision tools/fetch_addrlib.py
// pins (Mesa 25.1.0, commit 5c142e46f3f6e752ed745fd48912ebb8fad67145): src/gfx10/gfx10SwizzlePattern.h,
// src/gfx10/gfx10addrlib.cpp and src/core/addrlib2.cpp, each with the notice
//   Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT
// The pattern rows are restated below as data; no addrlib code is included in the default run.
#include "startup_stubs/address_texture.h"
#include "startup_stubs/gfx10_layout.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
using x2w_startup::gfx10::Layout;
int failures = 0;
bool thin = false;      // --mutate thin
bool flat_tail = false; // --mutate tail

void fail(const std::string& what)
{
    if (++failures <= 40) std::printf("FAIL %s\n", what.c_str());
}

D3D12_RESOURCE_DESC volume(UINT width, UINT height, UINT depth, DXGI_FORMAT format, UINT layout = 0x105, UINT mips = 1)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = static_cast<UINT16>(depth);
    d.MipLevels = static_cast<UINT16>(mips);
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(layout);
    return d;
}

// The layout under test; the controls read the volume as a thin array of its layers, or move every tail
// mip to the start of the tail block.
Layout layout_of(D3D12_RESOURCE_DESC d)
{
    if (thin) d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    Layout l(d);
    for (UINT mip = l.first_tail; flat_tail && mip < l.mips; ++mip) l.origin_x[mip] = l.origin_y[mip] = 0;
    return l;
}
std::uint64_t at(const Layout& l, UINT x, UINT y, UINT z, UINT mip = 0)
{
    return thin ? l.address(mip, x, y, z) : l.address(mip, x, y, 0, z);
}

// One format per element size: 1, 2, 4, 8 and 16 bytes.
constexpr DXGI_FORMAT by_size[5] = {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                                    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};

// GFX10_SW_4K_S3_PATINFO one-pipe rows by element size, address bits from bit 0 up; "0" is a bit inside
// the element.
const char* const rows_s3[5] = {
    "X0 X1 Z0 Y0 Z1 Y1 X2 Z2 Y2 X3 Z3 Y3", "0 X0 Z0 Y0 Z1 Y1 X1 Z2 Y2 X2 Z3 Y3", "0 0 X0 Y0 Z0 Y1 X1 Z1 Y2 X2 Z2 Y3",
    "0 0 0 X0 Z0 Y0 X1 Z1 Y1 X2 Z2 Y2", "0 0 0 0 Z0 Y0 X0 Z1 Y1 X1 Z2 Y2"};
// addrlib Block1K_3d, scaled to 4 KiB by ComputeThickBlockDimension (height and depth doubled).
constexpr UINT block_sides[5][3] = {{16, 16, 16}, {8, 16, 16}, {8, 16, 8}, {8, 8, 8}, {4, 8, 8}};

void check_equations()
{
    for (UINT size = 0; size < 5; ++size) {
        const Layout l = layout_of(volume(64, 64, 64, by_size[size]));
        const std::string name = "element " + std::to_string(1u << size);
        UINT bit = 0;
        for (const char* p = rows_s3[size]; *p; ++bit) {
            const char axis = *p++;
            UINT index = 0;
            if (axis != '0') index = static_cast<UINT>(*p++ - '0');
            if (*p == ' ') ++p;
            if (axis == '0') continue;
            const UINT v = 1u << index;
            const std::uint64_t got = axis == 'X' ? at(l, v, 0, 0) : axis == 'Y' ? at(l, 0, v, 0) : at(l, 0, 0, v);
            if (got != (1ull << bit))
                fail(name + " " + axis + std::to_string(index) + ": " + std::to_string(got) + ", expected bit " + std::to_string(bit));
        }
        if (l.block_bytes != 4096 || l.block_width != block_sides[size][0] || l.block_height != block_sides[size][1] ||
            l.block_depth != block_sides[size][2])
            fail(name + ": block " + std::to_string(l.block_width) + "x" + std::to_string(l.block_height) + "x" +
                 std::to_string(l.block_depth));
    }
}

// 17x17x17 RGBA8 in 8x16x8 blocks: 3 blocks across, 2 down and 3 layers of blocks.
void check_block_order()
{
    const Layout l = layout_of(volume(17, 17, 17, DXGI_FORMAT_R8G8B8A8_UNORM));
    const auto expect = [](const char* what, std::uint64_t got, std::uint64_t want) {
        if (got != want) fail(std::string("block order: ") + what + " " + std::to_string(got) + ", expected " + std::to_string(want));
    };
    expect("bytes", l.bytes(), 3ull * 2 * 3 * 4096);
    expect("next block in x", at(l, 8, 0, 0), 4096);
    expect("next row of blocks", at(l, 0, 16, 0), 3 * 4096);
    expect("next layer of blocks", at(l, 0, 0, 8), 6 * 4096);
    expect("last element", at(l, 16, 16, 16), (2ull * 6 + 1 * 3 + 2) * 4096);
}

// 32x32x32 RGBA8 with six mips, in 8x16x8 blocks. Mips 2 to 5 (8x8 and smaller, at most five from the
// end) share the tail block. A layer of blocks holds the tail block, mip 1 (2x1 blocks) and mip 0 (4x2
// blocks): 11 blocks. The tail mips sit at byte (k << 8) of the tail block for k = 4, 3, 2, 1, read as
// 4x4x4 micro blocks: (0,8), (4,4), (4,0) and (0,4).
void check_mip_chain()
{
    const Layout l = layout_of(volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0x105, 6));
    const auto expect = [](const char* what, std::uint64_t got, std::uint64_t want) {
        if (got != want) fail(std::string("mip chain: ") + what + " " + std::to_string(got) + ", expected " + std::to_string(want));
    };
    expect("bytes", l.bytes(), 4ull * 11 * 4096);
    expect("mip 0", at(l, 0, 0, 0, 0), 3 * 4096);
    expect("mip 0 next row of blocks", at(l, 0, 16, 0, 0), 7 * 4096);
    expect("mip 0 next layer of blocks", at(l, 0, 0, 8, 0), 14 * 4096);
    expect("mip 1", at(l, 0, 0, 0, 1), 4096);
    expect("mip 1 next block in x", at(l, 8, 0, 0, 1), 2 * 4096);
    expect("mip 1 next layer of blocks", at(l, 0, 0, 8, 1), 12 * 4096);
    expect("mip 2 at (0,8)", at(l, 0, 0, 0, 2), 0x800); // y 8: Y3, bit 11
    expect("mip 3 at (4,4)", at(l, 0, 0, 0, 3), 0x300); // x 4: X2, bit 9; y 4: Y2, bit 8
    expect("mip 4 at (4,0)", at(l, 0, 0, 0, 4), 0x200);
    expect("mip 5 at (0,4)", at(l, 0, 0, 0, 5), 0x100);
    expect("mip 2 next layer of blocks", at(l, 0, 0, 8, 2), 11 * 4096 + 0x800);
}

struct Shape {
    UINT width, height, depth;
    DXGI_FORMAT format;
    UINT mips = 1;
};
const Shape shapes[] = {
    {17, 17, 17, DXGI_FORMAT_R16G16B16A16_FLOAT}, {32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM}, {33, 33, 33, DXGI_FORMAT_R16G16B16A16_FLOAT},
    {32, 32, 32, DXGI_FORMAT_A8_UNORM},           {16, 16, 8, DXGI_FORMAT_R16_FLOAT},       {16, 16, 16, DXGI_FORMAT_R8G8_UNORM},
    {32, 32, 32, DXGI_FORMAT_R10G10B10A2_UNORM},  {5, 3, 7, DXGI_FORMAT_R32G32B32A32_FLOAT}, {1, 1, 2, DXGI_FORMAT_R8_UNORM},
    {64, 8, 40, DXGI_FORMAT_R32_FLOAT},
    // Mip chains: whole and partial, a tail that starts at mip 0, a chain with no tail, odd sides.
    {32, 32, 32, DXGI_FORMAT_R8_UNORM, 6},           {32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 6},
    {64, 64, 32, DXGI_FORMAT_R16G16_FLOAT, 7},       {128, 128, 128, DXGI_FORMAT_R16G16B16A16_FLOAT, 8},
    {16, 16, 16, DXGI_FORMAT_R32G32B32A32_FLOAT, 5}, {8, 4, 8, DXGI_FORMAT_R8G8B8A8_UNORM, 4},
    {64, 64, 64, DXGI_FORMAT_R16_FLOAT, 2},          {33, 17, 9, DXGI_FORMAT_R16G16B16A16_FLOAT, 6},
    {4, 4, 64, DXGI_FORMAT_R8_UNORM, 7},             {256, 32, 8, DXGI_FORMAT_R16_FLOAT, 9},
};

void check_distinct_and_round_trip()
{
    for (const Shape& s : shapes) {
        const D3D12_RESOURCE_DESC d = volume(s.width, s.height, s.depth, s.format, 0x105, s.mips);
        const std::string name = std::to_string(s.width) + "x" + std::to_string(s.height) + "x" + std::to_string(s.depth) +
                                 " format " + std::to_string(s.format) + " mips " + std::to_string(s.mips);
        if (!x2w_startup::gfx10::supports(d)) {
            fail(name + ": refused");
            continue;
        }
        const Layout l = layout_of(d);
        const UINT e = l.bytes_per_element;
        const std::uint64_t size = l.bytes();
        std::vector<unsigned char> used(static_cast<std::size_t>(size));
        // Every element of every mip has its own bytes.
        bool clash = false;
        for (UINT mip = 0; mip < s.mips; ++mip)
            for (UINT z = 0; z < l.layers(mip); ++z)
                for (UINT y = 0; y < l.rows(mip); ++y)
                    for (UINT x = 0; x < l.columns(mip); ++x) {
                        const std::uint64_t a = at(l, x, y, z, mip);
                        if (a % e || a + e > size) {
                            clash = true;
                            continue;
                        }
                        if (used[static_cast<std::size_t>(a)]) clash = true;
                        used[static_cast<std::size_t>(a)] = 1;
                    }
        if (clash) fail(name + ": elements share bytes or fall outside the texture");
        if (thin) continue;
        std::vector<unsigned char> tiled(static_cast<std::size_t>(size));
        for (UINT mip = 0; mip < s.mips; ++mip) {
            const UINT w = l.columns(mip), h = l.rows(mip), n = l.layers(mip);
            const std::size_t pitch = (std::size_t{w} * e + 255) & ~std::size_t{255}, layer = pitch * (h + 1);
            std::vector<unsigned char> rows(layer * n), back(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<unsigned char>(i * 131 + 7 + mip);
            l.tile(mip, 0, tiled.data(), rows.data(), pitch, layer);
            l.detile(mip, 0, tiled.data(), back.data(), pitch, layer);
            for (UINT z = 0; z < n; ++z)
                for (UINT y = 0; y < h; ++y)
                    if (std::memcmp(&rows[z * layer + y * pitch], &back[z * layer + y * pitch], std::size_t{w} * e)) {
                        fail(name + ": round trip differs in mip " + std::to_string(mip) + " at layer " + std::to_string(z) +
                             " row " + std::to_string(y));
                        z = n;
                        break;
                    }
        }
    }
}

void check_descriptions()
{
    const auto supported = [](const char* name, const D3D12_RESOURCE_DESC& d, bool want) {
        if (x2w_startup::gfx10::supports(d) != want) fail(std::string("layout: ") + name);
    };
    const D3D12_RESOURCE_DESC good = volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM);
    supported("4 KiB standard RGBA8", good, true);
    D3D12_RESOURCE_DESC d = good;
    d.MipLevels = 6;
    supported("a whole mip chain", d, true);
    d.MipLevels = 16;
    supported("more mips than a layout holds", d, false);
    d.MipLevels = 0;
    supported("no mips", d, false);
    supported("BC1", volume(32, 32, 32, DXGI_FORMAT_BC1_UNORM), false);
    supported("BC7", volume(32, 32, 32, DXGI_FORMAT_BC7_UNORM), false);
    supported("4 KiB display", volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0x106), false);
    supported("256 B standard", volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0x101), false);
    supported("64 KiB standard", volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0x109), false);
    supported("64 KiB display", volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0x10a), false);
    supported("linear", volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0x100), false);
    supported("desktop layout", volume(32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM, 0), false);
    supported("12-byte texels", volume(32, 32, 32, DXGI_FORMAT_R32G32B32_FLOAT), false);
    supported("too wide", volume(2049, 4, 4, DXGI_FORMAT_R8_UNORM), false);
    supported("too deep", volume(4, 4, 2049, DXGI_FORMAT_R8_UNORM), false);
    supported("no depth", volume(4, 4, 0, DXGI_FORMAT_R8_UNORM), false);
    d = good;
    d.SampleDesc.Count = 2;
    supported("multisampled", d, false);

    const auto address = [](const char* name, const D3D12_RESOURCE_DESC& d, bool want) {
        if (x2w_startup::console_address_texture(d) != want) fail(std::string("address texture: ") + name);
    };
    address("RGBA16F 17x17x17", volume(17, 17, 17, DXGI_FORMAT_R16G16B16A16_FLOAT), true);
    address("R8 32x32x32", volume(32, 32, 32, DXGI_FORMAT_R8_UNORM), true);
    address("typed depth", volume(32, 32, 32, DXGI_FORMAT_D32_FLOAT), false);
    d = good;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    address("unordered access", d, false);
    address("BC7", volume(32, 32, 32, DXGI_FORMAT_BC7_UNORM), false);
    d = good;
    d.MipLevels = 6;
    address("mip chain", d, true);
    d.MipLevels = 7;
    address("more mips than the sides allow", d, false);
    address("mips the depth allows", volume(4, 4, 64, DXGI_FORMAT_R8_UNORM, 0x105, 7), true);

    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    v.Texture3D.MipLevels = ~0u;
    if (!x2w_startup::address_view_fits(good, v)) fail("view: whole volume");
    v.Texture3D.MipLevels = 1;
    if (!x2w_startup::address_view_fits(good, v)) fail("view: one mip");
    v.Texture3D.MostDetailedMip = 1;
    if (x2w_startup::address_view_fits(good, v)) fail("view: mip past the volume");
    D3D12_RESOURCE_DESC chain = good;
    chain.MipLevels = 6;
    v.Texture3D.MostDetailedMip = 2;
    v.Texture3D.MipLevels = 4;
    if (!x2w_startup::address_view_fits(chain, v)) fail("view: the last four mips of a chain");
    v.Texture3D.MipLevels = 5;
    if (x2w_startup::address_view_fits(chain, v)) fail("view: mips past the chain");
    v.Texture3D.MipLevels = 1;
    v.Texture3D.MostDetailedMip = 0;
    D3D12_RESOURCE_DESC typeless = good;
    typeless.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    v.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    if (!x2w_startup::address_view_fits(typeless, v)) fail("view: sRGB of typeless storage");
    if (x2w_startup::address_view_fits(good, v)) fail("view: sRGB of UNORM storage");
    v.Format = DXGI_FORMAT_R32_FLOAT;
    if (x2w_startup::address_view_fits(good, v)) fail("view: other family");
    v.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    v.Texture2DArray = {0, ~0u, 0, 32, 0, 0.0f};
    if (x2w_startup::address_view_fits(good, v)) fail("view: 2D array on a volume");
}
} // namespace

#ifndef XBOX2WINDOWS_HAVE_ADDRLIB
int addrlib_check()
{
    std::puts("startup_gfx10_volume --addrlib: SKIP (built without addrlib; run tools/fetch_addrlib.py and configure with "
              "-DXBOX2WINDOWS_ADDRLIB_DIR=<mesa>/src/amd/addrlib)");
    return 77;
}
#else
#include <cstdlib>

#include "addrinterface.h"

namespace {
VOID* ADDR_API alloc_sys_mem(const ADDR_ALLOCSYSMEM_INPUT* in) { return std::malloc(in->sizeInBytes); }
ADDR_E_RETURNCODE ADDR_API free_sys_mem(const ADDR_FREESYSMEM_INPUT* in)
{
    std::free(in->pVirtAddr);
    return ADDR_OK;
}
} // namespace

// The shapes above and a sweep of small odd sides, for each element size, against Mesa's "navi21"
// (gfx10.3) configuration that startup_small_texture_4k_check and xg_swizzle_select_check use: family
// 0x8F, revision 0x28, GB_ADDR_CONFIG 0x444.
int addrlib_check()
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
    const AddrFormat addr_formats[5] = {ADDR_FMT_8, ADDR_FMT_16, ADDR_FMT_8_8_8_8, ADDR_FMT_16_16_16_16, ADDR_FMT_32_32_32_32};
    std::vector<D3D12_RESOURCE_DESC> cases;
    for (const Shape& s : shapes) cases.push_back(volume(s.width, s.height, s.depth, s.format, 0x105, s.mips));
    for (UINT size = 0; size < 5; ++size)
        for (UINT w : {1u, 3u, 9u, 24u})
            for (UINT h : {1u, 7u, 17u})
                for (UINT dep : {1u, 5u, 18u}) cases.push_back(volume(w, h, dep, by_size[size]));
    // Every mip count of a set of chains, so that each mip in turn is the first in the tail or the last
    // outside it.
    constexpr UINT chains[4][3] = {{64, 64, 64}, {40, 24, 12}, {17, 33, 5}, {8, 8, 32}};
    for (UINT size = 0; size < 5; ++size)
        for (const auto& side : chains)
            for (UINT mips = 2; ((std::max)({side[0], side[1], side[2]}) >> (mips - 1)) != 0; ++mips)
                cases.push_back(volume(side[0], side[1], side[2], by_size[size], 0x105, mips));
    unsigned compared = 0, elements = 0, mismatches = 0;
    for (const D3D12_RESOURCE_DESC& d : cases) {
        const Layout l(d);
        UINT size = 0;
        while ((1u << size) < l.bytes_per_element) ++size;
        ADDR2_COMPUTE_SURFACE_INFO_INPUT in{};
        ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out{};
        in.size = sizeof in;
        out.size = sizeof out;
        in.flags.texture = 1;
        in.swizzleMode = ADDR_SW_4KB_S;
        in.resourceType = ADDR_RSRC_TEX_3D;
        in.format = addr_formats[size];
        in.bpp = l.bytes_per_element * 8;
        in.width = l.width;
        in.height = l.height;
        in.numSlices = l.depth;
        in.numMipLevels = l.mips;
        in.numSamples = 1;
        in.numFrags = 1;
        ++compared;
        char name[64];
        std::snprintf(name, sizeof name, "%ux%ux%u %u-byte %u mips", l.width, l.height, l.depth, l.bytes_per_element, l.mips);
        if (Addr2ComputeSurfaceInfo(created.hLib, &in, &out) != ADDR_OK || out.surfSize != l.bytes()) {
            if (++mismatches <= 12)
                std::printf("FAIL %s: size 0x%llx, addrlib 0x%llx\n", name, static_cast<unsigned long long>(l.bytes()),
                            static_cast<unsigned long long>(out.surfSize));
            continue;
        }
        for (UINT mip = 0; mip < l.mips; ++mip)
            for (UINT z = 0; z < l.layers(mip); ++z)
                for (UINT y = 0; y < l.rows(mip); ++y)
                    for (UINT x = 0; x < l.columns(mip); ++x) {
                        ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT ain{};
                        ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT aout{};
                        ain.size = sizeof ain;
                        aout.size = sizeof aout;
                        ain.x = x;
                        ain.y = y;
                        ain.slice = z;
                        ain.mipId = mip;
                        ain.swizzleMode = ADDR_SW_4KB_S;
                        ain.flags.texture = 1;
                        ain.resourceType = ADDR_RSRC_TEX_3D;
                        ain.bpp = in.bpp;
                        ain.unalignedWidth = l.width;
                        ain.unalignedHeight = l.height;
                        ain.numSlices = l.depth;
                        ain.numMipLevels = l.mips;
                        ain.numSamples = 1;
                        ain.numFrags = 1;
                        ain.pitchInElement = out.pitch;
                        ++elements;
                        const std::uint64_t mine = l.address(mip, x, y, 0, z);
                        if (Addr2ComputeSurfaceAddrFromCoord(created.hLib, &ain, &aout) != ADDR_OK || aout.addr != mine) {
                            if (++mismatches <= 12)
                                std::printf("FAIL %s mip %u (%u,%u,%u): 0x%llx, addrlib 0x%llx\n", name, mip, x, y, z,
                                            static_cast<unsigned long long>(mine), static_cast<unsigned long long>(aout.addr));
                        }
                    }
    }
    AddrDestroy(created.hLib);
    std::printf("startup_gfx10_volume --addrlib: %u volumes, %u element addresses compared, %u mismatches (Mesa 25.1.0 addrlib, "
                "SW_4KB_S, 3D)\n", compared, elements, mismatches);
    return compared && elements && !mismatches ? 0 : 1;
}
#endif

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--addrlib") == 0) return addrlib_check();
    const bool mutate = argc == 3 && std::strcmp(argv[1], "--mutate") == 0;
    thin = mutate && std::strcmp(argv[2], "thin") == 0;
    flat_tail = mutate && std::strcmp(argv[2], "tail") == 0;
    if (argc != 1 && !thin && !flat_tail) {
        std::printf("usage: startup_gfx10_volume_check [--addrlib | --mutate thin | --mutate tail]\n");
        return 2;
    }
    check_equations();
    check_block_order();
    check_mip_chain();
    check_distinct_and_round_trip();
    check_descriptions();
    if (failures) {
        std::printf("startup_gfx10_volume: %d failures\n", failures);
        return 1;
    }
    std::printf("startup_gfx10_volume: PASS (5 element sizes, block order, a mip chain, %zu volumes round-tripped mip by mip, "
                "17 layouts, 8 address descriptions, 9 views)\n", sizeof(shapes) / sizeof(shapes[0]));
    return 0;
}
