// Checks the volume layout of src/startup_stubs/gfx10_layout.h (a single-mip 3D texture in the 4 KiB
// standard mode, 0x105, which addrlib lays out in thick blocks) without a device:
// - the address bit of every x, y and z bit against the one-pipe rows of addrlib's GFX10_SW_4K_S3_PATINFO,
//   restated below, and the block each element size gives;
// - the order of blocks: along x, then y, then a layer of blocks at a time along z, with the sides of
//   the texture padded to whole blocks;
// - that every element of a volume gets its own bytes inside the texture, and that a tile followed by a
//   detile (with padded row and layer pitches) returns the texels it started from;
// - the descriptions it takes and refuses, and the slot-70 views it allows on them.
// `--addrlib` compares the size and every element address of a set of volumes with
// Addr2ComputeSurfaceInfo and Addr2ComputeSurfaceAddrFromCoord (ADDR_RSRC_TEX_3D, ADDR_SW_4KB_S) and
// skips (77) when the build has no addrlib. `--mutate thin` lays each volume out as a thin 2D array of
// its depth layers and is the negative control: it must fail.
//
// Reference: AMD addrlib as Mesa carries it under src/amd/addrlib, at the revision tools/fetch_addrlib.py
// pins (Mesa 25.1.0, commit 5c142e46f3f6e752ed745fd48912ebb8fad67145): src/gfx10/gfx10SwizzlePattern.h,
// src/gfx10/gfx10addrlib.cpp and src/core/addrlib2.cpp, each with the notice
//   Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT
// The pattern rows are restated below as data; no addrlib code is included in the default run.
#include "startup_stubs/address_texture.h"
#include "startup_stubs/gfx10_layout.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
using x2w_startup::gfx10::Layout;
int failures = 0;
bool thin = false; // --mutate thin

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

// The layout under test; the control reads the volume as a thin array of its layers.
Layout layout_of(D3D12_RESOURCE_DESC d)
{
    if (thin) d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    return Layout(d);
}
std::uint64_t at(const Layout& l, UINT x, UINT y, UINT z)
{
    return thin ? l.address(0, x, y, z) : l.address(0, x, y, 0, z);
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

struct Shape {
    UINT width, height, depth;
    DXGI_FORMAT format;
};
const Shape shapes[] = {
    {17, 17, 17, DXGI_FORMAT_R16G16B16A16_FLOAT}, {32, 32, 32, DXGI_FORMAT_R8G8B8A8_UNORM}, {33, 33, 33, DXGI_FORMAT_R16G16B16A16_FLOAT},
    {32, 32, 32, DXGI_FORMAT_A8_UNORM},           {16, 16, 8, DXGI_FORMAT_R16_FLOAT},       {16, 16, 16, DXGI_FORMAT_R8G8_UNORM},
    {32, 32, 32, DXGI_FORMAT_R10G10B10A2_UNORM},  {5, 3, 7, DXGI_FORMAT_R32G32B32A32_FLOAT}, {1, 1, 2, DXGI_FORMAT_R8_UNORM},
    {64, 8, 40, DXGI_FORMAT_R32_FLOAT},
};

void check_distinct_and_round_trip()
{
    for (const Shape& s : shapes) {
        const D3D12_RESOURCE_DESC d = volume(s.width, s.height, s.depth, s.format);
        const std::string name = std::to_string(s.width) + "x" + std::to_string(s.height) + "x" + std::to_string(s.depth) +
                                 " format " + std::to_string(s.format);
        if (!x2w_startup::gfx10::supports(d)) {
            fail(name + ": refused");
            continue;
        }
        const Layout l = layout_of(d);
        const UINT e = l.bytes_per_element;
        const std::uint64_t size = l.bytes();
        std::vector<unsigned char> used(static_cast<std::size_t>(size));
        bool clash = false;
        for (UINT z = 0; z < s.depth; ++z)
            for (UINT y = 0; y < s.height; ++y)
                for (UINT x = 0; x < s.width; ++x) {
                    const std::uint64_t a = at(l, x, y, z);
                    if (a % e || a + e > size) {
                        clash = true;
                        continue;
                    }
                    if (used[static_cast<std::size_t>(a)]) clash = true;
                    used[static_cast<std::size_t>(a)] = 1;
                }
        if (clash) fail(name + ": elements share bytes or fall outside the texture");
        if (thin) continue;
        const std::size_t pitch = (std::size_t{s.width} * e + 255) & ~std::size_t{255}, layer = pitch * (s.height + 1);
        std::vector<unsigned char> rows(layer * s.depth), back(rows.size()), tiled(static_cast<std::size_t>(size));
        for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<unsigned char>(i * 131 + 7);
        l.tile(0, 0, tiled.data(), rows.data(), pitch, layer);
        l.detile(0, 0, tiled.data(), back.data(), pitch, layer);
        for (UINT z = 0; z < s.depth; ++z)
            for (UINT y = 0; y < s.height; ++y)
                if (std::memcmp(&rows[z * layer + y * pitch], &back[z * layer + y * pitch], std::size_t{s.width} * e)) {
                    fail(name + ": round trip differs at layer " + std::to_string(z) + " row " + std::to_string(y));
                    z = s.depth;
                    break;
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
    d.MipLevels = 2;
    supported("two mips", d, false);
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
    address("mip chain", d, false);

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
    for (const Shape& s : shapes) cases.push_back(volume(s.width, s.height, s.depth, s.format));
    for (UINT size = 0; size < 5; ++size)
        for (UINT w : {1u, 3u, 9u, 24u})
            for (UINT h : {1u, 7u, 17u})
                for (UINT dep : {1u, 5u, 18u}) cases.push_back(volume(w, h, dep, by_size[size]));
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
        in.numMipLevels = 1;
        in.numSamples = 1;
        in.numFrags = 1;
        ++compared;
        char name[64];
        std::snprintf(name, sizeof name, "%ux%ux%u %u-byte", l.width, l.height, l.depth, l.bytes_per_element);
        if (Addr2ComputeSurfaceInfo(created.hLib, &in, &out) != ADDR_OK || out.surfSize != l.bytes()) {
            if (++mismatches <= 12)
                std::printf("FAIL %s: size 0x%llx, addrlib 0x%llx\n", name, static_cast<unsigned long long>(l.bytes()),
                            static_cast<unsigned long long>(out.surfSize));
            continue;
        }
        for (UINT z = 0; z < l.depth; ++z)
            for (UINT y = 0; y < l.height; ++y)
                for (UINT x = 0; x < l.width; ++x) {
                    ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT ain{};
                    ADDR2_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT aout{};
                    ain.size = sizeof ain;
                    aout.size = sizeof aout;
                    ain.x = x;
                    ain.y = y;
                    ain.slice = z;
                    ain.swizzleMode = ADDR_SW_4KB_S;
                    ain.flags.texture = 1;
                    ain.resourceType = ADDR_RSRC_TEX_3D;
                    ain.bpp = in.bpp;
                    ain.unalignedWidth = l.width;
                    ain.unalignedHeight = l.height;
                    ain.numSlices = l.depth;
                    ain.numMipLevels = 1;
                    ain.numSamples = 1;
                    ain.numFrags = 1;
                    ain.pitchInElement = out.pitch;
                    ++elements;
                    const std::uint64_t mine = l.address(0, x, y, 0, z);
                    if (Addr2ComputeSurfaceAddrFromCoord(created.hLib, &ain, &aout) != ADDR_OK || aout.addr != mine) {
                        if (++mismatches <= 12)
                            std::printf("FAIL %s (%u,%u,%u): 0x%llx, addrlib 0x%llx\n", name, x, y, z,
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
    thin = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "thin") == 0;
    if (argc != 1 && !thin) {
        std::printf("usage: startup_gfx10_volume_check [--addrlib | --mutate thin]\n");
        return 2;
    }
    check_equations();
    check_block_order();
    check_distinct_and_round_trip();
    check_descriptions();
    if (failures) {
        std::printf("startup_gfx10_volume: %d failures\n", failures);
        return 1;
    }
    std::printf("startup_gfx10_volume: PASS (5 element sizes, block order, %zu round trips, 15 layouts, 6 address descriptions, "
                "7 views)\n", sizeof(shapes) / sizeof(shapes[0]));
    return 0;
}
