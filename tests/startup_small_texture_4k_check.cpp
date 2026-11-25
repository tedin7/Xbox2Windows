// Checks src/startup_stubs/small_texture_4k.h without a device: the 4 KiB answers for small 2D textures
// asked with a 4 KiB alignment, and the descriptions it leaves alone.
//
//   startup_small_texture_4k_check                            literal sizes and refused descriptions
//   startup_small_texture_4k_check --mutate whole-width-tail  expectations of a rule whose mip tail is a
//                                                             whole block wide (negative control: fails)
//   startup_small_texture_4k_check --addrlib                  every accepted shape against AMD addrlib's
//                                                             GFX10 4 KiB layout; 77 when built without it
//
// The literal sizes are worked by hand from the 4 KiB blocks (128x64 texels at 4 bits, 64x64 at 8, 64x32 at
// 16, 32x32 at 32, 32x16 at 64, 16x16 at 128) and the tail rule; the --addrlib run reproduces all of them.
// addrlib: Mesa 25.1.0 (5c142e46f3f6e752ed745fd48912ebb8fad67145) src/amd/addrlib, MIT, fetched by
// tools/fetch_addrlib.py; nothing is vendored.
#include "startup_stubs/small_texture_4k.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {
int failures;

D3D12_RESOURCE_DESC texture(unsigned format, unsigned width, unsigned height, unsigned mips = 1, UINT64 alignment = 0x1000)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = alignment;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = static_cast<UINT16>(mips);
    desc.Format = static_cast<DXGI_FORMAT>(format);
    desc.SampleDesc.Count = 1;
    return desc;
}

const D3D12_RESOURCE_ALLOCATION_INFO desktop{0x10000, 0x10000};

void answered(const char* name, const D3D12_RESOURCE_DESC& desc, unsigned long long expected)
{
    D3D12_RESOURCE_ALLOCATION_INFO info = desktop;
    const bool changed = x2w_startup::small_texture_4k(desc, info);
    if (!changed || info.SizeInBytes != expected || info.Alignment != 0x1000) {
        std::printf("FAIL %s: changed %d size 0x%llx alignment 0x%llx, expected 0x%llx at 0x1000\n", name, changed,
                    static_cast<unsigned long long>(info.SizeInBytes), static_cast<unsigned long long>(info.Alignment), expected);
        ++failures;
    }
}

void kept(const char* name, const D3D12_RESOURCE_DESC& desc)
{
    D3D12_RESOURCE_ALLOCATION_INFO info = desktop;
    if (x2w_startup::small_texture_4k(desc, info) || info.SizeInBytes != desktop.SizeInBytes || info.Alignment != desktop.Alignment) {
        std::printf("FAIL %s: the desktop answer was changed to 0x%llx at 0x%llx\n", name,
                    static_cast<unsigned long long>(info.SizeInBytes), static_cast<unsigned long long>(info.Alignment));
        ++failures;
    }
}

int literal(bool whole_width_tail)
{
    // whole_width_tail: what a tail one block wide (instead of half) would answer.
    const auto rule = [whole_width_tail](unsigned long long real, unsigned long long mutated) { return whole_width_tail ? mutated : real; };
    answered("32x16 R8G8B8A8", texture(28, 32, 16), 0x1000);
    answered("128x4 R16G16B16A16 (desktop at 4 KiB answers 0x2000)", texture(11, 128, 4), 0x4000);
    answered("4x4 BC1", texture(71, 4, 4), 0x1000);
    answered("128x128 R16 float", texture(54, 128, 128), 0x8000);
    answered("256x256 BC7, the 64 KiB bound", texture(98, 256, 256), 0x10000);
    answered("16x16 R32G32B32A32 float", texture(2, 16, 16), 0x1000);
    answered("128x128 BC1 6 mips: 2 blocks + tail", texture(71, 128, 128, 6), 0x3000);
    answered("64x128 BC1 5 mips: 2 blocks + tail", texture(71, 64, 128, 5), 0x3000);
    answered("64x64 BC7 5 mips: 1 block + tail", texture(98, 64, 64, 5), rule(0x2000, 0x1000));
    answered("128x64 BC7 5 mips: 2 + 1 + tail", texture(98, 128, 64, 5), rule(0x4000, 0x3000));
    answered("256x64 BC1 5 mips: 2 + 1 + tail", texture(71, 256, 64, 5), rule(0x4000, 0x3000));
    answered("128x32 BC1 4 mips: 1 + tail", texture(71, 128, 32, 4), rule(0x2000, 0x1000));
    answered("128x128 R8G8B8A8 8 mips: 16 + 4 + 1 + tail", texture(28, 128, 128, 8), rule(0x16000, 0x15000));
    answered("1024x1 R8 full chain (MipLevels 0): 16+8+4+2+1 + tail", texture(61, 1024, 1, 0), 0x20000);
    answered("64x64 BC1 full chain: all 7 levels in one tail block", texture(71, 64, 64, 7), 0x1000);

    // Refused: the desktop answer stays.
    kept("alignment 0", texture(28, 32, 16, 1, 0));
    kept("alignment 64 KiB", texture(28, 32, 16, 1, 0x10000));
    D3D12_RESOURCE_DESC d = texture(28, 32, 16);
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    kept("render target", d);
    d = texture(40, 32, 16);
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    kept("depth target", d);
    kept("256x128 R8G8B8A8, over the 64 KiB bound", texture(28, 256, 128));
    kept("512x256 BC7, over the bound", texture(98, 512, 256));
    d = texture(28, 32, 16);
    d.DepthOrArraySize = 2;
    kept("array of 2", d);
    kept("48x16 R8G8B8A8, width not a power of two", texture(28, 48, 16));
    kept("R32G32B32 float, 96 bits", texture(6, 32, 16));
    kept("32x16 with 7 mips, more than the chain", texture(28, 32, 16, 7));
    d = texture(28, 32, 16);
    d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    kept("64 KiB layout", d);
    d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x102);
    kept("console layout code", d);
    d = texture(28, 32, 16);
    d.SampleDesc.Count = 4;
    kept("4 samples", d);
    d = texture(28, 32, 16);
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    kept("3D texture", d);
    kept("depth with stencil R32G8X24 (two planes)", texture(20, 32, 16));
    kept("depth with stencil R24G8 (two planes)", texture(45, 32, 16));
    d = texture(61, 1, 1);
    d.Width = 1ull << 40;
    d.Height = 1u << 24;
    kept("2^40 x 2^24 R8, past the dimension limit", d);
    kept("32768x1 R8, past 16384", texture(61, 32768, 1));
    {
        D3D12_RESOURCE_ALLOCATION_INFO failed{~0ull, 0};
        if (x2w_startup::small_texture_4k(texture(71, 2, 2), failed) || failed.SizeInBytes != ~0ull) {
            std::printf("FAIL 2x2 BC1: a desktop failure was replaced\n");
            ++failures;
        }
    }

    // Bits per texel for every format value, listed from the DXGI format table (not from the helper's ranges).
    const unsigned bits128[] = {1, 2, 3, 4};
    const unsigned bits64[] = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const unsigned bits32[] = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                               67, 87, 88, 89, 90, 91, 92, 93};
    const unsigned bits16[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 85, 86, 115};
    const unsigned bits8[] = {60, 61, 62, 63, 64, 65, 73, 74, 75, 76, 77, 78, 82, 83, 84, 94, 95, 96, 97, 98, 99};
    const unsigned bits4[] = {70, 71, 72, 79, 80, 81};
    unsigned expected_bits[256] = {};
    for (unsigned f : bits128) expected_bits[f] = 128;
    for (unsigned f : bits64) expected_bits[f] = 64;
    for (unsigned f : bits32) expected_bits[f] = 32;
    for (unsigned f : bits16) expected_bits[f] = 16;
    for (unsigned f : bits8) expected_bits[f] = 8;
    for (unsigned f : bits4) expected_bits[f] = 4;
    for (unsigned f = 0; f < 256; ++f)
        if (x2w_startup::small_texture_4k_bits(f) != expected_bits[f]) {
            std::printf("FAIL format %u: %u bits per texel, expected %u\n", f, x2w_startup::small_texture_4k_bits(f), expected_bits[f]);
            ++failures;
        }

    // The limit the stand-in's own reads and writes use: the 4 KiB answer at Alignment 0x1000 only.
    const auto extent_is = [](const char* name, const D3D12_RESOURCE_DESC& desc, unsigned long long expected) {
        const unsigned long long got = x2w_startup::small_texture_4k_extent(desc);
        if (got != expected) {
            std::printf("FAIL extent %s: 0x%llx, expected 0x%llx\n", name, got, expected);
            ++failures;
        }
    };
    extent_is("32x32 R8G8B8A8 at alignment 0 (desktop answer kept)", texture(28, 32, 32, 1, 0), 0);
    extent_is("32x32 R8G8B8A8 at 4 KiB", texture(28, 32, 32), 0x1000);
    extent_is("32x32 R8G8B8A8 at 64 KiB", texture(28, 32, 32, 1, 0x10000), 0);
    extent_is("256x128 R8G8B8A8, over the bound", texture(28, 256, 128), 0);

    // The compact plan slot 2 describes, slot 8 writes and upload reads: levels back to back at exact row bytes.
    struct Level { unsigned long long offset, row_bytes, rows; };
    const auto plan_is = [](const char* name, const D3D12_RESOURCE_DESC& desc, std::initializer_list<Level> levels) {
        x2w_startup::SmallTexturePlan plan;
        bool ok = x2w_startup::small_texture_4k_plan(desc, plan) && plan.levels == levels.size();
        unsigned i = 0;
        for (const Level& l : levels) {
            ok = ok && plan.level[i].offset == l.offset && plan.level[i].row_bytes == l.row_bytes && plan.level[i].rows == l.rows &&
                 plan.level[i].bytes == l.row_bytes * l.rows;
            ++i;
        }
        if (!ok) {
            std::printf("FAIL plan %s\n", name);
            ++failures;
        }
    };
    plan_is("32x32 R8G8B8A8 2 mips", texture(28, 32, 32, 2), {{0, 128, 32}, {0x1000, 64, 16}});
    plan_is("128x128 BC1 3 mips", texture(71, 128, 128, 3), {{0, 256, 32}, {0x2000, 128, 16}, {0x2800, 64, 8}});
    plan_is("4x4 BC7 full chain", texture(98, 4, 4, 0), {{0, 16, 1}, {16, 16, 1}, {32, 16, 1}});
    {
        x2w_startup::SmallTexturePlan plan;
        if (x2w_startup::small_texture_4k_plan(texture(28, 32, 32, 1, 0), plan)) {
            std::printf("FAIL plan at alignment 0: a description without the 4 KiB answer got a compact plan\n");
            ++failures;
        }
    }

    // Every accepted shape (one format of each bit depth and both block sizes, power-of-two sides up to 4096,
    // every mip count): the compact plan ends inside the answer, and no tail holds more than 7 levels, so
    // addrlib's 8-level tail limit never applies to 4 KiB thin blocks.
    const unsigned formats[] = {71, 98, 61, 54, 28, 11, 2};
    unsigned shapes = 0, overflow = 0, long_tails = 0;
    for (unsigned f : formats)
        for (unsigned w = 1; w <= 4096; w <<= 1)
            for (unsigned h = 1; h <= 4096; h <<= 1)
                for (unsigned mips = 1; mips <= 13; ++mips) {
                    const D3D12_RESOURCE_DESC desc = texture(f, w, h, mips);
                    D3D12_RESOURCE_ALLOCATION_INFO info = desktop;
                    x2w_startup::SmallTexturePlan plan;
                    if (!x2w_startup::small_texture_4k(desc, info) || !x2w_startup::small_texture_4k_plan(desc, plan)) continue;
                    ++shapes;
                    if (plan.bytes > info.SizeInBytes) ++overflow;
                    const auto block = x2w_startup::small_texture_4k_detail::block(x2w_startup::small_texture_4k_bits(f));
                    for (unsigned m = 0; m < mips; ++m)
                        if (((w >> m) ? (w >> m) : 1u) <= block.width / 2 && ((h >> m) ? (h >> m) : 1u) <= block.height) {
                            if (mips - m > 7) ++long_tails;
                            break;
                        }
                }
    std::printf("compact plan: %u accepted shapes, %u past the answer, %u tails over 7 levels\n", shapes, overflow, long_tails);
    if (shapes != 4943 || overflow || long_tails) ++failures;
    if (failures) {
        std::printf("startup_small_texture_4k: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("startup_small_texture_4k: PASS (15 sizes, 19 refused descriptions, 256 format values, extents, compact plans)\n");
    return 0;
}
}  // namespace

#ifndef XBOX2WINDOWS_HAVE_ADDRLIB
int addrlib_check()
{
    std::puts("startup_small_texture_4k --addrlib: SKIP (built without addrlib; run tools/fetch_addrlib.py and configure with "
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
}  // namespace

// Every power-of-two shape and mip count the helper accepts, for one format of each bit depth, against
// Addr2ComputeSurfaceInfo with ADDR_SW_4KB_S on Mesa's "navi21" (gfx10.3) configuration, the one
// xg_swizzle_select_check uses: family 0x8F, revision 0x28, GB_ADDR_CONFIG 0x444.
int addrlib_check()
{
    ADDR_CREATE_INPUT create{};
    ADDR_CREATE_OUTPUT created{};
    create.size = sizeof create;
    created.size = sizeof created;
    create.chipEngine = 0x0000000D;  // CIASICIDGFXENGINE_ARCTICISLAND
    create.chipFamily = 0x8F;
    create.chipRevision = 0x28;
    create.callbacks.allocSysMem = alloc_sys_mem;
    create.callbacks.freeSysMem = free_sys_mem;
    create.regValue.gbAddrConfig = 0x40 | 4u | (4u << 8);
    if (AddrCreate(&create, &created) != ADDR_OK) return 1;
    struct { unsigned dxgi; AddrFormat addr; const char* name; } formats[] = {
        {71, ADDR_FMT_BC1, "BC1"}, {98, ADDR_FMT_BC7, "BC7"}, {61, ADDR_FMT_8, "R8"}, {54, ADDR_FMT_16, "R16"},
        {28, ADDR_FMT_8_8_8_8, "R8G8B8A8"}, {11, ADDR_FMT_16_16_16_16, "R16G16B16A16"}, {2, ADDR_FMT_32_32_32_32, "R32G32B32A32"}};
    unsigned compared = 0, mismatches = 0;
    for (const auto& f : formats)
        for (unsigned w = 1; w <= 4096; w <<= 1)
            for (unsigned h = 1; h <= 4096; h <<= 1)
                for (unsigned mips = 1; mips <= 13; ++mips) {
                    D3D12_RESOURCE_ALLOCATION_INFO info = desktop;
                    if (!x2w_startup::small_texture_4k(texture(f.dxgi, w, h, mips), info)) continue;
                    ADDR2_COMPUTE_SURFACE_INFO_INPUT in{};
                    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT out{};
                    in.size = sizeof in;
                    out.size = sizeof out;
                    in.flags.texture = 1;
                    in.swizzleMode = ADDR_SW_4KB_S;
                    in.resourceType = ADDR_RSRC_TEX_2D;
                    in.format = f.addr;
                    in.width = w;
                    in.height = h;
                    in.numSlices = 1;
                    in.numMipLevels = mips;
                    in.numSamples = 1;
                    in.numFrags = 1;
                    ++compared;
                    if (Addr2ComputeSurfaceInfo(created.hLib, &in, &out) != ADDR_OK || out.surfSize != info.SizeInBytes ||
                        out.baseAlign != info.Alignment) {
                        if (++mismatches <= 12)
                            std::printf("FAIL %s %ux%u %u mips: helper 0x%llx at 0x%llx, addrlib 0x%llx at 0x%x\n", f.name, w, h, mips,
                                        static_cast<unsigned long long>(info.SizeInBytes),
                                        static_cast<unsigned long long>(info.Alignment),
                                        static_cast<unsigned long long>(out.surfSize), out.baseAlign);
                    }
                }
    AddrDestroy(created.hLib);
    std::printf("startup_small_texture_4k --addrlib: %u shapes compared, %u mismatches (Mesa 25.1.0 addrlib, SW_4KB_S)\n", compared,
                mismatches);
    return compared && !mismatches ? 0 : 1;
}
#endif

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--addrlib") == 0) return addrlib_check();
    const bool mutate = argc > 2 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "whole-width-tail") == 0;
    if (argc > 1 && !mutate) return 2;
    return literal(mutate);
}
