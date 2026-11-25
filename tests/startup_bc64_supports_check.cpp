// x2w_startup::bc64::supports (src/startup_stubs/bc64_chain.h) without a device. XG texture computers
// and address SRVs use the 64 KiB standard chain only for the descriptions it accepts: 1024 wide, BC5_UNORM
// or BC7_UNORM, six mips at 1024x1024 with one slice or nine mips at 1024x512 or 1024x1024 with one or two
// slices, one sample, explicit layout 0x109, default alignment and zero flags. The ten accepted shapes
// are listed, and each rejected case differs from an accepted one in a single field.
// --mutate layout: expect layout UNKNOWN to be accepted (a negative control).
#include "startup_stubs/bc64_chain.h"

#include <cstdio>
#include <cstring>

namespace {
struct Case {
    const char* name;
    D3D12_RESOURCE_DESC desc;
    bool accepted;
};

constexpr UINT chain_layout = 0x109;

D3D12_RESOURCE_DESC shape(UINT16 mips, UINT height, UINT16 slices, DXGI_FORMAT format)
{
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1024, height, slices, mips, format, {1, 0},
                            D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    static_assert(sizeof(desc.Layout) == sizeof(chain_layout));
    std::memcpy(&desc.Layout, &chain_layout, sizeof(chain_layout));
    return desc;
}

D3D12_RESOURCE_DESC nine() { return shape(9, 1024, 1, DXGI_FORMAT_BC7_UNORM); }

template<typename Edit> D3D12_RESOURCE_DESC nine_with(Edit edit)
{
    D3D12_RESOURCE_DESC d = nine();
    edit(d);
    return d;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool layout = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "layout") == 0;
    if (argc != 1 && !layout) {
        std::printf("usage: startup_bc64_supports_check [--mutate layout]\n");
        return 2;
    }
    constexpr auto bc5 = DXGI_FORMAT_BC5_UNORM, bc7 = DXGI_FORMAT_BC7_UNORM;
    const Case cases[] = {
        {"six BC5 1024x1024x1", shape(6, 1024, 1, bc5), true},
        {"six BC7 1024x1024x1", shape(6, 1024, 1, bc7), true},
        {"nine BC5 1024x512x1", shape(9, 512, 1, bc5), true},
        {"nine BC5 1024x512x2", shape(9, 512, 2, bc5), true},
        {"nine BC5 1024x1024x1", shape(9, 1024, 1, bc5), true},
        {"nine BC5 1024x1024x2", shape(9, 1024, 2, bc5), true},
        {"nine BC7 1024x512x1", shape(9, 512, 1, bc7), true},
        {"nine BC7 1024x512x2", shape(9, 512, 2, bc7), true},
        {"nine BC7 1024x1024x1", shape(9, 1024, 1, bc7), true},
        {"nine BC7 1024x1024x2", shape(9, 1024, 2, bc7), true},
        // Shape: only the combinations above.
        {"six mips at 1024x512", shape(6, 512, 1, bc7), false},
        {"six mips with two slices", shape(6, 1024, 2, bc7), false},
        {"nine mips with three slices", shape(9, 1024, 3, bc7), false},
        {"seven mips", shape(7, 1024, 1, bc7), false},
        {"nine mips at 1024x256", shape(9, 256, 1, bc7), false},
        {"2048 wide", nine_with([](auto& d) { d.Width = 2048; }), false},
        // Every other field of the description.
        {"3D dimension", nine_with([](auto& d) { d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D; }), false},
        {"64 KiB alignment", nine_with([](auto& d) { d.Alignment = 65536; }), false},
        {"four samples", nine_with([](auto& d) { d.SampleDesc.Count = 4; }), false},
        {"sample quality 1", nine_with([](auto& d) { d.SampleDesc.Quality = 1; }), false},
        {"layout UNKNOWN", nine_with([](auto& d) { d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; }), layout},
        {"layout 64KB_STANDARD_SWIZZLE",
         nine_with([](auto& d) { d.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE; }), false},
        {"render-target flag", nine_with([](auto& d) { d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET; }), false},
        {"BC7_UNORM_SRGB", nine_with([](auto& d) { d.Format = DXGI_FORMAT_BC7_UNORM_SRGB; }), false},
        {"BC5_SNORM", nine_with([](auto& d) { d.Format = DXGI_FORMAT_BC5_SNORM; }), false},
        {"BC3_UNORM", nine_with([](auto& d) { d.Format = DXGI_FORMAT_BC3_UNORM; }), false},
    };
    int failures = 0, accepted = 0;
    for (const Case& c : cases) {
        const bool actual = x2w_startup::bc64::supports(c.desc);
        accepted += actual;
        if (actual != c.accepted) {
            std::printf("FAIL %s: %s, expected %s\n", c.name, actual ? "accepted" : "rejected",
                        c.accepted ? "accepted" : "rejected");
            ++failures;
        }
    }
    if (failures) return 1;

    // The chain arithmetic itself: the tail offsets and the in-tile bit interleave. The addrlib
    // reference test covers these only when the addrlib source is configured, so a build without it
    // had no check on `start()` or `address()` at all.
    {
        const x2w_startup::bc64::Chain chain(nine());
        struct Offset {
            const char* name;
            UINT mip;
            UINT expected;
        };
        const Offset offsets[] = {
            {"nine mips, tail mip 3", 3, 0x8000},
            {"nine mips, tail mip 4", 4, 0x4000},
            {"nine mips, tail mip 5", 5, 0x2000},
            {"nine mips, tail mip 6", 6, 0x1000},
            {"nine mips, tail mip 7", 7, 0x800},
            {"nine mips, tail mip 8", 8, 0x600},
        };
        for (const Offset& o : offsets) {
            if (chain.start(o.mip) != o.expected) {
                std::printf("FAIL %s: start(%u) = %#x, expected %#x\n", o.name, o.mip, chain.start(o.mip),
                            o.expected);
                ++failures;
            }
        }
        // Each whole-tile mip starts one step after the next one, its own tiles plus that step.
        for (UINT mip = 0; mip + 1 < x2w_startup::bc64::first_tail; ++mip) {
            const UINT step = chain.tiles_wide(mip + 1) * chain.tiles_high(mip + 1) * x2w_startup::bc64::tile_bytes;
            if (chain.start(mip) != chain.start(mip + 1) + step) {
                std::printf("FAIL mip %u starts %#x, expected %#x + %#x\n", mip, chain.start(mip),
                            chain.start(mip + 1), step);
                ++failures;
            }
        }
        // The interleave has to separate the axes: the same distance along x and along y lands in
        // different bytes, and both are inside the tile.
        const UINT at_origin = chain.address(0, 0, 0, 0), along_x = chain.address(0, 1, 0, 0),
                   along_y = chain.address(0, 0, 1, 0);
        if (at_origin != chain.start(0) || along_x == along_y) {
            std::printf("FAIL address: origin %#x, x %#x, y %#x\n", at_origin, along_x, along_y);
            ++failures;
        }
        if (along_x != at_origin + 64 || along_y != at_origin + 16) {
            std::printf("FAIL address: one texel along x should be +64 and along y +16, got %#x and %#x\n",
                        along_x - at_origin, along_y - at_origin);
            ++failures;
        }
        // The high bits of both coordinates go through the same interleave, so a coordinate of 4
        // (the first one that sets the bit-2 term) has to land 512 and 256 bytes out respectively.
        const UINT far_x = chain.address(0, 4, 0, 0), far_y = chain.address(0, 0, 4, 0);
        if (far_x - at_origin != 512 || far_y - at_origin != 256) {
            std::printf("FAIL address: four texels along x should be +512 and along y +256, got %#x and %#x\n",
                        far_x - at_origin, far_y - at_origin);
            ++failures;
        }
        // One slice is the tile offsets plus mip 0's whole tiles, and a second slice starts after it.
        if (chain.slice_bytes() != chain.start(0) + chain.tiles_wide(0) * chain.tiles_high(0) * x2w_startup::bc64::tile_bytes) {
            std::printf("FAIL slice_bytes %#x is not start(0) plus mip 0's %u tiles\n", chain.slice_bytes(),
                        chain.tiles_wide(0) * chain.tiles_high(0));
            ++failures;
        }
        // A second slice of the nine-mip shape starts one slice further along.
        if (chain.address(0, 0, 0, 1) != chain.slice_bytes() + at_origin) {
            std::printf("FAIL address: slice 1 starts %#x, expected %#x\n", chain.address(0, 0, 0, 1),
                        chain.slice_bytes() + at_origin);
            ++failures;
        }
    }
    if (failures) return 1;
    std::printf("bc64 supports: %d of %zu descriptions accepted, the rest rejected\n", accepted,
                sizeof(cases) / sizeof(cases[0]));
    return 0;
}
