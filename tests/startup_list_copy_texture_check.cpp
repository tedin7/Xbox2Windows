// Experimental (docs/startup-runtime.md, #870): the argument checks of command list slot 16,
// CopyTextureRegion (src/startup_stubs/list_copy_texture.h). Plain standard C++, synthetic sizes only.
// --mutate overflow drops the destination bounds check, --mutate slice reads an out-of-range array slice as valid; each run must then fail (negative controls).
#include "../src/startup_stubs/list_copy_texture.h"

#include <cstdio>
#include <cstring>

using namespace x2w_list_copy_texture;

namespace {
int failures;
void expect(Verdict got, Verdict want, const char* what)
{
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s: got %s, want %s\n", what, name(got), name(want));
        ++failures;
    }
}
constexpr std::uint32_t rgba8 = 28;  // DXGI_FORMAT_R8G8B8A8_UNORM
constexpr std::uint32_t bc1 = 71;    // DXGI_FORMAT_BC1_UNORM
Side texture(std::uint32_t w, std::uint32_t h, std::uint32_t block = 1)
{
    return {Kind::subresource, w, h, 1, 0, 0, 0, 0, 1, block, block, block > 1 ? bc1 : rgba8, 0, 1, 1, 1};
}
Side footprint(std::uint32_t w, std::uint32_t h, std::uint64_t offset, std::uint32_t pitch, std::uint32_t bpb,
               std::uint64_t bytes, std::uint32_t block = 1)
{
    return {Kind::placed_footprint, w, h, 1, offset, pitch, bpb, bytes, 1, block, block, block > 1 ? bc1 : rgba8, 0, 1, 1, 1};
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc > 2 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "overflow");
    const bool mutate_slice = argc > 2 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "slice");

    expect(check(texture(64, 64), 0, 0, 0, texture(64, 64), nullptr), Verdict::ok, "whole texture");
    expect(check(texture(64, 64), 32, 32, 0, texture(64, 64), &(const Box&)Box{0, 0, 0, 32, 32, 1}), Verdict::ok, "corner box");
    // The destination origin pushes the box past the destination.
    const Verdict past = check(texture(64, 64), 40, 0, 0, texture(64, 64), &(const Box&)Box{0, 0, 0, 32, 32, 1});
    expect(mutate ? Verdict::ok : past, Verdict::dest_overflow, "dest overflow");
    expect(check(texture(64, 64), 0, 0, 0, texture(64, 64), &(const Box&)Box{0, 0, 0, 65, 8, 1}), Verdict::source_overflow, "source overflow");
    expect(check(texture(64, 64), 0, 0, 0, texture(64, 64), &(const Box&)Box{8, 0, 0, 8, 8, 1}), Verdict::bad_box, "empty box");
    expect(check(footprint(16, 16, 0, 64, 4, 4096), 0, 0, 0, footprint(16, 16, 0, 64, 4, 4096), nullptr), Verdict::buffer_to_buffer, "two buffers");

    // Texture to a 16x16 RGBA8 footprint: 64-byte rows would be below 256, so the pitch is refused.
    expect(check(footprint(16, 16, 0, 64, 4, 1 << 20), 0, 0, 0, texture(16, 16), nullptr), Verdict::bad_pitch, "pitch not 256");
    expect(check(footprint(16, 16, 0, 256, 4, 1 << 20), 0, 0, 0, texture(16, 16), nullptr), Verdict::ok, "footprint ok");
    expect(check(footprint(16, 16, 256, 256, 4, 1 << 20), 0, 0, 0, texture(16, 16), nullptr), Verdict::bad_offset, "offset not 512");
    // 16 rows of 256 bytes, the last row 64 bytes: 15 * 256 + 64 = 3904 bytes from offset 512.
    expect(check(footprint(16, 16, 512, 256, 4, 512 + 3904), 0, 0, 0, texture(16, 16), nullptr), Verdict::ok, "last row not padded");
    expect(check(footprint(16, 16, 512, 256, 4, 512 + 3903), 0, 0, 0, texture(16, 16), nullptr), Verdict::footprint_overflow, "one byte short");
    // Row wider than the pitch.
    expect(check(footprint(128, 4, 0, 256, 4, 1 << 20), 0, 0, 0, texture(128, 4), nullptr), Verdict::bad_pitch, "row over pitch");

    // Block-compressed: 4x4 blocks of 8 bytes; a 6-texel start is not on a block.
    expect(check(texture(64, 64, 4), 6, 0, 0, texture(64, 64, 4), nullptr), Verdict::bad_alignment, "block alignment");
    expect(check(footprint(64, 64, 0, 256, 8, 256 * 16, 4), 0, 0, 0, texture(64, 64, 4), nullptr), Verdict::ok, "bc footprint");

    // The destination origin counts: a 1x1 RGBA8 copy at Y=63 of a 64x64, 256-pitch footprint ends at byte 16132.
    expect(check(footprint(64, 64, 0, 256, 4, 256), 0, 63, 0, texture(1, 1), nullptr), Verdict::footprint_overflow, "dest origin counts");
    expect(check(footprint(64, 64, 0, 256, 4, 16132), 0, 63, 0, texture(1, 1), nullptr), Verdict::ok, "dest origin fits");
    // The source box origin counts too.
    expect(check(texture(1, 1), 0, 0, 0, footprint(64, 64, 0, 256, 4, 256), &(const Box&)Box{0, 63, 0, 1, 64, 1}), Verdict::footprint_overflow, "box origin counts");
    // 3D: a slice is as tall as the footprint (64 rows), not as the copy (1 row).
    Side volume = footprint(64, 64, 0, 256, 4, 256 * 64);
    volume.depth = 4;
    expect(check(volume, 0, 0, 1, texture(1, 1), nullptr), Verdict::footprint_overflow, "slice stride from footprint height");

    Side ms = texture(64, 64);
    ms.samples = 4;
    expect(check(texture(64, 64), 0, 0, 0, ms, nullptr), Verdict::sample_mismatch, "multisampled source");
    Side ms2 = ms;
    ms2.samples = 2;
    expect(check(ms2, 0, 0, 0, ms, nullptr), Verdict::sample_mismatch, "2x against 4x");
    expect(check(ms, 0, 0, 0, ms, nullptr), Verdict::ok, "whole 4x subresource");
    const Box full_box{0, 0, 0, 64, 64, 1};
    expect(check(ms, 0, 0, 0, ms, &full_box), Verdict::partial_multisample, "4x explicit full box");
    Side depth = texture(64, 64);
    depth.depth_stencil = true;
    expect(check(depth, 0, 0, 0, depth, nullptr), Verdict::ok, "whole depth subresource");
    expect(check(depth, 0, 0, 0, depth, &full_box), Verdict::partial_depth_stencil, "depth explicit full box");
    expect(check(ms, 8, 0, 0, ms, &(const Box&)Box{0, 0, 0, 56, 64, 1}), Verdict::partial_multisample, "4x offset");
    expect(check(ms, 0, 0, 0, ms, &(const Box&)Box{0, 0, 0, 32, 32, 1}), Verdict::partial_multisample, "4x partial box");
    expect(check(footprint(64, 64, 0, 256, 4, 1 << 20), 0, 0, 0, ms, nullptr), Verdict::sample_mismatch, "4x to buffer");

    // Formats must be the same.
    Side bgra = texture(64, 64);
    bgra.format = 87;  // DXGI_FORMAT_B8G8R8A8_UNORM
    expect(check(bgra, 0, 0, 0, texture(64, 64), nullptr), Verdict::format_mismatch, "texture formats differ");
    Side fp_bc = footprint(64, 64, 0, 256, 4, 1 << 20);
    fp_bc.format = bc1;
    expect(check(fp_bc, 0, 0, 0, texture(64, 64), nullptr), Verdict::format_mismatch, "footprint format differs");

    // Subresource index = mip + slice * mips + plane * mips * array_size.
    Subresource parts{};
    expect(decode_subresource(7, 3, 2, 2, parts) && parts.mip == 1 && parts.slice == 0 && parts.plane == 1 ? Verdict::ok : Verdict::bad_box,
           Verdict::ok, "decode mip 1, slice 0, plane 1");
    expect(decode_subresource(5, 3, 2, 1, parts) && parts.mip == 2 && parts.slice == 1 && parts.plane == 0 ? Verdict::ok : Verdict::bad_box,
           Verdict::ok, "decode mip 2, slice 1");
    expect(decode_subresource(11, 3, 2, 2, parts) ? Verdict::ok : Verdict::bad_subresource, Verdict::ok, "last index of 3 x 2 x 2");
    expect(decode_subresource(12, 3, 2, 2, parts) ? Verdict::ok : Verdict::bad_subresource, Verdict::bad_subresource, "index past the planes");
    expect(decode_subresource(0, 0, 1, 1, parts) ? Verdict::ok : Verdict::bad_subresource, Verdict::bad_subresource, "zero mips");
    // A 4-mip, 3-slice texture has 12 subresources; index 12 is the first past the last slice, and the old
    // mip-only modulo would have read it as mip 0.
    Side arr = texture(64, 64);
    arr.mips = 4;
    arr.array_size = 3;
    arr.subresource = 11;
    expect(check(arr, 0, 0, 0, texture(64, 64), nullptr), Verdict::ok, "last array subresource");
    arr.subresource = 12;
    expect(mutate_slice ? Verdict::ok : check(arr, 0, 0, 0, texture(64, 64), nullptr), Verdict::bad_subresource, "slice out of range");
    expect(check(texture(64, 64), 0, 0, 0, arr, nullptr), Verdict::bad_subresource, "source slice out of range");
    arr.subresource = 0xffffffffu;
    expect(check(arr, 0, 0, 0, texture(64, 64), nullptr), Verdict::bad_subresource, "huge index");

    // Footprint arithmetic is checked at every multiply and add.
    Side wide = footprint(64, 0xffffffffu, 0, 256, 4, ~std::uint64_t{0});
    wide.depth = 0xffffffffu;
    Side big = texture(64, 0xffffffffu);
    big.depth = 0xffffffffu;
    expect(check(wide, 0, 0, 0xfffffffeu, big, &(const Box&)Box{0, 0, 0, 1, 1, 1}), Verdict::footprint_overflow, "slice stride overflow");
    expect(check(footprint(1, 3, ~std::uint64_t{0} - 511, 256, 4, ~std::uint64_t{0}), 0, 0, 0, texture(1, 3), nullptr),
           Verdict::footprint_overflow, "offset plus size wraps");

    if (failures) return 1;
    std::puts("ok");
    return 0;
}
