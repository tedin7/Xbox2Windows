#pragma once

#include <cstdint>
#include <dxgiformat.h>

struct IDXGIAdapter;

struct StartupBc7Input {
    const char* blocks;
    std::uint32_t width, height;
    bool tiled;
    // "resource format:resource flags:view format" in decimal, hex and decimal: instead of BC7 data,
    // a texture of that description is created through slot 133, cleared through a render-target
    // view of the view format and sampled through a shader-resource view of the view format.
    const char* packed = nullptr;
    // A 32x32 R8G8B8A8 texture with two mips at Alignment 0x1000 (src/startup_stubs/small_texture_4k.h): blocks
    // holds mip 0 then mip 1 as tight rows; both go through XG slot 8 into memory of exactly the size XG slot 2
    // reports, followed by a no-access guard page. The target is 48x32: mip 0 at the left, mip 1 top right.
    bool small4k = false;
    // A 256x256 R8G8B8A8 texture with 9 mips, every mip one solid colour, in the title's row layout: placed through slot 133
    // and sampled at level 0 by a translated draw (tests/startup_texture_drop_mips_draw_check.py).
    bool dropmips = false;
};

int check_stream_draw(IDXGIAdapter* adapter, void* guest, void* pso, const char* pixels,
                      const StartupBc7Input* bc7 = nullptr, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
                      std::uint32_t export_word = 0, bool two_targets = false);
