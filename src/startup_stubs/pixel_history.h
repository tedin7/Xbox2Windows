// Bounded per-draw texel history for the translated draw path (X2W_DRAW_DUMP_FRAME plus
// X2W_DRAW_DUMP_PIXEL). One sample is the texel after one translated draw, with the draw ordinal,
// the manifest key and the raw 32-bit word, so a reader can name the draw that last changed it.
#pragma once

#include <dxgiformat.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace x2w_startup {

struct PixelHistorySample {
    std::uint64_t draw{};
    long frame{};
    unsigned x{}, y{};
    std::uint64_t key{};
    unsigned count{};
    DXGI_FORMAT format{DXGI_FORMAT_UNKNOWN};
    std::uint32_t raw{};
    bool rgba{};
    unsigned char channels[4]{};
};

// Only the 8-bit four-channel targets the full draw dump already writes: a 1x1 copy of any other
// format would need its own block or element layout.
inline bool pixel_history_supported(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return true;
    default: return false;
    }
}

inline bool pixel_history_decode(PixelHistorySample& sample, const void* pixel)
{
    if (!pixel || !pixel_history_supported(sample.format)) return false;
    std::uint32_t word = 0;
    std::memcpy(&word, pixel, sizeof(word));
    sample.raw = word;
    const unsigned char* bytes = static_cast<const unsigned char*>(pixel);
    const bool bgra = sample.format == DXGI_FORMAT_B8G8R8A8_UNORM || sample.format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    sample.channels[0] = bytes[bgra ? 2 : 0];
    sample.channels[1] = bytes[bgra ? 1 : 1];
    sample.channels[2] = bytes[bgra ? 0 : 2];
    sample.channels[3] = bytes[3];
    sample.rgba = true;
    return true;
}

// Rewrites the report with every sample kept so far. The caller owns the bounded list; this helper
// stays device-free so a fixture can compare the exact text.
inline bool pixel_history_write(const char* folder, const PixelHistorySample* samples, std::size_t count)
{
    if (!folder || !*folder) return false;
    char path[512]{};
    const int length = std::snprintf(path, sizeof(path), "%s\\pixel_history.txt", folder);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(path)) return false;
    std::FILE* file = std::fopen(path, "wb");
    if (!file) return false;
    std::fprintf(file, "x2w-pixel-history 1 samples=%zu\n", count);
    for (std::size_t i = 0; i < count; ++i) {
        const PixelHistorySample& sample = samples[i];
        std::fprintf(file, "draw=%llu frame=%ld x=%u y=%u key=%016llx count=%u format=%d raw=%08x",
                     static_cast<unsigned long long>(sample.draw), sample.frame, sample.x, sample.y,
                     static_cast<unsigned long long>(sample.key), sample.count, static_cast<int>(sample.format), sample.raw);
        if (sample.rgba)
            std::fprintf(file, " rgba=%u,%u,%u,%u", sample.channels[0], sample.channels[1], sample.channels[2],
                         sample.channels[3]);
        std::fputc('\n', file);
    }
    return std::fclose(file) == 0;
}

}  // namespace x2w_startup
