// A minimal PNG writer for diagnostic pictures: 8-bit RGBA, filter 0, deflate "stored" blocks (no compression, no
// dependency). Readers (Python's zlib, every image viewer) accept stored blocks.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

namespace x2w_png {

inline std::uint32_t crc32(const unsigned char* data, std::size_t bytes, std::uint32_t crc = 0)
{
    static const auto table = [] {
        std::vector<std::uint32_t> t(256);
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < bytes; ++i) crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

inline void put32(std::vector<unsigned char>& out, std::uint32_t value)
{
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>(value >> shift));
}

inline void chunk(std::vector<unsigned char>& out, const char* tag, const std::vector<unsigned char>& data)
{
    put32(out, static_cast<std::uint32_t>(data.size()));
    const std::size_t start = out.size();
    out.insert(out.end(), tag, tag + 4);
    out.insert(out.end(), data.begin(), data.end());
    put32(out, crc32(out.data() + start, out.size() - start));
}

// `rgba`: height rows of width * 4 bytes. Returns the PNG file's bytes.
inline std::vector<unsigned char> encode(const unsigned char* rgba, std::uint32_t width, std::uint32_t height)
{
    std::vector<unsigned char> raw;
    raw.reserve(static_cast<std::size_t>(height) * (width * 4 + 1));
    for (std::uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + static_cast<std::size_t>(y) * width * 4, rgba + static_cast<std::size_t>(y + 1) * width * 4);
    }
    std::vector<unsigned char> z{0x78, 0x01};
    std::uint32_t a = 1, b = 0;
    for (unsigned char c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    for (std::size_t at = 0; at < raw.size() || at == 0;) {
        const std::size_t part = raw.size() - at > 65535 ? 65535 : raw.size() - at;
        z.push_back(at + part == raw.size() ? 1 : 0);
        z.push_back(static_cast<unsigned char>(part));
        z.push_back(static_cast<unsigned char>(part >> 8));
        z.push_back(static_cast<unsigned char>(~part));
        z.push_back(static_cast<unsigned char>(~part >> 8));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(at), raw.begin() + static_cast<std::ptrdiff_t>(at + part));
        at += part;
        if (!part) break;
    }
    put32(z, (b << 16) | a);
    std::vector<unsigned char> out{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<unsigned char> header;
    put32(header, width);
    put32(header, height);
    header.insert(header.end(), {8, 6, 0, 0, 0});
    chunk(out, "IHDR", header);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    return out;
}

}  // namespace x2w_png
