#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace x2w_local {

// A raw shader buffer view returns zero for every read past its end, and memory the title never wrote is zero. So the
// trailing all-zero part of a bound range need not be copied: the view over the shorter copy reads the same values. Returns the
// length of [data, data + bytes) without its trailing zero bytes, rounded up to `granule` (a power of two) and never below
// `granule` or above `bytes`. The scan walks backwards from the end and stops at the first non-zero word, so a buffer that is
// used to its end costs one word.
inline std::size_t used_length(const void* data, std::size_t bytes, std::size_t granule = 256) noexcept
{
    if (bytes <= granule) return bytes;
    const auto* base = static_cast<const unsigned char*>(data);
    std::size_t end = bytes;
    while (end % 8 && end > 0) {
        if (base[end - 1]) return (end + granule - 1) & ~(granule - 1) < bytes ? (end + granule - 1) & ~(granule - 1) : bytes;
        --end;
    }
    while (end >= 8) {
        std::uint64_t word;
        std::memcpy(&word, base + end - 8, 8);
        if (word) break;
        end -= 8;
    }
    if (end < granule) end = granule;
    const std::size_t rounded = (end + granule - 1) & ~(granule - 1);
    return rounded < bytes ? rounded : bytes;
}

// The backwards scan is a pass over the whole range when the tail is zero (about 3 ms for 16 MiB, more than the copy it saves).
// The per-frame cache below is OFF by default: nothing tells the runtime when the title writes into a buffer, so a cached length can
// miss data written between two draws of a frame (the shader would read zero). It stays as an opt-in for a caller that has a
// write signal. (original comment:)
// it is done once per frame per (address, size). Later draws of the frame re-check only where a writer would grow the data: the
// 64 KiB after the cached length and the last 4 KiB of the range; a non-zero byte there rescans. Data written elsewhere in the
// tail after the first draw of a frame is picked up on the next frame.
class ZeroTailCache {
public:
    std::size_t length(std::uint64_t frame, const void* data, std::size_t bytes, bool use_cache = false)
    {
        if (frame != frame_) {
            entries_.clear();
            frame_ = frame;
        }
        const std::uint64_t key = reinterpret_cast<std::uintptr_t>(data) * 1099511628211ull ^ bytes;
        if (use_cache) {
            const auto found = entries_.find(key);
            if (found != entries_.end() && found->second.data == data && found->second.bytes == bytes) {
                const std::size_t cached = found->second.used;
                const auto* base = static_cast<const unsigned char*>(data);
                std::size_t from = cached, to = cached + (64u << 10) < bytes ? cached + (64u << 10) : bytes;
                bool clean = all_zero(base + from, to - from);
                if (clean && bytes > (4u << 10) && to < bytes) {
                    from = bytes - (4u << 10) > to ? bytes - (4u << 10) : to;
                    clean = all_zero(base + from, bytes - from);
                }
                if (clean) {
                    ++hits_;
                    return cached;
                }
            }
        }
        ++scans_;
        const std::size_t used = used_length(data, bytes);
        if (use_cache) entries_[key] = Entry{data, bytes, used};
        return used;
    }
    long long scans() const { return scans_; }
    long long hits() const { return hits_; }

private:
    struct Entry {
        const void* data;
        std::size_t bytes, used;
    };
    static bool all_zero(const unsigned char* p, std::size_t n)
    {
        std::size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            std::uint64_t w[4];
            std::memcpy(w, p + i, 32);
            if (w[0] | w[1] | w[2] | w[3]) return false;
        }
        for (; i < n; ++i)
            if (p[i]) return false;
        return true;
    }
    std::uint64_t frame_ = ~std::uint64_t{0};
    std::unordered_map<std::uint64_t, Entry> entries_;
    long long scans_ = 0, hits_ = 0;
};

}  // namespace x2w_local
