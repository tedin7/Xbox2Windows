// Page store for record + replay (docs/startup-runtime.md, fork tooling; recordings hold title data and stay local).
//
// PageWriter turns a captured memory range into chunks in the capture framing {char tag[4]; u32 header bytes; u64 data bytes;
// header; data}:
//   PAGE  header {key (16 B)}, data 4 KiB: one distinct non-zero page, written the first time its content is seen.
//   MAP   header {address, bytes, seq, run count}, data: runs {u32 first page, u32 pages, u32 kind} each followed by `pages` keys
//         (16 B) when kind is 0. A zero page run (kind 1) has no keys. Pages are compared with the newest MAP stamp for
//         their address. A conflict at the same stamp forces the next later capture to write the page again.
// ImageReader replays the chunks in order and rebuilds the memory image a reader would see at each MAP.
// Pure C++: no Windows calls, so a fixture can compare the rebuilt image byte for byte.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

namespace x2w_replay {

constexpr std::size_t page_bytes = 4096;

struct PageKey {
    std::uint64_t a = 0, b = 0;
    bool operator==(const PageKey& o) const { return a == o.a && b == o.b; }
};
struct PageKeyHash {
    std::size_t operator()(const PageKey& k) const { return static_cast<std::size_t>(k.a ^ (k.b * 0x9e3779b97f4a7c15ull)); }
};

// Two independent 64-bit hashes of the page (different seeds and multipliers): a collision needs both to match.
inline PageKey page_key(const unsigned char* page)
{
    std::uint64_t a = 1469598103934665603ull, b = 0x84222325cbf29ce4ull;
    for (std::size_t i = 0; i < page_bytes; i += 8) {
        std::uint64_t word;
        std::memcpy(&word, page + i, 8);
        a = (a ^ word) * 1099511628211ull;
        a ^= a >> 32;
        b = (b + word + 1) * 0xff51afd7ed558ccdull;
        b ^= b >> 29;
    }
    return {a, b};
}
inline bool page_is_zero(const unsigned char* page)
{
    for (std::size_t i = 0; i < page_bytes; i += 8) {
        std::uint64_t word;
        std::memcpy(&word, page + i, 8);
        if (word) return false;
    }
    return true;
}

struct PageStats {
    std::uint64_t seen = 0, zero = 0, unchanged = 0, distinct = 0, duplicate = 0, bytes_written = 0;
};

struct MapHeader {
    std::uint64_t address, bytes, seq;  // seq: the recorder's call sequence number when the range was read
    std::uint32_t runs, pad;
};
struct RunHeader {
    std::uint32_t first, pages, kind;  // kind 0: keys follow; 1: zero pages
};

class PageWriter {
public:
    using Sink = std::function<void(const char* tag, const void* header, std::size_t header_bytes, const void* data, std::size_t data_bytes)>;
    explicit PageWriter(Sink sink) : sink_(std::move(sink)) {}

    // address and bytes must be multiples of the page size (title ranges come from VirtualAlloc). False: nothing written.
    // PAGE chunks for content never written before go to the sink first, then one MAP chunk with the changed runs.
    // Track the newest MAP stamp per page. Older writes leave it intact, and conflicting equal stamps
    // force the next later capture to write the page again.
    bool capture(std::uint64_t address, const void* data, std::size_t bytes, std::uint64_t seq)
    {
        if (!bytes || address % page_bytes || bytes % page_bytes) return false;
        const auto* base = static_cast<const unsigned char*>(data);
        const std::size_t pages = bytes / page_bytes;
        std::vector<unsigned char> runs;
        std::uint32_t run_count = 0;
        std::size_t first = 0;
        std::vector<PageKey> keys;
        bool open = false, run_zero = false;
        const auto close_run = [&] {
            if (!open) return;
            const RunHeader run{static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(run_zero ? run_pages_ : keys.size()), run_zero ? 1u : 0u};
            const auto* r = reinterpret_cast<const unsigned char*>(&run);
            runs.insert(runs.end(), r, r + sizeof(run));
            if (!run_zero) {
                const auto* k = reinterpret_cast<const unsigned char*>(keys.data());
                runs.insert(runs.end(), k, k + keys.size() * sizeof(PageKey));
            }
            ++run_count;
            keys.clear();
            open = false;
        };
        for (std::size_t i = 0; i < pages; ++i) {
            const unsigned char* page = base + i * page_bytes;
            const std::uint64_t at = address + i * page_bytes;
            ++stats_.seen;
            const bool zero = page_is_zero(page);
            const PageKey key = zero ? PageKey{} : page_key(page);
            const auto last = last_.find(at);
            if (last != last_.end() && last->second.zero == zero && last->second.key == key &&
                !(seq > last->second.seq && last->second.uncertain)) {
                ++stats_.unchanged;
                close_run();
                continue;
            }
            if (last == last_.end() || seq >= last->second.seq)
                last_[at] = {key, zero, seq, last != last_.end() && seq == last->second.seq};
            if (open && run_zero != zero) close_run();
            if (!open) { open = true; run_zero = zero; first = i; run_pages_ = 0; }
            ++run_pages_;
            if (zero) { ++stats_.zero; continue; }
            keys.push_back(key);
            if (written_.insert({key, true}).second) {
                ++stats_.distinct;
                sink_("PAGE", &key, sizeof(key), page, page_bytes);
                stats_.bytes_written += page_bytes;
            } else {
                ++stats_.duplicate;
            }
        }
        close_run();
        if (!run_count) return true;  // no page changes need a MAP at this stamp
        MapHeader header{address, bytes, seq, run_count, 0};
        sink_("MAP ", &header, sizeof(header), runs.data(), runs.size());
        return true;
    }

    const PageStats& stats() const { return stats_; }

private:
    struct Last { PageKey key; bool zero; std::uint64_t seq; bool uncertain; };
    Sink sink_;
    PageStats stats_;
    std::size_t run_pages_ = 0;
    std::unordered_map<std::uint64_t, Last> last_;
    std::unordered_map<PageKey, bool, PageKeyHash> written_;
};

class ImageReader {
public:
    // One chunk in file order. False: the chunk is malformed (a MAP naming a page never stored, a short run list).
    bool apply(const char* tag, const void* header, std::size_t header_bytes, const void* data, std::size_t data_bytes)
    {
        if (!std::memcmp(tag, "PAGE", 4)) {
            if (header_bytes != sizeof(PageKey) || data_bytes != page_bytes) return false;
            PageKey key;
            std::memcpy(&key, header, sizeof(key));
            auto& blob = pages_[key];
            blob.assign(static_cast<const unsigned char*>(data), static_cast<const unsigned char*>(data) + page_bytes);
            return true;
        }
        if (!std::memcmp(tag, "MAP ", 4)) {
            if (header_bytes != sizeof(MapHeader)) return false;
            MapHeader map;
            std::memcpy(&map, header, sizeof(map));
            const auto* p = static_cast<const unsigned char*>(data);
            std::size_t left = data_bytes;
            for (std::uint32_t r = 0; r < map.runs; ++r) {
                RunHeader run;
                if (left < sizeof(run)) return false;
                std::memcpy(&run, p, sizeof(run));
                p += sizeof(run); left -= sizeof(run);
                if (std::uint64_t{run.first} + run.pages > map.bytes / page_bytes) return false;
                for (std::uint32_t n = 0; n < run.pages; ++n) {
                    const std::uint64_t at = map.address + (std::uint64_t{run.first} + n) * page_bytes;
                    if (run.kind == 1) { image_[at] = PageKey{}; zero_.emplace(at, true); continue; }
                    if (left < sizeof(PageKey)) return false;
                    PageKey key;
                    std::memcpy(&key, p, sizeof(key));
                    p += sizeof(key); left -= sizeof(key);
                    if (!pages_.count(key)) return false;
                    image_[at] = key;
                    zero_.erase(at);
                }
            }
            return left == 0;
        }
        return true;  // other chunk kinds belong to other readers
    }

    // The bytes of [address, address+bytes) as the image holds them (page aligned); false when a page was never mapped.
    bool read(std::uint64_t address, void* out, std::size_t bytes) const
    {
        if (address % page_bytes || bytes % page_bytes) return false;
        auto* dst = static_cast<unsigned char*>(out);
        for (std::size_t at = 0; at < bytes; at += page_bytes) {
            const auto it = image_.find(address + at);
            if (it == image_.end()) return false;
            if (zero_.count(address + at)) std::memset(dst + at, 0, page_bytes);
            else std::memcpy(dst + at, pages_.at(it->second).data(), page_bytes);
        }
        return true;
    }

private:
    std::unordered_map<PageKey, std::vector<unsigned char>, PageKeyHash> pages_;
    std::unordered_map<std::uint64_t, PageKey> image_;
    std::unordered_map<std::uint64_t, bool> zero_;
};

}  // namespace x2w_replay
