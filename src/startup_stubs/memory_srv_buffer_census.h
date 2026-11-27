#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace x2w_startup {

enum class BufferRefusalReason {
    page_copy_stride, rounded_overlap, placed_stride, title_overlap, extent,
    not_covered, no_device, backing_failed, wrap_failed, view_fit,
    committed_stride, page_alias_stride, exact_copy_stride, count
};

struct BufferRefusalDetail {
    BufferRefusalReason reason = BufferRefusalReason::view_fit;
    std::uint64_t base = 0;
    std::uint64_t bytes = 0;
};

enum class BufferSrvBacking { committed, placed, page_copy, page_alias, exact_copy };

inline BufferRefusalDetail buffer_view_fit_refusal(std::uint64_t address, std::uint64_t stride,
    std::uint64_t base, std::uint64_t bytes, BufferSrvBacking backing)
{
    using Reason = BufferRefusalReason;
    constexpr Reason stride_reasons[] = {
        Reason::committed_stride, Reason::placed_stride, Reason::page_copy_stride,
        Reason::page_alias_stride, Reason::exact_copy_stride
    };
    const bool stride_mismatch = stride && address >= base && address - base < bytes &&
                                 (address - base) % stride != 0;
    return {stride_mismatch ? stride_reasons[static_cast<unsigned>(backing)] : Reason::view_fit, base, bytes};
}

inline BufferRefusalDetail buffer_overlap_refusal(std::uint64_t address, std::uint64_t span,
    std::uint64_t base, std::uint64_t bytes)
{
    using Reason = BufferRefusalReason;
    const bool touches = bytes && span &&
        (address < base ? base - address < span : address - base < bytes);
    return {touches ? Reason::title_overlap : Reason::rounded_overlap, base, bytes};
}

// Call under the existing unplaced-buffer mutex. Fixed open-addressed tables give exact
// counts until a probe limit is reached. After that the reported counts and cost are lower bounds.
class BufferSrvRefusalCensus {
public:
    static constexpr std::size_t slots = 4096;

    // Returns true at powers of two. Emit the example immediately so later reports stay compact.
    template<class Emit>
    bool note(BufferRefusalDetail detail, std::uint64_t address, std::uint64_t span, std::uint64_t stride, Emit emit)
    {
        const auto reason = static_cast<std::size_t>(detail.reason);
        const auto page = address & ~std::uint64_t{65535};
        if (++counts_[reason] <= 4) {
            char line[256];
            std::snprintf(line, sizeof(line),
                "x2w-slot70 buffer_unavailable example: reason=%c address=0x%llx size=%llu stride=%llu "
                "page_base=0x%llx conflict_base=0x%llx conflict_size=%llu",
                static_cast<int>('A' + reason), ull(address), ull(span), ull(stride), ull(page),
                ull(detail.base), ull(detail.bytes));
            emit(line);
        }

        const auto pages = (span >> 16) + ((span & 65535) != 0);
        auto* entry = find(addresses_, hash(address), Address{address, pages});
        if (!entry) {
            addresses_truncated_ = true;
        } else {
            if (!entry->used) {
                ++distinct_addresses_;
                entry->used = true;
                entry->value.address = address;
            }
            // A single exact copy for this start would need to cover the largest observed span.
            if (pages > entry->value.pages) {
                exact_copy_pages_ += pages - entry->value.pages;
                entry->value.pages = pages;
            }
        }

        const auto remainder = stride ? (address - page) % stride : 0;
        const Key key{page, stride, remainder};
        auto* key_entry = find(keys_, hash((page >> 16) ^ hash((stride << 16) ^ remainder)), key);
        if (!key_entry) {
            keys_truncated_ = true;
        } else if (!key_entry->used) {
            *key_entry = {true, key};
            ++distinct_keys_;
        }
        ++total_;
        return (total_ & (total_ - 1)) == 0;
    }

    template<class Emit>
    void report(Emit emit) const
    {
        char line[640];
        const auto& c = counts_;
        std::snprintf(line, sizeof(line),
            "x2w-slot70 buffer_unavailable split: A=%llu B=%llu C=%llu D=%llu "
            "E_extent=%llu F_not_covered=%llu G_no_device=%llu H_backing_failed=%llu "
            "I_wrap_failed=%llu J_view_fit=%llu K_committed_stride=%llu L_page_alias_stride=%llu M_exact_copy_stride=%llu "
            "distinct_addresses=%zu distinct_keys=%zu exact_copy_cost_mib=%.4f "
            "addresses_truncated=%u keys_truncated=%u",
            ull(c[0]), ull(c[1]), ull(c[2]), ull(c[3]), ull(c[4]), ull(c[5]), ull(c[6]), ull(c[7]),
            ull(c[8]), ull(c[9]), ull(c[10]), ull(c[11]), ull(c[12]), distinct_addresses_, distinct_keys_,
            static_cast<double>(exact_copy_pages_) / 16.0, unsigned(addresses_truncated_), unsigned(keys_truncated_));
        emit(line);
    }

private:
    struct Address {
        std::uint64_t address, pages;
        bool operator==(const Address& other) const { return address == other.address; }
    };

    struct Key {
        std::uint64_t page, stride, remainder;
        bool operator==(const Key& other) const
        {
            return page == other.page && stride == other.stride && remainder == other.remainder;
        }
    };

    template<class T>
    struct Entry {
        bool used;
        T value;
    };

    static unsigned long long ull(std::uint64_t n) { return static_cast<unsigned long long>(n); }

    static std::uint64_t hash(std::uint64_t n)
    {
        n ^= n >> 30;
        n *= 0xbf58476d1ce4e5b9ull;
        n ^= n >> 27;
        n *= 0x94d049bb133111ebull;
        return n ^ (n >> 31);
    }

    template<class T>
    static Entry<T>* find(std::array<Entry<T>, slots>& table, std::uint64_t start, const T& value)
    {
        for (std::size_t i = 0; i < 16; ++i) {
            auto& entry = table[(start + i) & (slots - 1)];
            if (!entry.used || entry.value == value) return &entry;
        }
        return nullptr;
    }

    std::array<Entry<Address>, slots> addresses_{};
    std::array<Entry<Key>, slots> keys_{};
    std::array<std::uint64_t, static_cast<std::size_t>(BufferRefusalReason::count)> counts_{};
    std::uint64_t total_ = 0, exact_copy_pages_ = 0;
    std::size_t distinct_addresses_ = 0, distinct_keys_ = 0;
    bool addresses_truncated_ = false, keys_truncated_ = false;
};

} // namespace x2w_startup
