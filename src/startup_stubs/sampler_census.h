#pragma once

#include "translated_sampler_bindings.h"

namespace x2w_startup {

// One process-wide census. The caller serializes access. Once the table fills,
// counts freeze at the last summary so repeats can never inflate the totals.
class SamplerCensus {
    static constexpr unsigned capacity = 8192;
    std::array<SamplerWords, capacity> seen_{};
    std::bitset<capacity> occupied_;

public:
    struct Counts {
        unsigned samplers, filter_min, filter_max, force_unnorm, force_degamma, skip_degamma,
                 trunc_coord, mc_trunc, no_cube_wrap, z_filter_point, z_filter_linear, lod_bias_sec;
    };

    const Counts& counts() const { return counts_; }

    // True when the caller should log the updated counts.
    bool observe(const SamplerWords& words)
    {
        if (counts_.samplers == capacity) return false;
        std::uint32_t hash = 2166136261u;
        for (auto word : words) {
            for (unsigned shift = 0; shift < 32; shift += 8)
                hash = (hash ^ ((word >> shift) & 255u)) * 16777619u;
        }
        unsigned slot = hash & (capacity - 1);
        while (occupied_.test(slot)) {
            if (seen_[slot] == words) return false;
            slot = (slot + 1) & (capacity - 1);
        }
        seen_[slot] = words;
        occupied_.set(slot);

        const auto w0 = words[0];
        const auto w2 = words[2];
        const auto filter_mode = (w0 >> 29) & 3u;
        const auto z_filter = (w2 >> 24) & 3u;
        counts_.filter_min += filter_mode == 1;
        counts_.filter_max += filter_mode == 2;
        counts_.force_unnorm += (w0 >> 15) & 1u;
        counts_.force_degamma += (w0 >> 20) & 1u;
        counts_.skip_degamma += w0 >> 31;
        counts_.trunc_coord += (w0 >> 27) & 1u;
        counts_.mc_trunc += (w0 >> 19) & 1u;
        counts_.no_cube_wrap += (w0 >> 28) & 1u;
        counts_.z_filter_point += z_filter == 1;
        counts_.z_filter_linear += z_filter == 2;
        counts_.lod_bias_sec += ((w2 >> 14) & 63u) != 0;
        const unsigned total = ++counts_.samplers;
        return (total & (total - 1)) == 0;
    }

private:
    Counts counts_{};
};

}  // namespace x2w_startup
