#pragma once

#include <bitset>
#include <optional>

namespace x2w_startup {

// The caller serializes access to each pool. Runs stay occupied until released.
template <unsigned SegmentCount>
class DescriptorRingSegments {
public:
    struct Run { unsigned first, count; };

    std::optional<Run> acquire(unsigned count)
    {
        if (!count || count > free_count()) return std::nullopt;
        unsigned consecutive = 0;
        for (unsigned i = first_free_; i < SegmentCount; ++i) {
            consecutive = held_[i] ? 0 : consecutive + 1;
            if (consecutive == count) {
                const Run run{i + 1 - count, count};
                for (unsigned j = run.first; j <= i; ++j) held_.set(j);
                used_ += count;
                while (first_free_ < SegmentCount && held_[first_free_]) ++first_free_;
                return run;
            }
        }
        return std::nullopt;
    }

    void release(Run run)
    {
        for (unsigned i = run.first; i < run.first + run.count; ++i) held_.reset(i);
        used_ -= run.count;
        if (run.first < first_free_) first_free_ = run.first;
    }

    unsigned used_count() const { return used_; }
    unsigned free_count() const { return SegmentCount - used_; }

private:
    std::bitset<SegmentCount> held_;
    unsigned used_ = 0;
    unsigned first_free_ = 0;
};

}  // namespace x2w_startup
