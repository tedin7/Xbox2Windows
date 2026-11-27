// Census of depth-stencil clears (list slot 51) and of draws whose decoded depth test is GREATER or GREATER_EQUAL
// (X2W_DEPTH_STATE), for a title whose depth looks wrong: which depth values it clears to, which clears the stand-in
// skipped, and whether a GREATER test meets a depth view last cleared to 1.0 (the desktop far value: nothing passes)
// or never cleared. A line goes to the sink each time a count reaches a power of two, so a long run prints few; the
// first 8 distinct (depth, flags) pairs are counted each, later ones together. Thread-safe (one SRW lock).
#pragma once

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace x2w_depth_census {

enum class Skip { unresolved_view, zero_flags, read_only_depth, read_only_stencil, stencil_without_stencil_format, unreadable_rects, count };
enum class LastClear { far_one, near_zero, other, never, count };
constexpr int distinct_pairs = 8;
constexpr unsigned clear_depth_flag = 1;  // D3D12_CLEAR_FLAG_DEPTH
constexpr int func_greater = 5, func_greater_equal = 7;  // D3D12_COMPARISON_FUNC_GREATER, _GREATER_EQUAL
using Sink = void (*)(const char* line);

inline bool power_of_two(std::uint64_t n)
{
    return n && !(n & (n - 1));
}

inline const char* skip_name(Skip why)
{
    static const char* const names[] = {"view not resolved", "zero clear flags", "depth clear on a read-only depth view",
                                        "stencil clear on a read-only stencil view", "stencil only, format without stencil",
                                        "unreadable rects"};
    return names[static_cast<int>(why)];
}

class DepthClearCensus {
public:
    explicit DepthClearCensus(Sink sink) : sink_(sink) {}

    // Every clear the title asks for, before the stand-in decides whether it runs.
    void requested(float depth, unsigned flags)
    {
        std::uint32_t bits;
        std::memcpy(&bits, &depth, sizeof(bits));
        char line[200];
        bool emit = false;
        AcquireSRWLockExclusive(&lock_);
        int slot = 0;
        while (slot < pairs_used_ && !(pairs_[slot].bits == bits && pairs_[slot].flags == flags)) ++slot;
        if (slot == pairs_used_ && pairs_used_ < distinct_pairs) pairs_[pairs_used_++] = {bits, flags, 0};
        if (slot < pairs_used_) {
            const std::uint64_t n = ++pairs_[slot].count;
            emit = power_of_two(n);
            if (emit)
                std::snprintf(line, sizeof(line), "x2w-render depth clear census: depth=%.6g flags=0x%x requested %llu times (pair %d of the first %d)",
                              depth, flags, static_cast<unsigned long long>(n), slot + 1, distinct_pairs);
        } else {
            const std::uint64_t n = ++other_pairs_;
            emit = power_of_two(n);
            if (emit)
                std::snprintf(line, sizeof(line), "x2w-render depth clear census: clears with a (depth, flags) pair after the first %d requested %llu times",
                              distinct_pairs, static_cast<unsigned long long>(n));
        }
        ReleaseSRWLockExclusive(&lock_);
        if (emit) sink_(line);
    }

    // A clear the stand-in did not pass on.
    void skipped(Skip why)
    {
        char line[160];
        AcquireSRWLockExclusive(&lock_);
        const std::uint64_t n = ++skipped_[static_cast<int>(why)];
        ReleaseSRWLockExclusive(&lock_);
        if (!power_of_two(n)) return;
        std::snprintf(line, sizeof(line), "x2w-render depth clear census: skipped (%s) %llu times", skip_name(why),
                      static_cast<unsigned long long>(n));
        sink_(line);
    }

    // A clear passed on to the desktop list: a depth clear sets what its target was last cleared to.
    void executed(const void* target, float depth, unsigned flags)
    {
        if (!target || !(flags & clear_depth_flag)) return;
        const LastClear value = depth == 1.0f ? LastClear::far_one : depth == 0.0f ? LastClear::near_zero : LastClear::other;
        AcquireSRWLockExclusive(&lock_);
        try {
            last_[target] = value;
        } catch (...) {  // out of memory: the target stays as it was
        }
        ReleaseSRWLockExclusive(&lock_);
    }

    // What a depth view was last cleared to; never when no clear reached it.
    LastClear last_clear(const void* target)
    {
        if (!target) return LastClear::never;
        AcquireSRWLockShared(&lock_);
        const auto found = last_.find(target);
        const LastClear value = found == last_.end() ? LastClear::never : found->second;
        ReleaseSRWLockShared(&lock_);
        return value;
    }

    // A draw with a decoded depth test on `target` (the bound depth resource); only GREATER and GREATER_EQUAL count.
    void depth_draw(const void* target, int func)
    {
        if (!target || (func != func_greater && func != func_greater_equal)) return;
        char line[240];
        AcquireSRWLockExclusive(&lock_);
        const auto found = last_.find(target);
        ++greater_[static_cast<int>(found == last_.end() ? LastClear::never : found->second)];
        const std::uint64_t total = ++greater_total_;
        const bool emit = power_of_two(total);
        if (emit)
            std::snprintf(line, sizeof(line),
                          "x2w-render depth clear census: %llu draws with a decoded GREATER/GREATER_EQUAL depth test: on a view last "
                          "cleared to 1.0 %llu, to 0.0 %llu, to another value %llu, never cleared %llu",
                          static_cast<unsigned long long>(total), static_cast<unsigned long long>(greater_[0]),
                          static_cast<unsigned long long>(greater_[1]), static_cast<unsigned long long>(greater_[2]),
                          static_cast<unsigned long long>(greater_[3]));
        ReleaseSRWLockExclusive(&lock_);
        if (emit) sink_(line);
    }

private:
    struct Pair {
        std::uint32_t bits;
        unsigned flags;
        std::uint64_t count;
    };
    Sink sink_;
    SRWLOCK lock_ = SRWLOCK_INIT;
    Pair pairs_[distinct_pairs]{};
    int pairs_used_ = 0;
    std::uint64_t other_pairs_ = 0;
    std::uint64_t skipped_[static_cast<int>(Skip::count)]{};
    std::unordered_map<const void*, LastClear> last_;
    std::uint64_t greater_[static_cast<int>(LastClear::count)]{};
    std::uint64_t greater_total_ = 0;
};

}  // namespace x2w_depth_census
