// Census of depth-stencil clears (list slot 51) and of draws whose decoded depth test is GREATER or GREATER_EQUAL
// (X2W_DEPTH_STATE), for a title whose depth looks wrong: which depth values it clears to, which clears the stand-in
// skipped, and whether a GREATER test meets a depth view last cleared to 1.0 (the desktop far value: nothing passes)
// or never cleared. A line goes to the sink each time a count reaches a power of two, so a long run prints few; the
// first 8 distinct (depth, flags) pairs are counted each, later ones together. The draws that write depth are counted by
// the state that decides whether they leave pixels (the depth resource, test, viewport, scissor, cull, clip, the view's
// last clear), the first distinct_writes alone and later ones together.
// Thread-safe (one SRW lock).
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

// A draw that writes depth (debug present's "depth writes" class): the state that decides whether it leaves pixels.
struct DepthWrite {
    std::uint64_t width = 0;  // the bound depth resource
    unsigned height = 0;
    int format = 0;  // its DSV format (DXGI_FORMAT)
    int func = 0;    // D3D12_COMPARISON_FUNC
    bool viewport_set = false;  // false: the stand-in's default, the full target with depth 0..1
    float vp_x = 0, vp_y = 0, vp_width = 0, vp_height = 0, min_depth = 0, max_depth = 1;  // viewport 0
    bool scissor_set = false;  // false: the stand-in's default, the full target
    long sc_left = 0, sc_top = 0, sc_right = 0, sc_bottom = 0;  // scissor rect 0
    int cull = 1;  // D3D12_CULL_MODE: 1 none, 2 front, 3 back
    bool front_ccw = false;
    bool depth_clip = true;
};
constexpr int distinct_writes = 32;

inline const char* last_clear_name(LastClear clear)
{
    static const char* const names[] = {"1.0", "0.0", "another value", "never (no depth clear seen)"};
    return names[static_cast<int>(clear)];
}

inline const char* func_name(int func)
{
    static const char* const names[] = {"none", "NEVER", "LESS", "EQUAL", "LESS_EQUAL", "GREATER", "NOT_EQUAL", "GREATER_EQUAL", "ALWAYS"};
    return func >= 0 && func <= 8 ? names[func] : "unknown";
}

inline const char* cull_name(int cull)
{
    static const char* const names[] = {"none", "front", "back"};
    return cull >= 1 && cull <= 3 ? names[cull - 1] : "unknown";
}

// Whether a fragment of a draw with this test can pass against the value the view was cleared to. A fragment's depth
// stays inside the viewport depth range (clipped or clamped), so a LESS test on a view cleared to 0.0 (reversed Z) or
// a GREATER test on one cleared to 1.0 passes nowhere. Only the first draw after the clear meets the cleared value;
// only 0.0 and 1.0 clears are known values.
inline const char* clear_test_verdict(int func, LastClear clear, float min_depth, float max_depth)
{
    if (func == 1) return "no (NEVER)";
    if (func == 8) return "yes (ALWAYS)";
    if (clear != LastClear::far_one && clear != LastClear::near_zero) return "unknown";
    const float c = clear == LastClear::far_one ? 1.0f : 0.0f;
    const float lo = min_depth < max_depth ? min_depth : max_depth, hi = min_depth < max_depth ? max_depth : min_depth;
    const char* const only = c == 0.0f ? "only at depth 0" : "only at depth 1";
    switch (func) {
    case 2: return lo < c ? "yes" : "no";                   // LESS
    case 3: return lo <= c && c <= hi ? only : "no";        // EQUAL
    case 4: return lo < c ? "yes" : lo == c ? only : "no";  // LESS_EQUAL
    case 5: return hi > c ? "yes" : "no";                   // GREATER
    case 6: return lo == c && hi == c ? "no" : "yes";       // NOT_EQUAL
    case 7: return hi > c ? "yes" : hi == c ? only : "no";  // GREATER_EQUAL
    default: return "unknown";
    }
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

    // A draw that writes depth on `target` (the bound depth resource): its distinct states, each with the value the view
    // was last cleared to, counted (the first distinct_writes alone, later ones together).
    void depth_write_draw(const void* target, const DepthWrite& w)
    {
        if (!target) return;
        char line[640];
        AcquireSRWLockExclusive(&lock_);
        const auto found = last_.find(target);
        const LastClear clear = found == last_.end() ? LastClear::never : found->second;
        int slot = 0;
        while (slot < writes_used_ &&
               !(writes_[slot].target == target && writes_[slot].clear == clear && same_write(writes_[slot].state, w)))
            ++slot;
        if (slot == writes_used_ && writes_used_ < distinct_writes) writes_[writes_used_++] = {target, w, clear, 0};
        const bool alone = slot < writes_used_;
        const std::uint64_t n = alone ? ++writes_[slot].count : ++other_writes_;
        const bool emit = power_of_two(n);
        if (emit && alone) write_line(line, sizeof(line), target, w, clear, n, slot);
        if (emit && !alone)
            std::snprintf(line, sizeof(line), "x2w-render depth writer census: depth-writing draws with a state after the first %d: %llu",
                          distinct_writes, static_cast<unsigned long long>(n));
        ReleaseSRWLockExclusive(&lock_);
        if (emit) sink_(line);
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
    static bool same_write(const DepthWrite& a, const DepthWrite& b)
    {
        const bool viewport = a.viewport_set == b.viewport_set &&
                              (!a.viewport_set || (a.vp_x == b.vp_x && a.vp_y == b.vp_y && a.vp_width == b.vp_width &&
                                                   a.vp_height == b.vp_height && a.min_depth == b.min_depth && a.max_depth == b.max_depth));
        const bool scissor = a.scissor_set == b.scissor_set &&
                             (!a.scissor_set || (a.sc_left == b.sc_left && a.sc_top == b.sc_top && a.sc_right == b.sc_right &&
                                                 a.sc_bottom == b.sc_bottom));
        return viewport && scissor && a.width == b.width && a.height == b.height && a.format == b.format && a.func == b.func &&
               a.cull == b.cull && a.front_ccw == b.front_ccw && a.depth_clip == b.depth_clip;
    }
    // x2w-render depth writer census: depth <resource> <w>x<h> format <dxgi>, func <test>, viewport <x>,<y> <w>x<h> depth
    // <min>..<max>, scissor <l>,<t>..<r>,<b>, cull <mode> (front <ccw|cw>), depth clip <0|1>, view last cleared to <value>,
    // a fragment can pass the test against the clear: <yes|no|only at depth N|unknown>: <n> draws (state <k> of the first 32)
    static void write_line(char* line, std::size_t size, const void* target, const DepthWrite& w, LastClear clear,
                           std::uint64_t n, int slot)
    {
        char viewport[128] = "none set (the full target, depth 0..1)";
        char scissor[96] = "none set (the full target)";
        if (w.viewport_set)
            std::snprintf(viewport, sizeof(viewport), "%g,%g %gx%g depth %g..%g", w.vp_x, w.vp_y, w.vp_width, w.vp_height,
                          w.min_depth, w.max_depth);
        if (w.scissor_set) std::snprintf(scissor, sizeof(scissor), "%ld,%ld..%ld,%ld", w.sc_left, w.sc_top, w.sc_right, w.sc_bottom);
        const float min_depth = w.viewport_set ? w.min_depth : 0.0f, max_depth = w.viewport_set ? w.max_depth : 1.0f;
        std::snprintf(line, size,
                      "x2w-render depth writer census: depth %p %llux%u format %d, func %s, viewport %s, scissor %s, cull %s "
                      "(front %s), depth clip %d, view last cleared to %s, a fragment can pass the test against the clear: %s: "
                      "%llu draws (state %d of the first %d)",
                      target, static_cast<unsigned long long>(w.width), w.height, w.format, func_name(w.func), viewport, scissor,
                      cull_name(w.cull), w.front_ccw ? "ccw" : "cw", w.depth_clip ? 1 : 0, last_clear_name(clear),
                      clear_test_verdict(w.func, clear, min_depth, max_depth), static_cast<unsigned long long>(n), slot + 1,
                      distinct_writes);
    }
    struct Write {
        const void* target;
        DepthWrite state;
        LastClear clear;
        std::uint64_t count;
    };
    Write writes_[distinct_writes]{};
    int writes_used_ = 0;
    std::uint64_t other_writes_ = 0;
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
