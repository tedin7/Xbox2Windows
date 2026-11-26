#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace x2w_local {

// Per acquiring site (function and line of a lock acquire): how long its holds took and how long it waited. The table has no lock of
// its own: it is written only by the thread that holds the lock it describes, and read (then cleared) by a caller that holds it too.
struct LockSite {
    const char* function;
    unsigned line;
    std::int64_t holds, hold_ticks, hold_max, wait_ticks;
};
constexpr std::size_t lock_site_slots = 128;

class LockSites {
public:
    void waited(const char* function, unsigned line, std::int64_t ticks)
    {
        if (LockSite* site = find(function, line)) site->wait_ticks += ticks;
    }
    void held(const char* function, unsigned line, std::int64_t ticks)
    {
        if (LockSite* site = find(function, line)) {
            ++site->holds;
            site->hold_ticks += ticks;
            if (ticks > site->hold_max) site->hold_max = ticks;
        }
    }
    // Top six sites by total hold time, one line; clears the table. False when nothing was held.
    bool top_line(char* out, std::size_t size, double ticks_per_ms)
    {
        LockSite copy[lock_site_slots];
        std::memcpy(copy, sites_, sizeof(copy));
        std::memset(sites_, 0, sizeof(sites_));
        int order[lock_site_slots], count = 0;
        for (int i = 0; i < static_cast<int>(lock_site_slots); ++i)
            if (copy[i].function && copy[i].holds) order[count++] = i;
        if (!count) return false;
        std::sort(order, order + count, [&](int a, int b) { return copy[a].hold_ticks > copy[b].hold_ticks; });
        int n = std::snprintf(out, size, "x2w-render draw_lock top holders (since the last such line; function:line holds, hold ms total/max, wait ms):");
        for (int k = 0; k < count && k < 6 && n > 0 && static_cast<std::size_t>(n) < size; ++k) {
            const LockSite& site = copy[order[k]];
            n += std::snprintf(out + n, size - n, " [%s:%u %lld, %.0f/%.1f, %.0f]", site.function, site.line, static_cast<long long>(site.holds),
                               site.hold_ticks / ticks_per_ms, site.hold_max / ticks_per_ms, site.wait_ticks / ticks_per_ms);
        }
        return true;
    }

private:
    LockSite* find(const char* function, unsigned line)
    {
        std::size_t at = (reinterpret_cast<std::uintptr_t>(function) / 8 * 31 + line) % lock_site_slots;
        for (std::size_t probe = 0; probe < lock_site_slots; ++probe, at = (at + 1) % lock_site_slots) {
            LockSite& site = sites_[at];
            if (!site.function) {
                site.function = function;
                site.line = line;
                return &site;
            }
            if (site.function == function && site.line == line) return &site;
        }
        return nullptr;
    }
    LockSite sites_[lock_site_slots]{};
};

}  // namespace x2w_local
