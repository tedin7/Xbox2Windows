// Experimental (docs/startup-runtime.md): where the time of a translated Dispatch / ExecuteIndirect goes, per phase, and
// how many threads queue for the draw lock. record_translated_dispatch holds that lock for the whole call, so a flood of
// dispatches from several recording threads serialises on it; these counters say whether the wait is long, which phase
// holds the lock, and how deep the queue gets. Counts are atomics, read once per report window.
#pragma once

#include <windows.h>

#include <cstdint>
#include <cstdio>

namespace x2w_phase {

enum Phase { lock_wait, resolve, values, record, hold, phase_count };  // hold = resolve + values + record (the lock held)
enum Kind { direct, indirect, refused, kind_count };                    // refused: ended before recording

inline const char* phase_name(int p)
{
    static const char* const names[phase_count] = {"lock-wait", "resolve", "values+upload", "record", "hold"};
    return names[p];
}
inline const char* kind_name(int k)
{
    static const char* const names[kind_count] = {"dispatch", "indirect", "refused"};
    return names[k];
}

inline std::uint64_t now_ticks() noexcept
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return static_cast<std::uint64_t>(t.QuadPart);
}
inline double ticks_per_us() noexcept
{
    static const double per_us = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return static_cast<double>(f.QuadPart) / 1e6;
    }();
    return per_us;
}

// Active only on the thread recording a measured copy. Existing lock timers
// can submit their elapsed ticks without taking a second timestamp.
struct LockWaitScope {
    inline static thread_local LockWaitScope* active = nullptr;
    LockWaitScope* previous = active;
    bool over_one_ms = false;

    LockWaitScope() noexcept { active = this; }
    ~LockWaitScope() { active = previous; }

    static void note(std::uint64_t ticks) noexcept
    {
        if (active && ticks > 1000 * ticks_per_us()) active->over_one_ms = true;
    }
};

inline void acquire_copy_lock(SRWLOCK* lock, bool exclusive = false) noexcept
{
    const auto start = LockWaitScope::active ? now_ticks() : 0;
    if (exclusive) AcquireSRWLockExclusive(lock);
    else AcquireSRWLockShared(lock);
    if (LockWaitScope::active) LockWaitScope::note(now_ticks() - start);
}

struct Totals {
    volatile LONG64 calls[kind_count] = {};
    volatile LONG64 ticks[kind_count][phase_count] = {};
    volatile LONG64 max_ticks[kind_count][phase_count] = {};
    volatile LONG queued = 0, queued_max = 0;  // threads between "asked for the lock" and "got it"

    void add(Kind kind, Phase phase, std::uint64_t t) noexcept
    {
        InterlockedExchangeAdd64(&ticks[kind][phase], static_cast<LONG64>(t));
        LONG64 seen = max_ticks[kind][phase];
        while (static_cast<LONG64>(t) > seen) {
            const LONG64 before = InterlockedCompareExchange64(&max_ticks[kind][phase], static_cast<LONG64>(t), seen);
            if (before == seen) break;
            seen = before;
        }
    }
    void enter() noexcept
    {
        const LONG depth = InterlockedIncrement(&queued);
        LONG seen = queued_max;
        while (depth > seen) {
            const LONG before = InterlockedCompareExchange(&queued_max, depth, seen);
            if (before == seen) break;
            seen = before;
        }
    }
    void leave() noexcept { InterlockedDecrement(&queued); }

    struct Window {
        std::uint64_t calls[kind_count] = {};
        double us[kind_count][phase_count] = {}, max_us[kind_count][phase_count] = {};
        LONG queued_max = 0;
    };
    // The numbers since the last take(), then a new window.
    Window take() noexcept
    {
        Window w;
        for (int k = 0; k < kind_count; ++k) {
            w.calls[k] = static_cast<std::uint64_t>(InterlockedExchange64(&calls[k], 0));
            for (int p = 0; p < phase_count; ++p) {
                w.us[k][p] = static_cast<double>(InterlockedExchange64(&ticks[k][p], 0)) / ticks_per_us();
                w.max_us[k][p] = static_cast<double>(InterlockedExchange64(&max_ticks[k][p], 0)) / ticks_per_us();
            }
        }
        w.queued_max = InterlockedExchange(&queued_max, queued);  // threads still queued count toward the next window
        return w;
    }
};

// One call's marks. start() before the lock is asked for, got_lock() once it is held, then one mark per phase, and
// finish() right before the lock is released.
struct Timer {
    std::uint64_t t_start = 0, t_lock = 0, t_resolve = 0, t_values = 0;
    Totals* totals = nullptr;

    void start(Totals& t) noexcept
    {
        totals = &t;
        t.enter();
        t_start = now_ticks();
    }
    void got_lock() noexcept
    {
        t_lock = now_ticks();
        totals->leave();
    }
    void resolved() noexcept { t_resolve = now_ticks(); }
    void valued() noexcept { t_values = now_ticks(); }
    void finish(Kind kind) noexcept
    {
        const std::uint64_t end = now_ticks();
        InterlockedIncrement64(&totals->calls[kind]);
        totals->add(kind, lock_wait, t_lock - t_start);
        // A refused call stops part-way: the marks it did not reach are the end time, so the time is not lost.
        const std::uint64_t r = t_resolve ? t_resolve : end, v = t_values ? t_values : end;
        totals->add(kind, resolve, r - t_lock);
        totals->add(kind, values, v - r);
        totals->add(kind, record, end - v);
        totals->add(kind, hold, end - t_lock);
    }
};

// "x2w-render dispatch phases (us: total / worst per call, calls): ..." for the kinds that ran in the window; the
// queue depth is the most threads waiting for the lock at once.
inline int format_window(const Totals::Window& w, char* out, std::size_t size) noexcept
{
    int n = std::snprintf(out, size, "x2w-render dispatch phases (queue max %ld)", static_cast<long>(w.queued_max));
    for (int k = 0; k < kind_count && n > 0 && static_cast<std::size_t>(n) < size; ++k) {
        if (!w.calls[k]) continue;
        n += std::snprintf(out + n, size - static_cast<std::size_t>(n), "; %s %llu calls:", kind_name(k), static_cast<unsigned long long>(w.calls[k]));
        for (int p = 0; p < phase_count && n > 0 && static_cast<std::size_t>(n) < size; ++p)
            n += std::snprintf(out + n, size - static_cast<std::size_t>(n), " %s %.0f/%.0f", phase_name(p), w.us[k][p], w.max_us[k][p]);
    }
    return n;
}

}  // namespace x2w_phase
