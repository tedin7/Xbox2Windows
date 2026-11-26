#pragma once

#include <windows.h>

namespace x2w_startup {

struct PipelineCreation {
    CONDITION_VARIABLE ready = CONDITION_VARIABLE_INIT;
    bool pending = false;
    bool finished = false;

    template<class Wait>
    void wait(Wait&& sleep)
    {
        while (pending) sleep(ready);
    }

    void finish()
    {
        finished = true;
        if (pending) {
            pending = false;
            WakeAllConditionVariable(&ready);
        }
    }
};

struct PipelineCreationTotals {
    LONG64 created = 0, failed = 0, outside = 0, ticks = 0, max_ticks = 0;

    // The caller holds the cache lock when recording and reading these counters.
    void record(HRESULT hr, bool unlocked, LONG64 elapsed)
    {
        if (SUCCEEDED(hr)) ++created;
        else ++failed;
        if (unlocked) ++outside;
        ticks += elapsed;
        if (elapsed > max_ticks) max_ticks = elapsed;
    }
};

// The entry is stable across insertions and lives until all users are done.
// The caller publishes the result and wakes waiters while holding the lock.
template<class Release, class Acquire, class Create>
HRESULT create_pipeline(PipelineCreation& entry, bool unlocked, PipelineCreationTotals& totals,
                        Release&& release, Acquire&& acquire, Create&& create)
{
    if (unlocked) {
        entry.pending = true;
        release();
    }
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    const HRESULT hr = create();
    QueryPerformanceCounter(&end);
    if (unlocked) acquire();
    totals.record(hr, unlocked, end.QuadPart - start.QuadPart);
    return hr;
}

}  // namespace x2w_startup
