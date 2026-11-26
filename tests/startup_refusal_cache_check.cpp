// src/startup_stubs/refusal_cache.h: 45,000 refused compute dispatches per frame (a pipeline with no translation) from four recording
// threads. The cached refusal never takes the global draw lock; the mutant (every refusal re-resolved under the lock, 0.3 ms each as
// measured in a tester run) must break the time bound. A published manifest (retry time passed, entry forgotten) is seen again.
// --mutate resolve-under-lock: the draw path ignores the cache.
#include "startup_stubs/refusal_cache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {
int failures;
void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}
std::mutex draw_lock;                 // stands for draw_lock
std::atomic<long> lock_acquisitions;  // refusals decided with the lock held
const char refusal[] = "no translation";

// Spends about `us` microseconds, as the descriptor walk under the lock did.
void busy_us(int us)
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    while (std::chrono::steady_clock::now() < end) {}
}
// One refused dispatch of pipeline `key`: the shape of record_translated_dispatch.
const char* dispatch(x2w_local::RefusalCache& cache, std::uint64_t key, std::uint64_t now_ms, bool use_cache)
{
    const char* reason;
    int kind;
    if (use_cache && cache.hit(key, now_ms, &reason, &kind)) return reason;
    std::lock_guard<std::mutex> lock(draw_lock);
    ++lock_acquisitions;
    busy_us(300);  // resolve under the lock
    cache.put(key, refusal, 1, now_ms + 250);
    return refusal;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "resolve-under-lock");
    x2w_local::RefusalCache cache;
    constexpr int threads = 4, per_thread = 11250;  // 45,000 dispatches
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t)
        workers.emplace_back([&, t] {
            for (int i = 0; i < (mutate ? per_thread / 10 : per_thread); ++i) dispatch(cache, 100 + (i + t) % 40, 1000, !mutate);
        });
    for (auto& w : workers) w.join();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    // The mutant runs a tenth of the dispatches and must still miss the bound: 4 x 1125 x 0.3 ms serialised is above 1 s.
    check("45,000 refused dispatches from 4 threads finish within 500 ms", ms < 500.0);
    check("the lock was taken once per pipeline, not once per dispatch", lock_acquisitions <= 40 * threads);

    // The entry expires at its retry time; the owner then resolves again (and can forget the entry once the manifest exists).
    x2w_local::RefusalCache timed;
    timed.put(7, refusal, 1, 2000);
    const char* reason;
    int kind;
    check("before the retry time the refusal is answered", timed.hit(7, 1999, &reason, &kind) && reason == refusal && kind == 1);
    check("at the retry time it is a miss (the owner looks again)", !timed.hit(7, 2000, &reason, &kind));
    timed.put(7, refusal, 1, 0);
    check("retry time 0 means until forgotten", timed.hit(7, 999999, &reason, &kind));
    timed.forget(7);
    check("a forgotten key is a miss", !timed.hit(7, 0, &reason, &kind));
    timed.put(8, refusal, 2, 0);
    timed.hit(8, 0, &reason, &kind);
    timed.hit(8, 0, &reason, &kind);
    check("hits are counted for the owner and taken once", timed.take_hits(8) == 2 && timed.take_hits(8) == 0);
    if (failures) return 1;
    std::printf("startup_refusal_cache: PASS (%.0f ms for 45,000 dispatches)\n", ms);
    return 0;
}
