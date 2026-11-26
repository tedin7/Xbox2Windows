// Per-phase dispatch counters (src/startup_stubs/dispatch_phases.h) and the multi-thread scaling method for the lock that
// record_translated_dispatch holds.
//
// A dispatch is modelled as `resolve` work (reads the title's descriptors; needs the shared draw state) followed by `record`
// work (writes the recording thread's own command list). `single` holds one SRWLOCK for both, as the stand-in does today;
// `split` holds it for resolve only. Both run through the real Timer / Totals, so the counters are checked under contention:
// one thread waits for nothing, four threads on the single lock queue three deep and gain no throughput, and the split model
// scales because record runs in parallel. Usage:
//   dispatch_phases_check                      the whole suite
//   dispatch_phases_check <model> <threads> <min speedup over one thread>   one run; exit 1 below the speedup (a control)
#include "startup_stubs/dispatch_phases.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int resolve_us = 30, record_us = 70, iterations = 400;
int failures = 0;

void require(bool ok, const char* what)
{
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

void spin_us(int us)
{
    const auto until = x2w_phase::now_ticks() + static_cast<std::uint64_t>(us * x2w_phase::ticks_per_us());
    while (x2w_phase::now_ticks() < until) {
    }
}

struct Run {
    double seconds = 0;
    x2w_phase::Totals::Window window;
};

Run run(bool split, int threads)
{
    static SRWLOCK lock = SRWLOCK_INIT;
    static x2w_phase::Totals totals;
    (void)totals.take();
    std::vector<std::thread> pool;
    const auto began = x2w_phase::now_ticks();
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            for (int i = 0; i < iterations; ++i) {
                x2w_phase::Timer timer;
                timer.start(totals);
                AcquireSRWLockExclusive(&lock);
                timer.got_lock();
                spin_us(resolve_us);
                timer.resolved();
                timer.valued();
                if (split) {
                    timer.finish(x2w_phase::direct);  // the lock covers resolve only
                    ReleaseSRWLockExclusive(&lock);
                    spin_us(record_us);
                } else {
                    spin_us(record_us);
                    timer.finish(x2w_phase::direct);
                    ReleaseSRWLockExclusive(&lock);
                }
            }
        });
    for (auto& thread : pool) thread.join();
    Run out;
    out.seconds = static_cast<double>(x2w_phase::now_ticks() - began) / (x2w_phase::ticks_per_us() * 1e6);
    out.window = totals.take();
    return out;
}

double speedup(bool split, int threads, Run* one_out = nullptr, Run* many_out = nullptr)
{
    const Run one = run(split, 1), many = run(split, threads);
    if (one_out) *one_out = one;
    if (many_out) *many_out = many;
    // Work done per second with `threads` threads over one thread's rate.
    return (threads * iterations / many.seconds) / (iterations / one.seconds);
}

void suite()
{
    Run one, four;
    const double serial = speedup(false, 4, &one, &four);
    const auto direct = static_cast<int>(x2w_phase::direct);
    std::printf("single lock: 1 thread %.3f s, 4 threads %.3f s, speedup %.2f, queue max %ld, wait/call %.1f us\n", one.seconds, four.seconds,
                serial, static_cast<long>(four.window.queued_max), four.window.us[direct][x2w_phase::lock_wait] / four.window.calls[direct]);
    require(one.window.calls[direct] == iterations && four.window.calls[direct] == 4u * iterations, "every call is counted once");
    require(one.window.queued_max == 1, "one thread never queues behind anyone");
    require(one.window.us[direct][x2w_phase::lock_wait] / one.window.calls[direct] < 20.0, "one thread waits for no lock");
    require(four.window.queued_max >= 2, "four threads on one lock queue");
    require(four.window.us[direct][x2w_phase::lock_wait] / four.window.calls[direct] > 2.0 * (resolve_us + record_us) * 0.5,
            "the queue shows up as lock wait per call");
    const double hold = one.window.us[direct][x2w_phase::hold] / one.window.calls[direct];
    require(hold >= resolve_us + record_us && hold < 2.0 * (resolve_us + record_us), "the hold time is the work done under the lock");
    require(one.window.us[direct][x2w_phase::resolve] / one.window.calls[direct] >= resolve_us, "resolve time is its own phase");
    require(one.window.us[direct][x2w_phase::record] / one.window.calls[direct] >= record_us, "record time is its own phase");
    require(serial < 1.25, "a single lock gives four threads no throughput");

    Run split_one, split_four;
    const double parallel = speedup(true, 4, &split_one, &split_four);
    std::printf("split lock:  1 thread %.3f s, 4 threads %.3f s, speedup %.2f, wait/call %.1f us, hold/call %.1f us\n", split_one.seconds,
                split_four.seconds, parallel, split_four.window.us[direct][x2w_phase::lock_wait] / split_four.window.calls[direct],
                split_four.window.us[direct][x2w_phase::hold] / split_four.window.calls[direct]);
    require(parallel >= 1.45, "holding the lock for resolve only lets record run in parallel");
    require(split_four.window.us[direct][x2w_phase::hold] / split_four.window.calls[direct] < 2.0 * resolve_us + 20,
            "the split model's lock hold is the resolve work");

    char line[1024];
    x2w_phase::format_window(four.window, line, sizeof(line));
    require(std::strstr(line, "queue max") && std::strstr(line, "dispatch ") && std::strstr(line, "lock-wait ") && std::strstr(line, "hold "),
            "the report line names the queue depth, the kind and the phases");
    require(!std::strstr(line, "indirect") && !std::strstr(line, "refused"), "a kind that did not run is left out");
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 4) {
        const bool split = std::strcmp(argv[1], "split") == 0;
        const double got = speedup(split, std::atoi(argv[2]));
        const double want = std::atof(argv[3]);
        std::printf("%s lock, %s threads: speedup %.2f (need %.2f)\n", argv[1], argv[2], got, want);
        return got >= want ? 0 : 1;
    }
    suite();
    if (failures) {
        std::printf("dispatch_phases: %d FAILED\n", failures);
        return 1;
    }
    std::puts("dispatch_phases: ok");
    return 0;
}
