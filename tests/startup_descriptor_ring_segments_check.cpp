#include "startup_stubs/descriptor_ring_segments.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

namespace {

using Pool = x2w_startup::DescriptorRingSegments<8>;
constexpr unsigned segment_size = 512;

bool allocation_order()
{
    Pool pool;
    if (pool.acquire(0) || pool.acquire(9) || pool.free_count() != 8) return false;
    const auto head = pool.acquire(2);
    const auto middle = pool.acquire(3);
    const auto tail = pool.acquire(3);
    if (!head || head->first != 0 || head->count != 2 ||
        !middle || middle->first != 2 || middle->count != 3 ||
        !tail || tail->first != 5 || tail->count != 3 ||
        pool.acquire(1) || pool.used_count() != 8 || pool.free_count()) return false;

    pool.release(*tail);
    pool.release(*head);
    if (pool.acquire(4) || pool.free_count() != 5) return false;  // enough free segments, but split
    const auto reused = pool.acquire(2);
    if (!reused || reused->first != head->first || reused->count != head->count) return false;
    pool.release(*middle);
    const auto joined = pool.acquire(6);
    if (!joined || joined->first != 2 || joined->count != 6) return false;
    pool.release(*reused);
    pool.release(*joined);
    return pool.used_count() == 0 && pool.free_count() == 8;
}

bool concurrent_owners()
{
    Pool pool;
    std::mutex lock;
    std::array<std::atomic<unsigned>, 8 * segment_size> owners{};
    for (auto& owner : owners) owner.store(0, std::memory_order_relaxed);
    std::atomic<unsigned> ready{0};
    std::atomic<bool> overlap{false};

    const auto exercise = [&](unsigned id) {
        ++ready;
        while (ready.load() != 2) std::this_thread::yield();
        for (unsigned cycle = 0; cycle < 100000; ++cycle) {
            std::optional<Pool::Run> run;
            {
                const std::lock_guard guard(lock);
                run = pool.acquire(1 + (cycle + id) % 3);
            }
            if (!run) {
                overlap.store(true, std::memory_order_relaxed);
                return;
            }
            const unsigned first = run->first * segment_size;
            const unsigned end = first + run->count * segment_size;
            for (unsigned i = first; i < end; ++i)
                if (owners[i].exchange(id, std::memory_order_relaxed) != 0)
                    overlap.store(true, std::memory_order_relaxed);
            if (cycle % 64 == 0) std::this_thread::yield();
            for (unsigned i = first; i < end; ++i)
                if (owners[i].exchange(0, std::memory_order_relaxed) != id)
                    overlap.store(true, std::memory_order_relaxed);
            const std::lock_guard guard(lock);
            pool.release(*run);
        }
    };

    std::thread first(exercise, 1), second(exercise, 2);
    first.join();
    second.join();
    if (overlap.load() || pool.used_count()) return false;
    for (const auto& owner : owners)
        if (owner.load(std::memory_order_relaxed)) return false;
    return true;
}

}  // namespace

int main()
{
    if (!allocation_order()) {
        std::puts("FAIL: descriptor segment allocation order, exhaustion or reuse");
        return 1;
    }
    if (!concurrent_owners()) {
        std::puts("FAIL: descriptor segment ownership under concurrent allocation");
        return 1;
    }
    std::puts("PASS: descriptor segment allocation and 200000 concurrent acquire/release cycles");
    return 0;
}
