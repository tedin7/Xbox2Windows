// Checks the resolved-view cache (src/startup_stubs/view_cache.h) without a device. An entry is served only while
// everything it was built over is unchanged:
//   id:        another id in the same slot replaces the entry; the old id then misses;
//   resource:  the same id over another resource address misses;
//   resource generation: release/retire of the resource (address reuse) drops every entry built over it;
//   stamp:     the caller's stamp (texture upload generation) differs: miss;
//   kind:      the same id asked as another descriptor kind misses;
//   write:     the write hook drops the entry of that id only;
//   flush:     a heap destroy / whole-cache flush drops everything built before it.
// A rebuilt entry (publish after a miss) hits again, and concurrent lookups and publishes keep the counters consistent.
// --mutate resource-generation|stamp|id|resource|kind|write|flush: expect the stale lookup to still hit. Each mutated run is a
// negative control and must fail.
#include "startup_stubs/view_cache.h"

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

int failures;
const char* mutation = "";

bool mutated(const char* name) { return std::strcmp(mutation, name) == 0; }

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

using namespace x2w_view_cache;

// `stale` is the lookup of an entry that must miss; a mutated run expects it to hit.
void expect_stale(const char* name, const char* mutation_name, bool lookup_hit)
{
    check(name, mutated(mutation_name) ? lookup_hit : !lookup_hit);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && !std::strcmp(argv[1], "--mutate")) mutation = argv[2];

    {
        Cache cache(64);
        check("a cold lookup misses", !cache.lookup(5, 0x1000, Kind::srv));
        cache.publish(5, 0x1000, Kind::srv);
        check("a published entry hits", cache.lookup(5, 0x1000, Kind::srv));
        check("repeated lookups keep hitting", cache.lookup(5, 0x1000, Kind::srv) && cache.lookup(5, 0x1000, Kind::srv));
        expect_stale("another resource address misses", "resource", cache.lookup(5, 0x2000, Kind::srv));
        expect_stale("another descriptor kind misses", "kind", cache.lookup(5, 0x1000, Kind::uav));
        expect_stale("another stamp misses", "stamp", cache.lookup(5, 0x1000, Kind::srv, 7));
        // A neighbour in another slot is untouched by all of the above.
        cache.publish(6, 0x1000, Kind::srv);
        check("a second id hits beside the first", cache.lookup(6, 0x1000, Kind::srv) && cache.lookup(5, 0x1000, Kind::srv));
    }
    {
        Cache cache(64);
        cache.publish(5, 0x1000, Kind::srv);
        cache.publish(5 + 64, 0x1000, Kind::srv);  // the same slot, another id
        expect_stale("the replaced id misses", "id", cache.lookup(5, 0x1000, Kind::srv));
        check("the replacing id hits", cache.lookup(5 + 64, 0x1000, Kind::srv));
        check("a replacement is counted", cache.stats().replaced == 1);
    }
    {
        Cache cache(64);
        cache.publish(1, 0x1000, Kind::srv);
        cache.publish(2, 0x1000, Kind::uav);
        cache.publish(3, 0x2000, Kind::srv);
        cache.note_resource_change(0x1000);  // released: its address may be reused
        if (mutated("resource-generation")) {
            check("mutated: a released resource's view still hits", cache.lookup(1, 0x1000, Kind::srv));
        } else {
            check("every view over a released resource misses", !cache.lookup(1, 0x1000, Kind::srv) && !cache.lookup(2, 0x1000, Kind::uav));
        }
        check("views over another resource keep hitting", cache.lookup(3, 0x2000, Kind::srv));
        cache.publish(1, 0x1000, Kind::srv);  // rebuilt over the new object
        check("a rebuilt entry hits again", cache.lookup(1, 0x1000, Kind::srv));
        cache.note_resource_change(0x1000);
        check("a second release drops it again", !cache.lookup(1, 0x1000, Kind::srv));
    }
    {
        Cache cache(64);
        cache.publish(1, 0x1000, Kind::srv);
        cache.publish(2, 0x1000, Kind::srv);
        cache.note_write(1);
        expect_stale("a written descriptor's entry misses", "write", cache.lookup(1, 0x1000, Kind::srv));
        check("a write to one id leaves the other", cache.lookup(2, 0x1000, Kind::srv));
        cache.note_write(99);  // an id with no entry
        check("a write to an unknown id changes nothing", cache.lookup(2, 0x1000, Kind::srv) && cache.stats().write_drops == 1);
    }
    {
        Cache cache(64);
        cache.publish(1, 0x1000, Kind::srv);
        cache.note_flush();
        expect_stale("a flush drops what was built before it", "flush", cache.lookup(1, 0x1000, Kind::srv));
        cache.publish(1, 0x1000, Kind::srv);
        check("an entry built after the flush hits", cache.lookup(1, 0x1000, Kind::srv));
    }
    {
        // Eight threads publish and look up their own ids while one thread releases a resource they all use: no lookup may
        // return a hit for an entry published before a release it did not see, and the counters add up exactly.
        Cache cache(1024);
        constexpr int kThreads = 8, kRounds = 2000;
        std::vector<std::thread> threads;
        std::atomic<int> bad{0};
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] {
                for (int r = 0; r < kRounds; ++r) {
                    const std::uint32_t id = static_cast<std::uint32_t>(t * 100 + r % 50);
                    if (!cache.lookup(id, 0x5000, Kind::srv)) cache.publish(id, 0x5000, Kind::srv);
                }
            });
        threads.emplace_back([&] {
            for (int r = 0; r < 200; ++r) cache.note_resource_change(0x5000);
        });
        for (auto& thread : threads) thread.join();
        const Stats stats = cache.stats();
        check("hits plus misses equal the lookups", stats.hits + stats.misses == std::uint64_t{kThreads} * kRounds);
        check("every miss published once", stats.publishes == stats.misses);
        check("no thread failed", bad.load() == 0);
        cache.publish(7, 0x5000, Kind::srv);
        cache.note_resource_change(0x5000);
        check("after the storm a release still drops entries", !cache.lookup(7, 0x5000, Kind::srv));
    }
    if (failures) return 1;
    std::puts("view cache checks passed");
    return 0;
}
