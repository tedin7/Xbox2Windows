// Unit test of src/replay/prefix_live.h: which prefix calls --prefix-live leaves out. Pure logic, no D3D12.
#include "replay/prefix_live.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using x2w_replay::LiveCall;
using x2w_replay::PrefixLive;
using x2w_replay::u64;

namespace {

int failures = 0;
void check(bool ok, const char* what, int line)
{
    if (ok) return;
    std::printf("FAIL line %d: %s\n", line, what);
    ++failures;
}
#define CHECK(condition) check((condition), #condition, __LINE__)

struct Feed {
    PrefixLive live;
    u64 seq = 0;
    // Adds one call and returns its add index.
    std::size_t add(std::uint16_t kind, std::uint16_t slot, u64 object = 0, std::vector<u64> args = {}, std::vector<u64> outputs = {})
    {
        LiveCall call{++seq, kind, slot, object, {}, outputs.empty() ? nullptr : outputs.data(), outputs.size()};
        for (std::size_t i = 0; i < args.size() && i < 11; ++i) call.args[i] = args[i];
        live.add(call);
        return static_cast<std::size_t>(seq - 1);
    }
};

constexpr std::uint16_t device = 0, heap = 5, resource = 7;

void descriptor_writes()
{
    // The destination handle sits at a different argument per slot: a later write to it replaces the earlier one.
    struct Slot { std::uint16_t slot; int at; };
    for (const Slot s : {Slot{17, 1}, Slot{18, 2}, Slot{19, 3}, Slot{20, 2}, Slot{21, 2}, Slot{22, 1}, Slot{70, 2}}) {
        Feed f;
        std::vector<u64> first(11, 0x1111), second(11, 0x2222);  // everything but the destination differs
        first[s.at] = second[s.at] = 0xd000;
        const auto a = f.add(device, s.slot, 1, first);
        const auto b = f.add(device, s.slot, 1, second);
        CHECK(f.live.left_out(a) && !f.live.left_out(b));
        CHECK(f.live.counts().superseded_writes == 1);
        // A different destination is not a replacement.
        std::vector<u64> other = second;
        other[s.at] = 0xd100;
        const auto c = f.add(device, s.slot, 1, other);
        CHECK(!f.live.left_out(b) && !f.live.left_out(c));
    }
    // Different slots, one destination: the later write wins whatever its slot.
    Feed f;
    const auto srv = f.add(device, 18, 1, {0, 0, 0xd000});
    const auto memory = f.add(device, 70, 1, {0, 0, 0xd000});
    CHECK(f.live.left_out(srv) && !f.live.left_out(memory));
    // A write without a destination handle is kept and never compared.
    const auto none1 = f.add(device, 20, 1, {0, 0, 0});
    const auto none2 = f.add(device, 20, 1, {0, 0, 0});
    CHECK(!f.live.left_out(none1) && !f.live.left_out(none2));
}

void copies_pin_writes()
{
    for (const std::uint16_t copy : {23, 24}) {
        Feed f;
        const auto before = f.add(device, 20, 1, {0, 0, 0xd000});
        f.add(device, copy, 1, {1, 0xe000, 0xd000});
        const auto after = f.add(device, 20, 1, {0, 0, 0xd000});
        CHECK(!f.live.left_out(before) && !f.live.left_out(after));  // the copy may have read the first one
        CHECK(f.live.counts().superseded_writes == 0);
        const auto later = f.add(device, 20, 1, {0, 0, 0xd000});
        CHECK(f.live.left_out(after) && !f.live.left_out(later));    // after the copy the usual rule holds again
        CHECK(f.live.counts().superseded_writes == 1);
    }
}

void teaching_calls()
{
    Feed f;
    const u64 r = 0x5000, r2 = 0x6000, h = 0x7000;
    f.add(device, 27, 1, {}, {r});                    // a creation: kept, and it names r
    f.add(device, 27, 1, {}, {r2});
    f.add(device, 14, 1, {}, {h});
    const auto map1 = f.add(resource, 8, r);
    const auto unmap1 = f.add(resource, 9, r);        // after the first Map: left out with the repeats
    const auto map2 = f.add(resource, 8, r);
    const auto unmap2 = f.add(resource, 9, r);
    CHECK(!f.live.left_out(map1) && f.live.left_out(unmap1) && f.live.left_out(map2) && f.live.left_out(unmap2));
    const auto early_unmap = f.add(resource, 9, r2);  // an Unmap before any Map of its resource stays
    const auto r2_map = f.add(resource, 8, r2);
    const auto r2_unmap = f.add(resource, 9, r2);
    CHECK(!f.live.left_out(early_unmap) && !f.live.left_out(r2_map) && f.live.left_out(r2_unmap));
    const auto gpu1 = f.add(resource, 11, r);
    const auto gpu2 = f.add(resource, 11, r);
    CHECK(!f.live.left_out(gpu1) && f.live.left_out(gpu2));
    const auto cpu1 = f.add(heap, 9, h);
    const auto cpu2 = f.add(heap, 9, h);
    const auto gpu_start1 = f.add(heap, 10, h);
    const auto gpu_start2 = f.add(heap, 10, h);
    CHECK(!f.live.left_out(cpu1) && f.live.left_out(cpu2) && !f.live.left_out(gpu_start1) && f.live.left_out(gpu_start2));
    // Resource slot 9 is Unmap, heap slot 9 is the CPU start: the kinds do not share state.
    const auto unrelated = f.add(heap, 9, r);
    CHECK(!f.live.left_out(unrelated));
    // A pointer the title gets again for a NEW object starts over.
    f.add(device, 27, 1, {}, {r});
    const auto map3 = f.add(resource, 8, r);
    const auto gpu3 = f.add(resource, 11, r);
    CHECK(!f.live.left_out(map3) && !f.live.left_out(gpu3));
    CHECK(f.live.counts().repeated_teaching == 7);
}

void queries_and_the_rest()
{
    Feed f;
    const auto query25 = f.add(device, 25, 1, {1, 2, 3});
    const auto query108 = f.add(device, 108, 1, {1, 2, 3});
    CHECK(f.live.left_out(query25) && f.live.left_out(query108) && f.live.counts().queries == 2);
    // Creations, AddRef, Release, QueryInterface, fences and everything else stay.
    const auto create = f.add(device, 27, 1, {}, {0x5000});
    const auto add_ref = f.add(resource, 1, 0x5000);
    const auto release = f.add(resource, 2, 0x5000);
    const auto query_interface = f.add(resource, 0, 0x5000, {}, {0x5008});
    const auto fence = f.add(2, 10, 0x9000);
    const auto list_call = f.add(4, 12, 0xa000);
    for (const auto kept : {create, add_ref, release, query_interface, fence, list_call}) CHECK(!f.live.left_out(kept));
    CHECK(f.live.counts().calls == 8);
    CHECK(!f.live.left_out(1000));  // an index that was never added is not left out
}

void mixed_load()
{
    // 3 writes to one handle, a copy, 2 writes to it, 3 Maps: 2 + 1 + 2 left out, the rest kept.
    Feed f;
    std::vector<std::size_t> all;
    for (int i = 0; i < 3; ++i) all.push_back(f.add(device, 70, 1, {0, 0, 0xd000}));
    all.push_back(f.add(device, 24, 1, {1, 0xe000, 0xd000}));
    for (int i = 0; i < 2; ++i) all.push_back(f.add(device, 70, 1, {0, 0, 0xd000}));
    f.add(device, 27, 1, {}, {0x5000});
    for (int i = 0; i < 3; ++i) all.push_back(f.add(resource, 8, 0x5000));
    std::size_t out = 0;
    for (std::size_t i = 0; i < f.live.counts().calls; ++i) out += f.live.left_out(i);
    CHECK(out == 2 + 1 + 2);
    CHECK(f.live.counts().superseded_writes == 3 && f.live.counts().repeated_teaching == 2);
}

}  // namespace

int main()
{
    descriptor_writes();
    copies_pin_writes();
    teaching_calls();
    queries_and_the_rest();
    mixed_load();
    if (failures) {
        std::printf("startup_replay_prefix_live_check: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("startup_replay_prefix_live_check: ok\n");
    return 0;
}
