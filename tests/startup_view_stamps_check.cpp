// Checks the validity stamps of the translate-once view cache (src/startup_stubs/view_stamps.h) without a device: every
// event that must drop a cached translation makes `current()` false, and nothing else does.
//   rewrite      the title writes the slot again (rule 1)
//   copy         a copy over a slot (rule 2): the destination is fresh, the source is untouched, a stale source stays stale
//   heap         a released heap invalidates all its slots (rule 3)
//   serial       a new heap never reuses a released heap's serial (rule 4)
//   resource     destroy / materialize / memory rewrite / re-place bumps the resource (rules 5-8)
//   epoch        reset invalidates everything (rule 9)
//   control      an unrelated slot, heap or resource keeps its entry (no over-invalidation)
// --mutate write|copy|heap|serial|resource|epoch|copy_stale: the matching rule is broken on purpose and the run must fail.
#include "startup_stubs/view_stamps.h"

#include <cstdio>
#include <cstring>

using x2w_startup::ViewStamp;
using x2w_startup::ViewStamps;

namespace {

int failures;

void check(bool ok, const char* what)
{
    std::printf("%s: %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

ViewStamps::Fault fault_of(const char* name)
{
    using F = ViewStamps::Fault;
    if (!std::strcmp(name, "write")) return F::write;
    if (!std::strcmp(name, "copy")) return F::copy;
    if (!std::strcmp(name, "heap")) return F::heap;
    if (!std::strcmp(name, "serial")) return F::serial;
    if (!std::strcmp(name, "resource")) return F::resource;
    if (!std::strcmp(name, "epoch")) return F::epoch;
    if (!std::strcmp(name, "copy_stale")) return F::copy_stale;
    return F::none;
}

constexpr std::uint64_t texture_a = 0x1000, texture_b = 0x2000;

}  // namespace

int main(int argc, char** argv)
{
    ViewStamps v;
    if (argc == 3 && !std::strcmp(argv[1], "--mutate")) v.fault = fault_of(argv[2]);

    // Rule 1: write view A into slot 5, cache it, then rewrite the slot (to B): the cached translation of A must not be used.
    const auto heap = v.new_heap();
    v.write_entry(heap, 5);
    v.bump_resource(texture_a);
    const ViewStamp cached_a = v.snapshot(heap, 5, texture_a);
    check(v.current(cached_a), "a fresh translation is current");
    v.write_entry(heap, 5);
    check(!v.current(cached_a), "rewrite: the slot was written again, the translation of the old view is stale");

    // Rule 2: a copy 5 -> 9 carries the source translation to the destination; a later rewrite of the source leaves the
    // copy's translation alone, and a copy of a stale source is never current.
    v.write_entry(heap, 5);
    v.write_entry(heap, 9);
    const ViewStamp old9 = v.snapshot(heap, 9, texture_b);
    const ViewStamp a5 = v.snapshot(heap, 5, texture_a);
    const ViewStamp copied = v.copy_entry(a5, heap, 9);
    check(!v.current(old9), "copy: what the destination slot held before is stale");
    check(v.current(copied), "copy: the destination slot holds the source's translation");
    check(v.current(a5), "copy: the source slot is untouched");
    v.write_entry(heap, 5);
    check(v.current(copied), "copy: rewriting the source later does not touch the copy");
    check(!v.current(a5), "copy: the source's own old translation is stale after its rewrite");
    const ViewStamp stale_copy = v.copy_entry(a5, heap, 10);
    check(!v.current(stale_copy), "copy: a copy of a stale source is never current");

    // Rule 5-8: the resource is destroyed / materialized / rewritten / re-placed.
    v.write_entry(heap, 7);
    const ViewStamp on_b = v.snapshot(heap, 7, texture_b);
    check(v.current(on_b), "resource: translation of an untouched resource is current");
    v.bump_resource(texture_b);
    check(!v.current(on_b), "resource: a destroyed / materialized / rewritten resource drops its translations");

    // Control: unrelated slots, heaps and resources are not over-invalidated.
    v.write_entry(heap, 20);
    v.bump_resource(texture_a);
    const ViewStamp on_a20 = v.snapshot(heap, 20, texture_a);
    v.write_entry(heap, 21);
    v.bump_resource(texture_b);
    check(v.current(on_a20), "control: writing another slot or bumping another resource keeps this entry");
    const auto other = v.new_heap();
    v.write_entry(other, 20);
    check(v.current(on_a20), "control: a second heap with the same slot number is a different entry");

    // Rule 3 + 4: a released heap drops all its entries; a heap created later never reuses its serial.
    const ViewStamp in_heap = v.snapshot(heap, 20, texture_a);
    v.release_heap(heap);
    check(!v.current(in_heap), "heap: all entries of a released heap are stale");
    check(!v.current(on_a20), "heap: and so is every other entry of it");
    const auto reborn = v.new_heap();
    v.write_entry(reborn, 20);
    check(!v.current(in_heap), "serial: a new heap at the same place does not revive the old heap's entries");
    check(reborn != heap, "serial: the new heap has a new serial");

    // Rule 9: reset.
    const ViewStamp fresh = v.snapshot(reborn, 20, 0);
    check(v.current(fresh), "a view that names no resource is current until something changes");
    v.reset_all();
    check(!v.current(fresh), "epoch: a device reset invalidates every translation");

    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("view stamps ok\n");
    return 0;
}
