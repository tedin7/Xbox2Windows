// Checks the negative cache of refused buffer memory SRVs (device slot 70, src/startup_stubs/srv_refusal_cache.h)
// without a device. Every cached refusal carries an exact proof:
//   independent: same request and placement generation;
//   fully_committed: the allocation is fully committed and its stamp is unchanged;
//   witness_page: the one page found uncommitted is still uncommitted in the same allocation.
// The reported page move (reserve 128 KiB, commit the upper 64 KiB, refuse; decommit it, commit the lower 64 KiB,
// retry at once; same base, size and committed total) must miss, because its witness page is now committed.
// --mutate generation: expect a changed placement generation to still hit.
// --mutate witness: expect the page move to still hit (the witness page is not asked).
// --mutate stamp: expect a fully_committed entry to hit under a changed stamp.
// --mutate key: expect a different view to hit an entry noted for another view.
// --mutate device: expect the same request on another device to hit an entry noted for the first device.
// --mutate age: expect an entry older than the age limit to still hit.
// --mutate hits: expect an entry used more than the hit limit to still hit.
// --mutate unproven: expect a refusal without a proof to be cached.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/srv_refusal_cache.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace {

int failures;
const char* mutation = "";

bool mutated(const char* name)
{
    return std::strcmp(mutation, name) == 0;
}

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

using namespace x2w_startup;

constexpr std::uint64_t kBase = 0x7000000000ull;
constexpr std::uint64_t kPage = 0x10000;  // 64 KiB

// A fake allocation of two 64 KiB pages: which are committed. The real question is a VirtualQuery of one page.
std::array<bool, 2> committed{};

bool page_still_uncommitted(const SrvRefusalProof& proof)
{
    if (mutated("witness")) return true;
    return !committed[(proof.witness - kBase) / kPage];
}

bool never_asked(const SrvRefusalProof&)
{
    check("only a witness_page entry asks about a page", false);
    return true;
}

SrvRefusalKey request(std::uint64_t address, std::uint64_t first_element)
{
    SrvRefusalKey key;
    key.device = 0x5000;
    key.address = address;
    key.span_bytes = 0x4000;
    key.view_a = 0x1234;
    key.view_b = first_element;
    key.view_c = 64;
    key.resource_a = 1;
    key.resource_b = 0x10000;
    return key;
}

SrvRefusalStamp stamp(std::uint64_t generation, std::uint64_t commit)
{
    SrvRefusalStamp s;
    s.valid = true;
    s.generation = generation;
    s.base = kBase;
    s.size = 2 * kPage;  // 128 KiB reservation
    s.commit = commit;
    s.flags = 1;
    return s;
}

SrvRefusalProof independent()
{
    return SrvRefusalProof{SrvRefusalKind::independent, 0, 0, true};
}

SrvRefusalProof fully_committed()
{
    return SrvRefusalProof{SrvRefusalKind::fully_committed, 0, 0, true};
}

SrvRefusalProof witness_page(std::uint64_t page)
{
    return SrvRefusalProof{SrvRefusalKind::witness_page, kBase + page * kPage, kBase, true};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) mutation = argv[2];

    {
        SrvRefusalCache cache;
        const auto key = request(kBase + 0x1000, 0);
        check("an empty cache misses", !cache.hit(key, stamp(5, 0), 100, never_asked));
        cache.note(key, stamp(5, 0), 100, independent());
        check("an independent refusal hits for the same request and generation", cache.hit(key, stamp(5, 0), 150, never_asked));
        check("a hit repeats", cache.hit(key, stamp(5, 0), 160, never_asked));
        check("an independent refusal ignores the memory", cache.hit(key, stamp(5, kPage), 170, never_asked));
        const bool hit = cache.hit(key, stamp(6, 0), 180, never_asked);
        check("a new placement generation misses (a resource may now cover the address)", mutated("generation") ? hit : !hit);
    }
    {
        // The reported sequence. Request at the allocation base; only the upper page is committed.
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        committed = {false, true};
        cache.note(key, stamp(5, kPage), 100, witness_page(0));
        check("the refusal still holds while the witness page is uncommitted", cache.hit(key, stamp(5, kPage), 101, page_still_uncommitted));
        // Decommit the upper page, commit the lower one: same base, same reservation, same committed total.
        committed = {true, false};
        check("the equal stamp does not hide the page move", stamp(5, kPage) == stamp(5, kPage));
        check("after the page move the full path runs (the witness page is committed)",
              !cache.hit(key, stamp(5, kPage), 102, page_still_uncommitted));
        // A request that starts in the upper page walks from the base too: the witness is still the lower page.
        committed = {false, true};
        const auto upper = request(kBase + kPage, 0);
        cache.note(upper, stamp(5, kPage), 110, witness_page(0));
        check("a request in the upper page is proven by the same witness", cache.hit(upper, stamp(5, kPage), 111, page_still_uncommitted));
        committed = {true, true};
        check("committing the witness page makes it miss", !cache.hit(upper, stamp(5, 2 * kPage), 112, page_still_uncommitted));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        committed = {false, true};
        cache.note(key, stamp(5, kPage), 100, witness_page(0));
        auto other_allocation = stamp(5, kPage);
        other_allocation.base += 0x100000;
        check("another allocation base misses", !cache.hit(key, other_allocation, 101, page_still_uncommitted));
        SrvRefusalStamp unreadable = stamp(5, kPage);
        unreadable.valid = false;
        check("an unreadable allocation stamp misses", !cache.hit(key, unreadable, 101, page_still_uncommitted));
        check("a new generation misses", !cache.hit(key, stamp(6, kPage), 101, page_still_uncommitted));
        cache.note(request(kBase, 1), unreadable, 100, witness_page(0));
        check("a witness entry that needs the allocation is not cached without a stamp",
              !cache.hit(request(kBase, 1), stamp(5, kPage), 101, page_still_uncommitted));
        // The address page itself was uncommitted: no allocation is involved (base 0), only that page is asked.
        SrvRefusalProof address_page{SrvRefusalKind::witness_page, kBase, 0, true};
        committed = {false, false};
        cache.note(request(kBase, 2), unreadable, 100, address_page);
        check("an uncommitted address hits without an allocation stamp", cache.hit(request(kBase, 2), unreadable, 101, page_still_uncommitted));
        committed = {true, false};
        check("a committed address misses", !cache.hit(request(kBase, 2), unreadable, 102, page_still_uncommitted));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        cache.note(key, stamp(5, 2 * kPage), 100, fully_committed());
        check("a fully committed allocation hits under the same stamp", cache.hit(key, stamp(5, 2 * kPage), 101, never_asked));
        const bool hit = cache.hit(key, stamp(5, kPage), 102, never_asked);
        check("a fully committed refusal misses when the stamp changes", mutated("stamp") ? hit : !hit);
        auto resized = stamp(5, 2 * kPage);
        resized.size += kPage;
        check("a resized allocation misses", !cache.hit(key, resized, 102, never_asked));
        auto moved = stamp(5, 2 * kPage);
        moved.base += kPage;
        check("another base misses", !cache.hit(key, moved, 102, never_asked));
    }
    {
        SrvRefusalCache cache;
        cache.note(request(kBase + 0x1000, 0), stamp(5, 0), 100, independent());
        const bool other_view = cache.hit(request(kBase + 0x1000, 8), stamp(5, 0), 110, never_asked);
        check("another view of the same address misses", mutated("key") ? other_view : !other_view);
        check("another address misses", !cache.hit(request(kBase + 0x2000, 0), stamp(5, 0), 110, never_asked));
        auto other_device = request(kBase + 0x1000, 0);
        other_device.device += 0x100;
        const bool device_hit = cache.hit(other_device, stamp(5, 0), 110, never_asked);
        check("the same request on another device misses (placement lookups are per device)",
              mutated("device") ? device_hit : !device_hit);
        auto wider = request(kBase + 0x1000, 0);
        wider.span_bytes += 4;
        check("another span misses", !cache.hit(wider, stamp(5, 0), 110, never_asked));
        auto resource = request(kBase + 0x1000, 0);
        resource.resource_b += 1;
        check("another resource width misses", !cache.hit(resource, stamp(5, 0), 110, never_asked));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        cache.note(key, stamp(5, 0), 100, independent());
        const bool old_hit = cache.hit(key, stamp(5, 0), 100 + SrvRefusalCache::kMaxAgeMs + 1, never_asked);
        check("an entry older than the age limit misses", mutated("age") ? old_hit : !old_hit);
        SrvRefusalCache fresh;
        fresh.note(key, stamp(5, 0), 100, independent());
        check("an entry exactly at the age limit still hits", fresh.hit(key, stamp(5, 0), 100 + SrvRefusalCache::kMaxAgeMs, never_asked));
        SrvRefusalCache backwards;
        backwards.note(key, stamp(5, 0), 100, independent());
        check("a clock that went backwards misses", !backwards.hit(key, stamp(5, 0), 50, never_asked));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        cache.note(key, stamp(5, 0), 100, independent());
        bool all = true;
        for (std::uint32_t i = 0; i < SrvRefusalCache::kMaxHits; ++i) all = cache.hit(key, stamp(5, 0), 110, never_asked) && all;
        check("the entry hits up to the hit limit", all);
        const bool over = cache.hit(key, stamp(5, 0), 110, never_asked);
        check("past the hit limit the full path runs again", mutated("hits") ? over : !over);
        cache.note(key, stamp(5, 0), 120, independent());  // the full path refused again and re-noted
        check("a re-noted entry hits again", cache.hit(key, stamp(5, 0), 121, never_asked));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        SrvRefusalProof none = independent();
        none.proven = mutated("unproven");
        cache.note(key, stamp(5, 0), 100, none);
        check("a refusal without a proof is never cached", !cache.hit(key, stamp(5, 0), 101, never_asked));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(kBase, 0);
        cache.note(key, stamp(5, 0), 100, independent());
        cache.clear();
        check("clear forgets everything", !cache.hit(key, stamp(5, 0), 110, never_asked));
        const auto before = srv_placement_generation().load();
        note_placement_change();
        check("note_placement_change advances the generation", srv_placement_generation().load() == before + 1);
        note_resource_reference(2);
        note_resource_reference(3);
        check("a reference added to a live resource keeps the generation", srv_placement_generation().load() == before + 1);
        note_resource_reference(1);
        check("a resource referenced again from zero advances the generation", srv_placement_generation().load() == before + 2);
    }

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("srv refusal cache checks passed\n");
    return 0;
}
