// AddressIndex and store_indexed (src/startup_stubs/resource_address_index.h) against a scan of every
// resource, the way the memory SRV lookups (d3d12_memory_srv.inl) run. Resources are never removed from the
// table or the index. A release only drops the live reference count to 0 and a later reference raises it
// again; each lookup filters on that count itself, as the real lookups do, so the index must answer what the
// scan answers across release, re-reference, address reuse (dead entries and live ones at one base, where
// two live matches mean none), overlapping live ranges, re-noting a resource with another size and
// unaligned bases. Directed cases pin the span end: a base at address + span - 1 is found, one at
// address + span is not. Allocation failures are injected into store_indexed (operator new is replaced):
// the record and the index must both hold the new resource or both hold the old state.
// --mutate skip   skips one index insert (a live resource the index lacks): must fail.
// --mutate unsync does the index insert first and leaves it behind when the record update fails (the
//                 unsynchronized order): must fail.
#include "startup_stubs/resource_address_index.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <new>
#include <random>

namespace {
long fail_after = -1;
bool injected;
} // namespace

void* operator new(std::size_t size)
{
    if (fail_after == 0) {
        fail_after = -1;
        injected = true;
        throw std::bad_alloc();
    }
    if (fail_after > 0) --fail_after;
    if (void* ptr = std::malloc(size ? size : 1)) return ptr;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
using x2w_startup::address_span_touches;
struct Resource {
    std::uint64_t base, bytes;
    int refs;
};
using Model = std::map<int, Resource>;
using Index = x2w_startup::AddressIndex<int>;
bool unknown_key; // the index returned a key with no record
bool unsync;      // --mutate unsync

const Resource& resource_of(const Model& model, int key)
{
    static const Resource none{};
    const auto found = model.find(key);
    unknown_key |= found == model.end();
    return found == model.end() ? none : found->second;
}
bool scan_touches(const Model& model, std::uint64_t address, std::uint64_t span)
{
    for (const auto& [key, r] : model)
        if (r.refs > 0 && address_span_touches(address, span, r.base, r.bytes)) return true;
    return false;
}
bool index_touches(const Index& index, const Model& model, std::uint64_t address, std::uint64_t span)
{
    bool found = false;
    index.for_each_reaching(address, span, [&](int key) {
        const Resource& r = resource_of(model, key);
        return found = r.refs > 0 && address_span_touches(address, span, r.base, r.bytes);
    });
    return found;
}
// -1: no live resource contains the address, -2: several do, else the key of the only one.
int scan_containing(const Model& model, std::uint64_t address)
{
    int match = -1;
    for (const auto& [key, r] : model) {
        if (r.refs <= 0 || address < r.base || address - r.base >= r.bytes) continue;
        match = match == -1 ? key : -2;
    }
    return match;
}
int index_containing(const Index& index, const Model& model, std::uint64_t address)
{
    int match = -1;
    index.for_each_reaching(address, 1, [&](int key) {
        const Resource& r = resource_of(model, key);
        if (r.refs <= 0 || address < r.base || address - r.base >= r.bytes) return false;
        if (match != -1) {
            match = -2;
            return true;
        }
        match = key;
        return false;
    });
    return match;
}
bool agree(const Index& index, const Model& model, std::uint64_t address, std::uint64_t span, const char* what)
{
    if (!unknown_key && scan_touches(model, address, span) == index_touches(index, model, address, span) &&
        scan_containing(model, address) == index_containing(index, model, address)) return true;
    std::fprintf(stderr, "address index (%s): address 0x%llx span %llu: scan and index disagree\n", what,
                 static_cast<unsigned long long>(address), static_cast<unsigned long long>(span));
    return false;
}
// Every record is in the index and the index holds nothing else (the table is never pruned).
bool synced(const Index& index, const Model& model)
{
    for (const auto& [key, r] : model)
        if (!index.contains(r.base, key)) return false;
    return index.size() == model.size();
}
bool mutated(const char* name, int argc, char** argv) { return argc > 2 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], name); }

// store_indexed into the model, the way note_resource does it. skip: leave the index insert out.
void note(Index& index, Model& model, int key, const Resource& r, bool skip = false)
{
    if (unsync) { // the order store_indexed exists to avoid
        index.insert(r.base, r.bytes, key);
        model.insert_or_assign(key, r);
        return;
    }
    x2w_startup::store_indexed(index, model, key, skip ? 0 : r.base, r.bytes, r);
}

int directed(bool skip_one)
{
    constexpr std::uint64_t span = 64;
    for (const std::uint64_t address : {0x10000ull, 0x10001ull, 0x12345ull}) {
        for (const std::uint64_t bytes : {0ull, 1ull, 100ull, 70000ull}) {
            // Alone, a base at address + span - 1 touches the span and one at address + span does not.
            for (const std::uint64_t base : {address + span - 1, address + span}) {
                Index index;
                Model model;
                note(index, model, 0, {base, bytes, 1}, skip_one && base == address + span - 1);
                const bool expect = base == address + span - 1;
                if (index_touches(index, model, address, span) != expect || scan_touches(model, address, span) != expect) {
                    std::fprintf(stderr, "address index: a base at address + span %s is %s\n", expect ? "- 1" : "",
                                 expect ? "not found" : "found");
                    return 1;
                }
            }
            Index index;
            Model model;
            int next = 0;
            for (const std::uint64_t base : {address + span - 1, address + span, address, address - 1, address + 1})
                note(index, model, next++, {base, bytes, 1});
            if (!agree(index, model, address, span, "directed")) return 1;
        }
    }
    return 0;
}

// An allocation failure at every point of a store_indexed (a new resource, then a re-note into another
// size group): the record and the index stay in step, and the retry succeeds.
int failures()
{
    int injections = 0;
    for (int renote = 0; renote < 2; ++renote) {
        for (long n = 0; n < 12; ++n) {
            Index index;
            Model model;
            note(index, model, 1, {0x4000, 0x100, 1});
            note(index, model, 2, {0x4000, 0x80, 0}); // a released twin at the same base
            const Resource fresh{renote ? 0x4000ull : 0x8000ull, renote ? 0x20000ull : 0x40ull, 1};
            const int key = renote ? 1 : 3;
            const std::size_t records = model.size();
            const std::uint64_t old_bytes = model.at(1).bytes;
            fail_after = n;
            injected = false;
            bool threw = false;
            try {
                note(index, model, key, fresh);
            } catch (const std::bad_alloc&) {
                threw = true;
            }
            fail_after = -1;
            if (injected != threw) {
                std::fprintf(stderr, "address index: injected failure not seen as a throw\n");
                return 1;
            }
            if (threw) {
                ++injections;
                // In neither: the old state, in both the record and the index.
                if (model.size() != records || model.at(1).bytes != old_bytes || !synced(index, model)) {
                    std::fprintf(stderr, "address index: a failed store left the record and the index out of step (n=%ld renote=%d)\n", n, renote);
                    return 1;
                }
                for (const std::uint64_t a : {0x4000ull, 0x4050ull, 0x8000ull, 0x4100ull, 0x9000ull})
                    if (!agree(index, model, a, 1, "after failure") || !agree(index, model, a, 0x30, "after failure")) return 1;
                note(index, model, key, fresh);
            }
            if (model.at(key).bytes != fresh.bytes || !synced(index, model)) {
                std::fprintf(stderr, "address index: the stored resource is not in both the record and the index\n");
                return 1;
            }
            for (const std::uint64_t a : {0x4000ull, 0x4050ull, 0x8000ull, 0x8040ull, 0x10000ull, 0x1ffffull, 0x24000ull})
                if (!agree(index, model, a, 1, "after store") || !agree(index, model, a, 0x30, "after store")) return 1;
        }
    }
    std::printf("address index: %d injected allocation failures left the record and the index in step\n", injections);
    return injections >= 3 ? 0 : 2; // new key: index node, record node; re-note: index node
}
} // namespace

int main(int argc, char** argv)
{
    const bool skip = mutated("skip", argc, argv);
    unsync = mutated("unsync", argc, argv);
    if (argc != 1 && !skip && !unsync) {
        std::fprintf(stderr, "usage: startup_resource_address_index_check [--mutate skip|unsync]\n");
        return 64;
    }
    if (const int rc = directed(skip)) return rc;
    if (const int rc = failures()) return rc;

    std::mt19937_64 rng(12345);
    const auto pick = [&](std::uint64_t n) { return rng() % n; };
    Index index;
    Model model;
    int next = 0, lookups = 0;
    bool armed = skip; // --mutate skip: one insert dropped after step 5000
    for (int step = 0; step < 30000; ++step) {
        const std::uint64_t op = pick(10);
        if (model.size() < 400 && (model.empty() || op < 4)) {
            // Bases cluster on a 4 KiB grid in a 16 MiB window (a third of them a few bytes off it) so
            // ranges overlap, and a third reuse the base of an earlier resource, live or released.
            const std::uint64_t reuse = !model.empty() && pick(3) == 0 ? std::next(model.begin(), pick(model.size()))->second.base : 0;
            const std::uint64_t base = reuse ? reuse : (1 + pick(4096)) * 4096 + (pick(3) == 0 ? 1 + pick(4095) : 0);
            const std::uint64_t kind = pick(4);
            const std::uint64_t bytes = kind == 0 ? 0 : kind == 1 ? 1 + pick(4096) : kind == 2 ? (1 + pick(64)) * 4096 : (1 + pick(2048)) * 4096;
            const bool dropped = armed && step >= 5000 && !pick(50);
            armed &= !dropped;
            note(index, model, next++, {base, bytes, 1}, dropped);
        } else if (op < 5) {
            // Re-note a resource: same base, another size.
            const auto it = std::next(model.begin(), pick(model.size()));
            note(index, model, it->first, {it->second.base, pick(2) ? pick(100000) : 0, it->second.refs});
        } else {
            // Release, or reference again (the entry stays either way).
            const auto it = std::next(model.begin(), pick(model.size()));
            it->second.refs = it->second.refs > 0 ? 0 : 1 + static_cast<int>(pick(2));
        }
        for (int i = 0; i < 6; ++i, ++lookups) {
            const std::uint64_t address = pick(0x1100000) + (pick(2) ? 0 : 4096 * pick(4096)), span = pick(3) ? pick(4096) : pick(64);
            if (!agree(index, model, address, span, "random")) return 1;
            // The edges of a random resource, live or released.
            const Resource& r = std::next(model.begin(), pick(model.size()))->second;
            for (const std::uint64_t a : {r.base, r.base + r.bytes - 1, r.base + r.bytes, r.base - 1})
                if (!agree(index, model, a, 1 + pick(8), "edge")) return 1;
        }
    }
    if (!synced(index, model)) {
        std::fprintf(stderr, "address index: a record and the index differ\n");
        return 1;
    }
    int live = 0;
    for (const auto& [key, r] : model) live += r.refs > 0;
    std::printf("address index: %d lookups agree with a scan of %zu resources (%d live) across release, re-reference, "
                "address reuse, re-note and span-end edges\n", lookups, model.size(), live);
    return 0;
}
