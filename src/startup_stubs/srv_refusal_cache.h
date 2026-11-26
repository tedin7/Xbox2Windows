#pragma once

// A bounded negative cache for refused buffer memory SRVs (device slot 70, reason buffer_unavailable).
//
// A refused SRV used to repeat the whole resolution on every call: the placement scans, the resource index
// scan, and the memory questions of placed_buffer_range.h (a region-information call and, for partly committed
// or mapped allocations, a VirtualQuery walk that costs milliseconds per GiB). A title that re-publishes the same
// view every frame paid that for every refusal.
//
// The answer is not remembered blindly: placed_buffer_range.h explains why title memory may change behind the
// stand-ins. A refusal is reused only while all of this still holds:
//   - the same request: address, span and every view / resource field the resolution reads (SrvRefusalKey);
//   - the same stand-in state: the placement generation, which advances whenever a resource is indexed, retired,
//     published or released, or an unplaced buffer is created or dropped (note_placement_change);
//   - an entry younger than kMaxAgeMs and used fewer than kMaxHits times.
// Three kinds of refusal exist, each with its own exact proof (SrvRefusalKind):
//   - independent: decided by the request and by stand-in state alone (a live resource overlaps, the span has no
//     page extent, a placed buffer owns the page, the found buffer is too small for the view). The key and the
//     placement generation prove it;
//   - fully_committed: the allocation holding the address is private and every byte of it is committed, and the
//     buffer does not fit the allocation. A region-information stamp (base, size, committed bytes, flags) with the
//     committed bytes equal to the size cannot hide a page move: there is no uncommitted page left to move;
//   - witness_page: the walk from the allocation base found a page that is not committed (or belongs to another
//     allocation) before the end of the buffer. The refusal holds exactly as long as that one page is still not
//     committed in the same allocation, so a hit re-asks only that page (the caller's still_refuses) and the
//     allocation base. Commit and decommit happen behind the stand-ins, and a committed-byte total cannot see pages
//     that move with the same total (reserve 128 KiB, commit the upper half, refuse; decommit it, commit the lower
//     half, retry at once: the totals are equal but the witness page is now committed, so the entry misses).
// A refusal decided by a later step that no proof covers is not cached. A miss runs the full resolution unchanged.
// A hit never refuses what the full path would publish.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace x2w_startup {

inline std::atomic<std::uint64_t>& srv_placement_generation()
{
    static std::atomic<std::uint64_t> generation{1};
    return generation;
}

// Call after any change of what existing_memory_buffer / unplaced_memory_buffer can find.
inline void note_placement_change()
{
    srv_placement_generation().fetch_add(1, std::memory_order_acq_rel);
}

// Call with the reference count a resource has just reached by add_ref / query_interface: the step from zero back to
// one makes it live again for existing_memory_buffer, so a refusal cached while it was released is stale.
inline void note_resource_reference(long refs_after_increment)
{
    if (refs_after_increment == 1) note_placement_change();
}

struct SrvRefusalKey {
    std::uint64_t device = 0;      // the device wrapper: every placement lookup is per device
    std::uint64_t address = 0;
    std::uint64_t span_bytes = 0;
    std::uint64_t view_a = 0;      // format, dimension, component mapping
    std::uint64_t view_b = 0;      // first element
    std::uint64_t view_c = 0;      // element count, stride, flags
    std::uint64_t resource_a = 0;  // dimension, flags
    std::uint64_t resource_b = 0;  // width
    bool operator==(const SrvRefusalKey& o) const
    {
        return device == o.device && address == o.address && span_bytes == o.span_bytes && view_a == o.view_a && view_b == o.view_b &&
               view_c == o.view_c && resource_a == o.resource_a && resource_b == o.resource_b;
    }
};

enum class SrvRefusalKind : std::uint8_t { independent, fully_committed, witness_page };

struct SrvRefusalProof {
    SrvRefusalKind kind = SrvRefusalKind::independent;
    std::uint64_t witness = 0;  // witness_page: an address of the page that is not committed
    std::uint64_t base = 0;     // witness_page: the allocation base the walk started from
    bool proven = false;        // false: nothing proves this refusal, it is not cached
};

struct SrvRefusalStamp {
    bool valid = false;  // the region information could be read
    std::uint64_t generation = 0;
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint64_t commit = 0;
    std::uint64_t flags = 0;
    bool operator==(const SrvRefusalStamp& o) const
    {
        return valid && o.valid && generation == o.generation && base == o.base && size == o.size &&
               commit == o.commit && flags == o.flags;
    }
};

class SrvRefusalCache {
public:
    static constexpr std::size_t kSlots = 256;
    static constexpr std::uint64_t kMaxAgeMs = 250;
    static constexpr std::uint32_t kMaxHits = 1024;

    // True when this request was refused under the same stand-in and memory state a moment ago.
    // still_refuses(proof) answers a witness_page entry: the page is still not committed in the allocation.
    template <class StillRefuses>
    bool hit(const SrvRefusalKey& key, const SrvRefusalStamp& stamp, std::uint64_t now_ms, StillRefuses&& still_refuses)
    {
        SrvRefusalProof proof;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const Entry& e = entries_[slot(key)];
            if (!e.used || !(e.key == key) || now_ms < e.noted_ms || now_ms - e.noted_ms > kMaxAgeMs || e.hits >= kMaxHits ||
                e.stamp.generation != stamp.generation)
                return false;
            proof = e.proof;
            if (proof.kind == SrvRefusalKind::fully_committed && !(e.stamp == stamp)) return false;
            // base 0: the address page itself was not committed, no allocation was involved.
            if (proof.kind == SrvRefusalKind::witness_page && proof.base != 0 && !(stamp.valid && stamp.base == proof.base))
                return false;
        }
        // The page question is asked outside the lock; the entry is re-checked before it counts a hit.
        if (proof.kind == SrvRefusalKind::witness_page && !still_refuses(proof)) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        Entry& e = entries_[slot(key)];
        if (!e.used || !(e.key == key) || e.hits >= kMaxHits || e.stamp.generation != stamp.generation) return false;
        ++e.hits;
        return true;
    }

    // Remember a refusal the full resolution just produced, with the proof that keeps it exact. A proof that needs the
    // memory stamp is refused when the stamp could not be read.
    void note(const SrvRefusalKey& key, const SrvRefusalStamp& stamp, std::uint64_t now_ms, const SrvRefusalProof& proof)
    {
        if (!proof.proven) return;
        const bool needs_stamp = proof.kind == SrvRefusalKind::fully_committed ||
                                 (proof.kind == SrvRefusalKind::witness_page && proof.base != 0);
        if (needs_stamp && !stamp.valid) return;
        if (proof.kind == SrvRefusalKind::witness_page && proof.witness == 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        entries_[slot(key)] = Entry{true, key, stamp, now_ms, 0, proof};
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (Entry& e : entries_) e = Entry{};
    }

private:
    struct Entry {
        bool used = false;
        SrvRefusalKey key;
        SrvRefusalStamp stamp;
        std::uint64_t noted_ms = 0;
        std::uint32_t hits = 0;
        SrvRefusalProof proof;
    };

    static std::size_t slot(const SrvRefusalKey& k)
    {
        std::uint64_t h = 1469598103934665603ull;
        for (std::uint64_t v : {k.device, k.address, k.span_bytes, k.view_a, k.view_b, k.view_c, k.resource_a, k.resource_b})
            h = (h ^ v) * 1099511628211ull;
        return static_cast<std::size_t>((h ^ (h >> 32)) % kSlots);
    }

    std::mutex mutex_;
    std::array<Entry, kSlots> entries_{};
};

} // namespace x2w_startup
