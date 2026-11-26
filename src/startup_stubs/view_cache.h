// A cache of resolved views: the desktop descriptor a draw or dispatch needs for a title view, built once and copied
// per call (CopyDescriptorsSimple of one descriptor) instead of rebuilt (CreateShaderResourceView and friends into the
// per-draw heap slot, after a view record lookup and the resource checks).
//
// A view id names its content: note_view() gives equal slot, resource, counter and description one id, so the descriptor
// built for an id never changes while the resource it names is the same object in the same state. The entry is therefore
// valid while three things hold, and a lookup that finds any of them changed is a miss:
//   - the same id sits in the entry (direct-mapped by id; another id in the slot replaces it);
//   - the resource generation recorded at build time is still the resource's current generation: the generation of a
//     resource advances when it is released or retired (its address can be reused by a different resource), when its
//     texture is uploaded again, or when its view-relevant state changes (note_resource_change);
//   - the caller's stamp recorded at build time equals the stamp it passes now (the generation of the texture upload
//     the descriptor was built over, 0 for a buffer or a view that has none);
//   - the cache generation recorded at build time is the current one: it advances when a heap is destroyed or the
//     whole cache is dropped (note_flush).
// The write hook is note_write(id): the title wrote or copied a descriptor, so the entry of that id is dropped before the
// next lookup. Content addressing makes this redundant for the descriptor words, but it keeps the contract simple and
// covers a view re-created with the same words over a different native object.
//
// The cache holds no descriptor memory itself; slot_of(id) names the slot in the caller's persistent CPU-only heap
// (one descriptor per slot). Switch: X2W_VIEW_CACHE=1 (default off until the bench shows a gain).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace x2w_view_cache {

enum class Kind : std::uint8_t { none, srv, uav, cbv, sampler };

struct Stats {
    std::uint64_t hits = 0, misses = 0, publishes = 0, write_drops = 0, resource_drops = 0, flushes = 0, replaced = 0;
};

class Cache {
public:
    explicit Cache(std::uint32_t slots = 1u << 16) : entries_(slots ? slots : 1) {}

    static bool enabled()
    {
        static const bool on = [] {
            const char* value = std::getenv("X2W_VIEW_CACHE");
            return value && value[0] == '1';
        }();
        return on;
    }

    std::uint32_t slot_count() const { return static_cast<std::uint32_t>(entries_.size()); }
    std::uint32_t slot_of(std::uint32_t id) const { return id % slot_count(); }

    // Draw / dispatch side. True on a hit: the descriptor in slot_of(id) is the one built for this id and resource.
    bool lookup(std::uint32_t id, std::uint64_t resource, Kind kind, std::uint64_t stamp = 0)
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        const Entry& e = entries_[slot_of(id)];
        const bool hit = e.valid && e.id == id && e.kind == kind && e.cache_generation == cache_generation_ &&
                         e.resource == resource && e.stamp == stamp && e.resource_generation == generation_of_locked(resource);
        (hit ? hits_ : misses_).fetch_add(1, std::memory_order_relaxed);
        return hit;
    }

    // What the descriptor is about to be built over. Take it BEFORE reading the resource and building: a release, flush or
    // other change that lands while the build runs makes the published entry miss, because publish() records the
    // generations seen here, not the ones current at publish time. (Recording them at publish time would stamp an entry built
    // over a released object with the new generation and serve it.) The caller takes its own stamp at the same moment.
    struct Ticket {
        std::uint64_t resource_generation = 0, cache_generation = 0;
    };
    Ticket begin(std::uint64_t resource) const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return Ticket{generation_of_locked(resource), cache_generation_};
    }

    // Called after the descriptor was built into slot_of(id), with the ticket taken before the build.
    void publish(std::uint32_t id, std::uint64_t resource, Kind kind, std::uint64_t stamp, const Ticket& ticket)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        Entry& e = entries_[slot_of(id)];
        if (e.valid && e.id != id) ++replaced_;
        e = Entry{true, kind, id, resource, ticket.resource_generation, ticket.cache_generation, stamp};
        ++publishes_;
    }

    // Write hook: a descriptor of this id was written or copied.
    void note_write(std::uint32_t id)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        Entry& e = entries_[slot_of(id)];
        if (e.valid && e.id == id) {
            e.valid = false;
            ++write_drops_;
        }
    }

    // The resource was released, retired, uploaded again, or changed what its views show.
    void note_resource_change(std::uint64_t resource)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        ++resource_generation_[resource];
        ++resource_drops_;
    }

    // A heap was destroyed or the cache is dropped as a whole.
    void note_flush()
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        ++cache_generation_;
        ++flushes_;
    }

    Stats stats() const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return Stats{hits_.load(), misses_.load(), publishes_, write_drops_, resource_drops_, flushes_, replaced_};
    }

private:
    struct Entry {
        bool valid = false;
        Kind kind = Kind::none;
        std::uint32_t id = 0;
        std::uint64_t resource = 0, resource_generation = 0, cache_generation = 0, stamp = 0;
    };

    std::uint64_t generation_of_locked(std::uint64_t resource) const
    {
        const auto found = resource_generation_.find(resource);
        return found == resource_generation_.end() ? 0 : found->second;
    }

    mutable std::shared_mutex mutex_;
    std::vector<Entry> entries_;
    std::unordered_map<std::uint64_t, std::uint64_t> resource_generation_;
    std::uint64_t cache_generation_ = 1;
    std::uint64_t publishes_ = 0, write_drops_ = 0, resource_drops_ = 0, flushes_ = 0, replaced_ = 0;
    std::atomic<std::uint64_t> hits_{0}, misses_{0};
};

inline Cache& view_cache()
{
    static Cache cache;
    return cache;
}

}  // namespace x2w_view_cache
