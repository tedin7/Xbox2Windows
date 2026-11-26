#pragma once

// Validity stamps for the translate-once view cache (pilot rules: when a cached translation of a descriptor must be dropped).
//
// A cached translation was built from three things: the descriptor entry the title wrote (heap slot), the resource behind it,
// and the heap the slot lives in. It is reusable only while none of them changed. Instead of erasing entries when an event
// happens (a heap release or a resource Release would have to scan every slot under a lock), each event BUMPS a counter and a
// lookup COMPARES: O(1) at the event, O(1) at the lookup, and a stale entry can never be returned because its stamp is behind.
//
//   heap serial   taken once per descriptor heap, never reused, so a new heap at a freed heap's title address cannot hit
//   entry stamp   bumped by every write or copy-destination of the slot (CreateXxxView, CopyDescriptors*)
//   resource stamp bumped when the resource is destroyed, materializes, has its memory rewritten, or is re-placed
//   epoch         bumped by a device reset / removal: everything older is stale
//
// The class owns no title or desktop objects; callers pass plain ids (heap serial, slot index, resource wrapper address).
// Thread safety: one mutex; a lookup is a handful of map reads, far cheaper than the rebuild it replaces.

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace x2w_startup {

struct ViewStamp {
    std::uint64_t epoch{};
    std::uint64_t heap_serial{};
    std::uint32_t slot{};
    std::uint64_t entry_stamp{};
    std::uint64_t resource{};  // 0: the view names no resource (null view, sampler, CBV of a title address)
    std::uint64_t resource_stamp{};
};

class ViewStamps {
public:
    // Deliberately wrong behaviour for the negative controls in tests/startup_view_stamps_check.cpp. Never set in the runtime.
    enum class Fault { none, write, copy, heap, serial, resource, epoch, copy_stale };
    Fault fault{Fault::none};

    std::uint64_t new_heap()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint64_t serial = fault == Fault::serial ? 1 : ++heap_counter_;  // the serial fault reuses one
        live_heaps_[serial] = true;
        return serial;
    }

    // Rule 3: a released heap invalidates all its entries at once (its serial is dead).
    void release_heap(std::uint64_t serial)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fault != Fault::heap) live_heaps_[serial] = false;
    }

    // Rule 1: the title wrote the slot (any CreateXxxView), also when the words are identical.
    void write_entry(std::uint64_t serial, std::uint32_t slot)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fault != Fault::write) entry_stamps_[key(serial, slot)] = ++clock_;
    }

    // Rule 2: a copy overwrites the destination slot. The destination gets a fresh stamp; what it now translates to is the
    // SOURCE's translation, rebased onto the destination, and only if the source was current when copied.
    // Returns the rebased stamp, or a stamp that is never current when the source was stale.
    ViewStamp copy_entry(const ViewStamp& source, std::uint64_t dest_serial, std::uint32_t dest_slot)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool source_ok = fault == Fault::copy_stale || current_locked(source);
        if (fault != Fault::copy) entry_stamps_[key(dest_serial, dest_slot)] = ++clock_;
        ViewStamp out = source;
        out.heap_serial = dest_serial;
        out.slot = dest_slot;
        out.entry_stamp = entry_stamps_[key(dest_serial, dest_slot)];
        if (!source_ok) out.epoch = ~std::uint64_t{0};  // never current
        return out;
    }

    // Rules 5-8: the resource was destroyed, materialized, had its memory rewritten, or was re-placed / aliased.
    void bump_resource(std::uint64_t resource)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fault != Fault::resource) resource_stamps_[resource] = ++clock_;
    }

    // Rule 9: device reset, removal or swap-chain resize.
    void reset_all()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fault != Fault::epoch) ++epoch_;
    }

    // The stamp to store next to a translation that was just built for (heap serial, slot) and its resource.
    ViewStamp snapshot(std::uint64_t serial, std::uint32_t slot, std::uint64_t resource)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ViewStamp s;
        s.epoch = epoch_;
        s.heap_serial = serial;
        s.slot = slot;
        s.entry_stamp = lookup(entry_stamps_, key(serial, slot));
        s.resource = resource;
        s.resource_stamp = resource ? lookup(resource_stamps_, resource) : 0;
        return s;
    }

    // May the cached translation stored with this stamp still be used?
    bool current(const ViewStamp& s)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_locked(s);
    }

private:
    static std::uint64_t key(std::uint64_t serial, std::uint32_t slot) { return serial << 32 ^ slot; }
    static std::uint64_t lookup(const std::unordered_map<std::uint64_t, std::uint64_t>& m, std::uint64_t k)
    {
        const auto it = m.find(k);
        return it == m.end() ? 0 : it->second;
    }
    bool current_locked(const ViewStamp& s)
    {
        if (s.epoch != epoch_) return false;
        const auto heap = live_heaps_.find(s.heap_serial);
        if (heap == live_heaps_.end() || !heap->second) return false;
        if (s.entry_stamp != lookup(entry_stamps_, key(s.heap_serial, s.slot))) return false;
        if (s.resource && s.resource_stamp != lookup(resource_stamps_, s.resource)) return false;
        return true;
    }

    std::mutex mutex_;
    std::uint64_t clock_{};
    std::uint64_t epoch_{};
    std::uint64_t heap_counter_{};
    std::unordered_map<std::uint64_t, bool> live_heaps_;
    std::unordered_map<std::uint64_t, std::uint64_t> entry_stamps_;
    std::unordered_map<std::uint64_t, std::uint64_t> resource_stamps_;
};

}  // namespace x2w_startup
