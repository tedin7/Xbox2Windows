#pragma once
// Slot bookkeeping for the shared slabs behind the physical pages (#1199). Standard C++ only, so it builds and
// is checked on Linux as well as Windows: the Windows calls that create a slab and map a slot stay in
// xmem_mapped_pages.cpp and are handed in as callables.
//
// A slab holds 4096 slots (256 MiB of 64 KiB pages). Slots are handed out in order from the newest slab; a
// released slot goes back to an ordered set and is reused lowest first, so pages that are reused together
// are adjacent slots again. Slabs are never closed while the object lives. A slab is only reserved; its slots
// are committed by the caller on first use. This class reports how many slabs are open and how many slots are used.
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace x2w_xmem {

template <class Handle>
class SlabSlots {
public:
    static constexpr std::uint64_t slots_per_slab = 4096;

    struct Slot {
        Handle slab{};
        std::size_t slab_index = 0;
        std::uint64_t offset = 0;  // bytes from the start of the slab
    };

    explicit SlabSlots(std::uint64_t slot_bytes) : slot_bytes_(slot_bytes) {}

    std::uint64_t slab_bytes() const { return slots_per_slab * slot_bytes_; }

    // The released slot with the lowest (slab, offset), left in place until take_lowest_free().
    bool lowest_free(Slot& out) const
    {
        if (free_.empty()) return false;
        const auto& slot = *free_.begin();
        out = Slot{slabs_[slot.first], slot.first, slot.second};
        return true;
    }

    void take_lowest_free()
    {
        if (!free_.empty()) free_.erase(free_.begin());
    }

    // The next never-used slot of the newest slab. When there is none, make(Handle&) creates a slab; if it
    // returns false the state is unchanged. If recording the new slab throws, discard(handle) releases it and
    // the exception continues.
    template <class Make, class Discard>
    bool take_fresh(Make&& make, Discard&& discard, Slot& out)
    {
        if (slabs_.empty() || fresh_used_ == slots_per_slab) {
            Handle slab{};
            if (!make(slab)) return false;
            try {
                slabs_.push_back(slab);
            } catch (...) {
                discard(slab);
                throw;
            }
            fresh_used_ = 0;
        }
        out = Slot{slabs_.back(), slabs_.size() - 1, fresh_used_ * slot_bytes_};
        ++fresh_used_;
        return true;
    }

    // Undoes the last take_fresh() when the caller could not record the slot. The slab stays open.
    void unwind_fresh(const Slot& slot)
    {
        if (!slabs_.empty() && fresh_used_ && slot.slab_index == slabs_.size() - 1 &&
            slot.offset == (fresh_used_ - 1) * slot_bytes_)
            --fresh_used_;
    }

    // Returns a slot. False, with nothing changed, for a slot that was never handed out or is already free.
    bool release(std::size_t slab_index, std::uint64_t offset)
    {
        if (slab_index >= slabs_.size() || offset % slot_bytes_ || offset >= slab_bytes()) return false;
        if (slab_index == slabs_.size() - 1 && offset / slot_bytes_ >= fresh_used_) return false;
        return free_.emplace(slab_index, offset).second;
    }

    const std::vector<Handle>& slabs() const { return slabs_; }
    std::uint64_t slab_count() const { return slabs_.size(); }
    // Slots handed out and not released.
    std::uint64_t slots_in_use() const { return handed_out() - free_.size(); }
    // Slots of the open slabs that are not in use: released ones and those never handed out.
    std::uint64_t slots_free() const { return slabs_.size() * slots_per_slab - slots_in_use(); }

private:
    std::uint64_t handed_out() const
    {
        return slabs_.empty() ? 0 : (slabs_.size() - 1) * slots_per_slab + fresh_used_;
    }

    std::uint64_t slot_bytes_;
    std::vector<Handle> slabs_;
    std::uint64_t fresh_used_ = 0;  // slots handed out from the newest slab
    std::set<std::pair<std::size_t, std::uint64_t>> free_;  // (slab index, offset)
};

}  // namespace x2w_xmem
