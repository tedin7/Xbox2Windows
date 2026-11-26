// Which desktop textures a command list or a submission names, and when the GPU is done with them (X2W_D3D12_RESIDENCY=1).
// Pure bookkeeping: no D3D12 calls. The device code calls note_use while a list records, discard when a list is
// reset or dropped, submit after ExecuteCommandLists once a fence value was signalled behind it, and complete with
// the value the fence reports. busy(id) is the fence rule of texture_residency.h: true from the first recorded use
// until the fence of the last submission that named the texture has completed.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace x2w_residency {

class Flights {
public:
    // A recorded draw, copy or dispatch on `list` names the texture.
    void note_use(const void* list, std::uint64_t id)
    {
        auto& ids = open_[list];
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    }

    // The list was reset, closed without submission or released: its record is gone.
    void discard(const void* list) { open_.erase(list); }

    // After ExecuteCommandLists of `lists` on `queue`, `value` was signalled behind them. The record moves to a flight.
    // `signalled` false means no fence could be recorded: the textures stay busy for the process.
    void submit(const std::vector<const void*>& lists, std::size_t queue, std::uint64_t value, bool signalled)
    {
        Flight flight{queue, value, {}};
        for (const void* list : lists) {
            const auto found = open_.find(list);
            if (found == open_.end()) continue;
            for (std::uint64_t id : found->second)
                if (std::find(flight.ids.begin(), flight.ids.end(), id) == flight.ids.end()) flight.ids.push_back(id);
            // A list may be submitted again without being re-recorded (a bundle, a list the title reuses): keep its record.
        }
        if (flight.ids.empty()) return;
        if (!signalled) {
            for (std::uint64_t id : flight.ids) kept_.push_back(id);
            return;
        }
        flights_.push_back(std::move(flight));
    }

    // The fence of `queue` has completed `completed`: flights up to it are over.
    void complete(std::size_t queue, std::uint64_t completed)
    {
        flights_.erase(std::remove_if(flights_.begin(), flights_.end(),
                                      [&](const Flight& f) { return f.queue == queue && f.value <= completed; }),
                       flights_.end());
    }

    // The texture is gone for good (its resource was released).
    void forget(std::uint64_t id)
    {
        for (auto& list : open_) list.second.erase(std::remove(list.second.begin(), list.second.end(), id), list.second.end());
        for (Flight& f : flights_) f.ids.erase(std::remove(f.ids.begin(), f.ids.end(), id), f.ids.end());
        kept_.erase(std::remove(kept_.begin(), kept_.end(), id), kept_.end());
    }

    bool busy(std::uint64_t id) const
    {
        if (std::find(kept_.begin(), kept_.end(), id) != kept_.end()) return true;
        for (const auto& list : open_)
            if (std::find(list.second.begin(), list.second.end(), id) != list.second.end()) return true;
        for (const Flight& f : flights_)
            if (std::find(f.ids.begin(), f.ids.end(), id) != f.ids.end()) return true;
        return false;
    }

    std::size_t open_lists() const { return open_.size(); }
    std::size_t flights() const { return flights_.size(); }

private:
    struct Flight {
        std::size_t queue;
        std::uint64_t value;
        std::vector<std::uint64_t> ids;
    };
    std::unordered_map<const void*, std::vector<std::uint64_t>> open_;
    std::vector<Flight> flights_;
    std::vector<std::uint64_t> kept_;
};

}  // namespace x2w_residency
