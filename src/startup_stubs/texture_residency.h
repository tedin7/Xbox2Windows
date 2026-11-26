// Decision logic for evicting desktop textures under video-memory pressure (X2W_D3D12_RESIDENCY=1).
// Pure code: no D3D12 calls and no locks. The caller (d3d12_xs_device.cpp / d3d12_draw.inl) fills the facts and
// performs the Evict / MakeResident calls named by the result. Public docs: ID3D12Device::Evict and MakeResident.
//
// Rules:
//  - a texture is a candidate only when the caller says it is safe: read-only, written by no GPU copy,
//    never named by the title's own Evict/MakeResident, not being materialized;
//  - a texture named by a submission whose fence has not completed is never a candidate (evicting a resource
//    the GPU may still read can remove the device);
//  - candidates unused for at least min_idle frames go oldest first, until the bytes over the low-water mark are covered;
//  - a texture that had to come back within pin_frames of its eviction is pinned for good (thrash guard).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace x2w_residency {

struct Config {
    std::uint32_t min_idle = 120;        // frames without a use before a texture may go
    std::uint32_t pin_frames = 240;      // coming back within this many frames of an eviction pins the texture
    std::uint32_t high_percent = 95;     // start evicting above this share of the budget
    std::uint32_t low_percent = 85;      // stop once usage would fall to this share
};

struct Texture {
    std::uint64_t id = 0;                // caller key (the resource pointer value)
    std::uint64_t bytes = 0;
    std::int64_t last_use = 0;           // frame of the last bind, copy or dispatch that named it
    bool resident = true;
    bool in_flight = false;              // a submission naming it has a fence that has not completed
    bool unsafe = false;                 // gpu_written, title Evict/MakeResident, render target, materialize in progress ...
    bool pinned = false;                 // thrash guard hit
    std::int64_t evicted_at = -1000000;  // frame of the last eviction
};

// Hysteresis: true from the sample that goes above high_percent until the sample that falls to low_percent or under.
class Pressure {
public:
    bool update(std::uint64_t used, std::uint64_t budget, const Config& config)
    {
        if (!budget) return active_ = false;
        const std::uint64_t pct = used * 100 / budget;
        if (pct > config.high_percent) active_ = true;
        else if (pct <= config.low_percent) active_ = false;
        return active_;
    }
    bool active() const { return active_; }
private:
    bool active_ = false;
};

// Ids to evict now, oldest use first. Returns nothing unless pressure is active. `used` and `budget` are the local group figures.
inline std::vector<std::uint64_t> plan_evictions(const std::vector<Texture>& textures, std::uint64_t used, std::uint64_t budget,
                                                 std::int64_t frame, const Config& config, bool pressure_active)
{
    std::vector<std::uint64_t> out;
    if (!pressure_active || !budget) return out;
    const std::uint64_t target = budget / 100 * config.low_percent;
    if (used <= target) return out;
    std::uint64_t to_free = used - target;
    std::vector<const Texture*> candidates;
    for (const Texture& t : textures) {
        if (!t.resident || t.in_flight || t.unsafe || t.pinned || !t.bytes) continue;
        if (frame - t.last_use < static_cast<std::int64_t>(config.min_idle)) continue;
        candidates.push_back(&t);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Texture* a, const Texture* b) { return a->last_use < b->last_use; });
    for (const Texture* t : candidates) {
        if (!to_free) break;
        out.push_back(t->id);
        to_free -= (std::min)(to_free, t->bytes);
    }
    return out;
}

// A used texture that is not resident must be made resident first. If it comes back soon after its eviction it is pinned.
// Returns true when the caller has to call MakeResident (the texture is updated to resident).
inline bool note_use(Texture& t, std::int64_t frame, const Config& config)
{
    t.last_use = frame;
    if (t.resident) return false;
    t.resident = true;
    if (frame - t.evicted_at < static_cast<std::int64_t>(config.pin_frames)) t.pinned = true;
    return true;
}

inline void note_evicted(Texture& t, std::int64_t frame)
{
    t.resident = false;
    t.evicted_at = frame;
}

}  // namespace x2w_residency
