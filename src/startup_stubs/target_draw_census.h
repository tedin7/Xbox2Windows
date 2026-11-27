#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace x2w_startup {

struct TargetDrawCensus {
    struct Surface {
        std::uintptr_t resource = 0;
        std::uint64_t width = 0;
        unsigned height = 0;
        int format = 0;
    };

    struct Target {
        Surface surface;
        long first = -1, last = -1, read = -1;
        bool drawn_in_prior_window = false;
        std::uint64_t draws = 0, poisoned = 0, generation = 0;
    };

    struct Refusal {
        std::uint64_t key = 0;
        std::uintptr_t resource = 0;
        const char* cause = nullptr;  // Static name of the most recent refusal site.
        long first = -1, last = -1, window_first = -1;
        std::uint64_t calls = 0, window = 0, since_draw = 0, generation = 0;
    };

    void draw(Surface surface, long frame, bool poisoned)
    {
        auto* row = track(surface);
        if (!row) {
            ++omitted_target_draws;
            return;
        }
        if (!row->draws) row->first = frame;
        row->last = frame;
        ++row->draws;
        row->poisoned += poisoned;
        ++row->generation;
    }

    void read(std::uintptr_t resource, long frame)
    {
        if (auto* row = find(resource); row && row->surface.resource)
            row->read = (std::max)(row->read, frame);
    }

    void refuse(Surface surface, std::uint64_t key, const char* cause, long frame)
    {
        auto* target = track(surface);
        if (!target) {
            ++omitted;
            return;
        }
        const auto hash = key ^ (key >> 32) ^ (surface.resource >> 4) ^ (surface.resource >> 10);
        for (std::size_t probe = 0; probe < refusals.size(); ++probe) {
            auto& row = refusals[(hash + probe) % refusals.size()];
            if (!row.calls) {
                row.key = key;
                row.resource = surface.resource;
                row.first = frame;
            } else if (row.key != key || row.resource != surface.resource) {
                continue;
            }
            if (!row.window++) row.window_first = frame;
            if (row.generation != target->generation) {
                row.since_draw = 0;
                row.generation = target->generation;
            }
            row.cause = cause;
            row.last = frame;
            ++row.calls;
            ++row.since_draw;
            return;
        }
        ++omitted;
    }

    void next_window()
    {
        for (auto& row : targets) {
            row.drawn_in_prior_window |= row.draws != 0;
            row.draws = row.poisoned = 0;
        }
        for (auto& row : refusals) row.window = 0;
        omitted = omitted_target_draws = 0;
    }

    static std::string label(Surface surface)
    {
        char text[96];
        std::snprintf(text, sizeof(text), "%llx (%llux%u format %d)",
                      static_cast<unsigned long long>(surface.resource),
                      static_cast<unsigned long long>(surface.width), surface.height, surface.format);
        return text;
    }

    template <typename Emit>
    void report(Emit emit, long frame, long threshold, bool starvation) const
    {
        for (const auto& target : targets) {
            if (!target.surface.resource) continue;
            const bool starved = starvation && target.drawn_in_prior_window && frame - target.last > threshold &&
                                 target.read > target.last && frame - target.read <= threshold;
            if (!target.draws && !starved &&
                std::none_of(refusals.begin(), refusals.end(), [&](const Refusal& row) {
                    return row.resource == target.surface.resource && row.window;
                })) continue;
            long first = target.draws ? target.first : frame;
            long last = target.draws ? target.last : -1;
            std::uint64_t total = 0;
            std::array<const Refusal*, 3> window{}, since{};
            for (const auto& row : refusals) {
                if (row.resource != target.surface.resource) continue;
                if (row.window) {
                    first = (std::min)(first, row.window_first);
                    last = (std::max)(last, row.last);
                    total += row.window;
                    rank(window, row, &Refusal::window);
                }
                if (starved && row.generation == target.generation && row.since_draw)
                    rank(since, row, &Refusal::since_draw);
            }
            const auto surface = label(target.surface);
            if (target.draws || total) {
                auto line = "x2w-render target " + surface + ": draws " + std::to_string(target.draws) +
                    ", with a null input " + std::to_string(target.poisoned) +
                    " frames " + std::to_string(first) + ".." + std::to_string(last);
                if (total)
                    line += ", not drawn " + std::to_string(total) + " by " + candidates(window, &Refusal::window);
                emit(line.c_str());
            }
            if (starved) {
                auto line = "x2w-render starved target " + surface + ": last drawn frame " + std::to_string(target.last);
                line += since[0] ? ", not drawn since: " + candidates(since, &Refusal::since_draw) : ", starved, no candidate";
                emit(line.c_str());
            }
        }
        if (omitted || omitted_target_draws) {
            const auto line = "x2w-render target census omitted: not drawn " + std::to_string(omitted) +
                              ", target draws " + std::to_string(omitted_target_draws);
            emit(line.c_str());
        }
    }

    std::string top_target(std::uint64_t key) const
    {
        const Refusal* best = nullptr;
        for (const auto& row : refusals)
            if (row.calls && row.key == key && (!best || row.calls > best->calls ||
                                               (row.calls == best->calls && row.resource < best->resource)))
                best = &row;
        if (!best) return {};
        for (const auto& row : targets)
            if (row.surface.resource == best->resource) return label(row.surface);
        return {};
    }

private:
    static constexpr std::size_t target_limit = 512;
    // Keep the table at most half full so reads of non-target textures remain
    // cheap after the tracking limit is reached.
    std::array<Target, target_limit * 2> targets{};
    std::array<Refusal, 1024> refusals{};
    std::size_t target_count = 0;
    std::uint64_t omitted = 0, omitted_target_draws = 0;

    Target* find(std::uintptr_t resource)
    {
        if (!resource) return nullptr;
        const auto hash = (resource >> 4) ^ (resource >> 10);
        for (std::size_t probe = 0; probe < targets.size(); ++probe) {
            auto& row = targets[(hash + probe) % targets.size()];
            if (!row.surface.resource || row.surface.resource == resource) return &row;
        }
        return nullptr;
    }

    Target* track(Surface surface)
    {
        auto* row = find(surface.resource);
        if (!row) return nullptr;
        if (!row->surface.resource) {
            if (target_count == target_limit) return nullptr;
            ++target_count;
        }
        row->surface = surface;
        return row;
    }

    using Count = std::uint64_t Refusal::*;

    static void rank(std::array<const Refusal*, 3>& top, const Refusal& row, Count count)
    {
        for (std::size_t i = 0; i < top.size(); ++i) {
            const auto* slot = top[i];
            if (!slot || row.*count > slot->*count ||
                (row.*count == slot->*count && row.key < slot->key)) {
                for (std::size_t j = top.size() - 1; j > i; --j) top[j] = top[j - 1];
                top[i] = &row;
                return;
            }
        }
    }

    static std::string candidates(const std::array<const Refusal*, 3>& top, Count count)
    {
        std::string result;
        for (const auto* row : top) {
            if (!row) break;
            char text[160];
            std::snprintf(text, sizeof(text), "%s%016llx x%llu (%s)", result.empty() ? "" : ", ",
                          static_cast<unsigned long long>(row->key),
                          static_cast<unsigned long long>(row->*count), row->cause);
            result += text;
        }
        return result;
    }
};

}  // namespace x2w_startup
