#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <unordered_map>

namespace x2w_startup {

inline std::optional<std::uint64_t> mesh_dispatch_groups(std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
    // The supported DispatchMesh path uses the base grid limits for all three dimensions.
    if (x >= 65536 || y >= 65536 || z >= 65536) return {};
    const auto groups = std::uint64_t{x} * y * z;
    if (groups > (1u << 22)) return {};
    return groups;
}

class RefusedDispatchCensus {
public:
    enum class Kind { compute, mesh57 };
    explicit RefusedDispatchCensus(Kind kind) : kind_(kind) {}

    // The caller serializes notes and reports. True at each power of two.
    bool note(std::uint64_t key, const char* reason, std::optional<std::uint64_t> groups = {})
    {
        auto& entry = keys_.try_emplace(key, Entry{reason}).first->second;
        ++entry.dispatches;
        if (groups) entry.groups += *groups;
        else if (kind_ == Kind::mesh57) ++entry.unknown_groups;
        ++total_;
        return (total_ & (total_ - 1)) == 0;
    }

    template<class Emit>
    void report(Emit emit) const
    {
        if (!total_) return;
        const char* kind = kind_ == Kind::compute ? "compute" : "mesh57";
        char line[384];
        std::snprintf(line, sizeof(line), "x2w-render %s refused by pipeline key: %zu keys, %llu dispatches",
                      kind, keys_.size(), static_cast<unsigned long long>(total_));
        emit(line);

        // Keep only the highest counts. Reporting can also run during DLL teardown, so use stack storage.
        std::array<const Keys::value_type*, 40> top{};
        auto end = top.begin();
        const auto before = [](const auto* a, const auto* b) {
            if (a->second.dispatches != b->second.dispatches) return a->second.dispatches > b->second.dispatches;
            return a->first < b->first;
        };
        for (const auto& item : keys_) {
            const auto at = std::lower_bound(top.begin(), end, &item, before);
            if (at == top.end()) continue;
            if (end == top.end()) --end;
            std::move_backward(at, end, end + 1);
            *at = &item;
            ++end;
        }
        for (auto at = top.begin(); at != end; ++at) {
            const auto& [key, entry] = **at;
            char group_text[96]{};
            if (kind_ == Kind::mesh57)
                std::snprintf(group_text, sizeof(group_text), ", %llu groups, %llu unknown group counts",
                              static_cast<unsigned long long>(entry.groups),
                              static_cast<unsigned long long>(entry.unknown_groups));
            std::snprintf(line, sizeof(line), "x2w-render %s refused key %016llx: %llu dispatches%s, first refusal: %s",
                          kind, static_cast<unsigned long long>(key), static_cast<unsigned long long>(entry.dispatches),
                          group_text, entry.first_refusal);
            emit(line);
        }
    }

private:
    struct Entry {
        const char* first_refusal;  // Runtime refusal strings have static storage.
        std::uint64_t dispatches = 0, groups = 0, unknown_groups = 0;
    };
    using Keys = std::unordered_map<std::uint64_t, Entry>;
    Kind kind_;
    Keys keys_;
    std::uint64_t total_ = 0;
};

}  // namespace x2w_startup
