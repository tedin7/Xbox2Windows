#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace x2w_startup {
template <class Resource>
class ResourceMapCounts {
    std::unordered_map<Resource, std::unordered_map<unsigned, std::uint64_t>> counts_;

public:
    void record_map(Resource resource, unsigned subresource)
    {
        ++counts_[resource][subresource];
    }

    bool consume_map(Resource resource, unsigned subresource)
    {
        const auto wrapper = counts_.find(resource);
        if (wrapper == counts_.end()) return false;
        const auto entry = wrapper->second.find(subresource);
        if (entry == wrapper->second.end()) return false;
        if (--entry->second == 0) wrapper->second.erase(entry);
        if (wrapper->second.empty()) counts_.erase(wrapper);
        return true;
    }

    void forget(Resource resource) { counts_.erase(resource); }
    std::size_t size() const { return counts_.size(); }
};
}
