#pragma once

#include "shader_compute_request.h"
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace x2w {
// Slot 67 publishes requests before their first dispatch. Keep recent captures
// until dispatch takes ownership; eviction leaves the normal file replay intact.
class FrozenComputeRequests {
    const size_t max_bytes_, max_entries_;
    size_t bytes_{};
    std::mutex lock_;
    std::list<ComputeRequest> requests_;
    std::unordered_map<uint64_t, decltype(requests_)::iterator> index_;
    uint64_t hits_{}, misses_{}, evicted_{};

    static size_t size(const ComputeRequest &r) {
        return r.code.capacity() + r.data.capacity() + r.root.capacity() +
               r.description.capacity() + r.sa.capacity() + r.sb.capacity();
    }

public:
    struct Stats { uint64_t hits, misses, evicted; size_t bytes, entries; };
    FrozenComputeRequests(size_t max_bytes = 32u << 20, size_t max_entries = 4096)
        : max_bytes_(max_bytes), max_entries_(max_entries) {}
    bool enabled() const { return max_entries_ && max_bytes_; }
    Stats stats() {
        std::lock_guard guard(lock_);
        return {hits_, misses_, evicted_, bytes_, requests_.size()};
    }

    void put(ComputeRequest request) {
        const auto bytes = size(request);
        if (!max_entries_ || bytes > max_bytes_) return;
        std::lock_guard guard(lock_);
        if (index_.count(request.key)) return;
        while (requests_.size() >= max_entries_ || bytes_ > max_bytes_ - bytes) {
            const auto &old = requests_.front();
            bytes_ -= size(old);
            index_.erase(old.key);
            requests_.pop_front();
            ++evicted_;
        }
        requests_.push_back(std::move(request));
        auto node = std::prev(requests_.end());
        try {
            index_.emplace(node->key, node);
        } catch (...) {
            requests_.pop_back();
            throw;
        }
        bytes_ += bytes;
    }

    std::optional<ComputeRequest> take(uint64_t key) {
        std::lock_guard guard(lock_);
        const auto found = index_.find(key);
        if (found == index_.end()) {
            ++misses_;
            return {};
        }
        ++hits_;
        const auto node = found->second;
        bytes_ -= size(*node);
        auto request = std::move(*node);
        requests_.erase(node);
        index_.erase(found);
        return request;
    }
};
}
