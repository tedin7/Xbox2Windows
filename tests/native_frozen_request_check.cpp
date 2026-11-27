#include "frozen_compute_requests.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
x2w::ComputeRequest fixture(uint64_t key) {
    x2w::ComputeRequest r{"code", "data", "root", std::string(96, 0), std::string(28, 0), {}, key};
    const auto length = [&](size_t offset, uint64_t n) { std::memcpy(r.description.data() + offset, &n, sizeof(n)); };
    length(48, r.code.size());
    length(64, r.data.size());
    length(16, r.sa.size());
    return r;
}
void require(bool ok, const char *reason) {
    if (!ok) throw std::runtime_error(reason);
}
}

int main() {
    try {
        x2w::FrozenComputeRequests pool(4096, 2);
        auto first = fixture(1);
        pool.put(first);
        first.code[0] = 'x';
        auto saved = pool.take(1);
        require(saved && saved->code == "code", "snapshot changed with caller");
        x2w::validate_compute_request_shape(*saved);
        require(!pool.take(1), "consumed snapshot still present");
        pool.put(first);
        pool.put(fixture(2));
        pool.put(fixture(2));
        pool.put(fixture(3));
        require(!pool.take(1) && pool.take(2) && pool.take(3), "FIFO eviction or duplicate key failed");
        auto stats = pool.stats();
        require(stats.hits == 3 && stats.misses == 2 && stats.evicted == 1 && !stats.bytes && !stats.entries,
                "handoff accounting mismatch");
        x2w::FrozenComputeRequests tiny(32), disabled(0);
        tiny.put(first);
        disabled.put(first);
        require(!tiny.take(1) && !disabled.enabled() && !disabled.take(1), "byte budget exceeded");
        auto broken = *saved;
        broken.description[48] ^= 1;
        try {
            x2w::validate_compute_request_shape(broken);
        } catch (const std::runtime_error &) {
            return 0;
        }
        throw std::runtime_error("inconsistent description accepted");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
