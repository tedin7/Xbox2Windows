#pragma once

#include <cstdint>

namespace x2w_startup {
// The caller supplies the platform operations; no allocator may be reused unless this returns reached.
enum class FenceWait { reached, removed, arm_failed, wait_failed };
enum class FenceSlice { signaled, timeout, failed };
template <class Operations>
FenceWait wait_for_submission(Operations& ops, std::uint64_t submitted)
{
    const auto start = ops.now();
    bool warned = false;
    bool armed = false;
    for (;;) {
        if (ops.removed()) return FenceWait::removed;
        if (ops.completed() >= submitted) return FenceWait::reached;
        if (!armed) {
            if (!ops.arm(submitted)) return FenceWait::arm_failed;
            armed = true;
        }
        const auto result = ops.wait();
        if (result == FenceSlice::failed) return FenceWait::wait_failed;
        if (result == FenceSlice::signaled) armed = false;
        if (!warned && ops.now() - start > 2000) {
            warned = true;
            ops.slow();
        }
    }
}
}  // namespace x2w_startup
