#pragma once

#include <cstdint>

namespace x2w_startup {

enum class FenceCompletion { pending, completed, removed };

constexpr FenceCompletion fence_completion(std::uint64_t observed, std::uint64_t target) noexcept
{
    // D3D12 returns UINT64_MAX on device removal, not successful completion.
    if (observed == UINT64_MAX) return FenceCompletion::removed;
    return observed >= target ? FenceCompletion::completed : FenceCompletion::pending;
}

} // namespace x2w_startup
