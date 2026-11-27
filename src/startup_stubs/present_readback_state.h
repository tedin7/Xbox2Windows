#pragma once

#include "startup_stubs/fence_completion.h"

namespace x2w_startup {

class PresentReadbackState {
    std::uint64_t target_ = 0;
    bool unusable_ = false;

    FenceCompletion classify(std::uint64_t observed) const noexcept
    {
        if (unusable_) return target_ ? FenceCompletion::removed : FenceCompletion::pending;
        return fence_completion(observed, target_);
    }

public:
    void submitted(std::uint64_t target) noexcept { target_ = target; }
    void signal_failed() noexcept { unusable_ = true; }
    bool pending() const noexcept { return target_ != 0; }
    bool unusable() const noexcept { return unusable_; }
    FenceCompletion observe(std::uint64_t observed) noexcept
    {
        const auto state = classify(observed);
        if (state == FenceCompletion::removed) unusable_ = true;
        if (state == FenceCompletion::completed) target_ = 0;
        return state;
    }
    // A stale drain may observe the fence but must not settle another submission.
    template <class Release>
    FenceCompletion retire(std::uint64_t observed, Release&& release, std::uint64_t expected = 0)
    {
        if (expected && expected != target_) return classify(observed);
        const bool held = pending();
        const auto state = observe(observed);
        if (state == FenceCompletion::completed && held) release();
        return state;
    }
};

} // namespace x2w_startup
