#include "startup_stubs/present_readback_state.h"

#include <cstdio>
#include <cstdlib>

namespace {
using x2w_startup::FenceCompletion;
using x2w_startup::PresentReadbackState;
using x2w_startup::fence_completion;

void require(bool ok, const char* message)
{
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}
}

int main()
{
    PresentReadbackState state;
    require(!state.pending() && !state.unusable(), "initial readback not reusable");
    state.submitted(7);
    require(state.pending(), "submission not retained");
    require(state.observe(6) == FenceCompletion::pending && state.pending(),
            "premature completion would release the submitted surface");
    require(state.observe(7) == FenceCompletion::completed && !state.pending() && !state.unusable(),
            "completed submission did not allow reuse");
    state.submitted(8);
    require(state.observe(7) == FenceCompletion::pending && state.pending(),
            "stale event would publish a later submission");
    require(state.observe(9) == FenceCompletion::completed && !state.pending(),
            "past-target completion did not settle submission");
    state.submitted(10);
    require(state.observe(UINT64_MAX) == FenceCompletion::removed && state.unusable(),
            "device removal would publish or allow reuse");

    PresentReadbackState failed;
    failed.signal_failed();
    require(failed.unusable(), "failed signal would allow reuse");
    require(fence_completion(UINT64_MAX - 1, UINT64_MAX - 1) == FenceCompletion::completed,
            "highest usable fence value rejected");
    require(fence_completion(UINT64_MAX, UINT64_MAX) == FenceCompletion::removed,
            "reserved fence value accepted as completion");

    // Ownership disposition: retire() must release the held references exactly once, only for a
    // confirmed completion, never while pending or removed, and a drain registered for one
    // submission must not retire a different pending target.
    {
        int released = 0;
        const auto release = [&] { ++released; };
        PresentReadbackState retired;
        require(retired.retire(5, release) == FenceCompletion::completed && released == 0,
                "a never-submitted state has no ownership to release");
        retired.submitted(20);
        require(retired.retire(19, release) == FenceCompletion::pending && released == 0 && retired.pending(),
                "a pending fence would retire ownership");
        require(retired.retire(20, release) == FenceCompletion::completed && released == 1 && !retired.pending(),
                "a completed fence did not retire ownership");
        require(retired.retire(20, release) == FenceCompletion::completed && released == 1,
                "a second completed observation retired ownership again");
        retired.submitted(21);
        require(retired.retire(UINT64_MAX, release) == FenceCompletion::removed && released == 1 && retired.unusable(),
                "device removal retired ownership");
        require(retired.retire(21, release) == FenceCompletion::removed && released == 1,
                "an unusable state retired ownership");
    }
    {
        int released = 0;
        PresentReadbackState drained;
        drained.submitted(30);
        // The drain was registered for 29; the fence completed but the pending target is a later
        // submission, so the drain must not release it.
        require(drained.retire(30, [&] { ++released; }, 29) == FenceCompletion::completed && released == 0 && drained.pending(),
                "a drain retired a submission it was not registered for");
        // The drain for the current target releases it exactly once.
        require(drained.retire(30, [&] { ++released; }, 30) == FenceCompletion::completed && released == 1 && !drained.pending(),
                "a drain for the current target did not retire ownership");
        // A drain registered for the settled submission observes completed without releasing again.
        drained.submitted(31);
        // Completed fence value, but the drain is for the settled earlier submission: it must
        // not release or settle the later target it does not own.
        require(drained.retire(31, [&] { ++released; }, 30) == FenceCompletion::completed && released == 1 && drained.pending(),
                "a stale drain retired or settled a later submission");
    }
    std::puts("present readback: pending, stale event, completion, failed signal, removal and "
              "ownership retirement controls passed");
}
