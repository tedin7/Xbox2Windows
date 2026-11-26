#pragma once

#include <cstdint>

// When a pipeline key was first and last used by a draw or a Dispatch, in milliseconds since the process started (the
// clock the debugger's events.txt uses), so a report can tell the keys used while loading from those used after it.
struct KeyUse {
    std::uint64_t first_ms = 0, last_ms = 0, uses = 0;
    std::uint64_t reported_uses = 0;  // the use count the last census line carried: only changed keys are printed again

    void note(std::uint64_t now_ms) noexcept
    {
        if (!uses) first_ms = now_ms;
        last_ms = now_ms;
        ++uses;
    }
};
