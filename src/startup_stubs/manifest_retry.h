// When to look again for a shader cache manifest (<key>.txt) that was absent.
//
// The graphics and compute translation paths look for a missing manifest again on a later draw or Dispatch, because
// the translator (run.ps1 -Translate) can publish it while the title runs. Each look opens a file in the shader cache
// while the caller holds draw_lock, which every draw and Dispatch takes. With a fixed 1 s retry, many keys that
// never get a translation each cost one file open per second, and every other list call waits on that lock.
//
// The wait starts at 1 s after the first miss and doubles with each further miss up to 32 s. A translation published
// during the run is still found, within 32 s at the latest; a key that is never translated costs one look every 32 s.
#pragma once

#include <windows.h>

namespace x2w_startup {

constexpr ULONGLONG manifest_retry_first_ms = 1000;
constexpr unsigned manifest_retry_max_doublings = 5;  // 1 s, 2 s, 4 s, 8 s, 16 s, then 32 s

// The wait after the (misses + 1)-th miss in a row.
constexpr ULONGLONG manifest_retry_delay_ms(unsigned misses) noexcept
{
    return manifest_retry_first_ms << (misses < manifest_retry_max_doublings ? misses : manifest_retry_max_doublings);
}

struct ManifestRetry {
    ULONGLONG at = 0;     // GetTickCount64 value of the next look, 0 when none is pending
    unsigned misses = 0;  // misses in a row since the manifest was last found

    bool pending() const noexcept { return at != 0; }
    bool due(ULONGLONG now) const noexcept { return at != 0 && now >= at; }
    void missed(ULONGLONG now) noexcept
    {
        at = now + manifest_retry_delay_ms(misses);
        if (misses < manifest_retry_max_doublings) ++misses;
    }
    void found() noexcept
    {
        at = 0;
        misses = 0;
    }
};

}  // namespace x2w_startup
