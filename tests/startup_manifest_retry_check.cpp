// The retry schedule for an absent shader cache manifest (startup_stubs/manifest_retry.h).
// The first look again comes 1 s after the first miss, each further miss doubles the wait, and the wait
// stops growing at 32 s. A found manifest starts the schedule over.
#include "startup_stubs/manifest_retry.h"

#include <cstdio>

namespace {

int failures = 0;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main()
{
    using x2w_startup::ManifestRetry;
    using x2w_startup::manifest_retry_delay_ms;

    expect(manifest_retry_delay_ms(0) == 1000, "first miss waits 1 s");
    expect(manifest_retry_delay_ms(1) == 2000, "second miss waits 2 s");
    expect(manifest_retry_delay_ms(4) == 16000, "fifth miss waits 16 s");
    expect(manifest_retry_delay_ms(5) == 32000, "sixth miss waits 32 s");
    expect(manifest_retry_delay_ms(6) == 32000, "the wait stops at 32 s");
    expect(manifest_retry_delay_ms(64) == 32000, "a large miss count does not overflow the shift");
    expect(manifest_retry_delay_ms(~0u) == 32000, "the largest miss count stays at 32 s");

    ManifestRetry retry;
    expect(!retry.pending(), "a fresh schedule has no retry pending");
    expect(!retry.due(0) && !retry.due(~0ull), "a fresh schedule is never due");

    retry.missed(10000);
    expect(retry.pending(), "a miss leaves a retry pending");
    expect(!retry.due(10999), "not due before 1 s");
    expect(retry.due(11000), "due at 1 s");

    retry.missed(11000);
    expect(!retry.due(12999) && retry.due(13000), "second miss: due after 2 s");

    for (int i = 0; i < 20; ++i) retry.missed(100000);
    expect(!retry.due(131999) && retry.due(132000), "many misses: due after 32 s");

    retry.found();
    expect(!retry.pending() && !retry.due(~0ull), "a found manifest clears the retry");
    retry.missed(200000);
    expect(!retry.due(200999) && retry.due(201000), "a later miss starts again at 1 s");

    // One key that never gets a translation: over ten minutes the old fixed 1 s retry looked it up about 600 times;
    // the schedule looks it up 23 times (6 lookups in the first 63 s, then one per 32 s).
    {
        ManifestRetry key;
        unsigned lookups = 0;
        unsigned long long now = 0;
        key.missed(now);
        ++lookups;
        for (; now <= 600000; now += 50)
            if (key.due(now)) {
                key.missed(now);
                ++lookups;
            }
        expect(lookups <= 25, "an absent key is looked up at most 25 times in ten minutes");
        expect(lookups >= 20, "an absent key is still looked up at least 20 times in ten minutes");
    }

    if (failures) return 1;
    std::printf("startup_manifest_retry: all checks passed\n");
    return 0;
}
