// Experimental (docs/startup-runtime.md): the game-save write log shows the first 20 writes in full, every failure (up
// to 20) and the totals every 25 writes, so a tester run tells whether the saves reached the disk.
#include "../src/startup_stubs/save_write_log.h"

#include <cstdio>
#include <string>

static int failed = 0;
static void expect(bool condition, const char* what)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        failed = 1;
    }
}
static bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

int main()
{
    x2w_save::WriteLog log;
    const std::string first = log.record("profile", 2, 4096, true, 0, 0);
    expect(has(first, "write #1: container \"profile\", 2 actions, 4096 bytes -> ok"), "the first write is logged in full");

    // Writes 2..20 ok, one of them failing: all logged in full, the failure marked with its Windows error.
    std::string failure;
    for (int i = 2; i <= 20; ++i) {
        const std::string text = i == 7 ? log.record("settings", 1, 10, false, 4, 3) : log.record("c", 1, 10, true, 0, 0);
        expect(!text.empty(), "writes up to 20 are logged");
        if (i == 7) failure = text;
    }
    expect(has(failure, "FAILED (status 4, Windows error 3)"), "a failed write shows its status and Windows error");

    // After 20: successes are silent, a failure is logged, the totals come at 25.
    expect(log.record("c", 1, 10, true, 0, 0).empty(), "write 21 succeeded: silent");
    expect(has(log.record("c", 1, 10, false, 4, 5), "write #22: container \"c\", 1 actions, 10 bytes -> FAILED"), "a later failure is logged");
    expect(log.record("c", 1, 10, true, 0, 0).empty(), "write 23 silent");
    expect(log.record("c", 1, 10, true, 0, 0).empty(), "write 24 silent");
    const std::string totals = log.record("c", 1, 10, true, 0, 0);
    // 25 writes, 2 failed, bytes of the 23 that succeeded: 4096 + 18 * 10 (writes 2..20 except 7) + 3 * 10 (21, 23, 24) + 10 (25).
    expect(has(totals, "totals: 25 writes, 2 failed, 4316 bytes written"), "the totals line at write 25");
    return failed;
}
