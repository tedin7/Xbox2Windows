// Checks src/startup_stubs/list_clear_depth.h without a device: which ClearDepthStencilView calls are forwarded, with
// which flags, and which are refused.
//
//   list_clear_depth_check [--mutate keep-stencil|allow-read-only]
//
// The expectations are written out case by case. Each mutation changes them to those of a wrong decision (a stencil
// clear kept for a format without stencil, a clear of a read-only plane forwarded); each must make the check fail.
#include "startup_stubs/list_clear_depth.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
using namespace x2w_clear_depth;
int failures, cases;
bool keep_stencil, allow_read_only;

void expect(const char* label, const Request& request, Verdict verdict, unsigned flags = 0, unsigned rects = 0)
{
    ++cases;
    const Plan got = plan(request);
    if (got.verdict == verdict && got.flags == flags && got.rect_count == rects) return;
    ++failures;
    std::printf("FAIL %s: got %s (flags %u, rects %u), expected %s (flags %u, rects %u)\n", label, reason(got.verdict), got.flags,
                got.rect_count, reason(verdict), flags, rects);
}
Request make(unsigned flags, bool stencil_format, float depth = 0.25f)
{
    Request request;
    request.flags = flags;
    request.format_has_stencil = stencil_format;
    request.depth = depth;
    return request;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) {
        keep_stencil = std::strcmp(argv[2], "keep-stencil") == 0;
        allow_read_only = std::strcmp(argv[2], "allow-read-only") == 0;
        if (!keep_stencil && !allow_read_only) return 2;
    } else if (argc != 1) {
        return 2;
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    expect("depth only, depth-only format", make(1, false), Verdict::forward, 1);
    expect("depth only, stencil format", make(1, true), Verdict::forward, 1);
    expect("stencil only, stencil format", make(2, true), Verdict::forward, 2);
    expect("both planes, stencil format", make(3, true), Verdict::forward, 3);
    // The stencil bit is dropped for a format without a stencil plane; what is left is still cleared.
    expect("both planes, depth-only format", make(3, false), Verdict::forward, keep_stencil ? 3u : 1u);
    expect("stencil only, depth-only format", make(2, false), keep_stencil ? Verdict::forward : Verdict::nothing_to_clear,
           keep_stencil ? 2u : 0u);
    expect("no flags", make(0, true), Verdict::nothing_to_clear);
    expect("unknown flag bits only", make(0xfc, true), Verdict::nothing_to_clear);
    expect("unknown bits dropped from a valid mask", make(0xff, true), Verdict::forward, 3);

    expect("depth 0", make(1, true, 0.0f), Verdict::forward, 1);
    expect("depth 1", make(1, true, 1.0f), Verdict::forward, 1);
    expect("depth above 1", make(1, true, 1.5f), Verdict::bad_depth);
    expect("depth below 0", make(1, true, -0.25f), Verdict::bad_depth);
    expect("depth NaN", make(1, true, nan), Verdict::bad_depth);
    expect("depth infinite", make(3, true, inf), Verdict::bad_depth);
    // The depth value is ignored when only stencil is cleared.
    expect("stencil only, junk depth", make(2, true, nan), Verdict::forward, 2);

    Request depth_ro = make(1, true);
    depth_ro.depth_read_only = true;
    expect("depth clear, depth plane read-only", depth_ro, allow_read_only ? Verdict::forward : Verdict::read_only_plane,
           allow_read_only ? 1u : 0u);
    depth_ro.flags = 2;
    expect("stencil clear, only the depth plane read-only", depth_ro, Verdict::forward, 2);
    Request stencil_ro = make(3, true);
    stencil_ro.stencil_read_only = true;
    expect("both planes, stencil plane read-only", stencil_ro, Verdict::read_only_plane);
    stencil_ro.flags = 1;
    expect("depth clear, only the stencil plane read-only", stencil_ro, Verdict::forward, 1);
    // A read-only stencil bit does not matter once a depth-only format dropped the stencil clear.
    Request dropped = make(3, false);
    dropped.stencil_read_only = true;
    expect("stencil dropped before the read-only check", dropped, Verdict::forward, 1);

    Request rects = make(1, true);
    rects.rect_count = 3;
    rects.rects_readable = true;
    expect("three readable rectangles", rects, Verdict::forward, 1, 3);
    rects.rects_readable = false;
    expect("three unreadable rectangles", rects, Verdict::unreadable_rects);
    rects.rect_count = 0;
    expect("no rectangles, pointer unreadable", rects, Verdict::forward, 1, 0);

    std::printf("%d cases, %d failures\n", cases, failures);
    if (failures) return 1;
    std::printf("list clear depth checks passed\n");
    return 0;
}
