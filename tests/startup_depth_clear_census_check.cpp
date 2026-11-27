// depth_clear_census.h: the depth values a title clears to, the clears the stand-in skipped, and what a decoded
// GREATER/GREATER_EQUAL depth test meets (a view last cleared to 1.0, to 0.0, or never), and the depth-writing draws by
// state (resource, test, viewport, scissor, cull, clip) and last clear, each printed when its count
// reaches a power of two.
#include "startup_stubs/depth_clear_census.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::vector<std::string> lines;
void sink(const char* line)
{
    lines.emplace_back(line);
}

bool has(const std::string& text)
{
    for (const auto& line : lines)
        if (line.find(text) != std::string::npos) return true;
    return false;
}

int failures = 0;
void expect(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void check_values()
{
    x2w_depth_census::DepthClearCensus census(sink);
    for (int i = 0; i < 3; ++i) census.requested(1.0f, 1);
    expect(has("depth=1 flags=0x1 requested 2 times (pair 1 of the first 8)"), "second clear to 1.0 is printed");
    expect(!has("requested 3 times"), "a third clear is not a power of two");
    for (int value = 1; value <= 8; ++value) census.requested(static_cast<float>(value) / 16.0f, 3);
    expect(has("depth=0.4375 flags=0x3 requested 1 times (pair 8 of the first 8)"), "eighth distinct pair is counted alone");
    expect(has("clears with a (depth, flags) pair after the first 8 requested 1 times"), "ninth pair goes to the rest");
    for (int i = 0; i < 4; ++i) census.skipped(x2w_depth_census::Skip::unresolved_view);
    expect(has("skipped (view not resolved) 4 times"), "skipped clears are counted by reason");
}

void check_greater_draws()
{
    lines.clear();
    x2w_depth_census::DepthClearCensus census(sink);
    const int a = 0, b = 0;
    census.depth_draw(&a, x2w_depth_census::func_greater_equal);
    expect(has("1 draws with a decoded GREATER/GREATER_EQUAL depth test: on a view last cleared to 1.0 0, to 0.0 0, "
               "to another value 0, never cleared 1"), "a view never cleared");
    census.executed(&a, 1.0f, x2w_depth_census::clear_depth_flag);
    census.depth_draw(&a, x2w_depth_census::func_greater_equal);
    expect(has("2 draws with a decoded GREATER/GREATER_EQUAL depth test: on a view last cleared to 1.0 1, to 0.0 0, "
               "to another value 0, never cleared 1"), "a view cleared to 1.0");
    census.executed(&b, 0.0f, 2);  // a stencil-only clear leaves the depth as it was
    census.depth_draw(&b, x2w_depth_census::func_greater);
    census.executed(&b, 0.0f, x2w_depth_census::clear_depth_flag);
    census.depth_draw(&b, x2w_depth_census::func_greater);
    expect(has("4 draws with a decoded GREATER/GREATER_EQUAL depth test: on a view last cleared to 1.0 1, to 0.0 1, "
               "to another value 0, never cleared 2"), "stencil-only clear is not a depth clear; a view cleared to 0.0");
    census.depth_draw(&a, 2);  // LESS: not counted
    for (int i = 0; i < 4; ++i) census.depth_draw(&a, x2w_depth_census::func_greater);
    expect(has("8 draws with a decoded GREATER/GREATER_EQUAL depth test: on a view last cleared to 1.0 5, to 0.0 1, "
               "to another value 0, never cleared 2"), "only GREATER and GREATER_EQUAL tests count");
}

// Draws that write depth, by the state that decides whether they leave pixels and the view's last clear, with whether a
// fragment can pass the test against that clear: a LESS test on a view cleared to 0.0 (reversed Z) passes nowhere.
void check_depth_writers()
{
    lines.clear();
    x2w_depth_census::DepthClearCensus census(sink);
    const int scene = 0, shadow = 1;  // two depth resources
    x2w_depth_census::DepthWrite less;
    less.width = 2560, less.height = 1440, less.format = 40, less.func = 2;
    less.viewport_set = true, less.vp_width = 2560, less.vp_height = 1440;
    less.scissor_set = true, less.sc_right = 2560, less.sc_bottom = 1440;
    less.cull = 3;
    census.depth_write_draw(&scene, less);
    expect(has(" 2560x1440 format 40, func LESS, viewport 0,0 2560x1440 depth 0..1, scissor 0,0..2560,1440, cull back (front cw), "
               "depth clip 1, view last cleared to never (no depth clear seen), a fragment can pass the test against the clear: "
               "unknown: 1 draws (state 1 of the first 32)"),
           "a depth-writing draw before any clear");
    census.executed(&scene, 0.0f, x2w_depth_census::clear_depth_flag);
    for (int i = 0; i < 4; ++i) census.depth_write_draw(&scene, less);
    expect(has("func LESS, viewport 0,0 2560x1440 depth 0..1, scissor 0,0..2560,1440, cull back (front cw), depth clip 1, view "
               "last cleared to 0.0, a fragment can pass the test against the clear: no: 4 draws (state 2 of the first 32)"),
           "LESS on a view cleared to 0.0 passes nowhere, and is its own state");
    expect(!has("no: 3 draws"), "a state's line comes at powers of two only");
    x2w_depth_census::DepthWrite greater = less;
    greater.func = 5, greater.front_ccw = true, greater.depth_clip = false;
    census.depth_write_draw(&scene, greater);
    expect(has("func GREATER, viewport 0,0 2560x1440 depth 0..1, scissor 0,0..2560,1440, cull back (front ccw), depth clip 0, view "
               "last cleared to 0.0, a fragment can pass the test against the clear: yes: 1 draws (state 3 of the first 32)"),
           "GREATER on a view cleared to 0.0 can pass");
    x2w_depth_census::DepthWrite far = less;
    far.func = 7, far.min_depth = 1;
    census.executed(&shadow, 1.0f, x2w_depth_census::clear_depth_flag);
    census.depth_write_draw(&shadow, far);
    expect(has("func GREATER_EQUAL, viewport 0,0 2560x1440 depth 1..1, scissor 0,0..2560,1440, cull back (front cw), depth clip 1, "
               "view last cleared to 1.0, a fragment can pass the test against the clear: only at depth 1: 1 draws (state 4 of"),
           "another depth resource is its own state");
    x2w_depth_census::DepthWrite unset;
    unset.func = 4;
    census.depth_write_draw(&scene, unset);
    expect(has(" 0x0 format 0, func LESS_EQUAL, viewport none set (the full target, depth 0..1), scissor none set (the full target), "
               "cull none (front cw), depth clip 1, view last cleared to 0.0, a fragment can pass the test against the clear: only "
               "at depth 0: 1 draws (state 5 of the first 32)"),
           "a draw with no viewport or scissor set");
    for (int i = 0; i < 28; ++i) {
        x2w_depth_census::DepthWrite range = less;
        range.max_depth = static_cast<float>(i) / 64.0f;
        census.depth_write_draw(&scene, range);
    }
    expect(has("1 draws (state 32 of the first 32)"), "32 states are counted alone");
    expect(has("x2w-render depth writer census: depth-writing draws with a state after the first 32: 1"),
           "the 33rd state goes to the rest");
}

}  // namespace

int main()
{
    check_values();
    check_greater_draws();
    check_depth_writers();
    if (failures) {
        for (const auto& line : lines) std::fprintf(stderr, "  %s\n", line.c_str());
        return 1;
    }
    std::printf("depth clear census: values, skips and GREATER tests by last clear\n");
    return 0;
}
