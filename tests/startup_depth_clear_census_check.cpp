// depth_clear_census.h: the depth values a title clears to, the clears the stand-in skipped, and what a decoded
// GREATER/GREATER_EQUAL depth test meets (a view last cleared to 1.0, to 0.0, or never), each printed when its count
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

}  // namespace

// The draw path keeps the desktop depth state for a GREATER test on a view last cleared to 1.0; it asks last_clear.
void check_last_clear()
{
    lines.clear();
    x2w_depth_census::DepthClearCensus census(sink);
    const int a = 0, b = 0, c = 0;
    expect(census.last_clear(&a) == x2w_depth_census::LastClear::never, "a view with no clear is never cleared");
    expect(census.last_clear(nullptr) == x2w_depth_census::LastClear::never, "no view is never cleared");
    census.executed(&a, 1.0f, x2w_depth_census::clear_depth_flag);
    census.executed(&b, 0.0f, x2w_depth_census::clear_depth_flag);
    census.executed(&c, 0.5f, x2w_depth_census::clear_depth_flag);
    expect(census.last_clear(&a) == x2w_depth_census::LastClear::far_one, "cleared to 1.0");
    expect(census.last_clear(&b) == x2w_depth_census::LastClear::near_zero, "cleared to 0.0");
    expect(census.last_clear(&c) == x2w_depth_census::LastClear::other, "cleared to another value");
    census.executed(&a, 0.0f, x2w_depth_census::clear_depth_flag);
    expect(census.last_clear(&a) == x2w_depth_census::LastClear::near_zero, "a later clear replaces the value");
}

int main()
{
    check_values();
    check_greater_draws();
    check_last_clear();
    if (failures) {
        for (const auto& line : lines) std::fprintf(stderr, "  %s\n", line.c_str());
        return 1;
    }
    std::printf("depth clear census: values, skips and GREATER tests by last clear\n");
    return 0;
}
