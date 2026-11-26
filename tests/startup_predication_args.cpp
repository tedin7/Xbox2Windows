#include "../src/startup_stubs/predication_args.h"

#include <cstdio>
#include <cstdlib>

using x2w_predication::check;
using x2w_predication::Verdict;

static void expect(bool ok, const char* what)
{
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

int main()
{
    expect(check(false, 3, 9, 0) == Verdict::clear, "no buffer clears predication whatever the other arguments are");
    expect(check(true, 0, 0, 8) == Verdict::set, "EQUAL_ZERO on the last 8 bytes is accepted");
    expect(check(true, 8, 1, 16) == Verdict::set, "NOT_EQUAL_ZERO at an aligned offset is accepted");
    expect(check(true, 0, 2, 64) == Verdict::bad_operation, "an operation above 1 is refused");
    expect(check(true, 4, 0, 64) == Verdict::bad_offset, "an offset that is not a multiple of 8 is refused");
    expect(check(true, 8, 0, 8) == Verdict::outside_buffer, "a value past the end is refused");
    expect(check(true, 16, 0, 8) == Verdict::outside_buffer, "an offset past the end cannot underflow");
    expect(check(true, 0, 0, 4) == Verdict::outside_buffer, "a buffer smaller than the value is refused");
    expect(check(true, ~std::uint64_t{15}, 0, ~std::uint64_t{0}) == Verdict::set, "a huge aligned offset does not overflow");
    std::puts("startup_predication_args: ok");
    return 0;
}
