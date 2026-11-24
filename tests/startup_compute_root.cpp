// Checks src/startup_stubs/list_compute_root.h without a device: the compute root signature and descriptor
// tables a command list records for a later Dispatch.
#include "startup_stubs/list_compute_root.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
using x2w_startup::ComputeRootState;
int failures;

void check(bool ok, const char* name)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

}  // namespace

int main()
{
    int root_a = 0, root_b = 0;

    {
        ComputeRootState s;
        check(!s.ready(), "a fresh list has no compute root");
        check(s.set_signature(true, &root_a) == ComputeRootState::Result::recorded && s.ready() && s.signature() == &root_a, "signature set");
        check(s.set_table(true, 2, 0x1000) == ComputeRootState::Result::recorded && s.table(2) == 0x1000, "table set");
        check(s.table(3) == 0, "an unset parameter has no table");
    }
    {
        ComputeRootState s;
        s.set_signature(true, &root_a);
        s.set_table(true, 0, 0x2000);
        s.set_table(true, 5, 0x3000);
        s.set_signature(true, &root_a);
        check(s.table(0) == 0x2000 && s.table(5) == 0x3000, "the same signature keeps the tables");
    }
    {
        ComputeRootState s;
        s.set_signature(true, &root_a);
        s.set_table(true, 0, 0x2000);
        s.set_table(true, 5, 0x3000);
        s.set_signature(true, &root_b);
        check(s.signature() == &root_b && s.table(0) == 0 && s.table(5) == 0, "a different signature leaves no table bound");
        s.set_signature(true, nullptr);
        check(!s.ready(), "a null signature clears the compute root");
    }
    {
        ComputeRootState s;
        s.set_signature(true, &root_a);
        s.set_table(true, 1, 0x4000);
        const auto at_limit = s.set_table(true, ComputeRootState::max_parameters - 1, 0x5000);
        const auto past = s.set_table(true, ComputeRootState::max_parameters, 0x6000);
        check(at_limit == ComputeRootState::Result::recorded && s.table(ComputeRootState::max_parameters - 1) == 0x5000, "the last parameter is kept");
        check(past == ComputeRootState::Result::out_of_range && s.out_of_range_count() == 1 && s.table(ComputeRootState::max_parameters) == 0,
              "a parameter out of range is refused and counted");
        s.set_table(true, 1000, 1);
        check(s.out_of_range_count() == 2 && s.table(1) == 0x4000, "a refused call leaves the other tables alone");
    }
    {
        ComputeRootState s;
        const auto closed_sig = s.set_signature(false, &root_a);
        check(closed_sig == ComputeRootState::Result::ignored_closed && !s.ready(), "a closed list ignores the signature");
        s.set_signature(true, &root_a);
        const auto closed_table = s.set_table(false, 0, 0x7000);
        check(closed_table == ComputeRootState::Result::ignored_closed && s.table(0) == 0, "a closed list ignores the table");
        check(s.closed_ignored_count() == 2, "each ignored signature and table call is counted");
    }
    std::printf("startup_compute_root: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
