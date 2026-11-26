// Checks the SOSetTargets rules (src/startup_stubs/stream_output_targets.h), no Windows types.
#include "startup_stubs/stream_output_targets.h"

#include <cstdio>

using namespace x2w_startup;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);                  \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

int main()
{
    const StreamOutputView a{0x1000, 256, 0x2000}, b{0x3000, 512, 0x4000};
    const StreamOutputView two[2] = {a, b};

    {  // views replace the slots from `start`
        StreamOutputTargets t;
        CHECK(set_stream_output_targets(t, 1, 2, two) == StreamOutputResult::applied);
        CHECK(!t.views[0].location && t.views[1].location == 0x1000 && t.views[2].size == 512);
        CHECK(t.views[2].filled_size_location == 0x4000 && t.bound_count() == 2);
    }
    {  // no view array unbinds only the named slots
        StreamOutputTargets t;
        set_stream_output_targets(t, 0, 2, two);
        CHECK(set_stream_output_targets(t, 1, 1, nullptr) == StreamOutputResult::unbound);
        CHECK(t.views[0].location == 0x1000 && !t.views[1].location && t.bound_count() == 1);
    }
    {  // a call past the fourth slot is refused and changes nothing
        StreamOutputTargets t;
        set_stream_output_targets(t, 0, 2, two);
        CHECK(set_stream_output_targets(t, 3, 2, two) == StreamOutputResult::refused);
        CHECK(set_stream_output_targets(t, 4, 0, nullptr) == StreamOutputResult::refused);
        CHECK(set_stream_output_targets(t, 0, 5, nullptr) == StreamOutputResult::refused);
        CHECK(t.views[0].location == 0x1000 && t.views[1].location == 0x3000 && !t.views[3].location);
    }
    {  // the last slot can be bound by itself
        StreamOutputTargets t;
        CHECK(set_stream_output_targets(t, 3, 1, &b) == StreamOutputResult::applied);
        CHECK(t.views[3].location == 0x3000 && t.bound_count() == 1);
    }
    if (!failures) std::printf("startup_stream_output_targets: ok\n");
    return failures ? 1 : 0;
}
