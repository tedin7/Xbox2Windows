// Unit check for src/startup_stubs/pixel_uav_support.h: a pixel image store needs typed UAV store, and a partial-dmask
// store (manifest 'uavload N') also typed UAV load. Run with --mutate no-load-check to see the control fail.
#include "startup_stubs/pixel_uav_support.h"

#include <cstdio>
#include <cstring>

namespace {
int failures = 0;

void expect(const char* label, const char* got, const char* want)
{
    const bool same = (got == nullptr || want == nullptr) ? got == want : std::strcmp(got, want) == 0;
    if (!same) {
        std::printf("FAIL %s: got '%s', expected '%s'\n", label, got ? got : "(accepted)", want ? want : "(accepted)");
        ++failures;
    }
}

// The control: the same decision with the load requirement removed.
const char* without_load_check(unsigned support2, bool needs_load)
{
    return x2w_startup::pixel_uav_support_reason(support2, false && needs_load);
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc > 2 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "no-load-check") == 0;
    const auto reason = [&](unsigned bits, bool load) {
        return mutate ? without_load_check(bits, load) : x2w_startup::pixel_uav_support_reason(bits, load);
    };
    using namespace x2w_startup;
    const char* no_store = "format has no typed UAV store on this device";
    const char* no_load = "format has no typed UAV load on this device";
    expect("store only, plain store", reason(kSupport2UavTypedStore, false), nullptr);
    expect("store only, read-back needed", reason(kSupport2UavTypedStore, true), no_load);
    expect("store and load, read-back needed", reason(kSupport2UavTypedStore | kSupport2UavTypedLoad, true), nullptr);
    expect("load only, plain store", reason(kSupport2UavTypedLoad, false), no_store);
    expect("no bits", reason(0, true), no_store);
    expect("other bits only", reason(0x3f, true), no_store);
    if (failures) return 1;
    std::printf("pixel_uav_support: 6 cases\n");
    return 0;
}
