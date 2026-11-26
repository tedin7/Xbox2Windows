// Device-free rules of the system-memory fallback (src/startup_stubs/sysmem_policy.h).
#include "startup_stubs/sysmem_policy.h"

#include <cstdio>

namespace {
int failures;

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
    using namespace x2w_sysmem;
    constexpr std::int32_t invalid_arg = static_cast<std::int32_t>(0x80070057);
    constexpr std::int32_t device_removed = static_cast<std::int32_t>(0x887a0005);

    // The retry needs all three: the switch, a default heap and E_OUTOFMEMORY.
    expect(wants_fallback(out_of_memory, true, true), "out of memory on a default heap with the switch on retries");
    expect(!wants_fallback(out_of_memory, false, true), "the switch off never retries");
    expect(!wants_fallback(out_of_memory, true, false), "a heap that is not the default one never retries");
    expect(!wants_fallback(invalid_arg, true, true), "an invalid-argument failure is not retried");
    expect(!wants_fallback(device_removed, true, true), "a device-removed failure is not retried");
    expect(!wants_fallback(0, true, true), "a success is not retried");

    // A shared-memory adapter may use the GPU-only page type first; a discrete one never gets it (it belongs to the
    // video-memory pool there) and needs a CPU-visible page type.
    const Candidates shared = candidate_pages(true);
    expect(shared.count == 3 && shared.pages[0] == not_available && shared.pages[1] == write_combine &&
               shared.pages[2] == write_back,
           "a shared-memory adapter tries not-available, write-combine, then write-back");
    const Candidates discrete = candidate_pages(false);
    expect(discrete.count == 2 && discrete.pages[0] == write_combine && discrete.pages[1] == write_back,
           "a discrete adapter tries write-combine, then write-back");
    for (int i = 0; i < discrete.count; ++i) expect(discrete.pages[i] != not_available, "a discrete adapter never gets not-available");
    // The values are D3D12_CPU_PAGE_PROPERTY's.
    expect(not_available == 1 && write_combine == 2 && write_back == 3, "the page property values match the D3D12 enum");

    if (failures) return 1;
    std::puts("sysmem policy checks passed");
    return 0;
}
