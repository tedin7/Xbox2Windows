// Checks acceleration-structure memory SRV validation and address resolution (device slot 70)
// without a device.
// --mutate alignment: expect an unaligned address to be accepted.
// --mutate state: expect a buffer that is not in the acceleration-structure state to be accepted.
// --mutate offset: expect the desktop address to be the buffer base, ignoring the offset.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/memory_srv_as.h"

#include <cstdio>
#include <cstring>

namespace {

using namespace x2w_startup;

int failures;
const char* mutation = "";

bool mutated(const char* name)
{
    return std::strcmp(mutation, name) == 0;
}

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

constexpr std::uint64_t title_base = 0x700000000ull;
constexpr std::uint64_t gpu_base = 0x5000000000ull;

D3D12_SHADER_RESOURCE_VIEW_DESC as_view(std::uint64_t location)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_UNKNOWN;
    v.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.RaytracingAccelerationStructure.Location = location;
    return v;
}

PlacedAsBuffer placed()
{
    PlacedAsBuffer p;
    p.found = true;
    p.title_base = title_base;
    p.gpu_base = gpu_base;
    p.bytes = 0x10000;
    p.initial_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    return p;
}

AsSrvRefusal resolve(std::uint64_t location, const PlacedAsBuffer& p, D3D12_SHADER_RESOURCE_VIEW_DESC& out)
{
    return resolve_as_srv_desc(location, p, as_view(location), out);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) mutation = argv[2];
    D3D12_SHADER_RESOURCE_VIEW_DESC out{};

    // The structure starts 0x300 bytes into the buffer: the view carries the desktop address of that byte.
    const std::uint64_t location = title_base + 0x300;
    check("accepted", resolve(location, placed(), out) == AsSrvRefusal::none);
    const std::uint64_t expected = mutated("offset") ? gpu_base : gpu_base + 0x300;
    check("desktop address", out.RaytracingAccelerationStructure.Location == expected);
    check("other fields kept", out.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE &&
                                   out.Shader4ComponentMapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);

    // The first and the last aligned byte of the buffer are inside it; one past the end is not.
    check("first byte", resolve(title_base, placed(), out) == AsSrvRefusal::none);
    check("last aligned byte", resolve(title_base + 0xff00, placed(), out) == AsSrvRefusal::none);
    check("one past the end", resolve(title_base + 0x10000, placed(), out) == AsSrvRefusal::as_not_placed);

    // Each refusal has its own reason and leaves the output alone.
    D3D12_SHADER_RESOURCE_VIEW_DESC untouched{};
    untouched.RaytracingAccelerationStructure.Location = 0x1234;
    D3D12_SHADER_RESOURCE_VIEW_DESC probe = untouched;
    const AsSrvRefusal unaligned = resolve(title_base + 0x301, placed(), probe);
    check("unaligned", mutated("alignment") ? unaligned == AsSrvRefusal::none : unaligned == AsSrvRefusal::as_unaligned);

    PlacedAsBuffer none;
    probe = untouched;
    check("not placed", resolve(location, none, probe) == AsSrvRefusal::as_not_placed);
    PlacedAsBuffer ambiguous = placed();
    ambiguous.ambiguous = true;
    check("ambiguous", resolve(location, ambiguous, probe) == AsSrvRefusal::as_not_placed);
    check("below the buffer", resolve(title_base - 0x100, placed(), probe) == AsSrvRefusal::as_not_placed);
    PlacedAsBuffer overflow = placed();
    overflow.gpu_base = UINT64_MAX - 0xff;
    check("desktop address overflow", resolve(location, overflow, probe) == AsSrvRefusal::as_not_placed);
    PlacedAsBuffer misaligned_gpu = placed();
    misaligned_gpu.gpu_base = gpu_base + 0x80;
    check("desktop address unaligned", resolve(location, misaligned_gpu, probe) == AsSrvRefusal::as_unaligned);

    PlacedAsBuffer wrong = placed();
    wrong.initial_state = D3D12_RESOURCE_STATE_COMMON;
    probe = untouched;
    check("wrong state", resolve(location, wrong, probe) == (mutated("state") ? AsSrvRefusal::none : AsSrvRefusal::as_wrong_state));
    if (!mutated("state")) check("refusal leaves output", probe.RaytracingAccelerationStructure.Location == 0x1234);

    // The address comes from Location, or from the call's address when Location is zero.
    check("location wins", as_srv_location(as_view(0x2000), 0x3000) == 0x2000);
    check("address fallback", as_srv_location(as_view(0), 0x3000) == 0x3000);
    check("both zero", as_srv_location(as_view(0), 0) == 0);
    check("dimension", is_as_srv(as_view(0)) && !is_as_srv(D3D12_SHADER_RESOURCE_VIEW_DESC{}));

    // Format must be UNKNOWN and Shader4ComponentMapping must be default.
    check("supported desc", supports_as_memory_srv(as_view(location)));
    auto bad_fmt = as_view(location);
    bad_fmt.Format = DXGI_FORMAT_R32_UINT;
    check("bad format refused", !supports_as_memory_srv(bad_fmt));
    auto bad_map = as_view(location);
    bad_map.Shader4ComponentMapping = 0;
    check("bad mapping refused", !supports_as_memory_srv(bad_map));
    check("non-as refused", !supports_as_memory_srv(D3D12_SHADER_RESOURCE_VIEW_DESC{}));

    // Trailing union bytes after Location are cleared in the resolved descriptor.
    auto dirty_tail = as_view(location);
    std::uint32_t words[6]{};
    std::memcpy(words, &dirty_tail.Buffer, sizeof(words));
    words[2] = 0xdeadbeefu;
    std::memcpy(&dirty_tail.Buffer, words, sizeof(words));
    D3D12_SHADER_RESOURCE_VIEW_DESC clean{};
    check("dirty tail accepted", resolve_as_srv_desc(location, placed(), dirty_tail, clean) == AsSrvRefusal::none);
    std::memcpy(words, &clean.Buffer, sizeof(words));
    check("tail cleared", words[2] == 0 && words[3] == 0 && words[4] == 0 && words[5] == 0);

    check("names", std::strcmp(as_srv_refusal_name(AsSrvRefusal::as_unaligned), "as_unaligned") == 0 &&
                       std::strcmp(as_srv_refusal_name(AsSrvRefusal::as_not_placed), "as_not_placed") == 0 &&
                       std::strcmp(as_srv_refusal_name(AsSrvRefusal::as_wrong_state), "as_wrong_state") == 0);

    if (failures) return 1;
    std::printf("startup_memory_srv_as_check: ok\n");
    return 0;
}
