// Checks the validation of the whole-resource command-list copies (CopyBufferRegion, CopyResource,
// ResolveSubresource; src/startup_stubs/list_copy.h) without a device.
// --mutate overflow: expect an offset and size that wrap to be accepted.
// --mutate typeless: expect a typeless half-float resolve format to stay typeless instead of resolving as float.
// --mutate samples: expect CopyResource between textures of different sample counts to be accepted.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/list_copy.h"

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

D3D12_RESOURCE_DESC buffer(UINT64 width)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = width;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture(UINT64 width, UINT height, DXGI_FORMAT format, UINT samples = 1, UINT mips = 1, UINT array = 1)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = static_cast<UINT16>(array);
    d.MipLevels = static_cast<UINT16>(mips);
    d.Format = format;
    d.SampleDesc.Count = samples;
    return d;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) mutation = argv[2];

    // CopyBufferRegion: ranges inside both buffers, and the first refusal in the documented order.
    const auto a = buffer(256), b = buffer(128), tex = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    check("buffer copy", check_copy_buffer_region(&a, 16, &b, 32, 64, false) == CopyRefusal::none);
    check("buffer copy to the end", check_copy_buffer_region(&a, 192, &b, 64, 64, false) == CopyRefusal::none);
    check("buffer null", check_copy_buffer_region(nullptr, 0, &b, 0, 4, false) == CopyRefusal::unresolved);
    check("buffer same", check_copy_buffer_region(&a, 0, &a, 64, 4, true) == CopyRefusal::same_resource);
    check("buffer texture", check_copy_buffer_region(&a, 0, &tex, 0, 4, false) == CopyRefusal::not_buffer);
    check("buffer empty", check_copy_buffer_region(&a, 0, &b, 0, 0, false) == CopyRefusal::zero_size);
    check("buffer past the source", check_copy_buffer_region(&a, 0, &b, 100, 64, false) == CopyRefusal::range_overflow);
    check("buffer past the destination", check_copy_buffer_region(&b, 100, &a, 0, 64, false) == CopyRefusal::range_overflow);
    const CopyRefusal wrapped = check_copy_buffer_region(&a, UINT64_MAX - 1, &b, 0, 8, false);
    check("buffer wrapping offset", mutated("overflow") ? wrapped == CopyRefusal::none : wrapped == CopyRefusal::range_overflow);

    // CopyResource: one shape.
    const auto rgba = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    const auto tmp0 = texture(4, 4, DXGI_FORMAT_R32_FLOAT);
    const auto tmp1 = texture(8, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    const auto tmp2 = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 2);
    const auto tmp3 = texture(8, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    check("resource copy", check_copy_resource(&rgba, &rgba, false) == CopyRefusal::none);
    check("resource same", check_copy_resource(&rgba, &rgba, true) == CopyRefusal::same_resource);
    check("resource null", check_copy_resource(&rgba, nullptr, false) == CopyRefusal::unresolved);
    const auto typeless = texture(4, 4, DXGI_FORMAT_R8G8B8A8_TYPELESS);
    check("resource typeless to typed", check_copy_resource(&rgba, &typeless, false) == CopyRefusal::none);
    check("resource other format", check_copy_resource(&rgba, &tmp0, false) ==
                                       CopyRefusal::format_mismatch);
    check("resource other size", check_copy_resource(&rgba, &tmp1, false) ==
                                     CopyRefusal::size_mismatch);
    check("resource other mips", check_copy_resource(&rgba, &tmp2, false) ==
                                     CopyRefusal::size_mismatch);
    check("resource buffer and texture", check_copy_resource(&a, &rgba, false) == CopyRefusal::dimension_mismatch);
    check("resource other buffer size", check_copy_resource(&a, &b, false) == CopyRefusal::size_mismatch);
    check("resource buffers", check_copy_resource(&a, &a, false) == CopyRefusal::none);
    const auto msaa = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    const CopyRefusal samples = check_copy_resource(&rgba, &msaa, false);
    check("resource other samples", mutated("samples") ? samples == CopyRefusal::none : samples == CopyRefusal::sample_mismatch);

    // ResolveSubresource.
    check("resolve", plan_resolve(&rgba, 0, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal == CopyRefusal::none);
    check("resolve format", plan_resolve(&rgba, 0, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).format == DXGI_FORMAT_R8G8B8A8_UNORM);
    check("resolve null", plan_resolve(nullptr, 0, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal == CopyRefusal::unresolved);
    check("resolve same", plan_resolve(&msaa, 0, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, true).refusal == CopyRefusal::same_resource);
    check("resolve from one sample", plan_resolve(&rgba, 0, &rgba, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal ==
                                         CopyRefusal::source_not_multisampled);
    check("resolve into samples", plan_resolve(&msaa, 0, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal ==
                                      CopyRefusal::destination_multisampled);
    check("resolve a buffer", plan_resolve(&a, 0, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal == CopyRefusal::not_texture);
    check("resolve other size", plan_resolve(&tmp3, 0, &msaa, 0,
                                             DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal == CopyRefusal::bad_subresource);
    check("resolve subresource out of range", plan_resolve(&rgba, 1, &msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal ==
                                                  CopyRefusal::bad_subresource);
    check("resolve other format", plan_resolve(&rgba, 0, &msaa, 0, DXGI_FORMAT_R32_FLOAT, false).refusal == CopyRefusal::format_mismatch);
    const auto hdr_source = texture(4, 4, DXGI_FORMAT_R16G16B16A16_TYPELESS, 4);
    const auto hdr_target = texture(4, 4, DXGI_FORMAT_R16G16B16A16_TYPELESS);
    const ResolvePlan hdr = plan_resolve(&hdr_target, 0, &hdr_source, 0, DXGI_FORMAT_R16G16B16A16_TYPELESS, false);
    check("resolve typeless half float", hdr.refusal == CopyRefusal::none);
    check("resolve typeless as float", hdr.format == (mutated("typeless") ? DXGI_FORMAT_R16G16B16A16_TYPELESS : DXGI_FORMAT_R16G16B16A16_FLOAT));
    check("resolve unknown over typeless", plan_resolve(&hdr_target, 0, &hdr_source, 0, DXGI_FORMAT_UNKNOWN, false).format ==
                                               DXGI_FORMAT_R16G16B16A16_FLOAT);
    const auto small_source = texture(4, 4, DXGI_FORMAT_R16_TYPELESS, 4);
    const auto small_target = texture(4, 4, DXGI_FORMAT_R16_TYPELESS);
    check("resolve typeless without a choice", plan_resolve(&small_target, 0, &small_source, 0, DXGI_FORMAT_R16_TYPELESS, false).refusal ==
                                                   CopyRefusal::bad_format);
    const auto mips_target = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 3);
    const auto mip_source = texture(2, 2, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    check("resolve a smaller mip", plan_resolve(&mips_target, 1, &mip_source, 0, DXGI_FORMAT_R8G8B8A8_UNORM, false).refusal == CopyRefusal::none);

    // The resolve format follows a fully typed endpoint exactly; a typeless endpoint takes any member of its family.
    const auto srgb_target = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    const auto typeless_msaa = texture(4, 4, DXGI_FORMAT_R8G8B8A8_TYPELESS, 4);
    const ResolvePlan typed_target = plan_resolve(&srgb_target, 0, &typeless_msaa, 0, DXGI_FORMAT_UNKNOWN, false);
    check("resolve unknown over a typed destination", typed_target.refusal == CopyRefusal::none &&
                                                          typed_target.format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    check("resolve format against a typed destination", plan_resolve(&srgb_target, 0, &typeless_msaa, 0, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                                     false).refusal == CopyRefusal::format_mismatch);
    const auto srgb_msaa = texture(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4);
    check("resolve typed endpoints that differ", plan_resolve(&rgba, 0, &srgb_msaa, 0, DXGI_FORMAT_UNKNOWN, false).refusal ==
                                                     CopyRefusal::format_mismatch);

    // Families beyond the colour formats the resolve choice covers.
    const auto bc7_typeless = texture(8, 8, DXGI_FORMAT_BC7_TYPELESS), bc7 = texture(8, 8, DXGI_FORMAT_BC7_UNORM),
               bc1 = texture(8, 8, DXGI_FORMAT_BC1_UNORM), r16_typeless = texture(4, 4, DXGI_FORMAT_R16_TYPELESS),
               r16 = texture(4, 4, DXGI_FORMAT_R16_UNORM), r8 = texture(4, 4, DXGI_FORMAT_R8_UNORM),
               r8_typeless = texture(4, 4, DXGI_FORMAT_R8_TYPELESS);
    check("copy BC7 typeless to typed", check_copy_resource(&bc7, &bc7_typeless, false) == CopyRefusal::none);
    check("copy BC7 to BC1", check_copy_resource(&bc7, &bc1, false) == CopyRefusal::format_mismatch);
    check("copy R16 typeless to typed", check_copy_resource(&r16, &r16_typeless, false) == CopyRefusal::none);
    check("copy R8 typeless to typed", check_copy_resource(&r8, &r8_typeless, false) == CopyRefusal::none);
    check("copy R8 to R16", check_copy_resource(&r8, &r16, false) == CopyRefusal::format_mismatch);

    check("names", std::strcmp(copy_refusal_name(CopyRefusal::heap_direction), "heap_direction") == 0 &&
                       std::strcmp(copy_refusal_name(CopyRefusal::initial_state), "initial_state") == 0 &&
                       std::strcmp(copy_refusal_name(CopyRefusal::title_backed_source), "title_backed_source") == 0 &&
                       std::strcmp(copy_refusal_name(CopyRefusal::range_overflow), "range_overflow") == 0 &&
                       std::strcmp(copy_refusal_name(CopyRefusal::none), "none") == 0);

    if (failures) return 1;
    std::printf("startup_list_copy_check: ok\n");
    return 0;
}
