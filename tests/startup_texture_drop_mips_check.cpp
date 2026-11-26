// Checks src/startup_stubs/texture_drop_mips.h without a device: when a texture may lose its top mip levels, the
// description that is created, the view that follows it, the manifest words and the counters' names.
//
//   startup_texture_drop_mips_check                     the rules
//   startup_texture_drop_mips_check --mutate <name>     the same expectations with one rule broken (must fail):
//       texel-allowed      a texel-addressed shader may lose levels
//       unknown-allowed    a draw with no hint may lose levels
//       view-start         a view that starts below the top level may lose levels
//       small-allowed      a texture under the byte threshold may lose levels
//       block-tail         a block-compressed level under one block may be dropped to
//       rt-allowed         a render target may lose levels
//       keep-view-mips     the view keeps the title's level count
#include "startup_stubs/texture_drop_mips.h"

#include <cstdio>
#include <cstring>

namespace dm = x2w_startup::drop_mips;

namespace {
int failures;
const char* mutation = "";
bool is(const char* name) { return std::strcmp(mutation, name) == 0; }

void check(const char* what, bool ok)
{
    std::printf("%s: %s\n", what, ok ? "ok" : "MISMATCH");
    if (!ok) ++failures;
}

D3D12_RESOURCE_DESC texture(UINT width, UINT height, UINT mips, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
                            D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = static_cast<UINT16>(mips);
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

// The rule under test, with the mutations applied on top of the real rule().
dm::Decision rule(const D3D12_RESOURCE_DESC& desc, const dm::Config& config, dm::Use use, const dm::ViewFacts& view)
{
    D3D12_RESOURCE_DESC effective = desc;
    dm::Use effective_use = use;
    dm::ViewFacts effective_view = view;
    dm::Config effective_config = config;
    if (is("texel-allowed") && use == dm::Use::texel) effective_use = dm::Use::sample;
    if (is("unknown-allowed") && use == dm::Use::unknown) effective_use = dm::Use::sample;
    if (is("view-start")) effective_view = {};
    if (is("small-allowed")) effective_config.min_bytes = 1;
    if (is("rt-allowed")) effective.Flags = D3D12_RESOURCE_FLAG_NONE;
    dm::Decision result = dm::decide(effective, effective_config, effective_use, effective_view);
    if (is("block-tail") && result.reason == dm::Reason::block_tail) result = {config.levels, dm::Reason::none};
    return result;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && !std::strcmp(argv[1], "--mutate")) mutation = argv[2];
    const dm::Config one{1, 4u << 20};
    const dm::ViewFacts top;

    // 2048x2048 RGBA8 with a full chain: about 21 MiB, 12 levels.
    const auto big = texture(2048, 2048, 12);
    const auto allowed = rule(big, one, dm::Use::sample, top);
    check("a big sampled 2D texture loses one level", allowed.levels == 1 && allowed.reason == dm::Reason::none);
    const auto reduced = dm::reduced_desc(big, allowed.levels ? allowed.levels : 1);
    check("the created texture is half size with one level less", reduced.Width == 1024 && reduced.Height == 1024 && reduced.MipLevels == 11);
    check("dropping two levels is a quarter", dm::reduced_desc(big, 2).Width == 512 && dm::reduced_desc(big, 2).MipLevels == 10);
    check("a BC7 4096x2048 chain is about 11 MiB, RGBA8 of the same size 43 MiB",
          dm::chain_estimate(texture(4096, 2048, 13, DXGI_FORMAT_BC7_UNORM)) > (10u << 20) &&
              dm::chain_estimate(texture(4096, 2048, 13, DXGI_FORMAT_BC7_UNORM)) < (12u << 20) &&
              dm::chain_estimate(texture(4096, 2048, 13)) > (42u << 20));
    check("the title's chain would have been about 21 MiB, one level saves about 15 MiB",
          dm::chain_estimate(big) > (20u << 20) && dm::saved_estimate(dm::chain_estimate(big), 1) > (15u << 20));

    check("off: no levels", dm::decide(big, {}, dm::Use::sample, top).reason == dm::Reason::off);
    check("no hint: nothing is dropped",
          rule(big, one, dm::Use::unknown, top).levels == 0 && rule(big, one, dm::Use::unknown, top).reason == dm::Reason::unknown_use);
    check("texel addressing keeps every level",
          rule(big, one, dm::Use::texel, top).levels == 0 && rule(big, one, dm::Use::texel, top).reason == dm::Reason::texel);
    check("GetDimensions keeps every level", rule(big, one, dm::Use::size, top).reason == dm::Reason::size);
    check("GetLOD keeps every level", rule(big, one, dm::Use::lod, top).reason == dm::Reason::lod);
    check("a small texture keeps every level",
          rule(texture(512, 512, 10), one, dm::Use::sample, top).reason == dm::Reason::below_min);
    check("a view that starts below the top level keeps every level",
          rule(big, one, dm::Use::sample, {2, 0.0f}).levels == 0 && rule(big, one, dm::Use::sample, {2, 0.0f}).reason == dm::Reason::view_clamp);
    check("a view with a minimum LOD clamp keeps every level", rule(big, one, dm::Use::sample, {0, 1.5f}).reason == dm::Reason::view_clamp);
    check("a render target keeps every level",
          rule(texture(2048, 2048, 12, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), one, dm::Use::sample, top).reason ==
              dm::Reason::flags);
    check("a texture with one level keeps it", rule(texture(2048, 2048, 1), one, dm::Use::sample, top).reason == dm::Reason::few_mips);
    auto array = big;
    array.DepthOrArraySize = 6;
    check("an array or cube keeps every level", rule(array, one, dm::Use::sample, top).reason == dm::Reason::not_2d);
    auto volume = big;
    volume.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    check("a volume keeps every level", rule(volume, one, dm::Use::sample, top).reason == dm::Reason::not_2d);
    auto multisampled = big;
    multisampled.SampleDesc.Count = 4;
    check("a multisampled texture keeps every level", rule(multisampled, one, dm::Use::sample, top).reason == dm::Reason::multisampled);
    check("planar video keeps every level", rule(texture(4096, 2048, 4, DXGI_FORMAT_NV12), one, dm::Use::sample, top).reason == dm::Reason::format);
    check("block compressed 4096x2048 may lose a level",
          rule(texture(4096, 2048, 13, DXGI_FORMAT_BC7_UNORM), one, dm::Use::sample, top).levels == 1);
    check("block compressed with a level under one block keeps every level",
          rule(texture(4096, 6, 13, DXGI_FORMAT_BC7_UNORM), {2, 1}, dm::Use::sample, top).levels == 0);

    // The view on the reduced texture.
    const unsigned view_levels = is("keep-view-mips") ? 12u : dm::view_mip_levels(12, 1);
    check("a view of all 12 levels sees the 11 that are left", view_levels == 11);
    check("a view of all levels stays all levels", dm::view_mip_levels(~0u, 1) == ~0u);
    check("a single-level view keeps one level", dm::view_mip_levels(1, 1) == 1);

    // Config.
    check("levels parse and clamp", dm::parse_config("1", nullptr).levels == 1 && dm::parse_config("9", nullptr).levels == 4 &&
                                    dm::parse_config("", nullptr).levels == 0 && dm::parse_config("x", nullptr).levels == 0);
    check("min bytes default 4 MiB, override", dm::parse_config("1", nullptr).min_bytes == (4u << 20) &&
                                               dm::parse_config("1", "65536").min_bytes == 65536);
    check("manifest words", dm::parse_use("sample") == dm::Use::sample && dm::parse_use("texel") == dm::Use::texel &&
                            dm::parse_use("size") == dm::Use::size && dm::parse_use("lod") == dm::Use::lod &&
                            dm::parse_use("samples") == dm::Use::unknown && dm::parse_use("") == dm::Use::unknown &&
                            dm::parse_use(nullptr) == dm::Use::unknown);

    // The bind hint: unknown outside a bind, restored after it, and only a sampling bind may receive a dropped texture.
    check("outside a bind nothing may receive a dropped texture", !dm::may_receive_dropped(dm::current_hint));
    {
        const dm::BindHint hint(dm::Use::sample, top);
        check("a sampling bind may", dm::may_receive_dropped(dm::current_hint));
        {
            const dm::BindHint inner(dm::Use::texel, top);
            check("a texel bind may not", !dm::may_receive_dropped(dm::current_hint));
        }
        check("the hint is restored", dm::may_receive_dropped(dm::current_hint));
    }
    check("the hint is gone after the bind", !dm::may_receive_dropped(dm::current_hint));
    check("a draw bind is told from a copy or a map", !dm::current_hint.draw);
    {
        const dm::BindHint hint(dm::Use::unknown, top);
        check("a draw with no hint is still a draw", dm::current_hint.draw && !dm::may_receive_dropped(dm::current_hint));
    }

    std::printf("%s\n", failures ? "FAIL" : "startup_texture_drop_mips: PASS");
    return failures ? 1 : 0;
}
