// Checks src/startup_stubs/depth_state_apply.h (X2W_DEPTH_STATE, #1999 item A) without a device: with the
// switch off the stand-in takes nothing from state block A, whatever the block holds, so the desktop state
// (depth and stencil off) and the pipeline key stay as they were; with it on, a decodable block is taken,
// a refused or short block is not, and the stencil reference is carried only for an enabled stencil test.
//
// --mutate switch: treats the switch as on in the "off" cases (the check must then fail).
#include "startup_stubs/depth_state_apply.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace x2w_pipeline_state;
int failures, cases;
bool mutate_switch;

void check(bool ok, const char* name)
{
    ++cases;
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

// A zeroed 0x200-byte state block A with the four depth-stencil words set.
std::vector<unsigned char> block(std::uint32_t depth, std::uint32_t stencil, std::uint32_t refmask, std::uint32_t refmask_bf)
{
    std::vector<unsigned char> a(0x200, 0);
    auto put = [&](std::size_t off, std::uint32_t v) { std::memcpy(a.data() + off, &v, 4); };
    put(a_depth, depth);
    put(a_stencil, stencil);
    put(a_refmask, refmask);
    put(a_refmask_bf, refmask_bf);
    return a;
}

// DB_DEPTH_CONTROL: Z_ENABLE [1], Z_WRITE_ENABLE [2], ZFUNC [6:4], STENCIL_ENABLE [0], STENCILFUNC [10:8] (Mesa gfx10.json).
constexpr std::uint32_t z_control(unsigned zfunc, bool write) { return 2u | (write ? 4u : 0u) | zfunc << 4 | 7u << 8 | 7u << 20; }
constexpr unsigned frag_greater = 4, frag_less = 1, frag_equal = 2;

bool desc_is_zero(const D3D12_DEPTH_STENCIL_DESC& d)
{
    static const D3D12_DEPTH_STENCIL_DESC zero{};
    return std::memcmp(&d, &zero, sizeof(d)) == 0;
}

void cases_all()
{
    const auto reversed = block(z_control(frag_greater, true), 0, 0, 0);
    // Switch off: nothing applied, nothing reported, the result object is the empty default.
    auto off = select_depth_stencil(mutate_switch, reversed.data(), reversed.size());
    check(!off.applied && !off.reason && !off.has_stencil_ref && desc_is_zero(off.desc), "switch off: reversed-Z block not taken");
    const auto refused = block(z_control(frag_greater, true) | 1u << 30, 0, 0, 0);
    off = select_depth_stencil(mutate_switch, refused.data(), refused.size());
    check(!off.applied && !off.reason, "switch off: refused block not even decoded");
    off = select_depth_stencil(mutate_switch, nullptr, 0);
    check(!off.applied && !off.reason, "switch off: no block");

    // Switch on: reversed-Z taken, function and write mask as decoded.
    auto on = select_depth_stencil(true, reversed.data(), reversed.size());
    check(on.applied && on.desc.DepthEnable && on.desc.DepthFunc == D3D12_COMPARISON_FUNC_GREATER &&
              on.desc.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL && !on.has_stencil_ref, "switch on: GREATER with write");
    // The compare direction is not symmetric: LESS must decode to a different function.
    const auto less = block(z_control(frag_less, true), 0, 0, 0);
    on = select_depth_stencil(true, less.data(), less.size());
    check(on.applied && on.desc.DepthFunc == D3D12_COMPARISON_FUNC_LESS, "switch on: LESS differs from GREATER");
    // Stencil on: the reference is carried; with the stencil test off it is not (the word's reference is then unused).
    const auto stencil = block(1u | frag_equal << 8 | 1u << 7 | frag_equal << 20, 0, 0x5a | 0xff << 8 | 0xff << 16 | 1u << 24, 0x5a | 0xff << 8 | 0xff << 16 | 1u << 24);
    on = select_depth_stencil(true, stencil.data(), stencil.size());
    check(on.applied && on.has_stencil_ref && on.stencil_ref == 0x5a && on.desc.StencilEnable &&
              on.desc.FrontFace.StencilFunc == D3D12_COMPARISON_FUNC_EQUAL, "switch on: stencil reference carried");
    const auto no_stencil = block(z_control(frag_greater, false), 0, 0x5a, 0x5a);
    on = select_depth_stencil(true, no_stencil.data(), no_stencil.size());
    check(on.applied && !on.has_stencil_ref && on.desc.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ZERO, "switch on: no reference without a stencil test");
    // Refusals keep the desktop state and say why.
    on = select_depth_stencil(true, refused.data(), refused.size());
    check(!on.applied && on.reason && desc_is_zero(on.desc), "switch on: colour-write tie refused with a reason");
    on = select_depth_stencil(true, reversed.data(), a_refmask_bf + 3);
    check(!on.applied && on.reason, "switch on: short block refused with a reason");
    on = select_depth_stencil(true, nullptr, 0x200);
    check(!on.applied && on.reason, "switch on: no block refused with a reason");
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc > 2 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "switch") == 0) {
        mutate_switch = true;
    } else if (argc > 1) {
        std::printf("usage: startup_depth_state_check [--mutate switch]\n");
        return 2;
    }
    cases_all();
    std::printf("%d of %d checks failed\n", failures, cases);
    return failures ? 1 : 0;
}
