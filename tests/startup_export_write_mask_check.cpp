// Export write masks from a translated pipeline manifest (src/startup_stubs/export_write_mask.h, #1674),
// without a device: the 'wmask' line forms that are accepted and refused, and how the masks enter a
// blend state.
//
// usage: startup_export_write_mask_check [--mutate shared-blend]
// --mutate shared-blend applies the masks to a shared blend state without expanding it first, so a
// mask on target 1 never reaches the draw; the check must then fail.
#include "startup_stubs/export_write_mask.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures = 0;

void expect(bool condition, const std::string& what)
{
    std::printf("%s: %s\n", what.c_str(), condition ? "ok" : "MISMATCH");
    failures += !condition;
}

bool accepts(const char* line, x2w_startup::ExportWriteMasks* out = nullptr)
{
    x2w_startup::ExportWriteMasks masks;
    const bool ok = masks.read_manifest_line(line);
    if (out) *out = masks;
    return ok;
}

D3D12_BLEND_DESC opaque_shared()
{
    D3D12_BLEND_DESC blend{};
    blend.RenderTarget[0] = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE,
                             D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    return blend;
}

void apply(D3D12_BLEND_DESC& blend, const x2w_startup::ExportWriteMasks& masks, bool mutate)
{
    if (!mutate) {
        expect(x2w_startup::apply_export_write_masks(blend, masks, 2), "mask applies to two targets");
        return;
    }
    for (unsigned target = 0; target < 8; ++target) blend.RenderTarget[target].RenderTargetWriteMask &= masks.mask[target];
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "shared-blend");
    if (argc != 1 && !mutate) {
        std::fprintf(stderr, "usage: startup_export_write_mask_check [--mutate shared-blend]\n");
        return 2;
    }

    // 1. Line forms. The writer emits 'wmask <target> 0x<mask>' with a mask of one to three channels.
    x2w_startup::ExportWriteMasks parsed;
    expect(accepts("wmask 0 0x3", &parsed) && parsed.mask[0] == 0x3 && parsed.any(), "wmask 0 0x3 accepted");
    expect(accepts("wmask 7 0xc", &parsed) && parsed.mask[7] == 0xC && parsed.mask[0] == 0xF, "wmask 7 0xc accepted");
    expect(accepts("wmask 2 0xf"), "full mask accepted");
    for (const char* bad : {"wmask 8 0x3", "wmask 0 0x0", "wmask 0 0x10", "wmask 0 3", "wmask -1 0x3", "wmask 0", "wmask",
                            "wmask 0 0x3 extra", "wmask0 0x3", "mask 0 0x3", "wmask 4294967296 0x3",
                            "wmask 0 0x100000003", "wmask 0 0x+3", "wmask +0 0x3"})
        expect(!accepts(bad), std::string("refused: '") + bad + "'");
    x2w_startup::ExportWriteMasks twice;
    expect(twice.read_manifest_line("wmask 1 0x3") && !twice.read_manifest_line("wmask 1 0xc"), "second line for target 1 refused");
    expect(!x2w_startup::ExportWriteMasks{}.any(), "no line: no mask");

    // 2. No mask leaves the blend state exactly as it was.
    D3D12_BLEND_DESC untouched = opaque_shared();
    apply(untouched, x2w_startup::ExportWriteMasks{}, mutate);
    const D3D12_BLEND_DESC reference = opaque_shared();
    expect(!std::memcmp(&untouched, &reference, sizeof(reference)), "no mask: blend state unchanged");

    // 3. A shared blend state is expanded to eight equal entries, then each target loses only its
    // unwritten channels.
    x2w_startup::ExportWriteMasks masks;
    masks.read_manifest_line("wmask 1 0x3");
    D3D12_BLEND_DESC shared = opaque_shared();
    apply(shared, masks, mutate);
    expect(shared.IndependentBlendEnable == TRUE, "shared blend state becomes independent");
    expect(shared.RenderTarget[0].RenderTargetWriteMask == 0xF, "target 0 keeps every channel");
    expect(shared.RenderTarget[1].RenderTargetWriteMask == 0x3, "target 1 writes channels 0 and 1");
    bool copied = true;
    for (unsigned t = 2; t < 8; ++t)
        copied &= !std::memcmp(&shared.RenderTarget[t], &shared.RenderTarget[0], sizeof(shared.RenderTarget[0]));
    expect(copied, "targets 2 to 7 copy target 0");

    // 4. An independent blend state keeps its per-target state; the mask only clears channels.
    D3D12_BLEND_DESC independent = opaque_shared();
    independent.IndependentBlendEnable = TRUE;
    independent.RenderTarget[0].BlendEnable = TRUE;
    independent.RenderTarget[0].RenderTargetWriteMask = 0x9;
    x2w_startup::ExportWriteMasks low;
    low.read_manifest_line("wmask 0 0x3");
    apply(independent, low, mutate);
    expect(independent.RenderTarget[0].BlendEnable == TRUE, "independent blend state kept");
    expect(independent.RenderTarget[0].RenderTargetWriteMask == 0x1, "console mask 0x9 and export mask 0x3 give 0x1");

    D3D12_BLEND_DESC logic = opaque_shared();
    logic.RenderTarget[0].LogicOpEnable = TRUE;
    logic.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_XOR;
    expect(!x2w_startup::apply_export_write_masks(logic, low, 2) && !logic.IndependentBlendEnable &&
           logic.RenderTarget[0].RenderTargetWriteMask == 0xF, "different masks with a shared logic op are refused");
    expect(x2w_startup::apply_export_write_masks(logic, low, 1) && !logic.IndependentBlendEnable &&
           logic.RenderTarget[0].RenderTargetWriteMask == 0x3 && logic.RenderTarget[0].LogicOpEnable,
           "one masked target retains its shared logic op");

    std::printf("startup export write mask checks: %d failed\n", failures);
    return failures ? 1 : 0;
}
