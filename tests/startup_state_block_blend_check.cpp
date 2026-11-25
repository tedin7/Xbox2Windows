// Checks src/startup_stubs/state_block_blend.h without a device: synthetic slot 57 state blocks
// carrying the four CB_BLENDn_CONTROL words measured on #1216, and the refusals.
#include "startup_stubs/state_block_blend.h"

#include <cstdio>
#include <vector>

namespace {
int failures;

std::vector<unsigned char> block_with(std::uint32_t target0, std::uint32_t others = 0, std::size_t bytes = 388)
{
    std::vector<unsigned char> block(bytes, 0xA5);  // unrelated state around the words
    for (unsigned target = 0; target < 8; ++target) {
        const std::uint32_t word = target ? others : target0;
        if (x2w_startup::state_block_blend_offset + 4 * target + 4 <= bytes)
            std::memcpy(block.data() + x2w_startup::state_block_blend_offset + 4 * target, &word, 4);
    }
    return block;
}

void fail(const char* name, const char* what)
{
    std::printf("FAIL %s: %s\n", name, what);
    ++failures;
}

void expect_target0(const char* name, std::uint32_t word, D3D12_BLEND src, D3D12_BLEND dest, D3D12_BLEND src_alpha, D3D12_BLEND dest_alpha)
{
    const auto block = block_with(word);
    D3D12_BLEND_DESC blend{};
    const char* reason = nullptr;
    unsigned refused = 0;
    if (!x2w_startup::state_block_blend(block.data(), block.size(), false, &blend, &reason, &refused)) return fail(name, reason ? reason : "refused");
    const auto& t = blend.RenderTarget[0];
    if (!blend.IndependentBlendEnable || !t.BlendEnable || t.SrcBlend != src || t.DestBlend != dest || t.BlendOp != D3D12_BLEND_OP_ADD ||
        t.SrcBlendAlpha != src_alpha || t.DestBlendAlpha != dest_alpha || t.BlendOpAlpha != D3D12_BLEND_OP_ADD ||
        t.RenderTargetWriteMask != D3D12_COLOR_WRITE_ENABLE_ALL)
        return fail(name, "render target 0 differs");
    // The other seven words are 0: written unblended with all channels.
    for (unsigned target = 1; target < 8; ++target)
        if (blend.RenderTarget[target].BlendEnable || blend.RenderTarget[target].RenderTargetWriteMask != D3D12_COLOR_WRITE_ENABLE_ALL)
            return fail(name, "a disabled target is blended or masked");
    if (refused) fail(name, "a target was reported refused");
}
}  // namespace

int main()
{
    // The four words measured at state block A +0xD0 (#1216).
    expect_target0("straight alpha", 0xE5040504, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA);
    expect_target0("straight colour, one alpha", 0xE5010504, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA);
    expect_target0("premultiplied", 0xE5010501, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA);
    expect_target0("additive", 0xE1000104, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_ONE);

    D3D12_BLEND_DESC blend{};
    const char* reason = nullptr;
    unsigned refused = 0;

    // ENABLE clear on target 0: a blend state is still answered, unblended.
    auto block = block_with(0x00040504);
    if (!x2w_startup::state_block_blend(block.data(), block.size(), false, &blend, &reason) || blend.RenderTarget[0].BlendEnable)
        fail("enable clear", "not answered as unblended");

    // Offset control: the same block read four bytes late sees target 1's word (0, unblended) as
    // target 0, so a hookup at the wrong offset cannot answer the measured state.
    block = block_with(0xE5040504);
    D3D12_BLEND_DESC shifted{};
    if (x2w_startup::state_block_blend(block.data() + 4, block.size() - 4, false, &shifted, &reason) && shifted.RenderTarget[0].BlendEnable)
        fail("offset control", "a block read at the wrong offset gave a blended target 0");

    // A block that ends before the eighth word, and no block.
    block = block_with(0xE5040504, 0, x2w_startup::state_block_blend_offset + 31);
    D3D12_BLEND_DESC untouched{};
    untouched.AlphaToCoverageEnable = 77;
    if (x2w_startup::state_block_blend(block.data(), block.size(), false, &untouched, &reason) || untouched.AlphaToCoverageEnable != 77 || !reason)
        fail("short block", "not refused, or the output was written");
    if (x2w_startup::state_block_blend(nullptr, 1024, false, &untouched, &reason)) fail("null block", "not refused");

    // A form D3D12 cannot express on target 0 (BOTH_SRC_ALPHA, factor 11): refused with the decoder's reason.
    block = block_with(x2w_cb_blend::control_word(11, 0, 5, 11, 0, 5, true, true));
    if (x2w_startup::state_block_blend(block.data(), block.size(), false, &untouched, &reason) || untouched.AlphaToCoverageEnable != 77 || !reason)
        fail("refused target 0", "not refused, or the output was written");

    // The same form on targets 1-7 only: the whole block is refused and every refused target is
    // reported. No target is given an unblended stand-in, which would change what it draws.
    block = block_with(0xE5040504, x2w_cb_blend::control_word(11, 0, 5, 11, 0, 5, true, true));
    reason = nullptr;
    if (x2w_startup::state_block_blend(block.data(), block.size(), false, &untouched, &reason, &refused) || refused != 0xFE ||
        untouched.AlphaToCoverageEnable != 77 || !reason)
        fail("refused other targets", "not refused, a refused target is unreported, or the output was written");

    // Dual-source factors (SRC1_*) read the second pixel shader output, which only target 0 has.
    // Target 0 may use them when no other target blends; any other target that blends with a SRC1_*
    // factor, or blends at all alongside a dual-source target 0, is refused and reported. The same
    // words with ENABLE clear use no factor and do not blend: not refused.
    const std::uint32_t dual_words[] = {x2w_cb_blend::control_word(15, 0, 1, 1, 0, 0, true, true), x2w_cb_blend::control_word(1, 0, 16, 1, 0, 0, true, true),
                                        x2w_cb_blend::control_word(1, 0, 0, 17, 0, 0, true, true), x2w_cb_blend::control_word(1, 0, 0, 1, 0, 18, true, true)};
    for (const std::uint32_t dual : dual_words) {
        block = block_with(dual);
        if (!x2w_startup::state_block_blend(block.data(), block.size(), false, &blend, &reason, &refused) || refused)
            fail("dual source on target 0", "refused");
        for (unsigned target = 1; target < 8; ++target) {
            block = block_with(0xE5040504);
            std::memcpy(block.data() + x2w_startup::state_block_blend_offset + 4 * target, &dual, 4);
            untouched.AlphaToCoverageEnable = 77;
            reason = nullptr;
            refused = 0;
            if (x2w_startup::state_block_blend(block.data(), block.size(), false, &untouched, &reason, &refused) || refused != 1u << target ||
                untouched.AlphaToCoverageEnable != 77 || !reason)
                fail("dual source on target > 0", "not refused, the target is unreported, or the output was written");
            const std::uint32_t off = dual & ~(1u << 30);
            std::memcpy(block.data() + x2w_startup::state_block_blend_offset + 4 * target, &off, 4);
            if (!x2w_startup::state_block_blend(block.data(), block.size(), false, &blend, &reason, &refused) || refused)
                fail("dual source, ENABLE clear", "a target that does not blend was refused");

            // Dual-source on target 0 cannot be combined with blending enabled on another target.
            block = block_with(dual);
            const std::uint32_t normal_blend = 0xE5040504;
            std::memcpy(block.data() + x2w_startup::state_block_blend_offset + 4 * target, &normal_blend, 4);
            untouched.AlphaToCoverageEnable = 77;
            reason = nullptr;
            refused = 0;
            if (x2w_startup::state_block_blend(block.data(), block.size(), false, &untouched, &reason, &refused) || refused != 1u << target ||
                untouched.AlphaToCoverageEnable != 77 || !reason)
                fail("dual source on target 0 with blending target > 0", "not refused, the target is unreported, or the output was written");
            const std::uint32_t normal_off = 0x00040504;
            std::memcpy(block.data() + x2w_startup::state_block_blend_offset + 4 * target, &normal_off, 4);
            if (!x2w_startup::state_block_blend(block.data(), block.size(), false, &blend, &reason, &refused) || refused)
                fail("dual source on target 0 with disabled target > 0", "a disabled target was refused");
        }
        block = block_with(dual, 0xE5040504);
        untouched.AlphaToCoverageEnable = 77;
        reason = nullptr;
        refused = 0;
        if (x2w_startup::state_block_blend(block.data(), block.size(), false, &untouched, &reason, &refused) || refused != 0xFE ||
            untouched.AlphaToCoverageEnable != 77 || !reason)
            fail("dual source on target 0 with all other targets blending", "not refused, a refused target is unreported, or the output was written");
    }

    if (failures) {
        std::printf("startup_state_block_blend: %d FAILED\n", failures);
        return 1;
    }
    std::printf("startup_state_block_blend: PASS (4 measured words, enable clear, 5 controls, dual source)\n");
    return 0;
}
