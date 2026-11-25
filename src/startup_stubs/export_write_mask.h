// Experimental (docs/startup-runtime.md): the channels a translated pixel stage writes per render
// target (#1674).
//
// gfx2hlsl writes 'wmask <target> 0x<mask>' to a pipeline manifest when an export leaves channels
// 'off' (bit c = channel c written). An 'off' channel is not written on the console, so the target
// keeps what it held there. The translated HLSL returns a float4 for every target, so the channels
// outside the mask have to be removed from that target's RenderTargetWriteMask instead.
// CB_TARGET_MASK has not been located (state_block_blend.h), so the export mask is the only channel
// mask known for a translated pipeline.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include <string_view>

namespace x2w_startup {
struct ExportWriteMasks {
    UINT8 mask[8] = {0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF};
    unsigned seen = 0;  // bit t: target t had a 'wmask' line

    // A 'wmask' manifest line. Returns false for a malformed line, a target above 7, a mask of no
    // channel or above 0xF, and a second line for the same target; the manifest is then rejected.
    bool read_manifest_line(std::string_view line)
    {
        if (line.size() != 11 || line.substr(0, 6) != "wmask " || line[6] < '0' || line[6] > '7' ||
            line.substr(7, 3) != " 0x") return false;
        const unsigned target = line[6] - '0';
        const char digit = line[10];
        unsigned value = 0;
        if (digit >= '1' && digit <= '9') value = digit - '0';
        if (digit >= 'a' && digit <= 'f') value = digit - 'a' + 10;
        if (digit >= 'A' && digit <= 'F') value = digit - 'A' + 10;
        if (value == 0 || (seen & (1u << target))) return false;
        seen |= 1u << target;
        mask[target] = static_cast<UINT8>(value);
        return true;
    }

    bool any() const { return seen != 0; }
};

// Removes the channels each export leaves unwritten from the pipeline's blend state. A shared blend
// state (IndependentBlendEnable FALSE) applies RenderTarget[0] to every target, so it is first
// expanded to eight equal entries; the per-target masks then differ only where the exports differ.
inline bool apply_export_write_masks(D3D12_BLEND_DESC& blend, const ExportWriteMasks& masks, UINT target_count)
{
    if (target_count > 8) return false;
    if (!masks.any() || !target_count) return true;
    if (!blend.IndependentBlendEnable) {
        const UINT8 original = blend.RenderTarget[0].RenderTargetWriteMask;
        const UINT8 first = original & masks.mask[0];
        bool same = true;
        for (UINT target = 1; target < target_count; ++target)
            same &= (original & masks.mask[target]) == first;
        if (same) {
            blend.RenderTarget[0].RenderTargetWriteMask = first;
            return true;
        }
        // D3D12 requires a shared blend state when a logic operation is enabled.
        if (blend.RenderTarget[0].LogicOpEnable) return false;
        for (unsigned target = 1; target < 8; ++target) blend.RenderTarget[target] = blend.RenderTarget[0];
        blend.IndependentBlendEnable = TRUE;
    }
    for (unsigned target = 0; target < target_count; ++target)
        blend.RenderTarget[target].RenderTargetWriteMask &= masks.mask[target];
    return true;
}
}  // namespace x2w_startup
