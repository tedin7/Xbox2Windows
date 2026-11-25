// Experimental (docs/startup-runtime.md): the decision behind X2W_DEPTH_STATE (#1999, item A).
//
// With the switch on, a pipeline created through device slot 57 takes its depth-stencil state from the
// register words of its state block A (pipeline_state_decode.h). With the switch off, or when the block
// is too short or the decoder refuses it, nothing is taken and the caller keeps the state it had before
// (the desktop default: depth and stencil off, no reference). Pure, so it can be checked without a device.
#pragma once

#include "startup_stubs/pipeline_state_decode.h"

namespace x2w_pipeline_state {

struct AppliedDepthStencil {
    bool applied = false;           // desc / stencil_ref / has_stencil_ref are valid and replace the desktop state
    const char* reason = nullptr;   // why not, when the switch is on (null when it is off)
    D3D12_DEPTH_STENCIL_DESC desc{};
    bool has_stencil_ref = false;   // the stencil test is enabled: stencil_ref is the list state the draws need
    std::uint8_t stencil_ref = 0;
    bool depth_bounds = false, masks_differ = false;  // decoder flags, reported by the caller
};

inline AppliedDepthStencil select_depth_stencil(bool switch_on, const unsigned char* block_a, std::size_t bytes)
{
    AppliedDepthStencil out;
    if (!switch_on) return out;
    DepthStencilWords words;
    if (!read_depth_stencil_words(block_a, bytes, words)) {
        out.reason = "state block A too short for the depth-stencil words";
        return out;
    }
    const DepthStencil decoded = decode_depth_stencil(words.depth, words.stencil, words.refmask, words.refmask_bf);
    if (!decoded.ok) {
        out.reason = decoded.reason;
        return out;
    }
    out.applied = true;
    out.desc = decoded.desc;
    out.has_stencil_ref = decoded.desc.StencilEnable != FALSE;
    out.stencil_ref = decoded.stencil_ref;
    out.depth_bounds = decoded.depth_bounds;
    out.masks_differ = decoded.masks_differ;
    return out;
}

}  // namespace x2w_pipeline_state
