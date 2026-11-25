// Experimental (docs/startup-runtime.md): the depth-stencil state of a pipeline created through
// device slot 57, decoded from the register words in its state block A (#1999, item A).
//
// State block A holds DB_DEPTH_CONTROL at +0xB8, DB_STENCIL_CONTROL at +0xBC and DB_STENCILREFMASK /
// DB_STENCILREFMASK_BF at +0xC0 / +0xC4, before the CB_BLENDn_CONTROL words at +0xD0 (#1216). The
// positions were measured locally on the vertex-pipeline layout, whose word at +0x004 is 4.
//
// Mesh pipelines use a second layout, told apart by the word at +0x004 being 6: every field from +0x058
// on sits 0x54 bytes later (so DB_DEPTH_CONTROL is at +0x10C), and the words before +0x058 keep their
// offsets. Also measured locally; a block with any other value at +0x004 is read at the vertex offsets.
//
// Field layouts and enumerations follow Mesa's src/amd/registers/gfx10.json (MIT): DB_DEPTH_CONTROL
// STENCIL_ENABLE [0], Z_ENABLE [1], Z_WRITE_ENABLE [2], DEPTH_BOUNDS_ENABLE [3], ZFUNC [6:4],
// BACKFACE_ENABLE [7], STENCILFUNC [10:8], STENCILFUNC_BF [22:20], ENABLE_COLOR_WRITES_ON_DEPTH_FAIL [30],
// DISABLE_COLOR_WRITES_ON_DEPTH_PASS [31]; DB_STENCIL_CONTROL STENCILFAIL [3:0], STENCILZPASS [7:4],
// STENCILZFAIL [11:8] and the _BF fields at [15:12], [19:16], [23:20]; DB_STENCILREFMASK STENCILTESTVAL
// [7:0], STENCILMASK [15:8], STENCILWRITEMASK [23:16], STENCILOPVAL [31:24]; CompareFrag FRAG_NEVER (0)
// .. FRAG_ALWAYS (7); StencilOp STENCIL_KEEP (0) .. STENCIL_XNOR (15).
//
// Pure functions, not used by the draw path yet. A form D3D12 cannot express is refused with a
// reason so the caller can keep the state it had before.
#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace x2w_pipeline_state {

// ---- state block A offsets (vertex-pipeline layout) ----
constexpr std::size_t a_depth = 0x0b8, a_stencil = 0x0bc, a_refmask = 0x0c0, a_refmask_bf = 0x0c4;
// The mesh-pipeline layout: the word at +0x004 and the shift applied from +0x058 on.
constexpr std::size_t a_layout = 0x004, a_mesh_shift_from = 0x058, a_mesh_shift = 0x54;
constexpr std::uint32_t a_layout_mesh = 6;

constexpr std::uint32_t bits(std::uint32_t v, unsigned lo, unsigned hi) { return (v >> lo) & ((1u << (hi - lo + 1)) - 1); }

// Bytes to add to a vertex-layout offset at or after +0x058 for this block (0 or a_mesh_shift).
inline std::size_t state_block_a_shift(const unsigned char* a, std::size_t bytes)
{
    if (!a || bytes < a_layout + 4) return 0;
    std::uint32_t layout;
    std::memcpy(&layout, a + a_layout, 4);
    return layout == a_layout_mesh ? a_mesh_shift : 0;
}

// The depth-stencil register words of one state block A.
struct DepthStencilWords {
    std::uint32_t depth = 0, stencil = 0, refmask = 0, refmask_bf = 0;
    bool mesh_layout = false;  // read at the mesh-pipeline offsets
};

// False, leaving w alone, when the block is too short for the words at its layout's offsets.
inline bool read_depth_stencil_words(const unsigned char* a, std::size_t bytes, DepthStencilWords& w)
{
    const std::size_t s = state_block_a_shift(a, bytes);
    if (!a || bytes < a_refmask_bf + s + 4) return false;
    w.mesh_layout = s != 0;
    auto u = [&](std::size_t off) { std::uint32_t v; std::memcpy(&v, a + off + s, 4); return v; };
    w.depth = u(a_depth);
    w.stencil = u(a_stencil);
    w.refmask = u(a_refmask);
    w.refmask_bf = u(a_refmask_bf);
    return true;
}

struct DepthStencil {
    bool ok = false;
    const char* reason = nullptr;
    D3D12_DEPTH_STENCIL_DESC desc{};
    std::uint8_t stencil_ref = 0;   // STENCILTESTVAL: D3D12 sets it on the command list (OMSetStencilRef)
    bool depth_bounds = false;      // DEPTH_BOUNDS_ENABLE: not applied (the bounds are separate state); flagged only
    bool masks_differ = false;      // back-face read/write masks or reference differ from the front: front used, flagged
};

// CompareFrag (FRAG_NEVER .. FRAG_ALWAYS) has the order of D3D12_COMPARISON_FUNC (NEVER = 1 .. ALWAYS = 8).
constexpr D3D12_COMPARISON_FUNC compare(std::uint32_t frag) { return static_cast<D3D12_COMPARISON_FUNC>((frag & 7) + 1); }

// StencilOp -> D3D12_STENCIL_OP; 0 when D3D12 has no equivalent. opval: STENCILOPVAL, the amount the
// ADD / SUB ops apply (D3D12's INCR / DECR forms add or subtract one).
inline D3D12_STENCIL_OP stencil_op(std::uint32_t op, std::uint32_t opval)
{
    switch (op) {
    case 0: return D3D12_STENCIL_OP_KEEP;
    case 1: return D3D12_STENCIL_OP_ZERO;
    case 3: return D3D12_STENCIL_OP_REPLACE;               // STENCIL_REPLACE_TEST: the test (reference) value
    case 5: return opval == 1 ? D3D12_STENCIL_OP_INCR_SAT : static_cast<D3D12_STENCIL_OP>(0);  // ADD_CLAMP
    case 6: return opval == 1 ? D3D12_STENCIL_OP_DECR_SAT : static_cast<D3D12_STENCIL_OP>(0);  // SUB_CLAMP
    case 7: return D3D12_STENCIL_OP_INVERT;
    case 8: return opval == 1 ? D3D12_STENCIL_OP_INCR : static_cast<D3D12_STENCIL_OP>(0);      // ADD_WRAP
    case 9: return opval == 1 ? D3D12_STENCIL_OP_DECR : static_cast<D3D12_STENCIL_OP>(0);      // SUB_WRAP
    default: return static_cast<D3D12_STENCIL_OP>(0);      // ONES, REPLACE_OP, AND .. XNOR
    }
}

inline DepthStencil decode_depth_stencil(std::uint32_t depth_control, std::uint32_t stencil_control, std::uint32_t refmask,
                                         std::uint32_t refmask_bf)
{
    DepthStencil out;
    D3D12_DEPTH_STENCIL_DESC& d = out.desc;
    if (bits(depth_control, 30, 31)) {
        out.reason = "colour writes tied to the depth result (DB_DEPTH_CONTROL bits 30-31)";
        return out;
    }
    const bool z = bits(depth_control, 1, 1) != 0;
    d.DepthEnable = z;
    d.DepthWriteMask = z && bits(depth_control, 2, 2) ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    d.DepthFunc = compare(bits(depth_control, 4, 6));
    out.depth_bounds = bits(depth_control, 3, 3) != 0;
    d.StencilEnable = bits(depth_control, 0, 0) != 0;
    const bool back_own = bits(depth_control, 7, 7) != 0;  // BACKFACE_ENABLE: else back faces use the front state
    d.StencilReadMask = static_cast<UINT8>(bits(refmask, 8, 15));
    d.StencilWriteMask = static_cast<UINT8>(bits(refmask, 16, 23));
    const std::uint32_t opval = bits(refmask, 24, 31), opval_bf = back_own ? bits(refmask_bf, 24, 31) : opval;
    auto face = [&](D3D12_DEPTH_STENCILOP_DESC& f, unsigned lo, std::uint32_t func, std::uint32_t val) {
        f.StencilFailOp = stencil_op(bits(stencil_control, lo, lo + 3), val);
        f.StencilPassOp = stencil_op(bits(stencil_control, lo + 4, lo + 7), val);
        f.StencilDepthFailOp = stencil_op(bits(stencil_control, lo + 8, lo + 11), val);
        f.StencilFunc = compare(func);
        return f.StencilFailOp && f.StencilPassOp && f.StencilDepthFailOp;
    };
    const bool front_ok = face(d.FrontFace, 0, bits(depth_control, 8, 10), opval);
    const bool back_ok = back_own ? face(d.BackFace, 12, bits(depth_control, 20, 22), opval_bf) : (d.BackFace = d.FrontFace, front_ok);
    if (d.StencilEnable && (!front_ok || !back_ok)) {
        out.reason = "stencil op without a D3D12 form (ONES, REPLACE_OP, logic op, or add / subtract by other than one)";
        return out;
    }
    if (!d.StencilEnable) {  // ignored by D3D12, but keep the description canonical (one PSO per real state)
        d.StencilReadMask = d.StencilWriteMask = 0xFF;
        d.FrontFace = d.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
    } else {
        out.stencil_ref = static_cast<std::uint8_t>(bits(refmask, 0, 7));
        // D3D12 has one reference and one pair of masks for both faces.
        if (back_own && bits(refmask_bf, 0, 23) != bits(refmask, 0, 23)) out.masks_differ = true;
    }
    if (!z) d.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;  // canonical: the function of a disabled test is ignored
    out.ok = true;
    return out;
}

}  // namespace x2w_pipeline_state
