// Checks src/startup_stubs/pipeline_state_decode.h by default: hand-built DB_DEPTH_CONTROL,
// DB_STENCIL_CONTROL and DB_STENCILREFMASK(_BF) words decoded to
// D3D12_DEPTH_STENCIL_DESC, the refusals, and a synthetic slot 57 state block.
//
// --mutate offset: expects the words one word later in the block (the check must then fail).
// --mutate mesh-shift: puts a mesh block's words at the vertex offsets (the check must then fail).
#include "startup_stubs/pipeline_state_decode.h"
#include "startup_pipeline_creation_warp.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace x2w_pipeline_state;
int failures, cases;
bool mutate_offset, mutate_shift;

void check(bool ok, const char* name)
{
    ++cases;
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

// DB_DEPTH_CONTROL from its fields.
constexpr std::uint32_t depth_control(bool stencil, bool z, bool z_write, bool bounds, unsigned zfunc, bool backface, unsigned sfunc,
                                      unsigned sfunc_bf)
{
    return std::uint32_t{stencil} | std::uint32_t{z} << 1 | std::uint32_t{z_write} << 2 | std::uint32_t{bounds} << 3 | (zfunc & 7) << 4 |
           std::uint32_t{backface} << 7 | (sfunc & 7) << 8 | (sfunc_bf & 7) << 20;
}
constexpr std::uint32_t stencil_control(unsigned fail, unsigned zpass, unsigned zfail, unsigned fail_bf, unsigned zpass_bf, unsigned zfail_bf)
{
    return fail | zpass << 4 | zfail << 8 | fail_bf << 12 | zpass_bf << 16 | zfail_bf << 20;
}
constexpr std::uint32_t refmask(unsigned test, unsigned mask, unsigned write, unsigned opval) { return test | mask << 8 | write << 16 | opval << 24; }

void depth_stencil_cases()
{
    // Depth off: the function the word carries is ignored (canonical ALWAYS), stencil off.
    auto d = decode_depth_stencil(depth_control(false, false, false, false, 1, true, 7, 7), 0, refmask(0, 0xff, 0, 1), refmask(0, 0xff, 0, 1));
    check(d.ok && !d.desc.DepthEnable && d.desc.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ZERO && d.desc.DepthFunc == D3D12_COMPARISON_FUNC_ALWAYS &&
              !d.desc.StencilEnable, "depth off");
    // Reversed depth: GEQUAL read-only.
    d = decode_depth_stencil(depth_control(false, true, false, false, 6, true, 7, 7), 0, refmask(0, 0xff, 0, 1), refmask(0, 0xff, 0, 1));
    check(d.ok && d.desc.DepthEnable && d.desc.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ZERO && d.desc.DepthFunc == D3D12_COMPARISON_FUNC_GREATER_EQUAL,
          "GEQUAL read");
    // Z_WRITE_ENABLE without Z_ENABLE writes nothing.
    d = decode_depth_stencil(depth_control(false, false, true, false, 1, false, 7, 7), 0, 0, 0);
    check(d.ok && !d.desc.DepthEnable && d.desc.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ZERO, "write without test");
    // Every compare function in order.
    const D3D12_COMPARISON_FUNC funcs[8] = {D3D12_COMPARISON_FUNC_NEVER, D3D12_COMPARISON_FUNC_LESS, D3D12_COMPARISON_FUNC_EQUAL,
                                            D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_COMPARISON_FUNC_GREATER, D3D12_COMPARISON_FUNC_NOT_EQUAL,
                                            D3D12_COMPARISON_FUNC_GREATER_EQUAL, D3D12_COMPARISON_FUNC_ALWAYS};
    for (unsigned f = 0; f < 8; ++f) {
        d = decode_depth_stencil(depth_control(true, true, true, false, f, true, f, 7 - f), 0, refmask(0, 0xff, 0xff, 1), refmask(0, 0xff, 0xff, 1));
        check(d.ok && d.desc.DepthFunc == funcs[f] && d.desc.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL && d.desc.FrontFace.StencilFunc == funcs[f] &&
                  d.desc.BackFace.StencilFunc == funcs[7 - f], "compare function order");
    }
    // Stencil: replace on depth pass, both faces, masks and the reference value.
    d = decode_depth_stencil(depth_control(true, true, true, false, 6, true, 7, 7), stencil_control(0, 3, 0, 0, 3, 0), refmask(0x5c, 0xff, 0xff, 1),
                             refmask(0x5c, 0xff, 0xff, 1));
    check(d.ok && d.desc.StencilEnable && d.desc.StencilReadMask == 0xff && d.desc.StencilWriteMask == 0xff && d.stencil_ref == 0x5c &&
              d.desc.FrontFace.StencilPassOp == D3D12_STENCIL_OP_REPLACE && d.desc.BackFace.StencilPassOp == D3D12_STENCIL_OP_REPLACE &&
              d.desc.FrontFace.StencilFailOp == D3D12_STENCIL_OP_KEEP && d.desc.FrontFace.StencilDepthFailOp == D3D12_STENCIL_OP_KEEP &&
              !d.masks_differ, "stencil replace");
    // Every op with a D3D12 form, front fail / back depth-fail; the add / subtract ops by one.
    const struct { unsigned op; D3D12_STENCIL_OP d3d; } ops[] = {{0, D3D12_STENCIL_OP_KEEP},     {1, D3D12_STENCIL_OP_ZERO},
                                                                 {3, D3D12_STENCIL_OP_REPLACE},  {5, D3D12_STENCIL_OP_INCR_SAT},
                                                                 {6, D3D12_STENCIL_OP_DECR_SAT}, {7, D3D12_STENCIL_OP_INVERT},
                                                                 {8, D3D12_STENCIL_OP_INCR},     {9, D3D12_STENCIL_OP_DECR}};
    for (const auto& o : ops) {
        d = decode_depth_stencil(depth_control(true, false, false, false, 7, true, 2, 2), stencil_control(o.op, 0, 0, 0, 0, o.op),
                                 refmask(0, 0x28, 0, 1), refmask(0, 0x28, 0, 1));
        check(d.ok && d.desc.FrontFace.StencilFailOp == o.d3d && d.desc.BackFace.StencilDepthFailOp == o.d3d && d.desc.StencilReadMask == 0x28 &&
                  d.desc.StencilWriteMask == 0, "stencil op mapping");
    }
    // No back-face state: back faces use the front state (the _BF words are ignored).
    d = decode_depth_stencil(depth_control(true, false, false, false, 7, false, 3, 0), stencil_control(0, 8, 0, 9, 9, 9), refmask(0, 0x80, 0x80, 1),
                             refmask(0, 0, 0, 7));
    check(d.ok && d.desc.BackFace.StencilFunc == D3D12_COMPARISON_FUNC_LESS_EQUAL && d.desc.BackFace.StencilPassOp == D3D12_STENCIL_OP_INCR &&
              !d.masks_differ, "back faces follow the front");
    // Negative: ops without a form, add by two, colour writes tied to the depth result.
    check(!decode_depth_stencil(depth_control(true, false, false, false, 7, true, 7, 7), stencil_control(2, 0, 0, 0, 0, 0), refmask(0, 0xff, 0xff, 1),
                                refmask(0, 0xff, 0xff, 1)).ok, "ONES refused");
    check(!decode_depth_stencil(depth_control(true, false, false, false, 7, true, 7, 7), stencil_control(0, 4, 0, 0, 4, 0), refmask(0, 0xff, 0xff, 1),
                                refmask(0, 0xff, 0xff, 1)).ok, "REPLACE_OP refused");
    check(!decode_depth_stencil(depth_control(true, false, false, false, 7, true, 7, 7), stencil_control(0, 8, 0, 0, 8, 0), refmask(0, 0xff, 0xff, 2),
                                refmask(0, 0xff, 0xff, 2)).ok, "add by two refused");
    check(!decode_depth_stencil(depth_control(true, false, false, false, 7, true, 7, 7), stencil_control(0, 0, 0, 0, 12, 0), refmask(0, 0xff, 0xff, 1),
                                refmask(0, 0xff, 0xff, 1)).ok, "back-face XOR refused");
    check(!decode_depth_stencil(depth_control(false, true, true, false, 6, false, 7, 7) | 1u << 30, 0, 0, 0).ok, "colour writes on depth fail refused");
    check(!decode_depth_stencil(depth_control(false, true, true, false, 6, false, 7, 7) | 1u << 31, 0, 0, 0).ok, "colour writes on depth pass refused");
    // An unusable op on a disabled stencil does not refuse the depth state.
    check(decode_depth_stencil(depth_control(false, true, true, false, 1, true, 7, 7), stencil_control(2, 2, 2, 2, 2, 2), 0, 0).ok,
          "ops of a disabled stencil ignored");
    // Flags.
    d = decode_depth_stencil(depth_control(false, true, false, true, 3, false, 7, 7), 0, 0, 0);
    check(d.ok && d.depth_bounds, "depth bounds flagged");
    d = decode_depth_stencil(depth_control(true, false, false, false, 7, true, 7, 7), 0, refmask(0, 0xff, 0xff, 1), refmask(0, 0x0f, 0xff, 1));
    check(d.ok && d.masks_differ && d.desc.StencilReadMask == 0xff, "back masks differ flagged, front kept");
    d = decode_depth_stencil(depth_control(true, false, false, false, 7, true, 7, 7), 0, refmask(1, 0xff, 0xff, 1), refmask(2, 0xff, 0xff, 1));
    check(d.ok && d.masks_differ && d.stencil_ref == 1, "back reference differs flagged, front kept");
}

void block_cases()
{
    {
        std::vector<unsigned char> a(0x184, 0x5a);  // unrelated state around the words
        const std::size_t s = mutate_offset ? 4 : 0;
        auto put = [&](std::size_t off, std::uint32_t v) { std::memcpy(a.data() + off + s, &v, 4); };
        const std::uint32_t dc = depth_control(true, true, true, false, 6, true, 7, 7), sc = stencil_control(0, 3, 0, 0, 3, 0);
        const std::uint32_t rm = refmask(7, 0xff, 0, 1), rm_bf = refmask(7, 0xfe, 0, 1);
        put(a_depth, dc);
        put(a_stencil, sc);
        put(a_refmask, rm);
        put(a_refmask_bf, rm_bf);
        DepthStencilWords w;
        const bool ok = read_depth_stencil_words(a.data(), a.size(), w);
        check(ok && w.depth == dc && w.stencil == sc && w.refmask == rm && w.refmask_bf == rm_bf, "block words");
        const auto d = decode_depth_stencil(w.depth, w.stencil, w.refmask, w.refmask_bf);
        check(d.ok && d.desc.DepthEnable && d.desc.DepthFunc == D3D12_COMPARISON_FUNC_GREATER_EQUAL && d.desc.StencilEnable &&
                  d.desc.FrontFace.StencilPassOp == D3D12_STENCIL_OP_REPLACE && d.stencil_ref == 7 && d.masks_differ,
              "block decodes");
        DepthStencilWords short_words;
        check(!read_depth_stencil_words(a.data(), a_refmask_bf + 3, short_words), "short block refused");
    }
    DepthStencilWords none;
    check(!read_depth_stencil_words(nullptr, 0x184, none) && !read_depth_stencil_words(std::vector<unsigned char>(4).data(), 4, none),
          "no block refused");
}

// The two layouts: +0x004 = 4 (vertex) reads the words at their offsets, +0x004 = 6 (mesh) 0x54 bytes later.
// Each block also holds a different, decodable depth state at the other layout's offsets, so reading the
// wrong offsets gives the wrong state instead of zeros.
void layout_cases()
{
    const std::uint32_t mesh_dc = depth_control(true, true, true, false, 6, true, 7, 7), mesh_sc = stencil_control(0, 3, 0, 0, 3, 0);
    const std::uint32_t mesh_rm = refmask(0x21, 0xf0, 0x0f, 1);
    const std::uint32_t vert_dc = depth_control(false, true, false, false, 1, false, 7, 7);
    for (std::uint32_t layout : {4u, 6u, 5u}) {
        std::vector<unsigned char> a(0x1d8, 0x5a);
        auto put = [&](std::size_t off, std::uint32_t v) { std::memcpy(a.data() + off, &v, 4); };
        put(a_layout, layout);
        const bool mesh = layout == a_layout_mesh;
        // The mesh state at the shifted offsets (or, mutated, at the vertex offsets), the vertex state at the others.
        const std::size_t ms = mutate_shift ? 0 : a_mesh_shift, vs = mutate_shift ? a_mesh_shift : 0;
        put(a_depth + ms, mesh_dc);
        put(a_stencil + ms, mesh_sc);
        put(a_refmask + ms, mesh_rm);
        put(a_refmask_bf + ms, mesh_rm);
        put(a_depth + vs, vert_dc);
        put(a_stencil + vs, 0);
        put(a_refmask + vs, 0);
        put(a_refmask_bf + vs, 0);
        DepthStencilWords w;
        const bool ok = read_depth_stencil_words(a.data(), a.size(), w);
        const auto d = decode_depth_stencil(w.depth, w.stencil, w.refmask, w.refmask_bf);
        if (mesh)
            check(ok && w.mesh_layout && w.depth == mesh_dc && d.ok && d.desc.StencilEnable && d.stencil_ref == 0x21 &&
                      d.desc.StencilReadMask == 0xf0 && d.desc.DepthFunc == D3D12_COMPARISON_FUNC_GREATER_EQUAL,
                  "mesh layout words");
        else
            check(ok && !w.mesh_layout && w.depth == vert_dc && d.ok && !d.desc.StencilEnable && d.desc.DepthFunc == D3D12_COMPARISON_FUNC_LESS,
                  layout == 4 ? "vertex layout words" : "other layout values read at the vertex offsets");
    }
    // A mesh block long enough for the vertex offsets but not the shifted ones is refused.
    std::vector<unsigned char> a(a_refmask_bf + 4 + 0x20, 0);
    std::memcpy(a.data() + a_layout, &a_layout_mesh, 4);
    DepthStencilWords w;
    check(!read_depth_stencil_words(a.data(), a.size(), w), "short mesh block refused");
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--creation-warp") == 0)
        return pipeline_creation_warp::run(false);
    if (argc == 2 && std::strcmp(argv[1], "--creation-mutant-warp") == 0)
        return pipeline_creation_warp::run(true);
    if (argc > 2 && std::strcmp(argv[1], "--mutate") == 0) {
        mutate_offset = std::strcmp(argv[2], "offset") == 0;
        mutate_shift = std::strcmp(argv[2], "mesh-shift") == 0;
        if (!mutate_offset && !mutate_shift) {
            std::printf("unknown mutation\n");
            return 2;
        }
    } else if (argc > 1) {
        std::printf("usage: startup_pipeline_state_check [--mutate offset|mesh-shift | --creation-warp | --creation-mutant-warp]\n");
        return 2;
    }
    depth_stencil_cases();
    block_cases();
    layout_cases();
    std::printf("%d of %d checks failed\n", failures, cases);
    return failures ? 1 : 0;
}
