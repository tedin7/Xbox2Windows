// Checks the static samplers src/startup_stubs/translated_sampler_bindings.h builds from manifest lines,
// without a device: a smpc line is a comparison sampler (D3D12 comparison filter and function from the
// SQ_IMG_SAMP words), a smp line stays an ordinary one, and the manifest order is the register order.
#include "startup_stubs/translated_sampler_bindings.h"

#include <cstdio>

namespace {
int failures;

void expect(bool ok, const char* what, int a = 0, int b = 0)
{
    if (!ok) {
        std::printf("FAIL %s (%d, %d)\n", what, a, b);
        ++failures;
    }
}

// SQ_TEX_DEPTH_COMPARE 0..7 in hardware order, written as the D3D12 enumerators.
const D3D12_COMPARISON_FUNC funcs[8] = {
    D3D12_COMPARISON_FUNC_NEVER, D3D12_COMPARISON_FUNC_LESS, D3D12_COMPARISON_FUNC_EQUAL,
    D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_COMPARISON_FUNC_GREATER, D3D12_COMPARISON_FUNC_NOT_EQUAL,
    D3D12_COMPARISON_FUNC_GREATER_EQUAL, D3D12_COMPARISON_FUNC_ALWAYS};
}  // namespace

int main()
{
    for (unsigned func = 0; func < 8; ++func)
        for (unsigned linear = 0; linear < 2; ++linear) {
            // CLAMP_LAST_TEXEL in X and Y, TRUNC_COORD with point filtering, the XY filter in word 2 bits 20..23.
            const unsigned w0 = 2u | 2u << 3 | func << 12 | (linear ? 0u : 1u << 27);
            const unsigned w2 = linear ? 1u << 20 | 1u << 22 : 0u;
            char line[96];
            std::snprintf(line, sizeof(line), "smpc 0 0x%x 0x0 0x%x 0x0", w0, w2);
            x2w_startup::TranslatedSamplers samplers;
            expect(samplers.read_manifest_line(line) && samplers.statics.size() == 1, "smpc line parsed", func, linear);
            const auto decoded = x2w_startup::decode_statics(samplers);
            if (decoded.size() != 1) continue;
            const auto& s = decoded[0];
            // D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT is 0x80, ..._MIN_MAG_LINEAR_MIP_POINT is 0x94.
            expect(s.Filter == (linear ? 0x94 : 0x80), "comparison filter", func, linear);
            expect(s.ComparisonFunc == funcs[func], "comparison function", func, linear);
            expect(s.AddressU == D3D12_TEXTURE_ADDRESS_MODE_CLAMP && s.AddressV == D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                   "clamp addressing", func, linear);
        }

    // An ordinary smp line is not a comparison sampler, whatever bits 14:12 hold, and keeps its register.
    x2w_startup::TranslatedSamplers mixed;
    expect(mixed.read_manifest_line("smp 0 0x7012 0x0 0x0 0x0") &&
           mixed.read_manifest_line("smpc 1 0x1012 0x0 0x0 0x0"), "mixed lines parsed");
    const auto both = x2w_startup::decode_statics(mixed);
    expect(both.size() == 2, "two samplers", int(both.size()));
    if (both.size() == 2) {
        expect(both[0].ShaderRegister == 0 && both[0].Filter == 0x0 && both[0].ComparisonFunc == D3D12_COMPARISON_FUNC_NEVER,
               "ordinary sampler at register 0", both[0].Filter, both[0].ComparisonFunc);
        expect(both[1].ShaderRegister == 1 && both[1].Filter == 0x80 && both[1].ComparisonFunc == D3D12_COMPARISON_FUNC_LESS,
               "comparison sampler at register 1", both[1].Filter, both[1].ComparisonFunc);
    }
    // An smpl line (image_sample_b) carries MIN_LOD (word 1 bits 11:0) and MAX_LOD (bits 23:12), u4.8, to MinLOD and
    // MaxLOD; the same words on an smp line keep MinLOD 0 and an unbounded MaxLOD.
    x2w_startup::TranslatedSamplers limits;
    expect(limits.read_manifest_line("smpl 0 0x0 0x100200 0x4000000 0x0") &&
           limits.read_manifest_line("smp 1 0x0 0x100200 0x4000000 0x0") &&
           limits.read_manifest_line("smpl 2 0x0 0xfff000 0x4000000 0x0") &&
           limits.read_manifest_line("smpl 3 0x0 0x180 0x4000000 0x0"), "limit lines parsed");
    const auto bounded = x2w_startup::decode_statics(limits);
    expect(bounded.size() == 4, "four limit samplers", int(bounded.size()));
    if (bounded.size() == 4) {
        expect(bounded[0].MinLOD == 2.0f && bounded[0].MaxLOD == 1.0f, "smpl MIN_LOD 2.0 and MAX_LOD 1.0",
               int(bounded[0].MinLOD * 256), int(bounded[0].MaxLOD * 256));
        expect(bounded[1].MinLOD == 0.0f && bounded[1].MaxLOD == D3D12_FLOAT32_MAX, "smp keeps unbounded limits",
               int(bounded[1].MinLOD), 0);
        expect(bounded[2].MaxLOD == 4095.0f / 256.0f, "smpl MAX_LOD 15.996", int(bounded[2].MaxLOD * 256), 0);
        expect(bounded[3].MinLOD == 1.5f && bounded[3].MaxLOD == 0.0f, "smpl MIN_LOD 1.5 and MAX_LOD 0",
               int(bounded[3].MinLOD * 256), int(bounded[3].MaxLOD * 256));
        expect(bounded[0].Filter == bounded[1].Filter && bounded[0].AddressU == bounded[1].AddressU,
               "smpl decodes the other fields like smp", bounded[0].Filter, bounded[1].Filter);
    }
    std::printf(failures ? "startup_sampler_comparison failed\n" : "startup_sampler_comparison ok\n");
    return failures ? 1 : 0;
}
