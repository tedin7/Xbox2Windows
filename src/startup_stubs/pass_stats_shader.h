// Pass thermometer reduction (docs/startup-runtime.md#pass-thermometer). One dispatch reads one snapshot: `count`
// texels (a texture's subresource 0, row-major over `width`) or 32-bit words (a buffer range, KIND 3). Each thread
// strides over the texels and keeps, per channel, min, max and sum of the finite values and the counts of exact zeros,
// NaN and Inf; then it folds them with buffer atomics into its group's record: 24 dwords at dword `out_word` +
// group * 24 of the result buffer, which starts zeroed. Min and max are stored as order-preserving words (0: no
// finite value; min as the complement of its word), the sum by compare-exchange. No group-shared memory: every D3D12
// shader compiler the stand-in meets accepts this. The CPU folds the groups (pass_stats_math.h combine).
// KIND: 0 float4, 1 uint4, 2 sint4, 3 raw words read as floats.
#pragma once

namespace x2w_pass_stats {

inline constexpr char reduction_hlsl[] = R"hlsl(
cbuffer Params : register(b0) { uint width; uint height; uint count; uint groups; uint out_word; uint3 unused; };
#if KIND == 3
ByteAddressBuffer source : register(t0);
#elif KIND == 2
Texture2D<int4> source : register(t0);
#elif KIND == 1
Texture2D<uint4> source : register(t0);
#else
Texture2D<float4> source : register(t0);
#endif
RWByteAddressBuffer partials : register(u0);

float4 texel(uint i)
{
#if KIND == 3
    return float4(asfloat(source.Load(i * 4)), 0, 0, 0);
#else
    return float4(source.Load(int3(i % width, i / width, 0)));
#endif
}

// A word whose unsigned order is the float order; never 0 for a finite value.
uint ordered(float value)
{
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

void add_float(uint at, float value)
{
    if (value == 0) return;
    uint expected = partials.Load(at);
    [allow_uav_condition] for (uint tries = 0; tries < 65536; ++tries) {
        uint original;
        partials.InterlockedCompareExchange(at, expected, asuint(asfloat(expected) + value), original);
        if (original == expected) break;
        expected = original;
    }
}

void fold(uint at, bool finite, float lo, float hi, float total, uint zeros, uint nans, uint infs)
{
    if (finite) {
        partials.InterlockedMax(at, ~ordered(lo));
        partials.InterlockedMax(at + 4, ordered(hi));
        add_float(at + 8, total);
    }
    if (zeros) partials.InterlockedAdd(at + 12, zeros);
    if (nans) partials.InterlockedAdd(at + 16, nans);
    if (infs) partials.InterlockedAdd(at + 20, infs);
}

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    float4 lo = asfloat(0x7f800000u), hi = asfloat(0xff800000u), total = 0;
    uint4 zeros = 0, nans = 0, infs = 0;
    for (uint i = group.x * 256 + lane; i < count; i += groups * 256) {
        const float4 v = texel(i);
        const uint4 magnitude = asuint(v) & 0x7fffffffu;
        const bool4 is_nan = magnitude > 0x7f800000u, is_inf = magnitude == 0x7f800000u;
        const bool4 finite = !is_nan && !is_inf;
        nans += is_nan ? 1 : 0;
        infs += is_inf ? 1 : 0;
        zeros += finite && magnitude == 0 ? 1 : 0;
        lo = finite ? min(lo, v) : lo;
        hi = finite ? max(hi, v) : hi;
        total += finite ? v : 0;
    }
    const uint at = (out_word + group.x * 24) * 4;
    const bool4 seen = lo <= hi;
    fold(at, seen.x, lo.x, hi.x, total.x, zeros.x, nans.x, infs.x);
    fold(at + 24, seen.y, lo.y, hi.y, total.y, zeros.y, nans.y, infs.y);
    fold(at + 48, seen.z, lo.z, hi.z, total.z, zeros.z, nans.z, infs.z);
    fold(at + 72, seen.w, lo.w, hi.w, total.w, zeros.w, nans.w, infs.w);
}
)hlsl";

}  // namespace x2w_pass_stats
