// Pass thermometer reduction (docs/startup-runtime.md#pass-thermometer). One dispatch reads one snapshot: `count`
// texels (a texture's subresource 0, row-major over `width`) or 32-bit words (a buffer range, KIND 3). Each group of
// 256 threads strides over the texels and writes, per channel, min, max and sum of the finite values and the counts
// of exact zeros, NaN and Inf: 24 dwords at dword `out_word` + group * 24 of the result buffer. The CPU folds the
// groups (pass_stats_math.h combine). KIND: 0 float4, 1 uint4, 2 sint4, 3 raw words read as floats.
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
groupshared float shared_min[256];
groupshared float shared_max[256];
groupshared float shared_sum[256];
groupshared uint shared_zero[256];
groupshared uint shared_nan[256];
groupshared uint shared_inf[256];

float4 texel(uint i)
{
#if KIND == 3
    return float4(asfloat(source.Load(i * 4)), 0, 0, 0);
#else
    return float4(source.Load(int3(i % width, i / width, 0)));
#endif
}

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    float lo[4], hi[4], total[4];
    uint zeros[4], nans[4], infs[4];
    [unroll] for (uint c = 0; c < 4; ++c) {
        lo[c] = asfloat(0x7f800000u);
        hi[c] = asfloat(0xff800000u);
        total[c] = 0;
        zeros[c] = 0;
        nans[c] = 0;
        infs[c] = 0;
    }
    for (uint i = group.x * 256 + lane; i < count; i += groups * 256) {
        const float4 v = texel(i);
        [unroll] for (uint c = 0; c < 4; ++c) {
            const uint magnitude = asuint(v[c]) & 0x7fffffffu;
            if (magnitude > 0x7f800000u) {
                nans[c] += 1;
            } else if (magnitude == 0x7f800000u) {
                infs[c] += 1;
            } else {
                zeros[c] += magnitude == 0 ? 1 : 0;
                lo[c] = min(lo[c], v[c]);
                hi[c] = max(hi[c], v[c]);
                total[c] += v[c];
            }
        }
    }
    [unroll] for (uint channel = 0; channel < 4; ++channel) {
        shared_min[lane] = lo[channel];
        shared_max[lane] = hi[channel];
        shared_sum[lane] = total[channel];
        shared_zero[lane] = zeros[channel];
        shared_nan[lane] = nans[channel];
        shared_inf[lane] = infs[channel];
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128; stride > 0; stride /= 2) {
            if (lane < stride) {
                shared_min[lane] = min(shared_min[lane], shared_min[lane + stride]);
                shared_max[lane] = max(shared_max[lane], shared_max[lane + stride]);
                shared_sum[lane] += shared_sum[lane + stride];
                shared_zero[lane] += shared_zero[lane + stride];
                shared_nan[lane] += shared_nan[lane + stride];
                shared_inf[lane] += shared_inf[lane + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) {
            const uint at = (out_word + (group.x * 4 + channel) * 6) * 4;
            partials.Store3(at, uint3(asuint(shared_min[0]), asuint(shared_max[0]), asuint(shared_sum[0])));
            partials.Store3(at + 12, uint3(shared_zero[0], shared_nan[0], shared_inf[0]));
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
)hlsl";

}  // namespace x2w_pass_stats
