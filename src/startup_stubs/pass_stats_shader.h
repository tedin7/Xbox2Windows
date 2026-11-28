// Pass thermometer reduction (docs/startup-runtime.md#pass-thermometer). One dispatch reads one snapshot: `count`
// texels (a texture's subresource 0, row-major over `width`) or 32-bit words (a buffer range, KIND 3). Each of the
// groups * 64 threads strides over the texels and writes its own record: per channel, min, max and sum of the finite
// values (+Inf, -Inf and 0 when it saw none) and the counts of exact zeros, NaN and Inf, 24 words at word `out_word` +
// thread * 24 of the result buffer. The CPU folds the records (pass_stats_math.h combine). Plain shader model 5: no
// group-shared memory, atomics, attributes or structured buffers, so Wine's builtin compiler accepts it too.
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

#define ACCUMULATE(c) { \
    uint magnitude = asuint(v.c) & 0x7fffffff; \
    if (magnitude > 0x7f800000) { nans.c = nans.c + 1; } \
    else if (magnitude == 0x7f800000) { infs.c = infs.c + 1; } \
    else { \
        if (magnitude == 0) { zeros.c = zeros.c + 1; } \
        lo.c = min(lo.c, v.c); \
        hi.c = max(hi.c, v.c); \
        total.c = total.c + v.c; \
    } }

#define STORE(c, at) { \
    partials.Store((at) * 4, asuint(lo.c)); \
    partials.Store((at + 1) * 4, asuint(hi.c)); \
    partials.Store((at + 2) * 4, asuint(total.c)); \
    partials.Store((at + 3) * 4, zeros.c); \
    partials.Store((at + 4) * 4, nans.c); \
    partials.Store((at + 5) * 4, infs.c); }

[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    float4 lo = asfloat(0x7f800000);
    float4 hi = asfloat(0xff800000);
    float4 total = float4(0, 0, 0, 0);
    uint4 zeros = uint4(0, 0, 0, 0);
    uint4 nans = uint4(0, 0, 0, 0);
    uint4 infs = uint4(0, 0, 0, 0);
    uint record = group.x * 64 + thread.x;
    for (uint i = record; i < count; i += groups * 64) {
        float4 v = texel(i);
        ACCUMULATE(x)
        ACCUMULATE(y)
        ACCUMULATE(z)
        ACCUMULATE(w)
    }
    uint at = out_word + record * 24;
    STORE(x, at)
    STORE(y, at + 6)
    STORE(z, at + 12)
    STORE(w, at + 18)
}
)hlsl";

}  // namespace x2w_pass_stats
