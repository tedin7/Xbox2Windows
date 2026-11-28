// Original corpus case: packed 16-bit fields in a compute stage.
RWStructuredBuffer<uint> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint packed = (id.x & 0xffffu) | ((id.y & 0xffffu) << 16);
    uint low = packed & 0xffffu;
    uint high = packed >> 16;
    Output[id.x] = low + high + WaveActiveCountBits(low != 0u);
}
