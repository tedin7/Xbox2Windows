cbuffer Entry : register(b0) {
    uint2 gates;
};

struct PixelInput {
    float4 position : SV_Position;
    nointerpolation uint side : SIDE;
};

struct PixelOutput {
    float4 first : SV_Target0;
    float4 second : SV_Target1;
    float4 third : SV_Target2;
};

PixelOutput main(PixelInput input) {
    if (any(gates != uint2(17, 29)) || input.side != 0 || input.position.x >= 2)
        discard;
    PixelOutput output;
    output.first = output.second = output.third = float4(1, 0, 0, 1);
    return output;
}
