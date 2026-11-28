// Original corpus case: implicit derivatives and fwidth in a pixel stage.
float4 main(float2 uv : TEXCOORD0) : SV_Target
{
    float2 dx = ddx(uv);
    float2 dy = ddy(uv);
    float width = fwidth(uv.x + uv.y);
    return float4(dx, dy.x + dy.y, width);
}
