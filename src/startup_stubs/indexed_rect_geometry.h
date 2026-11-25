#pragma once

#include <d3d12shader.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>
#include <string>

namespace x2w_startup {

// Only the interfaces emitted by the startup graphics translator are supported.
// The geometry stage consumes no root parameters and leaves the VS inputs unchanged.
inline HRESULT compile_indexed_rect_geometry(ID3DBlob* vs, ID3DBlob* ps, ID3DBlob** geometry, ID3DBlob** errors)
{
    using Microsoft::WRL::ComPtr;
    ComPtr<ID3D12ShaderReflection> vertex, pixel;
    if (!vs || !ps ||
        FAILED(D3DReflect(vs->GetBufferPointer(), vs->GetBufferSize(), IID_PPV_ARGS(&vertex))) ||
        FAILED(D3DReflect(ps->GetBufferPointer(), ps->GetBufferSize(), IID_PPV_ARGS(&pixel)))) return E_INVALIDARG;
    D3D12_SHADER_DESC out{}, in{};
    if (FAILED(vertex->GetDesc(&out)) || FAILED(pixel->GetDesc(&in)) ||
        !out.OutputParameters || out.OutputParameters > 32) return E_INVALIDARG;
    std::string source = "struct Vertex {\n";
    std::string flat, interpolated;
    bool position = false;
    UINT params = 0, flat_params = 0;
    for (UINT i = 0; i < out.OutputParameters; ++i) {
        D3D12_SIGNATURE_PARAMETER_DESC param{};
        if (FAILED(vertex->GetOutputParameterDesc(i, &param)) ||
            !param.SemanticName || param.Mask != 15 || param.Stream) return E_INVALIDARG;
        if (param.SystemValueType == D3D_NAME_POSITION && param.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 &&
            param.SemanticIndex == 0 && !position) {
            position = true;
            source += "float4 pos : SV_Position;\n";
            continue;
        }
        if (param.SystemValueType != D3D_NAME_UNDEFINED || param.SemanticIndex >= 32) return E_INVALIDARG;
        const bool smooth = _stricmp(param.SemanticName, "PARAM") == 0 &&
            param.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32;
        const bool constant = _stricmp(param.SemanticName, "FPARAM") == 0 &&
            param.ComponentType == D3D_REGISTER_COMPONENT_UINT32;
        if (!smooth && !constant) return E_INVALIDARG;
        auto& seen = smooth ? params : flat_params;
        if (seen & (1u << param.SemanticIndex)) return E_INVALIDARG;
        seen |= 1u << param.SemanticIndex;
        const std::string semantic = (smooth ? "PARAM" : "FPARAM") + std::to_string(param.SemanticIndex);
        source += (smooth ? "float4 " : "nointerpolation uint4 ") + semantic + " : " + semantic + ";\n";
        if (smooth) {
            interpolated += "fourth." + semantic + " = v[a]." + semantic + " + v[b]." +
                semantic + " - v[corner]." + semantic + ";\n";
        } else {
            flat += "v[1]." + semantic + " = v[2]." + semantic + " = v[0]." + semantic + ";\n";
        }
    }
    if (!position) return E_INVALIDARG;
    for (UINT i = 0; i < in.InputParameters; ++i) {
        D3D12_SIGNATURE_PARAMETER_DESC param{};
        if (FAILED(pixel->GetInputParameterDesc(i, &param)) || !param.SemanticName ||
            param.SemanticIndex >= 32) return E_INVALIDARG;
        if (param.SystemValueType == D3D_NAME_POSITION && param.SemanticIndex == 0) continue;
        if (param.SystemValueType != D3D_NAME_UNDEFINED) return E_INVALIDARG;
        const UINT semantic = 1u << param.SemanticIndex;
        if (_stricmp(param.SemanticName, "PARAM") == 0 &&
            param.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 && (params & semantic)) continue;
        if (_stricmp(param.SemanticName, "FPARAM") == 0 &&
            param.ComponentType == D3D_REGISTER_COMPONENT_UINT32 && (flat_params & semantic)) continue;
        return E_INVALIDARG;
    }
    source += R"(
};
bool corner_between(float2 p, float2 a, float2 b) {
    return (p.x == a.x && p.y == b.y && p.y != a.y && p.x != b.x) ||
           (p.x == b.x && p.y == a.y && p.y != b.y && p.x != a.x);
}
[maxvertexcount(6)]
void main(triangle Vertex input[3], inout TriangleStream<Vertex> stream) {
    Vertex v[3] = input;
    if (!all(isfinite(v[0].pos)) || !all(isfinite(v[1].pos)) || !all(isfinite(v[2].pos)) ||
        !(v[0].pos.w > 0 && v[0].pos.w == v[1].pos.w && v[0].pos.w == v[2].pos.w)) return;
    uint corner = 3;
    [unroll] for (uint i = 0; i < 3; ++i)
        if (corner_between(v[i].pos.xy, v[(i + 1) % 3].pos.xy, v[(i + 2) % 3].pos.xy)) corner = i;
    if (corner == 3) return;
    uint a = (corner + 1) % 3, b = (corner + 2) % 3;
)";
    source += flat;
    source += R"(
    Vertex fourth = v[0];
    fourth.pos = v[a].pos + v[b].pos - v[corner].pos;
    if (!all(isfinite(fourth.pos))) return;
)";
    source += interpolated;
    source += R"(
    stream.Append(v[0]);
    stream.Append(v[1]);
    stream.Append(v[2]);
    stream.RestartStrip();
    stream.Append(v[a]);
    stream.Append(fourth);
    stream.Append(v[b]);
    stream.RestartStrip();
}
)";
    return D3DCompile(source.data(), source.size(), "indexed rectangle geometry", nullptr, nullptr, "main", "gs_5_1",
                      D3DCOMPILE_IEEE_STRICTNESS, 0, geometry, errors);
}

}  // namespace x2w_startup
