#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../src/startup_stubs/pixel_history.h"
#include "desktop_gpu.h"

using Microsoft::WRL::ComPtr;

namespace {
constexpr DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr UINT width = 4, height = 4;
constexpr float black[4]{0, 0, 0, 0}, red[4]{1, 0, 0, 1}, green[4]{0, 1, 0, 1}, blue[4]{0, 0, 1, 1};

ComPtr<ID3DBlob> compile(const char* source, const char* entry, const char* target)
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(source, std::strlen(source), nullptr, nullptr, nullptr, entry, target,
                                  D3DCOMPILE_ENABLE_STRICTNESS, 0, &code, &errors);
    if (FAILED(hr)) {
        if (errors) std::fprintf(stderr, "shader %s: %s\n", entry, static_cast<const char*>(errors->GetBufferPointer()));
        check(hr, "compile shader");
    }
    return code;
}

ComPtr<ID3D12RootSignature> make_root(ID3D12Device* device)
{
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants.ShaderRegister = 0;
    parameter.Constants.Num32BitValues = 4;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 1;
    desc.pParameters = &parameter;
    ComPtr<ID3DBlob> blob, errors;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "serialize root signature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create root signature");
    return root;
}

ComPtr<ID3D12PipelineState> make_pipeline(ID3D12Device* device, ID3D12RootSignature* root)
{
    const char* vertex = R"(
struct VsOut { float4 position : SV_Position; };
VsOut main(uint id : SV_VertexID) {
    float2 corner = float2((id << 1) & 2, id & 2);
    VsOut vertex; vertex.position = float4(corner * 2.0 - 1.0, 0, 1); return vertex;
})";
    const char* pixel = R"(
cbuffer Color : register(b0) { float4 color; };
float4 main() : SV_Target { return color; }
)";
    const ComPtr<ID3DBlob> vs = compile(vertex, "main", "vs_5_1");
    const ComPtr<ID3DBlob> ps = compile(pixel, "main", "ps_5_1");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = format;
    desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)), "create pipeline");
    return pipeline;
}

struct Target {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Resource> texture, readback;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
};

Target make_target(IDXGIAdapter* adapter)
{
    Target target;
    check(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&target.device)), "create device");
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0},
                             D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{format, {0, 0, 0, 0}};
    check(target.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                 &clear, IID_PPV_ARGS(&target.texture)), "create target");
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    check(target.device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&target.rtv_heap)), "create rtv heap");
    target.rtv = target.rtv_heap->GetCPUDescriptorHandleForHeapStart();
    target.device->CreateRenderTargetView(target.texture.Get(), nullptr, target.rtv);
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC readback{D3D12_RESOURCE_DIMENSION_BUFFER, 0, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, 1, 1, 1,
                                 DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    check(target.device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&target.readback)),
          "create readback");
    return target;
}

// A fresh target keeps the control independent of the first history: the clear value is the resource's
// initial content, so no extra clear command is needed.
void renew_texture(Target& target)
{
    target.texture.Reset();
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0},
                             D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{format, {0, 0, 0, 0}};
    check(target.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                 &clear, IID_PPV_ARGS(&target.texture)), "renew target");
    target.device->CreateRenderTargetView(target.texture.Get(), nullptr, target.rtv);
}

struct Recorded {
    int draw{};
    std::uint64_t key{};
    x2w_startup::PixelHistorySample sample{};
};

// One full-screen triangle on a scissored area. After it completes, the chosen texel is copied and
// decoded exactly as the runtime history does.
void record_draw(Target& target, ComputeCommands& commands, ID3D12RootSignature* root, ID3D12PipelineState* pipeline,
                 const float* color, D3D12_RECT scissor, int draw, std::uint64_t key, Recorded& recorded)
{
    check(commands.list->Reset(commands.allocator.Get(), pipeline), "reset list");
    commands.list->SetGraphicsRootSignature(root);
    commands.list->SetPipelineState(pipeline);
    commands.list->SetGraphicsRoot32BitConstants(0, 4, color, 0);
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    commands.list->RSSetViewports(1, &viewport);
    commands.list->RSSetScissorRects(1, &scissor);
    commands.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commands.list->OMSetRenderTargets(1, &target.rtv, FALSE, nullptr);
    commands.list->DrawInstanced(3, 1, 0, 0);
    transition(commands.list.Get(), target.texture.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION to{target.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint.Footprint = {format, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
    const D3D12_TEXTURE_COPY_LOCATION from{target.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    const D3D12_BOX box{1, 1, 0, 2, 2, 1};
    commands.list->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
    transition(commands.list.Get(), target.texture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_RENDER_TARGET);
    commands.submit_and_wait(target.device.Get());
    void* mapped = nullptr;
    const D3D12_RANGE read{0, 4};
    check(target.readback->Map(0, &read, &mapped), "map readback");
    recorded.draw = draw;
    recorded.key = key;
    recorded.sample.draw = draw;
    recorded.sample.frame = 7;
    recorded.sample.x = 1;
    recorded.sample.y = 1;
    recorded.sample.key = key;
    recorded.sample.count = 3;
    recorded.sample.format = format;
    if (!x2w_startup::pixel_history_decode(recorded.sample, mapped)) std::fprintf(stderr, "decode failed\n");
    const D3D12_RANGE no_writes{};
    target.readback->Unmap(0, &no_writes);
}

bool has_rgba(const std::string& line, unsigned r, unsigned g, unsigned b, unsigned a)
{
    char wanted[64];
    std::snprintf(wanted, sizeof(wanted), "rgba=%u,%u,%u,%u", r, g, b, a);
    return line.find(wanted) != std::string::npos;
}

std::string line_for(const std::vector<std::string>& lines, int draw)
{
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "draw=%d ", draw);
    for (const auto& line : lines) if (line.rfind(prefix, 0) == 0) return line;
    return {};
}
}  // namespace

int main(int argc, char** argv)
{
    const bool hardware = argc >= 2 && std::strcmp(argv[1], "hardware") == 0;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    if (hardware) {
        ComPtr<IDXGIFactory6> factory6;
        if (FAILED(factory.As(&factory6))) return 3;
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT found = factory6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                                       IID_PPV_ARGS(&candidate));
            if (found == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(found)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 77;
    }
    DXGI_ADAPTER_DESC description{};
    if (FAILED(adapter->GetDesc(&description))) return 3;
    std::printf("adapter=%ls vendor=0x%x device=0x%x\n", description.Description, description.VendorId, description.DeviceId);
    std::fflush(stdout);

    Target target = make_target(adapter.Get());
    const auto root = make_root(target.device.Get());
    const auto pipeline = make_pipeline(target.device.Get(), root.Get());
    ComputeCommands commands(target.device.Get(), pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    check(commands.list->Close(), "close initial list");

    // Three source-built draws: the first and the third only cover the top-left texel, so the second
    // is the only one that changes the watched texel (1, 1).
    const D3D12_RECT corner{0, 0, 1, 1}, all{0, 0, width, height};
    std::vector<Recorded> recorded(3);
    record_draw(target, commands, root.Get(), pipeline.Get(), red, corner, 1, 0x1111111111111111ull, recorded[0]);
    record_draw(target, commands, root.Get(), pipeline.Get(), green, all, 2, 0x2222222222222222ull, recorded[1]);
    record_draw(target, commands, root.Get(), pipeline.Get(), blue, corner, 3, 0x3333333333333333ull, recorded[2]);

    std::vector<x2w_startup::PixelHistorySample> samples;
    for (const auto& item : recorded) samples.push_back(item.sample);
    const auto room = std::filesystem::temp_directory_path() / ("x2w-pixel-history-" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(room);
    const std::string path = (room / "pixel_history.txt").string();
    if (!x2w_startup::pixel_history_write(room.string().c_str(), samples.data(), samples.size())) return 4;
    std::ifstream input(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) lines.push_back(line);
    const auto first = line_for(lines, 1), second = line_for(lines, 2), third = line_for(lines, 3);
    std::printf("draw 1: %s\ndraw 2: %s\ndraw 3: %s\n", first.c_str(), second.c_str(), third.c_str());
    std::fflush(stdout);
    const bool first_black = has_rgba(first, 0, 0, 0, 0);
    const bool second_green = has_rgba(second, 0, 255, 0, 255);
    const bool third_green = has_rgba(third, 0, 255, 0, 255);
    const bool second_key = second.find("key=2222222222222222") != std::string::npos && second.find("frame=7") != std::string::npos &&
                            second.find("x=1 y=1") != std::string::npos;
    bool ok = first_black && second_green && third_green && second_key;

    // Negative control: without the second draw the watched texel stays black at draw 3.
    renew_texture(target);
    std::vector<Recorded> control(2);
    record_draw(target, commands, root.Get(), pipeline.Get(), red, corner, 1, 0x1111111111111111ull, control[0]);
    record_draw(target, commands, root.Get(), pipeline.Get(), blue, corner, 3, 0x3333333333333333ull, control[1]);
    std::vector<x2w_startup::PixelHistorySample> control_samples;
    for (const auto& item : control) control_samples.push_back(item.sample);
    if (!x2w_startup::pixel_history_write(room.string().c_str(), control_samples.data(), control_samples.size())) return 4;
    std::ifstream control_input(path);
    std::vector<std::string> control_lines;
    for (std::string line; std::getline(control_input, line);) control_lines.push_back(line);
    const auto control_third = line_for(control_lines, 3);
    std::printf("control without draw 2: %s\n", control_third.c_str());
    const bool control_has_no_second = line_for(control_lines, 2).empty();
    const bool control_black = has_rgba(control_third, 0, 0, 0, 0);
    ok = ok && control_has_no_second && control_black;
    std::printf("checks: first_black=%d second_green=%d third_green=%d second_key=%d no_second=%d control_black=%d\n",
                first_black, second_green, third_green, second_key, control_has_no_second, control_black);
    std::fflush(stdout);
    input.close();
    control_input.close();
    std::puts(ok ? "PASS: the history names draw 2; without it the texel stays black"
                 : "FAIL: pixel history report mismatch");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
