// The shared rectangle-expansion rule of src/startup_stubs/indexed_rect_geometry.h on a desktop device (#3131).
//
// The translated indexed rectangle draw binds this geometry stage, and the runtime compiles it from the
// vertex and pixel bytecode of the same translation. This fixture compiles the same rule and draws one
// rectangle that a rectangle list names with three corners, with back-face culling on, so the coverage it
// emits is measured directly:
//   axis        bottom-left, top-left, top-right: the corner test always finds a corner that shares an axis
//               with each neighbour, and the fourth corner is v0 + v2 - v1
//   swung       bottom-left, top-right, top-left: the same three corners in another order, which the stage
//               normalises to the same square
//   rotated     the square turned by 15 degrees: still a parallelogram whose fourth corner is v0 + v2 - v1 after
//               the right-angle corner is identified, and the corner the non-indexed path derives in the vertex
//               stage. No corner shares an axis with its neighbours, so the axis test cannot identify it.
// Each vertex record is 12 bytes (a mask at +0, XY at +4) and both orders must cover the pixels the two
// triangles of the literal rule cover. Without an adapter argument the fixture uses WARP. `--native-gs`
// replaces the HLSL stage with a geometry CSO emitted from src/gfx1013_rect_geometry.cpp.
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "startup_stubs/indexed_rect_geometry.h"

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT width = 8, height = 4;
constexpr DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
// R8G8B8A8 words: the low byte is red, then green and blue.
constexpr std::uint32_t red = 0xff0000ff, blue = 0xffff0000;

struct Vertex {
    float mask, x, y;
};
static_assert(sizeof(Vertex) == 12);

// The three corners of a rectangle, clockwise on screen (bottom-left, top-left, top-right).
constexpr std::array<Vertex, 3> corners{{
    {1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}};

int failures;
void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

HRESULT compile(const char* source, const char* profile, ComPtr<ID3DBlob>& blob)
{
    ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompile(source, std::strlen(source), "rect_geometry_fixture", nullptr, nullptr, "main", profile,
                                  0, 0, &blob, &errors);
    if (FAILED(hr) && errors) std::printf("compile %s: %s\n", profile, static_cast<const char*>(errors->GetBufferPointer()));
    return hr;
}

bool inside(const std::array<std::array<float, 2>, 3>& triangle, float x, float y, float& margin)
{
    float edges[3];
    for (int i = 0; i < 3; ++i) {
        const auto& a = triangle[i];
        const auto& b = triangle[(i + 1) % 3];
        edges[i] = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    }
    margin = (std::min)({std::fabs(edges[0]), std::fabs(edges[1]), std::fabs(edges[2])});
    const bool negative = edges[0] < 0 || edges[1] < 0 || edges[2] < 0;
    const bool positive = edges[0] > 0 || edges[1] > 0 || edges[2] > 0;
    return !(negative && positive);
}

// The pixels the rectangle covers: (v0, v1, v2) and (v0, v2, v3) of its four corners. For the axis order this
// is the literal rule (v3 = v0 + v2 - v1); a swung order names the same four corners, which is the order the
// existing indirect fixture pins for the geometry stage.
std::vector<std::uint32_t> reference(const std::array<std::pair<float, float>, 4>& quad)
{
    const auto at = [&](int i, int axis) { return axis ? quad[i].second : quad[i].first; };
    const std::array<std::array<float, 2>, 3> first{at(0, 0), at(0, 1), at(1, 0), at(1, 1), at(2, 0), at(2, 1)};
    const std::array<std::array<float, 2>, 3> second{at(0, 0), at(0, 1), at(2, 0), at(2, 1), at(3, 0), at(3, 1)};
    std::vector<std::uint32_t> pixels(width * height, blue);  // the clear colour
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const float centre_x = (2.0f * x + 1.0f) / width - 1.0f;
            const float centre_y = 1.0f - (2.0f * y + 1.0f) / height;
            float margin = 0;
            const bool covered = inside(first, centre_x, centre_y, margin) || inside(second, centre_x, centre_y, margin);
            expect(margin > 1e-3f, "a pixel centre lies on a rectangle edge");
            pixels[y * width + x] = covered ? red : blue;
        }
    }
    return pixels;
}
}

int main(int argc, char** argv)
{
    bool hardware = false;
    const char* native_vs = nullptr;
    const char* native_ps = nullptr;
    const char* native_gs = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "hardware") == 0) {
            hardware = true;
        } else if (std::strcmp(argv[i], "--native-gs") == 0 && i + 1 < argc) {
            native_gs = argv[++i];
        } else if (std::strcmp(argv[i], "--native-vs") == 0 && i + 1 < argc) {
            native_vs = argv[++i];
        } else if (std::strcmp(argv[i], "--native-ps") == 0 && i + 1 < argc) {
            native_ps = argv[++i];
        } else {
            std::printf("usage: startup_rect_geometry [hardware] [--native-gs FILE]\n");
            return 2;
        }
    }
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 4;
    if (hardware) {
        ComPtr<IDXGIFactory6> factory6;
        if (FAILED(factory.As(&factory6))) return 77;
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory6->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 4;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 4;
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 4;
    }
    ComPtr<ID3D12Device> device;
    {
        ComPtr<ID3D12Debug> debug;
        const HRESULT hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
        if (SUCCEEDED(hr)) debug->EnableDebugLayer();
        std::printf("debug_layer=%s\n", SUCCEEDED(hr) ? "on" : "unavailable");
    }
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 5;
    DXGI_ADAPTER_DESC desc{};
    LARGE_INTEGER driver{};
    adapter->GetDesc(&desc);
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
    std::printf("adapter=%ls driver=%u.%u.%u.%u\n", desc.Description, HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                HIWORD(driver.LowPart), LOWORD(driver.LowPart));

    // The vertex stage exports the 12-byte record's XY as the position, which is what the geometry stage reads
    // back through its signature. The mask at +0 is not named by the layout: this fixture measures coverage.
    static const char vs_text[] = R"(
struct In { float2 xy : POSITION; };
float4 main(In input) : SV_Position { return float4(input.xy, 0, 1); })";
    static const char ps_text[] = R"(
float4 main(float4 pos : SV_Position) : SV_Target { return float4(1, 0, 0, 1); })";
    ComPtr<ID3DBlob> vs, ps, gs, gs_errors;
    const auto load_blob = [](const char* path, ComPtr<ID3DBlob>& blob) {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        const std::string bytes{std::istreambuf_iterator<char>(file), {}};
        if (FAILED(D3DCreateBlob(bytes.size(), &blob))) return false;
        std::memcpy(blob->GetBufferPointer(), bytes.data(), bytes.size());
        return true;
    };
    const bool native = native_vs || native_ps || native_gs;
    if (native && !(native_vs && native_ps && native_gs)) {
        std::printf("--native-vs, --native-ps and --native-gs must be used together\n");
        return 2;
    }
    if (native) {
        if (!load_blob(native_vs, vs) || !load_blob(native_ps, ps) || !load_blob(native_gs, gs)) {
            std::printf("missing native stage\n");
            return 7;
        }
        std::printf("geometry: native %zu bytes\n", static_cast<std::size_t>(gs->GetBufferSize()));
    } else if (FAILED(compile(vs_text, "vs_5_1", vs)) || FAILED(compile(ps_text, "ps_5_1", ps))) {
        return 6;
    }
    if (!native && FAILED(x2w_startup::compile_indexed_rect_geometry(vs.Get(), ps.Get(), &gs, &gs_errors))) {
        if (gs_errors) std::printf("geometry: %s\n", static_cast<const char*>(gs_errors->GetBufferPointer()));
        return 7;
    }

    const D3D12_INPUT_ELEMENT_DESC elements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 4, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    // A pipeline state that sets an input layout needs this flag on its root signature.
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> root_blob;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &root_blob, nullptr))) return 8;
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                           IID_PPV_ARGS(&root)))) return 8;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.Get();
    pso_desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pso_desc.GS = {gs->GetBufferPointer(), gs->GetBufferSize()};
    pso_desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pso_desc.InputLayout = {elements, 1};
    pso_desc.BlendState.RenderTarget[0] = {TRUE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
                                           D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
                                           D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    pso_desc.RasterizerState.FrontCounterClockwise = FALSE;  // clockwise on screen is front, as in the source
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    // A zero-filled depth-stencil description holds invalid enum values (stencil op 0), so the fields a
    // depth-less pipeline still needs are spelled out.
    pso_desc.DepthStencilState.DepthEnable = FALSE;
    pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pso_desc.DepthStencilState.StencilEnable = FALSE;
    pso_desc.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    pso_desc.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    pso_desc.DepthStencilState.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                            D3D12_COMPARISON_FUNC_ALWAYS};
    pso_desc.DepthStencilState.BackFace = pso_desc.DepthStencilState.FrontFace;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = format;
    pso_desc.SampleDesc.Count = 1;
    pso_desc.SampleMask = UINT_MAX;
    ComPtr<ID3D12PipelineState> pso;
    const HRESULT pso_hr = device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso));
    if (FAILED(pso_hr)) {
        std::printf("CreateGraphicsPipelineState: 0x%08lx\n", static_cast<unsigned long>(pso_hr));
        ComPtr<ID3D12InfoQueue> queue;
        if (SUCCEEDED(device.As(&queue))) {
            const UINT64 count = queue->GetNumStoredMessages();
            for (UINT64 i = 0; i < count; ++i) {
                SIZE_T size = 0;
                queue->GetMessage(i, nullptr, &size);
                std::vector<char> message(size);
                auto* entry = reinterpret_cast<D3D12_MESSAGE*>(message.data());
                if (SUCCEEDED(queue->GetMessage(i, entry, &size)))
                    std::printf("debug layer: %s\n", entry->pDescription);
            }
        }
        return 9;
    }

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format,
                                          {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const float clear[4] = {0, 0, 1, 1};
    const D3D12_CLEAR_VALUE optimized{format, {clear[0], clear[1], clear[2], clear[3]}};
    ComPtr<ID3D12Resource> target;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &target_desc, D3D12_RESOURCE_STATE_COMMON,
                                               &optimized, IID_PPV_ARGS(&target)))) return 10;
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    if (FAILED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)))) return 10;
    device->CreateRenderTargetView(target.Get(), nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());

    struct Scene {
        const char* name;
        std::array<Vertex, 6> records;
        // The four corners of the rectangle the three records name, clockwise on screen from the bottom left.
        std::array<std::pair<float, float>, 4> quad;
        float angle;
        // False when the stage must draw nothing: the swung order is rejected by back-face culling, the
        // skewed triangle by the rectangle rule's right-angle guard.
        bool covered;
    };
    const auto turned = [](std::array<Vertex, 3> rectangle, float degrees) {
        const float c = std::cos(degrees * 3.14159265358979324f / 180.0f);
        const float s = std::sin(degrees * 3.14159265358979324f / 180.0f);
        for (Vertex& v : rectangle) {
            const float x = v.x, y = v.y;
            v.x = c * x - s * y;
            v.y = s * x + c * y;
        }
        return rectangle;
    };
    const std::array<std::pair<float, float>, 4> square{{
        {-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, -1.0f}}};
    std::array<std::pair<float, float>, 4> turned_square = square;
    {
        const float c = std::cos(15.0f * 3.14159265358979324f / 180.0f);
        const float s = std::sin(15.0f * 3.14159265358979324f / 180.0f);
        for (auto& p : turned_square) p = {c * p.first - s * p.second, s * p.first + c * p.second};
    }
    std::array<Vertex, 3> skewed = corners;
    skewed[1].x = -0.5f;  // the top-left corner moved a quarter of the way to the top-right
    std::array<Scene, 4> scenes{{
        {"axis", {}, square, 0.0f, true},
        {"swung", {}, square, 0.0f, false},
        {"rotated", {}, turned_square, 15.0f, true},
        {"skewed", {}, square, 0.0f, false},
    }};
    // The three corners a rectangle list names per rectangle: v0, v1 and v2. The fourth is derived. Both
    // records of the six-vertex dispatch name the same rectangle.
    const std::array<std::array<Vertex, 3>, 4> records{{
        {corners[0], corners[1], corners[2]},                 // bottom left, top left, top right
        {corners[0], corners[2], corners[1]},                 // bottom left, top right, top left
        {turned(corners, 15.0f)[0], turned(corners, 15.0f)[1], turned(corners, 15.0f)[2]},
        {skewed[0], skewed[1], skewed[2]},                    // no axis corner and no right angle
    }};
    for (auto& scene : scenes) {
        const auto& three = records[&scene - scenes.data()];
        for (int i = 0; i < 3; ++i) scene.records[i] = three[i];
        for (int i = 0; i < 3; ++i) scene.records[3 + i] = three[i];
    }

    D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const auto make_buffer = [&](UINT64 bytes, D3D12_RESOURCE_STATES state, ID3D12Resource** out) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        return device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                              IID_PPV_ARGS(out));
    };
    ComPtr<ID3D12Resource> vertices;
    void* mapped = nullptr;
    if (FAILED(make_buffer(sizeof(Scene::records), D3D12_RESOURCE_STATE_GENERIC_READ, &vertices)) ||
        FAILED(vertices->Map(0, nullptr, &mapped))) return 11;
    UINT64 total = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, nullptr, nullptr, nullptr, &total);
    ComPtr<ID3D12Resource> readback;
    {
        D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if (FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))))
            return 12;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);

    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, width, height};
    const D3D12_VERTEX_BUFFER_VIEW vertex_view{vertices->GetGPUVirtualAddress(), sizeof(Scene::records), sizeof(Vertex)};
    for (const auto& scene : scenes) {
        std::memcpy(mapped, scene.records.data(), sizeof(scene.records));
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pso.Get(), IID_PPV_ARGS(&list))))
            return 13;
        const auto transition = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
            list->ResourceBarrier(1, &barrier);
        };
        transition(target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ClearRenderTargetView(rtv_heap->GetCPUDescriptorHandleForHeapStart(), clear, 0, nullptr);
        list->OMSetRenderTargets(1, &rtv_heap->GetCPUDescriptorHandleForHeapStart(), FALSE, nullptr);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->IASetVertexBuffers(0, 1, &vertex_view);
        list->DrawInstanced(6, 1, 0, 0);
        transition(target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION from{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (FAILED(list->Close())) return 14;
        ComPtr<ID3D12CommandQueue> queue;
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        ComPtr<ID3D12Fence> fence;
        if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return 15;
        ID3D12CommandList* lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.Get(), 1))) return 15;
        while (fence->GetCompletedValue() != 1) Sleep(1);
        void* pixels = nullptr;
        if (FAILED(readback->Map(0, nullptr, &pixels))) return 16;
        std::vector<std::uint32_t> actual(width * height);
        for (UINT y = 0; y < height; ++y)
            std::memcpy(&actual[y * width], static_cast<const unsigned char*>(pixels) + y * footprint.Footprint.RowPitch,
                        width * 4);
        readback->Unmap(0, nullptr);
        auto want = reference(scene.quad);
        if (!scene.covered) std::fill(want.begin(), want.end(), blue);
        UINT differing = 0;
        for (UINT i = 0; i < actual.size(); ++i) {
            if (actual[i] != want[i]) {
                if (!differing)
                    std::printf("  %s: pixel (%u, %d) is %08x, expected %08x\n", scene.name, i % width,
                                static_cast<int>(i / width), actual[i], want[i]);
                ++differing;
            }
        }
        expect(differing == 0, scene.name);
        std::printf("%s: %u of %u pixels differ from the literal rectangle rule (angle %.1f)\n", scene.name, differing,
                    width * height, scene.angle);
    }
    vertices->Unmap(0, nullptr);
    std::printf("rect geometry: %s, %d failure(s)\n", failures ? "failed" : "passed", failures);
    return failures ? 1 : 0;
}
