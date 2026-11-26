#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "desktop_gpu.h"
#include "xmem_section_lease.h"
#include "startup_stubs/rect_indirect_args.h"
#include "startup_indirect_mesh_mesh.h"
#include "startup_indirect_mesh_amplify.h"
#include "startup_indirect_mesh_amplified_mesh.h"
#include "startup_indirect_mesh_pixel.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_distinct_rect_scratch(void*, void*);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_census_report(char* out, std::size_t capacity);
extern "C" __declspec(dllimport) LONG64 x2w_d3d12_mesh_census_lock_probe(void (*exercise)(void*), void* context);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_not_drawn_keys_report(char* out, std::size_t capacity);

#if defined(OFFSCREEN_AGILITY_620)
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 620;
__declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

namespace {
using Microsoft::WRL::ComPtr;
using Words = std::array<std::uint32_t, 9>;
constexpr UINT width = 4, height = 2;
constexpr auto format = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr std::uint32_t black = 0xff000000, red = 0xff0000ff, green = 0xff00ff00;
constexpr Words initial{0xa5a5a5a5, 0, 0, 0, 0, 0, 0x5a5a5a5a, 0x12345678, 0x9abcdef0};
constexpr std::array<std::uint16_t, 10> indices{99, 99, 99, 0, 1, 2, 3, 4, 5, 99};
constexpr std::array<std::uint32_t, 10> wide_indices{99, 99, 99, 0, 1, 2, 3, 4, 5, 99};
constexpr UINT argument_offset = sizeof(std::uint32_t);
constexpr UINT64 replay_copy_offset = 0xbfffa0, replay_copy_bytes = 0x44;

constexpr char shader[] = R"(
RWByteAddressBuffer data : register(u0);
ByteAddressBuffer retained_input : register(t0);
[numthreads(1,1,1)]
void consume_retained_compute_root() {
    data.Store(0, retained_input.Load(0));
}

[numthreads(1, 1, 1)]
void write_left() {
    data.Store4(4, uint4(3, 1, 1, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_right() {
    data.Store4(4, uint4(3, 1, 4, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_indexed_base() {
    data.Store4(8, uint4(3, 1, 1, 3));
    data.Store(24, 0);
}

[numthreads(1, 1, 1)]
void write_four_rectangle_indices() {
    data.Store4(4, uint4(4, 1, 1, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_indexed_overrun() {
    data.Store4(8, uint4(3, 1, 1, 5));
    data.Store(24, 0);
}

[numthreads(1, 1, 1)]
void write_one_triangle() {
    data.Store4(4, uint4(3, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_two_triangles() {
    data.Store4(4, uint4(6, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_buffer_colour() {
    data.Store(16, 0xff00ff00u);
    data.Store(32, 0xff0000ffu);
}

[numthreads(1, 1, 1)]
void write_large_buffer_colour() {
    data.Store(0x100000c, 0xff00ff00u);
    data.Store(0x100001c, 0xff0000ffu);
}

[numthreads(1, 1, 1)]
void write_replay_colour() {
    data.Store(0, 0xff00ff00u);
    data.Store(4, 0xff00ff00u);
    data.Store4(0xbfffa0, uint4(0xff00ff00u, 0xff0000ffu, 0, 0));
    data.Store4(0xbfffb0, uint4(0, 0, 0, 0));
    data.Store4(0xbfffc0, uint4(0, 0, 0, 0));
    data.Store4(0xbfffd0, uint4(0, 0, 0, 0));
    data.Store(0xbfffe0, 0);
}

[numthreads(1, 1, 1)]
void write_one_mesh_group() {
    data.Store3(4, uint3(1, 1, 1));
}

[numthreads(1, 1, 1)]
void write_two_mesh_groups() {
    data.Store3(4, uint3(2, 1, 1));
}

[numthreads(1, 1, 1)]
void initialize_mesh_record() {
    data.Store4(0, uint4(0, 1, 1, 0));
}

[numthreads(2, 1, 1)]
void append_mesh_groups() {
    uint previous;
    data.InterlockedAdd(0, 1u, previous);
}

[numthreads(1, 1, 1)]
void write_one_rect() {
    data.Store4(4, uint4(3, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_two_rects() {
    data.Store4(4, uint4(6, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_overflowing_rects() {
    data.Store4(4, uint4(0xffffffff, 1, 0, 0));
}

struct Varying {
    float4 position : SV_Position;
    nointerpolation uint side : SIDE;
};

Varying vertex(uint id : SV_VertexID) {
    Varying result;
    uint corner = id % 3;
    result.position = float4(corner == 2 ? 3 : -1, corner == 1 ? 3 : -1, 0, 1);
    result.side = id / 3;
    return result;
}

// One rectangle is three console vertices (bottom-left, bottom-right, top-right); the desktop draw spends six
// vertices on it: the triangles (v0, v1, v2) and (v0, v2, v3) with v3 = v0 + v2 - v1.
Varying vertex_rect(uint id : SV_VertexID) {
    Varying result;
    const uint rect = id / 6;
    // No local array indexed by SV_VertexID: some Gen9 drivers mis-lower that. The corner sequence
    // 0,1,2,0,2,3 is computed with plain arithmetic and the corner is picked with scalar selects.
    const uint k = id % 6;
    const uint c = (k < 3u) ? k : ((k == 3u) ? 0u : (k - 2u));
    const float x0 = (rect == 0) ? -1.0 : 0.0;
    const float x1 = (rect == 0) ? 0.0 : 1.0;
    // v0 = (x0, -1), v1 = (x1, -1), v2 = (x1, 1), v3 = v0 + v2 - v1 = (x0, 1)
    const float x = (c == 0u || c == 3u) ? x0 : x1;
    const float y = (c >= 2u) ? 1.0 : -1.0;
    result.position = float4(x, y, 0, 1);
    result.side = rect;
    return result;
}

float4 pixel(Varying input) : SV_Target {
    if (input.side == 0 && input.position.x < 2) return float4(1, 0, 0, 1);
    if (input.side == 1 && input.position.x >= 2) return float4(0, 1, 0, 1);
    discard;
    return float4(0, 0, 0, 1);
}
cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; };
Texture2D<float4> translated_texture : register(t0);
ByteAddressBuffer translated_buffer : register(t0);
Texture2D<float4> translated_indexed_texture : register(t1);

float4 translated_vertex(uint id : SV_VertexID) : SV_Position {
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}

float4 translated_vertex_with_start(uint id : SV_VertexID) : SV_Position {
    return translated_vertex(id + ud1);
}

float4 translated_pixel() : SV_Target {
    return translated_texture.Load(int3(0, 0, 0));
}

float4 translated_buffer_pixel() : SV_Target {
    uint value = translated_buffer.Load(ud2);
    return float4(value & 255, (value >> 8) & 255, (value >> 16) & 255, 255) / 255.0;
}

float4 translated_buffer_tail_pixel() : SV_Target {
    uint value = translated_buffer.Load(0x100000c);
    return float4(value & 255, (value >> 8) & 255, (value >> 16) & 255, 255) / 255.0;
}

struct RectVarying {
    float4 pos : SV_Position;
    float4 uv : PARAM0;
    nointerpolation uint4 flat : FPARAM0;
};
RectVarying translated_rect_vertex(uint id : SV_VertexID) {
    RectVarying o = (RectVarying)0;
    if (ud0) return o;
    const uint offset = (id + ud1) * ud2;
    o.pos = asfloat(translated_buffer.Load4(offset));
    const uint uv = translated_buffer.Load(offset + 16);
    o.uv.xy = float2(f16tof32(uv & 65535), f16tof32(uv >> 16));
    o.flat.x = translated_buffer.Load(offset + 20);
    return o;
}
float4 translated_rect_pixel(RectVarying input) : SV_Target {
    const float2 quantizedUv = floor(input.uv.xy * 255 + .5) / 255;
    return float4(quantizedUv, float(input.flat.x) / 255, 1);
}

Varying translated_indexed_vertex(uint id : SV_VertexID) {
    return vertex(translated_buffer.Load((id + ud1) * ud2));
}

float4 translated_indexed_pixel(Varying input) : SV_Target {
    if ((input.side == 0 && input.position.x < 2) || (input.side == 1 && input.position.x >= 2))
        return translated_indexed_texture.Load(int3(0, 0, 0));
    discard;
    return 0;
}

)";

template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object, bool allow_missing = false)
{
    // The guest forwards SetPrivateDataInterface with the underlying desktop argument.
    constexpr GUID key{0x18097577, 0x0a4a, 0x427c, {0xac, 0xd5, 0xac, 0x1d, 0x98, 0x90, 0x65, 0x85}};
    check(guest->SetPrivateDataInterface(key, object), "publish underlying object");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear underlying object");
    if (allow_missing && (read == DXGI_ERROR_NOT_FOUND || (SUCCEEDED(read) && !stored))) return {};
    check(read, "retrieve underlying object");
    ComPtr<T> result;
    check(stored.As(&result), "query underlying object");
    return result;
}

ComPtr<ID3DBlob> compile(const char* entry, const char* profile)
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, entry, profile, 0, 0,
                                  &code, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    check(hr, entry);
    return code;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& desc,
                                       ComPtr<ID3DBlob>* serialized = nullptr)
{
    ComPtr<ID3DBlob> code;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &code, nullptr), "serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create root");
    if (serialized) *serialized = code;
    return root;
}

enum class TableSample { texture, buffer, buffer_tail };

ComPtr<IUnknown> translated_pipeline(ID3D12Device* guest, ID3D12RootSignature* root, ID3DBlob* root_blob,
                                    bool needs_start, TableSample sample = TableSample::texture, bool indexed = false, bool rect = false)
{
    // The opaque guest programs identify synthetic cache entries. The actual bound shaders are source-built DXBC.
    constexpr char revision[] = "x2w-startup-shader-28";
    static std::uint32_t vertex_ids[]{0xe0000101, 0xe0000102, 0xe0000103};
    static std::uint32_t pixel_ids[]{0xe0000202, 0xe0000203, 0xe0000205};
    static std::uint32_t indexed_pixel_id = 0xe0000204;
    auto& vertex_id = vertex_ids[indexed ? 2 : needs_start];
    auto& pixel_id = indexed ? indexed_pixel_id : pixel_ids[static_cast<unsigned>(sample)];
    std::uint64_t key = 1469598103934665603ull;
    const auto add = [&key](const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) key = (key ^ p[i]) * 1099511628211ull;
    };
    add(revision, sizeof(revision) - 1);
    add(&vertex_id, sizeof(vertex_id));
    add(&pixel_id, sizeof(pixel_id));
    add(root_blob->GetBufferPointer(), root_blob->GetBufferSize());
    char stem[32];
    std::snprintf(stem, sizeof(stem), "%016llx", static_cast<unsigned long long>(key));
    char directory[MAX_PATH];
    const DWORD size = GetEnvironmentVariableA("X2W_SHADER_CACHE", directory, MAX_PATH);
    check(size && size < MAX_PATH ? S_OK : E_FAIL, "find fixture cache");
    const std::string prefix = std::string(directory) + "\\" + stem;
    const auto write = [&](const char* suffix, const void* bytes, std::size_t count) {
        std::ofstream file(prefix + suffix, std::ios::binary | std::ios::trunc);
        file.write(static_cast<const char*>(bytes), count);
        check(file.good() ? S_OK : E_FAIL, "write fixture cache");
    };
    const auto vs = compile(rect ? "translated_rect_vertex" : indexed ? "translated_indexed_vertex" : needs_start ? "translated_vertex_with_start" : "translated_vertex", "vs_5_1");
    const char* table_pixels[]{"translated_pixel", "translated_buffer_pixel", "translated_buffer_tail_pixel"};
    const auto ps = compile(rect ? "translated_rect_pixel" : indexed ? "translated_indexed_pixel" : table_pixels[static_cast<unsigned>(sample)], "ps_5_1");
    write(".vs.cso", vs->GetBufferPointer(), vs->GetBufferSize());
    write(".ps.cso", ps->GetBufferPointer(), ps->GetBufferSize());
    constexpr char manifest[] = "x2w-manifest 1\nud 2\nudsrc 0 rect\nudsrc 1 basevertex\nsrv 0 tex2d table 0 0\nparams 0\ntargets 0\n";
    constexpr char buffer_manifest[] = "x2w-manifest 1\nud 3\nudsrc 0 rect\nudsrc 1 basevertex\nudsrc 2 stride 0\nsrv 0 buf table 0 0\nparams 0\ntargets 0\n";
    constexpr char indexed_manifest[] = "x2w-manifest 1\nud 3\nudsrc 0 rect\nudsrc 1 basevertex\nudsrc 2 stride 0\nsrv 0 buf vb 0\nsrv 1 tex2d table 0 0\nparams 1\ntargets 0\n";
    if (indexed) write(".txt", indexed_manifest, sizeof(indexed_manifest) - 1);
    else if (sample != TableSample::texture) write(".txt", buffer_manifest, sizeof(buffer_manifest) - 1);
    else write(".txt", manifest, sizeof(manifest) - 1);
    const std::array<UINT64, 0x300 / sizeof(UINT64)> desc{
        reinterpret_cast<UINT64>(root), 0, 0, 0, 0,
        reinterpret_cast<UINT64>(&vertex_id), sizeof(vertex_id), 0, 0,
        reinterpret_cast<UINT64>(&pixel_id), sizeof(pixel_id), 0};
    using Create = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
    ComPtr<IUnknown> result;
    check(slot<Create>(guest, 57)(guest, desc.data(), __uuidof(ID3D12PipelineState),
        reinterpret_cast<void**>(result.GetAddressOf())), "create translated guest pipeline");
    return result;
}

ComPtr<ID3D12PipelineState> writer_pipeline(ID3D12Device* device, ID3D12RootSignature* root, const char* entry)
{
    const auto cs = compile(entry, "cs_5_0");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    ComPtr<ID3D12PipelineState> result;
    check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result)), "create writer pipeline");
    return result;
}

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Kind;
    T value{};
};

enum class StreamDepth { none, writable, read_only };

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Stage, bool CacheControl = false>
ComPtr<IUnknown> guest_pipeline(ID3D12Device* guest, ID3D12RootSignature* root,
                               D3D12_SHADER_BYTECODE source, D3D12_SHADER_BYTECODE pixel,
                               D3D12_SHADER_BYTECODE amplification = {}, UINT targets = 1, StreamDepth depth = StreamDepth::none)
{
    using Depth = std::conditional_t<CacheControl, D3D12_DEPTH_STENCIL_DESC1, D3D12_DEPTH_STENCIL_DESC>;
    constexpr auto depth_type = CacheControl ? D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1
                                             : D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
    struct {
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
        Subobject<Stage, D3D12_SHADER_BYTECODE> source;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
        Subobject<depth_type, Depth> depth;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> targets;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> samples;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> dsv;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE> as;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT> sample_mask;
    } stream;
    static_assert(offsetof(decltype(stream), as) == 16 + 24 + 24 + 48 + (CacheControl ? 64 : 56) + 8 + 40 + 16 + 8);
    stream.root.value = root;
    stream.source.value = source;
    stream.ps.value = pixel;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.depth.value.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    stream.depth.value.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    if (depth != StreamDepth::none) {
        stream.depth.value.DepthEnable = TRUE;
        if (depth == StreamDepth::writable)
            stream.depth.value.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        stream.depth.value.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        stream.dsv.value = DXGI_FORMAT_D16_UNORM;
    }
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    std::fill_n(stream.targets.value.RTFormats, targets, format);
    stream.targets.value.NumRenderTargets = targets;
    stream.samples.value.Count = 1;
    stream.as.value = amplification;
    using Create = HRESULT (WINAPI*)(IUnknown*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    ComPtr<IUnknown> result;
    const auto create = [&](void* bytes, std::size_t size) {
        const D3D12_PIPELINE_STATE_STREAM_DESC desc{size, bytes};
        check(slot<Create>(guest, 86)(guest, &desc, __uuidof(ID3D12PipelineState),
                                     reinterpret_cast<void**>(result.GetAddressOf())), "create guest graphics pipeline");
    };
    if constexpr (CacheControl) {
        constexpr auto masks = offsetof(Depth, StencilWriteMask) + 1;
        constexpr auto faces = offsetof(Depth, FrontFace);
        constexpr auto write_mask = offsetof(D3D12_RENDER_TARGET_BLEND_DESC, RenderTargetWriteMask) + 1;
        const int padding = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_PADDING", nullptr, 0) ? 0xa5 : 0;
        std::memset(reinterpret_cast<unsigned char*>(&stream.depth.value) + masks, padding, faces - masks);
        for (auto& blend : stream.blend.value.RenderTarget) {
            blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
            blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
            blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            blend.LogicOp = D3D12_LOGIC_OP_NOOP;
            blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            std::memset(reinterpret_cast<unsigned char*>(&blend) + write_mask, padding, sizeof(blend) - write_mask);
        }
        stream.sample_mask.value = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_SAMPLE_MASK", nullptr, 0)
            ? UINT_MAX ^ 1u : UINT_MAX;

        // Omit the amplification record from this cache profile.
        constexpr auto prefix = offsetof(decltype(stream), as);
        constexpr auto suffix = offsetof(decltype(stream), blend);
        alignas(void*) std::array<unsigned char, sizeof(stream) - sizeof(stream.as)> bytes;
        std::memcpy(bytes.data(), &stream, prefix);
        std::memcpy(bytes.data() + prefix, &stream.blend, sizeof(stream) - suffix);
        create(bytes.data(), bytes.size());
    } else {
        create(&stream, amplification.pShaderBytecode ? offsetof(decltype(stream), blend) : offsetof(decltype(stream), as));
    }
    return result;
}

struct FreeMemory {
    void operator()(void* pointer) const { VirtualFree(pointer, 0, MEM_RELEASE); }
};

struct PlacedBuffer {
    PlacedBuffer(ID3D12Device* device, ID3D12Device* guest, UINT width, D3D12_RESOURCE_FLAGS flags)
        : backing(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
    {
        if (!backing) throw std::bad_alloc();
        // A deliberately unaligned placed address selects the runtime's committed stand-in.
        // Its GPU data is independent of the zero-filled guest virtual address.
        address = static_cast<char*>(backing.get()) + 4;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = width;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;
        using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
        check(slot<Place>(guest, 133)(guest, address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                      reinterpret_cast<void**>(wrapped.GetAddressOf())), "place guest buffer");
        native = native_object<ID3D12Resource>(device, guest, wrapped.Get());
    }

    std::unique_ptr<void, FreeMemory> backing;
    void* address{};
    ComPtr<IUnknown> wrapped;
    ComPtr<ID3D12Resource> native;
};

ComPtr<IUnknown> begin_guest_list(ID3D12Device* guest, ID3D12CommandAllocator* allocator,
                                 D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT)
{
    using CreateList = HRESULT (WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    const UINT list_desc[]{static_cast<UINT>(type), 2, 1, 0};
    ComPtr<IUnknown> result;
    check(slot<CreateList>(guest, 73)(guest, list_desc, allocator, nullptr, __uuidof(ID3D12GraphicsCommandList),
                                     reinterpret_cast<void**>(result.GetAddressOf())), "create guest command list");
    using Reset = HRESULT (WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(result.Get(), 139)(result.Get(), allocator, nullptr, nullptr), "reset guest command list");
    return result;
}

void submit(ID3D12Device* device, ID3D12GraphicsCommandList* list)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "create queue");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create completion fence");
    ID3D12CommandList* submitted[]{list};
    queue->ExecuteCommandLists(1, submitted);
    check(queue->Signal(fence.Get(), 1), "signal completion fence");
    check(fence->SetEventOnCompletion(1, nullptr), "wait for completion");
    check(device->GetDeviceRemovedReason(), "GPU completion");
}

// The lease fixtures hold a writer queue behind a CPU-signalled gate while run() clears and reads back
// its targets. With NVIDIA driver 32.0.16.2124, releasing a queue during that hold blocks until the test
// times out when another process releases a queue during its own hold. Those clears and copies therefore
// share one DIRECT queue per device that lives until the process exits.
ID3D12CommandQueue* kept_direct_queue(ID3D12Device* device)
{
    static std::mutex lock;
    static std::vector<std::pair<ID3D12Device*, ComPtr<ID3D12CommandQueue>>> queues;
    const std::lock_guard guard(lock);
    for (const auto& [owner, queue] : queues)
        if (owner == device) return queue.Get();
    const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "create kept readback queue");
    queues.emplace_back(device, queue);
    return queue.Get();
}

struct MeshLeaseDraw {
    UINT64 address;
    IUnknown* queue;
    UINT expected;
    bool drop_recording = false;
    std::function<void(ID3D12Fence*)> queued;
    IUnknown* argument{};
    UINT64 argument_offset{};
    std::optional<UINT> expected_right;
    std::function<void()> before_submit;
};

bool mesh_fence_reached(ID3D12Fence* fence, DWORD milliseconds)
{
    const auto deadline = GetTickCount64() + milliseconds;
    do {
        const auto completed = fence->GetCompletedValue();
        if (completed == 1) return true;
        if (completed == UINT64_MAX || GetTickCount64() >= deadline) return false;
        Sleep(1);
    } while (true);
}

void submit_wrapped(IUnknown* queue, IUnknown* list)
{
    using Execute = void (WINAPI*)(IUnknown*, UINT, IUnknown* const*);
    slot<Execute>(queue, 10)(queue, list ? 1u : 0u, list ? &list : nullptr);
}

// Two runs on two threads share one guest device, whose recording state is not meant to be driven from two threads
// at once. The gate serializes everything except the GPU phase and releases both submits together, so the two
// queues run their conversions at the same time.
struct Gate {
    std::mutex state;
    std::mutex wait_lock;
    std::condition_variable changed;
    IUnknown* first = nullptr;
    unsigned generation = 0;
    bool aborted = false;
    bool meet(IUnknown* list) {
        std::unique_lock<std::mutex> guard(wait_lock);
        if (aborted) return false;
        if (first) {
            if (!x2w_d3d12_distinct_rect_scratch(first, list)) {
                std::fprintf(stderr, "rectangle scratch is missing or shared between recordings\n");
                aborted = true;
            }
            first = nullptr;
            ++generation;
            changed.notify_all();
            return !aborted;
        }
        first = list;
        const unsigned mine = generation;
        changed.wait(guard, [&] { return aborted || generation != mine; });
        return !aborted;
    }
    void abort() {
        std::lock_guard<std::mutex> guard(wait_lock);
        aborted = true;
        changed.notify_all();
    }
};

enum class Refusal { none, index_upload_cross, index_upload_foreign, index_upload_unaligned, index_unregistered,
                     alias_no_heap, alias_overlaps, alias_cross_page, alias_off, invalid_index, short_argument, missing_viewport, wrong_root, equivalent_root_disabled, missing_texture, start_vertex,
                     insufficient_table_range, copied_table_buffer, upload_range_crossing, unproven_copy,
                     indexed_base_overrun, indexed_vertex_range, invalid_mesh_tag,
                     rect_bad_index, rect_bad_count, rect_bad_w, rect_bad_shape,
                     incomplete_mesh_constants, mesh_attachment_count, mesh_pending, overflowing_count, rect_live_compute };
enum class DrawKind { indexed, vertices, mesh, indexed_rect, rect };

ComPtr<ID3D12Resource> make_gpu_table_buffer(ID3D12Device* guest, UINT64 bytes)
{
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    using Create = HRESULT (WINAPI*)(IUnknown*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*,
        UINT, const void*, void*, UINT, const void*, REFIID, void**);
    ComPtr<ID3D12Resource> result;
    check(slot<Create>(guest, 127)(guest, &props, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr,
        0, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(result.GetAddressOf())), "create GPU table buffer");
    return result;
}

struct TranslatedDrawFixture {
    TranslatedDrawFixture(ID3D12Device* guest, std::uint32_t colour, ID3D12Device* desktop = nullptr,
                          bool short_view = false, bool copied = false, bool replay = false, bool allow_rt = false,
                          UINT committed_bytes = 64)
        : wrong_root(create_root(guest, {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE})),
          backing(VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
    {
        if (!backing) throw std::bad_alloc();
        if (desktop && !copied) {
            sampled = make_gpu_table_buffer(guest, committed_bytes);
        } else {
            const D3D12_RESOURCE_DESC desc = copied ?
                D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_BUFFER, 0, replay ? replay_copy_bytes : 64u, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS} :
                D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, allow_rt ? width : 1u, allow_rt ? height : 1u, 1, 1, format, {1, 0},
                                    D3D12_TEXTURE_LAYOUT_UNKNOWN, allow_rt ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_NONE};
            auto* address = static_cast<unsigned char*>(backing.get()) + (copied ? 16 : 0);
            std::memcpy(address + (replay ? 4 : copied ? 32 : 0), &colour, sizeof(colour));
            using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
            check(slot<Place>(guest, 133)(guest, address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                __uuidof(ID3D12Resource), reinterpret_cast<void**>(sampled.GetAddressOf())), "place sampled resource");
        }
        if (desktop) gpu_buffer = native_object<ID3D12Resource>(desktop, guest, sampled.Get());
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
        using CreateHeap = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
        check(slot<CreateHeap>(guest, 14)(guest, &heap_desc, __uuidof(ID3D12DescriptorHeap),
            reinterpret_cast<void**>(heap.GetAddressOf())), "create sampled texture heap");
        struct {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            UINT64 address = 0, tail[2]{};
        } storage;
        auto& srv = storage.view;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (desktop) {
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Buffer.FirstElement = replay ? 0 : 1;
            srv.Buffer.NumElements = replay ? static_cast<UINT>(replay_copy_bytes / 4) : short_view ? 1 : 2;
            srv.Buffer.StructureByteStride = replay ? 4 : 16;
            if (committed_bytes > 64)
                srv.Buffer.NumElements = (committed_bytes / 16) - 1 - (short_view ? 1 : 0);
        } else {
            srv.Format = format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = 1;
        }
        using Handle = UINT64 (WINAPI*)(IUnknown*, void*);
        cpu = slot<Handle>(heap.Get(), 9)(heap.Get(), nullptr);
        gpu = slot<Handle>(heap.Get(), 10)(heap.Get(), nullptr);
        check(cpu && gpu ? S_OK : E_FAIL, "get guest texture handles");
        using View = void (WINAPI*)(IUnknown*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, UINT64, UINT64);
        slot<View>(guest, 18)(guest, sampled.Get(), &srv, cpu, 0);
    }

    ComPtr<ID3D12RootSignature> wrong_root;
    std::unique_ptr<void, FreeMemory> backing;
    ComPtr<ID3D12Resource> sampled;
    ComPtr<ID3D12Resource> gpu_buffer;
    ComPtr<ID3D12PipelineState> buffer_writer;
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT64 cpu = 0;
    UINT64 gpu = 0;
};

struct ReplayCopy {
    ID3D12Resource* producer;
    ID3D12Resource* native;
    ID3D12PipelineState* writer;
    UINT64 bytes;
    bool uncertified_overwrite = false;
};

constexpr UINT command_stride(DrawKind kind)
{
    switch (kind) {
    case DrawKind::indexed_rect:
    case DrawKind::indexed: return sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
    case DrawKind::vertices:
    case DrawKind::rect: return sizeof(D3D12_DRAW_ARGUMENTS);
    case DrawKind::mesh: return sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
    }
    return 0;
}

bool supports_mesh(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    return SUCCEEDED(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))) &&
        options.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
}

// Failure-path diagnostic for the rectangle-list indirect conversion. Runs the same GPU conversion pass on a copy of
// the four draw words that the title's buffer held and prints the words it wrote, so a log shows the exact
// DrawInstanced arguments (vertex count, instance count, start vertex, start instance) the adapter produced.
void report_rect_conversion(ID3D12Device* device, const Words& observed, UINT word)
{
    const std::uint32_t source_words[4]{observed[word], observed[word + 1], observed[word + 2], observed[word + 3]};
    const std::uint32_t cpu_vertices = source_words[0] / 3 > 0x2AAAAAAAu ? 0 : source_words[0] / 3 * 6;
    std::fprintf(stderr, "rect conversion diagnostic: title words %u %u %u %u, CPU reference vertices %u\n",
                 source_words[0], source_words[1], source_words[2], source_words[3], cpu_vertices);
    const auto source = make_buffer(device, 16, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const auto readback = make_buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto scratch = x2w_startup::RectIndirectArguments::make_scratch(device);
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    if (!scratch || FAILED(source->Map(0, &none, &mapped))) {
        std::fprintf(stderr, "rect conversion diagnostic: scratch or source unavailable\n");
        return;
    }
    std::memcpy(mapped, source_words, sizeof(source_words));
    source->Unmap(0, nullptr);
    ComputeCommands commands(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
    if (!x2w_startup::rect_indirect_arguments().convert(commands.list.Get(), source.Get(), 0, scratch.Get())) {
        std::fprintf(stderr, "rect conversion diagnostic: conversion pass unavailable\n");
        return;
    }
    transition(commands.list.Get(), scratch.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands.list->CopyBufferRegion(readback.Get(), 0, scratch.Get(), 0, 16);
    commands.submit_and_wait(device);
    const D3D12_RANGE read{0, 16};
    std::uint32_t converted[4]{};
    if (FAILED(readback->Map(0, &read, &mapped))) return;
    std::memcpy(converted, mapped, sizeof(converted));
    readback->Unmap(0, &none);
    std::fprintf(stderr, "rect conversion diagnostic: GPU scratch DrawInstanced words %u %u %u %u (%s)\n", converted[0],
                 converted[1], converted[2], converted[3],
                 converted[0] == cpu_vertices && converted[1] == source_words[1] && converted[2] == source_words[2] &&
                 converted[3] == source_words[3] ? "matches the CPU reference" : "differs from the CPU reference");
}

bool run(ID3D12Device* device, ID3D12Device* guest, ID3D12RootSignature* writer_root, ID3D12PipelineState* writer,
         ID3D12RootSignature* graphics_root, IUnknown* graphics, ID3D12CommandSignature* signature,
         DrawKind kind, bool draw_right, Refusal refusal, const char* label,
         const TranslatedDrawFixture* translated = nullptr, bool upload_table = false, const ReplayCopy* replay = nullptr,
         bool upload_index = false, bool alias_index = false,
         UINT cached_mesh_targets = 0, bool cached_depth_only = false, bool restore_compute = false,
         const MeshLeaseDraw* leased = nullptr, Gate* gate = nullptr, bool cached_depth_pixel = false)
{
    std::unique_lock<std::mutex> recording;
    if (gate) recording = std::unique_lock<std::mutex>(gate->state);
    const bool indexed = kind == DrawKind::indexed || kind == DrawKind::indexed_rect;
    const bool translated_indexed = translated && indexed;
    const UINT indexed_argument_offset = translated_indexed && draw_right ? 8 : argument_offset;
    const bool mesh = kind == DrawKind::mesh;
    const UINT index_bytes = indexed ? (draw_right ? sizeof(wide_indices) : sizeof(indices)) : 0;
    const UINT index_offset = 2 * (draw_right ? sizeof(std::uint32_t) : sizeof(std::uint16_t));
    const bool rect = kind == DrawKind::indexed_rect;
    auto narrow = indices;
    auto wide = wide_indices;
    if (rect) {
        narrow[3] = 2;
        narrow[4] = 0;
        narrow[5] = refusal == Refusal::rect_bad_index ? 9 : 1;
        wide[3] = 0;
        wide[4] = 2;
        wide[5] = 1;
    }
    const void* index_data = draw_right ? static_cast<const void*>(wide.data()) : narrow.data();
    Words initial_words = initial;
    if (!indexed) initial_words[1 + command_stride(kind) / sizeof(std::uint32_t)] = 0x3c3c3c3c;
    const bool external_arguments = leased && leased->argument;
    std::optional<PlacedBuffer> arguments, index_buffer;
    if (!external_arguments)
        arguments.emplace(device, guest, static_cast<UINT>(sizeof(initial_words)), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (indexed && !upload_index && !alias_index) index_buffer.emplace(device, guest, index_bytes, D3D12_RESOURCE_FLAG_NONE);
    ComPtr<ID3D12Resource> upload, arg_readback;
    void* memory = nullptr;
    const D3D12_RANGE untouched{0, 0};
    if (arguments) {
        upload = make_buffer(device, sizeof(initial_words) + index_bytes, D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
        arg_readback = make_buffer(device, sizeof(initial), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        check(upload->Map(0, &untouched, &memory), "map upload");
        std::memcpy(memory, initial_words.data(), sizeof(initial_words));
        if (indexed) std::memcpy(static_cast<char*>(memory) + sizeof(initial_words), index_data, index_bytes);
        upload->Unmap(0, nullptr);
    }

    const auto target_format = cached_depth_only ? DXGI_FORMAT_D16_UNORM : format;
    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, target_format, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN, cached_depth_only ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL :
                                                          D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    constexpr float clear[]{0, 0, 0, 1};
    D3D12_CLEAR_VALUE clear_value{target_format, {0, 0, 0, 1}};
    if (cached_depth_only) clear_value.DepthStencil = {1, 0};
    const UINT target_count = cached_mesh_targets ? cached_mesh_targets : 1;
    std::vector<ComPtr<ID3D12Resource>> targets;
    const auto heap_kind = cached_depth_only ? D3D12_DESCRIPTOR_HEAP_TYPE_DSV : D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{heap_kind, target_count,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check((cached_depth_only ? guest : device)->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)),
        "create attachment heap");
    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> rtvs;
    D3D12_CPU_DESCRIPTOR_HANDLE native_dsv{};
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    if (cached_depth_only) {
        using GetCpu = UINT64(WINAPI*)(IUnknown*, void*);
        rtv.ptr = slot<GetCpu>(heap.Get(), 9)(heap.Get(), nullptr);
        native_dsv = native_object<ID3D12DescriptorHeap>(device, guest, heap.Get())->GetCPUDescriptorHandleForHeapStart();
    } else {
        rtv = heap->GetCPUDescriptorHandleForHeapStart();
    }
    for (UINT i = 0; i < target_count; ++i) {
        targets.push_back(resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &clear_value));
        if (cached_depth_only) {
            D3D12_DEPTH_STENCIL_VIEW_DESC depth_view{};
            depth_view.Format = target_format;
            depth_view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            guest->CreateDepthStencilView(targets.back().Get(), &depth_view, rtv);
        } else guest->CreateRenderTargetView(targets.back().Get(), nullptr, rtv);
        rtvs.push_back(rtv);
        rtv.ptr += device->GetDescriptorHandleIncrementSize(heap_kind);
    }

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    auto wrapped_list = begin_guest_list(guest, allocator.Get());
    auto* wrapped = wrapped_list.Get();
    const auto native_list = native_object<ID3D12GraphicsCommandList>(device, guest, wrapped);
    auto* list = native_list.Get();
    constexpr UINT64 alias_page_bytes = 65536, alias_ib_offset = 0x224, alias_vb_offset = 0x490;
    static std::vector<std::unique_ptr<void, FreeMemory>> alias_owners;
    char* alias_cpu = nullptr;
    ComPtr<IUnknown> alias_seed, alias_overlap;
    if (alias_index) {
        void* base = VirtualAlloc(nullptr, 2 * alias_page_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!base) return false;
        alias_owners.emplace_back(base);
        alias_cpu = static_cast<char*>(base);
        const auto place = [&](UINT64 offset, IUnknown** result) {
            using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
            const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 16, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
            return slot<Place>(guest, 133)(guest, alias_cpu + offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                __uuidof(ID3D12Resource), reinterpret_cast<void**>(result));
        };
        if (refusal != Refusal::alias_no_heap)
            check(place(alias_page_bytes, alias_seed.GetAddressOf()), "create reference placed alias in second page");
        if (refusal == Refusal::alias_overlaps)
            check(place(0, alias_overlap.GetAddressOf()), "create overlapping alias on first page");
    }
    if (upload_table) {
        using Allocate = void* (WINAPI*)(IUnknown*, UINT64, UINT64);
        void* allocation = slot<Allocate>(wrapped, 148)(wrapped, 32, 16);
        if (!allocation) return false;
        const std::array<std::uint32_t, 8> words{green, 0, 0, 0, red, 0, 0, 0};
        std::memcpy(allocation, words.data(), sizeof(words));
        const UINT64 address = reinterpret_cast<UINT64>(allocation);
        const std::array<std::uint32_t, 4> view{static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>(address >> 32) | (16u << 16),
            refusal == Refusal::upload_range_crossing ? 3u : 2u, 0};
        std::memcpy(reinterpret_cast<void*>(translated->cpu), view.data(), sizeof(view));
    }
    if (index_buffer) {
        transition(list, index_buffer->native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(index_buffer->native.Get(), 0, upload.Get(), sizeof(initial_words), index_bytes);
        transition(list, index_buffer->native.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
    if (arguments) {
        transition(list, arguments->native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(arguments->native.Get(), 0, upload.Get(), 0, sizeof(initial));
        auto argument_state = D3D12_RESOURCE_STATE_COPY_DEST;
        if (writer) {
            transition(list, arguments->native.Get(), argument_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            argument_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list->SetPipelineState(writer);
            list->SetComputeRootSignature(writer_root);
            list->SetComputeRootUnorderedAccessView(0, arguments->native->GetGPUVirtualAddress());
            list->Dispatch(1, 1, 1);
        }
        transition(list, arguments->native.Get(), argument_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(arg_readback.Get(), 0, arguments->native.Get(), 0, sizeof(initial));
        // The wrapper owns the subsequent COMMON -> INDIRECT_ARGUMENT transition.
        transition(list, arguments->native.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    if (replay) {
        transition(list, replay->native, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(replay->writer);
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, replay->native->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, replay->native, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        using Copy = void (WINAPI*)(IUnknown*, IUnknown*, UINT64, IUnknown*, UINT64, UINT64);
        slot<Copy>(wrapped, 15)(wrapped, translated->sampled.Get(), 0, replay->producer, replay_copy_offset, replay->bytes);
        if (replay->uncertified_overwrite)
            slot<Copy>(wrapped, 15)(wrapped, translated->sampled.Get(), 0, replay->native, 0, 4);
    }
    if (translated && translated->gpu_buffer && translated->buffer_writer) {
        auto* table_resource = translated->gpu_buffer.Get();
        transition(list, table_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(translated->buffer_writer.Get());
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, table_resource->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, table_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    }

    using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
    using Clear = void (WINAPI*)(IUnknown*, SIZE_T, const float*, UINT, const D3D12_RECT*);
    using Targets = void (WINAPI*)(IUnknown*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL,
                                  const D3D12_CPU_DESCRIPTOR_HANDLE*);
    using Topology = void (WINAPI*)(IUnknown*, D3D12_PRIMITIVE_TOPOLOGY);
    using Viewport = void (WINAPI*)(IUnknown*, UINT, const D3D12_VIEWPORT*);
    using Scissor = void (WINAPI*)(IUnknown*, UINT, const D3D12_RECT*);
    using Index = void (WINAPI*)(IUnknown*, const D3D12_INDEX_BUFFER_VIEW*);
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
    const D3D12_RECT scissor{rect ? 1 : 0, 0, width, height};
    if (cached_depth_only) {
        slot<Targets>(wrapped, 50)(wrapped, 0, nullptr, FALSE, rtvs.data());
        list->ClearDepthStencilView(native_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        if (refusal == Refusal::mesh_attachment_count)
            slot<Targets>(wrapped, 50)(wrapped, 0, nullptr, FALSE, nullptr);
    } else {
        if (leased && leased->before_submit) {
            // A submission-level refusal skips the entire wrapped list,
            // including its clear. Initialize the readback target separately.
            ComputeCommands clear_targets(device, kept_direct_queue(device), nullptr);
            for (std::size_t i = 0; i < targets.size(); ++i) {
                auto* target = targets[i].Get();
                transition(clear_targets.list.Get(), target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
                clear_targets.list->ClearRenderTargetView(rtvs[i], clear, 0, nullptr);
                transition(clear_targets.list.Get(), target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
            }
            clear_targets.submit_and_wait(device);
        }
        for (const auto target_view : rtvs) slot<Clear>(wrapped, 52)(wrapped, target_view.ptr, clear, 0, nullptr);
        const UINT bound_targets = target_count - (refusal == Refusal::mesh_attachment_count ? 1 : 0);
        slot<Targets>(wrapped, 50)(wrapped, bound_targets, rtvs.data(), FALSE, nullptr);
    }
    slot<Bind>(wrapped, 25)(wrapped, graphics);
    slot<Bind>(wrapped, 34)(wrapped, refusal == Refusal::wrong_root ? translated->wrong_root.Get() : graphics_root);
    if (leased) {
        using RootSrv = void (WINAPI*)(IUnknown*, UINT, UINT64);
        slot<RootSrv>(wrapped, 44)(wrapped, 0, leased->address);
    } else if (cached_mesh_targets || cached_depth_only) {
        using Constants = void (WINAPI*)(IUnknown*, UINT, UINT, const UINT*, UINT);
        const UINT values[]{17, 29};
        slot<Constants>(wrapped, 40)(wrapped, 0, 1, values, 0);
        if (refusal != Refusal::incomplete_mesh_constants)
            slot<Constants>(wrapped, 40)(wrapped, 0, 1, values + 1, 1);
    }
    if (translated && refusal != Refusal::missing_texture) {
        ID3D12DescriptorHeap* heaps[]{translated->heap.Get()};
        using Heaps = void (WINAPI*)(IUnknown*, UINT, ID3D12DescriptorHeap* const*);
        using Table = void (WINAPI*)(IUnknown*, UINT, UINT64);
        slot<Heaps>(wrapped, 32)(wrapped, 1, heaps);
        slot<Table>(wrapped, 36)(wrapped, 0, translated->gpu);
    }
    if (!mesh) slot<Topology>(wrapped, 20)(wrapped,
        rect || kind == DrawKind::rect ? static_cast<D3D12_PRIMITIVE_TOPOLOGY>(0x11)
                                       : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (refusal != Refusal::missing_viewport) slot<Viewport>(wrapped, 21)(wrapped, 1, &viewport);
    slot<Scissor>(wrapped, 22)(wrapped, 1, &scissor);
    if (translated_indexed) {
        using Allocate = void* (WINAPI*)(IUnknown*, UINT64, UINT64);
        using VertexBuffer = void (WINAPI*)(IUnknown*, UINT, UINT, const D3D12_VERTEX_BUFFER_VIEW*);
        constexpr std::uint32_t vertex_values[]{0, 1, 2, 3, 4, 5};
        struct RectVertex { float x, y, z, w; std::uint16_t u, v; UINT flat; };
        static_assert(sizeof(RectVertex) == 24);
        constexpr RectVertex top_left{-2, 1, 0.5f, 1, 0x0000, 0x3400, 53};
        constexpr RectVertex bottom_left{-2, -1, 0.5f, 1, 0x0000, 0x3a00, 7};
        constexpr RectVertex bottom_right{0.5f, -1, 0.5f, 1, 0x3900, 0x3a00, 11};
        constexpr RectVertex top_right{0.5f, 1, 0.5f, 1, 0x3900, 0x3400, 29};
        RectVertex rectangle[]{bottom_left, bottom_right, top_left, top_left, top_right, bottom_left};
        if (refusal == Refusal::rect_bad_w) rectangle[1].w = 2;
        if (refusal == Refusal::rect_bad_shape) rectangle[0].y = -0.25f;
        const UINT vertex_bytes = rect ? sizeof(rectangle) : sizeof(vertex_values);
        void* allocation = alias_cpu ? alias_cpu + alias_vb_offset : slot<Allocate>(wrapped, 148)(wrapped, vertex_bytes, 16);
        if (!allocation) return false;
        const void* contents = rect ? static_cast<const void*>(rectangle) : vertex_values;
        std::memcpy(allocation, contents, vertex_bytes);
        const UINT bytes = vertex_bytes + (refusal == Refusal::indexed_vertex_range ? 4 : 0);
        const D3D12_VERTEX_BUFFER_VIEW vertex{reinterpret_cast<UINT64>(allocation), bytes,
            rect ? sizeof(RectVertex) : sizeof(std::uint32_t)};
        slot<VertexBuffer>(wrapped, 48)(wrapped, 0, 1, &vertex);
    }
    if (index_buffer) {
        const auto address = reinterpret_cast<UINT64>(index_buffer->address);
        const auto start = refusal == Refusal::invalid_index ? index_bytes : index_offset;
        const D3D12_INDEX_BUFFER_VIEW view{address + start, index_bytes - index_offset,
                                         draw_right ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT};
        slot<Index>(wrapped, 47)(wrapped, &view);
    }
    ComPtr<ID3D12CommandAllocator> foreign_allocator;
    ComPtr<IUnknown> foreign_list;
    std::array<std::uint32_t, 4> unregistered{};
    void* late_index_write = nullptr;
    if (alias_index) {
        constexpr std::uint16_t narrow_alias[]{99, 0, 1, 2};
        constexpr std::uint32_t wide_alias[]{99, 0, 1, 2};
        const UINT size = draw_right ? sizeof(wide_alias) : sizeof(narrow_alias);
        const UINT64 at = refusal == Refusal::alias_cross_page ? alias_page_bytes - 4 : alias_ib_offset;
        if (draw_right) std::memcpy(alias_cpu + at, wide_alias, size);
        else std::memcpy(alias_cpu + at, narrow_alias, size);
        const D3D12_INDEX_BUFFER_VIEW view{reinterpret_cast<UINT64>(alias_cpu + at), size,
            draw_right ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT};
        slot<Index>(wrapped, 47)(wrapped, &view);
    }
    if (upload_index) {
        using Allocate = void* (WINAPI*)(IUnknown*, UINT64, UINT64);
        const UINT element_size = draw_right ? sizeof(std::uint32_t) : sizeof(std::uint16_t);
        const UINT size = 4 * element_size;
        auto* owner = wrapped;
        if (refusal == Refusal::index_upload_foreign) {
            check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&foreign_allocator)),
                "create another recording allocator");
            foreign_list = begin_guest_list(guest, foreign_allocator.Get());
            owner = foreign_list.Get();
        }
        const UINT subview_offset = refusal == Refusal::index_upload_unaligned ? 2 : 4;
        const UINT allocation_size = subview_offset + size - (refusal == Refusal::index_upload_cross ? 2 : 0);
        auto* allocation = refusal == Refusal::index_unregistered ? reinterpret_cast<char*>(unregistered.data())
            : static_cast<char*>(slot<Allocate>(owner, 148)(owner, allocation_size, 16));
        if (!allocation) return false;
        // An invalid index is deliberately visible while ExecuteIndirect records.
        // The current recording upload is GPU-readable and the caller finalizes
        // the contents before submission.
        if (refusal == Refusal::none) {
            std::memset(allocation + subview_offset, 0xff, size);
            late_index_write = allocation + subview_offset;
        }
        const D3D12_INDEX_BUFFER_VIEW view{reinterpret_cast<UINT64>(allocation + subview_offset), size,
            draw_right ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT};
        slot<Index>(wrapped, 47)(wrapped, &view);
    }
    ComPtr<ID3D12RootSignature> retained_compute_root;
    ComPtr<ID3D12PipelineState> retained_compute_pipeline;
    ComPtr<ID3D12Resource> retained_result;
    ComPtr<ID3D12Resource> retained_readback;
    const UINT retained_value = draw_right ? 0xcafe0489 : 0x31ac1579;
    if (restore_compute) {
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        retained_compute_root = create_root(guest, {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE});
        const auto native_root = native_object<ID3D12RootSignature>(device, guest, retained_compute_root.Get());
        retained_compute_pipeline = writer_pipeline(device, native_root.Get(), "consume_retained_compute_root");
        retained_result = make_buffer(device, 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        retained_readback = make_buffer(device, 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        using Root = void (WINAPI*)(IUnknown*, IUnknown*);
        using Allocate = void* (WINAPI*)(IUnknown*, UINT64, UINT64);
        using Srv = void (WINAPI*)(IUnknown*, UINT, UINT64);
        slot<Root>(wrapped, 33)(wrapped, retained_compute_root.Get());
        auto* const input = static_cast<UINT*>(slot<Allocate>(wrapped, 148)(wrapped, 16, 16));
        if (!input) return false;
        *input = retained_value;
        slot<Srv>(wrapped, 43)(wrapped, 0, reinterpret_cast<UINT64>(input));
        if (draw_right) slot<Root>(wrapped, 33)(wrapped, nullptr);
        else if (kind == DrawKind::rect) slot<Root>(wrapped, 33)(wrapped, retained_compute_root.Get());
    }
    const UINT64 offset = refusal == Refusal::short_argument
        ? sizeof(initial_words) - command_stride(kind) + sizeof(std::uint32_t) : indexed_argument_offset;
    slot<Execute>(wrapped, 61)(wrapped, signature, 1, external_arguments ? leased->argument : arguments->wrapped.Get(),
                               external_arguments ? leased->argument_offset : offset, nullptr, 0);
    if (restore_compute) {
        // Leave the previously forwarded root + SRV alone. Only install the
        // consumer PSO and the output UAV that this fixture owns.
        transition(list, retained_result.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(retained_compute_pipeline.Get());
        list->SetComputeRootUnorderedAccessView(1, retained_result->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, retained_result.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(retained_readback.Get(), 0, retained_result.Get(), 0, 4);
    }
    if (late_index_write) {
        constexpr std::uint16_t narrow_upload[]{99, 0, 1, 2};
        constexpr std::uint32_t wide_upload[]{99, 0, 1, 2};
        if (draw_right) std::memcpy(late_index_write, wide_upload, sizeof(wide_upload));
        else std::memcpy(late_index_write, narrow_upload, sizeof(narrow_upload));
    }
    if (foreign_list) check(slot<HRESULT (WINAPI*)(IUnknown*)>(foreign_list.Get(), 9)(foreign_list.Get()),
        "close other recording");
    const HRESULT closed = slot<HRESULT (WINAPI*)(IUnknown*)>(wrapped, 9)(wrapped);
    if (FAILED(closed)) {
        ComPtr<ID3D12InfoQueue> messages;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&messages)))) {
            for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
                SIZE_T size = 0;
                if (FAILED(messages->GetMessage(i, nullptr, &size))) continue;
                std::vector<char> storage(size);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                if (SUCCEEDED(messages->GetMessage(i, message, &size))) std::fprintf(stderr, "%s\n", message->pDescription);
            }
        }
    }
    check(closed, "close guest command list");
    if (leased) {
        const auto queue = native_object<ID3D12CommandQueue>(device, guest, leased->queue);
        ComPtr<ID3D12Fence> completion;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion)), "create mesh consumer fence");
        if (leased->before_submit) leased->before_submit();
        submit_wrapped(leased->queue, wrapped);
        check(queue->Signal(completion.Get(), 1), "signal mesh consumer completion");
        if (leased->drop_recording) wrapped_list.Reset();
        if (leased->queued) leased->queued(completion.Get());
        check(mesh_fence_reached(completion.Get(), 5000) ? S_OK : E_FAIL, "wait for mesh consumer");
        check(device->GetDeviceRemovedReason(), "leased mesh GPU completion");
    } else {
        if (gate) {
            recording.unlock();
            if (!gate->meet(wrapped)) return false;
        }
        submit(device, list);
        if (gate) recording.lock();
    }

    if (restore_compute) {
        const D3D12_RANGE read{0, 4}, none{0, 0};
        void* mapped = nullptr;
        check(retained_readback->Map(0, &read, &mapped), "map restored compute result");
        UINT actual = 0;
        std::memcpy(&actual, mapped, 4);
        retained_readback->Unmap(0, &none);
        if (actual != retained_value) {
            std::fprintf(stderr, "restored compute readback=%08x expected=%08x\n", actual, retained_value);
            return false;
        }
        std::printf("restored compute root and one uploaded SRV: GPU readback %08x\n", actual);
    }
    if (translated) {
        constexpr Words empty{};
        if (std::memcmp(arguments->address, empty.data(), sizeof(empty))) {
            std::fprintf(stderr, "%s: CPU argument backing changed\n", label);
            return false;
        }
    }

    Words observed{};
    if (arguments) {
        const D3D12_RANGE arg_range{0, sizeof(initial)};
        check(arg_readback->Map(0, &arg_range, &memory), "map argument readback");
        std::memcpy(observed.data(), memory, sizeof(observed));
        arg_readback->Unmap(0, &untouched);
        Words expected = initial_words;
        if (writer) {
            const UINT word = indexed_argument_offset / sizeof(std::uint32_t);
            expected[word] = refusal == Refusal::overflowing_count ? 0xffffffffu
                           : refusal == Refusal::rect_bad_count ? 4
                           : !translated && !indexed && draw_right ? 6 : 3;
            expected[word + 1] = 1;
            if (indexed) expected[word + 2] = draw_right && !translated_indexed ? 4 : 1;
            if (translated_indexed && draw_right) {
                expected[word + 3] = refusal == Refusal::indexed_base_overrun ? 5 : 3;
                expected[word + 4] = 0;
            }
            if (mesh) {
                expected[1] = draw_right ? 2 : 1;
                expected[3] = 1;
            }
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (observed[i] == expected[i]) continue;
            std::fprintf(stderr, "%s: argument word %zu: got 0x%08x, expected 0x%08x\n", label, i, observed[i], expected[i]);
            return false;
        }
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    for (const auto& target : targets) {
        const auto pixel_readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        ComputeCommands copy(device, kept_direct_queue(device), nullptr);
        transition(copy.list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        D3D12_TEXTURE_COPY_LOCATION destination{pixel_readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        destination.PlacedFootprint = footprint;
        copy.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        copy.submit_and_wait(device);

        const D3D12_RANGE pixel_range{0, static_cast<SIZE_T>(bytes)};
        check(pixel_readback->Map(0, &pixel_range, &memory), "map pixel readback");
        UINT painted = 0;
        bool pixels_match = true;
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x) {
                std::uint32_t actual = 0;
                const auto pixel_bytes = cached_depth_only ? 2u : sizeof(actual);
                std::memcpy(&actual, static_cast<char*>(memory) + y * footprint.Footprint.RowPitch + x * pixel_bytes,
                            pixel_bytes);
                const bool right_half = x >= width / 2;
                const bool covered = (cached_depth_only && !cached_depth_pixel) ||
                    (indexed ? right_half == draw_right : translated || !right_half || draw_right);
                const bool drawn = writer && refusal == Refusal::none && covered;  // overflowing_count draws nothing
                const std::uint32_t background = cached_depth_only ? 0xffff : black;
                const auto depth_word = cached_depth_pixel ? 0x4000u : 0u; // D16_UNORM encodes SV_Depth 0.25.
                auto wanted = drawn ? (cached_depth_only ? depth_word :
                    translated ? (draw_right ? green : red) : (right_half ? green : red)) : background;
                if (leased) wanted = right_half ? leased->expected_right.value_or(leased->expected) : leased->expected;
                if (rect) {
                    // NDC pixel centers are x=(-.75,-.25,.25,.75), y=(.5,-.5).
                    // UV follows the clipped original rectangle, not the viewport/scissor.
                    constexpr UINT u[]{80, 112, 143, 175}, v[]{96, 159};
                    wanted = writer && refusal == Refusal::none && (x == 1 || x == 2)
                        ? (0xff350000 | (v[y] << 8) | u[x]) : black;
                }
                if (actual != background) ++painted;
                if (actual == wanted) continue;
                std::fprintf(stderr, "%s: pixel (%u,%u): got 0x%08x, expected 0x%08x\n", label, x, y, actual, wanted);
                pixels_match = false;
            }
        pixel_readback->Unmap(0, &untouched);
        std::printf("%s: %u/%u non-clear pixels%s\n", label, painted, width * height,
                    external_arguments ? "" : ", argument guards intact");
        if (!pixels_match) {
            if (kind == DrawKind::rect) report_rect_conversion(device, observed, indexed_argument_offset / sizeof(std::uint32_t));
            return false;
        }
    }
    return true;
}
// Calls refused before any argument is read: one per counted reason, and one with two commands, which is
// counted as other under its signature's kind. The list is closed and dropped.
void refuse_each_reason(ID3D12Device* guest, ID3D12CommandSignature* draw_indexed)
{
    D3D12_INDIRECT_ARGUMENT_DESC dispatch{};
    dispatch.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const D3D12_COMMAND_SIGNATURE_DESC dispatch_desc{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &dispatch, 0};
    ComPtr<ID3D12CommandSignature> dispatch_signature;
    check(guest->CreateCommandSignature(&dispatch_desc, nullptr, IID_PPV_ARGS(&dispatch_signature)), "create guest dispatch signature");
    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    const auto list = begin_guest_list(guest, allocator.Get());
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const auto execute = slot<Execute>(list.Get(), 61);
    execute(list.Get(), nullptr, 1, nullptr, 0, nullptr, 0);                    // placeholder signature
    execute(list.Get(), draw_indexed, 1, nullptr, 0, nullptr, 0);               // no desktop graphics pipeline bound
    execute(list.Get(), dispatch_signature.Get(), 1, nullptr, 0, nullptr, 0);   // no desktop compute pipeline bound
    execute(list.Get(), draw_indexed, 2, nullptr, 0, nullptr, 0);               // two commands
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(list.Get(), 9)(list.Get()), "close refusal list");
}

// 300 empty console presents (queue slot 32) reach the render-totals heartbeat; nothing is shown.
std::string indirect_line(ID3D12Device* guest, const std::wstring& log_path)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<IUnknown> queue;
    check(guest->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(queue.GetAddressOf())),
          "create guest queue");
    using Present = HRESULT (WINAPI*)(IUnknown*, UINT, const void*, const void*);
    const std::array<unsigned char, 0x50> planes{};
    const std::array<unsigned char, 0x40> present{};
    for (int frame = 0; frame < 300; ++frame) slot<Present>(queue.Get(), 32)(queue.Get(), 0, planes.data(), present.data());
    std::ifstream file(log_path, std::ios::binary);
    std::string line, found;
    while (std::getline(file, line))
        if (line.rfind("x2w-render indirect:", 0) == 0) found = line;
    while (!found.empty() && (found.back() == '\r' || found.back() == '\n')) found.pop_back();
    return found;
}

bool check_translated_refusal_census(const std::wstring& log_path)
{
    const std::size_t needed = x2w_d3d12_census_report(nullptr, 0);
    std::string report(needed + 1, '\0');
    x2w_d3d12_census_report(report.data(), report.size());
    report.resize(std::strlen(report.c_str()));
    std::string summary;
    std::vector<std::string> entries;
    std::size_t pos = 0;
    while (pos < report.size()) {
        std::size_t end = report.find('\n', pos);
        if (end == std::string::npos) end = report.size();
        std::string line = report.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("x2w-d3d12 indirect-draw refusals:", 0) == 0) summary = line;
        else if (line.rfind("x2w-d3d12 indirect-draw refusal ", 0) == 0) entries.push_back(line);
        pos = end + 1;
    }
    const std::string expected =
        "x2w-d3d12 indirect-draw refusals: total 6; distinct keys 3, distinct pairs 4 (overflow calls 0)";
    if (summary != expected) {
        std::fprintf(stderr, "indirect-draw refusal census mismatch:\n  got      '%s'\n  expected '%s'\n",
                     summary.c_str(), expected.c_str());
        return false;
    }
    if (entries.size() != 4) {
        std::fprintf(stderr, "indirect-draw refusal census expected 4 detail lines, got %zu\n", entries.size());
        return false;
    }
    struct ParsedEntry {
        unsigned index = 0;
        std::string key;
        std::string reason;
        unsigned long long calls = 0;
    };
    std::vector<ParsedEntry> parsed;
    for (const auto& line : entries) {
        unsigned index = 0;
        char key[32]{};
        char reason[64]{};
        unsigned long long calls = 0;
        if (std::sscanf(line.c_str(), "x2w-d3d12 indirect-draw refusal %u: key=%16[0-9a-f] reason=%63s calls=%llu",
                        &index, key, reason, &calls) != 4 || std::strlen(key) != 16) {
            std::fprintf(stderr, "malformed indirect-draw refusal detail line: '%s'\n", line.c_str());
            return false;
        }
        parsed.push_back({index, key, reason, calls});
    }
    const bool entries_ok =
        parsed[0].index == 1 && parsed[0].reason == "missing_guest_root" && parsed[0].calls == 2 &&
        parsed[1].index == 2 && parsed[1].key == parsed[0].key &&
        parsed[1].reason == "sampled_texture_mismatch" && parsed[1].calls == 1 &&
        parsed[2].index == 3 && parsed[2].key != parsed[0].key &&
        parsed[2].reason == "unsupported_resource_or_entry" && parsed[2].calls == 1 &&
        parsed[3].index == 4 && parsed[3].key != parsed[0].key && parsed[3].key != parsed[2].key &&
        parsed[3].reason == "incomplete_table_buffer" && parsed[3].calls == 2;
    if (!entries_ok) {
        std::fprintf(stderr, "indirect-draw refusal census entries did not match expected per-(key, reason) rows:\n");
        for (const auto& line : entries) std::fprintf(stderr, "  %s\n", line.c_str());
        return false;
    }
    // The same refusals ranked by pipeline key (the lines the tester pack reads): each key carries its own draws and sites.
    {
        const std::size_t keyed_needed = x2w_d3d12_not_drawn_keys_report(nullptr, 0);
        std::string keyed(keyed_needed + 1, '\0');
        x2w_d3d12_not_drawn_keys_report(keyed.data(), keyed.size());
        keyed.resize(std::strlen(keyed.c_str()));
        const auto line_of = [&](const std::string& key) {
            const std::size_t at = keyed.find("x2w-render not drawn key " + key + ":");
            return at == std::string::npos ? std::string() : keyed.substr(at, keyed.find('\n', at) - at);
        };
        const std::string one = line_of(parsed[0].key), two = line_of(parsed[2].key), three = line_of(parsed[3].key);
        const bool keyed_ok = keyed.rfind("x2w-render not drawn by pipeline key: ", 0) == 0 &&
            one.find(": 3 (") != std::string::npos && one.find("missing_guest_root 2") != std::string::npos &&
            one.find("sampled_texture_mismatch 1") != std::string::npos &&
            two.find(": 1 (unsupported_resource_or_entry 1)") != std::string::npos &&
            three.find(": 2 (incomplete_vertex_or_table_buffer 2)") != std::string::npos;
        if (!keyed_ok) {
            std::fprintf(stderr, "not-drawn-by-key lines did not match the per-key refusals:\n%s\n", keyed.c_str());
            return false;
        }
    }
    std::ifstream file(log_path, std::ios::binary);
    std::string log_line;
    bool saw_root_log = false, saw_tex_log = false;
    const std::string want_root = "x2w-render translated ExecuteIndirect Draw refused, shader " + parsed[0].key +
        ": translated draw requires the guest pipeline's root signature";
    const std::string want_tex = "x2w-render translated ExecuteIndirect Draw refused, shader " + parsed[0].key +
        ": translated draw requires a matching sampled texture distinct from the target";
    while (std::getline(file, log_line)) {
        if (!log_line.empty() && log_line.back() == '\r') log_line.pop_back();
        if (log_line == want_root) saw_root_log = true;
        if (log_line == want_tex) saw_tex_log = true;
    }
    if (!saw_root_log || !saw_tex_log) {
        std::fprintf(stderr, "indirect-draw refusal log lines missing for key %s (root=%d, texture=%d)\n",
                     parsed[0].key.c_str(), saw_root_log ? 1 : 0, saw_tex_log ? 1 : 0);
        return false;
    }
    std::printf("%s\n", summary.c_str());
    for (const auto& line : entries) std::printf("%s\n", line.c_str());
    return true;
}

bool check_translated_targets(ID3D12Device* device, ID3D12Device* guest, ID3D12RootSignature* writer_root,
                              ID3D12PipelineState* writer, ID3D12RootSignature* title_root, IUnknown* pso,
                              ID3D12CommandSignature* signature, const TranslatedDrawFixture& red_binding)
{
    const TranslatedDrawFixture alias_binding(guest, red, nullptr, false, false, false, true);
    enum class Mode { mrt_two, color_and_dsv, depth_only, no_attachments, alias_rtv1 };
    const auto run_mode = [&](Mode mode, const char* label) -> bool {
        const auto& binding = mode == Mode::alias_rtv1 ? alias_binding : red_binding;
        PlacedBuffer arguments(device, guest, static_cast<UINT>(sizeof(initial)), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Words initial_words = initial;
        initial_words[1 + command_stride(DrawKind::vertices) / sizeof(std::uint32_t)] = 0x3c3c3c3c;
        const auto upload = make_buffer(device, sizeof(initial_words), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* memory = nullptr;
        const D3D12_RANGE untouched{0, 0};
        check(upload->Map(0, &untouched, &memory), "map targets upload");
        std::memcpy(memory, initial_words.data(), sizeof(initial_words));
        upload->Unmap(0, nullptr);

        const D3D12_RESOURCE_DESC color_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0},
            D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
        const D3D12_RESOURCE_DESC depth_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_D16_UNORM, {1, 0},
            D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
        constexpr float clear[]{0, 0, 0, 1};
        D3D12_CLEAR_VALUE color_clear{format, {0, 0, 0, 1}};
        D3D12_CLEAR_VALUE depth_clear{DXGI_FORMAT_D16_UNORM, {0, 0, 0, 1}};
        depth_clear.DepthStencil = {1.0f, 0};

        const auto color0 = resource(device, color_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &color_clear);
        const auto color1 = resource(device, color_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &color_clear);
        const auto depth = resource(device, depth_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &depth_clear);

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        ComPtr<ID3D12DescriptorHeap> rtv_heap;
        check(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap)), "create targets rtv heap");
        const UINT rtv_inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2]{
            rtv_heap->GetCPUDescriptorHandleForHeapStart(),
            {rtv_heap->GetCPUDescriptorHandleForHeapStart().ptr + rtv_inc}};
        guest->CreateRenderTargetView(color0.Get(), nullptr, rtvs[0]);
        ID3D12Resource* rtv1_res = mode == Mode::alias_rtv1 ? binding.sampled.Get() : color1.Get();
        guest->CreateRenderTargetView(rtv1_res, nullptr, rtvs[1]);

        const D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        ComPtr<ID3D12DescriptorHeap> dsv_heap;
        check(guest->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap)), "create targets dsv heap");
        using GetCpu = UINT64(WINAPI*)(IUnknown*, void*);
        D3D12_CPU_DESCRIPTOR_HANDLE guest_dsv{slot<GetCpu>(dsv_heap.Get(), 9)(dsv_heap.Get(), nullptr)};
        const D3D12_CPU_DESCRIPTOR_HANDLE native_dsv =
            native_object<ID3D12DescriptorHeap>(device, guest, dsv_heap.Get())->GetCPUDescriptorHandleForHeapStart();
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc{};
        dsv_desc.Format = DXGI_FORMAT_D16_UNORM;
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        guest->CreateDepthStencilView(depth.Get(), &dsv_desc, guest_dsv);

        ComPtr<ID3D12CommandAllocator> allocator;
        check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create targets allocator");
        auto wrapped_list = begin_guest_list(guest, allocator.Get());
        auto* wrapped = wrapped_list.Get();
        const auto native_list = native_object<ID3D12GraphicsCommandList>(device, guest, wrapped);
        auto* list = native_list.Get();

        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(arguments.native.Get(), 0, upload.Get(), 0, sizeof(initial_words));
        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(writer);
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, arguments.native->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

        using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
        using Clear = void (WINAPI*)(IUnknown*, SIZE_T, const float*, UINT, const D3D12_RECT*);
        using Targets = void (WINAPI*)(IUnknown*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL, const D3D12_CPU_DESCRIPTOR_HANDLE*);
        using Topology = void (WINAPI*)(IUnknown*, D3D12_PRIMITIVE_TOPOLOGY);
        using Viewport = void (WINAPI*)(IUnknown*, UINT, const D3D12_VIEWPORT*);
        using Scissor = void (WINAPI*)(IUnknown*, UINT, const D3D12_RECT*);
        using Heaps = void (WINAPI*)(IUnknown*, UINT, ID3D12DescriptorHeap* const*);
        using Table = void (WINAPI*)(IUnknown*, UINT, UINT64);
        using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);

        slot<Clear>(wrapped, 52)(wrapped, rtvs[0].ptr, clear, 0, nullptr);
        if (mode == Mode::mrt_two) slot<Clear>(wrapped, 52)(wrapped, rtvs[1].ptr, clear, 0, nullptr);
        if (mode == Mode::mrt_two || mode == Mode::alias_rtv1)
            slot<Targets>(wrapped, 50)(wrapped, 2, rtvs, FALSE, nullptr);
        else if (mode == Mode::color_and_dsv) {
            slot<Targets>(wrapped, 50)(wrapped, 1, rtvs, FALSE, &guest_dsv);
            list->ClearDepthStencilView(native_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        } else if (mode == Mode::depth_only) {
            slot<Targets>(wrapped, 50)(wrapped, 0, nullptr, FALSE, &guest_dsv);
            list->ClearDepthStencilView(native_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        } else {
            slot<Targets>(wrapped, 50)(wrapped, 0, nullptr, FALSE, nullptr);
        }

        ComPtr<ID3D12QueryHeap> query_heap;
        ComPtr<ID3D12Resource> query_readback;
        if (mode == Mode::depth_only) {
            const D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 0};
            check(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&query_heap)), "create depth-only occlusion query heap");
            query_readback = make_buffer(device, sizeof(std::uint64_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        }

        const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
        const D3D12_RECT scissor{0, 0, width, height};
        slot<Bind>(wrapped, 25)(wrapped, pso);
        slot<Bind>(wrapped, 34)(wrapped, title_root);
        ID3D12DescriptorHeap* heaps[]{binding.heap.Get()};
        slot<Heaps>(wrapped, 32)(wrapped, 1, heaps);
        slot<Table>(wrapped, 36)(wrapped, 0, binding.gpu);
        slot<Topology>(wrapped, 20)(wrapped, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        slot<Viewport>(wrapped, 21)(wrapped, 1, &viewport);
        slot<Scissor>(wrapped, 22)(wrapped, 1, &scissor);
        if (mode == Mode::depth_only) list->BeginQuery(query_heap.Get(), D3D12_QUERY_TYPE_OCCLUSION, 0);
        slot<Execute>(wrapped, 61)(wrapped, signature, 1, arguments.wrapped.Get(), argument_offset, nullptr, 0);
        if (mode == Mode::depth_only) {
            list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_OCCLUSION, 0);
            list->ResolveQueryData(query_heap.Get(), D3D12_QUERY_TYPE_OCCLUSION, 0, 1, query_readback.Get(), 0);
        }
        check(slot<HRESULT(WINAPI*)(IUnknown*)>(wrapped, 9)(wrapped), "close targets list");

        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        ComPtr<IUnknown> guest_queue;
        check(guest->CreateCommandQueue(&queue_desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(guest_queue.GetAddressOf())),
              "create targets queue");
        IUnknown* lists[]{wrapped};
        slot<void (WINAPI*)(IUnknown*, UINT, IUnknown* const*)>(guest_queue.Get(), 10)(guest_queue.Get(), 1, lists);
        const auto native_queue = native_object<ID3D12CommandQueue>(device, guest, guest_queue.Get());
        ComPtr<ID3D12Fence> fence;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create targets fence");
        check(native_queue->Signal(fence.Get(), 1), "signal targets fence");
        HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        check(fence->SetEventOnCompletion(1, done), "set targets fence event");
        WaitForSingleObject(done, INFINITE);
        CloseHandle(done);

        if (mode == Mode::depth_only) {
            std::uint64_t samples = 0;
            const D3D12_RANGE query_range{0, sizeof(samples)};
            check(query_readback->Map(0, &query_range, &memory), "map depth-only occlusion query");
            std::memcpy(&samples, memory, sizeof(samples));
            query_readback->Unmap(0, &untouched);
            const auto expected_samples = static_cast<std::uint64_t>(width) * height;
            if (samples != expected_samples) {
                std::fprintf(stderr, "%s: expected %llu rasterized samples, got %llu\n",
                             label, static_cast<unsigned long long>(expected_samples),
                             static_cast<unsigned long long>(samples));
                return false;
            }
        }

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 bytes = 0;
        device->GetCopyableFootprints(&color_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        const auto read_pixels = [&](ID3D12Resource* target, std::uint32_t expected_colour, const char* which) -> bool {
            const auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            ComputeCommands copy(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
            transition(copy.list.Get(), target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION src{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            dst.PlacedFootprint = footprint;
            copy.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            copy.submit_and_wait(device);
            const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
            check(readback->Map(0, &range, &memory), "map targets readback");
            bool ok = true;
            for (UINT y = 0; y < height; ++y)
                for (UINT x = 0; x < width; ++x) {
                    std::uint32_t actual = 0;
                    std::memcpy(&actual, static_cast<char*>(memory) + y * footprint.Footprint.RowPitch + x * sizeof(actual), sizeof(actual));
                    if (actual != expected_colour) {
                        std::fprintf(stderr, "%s (%s) pixel (%u,%u): got 0x%08x, expected 0x%08x\n",
                                     label, which, x, y, actual, expected_colour);
                        ok = false;
                    }
                }
            readback->Unmap(0, &untouched);
            return ok;
        };
        const bool expect_draw = mode == Mode::mrt_two || mode == Mode::color_and_dsv;
        if (!read_pixels(color0.Get(), expect_draw ? red : black, "rtv0")) return false;
        if (mode == Mode::mrt_two && !read_pixels(color1.Get(), black, "rtv1")) return false;
        std::printf("%s: OK\n", label);
        return true;
    };
    const bool ok =
        run_mode(Mode::mrt_two, "translated indirect MRT (2 color targets)") &&
        run_mode(Mode::color_and_dsv, "translated indirect color + DSV") &&
        run_mode(Mode::depth_only, "translated indirect depth-only (0 RTV + DSV)") &&
        run_mode(Mode::no_attachments, "translated indirect refuses 0 RTV + no DSV") &&
        run_mode(Mode::alias_rtv1, "translated indirect refuses sampled texture aliasing RTV 1");
    if (!ok) return false;
    const std::size_t needed = x2w_d3d12_census_report(nullptr, 0);
    std::string report(needed + 1, '\0');
    x2w_d3d12_census_report(report.data(), report.size());
    if (report.find("reason=sampled_texture_mismatch calls=1") == std::string::npos) {
        std::fprintf(stderr, "expected sampled_texture_mismatch refusal for RTV 1 alias in census report:\n%s\n", report.c_str());
        return false;
    }
    return true;
}

bool check_indirect_other_refusal_census(const std::wstring& log_path)
{
    const std::size_t needed = x2w_d3d12_census_report(nullptr, 0);
    std::string report(needed + 1, '\0');
    x2w_d3d12_census_report(report.data(), report.size());
    report.resize(std::strlen(report.c_str()));
    std::string summary;
    std::vector<std::string> entries;
    bool saw_indirect_draw = false;
    std::size_t pos = 0;
    while (pos < report.size()) {
        std::size_t end = report.find('\n', pos);
        if (end == std::string::npos) end = report.size();
        std::string line = report.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("x2w-d3d12 indirect-other refusals:", 0) == 0) summary = line;
        else if (line.rfind("x2w-d3d12 indirect-other refusal ", 0) == 0) entries.push_back(line);
        else if (line.rfind("x2w-d3d12 indirect-draw refusal", 0) == 0) saw_indirect_draw = true;
        pos = end + 1;
    }
    if (saw_indirect_draw) {
        std::fprintf(stderr, "indirect-other refusal census saw unexpected indirect-draw refusal lines\n");
        return false;
    }
    const std::string expected =
        "x2w-d3d12 indirect-other refusals: total 4; distinct keys 2, distinct pairs 3 (overflow calls 0)";
    if (summary != expected) {
        std::fprintf(stderr, "indirect-other refusal census mismatch:\n  got      '%s'\n  expected '%s'\n",
                     summary.c_str(), expected.c_str());
        return false;
    }
    if (entries.size() != 3) {
        std::fprintf(stderr, "indirect-other refusal census expected 3 detail lines, got %zu\n", entries.size());
        return false;
    }
    struct ParsedEntry {
        unsigned index = 0;
        std::string key;
        std::string reason;
        unsigned long long calls = 0;
    };
    std::vector<ParsedEntry> parsed;
    for (const auto& line : entries) {
        unsigned index = 0;
        char key[32]{};
        char reason[64]{};
        unsigned long long calls = 0;
        if (std::sscanf(line.c_str(), "x2w-d3d12 indirect-other refusal %u: key=%16[0-9a-f] reason=%63s calls=%llu",
                        &index, key, reason, &calls) != 4 || std::strlen(key) != 16) {
            std::fprintf(stderr, "malformed indirect-other refusal detail line: '%s'\n", line.c_str());
            return false;
        }
        parsed.push_back({index, key, reason, calls});
    }
    const std::string zero_key = "0000000000000000";
    const bool entries_ok =
        parsed[0].index == 1 && parsed[0].key != zero_key &&
        parsed[0].reason == "missing_viewport_or_scissor" && parsed[0].calls == 2 &&
        parsed[1].index == 2 && parsed[1].key == parsed[0].key &&
        parsed[1].reason == "argument_outside_placed_buffer" && parsed[1].calls == 1 &&
        parsed[2].index == 3 && parsed[2].key == zero_key &&
        parsed[2].reason == "graphics_multi_command" && parsed[2].calls == 1;
    if (!entries_ok) {
        std::fprintf(stderr, "indirect-other refusal census entries did not match expected per-(key, reason) rows:\n");
        for (const auto& line : entries) std::fprintf(stderr, "  %s\n", line.c_str());
        return false;
    }
    std::ifstream file(log_path, std::ios::binary);
    std::string log_line;
    bool saw_vp_log = false, saw_arg_log = false, saw_multi_log = false;
    const std::string want_vp = "x2w-render ExecuteIndirect refused, key " + parsed[0].key +
        " [missing_viewport_or_scissor]: translated indirect draw requires bound target or depth view, viewport and scissor";
    const std::string want_arg = "x2w-render ExecuteIndirect refused, key " + parsed[0].key +
        " [argument_outside_placed_buffer]: argument record is outside a unique placed buffer";
    const std::string want_multi = "x2w-render ExecuteIndirect refused, key " + zero_key +
        " [graphics_multi_command]: indirect graphics requires one command and no count buffer";
    while (std::getline(file, log_line)) {
        if (!log_line.empty() && log_line.back() == '\r') log_line.pop_back();
        if (log_line == want_vp) saw_vp_log = true;
        if (log_line == want_arg) saw_arg_log = true;
        if (log_line == want_multi) saw_multi_log = true;
    }
    if (!saw_vp_log || !saw_arg_log || !saw_multi_log) {
        std::fprintf(stderr, "indirect-other refusal log lines missing (vp=%d, arg=%d, multi=%d)\n",
                     saw_vp_log ? 1 : 0, saw_arg_log ? 1 : 0, saw_multi_log ? 1 : 0);
        return false;
    }
    std::printf("%s\n", summary.c_str());
    for (const auto& line : entries) std::printf("%s\n", line.c_str());
    return true;
}
bool fixed_mesh_create_contract(ID3D12Device* guest)
{
    const D3D12_ROOT_SIGNATURE_DESC root_desc{};
    const auto root = create_root(guest, root_desc);
    const std::array<UINT, 7> ms{0xbfa00003, 0xbf900009, 0xf80008cf, 0x03020100, 0xf8000941, 0, 0xbf810000};
    const std::array<UINT, 4> ps{0xbfa00003, 0xf800080f, 0x03020100, 0xbf810000};
    using Fields = std::array<UINT64, 0x300 / sizeof(UINT64)>;
    Fields pair{};
    pair[0] = reinterpret_cast<UINT64>(root.Get());
    pair[5] = reinterpret_cast<UINT64>(ps.data());
    pair[6] = sizeof(ps);
    pair[13] = reinterpret_cast<UINT64>(ms.data());
    pair[14] = sizeof(ms);
    Fields mesh_only = pair;
    mesh_only[5] = mesh_only[6] = 0;
    Fields legacy = pair;
    legacy[9] = reinterpret_cast<UINT64>(ps.data());
    legacy[10] = sizeof(ps);
    legacy[13] = legacy[14] = 0;
    const auto pipeline_iid = __uuidof(ID3D12PipelineState);
    const auto wrong_iid = __uuidof(ID3D12Resource);
    using Create = HRESULT (WINAPI*)(IUnknown*, const void*, const GUID*, void**);
    const auto create = slot<Create>(guest, 57);
    const auto expect = [&](const char* label, const void* desc, const GUID& iid, HRESULT expected) {
        void* result = guest;
        const HRESULT hr = create(guest, desc, &iid, &result);
        const bool output_ok = expected == S_OK ? result && result != guest : !result;
        if (result && result != guest) static_cast<IUnknown*>(result)->Release();
        if (hr == expected && output_ok) return true;
        std::fprintf(stderr, "%s: got HRESULT 0x%08lx with %s output, expected 0x%08lx\n",
            label, static_cast<unsigned long>(hr), output_ok ? "expected" : "unexpected",
            static_cast<unsigned long>(expected));
        return false;
    };
    check(SetEnvironmentVariableA("X2W_REFUSE_FIXED_MESH", nullptr) ? S_OK : E_FAIL, "clear fixed mesh switch");
    if (!expect("default MS+PS", pair.data(), pipeline_iid, S_OK) ||
        !expect("default MS-only", mesh_only.data(), pipeline_iid, S_OK) ||
        !expect("default wrong IID", pair.data(), wrong_iid, E_NOINTERFACE)) return false;
    check(SetEnvironmentVariableA("X2W_REFUSE_FIXED_MESH", "1") ? S_OK : E_FAIL, "enable fixed mesh switch");
    if (!expect("opt-in MS+PS", pair.data(), pipeline_iid, E_NOTIMPL) ||
        !expect("opt-in MS-only", mesh_only.data(), pipeline_iid, E_NOTIMPL) ||
        !expect("opt-in wrong IID takes precedence", pair.data(), wrong_iid, E_NOINTERFACE) ||
        !expect("ordinary VS+PS", legacy.data(), pipeline_iid, S_OK) ||
        !expect("null description keeps legacy result", nullptr, pipeline_iid, S_OK) ||
        !expect("unreadable description keeps legacy result", reinterpret_cast<const void*>(1), pipeline_iid, S_OK))
        return false;
    const auto field = [&](const char* label, std::size_t index, UINT64 value) {
        auto invalid = pair;
        invalid[index] = value;
        return expect(label, invalid.data(), pipeline_iid, S_OK);
    };
    auto wrong_root = pair;
    wrong_root[0] = reinterpret_cast<UINT64>(guest);
    if (!expect("registered object with the wrong root kind", wrong_root.data(), pipeline_iid, S_OK) ||
        !field("missing root", 0, 0) ||
        !field("populated VS pointer", 9, pair[13]) ||
        !field("populated VS length", 10, sizeof(ms)) ||
        !field("missing MS pointer", 13, 0) ||
        !field("unreadable MS span", 13, 1) ||
        !field("oversized MS span", 14, 0x10001) ||
        !field("truncated MS span", 14, 15) ||
        !field("missing PS pointer", 5, 0) ||
        !field("missing PS length", 6, 0) ||
        !field("unreadable PS span", 5, 1) ||
        create(guest, pair.data(), &pipeline_iid, nullptr) != E_POINTER) return false;
    std::puts("slot 57 creation result and output matched all controls");
    return true;
}
#include "startup_indirect_mesh_lease.inl"
}  // namespace

int main(int argc, char** argv) try
{
    DrawKind kind = DrawKind::indexed;
    const bool lease_writer = argc == 3 && std::strcmp(argv[2], "mesh-lease-writer") == 0;
    const bool skip_argument_wait = argc == 3 && std::strcmp(argv[2], "mesh-xmem-arguments-skip-wait") == 0;
    const bool lease_arguments = skip_argument_wait || (argc == 3 && std::strcmp(argv[2], "mesh-xmem-arguments") == 0);
    if (lease_arguments) {
        check(SetEnvironmentVariableA("X2W_MESH_INDIRECT_XMEM", "1") ? S_OK : E_FAIL, "enable XMem mesh arguments");
        check(SetEnvironmentVariableA("X2W_TEST_MESH_XMEM_SKIP_WAIT", skip_argument_wait ? "1" : nullptr) ? S_OK : E_FAIL,
              "configure the XMem argument wait control");
    }
    wchar_t lease_xmem[MAX_PATH]{};
    const bool lease_fixture = GetEnvironmentVariableW(L"X2W_MESH86_FIXTURE_XMEM", lease_xmem, MAX_PATH) != 0;
    const bool dual_lease = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_DUAL", nullptr, 0) != 0;
    if (dual_lease) check(lease_fixture && !lease_arguments ? S_OK : E_FAIL, "configure combined mesh XMem fixture");
    const bool loose_root = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_LOOSE_ROOT", nullptr, 0) != 0;
    if (loose_root)
        check(lease_fixture && SetEnvironmentVariableA("X2W_MESH86_LOOSE_XMEM_SRV", "1") &&
            SetEnvironmentVariableA("X2W_MESH86_XMEM_SRV", nullptr) ? S_OK : E_FAIL, "enable loose XMem root SRV");
    else if (lease_fixture && !lease_arguments)
        check(SetEnvironmentVariableA("X2W_MESH86_XMEM_SRV", "1") ? S_OK : E_FAIL, "enable mesh XMem handoff");
    const bool console_mesh = argc == 3 && std::strcmp(argv[2], "mesh-console") == 0;
    const bool desktop_as = argc == 3 && std::strcmp(argv[2], "mesh-amplified") == 0;
    const bool copy_on = argc == 3 && std::strcmp(argv[2], "copy-provenance") == 0;
    const bool copy_off = argc == 3 && std::strcmp(argv[2], "copy-provenance-off") == 0;
    const bool refusal_census_control = argc == 3 && std::strcmp(argv[2], "translated-refusal-census-control") == 0;
    const bool refusal_census = refusal_census_control || (argc == 3 && std::strcmp(argv[2], "translated-refusal-census") == 0);
    const bool translated_targets = argc == 3 && std::strcmp(argv[2], "translated-targets") == 0;
    const bool other_refusal_census_control = argc == 3 && std::strcmp(argv[2], "indirect-other-refusal-census-control") == 0;
    const bool other_refusal_census = other_refusal_census_control || (argc == 3 && std::strcmp(argv[2], "indirect-other-refusal-census") == 0);
    const bool root_fix_control = argc == 3 && std::strcmp(argv[2], "translated-root-control") == 0;
    const bool root_fix = root_fix_control || (argc == 3 && std::strcmp(argv[2], "translated-root") == 0);
    const bool upload_index = argc == 3 && std::strcmp(argv[2], "translated-indexed-upload") == 0;
    const bool rect = argc == 3 && std::strcmp(argv[2], "translated-indexed-rect") == 0;
    const bool alias_disabled = argc == 3 && std::strcmp(argv[2], "translated-indexed-alias-off") == 0;
    const bool alias_index = alias_disabled || (argc == 3 && std::strcmp(argv[2], "translated-indexed-alias") == 0);
    const bool translated_indexed = alias_index || upload_index || rect || (argc == 3 && std::strcmp(argv[2], "translated-indexed") == 0);
    const bool translated = root_fix || copy_on || copy_off || refusal_census || other_refusal_census || translated_targets || translated_indexed || (argc == 3 && std::strcmp(argv[2], "translated") == 0);
    // "count-control" is the negative control: it expects one forwarded call too many.
    const bool count_control = argc == 3 && std::strcmp(argv[2], "count-control") == 0;
    if ((translated && !translated_indexed) || (argc == 3 && std::strcmp(argv[2], "draw") == 0)) kind = DrawKind::vertices;
    else if (argc == 3 && std::strcmp(argv[2], "rect") == 0) kind = DrawKind::rect;
    else if (lease_arguments || console_mesh || desktop_as || (argc == 3 && std::strcmp(argv[2], "mesh") == 0))
        kind = DrawKind::mesh;
    else if (rect) kind = DrawKind::indexed_rect;
    else if (argc != 2 && !count_control && !translated_indexed && !lease_writer) return 2;
    const bool indexed = kind == DrawKind::indexed || kind == DrawKind::indexed_rect;
    const bool mesh = kind == DrawKind::mesh;
    const bool fixed_create_fixture = mesh && GetEnvironmentVariableA("X2W_FIXED_MESH_CREATE_FIXTURE", nullptr, 0);
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (!warp && std::strcmp(argv[1], "hardware") != 0) return 2;
    ComPtr<ID3D12Debug> debug;
    const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "get debug layer");
    if (debug) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "find WARP");
    } else {
        for (UINT index = 0;; ++index) {
            const HRESULT hr = factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) {
                std::puts("indirect graphics capability skip: no compatible physical adapter");
                return 77;
            }
            check(hr, "enumerate physical adapters");
            DXGI_ADAPTER_DESC1 desc{};
            check(adapter->GetDesc1(&desc), "get physical adapter description");
            const bool basic_render_driver = desc.VendorId == 0x1414 && desc.DeviceId == 0x008c;
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || basic_render_driver) continue;
            ComPtr<ID3D12Device> probe;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe))) &&
                (!mesh || fixed_create_fixture || supports_mesh(probe.Get()))) break;
        }
    }
    wchar_t directory[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, directory)) return 4;
    const std::wstring log_path = std::wstring(directory) + L"x2w_indirect_counts_" + std::to_wstring(GetCurrentProcessId()) + L".log";
    if (indexed || lease_arguments || dual_lease || loose_root || refusal_census || other_refusal_census) {
        DeleteFileW(log_path.c_str());
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
        SetEnvironmentVariableA("X2W_PRESENT_SIGNATURE_EVERY", "0");
    }
    DXGI_ADAPTER_DESC1 desc{};
    check(adapter->GetDesc1(&desc), "get selected adapter description");
    std::printf("adapter: %ls, debug layer %s\n", desc.Description, debug ? "enabled" : "unavailable");
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver: %u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                    HIWORD(driver.LowPart), LOWORD(driver.LowPart));

    ComPtr<ID3D12Device> device, guest;
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create desktop device through d3d12_xs");
    if (mesh && !fixed_create_fixture && !supports_mesh(device.Get())) {
        std::puts("indirect mesh capability skip: mesh shaders unavailable");
        return 77;
    }
    check(SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable guest");
    check(SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable guest list");
    check(SetEnvironmentVariableA("X2W_D3D12_RENDER", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable recording");
    const std::array<unsigned char, 64> parameters{};
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    if (fixed_create_fixture) return fixed_mesh_create_contract(guest.Get()) ? 0 : 1;
    ComPtr<ID3D12InfoQueue> info;
    if (debug) check(device.As(&info), "get debug message queue");
    if ((lease_fixture || lease_arguments) && !supports_lease_heap(device.Get())) {
        std::puts("mesh XMem capability skip: existing heaps unavailable");
        return 77;
    }
    if (lease_writer) {
        if (!lease_fixture) return 2;
        MeshLeaseWriter source(device.Get(), guest.Get(), lease_xmem);
        source.start();
        source.finish();
        source.free_original();
        source.check_retirement("completed writer releases the first XMem slot");
        check_lease_debug(info.Get());
        return 0;
    }
    const D3D12_ROOT_PARAMETER uav{D3D12_ROOT_PARAMETER_TYPE_UAV, {}, D3D12_SHADER_VISIBILITY_ALL};
    const D3D12_ROOT_SIGNATURE_DESC writer_desc{1, &uav, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_flags = mesh ? D3D12_ROOT_SIGNATURE_FLAG_NONE : D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const D3D12_ROOT_SIGNATURE_DESC graphics_desc{0, nullptr, 0, nullptr, root_flags};
    const auto writer = create_root(device.Get(), writer_desc);
    wchar_t mesh86_dir[MAX_PATH]{};
    if (mesh && GetEnvironmentVariableW(L"X2W_MESH86_FIXTURE_DIR", mesh86_dir, MAX_PATH)) {
        const bool depth_only = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_DEPTH", nullptr, 0) != 0;
        char depth_mode[16]{};
        const bool depth_pixel = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_DEPTH_PIXEL", depth_mode, sizeof(depth_mode)) != 0;
        auto depth = depth_only ? StreamDepth::writable : StreamDepth::none;
        if (std::strcmp(depth_mode, "no-dsv") == 0) depth = StreamDepth::none;
        if (std::strcmp(depth_mode, "read-only") == 0) depth = StreamDepth::read_only;
        const auto input = std::filesystem::path(mesh86_dir);
        const auto read_stage = [&](const char* name) {
            std::ifstream file(input / name, std::ios::binary);
            check(file ? S_OK : E_FAIL, name);
            return std::vector<char>(std::istreambuf_iterator<char>(file), {});
        };
        const auto ms = read_stage("source.ms.dxbc");
        const auto ps = depth_only && !depth_pixel ? std::vector<char>{} : read_stage("source.ps.dxbc");
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = lease_fixture ? D3D12_ROOT_PARAMETER_TYPE_SRV : D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameter.ShaderVisibility = lease_fixture ? D3D12_SHADER_VISIBILITY_MESH : D3D12_SHADER_VISIBILITY_ALL;
        if (!lease_fixture) parameter.Constants = {0, 0, 2};
        const D3D12_ROOT_SIGNATURE_DESC root_desc{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root = create_root(guest.Get(), root_desc);
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        const D3D12_COMMAND_SIGNATURE_DESC signature_desc{sizeof(D3D12_DISPATCH_MESH_ARGUMENTS), 1, &argument, 0};
        ComPtr<ID3D12CommandSignature> signature;
        check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create cached mesh signature");
        const D3D12_SHADER_BYTECODE ms_code{ms.data(), ms.size()};
        const D3D12_SHADER_BYTECODE ps_code{ps.empty() ? nullptr : ps.data(), ps.size()};
        const auto pso = lease_fixture
            ? guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS>(guest.Get(), root.Get(), ms_code, ps_code, {}, 1, depth)
            : guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, true>(guest.Get(), root.Get(), ms_code, ps_code, {},
                                                                         depth_only ? 0 : 3, depth);
        if (dual_lease) {
            const bool init_only = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_INIT_ONLY", nullptr, 0) != 0;
            const auto initialize = writer_pipeline(device.Get(), writer.Get(), "initialize_mesh_record");
            ComPtr<ID3D12PipelineState> append;
            if (!init_only) append = writer_pipeline(device.Get(), writer.Get(), "append_mesh_groups");
            return check_mesh_dual_xmem(device.Get(), guest.Get(), lease_xmem, info.Get(), log_path,
                                        writer.Get(), initialize.Get(), append.Get(), root.Get(), pso.Get(), signature.Get()) ? 0 : 1;
        }
        const auto work = writer_pipeline(device.Get(), writer.Get(), "write_one_mesh_group");
        if (loose_root) {
            const bool ready = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_READY", nullptr, 0) != 0;
            MeshLeaseWriter source(device.Get(), guest.Get(), lease_xmem, nullptr, nullptr, nullptr, nullptr, true);
            const auto queue = source.create_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
            const auto verify = [&](const MeshLeaseDraw& draw, const char* label) {
                return run(device.Get(), guest.Get(), writer.Get(), work.Get(), root.Get(), pso.Get(), signature.Get(),
                    DrawKind::mesh, false, Refusal::none, label, nullptr, false, nullptr, false, false, 1, false, false,
                    &draw);
            };
            if (!ready) {
                const bool ok = verify({source.address(), queue.Get(), black}, "loose mesh requires shader proof");
                source.free_original();
                return ok ? 0 : 1;
            }
            if (!verify({source.address(), queue.Get(), green}, "loose mesh reads ordinary XMem root SRV") ||
                !verify({source.address() + 4, queue.Get(), black}, "loose mesh refuses misaligned root") ||
                !verify({source.address() + 2 * MeshLeaseWriter::page, queue.Get(), black}, "loose mesh refuses unmapped page"))
                return 1;
            const std::unique_ptr<void, FreeMemory> unrelated(
                VirtualAlloc(nullptr, MeshLeaseWriter::page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            if (!unrelated || !verify({reinterpret_cast<UINT64>(unrelated.get()), queue.Get(), black},
                "loose mesh refuses readable untracked memory")) return 1;
            const auto registered_desc = D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_BUFFER, 0, MeshLeaseWriter::page,
                1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
            using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
            ComPtr<IUnknown> placed;
            check(slot<Place>(guest.Get(), 133)(guest.Get(), source.cpu, &registered_desc, D3D12_RESOURCE_STATE_COMMON,
                nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(placed.GetAddressOf())), "register overlapping buffer");
            if (!verify({source.address(), queue.Get(), black}, "loose mesh refuses registered resource without a publisher")) return 1;
            placed.Reset();

            MeshLeaseDraw late{source.address(), queue.Get(), red};
            late.before_submit = [&] { std::memcpy(source.cpu + source.record_offset, &red, sizeof(red)); };
            if (!verify(late, "loose mesh sees a CPU write after Close")) return 1;
            // The queue rejects the entire stale recording, so use already
            // initialized external DispatchMesh arguments for this control.
            PlacedBuffer args(device.Get(), guest.Get(), sizeof(MeshLeaseWriter::old_groups), D3D12_RESOURCE_FLAG_NONE);
            const auto upload = make_buffer(device.Get(), sizeof(MeshLeaseWriter::old_groups), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ);
            const D3D12_RANGE empty{};
            void* memory{};
            check(upload->Map(0, &empty, &memory), "map loose-root dispatch arguments");
            std::memcpy(memory, MeshLeaseWriter::old_groups.data(), sizeof(MeshLeaseWriter::old_groups));
            upload->Unmap(0, nullptr);
            ComputeCommands seed(device.Get(), nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
            seed.list->CopyBufferRegion(args.native.Get(), 0, upload.Get(), 0, sizeof(MeshLeaseWriter::old_groups));
            seed.submit_and_wait(device.Get());
            MeshLeaseDraw remap{source.address(), queue.Get(), black};
            remap.argument = args.wrapped.Get();
            remap.before_submit = [&] { source.replace_loose_page(); };
            if (!verify(remap, "loose mesh refuses a remapped page on submit") ||
                !verify({source.address(), queue.Get(), green}, "new loose mesh recording reads replacement page")) return 1;
            source.free_original();
            check_lease_debug(info.Get());

            std::ifstream log{std::filesystem::path(log_path)};
            const std::string lines{std::istreambuf_iterator<char>(log), {}};
            log.close();
            DeleteFileW(log_path.c_str());
            check(lines.find("mesh XMem loose root bind") != std::string::npos &&
                  lines.find("mesh XMem submit refused: writer or live page identity changed") != std::string::npos &&
                  lines.find("mesh XMem loose root SRV has no usable single-page lease") != std::string::npos &&
                  lines.find("mesh XMem read address is missing or misaligned") != std::string::npos ? S_OK : E_FAIL,
                  "observe loose binding, stale submission and invalid range refusals");
            return 0;
        }
        if (lease_fixture) {
            MeshLeaseWriter source(device.Get(), guest.Get(), lease_xmem);
            const auto queue = source.create_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
            const auto verify = [&](const MeshLeaseDraw& draw, const char* label) {
                return run(device.Get(), guest.Get(), writer.Get(), work.Get(), root.Get(), pso.Get(), signature.Get(),
                           DrawKind::mesh, false, Refusal::none, label, nullptr, false, nullptr, false, false, 1, false, false,
                           &draw);
            };
            const MeshLeaseDraw no_writer{source.address(), queue.Get(), black};
            if (!verify(no_writer, "mesh lease missing writer or proof")) return 1;
            if (GetEnvironmentVariableA("X2W_MESH86_FIXTURE_REJECT_MARKER", nullptr, 0)) {
                source.start();
                source.finish();
                return verify(no_writer, "mesh lease rejects invalid marker even with a writer") ? 0 : 1;
            }
            if (!GetEnvironmentVariableA("X2W_MESH86_FIXTURE_READY", nullptr, 0)) return 0;
            source.start();
            const MeshLeaseDraw wrong_page{source.address() + MeshLeaseWriter::page, queue.Get(), black};
            if (!verify(wrong_page, "mesh lease different page refused")) return 1;
            const MeshLeaseDraw wrong_offset{source.address() + 16, queue.Get(), black};
            if (!verify(wrong_offset, "mesh lease same-page byte range refused")) return 1;
            const bool skip_wait = GetEnvironmentVariableA("X2W_TEST_MESH_XMEM_SKIP_WAIT", nullptr, 0) != 0;
            MeshLeaseDraw ordered{source.address(), queue.Get(), skip_wait ? green : red};
            if (!skip_wait) ordered.queued = [&](ID3D12Fence* done) { source.open_after_wait_probe(done); };
            if (!verify(ordered, skip_wait ? "mesh control reads stale bytes" : "mesh reads the published XMem write")) return 1;
            source.finish();
            if (!skip_wait) {
                const auto consumer_gate = source.create_fence();
                struct ReleaseGate {
                    ID3D12Fence* gate;
                    ~ReleaseGate() { gate->Signal(1); }
                } release{consumer_gate.Get()};
                auto native = native_object<ID3D12CommandQueue>(device.Get(), guest.Get(), queue.Get());
                check(native->Wait(consumer_gate.Get(), 1), "hold leased consumer before draw");
                MeshLeaseDraw pinned{source.address(), queue.Get(), red, true};
                pinned.queued = [&](ID3D12Fence*) {
                    source.free_original();
                    check(source.allocate_probe(2) != source.published_page ? S_OK : E_FAIL,
                          "in-flight mesh consumer pins the original slot");
                    check(consumer_gate->Signal(1), "release the mesh consumer");
                };
                if (!verify(pinned, "submitted mesh still reads the leased page after unmapping")) return 1;
                source.check_retirement("mesh completion returns the original slot to XMem");
            } else {
                source.free_original();
            }
            check_lease_debug(info.Get());
            return 0;
        }
        const auto verify = [&](Refusal control, const char* label) {
            return run(device.Get(), guest.Get(), writer.Get(), work.Get(), root.Get(), pso.Get(), signature.Get(),
                       DrawKind::mesh, false, control, label, nullptr, false, nullptr, false, false,
                       depth_only ? 0 : 3, depth_only, false, nullptr, nullptr, depth_pixel);
        };
        const bool ready = GetEnvironmentVariableA("X2W_MESH86_FIXTURE_READY", nullptr, 0) != 0;
        const bool invalid_depth_pixel = depth_pixel && depth != StreamDepth::writable;
        if (depth_pixel) {
            const bool native = bool(native_object<ID3D12PipelineState>(device.Get(), guest.Get(), pso.Get(), true));
            if (native != (ready && !invalid_depth_pixel)) {
                std::fprintf(stderr, "depth-only pixel mode %s: unexpected native PSO presence %d\n", depth_mode, native);
                return 1;
            }
        }
        const bool ok = !ready || invalid_depth_pixel
            ? verify(Refusal::mesh_pending, invalid_depth_pixel ? depth_mode : "unavailable cached mesh")
            : verify(Refusal::none, depth_pixel ? "cached mesh depth-only pixel stage" :
                depth_only ? "cached mesh depth-only" : "cached translated mesh with three attachments") &&
              verify(Refusal::incomplete_mesh_constants, "cached mesh incomplete root constants") &&
              verify(Refusal::mesh_attachment_count, "cached mesh missing attachment");
        if (depth_pixel) check_lease_debug(info.Get());
        return ok ? 0 : 1;
    }
    if (mesh && GetEnvironmentVariableA("X2W_MESH_REQUEST_FIXTURE", nullptr, 0)) {
        wchar_t exchange[MAX_PATH]{};
        const DWORD exchange_size = GetEnvironmentVariableW(L"X2W_TEST_MESH57_EXCHANGE", exchange, MAX_PATH);
        if (exchange_size >= MAX_PATH) return 1;
        const auto cache = exchange_size ? std::filesystem::path(exchange) / "requests" :
            std::filesystem::current_path() / ("slot57-mesh-" + std::to_string(GetCurrentProcessId()));
        std::filesystem::create_directory(cache);
        check(SetEnvironmentVariableW(L"X2W_SHADER_CACHE", cache.c_str()) ? S_OK : E_FAIL, "set mesh fixture cache");
        const D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &srv};
        const D3D12_ROOT_SIGNATURE_DESC source_root{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto& desc = exchange_size ? source_root : graphics_desc;
        ComPtr<ID3DBlob> root_blob;
        const auto root = create_root(guest.Get(), desc, &root_blob);
        const auto different_root = create_root(guest.Get(), desc);
        // Distinct, source-owned bounded spans. The request publisher must preserve their stage roles and lengths.
        const std::array<UINT, 4> default_ms{0xbfa00003, 0xbf900009, 0xbf810000, 0};
        const std::array<UINT, 2> default_ps{0xbfa00003, 0xbf810000};
        const auto read_stage = [&](const wchar_t* name) {
            std::ifstream file(std::filesystem::path(exchange) / name, std::ios::binary);
            return std::vector<char>(std::istreambuf_iterator<char>(file), {});
        };
        const auto source_ms = exchange_size ? read_stage(L"ms.bin") : std::vector<char>{};
        const auto source_ps = exchange_size ? read_stage(L"ps.bin") : std::vector<char>{};
        if (exchange_size && (source_ms.empty() || source_ps.empty())) return 1;
        const void* ms = exchange_size ? static_cast<const void*>(source_ms.data()) : default_ms.data();
        const void* ps = exchange_size ? static_cast<const void*>(source_ps.data()) : default_ps.data();
        const auto ms_size = exchange_size ? source_ms.size() : sizeof(default_ms);
        const auto ps_size = exchange_size ? source_ps.size() : sizeof(default_ps);
        const auto make_pso = [&](bool with_pixel, bool with_vertex, bool incomplete_pixel) {
            std::array<UINT64, 0x300 / sizeof(UINT64)> fields{};
            fields[0] = reinterpret_cast<UINT64>(root.Get());
            fields[5] = with_pixel ? reinterpret_cast<UINT64>(ps) : 0;
            fields[6] = with_pixel && !incomplete_pixel ? ps_size : 0;
            fields[9] = with_vertex ? reinterpret_cast<UINT64>(ps) : 0;
            fields[10] = with_vertex ? ps_size : 0;
            fields[13] = reinterpret_cast<UINT64>(ms);
            fields[14] = ms_size;
            using Create = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
            ComPtr<IUnknown> pso;
            check(slot<Create>(guest.Get(), 57)(guest.Get(), fields.data(), __uuidof(ID3D12PipelineState),
                reinterpret_cast<void**>(pso.GetAddressOf())), "create fixed mesh placeholder");
            return pso;
        };
        D3D12_INDIRECT_ARGUMENT_DESC mesh_argument{};
        mesh_argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        const D3D12_COMMAND_SIGNATURE_DESC signature_desc{sizeof(D3D12_DISPATCH_MESH_ARGUMENTS), 1, &mesh_argument, 0};
        ComPtr<ID3D12CommandSignature> signature;
        check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create mesh request signature");
        const auto dispatch = [&](IUnknown* pso, ID3D12RootSignature* bound_root, const char* label) {
            return run(device.Get(), guest.Get(), writer.Get(), nullptr, bound_root, pso, signature.Get(),
                       DrawKind::mesh, false, Refusal::none, label);
        };
        const auto pair = make_pso(true, false, false);
        const auto depth = make_pso(false, false, false);
        const auto aliased_vertex = make_pso(true, true, false);
        const auto incomplete = make_pso(true, false, true);
        if (!dispatch(pair.Get(), different_root.Get(), "mesh with a different root object") ||
            !dispatch(aliased_vertex.Get(), root.Get(), "mesh descriptor with a vertex slot") ||
            !dispatch(incomplete.Get(), root.Get(), "mesh descriptor with an incomplete PS span") ||
            !std::filesystem::is_empty(cache)) return 1;
        const bool control = GetEnvironmentVariableA("X2W_MESH_REQUEST_CONTROL", nullptr, 0) != 0;
        if (!dispatch(pair.Get(), root.Get(), "mesh with PS") ||
            !dispatch(depth.Get(), root.Get(), "mesh with no PS") ||
            (!control && !dispatch(pair.Get(), root.Get(), "repeat mesh with PS"))) return 1;
        unsigned pair_uses = 2, depth_uses = 1;
        if (!control && !exchange_size) {
            constexpr unsigned pair_calls = 4000, depth_calls = 1000;
            using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
            using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
            struct Recording {
                ComPtr<ID3D12CommandAllocator> allocator;
                ComPtr<IUnknown> list;
            };
            const PlacedBuffer arguments(device.Get(), guest.Get(), sizeof(initial), D3D12_RESOURCE_FLAG_NONE);
            const auto execute = [&](IUnknown* list) {
                slot<Execute>(list, 61)(list, signature.Get(), 1, arguments.wrapped.Get(), argument_offset, nullptr, 0);
            };
            const auto record = [&](Recording& recording, IUnknown* pipeline) {
                check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&recording.allocator)),
                      "create concurrent mesh census allocator");
                recording.list = begin_guest_list(guest.Get(), recording.allocator.Get());
                auto* list = recording.list.Get();
                slot<Bind>(list, 25)(list, pipeline);
                slot<Bind>(list, 34)(list, root.Get());
                execute(list);  // Resolve the request key before measuring repeated refusals.
            };
            Recording lists[2][2];
            for (auto& worker_lists : lists) {
                record(worker_lists[0], pair.Get());
                record(worker_lists[1], depth.Get());
            }

            struct Probe {
                std::atomic<bool> start{false};
                HANDLE threads[2]{};
                DWORD wait_result = WAIT_FAILED;
            } probe;
            const auto worker = [&](Recording* recordings) {
                while (!probe.start.load(std::memory_order_acquire)) std::this_thread::yield();
                for (unsigned i = 0; i < pair_calls; ++i) execute(recordings[0].list.Get());
                for (unsigned i = 0; i < depth_calls; ++i) execute(recordings[1].list.Get());
            };
            std::thread first(worker, lists[0]);
            std::thread second(worker, lists[1]);
            probe.threads[0] = first.native_handle();
            probe.threads[1] = second.native_handle();
            const auto holds = x2w_d3d12_mesh_census_lock_probe([](void* context) {
                auto& state = *static_cast<Probe*>(context);
                state.start.store(true, std::memory_order_release);
                state.wait_result = WaitForMultipleObjects(2, state.threads, TRUE, 5000);
            }, &probe);
            first.join();
            second.join();
            for (auto& worker_lists : lists)
                for (auto& recording : worker_lists)
                    check(slot<HRESULT (WINAPI*)(IUnknown*)>(recording.list.Get(), 9)(recording.list.Get()),
                          "close concurrent mesh census list");
            if (probe.wait_result != WAIT_OBJECT_0 || holds != 0) {
                std::fprintf(stderr, "mesh census lock probe failed: wait result %lu, draw_lock holds %+lld\n",
                             probe.wait_result, holds);
                return 1;
            }
            pair_uses += 2 * (1 + pair_calls);
            depth_uses += 2 * (1 + depth_calls);
            std::puts("10000 concurrent mesh refusals completed during a draw_lock hold with no additional holds");
        }
        std::string report(x2w_d3d12_census_report(nullptr, 0) + 1, '\0');
        x2w_d3d12_census_report(report.data(), report.size());
        unsigned requests = 0, with_ps = 0, files = 0;
        const auto matches = [](const std::filesystem::path& path, const void* bytes, std::size_t size) {
            std::ifstream stream(path, std::ios::binary);
            if (!stream || std::filesystem::file_size(path) != size) return false;
            std::vector<char> contents(size);
            return !size || (stream.read(contents.data(), size) && std::memcmp(contents.data(), bytes, size) == 0);
        };
        for (const auto& file : std::filesystem::directory_iterator(cache)) {
            auto path = file.path();
            if (path.filename() == "revision.txt") continue;  // the runtime's revisions, written with the first request
            ++files;
            if (path.extension() != ".m57") continue;
            ++requests;
            if (!matches(path, ms, ms_size) ||
                !matches(path.replace_extension(".root"), root_blob->GetBufferPointer(), root_blob->GetBufferSize())) return 1;
            const bool has_ps = matches(path.replace_extension(".ps57"), ps, ps_size);
            if (has_ps) ++with_ps;
            else if (!matches(path, nullptr, 0)) return 1;
            const auto key = path.stem().string().substr(4);  // req_<key>
            const auto count = std::to_string(has_ps ? pair_uses : depth_uses);
            const auto use = report.find("x2w-render key use mesh " + key + " first ");
            const auto use_count = report.find(" uses ", use);
            const auto expected_use = std::string(" uses ") + count + "\n";
            const bool correct_use = use_count != std::string::npos && use_count < report.find('\n', use) &&
                report.compare(use_count, expected_use.size(), expected_use) == 0;
            const bool correct_refusal = report.find("key=" + key + " reason=mesh no translation calls=" +
                                                     count + "\n") != std::string::npos;
            if (!correct_use || !correct_refusal) {
                std::fprintf(stderr, "mesh request census mismatch for %s (expected %s uses and refusals)\n%s",
                             key.c_str(), count.c_str(), report.c_str());
                return 1;
            }
        }
        if (requests != 2 || with_ps != 1 || files != 6) return 1;
        if (!exchange_size) std::filesystem::remove_all(cache);
        std::puts("slot 57 MS+PS and MS-only requests preserved stages and the matching root");
        return 0;
    }
    if (translated) {
        const auto cache = std::filesystem::current_path() / ("indirect-translated-" + std::to_string(GetCurrentProcessId()));
        std::filesystem::create_directory(cache);
        check(SetEnvironmentVariableW(L"X2W_SHADER_CACHE", cache.c_str()) ? S_OK : E_FAIL, "set fixture cache");
        if (alias_index)
            check(SetEnvironmentVariableA("X2W_TRANSLATED_INDIRECT_ADDRESS_ALIAS", alias_disabled ? "0" : "1"),
                "set existing heap alias control");
        check(SetEnvironmentVariableA("X2W_TRANSLATED_INDIRECT_DRAW", "1") ? S_OK : E_FAIL, "enable fixture path");
        check(SetEnvironmentVariableA("X2W_TRANSLATED_INDIRECT_COPY_RANGES", copy_on ? "1" : "0") ? S_OK : E_FAIL,
              "set copy-range provenance control");
        const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER table{};
        table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table.DescriptorTable = {1, &range};
        const D3D12_ROOT_SIGNATURE_DESC title_desc{1, &table, 0, nullptr, root_flags};
        ComPtr<ID3DBlob> root_blob;
        const auto title_root = create_root(guest.Get(), title_desc, &root_blob);
        if (translated_indexed) {
            const auto pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), true, TableSample::texture, true, rect);
            const auto left = writer_pipeline(device.Get(), writer.Get(), "write_left");
            const auto right = writer_pipeline(device.Get(), writer.Get(), "write_indexed_base");
            const auto invalid = writer_pipeline(device.Get(), writer.Get(), "write_indexed_overrun");
            D3D12_INDIRECT_ARGUMENT_DESC argument{};
            argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
            const D3D12_COMMAND_SIGNATURE_DESC desc{sizeof(D3D12_DRAW_INDEXED_ARGUMENTS), 1, &argument, 0};
            ComPtr<ID3D12CommandSignature> signature;
            check(guest->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature)), "create translated indexed signature");
            const TranslatedDrawFixture red_binding(guest.Get(), red), green_binding(guest.Get(), green);
            const auto verify = [&](ID3D12PipelineState* producer, bool green, Refusal refusal, const char* label, bool restore_compute = false) {
                return run(device.Get(), guest.Get(), writer.Get(), producer, title_root.Get(), pso.Get(), signature.Get(),
                    kind, green, refusal, label, green ? &green_binding : &red_binding, false, nullptr,
                    upload_index, alias_index, 0, false, restore_compute);
            };
            if (alias_index) {
                const bool ok = alias_disabled
                    ? verify(left.Get(), false, Refusal::alias_off, "unregistered spans refused when alias route is off")
                    : verify(left.Get(), false, Refusal::alias_no_heap, "unproven external heap refused") &&
                      verify(left.Get(), false, Refusal::alias_overlaps, "first page overlaps an existing GPU alias") &&
                      verify(left.Get(), false, Refusal::alias_cross_page, "raw index view crosses its imported page") &&
                      verify(left.Get(), false, Refusal::none, "R16 eight-byte index and VB0 from one unregistered GPU-aliased page") &&
                      verify(right.Get(), true, Refusal::none, "R32 with base 3 and both inputs on one GPU-aliased page");
                if (ok) std::filesystem::remove_all(cache);
                return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
            }
            if (upload_index) {
                const bool ok =
                    verify(left.Get(), false, Refusal::none, "R16 eight-byte current recording index span with late write") &&
                    verify(right.Get(), true, Refusal::none, "R32 current recording index span with offset and base") &&
                    verify(left.Get(), false, Refusal::index_upload_cross, "index view crosses logical upload span") &&
                    verify(left.Get(), false, Refusal::index_upload_foreign, "index view owned by a different list recording") &&
                    verify(left.Get(), false, Refusal::index_upload_unaligned, "R16 index upload starts at a half-word offset") &&
                    verify(left.Get(), false, Refusal::index_unregistered, "readable CPU memory without recording ownership refused");
                if (ok) std::filesystem::remove_all(cache);
                return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
            }
            if (rect) {
                const auto wrong_count = writer_pipeline(device.Get(), writer.Get(), "write_four_rectangle_indices");
                const bool ok =
                    verify(left.Get(), false, Refusal::none, "R16 rectangle with corner 1, permuted indices, FP16 UV, flat and scissor") &&
                    verify(right.Get(), true, Refusal::none, "R32 rectangle with corner 0, base 3 and clipping") &&
                    verify(left.Get(), false, Refusal::rect_bad_index, "refuse an out-of-range rectangle index") &&
                    verify(wrong_count.Get(), false, Refusal::rect_bad_count, "refuse four rectangle indices") &&
                    verify(left.Get(), false, Refusal::rect_bad_w, "refuse rectangle with differing clip-space w") &&
                    verify(left.Get(), false, Refusal::rect_bad_shape, "refuse an unmatched rectangle corner");
                if (ok) std::filesystem::remove_all(cache);
                return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
            }
            const bool ok = verify(left.Get(), false, Refusal::none, "translated indexed R16, base zero") &&
                verify(right.Get(), true, Refusal::none, "translated indexed R32, base 3, new argument and IB offsets") &&
                verify(invalid.Get(), true, Refusal::indexed_base_overrun, "GPU rejects vertex index past the bound stream") &&
                verify(right.Get(), true, Refusal::indexed_vertex_range, "refuse a vertex extent that crosses its allocation") &&
                verify(left.Get(), false, Refusal::invalid_index, "refuse an index view outside its placement") &&
                verify(left.Get(), false, Refusal::short_argument, "refuse a short argument record") &&
                verify(nullptr, false, Refusal::none, "omitted GPU writer leaves target clear") &&
                verify(left.Get(), false, Refusal::none, "draw and dispatch retain an uploaded compute SRV", true) &&
                verify(right.Get(), true, Refusal::none, "a guest root call without a desktop root retains native compute arguments", true);
            if (ok) std::filesystem::remove_all(cache);
            DeleteFileW(log_path.c_str());
            return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
        }
        const auto pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), false);
        const auto start_pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), true);
        const auto buffer_pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), false, TableSample::buffer);
        const auto producer = writer_pipeline(device.Get(), writer.Get(), "write_one_triangle");
        const auto buffer_writer = writer_pipeline(device.Get(), writer.Get(), "write_buffer_colour");
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        const D3D12_COMMAND_SIGNATURE_DESC signature_desc{sizeof(D3D12_DRAW_ARGUMENTS), 1, &argument, 0};
        ComPtr<ID3D12CommandSignature> signature;
        check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create translated draw signature");
        const TranslatedDrawFixture red_binding(guest.Get(), red), green_binding(guest.Get(), green);
        if (root_fix) {
            check(SetEnvironmentVariableA("X2W_INDIRECT_ROOT_FIX", "1") ? S_OK : E_FAIL, "enable indirect root fix");
            if (root_fix_control)
                check(SetEnvironmentVariableA("X2W_INDIRECT_ROOT_FIX", nullptr) ? S_OK : E_FAIL, "unset indirect root fix");
            ComPtr<ID3D12RootSignature> equivalent_root;
            check(guest->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                             IID_PPV_ARGS(&equivalent_root)), "create second guest root from the same blob");
            const auto refusal = root_fix_control ? Refusal::equivalent_root_disabled : Refusal::none;
            const bool pixels_ok = run(device.Get(), guest.Get(), writer.Get(), producer.Get(), equivalent_root.Get(),
                pso.Get(), signature.Get(), DrawKind::vertices, false, refusal,
                root_fix_control ? "equivalent root refused with switch unset" : "equivalent root draws with switch on",
                &red_binding);
            const auto needed = x2w_d3d12_census_report(nullptr, 0);
            std::vector<char> report(needed + 1);
            x2w_d3d12_census_report(report.data(), report.size());
            const std::string_view census(report.data());
            const bool refused = census.find("reason=missing_guest_root calls=1") != std::string_view::npos;
            if (refused != root_fix_control) std::fprintf(stderr, "unexpected root refusal census:\n%s", report.data());
            std::filesystem::remove_all(cache);
            return pixels_ok && refused == root_fix_control && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
        }
        TranslatedDrawFixture full_buffer(guest.Get(), 0, device.Get()), short_buffer(guest.Get(), 0, device.Get(), true),
            copied_buffer(guest.Get(), green, device.Get(), false, true);
        const TranslatedDrawFixture raw_upload(guest.Get(), 0);
        full_buffer.buffer_writer = short_buffer.buffer_writer = copied_buffer.buffer_writer = buffer_writer;
        const auto verify = [&](ID3D12PipelineState* gpu_writer, bool use_green, Refusal control, const char* label) {
            return run(device.Get(), guest.Get(), writer.Get(), gpu_writer, title_root.Get(),
                control == Refusal::start_vertex ? start_pso.Get() : pso.Get(), signature.Get(),
                DrawKind::vertices, use_green, control, label, use_green ? &green_binding : &red_binding);
        };
        if (refusal_census) {
            bool ok = verify(producer.Get(), false, Refusal::wrong_root, "census key1 missing_guest_root #1") &&
                verify(producer.Get(), false, refusal_census_control ? Refusal::none : Refusal::missing_texture,
                       refusal_census_control ? "census control valid draw replaces second reason" : "census key1 sampled_texture_mismatch") &&
                verify(producer.Get(), false, Refusal::wrong_root, "census key1 missing_guest_root #2") &&
                verify(producer.Get(), false, Refusal::start_vertex, "census key2 unsupported_resource_or_entry") &&
                run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                    DrawKind::vertices, false, Refusal::upload_range_crossing, "census key3 incomplete_table_buffer #1", &raw_upload, true) &&
                run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                    DrawKind::vertices, false, Refusal::copied_table_buffer, "census key3 incomplete_table_buffer #2", &copied_buffer) &&
                verify(producer.Get(), false, Refusal::none, "census valid draw does not increment refusals");
            if (ok) ok = check_translated_refusal_census(log_path);
            std::filesystem::remove_all(cache);
            DeleteFileW(log_path.c_str());
            return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
        }
        if (translated_targets) {
            const bool ok = check_translated_targets(device.Get(), guest.Get(), writer.Get(), producer.Get(),
                                                     title_root.Get(), pso.Get(), signature.Get(), red_binding);
            std::filesystem::remove_all(cache);
            bool debug_ok = true;
            if (info) {
                for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
                    SIZE_T bytes = 0;
                    info->GetMessage(i, nullptr, &bytes);
                    std::vector<char> storage(bytes);
                    auto* msg = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                    info->GetMessage(i, msg, &bytes);
                    // Message 679 (PS expects RTV0 with NumRenderTargets=0) is an expected benign warning when
                    // reusing the colour-writing pixel shader for the depth-only pass.
                    if (msg->Severity <= D3D12_MESSAGE_SEVERITY_ERROR ||
                        (msg->Severity == D3D12_MESSAGE_SEVERITY_WARNING && msg->ID != 679)) {
                        std::fprintf(stderr, "D3D12 debug message %u (sev %d): %s\n",
                                     static_cast<unsigned>(msg->ID), static_cast<int>(msg->Severity), msg->pDescription);
                        debug_ok = false;
                    }
                }
            }
            return ok && debug_ok ? 0 : 1;
        }
        if (other_refusal_census) {
            bool ok = verify(producer.Get(), false, Refusal::none, "other census warm up translated pipeline key") &&
                verify(producer.Get(), false, Refusal::missing_viewport, "other census key1 missing_viewport_or_scissor #1") &&
                verify(producer.Get(), false, other_refusal_census_control ? Refusal::none : Refusal::short_argument,
                       other_refusal_census_control ? "other census control valid draw replaces second reason"
                                                    : "other census key1 argument_outside_placed_buffer") &&
                verify(producer.Get(), false, Refusal::missing_viewport, "other census key1 missing_viewport_or_scissor #2");
            if (ok) {
                refuse_each_reason(guest.Get(), signature.Get());
                ok = check_indirect_other_refusal_census(log_path);
            }
            std::filesystem::remove_all(cache);
            DeleteFileW(log_path.c_str());
            return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
        }
        bool ok = verify(producer.Get(), false, Refusal::none, "translated red texture") &&
            verify(producer.Get(), true, Refusal::none, "translated green texture") &&
            verify(nullptr, false, Refusal::none, "translated omitted GPU writer") &&
            verify(producer.Get(), false, Refusal::wrong_root, "translated wrong guest root") &&
            verify(producer.Get(), false, Refusal::short_argument, "translated short argument range") &&
            verify(producer.Get(), false, Refusal::missing_texture, "translated missing texture") &&
            verify(producer.Get(), false, Refusal::start_vertex, "translated uses start vertex") &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::none, "translated GPU buffer range with nonzero offset and stride", &full_buffer) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::insufficient_table_range, "translated table exposes only its own range", &short_buffer) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::copied_table_buffer, "translated refuses copied GPU buffer and CPU shadow", &copied_buffer) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::none, "translated raw table uses its recording upload page", &raw_upload, true) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::upload_range_crossing, "translated raw table refuses crossing its upload allocation", &raw_upload, true);
        auto deferred_desc = title_desc;
        deferred_desc.Flags |= D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        ComPtr<ID3DBlob> deferred_blob;
        ComPtr<ID3D12RootSignature> deferred_root;
        if (SUCCEEDED(D3D12SerializeRootSignature(&deferred_desc, D3D_ROOT_SIGNATURE_VERSION_1, &deferred_blob, nullptr)) &&
            SUCCEEDED(guest->CreateRootSignature(0, deferred_blob->GetBufferPointer(), deferred_blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&deferred_root)))) {
            const auto deferred_pso = translated_pipeline(guest.Get(), deferred_root.Get(), deferred_blob.Get(), false, TableSample::buffer);
            if (ok) ok = run(device.Get(), guest.Get(), writer.Get(), producer.Get(), deferred_root.Get(), deferred_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::none, "translated table with deferred guest root", &full_buffer);
        } else {
            std::puts("deferred guest root capability unavailable");
        }
        if (ok && !copy_on && !copy_off) {
            const auto tail_pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), false, TableSample::buffer_tail);
            const auto tail_writer = writer_pipeline(device.Get(), writer.Get(), "write_large_buffer_colour");
            // Both resources contain the sentinel. Only the full view includes the final DWORD.
            for (bool short_view : {false, true}) {
                TranslatedDrawFixture large(guest.Get(), 0, device.Get(), short_view, false, false, false, 0x1000020);
                large.buffer_writer = tail_writer;
                ok = run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), tail_pso.Get(), signature.Get(),
                    DrawKind::vertices, false, short_view ? Refusal::insufficient_table_range : Refusal::none,
                    short_view ? "large buffer tail outside view" : "large buffer tail inside view", &large);
                if (!ok) break;
            }
        }
        if (ok && (copy_on || copy_off)) {
            const TranslatedDrawFixture copied(guest.Get(), green, device.Get(), false, true, true);
            const auto source = make_gpu_table_buffer(guest.Get(), replay_copy_offset + replay_copy_bytes);
            const auto native = native_object<ID3D12Resource>(device.Get(), guest.Get(), source.Get());
            const auto replay_writer = writer_pipeline(device.Get(), writer.Get(), "write_replay_colour");
            const ReplayCopy complete{source.Get(), native.Get(), replay_writer.Get(), replay_copy_bytes};
            const ReplayCopy partial{source.Get(), native.Get(), replay_writer.Get(), replay_copy_bytes - 4};
            const ReplayCopy overwritten{source.Get(), native.Get(), replay_writer.Get(), replay_copy_bytes, true};
            const auto draw = [&](const ReplayCopy* copy, Refusal refusal, const char* label) {
                return run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                    DrawKind::vertices, false, refusal, label, &copied, false, copy);
            };
            ok = draw(&complete, copy_on ? Refusal::none : Refusal::unproven_copy, "complete 68-byte GPU copy") &&
                 draw(nullptr, Refusal::unproven_copy, "new recording cannot reuse the previous GPU copy proof") &&
                 draw(&partial, Refusal::unproven_copy, "64-byte copy cannot authorize the 68-byte view") &&
                 draw(&overwritten, Refusal::unproven_copy, "untracked overlap invalidates the prior copy range");
            std::uint32_t title_word = 0;
            std::memcpy(&title_word, static_cast<unsigned char*>(copied.backing.get()) + 20, sizeof(title_word));
            if (title_word != green) {
                std::fprintf(stderr, "title-side green sentinel changed to %08x\n", title_word);
                return 1;
            }
        }
        if (ok) std::filesystem::remove_all(cache);
        return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
    }
    const auto graphics = create_root(guest.Get(), graphics_desc);
    const char* left_entry = indexed ? "write_left" : "write_one_triangle";
    const char* right_entry = indexed ? "write_right" : "write_two_triangles";
    if (kind == DrawKind::rect) {
        left_entry = "write_one_rect";
        right_entry = "write_two_rects";
    }
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    ComPtr<IUnknown> pso;
    if (mesh) {
        left_entry = "write_one_mesh_group";
        right_entry = "write_two_mesh_groups";
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        const D3D12_SHADER_BYTECODE ps{startup_indirect_mesh_pixel, sizeof(startup_indirect_mesh_pixel)};
        const D3D12_SHADER_BYTECODE ms{startup_indirect_mesh_mesh, sizeof(startup_indirect_mesh_mesh)};
        if (console_mesh) {
            // Xbox MS uses the same numeric tag as desktop AS.
            pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS>(guest.Get(), graphics.Get(), ms, ps);
        } else if (desktop_as) {
            pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS>(guest.Get(), graphics.Get(),
                {startup_indirect_mesh_amplified_mesh, sizeof(startup_indirect_mesh_amplified_mesh)}, ps,
                {startup_indirect_mesh_amplify, sizeof(startup_indirect_mesh_amplify)});
        } else {
            pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS>(guest.Get(), graphics.Get(), ms, ps);
        }
    } else {
        const auto vs = compile(kind == DrawKind::rect ? "vertex_rect" : "vertex", "vs_5_0");
        const auto ps = compile("pixel", "ps_5_0");
        pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS>(guest.Get(), graphics.Get(),
            {vs->GetBufferPointer(), vs->GetBufferSize()}, {ps->GetBufferPointer(), ps->GetBufferSize()});
    }
    const auto left = writer_pipeline(device.Get(), writer.Get(), left_entry);
    const auto right = writer_pipeline(device.Get(), writer.Get(), right_entry);
    const auto overflow = kind == DrawKind::rect ? writer_pipeline(device.Get(), writer.Get(), "write_overflowing_rects") : nullptr;
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{command_stride(kind), 1, &argument, 0};
    ComPtr<ID3D12CommandSignature> signature;
    check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create guest draw signature");

    if (lease_arguments) {
        if (!lease_fixture) return 2;
        return check_mesh_indirect_xmem(device.Get(), guest.Get(), lease_xmem, info.Get(), log_path,
                                        writer.Get(), right.Get(), graphics.Get(), pso.Get(), signature.Get()) ? 0 : 1;
    }

    const auto verify = [&](ID3D12PipelineState* producer, bool draw_right, Refusal refusal, const char* label) {
        return run(device.Get(), guest.Get(), writer.Get(), producer, graphics.Get(), pso.Get(), signature.Get(),
                   kind, draw_right, refusal, label);
    };
    const char* left_label = indexed ? "left writer" : "one triangle";
    const char* right_label = indexed ? "right writer" : "two triangles";
    if (kind == DrawKind::rect) {
        left_label = "one rectangle";
        right_label = "two rectangles";
    }
    if (mesh) {
        left_label = "one mesh group";
        right_label = "two mesh groups";
    }
    if (!verify(left.Get(), false, Refusal::none, left_label) ||
        !verify(right.Get(), true, Refusal::none, right_label) ||
        !verify(nullptr, false, Refusal::none, "omitted writer") ||
        !verify(left.Get(), false, Refusal::short_argument, "short argument range")) return 1;
    if (indexed ? !verify(left.Get(), false, Refusal::invalid_index, "invalid index range")
                : !verify(left.Get(), false, Refusal::missing_viewport, "missing viewport")) return 1;
    if (console_mesh) {
        const D3D12_SHADER_BYTECODE ps{startup_indirect_mesh_pixel, sizeof(startup_indirect_mesh_pixel)};
        pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS>(guest.Get(), graphics.Get(), ps, ps);
        if (!verify(left.Get(), false, Refusal::invalid_mesh_tag, "pixel DXIL under shared mesh tag")) return 1;
    }
    if (indexed) {
        // Forwarded: the three unrefused runs. Other reasons: the short argument range, the invalid index range and two commands.
        refuse_each_reason(guest.Get(), signature.Get());
        const std::string line = indirect_line(guest.Get(), log_path);
        DeleteFileW(log_path.c_str());
        const std::string expected = std::string("x2w-render indirect: forwarded: drawindexed ") + (count_control ? "4" : "3") +
            "; refused graphics-pipeline: drawindexed 1; compute-pipeline: dispatch 1; placeholder-signature: other 1; "
            "other: drawindexed 3";
        if (line != expected) {
            std::fprintf(stderr, "indirect line mismatch:\n  got      '%s'\n  expected '%s'\n", line.c_str(), expected.c_str());
            return 1;
        }
        std::printf("%s\n", line.c_str());
    }
    if (kind == DrawKind::rect &&
        !run(device.Get(), guest.Get(), writer.Get(), overflow.Get(), graphics.Get(), pso.Get(), signature.Get(), kind,
             false, Refusal::overflowing_count, "overflowing rectangle count")) return 1;
    if (kind == DrawKind::rect) {
        if (!run(device.Get(), guest.Get(), writer.Get(), left.Get(), graphics.Get(), pso.Get(), signature.Get(), kind,
                 false, Refusal::rect_live_compute, "rectangle with rebound compute root", nullptr, false, nullptr,
                 false, false, 0, false, true) ||
            !run(device.Get(), guest.Get(), writer.Get(), right.Get(), graphics.Get(), pso.Get(), signature.Get(), kind,
                 true, Refusal::rect_live_compute, "rectangle with retained desktop compute root", nullptr, false, nullptr,
                 false, false, 0, false, true)) return 1;
        // Check that the closed recordings own distinct scratch before releasing both submissions. Then each
        // queue must draw its own argument count even if the GPU overlaps the two conversions.
        std::atomic<bool> concurrent_ok{true};
        Gate gate;
        const auto worker = [&](ID3D12PipelineState* producer, bool draw_right, const char* label) {
            try {
                for (int round = 0; round < 24 && concurrent_ok; ++round)
                    if (!run(device.Get(), guest.Get(), writer.Get(), producer, graphics.Get(), pso.Get(), signature.Get(),
                             kind, draw_right, Refusal::none, label, nullptr, false, nullptr,
                             false, false, 0, false, false, nullptr, &gate)) concurrent_ok = false;
            } catch (...) {
                concurrent_ok = false;
            }
            if (!concurrent_ok) gate.abort();
        };
        std::thread one(worker, left.Get(), false, "concurrent one rectangle");
        std::thread two(worker, right.Get(), true, "concurrent two rectangles");
        one.join();
        two.join();
        if (!concurrent_ok) return 1;
    }
    if (info) {
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T bytes = 0;
            check(info->GetMessage(i, nullptr, &bytes), "get debug message size");
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &bytes), "get debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::fprintf(stderr, "D3D12 debug message %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
                return 1;
            }
        }
    }
    return 0;
}
catch (const std::exception&) {
    return 3;
}
