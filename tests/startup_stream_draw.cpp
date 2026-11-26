#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <iterator>
#include <vector>

#include "startup_stream_draw.h"
#include "../src/startup_stubs/packed_target_format.h"
#include "startup_stream_setup.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT XGCreateTextureComputer(const D3D12_RESOURCE_DESC*, void***, UINT64, UINT64);

// The components under test receive the guest ABI's vtable slots. The observation
// queue and its readback buffer are desktop objects owned only by the fixture.
using Word = std::uint64_t;
template<class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }




int draw_logical_tables(ID3D12Device* desktop, void* guest, void* list)
{
    constexpr UINT count = 1245184;
    auto accepts = [&](UINT n) {
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, n,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> probe;
        return SUCCEEDED(desktop->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&probe)));
    };
    if (accepts(count)) {
        std::printf("adapter accepts %u descriptors directly; the logical view tables are not needed here\n", count);
        return 77;
    }
    UINT boundary = 0, upper = 1048577;
    while (upper - boundary > 1) {
        const UINT n = boundary + (upper - boundary) / 2;
        if (accepts(n)) boundary = n; else upper = n;
    }
    if (!boundary) return 34;
    std::printf("logical views: physical capacity %u, logical count %u\n", boundary, count);
    const UINT stride = static_cast<UINT>(invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    if (stride < 32) return 34;
    struct Extended {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word cpu, bytes;
    } info{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void* heaps[2]{};
    Word handles[2]{};
    Extended staging_info{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2}, 0, 0};
    void* staging = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&staging_info), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&staging))))) return 34;
    const Word staging_cpu = invoke(staging, 9);
    D3D12_SHADER_RESOURCE_VIEW_DESC null_srv{};
    null_srv.Format = DXGI_FORMAT_R32_TYPELESS;
    null_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    null_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    null_srv.Buffer.NumElements = 1;
    null_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    alignas(256) static std::array<std::array<UINT, 64>, 6> words{};
    for (UINT h = 0; h < 2; ++h) {
        if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&info), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heaps[h]))))) return 34;
        const Word cpu = invoke(heaps[h], 9);
        handles[h] = invoke(heaps[h], 10);
        if (!cpu || !handles[h]) return 34;
        const UINT slots[] = {boundary - 1, boundary, count - 1};
        for (UINT i = 0; i < 3; ++i) {
            auto& data = words[3 * h + i];
            data[0] = 16 + 96 * h + 16 * i;
            const D3D12_CONSTANT_BUFFER_VIEW_DESC cb{arg(data.data()), static_cast<UINT>(sizeof(data))};
            // Initialize a real native descriptor as well as the fixture's guest view token.
            const Word destination = i < 2 ? staging_cpu + Word{i} * stride : cpu + Word{slots[i]} * stride;
            invoke(guest, 18, 0, arg(&null_srv), destination);
            invoke(guest, 17, arg(&cb), destination);
        }
        const Word destination = cpu + Word{boundary - 1} * stride;
        const UINT two = 2;
        if (h == 0) {
            invoke(guest, 24, two, destination, staging_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        } else {
            invoke(guest, 23, 1, arg(&destination), arg(&two), 1, arg(&staging_cpu), arg(&two), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        // Reject the complete copy: even its first entry must leave the final sentinel intact.
        invoke(guest, 24, two, cpu + Word{count - 1} * stride, staging_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    const auto bind = [&](UINT h) { invoke(list, 32, 1, arg(&heaps[h])); };
    const auto table = [&](UINT h, UINT i) { invoke(list, 36, 0, handles[h] + Word{i} * stride); };
    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, arg(&strip));
        invoke(list, 12, 3, 1);
    };
    bind(0);
    table(0, boundary - 1); draw(0);
    table(0, boundary); draw(1);
    table(0, count - 1); draw(2);
    table(0, count); draw(3);
    bind(1);
    table(1, boundary); draw(4);
    bind(0); draw(5);                   // Changing heaps invalidates its previous table.
    table(1, boundary); draw(6);        // A handle belonging to the other heap.
    table(0, boundary); draw(7);
    return 0;
}

namespace {
struct Bc7Binding {
    std::vector<unsigned char> backing, views;
    UINT64 table = 0, rtv = 0;
};

bool populate_tiled_bc7(const StartupBc7Input& input, const D3D12_RESOURCE_DESC& desc,
                        const std::vector<unsigned char>& blocks, UINT row_bytes, Bc7Binding& binding)
{
    void** computer = nullptr;
    if (FAILED(XGCreateTextureComputer(&desc, &computer, 0, 0)) || !computer) return false;
    const auto table = static_cast<void**>(computer[0]);
    using Query = HRESULT(*)(void*, unsigned char*);
    using Write = HRESULT(*)(void*, unsigned char*, UINT, UINT, const unsigned char*, UINT, UINT);
    std::array<unsigned char, 0x18 + 4 * 0x6c8 + 8> layout{};
    UINT64 span = 0;
    HRESULT hr = reinterpret_cast<Query>(table[2])(computer, layout.data());
    if (SUCCEEDED(hr)) std::memcpy(&span, layout.data(), sizeof span);
    const UINT64 expected = input.width == 64 ? 4096 : 16384;
    if (span != expected) hr = E_FAIL;
    if (SUCCEEDED(hr)) {
        binding.backing.resize(static_cast<std::size_t>(span), 0xa5);
        hr = reinterpret_cast<Write>(table[8])(computer, binding.backing.data(), 0, 0, blocks.data(),
                                                row_bytes, static_cast<UINT>(blocks.size()));
    }
    reinterpret_cast<UINT64(*)(void*)>(table[1])(computer);
    return SUCCEEDED(hr);
}

// Clear colour of the packed-target case; its 8-, 5- and 4-bit roundings differ (tests/startup_packed_target_draw_check.py).
const float packed_clear[] = {0.45f, 0.0f, 0.8f, 1.0f};

// The 16-bit texel the "fill" variant of the packed case writes into the title's memory at (x, y): distinct in every
// bit field of both 16-bit layouts (tests/startup_packed_target_draw_check.py computes the same).
std::uint16_t packed_fill_texel(UINT x, UINT y)
{
    return static_cast<std::uint16_t>((x * 257u + y * 131u + 0x1234u) & 0xffffu);
}

// True when the packed spec ends in ":1": the title memory holds known 16-bit texels and nothing clears the texture.
bool packed_fill(const char* spec)
{
    unsigned a = 0, b = 0, c = 0, fill = 0;
    return std::sscanf(spec, "%u:%x:%u:%u", &a, &b, &c, &fill) == 4 && fill == 1;
}

// 0: the texture, both views and the table exist. Otherwise an exit code that names the step.
int bind_packed(ID3D12Device* desktop, void* guest, const StartupBc7Input& input, Bc7Binding& binding)
{
    unsigned resource_format = 0, flags = 0, view_format = 0;
    if (std::sscanf(input.packed, "%u:%x:%u", &resource_format, &flags, &view_format) != 3) return 2;
    const bool fill = packed_fill(input.packed);
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, input.width, input.height, 1, 1,
        static_cast<DXGI_FORMAT>(resource_format), {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, static_cast<D3D12_RESOURCE_FLAGS>(flags)};
    // What the adapter says about the description, and what the helper makes of it.
    const auto support = [desktop](DXGI_FORMAT format) -> UINT {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT query{format};
        return SUCCEEDED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &query, sizeof(query))) ? static_cast<UINT>(query.Support1) : 0u;
    };
    const DXGI_FORMAT created = x2w_startup::packed_target_stand_in(desc, support);
    std::printf("packed: resource format %u flags 0x%x support1 0x%x view format %u expected texture format %u\n", resource_format, flags,
                support(desc.Format), view_format, static_cast<unsigned>(created));
    std::fflush(stdout);
    binding.backing.resize(static_cast<std::size_t>(input.width) * input.height * 4 + 0x10000);
    if (fill) {
        // Tight rows of 16-bit texels, as a title writes them for a linear texture of that description.
        for (UINT y = 0; y < input.height; ++y)
            for (UINT x = 0; x < input.width; ++x) {
                const std::uint16_t texel = packed_fill_texel(x, y);
                const std::size_t at = (static_cast<std::size_t>(y) * input.width + x) * 2;
                binding.backing[at] = static_cast<unsigned char>(texel & 0xff);
                binding.backing[at + 1] = static_cast<unsigned char>(texel >> 8);
            }
    }
    void* texture = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, arg(binding.backing.data()), arg(&desc),
        D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&texture)))) || !texture) return 40;

    const UINT increment = desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT entries = (32 + increment - 1) / increment;
    binding.views.resize(static_cast<std::size_t>(entries) * increment);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, entries, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
                arg(binding.views.data()), binding.views.size()},
      rtv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    void* heap = nullptr;
    void* rtv_heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heap)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&rtv_heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&rtv_heap)))))
        return 41;
    const auto cpu = invoke(heap, 9);
    if (cpu != arg(binding.views.data())) return 41;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{static_cast<DXGI_FORMAT>(view_format), D3D12_SRV_DIMENSION_TEXTURE2D,
                                        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
    srv.Texture2D.MipLevels = 1;
    const D3D12_RENDER_TARGET_VIEW_DESC rtv{static_cast<DXGI_FORMAT>(view_format), D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    binding.rtv = invoke(rtv_heap, 9);
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) invoke(guest, 20, arg(texture), arg(&rtv), binding.rtv);
    invoke(guest, 18, arg(texture), arg(&srv), cpu);
    binding.table = invoke(heap, 10);
    return binding.table ? 0 : 41;
}

bool bind_table(ID3D12Device* desktop, void* guest, void* texture, Bc7Binding& binding);

bool bind_bc7(ID3D12Device* desktop, void* guest, const StartupBc7Input& input, Bc7Binding& binding)
{
    std::ifstream file(input.blocks, std::ios::binary);
    const std::vector<unsigned char> blocks{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>{}};
    const UINT columns = (input.width + 3) / 4, rows = (input.height + 3) / 4, row_bytes = columns * 16;
    if (blocks.size() != static_cast<std::size_t>(row_bytes) * rows) return false;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, input.width, input.height, 1, 1,
        DXGI_FORMAT_BC7_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    if (input.tiled) {
        if (!populate_tiled_bc7(input, desc, blocks, row_bytes, binding)) return false;
    } else {
        auto aligned = desc;
        aligned.Width = columns * 4;
        aligned.Height = rows * 4;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 size = 0;
        desktop->GetCopyableFootprints(&aligned, 0, 1, 0, &footprint, nullptr, nullptr, &size);
        if (!size || size == UINT64_MAX || footprint.Footprint.RowPitch < row_bytes) return false;
        binding.backing.resize(static_cast<std::size_t>(size));
        for (UINT y = 0; y < rows; ++y)
            std::memcpy(binding.backing.data() + footprint.Offset + y * footprint.Footprint.RowPitch,
                        blocks.data() + y * row_bytes, row_bytes);
    }

    void* texture = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, arg(binding.backing.data()), arg(&desc),
        D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&texture))))) return false;
    return bind_table(desktop, guest, texture, binding);
}

bool bind_small4k(ID3D12Device* desktop, void* guest, const StartupBc7Input& input, Bc7Binding& binding)
{
    std::ifstream file(input.blocks, std::ios::binary);
    const std::vector<unsigned char> texels{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>{}};
    if (texels.size() != 32 * 32 * 4 + 16 * 16 * 4) return false;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0x1000, 32, 32, 1, 2,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    void** computer = nullptr;
    if (FAILED(XGCreateTextureComputer(&desc, &computer, 0, 0)) || !computer) return false;
    const auto table = static_cast<void**>(computer[0]);
    using Query = HRESULT(*)(void*, unsigned char*);
    using Write = HRESULT(*)(void*, unsigned char*, UINT, UINT, const unsigned char*, UINT, UINT);
    std::array<unsigned char, 0x18 + 4 * 0x6c8 + 8> layout{};
    UINT64 total = 0, alignment = 0;
    HRESULT hr = reinterpret_cast<Query>(table[2])(computer, layout.data());
    std::memcpy(&total, layout.data(), sizeof total);
    std::memcpy(&alignment, layout.data() + 8, sizeof alignment);
    std::printf("small4k: slot 2 total 0x%llx alignment 0x%llx\n", static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(alignment));
    if (SUCCEEDED(hr) && (total != 0x2000 || alignment != 0x1000)) hr = E_FAIL;
    auto* memory = SUCCEEDED(hr) ? static_cast<unsigned char*>(VirtualAlloc(nullptr, 0x3000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) : nullptr;
    DWORD old = 0;
    if (!memory || !VirtualProtect(memory + 0x2000, 0x1000, PAGE_NOACCESS, &old)) hr = E_FAIL;
    if (SUCCEEDED(hr)) hr = reinterpret_cast<Write>(table[8])(computer, memory, 0, 0, texels.data(), 128, 0);
    if (SUCCEEDED(hr)) hr = reinterpret_cast<Write>(table[8])(computer, memory, 0, 1, texels.data() + 32 * 32 * 4, 64, 0);
    reinterpret_cast<UINT64(*)(void*)>(table[1])(computer);
    if (FAILED(hr)) return false;
    void* texture = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, arg(memory), arg(&desc),
        D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&texture))))) return false;
    return bind_table(desktop, guest, texture, binding);
}

// A shader-visible heap whose first entry is a view of every mip of texture; binding.table is its GPU handle.
bool bind_table(ID3D12Device* desktop, void* guest, void* texture, Bc7Binding& binding)
{
    const UINT increment = desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT entries = (32 + increment - 1) / increment;
    binding.views.resize(static_cast<std::size_t>(entries) * increment);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, entries, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
                arg(binding.views.data()), binding.views.size()};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heap)))))
        return false;
    const auto cpu = invoke(heap, 9);
    if (cpu != arg(binding.views.data())) return false;
    invoke(guest, 18, arg(texture), 0, cpu);
    binding.table = invoke(heap, 10);
    return binding.table != 0;
}
}  // namespace

int check_stream_draw(IDXGIAdapter* adapter, void* guest, void* pso, const char* pixels, const StartupBc7Input* bc7,
                      DXGI_FORMAT format, std::uint32_t export_word, bool two_targets)
{
    const bool console_view = GetEnvironmentVariableW(L"X2W_TEST_CONSOLE_VIEW", nullptr, 0) != 0;
    const bool console_gate = GetEnvironmentVariableW(L"X2W_TEST_CONSOLE_GATE", nullptr, 0) != 0;
    const bool bc = console_view || console_gate || GetEnvironmentVariableW(L"X2W_TEST_BC_VIEW", nullptr, 0) != 0;
    const bool bc_chain = GetEnvironmentVariableW(L"X2W_TEST_BC_CHAIN", nullptr, 0) != 0;
    StreamMrtFormat mrt{};
    const bool typed = stream_mrt_format(format, mrt);
    // The translated MRT formats draw one pixel; the other formats keep the 8x4 target.
    const bool single = typed;
    const bool legacy = !single;
    const UINT width = bc_chain ? 384 : bc7 ? (bc7->small4k ? 48 : bc7->width) : bc ? 4 * 128 : single ? 1 : 8;
    const UINT height = bc_chain ? 32 : bc7 ? bc7->height : bc ? 128 : single ? 1 : 4;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 20;
    Bc7Binding binding;
    if (bc7 && bc7->packed) {
        if (const int rc = bind_packed(desktop.Get(), guest, *bc7, binding)) return rc;
    } else if (bc7 && bc7->small4k) {
        if (!bind_small4k(desktop.Get(), guest, *bc7, binding)) return 35;
    } else if (bc7 && !bind_bc7(desktop.Get(), guest, *bc7, binding)) return 34;
    StreamTarget frame;
    if (two_targets) frame.count = 2;  // a second RGBA8 target of the same size, written after the first
    if (const int rc = open_stream_target(desktop.Get(), guest, pso, width, height, format,
                                          legacy && !bc7 && !bc_chain ? width / 2 : width, frame)) return rc;
    auto* list = frame.list;
    auto* queue = frame.queue;
    auto* fence = frame.fence;
    if (export_word) invoke(list, 40, 0, 1, arg(&export_word), 0);
    char index_probe[32]{};
    GetEnvironmentVariableA("X2W_TEST_INDEX_PROBE", index_probe, sizeof(index_probe));
    void* index_resource = nullptr;
    if (index_probe[0]) {
        bool initial_visible = false, final_visible = false;
        const bool wide_indices = !std::strcmp(index_probe, "late-visible-r32");
        if (!std::strcmp(index_probe, "visible")) initial_visible = final_visible = true;
        else if (!std::strcmp(index_probe, "late-visible") || wide_indices) final_visible = true;
        else if (!std::strcmp(index_probe, "late-clipped")) initial_visible = true;
        else if (std::strcmp(index_probe, "clipped")) return 70;
        D3D12_FEATURE_DATA_EXISTING_HEAPS capability{};
        if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &capability, sizeof(capability))) || !capability.Supported)
            return 77;
        void* index_memory = VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!index_memory) return 71;
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, arg(index_memory), arg(&desc),
                D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&index_resource)))) ||
            !index_resource || invoke(index_resource, 11) != arg(index_memory)) return 72;
        const auto write = [&](bool visible) {
            const UINT first = visible ? 0 : 3;
            if (wide_indices) {
                auto* words = static_cast<std::uint32_t*>(index_memory);
                words[0] = 9; // Not part of the draw. A binding that misses the start offset sees this index.
                for (UINT i = 0; i < 3; ++i) words[i + 1] = first + i;
            } else {
                auto* words = static_cast<std::uint16_t*>(index_memory);
                for (UINT i = 0; i < 3; ++i) words[i] = static_cast<std::uint16_t>(first + i);
            }
        };
        write(initial_visible);
        const D3D12_INDEX_BUFFER_VIEW index_view{arg(index_memory), wide_indices ? 16u : 6u,
                                                wide_indices ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT};
        const D3D12_RECT full{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        invoke(list, 22, 1, arg(&full));
        invoke(list, 47, arg(&index_view));
        invoke(list, 13, 3, 1, wide_indices ? 1 : 0, 0, 0);
        write(final_visible);
    } else if (GetEnvironmentVariableW(L"X2W_TEST_LOGICAL_VIEW", nullptr, 0)) {
        if (int result = draw_logical_tables(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_PLACED_VERTEX", nullptr, 0)) {
        extern int draw_placed_vertices(ID3D12Device*, void*, void*);
        if (const int result = draw_placed_vertices(desktop.Get(), guest, list)) return result;
    } else if (bc_chain) {
        extern int draw_bc_chain(void*, void*);
        if (const int result = draw_bc_chain(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_ROOT_CBV", nullptr, 0)) {
        extern int root_heap_control(ID3D12Device*);
        extern int draw_root_cbv(void*);
        if (const int result = root_heap_control(desktop.Get())) return result;
        if (const int result = draw_root_cbv(list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_1D_SRV", nullptr, 0)) {
        extern int draw_1d_srvs(void*, void*);
        if (const int result = draw_1d_srvs(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_JOINED_TABLE", nullptr, 0) || GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_SRV", nullptr, 0)) {
        extern int draw_address_srvs(ID3D12Device*, void*, void*);
        if (const int result = draw_address_srvs(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_MEMORY_SRV_DIAG", nullptr, 0)) {
        extern int draw_memory_srv_diagnostics(ID3D12Device*, void*, void*);
        if (const int result = draw_memory_srv_diagnostics(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_SINGLE_ELEMENT", nullptr, 0)) {
        extern int draw_single_element_srvs(void*, void*);
        if (const int result = draw_single_element_srvs(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_FORMATTED_VIEW", nullptr, 0)) {
        extern int draw_formatted_view(void*, void*);
        if (const int result = draw_formatted_view(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_COPIED_VERTEX_SRV", nullptr, 0)) {
        extern int draw_copied_vertex_buffer_srv(void*, void*);
        if (const int result = draw_copied_vertex_buffer_srv(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_COPIED_VERTEX_SRV_PAIR", nullptr, 0)) {
        extern int draw_copied_vertex_srv_pair(void*, void*);
        if (const int result = draw_copied_vertex_srv_pair(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_BUFFER_SRV", nullptr, 0)) {
        extern int draw_address_buffer_srvs(ID3D12Device*, void*, void*);
        if (const int result = draw_address_buffer_srvs(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_FORMATTED_BUFFER", nullptr, 0)) {
        extern int draw_formatted_buffer(void*, void*);
        if (const int result = draw_formatted_buffer(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_VOLUME_SRV", nullptr, 0)) {
        extern int draw_volume_address_srvs(ID3D12Device*, void*, void*);
        if (const int result = draw_volume_address_srvs(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_D16_SRV", nullptr, 0)) {
        extern int draw_d16_srvs(ID3D12Device*, void*, void*);
        if (const int result = draw_d16_srvs(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_PIXEL_UAV", nullptr, 0)) {
        extern int draw_pixel_uavs(ID3D12Device*, void*, void*);
        if (const int result = draw_pixel_uavs(desktop.Get(), guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_VERTEX_FETCH", nullptr, 0)) {
        // TEXCOORD float3 in slot 0 and POSITION float3 in slot 1, 12 bytes each; vertices 0-2 are unused. Vertices
        // 3-8 are drawn from start vertex 3: positions cover pixels [1, 3) x [1, 3) of the 8x4 target, texture
        // coordinates read as positions would cover [0, 4) x [0, 1). Vertices 9-14 are drawn through indices 4-9 with
        // base vertex 5: positions cover column 0, texture coordinates read as positions pixel (3, 3).
        struct Rect { float x0, y0, x1, y1; };
        const auto corners = [](float (*out)[3], Rect r) {
            const float c[6][2] = {{r.x0, r.y0}, {r.x1, r.y0}, {r.x0, r.y1}, {r.x1, r.y0}, {r.x1, r.y1}, {r.x0, r.y1}};
            for (int v = 0; v < 6; ++v) { out[v][0] = c[v][0]; out[v][1] = c[v][1]; out[v][2] = 0.5f; }
        };
        static float texcoords[15][3], positions[15][3];
        corners(texcoords + 3, {-1.0f, 1.0f, 0.0f, 0.5f});
        corners(positions + 3, {-0.75f, 0.5f, -0.25f, -0.5f});
        corners(texcoords + 9, {-0.25f, -0.5f, 0.0f, -1.0f});
        corners(positions + 9, {-1.0f, 1.0f, -0.75f, -1.0f});
        static const std::uint16_t indices[6] = {4, 5, 6, 7, 8, 9};
        const D3D12_VERTEX_BUFFER_VIEW views[2] = {
            {reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(texcoords), sizeof(texcoords), sizeof(texcoords[0])},
            {reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(positions), sizeof(positions), sizeof(positions[0])}};
        const D3D12_INDEX_BUFFER_VIEW index_view{reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(indices), sizeof(indices), DXGI_FORMAT_R16_UINT};
        invoke(list, 48, 0, 2, arg(views));
        invoke(list, 12, 6, 1, 3, 0);
        invoke(list, 47, arg(&index_view));
        invoke(list, 13, 6, 1, 0, 5, 0);
    } else if (GetEnvironmentVariableW(L"X2W_TEST_ROOT_RESET", nullptr, 0)) {
        extern int draw_root_reset(void*, void*);
        if (const int result = draw_root_reset(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_FRAME_COPY", nullptr, 0)) {
        extern int draw_frame_copies(ID3D12Device*, void*, void*, UINT, UINT);
        if (const int result = draw_frame_copies(desktop.Get(), guest, list, width, height)) return result;
    } else {
        if (bc7 && bc7->packed && !packed_fill(bc7->packed)) invoke(list, 52, binding.rtv, arg(packed_clear));  // render into the texture, then sample it
        if (bc7) invoke(list, 36, 0, binding.table);
        invoke(list, 12, 3, 1);
    }
    if (console_view) {
        extern int draw_console_address_srv(void*, void*);
        if (const int result = draw_console_address_srv(guest, list)) return result;
    } else if (console_gate) {
        // X2W_TEST_CONSOLE_GATE: the queue waits on a fence this fixture holds while eighteen lists are
        // recorded and submitted on one list object, each after a new payload was written at the same
        // address (draw_gated_console_payload). Nothing may complete before the fence is signalled; the
        // target is read back after the last submit has completed.
        extern int draw_gated_console_payload(void*, void*, unsigned, bool);
        constexpr unsigned payloads = 18;
        void* gate = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&gate))))) return 44;
        D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(frame.rtv_heap, 9))};
        const auto next_recording = [&] {
            if (FAILED(static_cast<HRESULT>(invoke(frame.allocator, 8))) ||
                FAILED(static_cast<HRESULT>(invoke(list, 139, arg(frame.allocator))))) return false;
            invoke(list, 50, 1, arg(&rtv));
            invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            invoke(list, 25, arg(pso));
            return true;
        };
        // Retrying the same bytes after abandoning a recording still needs an upload. Keep the
        // shared-ring control's first upload submitted so its mismatch comes from ring wrap.
        if (!GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_UPLOAD_RING", nullptr, 0)) {
            if (const int result = draw_gated_console_payload(guest, list, 0, true)) return result;
            if (FAILED(static_cast<HRESULT>(invoke(list, 9))) || !next_recording()) return 35;
        }
        if (FAILED(static_cast<HRESULT>(invoke(queue, 17, arg(gate), 1)))) return 45;
        for (unsigned payload = 0; payload < payloads; ++payload) {
            if (payload && !next_recording()) return 35;
            if (const int result = draw_gated_console_payload(guest, list, payload, payload == 0)) return result;
            invoke(list, 9);
            invoke(queue, 10, 1, arg(&list));
            if (FAILED(static_cast<HRESULT>(invoke(queue, 14, arg(fence), payload + 1)))) return 24;
        }
        Sleep(250);
        if (invoke(fence, 8) != 0) return 46;  // a submit completed although the queue was held
        if (FAILED(static_cast<HRESULT>(invoke(gate, 10, 1)))) return 47;
        // One more recording without a draw: its submit is where the stand-in visits the completed ones.
        const auto reached = [&](UINT64 value) {
            const auto deadline = GetTickCount64() + 60000;
            while (invoke(fence, 8) < value && GetTickCount64() < deadline) Sleep(1);
            return invoke(fence, 8) == value;
        };
        if (!reached(payloads)) return 25;
        if (!next_recording()) return 35;
        invoke(list, 9);
        invoke(queue, 10, 1, arg(&list));
        if (FAILED(static_cast<HRESULT>(invoke(queue, 14, arg(fence), payloads + 1))) || !reached(payloads + 1)) return 25;
        return read_stream_target(desktop.Get(), frame, pixels);
    } else if (bc) {
        extern int draw_bc_address_srvs(void*, void*);
        if (const int result = draw_bc_address_srvs(guest, list)) return result;
    }
    const auto submit = [&](UINT64 value) {
        invoke(list, 9);
        invoke(queue, 10, 1, arg(&list));
        if (FAILED(static_cast<HRESULT>(invoke(queue, 14, arg(fence), value)))) return 24;
        const auto deadline = GetTickCount64() + 10000;
        while (invoke(fence, 8) < value && GetTickCount64() < deadline) Sleep(1);
        return invoke(fence, 8) == value ? 0 : 25;
    };
    // X2W_TEST_VIEWPORT=reset: the list above drew inside a small rectangle. Record a second list on the same
    // object after Reset that sets neither a viewport nor a scissor; it must not inherit the first rectangle.
    wchar_t viewport_mode[8] = L"";
    GetEnvironmentVariableW(L"X2W_TEST_VIEWPORT", viewport_mode, 8);
    UINT64 last = 1;
    // X2W_TEST_VIEWPORT=grow: no viewport or scissor anywhere. After the draw above on the small target, the same
    // list switches to a target twice as large and draws again; the default must follow the new target.
    if (std::wcscmp(viewport_mode, L"grow") == 0) {
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word backing, bytes;
        } big_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
        void* big_heap = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&big_heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&big_heap))))) return 36;
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width * 2, height * 2, 1, 1,
            format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
        const float clear[] = {0, 0, 1, 1};
        const D3D12_CLEAR_VALUE optimized_clear{format, {clear[0], clear[1], clear[2], clear[3]}};
        StreamAttachment big;
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, &optimized_clear, IID_PPV_ARGS(&big.texture)))) return 37;
        D3D12_RENDER_TARGET_VIEW_DESC view{format, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
        D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(big_heap, 9))};
        invoke(guest, 20, arg(big.texture.Get()), arg(&view), rtv.ptr);
        big.row_bytes = width * 2 * 4;
        invoke(list, 52, rtv.ptr, arg(clear));
        invoke(list, 50, 1, arg(&rtv));
        invoke(list, 12, 3, 1);
        // Read back the larger target only; the small one stays alive in slot 1 until the list has executed.
        frame.attachments[1] = std::move(frame.attachments[0]);
        frame.attachments[0] = std::move(big);
        frame.count = 1;
    }
    if (std::wcscmp(viewport_mode, L"reset") == 0) {
        if (const int rc = submit(1)) return rc;
        const float clear[] = {0, 0, 1, 1};
        D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(frame.rtv_heap, 9))};
        if (FAILED(static_cast<HRESULT>(invoke(frame.allocator, 8))) ||
            FAILED(static_cast<HRESULT>(invoke(list, 139, arg(frame.allocator))))) return 35;
        invoke(list, 52, rtv.ptr, arg(clear));
        invoke(list, 50, 1, arg(&rtv));
        invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(list, 25, arg(pso));
        invoke(list, 12, 3, 1);
        last = 2;
    }
    if (const int rc = submit(last)) return rc;
    if (index_resource) invoke(index_resource, 2); // The backing heap can outlive the resource.
    if (GetEnvironmentVariableW(L"X2W_TEST_PIXEL_UAV", nullptr, 0)) {
        extern int report_pixel_uavs(const char*);
        if (const int rc = report_pixel_uavs(pixels)) return rc;
    }
    if (GetEnvironmentVariableW(L"X2W_TEST_ROOT_CBV", nullptr, 0)) {
        extern int root_heap_report();
        if (const int rc = root_heap_report()) return rc;
    }

    return read_stream_target(desktop.Get(), frame, pixels);
}
