#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <array>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <vector>

using Word = std::uint64_t;
Word invoke(void*, unsigned, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0);
extern "C" __declspec(dllimport) void x2w_d3d12_texture_written(void* base, unsigned long long bytes);
extern "C" __declspec(dllimport) void x2w_d3d12_bc7_4k_s_written(void* base, UINT width, UINT height);
extern "C" __declspec(dllimport) void x2w_d3d12_fail_wraps(LONG count);
namespace {
template<class T> Word ptr(T* value) { return reinterpret_cast<Word>(value); }
struct ViewAt {
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    Word address;
    Word tail[2];
};
static_assert(offsetof(ViewAt, address) == 0x28 && sizeof(ViewAt) == 0x40);

bool failed_publication_keeps_view(void* guest, const D3D12_RESOURCE_DESC& desc, const ViewAt& view, Word cpu)
{
    const auto* slot = reinterpret_cast<const void*>(cpu);
    std::array<unsigned char, 32> previous;
    std::memcpy(previous.data(), slot, previous.size());
    x2w_d3d12_fail_wraps(1);
    invoke(guest, 70, ptr(&desc), ptr(&view), cpu);
    x2w_d3d12_fail_wraps(0);
    return std::memcmp(previous.data(), slot, previous.size()) == 0;
}

bool bind_view_table(void* guest, void* list, Word& cpu, UINT count = 1, Word address = 0, Word bytes = 0)
{
    struct Extended {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word address, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, address, bytes};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&heap))))) return false;
    cpu = invoke(heap, 9);
    invoke(list, 32, 1, ptr(&heap));
    invoke(list, 36, 0, invoke(heap, 10));
    return true;
}
}

int draw_1d_address_srvs(ID3D12Device* desktop, void* guest, void* list)
{
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu, 2)) return 42;
    const Word stride = invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (stride < 32) return 42;

    static std::vector<unsigned char> backing[2];
    for (UINT layers : {1u, 2u}) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE1D, 0, 8, 1,
            static_cast<UINT16>(layers), 1, DXGI_FORMAT_R8_UNORM, {1, 0},
            D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2]{};
        UINT64 bytes = 0;
        desktop->GetCopyableFootprints(&desc, 0, layers, 0, footprints, nullptr, nullptr, &bytes);
        if (!bytes || bytes == ~UINT64{0}) return 40;
        auto& texels = backing[layers - 1];
        texels.resize(static_cast<std::size_t>(bytes));
        for (UINT layer = 0; layer < layers; ++layer)
            for (UINT x = 0; x < desc.Width; ++x)
                texels[footprints[layer].Offset + x] = static_cast<unsigned char>(16 + 24 * x + 8 * layer);
        void* placed = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(texels.data()), ptr(&desc), D3D12_RESOURCE_STATE_COMMON,
                                               0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 41;
        const Word slot = cpu + (layers - 1) * stride;
        ViewAt view{};
        view.address = ptr(texels.data());
        view.view.Format = desc.Format;
        view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (layers == 1) {
            view.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            view.view.Texture1D.MipLevels = 1;
        } else {
            view.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            view.view.Texture1DArray = {0, 1, 1, 1, 0};
        }
        invoke(guest, 18, 0, ptr(&view.view), slot);
        const auto* descriptor = reinterpret_cast<const void*>(slot);
        std::array<unsigned char, 32> previous;
        std::memcpy(previous.data(), descriptor, previous.size());
        invoke(guest, 70, ptr(&desc), ptr(&view), slot);
        std::printf("slot 70 texture1d layers=%u descriptor_changed=%d\n", layers,
                    std::memcmp(previous.data(), descriptor, previous.size()) != 0);
        if (layers == 2) {
            // The second slot is outside the draw's table entry. Check that a
            // slice range extending beyond the placed array leaves it intact.
            std::memcpy(previous.data(), descriptor, previous.size());
            view.view.Texture1DArray.ArraySize = 2;
            invoke(guest, 70, ptr(&desc), ptr(&view), slot);
            if (std::memcmp(previous.data(), descriptor, previous.size()) != 0) return 43;
        }
    }
    const D3D12_RECT full{0, 0, 8, 4};
    invoke(list, 22, 1, ptr(&full));
    invoke(list, 12, 3, 1);
    return 0;
}

extern "C" __declspec(dllimport) std::size_t x2w_d3d12_census_report(char* out, std::size_t capacity);

static void print_slot70_switch_census()
{
    std::string report(x2w_d3d12_census_report(nullptr, 0) + 1, '\0');
    x2w_d3d12_census_report(report.data(), report.size());
    const std::string prefix = "x2w-d3d12 memory SRV ";
    for (auto first = report.find(prefix); first != std::string::npos;) {
        const auto end = report.find('\n', first);
        const auto kind = first + prefix.size();
        if (!report.compare(kind, 9, "switches:") || !report.compare(kind, 10, "diagnostic"))
            std::puts(report.substr(first, end - first).c_str());
        if (end == std::string::npos) break;
        first = report.find(prefix, end + 1);
    }
}

int draw_unregistered_1d_address_srv(void* guest, void* list)
{
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 44;
    const bool mutant = GetEnvironmentVariableW(L"X2W_TEST_PLACED_1D_WRONG_WIDTH", nullptr, 0) != 0;
    wchar_t console_case[2]{};
    GetEnvironmentVariableW(L"X2W_TEST_PLACED_1D_CONSOLE_LAYOUT", console_case, 2);
    const bool console_layout = console_case[0] != L'\0';
    const bool multiple_mips = console_case[0] == L'm';
    const bool array = console_case[0] == L'a';
    const bool srgb = console_layout || GetEnvironmentVariableW(L"X2W_TEST_PLACED_1D_SRGB", nullptr, 0) != 0;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE1D, 0, multiple_mips ? 8u : mutant ? 4u : 7u,
        1, array ? UINT16{2} : UINT16{1}, multiple_mips ? UINT16{2} : UINT16{1},
        srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8_UNORM, {1, 0},
        console_layout ? static_cast<D3D12_TEXTURE_LAYOUT>(console_case[0] == L'u' ? 0x105 : 0x100)
                       : D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_NONE};
    // Leave backing for the desktop mip and array offsets so the refusal controls exercise the layout gate.
    static std::array<unsigned char, 65536 + 8 * 4> memory{};
    ViewAt view{};
    view.address = ptr(memory.data());
    view.view.Format = desc.Format;
    view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (array) {
        view.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
        view.view.Texture1DArray.MipLevels = desc.MipLevels;
        view.view.Texture1DArray.ArraySize = desc.DepthOrArraySize;
    } else {
        view.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
        view.view.Texture1D.MipLevels = desc.MipLevels;
    }
    invoke(guest, 18, 0, ptr(&view.view), cpu);
    const auto* slot = reinterpret_cast<const void*>(cpu);
    std::array<unsigned char, 32> previous;
    std::memcpy(previous.data(), slot, previous.size());

    const D3D12_VIEWPORT viewport{0, 0, 8, 4, 0, 1};
    invoke(list, 21, 1, ptr(&viewport));
    for (LONG phase = 0; phase < 2; ++phase) {
        for (UINT x = 0; x < 8; ++x) {
            if (srgb) memory[4 * x] = ((x + phase) % 3 == 1) ? 255 : 0;
            else memory[x] = static_cast<unsigned char>(11 + 19 * x + 73 * phase);
        }
        // Republish the same address after a title-side write, in the same frame.
        invoke(guest, 70, ptr(&desc), ptr(&view), cpu);
        const D3D12_RECT rows{0, phase * 2, 8, phase * 2 + 2};
        invoke(list, 22, 1, ptr(&rows));
        invoke(list, 12, 3, 1);
    }
    std::printf("unregistered 1D descriptor_changed=%d\n", std::memcmp(previous.data(), slot, previous.size()) != 0);
    print_slot70_switch_census();
    return 0;
}

// Two separately placed R8 planes have the same descriptor and different row
// data. The producer and oracle both use the documented desktop footprint.
int draw_address_srvs(ID3D12Device* desktop, void* guest, void* list)
{
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 2, 2, 1, 1,
        DXGI_FORMAT_R8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, bytes = 0;
    desktop->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &bytes);
    if (rows != 2 || row_bytes != 2 || bytes < footprint.Footprint.RowPitch + 2) return 40;
    std::array<std::vector<unsigned char>, 2> planes;
    void* placed[2]{};
    for (unsigned i = 0; i < planes.size(); ++i) {
        planes[i].resize(static_cast<std::size_t>(bytes));
        for (unsigned y = 0; y < rows; ++y)
            for (unsigned x = 0; x < row_bytes; ++x)
                planes[i][y * footprint.Footprint.RowPitch + x] = static_cast<unsigned char>(16 + 128 * i + 32 * y + 7 * x);
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(planes[i].data()), ptr(&desc), D3D12_RESOURCE_STATE_COMMON,
                                               0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed[i]))))) return 41;
    }
    Word cpu = 0;
    const bool joined_table = GetEnvironmentVariableW(L"X2W_TEST_JOINED_TABLE", nullptr, 0) != 0;
    const Word stride = joined_table ? invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) : 0;
    static std::vector<unsigned char> title_table;  // The bound heap retains this backing memory.
    if (joined_table) {
        if (stride < 32) return 42;
        title_table.resize(2 * stride);
    }
    if (!bind_view_table(guest, list, cpu, joined_table ? 2 : 1, joined_table ? ptr(title_table.data()) : 0,
                         joined_table ? title_table.size() : 0)) return 42;
    if (joined_table && cpu != ptr(title_table.data())) return 42;
    ViewAt v{};
    v.view.Format = DXGI_FORMAT_R8_UNORM;
    v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.view.Texture2D.MipLevels = 1;
    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    if (joined_table) {
        for (UINT swap = 0; swap < 2; ++swap) {
            for (UINT i = 0; i < 2; ++i)
                invoke(guest, 18, ptr(placed[i ^ swap]), ptr(&v.view), cpu + i * stride);
            // The shader uses packed 32-byte records. Publish both views at native
            // slot boundaries, then pack the second token in the guest-owned table.
            std::memmove(title_table.data() + 32, title_table.data() + stride, 32);
            for (UINT key = 0; key < 4; ++key) {
                invoke(list, 40, 1, 1, ptr(&key), 0);
                draw(static_cast<LONG>(swap * 4 + key));
            }
        }
        return 0;
    }
    const auto publish = [&](const D3D12_RESOURCE_DESC& resource_desc, Word address) {
        v.address = address;
        invoke(guest, 70, ptr(&resource_desc), ptr(&v), cpu);
    };
    invoke(guest, 18, ptr(placed[0]), ptr(&v.view), cpu);
    draw(0); // Ordinary resource-object SRV is the independent control.
    publish(desc, ptr(planes[1].data())); draw(1);
    publish(desc, ptr(planes[0].data())); draw(2);
    publish(desc, 0); draw(3); // Rejection preserves the previous descriptor.
    publish(desc, ptr(planes[1].data())); draw(4);
    auto wrong_desc = desc;
    wrong_desc.Width = 1;
    publish(wrong_desc, ptr(planes[0].data())); draw(5);
    publish(desc, ptr(planes[0].data()) + 1); // Interior address, not the registered base.
    // Refusals that name their cause in the log; draw 6 shows each left the descriptor alone.
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = 256;
    buffer_desc.Height = buffer_desc.DepthOrArraySize = buffer_desc.MipLevels = buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ViewAt buffer_view{};
    buffer_view.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    buffer_view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    buffer_view.view.Buffer.NumElements = 64;
    buffer_view.view.Buffer.StructureByteStride = 4;
    buffer_view.address = ptr(planes[0].data());
    invoke(guest, 70, ptr(&buffer_desc), ptr(&buffer_view), cpu);
    v.address = ptr(planes[0].data());
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu + 1); // Not the start of a descriptor in the view heap.
    draw(6);
    publish(desc, ptr(planes[0].data())); draw(7);
    return 0;
}

// Exercise the same four columns twice. The fourth view deliberately names a 2D resource.
int draw_1d_srvs(void* guest, void* list)
{
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE1D, 0, 4, 1, 1, 1,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    static std::array<std::array<unsigned char, 16>, 4> data;
    void* resources[4]{};
    for (unsigned i = 0; i < data.size(); ++i) {
        auto shape = desc;
        if (i == 3) shape.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        for (unsigned x = 0; x < 4; ++x) {
            data[i][4*x] = static_cast<unsigned char>(7 + 3*x);
            data[i][4*x+3] = static_cast<unsigned char>(24 + 48*i + 8*x);
        }
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(data[i].data()), ptr(&shape),
                    D3D12_RESOURCE_STATE_COMMON, 0, ptr(&__uuidof(ID3D12Resource)), ptr(&resources[i]))))) return 41;
    }
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    for (unsigned x = 0; x < 8; ++x) {
        const unsigned i = x % 4;
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = desc.Format;
        view.ViewDimension = i == 3 ? D3D12_SRV_DIMENSION_TEXTURE2D : D3D12_SRV_DIMENSION_TEXTURE1D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture1D.MipLevels = 1;
        invoke(guest, 18, ptr(resources[i]), i == 1 ? 0 : ptr(&view), cpu);
        const D3D12_RECT strip{LONG(x), 0, LONG(x + 1), 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    }
    return 0;
}

// Columns 0..5 sample a single cube. Column 6 reuses that resource with an array view.
// Column 7 uses a cube view over a resource containing two cubes.
int draw_cube_srvs(void* guest, void* list)
{
    constexpr std::size_t slice_stride = 64 * 1024;  // Placed title texture slice alignment.
    alignas(4096) static std::array<std::array<unsigned char, 12 * slice_stride>, 2> data{};
    void* textures[2]{};
    for (unsigned i = 0; i < data.size(); ++i) {
        const auto slices = static_cast<UINT16>(6 * (i + 1));
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, slices, 1,
            DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        for (unsigned face = 0; face < slices; ++face) {
            const unsigned char color[] = {static_cast<unsigned char>(31 + 29 * (face % 6)), 73, 181, 255};
            std::memcpy(data[i].data() + face * slice_stride, color, sizeof(color));
        }
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(data[i].data()), ptr(&desc),
                D3D12_RESOURCE_STATE_COMMON, 0, ptr(&__uuidof(ID3D12Resource)), ptr(&textures[i]))))) return 41;
    }

    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    D3D12_SHADER_RESOURCE_VIEW_DESC cube{};
    cube.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    cube.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    cube.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    cube.TextureCube.MipLevels = 1;
    auto array = cube;
    array.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    array.Texture2DArray = {0, 1, 0, 6, 0, 0};

    const auto draw = [&](LONG left, LONG right) {
        const D3D12_RECT strip{left, 0, right, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    invoke(guest, 18, ptr(textures[0]), ptr(&cube), cpu);
    draw(0, 6);
    invoke(guest, 18, ptr(textures[0]), ptr(&array), cpu);
    draw(6, 7);
    invoke(guest, 18, ptr(textures[1]), ptr(&cube), cpu);
    draw(7, 8);
    return 0;
}

int draw_desktop_array_srvs(void* guest, void* list)
{
    constexpr std::size_t slice_stride = 64 * 1024;
    constexpr auto color_format = DXGI_FORMAT_R8G8B8A8_UNORM;
    alignas(4096) static std::array<unsigned char, 6 * slice_stride> cube_small{};
    alignas(4096) static std::array<unsigned char, 12 * slice_stride> cube_large{};
    const auto fill = [slice_stride](unsigned char* data, UINT slices, UINT side, UINT red) {
        for (UINT slice = 0; slice < slices; ++slice)
            for (UINT y = 0; y < side; ++y)
                for (UINT x = 0; x < side; ++x) {
                    const unsigned char color[] = {static_cast<unsigned char>(red + 13 * slice),
                        static_cast<unsigned char>(slice < 6 ? 53 : 53 + 61 * y),
                        static_cast<unsigned char>(slice < 6 ? 197 : 197 - 73 * x), 255};
                    // Using an unrounded footprint stride reads zero padding for faces 1 through 5.
                    std::memcpy(data + slice * slice_stride + y * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT + 4 * x,
                                color, sizeof(color));
                }
    };
    fill(cube_small.data(), 6, 1, 19);
    fill(cube_large.data(), 12, 2, 31);

    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    D3D12_RESOURCE_DESC shape{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 2, 2, 12, 1, color_format,
        {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, static_cast<D3D12_RESOURCE_FLAGS>(0x300000)};
    ViewAt binding{};
    binding.view.Format = color_format;
    binding.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    binding.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    binding.view.TextureCube.MipLevels = 1;
    binding.address = ptr(cube_large.data());

    const auto draw = [&](LONG top, LONG bottom) {
        invoke(guest, 70, ptr(&shape), ptr(&binding), cpu);
        const D3D12_RECT region{0, top, 8, bottom};
        invoke(list, 22, 1, ptr(&region));
        invoke(list, 12, 3, 1);
    };
    if (GetEnvironmentVariableW(L"X2W_TEST_DESKTOP_ARRAY_SLICES", nullptr, 0)) {
        binding.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        binding.view.Texture2DArray = {0, 1, 6, 6, 0, 0};
        draw(0, 4);
    } else {
        draw(2, 4);
        shape.Width = shape.Height = 1;
        shape.DepthOrArraySize = 6;
        binding.address = ptr(cube_small.data());
        draw(0, 2);

        const void* slot = reinterpret_cast<const void*>(cpu);
        std::array<unsigned char, 32> previous;
        std::memcpy(previous.data(), slot, previous.size());
        shape.DepthOrArraySize = 5;
        invoke(guest, 70, ptr(&shape), ptr(&binding), cpu);
        if (std::memcmp(previous.data(), slot, previous.size()) != 0) return 48;
        shape.DepthOrArraySize = 6;
        shape.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        invoke(guest, 70, ptr(&shape), ptr(&binding), cpu);
        if (std::memcmp(previous.data(), slot, previous.size()) != 0) return 48;
    }
    return 0;
}

int draw_bc_address_srvs(void* guest, void* list)
{
    wchar_t path[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableW(L"X2W_TEST_BC_VIEW", path, MAX_PATH);
    if (!length || length >= MAX_PATH) return 43;
    const bool bc3 = GetEnvironmentVariableW(L"X2W_TEST_BC3", nullptr, 0) != 0;
    // X2W_TEST_BC_SIZE=<width>x<height> (default 128x128) keeps the 2x2 tiles; a smaller size leaves the
    // last tile column and row partial, so the padding blocks must stay outside the texture.
    constexpr UINT tile = 4096, panel = 128;
    UINT width = panel, height = panel;
    wchar_t size[16]{};
    if (GetEnvironmentVariableW(L"X2W_TEST_BC_SIZE", size, 16) &&
        (swscanf_s(size, L"%ux%u", &width, &height) != 2 || width > panel || height > panel)) return 43;
    alignas(4096) static std::array<std::array<char, 4 * tile>, 2> backing{};
    std::ifstream input(path, std::ios::binary);
    if (!input.read(backing[0].data(), 4 * tile) || input.peek() != std::char_traits<char>::eof()) return 43;
    for (UINT i = 0; i < 4; ++i)
        std::memcpy(backing[1].data() + i * tile, backing[0].data() + (i ^ 1) * tile, tile);
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        bc3 ? DXGI_FORMAT_BC3_UNORM : DXGI_FORMAT_BC7_UNORM, {1, 0}, static_cast<D3D12_TEXTURE_LAYOUT>(0x105),
        bc3 ? D3D12_RESOURCE_FLAG_NONE : static_cast<D3D12_RESOURCE_FLAGS>(0x300000)};
    ViewAt v{};
    v.view.Format = desc.Format;
    v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.view.Texture2D.MipLevels = 1;
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    const Word addresses[] = {ptr(backing[0].data()), ptr(backing[1].data()), 0, ptr(backing[0].data())};
    for (LONG i = 0; i < 4; ++i) {
        v.address = addresses[i];
        invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
        // Keep the already-published descriptor while the backing changes to a row layout.
        if (i == 3 && (width % 64 || height % 64))
            x2w_d3d12_texture_written(backing[0].data(), backing[0].size());
        const D3D12_RECT region{i * LONG(panel), 0, (i + 1) * LONG(panel), LONG(panel)};
        invoke(list, 22, 1, ptr(&region));
        invoke(list, 12, 3, 1);
    }
    // Refused, with nothing drawn after them (the check reads the log): not whole 4x4 blocks in either
    // direction, and padded tiles past the size XG reports (1028 x 1028 pads to 1,183,744 bytes; the
    // footprint rounded to 64 KiB is 1,179,648).
    const auto refused = [&](UINT64 w, UINT h) {
        auto shape = desc;
        shape.Width = w;
        shape.Height = h;
        v.address = ptr(backing[1].data());
        invoke(guest, 70, ptr(&shape), ptr(&v), cpu);
    };
    refused(width - 2, height);
    refused(width, height - 2);
    refused(1028, 1028);
    // Published at the boundary: 256 x 196 (only the height partial) pads to 64 KiB, exactly the reported size.
    alignas(4096) static std::array<char, 16 * tile> boundary{};
    auto exact = desc;
    exact.Width = 256;
    exact.Height = 196;
    v.address = ptr(boundary.data());
    invoke(guest, 70, ptr(&exact), ptr(&v), cpu);
    // A partial-tile view over memory the XG stand-in announced as copied rows stays refused; over blocks
    // its opt-in BC7 4 KiB copy announced as tiled, it is published. Only partial sizes are affected.
    if (width % 64 || height % 64) {
        alignas(4096) static std::array<std::array<char, 4 * tile>, 2> announced{};
        x2w_d3d12_texture_written(announced[0].data(), announced[0].size());
        x2w_d3d12_bc7_4k_s_written(announced[1].data(), width, height);
        for (const auto& backing : announced) {
            v.address = ptr(backing.data());
            invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
        }
        // Published first, then overwritten by an XG row copy: a second request is refused and leaves the
        // destination naming the view it held before (the second backing's).
        alignas(4096) static std::array<char, 4 * tile> reused{};
        v.address = ptr(reused.data());
        invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
        v.address = ptr(backing[1].data());
        invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
        v.address = ptr(reused.data());
        std::array<char, 32> written{};
        std::memcpy(written.data(), reinterpret_cast<const void*>(cpu), written.size());
        x2w_d3d12_texture_written(reused.data(), reused.size());
        invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
        if (std::memcmp(written.data(), reinterpret_cast<const void*>(cpu), written.size())) return 44;
    }
    return 0;
}

// X2W_TEST_CONSOLE_VIEW names a file of texels and X2W_TEST_CONSOLE_DESC the texture they belong to
// ("width,height,mips,format,layout,viewed mip", and a depth for a volume viewed as a Texture3D). No
// resource object exists at the address; one mip is viewed and drawn over the whole target. With X2W_TEST_CONSOLE_XG_ROWS=1 the address is first announced
// the way the XG stand-in announces a copy of desktop-footprint rows; with =after the announcement comes
// between the publication and the draw.
int draw_console_address_srv(void* guest, void* list)
{
    wchar_t path[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableW(L"X2W_TEST_CONSOLE_VIEW", path, MAX_PATH);
    char text[96]{};
    unsigned width = 0, height = 0, mips = 0, format = 0, layout = 0, viewed = 0, depth = 0;
    if (!length || length >= MAX_PATH || !GetEnvironmentVariableA("X2W_TEST_CONSOLE_DESC", text, sizeof(text)) ||
        std::sscanf(text, "%u,%u,%u,%u,%u,%u,%u", &width, &height, &mips, &format, &layout, &viewed, &depth) < 6) return 43;
    alignas(4096) static std::array<char, 1 << 20> backing{};
    std::ifstream input(path, std::ios::binary);
    input.read(backing.data(), backing.size());
    if (!input.gcount() || !input.eof()) return 43;
    const D3D12_RESOURCE_DESC desc{depth ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width,
        height, static_cast<UINT16>(depth ? depth : 1), static_cast<UINT16>(mips), static_cast<DXGI_FORMAT>(format), {1, 0},
        static_cast<D3D12_TEXTURE_LAYOUT>(layout), D3D12_RESOURCE_FLAG_NONE};
    ViewAt v{};
    v.view.Format = desc.Format;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (depth) {
        v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        v.view.Texture3D.MostDetailedMip = viewed;
        v.view.Texture3D.MipLevels = 1;
    } else {
        v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.view.Texture2D.MostDetailedMip = viewed;
        v.view.Texture2D.MipLevels = 1;
    }
    v.address = ptr(backing.data());
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    wchar_t rows[8]{};
    GetEnvironmentVariableW(L"X2W_TEST_CONSOLE_XG_ROWS", rows, 8);
    if (!std::wcscmp(rows, L"1")) x2w_d3d12_texture_written(backing.data(), backing.size());
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
    // "after": the announcement follows the publication, before the draw samples the view.
    if (!std::wcscmp(rows, L"after")) x2w_d3d12_texture_written(backing.data(), backing.size());
    const D3D12_RECT whole{0, 0, 512, 128};
    invoke(list, 22, 1, ptr(&whole));
    invoke(list, 12, 3, 1);
    return 0;
}

// X2W_TEST_CONSOLE_GATE: one 2048x2048 RGBA8 texture in the 64 KiB standard layout (16 MiB), republished
// from the same address with a new payload before each draw. Every texel of a payload has one colour
// (gate_colour in tests/startup_address_console_check.py), so the bytes do not depend on the layout. The
// draw covers the left or the right half of the 512x128 target.
int draw_gated_console_payload(void* guest, void* list, unsigned payload, bool left)
{
    constexpr unsigned side = 2048;
    constexpr std::size_t bytes = std::size_t{side} * side * 4;
    static auto* const backing = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!backing) return 43;
    const unsigned char colour[4] = {static_cast<unsigned char>(17 * payload + 3), static_cast<unsigned char>(255 - 9 * payload),
                                     static_cast<unsigned char>(payload ^ 0x5a), 255};
    for (std::size_t at = 0; at < bytes; at += 4) std::memcpy(backing + at, colour, 4);
    static void* heap;
    static Word cpu;
    if (!heap) {
        struct Extended {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word address, size;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
        if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&heap))))) return 42;
        cpu = invoke(heap, 9);
    }
    invoke(list, 32, 1, ptr(&heap));
    invoke(list, 36, 0, invoke(heap, 10));
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
        {1, 0}, static_cast<D3D12_TEXTURE_LAYOUT>(0x109), D3D12_RESOURCE_FLAG_NONE};
    ViewAt v{};
    v.view.Format = desc.Format;
    v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.view.Texture2D.MipLevels = 1;
    v.address = ptr(backing);
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
    const D3D12_RECT half{left ? 0 : 256, 0, left ? 256 : 512, 128};
    invoke(list, 22, 1, ptr(&half));
    invoke(list, 12, 3, 1);
    return 0;
}

// The backing is produced by XG CopyIntoSubresource from a source-built block texture.
int draw_bc_chain(void* guest, void* list)
{
    wchar_t path[MAX_PATH]{};
    const DWORD size = GetEnvironmentVariableW(L"X2W_TEST_BC_CHAIN", path, MAX_PATH);
    if (!size || size >= MAX_PATH) return 43;
    const bool array = GetEnvironmentVariableW(L"X2W_TEST_BC_ARRAY", nullptr, 0) != 0;
    const bool rectangular = GetEnvironmentVariableW(L"X2W_TEST_BC_RECT", nullptr, 0) != 0;
    const UINT height = rectangular ? 512 : 1024;
    const UINT16 slices = array ? 2 : 1;
    const UINT16 mips = array ? 9 : 6;
    const UINT bytes = slices * (rectangular ? 0xc0000 : 0x160000);
    static std::array<char, 2 * 0x160000 + 0xffff> storage{};
    char* backing = reinterpret_cast<char*>((ptr(storage.data()) + 0xffff) & ~Word{0xffff});
    std::ifstream input(path, std::ios::binary);
    if (!input.read(backing, bytes) || input.peek() != std::char_traits<char>::eof()) return 43;
    const bool bc5 = GetEnvironmentVariableW(L"X2W_TEST_BC5", nullptr, 0) != 0;
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1024, height, slices, mips,
        bc5 ? DXGI_FORMAT_BC5_UNORM : DXGI_FORMAT_BC7_UNORM, {1, 0}, static_cast<D3D12_TEXTURE_LAYOUT>(0x109),
        D3D12_RESOURCE_FLAG_NONE};
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    ViewAt v{};
    v.address = ptr(backing);
    v.view.Format = desc.Format;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (array) {
        v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        v.view.Texture2DArray.ArraySize = slices;
        v.view.Texture2DArray.MipLevels = mips;
    } else {
        v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.view.Texture2D.MipLevels = mips;
    }
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
    invoke(list, 12, 3, 1);
    return 0;
}

namespace {
struct PlacedPair {
    std::array<DWORD*, 2> memory{};
    std::array<void*, 2> resource{};
    bool rebind{};
} placed_pair;
}

int draw_placed_vertices(ID3D12Device* desktop, void* guest, void* list)
{
    auto& p = placed_pair;
    D3D12_FEATURE_DATA_EXISTING_HEAPS heaps{};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &heaps, sizeof(heaps))) || !heaps.Supported)
        return 77;
    if (!p.memory[0]) {
        constexpr auto flags = MEM_RESERVE | MEM_COMMIT;
        for (Word base = 0x1600000000ull; base < 0x1700000000ull; base += 16ull << 20) {
            auto* low = static_cast<DWORD*>(VirtualAlloc(reinterpret_cast<void*>(base), 65536, flags, PAGE_READWRITE));
            if (!low) continue;
            auto* high = static_cast<DWORD*>(VirtualAlloc(reinterpret_cast<void*>(base + (1ull << 40)), 65536, flags, PAGE_READWRITE));
            if (high) { p.memory = {low, high}; break; }
            VirtualFree(low, 0, MEM_RELEASE);
        }
        if (!p.memory[0]) return 45;
    }
    for (UINT i = 0; i < 2; ++i) {
        if (p.resource[i]) continue;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = p.rebind && i ? 516 : 260;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const auto hr = static_cast<HRESULT>(invoke(guest, 133, ptr(p.memory[i]), ptr(&desc),
            D3D12_RESOURCE_STATE_COMMON, 0, ptr(&__uuidof(ID3D12Resource)), ptr(&p.resource[i])));
        if (FAILED(hr) || !p.resource[i] || invoke(p.resource[i], 11) != ptr(p.memory[i])) return 46;
    }
    const UINT hi = p.rebind ? 128 : 64;
    p.memory[0][64] = 10; p.memory[1][hi] = 20;
    auto draw = [&](LONG column, DWORD* word, UINT bytes) {
        const D3D12_VERTEX_BUFFER_VIEW view{ptr(word), bytes, 4};
        invoke(list, 48, 0, 1, ptr(&view));
        const D3D12_RECT rect{column, 0, column + 1, 4};
        invoke(list, 22, 1, ptr(&rect));
        invoke(list, 12, 3, 1);
    };
    for (LONG col = 0; col < 8; ++col) {
        switch (col % 4) {
        case 0: draw(col, p.memory[0] + 64, 4); break;
        case 1: draw(col, p.memory[1] + hi, 4); break;
        case 2: draw(col, p.memory[1] + hi, 8); break; // Reject before requesting data; shader only loads byte 0.
        case 3: draw(col, p.rebind ? p.memory[1] + 64 : p.memory[0] + 64, 4); break;
        }
    }
    // Legitimate writes to the last DWORD of each buffer, before queue submission.
    // A snapshot made when recording the list would publish the previous values.
    p.memory[0][64] = p.rebind ? 49 : 41;
    p.memory[1][hi] = p.rebind ? 153 : 135;
    if (p.rebind) p.memory[1][64] = 179; // Reused address overlaps the retired smaller registration.
    return 0;
}

void release_placed_vertices(bool done)
{
    auto& p = placed_pair;
    invoke(p.resource[1], 2);
    p.resource[1] = nullptr;
    p.rebind = true;
    if (done) {
        invoke(p.resource[0], 2);
        p.resource[0] = nullptr;
    }
}

// X2W_TEST_FORMATTED_VIEW=<DXGI format>:<mapping>:<mode>:<64 hex bytes>. Modes 0, 1 and 2
// create a typed view, raw view and oversized R32_UINT view respectively. The Python fixture
// reads the selected element through buffer_load_format_xyzw and checks the target readback.
int draw_formatted_view(void* guest, void* list)
{
    char spec[256]{};
    if (!GetEnvironmentVariableA("X2W_TEST_FORMATTED_VIEW", spec, sizeof(spec))) return 48;
    unsigned format = 0, mapping = 0, mode = 0;
    char hex[129]{};
    if (std::sscanf(spec, "%u:%x:%u:%128s", &format, &mapping, &mode, hex) != 4 || std::strlen(hex) != 128 || mode > 2) return 48;
    static std::array<std::uint8_t, 65536 + 256> backing{};
    constexpr UINT over_limit = (16u << 20) + 4u;
    const bool oversized = mode == 2;
    if (oversized && format != DXGI_FORMAT_R32_UINT) return 48;
    auto* const bytes = oversized
        ? static_cast<std::uint8_t*>(VirtualAlloc(nullptr, over_limit + 65536u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
        : reinterpret_cast<std::uint8_t*>((ptr(backing.data()) + 0xffff) & ~Word{0xffff});
    if (!bytes) return 48;
    for (unsigned i = 0; i < 64; ++i) {
        unsigned value = 0;
        if (std::sscanf(hex + 2 * i, "%2x", &value) != 1) return 48;
        bytes[i] = static_cast<std::uint8_t>(value);
    }
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = oversized ? over_limit : 256u;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* placed = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(bytes), ptr(&desc), D3D12_RESOURCE_STATE_COMMON,
                                           0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 47;
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    D3D12_SHADER_RESOURCE_VIEW_DESC view{static_cast<DXGI_FORMAT>(format), D3D12_SRV_DIMENSION_BUFFER, mapping};
    view.Buffer.NumElements = oversized ? over_limit / 4u : 4u;
    view.Buffer.Flags = mode == 1 ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
    invoke(guest, 18, ptr(placed), ptr(&view), cpu);
    invoke(list, 12, 3, 1);
    char repeat[2]{};
    if (GetEnvironmentVariableA("X2W_TEST_FORMATTED_VIEW_REPEAT", repeat, sizeof(repeat)) == 1 && repeat[0] == '1')
        invoke(list, 12, 3, 1);
    return 0;
}

// X2W_TEST_TABLE_ALIAS_WRITE: a buffer view the title creates over a placed buffer, at table entry 0
// (tests/startup_table_alias_check.py).
//   1             a raw view of the whole 256-byte buffer. The title writes the record, draws the left half, writes
//                 it again, draws the right half, and writes it a third time after recording, before the list runs.
//                 A per-draw copy shows the first two values; a live alias shows the third in both halves.
//   shared-index  the same buffer is also the index buffer of one indexed draw (indices 0-2 at offset 0). The alias
//                 must be in both the index-buffer and the shader-resource states at that draw.
//   odd-size      a 254-byte buffer under a structured view of 127 two-byte records. The desktop raw view covers
//                 256 bytes, past the alias, so the draw takes a copy: it shows the value written before recording.
// With X2W_TEST_DEBUG_LAYER, errors the debug layer stored for the device from the fixture's start
// (ID3D12InfoQueue; the message callback of ID3D12InfoQueue1 needs a newer runtime).
ID3D12InfoQueue* table_alias_messages = nullptr;

void table_alias_debug_report()
{
    if (!table_alias_messages) return;
    int errors = 0;
    const UINT64 stored = table_alias_messages->GetNumStoredMessages();
    for (UINT64 i = 0; i < stored; ++i) {
        SIZE_T bytes = 0;
        if (FAILED(table_alias_messages->GetMessage(i, nullptr, &bytes)) || !bytes) continue;
        std::vector<unsigned char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(table_alias_messages->GetMessage(i, message, &bytes)) || message->Severity > D3D12_MESSAGE_SEVERITY_ERROR)
            continue;
        if (++errors <= 8) std::printf("debug layer id %d: %.300s\n", static_cast<int>(message->ID), message->pDescription);
    }
    std::printf("table alias debug layer errors: %d\n", errors);
}

int draw_table_alias_writes(ID3D12Device* desktop, void* guest, void* list)
{
    D3D12_FEATURE_DATA_EXISTING_HEAPS heaps{};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &heaps, sizeof(heaps))) || !heaps.Supported)
        return 77;
    if (GetEnvironmentVariableW(L"X2W_TEST_DEBUG_LAYER", nullptr, 0)) {
        if (FAILED(desktop->QueryInterface(IID_PPV_ARGS(&table_alias_messages)))) {
            std::printf("no debug layer message queue: skipped\n");
            return 77;
        }
        table_alias_messages->SetMessageCountLimit(4096);
        table_alias_messages->ClearStoredMessages();
    }
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_TABLE_ALIAS_WRITE", mode, sizeof(mode));
    const bool shared_index = !std::strcmp(mode, "shared-index");
    const bool odd_size = !std::strcmp(mode, "odd-size");
    if (!shared_index && !odd_size && std::strcmp(mode, "1")) return 70;
    // VirtualAlloc memory, so the placement can alias it (an existing heap cannot open image memory).
    auto* const records = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!records) return 71;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, odd_size ? 254u : 256u, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    void* placed = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(records), ptr(&desc), D3D12_RESOURCE_STATE_COMMON,
                                           0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 47;
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    D3D12_SHADER_RESOURCE_VIEW_DESC view{odd_size ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_TYPELESS,
                                         D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
    view.Buffer.NumElements = odd_size ? 127 : 64;
    view.Buffer.StructureByteStride = odd_size ? 2 : 0;
    view.Buffer.Flags = odd_size ? D3D12_BUFFER_SRV_FLAG_NONE : D3D12_BUFFER_SRV_FLAG_RAW;
    invoke(guest, 18, ptr(placed), ptr(&view), cpu);
    if (shared_index || odd_size) {
        auto* const indices = reinterpret_cast<std::uint16_t*>(records);
        for (std::uint16_t i = 0; i < 3; ++i) indices[i] = i;
        records[4] = 40;
        records[5] = 41;
        const D3D12_RECT full{0, 0, 8, 4};
        invoke(list, 22, 1, ptr(&full));
        if (shared_index) {
            const D3D12_INDEX_BUFFER_VIEW index_view{ptr(records), 6, DXGI_FORMAT_R16_UINT};
            invoke(list, 47, ptr(&index_view));
            invoke(list, 13, 3, 1, 0, 0, 0);
        } else {
            invoke(list, 12, 3, 1);
            records[4] = 120;  // after recording: only a live alias would show it
            records[5] = 121;
        }
        return 0;
    }
    const auto write = [&](std::uint32_t value) {
        records[4] = value;
        records[5] = value + 1;
    };
    const auto draw = [&](LONG left, LONG right) {
        const D3D12_RECT strip{left, 0, right, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    write(40);
    draw(0, 4);
    write(80);
    draw(4, 8);
    write(120);  // after recording: only a binding that reads at execution sees it
    return 0;
}

int draw_copied_vertex_buffer_srv(void* guest, void* list)
{
    constexpr UINT full_bytes = 512;
    constexpr UINT short_bytes = 256;
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, full_bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    std::uint32_t* records = nullptr;
    void* placed = nullptr;

    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_COPIED_VERTEX_SRV", mode, sizeof(mode));
    if (std::strcmp(mode, "copy15") == 0) {
        // Use fresh, deliberately unaligned placements so the draw snapshots
        // the title destination. Give the untouched destination a different
        // color so a skipped copy fails the pixel comparison.
        static constexpr std::size_t page = 4096;
        auto allocate = [] {
            auto* base = static_cast<unsigned char*>(VirtualAlloc(nullptr, 2 * page,
                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            return base ? reinterpret_cast<std::uint32_t*>(base + page) : nullptr;
        };
        auto* source = allocate();
        records = allocate();
        void* from = nullptr;
        if (!source || !records ||
            FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(source), ptr(&buffer),
                D3D12_RESOURCE_STATE_COMMON, 0, ptr(&__uuidof(ID3D12Resource)), ptr(&from)))) ||
            FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(records), ptr(&buffer),
                D3D12_RESOURCE_STATE_COMMON, 0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 47;
        source[224 / sizeof(*source)] = 73;
        source[288 / sizeof(*source)] = 149;
        for (unsigned i = 0; i < 2000; ++i) {
            const Word offset = (i % 8) * 64;
            invoke(list, 15, ptr(placed), offset, ptr(from), offset, 64);
        }
    } else {
        static std::array<std::uint8_t, 65536 + full_bytes> storage{};
        records = reinterpret_cast<std::uint32_t*>((ptr(storage.data()) + 0xffff) & ~Word{0xffff});
        records[224 / sizeof(*records)] = 57;
        records[288 / sizeof(*records)] = 181;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(records), ptr(&buffer), D3D12_RESOURCE_STATE_COMMON,
                                               0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 47;
    }

    const Word stride = invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    alignas(64) static std::array<std::uint8_t, 256> descriptors{};
    Word cpu = 0;
    if (!stride || stride > descriptors.size() / 3 ||
        !bind_view_table(guest, list, cpu, 3, ptr(descriptors.data()), 3 * stride) ||
        cpu != ptr(descriptors.data())) return 42;

    ViewAt view{};
    view.view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    view.address = ptr(records);
    view.view.Buffer.NumElements = full_bytes / sizeof(*records);
    invoke(guest, 70, ptr(&buffer), ptr(&view), cpu + stride);
    view.view.Buffer.NumElements = short_bytes / sizeof(*records);
    invoke(guest, 70, ptr(&buffer), ptr(&view), cpu + 2 * stride);
    if (std::memcmp(descriptors.data() + stride, descriptors.data() + 2 * stride, stride) == 0) return 54;

    const auto copy_and_draw = [&](Word source, LONG left, LONG right) {
        invoke(guest, 24, 1, cpu, cpu + source * stride, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (std::memcmp(descriptors.data(), descriptors.data() + source * stride, stride) != 0) return false;
        const D3D12_RECT strip{left, 0, right, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
        return true;
    };
    if (!copy_and_draw(1, 0, 2) || !copy_and_draw(2, 2, 6) || !copy_and_draw(1, 6, 8)) return 55;
    std::printf("copied vertex buffer SRV with %llu-byte descriptors\n", static_cast<unsigned long long>(stride));
    return 0;
}

int draw_copied_vertex_srv_pair(void* guest, void* list)
{
    constexpr UINT bytes = 512;
    constexpr Word record_bytes = 32;
    static std::uint8_t inputs[2][65536 + bytes]{};
    constexpr std::uint32_t values[]{181, 211};
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    const Word stride = invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    alignas(64) static std::array<std::uint8_t, 256> descriptors{};
    Word cpu = 0;
    if (stride < record_bytes || stride > descriptors.size() / 4 ||
        !bind_view_table(guest, list, cpu, 4, ptr(descriptors.data()), 4 * stride) ||
        cpu != ptr(descriptors.data())) return 42;

    ViewAt view{};
    view.view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    view.view.Buffer.NumElements = bytes / sizeof(std::uint32_t);
    for (Word i = 0; i < 2; ++i) {
        void* placed = nullptr;
        view.address = (ptr(inputs[i]) + 0xffff) & ~Word{0xffff};
        reinterpret_cast<std::uint32_t*>(view.address)[288 / sizeof(std::uint32_t)] = values[i];
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, view.address, ptr(&buffer), D3D12_RESOURCE_STATE_COMMON,
                                               0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 47;
        invoke(guest, 70, ptr(&buffer), ptr(&view), cpu + (2 + i) * stride);
    }
    auto* const records = descriptors.data();
    if (std::memcmp(records + 2 * stride, records + 3 * stride, record_bytes) == 0) return 54;

    const auto draw = [&](Word first, Word second, LONG left, LONG right) {
        invoke(guest, 24, 1, cpu, cpu + first * stride, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        invoke(guest, 24, 1, cpu + stride, cpu + second * stride, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // The shader reads fixed-width guest records. Copy via the reported descriptor stride first.
        std::memmove(records + record_bytes, records + stride, record_bytes);
        if (std::memcmp(records, records + first * stride, record_bytes) != 0 ||
            std::memcmp(records + record_bytes, records + second * stride, record_bytes) != 0) return false;
        const D3D12_RECT strip{left, 0, right, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
        return true;
    };
    if (!draw(2, 3, 0, 2) || !draw(3, 2, 2, 6) || !draw(2, 3, 6, 8)) return 55;
    return 0;
}

int draw_address_buffer_srvs(ID3D12Device* desktop, void* guest, void* list)
{
    static std::array<std::uint8_t, 65536 + 256> raw_buf0{};
    static auto* const buf1 = static_cast<std::uint32_t*>(
        VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    auto* const buf0 = reinterpret_cast<std::uint32_t*>((ptr(raw_buf0.data()) + 0xffff) & ~Word{0xffff});
    std::memset(buf0, 0, 256);
    std::memset(buf1, 0, 256);
    buf0[0] = 17;
    buf0[1] = 99;
    buf0[4] = 63;
    buf1[4] = 141;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    void* placed = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(buf0), ptr(&desc), D3D12_RESOURCE_STATE_COMMON,
                                           0, ptr(&__uuidof(ID3D12Resource)), ptr(&placed))))) return 47;
    // buf1 is intentionally not placed in slot 133 to verify unplaced buffer backing in device slot 70.

    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;

    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };

    ViewAt raw{};
    raw.view.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    raw.view.Buffer.NumElements = 4;

    raw.address = ptr(buf0);
    invoke(guest, 70, ptr(&desc), ptr(&raw), cpu);
    draw(0);

    raw.address = ptr(buf0 + 4);
    invoke(guest, 70, ptr(&desc), ptr(&raw), cpu);
    draw(1);

    raw.address = ptr(buf1);
    raw.view.Buffer.FirstElement = 4;
    if (!failed_publication_keeps_view(guest, desc, raw, cpu)) return 54;
    invoke(guest, 70, ptr(&desc), ptr(&raw), cpu);
    draw(2);

    // Unaligned interior address for a raw buffer must be refused and keep the previous descriptor.
    raw.view.Buffer.FirstElement = 0;
    raw.address = ptr(buf0 + 1);
    invoke(guest, 70, ptr(&desc), ptr(&raw), cpu);
    draw(3);

    ViewAt structured{};
    structured.view.Format = DXGI_FORMAT_UNKNOWN;
    structured.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    structured.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    structured.view.Buffer.FirstElement = 1;
    structured.view.Buffer.NumElements = 1;
    structured.view.Buffer.StructureByteStride = 4;
    structured.address = ptr(buf0);
    invoke(guest, 70, ptr(&desc), ptr(&structured), cpu);
    // An acceleration-structure view of a placed buffer that is not in the acceleration-structure state is
    // refused (as_wrong_state); draw(4) proves the structured descriptor stayed in place.
    ViewAt as_view{};
    as_view.view.Format = DXGI_FORMAT_UNKNOWN;
    as_view.view.ViewDimension = static_cast<D3D12_SRV_DIMENSION>(11);
    as_view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    as_view.address = ptr(buf0);
    invoke(guest, 70, ptr(&desc), ptr(&as_view), cpu);
    draw(4);

    raw.address = ptr(buf0);
    raw.view.Buffer.FirstElement = 0;
    invoke(guest, 70, ptr(&desc), ptr(&raw), cpu);
    draw(5);

    // An unplaced structured view whose address is not a whole number of 12-byte elements past its 64 KiB page start
    // (page offset 0x104): published from a copy that starts at the address, so the shader reads 201 at element 0.
    static auto* const suballoc = static_cast<unsigned char*>(VirtualAlloc(nullptr, 2 * 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!suballoc) return 56;
    auto* const odd = reinterpret_cast<std::uint32_t*>((reinterpret_cast<std::uintptr_t>(suballoc) & ~std::uintptr_t{0xffff}) + 65536 + 0x104);
    odd[0] = 201;
    structured.view.Buffer.FirstElement = 0;
    structured.view.Buffer.StructureByteStride = 12;
    structured.address = ptr(odd);
    invoke(guest, 70, ptr(&desc), ptr(&structured), cpu);
    draw(6);

    // An unplaced buffer in an allocation above the 256 MiB limit is not wrapped as an existing heap; its
    // upload copy must still publish the bytes at the address.
    static auto* const large = static_cast<std::uint32_t*>(
        VirtualAlloc(nullptr, (256ull << 20) + 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!large) return 53;
    large[4] = 77;
    raw.address = ptr(large);
    raw.view.Buffer.FirstElement = 4;
    invoke(guest, 70, ptr(&desc), ptr(&raw), cpu);
    draw(7);
    return 0;
}

// One-texel textures published by address (X2W_TEST_SINGLE_ELEMENT=2d or 3d), one 1x4 column per draw. The texel
// sits at offset 0 of its memory whatever the layout field says. Before column 3 the column-2 texel is rewritten
// without a new publication, then sixteen calls are refused; each names memory holding another texel, so any of
// them taking the descriptor would change column 3. Each refused call must also leave the destination's 32 bytes
// (plain memory behind a view heap created without title memory) unchanged; the first that does not returns 60+.
int draw_single_element_srvs(void* guest, void* list)
{
    wchar_t mode[4]{};
    GetEnvironmentVariableW(L"X2W_TEST_SINGLE_ELEMENT", mode, 4);
    const bool volume = mode[0] == L'3';
    alignas(256) static unsigned char memory[8][256];
    const unsigned char a[] = {0x11, 0x22, 0x33, 0xff}, b[] = {0xc0, 0x40, 0x80, 0xff}, c[] = {0x5a, 0xa5, 0x3c, 0xff};
    const unsigned char d[] = {0x81, 0x18, 0xe7, 0xff}, other[] = {0xde, 0xad, 0xbe, 0xef};
    const unsigned char bgra[] = {0x10, 0x20, 0xf0, 0xff};
    const std::uint16_t half[] = {0x3400, 0x3a00, 0x3c00, 0x3c00};  // 0.25, 0.75, 1, 1
    std::memcpy(memory[0], a, 4);
    std::memcpy(memory[1], b, 4);
    std::memcpy(memory[2], bgra, 4);
    std::memcpy(memory[3], half, 8);
    for (int row : {4, 5, 7}) std::memcpy(memory[row], other, 4);
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    const auto texel = [&](DXGI_FORMAT format, UINT layout) {
        return D3D12_RESOURCE_DESC{volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1,
            format, {1, 0}, static_cast<D3D12_TEXTURE_LAYOUT>(layout), static_cast<D3D12_RESOURCE_FLAGS>(0x300000)};
    };
    const auto view_of = [&](const D3D12_RESOURCE_DESC& desc, Word address) {
        ViewAt v{};
        v.view.Format = desc.Format;
        v.view.ViewDimension = volume ? D3D12_SRV_DIMENSION_TEXTURE3D : D3D12_SRV_DIMENSION_TEXTURE2D;
        v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.view.Texture2D.MipLevels = 1;  // Texture3D.MipLevels is at the same offset
        v.address = address;
        return v;
    };
    const auto publish = [&](const D3D12_RESOURCE_DESC& desc, const ViewAt& v) { invoke(guest, 70, ptr(&desc), ptr(&v), cpu); };
    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    const auto rgba = texel(DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    publish(rgba, view_of(rgba, ptr(memory[0]))); draw(0);
    const auto swizzled = texel(DXGI_FORMAT_R8G8B8A8_UNORM, 0x105);
    const auto second_view = view_of(swizzled, ptr(memory[1]));
    if (!failed_publication_keeps_view(guest, swizzled, second_view, cpu)) return 54;
    publish(swizzled, second_view); draw(1);
    std::memcpy(memory[0], c, 4);  // rewritten, then published again at the same address
    publish(rgba, view_of(rgba, ptr(memory[0]))); draw(2);
    std::memcpy(memory[0], d, 4);  // rewritten without a new publication: column 3 must read it
    unsigned char published[32];
    std::memcpy(published, reinterpret_cast<const void*>(cpu), 32);
    int refusals = 0, changed = 0;
    const auto unchanged = [&] {
        ++refusals;
        if (!changed && std::memcmp(published, reinterpret_cast<const void*>(cpu), 32)) changed = refusals;
    };
    const auto refuse = [&](auto edit) {
        auto desc = rgba;
        auto v = view_of(desc, ptr(memory[5]));
        edit(desc, v);
        publish(desc, v);
        unchanged();
    };
    refuse([](auto& d, auto&) { d.Width = 2; });
    refuse([](auto& d, auto&) { d.MipLevels = 2; });
    refuse([](auto& d, auto&) { d.DepthOrArraySize = 2; });
    refuse([](auto& d, auto&) { d.SampleDesc.Count = 4; });
    refuse([](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x115); });  // 4 KiB S with XOR
    refuse([](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x10b); });  // reserved mode
    refuse([](auto& d, auto&) { d.Alignment = 0x400000; });
    refuse([](auto& d, auto& v) { d.Format = v.view.Format = DXGI_FORMAT_BC1_UNORM; });
    refuse([](auto& d, auto& v) { d.Format = v.view.Format = DXGI_FORMAT_R8_UNORM; });   // placed-texture path
    refuse([](auto& d, auto&) { d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; });
    refuse([&](auto&, auto& v) {
        v.view.ViewDimension = volume ? D3D12_SRV_DIMENSION_TEXTURE2D : D3D12_SRV_DIMENSION_TEXTURE3D;
    });
    refuse([](auto&, auto& v) { v.view.Texture2D.MipLevels = 2; });
    refuse([](auto&, auto& v) { v.address = 0; });
    // Placed UAV resources are written on the desktop side. A flagless view at the base of a 1x1 UAV texture,
    // one 256 bytes into a 512-byte UAV buffer, and one that straddles the buffer base stay refused.
    auto written = rgba;
    written.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_RESOURCE_DESC written_buffer{};
    written_buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    written_buffer.Width = 512;
    written_buffer.Height = written_buffer.DepthOrArraySize = written_buffer.MipLevels = written_buffer.SampleDesc.Count = 1;
    written_buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    written_buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    void* uav[2]{};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(memory[4]), ptr(&written), D3D12_RESOURCE_STATE_COMMON, 0,
                                           ptr(&__uuidof(ID3D12Resource)), ptr(&uav[0])))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(memory[6]), ptr(&written_buffer), D3D12_RESOURCE_STATE_COMMON, 0,
                                           ptr(&__uuidof(ID3D12Resource)), ptr(&uav[1]))))) return 48;
    publish(rgba, view_of(rgba, ptr(memory[4])));
    unchanged();
    publish(rgba, view_of(rgba, ptr(memory[7])));
    unchanged();
    publish(rgba, view_of(rgba, ptr(memory[5] + 254)));  // two texel bytes precede the UAV buffer base
    unchanged();
    if (changed) return 60 + changed;
    draw(3);
    const auto bgra_desc = texel(DXGI_FORMAT_B8G8R8A8_UNORM, 0x100);
    publish(bgra_desc, view_of(bgra_desc, ptr(memory[2]))); draw(4);
    const auto half_desc = texel(DXGI_FORMAT_R16G16B16A16_FLOAT, 0x105);
    publish(half_desc, view_of(half_desc, ptr(memory[3]))); draw(5);
    draw(6);  // the same view again
    auto aligned = rgba;
    aligned.Alignment = 0x10000;
    publish(aligned, view_of(aligned, ptr(memory[0]))); draw(7);
    return 0;
}

// Exercise runtime bound-SRV format validation and d16unorm16 mode selection on a shader carrying
// 'd16srv 0 exact' and 'udsrc 3 d16unorm16 0' (#2023):
// - col 0: R16_UNORM (0xffef) + explicit R16_UNORM SRV -> d16unorm16=1 gives exact 0x3bff (yellow 255, 255, 0)
// - col 1: R32_FLOAT (0x3f7ff000 = binary32(0xffef/65535)) + default SRV -> d16unorm16=0 gives 0x3c00 (red 255, 0, 0)
// - col 2: R16_TYPELESS (0xaaaa) + explicit R16_UNORM SRV -> d16unorm16=1 gives 0x3955 (170, 0, 0)
// - cols 3..6: R16_UINT, R16_SNORM, null-resource SRV and unwritten descriptor -> refused (clear blue 0, 0, 255)
// - col 7: rebind R16_UNORM (0xffef) + default SRV -> d16unorm16=1 recovers cleanly (yellow 255, 255, 0)
int draw_d16_srvs(ID3D12Device* desktop, void* guest, void* list)
{
    D3D12_RESOURCE_DESC probe16{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1,
        DXGI_FORMAT_R16_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    D3D12_RESOURCE_DESC probe32 = probe16;
    probe32.Format = DXGI_FORMAT_R32_FLOAT;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp16{}, fp32{};
    UINT64 total16 = 0, total32 = 0;
    desktop->GetCopyableFootprints(&probe16, 0, 1, 0, &fp16, nullptr, nullptr, &total16);
    desktop->GetCopyableFootprints(&probe32, 0, 1, 0, &fp32, nullptr, nullptr, &total32);
    if (!total16 || total16 == ~UINT64{0} || !total32 || total32 == ~UINT64{0}) return 48;

    static std::array<std::vector<unsigned char>, 5> planes;
    const DXGI_FORMAT resource_formats[5] = {
        DXGI_FORMAT_R16_UNORM,
        DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R16_TYPELESS,
        DXGI_FORMAT_R16_UINT,
        DXGI_FORMAT_R16_SNORM,
    };
    const std::uint16_t u16_unorm = 0xffefu, u16_typeless = 0xaaaau, u16_uint = 200u, u16_snorm = 0x4000u;
    const std::uint32_t f32_bits = 0x3f7ff000u;  // binary32(65519 / 65535.0): rounds to 0x3c00 under pack16_rne
    void* textures[5]{};
    for (int i = 0; i < 5; ++i) {
        const bool is32 = i == 1;
        const UINT64 total = is32 ? total32 : total16;
        const UINT64 offset = is32 ? fp32.Offset : fp16.Offset;
        planes[i].assign(static_cast<std::size_t>(total), 0);
        if (i == 0) std::memcpy(planes[i].data() + offset, &u16_unorm, sizeof(std::uint16_t));
        else if (i == 1) std::memcpy(planes[i].data() + offset, &f32_bits, sizeof(std::uint32_t));
        else if (i == 2) std::memcpy(planes[i].data() + offset, &u16_typeless, sizeof(std::uint16_t));
        else if (i == 3) std::memcpy(planes[i].data() + offset, &u16_uint, sizeof(std::uint16_t));
        else std::memcpy(planes[i].data() + offset, &u16_snorm, sizeof(std::uint16_t));
        auto desc = is32 ? probe32 : probe16;
        desc.Format = resource_formats[i];
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(planes[i].data()), ptr(&desc),
                                               D3D12_RESOURCE_STATE_COMMON, 0,
                                               ptr(&__uuidof(ID3D12Resource)), ptr(&textures[i])))))
            return 49;
    }

    struct Extended {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word address, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 9, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&heap)))))
        return 42;
    const Word cpu = invoke(heap, 9);
    const Word gpu = invoke(heap, 10);
    const UINT stride = static_cast<UINT>(invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    if (!cpu || !gpu || stride < 32) return 42;
    invoke(list, 32, 1, ptr(&heap));

    const auto make_view = [](DXGI_FORMAT fmt) {
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = fmt;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Texture2D.MipLevels = 1;
        return v;
    };
    const auto v_unorm = make_view(DXGI_FORMAT_R16_UNORM);
    const auto v_uint = make_view(DXGI_FORMAT_R16_UINT);
    const auto v_snorm = make_view(DXGI_FORMAT_R16_SNORM);

    invoke(guest, 18, ptr(textures[0]), ptr(&v_unorm), cpu + 0 * stride);  // 0: R16_UNORM (0xffef) + explicit R16_UNORM
    invoke(guest, 18, ptr(textures[1]), 0,             cpu + 1 * stride);  // 1: R32_FLOAT (0x3f7ff000) + default SRV
    invoke(guest, 18, ptr(textures[2]), ptr(&v_unorm), cpu + 2 * stride);  // 2: R16_TYPELESS (0xaaaa) + explicit R16_UNORM
    invoke(guest, 18, ptr(textures[2]), ptr(&v_uint),  cpu + 3 * stride);  // 3: R16_TYPELESS + explicit R16_UINT (refused)
    invoke(guest, 18, ptr(textures[3]), 0,             cpu + 4 * stride);  // 4: R16_UINT + default SRV (refused)
    invoke(guest, 18, ptr(textures[4]), ptr(&v_snorm), cpu + 5 * stride);  // 5: R16_SNORM + explicit R16_SNORM (refused)
    invoke(guest, 18, 0,                ptr(&v_unorm), cpu + 6 * stride);  // 6: null resource SRV (refused)
    // Slot 7 is intentionally left unwritten (missing view token -> refused).
    invoke(guest, 18, ptr(textures[0]), 0,             cpu + 8 * stride);  // 8: R16_UNORM (0xffef) + default SRV

    const auto draw_at = [&](LONG x, UINT slot) {
        invoke(list, 36, 0, gpu + Word{slot} * stride);
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    for (UINT col = 0; col < 7; ++col) draw_at(static_cast<LONG>(col), col);
    draw_at(6, 7);  // Also test unwritten descriptor slot on column 6; must stay clear.
    draw_at(7, 8);  // Rebind R16_UNORM default SRV on column 7; must draw yellow (0x3bff) cleanly.
    return 0;
}

// One translated pixel shader uses the same table slot across five draws. The clear color is opaque blue.
int draw_formatted_buffer(void* guest, void* list)
{
    alignas(16) static const std::uint32_t zeros[4] = {};
    alignas(16) static const std::uint32_t red[4] = {0x3f800000u};
    const auto draw_at = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };
    draw_at(0);  // No table binding.
    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;
    const auto store = [&](const std::uint32_t* data, std::uint32_t format) {
        const auto address = ptr(data);
        const std::uint32_t descriptor[4] = {static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>(address >> 32) | (4u << 16), 4u, (format << 12) | 4u};
        std::memcpy(reinterpret_cast<void*>(cpu), descriptor, sizeof(descriptor));
    };
    store(red, 3);
    draw_at(1);  // FORMAT 3 has no decode even in the default extended table; readable descriptor and range.
    store(zeros, 22);
    draw_at(2);  // Real zero data with a supported R32_FLOAT descriptor.
    store(nullptr, 22);
    draw_at(3);  // Supported metadata with no resolved buffer.
    store(red, 22);
    // Expand the viewport from the fixture's initial four columns to all eight.
    const D3D12_VIEWPORT viewport{0, 0, 8, 4, 0, 1};
    invoke(list, 21, 1, ptr(&viewport));
    draw_at(4);  // Nonzero data proves the accepted format is sampled.
    return 0;
}

// 3D volume memory SRV (device slot 70) with 2 mips (4x4x2 and 2x2x1) in RGBA8_UNORM format.
int draw_volume_address_srvs(ID3D12Device* desktop, void* guest, void* list)
{
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE3D, 0, 4, 4, 2, 2,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2]{};
    UINT rows[2]{};
    UINT64 row_bytes[2]{};
    UINT64 total_bytes = 0;
    desktop->GetCopyableFootprints(&desc, 0, 2, 0, footprints, rows, row_bytes, &total_bytes);
    if (!total_bytes || total_bytes == ~UINT64{0}) return 50;
    // Footprint row pitch (256 bytes) is larger than packed row bytes (4 texels * 4 bytes = 16 bytes).
    if (footprints[0].Footprint.RowPitch <= row_bytes[0]) return 52;

    static std::vector<unsigned char> backing;
    backing.assign(static_cast<std::size_t>(total_bytes), 0);

    const auto write_voxel = [&](UINT mip, UINT z, UINT y, UINT x, const unsigned char rgba[4]) {
        const auto& fp = footprints[mip].Footprint;
        const UINT64 slice_pitch = fp.RowPitch * rows[mip];
        const UINT64 offset = footprints[mip].Offset + z * slice_pitch + y * fp.RowPitch + x * 4;
        if (offset + 4 <= backing.size())
            std::memcpy(backing.data() + offset, rgba, 4);
    };

    const unsigned char c0[4] = {200, 50, 10, 255};   // mip 0, slice 0
    const unsigned char c1[4] = {50, 200, 10, 255};   // mip 0, slice 1
    const unsigned char c2[4] = {10, 50, 200, 255};   // mip 1, slice 0
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) {
            write_voxel(0, 0, y, x, c0);
            write_voxel(0, 1, y, x, c1);
        }
    for (UINT y = 0; y < 2; ++y)
        for (UINT x = 0; x < 2; ++x)
            write_voxel(1, 0, y, x, c2);

    void* placed = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(backing.data()), ptr(&desc),
                                           D3D12_RESOURCE_STATE_COMMON, 0,
                                           ptr(&__uuidof(ID3D12Resource)), ptr(&placed)))))
        return 51;

    Word cpu = 0;
    if (!bind_view_table(guest, list, cpu)) return 42;

    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, ptr(&strip));
        invoke(list, 12, 3, 1);
    };

    ViewAt v{};
    v.view.Format = desc.Format;
    v.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    v.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.view.Texture3D.MostDetailedMip = 0;
    v.view.Texture3D.MipLevels = 2;
    v.address = ptr(backing.data());

    // Col 0: view of mip 0+1, shader samples (0, 0, slice 0, mip 0)
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
    draw(0);

    // Col 1: shader samples (0, 0, slice 1, mip 0)
    draw(1);

    // Col 2: view restricted to mip 1
    v.view.Texture3D.MostDetailedMip = 1;
    v.view.Texture3D.MipLevels = 1;
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
    draw(2);

    // Refusals before column 3:
    // 1. Wrong view dimension (2D view on 3D resource)
    ViewAt bad_dim = v;
    bad_dim.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    invoke(guest, 70, ptr(&desc), ptr(&bad_dim), cpu);

    // 2. Short backing / unregistered interior address
    ViewAt bad_addr = v;
    bad_addr.address = ptr(backing.data()) + 1;
    invoke(guest, 70, ptr(&desc), ptr(&bad_addr), cpu);

    // 3. Mip range out of bounds
    ViewAt bad_mip = v;
    bad_mip.view.Texture3D.MostDetailedMip = 2;
    invoke(guest, 70, ptr(&desc), ptr(&bad_mip), cpu);

    // 4. Explicit non-zero LOD clamp
    ViewAt bad_clamp = v;
    bad_clamp.view.Texture3D.ResourceMinLODClamp = 1.0f;
    invoke(guest, 70, ptr(&desc), ptr(&bad_clamp), cpu);

    // 5. Wrong layout (console layout 0x105)
    D3D12_RESOURCE_DESC bad_layout = desc;
    bad_layout.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    invoke(guest, 70, ptr(&bad_layout), ptr(&v), cpu);

    // Col 3: all refusals preserved the previous descriptor (mip 1 view)
    draw(3);

    // Col 4-7: rebind mip 0+1
    v.view.Texture3D.MostDetailedMip = 0;
    v.view.Texture3D.MipLevels = 2;
    invoke(guest, 70, ptr(&desc), ptr(&v), cpu);
    draw(4);
    draw(5);
    draw(6);
    draw(7);

    return 0;
}

// An R8 0x105 slice has a 4 KiB mip tail followed by one 4 KiB tile for mip 0.
// The request names the first two mips in the tail of the first or second slice.
int draw_slot70_texture_alias(void* guest, void* list)
{
    constexpr std::size_t slice_bytes = 8192;
    static std::array<unsigned char, 2 * slice_bytes> backing{};
    constexpr auto layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    const D3D12_RESOURCE_DESC owner{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 2, 4, DXGI_FORMAT_R8_UNORM,
        {1, 0}, layout, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_RESOURCE_DESC alias{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 32, 32, 1, 2, DXGI_FORMAT_R8_UNORM,
        {1, 0}, layout, D3D12_RESOURCE_FLAG_NONE};
    void* texture = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr(backing.data()), ptr(&owner), D3D12_RESOURCE_STATE_COMMON,
                                           0, ptr(&__uuidof(ID3D12Resource)), ptr(&texture))))) return 40;

    Word srv = 0;
    if (!bind_view_table(guest, list, srv)) return 41;
    wchar_t mode[4]{};
    GetEnvironmentVariableW(L"X2W_TEST_SLOT70_TEXTURE_ALIAS", mode, 4);
    const Word offset = mode[0] == L'2' ? slice_bytes : mode[0] == L'3' ? 1 : 0;
    ViewAt view{};
    view.address = ptr(backing.data()) + offset;
    view.view.Format = alias.Format;
    view.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    view.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.view.Texture2DArray = {0, UINT_MAX, 0, 1, 0, 0};
    auto null_view = view.view;
    null_view.Texture2DArray.MipLevels = 2;
    invoke(guest, 18, 0, ptr(&null_view), srv);
    std::array<unsigned char, 32> previous{};
    const auto* descriptor = reinterpret_cast<const void*>(srv);
    std::memcpy(previous.data(), descriptor, previous.size());
    invoke(guest, 70, ptr(&alias), ptr(&view), srv);
    std::printf("slot70 alias descriptor changed=%d\n", std::memcmp(previous.data(), descriptor, previous.size()) != 0);
    print_slot70_switch_census();

    struct ExtendedHeap {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word address, bytes;
    };
    const ExtendedHeap heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8}, 0, 0};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)),
                                           ptr(&heap))))) return 42;
    const Word rtv_start = invoke(heap, 9);
    const Word stride = invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    // Write after publishing the alias. Distinct colours identify incorrect slice and mip remapping.
    for (UINT slice = 0; slice < 2; ++slice) {
        for (UINT mip = 0; mip < owner.MipLevels; ++mip) {
            D3D12_RENDER_TARGET_VIEW_DESC target{};
            target.Format = owner.Format;
            target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            target.Texture2DArray = {mip, slice, 1, 0};
            const Word handle = rtv_start + (slice * owner.MipLevels + mip) * stride;
            invoke(guest, 20, ptr(texture), ptr(&target), handle);
            const float color[]{(17 + 100 * slice + 31 * mip) / 255.0f, 0, 0, 1};
            invoke(list, 52, handle, ptr(color));
        }
    }
    const D3D12_RECT full{0, 0, 8, 4};
    invoke(list, 22, 1, ptr(&full));
    invoke(list, 12, 3, 1);
    return 0;
}
