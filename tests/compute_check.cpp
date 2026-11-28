#include "desktop_gpu.h"
#include "compute_root.h"
#include "compute_tables.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT count = 257;
constexpr UINT capacity = 320;
constexpr UINT bytes = capacity * sizeof(UINT);
constexpr UINT constants_offset = 2 * bytes;
constexpr UINT alignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
static_assert(constants_offset % alignment == 0);
constexpr UINT sentinel = 0xc5c5c5c5u;
constexpr std::array<UINT, 16> constants{count, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 23, 29};
constexpr UINT uavs = 13, srvs = 21;
// The image case header packs the dispatch count and a format/capability selector. The payload
// carries raw texels, one word each for RGBA8 and scalar R32, or four for RGBA32.
constexpr UINT texture_rgba8_flag = 1u << 24;
// Read-only image case selectors. The dispatch dimensions each fit in three bits.
constexpr UINT texture_mip_zero_sampler_flag = 1u << 22;
constexpr UINT texture_comparison_flag = 1u << 23;
constexpr UINT texture_atomic_uint_flag = 1u << 25;
constexpr UINT texture_atomic_sint_minmax_flag = 1u << 26;
constexpr UINT texture_atomic_uint_minmax_flag = 1u << 27;
constexpr UINT texture_atomic_exchange_flag = 1u << 28;
constexpr UINT texture_atomic_cas_flag = 1u << 29;
constexpr UINT texture_atomic_mask = texture_atomic_uint_flag | texture_atomic_sint_minmax_flag |
                                     texture_atomic_uint_minmax_flag | texture_atomic_exchange_flag |
                                     texture_atomic_cas_flag;
constexpr UINT texture_rgba32_float_flag = 1u << 30;
// Image-load cases only: bind the t0 SRV from resource mip 1 (MostDetailedMip = 1).
constexpr UINT texture_view_mip1_flag = 1u << 31;
// Texture-store cases use only scalar X group counts of 1 or 2, leaving bits 21 and 22
// free below the format and view selectors.
constexpr UINT texture_uav_load_flag = 1u << 21;
constexpr UINT texture_rgba16_float_flag = 1u << 22;
constexpr UINT texture_array_flag = 1u << 31;
constexpr UINT texture_volume_flag = texture_array_flag | texture_rgba32_float_flag;
constexpr UINT texture_1d_flag = 1u << 23; // The image case uses only scalar X group counts.
constexpr UINT texture_format_mask = texture_1d_flag | texture_rgba8_flag | texture_atomic_mask |
                                     texture_volume_flag | texture_rgba16_float_flag | texture_uav_load_flag;
constexpr UINT texture_width = 32, texture_height = 8, texture_channels = 4;
constexpr UINT texture_mip_levels = 3, texture_layers = 4;

UINT texture_subresources(const D3D12_RESOURCE_DESC& desc)
{
    // A volume mip contains all of its depth slices in one subresource.
    return desc.MipLevels * (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : desc.DepthOrArraySize);
}

UINT texture_words_per_texel(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return texture_channels;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return texture_channels / 2;
    default: return 1;
    }
}
}

static UINT input_word(UINT i) { return 0x8b02a7f1u + i * 0x413b1u; }

// source replaces the generated t0 words when a store case supplies its own, initial replaces the
// sentinel words that seed the output buffer, and root_constants replaces the b0 CBV constants.
static void fill_upload(ID3D12Resource* upload, const std::array<UINT, capacity>* source = nullptr,
                        const std::array<UINT, capacity>* initial = nullptr,
                        const std::array<UINT, constants.size()>* root_constants = nullptr)
{
    void* mapped = nullptr;
    const D3D12_RANGE no_reads{};
    check(upload->Map(0, &no_reads, &mapped), "Map compute upload");
    auto* words = static_cast<UINT*>(mapped);
    for (UINT i = 0; i < capacity; ++i) {
        words[i] = source ? (*source)[i] : input_word(i);
        words[capacity + i] = initial ? (*initial)[i] : sentinel;
    }
    auto* cb = static_cast<unsigned char*>(mapped) + constants_offset;
    std::memset(cb, 0, 4 * alignment);
    const auto& cb0 = root_constants ? *root_constants : constants;
    std::memcpy(cb, cb0.data(), sizeof(cb0));
    for (UINT i = 1; i < 4; ++i) std::memcpy(cb + i * alignment, &constants[12 + i], sizeof(UINT));
    upload->Unmap(0, nullptr);
}

// Upload the source texture and leave it in the compute read state. The caller keeps the staging
// buffer alive until the command list has executed.
static ComPtr<ID3D12Resource> upload_texture(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                                             const std::vector<UINT>& texels,
                                             DXGI_FORMAT format, ComPtr<ID3D12Resource>& staging, bool writable = false,
                                             D3D12_UAV_DIMENSION dimension = D3D12_UAV_DIMENSION_TEXTURE2D)
{
    const UINT words_per_texel = texture_words_per_texel(format);
    const bool one_dimensional = dimension == D3D12_UAV_DIMENSION_TEXTURE1D;
    const UINT height = one_dimensional ? 1 : texture_height;
    const UINT base_words = texture_width * height * words_per_texel;
    // Two base-level images in a row are the slices of a 2D array (a compute gather case), with one mip each.
    const bool two_slices = !writable && !one_dimensional && dimension == D3D12_UAV_DIMENSION_TEXTURE2D &&
        texels.size() == 2 * static_cast<std::size_t>(base_words);
    const UINT mips = texels.size() == base_words || two_slices ? 1 : texture_mip_levels;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = one_dimensional ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
        : dimension == D3D12_UAV_DIMENSION_TEXTURE3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
        : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = texture_width;
    desc.Height = height;
    desc.DepthOrArraySize = dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY ||
        dimension == D3D12_UAV_DIMENSION_TEXTURE3D ? texture_layers : two_slices ? 2 : 1;
    desc.SampleDesc.Count = 1;
    desc.MipLevels = static_cast<UINT16>(mips);
    desc.Format = format;
    if (writable) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto texture = resource(device, desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(texture_subresources(desc));
    UINT64 upload_bytes = 0;
    device->GetCopyableFootprints(&desc, 0, static_cast<UINT>(layouts.size()), 0,
                                  layouts.data(), nullptr, nullptr, &upload_bytes);
    staging = make_buffer(device, upload_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE no_reads{};
    check(staging->Map(0, &no_reads, &mapped), "Map texture upload");
    std::memset(mapped, 0, static_cast<std::size_t>(upload_bytes));
    std::size_t word = 0;
    for (const auto& layout : layouts) {
        const auto& footprint = layout.Footprint;
        const UINT row_words = footprint.Width * words_per_texel;
        for (UINT z = 0; z < footprint.Depth; ++z) {
            for (UINT y = 0; y < footprint.Height; ++y) {
                auto* row = static_cast<unsigned char*>(mapped) + layout.Offset +
                            (z * footprint.Height + y) * footprint.RowPitch;
                std::memcpy(row, texels.data() + word, row_words * sizeof(UINT));
                word += row_words;
            }
        }
    }
    staging->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION dest{}, src{};
    dest.pResource = texture.Get();
    dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = staging.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT subresource = 0; subresource < layouts.size(); ++subresource) {
        dest.SubresourceIndex = subresource;
        src.PlacedFootprint = layouts[subresource];
        list->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               writable ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return texture;
}

static ComPtr<ID3D12DescriptorHeap> bind_tables(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                                         ID3D12Resource* input, ID3D12Resource* output,
                                         ID3D12Resource* texture = nullptr, UINT output_register = uavs - 1, UINT output_mip = 0,
                                         UINT input_register = srvs - 1, UINT texture_first_mip = 0,
                                         UINT output_words = capacity)
{
    if (output_words == 0 || output_words > capacity)
        throw std::runtime_error("Invalid compute UAV word count");
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.NumDescriptors = uavs + srvs;
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)), "Create compute descriptor heap");
    const UINT stride = device->GetDescriptorHandleIncrementSize(desc.Type);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = output_words;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    if (output_register >= uavs) throw std::runtime_error("Invalid output register");
    const auto output_desc = output->GetDesc();
    for (UINT i = 0; i < uavs; ++i, cpu.ptr += stride) {
        if (i == output_register && output_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC image{};
            image.Format = output_desc.Format;
            if (output_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
                image.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                image.Texture3D = {output_mip, 0, ~0u};
            } else if (output_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
                image.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                image.Texture1D.MipSlice = output_mip;
            } else if (output_desc.DepthOrArraySize > 1) {
                image.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                image.Texture2DArray = {output_mip, 0, output_desc.DepthOrArraySize, 0};
            } else {
                image.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                image.Texture2D.MipSlice = output_mip;
            }
            device->CreateUnorderedAccessView(output, nullptr, &image, cpu);
        } else {
            device->CreateUnorderedAccessView(i == output_register ? output : nullptr, nullptr, &uav, cpu);
        }
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = capacity;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    for (UINT i = 0; i < srvs; ++i, cpu.ptr += stride) {
        // A texture SRV at t0 selects its own format; every other slot stays a raw buffer or null.
        if (texture && i == 0 && texture_first_mip) {
            // A view that starts below the base level, so view-relative mip numbers differ from the resource's.
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = texture->GetDesc().Format;
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view.Texture2D.MostDetailedMip = texture_first_mip;
            view.Texture2D.MipLevels = static_cast<UINT>(-1);
            device->CreateShaderResourceView(texture, &view, cpu);
        } else if (texture && i == 0)
            device->CreateShaderResourceView(texture, nullptr, cpu);
        else
            device->CreateShaderResourceView(!texture && i == input_register ? input : nullptr, &srv, cpu);
    }
    ID3D12DescriptorHeap* heaps[]{heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += UINT64{uavs} * stride;
    list->SetComputeRootDescriptorTable(1, gpu);
    for (UINT i = 0; i < 4; ++i)
        list->SetComputeRootConstantBufferView(2 + i, input->GetGPUVirtualAddress() + constants_offset + i * alignment);
    return heap;
}

struct StoreCase {
    // Group counts: X in bits 0-7, Y in bits 8-15 and Z in bits 16-23; zero Y or Z means one.
    // Bit 31 declares that the case file carries 320 initial output words right after the
    // expected words, so the output buffer starts at a caller-chosen nonzero state.
    UINT groups;
    std::array<UINT, capacity> expected;
    std::array<UINT, constants.size()> root_constants = constants;
    // Optional t0 words after the constants; otherwise t0 holds input_word(i).
    std::array<UINT, capacity> source{};
    bool has_source = false;
    // The flagged case's output words; otherwise every output word starts at the sentinel.
    std::array<UINT, capacity> initial{};
    bool has_initial = false;
    UINT axis(UINT index) const
    {
        const UINT n = (groups >> (8 * index)) & 0xff;
        return index && !n ? 1 : n;
    }
};

static ComputeReadbackProgram create_compute_program(ID3D12Device* device, D3D12_SHADER_BYTECODE shader,
                                                     D3D12_SHADER_BYTECODE rootBytes)
{
    ComputeReadbackProgram program;
    check(device->CreateRootSignature(0, rootBytes.pShaderBytecode, rootBytes.BytecodeLength, IID_PPV_ARGS(&program.signature)),
          "Create compute root signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = program.signature.Get();
    pso.CS = shader;
    check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&program.pipeline)), "Create compute pipeline");
    return program;
}

static void run_compute(ID3D12Device* device, bool tables, const ComputeReadbackProgram& program,
                        bool conversion_pair = false,
                        const StoreCase* store = nullptr, const std::vector<UINT>* texels = nullptr,
                        DXGI_FORMAT texture_format = DXGI_FORMAT_UNKNOWN, ID3D12Resource* rendered_texture = nullptr,
                        bool dpp_case = false, bool sized_views = false, UINT texture_first_mip = 0,
                        UINT uav_words = capacity)
{
    if ((texels || rendered_texture) && !tables) throw std::runtime_error("An image load needs the descriptor-table signature");
    if (uav_words == 0 || uav_words > capacity) throw std::runtime_error("Invalid compute UAV word count");
    auto input = make_buffer(device, constants_offset + 4 * alignment, D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    fill_upload(input.Get(), store && store->has_source ? &store->source : nullptr,
                store && store->has_initial ? &store->initial : nullptr,
                store ? &store->root_constants : nullptr);
    ComputeCommands commands(device, program.pipeline.Get());
    auto& list = commands.list;
    ComPtr<ID3D12Resource> texture_upload, texture;
    if (texels) texture = upload_texture(device, list.Get(), *texels, texture_format, texture_upload);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(output.Get(), 0, input.Get(), bytes, bytes);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(program.signature.Get());
    ComPtr<ID3D12DescriptorHeap> heap;
    if (tables) {
        // An image-load case binds its output at u0 and the source texture at t0; the table
        // signature's UAV range starts at u0, while the root-descriptor path keeps u12.
        auto* image = rendered_texture ? rendered_texture : texture.Get();
        // A sized-view case binds both raw buffers at u0 and t0, so GetDimensions reports the view's byte size.
        heap = sized_views ? bind_tables(device, list.Get(), input.Get(), output.Get(), nullptr, 0, 0, 0, 0, uav_words)
                           : bind_tables(device, list.Get(), input.Get(), output.Get(), image, image ? 0 : uavs - 1,
                                         0, srvs - 1, texture_first_mip);
    } else {
        list->SetComputeRootShaderResourceView(0, input->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        const auto& words = store ? store->root_constants : constants;
        list->SetComputeRoot32BitConstants(2, static_cast<UINT>(words.size()), words.data(), 0);
    }
    if (store) list->Dispatch(store->axis(0), store->axis(1), store->axis(2));
    else list->Dispatch(capacity / 64, 1, 1);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, bytes);
    commands.submit_and_wait(device);
    void* mapped = nullptr;
    const D3D12_RANGE read{0, bytes}, no_writes{};
    check(readback->Map(0, &read, &mapped), "Map compute readback");
    const auto* words = static_cast<const UINT*>(mapped);
    UINT mismatches = 0;
    constexpr UINT key = constants[13] + 3 * constants[14] + 5 * constants[15];
    if (conversion_pair)
        std::printf("immediate_uint=0x%08x immediate_int=0x%08x\n", words[0], words[1]);
    const UINT active = conversion_pair ? 2 : count;
    for (UINT i = 0; i < capacity; ++i) {
        UINT expected = store ? store->expected[i] : i >= active ? sentinel : conversion_pair ? (i == 0 ? 0xffffffffu : 0x7fffffffu)
            : (input_word(i) * 33u) ^ (key + i * constants[1]);
        bool mismatch = words[i] != expected;
        if (dpp_case) {
            if (i < 32) {
                const UINT logical_thread = words[32 + i];
                constexpr UINT perm[4]{1, 0, 3, 2};
                mismatch = logical_thread != i;
                const UINT source_thread = (logical_thread & ~3u) | perm[logical_thread & 3u];
                expected = input_word(source_thread) | logical_thread;
                mismatch = mismatch || words[i] != expected;
            } else if (i < 64) {
                expected = i - 32;
                mismatch = words[i] != expected;
            } else {
                expected = sentinel;
                mismatch = words[i] != expected;
            }
        }
        if (mismatch) {
            if (!mismatches)
                std::fprintf(stderr, "word=%u expected=%08x actual=%08x\n", i, expected, words[i]);
            ++mismatches;
        }
    }
    readback->Unmap(0, &no_writes);
    if (store && store->groups <= 0xff)
        std::printf("groups=%u compared_words=%u mismatches=%u\n", store->groups, capacity, mismatches);
    else if (store)
        std::printf("groups=%ux%ux%u compared_words=%u mismatches=%u\n", store->axis(0), store->axis(1),
                    store->axis(2), capacity, mismatches);
    else std::printf("layout=%s output_words=%u guard_words=%u mismatches=%u\n",
                tables ? "tables" : "root", active, capacity - active, mismatches);
    if (mismatches) throw ReadbackMismatch("Compute readback differs from CPU reference");
}

void check_compute(ID3D12Device* device, bool tables)
{
    const D3D12_SHADER_BYTECODE shader = tables ? D3D12_SHADER_BYTECODE{compute_tables, sizeof(compute_tables)} :
                                                  D3D12_SHADER_BYTECODE{compute_root, sizeof(compute_root)};
    run_compute(device, tables, create_compute_program(device, shader, shader));
}

static std::vector<char> read_compute_shader(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("Could not open compute shader: ") + path);
    const auto size = file.tellg();
    if (size <= 0) throw std::runtime_error(std::string("Empty compute shader: ") + path);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error(std::string("Could not read compute shader: ") + path);
    return bytes;
}

static StoreCase read_store_reference(const char* reference, bool dpp_case)
{
    StoreCase store{};
    std::ifstream file(reference, std::ios::binary);
    file.read(reinterpret_cast<char*>(&store.groups), sizeof(store.groups));
    file.read(reinterpret_cast<char*>(store.expected.data()), sizeof(store.expected));
    // Bit 31 declares the trailing initial output segment; no other bit above the
    // group counts is known, so an unflagged case keeps its historic bounds check.
    constexpr UINT initial_data_flag = 1u << 31;
    store.has_initial = (store.groups & initial_data_flag) != 0;
    store.groups &= ~initial_data_flag;
    if (!file || store.groups >> 24 || !store.axis(0) || store.axis(0) > 5 || store.axis(1) > 5 ||
        store.axis(2) > 5)
        throw std::runtime_error("Invalid compute store reference");
    if (dpp_case && store.groups != 1)
        throw std::runtime_error("Invalid compute DPP reference dispatch");
    if (store.has_initial) {
        file.read(reinterpret_cast<char*>(store.initial.data()), sizeof(store.initial));
        if (!file)
            throw std::runtime_error("Invalid compute store initial words");
    }
    if (file.peek() != std::char_traits<char>::eof()) {
        file.read(reinterpret_cast<char*>(store.root_constants.data()), sizeof(store.root_constants));
        if (!file)
            throw std::runtime_error("Invalid compute store constants");
        if (file.peek() != std::char_traits<char>::eof()) {
            file.read(reinterpret_cast<char*>(store.source.data()), sizeof(store.source));
            if (!file || file.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Invalid compute store source words");
            store.has_source = true;
        }
    }
    return store;
}

ComputeReadbackProgram load_compute_store_program(ID3D12Device* device, const char* shader)
{
    const auto code = read_compute_shader(shader);
    return create_compute_program(device, {code.data(), code.size()}, {compute_root, sizeof(compute_root)});
}

void check_compute_store_reference(ID3D12Device* device, const ComputeReadbackProgram& program, const char* reference)
{
    const StoreCase store = read_store_reference(reference, false);
    run_compute(device, false, program, false, &store);
}

void check_compute_file(ID3D12Device* device, const char* path, bool conversion_pair, const char* reference,
                        bool dpp_case, bool sized_views, UINT uav_words)
{
    if (sized_views && uav_words > capacity)
        throw std::runtime_error("Invalid compute UAV word count");
    const auto bytes = read_compute_shader(path);
    const D3D12_SHADER_BYTECODE shader{bytes.data(), bytes.size()};
    if (reference) {
        const StoreCase store = read_store_reference(reference, dpp_case);
        if (sized_views) {
            const UINT effective_uav_words = uav_words ? uav_words : capacity;
            if (effective_uav_words > capacity)
                throw std::runtime_error("Invalid compute UAV word count");
            const auto program = create_compute_program(device, shader, {compute_tables, sizeof(compute_tables)});
            run_compute(device, true, program, false, &store, nullptr,
                        DXGI_FORMAT_UNKNOWN, nullptr, dpp_case, true, 0, effective_uav_words);
        } else {
            const auto program = create_compute_program(device, shader, {compute_root, sizeof(compute_root)});
            run_compute(device, false, program, false, &store, nullptr,
                        DXGI_FORMAT_UNKNOWN, nullptr, dpp_case);
        }
    } else {
        run_compute(device, false, create_compute_program(device, shader, shader), conversion_pair);
    }
}

static ComPtr<ID3DBlob> compute_texture_sampler_root(const char* manifest)
{
    std::ifstream source(manifest);
    if (!source) throw std::runtime_error("Could not open compute texture sampler manifest");
    x2w_startup::TranslatedSamplers samplers;
    for (std::string line; std::getline(source, line);)
        if (!samplers.read_manifest_line(line.c_str()))
            throw std::runtime_error("Invalid compute texture sampler manifest");
    if (samplers.heap || samplers.statics.size() != 1)
        throw std::runtime_error("Expected one static compute texture sampler");
    const auto statics = x2w_startup::decode_statics(samplers);

    ComPtr<ID3D12VersionedRootSignatureDeserializer> embedded;
    check(D3D12CreateVersionedRootSignatureDeserializer(compute_tables, sizeof(compute_tables),
          IID_PPV_ARGS(&embedded)), "Read embedded compute table signature");
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* original = nullptr;
    check(embedded->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &original),
          "Read compute table layout");
    auto root = *original;
    root.Desc_1_1.NumStaticSamplers = static_cast<UINT>(statics.size());
    root.Desc_1_1.pStaticSamplers = statics.data();
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeVersionedRootSignature(&root, &blob, nullptr), "Serialize compute texture sampler root");
    return blob;
}

// The image load uses a descriptor-table signature. Its case file appends the source texels.
void check_compute_texture_file(ID3D12Device* device, const char* path, const char* reference,
                                const char* sampler_manifest)
{
    const auto code = read_compute_shader(path);
    const D3D12_SHADER_BYTECODE shader{code.data(), code.size()};
    StoreCase store{};
    std::ifstream source(reference, std::ios::binary);
    if (!source) throw std::runtime_error(std::string("Could not open compute reference: ") + reference);
    source.read(reinterpret_cast<char*>(&store.groups), sizeof(store.groups));
    source.read(reinterpret_cast<char*>(store.expected.data()), sizeof(store.expected));
    const bool rgba8 = (store.groups & texture_rgba8_flag) != 0;
    const bool rgba32_float = (store.groups & texture_rgba32_float_flag) != 0;
    const bool view_mip1 = (store.groups & texture_view_mip1_flag) != 0;
    const bool mip_zero = (store.groups & texture_mip_zero_sampler_flag) != 0;
    const bool comparison = (store.groups & texture_comparison_flag) != 0;
    store.groups &= ~(texture_rgba8_flag | texture_rgba32_float_flag | texture_view_mip1_flag |
                      texture_mip_zero_sampler_flag | texture_comparison_flag);
    const auto format = comparison ? DXGI_FORMAT_R32_FLOAT : rgba8 ? DXGI_FORMAT_R8G8B8A8_UINT
        : rgba32_float ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32G32B32A32_UINT;
    const UINT base_words = texture_width * texture_height * texture_words_per_texel(format);
    std::vector<UINT> texels(base_words);
    source.read(reinterpret_cast<char*>(texels.data()), static_cast<std::streamsize>(texels.size() * sizeof(UINT)));
    // Older cases stop after the base image. A mip case appends levels one and two; an array case appends
    // the second slice (the file's remaining size tells them apart).
    const auto here = source.tellg();
    source.seekg(0, std::ios::end);
    const auto remaining = static_cast<std::size_t>(source.tellg() - here);
    source.seekg(here);
    if (source && remaining == base_words * sizeof(UINT)) {
        texels.resize(2 * static_cast<std::size_t>(base_words));
        source.read(reinterpret_cast<char*>(texels.data() + base_words), base_words * sizeof(UINT));
    } else if (source && source.peek() != std::char_traits<char>::eof()) {
        const UINT extra_words = base_words / 4 + base_words / 16;
        texels.resize(base_words + extra_words);
        source.read(reinterpret_cast<char*>(texels.data() + base_words), extra_words * sizeof(UINT));
    }
    if (!source || source.peek() != std::char_traits<char>::eof() || store.groups >> 24 || !store.axis(0) ||
        store.axis(0) > 5 || store.axis(1) > 5 || store.axis(2) > 5 || (rgba8 && rgba32_float) ||
        (comparison && (rgba8 || rgba32_float || mip_zero)) ||
        (view_mip1 && texels.size() <= base_words * 2 && texels.size() % base_words == 0))
        throw std::runtime_error("Invalid compute image load reference");
    D3D12_SHADER_BYTECODE root{compute_tables, sizeof(compute_tables)};
    ComPtr<ID3DBlob> case_root;
    if (sampler_manifest) {
        case_root = compute_texture_sampler_root(sampler_manifest);
        root = {case_root->GetBufferPointer(), case_root->GetBufferSize()};
    } else if (comparison || mip_zero) {
        ComPtr<ID3D12VersionedRootSignatureDeserializer> embedded;
        check(D3D12CreateVersionedRootSignatureDeserializer(root.pShaderBytecode, root.BytecodeLength,
              IID_PPV_ARGS(&embedded)), "Read embedded compute table signature");
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* original = nullptr;
        check(embedded->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &original),
              "Get v1.1 compute table signature");
        auto desc = *original;
        if (desc.Desc_1_1.NumStaticSamplers != 1)
            throw std::runtime_error("Expected one static sampler in the compute table signature");
        auto sampler = desc.Desc_1_1.pStaticSamplers[0];
        if (comparison) {
            sampler.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
            sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS;
        } else {
            sampler.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
            sampler.MaxLOD = 0;
        }
        desc.Desc_1_1.pStaticSamplers = &sampler;
        check(D3D12SerializeVersionedRootSignature(&desc, &case_root, nullptr), "Serialize compute case signature");
        root = {case_root->GetBufferPointer(), case_root->GetBufferSize()};
    }
    const auto program = create_compute_program(device, shader, root);
    run_compute(device, true, program, false, &store, &texels, format,
                nullptr, false, false, view_mip1 ? 1 : 0);
}

static ComPtr<ID3D12Resource> render_msaa_texture(ID3D12Device* device, D3D12_SHADER_BYTECODE vs,
                                               D3D12_SHADER_BYTECODE ps)
{
    constexpr UINT side = 8;
    constexpr DXGI_SAMPLE_DESC samples{4, 0};
    D3D12_FEATURE_DATA_FORMAT_SUPPORT format{DXGI_FORMAT_R32_UINT};
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{format.Format, samples.Count};
    constexpr UINT required = D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_RENDER_TARGET |
        D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD;
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format, sizeof(format))) ||
        (format.Support1 & required) != required ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality, sizeof(quality))) ||
        !quality.NumQualityLevels)
        return {};

    D3D12_RESOURCE_DESC image{};
    image.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image.Width = image.Height = side;
    image.DepthOrArraySize = image.MipLevels = 1;
    image.Format = format.Format;
    image.SampleDesc = samples;
    image.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    auto texture = resource(device, image, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create MSAA RTV heap");
    const auto rtv = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(texture.Get(), nullptr, rtv);

    D3D12_ROOT_SIGNATURE_DESC layout{};
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize MSAA draw root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create MSAA draw root");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.Get();
    desc.VS = vs;
    desc.PS = ps;
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.RasterizerState.MultisampleEnable = TRUE;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = format.Format;
    desc.SampleDesc = samples;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create MSAA input draw pipeline");
    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto& list = commands.list;
    list->SetGraphicsRootSignature(root.Get());
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list.Get(), texture.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    commands.submit_and_wait(device);
    return texture;
}

bool check_compute_texture_msaa_file(ID3D12Device* device, const char* path, const char* reference,
                                     const char* vs_path, const char* ps_path)
{
    StoreCase store{};
    std::ifstream source(reference, std::ios::binary);
    source.read(reinterpret_cast<char*>(&store.groups), sizeof(store.groups));
    source.read(reinterpret_cast<char*>(store.expected.data()), sizeof(store.expected));
    if (!source || source.peek() != std::char_traits<char>::eof() || store.groups != 1)
        throw std::runtime_error("Invalid compute MSAA load reference");
    const auto code = read_compute_shader(path);
    const auto vs = read_compute_shader(vs_path), ps = read_compute_shader(ps_path);
    auto texture = render_msaa_texture(device, {vs.data(), vs.size()}, {ps.data(), ps.size()});
    if (!texture) return false;
    const auto program = create_compute_program(device, {code.data(), code.size()}, {compute_tables, sizeof(compute_tables)});
    run_compute(device, true, program, false, &store, nullptr, DXGI_FORMAT_UNKNOWN, texture.Get());
    return true;
}

bool check_compute_texture_store_file(ID3D12Device* device, const char* path, const char* reference)
{
    UINT groups = 0, mip = 0;
    std::ifstream source(reference, std::ios::binary);
    source.read(reinterpret_cast<char*>(&groups), sizeof(groups));
    source.read(reinterpret_cast<char*>(&mip), sizeof(mip));
    const bool uav_load = (groups & texture_uav_load_flag) != 0;
    const UINT selector = groups & (texture_format_mask & ~texture_uav_load_flag);
    groups &= ~texture_format_mask;
    DXGI_FORMAT format = uav_load ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32G32B32A32_UINT;
    D3D12_UAV_DIMENSION dimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    UINT required = D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
    switch (selector) {
    case 0: break;
    case texture_rgba32_float_flag: format = DXGI_FORMAT_R32G32B32A32_FLOAT; break;
    case texture_1d_flag:
        format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        break;
    case texture_array_flag:
        format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        break;
    case texture_volume_flag:
        format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        break;
    case texture_rgba16_float_flag: format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
    case texture_rgba8_flag: format = DXGI_FORMAT_R8G8B8A8_UINT; break;
    case texture_atomic_uint_flag:
        format = DXGI_FORMAT_R32_UINT;
        required = D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD |
                   D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS;
        break;
    case texture_atomic_sint_minmax_flag:
        format = DXGI_FORMAT_R32_SINT;
        required = D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX;
        break;
    case texture_atomic_uint_minmax_flag:
        format = DXGI_FORMAT_R32_UINT;
        required = D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX;
        break;
    case texture_atomic_exchange_flag:
        format = DXGI_FORMAT_R32_UINT;
        required = D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_EXCHANGE |
                   D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE;
        break;
    case texture_atomic_cas_flag:
        format = DXGI_FORMAT_R32_UINT;
        required = D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE;
        break;
    default: throw std::runtime_error("Invalid compute image store format selector");
    }
    if (uav_load) {
        if (format != DXGI_FORMAT_R32G32B32A32_FLOAT)
            throw std::runtime_error("Typed load fixture requires RGBA32 float");
        required |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD;
    }
    const bool one_dimensional = dimension == D3D12_UAV_DIMENSION_TEXTURE1D;
    const bool separate_srv = !uav_load && (selector == texture_rgba32_float_flag || one_dimensional);
    const bool layered = dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY || dimension == D3D12_UAV_DIMENSION_TEXTURE3D;
    const bool half_image = selector == texture_rgba16_float_flag;
    const UINT words_per_texel = texture_words_per_texel(format);
    UINT all_words = 0;
    for (UINT level = 0; level < texture_mip_levels; ++level) {
        UINT depth = layered ? texture_layers : 1;
        if (dimension == D3D12_UAV_DIMENSION_TEXTURE3D) depth >>= level;
        const UINT height = one_dimensional ? 1 : texture_height >> level;
        all_words += (texture_width >> level) * height * depth * words_per_texel;
    }
    std::vector<UINT> initial(all_words), expected(all_words);
    source.read(reinterpret_cast<char*>(initial.data()), all_words * sizeof(UINT));
    source.read(reinterpret_cast<char*>(expected.data()), all_words * sizeof(UINT));
    std::vector<UINT> input_texels(separate_srv ? all_words : 0);
    if (separate_srv) source.read(reinterpret_cast<char*>(input_texels.data()), all_words * sizeof(UINT));
    if (!source || source.peek() != std::char_traits<char>::eof() || !groups || groups > 2 || mip >= texture_mip_levels
        || (((selector & texture_atomic_mask) || uav_load) && mip != 0))
        throw std::runtime_error("Invalid compute image store reference");

    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
    check(check_feature(device, D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)),
          "Check texture UAV format");
    if (uav_load) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)),
              "Check typed UAV loads for additional formats");
        if (!options.TypedUAVLoadAdditionalFormats) {
            std::puts("R32G32B32A32_FLOAT requires TypedUAVLoadAdditionalFormats");
            return false;
        }
    }
    const UINT required_shape = dimension == D3D12_UAV_DIMENSION_TEXTURE3D ? D3D12_FORMAT_SUPPORT1_TEXTURE3D
        : one_dimensional ? D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_MIP
        : layered ? D3D12_FORMAT_SUPPORT1_TEXTURE2D : 0;
    if ((support.Support2 & required) != required || (support.Support1 & required_shape) != required_shape) {
        std::printf("format=%u dimension=%u required_support1=0x%x required_support2=0x%x support1=0x%x support2=0x%x\n",
                    static_cast<UINT>(format), static_cast<UINT>(dimension), required_shape, required,
                    static_cast<UINT>(support.Support1),
                    static_cast<UINT>(support.Support2));
        return false;
    }

    const auto code = read_compute_shader(path);
    ComPtr<ID3D12RootSignature> signature;
    check(device->CreateRootSignature(0, compute_tables, sizeof(compute_tables), IID_PPV_ARGS(&signature)),
          "Create image store root signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = signature.Get();
    pso.CS = {code.data(), code.size()};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create image store pipeline");
    ComputeCommands commands(device, pipeline.Get());
    auto* list = commands.list.Get();
    ComPtr<ID3D12Resource> staging, input_staging, input_image;
    auto texture = upload_texture(device, list, initial, format, staging, true, dimension);
    if (separate_srv) input_image = upload_texture(device, list, input_texels, format, input_staging, false, dimension);
    auto constants_buffer = make_buffer(device, constants_offset + 4 * alignment, D3D12_HEAP_TYPE_UPLOAD,
                                        D3D12_RESOURCE_STATE_GENERIC_READ);
    fill_upload(constants_buffer.Get());
    list->SetComputeRootSignature(signature.Get());
    auto heap = bind_tables(device, list, constants_buffer.Get(), texture.Get(), input_image.Get(), 0, mip);
    list->Dispatch(groups, 1, 1);

    transition(list, texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto desc = texture->GetDesc();
    const UINT subresources = texture_subresources(desc);
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(subresources);
    UINT64 readback_bytes = 0;
    device->GetCopyableFootprints(&desc, 0, subresources, 0, layouts.data(), nullptr, nullptr, &readback_bytes);
    auto readback = make_buffer(device, readback_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION src{}, dest{};
    src.pResource = texture.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dest.pResource = readback.Get();
    dest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    for (UINT subresource = 0; subresource < subresources; ++subresource) {
        src.SubresourceIndex = subresource;
        dest.PlacedFootprint = layouts[subresource];
        list->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
    }
    commands.submit_and_wait(device);

    void* mapped = nullptr;
    const D3D12_RANGE read{0, static_cast<SIZE_T>(readback_bytes)}, no_writes{};
    check(readback->Map(0, &read, &mapped), "Map image store readback");
    UINT word = 0, mismatches = 0;
    for (UINT subresource = 0; subresource < subresources; ++subresource) {
        const auto& layout = layouts[subresource];
        const auto& footprint = layout.Footprint;
        const UINT level = subresource % desc.MipLevels;
        const UINT row_words = footprint.Width * words_per_texel;
        for (UINT z = 0; z < footprint.Depth; ++z) {
            for (UINT y = 0; y < footprint.Height; ++y) {
                const auto* row = reinterpret_cast<const UINT*>(static_cast<const unsigned char*>(mapped) +
                    layout.Offset + (z * footprint.Height + y) * footprint.RowPitch);
                for (UINT x = 0; x < row_words; ++x, ++word) {
                    if (row[x] == expected[word]) continue;
                    if (half_image) {
                        for (UINT half = 0; half < 2; ++half) {
                            const UINT shift = 16u * half;
                            const UINT want = (expected[word] >> shift) & 0xffffu;
                            const UINT got = (row[x] >> shift) & 0xffffu;
                            if (want == got) continue;
                            if (!mismatches)
                                std::fprintf(stderr, "mip=%u half=%u expected=%04x actual=%04x\n",
                                             level, 2 * word + half, want, got);
                            ++mismatches;
                        }
                    } else {
                        if (!mismatches)
                            std::fprintf(stderr, "mip=%u word=%u expected=%08x actual=%08x\n",
                                         level, word, expected[word], row[x]);
                        ++mismatches;
                    }
                }
            }
        }
    }
    readback->Unmap(0, &no_writes);
    std::printf("groups=%u view_mip=%u compared_%s=%u mismatches=%u\n", groups, mip,
                half_image ? "halves" : "words", half_image ? 2 * word : word, mismatches);
    if (mismatches) throw std::runtime_error("Image store readback differs from CPU reference");
    return true;
}

void check_bvh_root_roundtrip(ID3D12Device* device)
{
    const D3D12_SHADER_BYTECODE shader{compute_root, sizeof(compute_root)};
    ComPtr<ID3D12VersionedRootSignatureDeserializer> embedded;
    check(D3D12CreateVersionedRootSignatureDeserializer(shader.pShaderBytecode, shader.BytecodeLength,
          IID_PPV_ARGS(&embedded)), "Read embedded control signature");
    auto desc = *embedded->GetUnconvertedRootSignatureDesc();
    if (desc.Version != D3D_ROOT_SIGNATURE_VERSION_1_1) throw std::runtime_error("Expected a v1.1 control");
    auto& flags = desc.Desc_1_1.Flags;
    constexpr UINT tags = 0x03000000;
    if (flags & tags) throw std::runtime_error("Control already contains the BVH flags");
    ComPtr<ID3DBlob> control, rejected, normalized, roundtrip;
    check(D3D12SerializeVersionedRootSignature(&desc, &control, nullptr), "Serialize control signature");
    flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags | tags);
    const HRESULT tagged = D3D12SerializeVersionedRootSignature(&desc, &rejected, nullptr);
    std::printf("tagged_flags=0x%08x serialize_hr=0x%08lx\n", static_cast<UINT>(flags), tagged);
    if (tagged != E_INVALIDARG || rejected.Get()) throw std::runtime_error("Unexpected tagged signature result");

    flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~tags);
    check(D3D12SerializeVersionedRootSignature(&desc, &normalized, nullptr), "Serialize normalized signature");
    ComPtr<ID3D12VersionedRootSignatureDeserializer> decoded;
    check(D3D12CreateVersionedRootSignatureDeserializer(normalized->GetBufferPointer(), normalized->GetBufferSize(),
          IID_PPV_ARGS(&decoded)), "Deserialize normalized signature");
    check(D3D12SerializeVersionedRootSignature(decoded->GetUnconvertedRootSignatureDesc(), &roundtrip, nullptr),
          "Reserialize normalized signature");
    if (roundtrip->GetBufferSize() != control->GetBufferSize() ||
        std::memcmp(roundtrip->GetBufferPointer(), control->GetBufferPointer(), control->GetBufferSize()) != 0)
        throw std::runtime_error("Normalized signature differs from independent control serialization");
    const auto program = create_compute_program(device, shader, {roundtrip->GetBufferPointer(), roundtrip->GetBufferSize()});
    run_compute(device, false, program);
    std::puts("normalized_signature=matched_control root_creation=passed pipeline_creation=passed dispatch=passed");
}

int check_compute_waterfall_oracle(ID3D12Device* device, const char* shader_path)
{
    const auto code = read_compute_shader(shader_path);
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, code.data(), code.size(), IID_PPV_ARGS(&root)), "Create waterfall oracle root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.Get();
    pso_desc.CS = {code.data(), code.size()};
    ComPtr<ID3D12PipelineState> pso;
    check(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso)), "Create waterfall oracle pipeline");

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 4;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create oracle descriptor heap");
    const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    const auto buf_desc = [](UINT64 size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = size;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = flags;
        return d;
    };
    const auto write_upload = [](ID3D12Resource* res, const void* data, UINT sz) {
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        check(res->Map(0, &none, &mapped), "Map upload buffer");
        std::memcpy(mapped, data, sz);
        res->Unmap(0, nullptr);
    };

    std::array<ComPtr<ID3D12Resource>, 4> sources;
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT rec = 0; rec < 4; ++rec) {
        const std::array<UINT, 4> data{
            1000u * (rec + 1u), 1000u * (rec + 1u) + 17u, 1000u * (rec + 1u) + 34u, 1000u * (rec + 1u) + 51u};
        sources[rec] = resource(device, buf_desc(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        write_upload(sources[rec].Get(), data.data(), 16);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R32_TYPELESS;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = 4;
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device->CreateShaderResourceView(sources[rec].Get(), &srv, handle);
        handle.ptr += inc;
    }

    auto keys_res = resource(device, buf_desc(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto init_res = resource(device, buf_desc(16), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto out_res = resource(device, buf_desc(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    auto rb_res = resource(device, buf_desc(16), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    constexpr std::array<UINT, 4> sentinels{0xc001cafeu, 0xc001cafeu, 0xc001cafeu, 0xc001cafeu};
    write_upload(init_res.Get(), sentinels.data(), 16);

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    check(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue)), "Create oracle queue");
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)), "Create oracle alloc");
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list)),
          "Create oracle list");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create oracle fence");
    check(list->Close(), "Close initial list");

    const std::array<std::array<UINT, 4>, 5> inputs{{{0, 1, 2, 3}, {3, 2, 1, 0}, {2, 1, 2, 3}, {4, 9, 14, 19}, {7, 7, 7, 7}}};
    UINT matches = 0, mismatches = 0;
    UINT64 fence_value = 0;
    const auto barrier = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {out_res.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &b);
    };

    for (const auto& row : inputs) {
        write_upload(keys_res.Get(), row.data(), 16);
        for (UINT active = 0; active <= 4; ++active) {
            check(alloc->Reset(), "Reset oracle alloc");
            check(list->Reset(alloc.Get(), pso.Get()), "Reset oracle list");
            barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(out_res.Get(), 0, init_res.Get(), 0, 16);
            barrier(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            list->SetDescriptorHeaps(1, heaps);
            list->SetComputeRootSignature(root.Get());
            list->SetComputeRootShaderResourceView(0, keys_res->GetGPUVirtualAddress());
            list->SetComputeRootDescriptorTable(1, heap->GetGPUDescriptorHandleForHeapStart());
            list->SetComputeRootUnorderedAccessView(2, out_res->GetGPUVirtualAddress());
            list->SetComputeRoot32BitConstant(3, active, 0);
            list->Dispatch(1, 1, 1);
            barrier(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(rb_res.Get(), 0, out_res.Get(), 0, 16);
            barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            check(list->Close(), "Close oracle list");
            ID3D12CommandList* cmd = list.Get();
            queue->ExecuteCommandLists(1, &cmd);
            check(queue->Signal(fence.Get(), ++fence_value), "Signal oracle fence");
            while (fence->GetCompletedValue() < fence_value) Sleep(1);

            void* mapped = nullptr;
            const D3D12_RANGE range{0, 16};
            const D3D12_RANGE no_writes{};
            check(rb_res->Map(0, &range, &mapped), "Map oracle readback");
            std::array<UINT, 4> got{};
            std::memcpy(got.data(), mapped, 16);
            rb_res->Unmap(0, &no_writes);

            std::array<UINT, 4> expected{};
            for (UINT lane = 0; lane < 4; ++lane)
                expected[lane] = lane < active ? 1000u * ((row[lane] & 3u) + 1u) + 17u * lane : 13u;
            if (got == expected) ++matches;
            else ++mismatches;
        }
    }
    std::printf("dispatches=25 matches=%u mismatches=%u\n", matches, mismatches);
    return 0;
}
