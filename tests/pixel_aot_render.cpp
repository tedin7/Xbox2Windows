#include "bvh_fixture.h"
#include "desktop_gpu.h"
#include "pixel_aot_vs.h"
#include "pixel_aot_vs_linear.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <optional>
#include <string_view>

using namespace bvh_fixture;

namespace {
// Includes UNORM values on both sides of a binary16 rounding midpoint.
constexpr std::array<UINT, 8> ramp{0, 16, 37, 64, 101, 153, 219, 254};

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = format;
    desc.Flags = flags;
    return desc;
}

float expected_channel(UINT byte)
{
    if (!byte) return 0;
    // Fixture inputs are zero or normal positive half values. Truncate the exact
    // UNORM fraction to the 11 significant bits of a binary16 MRT export.
    const double value = byte / 255.0;
    int exponent;
    std::frexp(value, &exponent);
    const double scale = std::ldexp(1.0, 11 - exponent);
    return static_cast<float>(std::floor(value * scale) / scale);
}
using Color = std::array<float, 4>;

// Aliased buffers bound after the title range table for flat and global loads; tools/pixel_aot.py
// TITLE_ALIAS_SLOTS declares the same count.
constexpr UINT titleAliasSlots = 8;
constexpr size_t titleRangeWords = 6;  // TitleRange in src/title_va_lookup.hlsli

struct PixelInput {
    UINT width, height, scale;
    DXGI_FORMAT format;
    std::vector<BYTE> rows;
    std::vector<Color> expected;
    std::vector<BYTE> secondRows;
    std::vector<Word> constants = std::vector<Word>(64);
    std::vector<Word> extraConstants;   // 128 words for b1 then b2, empty unless trailer flag bit 3 is set
    UINT8 writeMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    bool maskedCase = false;
    std::vector<BYTE> coverage;
    bool integerRT = false;      // bit 0 of the optional case-trailer flags byte
    std::vector<BYTE> nanMask;   // 64 per-texel NaN-class mask bytes, empty without a trailer
    std::optional<D3D12_VIEWPORT> viewport;
    std::vector<Word> titleRanges;                // TitleRange records, empty without title memory
    std::vector<std::vector<Word>> titleAliases;  // one buffer per alias slot
    bool linearMips = false;     // sample through the MIN_MAG_POINT_MIP_LINEAR root signature
    float tolerance = 0;         // per-channel absolute error; 0 compares bit for bit
};

// Flag bit 2 of the case trailer: the range count and the slot count (titleAliasSlots), the ranges,
// then for each slot its length in words followed by the words.
void read_title_memory(std::istream& file, PixelInput& input)
{
    std::array<Word, 2> counts{};
    file.read(reinterpret_cast<char*>(counts.data()), sizeof(counts));
    const auto [rangeCount, slotCount] = counts;
    if (!file || !rangeCount || rangeCount > 4096 || slotCount != titleAliasSlots)
        throw std::runtime_error("Invalid title memory counts");
    input.titleRanges.resize(size_t(rangeCount) * titleRangeWords);
    file.read(reinterpret_cast<char*>(input.titleRanges.data()),
              static_cast<std::streamsize>(input.titleRanges.size() * sizeof(Word)));
    for (size_t range = 0; file && range < rangeCount; ++range) {
        const Word* fields = input.titleRanges.data() + range * titleRangeWords;
        if (fields[4] >= slotCount || fields[5])
            throw std::runtime_error("Invalid title range slot");
    }
    for (UINT slot = 0; file && slot < slotCount; ++slot) {
        Word words = 0;
        file.read(reinterpret_cast<char*>(&words), sizeof(words));
        if (!file || !words || words > (1u << 22))
            throw std::runtime_error("Invalid title alias length");
        auto& alias = input.titleAliases.emplace_back(words);
        file.read(reinterpret_cast<char*>(alias.data()), static_cast<std::streamsize>(alias.size() * sizeof(Word)));
    }
    if (!file) throw std::runtime_error("Truncated title memory");
}

// pixel_aot_vs's root signature followed by a pixel-visible table of the title range table (t0, space1)
// and the aliased buffers (t1, space1 onwards).
ComPtr<ID3D12RootSignature> title_root_signature(ID3D12Device* device)
{
    const D3D12_DESCRIPTOR_RANGE ranges[2]{
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1 + titleAliasSlots, 0, 1, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
    };
    D3D12_ROOT_PARAMETER parameters[5]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable.NumDescriptorRanges = 1;
    parameters[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    for (UINT cbv = 0; cbv < 3; ++cbv) {
        auto& param = parameters[1 + cbv];
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        param.Descriptor.ShaderRegister = cbv;
        param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    parameters[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[4].DescriptorTable.NumDescriptorRanges = 1;
    parameters[4].DescriptorTable.pDescriptorRanges = &ranges[1];
    parameters[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    // The static sampler of tests/pixel_aot_vs.hlsl, with the HLSL defaults for the fields it leaves out.
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 16;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC signature{};
    signature.NumParameters = 5;
    signature.pParameters = parameters;
    signature.NumStaticSamplers = 1;
    signature.pStaticSamplers = &sampler;
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr),
          "Serialize title memory root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&root)), "Title memory root");
    return root;
}

struct MipDraw {
    UINT sampledLevel;
    UINT scale;
    UINT requestedMip = 0;
    bool isTailDraw = false;
    bool partialQuad = false;
    bool allowUpperRawLodStep = false;
};

struct Extent {
    UINT width, height;
    Extent mip(UINT level) const {
        return {std::max<UINT>(1, width >> level), std::max<UINT>(1, height >> level)};
    }
};

void draw_verified(ID3D12Device* device, const char* path, const std::vector<PixelInput>& levels,
                   Extent base, UINT firstResident, const std::vector<MipDraw>& draws, const char* capture,
                   ID3D12Resource* gpuConstants = nullptr, const char* rawReadback = nullptr,
                   ID3D12RootSignature* rootOverride = nullptr, const char* vsPath = nullptr)
{
    const auto& input = levels.front();
    ComPtr<ID3D12Resource> uploadedConstants;
    if (!gpuConstants) {
        uploadedConstants = upload(device, input.constants);
        gpuConstants = uploadedConstants.Get();
    }
    // b1 and b2 are bound only for a case that supplies them. No other fixture shader declares them.
    std::array<ComPtr<ID3D12Resource>, 2> extraConstants;
    if (!input.extraConstants.empty()) {
        if (input.extraConstants.size() != 128) throw std::runtime_error("Expected two 256-byte extra constant buffers");
        for (size_t i = 0; i < extraConstants.size(); ++i)
            extraConstants[i] = upload(device, std::vector<Word>(input.extraConstants.begin() + 64 * i,
                                                                 input.extraConstants.begin() + 64 * (i + 1)));
    }
    std::ifstream file(path, std::ios::binary);
    std::vector<char> ps{std::istreambuf_iterator<char>(file), {}};
    if (ps.empty() || file.bad()) throw std::runtime_error("Cannot read compiled pixel shader");
    const bool titleMemory = !input.titleRanges.empty();
    ComPtr<ID3D12RootSignature> root;
    std::vector<char> vsFile;
    D3D12_SHADER_BYTECODE vs = input.linearMips
        ? D3D12_SHADER_BYTECODE{pixel_aot_vs_linear, sizeof(pixel_aot_vs_linear)}
        : D3D12_SHADER_BYTECODE{pixel_aot_vs, sizeof(pixel_aot_vs)};
    if (vsPath) {
        // A comparison gather compiles its pixel stage with FXC ps_5_1 too, which needs a matching
        // vs_5_1 blob instead of the embedded DXC vertex shader.
        std::ifstream file(vsPath, std::ios::binary);
        vsFile.assign(std::istreambuf_iterator<char>(file), {});
        if (vsFile.empty() || file.bad()) throw std::runtime_error("Cannot read compiled vertex shader");
        vs = {vsFile.data(), vsFile.size()};
    }
    // The title root signature has its own point sampler.
    if (titleMemory && input.linearMips)
        throw std::runtime_error("The mip-linear sampler is not available with title memory");
    if (rootOverride)
        root = rootOverride;  // the caller owns the comparison-sampler root and keeps it alive
    else if (titleMemory)
        root = title_root_signature(device);
    else
        check(device->CreateRootSignature(0, vs.pShaderBytecode, vs.BytecodeLength, IID_PPV_ARGS(&root)), "Pixel root");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.Get();
    desc.VS = vs;
    desc.PS = {ps.data(), ps.size()};
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    auto& blend = desc.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = input.writeMask;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = desc.SampleDesc.Count = 1;
    desc.RTVFormats[0] = input.integerRT ? DXGI_FORMAT_R32G32B32A32_UINT
                                         : DXGI_FORMAT_R32G32B32A32_FLOAT;
    ComPtr<ID3D12PipelineState> pso;
    check(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)), "Translated pixel pipeline");

    auto srcDesc = texture_desc(base.width, base.height, input.format, D3D12_RESOURCE_FLAG_NONE);
    if (input.format == DXGI_FORMAT_BC7_UNORM && (srcDesc.Width % 4 || srcDesc.Height % 4))
        throw std::runtime_error("Expected block-aligned BC7 base dimensions");
    const UINT residentCount = static_cast<UINT>(levels.size());
    srcDesc.MipLevels = static_cast<UINT16>(firstResident + residentCount);
    auto texture = resource(device, srcDesc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_CLEAR_VALUE clear{desc.RTVFormats[0], {}};
    if (input.integerRT) {
        // Integer masked cases use distinct nonzero words so a wrong PSO mask is visible.
        clear.Color[0] = 1.0f;
        clear.Color[1] = 2.0f;
        clear.Color[2] = 3.0f;
        clear.Color[3] = 4.0f;
    } else {
        clear.Color[0] = -1.0f;
        clear.Color[1] = input.maskedCase ? 2.0f : -1.0f;
        clear.Color[2] = input.maskedCase ? -3.0f : -1.0f;
        clear.Color[3] = input.maskedCase ? 4.0f : -1.0f;
    }
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(residentCount);
    UINT64 sourceBytes = 0;
    std::vector<UINT> rows(residentCount);
    std::vector<UINT64> rowBytes(residentCount);
    device->GetCopyableFootprints(&srcDesc, firstResident, residentCount, 0, layouts.data(), rows.data(), rowBytes.data(), &sourceBytes);
    std::vector<BYTE> colors(sourceBytes);
    for (UINT level = 0; level < residentCount; ++level) {
        const auto& packed = levels[level].rows;
        if (rowBytes[level] * rows[level] != packed.size()) throw std::runtime_error("Invalid texture row size");
        for (UINT y = 0; y < rows[level]; ++y) {
            auto offset = layouts[level].Offset + size_t(y) * layouts[level].Footprint.RowPitch;
            std::memcpy(colors.data() + offset, packed.data() + y * rowBytes[level], rowBytes[level]);
        }
    }
    auto staging = upload(device, colors);
    ComPtr<ID3D12Resource> secondTexture, secondStaging;
    if (!input.secondRows.empty()) {
        if (residentCount != 1 || firstResident || input.secondRows.size() != input.rows.size())
            throw std::runtime_error("Invalid second texture extent");
        for (UINT y = 0; y < rows[0]; ++y)
            std::memcpy(colors.data() + y * layouts[0].Footprint.RowPitch,
                        input.secondRows.data() + y * rowBytes[0], rowBytes[0]);
        secondStaging = upload(device, colors);
        secondTexture = resource(device, srcDesc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.NumDescriptors = 1;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ComPtr<ID3D12DescriptorHeap> rtvs, srvs;
    check(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&rtvs)), "Pixel target heap");
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = titleMemory ? 3 + titleAliasSlots : 2;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    check(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&srvs)), "Pixel texture heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = input.format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MostDetailedMip = firstResident;
    view.Texture2D.MipLevels = residentCount;
    device->CreateShaderResourceView(texture.Get(), &view, srvs->GetCPUDescriptorHandleForHeapStart());
    auto secondHandle = srvs->GetCPUDescriptorHandleForHeapStart();
    secondHandle.ptr += device->GetDescriptorHandleIncrementSize(heapDesc.Type);
    device->CreateShaderResourceView(secondTexture.Get(), &view, secondHandle);
    // Title memory follows the two textures: the range table, then one raw view per alias slot.
    const UINT srvIncrement = device->GetDescriptorHandleIncrementSize(heapDesc.Type);
    std::vector<ComPtr<ID3D12Resource>> titleBuffers;
    if (titleMemory) {
        auto handle = secondHandle;
        handle.ptr += srvIncrement;
        titleBuffers.push_back(upload(device, input.titleRanges));
        D3D12_SHADER_RESOURCE_VIEW_DESC table{};
        table.Format = DXGI_FORMAT_UNKNOWN;
        table.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        table.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        table.Buffer.NumElements = static_cast<UINT>(input.titleRanges.size() / titleRangeWords);
        table.Buffer.StructureByteStride = static_cast<UINT>(titleRangeWords * sizeof(Word));
        device->CreateShaderResourceView(titleBuffers.back().Get(), &table, handle);
        for (const auto& alias : input.titleAliases) {
            handle.ptr += srvIncrement;
            titleBuffers.push_back(upload(device, alias));
            D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
            raw.Format = DXGI_FORMAT_R32_TYPELESS;
            raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            raw.Buffer.NumElements = static_cast<UINT>(alias.size());
            raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            device->CreateShaderResourceView(titleBuffers.back().Get(), &raw, handle);
        }
    }

    bool textureUploaded = false;
    for (const auto& req : draws) {
        if (req.isTailDraw)
            std::printf("requested_mip=%u sampled_mip=%u ", req.requestedMip, req.sampledLevel + firstResident);
        const auto& sampled = levels.at(req.sampledLevel);
        ComPtr<ID3D12Resource> partialTriangle;
        if (req.partialQuad)
            partialTriangle = upload(device, std::vector<Word>{3, 4, 5});
        const auto [width, height] = req.isTailDraw ? base.mip(req.requestedMip)
                                                   : Extent{sampled.width * req.scale, sampled.height * req.scale};
        auto dstDesc = texture_desc(width, height, desc.RTVFormats[0], D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        auto target = resource(device, dstDesc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT resultLayout{};
        UINT64 resultBytes = 0;
        device->GetCopyableFootprints(&dstDesc, 0, 1, 0, &resultLayout, nullptr, nullptr, &resultBytes);
        auto back = make_buffer(device, resultBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        device->CreateRenderTargetView(target.Get(), nullptr, rtv);
        ComputeCommands cmd(device, pso.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
        auto* list = cmd.list.Get();
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        if (!textureUploaded) {
            src.pResource = staging.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.pResource = texture.Get();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            for (UINT level = 0; level < residentCount; ++level) {
                src.PlacedFootprint = layouts[level];
                dst.SubresourceIndex = firstResident + level;
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            if (secondTexture) {
                src.pResource = secondStaging.Get();
                dst.pResource = secondTexture.Get();
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                transition(list, secondTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }
            textureUploaded = true;
        }
        ID3D12DescriptorHeap* heaps[]{srvs.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootSignature(root.Get());
        list->SetGraphicsRootDescriptorTable(0, srvs->GetGPUDescriptorHandleForHeapStart());
        list->SetGraphicsRootConstantBufferView(1, gpuConstants->GetGPUVirtualAddress());
        for (UINT i = 0; i < extraConstants.size(); ++i)
            if (extraConstants[i])
                list->SetGraphicsRootConstantBufferView(2 + i, extraConstants[i]->GetGPUVirtualAddress());
        if (titleMemory) {
            auto titleTable = srvs->GetGPUDescriptorHandleForHeapStart();
            titleTable.ptr += 2 * UINT64(srvIncrement);
            list->SetGraphicsRootDescriptorTable(4, titleTable);
        }
        // WARP and hardware can use either the rounded SRV extent or the
        // unrounded base extent for implicit LOD. Their geometric mean is
        // within half a mip of both, since rounding discards less than one texel.
        const auto viewportAxis = [&](UINT target, UINT resident, UINT baseAxis) {
            const float unrounded = std::ldexp(float(baseAxis), -static_cast<int>(firstResident));
            const float span = std::sqrt(resident * (std::max)(1.0f, unrounded));
            return (std::max)(float(target), std::ldexp(span, -static_cast<int>(req.sampledLevel)));
        };
        const float viewportWidth = viewportAxis(width, input.width, base.width);
        const float viewportHeight = viewportAxis(height, input.height, base.height);
        const D3D12_VIEWPORT viewport = input.viewport.value_or(
            D3D12_VIEWPORT{(width - viewportWidth) * 0.5f, (height - viewportHeight) * 0.5f,
                           viewportWidth, viewportHeight, 0, 1});
        const D3D12_RECT scissor{0,0,LONG(width),LONG(height)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ComPtr<ID3D12QueryHeap> queries;
        ComPtr<ID3D12Resource> queryResults;
        if (!input.coverage.empty()) {
            const UINT pixels = width * height;
            if (pixels != input.coverage.size()) throw std::runtime_error("Coverage reference extent");
            D3D12_QUERY_HEAP_DESC queryDesc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, pixels + 1, 0};
            check(device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queries)), "Coverage queries");
            queryResults = make_buffer(device, (pixels + 1) * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
            // First measure a full draw with neighboring lanes active, then locate
            // each surviving sample with a one-pixel scissor.
            for (UINT i = 0; i <= pixels; ++i) {
                if (i) {
                    LONG x = (i - 1) % width, y = (i - 1) / width;
                    const D3D12_RECT pixel{x, y, x + 1, y + 1};
                    list->RSSetScissorRects(1, &pixel);
                }
                list->BeginQuery(queries.Get(), D3D12_QUERY_TYPE_OCCLUSION, i);
                list->DrawInstanced(3, 1, 0, 0);
                list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_OCCLUSION, i);
            }
            list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_OCCLUSION, 0, pixels + 1, queryResults.Get(), 0);
        } else if (partialTriangle) {
            const D3D12_INDEX_BUFFER_VIEW indices{partialTriangle->GetGPUVirtualAddress(),
                                                   3 * sizeof(Word), DXGI_FORMAT_R32_UINT};
            list->IASetIndexBuffer(&indices);
            list->DrawIndexedInstanced(3, 1, 0, 0, 0);
        } else {
            list->DrawInstanced(3,1,0,0);
        }
        transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src.pResource = target.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        dst.pResource = back.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = resultLayout;
        list->CopyTextureRegion(&dst, 0,0,0, &src, nullptr);
        cmd.submit_and_wait(device);
        if (queryResults) {
            auto counts = read(queryResults.Get(), (input.coverage.size() + 1) * 2);
            UINT sum = 0;
            for (size_t i = 0; i < input.coverage.size(); ++i) {
                sum += input.coverage[i];
                if (counts[2*i+2] != input.coverage[i] || counts[2*i+3]) {
                    std::fprintf(stderr, "coverage pixel=%zu expected=%u actual=%u\n", i, unsigned(input.coverage[i]), counts[2*i+2]);
                    throw std::runtime_error("Sample coverage mismatch");
                }
            }
            if (counts[0] != sum || counts[1]) throw std::runtime_error("Full draw coverage mismatch");
            std::printf("covered_samples=%u checked_pixels=%zu\n", sum, input.coverage.size());
        }
        auto words = read(back.Get(), resultBytes/4);
        if (rawReadback) {
            std::ofstream output(rawReadback, std::ios::binary);
            for (UINT y = 0; y < height; ++y) {
                const auto* row = words.data() + y * resultLayout.Footprint.RowPitch / sizeof(UINT);
                output.write(reinterpret_cast<const char*>(row), width * sizeof(Color));
            }
            output.close();
            if (!output) throw std::runtime_error("Cannot save raw pixel readback");
        }
        float maxError = 0;
        std::array<UINT, 4> firstBits{};
        std::vector<BYTE> image;
        if (capture) image.reserve(size_t(width) * height * 4);
        for (UINT y = 0; y < height; ++y) {
            const UINT numeratorY = (2 * y + 1) * sampled.height;
            const UINT sampleY = numeratorY / (2 * height);
            for (UINT x = 0; x < width; ++x) {
                const UINT numeratorX = (2 * x + 1) * sampled.width;
                const UINT sampleX = numeratorX / (2 * width);
                const auto& expected = sampled.expected[sampleY * sampled.width + sampleX];
                const auto offset = y * resultLayout.Footprint.RowPitch / 4 + x * 4;
                Color actual;
                std::memcpy(actual.data(), words.data() + offset, sizeof(actual));
                const auto bits = [](float value) {
                    UINT word;
                    std::memcpy(&word, &value, sizeof(word));
                    return word;
                };
                const auto is_nan_bits = [](UINT word) {
                    return (word & 0x7f800000u) == 0x7f800000u && (word & 0x007fffffu) != 0u;
                };
                // Per-channel compare against one expected texel. Unmasked channels are strict
                // bitwise. A masked channel accepts a NaN against an expected NaN of any sign or
                // payload and still logs both raw words; masking a channel whose expected word is
                // not NaN is a case-format error.
                float pixelError = 0;
                const auto matches_at = [&](size_t index) {
                    const auto& reference = sampled.expected[index];
                    float candidateError = 0;
                    const BYTE mask = index < sampled.nanMask.size() ? sampled.nanMask[index] : BYTE{0};
                    for (size_t c = 0; c < actual.size(); ++c) {
                        const UINT expected_bits = bits(reference[c]);
                        const UINT actual_bits = bits(actual[c]);
                        if (mask >> c & 1) {
                            if (!is_nan_bits(expected_bits))
                                throw std::runtime_error("NaN-class mask on a channel whose expected word is not NaN");
                            std::printf("nan_class x=%u y=%u ch=%zu expected=%08x actual=%08x\n",
                                        x, y, c, expected_bits, actual_bits);
                            if (!is_nan_bits(actual_bits)) return false;
                        } else if (expected_bits != actual_bits) {
                            // Only the rendered raw-LOD channel in the upper-clamp fixture.
                            if (req.allowUpperRawLodStep && c == 0 && expected_bits == 0) {
                                constexpr float step = 1.0f / (1u << D3D12_MIP_LOD_FRACTIONAL_BIT_COUNT);
                                const float error = std::fabs(actual[c]);
                                if (!(error <= step)) return false;
                                candidateError = (std::max)(candidateError, error);
                                continue;
                            }
                            if (!input.tolerance) return false;
                            const float error = std::fabs(actual[c] - reference[c]);
                            if (!(error <= input.tolerance)) return false;   // NaN fails too
                            candidateError = (std::max)(candidateError, error);
                        }
                    }
                    pixelError = candidateError;
                    return true;
                };
                bool matched = matches_at(sampleY * sampled.width + sampleX);
                // The exact texel is always tried first. Neighbors are admitted only
                // for magnified tail draws, which sample through the root signature's
                // single static sampler (MIN_MAG_MIP_POINT, clamp addressing).
                if (!matched && req.isTailDraw && req.scale > 1) {
                    // Per axis, at most one immediate neighbor in the same mip:
                    // the previous texel for a center exactly on a boundary, or
                    // the next texel for a center up to 1/512 texel below a
                    // boundary (tie included), where rounding the scaled
                    // coordinate to 8 fractional bits crosses it. The last texel
                    // has no boundary above it under clamp addressing.
                    const auto neighbor = [](UINT numerator, UINT denominator, UINT sample, UINT extent) {
                        const UINT remainder = numerator % denominator;
                        if (!remainder) return sample - 1;
                        const bool roundsUp = UINT64(remainder) * 512 >= UINT64(denominator) * 511;
                        return roundsUp && sample + 1 < extent ? sample + 1 : sample;
                    };
                    const UINT nearX = neighbor(numeratorX, 2 * width, sampleX, sampled.width);
                    const UINT nearY = neighbor(numeratorY, 2 * height, sampleY, sampled.height);
                    // Each candidate must match as a whole RGBA texel.
                    matched = matches_at(sampleY * sampled.width + nearX) ||
                              matches_at(nearY * sampled.width + sampleX) ||
                              matches_at(nearY * sampled.width + nearX);
                }
                if (!matched) {
                    std::fprintf(stderr, "pixel_mismatch x=%u y=%u\n", x, y);
                    for (size_t c = 0; c < actual.size(); ++c)
                        std::fprintf(stderr, "channel=%zu expected=%g actual=%g expected_bits=%08x actual_bits=%08x\n",
                                     c, expected[c], actual[c], bits(expected[c]), bits(actual[c]));
                    throw PixelMismatch();
                }
                maxError = (std::max)(maxError, pixelError);
                if (!x && !y)
                    for (size_t c = 0; c < actual.size(); ++c) firstBits[c] = bits(actual[c]);
                if (capture) {
                    for (float value : actual) image.push_back(static_cast<BYTE>(std::lround(value * 255)));
                }
            }
        }
        // Filtered results may differ by adapter within the tolerance; the raw bits let callers compare them.
        if (input.tolerance)
            std::printf("first_pixel_bits=%08x,%08x,%08x,%08x\n", firstBits[0], firstBits[1], firstBits[2], firstBits[3]);
        if (capture) {
            std::ofstream output(capture, std::ios::binary);
            output.write(reinterpret_cast<const char*>(image.data()), image.size());
            output.close();
            if (!output) throw std::runtime_error("Cannot save GPU readback");
        }
        std::printf("pixels=%u channels=%u max_absolute_error=%g\n", width * height, width * height * 4, maxError);
    }
}

void draw_verified(ID3D12Device* device, const char* path, const std::vector<PixelInput>& levels,
                   UINT firstResident, UINT sampledLevel, const char* capture,
                   ID3D12Resource* gpuConstants = nullptr, const char* rawReadback = nullptr,
                   ID3D12RootSignature* rootOverride = nullptr, const char* vsPath = nullptr)
{
    const auto& top = levels.front();
    draw_verified(device, path, levels, {top.width << firstResident, top.height << firstResident},
                  firstResident, {{sampledLevel, levels.at(sampledLevel).scale}}, capture, gpuConstants, rawReadback,
                  rootOverride, vsPath);
}

ComPtr<ID3D12Resource> produce_pixel_constants(ID3D12Device* device, const char* path)
{
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> shader{std::istreambuf_iterator<char>(file), {}};
    if (shader.empty() || file.bad()) throw std::runtime_error("Cannot read compiled compute producer");

    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC signature{};
    signature.NumParameters = 1;
    signature.pParameters = &parameter;
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr),
          "Serialize compute producer root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&root)), "Compute producer root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.Get();
    desc.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Compute producer pipeline");

    auto constants = make_buffer(device, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    commands.list->SetComputeRootSignature(root.Get());
    commands.list->SetComputeRootUnorderedAccessView(0, constants->GetGPUVirtualAddress());
    commands.list->Dispatch(1, 1, 1);
    transition(commands.list.Get(), constants.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    commands.submit_and_wait(device);
    return constants;
}
}

static void check_pixel(ID3D12Device* device, const char* path, bool swapRB, const char* reference,
                        UINT8 writeMask, ID3D12Resource* gpuConstants, const char* rawReadback = nullptr)
{
    PixelInput input{8, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM};
    input.writeMask = writeMask;
    for (auto g : ramp) {
        for (auto r : ramp) {
            input.rows.insert(input.rows.end(), {BYTE(r), BYTE(g), 128, 255});
            Color expected{expected_channel(r), expected_channel(g), expected_channel(128), 1};
            if (swapRB) std::swap(expected[0], expected[2]);
            input.expected.push_back(expected);
        }
    }
    if (reference) {
        std::ifstream file(reference, std::ios::binary);
        file.read(reinterpret_cast<char*>(input.constants.data()), input.constants.size() * sizeof(Word));
        file.read(reinterpret_cast<char*>(input.expected.data()), input.expected.size() * sizeof(Color));
        if (!file)
            throw std::runtime_error("Expected a 256-byte constant buffer and 64 float4 reference colors");
        if (file.peek() != std::char_traits<char>::eof()) {
            BYTE flags = 0;
            file.read(reinterpret_cast<char*>(&flags), 1);
            if ((flags & 4u) && (flags & 0xF0u))
                throw std::runtime_error("Unknown case trailer flag bits");
            input.integerRT = (flags & 1u) != 0;
            const UINT8 caseMask = flags >> 4;
            if (caseMask) {
                input.maskedCase = true;
                input.writeMask = caseMask;
            }
            if (flags & 2u) {
                std::array<UINT, 4> rect{};
                file.read(reinterpret_cast<char*>(rect.data()), sizeof(rect));
                const auto [left, top, width, height] = rect;
                if (!file || !width || !height || left >= input.width || top >= input.height ||
                    width > input.width - left || height > input.height - top)
                    throw std::runtime_error("Invalid pixel case viewport");
                // Explicit viewport cases provide one reference color per target pixel.
                input.scale = 1;
                input.viewport = D3D12_VIEWPORT{float(left), float(top), float(width), float(height), 0, 1};
            }
            if (flags & 4u) read_title_memory(file, input);
            if (flags & 8u) {
                input.extraConstants.resize(128);
                file.read(reinterpret_cast<char*>(input.extraConstants.data()),
                          static_cast<std::streamsize>(input.extraConstants.size() * sizeof(Word)));
                if (!file)
                    throw std::runtime_error("Expected two 256-byte constant buffers for b1 and b2");
            }
            if (file.peek() != std::char_traits<char>::eof()) {
                if (input.integerRT)
                    throw std::runtime_error("NaN-class masks require a floating-point render target");
                input.nanMask.resize(64);
                file.read(reinterpret_cast<char*>(input.nanMask.data()),
                          static_cast<std::streamsize>(input.nanMask.size()));
                if (!file || file.peek() != std::char_traits<char>::eof())
                    throw std::runtime_error("Expected exactly 64 NaN mask bytes after the case trailer fields");
            }
        }
    }
    std::vector<PixelInput> levels;
    levels.push_back(std::move(input));
    draw_verified(device, path, levels, 0, 0, nullptr, gpuConstants, rawReadback);
}

void check_translated_pixel(ID3D12Device* device, const char* path, bool swapRB, const char* reference,
                            UINT8 writeMask, const char* rawReadback)
{
    check_pixel(device, path, swapRB, reference, writeMask, nullptr, rawReadback);
}

void check_compute_to_pixel(ID3D12Device* device, const char* compute, const char* pixel,
                            const char* reference, const char* previous)
{
    auto stale = previous ? produce_pixel_constants(device, previous) : ComPtr<ID3D12Resource>{};
    auto current = produce_pixel_constants(device, compute);
    check_pixel(device, pixel, false, reference, D3D12_COLOR_WRITE_ENABLE_ALL,
                stale ? stale.Get() : current.Get());
}

void check_pixel_gather(ID3D12Device* device, const char* shader, const char* reference)
{
    constexpr UINT side = 8;
    std::ifstream file(reference, std::ios::binary);
    std::array<UINT, 2> view{};
    file.read(reinterpret_cast<char*>(view.data()), sizeof(view));
    const auto [first, count] = view;
    if (!file || first > 1 || !count || count > 3)
        throw std::runtime_error("Invalid pixel gather view");
    std::vector<PixelInput> levels;
    for (UINT mip = 0; mip < count; ++mip) {
        const UINT extent = side >> mip;
        PixelInput input{extent, extent, 1, DXGI_FORMAT_R32G32B32A32_FLOAT};
        input.rows.resize(size_t(extent) * extent * sizeof(Color));
        file.read(reinterpret_cast<char*>(input.rows.data()), static_cast<std::streamsize>(input.rows.size()));
        levels.push_back(std::move(input));
    }
    auto& expected = levels.front().expected;
    expected.resize(side * side);
    file.read(reinterpret_cast<char*>(expected.data()), side * side * sizeof(Color));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Invalid pixel gather case data");
    draw_verified(device, shader, levels, first, 0, nullptr);
}

// Comparison gathers bind a SamplerComparisonState built from the translated smpc line and a
// single-component R32_FLOAT view (R32G32B32A32_FLOAT does not advertise SHADER_GATHER_COMPARISON).
// The case layout is the gather case with one float per texel instead of four.
void check_pixel_compare_gather(ID3D12Device* device, const char* shader, const char* reference,
                                const char* sampler_line, const char* vsPath)
{
    constexpr UINT side = 8;
    std::ifstream file(reference, std::ios::binary);
    std::array<UINT, 2> view{};
    file.read(reinterpret_cast<char*>(view.data()), sizeof(view));
    const auto [first, count] = view;
    if (!file || first > 1 || !count || count > 3)
        throw std::runtime_error("Invalid comparison gather view");
    std::vector<PixelInput> levels;
    for (UINT mip = 0; mip < count; ++mip) {
        const UINT extent = side >> mip;
        PixelInput input{extent, extent, 1, DXGI_FORMAT_R32_FLOAT};
        input.rows.resize(size_t(extent) * extent * sizeof(float));
        file.read(reinterpret_cast<char*>(input.rows.data()), static_cast<std::streamsize>(input.rows.size()));
        levels.push_back(std::move(input));
    }
    auto& expected = levels.front().expected;
    expected.resize(side * side);
    file.read(reinterpret_cast<char*>(expected.data()), side * side * sizeof(Color));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Invalid comparison gather case data");

    x2w_startup::TranslatedSamplers samplers;
    if (!samplers.read_manifest_line(sampler_line) || samplers.statics.size() != 1 ||
        samplers.comparison.size() != 1 || !samplers.comparison[0])
        throw std::runtime_error("Expected exactly one smpc line for the comparison gather");
    const auto statics = x2w_startup::decode_statics(samplers);

    // Same parameter layout as tests/pixel_aot_vs.hlsl, with the comparison sampler in place of its
    // ordinary point sampler: table t0 at 0, b0..b2 at 1..3, one static sampler at s0.
    D3D12_DESCRIPTOR_RANGE srv{};
    srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv.NumDescriptors = 2;
    D3D12_ROOT_PARAMETER parameters[4]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {1, &srv};
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    for (UINT i = 1; i < 4; ++i) {
        parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[i].Descriptor.ShaderRegister = i - 1;
        parameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    D3D12_ROOT_SIGNATURE_DESC signature{};
    signature.NumParameters = 4;
    signature.pParameters = parameters;
    signature.NumStaticSamplers = 1;
    signature.pStaticSamplers = statics.data();
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
          "Serialize comparison gather root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Comparison gather root");
    draw_verified(device, shader, levels, first, 0, nullptr, nullptr, nullptr, root.Get(), vsPath);
}

void check_pixel_quad_lod(ID3D12Device* device, const char* shader, const char* rawReadback,
                          bool allowUpperRawLodStep)
{
    PixelInput top{8, 8, 1, DXGI_FORMAT_R8G8B8A8_UNORM};
    PixelInput next{4, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM};
    const float scale = 4.0f;
    std::memcpy(&top.constants[2], &scale, sizeof(scale));
    for (UINT y = 0; y < top.height; ++y) {
        for (UINT x = 0; x < top.width; ++x) {
            top.rows.insert(top.rows.end(), {255, 0, 0, 255});
            // Pixel centers have x + y + 1 below the triangle's 7.75-pixel edge.
            top.expected.push_back(x + y <= 6 ? Color{0, 1, 0, 1} : Color{-1, -1, -1, -1});
        }
    }
    for (UINT texel = 0; texel < next.width * next.height; ++texel)
        next.rows.insert(next.rows.end(), {0, 255, 0, 255});
    MipDraw draw{0, 1};
    draw.partialQuad = true;
    draw.allowUpperRawLodStep = allowUpperRawLodStep;
    draw_verified(device, shader, {std::move(top), std::move(next)}, {8, 8}, 0, {draw}, nullptr,
                  nullptr, rawReadback);
}

namespace {
// One solid color per mip, so the color alone identifies the level a sample read.
constexpr std::array<std::array<BYTE, 4>, 4> kMipColors{{
    {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}}};

std::vector<PixelInput> solid_mips()
{
    std::vector<PixelInput> levels;
    for (UINT level = 0; level < kMipColors.size(); ++level) {
        PixelInput input{8u >> level, 8u >> level, 1, DXGI_FORMAT_R8G8B8A8_UNORM};
        for (UINT texel = 0; texel < input.width * input.height; ++texel)
            input.rows.insert(input.rows.end(), kMipColors[level].begin(), kMipColors[level].end());
        levels.push_back(std::move(input));
    }
    return levels;
}
}

void check_pixel_mip_clamp(ID3D12Device* device, const char* shader, UINT expectedLevel)
{
    if (expectedLevel >= kMipColors.size()) throw std::runtime_error("Expected a mip level from 0 to 3");
    auto levels = solid_mips();
    const auto& rgba = kMipColors[expectedLevel];
    const Color expected{expected_channel(rgba[0]), expected_channel(rgba[1]),
                         expected_channel(rgba[2]), expected_channel(rgba[3])};
    levels.front().expected.assign(64, expected);
    draw_verified(device, shader, levels, {8, 8}, 0, {MipDraw{0, 1}}, nullptr);
}

void check_pixel_mip_blend(ID3D12Device* device, const char* shader, float clamp, float tolerance)
{
    if (!(clamp >= 0 && clamp <= 8) || !(tolerance > 0 && tolerance < 0.5f))
        throw std::runtime_error("Expected a clamp from 0 to 8 and a tolerance below 0.5");
    // The constant coordinate has no derivatives, so the LOD is the clamp, limited to the last mip.
    const float lod = (std::min)(clamp, float(kMipColors.size() - 1));
    const UINT low = UINT(lod);
    const UINT high = (std::min)(low + 1, UINT(kMipColors.size() - 1));
    const float weight = lod - float(low);
    Color expected{};
    for (size_t c = 0; c < expected.size(); ++c)
        expected[c] = (1 - weight) * (kMipColors[low][c] / 255.0f) + weight * (kMipColors[high][c] / 255.0f);
    auto levels = solid_mips();
    levels.front().expected.assign(64, expected);
    levels.front().linearMips = true;
    levels.front().tolerance = tolerance;
    draw_verified(device, shader, levels, {8, 8}, 0, {MipDraw{0, 1}}, nullptr);
}

void check_attribute_pixel(ID3D12Device* device, const char* shader, const char* reference, bool rg)
{
    PixelInput input{32, 32, 1, DXGI_FORMAT_R8G8B8A8_UNORM};
    if (rg) input.writeMask = D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN;
    input.rows.resize(input.width * input.height * 4);
    input.expected.resize(input.width * input.height);
    std::ifstream file(reference, std::ios::binary);
    file.read(reinterpret_cast<char*>(input.constants.data()), input.constants.size() * sizeof(Word));
    file.read(reinterpret_cast<char*>(input.expected.data()), input.expected.size() * sizeof(Color));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Expected a 256-byte constant buffer and 32 by 32 float4 colors");
    draw_verified(device, shader, {std::move(input)}, 0, 0, nullptr);
}

void check_sampled_coverage(ID3D12Device* device, const char* shader, const char* reference)
{
    PixelInput input{8, 8, 1, DXGI_FORMAT_R8G8B8A8_UNORM};
    for (UINT y = 0; y < 8; ++y) {
        for (UINT x = 0; x < 8; ++x) {
            BYTE alpha = BYTE((x*41 + y*19) & 255);
            input.rows.insert(input.rows.end(), {255, 0, 128, alpha});
            input.secondRows.insert(input.secondRows.end(), {0, 255, 128, BYTE(255-alpha)});
        }
    }
    input.expected.assign(64, Color{-1, -1, -1, -1});
    input.coverage.resize(64);
    std::ifstream file(reference, std::ios::binary);
    file.read(reinterpret_cast<char*>(input.constants.data()), input.constants.size() * sizeof(Word));
    file.read(reinterpret_cast<char*>(input.coverage.data()), input.coverage.size());
    if (!file || file.peek() != std::char_traits<char>::eof() ||
        std::any_of(input.coverage.begin(), input.coverage.end(), [](BYTE v){ return v > 1; }))
        throw std::runtime_error("Invalid coverage case");
    draw_verified(device, shader, {input}, 0, 0, nullptr);
}

void check_dual_pixel(ID3D12Device* device, const char* shader, const char* capture, bool swap)
{
    PixelInput input{8, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM};
    for (auto blue : ramp) {
        for (auto red : ramp) {
            input.rows.insert(input.rows.end(), {BYTE(red), 0, 0, 255});
            input.secondRows.insert(input.secondRows.end(), {0, 0, BYTE(blue), 255});
            // The two nonzero channels use exact power-of-two scaling after UNORM conversion.
            input.expected.push_back({expected_channel(red * 3) / 4, 0, expected_channel(blue) / 4, 1});
        }
    }
    if (swap) std::swap(input.rows, input.secondRows);
    draw_verified(device, shader, {input}, 0, 0, capture);
}

namespace {
PixelInput read_bc7_case(const char* fixture)
{
    std::ifstream file(fixture, std::ios::binary);
    std::array<char,4> magic{};
    UINT w = 0, h = 0;
    file.read(magic.data(), 4);
    file.read(reinterpret_cast<char*>(&w), sizeof(w));
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!file || std::memcmp(magic.data(), "BC7C", 4) || !w || !h ||
        w > 4096 || h > 4096 || size_t(w) * h > 4 * 1024 * 1024)
        throw std::runtime_error("Invalid single-level BC7 case dimensions or header");
    PixelInput input{w, h, 1, DXGI_FORMAT_BC7_UNORM};
    const size_t pixels = size_t(w) * h;
    input.rows.resize(((w + 3) / 4) * ((h + 3) / 4) * 16);
    file.read(reinterpret_cast<char*>(input.rows.data()), input.rows.size());
    std::vector<std::array<BYTE,4>> reference(pixels);
    file.read(reinterpret_cast<char*>(reference.data()), pixels * 4);
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("BC7 block rows or RGBA reference have incorrect size");
    input.expected.reserve(pixels);
    for (const auto& rgba : reference) {
        input.expected.push_back({expected_channel(rgba[0]), expected_channel(rgba[1]),
                                  expected_channel(rgba[2]), expected_channel(rgba[3])});
    }
    return input;
}

bool is_next_mip(const PixelInput& upper, const PixelInput& lower)
{
    return (upper.width > 1 || upper.height > 1) &&
           lower.width == std::max<UINT>(1, upper.width / 2) &&
           lower.height == std::max<UINT>(1, upper.height / 2);
}
}

void check_bc7_pixel(ID3D12Device* device, const char* shader, const char* fixture,
                     const char* capture, const char* mip1)
{
    std::vector<PixelInput> levels;
    levels.push_back(read_bc7_case(fixture));
    if (mip1) {
        auto lower = read_bc7_case(mip1);
        if (!is_next_mip(levels.front(), lower))
            throw std::runtime_error("Expected a half-size BC7 mip");
        levels.push_back(std::move(lower));
    }
    draw_verified(device, shader, levels, 0, static_cast<UINT>(levels.size() - 1), capture);
}

void check_bc7_tail(ID3D12Device* device, const char* shader, const char* firstMip, int count, char** paths)
{
    const std::string_view arg(firstMip);
    UINT first = 0;
    const auto parsed = std::from_chars(arg.data(), arg.data() + arg.size(), first);
    if (parsed.ec != std::errc{} || parsed.ptr != arg.data() + arg.size() || first > 12)
        throw std::runtime_error("Invalid first resident mip");
    Extent base{};
    const bool explicitBase = count > 0 && std::string_view(paths[0]) == "--base-size";
    if (explicitBase) {
        if (count < 4) throw std::runtime_error("Expected --base-size WIDTH HEIGHT followed by BC7 cases");
        const auto dimension = [](std::string_view text) -> UINT {
            UINT value = 0;
            const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
            if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !value)
                throw std::runtime_error("Invalid BC7 base dimension");
            return value;
        };
        base = {dimension(paths[1]), dimension(paths[2])};
        paths += 3;
        count -= 3;
    }
    std::vector<PixelInput> levels;
    for (int i = 0; i < count; ++i) {
        auto level = read_bc7_case(paths[i]);
        if (levels.empty()) {
            if (!explicitBase) base = {level.width << first, level.height << first};
            if (base.width > 4096 || base.height > 4096 || UINT64(base.width) * base.height > 4 * 1024 * 1024)
                throw std::runtime_error("Logical BC7 texture exceeds fixture bounds");
            const auto resident = base.mip(first);
            if (!(std::max<UINT>(base.width, base.height) >> first) ||
                resident.width != level.width || resident.height != level.height)
                throw std::runtime_error("BC7 base size is incompatible with the first resident mip");
        } else if (!is_next_mip(levels.back(), level)) {
            throw std::runtime_error("Expected consecutive BC7 mip sizes");
        }
        levels.push_back(std::move(level));
    }
    // Changing the viewport exercises the copy shader's implicit LOD with the same SRV range.
    std::vector<MipDraw> draws;
    for (UINT mip = 0; mip < first + levels.size(); ++mip) {
        const UINT scale = mip < first ? 1u << (first - mip) : 1u;
        const UINT selected = mip < first ? 0 : mip - first;
        draws.push_back({selected, scale, mip, true});
    }
    draw_verified(device, shader, levels, base, first, draws, nullptr);
}
