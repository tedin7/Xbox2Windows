#include "dxr_fixture.h"
#include "dxr_placed_build.h"
#include "era_title_va.h"
#include "ray_candidates.h"
#include "ray_candidates_pipeline.h"
#include "ray_pipeline.h"
#include "ray_query.h"
#include "ray_query_table.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

using Microsoft::WRL::ComPtr;

namespace dxr_fixture {
bool Capabilities::supports(Backend backend) const
{
    if (backend == Backend::pipeline)
        return tier >= D3D12_RAYTRACING_TIER_1_0 && model >= D3D_SHADER_MODEL_6_3;
    return tier >= D3D12_RAYTRACING_TIER_1_1 && model >= D3D_SHADER_MODEL_6_5;
}

Capabilities capabilities(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
    const HRESULT tier = check_feature(device, D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options));
    if (tier != E_INVALIDARG) check(tier, "Query DXR tier");
    D3D_SHADER_MODEL highest{};
    // Older runtimes can reject a shader-model enum they do not recognize.
    for (auto requested : {D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_3}) {
        D3D12_FEATURE_DATA_SHADER_MODEL model{requested};
        const HRESULT hr = check_feature(device, D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model));
        if (hr == E_INVALIDARG) continue;
        check(hr, "Query DXR shader model");
        highest = model.HighestShaderModel;
        break;
    }
    std::printf("dxr_tier=%u shader_model_capped_at_6_5=0x%x\n", options.RaytracingTier, highest);
    return {options.RaytracingTier, highest};
}

bool supported(ID3D12Device* device, Backend backend)
{
    if (capabilities(device).supports(backend)) return true;
    std::puts(backend == Backend::pipeline
        ? "skip: DispatchRays requires DXR Tier 1.0 and Shader Model 6.3"
        : "skip: inline RayQuery requires DXR Tier 1.1 and Shader Model 6.5");
    return false;
}

D3D12_RAYTRACING_INSTANCE_DESC instance(UINT id, UINT mask, float translateX) {
    D3D12_RAYTRACING_INSTANCE_DESC result{};
    result.Transform[0][0] = result.Transform[1][1] = result.Transform[2][2] = 1;
    result.Transform[0][3] = translateX;
    result.InstanceID = id;
    result.InstanceMask = mask;
    return result;
}

struct RayPipeline {
    ComPtr<ID3D12StateObject> state;
    ComPtr<ID3D12Resource> shaderTable;
    D3D12_DISPATCH_RAYS_DESC dispatch{};
};

static RayPipeline make_ray_pipeline(ID3D12Device5* device, ID3D12RootSignature* root,
                                     UINT rayCount, UINT opacityGeometries)
{
    const D3D12_SHADER_BYTECODE bytecode = opacityGeometries
        ? D3D12_SHADER_BYTECODE{ray_candidates_pipeline, sizeof(ray_candidates_pipeline)}
        : D3D12_SHADER_BYTECODE{ray_pipeline, sizeof(ray_pipeline)};
    const D3D12_DXIL_LIBRARY_DESC library{bytecode, 0, nullptr};
    LPCWSTR hitGroupName = L"TriangleHitGroup";
    D3D12_HIT_GROUP_DESC hitGroup{};
    hitGroup.HitGroupExport = hitGroupName;
    hitGroup.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    hitGroup.ClosestHitShaderImport = L"ClosestHit";
    ComPtr<ID3D12RootSignature> hitRoot;
    if (opacityGeometries) {
        hitGroup.AnyHitShaderImport = L"AnyHit";
        D3D12_ROOT_PARAMETER geometryIndex{};
        geometryIndex.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        geometryIndex.Constants = {0, 0, 1};
        const D3D12_ROOT_SIGNATURE_DESC layout{1, &geometryIndex, 0, nullptr,
                                               D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE};
        ComPtr<ID3DBlob> blob;
        check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
              "Serialize DXR hit group root");
        check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&hitRoot)),
              "Create DXR hit group root");
    }
    const D3D12_RAYTRACING_SHADER_CONFIG shaderConfig{sizeof(bvh_fixture::Row), 2 * sizeof(float)};
    const D3D12_GLOBAL_ROOT_SIGNATURE globalRoot{root};
    const D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig{1};
    const D3D12_LOCAL_ROOT_SIGNATURE localRoot{hitRoot.Get()};
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION hitRootAssociation{};
    const D3D12_STATE_SUBOBJECT subobjects[]{
        {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
        {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hitGroup},
        {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig},
        {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &globalRoot},
        {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipelineConfig},
        {D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &localRoot},
        {D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &hitRootAssociation},
    };
    const UINT commonSubobjectCount = static_cast<UINT>(std::size(subobjects)) - 2;
    hitRootAssociation = {&subobjects[commonSubobjectCount], 1, &hitGroupName};
    const UINT subobjectCount = opacityGeometries ? static_cast<UINT>(std::size(subobjects)) : commonSubobjectCount;
    const D3D12_STATE_OBJECT_DESC desc{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
        subobjectCount, subobjects};
    RayPipeline pipeline;
    check(device->CreateStateObject(&desc, IID_PPV_ARGS(&pipeline.state)), "Create DXR ray pipeline");

    ComPtr<ID3D12StateObjectProperties> properties;
    check(pipeline.state.As(&properties), "Get DXR state object properties");
    constexpr UINT tableAlignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    constexpr UINT identifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT hitCount = (std::max)(1u, opacityGeometries);
    std::vector<std::array<BYTE, tableAlignment>> records(2 + hitCount);
    const auto writeIdentifier = [&](auto& record, LPCWSTR name) {
        const void* identifier = properties->GetShaderIdentifier(name);
        if (!identifier) throw std::runtime_error("Missing DXR shader identifier");
        std::memcpy(record.data(), identifier, identifierSize);
    };
    writeIdentifier(records[0], L"RayGen");
    writeIdentifier(records[1], L"Miss");
    // TraceRay uses multiplier 1 and all instances use offset 0, so
    // each geometry selects its own record with one DWORD of local data.
    static_assert(identifierSize + sizeof(UINT) <= tableAlignment);
    for (UINT g = 0; g < hitCount; ++g) {
        auto& record = records[2 + g];
        writeIdentifier(record, hitGroupName);
        if (opacityGeometries) std::memcpy(record.data() + identifierSize, &g, sizeof(g));
    }
    pipeline.shaderTable = bvh_fixture::upload(device, records);
    const auto base = pipeline.shaderTable->GetGPUVirtualAddress();
    pipeline.dispatch.RayGenerationShaderRecord = {base, identifierSize};
    pipeline.dispatch.MissShaderTable = {base + tableAlignment, identifierSize, identifierSize};
    const UINT hitStride = opacityGeometries ? tableAlignment : identifierSize;
    pipeline.dispatch.HitGroupTable = {base + 2 * tableAlignment, hitCount * hitStride, hitStride};
    pipeline.dispatch.Width = rayCount;
    pipeline.dispatch.Height = pipeline.dispatch.Depth = 1;
    return pipeline;
}

// This fixture registers two distinct, live AS buffers whose synthetic title
// bases have the same low 40 bits. Each AS starts inside its mapped range.
struct AddressScene {
    std::vector<Vertex> otherTriangle;
    bool truncate;
    // title: build and instance addresses go through the title range table (the path under test).
    // direct: the native control; the builds and the TLAS use the two BLAS GPU addresses without the table, and
    //   with truncate both instances name the left BLAS explicitly.
    // misroute_high: the title path, but the high instance is deliberately looked up at the low title base.
    enum class Route { title, direct, misroute_high } route = Route::title;
};
constexpr UINT64 lowTitle = 0x2300000000ull;
constexpr UINT64 highTitle = lowTitle + (1ull << 40);
constexpr UINT64 asOffset = 2 * D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;

static std::vector<bvh_fixture::Word> trace_scene(ID3D12Device* device, const std::vector<Vertex>& vertices,
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances, const std::vector<Ray>& rays,
    Backend backend, const SceneUpdate* update, const std::vector<D3D12_RAYTRACING_GEOMETRY_FLAGS>* triangleGeometries,
    const AddressScene* addressScene, PlacedBuildScene* placedScene = nullptr, const ResolvedBatch* resolved = nullptr,
    BuildCounts* counts = nullptr) {
    const bool usePipeline = backend == Backend::pipeline;
    if (usePipeline) {
        if (triangleGeometries && (triangleGeometries->empty() || triangleGeometries->size() > 4))
            throw std::runtime_error("The DXR opacity pipeline requires one to four geometries");
        const auto usesHitGroupOffset = [](const auto& descriptors) {
            return std::any_of(descriptors.begin(), descriptors.end(), [](const auto& desc) {
                return desc.InstanceContributionToHitGroupIndex != 0;
            });
        };
        if (usesHitGroupOffset(instances) || (update && usesHitGroupOffset(update->instances)))
            throw std::runtime_error("The DXR pipeline fixture requires instance hit-group offset 0");
        if (std::any_of(rays.begin(), rays.end(), [triangleGeometries](const Ray& ray) {
                if (triangleGeometries) {
                    constexpr UINT opacityFlags = D3D12_RAY_FLAG_FORCE_OPAQUE | D3D12_RAY_FLAG_FORCE_NON_OPAQUE |
                                                  D3D12_RAY_FLAG_CULL_OPAQUE | D3D12_RAY_FLAG_CULL_NON_OPAQUE;
                    constexpr UINT allowed = opacityFlags | D3D12_RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | 0xf0000u;
                    const UINT opacity = ray.flags & opacityFlags;
                    return (ray.flags & ~allowed) != 0 || (opacity & (opacity - 1)) != 0;
                }
                constexpr UINT cullFlags = D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES |
                                           D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
                return (ray.flags & ~(report_facing | cullFlags)) != 0 ||
                       (ray.flags & cullFlags) == cullFlags;
            }))
            throw std::runtime_error("Unsupported flags for the selected DXR pipeline fixture");
    }
    std::puts(usePipeline ? "dxr_backend=dispatch_rays" : "dxr_backend=inline_ray_query");
    ComPtr<ID3D12Device5> dxr;
    check(device->QueryInterface(IID_PPV_ARGS(&dxr)), "Get DXR device");
    auto vertexBuffer = bvh_fixture::upload(device, vertices);
    auto rayBuffer = bvh_fixture::upload(device, rays);
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries;
    const auto addGeometry = [&](UINT first, UINT count, D3D12_RAYTRACING_GEOMETRY_FLAGS flags) {
        D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
        geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry.Flags = flags;
        geometry.Triangles.VertexBuffer = {vertexBuffer->GetGPUVirtualAddress() + first * sizeof(Vertex), sizeof(Vertex)};
        geometry.Triangles.VertexCount = count;
        geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geometries.push_back(geometry);
    };
    if (triangleGeometries) {
        if (update || triangleGeometries->size() * 3 != vertices.size())
            throw std::runtime_error("Per-triangle geometries need one flags entry per triangle and no update");
        for (UINT i = 0; i < triangleGeometries->size(); ++i) addGeometry(3 * i, 3, (*triangleGeometries)[i]);
    } else {
        addGeometry(0, static_cast<UINT>(vertices.size()), D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE);
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottom{}, top{};
    bottom.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottom.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottom.Inputs.NumDescs = static_cast<UINT>(geometries.size());
    bottom.Inputs.pGeometryDescs = geometries.data();
    top.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    top.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    top.Inputs.NumDescs = static_cast<UINT>(instances.size());
    if (update) {
        if (update->vertices.size() != vertices.size() || update->instances.size() != instances.size())
            throw std::runtime_error("DXR update requires unchanged fixture topology");
        bottom.Inputs.Flags = top.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottom_size{}, top_size{};
    dxr->GetRaytracingAccelerationStructurePrebuildInfo(&bottom.Inputs, &bottom_size);
    dxr->GetRaytracingAccelerationStructurePrebuildInfo(&top.Inputs, &top_size);
    if (!bottom_size.ResultDataMaxSizeInBytes || !top_size.ResultDataMaxSizeInBytes)
        throw std::runtime_error("No acceleration structure prebuild size");
    constexpr auto writable = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    constexpr auto as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    const UINT64 prefix = addressScene ? asOffset : 0;
    const UINT64 blasBytes = prefix + bottom_size.ResultDataMaxSizeInBytes;
    ComPtr<ID3D12Resource> blas, tlas, scratch;
    if (resolved) {
        const auto apply = [](const ResolvedBuild& input, const auto& measured, auto& build) {
            if (input.sizes.ResultDataMaxSizeInBytes != measured.ResultDataMaxSizeInBytes ||
                input.sizes.ScratchDataSizeInBytes != measured.ScratchDataSizeInBytes)
                throw std::runtime_error("Submitted geometry sizes differ from the prevalidated batch");
            build.DestAccelerationStructureData = input.dest;
            build.ScratchAccelerationStructureData = input.scratch;
        };
        apply(resolved->at(0), bottom_size, bottom);
        apply(resolved->at(1), top_size, top);
    } else if (placedScene) {
        placedScene->prepare(device, bottom_size, top_size, bottom, top);
    } else {
        blas = make_buffer(device, blasBytes, D3D12_HEAP_TYPE_DEFAULT, as_state, writable);
        tlas = make_buffer(device, prefix + top_size.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, as_state, writable);
        scratch = make_buffer(device, (std::max)({bottom_size.ScratchDataSizeInBytes, top_size.ScratchDataSizeInBytes,
                              bottom_size.UpdateScratchDataSizeInBytes, top_size.UpdateScratchDataSizeInBytes}),
                              D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, writable);
        bottom.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        top.DestAccelerationStructureData = tlas->GetGPUVirtualAddress() + prefix;
        bottom.ScratchAccelerationStructureData = top.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    }
    ComPtr<ID3D12Resource> otherBlas, otherVertices;
    auto otherGeometry = geometries.front();
    auto otherBuild = bottom;
    if (addressScene) {
        // Keep both geometries and BLAS alive until the GPU has completed every ray.
        otherBlas = make_buffer(device, blasBytes, D3D12_HEAP_TYPE_DEFAULT, as_state, writable);
        otherVertices = bvh_fixture::upload(device, addressScene->otherTriangle);
        otherGeometry.Triangles.VertexBuffer.StartAddress = otherVertices->GetGPUVirtualAddress();
        otherBuild.Inputs.pGeometryDescs = &otherGeometry;
        const UINT64 lowBlas = blas->GetGPUVirtualAddress() + prefix, highBlas = otherBlas->GetGPUVirtualAddress() + prefix;
        if (addressScene->route == AddressScene::Route::direct) {
            bottom.DestAccelerationStructureData = lowBlas;
            otherBuild.DestAccelerationStructureData = highBlas;
            for (auto& desc : instances) {
                const UINT64 title = desc.AccelerationStructure;
                if (title != lowTitle + prefix && title != highTitle + prefix)
                    throw std::runtime_error("Direct DXR control expects the two fixture title addresses");
                desc.AccelerationStructure = title == highTitle + prefix && !addressScene->truncate ? highBlas : lowBlas;
            }
        }
    }
    const bool decoy = placedScene && placedScene->misroute;
    if (decoy) {
        // A triangle 4 units right of the scene's, so the first ray misses whatever routes to it.
        otherVertices = bvh_fixture::upload(device, std::vector<Vertex>{{3,-1,1}, {5,-1,1}, {4,1,1}});
        otherGeometry.Triangles.VertexBuffer.StartAddress = otherVertices->GetGPUVirtualAddress();
        otherBuild.Inputs.pGeometryDescs = &otherGeometry;
        placedScene->prepare_decoy(bottom_size, otherBuild);
    }
    if (addressScene && addressScene->route != AddressScene::Route::direct) {
        era_title_va::Table ranges;
        if (!ranges.add(lowTitle, blasBytes, blas->GetGPUVirtualAddress()) ||
            !ranges.add(highTitle, blasBytes, otherBlas->GetGPUVirtualAddress()))
            throw std::runtime_error("DXR fixture could not register AS ranges");
        const UINT64 bytes = bottom_size.ResultDataMaxSizeInBytes;
        bottom.DestAccelerationStructureData = ranges.translate(lowTitle + prefix, bytes);
        otherBuild.DestAccelerationStructureData = ranges.translate(highTitle + prefix, bytes);
        if (bottom.DestAccelerationStructureData != blas->GetGPUVirtualAddress() + prefix ||
            otherBuild.DestAccelerationStructureData != otherBlas->GetGPUVirtualAddress() + prefix)
            throw std::runtime_error("DXR range lookup lost the AS allocation or its interior offset");
        for (auto& desc : instances) {
            const UINT64 title = desc.AccelerationStructure;
            UINT64 key = addressScene->truncate ? title & ((1ull << 40) - 1) : title;
            if (addressScene->route == AddressScene::Route::misroute_high && key == highTitle + prefix)
                key = lowTitle + prefix;
            desc.AccelerationStructure = ranges.translate(key, bytes);
            if (!desc.AccelerationStructure) throw std::runtime_error("Unmapped fixture BLAS");
        }
    } else if (!addressScene) {
        const UINT64 target = decoy ? otherBuild.DestAccelerationStructureData : bottom.DestAccelerationStructureData;
        for (auto& desc : instances) desc.AccelerationStructure = target;
    }
    auto instanceBuffer = bvh_fixture::upload(device, instances);
    top.Inputs.InstanceDescs = instanceBuffer->GetGPUVirtualAddress();
    const UINT result_bytes = static_cast<UINT>(rays.size() * sizeof(bvh_fixture::Row));
    const bvh_fixture::Row sentinel{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    auto initial = bvh_fixture::upload(device, std::vector<bvh_fixture::Row>(rays.size(), sentinel));
    auto results = make_buffer(device, result_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, writable);
    auto readback = make_buffer(device, result_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    // Per-triangle geometries are traced by the shader that handles non-opaque candidates.
    const bool viaView = placedScene && placedScene->slot70;
    if (viaView && (usePipeline || triangleGeometries || update))
        throw std::runtime_error("The slot-70 view fixture traces one placed scene with the inline query shader");
    const D3D12_SHADER_BYTECODE shader = triangleGeometries
        ? D3D12_SHADER_BYTECODE{ray_candidates, sizeof(ray_candidates)}
        : viaView ? D3D12_SHADER_BYTECODE{ray_query_table, sizeof(ray_query_table)}
                  : D3D12_SHADER_BYTECODE{ray_query, sizeof(ray_query)};
    ComPtr<ID3D12DescriptorHeap> asHeap;
    if (viaView) asHeap = placedScene->publish_as_view(device);
    ComPtr<ID3D12RootSignature> root;
    // Both backends share the global root signature embedded in the compute shader.
    check(device->CreateRootSignature(0, shader.pShaderBytecode, shader.BytecodeLength, IID_PPV_ARGS(&root)),
          "Create DXR fixture root");
    const auto rayCount = static_cast<UINT>(rays.size());
    RayPipeline rayPipeline;
    ComPtr<ID3D12PipelineState> compute;
    if (usePipeline) {
        const UINT opacityGeometries = triangleGeometries ? static_cast<UINT>(geometries.size()) : 0;
        rayPipeline = make_ray_pipeline(dxr.Get(), root.Get(), rayCount, opacityGeometries);
    } else {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = shader;
        check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&compute)), "Create ray query pipeline");
    }
    ComputeCommands commands(device, compute.Get());
    ComPtr<ID3D12GraphicsCommandList4> list;
    check(commands.list.As(&list), "Get DXR command list");
    transition(list.Get(), results.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(results.Get(), 0, initial.Get(), 0, result_bytes);
    transition(list.Get(), results.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (scratch) transition(list.Get(), scratch.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_RESOURCE_BARRIER order{};
    order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    // Order both the BLAS write and the reused scratch storage before building the TLAS.
    list->BuildRaytracingAccelerationStructure(&bottom, 0, nullptr);
    if (counts) ++counts->records;
    list->ResourceBarrier(1, &order);
    if (addressScene || decoy) {
        list->BuildRaytracingAccelerationStructure(&otherBuild, 0, nullptr);
        list->ResourceBarrier(1, &order);
    }
    list->BuildRaytracingAccelerationStructure(&top, 0, nullptr);
    if (counts) ++counts->records;
    ComPtr<ID3D12Resource> updatedVertices, updatedInstances;
    if (update) {
        updatedVertices = bvh_fixture::upload(device, update->vertices);
        auto descriptors = update->instances;
        for (auto& desc : descriptors) desc.AccelerationStructure = bottom.DestAccelerationStructureData;
        updatedInstances = bvh_fixture::upload(device, descriptors);
        geometries[0].Triangles.VertexBuffer.StartAddress = updatedVertices->GetGPUVirtualAddress();
        top.Inputs.InstanceDescs = updatedInstances->GetGPUVirtualAddress();
        // Keep ALLOW_UPDATE on the refit. Dropping it caused device removal on the GTX 1060.
        bottom.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        top.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        bottom.SourceAccelerationStructureData = bottom.DestAccelerationStructureData;
        top.SourceAccelerationStructureData = top.DestAccelerationStructureData;
        list->ResourceBarrier(1, &order);
        list->BuildRaytracingAccelerationStructure(&bottom, 0, nullptr);
        list->ResourceBarrier(1, &order);
        list->BuildRaytracingAccelerationStructure(&top, 0, nullptr);
    }
    order.UAV.pResource = tlas.Get();
    list->ResourceBarrier(1, &order);
    list->SetComputeRootSignature(root.Get());
    if (asHeap) {
        ID3D12DescriptorHeap* heaps[]{asHeap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootDescriptorTable(0, asHeap->GetGPUDescriptorHandleForHeapStart());
    } else {
        list->SetComputeRootShaderResourceView(0, top.DestAccelerationStructureData);
    }
    list->SetComputeRootUnorderedAccessView(1, results->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(2, rayBuffer->GetGPUVirtualAddress());
    if (usePipeline) {
        list->SetPipelineState1(rayPipeline.state.Get());
        list->DispatchRays(&rayPipeline.dispatch);
    } else {
        list->Dispatch(rayCount, 1, 1);
    }
    transition(list.Get(), results.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, results.Get(), 0, result_bytes);
    commands.submit_and_wait(device);
    if (counts) ++counts->submissions;

    return bvh_fixture::read(readback.Get(), rays.size()*4);
}

std::vector<bvh_fixture::Word> trace(ID3D12Device* device, const std::vector<Vertex>& vertices,
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances, const std::vector<Ray>& rays,
    Backend backend, const SceneUpdate* update, const std::vector<D3D12_RAYTRACING_GEOMETRY_FLAGS>* triangleGeometries) {
    return trace_scene(device, vertices, std::move(instances), rays, backend, update, triangleGeometries, nullptr);
}

}

bool check_dxr_address_ranges(ID3D12Device* device) {
    using namespace dxr_fixture;
    if (!supported(device, Backend::pipeline)) return false;
    const std::vector<Vertex> left{{-1,-1,1}, {1,-1,1}, {0,1,1}};
    AddressScene scene{{{3,-1,2}, {5,-1,2}, {4,1,2}}, false};
    auto low = instance(7, 1), high = instance(11, 2);
    low.AccelerationStructure = lowTitle + asOffset;
    high.AccelerationStructure = highTitle + asOffset;
    const std::vector<Ray> rays{{0,0,0,.01f,0,0,1,4,1}, {4,0,0,.01f,0,0,1,4,1},
                               {0,0,0,.01f,0,0,1,4,2}, {4,0,0,.01f,0,0,1,4,2}};
    const auto probe = [&](bool truncate, AddressScene::Route route) {
        scene.truncate = truncate;
        scene.route = route;
        return trace_scene(device, left, {low, high}, rays, Backend::pipeline, nullptr, nullptr, &scene);
    };
    using Route = AddressScene::Route;
    const auto full = probe(false, Route::title);
    const auto truncated = probe(true, Route::title); // Both instance masks now select the valid left BLAS.
    // The native control: the same triangles, masks, rays and transforms, with the BLAS GPU addresses given directly.
    const auto directFull = probe(false, Route::direct);
    const auto directTruncated = probe(true, Route::direct);
    // Negative control: the high instance deliberately routed to the low backing.
    const auto misrouted = probe(false, Route::misroute_high);
    // Fixed geometry expectations: hit, instance ID and primitive per ray. The distance word is compared with the
    // direct-address result below rather than with a constant, since a device may round t differently.
    using Geometry = std::array<bvh_fixture::Word, 3>;
    constexpr Geometry miss{0, 0, 0}, hitLow{1, 7, 0}, hitHigh{1, 11, 0};
    const std::array<Geometry, 4> expected{hitLow, miss, miss, hitHigh};
    const std::array<Geometry, 4> wrong{hitLow, miss, hitHigh, miss};
    const auto print = [](const char* label, const std::vector<bvh_fixture::Word>& words) {
        std::printf("dxr_address_rows %s", label);
        for (std::size_t i = 0; i + 3 < words.size(); i += 4)
            std::printf(" [%u,%u,%u,0x%08x]", words[i], words[i + 1], words[i + 2], words[i + 3]);
        std::puts("");
    };
    print("title_full", full);
    print("direct_full", directFull);
    print("title_truncated", truncated);
    print("direct_truncated", directTruncated);
    print("misroute_high", misrouted);
    const auto distance = [](const std::vector<bvh_fixture::Word>& words, std::size_t ray) { return words[4 * ray + 3]; };
    std::printf("dxr_address_distance ray0_t=0x%08x (1.0=0x3f800000) ray3_t=0x%08x (2.0=0x40000000) "
                "truncated_ray2_t=0x%08x (1.0=0x3f800000)\n",
                distance(full, 0), distance(full, 3), distance(truncated, 2));
    const auto geometry_mismatch = [](const std::array<Geometry, 4>& rows, const std::vector<bvh_fixture::Word>& words) {
        for (std::size_t i = 0; i < rows.size(); ++i)
            if (!std::equal(rows[i].begin(), rows[i].end(), words.begin() + 4 * i)) return static_cast<int>(i);
        return -1;
    };
    const auto first_difference = [&](const std::vector<bvh_fixture::Word>& a, const std::vector<bvh_fixture::Word>& b) {
        for (std::size_t i = 0; i < rays.size(); ++i)
            if (!std::equal(a.begin() + 4 * i, a.begin() + 4 * i + 4, b.begin() + 4 * i)) return static_cast<int>(i);
        return -1;
    };
    const auto fail = [](const char* what, int ray) {
        throw std::runtime_error(std::string(what) + " at ray " + std::to_string(ray));
    };
    for (const auto* set : {&full, &directFull})
        if (const int ray = geometry_mismatch(expected, *set); ray >= 0) fail("DXR address range geometry mismatch", ray);
    for (const auto* set : {&truncated, &directTruncated})
        if (const int ray = geometry_mismatch(wrong, *set); ray >= 0) fail("DXR truncation-control geometry mismatch", ray);
    // All four words, distance bits included, must equal the native direct-address result.
    if (const int ray = first_difference(full, directFull); ray >= 0)
        fail("DXR title-range result differs from the direct-address control", ray);
    if (const int ray = first_difference(truncated, directTruncated); ray >= 0)
        fail("DXR truncation result differs from the direct-address truncation control", ray);
    // The misrouted high instance must fail the same comparison, so it cannot pass by accident.
    if (first_difference(misrouted, directFull) < 0)
        throw std::runtime_error("Routing the high instance to the low backing still matched the direct control");
    std::puts("DXR address ranges: interior offset=512 title rows equal direct-address rows (full and truncated), "
              "misrouted high instance differs");
    return true;
}

std::vector<UINT32> dxr_fixture::trace_placed_triangle(ID3D12Device* device, const ResolvedBatch* resolved,
                                                       PlacedBuildScene* placement, BuildCounts* counts) {
    using namespace dxr_fixture;
    const std::vector<Vertex> vertices{{-1,-1,1}, {1,-1,1}, {0,1,1}};
    const std::vector<Ray> rays{{0,0,0,.01f,0,0,1,4,1}, {4,0,0,.01f,0,0,1,4,1}, {0,0,0,.01f,0,0,1,4,2}};
    return trace_scene(device, vertices, {instance(7,1)}, rays, Backend::pipeline,
                       nullptr, nullptr, nullptr, placement, resolved, counts);
}

bool check_dxr_placed_build(ID3D12Device* device) {
    using namespace dxr_fixture;
    if (!supported(device, Backend::pipeline)) return false;
    PlacedBuildScene placement;
    const auto output = trace_placed_triangle(device, nullptr, &placement);
    // The reference (issue #874): the same geometry, build and instance inputs, rays and readback on the same
    // adapter, with desktop GPU addresses given directly; only the placed-address resolver is bypassed.
    const auto direct = trace_placed_triangle(device, nullptr, nullptr);
    // Wrong-routing control: the instance routed to a decoy BLAS placed through the same resolver.
    PlacedBuildScene misrouted_placement;
    misrouted_placement.misroute = true;
    const auto misrouted = trace_placed_triangle(device, nullptr, &misrouted_placement);
    const auto print = [](const char* label, const std::vector<bvh_fixture::Word>& words) {
        std::printf("placed_dxr_rows %s", label);
        for (std::size_t i = 0; i + 3 < words.size(); i += 4)
            std::printf(" [%u,%u,%u,0x%08x]", words[i], words[i + 1], words[i + 2], words[i + 3]);
        std::puts("");
    };
    print("placed", output);
    print("direct", direct);
    print("misrouted", misrouted);
    // Fixed expectations: hit7, miss, masked miss with zero miss rows. The hit distance is compared with the
    // adapter's direct-address result instead of a constant and reported against the mathematical 1.0.
    const std::vector<bvh_fixture::Word> expected{1,7,0,0, 0,0,0,0, 0,0,0,0};
    const auto fixed = [&](const std::vector<bvh_fixture::Word>& words) {
        if (words.size() != expected.size()) return false;
        for (std::size_t i = 0; i < words.size(); ++i)
            if (i != 3 && words[i] != expected[i]) return false;
        return true;
    };
    if (!fixed(output) || !fixed(direct))
        throw std::runtime_error("Placed BLAS/TLAS ray output differs from hit7, miss, masked miss");
    std::printf("placed_dxr_distance placed=0x%08x direct=0x%08x (1.0=0x3f800000)\n", output[3], direct[3]);
    if (output != direct)
        throw std::runtime_error("Placed BLAS/TLAS rays differ from the direct-address reference");
    if (misrouted == direct)
        throw std::runtime_error("Routing the instance to the decoy BLAS still matched the direct reference");
    std::printf("Slot 133 placed BLAS/TLAS: hit7 (distance=0x%08x), miss, masked miss; all 12 words equal the "
                "direct-address reference, decoy routing differs\n", output[3]);
    return true;
}

// Device slot 70 (issue #1938): the TLAS is placed at a nonzero offset, published as an acceleration-structure
// view through slot 70 and consumed from a descriptor table by an inline ray query, on the same adapter.
bool check_dxr_slot70_as(ID3D12Device* device) {
    using namespace dxr_fixture;
    if (!supported(device, Backend::inline_query)) return false;
    const std::vector<Vertex> vertices{{-1,-1,1}, {1,-1,1}, {0,1,1}};
    const std::vector<Ray> rays{{0,0,0,.01f,0,0,1,4,1}, {4,0,0,.01f,0,0,1,4,1}, {0,0,0,.01f,0,0,1,4,2}};
    const auto run = [&](bool misroute) {
        PlacedBuildScene placement;
        placement.slot70 = true;
        placement.misroute = misroute;
        return trace_scene(device, vertices, {instance(7,1)}, rays, Backend::inline_query,
                           nullptr, nullptr, nullptr, &placement);
    };
    const auto viaView = run(false);
    // The reference: the same builds and rays with the TLAS GPU address given as a root SRV; only slot 70 is bypassed.
    const auto root = trace(device, vertices, {instance(7,1)}, rays);
    // Distinct-AS control: the view names a TLAS whose instance routes to a decoy BLAS.
    const auto decoy = run(true);
    const auto print = [](const char* label, const std::vector<bvh_fixture::Word>& words) {
        std::printf("slot70_as_rows %s", label);
        for (std::size_t i = 0; i + 3 < words.size(); i += 4)
            std::printf(" [%u,%u,%u,0x%08x]", words[i], words[i + 1], words[i + 2], words[i + 3]);
        std::puts("");
    };
    print("view", viaView);
    print("root", root);
    print("decoy", decoy);
    const std::vector<bvh_fixture::Word> expected{1,7,0,0, 0,0,0,0, 0,0,0,0};
    const auto fixed = [&](const std::vector<bvh_fixture::Word>& words) {
        if (words.size() != expected.size()) return false;
        for (std::size_t i = 0; i < words.size(); ++i)
            if (i != 3 && words[i] != expected[i]) return false;
        return true;
    };
    if (!fixed(viaView) || !fixed(root))
        throw std::runtime_error("Slot-70 view rays differ from hit7, miss, masked miss");
    if (viaView != root)
        throw std::runtime_error("Slot-70 view rays differ from the root-address reference");
    if (decoy == root) throw std::runtime_error("The decoy-routed view still matched the reference");
    std::printf("Slot 70 acceleration-structure view: hit7 (distance=0x%08x), miss, masked miss; all 12 words equal the "
                "root-address reference after three refused calls kept the descriptor and the decoy differs\n", viaView[3]);
    return true;
}

bool check_dxr_rays(ID3D12Device* device, bool usePipeline) {
    using namespace dxr_fixture;
    const auto backend = usePipeline ? Backend::pipeline : Backend::inline_query;
    if (!supported(device, backend)) return false;
    const std::vector<Vertex> vertices{{-1,-1,1}, {1,-1,1}, {0,1,1}};
    std::vector<Ray> rays;
    for (UINT i = 0; i < 5; ++i)
        rays.push_back({i >= 3 ? 3.f : (i == 1 ? 1.5f : 0.f), 0, 0, .01f, 0, 0, 1,
                        i == 2 ? .5f : 5.f, i == 3 ? 2u : 1u});
    auto words = trace(device, vertices, {instance(7, 1), instance(11, 2, 3)}, rays, backend);
    constexpr UINT expected_id[]{7, 0, 0, 11, 0};
    UINT mismatches = 0;
    for (UINT i = 0; i < 5; ++i) {
        const UINT* value = words.data() + 4 * i;
        float distance;
        std::memcpy(&distance, value + 3, sizeof(distance));
        const bool hit = expected_id[i] != 0;
        const bool distance_matches = hit ? std::abs(distance - 1.f) < 1e-5f : value[3] == 0;
        if (value[0] != static_cast<UINT>(hit) || value[1] != expected_id[i] || value[2] != 0 || !distance_matches)
            ++mismatches;
        std::printf("ray=%u hit=%u instance=%u primitive=%u distance=%g\n", i, value[0], value[1], value[2], distance);
    }
    if (mismatches) throw std::runtime_error("DXR results disagree with triangle, distance or mask reference");
    return true;
}

bool check_dxr_update(ID3D12Device* device, bool usePipeline) {
    using namespace dxr_fixture;
    const auto backend = usePipeline ? Backend::pipeline : Backend::inline_query;
    if (!supported(device, backend)) return false;
    const std::vector<Vertex> initial{{-1,-1,1}, {1,-1,1}, {0,1,1}};
    const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances{instance(7,1), instance(11,2,4)};
    const SceneUpdate update{{{-1,-1,3}, {1,-1,3}, {0,1,3}}, {instance(7,1,2), instance(11,4,-3)}};
    std::vector<Ray> rays;
    struct Probe { float x; UINT mask; };
    const Probe probes[]{{0,1}, {4,2}, {2,1}, {-3,4}, {-3,2}, {2,4}};
    for (const auto& probe : probes)
        rays.push_back({probe.x,0,0,0.01f,0,0,1,8,probe.mask});
    rays.push_back({2,0,0,0.01f,0,0,1,2,1});
    const auto before = trace(device, initial, instances, rays, backend);
    const auto after = trace(device, initial, instances, rays, backend, &update);
    const auto rebuilt = trace(device, update.vertices, update.instances, rays, backend);
    const auto mismatch = std::mismatch(after.begin(), after.end(), rebuilt.begin());
    if (mismatch.first != after.end()) {
        const auto word = static_cast<UINT>(mismatch.first - after.begin());
        std::fprintf(stderr, "ray=%u field=%u rebuilt=%08x updated=%08x\n",
                     word / 4, word % 4, *mismatch.second, *mismatch.first);
        throw std::runtime_error("In-place update differs from a fresh DXR build");
    }
    const UINT idsBefore[]{7,11,0,0,0,0,0}, idsAfter[]{0,0,7,11,0,0,0};
    for (UINT i = 0; i < rays.size(); ++i) {
        for (bool updated : {false, true}) {
            const auto& values = updated ? after : before;
            const UINT id = updated ? idsAfter[i] : idsBefore[i];
            const float distance = id ? (updated ? 3.f : 1.f) : 0.f;
            UINT distanceBits;
            std::memcpy(&distanceBits, &distance, sizeof(distanceBits));
            const bvh_fixture::Row expected{static_cast<UINT>(id != 0), id, 0, distanceBits};
            for (UINT field = 0; field < 4; ++field) {
                if (field == 3 && id != 0) {
                    float actualDistance;
                    std::memcpy(&actualDistance, &values[4 * i + field], sizeof(actualDistance));
                    if (std::abs(actualDistance - distance) < 1e-4f) continue;
                }
                bvh_fixture::equal(values[4 * i + field], expected[field], i, field);
            }
        }
    }
    const auto checkStale = [&](const char* name, const std::vector<bvh_fixture::Word>& stale, UINT expectedMask) {
        UINT mismatchMask = 0;
        for (UINT i = 0; i < rays.size(); ++i) {
            const auto row = stale.begin() + 4 * i;
            if (!std::equal(row, row + 4, after.begin() + 4 * i))
                mismatchMask |= 1u << i;
        }
        std::printf("dxr_update control=%s mismatch_ray_mask=0x%02x expected=0x%02x\n",
                    name, mismatchMask, expectedMask);
        if (mismatchMask != expectedMask)
            throw std::runtime_error("DXR update stale-result control differs from expected rays");
    };
    // Build valid controls with old inputs. Tracing an old TLAS after changing its BLAS is invalid.
    checkStale("stale_scene", before, 0x0f); // Rays 0-3 change hit status.
    checkStale("stale_vertices", trace(device, initial, update.instances, rays, backend),
               0x4c); // Rays 2 and 3 change distance, and ray 6 hits before clipping.
    checkStale("stale_instances", trace(device, update.vertices, instances, rays, backend),
               0x0f); // Rays 0-3 change hit status.
    std::puts("dxr_update rays=7 changed_hits=4 in_place_matches_rebuild=1");
    return true;
}
