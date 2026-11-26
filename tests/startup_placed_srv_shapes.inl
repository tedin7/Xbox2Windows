// Uses the retire suite's native queue and readback to test the descriptor published by guest slot 70.
bool case_placed_srv_shape(Context& c, const char* shape, bool enabled, void* texture_memory, void* buffer_memory)
{
    const bool typed = !std::strcmp(shape, "typed");
    const bool structured = !std::strcmp(shape, "structured");
    const bool raw = !std::strcmp(shape, "raw");
    const bool buffer = typed || structured || raw;
    const bool volume = !std::strcmp(shape, "volume");
    const bool array = !std::strcmp(shape, "array");
    const bool cube = !std::strcmp(shape, "cube");
    const bool cube_array = !std::strcmp(shape, "cube_array");
    if (!buffer && !volume && !array && !cube && !cube_array && std::strcmp(shape, "mip")) return false;
    const UINT16 depth = volume ? 4 : cube_array ? 12 : cube ? 6 : array ? 3 : 1;
    D3D12_RESOURCE_DESC resource{volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        0, 2, 2, depth, 2, volume ? DXGI_FORMAT_R8_TYPELESS : DXGI_FORMAT_R8_UNORM,
        {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    MemorySrvInput input{};
    auto& view = input.view;
    view.Format = DXGI_FORMAT_R8_UNORM;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Texture2D = {1, 1, 0, 0};
    const char* declaration = "Texture2D<float> source : register(t0);";
    const char* expression = "round(source.Load(int3(0,0,0))*255)";
    const auto texel = [](UINT subresource, UINT z) { return 17u + 5u * subresource + z; };
    UINT reference = texel(1, 0);
    if (array) {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray = {1, 1, 2, 1, 0, 0};
        declaration = "Texture2DArray<float> source : register(t0);";
        expression = "round(source.Load(int4(0,0,0,0))*255)";
        reference = texel(5, 0);
    } else if (cube) {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        view.TextureCube = {1, 1, 0};
        declaration = "TextureCube<float> source : register(t0);";
        expression = "round(source.SampleLevel(pointSampler, float3(-1,0,0), 0)*255)";
        reference = texel(3, 0);  // -X, the second face
    } else if (cube_array) {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        view.TextureCubeArray = {1, 1, 6, 1, 0};
        declaration = "TextureCubeArray<float> source : register(t0);";
        expression = "round(source.SampleLevel(pointSampler, float4(-1,0,0,0), 0)*255)";
        reference = texel(15, 0);  // -X of the second cube
    } else if (volume) {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        view.Texture3D = {1, 1, 0};
        declaration = "Texture3D<float> source : register(t0);";
        expression = "round(source.Load(int4(0,0,1,0))*255)";
        reference = texel(1, 1);
    } else if (buffer) {
        resource = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, 128, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                    {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        view.Format = typed ? DXGI_FORMAT_R32_UINT : raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        view.Buffer = {raw ? 4u : 1u, 1, structured ? 12u : 0u,
                       raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE};
        declaration = typed ? "Buffer<uint> source : register(t0);" :
                      raw ? "ByteAddressBuffer source : register(t0);" :
                            "StructuredBuffer<uint3> source : register(t0);";
        expression = typed ? "source[0]" : raw ? "source.Load(0)" : "source[0].x";
        auto* window = static_cast<unsigned char*>(buffer_memory) + 32;
        input.address = reinterpret_cast<UINT64>(window);
        reference = 0x13579bdf;
        std::memcpy(window + (raw ? 16u : structured ? 12u : 4u), &reference, sizeof(reference));
    }

    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &range};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3D12RootSignature> root;
    if (!require(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "SRV root blob") ||
        !require(c.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "SRV root"))
        return false;
    const std::string source = std::string(declaration) +
        "SamplerState pointSampler : register(s0);"
        "RWByteAddressBuffer result : register(u0);"
        "[numthreads(1,1,1)] void main() { result.Store(0, (uint)(" + expression + ")); }";
    const auto shader = compile_stage(source.c_str(), "main", "cs_5_1");
    if (!shader) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.Get();
    pso_desc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pipeline;
    if (!require(c.device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline)), "SRV PSO")) return false;

    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 address, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    ComPtr<IUnknown> guest_heap, guest_texture;
    if (!require(invoke(c.guest, 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), guest_heap.GetAddressOf()), "SRV heap"))
        return false;
    const auto heap = desktop_object<ID3D12DescriptorHeap>(c.guest, c.device, guest_heap.Get());
    if (!heap) return false;
    // Prime with a dimension-compatible null SRV. A refusal must leave that descriptor intact.
    auto null_view = view;
    if (buffer) null_view.Buffer.FirstElement = 0;
    else if (cube_array) {
        null_view.TextureCubeArray.MostDetailedMip = 0;
        null_view.TextureCubeArray.First2DArrayFace = 0;
    } else if (cube) null_view.TextureCube.MostDetailedMip = 0;
    else if (volume) null_view.Texture3D.MostDetailedMip = 0;
    else if (array) {
        null_view.Texture2DArray.MostDetailedMip = 0;
        null_view.Texture2DArray.FirstArraySlice = 0;
    } else null_view.Texture2D.MostDetailedMip = 0;
    c.device->CreateShaderResourceView(nullptr, &null_view, heap->GetCPUDescriptorHandleForHeapStart());

    if (!require(c.own_allocator->Reset(), "SRV allocator") ||
        !require(c.own_list->Reset(c.own_allocator.Get(), pipeline.Get()), "SRV list")) return false;
    const auto transition = [&](ID3D12Resource* target, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
        barrier.Transition = {target, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        c.own_list->ResourceBarrier(1, &barrier);
    };
    ComPtr<ID3D12Resource> texture;
    if (!buffer) {
        input.address = reinterpret_cast<UINT64>(texture_memory);
        if (!require(invoke(c.guest, 133, texture_memory, &resource, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &__uuidof(ID3D12Resource), guest_texture.GetAddressOf()), "placed SRV texture"))
            return false;
        texture = desktop_object<ID3D12Resource>(c.guest, c.device, guest_texture.Get());
        if (!texture) return false;
        const UINT count = resource.MipLevels * (volume ? 1u : depth);
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(count);
        UINT64 bytes = 0;
        c.device->GetCopyableFootprints(&resource, 0, count, 0, footprints.data(), nullptr, nullptr, &bytes);
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        if (bytes > image_bytes || !require(c.upload->Map(0, &none, &mapped), "SRV texture upload")) return false;
        transition(texture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        for (UINT i = 0; i < count; ++i) {
            const auto& fp = footprints[i];
            for (UINT z = 0; z < fp.Footprint.Depth; ++z)
                for (UINT y = 0; y < fp.Footprint.Height; ++y)
                    std::memset(static_cast<unsigned char*>(mapped) + fp.Offset +
                        (z * fp.Footprint.Height + y) * fp.Footprint.RowPitch, texel(i, z), fp.Footprint.Width);
            D3D12_TEXTURE_COPY_LOCATION dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            dst.SubresourceIndex = i;
            D3D12_TEXTURE_COPY_LOCATION src{c.upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            src.PlacedFootprint = fp;
            c.own_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        c.upload->Unmap(0, nullptr);
        transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    const auto cpu = invoke<UINT64>(guest_heap.Get(), 9, UINT64{0});
    invoke<void>(c.guest, 70, &resource, &input, cpu);

    const D3D12_HEAP_PROPERTIES local{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC output_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(UINT), 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    ComPtr<ID3D12Resource> output;
    if (!require(c.device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &output_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&output)), "SRV result buffer")) return false;
    ID3D12DescriptorHeap* heaps[]{heap.Get()};
    c.own_list->SetDescriptorHeaps(1, heaps);
    c.own_list->SetComputeRootSignature(root.Get());
    c.own_list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    c.own_list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
    c.own_list->Dispatch(1, 1, 1);
    transition(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    c.own_list->CopyBufferRegion(c.readback.Get(), 0, output.Get(), 0, sizeof(UINT));
    if (!c.own_run()) return false;
    void* mapped = nullptr;
    const D3D12_RANGE result{0, sizeof(UINT)}, none{};
    if (!require(c.readback->Map(0, &result, &mapped), "SRV result map")) return false;
    const UINT actual = *static_cast<const UINT*>(mapped);
    c.readback->Unmap(0, &none);
    const UINT expected = enabled ? reference : 0;
    std::printf("%s SRV %s: GPU=%08x CPU=%08x\n", shape, enabled ? "enabled" : "off", actual, expected);
    return actual == expected && SUCCEEDED(c.device->GetDeviceRemovedReason());
}
