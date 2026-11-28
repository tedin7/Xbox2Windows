// Pass thermometer scene (tests/startup_pass_stats_check.py, docs/startup-runtime.md#pass-thermometer). Included by
// startup_stream_consumer.cpp after create_slot67, whose slot 67 pipeline is the check's exposure program: each
// thread loads the texel of texture A (table byte 0x80) at its position, multiplies the four channels by root
// constant 0 (the exposure) and stores them to texture T (table byte 0x40). One recording on 32x8 RGBA8 textures:
//   pass A: ClearRenderTargetView of A to (0.25, 0.5, 0.75, 1);
//   pass B: Dispatch(8, 4, 1) of the exposure program, A to T;
//   pass C: CopyResource of T to S, the surface the queue then presents.
// The recording also copies A, T and S to a readback buffer; X2W_TEST_DISPATCH_OUT receives their 32x8 rows, in
// that order. pass-stats: exposure 0; pass-stats-bright: exposure 1.
static int pass_stats_slot67(void* device, const std::string& mode)
{
    constexpr UINT width = 32, height = 8, pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, image_bytes = pitch * height;
    constexpr DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    const auto word = [](const void* value) { return reinterpret_cast<StreamWord>(value); };
    const float exposure = mode == "pass-stats-bright" ? 1.0f : 0.0f;
    std::uint32_t exposure_bits = 0;
    std::memcpy(&exposure_bits, &exposure, sizeof(exposure_bits));
    const auto texture = [&](D3D12_RESOURCE_FLAGS flags) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_UNKNOWN, flags};
        return make_committed(device, D3D12_HEAP_TYPE_DEFAULT, desc, D3D12_RESOURCE_STATE_COMMON);
    };
    void* const source = texture(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    void* const exposed = texture(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    void* const presented = texture(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 3 * image_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                          {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    void* const readback = make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!source || !exposed || !presented || !readback) return 71;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } view_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0},
      rtv_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    void *views = nullptr, *rtvs = nullptr, *queue = nullptr, *allocator = nullptr, *list = nullptr, *fence = nullptr;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, word(&view_desc), word(&__uuidof(ID3D12DescriptorHeap)), word(&views)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 14, word(&rtv_desc), word(&__uuidof(ID3D12DescriptorHeap)), word(&rtvs)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 8, word(&queue_desc), word(&__uuidof(ID3D12CommandQueue)), word(&queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, word(&__uuidof(ID3D12CommandAllocator)), word(&allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 73, word(list_desc), word(allocator), 0, word(&__uuidof(ID3D12GraphicsCommandList)), word(&list)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 36, 0, 0, word(&__uuidof(ID3D12Fence)), word(&fence))))) return 72;
    const StreamWord cpu = invoke(views, 9), gpu = invoke(views, 10), rtv = invoke(rtvs, 9);
    if (!cpu || !gpu || !rtv) return 73;
    const D3D12_UNORDERED_ACCESS_VIEW_DESC write_view{format, D3D12_UAV_DIMENSION_TEXTURE2D};
    D3D12_SHADER_RESOURCE_VIEW_DESC read_view{format, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
    read_view.Texture2D.MipLevels = 1;
    const D3D12_RENDER_TARGET_VIEW_DESC target_view{format, D3D12_RTV_DIMENSION_TEXTURE2D};
    invoke(device, 19, word(exposed), 0, word(&write_view), cpu + 0x40);
    invoke(device, 18, word(source), word(&read_view), cpu + 0x80);
    invoke(device, 20, word(source), word(&target_view), rtv);

    if (FAILED(static_cast<HRESULT>(invoke(list, 139, word(allocator))))) return 74;
    const float colour[]{0.25f, 0.5f, 0.75f, 1.0f};
    invoke(list, 52, rtv, word(colour));
    invoke(list, 32, 1, word(&views));
    invoke(list, 25, word(slot67_pipeline));
    invoke(list, 33, word(slot67_root));
    const UINT constants[3]{exposure_bits, 0, 0};
    invoke(list, 39, 0, 3, word(constants), 0);
    invoke(list, 35, 1, gpu);
    invoke(list, 14, width / 4, height / 2, 1);
    invoke(list, 17, word(presented), word(exposed));
    void* const images[]{source, exposed, presented};
    for (UINT i = 0; i < 3; ++i) {
        D3D12_TEXTURE_COPY_LOCATION image{static_cast<ID3D12Resource*>(images[i]), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION rows{static_cast<ID3D12Resource*>(readback), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        rows.PlacedFootprint = {UINT64{i} * image_bytes, {format, width, height, 1, pitch}};
        invoke(list, 16, word(&rows), 0, 0, 0, word(&image), 0);
    }
    if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 75;
    invoke(queue, 10, 1, word(&list));
    // Plane parameters: +0x00 frame token, +0x08 resource count, +0x10 pointer to the resources.
    static std::uint64_t resources[1], plane[9], present[8];
    resources[0] = word(presented);
    plane[1] = 1;
    plane[2] = word(resources);
    if (FAILED(static_cast<HRESULT>(invoke(queue, 32, 1, word(plane), word(present))))) return 81;
    if (FAILED(static_cast<HRESULT>(invoke(queue, 14, word(fence), 1)))) return 76;
    for (const ULONGLONG deadline = GetTickCount64() + 10000; invoke(fence, 8) < 1;) {
        if (GetTickCount64() >= deadline) return 78;
        Sleep(1);
    }
    char path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("X2W_TEST_DISPATCH_OUT", path, MAX_PATH);
    void* memory = nullptr;
    if (!n || n >= MAX_PATH || FAILED(static_cast<HRESULT>(invoke(readback, 8, 0, 0, word(&memory)))) || !memory) return 2;
    std::ofstream output(path, std::ios::binary);
    for (UINT i = 0; i < 3; ++i)
        for (UINT y = 0; y < height; ++y)
            output.write(static_cast<const char*>(memory) + i * image_bytes + y * pitch, width * 4);
    invoke(readback, 9, 0, 0);
    return output ? 0 : 2;
}
