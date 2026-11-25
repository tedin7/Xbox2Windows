// Experimental (docs/startup-runtime.md): shows the presented surface in the title's window.
// Included inside d3d12_xs_device.cpp's anonymous namespace, before queue_present.
//
// A flip-model swap chain on the title window, created on the queue the title presents with; each
// present draws the presented surface into the back buffer with a full-window triangle (the surface
// is usually larger than the window) and flips. Diagnostics go to the window title, because GDI
// drawing does not mix with a flip-model swap chain.

struct WindowPresenter {
    IDXGISwapChain3* chain = nullptr;
    ID3D12DescriptorHeap* targets = nullptr;
    ID3D12DescriptorHeap* textures = nullptr;
    ID3D12Resource* buffers[2] = {};
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pipeline = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE reached = nullptr;
    UINT64 fence_value = 0;
    UINT width = 0, height = 0;
    bool failed = false;
};
WindowPresenter window_presenter;
LONG window_presents;

// The output window belongs to this layer and has its own thread and message loop; the title's
// window is left alone (a swap chain on it made the title stop at one of its own fatal checks).
// With X2W_HIDE_TITLE_WINDOW=1, a timer on this thread hides the title's own window while frames
// are shown here (x2w_startup::TitleWindowHider, docs/startup-runtime.md#switches).
HWND output_handle;
LONG last_frame_shown;  // whether the latest present reached this window (set by queue_present)
x2w_startup::TitleWindowHider title_hider;  // used on the output window's thread only
DWORD WINAPI output_window_thread(void* ready)
{
    WNDCLASSW cls{};
    cls.lpfnWndProc = [](HWND window, UINT message, WPARAM w, LPARAM l) -> LRESULT {
        if (message == WM_CLOSE) {
            ShowWindow(window, SW_MINIMIZE);
            return 0;
        }
        if (message == WM_TIMER && w == x2w_startup::hide_title_timer) {
            bool refused = false;
            const int changed = title_hider.step(window, InterlockedCompareExchange(&window_presents, 0, 0),
                                                 InterlockedCompareExchange(&last_frame_shown, 0, 0), &refused);
            // These flags are shared by every output window, so each line is logged once per process,
            // not once per window. That is fine while there is one output window.
            static bool told_refused, told_hidden, told_shown;
            if (refused && !told_refused) {
                told_refused = true;  // keyboard input reaches the title only while this process is in front
                log("x2w-present the output window could not take the foreground from the title's window");
            }
            if (changed > 0 && !told_hidden) {
                told_hidden = true;
                log("x2w-present X2W_HIDE_TITLE_WINDOW: hid %d title window(s)", changed);
            }
            if (changed < 0 && !told_shown) {
                told_shown = true;
                log("x2w-present X2W_HIDE_TITLE_WINDOW: frames no longer shown, %d title window(s) shown again", -changed);
            }
            return 0;
        }
        return DefWindowProcW(window, message, w, l);
    };
    cls.hInstance = GetModuleHandleW(nullptr);
    cls.lpszClassName = x2w_startup::presenter_class;
    cls.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    RegisterClassW(&cls);
    RECT area{0, 0, 1600, 900};
    AdjustWindowRect(&area, WS_OVERLAPPEDWINDOW, FALSE);
    wchar_t title[80];  // 22 fixed characters and the revision (at most 46)
    swprintf(title, 80, L"Xbox2Windows %ls - output", x2w_build_rev_w);
    output_handle = CreateWindowExW(0, cls.lpszClassName, title, WS_OVERLAPPEDWINDOW, 60, 60,
                                    area.right - area.left, area.bottom - area.top, nullptr, nullptr, cls.hInstance, nullptr);
    if (output_handle) ShowWindow(output_handle, SW_SHOWNOACTIVATE);
    if (output_handle) x2w_startup::TitleWindowHider::start(output_handle);
    SetEvent(static_cast<HANDLE>(ready));
    MSG message;
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return 0;
}
HWND output_window()
{
    static bool started = false;
    if (!started) {
        started = true;
        HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (HANDLE thread = CreateThread(nullptr, 0, output_window_thread, ready, 0, nullptr)) {
            WaitForSingleObject(ready, 5000);
            CloseHandle(thread);
        }
        CloseHandle(ready);
    }
    return output_handle;
}

bool create_window_presenter(ID3D12CommandQueue* queue, HWND window)
{
    WindowPresenter& p = window_presenter;
    ID3D12Device* device = device_of(the_device);
    RECT client{};
    GetClientRect(window, &client);
    p.width = client.right > 0 ? client.right : 1280;
    p.height = client.bottom > 0 ? client.bottom : 720;
    IDXGIFactory4* factory = nullptr;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    IDXGISwapChain1* chain1 = nullptr;
    if (SUCCEEDED(hr)) {
        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = p.width;
        desc.Height = p.height;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 2;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        hr = factory->CreateSwapChainForHwnd(queue, window, &desc, nullptr, nullptr, &chain1);
        if (SUCCEEDED(hr)) factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
    }
    if (SUCCEEDED(hr)) hr = chain1->QueryInterface(IID_PPV_ARGS(&p.chain));
    if (chain1) chain1->Release();
    if (factory) factory->Release();
    const D3D12_DESCRIPTOR_HEAP_DESC target_heap{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC texture_heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (SUCCEEDED(hr)) hr = device->CreateDescriptorHeap(&target_heap, IID_PPV_ARGS(&p.targets));
    if (SUCCEEDED(hr)) hr = device->CreateDescriptorHeap(&texture_heap, IID_PPV_ARGS(&p.textures));
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (UINT i = 0; SUCCEEDED(hr) && i < 2; ++i) {
        hr = p.chain->GetBuffer(i, IID_PPV_ARGS(&p.buffers[i]));
        if (SUCCEEDED(hr))
            device->CreateRenderTargetView(p.buffers[i], nullptr, D3D12_CPU_DESCRIPTOR_HANDLE{p.targets->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{i} * step});
    }
    if (SUCCEEDED(hr)) hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&p.allocator));
    if (SUCCEEDED(hr)) hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, p.allocator, nullptr, IID_PPV_ARGS(&p.list));
    if (SUCCEEDED(hr)) hr = p.list->Close();
    if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&p.fence));
    p.reached = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    static const char source[] =
        "Texture2D<float4> surface : register(t0); SamplerState smooth : register(s0);\n"
        "struct V { float4 position : SV_Position; float2 uv : TEXCOORD0; };\n"
        "V vs(uint id : SV_VertexID) { V o; o.uv = float2((id << 1) & 2, id & 2); o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1); return o; }\n"
        "float4 ps(V i) : SV_Target0 { return float4(surface.Sample(smooth, i.uv).rgb, 1); }\n";
    ID3DBlob* vs = nullptr;
    ID3DBlob* ps = nullptr;
    ID3DBlob* blob = nullptr;
    if (SUCCEEDED(hr)) hr = D3DCompile(source, sizeof(source) - 1, "present", nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vs, nullptr);
    if (SUCCEEDED(hr)) hr = D3DCompile(source, sizeof(source) - 1, "present", nullptr, nullptr, "ps", "ps_5_0", 0, 0, &ps, nullptr);
    if (SUCCEEDED(hr)) {
        const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {1, &range};
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        const D3D12_ROOT_SIGNATURE_DESC desc{1, &parameter, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, nullptr);
        if (SUCCEEDED(hr)) hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&p.root));
    }
    if (SUCCEEDED(hr)) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = p.root;
        desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
        desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.SampleMask = ~0u;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&p.pipeline));
    }
    if (vs) vs->Release();
    if (ps) ps->Release();
    if (blob) blob->Release();
    log("x2w-present window swap chain %ux%u -> 0x%08lx", p.width, p.height, static_cast<unsigned long>(hr));
    check_device("window presenter", 0);
    return SUCCEEDED(hr);
}

// Draws `surface` (resting in COMMON) into the window and flips. Returns false if not shown.
bool present_to_window(ID3D12CommandQueue* queue, ID3D12Resource* surface)
{
    X2W_ZONE("window present (blit + fence wait)");
    WindowPresenter& p = window_presenter;
    if (p.failed || !queue || !surface) return false;
    const D3D12_RESOURCE_DESC desc = surface->GetDesc();
    if (!x2w_startup::present_surface_supported(desc)) {
        static LONG refused;
        if (first_calls(refused, 4))
            log("x2w-present surface dim=%d samples=%u flags=0x%x format=%d not supported (present_view_format.h)",
                desc.Dimension, desc.SampleDesc.Count, desc.Flags, desc.Format);
        return false;
    }
    HWND window = output_window();
    if (!window) return false;
    if (!p.chain && !create_window_presenter(queue, window)) {
        p.failed = true;
        return false;
    }
    ID3D12Device* device = device_of(the_device);
    const DXGI_FORMAT view_format = x2w_startup::present_view_format(desc.Format);
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = view_format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(surface, &view, p.textures->GetCPUDescriptorHandleForHeapStart());
    const UINT index = p.chain->GetCurrentBackBufferIndex();
    const D3D12_CPU_DESCRIPTOR_HANDLE target{p.targets->GetCPUDescriptorHandleForHeapStart().ptr +
                                             SIZE_T{index} * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator, p.pipeline))) return false;
    transition(p.list, surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    transition(p.list, p.buffers[index], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(p.width), static_cast<float>(p.height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(p.width), static_cast<LONG>(p.height)};
    p.list->SetGraphicsRootSignature(p.root);
    p.list->SetDescriptorHeaps(1, &p.textures);
    p.list->SetGraphicsRootDescriptorTable(0, p.textures->GetGPUDescriptorHandleForHeapStart());
    p.list->RSSetViewports(1, &viewport);
    p.list->RSSetScissorRects(1, &scissor);
    p.list->OMSetRenderTargets(1, &target, FALSE, nullptr);
    p.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    p.list->DrawInstanced(3, 1, 0, 0);
    transition(p.list, p.buffers[index], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    transition(p.list, surface, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (FAILED(p.list->Close())) {
        p.failed = true;
        return false;
    }
    ID3D12CommandList* lists[] = {p.list};
    queue->ExecuteCommandLists(1, lists);
    const HRESULT hr = p.chain->Present(0, 0);
    // The allocator is reused next frame: wait until this frame's commands are done.
    if (SUCCEEDED(queue->Signal(p.fence, ++p.fence_value)) && p.fence->GetCompletedValue() < p.fence_value &&
        SUCCEEDED(p.fence->SetEventOnCompletion(p.fence_value, p.reached)))
        WaitForSingleObject(p.reached, 2000);
    static LONG failures;
    if (FAILED(hr) && first_calls(failures, 4)) log("x2w-present Present failed 0x%08lx", static_cast<unsigned long>(hr));
    check_device("window present", 0);
    if (SUCCEEDED(hr)) InterlockedIncrement(&window_presents);
    return SUCCEEDED(hr);
}
