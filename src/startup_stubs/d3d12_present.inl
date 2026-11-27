// Experimental (docs/startup-runtime.md): shows the presented surface in the title's window.
// Included inside d3d12_xs_device.cpp's anonymous namespace, before queue_present.
//
// A flip-model swap chain on the title window, created on the queue the title presents with; each
// present draws its planes into the back buffer, bottom to top, with the plane compositor
// (plane_compositor.h: one full-window triangle a plane; a surface is usually larger than the
// window), and flips. Plane 0 alone by default; every plane with X2W_PRESENT_COMPOSITE=1. Diagnostics go to the window title, because GDI drawing does not mix with a
// flip-model swap chain.

struct WindowPresenter {
    IDXGISwapChain3* chain = nullptr;
    ID3D12DescriptorHeap* targets = nullptr;
    ID3D12Resource* buffers[2] = {};
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    x2w_startup::PlaneCompositor compositor;  // shaders, pipelines and plane views
    ID3D12Fence* fence = nullptr;
    HANDLE reached = nullptr;
    UINT64 fence_value = 0;
    UINT width = 0, height = 0;
    bool failed = false;
};
constexpr UINT window_planes = 4;
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

bool create_window_presenter(ID3D12CommandQueue* queue, HWND window, UINT plane_capacity)
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
    if (SUCCEEDED(hr)) hr = device->CreateDescriptorHeap(&target_heap, IID_PPV_ARGS(&p.targets));
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
    if (SUCCEEDED(hr)) hr = p.compositor.create(device, DXGI_FORMAT_R8G8B8A8_UNORM, plane_capacity);
    log("x2w-present window swap chain %ux%u -> 0x%08lx", p.width, p.height, static_cast<unsigned long>(hr));
    check_device("window presenter", 0);
    return SUCCEEDED(hr);
}

// Whether the window presenter can draw `surface` (present_view_format.h); logs the first refusals.
bool presentable_surface(ID3D12Resource* surface)
{
    if (!surface) return false;
    const D3D12_RESOURCE_DESC desc = surface->GetDesc();
    if (x2w_startup::present_surface_supported(desc)) return true;
    static LONG refused;
    if (first_calls(refused, 4))
        log("x2w-present surface dim=%d samples=%u flags=0x%x format=%d not supported (present_view_format.h)",
            desc.Dimension, desc.SampleDesc.Count, desc.Flags, desc.Format);
    return false;
}

// Composites the steps of a present (present_planes.h composite_steps) into the window and flips.
// plane_surfaces[step.plane] is that plane's surface (resting in COMMON and presentable), or null to
// leave it out. Returns false if not shown.
bool present_to_window(ID3D12CommandQueue* queue, const x2w_startup::PlaneStep* steps, unsigned count,
                       ID3D12Resource* const* plane_surfaces, UINT plane_capacity)
{
    X2W_ZONE("window present (blit + fence wait)");
    WindowPresenter& p = window_presenter;
    if (p.failed || !queue || !steps || !count || !plane_surfaces) return false;
    HWND window = output_window();
    if (!window) return false;
    if (!p.chain && !create_window_presenter(queue, window, plane_capacity)) {
        p.failed = true;
        return false;
    }
    ID3D12Device* device = device_of(the_device);
    const UINT index = p.chain->GetCurrentBackBufferIndex();
    const D3D12_CPU_DESCRIPTOR_HANDLE target{p.targets->GetCPUDescriptorHandleForHeapStart().ptr +
                                             SIZE_T{index} * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator, p.compositor.initial_pipeline()))) return false;
    transition(p.list, p.buffers[index], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    const UINT drawn = p.compositor.record(device, p.list, target, p.width, p.height, steps, count, plane_surfaces, log);
    transition(p.list, p.buffers[index], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    if (!drawn) {
        p.list->Close();
        return false;
    }
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
