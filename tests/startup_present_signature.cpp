// Verifies the presented-picture signature (src/startup_stubs/present_signature_sampler.h) on WARP:
// the sampler copies only the sampled rows of a presented surface, publishes a grid equal to the CPU
// reduction of those rows, and the published signal changes once for two different pictures and
// never for repeated identical pictures (negative control). Also checks the sampling interval, that a
// bar of about 1 % of the frame between two base rows is missed by the 32 x 18 grid and caught by the
// fine 64 x 72 grid, prints the CPU-side cost per sampled present at 3840 x 2160, and checks that a
// sampler made with always = false records nothing until the block asks for samples.
//
// With "held_source" it runs the lifetime check under the D3D12 debug layer: the queue is stalled on
// a fence, the sampler records its copy, the caller releases its only reference to the surface,
// then the queue is released. The signature must still equal the picture, the sampler must drop its
// reference once the copy's fence has passed, and the debug layer must report no error. A queue whose
// Signal fails after submission must leave the sampler stranded, still holding the surface.
// With "unheld_control" it runs the same stall with a plain copy of its own that does not hold the
// surface (the defect the sampler avoids); it passes only if the debug layer reports that.
//
// usage: startup_present_signature [held_source | unheld_control]
//        (exit 77 when no WARP adapter, or for the two modes no debug layer, is available)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "startup_stubs/input_script.h"
#include "startup_stubs/present_signature_sampler.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

constexpr UINT width = 200, height = 120;

// Picture contents, 4 bytes a texel in the surface's byte order.
using Picture = std::vector<unsigned char>;
Picture gradient(bool inverted)
{
    Picture p(std::size_t{width} * height * 4);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
            unsigned char* t = &p[(std::size_t{y} * width + x) * 4];
            const auto v = static_cast<unsigned char>((x * 255) / (width - 1));
            t[0] = inverted ? 255 - v : v;
            t[1] = static_cast<unsigned char>((y * 255) / (height - 1));
            t[2] = inverted ? v : static_cast<unsigned char>(255 - v);
            t[3] = 255;
        }
    return p;
}

struct Gpu {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 value = 0;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    void idle()
    {
        queue->Signal(fence, ++value);
        if (fence->GetCompletedValue() < value) {
            fence->SetEventOnCompletion(value, event);
            WaitForSingleObject(event, 10000);
        }
    }
};

// A texture in COMMON holding `picture` (w x h texels).
ID3D12Resource* surface(Gpu& gpu, DXGI_FORMAT format, const Picture& picture, UINT w = width, UINT h = height)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* texture = nullptr;
    if (FAILED(gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&texture))))
        return nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* upload = nullptr;
    gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    void* mapped = nullptr;
    upload->Map(0, nullptr, &mapped);
    for (UINT y = 0; y < h; ++y)
        std::memcpy(static_cast<unsigned char*>(mapped) + footprint.Offset + std::size_t{y} * footprint.Footprint.RowPitch,
                    &picture[std::size_t{y} * w * 4], std::size_t{w} * 4);
    upload->Unmap(0, nullptr);
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION to{}, from{};
    to.pResource = texture;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.pResource = upload;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);
    list->Close();
    ID3D12CommandList* lists[] = {list};
    gpu.queue->ExecuteCommandLists(1, lists);
    gpu.idle();
    list->Release();
    allocator->Release();
    upload->Release();
    return texture;
}

// The grid the sampler must publish for `picture`, reduced on the CPU from the same rows.
void expected_grid(const Picture& picture, x2w_present_signature::Layout layout, std::uint8_t (&grid)[x2w_present_signature::cells])
{
    for (int band = 0; band < x2w_present_signature::cells_y; ++band)
        x2w_present_signature::reduce_row(&picture[std::size_t{x2w_present_signature::sampled_row(band, height)} * width * 4], width, layout,
                                          grid + band * x2w_present_signature::cells_x);
}

// Presents `surfaces` in order through a sampler with interval 1, letting the GPU finish between
// presents, then one empty present to publish the last copy. Returns the published grids.
std::vector<x2w_present_signature::Snapshot> present_all(Gpu& gpu, x2w_present_signature::Block* block, const std::vector<ID3D12Resource*>& surfaces)
{
    std::vector<x2w_present_signature::Snapshot> published;
    x2w_present_signature::Sampler sampler(1);
    LONG64 present = 0;
    const std::uint64_t start = static_cast<std::uint64_t>(block->sequence / 2);
    const auto record = [&] {
        x2w_present_signature::Snapshot s{};
        if (x2w_present_signature::read(block, s) && s.sequence > start && (published.empty() || s.sequence != published.back().sequence))
            published.push_back(s);
    };
    for (ID3D12Resource* s : surfaces) {
        sampler.present(gpu.device, gpu.queue, s, ++present, block);
        record();
        gpu.idle();
    }
    sampler.present(gpu.device, gpu.queue, nullptr, ++present, block);
    record();
    check(!sampler.failed(), "sampler setup");
    return published;
}

// Forwards every call to a real queue, except that Signal fails once `fail_signal` is set: the
// "submitted, but no fence" case the sampler has to survive.
class SignalFailingQueue : public ID3D12CommandQueue {
public:
    explicit SignalFailingQueue(ID3D12CommandQueue* real) : real_(real) {}
    bool fail_signal = false;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override { return real_->QueryInterface(riid, out); }
    ULONG STDMETHODCALLTYPE AddRef() override { return 2; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID g, UINT* n, void* d) override { return real_->GetPrivateData(g, n, d); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID g, UINT n, const void* d) override { return real_->SetPrivateData(g, n, d); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID g, const IUnknown* d) override { return real_->SetPrivateDataInterface(g, d); }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) override { return real_->SetName(name); }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override { return real_->GetDevice(riid, out); }
    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource* r, UINT n, const D3D12_TILED_RESOURCE_COORDINATE* c,
                                              const D3D12_TILE_REGION_SIZE* s, ID3D12Heap* h, UINT rn, const D3D12_TILE_RANGE_FLAGS* f,
                                              const UINT* o, const UINT* cnt, D3D12_TILE_MAPPING_FLAGS fl) override
    {
        real_->UpdateTileMappings(r, n, c, s, h, rn, f, o, cnt, fl);
    }
    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource* d, const D3D12_TILED_RESOURCE_COORDINATE* dc, ID3D12Resource* s,
                                            const D3D12_TILED_RESOURCE_COORDINATE* sc, const D3D12_TILE_REGION_SIZE* rs,
                                            D3D12_TILE_MAPPING_FLAGS f) override
    {
        real_->CopyTileMappings(d, dc, s, sc, rs, f);
    }
    void STDMETHODCALLTYPE ExecuteCommandLists(UINT n, ID3D12CommandList* const* lists) override { real_->ExecuteCommandLists(n, lists); }
    void STDMETHODCALLTYPE SetMarker(UINT m, const void* d, UINT n) override { real_->SetMarker(m, d, n); }
    void STDMETHODCALLTYPE BeginEvent(UINT m, const void* d, UINT n) override { real_->BeginEvent(m, d, n); }
    void STDMETHODCALLTYPE EndEvent() override { real_->EndEvent(); }
    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* fence, UINT64 value) override { return fail_signal ? E_FAIL : real_->Signal(fence, value); }
    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* fence, UINT64 value) override { return real_->Wait(fence, value); }
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* f) override { return real_->GetTimestampFrequency(f); }
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* g, UINT64* c) override { return real_->GetClockCalibration(g, c); }
#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override { return real_->GetDesc(); }
#else
    D3D12_COMMAND_QUEUE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_COMMAND_QUEUE_DESC* out) override { return real_->GetDesc(out); }
#endif

private:
    ID3D12CommandQueue* real_;
};

// Error and corruption messages the debug layer has stored since the last call; prints each one.
// `deleted_in_use` is set when one of them reports an object released while the GPU still used it.
int debug_errors(ID3D12InfoQueue* info, bool* deleted_in_use = nullptr)
{
    int errors = 0;
    const UINT64 stored = info->GetNumStoredMessages();
    for (UINT64 i = 0; i < stored; ++i) {
        SIZE_T bytes = 0;
        info->GetMessage(i, nullptr, &bytes);
        std::vector<unsigned char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(info->GetMessage(i, message, &bytes))) continue;
        if (message->Severity != D3D12_MESSAGE_SEVERITY_ERROR && message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION) continue;
        ++errors;
        if (deleted_in_use && message->ID == D3D12_MESSAGE_ID_OBJECT_DELETED_WHILE_STILL_IN_USE) *deleted_in_use = true;
        std::printf("debug layer: %.*s\n", static_cast<int>(message->DescriptionByteLength), message->pDescription);
    }
    info->ClearStoredMessages();
    return errors;
}

// Builds a stall: the queue waits on `gate` (not yet signalled) before anything submitted after it.
ID3D12Fence* stall(Gpu& gpu)
{
    ID3D12Fence* gate = nullptr;
    gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate));
    gpu.queue->Wait(gate, 1);
    return gate;
}

int lifetime(Gpu& gpu, ID3D12InfoQueue* info, x2w_present_signature::Block* block, bool control)
{
    const Picture picture = gradient(false);
    std::uint8_t want[x2w_present_signature::cells];
    expected_grid(picture, x2w_present_signature::Layout::rgba8, want);
    debug_errors(info);  // setup noise, if any, is not part of the check
    if (control) {
        // The defect: a copy whose source nobody holds. The test records it itself, releases its only
        // reference while the queue is stalled, then lets the queue run.
        ID3D12Resource* source = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        ID3D12Fence* gate = stall(gpu);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 1024;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* readback = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
        gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = source;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.pResource = readback;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, width, 1, 1, 1024};
        const D3D12_BOX row{0, 0, 0, width, 1, 1};
        list->CopyTextureRegion(&to, 0, 0, 0, &from, &row);
        list->Close();
        ID3D12CommandList* lists[] = {list};
        gpu.queue->ExecuteCommandLists(1, lists);
        source->Release();  // nobody holds the source of the stalled copy now
        gate->Signal(1);
        gpu.idle();
        bool deleted_in_use = false;
        const int errors = debug_errors(info, &deleted_in_use);
        std::printf("control: %d debug-layer errors, released while in use %s\n", errors, deleted_in_use ? "reported" : "not reported");
        check(deleted_in_use, "the debug layer reports a stalled copy whose source was released (the check can fail)");
        list->Release();
        allocator->Release();
        readback->Release();
        gate->Release();
        return 0;
    }

    // The sampler: its copy is stalled, the caller drops its reference, then the queue runs.
    {
        ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        x2w_present_signature::Sampler sampler(1);
        ID3D12Fence* gate = stall(gpu);
        const LONG64 before = block->sequence / 2;
        sampler.present(gpu.device, gpu.queue, presented, 1, block);
        check(sampler.in_flight() && sampler.holds_source(), "the stalled copy is in flight and the sampler holds its surface");
        const ULONG left = presented->Release();  // the caller's only reference, right after the present
        check(left == 1, "after the caller's release, only the sampler's reference is left");
        sampler.present(gpu.device, gpu.queue, nullptr, 2, block);
        check(block->sequence / 2 == before && sampler.holds_source(), "nothing is published and the surface is kept while the queue is stalled");
        gate->Signal(1);
        gpu.idle();
        sampler.present(gpu.device, gpu.queue, nullptr, 3, block);
        x2w_present_signature::Snapshot snapshot{};
        check(block->sequence / 2 == before + 1 && x2w_present_signature::read(block, snapshot) && snapshot.sample_present == 1 &&
                  std::memcmp(snapshot.luma, want, sizeof(want)) == 0,
              "after the stall the published signature equals the released surface's picture");
        check(!sampler.holds_source() && !sampler.in_flight(), "the sampler drops the surface once the copy's fence has passed");
        gate->Release();
    }
    // Signal fails after the list was submitted: the sampler stops and keeps everything the copy uses.
    {
        ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        SignalFailingQueue failing(gpu.queue);
        failing.fail_signal = true;
        ID3D12Fence* gate = stall(gpu);
        auto* sampler = new x2w_present_signature::Sampler(1);
        sampler->present(gpu.device, &failing, presented, 1, block);
        check(sampler->failed() && sampler->stranded() && sampler->holds_source(), "a failed Signal strands the sampler with the surface held");
        const ULONG left = presented->Release();
        check(left == 1, "the stranded sampler still holds the surface after the caller's release");
        gate->Signal(1);
        gpu.idle();
        sampler->present(gpu.device, &failing, presented, 2, block);
        check(sampler->stranded() && !sampler->in_flight() && sampler->holds_source(), "a stranded sampler records nothing more and keeps the surface");
        delete sampler;  // a stranded sampler releases nothing it may still be using
        gate->Release();
    }
    const int errors = debug_errors(info);
    std::printf("held source: %d debug-layer errors\n", errors);
    check(errors == 0, "the debug layer reports no error for the sampler's stalled copy");
    return 0;
}

int changes(const std::vector<x2w_present_signature::Snapshot>& grids)
{
    int n = 0;
    for (std::size_t i = 1; i < grids.size(); ++i)
        n += x2w_input_script::signal_distance(grids[i - 1].luma, grids[i].luma) >= x2w_input_script::changed_cells;
    return n;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool held = argc > 1 && std::strcmp(argv[1], "held_source") == 0;
    const bool control = argc > 1 && std::strcmp(argv[1], "unheld_control") == 0;
    if (argc > 1 && !held && !control) {
        std::printf("unknown mode %s\n", argv[1]);
        return 2;
    }
    if (held || control) {
        ID3D12Debug* debug = nullptr;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            std::printf("SKIP: no D3D12 debug layer\n");
            return 77;
        }
        debug->EnableDebugLayer();
        debug->Release();
    }
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter* warp = nullptr;
    Gpu gpu;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&gpu.device)))) {
        std::printf("SKIP: no WARP device\n");
        return 77;
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    gpu.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&gpu.queue));
    gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence));

    // A block of its own: this test's process id plus a high bit, so it never meets a run's block.
    x2w_present_signature::Block* block = x2w_present_signature::create_block(GetCurrentProcessId() | 0x80000000u, 1);
    check(block != nullptr, "create the signature block");
    if (!block) return 1;

    if (held || control) {
        ID3D12InfoQueue* info = nullptr;
        if (FAILED(gpu.device->QueryInterface(IID_PPV_ARGS(&info)))) {
            std::printf("SKIP: the debug layer gave no info queue\n");
            return 77;
        }
        lifetime(gpu, info, block, control);
        info->Release();
        std::printf("%s\n", failures ? "FAIL" : "pass");
        return failures ? 1 : 0;
    }

    const Picture a = gradient(false), b = gradient(true);
    for (DXGI_FORMAT format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_TYPELESS}) {
        const auto layout = format == DXGI_FORMAT_R8G8B8A8_UNORM ? x2w_present_signature::Layout::rgba8 : x2w_present_signature::Layout::bgra8;
        ID3D12Resource* a1 = surface(gpu, format, a);
        ID3D12Resource* a2 = surface(gpu, format, a);
        ID3D12Resource* b1 = surface(gpu, format, b);
        ID3D12Resource* b2 = surface(gpu, format, b);
        check(a1 && a2 && b1 && b2, "create the surfaces");
        if (!(a1 && a2 && b1 && b2)) return 1;

        // Two different pictures (A, A, B, B): the signal changes exactly once.
        const auto changing = present_all(gpu, block, {a1, a2, b1, b2});
        std::uint8_t grid_a[x2w_present_signature::cells], grid_b[x2w_present_signature::cells];
        expected_grid(a, layout, grid_a);
        expected_grid(b, layout, grid_b);
        check(changing.size() == 4, "four samples published for four presents");
        if (changing.size() == 4) {
            check(std::memcmp(changing[0].luma, grid_a, sizeof(grid_a)) == 0 && std::memcmp(changing[3].luma, grid_b, sizeof(grid_b)) == 0,
                  "published grids equal the CPU reduction of the sampled rows");
            check(changing[0].sample_present == 1 && changing[3].sample_present == 4, "samples carry their present numbers");
        }
        check(changes(changing) == 1, "two different pictures: the signal changes once");
        // Negative control: the same picture presented four times never changes the signal.
        const auto steady = present_all(gpu, block, {a1, a2, a1, a2});
        check(steady.size() == 4 && changes(steady) == 0, "identical pictures: the signal never changes");
        for (ID3D12Resource* r : {a1, a2, b1, b2}) r->Release();
    }

    // Opt-in: a sampler made with always = false creates no GPU objects and records no copy until the
    // block asks for samples; then it samples as usual.
    {
        ID3D12Resource* s = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, a);
        x2w_present_signature::Sampler sampler(1, false);
        InterlockedExchange(&block->sampling_wanted, 0);
        const LONG64 before = block->sequence / 2;
        for (LONG64 present = 1; present <= 4; ++present) {
            sampler.present(gpu.device, gpu.queue, s, present, block);
            gpu.idle();
        }
        check(!sampler.set_up() && !sampler.in_flight() && block->sequence / 2 == before && block->presents == 4,
              "without a request: no GPU objects, no copies, presents still counted");
        InterlockedExchange(&block->sampling_wanted, 1);
        sampler.present(gpu.device, gpu.queue, s, 5, block);
        gpu.idle();
        sampler.present(gpu.device, gpu.queue, nullptr, 6, block);
        x2w_present_signature::Snapshot last{};
        check(sampler.set_up() && block->sequence / 2 == before + 1 && x2w_present_signature::read(block, last) && last.sample_present == 5,
              "after the request: the next present is sampled");
        InterlockedExchange(&block->sampling_wanted, 0);
        s->Release();
    }
    // Interval: with every = 4, eight presents publish two samples (presents 1 and 5).
    {
        ID3D12Resource* s = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, a);
        x2w_present_signature::Sampler sampler(4);
        const LONG64 before = block->sequence / 2;
        for (LONG64 present = 1; present <= 8; ++present) {
            sampler.present(gpu.device, gpu.queue, s, present, block);
            gpu.idle();
        }
        sampler.present(gpu.device, gpu.queue, nullptr, 9, block);
        x2w_present_signature::Snapshot last{};
        check(block->sequence / 2 - before == 2 && x2w_present_signature::read(block, last) && last.sample_present == 5 && last.presents == 9,
              "every 4 presents: two samples from presents 1 and 5, latest present 9");
        s->Release();
    }
    // A format the signature does not read is refused, not guessed.
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 16;
        desc.Height = 16;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.SampleDesc.Count = 1;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        ID3D12Resource* half = nullptr;
        gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&half));
        x2w_present_signature::Sampler sampler(1);
        sampler.present(gpu.device, gpu.queue, half, 1, block);
        check(half && sampler.unsupported_format() == DXGI_FORMAT_R16G16B16A16_FLOAT && !sampler.in_flight(),
              "16-bit float surface refused without a copy");
        if (half) half->Release();
    }
    // Resolution: a bar 60 x 4 texels (30 % x 3.3 %, 1 % of the frame) at rows 11..14, between the base
    // rows 10 and 16, switches from black to white. The base grid must not see it; the fine grid must,
    // and its published cells must equal the CPU reduction of the 72 fine rows.
    {
        Picture off = gradient(false), on = off;
        for (UINT y = 11; y < 15; ++y)
            for (UINT x = 20; x < 80; ++x)
                for (int c = 0; c < 3; ++c) {
                    off[(std::size_t{y} * width + x) * 4 + c] = 0;
                    on[(std::size_t{y} * width + x) * 4 + c] = 255;
                }
        ID3D12Resource* s_off = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, off);
        ID3D12Resource* s_on = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, on);
        InterlockedExchange(&block->fine_wanted, 1);
        const auto grids = present_all(gpu, block, {s_off, s_on});
        InterlockedExchange(&block->fine_wanted, 0);
        check(grids.size() == 2 && grids[0].fine_valid && grids[1].fine_valid, "fine grid published while asked for");
        if (grids.size() == 2) {
            x2w_input_script::Wait base_wait{}, fine_wait{};
            base_wait.level = fine_wait.level = x2w_input_script::cell_tolerance;
            fine_wait.fine = true;
            const int base_cells = x2w_input_script::wait_distance(base_wait, grids[0].luma, grids[1].luma);
            const int fine_cells = x2w_input_script::wait_distance(fine_wait, grids[0].fine, grids[1].fine);
            std::printf("1 %% bar: base grid %d changed cells, fine grid %d changed cells\n", base_cells, fine_cells);
            check(base_cells == 0, "the 32 x 18 grid misses a 1 % bar between its rows");
            check(fine_cells >= x2w_input_script::changed_cells, "the 64 x 72 grid sees the 1 % bar in at least 16 cells");
            std::uint8_t want[x2w_present_signature::fine_cells];
            for (int band = 0; band < x2w_present_signature::fine_y; ++band)
                x2w_present_signature::reduce_row(&on[std::size_t{x2w_present_signature::sampled_row(band, height, x2w_present_signature::fine_y)} * width * 4],
                                                  width, x2w_present_signature::Layout::rgba8, want + band * x2w_present_signature::fine_x,
                                                  x2w_present_signature::fine_x);
            check(std::memcmp(grids[1].fine, want, sizeof(want)) == 0, "published fine cells equal the CPU reduction of the fine rows");
        }
        const auto plain = present_all(gpu, block, {s_off, s_on});
        check(plain.size() == 2 && !plain[0].fine_valid && !plain[1].fine_valid, "no fine grid while nobody asks");
        s_off->Release();
        s_on->Release();
    }
    // Cost at 3840 x 2160: CPU time of the present() calls (record the row copies; map, reduce and
    // publish the previous copy) with and without the fine grid. The GPU copies 18 or 90 rows.
    {
        constexpr UINT big_w = 3840, big_h = 2160;
        Picture big(std::size_t{big_w} * big_h * 4, 128);
        ID3D12Resource* s = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, big, big_w, big_h);
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        double ms[2] = {};
        for (int fine = 0; fine < 2; ++fine) {
            InterlockedExchange(&block->fine_wanted, fine);
            x2w_present_signature::Sampler sampler(1);
            constexpr int rounds = 24;
            LONGLONG ticks = 0;
            for (int i = 0; i < rounds + 2; ++i) {
                LARGE_INTEGER a, b;
                QueryPerformanceCounter(&a);
                sampler.present(gpu.device, gpu.queue, s, i + 1, block);
                QueryPerformanceCounter(&b);
                if (i >= 2) ticks += b.QuadPart - a.QuadPart;  // the first two set up the buffers
                gpu.idle();
            }
            ms[fine] = ticks * 1000.0 / frequency.QuadPart / rounds;
        }
        InterlockedExchange(&block->fine_wanted, 0);
        const double row_kib = big_w * 4 / 1024.0;
        std::printf("cost per sampled present at %ux%u (CPU side, WARP): base %.3f ms (18 rows, %.0f KiB), fine %.3f ms (90 rows, %.0f KiB); "
                    "a full frame is %.0f KiB\n",
                    big_w, big_h, ms[0], 18 * row_kib, ms[1], 90 * row_kib, big_h * row_kib);
        s->Release();
    }
    std::printf("%s\n", failures ? "FAIL" : "pass");
    return failures ? 1 : 0;
}
