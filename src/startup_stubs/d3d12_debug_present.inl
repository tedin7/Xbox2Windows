// X2W_DEBUG_PRESENT_TARGET = depth | largest-rt | rt:<W>x<H>:<fmt> (docs/startup-runtime.md#debug-present).
// Opt-in. At every title Present the picture the title presents is replaced, before anything reads it (the picture
// signature, X2W_PRESENT_PICTURES, the capture, the output window), by a picture of the chosen target of the last frame:
//   depth       the biggest depth buffer a pass bound, as grey (debug_present_convert.h: reversed-Z aware, near is bright)
//   largest-rt  the biggest colour target a pass bound, tone-mapped to 8 bits
//   rt:WxH:fmt  the colour target (or depth buffer) of exactly that size and DXGI format
// The target is stretched to the presented surface's size. Targets are found where a pass binds them (OMSetRenderTargets),
// so a clear alone shows too. The copy runs on the presenting queue and is waited for: a diagnostic, a GPU round trip.
// A resource is AddRef'd while it is the choice, so the title releasing it cannot free it under the copy.
#include "startup_stubs/debug_present_convert.h"

namespace debug_present {

struct Tracker {
    SRWLOCK lock = SRWLOCK_INIT;
    x2w_debug_present::Spec spec;
    ID3D12Resource* best = nullptr;  // holds a reference
    x2w_debug_present::Candidate info;

    explicit Tracker(const x2w_debug_present::Spec& s = {}) : spec(s) {}
    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;
    ~Tracker() { if (best) best->Release(); }

    bool enabled() const { return spec.mode != x2w_debug_present::Mode::off; }

    // A pass bound `resource` (a colour target, or the depth buffer when `depth`).
    void note(ID3D12Resource* resource, unsigned long long width, unsigned height, int format, bool depth)
    {
        if (!enabled() || !resource) return;
        const x2w_debug_present::Candidate next{width, height, format, depth};
        AcquireSRWLockExclusive(&lock);
        if (resource == best) {
            info = next;  // the same object bound again: its format may be a different view
        } else if (x2w_debug_present::wants(spec, info, next)) {
            resource->AddRef();
            if (best) best->Release();
            best = resource;
            info = next;
        }
        ReleaseSRWLockExclusive(&lock);
    }

    // The choice with its own reference, or null.
    ID3D12Resource* take(x2w_debug_present::Candidate& out)
    {
        AcquireSRWLockExclusive(&lock);
        ID3D12Resource* resource = best;
        out = info;
        if (resource) resource->AddRef();
        ReleaseSRWLockExclusive(&lock);
        return resource;
    }
};

// Copy objects for the readback of the chosen target (the readback buffer grows to the biggest target).
struct Copier {
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE event = nullptr;
    ID3D12Resource* readback = nullptr;
    UINT64 readback_bytes = 0, fence_value = 0;
    bool failed = false;

    bool setup(ID3D12Device* device, UINT64 bytes)
    {
        if (!allocator && (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
                           FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
                           FAILED(list->Close()) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
                           !(event = CreateEventW(nullptr, FALSE, FALSE, nullptr))))
            return false;
        if (readback_bytes >= bytes) return true;
        if (readback) readback->Release();
        readback = nullptr;
        readback_bytes = 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&readback))))
            return false;
        x2w_budget::committed(device, readback, D3D12_HEAP_TYPE_READBACK);
        readback_bytes = bytes;
        return true;
    }
};

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

struct Painter {
    Copier copier;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Resource* upload = nullptr;
    UINT64 upload_bytes = 0;
    std::vector<unsigned char> rgba, stretched;

    bool setup(ID3D12Device* device, UINT64 bytes)
    {
        if (!allocator && (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
                           FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
                           FAILED(list->Close())))
            return false;
        if (upload_bytes >= bytes) return true;
        if (upload) upload->Release();
        upload = nullptr;
        upload_bytes = 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                   IID_PPV_ARGS(&upload))))
            return false;
        x2w_budget::committed(device, upload, D3D12_HEAP_TYPE_UPLOAD);
        upload_bytes = bytes;
        return true;
    }
};

// Submits the list the painter holds on `queue` and waits for it.
bool submit_and_wait(Copier& c, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list)
{
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(c.fence, ++c.fence_value)) || FAILED(c.fence->SetEventOnCompletion(c.fence_value, c.event)) ||
        WaitForSingleObject(c.event, 10000) != WAIT_OBJECT_0) {
        c.failed = true;  // may still be running: never reuse these objects
        return false;
    }
    return true;
}

// The row of `rgba` as the presented surface stores it.
void store_row(const unsigned char* in, unsigned char* out, UINT width, x2w_present_signature::Layout layout)
{
    for (UINT x = 0; x < width; ++x, in += 4, out += 4) {
        if (layout == x2w_present_signature::Layout::bgra8) {
            out[0] = in[2], out[1] = in[1], out[2] = in[0], out[3] = 255;
        } else if (layout == x2w_present_signature::Layout::rgb10a2) {
            const std::uint32_t v = (std::uint32_t{in[0]} << 2) | (std::uint32_t{in[1]} << 12) | (std::uint32_t{in[2]} << 22) | (3u << 30);
            std::memcpy(out, &v, 4);
        } else {
            out[0] = in[0], out[1] = in[1], out[2] = in[2], out[3] = 255;
        }
    }
}

// Copies `source` (a target in COMMON state) to the readback, converts it and writes the picture into `surface` (COMMON).
bool show(Painter& p, ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* source,
          const x2w_debug_present::Candidate& info, ID3D12Resource* surface, const char*& reason)
{
    const D3D12_RESOURCE_DESC sdesc = source->GetDesc(), tdesc = surface->GetDesc();
    x2w_present_signature::Layout layout;
    if (sdesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || sdesc.SampleDesc.Count != 1 ||
        !x2w_debug_present::plane_bytes(info.format) || sdesc.Width > 16384 || sdesc.Height > 16384) {
        reason = "the chosen target's format, size or sample count is not read";
        return false;
    }
    if (tdesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !x2w_present_signature::layout_of(tdesc.Format, layout) ||
        tdesc.Width > 16384 || tdesc.Height > 16384) {
        reason = "the presented surface's format or size is not written";
        return false;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT from_fp{}, to_fp{};
    UINT64 from_total = 0, to_total = 0;
    device->GetCopyableFootprints(&sdesc, 0, 1, 0, &from_fp, nullptr, nullptr, &from_total);
    device->GetCopyableFootprints(&tdesc, 0, 1, 0, &to_fp, nullptr, nullptr, &to_total);
    auto& c = p.copier;
    if (c.failed || !c.setup(device, from_total) || !p.setup(device, to_total) || FAILED(c.allocator->Reset()) ||
        FAILED(c.list->Reset(c.allocator, nullptr))) {
        reason = "copy objects could not be made, or an earlier copy did not finish";
        return false;
    }
    transition(c.list, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, to{c.readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = from_fp;
    c.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(c.list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (!submit_and_wait(c, queue, c.list)) {
        reason = "the copy of the chosen target did not finish in 10 s";
        return false;
    }
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(from_total)};
    if (FAILED(c.readback->Map(0, &range, &mapped)) || !mapped) {
        reason = "the readback did not map";
        return false;
    }
    const bool read = x2w_debug_present::to_rgba(static_cast<const unsigned char*>(mapped) + from_fp.Offset, from_fp.Footprint.RowPitch,
                                                 static_cast<UINT>(sdesc.Width), sdesc.Height, info.format, p.rgba);
    const D3D12_RANGE none{0, 0};
    c.readback->Unmap(0, &none);
    if (!read) {
        reason = "the chosen target's format is not read";
        return false;
    }
    const auto tw = static_cast<UINT>(tdesc.Width), th = tdesc.Height;
    x2w_debug_present::stretch(p.rgba, static_cast<UINT>(sdesc.Width), sdesc.Height, tw, th, p.stretched);
    void* up = nullptr;
    const D3D12_RANGE write_range{0, static_cast<SIZE_T>(to_total)};
    if (FAILED(p.upload->Map(0, &none, &up)) || !up) {
        reason = "the upload buffer did not map";
        return false;
    }
    for (UINT y = 0; y < th; ++y)
        store_row(&p.stretched[static_cast<std::size_t>(y) * tw * 4],
                  static_cast<unsigned char*>(up) + to_fp.Offset + static_cast<UINT64>(y) * to_fp.Footprint.RowPitch, tw, layout);
    p.upload->Unmap(0, &write_range);
    if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator, nullptr))) {
        reason = "the upload list did not reset";
        return false;
    }
    transition(p.list, surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst{surface, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, src{p.upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint = to_fp;
    p.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(p.list, surface, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    if (!submit_and_wait(c, queue, p.list)) {
        reason = "the picture copy into the presented surface did not finish in 10 s";
        return false;
    }
    return true;
}

Tracker& global_tracker()
{
    static Tracker tracker([] {
        char value[64]{};
        const DWORD n = GetEnvironmentVariableA("X2W_DEBUG_PRESENT_TARGET", value, sizeof(value));
        const x2w_debug_present::Spec spec = n && n < sizeof(value) ? x2w_debug_present::parse_spec(value) : x2w_debug_present::Spec{};
        if (n && spec.mode == x2w_debug_present::Mode::off)
            log("x2w-render debug present: X2W_DEBUG_PRESENT_TARGET=%s is not depth, largest-rt or rt:<W>x<H>:<fmt>; off", value);
        else if (spec.mode != x2w_debug_present::Mode::off)
            log("x2w-render debug present: on, target %s", value);
        return spec;
    }());
    return tracker;
}

// What the picture shows, for the log line: "depth 2560x1440 fmt 20".
std::string describe(const x2w_debug_present::Candidate& info)
{
    return std::string(info.depth ? "depth " : "colour ") + std::to_string(info.width) + "x" + std::to_string(info.height) +
           " fmt " + std::to_string(info.format);
}

// A pass bound these targets.
void note_bound(ID3D12Resource* const* targets, const x2w_startup::TargetDrawCensus::Surface* surfaces, UINT count,
                ID3D12Resource* depth, const x2w_startup::TargetDrawCensus::Surface& depth_surface)
{
    Tracker& tracker = global_tracker();
    if (!tracker.enabled()) return;
    for (UINT i = 0; i < count; ++i)
        tracker.note(targets[i], surfaces[i].width, surfaces[i].height, surfaces[i].format, false);
    if (depth) tracker.note(depth, depth_surface.width, depth_surface.height, depth_surface.format, true);
}

// At a title Present, before the picture is read: replace plane 0's surface content by the chosen target.
void present(Wrapper* queue, Wrapper* surface)
{
    Tracker& tracker = global_tracker();
    if (!tracker.enabled()) return;
    static SRWLOCK lock = SRWLOCK_INIT;
    static Painter painter;
    static LONG refused, shown;
    static ID3D12Resource* last_shown;
    static x2w_debug_present::Candidate last_info;
    x2w_debug_present::Candidate info;
    ID3D12Resource* source = tracker.take(info);
    const char* reason = !source ? "no pass has bound a matching target yet" : "no presented surface";
    bool ok = false;
    if (source && surface && surface->own().kind == resource_kind && surface->own().real) {
        AcquireSRWLockExclusive(&lock);
        ok = show(painter, device_of(the_device), static_cast<ID3D12CommandQueue*>(queue->own().real), source, info,
                  static_cast<ID3D12Resource*>(surface->own().real), reason);
        if (ok && (source != last_shown || info.width != last_info.width || info.format != last_info.format)) {
            log("x2w-render debug present: showing %s", describe(info).c_str());
            last_shown = source;
            last_info = info;
        }
        ReleaseSRWLockExclusive(&lock);
    }
    if (source) source->Release();
    if (ok) InterlockedIncrement(&shown);
    else if (first_calls(refused, 6)) log("x2w-render debug present: frame picture left as the title made it: %s", reason);
}

}  // namespace debug_present

// WARP fixture (tests/startup_debug_present.cpp): the production selection, copy and conversion on desktop objects. `spec`
// is the setting's text; the colour target and the depth buffer are "bound" in that order. Returns 0 and the log text in
// `what` when a picture was written, 1 when nothing matched, 2 when the picture could not be made (`what` has the reason).
extern "C" __declspec(dllexport) int x2w_d3d12_fixture_debug_present(ID3D12Device* device, ID3D12CommandQueue* queue, const char* spec,
        ID3D12Resource* color, int color_format, ID3D12Resource* depth, int depth_format, ID3D12Resource* surface,
        char* what, std::size_t capacity)
{
    debug_present::Tracker tracker(x2w_debug_present::parse_spec(spec));
    const auto note = [&](ID3D12Resource* resource, int format, bool is_depth) {
        if (!resource) return;
        const D3D12_RESOURCE_DESC desc = resource->GetDesc();
        tracker.note(resource, desc.Width, desc.Height, format, is_depth);
    };
    note(color, color_format, false);
    note(depth, depth_format, true);
    x2w_debug_present::Candidate info;
    ID3D12Resource* source = tracker.take(info);
    if (!source) {
        std::snprintf(what, capacity, "no match");
        return 1;
    }
    debug_present::Painter painter;
    const char* reason = "";
    const bool ok = debug_present::show(painter, device, queue, source, info, surface, reason);
    source->Release();
    std::snprintf(what, capacity, "%s", ok ? debug_present::describe(info).c_str() : reason);
    return ok ? 0 : 2;
}
