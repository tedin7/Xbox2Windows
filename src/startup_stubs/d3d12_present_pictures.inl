// X2W_PRESENT_PICTURES=<dir> (docs/startup-runtime.md#d3d12-replay-record): every present writes plane 0's surface
// to <dir>\present_<nnnn>.png, n = the present's number. A recorded run and its replay write the same names, so the
// pictures compare one to one. The copy is waited for on the presenting queue (a diagnostic: it costs a GPU round trip
// per present). Formats: the 4-byte colour formats the picture signature reads (RGBA8, BGRA8/X8, RGB10A2 -> 8 bits).
namespace present_pictures {

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

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* surface, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {surface, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

// Converts one row of the copied surface to RGBA8.
void to_rgba(const unsigned char* in, unsigned char* out, UINT width, x2w_present_signature::Layout layout)
{
    for (UINT x = 0; x < width; ++x, in += 4, out += 4) {
        if (layout == x2w_present_signature::Layout::bgra8) {
            out[0] = in[2], out[1] = in[1], out[2] = in[0], out[3] = in[3];
        } else if (layout == x2w_present_signature::Layout::rgb10a2) {
            std::uint32_t v;
            std::memcpy(&v, in, 4);
            out[0] = static_cast<unsigned char>((v >> 2) & 0xff), out[1] = static_cast<unsigned char>((v >> 12) & 0xff);
            out[2] = static_cast<unsigned char>((v >> 22) & 0xff), out[3] = static_cast<unsigned char>((v >> 30) * 85);
        } else {
            std::memcpy(out, in, 4);
        }
    }
}

// Copies `surface` on `queue`, waits, and writes the PNG. Returns false (with a reason) when it cannot.
bool write(Copier& c, ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* surface, const char* path,
           const char*& reason)
{
    const D3D12_RESOURCE_DESC desc = surface->GetDesc();
    x2w_present_signature::Layout layout;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !x2w_present_signature::layout_of(desc.Format, layout) ||
        desc.Width > 16384 || desc.Height > 16384) {
        reason = "surface format or size not written";
        return false;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    if (!c.setup(device, total) || FAILED(c.allocator->Reset()) || FAILED(c.list->Reset(c.allocator, nullptr))) {
        reason = "copy objects could not be made";
        return false;
    }
    transition(c.list, surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{surface, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION to{c.readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = footprint;
    c.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(c.list, surface, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (FAILED(c.list->Close())) {
        reason = "copy list did not close";
        return false;
    }
    ID3D12CommandList* lists[] = {c.list};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(c.fence, ++c.fence_value)) || FAILED(c.fence->SetEventOnCompletion(c.fence_value, c.event)) ||
        WaitForSingleObject(c.event, 10000) != WAIT_OBJECT_0) {
        c.failed = true;  // the copy may still be running: never reuse its objects
        reason = "copy did not finish in 10 s";
        return false;
    }
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    if (FAILED(c.readback->Map(0, &range, &mapped)) || !mapped) {
        reason = "readback did not map";
        return false;
    }
    const auto width = static_cast<UINT>(desc.Width);
    std::vector<unsigned char> rgba(static_cast<std::size_t>(width) * desc.Height * 4);
    for (UINT y = 0; y < desc.Height; ++y)
        to_rgba(static_cast<const unsigned char*>(mapped) + footprint.Offset + static_cast<UINT64>(y) * footprint.Footprint.RowPitch,
                rgba.data() + static_cast<std::size_t>(y) * width * 4, width, layout);
    const D3D12_RANGE none{0, 0};
    c.readback->Unmap(0, &none);
    const std::vector<unsigned char> png = x2w_png::encode(rgba.data(), width, desc.Height);
    std::FILE* f = std::fopen(path, "wb");
    const bool ok = f && std::fwrite(png.data(), 1, png.size(), f) == png.size();
    if (f) std::fclose(f);
    if (!ok) reason = "file not written";
    return ok;
}

void present(Wrapper* queue, Wrapper* surface, LONG frame)
{
    static char dir[MAX_PATH - 32]{};
    static const bool wanted = [] {
        const DWORD n = GetEnvironmentVariableA("X2W_PRESENT_PICTURES", dir, sizeof(dir));
        if (n && n < sizeof(dir)) log("x2w-present pictures: every present -> %s\\present_<nnnn>.png", dir);
        return n && n < sizeof(dir);
    }();
    if (!wanted) return;
    static SRWLOCK lock = SRWLOCK_INIT;
    static Copier copier;
    static LONG written, refused;
    const char* reason = "no resource on plane 0";
    char path[MAX_PATH];
    std::snprintf(path, sizeof(path), "%s\\present_%04ld.png", dir, frame);
    AcquireSRWLockExclusive(&lock);
    const bool ok = !copier.failed && surface && surface->own().kind == resource_kind &&
                    write(copier, device_of(the_device), static_cast<ID3D12CommandQueue*>(queue->own().real),
                          static_cast<ID3D12Resource*>(surface->own().real), path, reason);
    if (copier.failed && !ok) reason = "an earlier copy did not finish; pictures stopped";
    ReleaseSRWLockExclusive(&lock);
    if (ok && first_calls(written, 1)) log("x2w-present pictures: first picture %s", path);
    if (!ok && first_calls(refused, 4)) log("x2w-present pictures: present %ld not written: %s", frame, reason);
}

}  // namespace present_pictures
