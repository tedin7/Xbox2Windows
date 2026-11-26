// Included by d3d12_xs_device.cpp after d3d12_draw.inl. Command list slot 16, CopyTextureRegion (#870).
// Each location's resource is mapped to its desktop resource, the call is checked by
// list_copy_texture.h, the two resources move to COPY_SOURCE / COPY_DEST for the call (the list's Close
// puts them back to COMMON), and a call the desktop would reject is counted and logged, not forwarded.
#include "list_copy_texture.h"

volatile LONG list_copy_texture_forwarded, list_copy_texture_refused;

namespace {

// Memory planes of a format on the resource's device (1 when the device cannot say).
UINT copy_texture_planes(ID3D12Resource* resource, DXGI_FORMAT format)
{
    ID3D12Device* device = nullptr;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(&device))) || !device) return 1;
    D3D12_FEATURE_DATA_FORMAT_INFO info{format, 1};
    const bool known = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info)));
    device->Release();
    return known && info.PlaneCount ? info.PlaneCount : 1;
}

// One side of the copy, read from the desktop resource it names. False when the location is unusable.
bool copy_texture_side(const D3D12_TEXTURE_COPY_LOCATION* location, ID3D12Resource*& real, x2w_list_copy_texture::Side& side,
                       DXGI_FORMAT& format, bool& bad_subresource)
{
    Wrapper* w = location && x2w_local::readable(location, sizeof(*location)) ? as_wrapper(location->pResource) : nullptr;
    if (!w || w->own().kind != resource_kind || !w->own().real) return false;
    real = static_cast<ID3D12Resource*>(w->own().real);
    const D3D12_RESOURCE_DESC desc = real->GetDesc();
    side = {};
    side.block_w = side.block_h = 1;
    side.samples = desc.SampleDesc.Count ? desc.SampleDesc.Count : 1;
    side.depth_stencil = (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0 ||
        desc.Format == DXGI_FORMAT_D16_UNORM || desc.Format == DXGI_FORMAT_D32_FLOAT;
    side.mips = side.array_size = side.planes = 1;
    if (location->Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return false;
        const auto& fp = location->PlacedFootprint;
        side.kind = x2w_list_copy_texture::Kind::placed_footprint;
        side.width = fp.Footprint.Width;
        side.height = fp.Footprint.Height;
        side.depth = fp.Footprint.Depth;
        side.offset = fp.Offset;
        side.row_pitch = fp.Footprint.RowPitch;
        side.resource_bytes = desc.Width;
        side.format = fp.Footprint.Format;
        format = fp.Footprint.Format;
        return true;
    }
    if (location->Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX || desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return false;
    side.mips = desc.MipLevels ? desc.MipLevels : 1;
    side.array_size = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : desc.DepthOrArraySize;
    side.planes = copy_texture_planes(real, desc.Format);
    side.subresource = location->SubresourceIndex;
    side.format = desc.Format;
    // The index must lie inside mips * array slices * planes; the mip it names sizes this side.
    x2w_list_copy_texture::Subresource parts{};
    if (!x2w_list_copy_texture::decode_subresource(side.subresource, side.mips, side.array_size, side.planes, parts)) {
        bad_subresource = true;
        return false;
    }
    const UINT mip = parts.mip;
    side.kind = x2w_list_copy_texture::Kind::subresource;
    side.width = static_cast<UINT>((std::max<UINT64>)(1, desc.Width >> mip));
    side.height = (std::max<UINT>)(1, desc.Height >> mip);
    side.depth = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? (std::max<UINT>)(1, desc.DepthOrArraySize >> mip) : 1;
    format = desc.Format;
    return true;
}

// Texels per block and bytes per block, from the desktop's own footprint for a one-subresource copy.
void copy_texture_block(ID3D12Resource* texture, x2w_list_copy_texture::Side& side)
{
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    ID3D12Device* device = nullptr;
    if (FAILED(texture->GetDevice(IID_PPV_ARGS(&device))) || !device) return;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &row_bytes, &total);
    device->Release();
    const bool bc = (desc.Format >= DXGI_FORMAT_BC1_TYPELESS && desc.Format <= DXGI_FORMAT_BC5_SNORM) ||
                    (desc.Format >= DXGI_FORMAT_BC6H_TYPELESS && desc.Format <= DXGI_FORMAT_BC7_UNORM_SRGB);
    side.block_w = side.block_h = bc ? 4 : 1;
    const UINT64 blocks = (std::max<UINT64>)(1, (desc.Width + side.block_w - 1) / side.block_w);
    side.bytes_per_block = static_cast<std::uint32_t>(row_bytes / blocks);
}

// Upload buffers stay in GENERIC_READ and readback buffers in COPY_DEST: the desktop refuses a barrier on either.
bool fixed_heap_state(ID3D12Resource* resource, D3D12_HEAP_TYPE& type)
{
    D3D12_HEAP_PROPERTIES props{};
    D3D12_HEAP_FLAGS flags{};
    if (FAILED(resource->GetHeapProperties(&props, &flags))) return false;
    type = props.Type;
    return type == D3D12_HEAP_TYPE_UPLOAD || type == D3D12_HEAP_TYPE_READBACK;
}

}  // namespace

// List slot 16 (unshifted, like slots 12 to 25), CopyTextureRegion(dst, x, y, z, src, box).
void __stdcall list_copy_texture(Wrapper* self, const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                 const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box)
{
    if (!self->own().list_open) return;
    static LONG logged;
    const auto refuse = [&](const char* why) {
        InterlockedIncrement(&list_copy_texture_refused);
        if (first_calls(logged, 6)) log("x2w-d3d12 CopyTextureRegion refused: %s", why);
    };
    ID3D12Resource *real_dst = nullptr, *real_src = nullptr;
    x2w_list_copy_texture::Side side_dst{}, side_src{};
    DXGI_FORMAT format_dst = DXGI_FORMAT_UNKNOWN, format_src = DXGI_FORMAT_UNKNOWN;
    bool bad_subresource = false;
    if (!copy_texture_side(dst, real_dst, side_dst, format_dst, bad_subresource) ||
        !copy_texture_side(src, real_src, side_src, format_src, bad_subresource))
        return refuse(bad_subresource ? "bad_subresource" : "unknown_resource");
    // Planar (depth-stencil, video) copies name a plane footprint format; only single-plane copies are forwarded.
    if (side_dst.planes > 1 || side_src.planes > 1) return refuse("planar_unsupported");
    ID3D12Resource* texture = side_dst.kind == x2w_list_copy_texture::Kind::subresource ? real_dst : real_src;
    if (side_dst.kind == x2w_list_copy_texture::Kind::subresource || side_src.kind == x2w_list_copy_texture::Kind::subresource) {
        x2w_list_copy_texture::Side blocks{};
        copy_texture_block(texture, blocks);
        for (auto* s : {&side_dst, &side_src}) {
            s->block_w = blocks.block_w ? blocks.block_w : 1;
            s->block_h = blocks.block_h ? blocks.block_h : 1;
            if (s->kind == x2w_list_copy_texture::Kind::placed_footprint) s->bytes_per_block = blocks.bytes_per_block;
        }
    }
    x2w_list_copy_texture::Box copy_box{};
    if (box && x2w_local::readable(box, sizeof(*box))) copy_box = {box->left, box->top, box->front, box->right, box->bottom, box->back};
    const auto verdict = x2w_list_copy_texture::check(side_dst, x, y, z, side_src, box ? &copy_box : nullptr);
    if (verdict != x2w_list_copy_texture::Verdict::ok) return refuse(x2w_list_copy_texture::name(verdict));
    // One resource tracks one state, so two subresources of it cannot be COPY_SOURCE and COPY_DEST at once.
    if (real_src == real_dst) return refuse("same_resource");
    D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT;
    const bool src_fixed = fixed_heap_state(real_src, heap);
    if (src_fixed && heap == D3D12_HEAP_TYPE_READBACK) return refuse("bad_heap");
    const bool dst_fixed = fixed_heap_state(real_dst, heap);
    if (dst_fixed && heap == D3D12_HEAP_TYPE_UPLOAD) return refuse("bad_heap");
    Wrapper* const src_wrapper = as_wrapper(src->pResource);
    Wrapper* const dst_wrapper = as_wrapper(dst->pResource);
    // A copied (non-aliased) buffer has no refresh path here, so its desktop bytes may be stale.
    if (side_src.kind == x2w_list_copy_texture::Kind::placed_footprint) {
        const auto* info = resource_info(src_wrapper);
        if (info && info->placement == Placement::copied_buffer) return refuse("stale_source");
    }
    // Bring a texture's desktop copy up to date with its guest backing before the copy reads or partly overwrites it,
    // and record that upload so a later sample does not replace the copied texels.
    // A resource of the retire class that has no title memory has nothing to upload and leaves no texture_uploads()
    // entry behind; every other resource uploads exactly as it did before the switch (switch off: nothing changes).
    const auto needs_upload = [](const Wrapper* wrapper) {
        return !wrapper->own().retire_when_idle || wrapper->own().title_memory;
    };
    if (side_src.kind == x2w_list_copy_texture::Kind::subresource && needs_upload(src_wrapper)) {
        AcquireSRWLockExclusive(&draw_lock);
        const bool uploaded = upload_texture(self, src_wrapper, real_src);
        ReleaseSRWLockExclusive(&draw_lock);
        // A source whose upload could not be staged holds no (or half the) pixels: do not copy it; the next recording retries.
        if (!uploaded) return refuse("source_not_uploaded");
    }
    if (side_dst.kind == x2w_list_copy_texture::Kind::subresource && needs_upload(dst_wrapper)) {
        AcquireSRWLockExclusive(&draw_lock);
        upload_texture(self, dst_wrapper, real_dst);
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (!src_fixed) require_state(self, real_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (!dst_fixed) require_state(self, real_dst, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION to = *dst, from = *src;
    to.pResource = real_dst;
    from.pResource = real_src;
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)->CopyTextureRegion(&to, x, y, z, &from, box ? box : nullptr);
    retire_hold(self, dst_wrapper);
    retire_hold(self, src_wrapper);
    InterlockedIncrement(&list_copy_texture_forwarded);
}

extern "C" __declspec(dllexport) void x2w_d3d12_list_copy_texture_stats(LONG* forwarded, LONG* refused)
{
    if (forwarded) *forwarded = list_copy_texture_forwarded;
    if (refused) *refused = list_copy_texture_refused;
}
