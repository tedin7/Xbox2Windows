namespace bc64 = x2w_startup::bc64;

u64 __stdcall bc64_add_ref(Computer* self) { return InterlockedIncrement(&self->refs); }

long __stdcall bc64_layout(Computer* self, unsigned char* out)
{
    const bc64::Chain chain(self->desc);
    if (!out) return E_POINTER;
    std::memset(out, 0, layout_header + 4 * plane_bytes + 8);
    put64(out, chain.bytes());
    put64(out + 8, 65536);
    put32(out + 0x10, chain.levels);
    put32(out + 0x14, 1);
    put32(out + 0x1b38, D3D12_RESOURCE_DIMENSION_TEXTURE2D);
    auto* plane = out + layout_header;
    put32(plane, 1);
    put64(plane + 8, chain.bytes());
    put64(plane + 0x10, chain.slice_bytes());
    put64(plane + 0x20, 65536);
    put32(plane + 0x28, 16);
    put32(plane + 0x2c, bc64::first_tail);
    put32(plane + 0x30, chain.levels - bc64::first_tail);
    for (UINT level = 0; level < chain.levels; ++level) {
        auto* mip = plane + mip_base + level * mip_bytes;
        const UINT columns = chain.columns(level), rows = chain.rows(level);
        const bool tail = level >= bc64::first_tail;
        const UINT padded_columns = tail ? (std::max)(4u, 256u >> level) : chain.tiles_wide(level) * 64;
        const UINT padded_rows = tail ? (std::max)(4u, 512u >> level) : chain.tiles_high(level) * 64;
        const UINT span = 16 * padded_columns * padded_rows;
        put64(mip, span);
        put64(mip + 8, chain.start(level));
        put64(mip + 0x10, span);
        put32(mip + 0x18, 4 * padded_columns);
        put32(mip + 0x1c, 16 * padded_columns);
        put32(mip + 0x20, tail ? 256 : 65536);
        put32(mip + 0x24, padded_columns);
        put32(mip + 0x28, padded_rows);
        put32(mip + 0x2c, chain.slices);
        put32(mip + 0x30, columns);
        put32(mip + 0x34, rows);
        put32(mip + 0x38, 1024u >> level);
        put32(mip + 0x3c, chain.height >> level);
        put32(mip + 0x40, chain.slices);
        put32(mip + 0x44, 1);
        put32(mip + 0x48, tail ? 1 : 9);
        put32(mip + 0x60, 1);
        put32(mip + 0x64, tail);
    }
    return S_OK;
}

template<bool into>
long __stdcall bc64_copy(Computer* self, unsigned char* target, UINT plane, UINT subresource, const unsigned char* source,
                         UINT pitch, UINT slice_pitch)
{
    const bc64::Chain chain(self->desc);
    if (plane || subresource >= chain.levels * chain.slices || !target || !source) return E_INVALIDARG;
    const UINT level = subresource % chain.levels, slice = subresource / chain.levels;
    const u64 row_bytes = 16 * chain.columns(level);
    const u64 linear_bytes = u64{chain.rows(level) - 1} * pitch + row_bytes;
    if (pitch < row_bytes || (slice_pitch && slice_pitch < linear_bytes)) return E_INVALIDARG;
    const auto dest = reinterpret_cast<std::uintptr_t>(target);
    const auto src = reinterpret_cast<std::uintptr_t>(source);
    const auto dest_bytes = into ? chain.bytes() : linear_bytes;
    const auto src_bytes = into ? linear_bytes : chain.bytes();
    if (dest <= src ? src - dest < dest_bytes : dest - src < src_bytes) return E_INVALIDARG;
    if (!x2w_local::writable(target, dest_bytes) || !x2w_local::readable(source, src_bytes)) return E_FAIL;
    chain.copy<into>(level, slice, target, source, pitch);
    if constexpr (into) notify_texture_write(target, chain.bytes());
    return S_OK;
}
long __stdcall bc64_copy_from(Computer* self, unsigned char* target, UINT plane, UINT subresource,
                              UINT scarlett_option, const unsigned char* source, UINT pitch, UINT slice_pitch)
{
    // DirectXTex's Scarlett copy-out call passes zero between the subresource and the tiled base.
    // Reject unknown options before interpreting the remaining arguments or touching either buffer.
    if (scarlett_option != 0) return E_NOTIMPL;
    return bc64_copy<false>(self, target, plane, subresource, source, pitch, slice_pitch);
}

u64 __stdcall bc64_size(Computer* self) { return bc64::Chain(self->desc).bytes(); }
u64 __stdcall bc64_alignment(Computer*) { return 65536; }
u64 __stdcall bc64_offset(Computer* self, UINT plane, UINT level)
{
    const bc64::Chain chain(self->desc);
    return !plane && level < chain.levels ? chain.start(level) : 0;
}
u64 __stdcall bc64_element(Computer* self, UINT plane, UINT level, u64 x, UINT y, UINT slice, UINT sample, UINT* bit)
{
    if (bit) *bit = 0;
    const bc64::Chain chain(self->desc);
    if (plane || level >= chain.levels || slice >= chain.slices || sample ||
        x >= chain.columns(level) || y >= chain.rows(level)) return 0;
    return chain.address(level, static_cast<UINT>(x), y, slice);
}

long __stdcall bc64_tiling(Computer* self, UINT* tiles, D3D12_PACKED_MIP_INFO* packed, D3D12_TILE_SHAPE* shape,
                           UINT* count, UINT first, D3D12_SUBRESOURCE_TILING* tilings)
{
    const auto& desc = self->desc;
    const bc64::Chain chain(desc);
    const UINT slice_tiles = chain.slice_bytes() / bc64::tile_bytes;
    if (tiles) *tiles = slice_tiles * desc.DepthOrArraySize;
    if (packed) *packed = {static_cast<UINT8>(bc64::first_tail),
                           static_cast<UINT8>(desc.MipLevels - bc64::first_tail), 1, 0};
    if (shape) *shape = {bc64::tile_side, bc64::tile_side, 1};
    if (count) {
        const UINT subresources = UINT{desc.MipLevels} * desc.DepthOrArraySize;
        const UINT available = first < subresources ? subresources - first : 0;
        const UINT requested = (std::min)(*count, available);
        UINT written = 0;
        for (UINT i = 0; i < requested; ++i) {
            const UINT subresource = first + i;
            const UINT level = subresource % desc.MipLevels;
            if (level >= bc64::first_tail) continue;
            if (tilings) {
                const UINT start = (subresource / desc.MipLevels) * slice_tiles + chain.start(level) / bc64::tile_bytes;
                tilings[i] = {chain.tiles_wide(level), static_cast<UINT16>(chain.tiles_high(level)), 1, start};
            }
            ++written;
        }
        *count = written;
    }
    return S_OK;
}
