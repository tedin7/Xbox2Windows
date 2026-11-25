// Experimental (docs/startup-runtime.md, #870): the argument checks for command list slot 16,
// CopyTextureRegion(dst, x, y, z, src, box). Plain standard C++ so a Linux fixture can run it; the
// handler (list_copy_texture.inl) maps the title's resources and forwards only what passes here.
// A call the desktop runtime would reject is refused with a named reason instead.
#pragma once

#include <cstdint>
#include <initializer_list>

namespace x2w_list_copy_texture {

enum class Kind : std::uint32_t { subresource = 0, placed_footprint = 1 };

// What the check needs to know about one side of the copy. For a subresource side, width/height/depth are
// the subresource's size in blocks (texels for an uncompressed format). For a footprint side, the footprint
// size in blocks, the row pitch in bytes, the byte offset and the bytes per block are used, and
// resource_bytes is the size of the buffer.
struct Side {
    Kind kind;
    std::uint32_t width, height, depth;
    std::uint64_t offset;
    std::uint32_t row_pitch, bytes_per_block;
    std::uint64_t resource_bytes;
    std::uint32_t samples;           // sample count; a footprint (buffer) side is always 1
    std::uint32_t block_w, block_h;  // texels per block (4 x 4 for block-compressed formats, else 1 x 1)
    std::uint32_t format;            // DXGI_FORMAT value of the subresource, or of the footprint
    // A subresource side: its index and the resource's counts, for D3D12's
    // index = mip + slice * mips + plane * mips * array_size.
    std::uint32_t subresource, mips, array_size, planes;
    bool depth_stencil = false;
};

struct Subresource {
    std::uint32_t mip, slice, plane;
};

// Splits a subresource index by the desktop's math. False when the index is outside
// mips * array_size * planes (or a count is zero), so no mip, slice or plane aliases another.
inline bool decode_subresource(std::uint32_t index, std::uint32_t mips, std::uint32_t array_size, std::uint32_t planes,
                               Subresource& out) noexcept
{
    if (!mips || !array_size || !planes) return false;
    const std::uint64_t per_plane = std::uint64_t{mips} * array_size;
    if (index >= per_plane * planes) return false;
    out = {static_cast<std::uint32_t>(index % mips), static_cast<std::uint32_t>((index / mips) % array_size),
           static_cast<std::uint32_t>(index / per_plane)};
    return true;
}

struct Box {
    std::uint32_t left, top, front, right, bottom, back;
};

enum class Verdict {
    ok,
    buffer_to_buffer,   // a footprint on both sides: the desktop wants CopyBufferRegion
    bad_box,            // an empty or inverted source box
    source_overflow,    // the source box leaves the source subresource
    dest_overflow,      // the destination origin plus box size leaves the destination subresource
    bad_pitch,          // a footprint row pitch below one row, or not a multiple of 256
    bad_offset,         // a footprint offset that is not a multiple of 512
    footprint_overflow, // the footprint reaches past the end of its buffer
    sample_mismatch,    // the two sides differ in sample count
    bad_alignment,      // a block-compressed copy that does not start on a block
    bad_subresource,    // an index outside the resource's mips * array slices * planes
    format_mismatch,    // the two sides name different formats
    partial_multisample, // a multisampled copy that is not the whole subresource at the origin
    partial_depth_stencil
};

inline const char* name(Verdict v) noexcept
{
    switch (v) {
    case Verdict::ok: return "ok";
    case Verdict::buffer_to_buffer: return "buffer_to_buffer";
    case Verdict::bad_box: return "bad_box";
    case Verdict::source_overflow: return "source_overflow";
    case Verdict::dest_overflow: return "dest_overflow";
    case Verdict::bad_pitch: return "bad_pitch";
    case Verdict::bad_offset: return "bad_offset";
    case Verdict::footprint_overflow: return "footprint_overflow";
    case Verdict::sample_mismatch: return "sample_mismatch";
    case Verdict::bad_alignment: return "bad_alignment";
    case Verdict::bad_subresource: return "bad_subresource";
    case Verdict::format_mismatch: return "format_mismatch";
    case Verdict::partial_multisample: return "partial_multisample";
    case Verdict::partial_depth_stencil: return "partial_depth_stencil";
    }
    return "unknown";
}

// a * b and a + b with the overflow reported, so no step of the footprint arithmetic wraps.
inline bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept
{
    if (a && b > ~std::uint64_t{0} / a) return false;
    out = a * b;
    return true;
}
inline bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept
{
    if (b > ~std::uint64_t{0} - a) return false;
    out = a + b;
    return true;
}

enum class End { ok, over_pitch, overflow };

// Bytes a footprint reaches for a w x h x d copy that starts at texel (ox, oy, oz) inside it, from its offset
// (the last row is not padded; slices are as tall as the footprint, not the copy). Every multiply and add is checked.
inline End footprint_end(const Side& s, std::uint32_t w, std::uint32_t h, std::uint32_t d, std::uint32_t ox, std::uint32_t oy,
                         std::uint32_t oz, std::uint64_t& end) noexcept
{
    if (!w || !h || !d) { end = s.offset; return End::ok; }
    const std::uint64_t blocks_w = (std::uint64_t{w} + s.block_w - 1) / s.block_w;
    const std::uint64_t rows = (std::uint64_t{h} + s.block_h - 1) / s.block_h;
    const std::uint64_t start_row = oy / s.block_h;
    const std::uint64_t slice_rows = (std::uint64_t{s.height} + s.block_h - 1) / s.block_h;
    std::uint64_t row = 0, start_x = 0, row_end = 0;
    if (!checked_mul(blocks_w, s.bytes_per_block, row) || !checked_mul(ox / s.block_w, s.bytes_per_block, start_x) ||
        !checked_add(start_x, row, row_end))
        return End::overflow;
    if (row_end > s.row_pitch) return End::over_pitch;
    std::uint64_t last_row = 0, row_bytes = 0, slice = 0, slice_bytes = 0, slices = 0, slices_bytes = 0, total = 0;
    if (!checked_add(start_row, rows - 1, last_row) || !checked_mul(s.row_pitch, last_row, row_bytes) ||
        !checked_add(row_bytes, row_end, slice) || !checked_mul(s.row_pitch, slice_rows, slice_bytes) ||
        !checked_add(oz, d - 1, slices) || !checked_mul(slice_bytes, slices, slices_bytes) ||
        !checked_add(slices_bytes, slice, total) || !checked_add(s.offset, total, end))
        return End::overflow;
    return End::ok;
}

// box == nullptr copies the whole source (for a footprint source, the whole footprint).
inline Verdict check(const Side& dst, std::uint32_t x, std::uint32_t y, std::uint32_t z, const Side& src, const Box* box) noexcept
{
    if (dst.kind == Kind::placed_footprint && src.kind == Kind::placed_footprint) return Verdict::buffer_to_buffer;
    for (const Side* s : {&dst, &src}) {
        Subresource unused{};
        if (s->kind == Kind::subresource && !decode_subresource(s->subresource, s->mips, s->array_size, s->planes, unused))
            return Verdict::bad_subresource;
    }
    if (dst.samples != src.samples) return Verdict::sample_mismatch;
    if (dst.format != src.format) return Verdict::format_mismatch;
    const Box whole{0, 0, 0, src.width, src.height, src.depth};
    const Box use = box ? *box : whole;
    if (use.left >= use.right || use.top >= use.bottom || use.front >= use.back) return Verdict::bad_box;
    if (use.right > src.width || use.bottom > src.height || use.back > src.depth) return Verdict::source_overflow;
    const Side& tex = dst.kind == Kind::subresource ? dst : src;  // the side that knows the block size
    if (x % tex.block_w || use.left % tex.block_w || y % tex.block_h || use.top % tex.block_h) return Verdict::bad_alignment;
    const std::uint32_t w = use.right - use.left, h = use.bottom - use.top, d = use.back - use.front;
    // D3D12 requires a null source box even if an explicit box would cover the whole subresource.
    const bool depth_stencil = src.depth_stencil || dst.depth_stencil;
    if ((src.samples > 1 || depth_stencil) &&
        (box || x || y || z || w != dst.width || h != dst.height || d != dst.depth))
        return src.samples > 1 ? Verdict::partial_multisample : Verdict::partial_depth_stencil;
    if (std::uint64_t{x} + w > dst.width || std::uint64_t{y} + h > dst.height || std::uint64_t{z} + d > dst.depth)
        return Verdict::dest_overflow;
    for (const Side* s : {&dst, &src}) {
        if (s->kind != Kind::placed_footprint) continue;
        if (!s->row_pitch || s->row_pitch % 256) return Verdict::bad_pitch;
        if (s->offset % 512) return Verdict::bad_offset;
        std::uint64_t end = 0;
        const bool is_dst = s == &dst;
        const End reach = footprint_end(*s, w, h, d, is_dst ? x : use.left, is_dst ? y : use.top, is_dst ? z : use.front, end);
        if (reach == End::over_pitch) return Verdict::bad_pitch;
        if (reach == End::overflow) return Verdict::footprint_overflow;
        if (end > s->resource_bytes) return Verdict::footprint_overflow;
    }
    return Verdict::ok;
}

}  // namespace x2w_list_copy_texture
