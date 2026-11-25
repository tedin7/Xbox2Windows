#pragma once

namespace x2w_startup {
inline constexpr char indexed_indirect_guard[] = R"guard(
// Validation operates on the desktop GPU buffers used by the draw.
cbuffer Bounds : register(b0) {
    uint arg_offset, arg_size, index_offset, index_size;
    uint index_backing_size, index_element_size, vertices, reserved;
};
ByteAddressBuffer guest_args : register(t0);
ByteAddressBuffer guest_indices : register(t1);
RWByteAddressBuffer verified : register(u0);

[numthreads(1, 1, 1)]
void guard() {
    verified.Store(0, 0); // count for ExecuteIndirect
    if ((arg_offset & 3) || arg_size < 20 || arg_offset > arg_size - 20) return;
    if (index_element_size != 2 && index_element_size != 4) return;
    if (index_offset % index_element_size || index_size % index_element_size ||
        index_offset > index_backing_size || index_size > index_backing_size - index_offset) return;
    uint4 args = guest_args.Load4(arg_offset);
    uint start_instance = guest_args.Load(arg_offset + 16);
    uint max_indices = index_size / index_element_size;
    if (args.x < 3 || args.x > 4096 || args.x % 3 || args.y != 1 || start_instance ||
        args.z > max_indices || args.x > max_indices - args.z) return;
    int base = asint(args.w);
    for (uint i = 0; i < args.x; ++i) {
        uint at = index_offset + (args.z + i) * index_element_size;
        uint word_at = at & ~3u;
        if (index_backing_size < 4 || word_at > index_backing_size - 4) return;
        uint idx = guest_indices.Load(word_at);
        if (index_element_size == 2) idx = (idx >> ((at & 2) * 8)) & 65535;
        uint adjusted = idx + args.w;
        if ((base < 0 && adjusted > idx) || (base >= 0 && adjusted < idx) || adjusted >= vertices) return;
    }
    verified.Store(4, args.w);     // graphic root constant: translated base vertex
    verified.Store4(8, args);      // D3D12_DRAW_INDEXED_ARGUMENTS
    verified.Store(24, start_instance);
    verified.Store(0, 1);
}
)guard";
}
