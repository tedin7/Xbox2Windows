#pragma once

#include <d3d12.h>

#include <cstddef>

namespace x2w_replay_slots {

inline constexpr char setting[] = "X2W_REPLAY_SLOT_LAYOUTS";
inline constexpr int slot_count = 512;
inline constexpr int device = 0, list = 4;

struct Layout {
    int kind, slot;
    const char* name;
    bool known_memory;  // false means the recorder has no declared argument memory
    std::size_t memory[5];  // exact top-level byte counts, from rdx onward
    int wrapper_output = -1;
    int follow = -1;  // argument containing a pipeline stream pointer and size
};

// Slot -> stand-in name -> argument layout (after this). Pointer sizes are for x64.
// An opaque no-op has no established console layout. Its name describes the
// fallback entry when DEVICE_NOOP or LIST_NOOP is enabled.
// Device 70  CreateMemoryShaderResourceView(desc*, viewAndAddress*, CPU handle).
// Device 71  OpaqueNoop71(unknown).
// Device 72  OpaqueNoop72(unknown).
// Device 73  CreateCommandListX(desc16*, allocator, pipeline, IID*, out list**).
// Device 74  OpaqueNoop74(unknown).
// Device 75  OpaqueNoop75(unknown).
// Device 80  WaitFrameEvent(type u32, timeout u32, reserved, flags u32, out token u64*).
// Device 85  OpaqueNoop85(unknown).
// Device 86  CreatePipelineStateStreamX(stream desc*, IID*, out pipeline**).
// Device 101 OpaqueNoop101(unknown).
// Device 126 OpaqueNoop126(unknown).
// Device 129 OpaqueNoop129(unknown).
// Device 131 OpaqueNoop131(unknown).
// Device 132 OpaqueNoop132(unknown).
// List 28    BeginQuery(heap, type u32, index u32).
// List 29    EndQuery(heap, type u32, index u32).
// List 30    ResolveQueryData(heap, type u32, start u32, count u32, buffer, offset u64).
//            These three bindings are hypotheses enabled by X2W_D3D12_QUERY_SLOTS.
// List 142   OpaqueNoop142(unknown).
// List 146   BarrierLikeNoop(count?, array?, unknown tail). The separate capture
//            probe samples r8 as an array. The replay stand-in consumes no array.
// List 148   AllocateUpload(bytes u64, alignment u64) -> CPU pointer.
inline constexpr Layout layouts[] = {
    {device, 70, "CreateMemoryShaderResourceView", true, {sizeof(D3D12_RESOURCE_DESC), 0x30}},
    {device, 71, "OpaqueNoop71", false, {}},
    {device, 72, "OpaqueNoop72", false, {}},
    {device, 73, "CreateCommandListX", true, {16, 0, 0, sizeof(GUID), sizeof(void*)}, 4},
    {device, 74, "OpaqueNoop74", false, {}},
    {device, 75, "OpaqueNoop75", false, {}},
    {device, 80, "WaitFrameEvent", true, {0, 0, 0, 0, sizeof(UINT64)}},
    {device, 85, "OpaqueNoop85", false, {}},
    {device, 86, "CreatePipelineStateStreamX", true, {sizeof(D3D12_PIPELINE_STATE_STREAM_DESC), sizeof(GUID), sizeof(void*)}, 2, 0},
    {device, 101, "OpaqueNoop101", false, {}},
    {device, 126, "OpaqueNoop126", false, {}},
    {device, 129, "OpaqueNoop129", false, {}},
    {device, 131, "OpaqueNoop131", false, {}},
    {device, 132, "OpaqueNoop132", false, {}},
    {list, 28, "BeginQuery", true, {}},
    {list, 29, "EndQuery", true, {}},
    {list, 30, "ResolveQueryData", true, {}},
    {list, 142, "OpaqueNoop142", false, {}},
    {list, 146, "BarrierLikeNoop", true, {}},
    {list, 148, "AllocateUpload", true, {}},
};

inline constexpr const Layout* find(int kind, int slot)
{
    for (const auto& layout : layouts)
        if (layout.kind == kind && layout.slot == slot) return &layout;
    return nullptr;
}

// The argument that receives a created or queried object, or -1: QueryInterface's on every kind, the device creations
// the recorder captures with create()/output() (d3d12_record.inl), and the declared wrapper outputs. The player gives
// one the file does not hold a buffer of its own instead of the recorded process's (stack) address, and does not play
// such a call when it was recorded as a failure.
// An argument the stand-in reads as memory (the recorder's declared captures): the player must not pass the recorded
// process's address for it when the file holds no copy.
inline constexpr bool declared_input(int kind, int slot, int arg)
{
    if (slot == 0) return arg == 0;  // QueryInterface's IID
    if (const Layout* layout = find(kind, slot); layout && layout->known_memory)
        return arg < 5 && layout->memory[arg] && arg != layout->wrapper_output;
    if (kind != device) return false;
    switch (slot) {
    case 57: case 67: return arg == 0 || arg == 1;
    case 55: return arg == 2 || arg == 3;
    case 25: return arg == 3;
    case 108: return arg == 3 || arg == 4;
    case 133: return arg == 1 || arg == 4;
    default: return false;
    }
}

inline constexpr int output_arg(int kind, int slot)
{
    if (slot == 0) return 1;  // QueryInterface(riid, out)
    if (const Layout* layout = find(kind, slot); layout && layout->wrapper_output >= 0) return layout->wrapper_output;
    if (kind != device) return -1;
    switch (slot) {
    case 8: case 9: case 14: case 28: case 39: case 57: case 67: return 2;
    case 36: return 3;
    case 16: case 30: case 55: return 4;
    case 12: case 133: return 5;
    case 27: case 29: case 76: return 6;
    case 127: return 9;
    default: return -1;
    }
}

}  // namespace x2w_replay_slots
