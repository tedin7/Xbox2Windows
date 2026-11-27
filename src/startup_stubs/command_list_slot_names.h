// Experimental (docs/startup-runtime.md, #870): names for the slots of the command list, used by the
// `x2w-noop` log so a line reads "CommandList::slot14(Dispatch)" instead of "CommandList::slot14".
// Plain standard C++ with no Windows types, so the table can be compiled and tested on its own.
//
// What is claimed, and why:
// - Slots 0 to 9 and 12 to 25 follow the public desktop ID3D12GraphicsCommandList order. The
//   implemented slots agree with it at both ends of every run: 7 (GetDevice), 9 (Close), 12 and 13
//   (the draws), 20 to 23 (topology, viewports, scissors, blend factor) and 25 (pipeline state), so
//   the slots between them (0 to 6, 8, 14 to 19, 24) take the desktop names.
// - Slots 10 and 11 are not named. The desktop order puts Reset and ClearState there, but this
//   console list resets through slot 139, so the desktop order is not known to hold at 10 and 11.
// - Slot 26 accepts the UAV form of ResourceBarrier. Slots 27 to 31 are not named here. The slots
//   that follow are the desktop ones moved up by four
//   (32 is SetDescriptorHeaps, desktop 28), so four console methods sit somewhere in 26 to 31.
//   Slots 28, 29 and 30 are BeginQuery, EndQuery and ResolveQueryData candidates. The census
//   counts calls that name a wrapped heap from device slot 39 with a compatible type and in-range
//   indices. Resolve additionally needs a buffer wrapper. Fits measure compatibility with the
//   proposed argument layout. The counts alone cannot establish slot identities or semantics.
//   X2W_QUERY_RESULTS=forward (also X2W_D3D12_QUERY_SLOTS=1) opts into desktop occlusion queries.
//   X2W_QUERY_RESULTS=visible opts into synthetic results in writable placed buffer memory.
// - Slots 32 to 59 are named as the desktop slots 28 to 55 plus four. Implemented and checked
//   against the desktop order: 32, 34, 36, 40, 42, 44, 47, 48, 50 and 52. Past 52 the rule is not
//   checked: ExecuteIndirect (desktop 59) is at 61, desktop plus two, so the run of four has ended
//   by then. The 56 to 58 names below are kept as the rule gives them.
// - Slots 61 (ExecuteIndirect) and 139 (Reset) are implemented in d3d12_xs_device.cpp.
// - Other slots have no confirmed meaning.
#pragma once

#include <cstddef>

namespace x2w_list_slots {

// The public desktop ID3D12GraphicsCommandList methods from Close (9) to ExecuteIndirect (59).
inline constexpr const char* desktop_names[] = {
    "Close", "Reset", "ClearState", "DrawInstanced", "DrawIndexedInstanced", "Dispatch", "CopyBufferRegion",
    "CopyTextureRegion", "CopyResource", "CopyTiles", "ResolveSubresource", "IASetPrimitiveTopology",
    "RSSetViewports", "RSSetScissorRects", "OMSetBlendFactor", "OMSetStencilRef", "SetPipelineState",
    "ResourceBarrier", "ExecuteBundle", "SetDescriptorHeaps", "SetComputeRootSignature",
    "SetGraphicsRootSignature", "SetComputeRootDescriptorTable", "SetGraphicsRootDescriptorTable",
    "SetComputeRoot32BitConstant", "SetGraphicsRoot32BitConstant", "SetComputeRoot32BitConstants",
    "SetGraphicsRoot32BitConstants", "SetComputeRootConstantBufferView", "SetGraphicsRootConstantBufferView",
    "SetComputeRootShaderResourceView", "SetGraphicsRootShaderResourceView", "SetComputeRootUnorderedAccessView",
    "SetGraphicsRootUnorderedAccessView", "IASetIndexBuffer", "IASetVertexBuffers", "SOSetTargets",
    "OMSetRenderTargets", "ClearDepthStencilView", "ClearRenderTargetView", "ClearUnorderedAccessViewUint",
    "ClearUnorderedAccessViewFloat", "DiscardResource", "BeginQuery", "EndQuery", "ResolveQueryData",
    "SetPredication", "SetMarker", "BeginEvent", "EndEvent", "ExecuteIndirect"};
inline constexpr int first_desktop_method = 9;
inline constexpr int desktop_method_count = static_cast<int>(sizeof(desktop_names) / sizeof(desktop_names[0]));
static_assert(desktop_method_count == 51, "ID3D12GraphicsCommandList has 51 methods of its own");

inline constexpr int console_shift = 4;  // console slot = desktop slot + 4, from SetDescriptorHeaps on
inline constexpr int resource_barrier_slot = 26;
inline constexpr int first_shifted_slot = 32;
inline constexpr int last_shifted_slot = 59;
inline constexpr int execute_indirect_slot = 61;
inline constexpr int reset_slot = 139;

// The name of a command list slot, or nullptr when it is not known.
inline const char* name(int slot) noexcept
{
    const auto desktop = [](int index) { return desktop_names[index - first_desktop_method]; };
    static constexpr const char* iunknown[] = {"QueryInterface", "AddRef", "Release", "GetPrivateData", "SetPrivateData",
                                                "SetPrivateDataInterface", "SetName", "GetDevice", "GetType"};
    if (slot >= 0 && slot < first_desktop_method) return iunknown[slot];
    if (slot == first_desktop_method || (slot >= 12 && slot <= resource_barrier_slot)) return desktop(slot);
    if (slot >= first_shifted_slot && slot <= last_shifted_slot) return desktop(slot - console_shift);
    if (slot == execute_indirect_slot) return "ExecuteIndirect";
    if (slot == reset_slot) return "Reset";
    return nullptr;
}

}  // namespace x2w_list_slots
