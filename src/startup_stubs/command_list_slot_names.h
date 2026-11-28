// Experimental (docs/startup-runtime.md, #870): names for the slots of the command list, used by the
// `x2w-noop` log so a line reads "CommandList::slot14(Dispatch)" instead of "CommandList::slot14".
// Plain standard C++ with no Windows types, so the table can be compiled and tested on its own.
//
// The 0 to 61, 139 and 148 rows were checked against the GXDK 261000
// ID3D12GraphicsCommandList8 function table through a local oracle outside the repository. The
// header itself is not copied into this repository; only the public desktop method names and the
// slot numbers are recorded.
// - Slots 0 to 8 are IUnknown, ID3D12Object and ID3D12DeviceChild.
// - 9 Close, 10 Reset and 11 ClearState.
// - 12 to 27 follow the desktop ID3D12GraphicsCommandList order through ExecuteBundle.
// - 28 to 31 are BeginQuery, EndQuery, ResolveQueryData and SetPredication. They sit before the
//   desktop SetDescriptorHeaps rather than after DiscardResource.
// - 32 to 55 are the desktop slots 28 to 51 plus four: SetDescriptorHeaps through DiscardResource.
// - 56 to 58 are the unshifted desktop SetMarker, BeginEvent and EndEvent.
// - 59 and 60 are console-only marker methods and stay unnamed in this repository.
// - 61 is ExecuteIndirect (desktop 59, plus two).
// - 139 is the console reset path the title calls. The base Reset is at 10; the log calls the
//   console path "Reset (console)" so both rows have distinct names.
// - 62 to 138 and 140 and above are console-only methods. They stay unnamed because their names
//   and signatures are not needed by the current log. Slot 148 is the allocation path used by
//   list_allocate_upload; it stays unnamed here for the same reason.
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
inline constexpr int last_shifted_slot = 55;
inline constexpr int first_desktop_marker_slot = 56;
inline constexpr int last_desktop_marker_slot = 58;
inline constexpr int execute_indirect_slot = 61;
inline constexpr int reset_slot = 139;

// The name of a command list slot, or nullptr when it is not known.
inline const char* name(int slot) noexcept
{
    const auto desktop = [](int index) { return desktop_names[index - first_desktop_method]; };
    static constexpr const char* iunknown[] = {"QueryInterface", "AddRef", "Release", "GetPrivateData", "SetPrivateData",
                                                "SetPrivateDataInterface", "SetName", "GetDevice", "GetType"};
    static constexpr const char* query[] = {"BeginQuery", "EndQuery", "ResolveQueryData", "SetPredication"};
    if (slot >= 0 && slot < first_desktop_method) return iunknown[slot];
    if (slot >= first_desktop_method && slot <= 27) return desktop(slot);
    if (slot >= 28 && slot <= 31) return query[slot - 28];
    if (slot >= first_shifted_slot && slot <= last_shifted_slot) return desktop(slot - console_shift);
    if (slot >= first_desktop_marker_slot && slot <= last_desktop_marker_slot) return desktop(slot);
    if (slot == execute_indirect_slot) return "ExecuteIndirect";
    if (slot == reset_slot) return "Reset (console)";
    return nullptr;
}

}  // namespace x2w_list_slots
