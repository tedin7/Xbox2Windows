// Experimental (docs/startup-runtime.md, #870): the names the x2w-noop log gives to command list slots
// (src/startup_stubs/command_list_slot_names.h). Plain standard C++; the table is checked against names
// written out here, slot by slot, for the slots the implementation already handles and for the runs
// between them, and the slots that must stay unnamed are listed. Synthetic: no title data.
#include "../src/startup_stubs/command_list_slot_names.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {
int failures;

void expect(bool ok, const std::string& what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

bool is(int slot, const char* want)
{
    const char* got = x2w_list_slots::name(slot);
    return got && std::strcmp(got, want) == 0;
}
}  // namespace

int main()
{
    // Slots the implementation in d3d12_xs_device.cpp already handles, with the desktop method each one is.
    const std::vector<std::pair<int, const char*>> handled = {
        {7, "GetDevice"},
        {9, "Close"},
        {12, "DrawInstanced"},
        {13, "DrawIndexedInstanced"},
        {20, "IASetPrimitiveTopology"},
        {21, "RSSetViewports"},
        {22, "RSSetScissorRects"},
        {23, "OMSetBlendFactor"},
        {25, "SetPipelineState"},
        {26, "ResourceBarrier"},
        {32, "SetDescriptorHeaps"},
        {34, "SetGraphicsRootSignature"},
        {36, "SetGraphicsRootDescriptorTable"},
        {40, "SetGraphicsRoot32BitConstants"},
        {42, "SetGraphicsRootConstantBufferView"},
        {44, "SetGraphicsRootShaderResourceView"},
        {47, "IASetIndexBuffer"},
        {48, "IASetVertexBuffers"},
        {50, "OMSetRenderTargets"},
        {52, "ClearRenderTargetView"},
        {61, "ExecuteIndirect"},
        {139, "Reset"},
    };
    for (const auto& entry : handled) expect(is(entry.first, entry.second), "slot " + std::to_string(entry.first) + " is " + entry.second);

    // The runs between the handled slots take the desktop order.
    const std::vector<std::pair<int, const char*>> between = {
        {0, "QueryInterface"}, {1, "AddRef"}, {2, "Release"}, {3, "GetPrivateData"}, {4, "SetPrivateData"},
        {5, "SetPrivateDataInterface"}, {6, "SetName"}, {8, "GetType"},
        {14, "Dispatch"}, {15, "CopyBufferRegion"}, {16, "CopyTextureRegion"}, {17, "CopyResource"}, {18, "CopyTiles"},
        {19, "ResolveSubresource"}, {24, "OMSetStencilRef"},
    };
    for (const auto& entry : between) expect(is(entry.first, entry.second), "slot " + std::to_string(entry.first) + " is " + entry.second);

    // From SetDescriptorHeaps on the console slot is the desktop slot plus four: 28 desktop methods in a row.
    const std::vector<const char*> shifted = {
        "SetDescriptorHeaps", "SetComputeRootSignature", "SetGraphicsRootSignature", "SetComputeRootDescriptorTable",
        "SetGraphicsRootDescriptorTable", "SetComputeRoot32BitConstant", "SetGraphicsRoot32BitConstant",
        "SetComputeRoot32BitConstants", "SetGraphicsRoot32BitConstants", "SetComputeRootConstantBufferView",
        "SetGraphicsRootConstantBufferView", "SetComputeRootShaderResourceView", "SetGraphicsRootShaderResourceView",
        "SetComputeRootUnorderedAccessView", "SetGraphicsRootUnorderedAccessView", "IASetIndexBuffer", "IASetVertexBuffers",
        "SOSetTargets", "OMSetRenderTargets", "ClearDepthStencilView", "ClearRenderTargetView",
        "ClearUnorderedAccessViewUint", "ClearUnorderedAccessViewFloat", "DiscardResource", "BeginQuery", "EndQuery",
        "ResolveQueryData", "SetPredication"};
    expect(shifted.size() == 28, "the shifted run has 28 methods");
    for (std::size_t i = 0; i < shifted.size(); ++i)
        expect(is(32 + static_cast<int>(i), shifted[i]), "slot " + std::to_string(32 + i) + " is " + shifted[i]);
    // The run ends at 59 with the desktop slot 55; the desktop methods after it (56 to 59, plus four) are not named.
    expect(is(59, "SetPredication"), "slot 59 is the desktop slot 55 plus four");

    // Slots that are not known stay unnamed: a name there would be a guess.
    // 10 and 11: this console list resets through 139. 27 to 31: the remaining unconfirmed methods before the shift.
    // Other slots from 60 up have unconfirmed meanings. -1 and 4096: outside the table.
    std::vector<int> unknown = {10, 11, 27, 28, 29, 30, 31, 60, 62, 63, 64, 66, 68, 70, 71, 72, 74, 75, 100, 126,
                                131, 132, 138, 140, 142, 146, 255, 256, 999, 4096, -1, -100};
    for (const int slot : unknown) expect(x2w_list_slots::name(slot) == nullptr, "slot " + std::to_string(slot) + " has no name");

    // Names are unique within the table and never empty, so a log line says which method it was.
    std::vector<std::string> seen;
    for (int slot = 0; slot < 200; ++slot) {
        const char* got = x2w_list_slots::name(slot);
        if (!got) continue;
        expect(*got != '\0', "slot " + std::to_string(slot) + " has a non-empty name");
        for (const std::string& earlier : seen) expect(earlier != got, std::string("the name ") + got + " is used once");
        seen.push_back(got);
    }
    expect(seen.size() == 10 + 15 + 28 + 2, "the named slots are 0 to 9, 12 to 26, 32 to 59, 61 and 139");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("command list slot name checks passed\n");
    return 0;
}
