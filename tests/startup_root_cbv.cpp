#include <windows.h>

#include <cstdint>
#include <cstring>

#include "startup_stream_setup.h"

// Draws with a translated pixel shader that reads four floats through root parameter 0. X2W_TEST_ROOT_CBV
// selects what the title does with that parameter before the draw:
//   bound       SetGraphicsRootConstantBufferView(0, address of the colour)
//   unbound     no call
//   null        SetGraphicsRootConstantBufferView(0, address) then (0, 0) to clear it
//   same_root  repeat the original root signature after binding
//   switch_root change to a different signature and back without rebinding the descriptor
//   wrong_type  parameter 0 is declared as an SRV; the same call is made twice
//   placed_copy  the colour is in a placed buffer that does not alias title memory (an address that is not 64 KiB aligned in
//                its allocation); a recorded CopyBufferRegion from an aliased placed buffer writes the new colour into it, so
//                title memory still holds the old one. The draw must read the new colour.
int draw_root_cbv(void* guest, void* list)
{
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_ROOT_CBV", mode, sizeof(mode));
    alignas(256) static float colour[4] = {1.0f, 1.0f, 0.0f, 1.0f};
    auto address = static_cast<StreamWord>(reinterpret_cast<std::uintptr_t>(colour));
    if (!std::strcmp(mode, "placed_copy")) {
        // One 128 KiB allocation: the source buffer at its 64 KiB aligned start (aliased), the destination 0x1100 bytes in
        // (not 64 KiB aligned, so the placed buffer falls back to a committed copy that title memory does not back).
        auto* memory = static_cast<unsigned char*>(VirtualAlloc(nullptr, 1u << 17, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!memory) return 41;
        const float fresh[4] = {0.0f, 1.0f, 1.0f, 1.0f};
        std::memcpy(memory, fresh, sizeof(fresh));
        std::memcpy(memory + 0x1100, colour, sizeof(colour));
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        const auto iid = reinterpret_cast<StreamWord>(&__uuidof(ID3D12Resource));
        void* source = nullptr;
        void* destination = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, reinterpret_cast<StreamWord>(memory), reinterpret_cast<StreamWord>(&desc),
                D3D12_RESOURCE_STATE_COMMON, 0, iid, reinterpret_cast<StreamWord>(&source)))) || !source) return 77;  // no existing-heap support
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, reinterpret_cast<StreamWord>(memory + 0x1100), reinterpret_cast<StreamWord>(&desc),
                D3D12_RESOURCE_STATE_COMMON, 0, iid, reinterpret_cast<StreamWord>(&destination)))) || !destination) return 42;
        invoke(list, 15, reinterpret_cast<StreamWord>(destination), 0, reinterpret_cast<StreamWord>(source), 0, 256);  // CopyBufferRegion
        address = reinterpret_cast<StreamWord>(memory + 0x1100);
    }
    const StreamWord root = reinterpret_cast<std::uintptr_t>(stream_root_signature());
    if (!root) return 40;
    invoke(list, 34, root);  // SetGraphicsRootSignature
    if (!std::strcmp(mode, "bound")) {
        invoke(list, 42, 0, address);
    } else if (!std::strcmp(mode, "null")) {
        invoke(list, 42, 0, address);
        invoke(list, 42, 0, 0);
    } else if (!std::strcmp(mode, "same_root")) {
        invoke(list, 42, 0, address);
        invoke(list, 34, root);
    } else if (!std::strcmp(mode, "switch_root")) {
        const StreamWord alternate = reinterpret_cast<std::uintptr_t>(stream_alternate_root_signature());
        if (!alternate) return 40;
        invoke(list, 42, 0, address);
        invoke(list, 34, alternate);
        invoke(list, 34, root);
    } else if (!std::strcmp(mode, "wrong_type")) {
        invoke(list, 42, 0, address);
        invoke(list, 42, 0, address);
    } else if (!std::strcmp(mode, "placed_copy")) {
        invoke(list, 42, 0, address);
    } else if (std::strcmp(mode, "unbound")) {
        return 2;
    }
    invoke(list, 12, 3, 1);
    return 0;
}
