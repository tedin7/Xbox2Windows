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
int draw_root_cbv(void* list)
{
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_ROOT_CBV", mode, sizeof(mode));
    alignas(256) static const float colour[4] = {1.0f, 1.0f, 0.0f, 1.0f};
    const auto address = static_cast<StreamWord>(reinterpret_cast<std::uintptr_t>(colour));
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
    } else if (std::strcmp(mode, "unbound")) {
        return 2;
    }
    invoke(list, 12, 3, 1);
    return 0;
}
