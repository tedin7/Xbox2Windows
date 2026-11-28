#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "startup_stream_setup.h"

extern "C" __declspec(dllimport) void x2w_d3d12_root_srv_copies(LONG* skipped, LONG* copied);

// A translated draw that reads a graphics root SRV set through list slot 44, at a title address without an upload
// mapping (X2W_TEST_ROOT_SRV_DRAW=write-after). Slot 44 sees the buffer holding red; the title then writes green and
// draws. The translated draw reads the SRV from its title address at the draw, so it draws green whether or not slot 44
// also copied the bytes for the desktop binding (skipped with X2W_D3D12_ROOT_SRV_AT_DRAW=1). After the draw is recorded
// it prints the stand-in's copy counts (x2w_d3d12_root_srv_copies). The stream root's parameter 0 is a root SRV in this
// mode (startup_stream_setup.cpp).
int draw_root_srv(void* list)
{
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_ROOT_SRV_DRAW", mode, sizeof(mode));
    if (std::strcmp(mode, "write-after")) return 2;
    const StreamWord root = reinterpret_cast<std::uintptr_t>(stream_root_signature());
    if (!root) return 40;
    // Title memory outside any resource: slot 44 finds no upload mapping.
    auto* colour = static_cast<float*>(VirtualAlloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!colour) return 41;
    const float red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1};
    std::memcpy(colour, red, sizeof(red));
    invoke(list, 34, root);                                         // SetGraphicsRootSignature
    invoke(list, 44, 0, reinterpret_cast<std::uintptr_t>(colour));  // SetGraphicsRootShaderResourceView(0, address)
    std::memcpy(colour, green, sizeof(green));
    invoke(list, 12, 3, 1);  // DrawInstanced(3, 1)
    // The stand-in's slot 44 copy counts after the draw was recorded, for the check script.
    LONG skipped = -1, copied = -1;
    x2w_d3d12_root_srv_copies(&skipped, &copied);
    std::printf("root SRV copies: skipped %ld, copied %ld\n", skipped, copied);
    return 0;  // The buffer stays allocated until the process ends.
}
