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

// Before the draw, the title sets root constants with SetGraphicsRoot32BitConstant (list slot 38). The check feeds
// one word, `word`, and X2W_TEST_ROOT_CONSTANT names what is done with it:
//   set          root signature, then (parameter 0, word, offset 0)
//   overwrite    a first value at (0, 0), then the word at (0, 0)
//   offset3      the word at (0, 3)
//   preserve     slot 40 sets (0, 0..1) with word at offset 0, then slot 38 writes (0, 3): both offsets stay intact
//   corner       the word at the last stored slot, (15, 7)
//   outside      (16, word, 0), (0xffffffff, word, 0), (0, word, 8) and (0, word, 0xffffffff): all dropped
//   same_root    the word at (0, 0), then the same root signature again: kept
//   switch_root  the word at (0, 0), then a different root signature and back: cleared
//   unbound      no call
int draw_root_constant(void* list, std::uint32_t word)
{
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_ROOT_CONSTANT", mode, sizeof(mode));
    const StreamWord root = reinterpret_cast<std::uintptr_t>(stream_root_signature());
    if (!root) return 40;
    invoke(list, 34, root);  // SetGraphicsRootSignature
    if (!std::strcmp(mode, "set")) {
        invoke(list, 38, 0, word, 0);
    } else if (!std::strcmp(mode, "overwrite")) {
        invoke(list, 38, 0, 0x5a5a5a5au, 0);
        invoke(list, 38, 0, word, 0);
    } else if (!std::strcmp(mode, "offset3")) {
        invoke(list, 38, 0, word, 3);
    } else if (!std::strcmp(mode, "preserve")) {
        const std::uint32_t pair[2] = {word, 0x01000000u};
        invoke(list, 40, 0, 2, reinterpret_cast<StreamWord>(pair), 0);
        invoke(list, 38, 0, 0x01800000u, 3);
    } else if (!std::strcmp(mode, "corner")) {
        invoke(list, 38, 15, word, 7);
    } else if (!std::strcmp(mode, "outside")) {
        invoke(list, 38, 16, word, 0);
        invoke(list, 38, 0xffffffffu, word, 0);
        invoke(list, 38, 0, word, 8);
        invoke(list, 38, 0, word, 0xffffffffu);
    } else if (!std::strcmp(mode, "same_root")) {
        invoke(list, 38, 0, word, 0);
        invoke(list, 34, root);
    } else if (!std::strcmp(mode, "switch_root")) {
        const StreamWord alternate = reinterpret_cast<std::uintptr_t>(stream_alternate_root_signature());
        if (!alternate) return 40;
        invoke(list, 38, 0, word, 0);
        invoke(list, 34, alternate);
        invoke(list, 34, root);
    } else if (std::strcmp(mode, "unbound")) {
        return 2;
    }
    invoke(list, 12, 3, 1);
    return 0;
}
