// Verifies that GameInput.dll plays the final entry of a long input script.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr std::uint32_t kind_gamepad = 0x00040000;
constexpr std::uint32_t dpad_right = 0x200;

struct GamepadState {
    std::uint32_t buttons;
    float leftTrigger, rightTrigger;
    float leftThumbstickX, leftThumbstickY, rightThumbstickX, rightThumbstickY;
};
static_assert(sizeof(GamepadState) == 28, "GamepadState layout");

struct IGameInputDevice;

struct IGameInputReading {
    void* const* vtbl;

    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputReading*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
    }
    bool GetGamepadState(GamepadState* state)
    {
        using Fn = bool(__stdcall*)(IGameInputReading*, GamepadState*);
        return reinterpret_cast<Fn>(vtbl[22])(this, state);
    }
};

struct IGameInput {
    void* const* vtbl;

    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInput*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
    }
    HRESULT GetCurrentReading(std::uint32_t kind, IGameInputDevice* device, IGameInputReading** reading)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, std::uint32_t, IGameInputDevice*, IGameInputReading**);
        return reinterpret_cast<Fn>(vtbl[4])(this, kind, device, reading);
    }
};

int fail(const char* reason)
{
    std::printf("FAIL: %s\n", reason);
    return 1;
}

bool read_gamepad(IGameInput* input, GamepadState* state)
{
    IGameInputReading* reading = nullptr;
    const HRESULT hr = input->GetCurrentReading(kind_gamepad, nullptr, &reading);
    const bool ok = SUCCEEDED(hr) && reading && reading->GetGamepadState(state);
    if (reading) reading->Release();
    return ok;
}

const char* check_script(IGameInput* input)
{
    // The first GetCurrentReading calls capture_current(), which calls sample()
    // and starts the script clock. GetGamepadState copies the captured state.
    const ULONGLONG start = GetTickCount64();
    GamepadState state{};
    do {
        if (!read_gamepad(input, &state)) return "could not read initial gamepad state";
        if (state.buttons & dpad_right) break;
        Sleep(25);
    } while (GetTickCount64() - start < 200);
    if (!(state.buttons & dpad_right)) return "script did not press D-pad right";

    const ULONGLONG elapsed = GetTickCount64() - start;
    if (elapsed < 600) Sleep(static_cast<DWORD>(600 - elapsed));
    if (!read_gamepad(input, &state)) return "could not read gamepad state after the press";
    if (state.buttons & dpad_right) return "script did not release D-pad right";
    return nullptr;
}

}  // namespace

int usage_error(const char* reason)
{
    std::printf("ERROR: %s\n", reason);
    return 2;
}

int main(int argc, char** argv)
{
    // startup_gameinput_script <GameInput.dll> long|entries
    //   long:    41 entries, 725 characters. Before this change the stand-in read the value into a
    //            512-byte buffer, which then stayed empty, so no entry ran and bit 0x200 was never set.
    //   entries: 33 entries, 229 characters. Before this change only the first 32 entries were kept,
    //            so the last one, the only one due at second 0, was dropped.
    if (argc < 3) return usage_error("usage: startup_gameinput_script <GameInput.dll> long|entries");
    std::string script;
    if (std::strcmp(argv[2], "long") == 0) {
        for (int i = 0; i < 40; ++i) script += "00000001@9000.000,";
    } else if (std::strcmp(argv[2], "entries") == 0) {
        for (int i = 0; i < 32; ++i) script += "1@9000,";
    } else {
        return usage_error("the mode must be long or entries");
    }
    script += "200@0";
    if (!SetEnvironmentVariableA("X2W_INPUT_SCRIPT", script.c_str()))
        return usage_error("could not set X2W_INPUT_SCRIPT");
    if (!SetEnvironmentVariableA("X2W_INPUT_PRESS", nullptr) && GetLastError() != ERROR_ENVVAR_NOT_FOUND)
        return usage_error("could not clear X2W_INPUT_PRESS");

    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return usage_error("LoadLibraryA failed");

    using CreateFn = HRESULT(__stdcall*)(IGameInput**);
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(dll, "GameInputCreate"));
    if (!create) {
        FreeLibrary(dll);
        return usage_error("GetProcAddress(GameInputCreate) failed");
    }

    IGameInput* input = nullptr;
    if (FAILED(create(&input)) || !input) {
        FreeLibrary(dll);
        return usage_error("GameInputCreate failed");
    }

    const char* error = check_script(input);
    input->Release();
    FreeLibrary(dll);
    if (error) return fail(error);
    std::printf("PASS: startup_gameinput_script %s\n", argv[2]);
    return 0;
}
