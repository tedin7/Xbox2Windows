// Source-built verification of GameInput.dll keyboard readings, GetPreviousReading/GetNextReading
// history chain, GetKeyCount/GetKeyState, and negative controls.
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

namespace {

constexpr std::uint32_t kind_keyboard = 0x00000010;
constexpr std::uint32_t kind_gamepad = 0x00040000;
constexpr std::uint32_t kind_mouse = 0x00000020;
constexpr std::uint32_t status_connected = 0x00000001;
constexpr std::uint32_t blocking_enumeration = 2;
constexpr HRESULT reading_not_found = static_cast<HRESULT>(0x838A0003L);
constexpr HRESULT device_not_found = static_cast<HRESULT>(0x838A0002L);

struct KeyState {
    std::uint32_t scanCode;
    std::uint32_t codePoint;
    std::uint8_t virtualKey;
    bool isDeadKey;
};
static_assert(sizeof(KeyState) == 12, "KeyState layout");

struct GamepadState {
    std::uint32_t buttons;
    float leftTrigger, rightTrigger;
    float leftThumbstickX, leftThumbstickY, rightThumbstickX, rightThumbstickY;
};
static_assert(sizeof(GamepadState) == 28, "GamepadState layout");

// Minimal vtable-matching C++ views of GameInput v0 COM interfaces.
struct IGameInputDeviceVtbl;
struct IGameInputDevice {
    void* const* vtbl;
};

struct IGameInputReading {
    void* const* vtbl;

    std::uint32_t AddRef()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputReading*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[1])(this));
    }
    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputReading*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
    }
    std::uint32_t GetInputKind()
    {
        using Fn = std::uint32_t(__stdcall*)(IGameInputReading*);
        return reinterpret_cast<Fn>(vtbl[3])(this);
    }
    std::uint64_t GetSequenceNumber(std::uint32_t kind)
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputReading*, std::uint32_t);
        return reinterpret_cast<Fn>(vtbl[4])(this, kind);
    }
    void GetDevice(IGameInputDevice** device)
    {
        using Fn = void(__stdcall*)(IGameInputReading*, IGameInputDevice**);
        reinterpret_cast<Fn>(vtbl[6])(this, device);
    }
    std::uint32_t GetKeyCount()
    {
        using Fn = std::uint32_t(__stdcall*)(IGameInputReading*);
        return reinterpret_cast<Fn>(vtbl[14])(this);
    }
    std::uint32_t GetKeyState(std::uint32_t count, KeyState* states)
    {
        using Fn = std::uint32_t(__stdcall*)(IGameInputReading*, std::uint32_t, KeyState*);
        return reinterpret_cast<Fn>(vtbl[15])(this, count, states);
    }
    bool GetGamepadState(GamepadState* state)
    {
        using Fn = bool(__stdcall*)(IGameInputReading*, GamepadState*);
        return reinterpret_cast<Fn>(vtbl[22])(this, state);
    }
};

using DeviceCallback = void(__stdcall*)(std::uint64_t token, void* context, IGameInputDevice* device,
                                        std::uint64_t timestamp, std::uint32_t currentStatus,
                                        std::uint32_t previousStatus);

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
    HRESULT GetNextReading(IGameInputReading* reference, std::uint32_t kind, IGameInputDevice* device,
                           IGameInputReading** reading)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputReading*, std::uint32_t, IGameInputDevice*,
                                       IGameInputReading**);
        return reinterpret_cast<Fn>(vtbl[5])(this, reference, kind, device, reading);
    }
    HRESULT GetPreviousReading(IGameInputReading* reference, std::uint32_t kind, IGameInputDevice* device,
                               IGameInputReading** reading)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputReading*, std::uint32_t, IGameInputDevice*,
                                       IGameInputReading**);
        return reinterpret_cast<Fn>(vtbl[6])(this, reference, kind, device, reading);
    }
    HRESULT RegisterDeviceCallback(IGameInputDevice* device, std::uint32_t kind, std::uint32_t statusFilter,
                                   std::uint32_t enumerationKind, void* context, DeviceCallback callback,
                                   std::uint64_t* token)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputDevice*, std::uint32_t, std::uint32_t,
                                       std::uint32_t, void*, DeviceCallback, std::uint64_t*);
        return reinterpret_cast<Fn>(vtbl[9])(this, device, kind, statusFilter, enumerationKind, context, callback,
                                             token);
    }
    HRESULT FindDevice(int slot, const void* arg, IGameInputDevice** device)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, const void*, IGameInputDevice**);
        return reinterpret_cast<Fn>(vtbl[slot])(this, arg, device);
    }
    bool UnregisterCallback(std::uint64_t token, std::uint64_t timeoutUs)
    {
        using Fn = bool(__stdcall*)(IGameInput*, std::uint64_t, std::uint64_t);
        return reinterpret_cast<Fn>(vtbl[13])(this, token, timeoutUs);
    }
};

struct CallbackCapture {
    int count = 0;
    IGameInputDevice* devices[4] = {};
    std::uint32_t supported_inputs[4] = {};
};

void __stdcall on_device_callback(std::uint64_t, void* context, IGameInputDevice* device, std::uint64_t,
                                  std::uint32_t current_status, std::uint32_t)
{
    auto* cap = static_cast<CallbackCapture*>(context);
    if (!cap || (current_status & status_connected) == 0 || cap->count >= 4) return;
    using GetInfoFn = const unsigned char*(__stdcall*)(IGameInputDevice*);
    const unsigned char* info = reinterpret_cast<GetInfoFn>(device->vtbl[3])(device);
    std::uint32_t supported = 0;
    if (info) std::memcpy(&supported, info + 0x68, sizeof(supported));
    cap->devices[cap->count] = device;
    cap->supported_inputs[cap->count] = supported;
    ++cap->count;
}

int check(bool cond, const char* msg)
{
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* dll_path = argc > 1 ? argv[1] : "GameInput.dll";
    HMODULE mod = LoadLibraryA(dll_path);
    if (!mod) {
        std::fprintf(stderr, "FAIL: LoadLibraryA(%s) failed (%lu)\n", dll_path, GetLastError());
        return 1;
    }

    using CreateFn = HRESULT(__stdcall*)(void**);
    auto create_fn = reinterpret_cast<CreateFn>(GetProcAddress(mod, MAKEINTRESOURCEA(1)));
    if (!create_fn) {
        std::fprintf(stderr, "FAIL: GameInputCreate ordinal 1 not found\n");
        FreeLibrary(mod);
        return 1;
    }

    int failures = 0;
    void* raw_input = nullptr;
    failures += check(create_fn(nullptr) == E_POINTER, "GameInputCreate(nullptr) should return E_POINTER");
    failures += check(create_fn(&raw_input) == S_OK && raw_input != nullptr, "GameInputCreate(&input) should succeed");
    auto* input = static_cast<IGameInput*>(raw_input);

    // 1. RegisterDeviceCallback enumerates keyboard and gamepad devices.
    CallbackCapture cap{};
    std::uint64_t token = 0;
    HRESULT hr = input->RegisterDeviceCallback(nullptr, kind_keyboard | kind_gamepad, status_connected,
                                               blocking_enumeration, &cap, &on_device_callback, &token);
    failures += check(hr == S_OK && token != 0, "RegisterDeviceCallback(keyboard|gamepad) should succeed");
    failures += check(cap.count == 2, "RegisterDeviceCallback should enumerate 2 devices (gamepad + keyboard)");
    IGameInputDevice* kbd_device = nullptr;
    IGameInputDevice* pad_device = nullptr;
    for (int i = 0; i < cap.count; ++i) {
        if (cap.supported_inputs[i] & kind_keyboard) kbd_device = cap.devices[i];
        if (cap.supported_inputs[i] & kind_gamepad) pad_device = cap.devices[i];
    }
    failures += check(kbd_device != nullptr && pad_device != nullptr && kbd_device != pad_device,
                      "Distinct keyboard and gamepad devices should be enumerated");
    failures += check(input->UnregisterCallback(token, 0), "UnregisterCallback should succeed");

    // 2. Empty keyboard reading (no X2W_INPUT_KEYS).
    SetEnvironmentVariableA("X2W_INPUT_KEYS", nullptr);
    IGameInputReading* r0 = nullptr;
    hr = input->GetCurrentReading(kind_keyboard, kbd_device, &r0);
    failures += check(hr == S_OK && r0 != nullptr, "GetCurrentReading(kind_keyboard) should succeed");
    if (r0) {
        failures += check((r0->GetInputKind() & kind_keyboard) != 0, "r0 kind should include keyboard");
        failures += check(r0->GetKeyCount() == 0, "r0 key count should be 0 when no keys are pressed");
        KeyState buf[4]{};
        failures += check(r0->GetKeyState(4, buf) == 0, "r0 GetKeyState should return 0 when empty");
        failures += check(r0->GetKeyState(0, buf) == 0, "r0 GetKeyState(0, buf) should return 0");
        failures += check(r0->GetKeyState(4, nullptr) == 0, "r0 GetKeyState(4, nullptr) should return 0");
        GamepadState gp{};
        failures += check(!r0->GetGamepadState(&gp), "keyboard-only reading should reject GetGamepadState");
    }

    // 3. Inject first key set ("41,57" -> 'A', 'W') and verify GetKeyCount / GetKeyState.
    SetEnvironmentVariableA("X2W_INPUT_KEYS", "41,57");
    IGameInputReading* r1 = nullptr;
    hr = input->GetCurrentReading(kind_keyboard, kbd_device, &r1);
    failures += check(hr == S_OK && r1 != nullptr, "r1 GetCurrentReading should succeed");
    if (r1) {
        failures += check(r1->GetKeyCount() == 2, "r1 GetKeyCount should be 2");
        KeyState single[1]{};
        failures += check(r1->GetKeyState(1, single) == 1 && single[0].virtualKey == 0x41,
                          "r1 GetKeyState(1) should truncate to first key (0x41)");
        KeyState keys[4]{};
        const std::uint32_t got = r1->GetKeyState(4, keys);
        failures += check(got == 2, "r1 GetKeyState(4) should return 2");
        failures += check(keys[0].virtualKey == 0x41 && keys[1].virtualKey == 0x57,
                          "r1 keys should be 0x41 and 0x57");
        failures += check(keys[0].scanCode != 0 && !keys[0].isDeadKey, "r1 key 0 scanCode should be populated");
        if (r0) {
            failures += check(r1->GetSequenceNumber(kind_keyboard) > r0->GetSequenceNumber(kind_keyboard),
                              "r1 sequence should advance after key change");
        }
    }

    // 4. Inject second key set ("42" -> 'B') and test GetPreviousReading / GetNextReading chain.
    SetEnvironmentVariableA("X2W_INPUT_KEYS", "42");
    IGameInputReading* r2 = nullptr;
    hr = input->GetCurrentReading(kind_keyboard, kbd_device, &r2);
    failures += check(hr == S_OK && r2 != nullptr, "r2 GetCurrentReading should succeed");
    if (r2) {
        failures += check(r2->GetKeyCount() == 1, "r2 GetKeyCount should be 1");
        KeyState k2[2]{};
        failures += check(r2->GetKeyState(2, k2) == 1 && k2[0].virtualKey == 0x42, "r2 key should be 0x42");

        IGameInputReading* prev = nullptr;
        hr = input->GetPreviousReading(r2, kind_keyboard, kbd_device, &prev);
        failures += check(hr == S_OK && prev != nullptr, "GetPreviousReading(r2) should succeed");
        if (prev) {
            KeyState pk[4]{};
            failures += check(prev->GetKeyCount() == 2, "prev key count should be 2");
            failures += check(prev->GetKeyState(4, pk) == 2 && pk[0].virtualKey == 0x41 && pk[1].virtualKey == 0x57,
                              "prev keys should match r1 (0x41, 0x57)");
            if (r1) {
                failures += check(prev->GetSequenceNumber(kind_keyboard) == r1->GetSequenceNumber(kind_keyboard),
                                  "prev sequence should match r1");
            }

            IGameInputReading* next = nullptr;
            hr = input->GetNextReading(prev, kind_keyboard, kbd_device, &next);
            failures += check(hr == S_OK && next != nullptr, "GetNextReading(prev) should succeed");
            if (next) {
                KeyState nk[2]{};
                failures += check(next->GetKeyCount() == 1 && next->GetKeyState(2, nk) == 1 &&
                                      nk[0].virtualKey == 0x42,
                                  "next keys should match r2 (0x42)");
                next->Release();
            }
            prev->Release();
        }
    }

    // 5. Combined kind_keyboard | kind_gamepad reading + gamepad preservation.
    IGameInputReading* combo = nullptr;
    hr = input->GetCurrentReading(kind_keyboard | kind_gamepad, nullptr, &combo);
    failures += check(hr == S_OK && combo != nullptr, "GetCurrentReading(keyboard|gamepad) should succeed");
    if (combo) {
        GamepadState gp{};
        failures += check(combo->GetGamepadState(&gp), "combo reading should support GetGamepadState");
        failures += check(combo->GetKeyCount() == 1, "combo reading should report 1 pressed key");
        combo->Release();
    }

    // 5b. Extended scan-code prefixes survive; generic modifier aliases are not reported as keys.
    SetEnvironmentVariableA("X2W_INPUT_KEYS", "25,64,A3,A5");  // Left, Numpad4, RCtrl, RAlt
    IGameInputReading* ext = nullptr;
    hr = input->GetCurrentReading(kind_keyboard, kbd_device, &ext);
    failures += check(hr == S_OK && ext != nullptr, "extended-key reading should succeed");
    if (ext) {
        KeyState ek[8]{};
        failures += check(ext->GetKeyState(8, ek) == 4, "extended-key reading should report 4 keys");
        failures += check(ek[0].scanCode == 0xE04B, "Left Arrow scan code should keep the E0 prefix");
        failures += check(ek[1].scanCode == 0x4B, "Numpad 4 scan code should have no prefix");
        failures += check(ek[2].scanCode == 0xE01D, "Right Ctrl scan code should keep the E0 prefix");
        failures += check(ek[3].scanCode == 0xE038, "Right Alt scan code should keep the E0 prefix");
        ext->Release();
    }
    SetEnvironmentVariableA("X2W_INPUT_KEYS", "10,11,12,A0,A2");  // generic Shift/Ctrl/Alt, then LShift, LCtrl
    hr = input->GetCurrentReading(kind_keyboard, kbd_device, &ext);
    failures += check(hr == S_OK && ext != nullptr, "modifier reading should succeed");
    if (ext) {
        KeyState mk[8]{};
        const std::uint32_t n = ext->GetKeyState(8, mk);
        failures += check(n == 2 && ext->GetKeyCount() == 2, "generic modifier aliases should be skipped");
        failures += check(n == 2 && mk[0].virtualKey == 0xA0 && mk[1].virtualKey == 0xA2, "left modifiers should remain");
        ext->Release();
    }

    // 5c. History traversal is filtered by device: keyboard-only changes are invisible to a gamepad caller.
    SetEnvironmentVariableA("X2W_INPUT_KEYS", nullptr);
    HANDLE pad_map = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 16, L"Local\\x2w_input_pad");
    auto* pad_words = pad_map ? static_cast<volatile std::uint32_t*>(MapViewOfFile(pad_map, FILE_MAP_ALL_ACCESS, 0, 0, 16)) : nullptr;
    failures += check(pad_words != nullptr, "shared pad block should be mappable");
    if (pad_words) {
        pad_words[1] = 2;  // buttons: View
        IGameInputReading *pad_a = nullptr, *pad_b = nullptr, *pad_c = nullptr, *tmp = nullptr;
        failures += check(input->GetCurrentReading(kind_gamepad, pad_device, &pad_a) == S_OK && pad_a, "pad_a reading");
        const std::uint64_t base = pad_a ? pad_a->GetSequenceNumber(kind_gamepad) : 0;
        SetEnvironmentVariableA("X2W_INPUT_KEYS", "51");
        input->GetCurrentReading(kind_keyboard, kbd_device, &tmp);
        const std::uint64_t key51 = tmp ? tmp->GetSequenceNumber(kind_keyboard) : 0;
        if (tmp) tmp->Release();
        SetEnvironmentVariableA("X2W_INPUT_KEYS", "52");
        input->GetCurrentReading(kind_keyboard, kbd_device, &tmp);
        if (tmp) tmp->Release();
        tmp = nullptr;
        // Control: repeated keyboard-only reads never move the gamepad kind's sequence number.
        for (int i = 0; i < 3; ++i) {
            SetEnvironmentVariableA("X2W_INPUT_KEYS", i % 2 ? "53" : "54");
            input->GetCurrentReading(kind_keyboard, kbd_device, &tmp);
            if (tmp) tmp->Release();
            tmp = nullptr;
        }
        SetEnvironmentVariableA("X2W_INPUT_KEYS", "52");
        IGameInputReading* pad_same = nullptr;
        failures += check(input->GetCurrentReading(kind_gamepad, pad_device, &pad_same) == S_OK && pad_same &&
                              pad_same->GetSequenceNumber(kind_gamepad) == base,
                          "gamepad sequence is unchanged after repeated keyboard-only reads");
        if (pad_same) pad_same->Release();
        // Control: keyboard-only traffic beyond the ring size must not age out the gamepad reference.
        for (int i = 0; i < 100; ++i) {
            char keys[8];
            std::snprintf(keys, sizeof keys, "%X", 0x30 + i % 2);
            SetEnvironmentVariableA("X2W_INPUT_KEYS", keys);
            input->GetCurrentReading(kind_keyboard, kbd_device, &tmp);
            if (tmp) tmp->Release();
            tmp = nullptr;
        }
        SetEnvironmentVariableA("X2W_INPUT_KEYS", "52");
        input->GetCurrentReading(kind_keyboard, kbd_device, &tmp);
        const std::uint64_t key52 = tmp ? tmp->GetSequenceNumber(kind_keyboard) : 0;
        if (tmp) tmp->Release();
        tmp = nullptr;
        failures += check(input->GetNextReading(pad_a, kind_gamepad, pad_device, &tmp) == reading_not_found && tmp == nullptr,
                          "keyboard-only changes must not be a next gamepad reading");
        pad_words[1] = 4;  // buttons: A
        input->GetCurrentReading(kind_keyboard, kbd_device, &pad_c);  // records the pad change
        failures += check(input->GetNextReading(pad_a, kind_gamepad, pad_device, &pad_b) == S_OK && pad_b, "next gamepad reading");
        if (pad_b) {
            GamepadState gs{};
            failures += check(pad_b->GetGamepadState(&gs) && gs.buttons == 4, "next gamepad reading carries the new buttons");
            failures += check(pad_b->GetSequenceNumber(kind_gamepad) == base + 1, "next gamepad reading carries the next gamepad sequence");
            IGameInputReading* back = nullptr;
            failures += check(input->GetPreviousReading(pad_b, kind_gamepad, pad_device, &back) == S_OK && back, "previous gamepad reading");
            if (back) {
                failures += check(back->GetSequenceNumber(kind_gamepad) == base, "previous gamepad reading skips keyboard-only entries");
                back->Release();
            }
            IGameInputReading* kback = nullptr;
            failures += check(input->GetPreviousReading(pad_b, kind_keyboard, kbd_device, &kback) == S_OK && kback, "previous keyboard reading");
            if (kback) {
                failures += check(kback->GetSequenceNumber(kind_keyboard) == key52 && key52 > key51,
                                  "previous keyboard reading skips pad-only entries");
                IGameInputReading* earlier_key = nullptr;
                failures += check(input->GetPreviousReading(kback, kind_keyboard, kbd_device, &earlier_key) == S_OK &&
                                      earlier_key && earlier_key->GetSequenceNumber(kind_keyboard) == key52 - 1,
                                  "previous keyboard reading steps back from a keyboard entry");
                if (earlier_key) earlier_key->Release();
                kback->Release();
            }
            pad_b->Release();
        }
        if (pad_a) pad_a->Release();
        if (pad_c) pad_c->Release();
        pad_words[1] = 0;
        UnmapViewOfFile(const_cast<std::uint32_t*>(pad_words));
    }
    if (pad_map) CloseHandle(pad_map);
    SetEnvironmentVariableA("X2W_INPUT_KEYS", "42");

    // 5d. Device lookups: by id, and an opaque platform handle that must never be dereferenced.
    using GetInfoFn = const unsigned char*(__stdcall*)(IGameInputDevice*);
    if (kbd_device) {
        const unsigned char* kinfo = reinterpret_cast<GetInfoFn>(kbd_device->vtbl[3])(kbd_device);
        IGameInputDevice* found = nullptr;
        failures += check(input->FindDevice(16, kinfo + 0x20, &found) == S_OK && found == kbd_device, "FindDeviceFromId(keyboard id)");
        found = nullptr;
        failures += check(input->FindDevice(18, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(4)), &found) == device_not_found &&
                              found == nullptr,
                          "FindDeviceFromPlatformHandle(0x4) should fail without dereferencing the handle");
    }

    // 6. Negative controls: null out, wrong kind, mismatched device, null reference for GetPreviousReading.
    IGameInputReading* bad = nullptr;
    failures += check(input->GetCurrentReading(kind_keyboard, nullptr, nullptr) == E_POINTER,
                      "GetCurrentReading(null out) should return E_POINTER");
    failures += check(input->GetCurrentReading(kind_mouse, nullptr, &bad) == reading_not_found && bad == nullptr,
                      "GetCurrentReading(mouse) should return reading_not_found");
    failures += check(input->GetCurrentReading(kind_keyboard, pad_device, &bad) == reading_not_found && bad == nullptr,
                      "GetCurrentReading(keyboard, pad_device) should return reading_not_found");
    failures += check(input->GetCurrentReading(kind_gamepad, kbd_device, &bad) == reading_not_found && bad == nullptr,
                      "GetCurrentReading(gamepad, kbd_device) should return reading_not_found");
    failures += check(input->GetPreviousReading(nullptr, kind_keyboard, nullptr, &bad) == E_INVALIDARG && bad == nullptr,
                      "GetPreviousReading(null ref) should return E_INVALIDARG");
    failures += check(input->GetPreviousReading(r2, kind_keyboard, nullptr, nullptr) == E_POINTER,
                      "GetPreviousReading(null out) should return E_POINTER");
    if (r2) {
        // Unknown device pointer should fail with device_not_found or reading_not_found.
        IGameInputDevice fake_device{nullptr};
        hr = input->GetCurrentReading(kind_keyboard, &fake_device, &bad);
        failures += check((hr == device_not_found || hr == reading_not_found) && bad == nullptr,
                          "GetCurrentReading(fake_device) should fail");
        // No next reading past latest.
        IGameInputReading* latest = nullptr;
        failures += check(input->GetCurrentReading(kind_keyboard, kbd_device, &latest) == S_OK && latest, "latest reading");
        failures += check(input->GetNextReading(latest, kind_keyboard, kbd_device, &bad) == reading_not_found &&
                              bad == nullptr,
                          "GetNextReading(latest) should return reading_not_found");
        if (latest) latest->Release();
    }

    if (r0) r0->Release();
    if (r1) r1->Release();
    if (r2) r2->Release();
    SetEnvironmentVariableA("X2W_INPUT_KEYS", nullptr);
    input->Release();
    FreeLibrary(mod);
    if (failures == 0) std::printf("PASS: startup_gameinput_keys\n");
    return failures == 0 ? 0 : 1;
}
