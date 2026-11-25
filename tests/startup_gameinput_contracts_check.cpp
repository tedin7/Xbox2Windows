// COM contract, output-pointer validation, and cross-kind refusal checks for GameInput.dll.
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
constexpr std::uint32_t status_connected = 0x47;
constexpr std::uint32_t blocking_enumeration = 2;
constexpr HRESULT device_not_found = static_cast<HRESULT>(0x838A0002L);
constexpr GUID dummy_iid = {0x11223344u, 0x5566u, 0x7788u, {0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00}};

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

struct IGameInputDevice {
    const void* const* vtbl;

    HRESULT QueryInterface(const GUID* iid, void** out)
    {
        using Fn = HRESULT(__stdcall*)(IGameInputDevice*, const GUID*, void**);
        return reinterpret_cast<Fn>(vtbl[0])(this, iid, out);
    }
    std::uint32_t AddRef()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputDevice*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[1])(this));
    }
    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputDevice*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
    }
    const unsigned char* GetDeviceInfo()
    {
        using Fn = const unsigned char*(__stdcall*)(IGameInputDevice*);
        return reinterpret_cast<Fn>(vtbl[3])(this);
    }
    std::uint32_t GetDeviceStatus()
    {
        using Fn = std::uint32_t(__stdcall*)(IGameInputDevice*);
        return reinterpret_cast<Fn>(vtbl[4])(this);
    }
    void SetRumbleState(const void* params)
    {
        using Fn = void(__stdcall*)(IGameInputDevice*, const void*);
        reinterpret_cast<Fn>(vtbl[10])(this, params);
    }
};

struct IGameInputReading {
    const void* const* vtbl;

    HRESULT QueryInterface(const GUID* iid, void** out)
    {
        using Fn = HRESULT(__stdcall*)(IGameInputReading*, const GUID*, void**);
        return reinterpret_cast<Fn>(vtbl[0])(this, iid, out);
    }
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
    std::uint64_t GetTimestamp()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputReading*);
        return reinterpret_cast<Fn>(vtbl[5])(this);
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
    const void* const* vtbl;

    HRESULT QueryInterface(const GUID* iid, void** out)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, const GUID*, void**);
        return reinterpret_cast<Fn>(vtbl[0])(this, iid, out);
    }
    std::uint32_t AddRef()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInput*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[1])(this));
    }
    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInput*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
    }
    std::uint64_t GetCurrentTimestamp()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInput*);
        return reinterpret_cast<Fn>(vtbl[3])(this);
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
    HRESULT RegisterReadingCallback(IGameInputDevice* device, std::uint32_t kind, float analogThreshold, void* context,
                                    void* callback, std::uint64_t* token)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputDevice*, std::uint32_t, float, void*, void*,
                                       std::uint64_t*);
        return reinterpret_cast<Fn>(vtbl[8])(this, device, kind, analogThreshold, context, callback, token);
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
    HRESULT RegisterGuideButtonCallback(IGameInputDevice* device, void* context, void* callback, std::uint64_t* token)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputDevice*, void*, void*, std::uint64_t*);
        return reinterpret_cast<Fn>(vtbl[10])(this, device, context, callback, token);
    }
    HRESULT RegisterKeyboardLayoutCallback(IGameInputDevice* device, void* context, void* callback,
                                           std::uint64_t* token)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputDevice*, void*, void*, std::uint64_t*);
        return reinterpret_cast<Fn>(vtbl[11])(this, device, context, callback, token);
    }
    void StopCallback(std::uint64_t token)
    {
        using Fn = void(__stdcall*)(IGameInput*, std::uint64_t);
        reinterpret_cast<Fn>(vtbl[12])(this, token);
    }
    bool UnregisterCallback(std::uint64_t token, std::uint64_t timeoutUs)
    {
        using Fn = bool(__stdcall*)(IGameInput*, std::uint64_t, std::uint64_t);
        return reinterpret_cast<Fn>(vtbl[13])(this, token, timeoutUs);
    }

    // GameInput.h v0 slot 7: GetTemporalReading(uint64_t timestamp, IGameInputDevice* device,
    // IGameInputReading** reading). Documented as never implemented in v0 (removed in v1).
    HRESULT GetTemporalReading(std::uint64_t timestamp, IGameInputDevice* device, IGameInputReading** reading)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, std::uint64_t, IGameInputDevice*, IGameInputReading**);
        return reinterpret_cast<Fn>(vtbl[7])(this, timestamp, device, reading);
    }
    HRESULT FindDevice(int slot, const void* arg, IGameInputDevice** device)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, const void*, IGameInputDevice**);
        return reinterpret_cast<Fn>(vtbl[slot])(this, arg, device);
    }
    void SetFocusPolicy(std::uint32_t policy)
    {
        using Fn = void(__stdcall*)(IGameInput*, std::uint32_t);
        reinterpret_cast<Fn>(vtbl[21])(this, policy);
    }
};

struct DeviceCapture {
    int count = 0;
    IGameInputDevice* gamepad = nullptr;
    IGameInputDevice* keyboard = nullptr;
};

void __stdcall on_device_callback(std::uint64_t, void* context, IGameInputDevice* device, std::uint64_t,
                                  std::uint32_t current_status, std::uint32_t)
{
    auto* cap = static_cast<DeviceCapture*>(context);
    if (!cap || (current_status & status_connected) == 0 || !device) return;
    ++cap->count;
    const unsigned char* info = device->GetDeviceInfo();
    std::uint32_t supported = 0;
    if (info) std::memcpy(&supported, info + 0x68, sizeof(supported));
    if (supported & kind_gamepad) cap->gamepad = device;
    if (supported & kind_keyboard) cap->keyboard = device;
}

int failures = 0;

void check(bool cond, const char* label)
{
    if (cond) return;
    ++failures;
    std::printf("FAIL: %s\n", label);
}

}  // namespace

int main(int argc, char** argv)
{
    const char* dll_path = argc > 1 ? argv[1] : "GameInput.dll";
    HMODULE mod = LoadLibraryA(dll_path);
    if (!mod) return std::printf("FAIL: LoadLibraryA(%s) failed (%lu)\n", dll_path, GetLastError()), 2;

    using CreateFn = HRESULT(__stdcall*)(void**);
    auto create_fn = reinterpret_cast<CreateFn>(GetProcAddress(mod, MAKEINTRESOURCEA(1)));
    if (!create_fn) {
        FreeLibrary(mod);
        return std::printf("FAIL: GameInputCreate ordinal 1 not found\n"), 2;
    }

    // Allocate one read-only page and one RW->RO straddling page for unwritable-pointer negative controls.
    void* ro_page = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY);
    check(ro_page != nullptr, "VirtualAlloc PAGE_READONLY");
    void* split_pages = VirtualAlloc(nullptr, 0x2000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    void* edge_ptr = nullptr;
    if (split_pages) {
        DWORD old_prot = 0;
        if (VirtualProtect(static_cast<unsigned char*>(split_pages) + 0x1000, 0x1000, PAGE_READONLY, &old_prot)) {
            edge_ptr = static_cast<unsigned char*>(split_pages) + 0x1000 - 4;
        }
    }
    check(edge_ptr != nullptr, "VirtualAlloc straddling RW->RO boundary");

    // 1. GameInputCreate pointer validation and IUnknown lifecycle.
    check(create_fn(nullptr) == E_POINTER, "GameInputCreate(null) -> E_POINTER");
    if (ro_page) {
        check(create_fn(static_cast<void**>(ro_page)) == E_POINTER, "GameInputCreate(PAGE_READONLY) -> E_POINTER");
    }
    if (edge_ptr) {
        check(create_fn(static_cast<void**>(edge_ptr)) == E_POINTER, "GameInputCreate(straddling) -> E_POINTER");
    }

    void* raw_input = nullptr;
    check(create_fn(&raw_input) == S_OK && raw_input != nullptr, "GameInputCreate succeeds");
    auto* input = static_cast<IGameInput*>(raw_input);
    if (!input) {
        if (ro_page) VirtualFree(ro_page, 0, MEM_RELEASE);
        if (split_pages) VirtualFree(split_pages, 0, MEM_RELEASE);
        FreeLibrary(mod);
        return 1;
    }

    void* qi_out = nullptr;
    check(input->QueryInterface(&dummy_iid, nullptr) == E_POINTER, "IGameInput::QueryInterface(null) -> E_POINTER");
    if (ro_page) {
        check(input->QueryInterface(&dummy_iid, static_cast<void**>(ro_page)) == E_POINTER,
              "IGameInput::QueryInterface(PAGE_READONLY) -> E_POINTER");
    }
    check(input->QueryInterface(&dummy_iid, &qi_out) == S_OK && qi_out == input,
          "IGameInput::QueryInterface succeeds");
    const std::uint32_t refs_after_add = input->AddRef();
    check(refs_after_add >= 3, "IGameInput::AddRef increments reference count");
    check(input->Release() == refs_after_add - 1, "IGameInput::Release decrements reference count");
    if (qi_out) static_cast<IGameInput*>(qi_out)->Release();

    // 2. GetCurrentTimestamp and SetFocusPolicy.
    const std::uint64_t ts0 = input->GetCurrentTimestamp();
    check(ts0 > 0, "GetCurrentTimestamp returns non-zero microseconds");
    input->SetFocusPolicy(1);

    // GetTemporalReading (v0 slot 7) was never implemented by Microsoft and is documented to
    // return E_NOTIMPL; the stand-in keeps that refusal and never produces a reading.
    {
        IGameInputReading* temporal = reinterpret_cast<IGameInputReading*>(static_cast<std::uintptr_t>(1));
        check(input->GetTemporalReading(ts0, nullptr, &temporal) == E_NOTIMPL,
              "GetTemporalReading -> E_NOTIMPL (never implemented in v0)");
        check(temporal == nullptr, "GetTemporalReading never produces a reading");
        check(input->GetTemporalReading(ts0, nullptr, nullptr) == E_POINTER,
              "GetTemporalReading(null out) -> E_POINTER");
        if (ro_page) {
            check(input->GetTemporalReading(ts0, nullptr, reinterpret_cast<IGameInputReading**>(ro_page)) == E_POINTER,
                  "GetTemporalReading(PAGE_READONLY) -> E_POINTER");
        }
    }

    // 3. Callback registration, StopCallback, UnregisterCallback, and unwritable token guards.
    DeviceCapture cap{};
    std::uint64_t dev_token = 0;
    check(input->RegisterDeviceCallback(nullptr, kind_gamepad | kind_keyboard, status_connected, blocking_enumeration,
                                        &cap, &on_device_callback, &dev_token) == S_OK &&
              dev_token != 0,
          "RegisterDeviceCallback enumerates virtual devices");
    check(cap.count == 2 && cap.gamepad != nullptr && cap.keyboard != nullptr && cap.gamepad != cap.keyboard,
          "RegisterDeviceCallback reports distinct gamepad and keyboard devices");

    DeviceCapture optional_cap{};
    check(input->RegisterDeviceCallback(nullptr, kind_gamepad, status_connected, blocking_enumeration, &optional_cap,
                                        &on_device_callback, nullptr) == S_OK &&
              optional_cap.count == 1,
          "RegisterDeviceCallback allows null optional token");

    if (ro_page) {
        DeviceCapture refused_cap{};
        check(input->RegisterDeviceCallback(nullptr, kind_gamepad | kind_keyboard, status_connected,
                                            blocking_enumeration, &refused_cap, &on_device_callback,
                                            static_cast<std::uint64_t*>(ro_page)) == E_POINTER &&
                  refused_cap.count == 0,
              "RegisterDeviceCallback(PAGE_READONLY token) -> E_POINTER without invoking callback");
        check(input->RegisterReadingCallback(nullptr, kind_gamepad, 0.0f, nullptr, nullptr,
                                             static_cast<std::uint64_t*>(ro_page)) == E_POINTER,
              "RegisterReadingCallback(PAGE_READONLY token) -> E_POINTER");
        check(input->RegisterGuideButtonCallback(nullptr, nullptr, nullptr,
                                                 static_cast<std::uint64_t*>(ro_page)) == E_POINTER,
              "RegisterGuideButtonCallback(PAGE_READONLY token) -> E_POINTER");
        check(input->RegisterKeyboardLayoutCallback(nullptr, nullptr, nullptr,
                                                    static_cast<std::uint64_t*>(ro_page)) == E_POINTER,
              "RegisterKeyboardLayoutCallback(PAGE_READONLY token) -> E_POINTER");
    }
    if (edge_ptr) {
        check(input->RegisterDeviceCallback(nullptr, kind_gamepad, status_connected, 0, nullptr, nullptr,
                                            static_cast<std::uint64_t*>(edge_ptr)) == E_POINTER,
              "RegisterDeviceCallback(straddling token) -> E_POINTER");
        check(input->RegisterGuideButtonCallback(nullptr, nullptr, nullptr,
                                                 static_cast<std::uint64_t*>(edge_ptr)) == E_POINTER,
              "RegisterGuideButtonCallback(straddling token) -> E_POINTER");
    }

    std::uint64_t read_token = 0;
    std::uint64_t guide_token = 0;
    std::uint64_t layout_token = 0;
    check(input->RegisterReadingCallback(nullptr, kind_gamepad, 0.0f, nullptr, nullptr, &read_token) == S_OK &&
              read_token >= 0x1000,
          "RegisterReadingCallback succeeds and writes token");
    check(input->RegisterGuideButtonCallback(nullptr, nullptr, nullptr, &guide_token) == S_OK && guide_token >= 0x2000,
          "RegisterGuideButtonCallback succeeds and writes token");
    check(input->RegisterKeyboardLayoutCallback(nullptr, nullptr, nullptr, &layout_token) == S_OK &&
              layout_token >= 0x2000 && layout_token != guide_token,
          "RegisterKeyboardLayoutCallback succeeds and writes distinct token");
    input->StopCallback(dev_token);
    check(input->UnregisterCallback(dev_token, 0), "UnregisterCallback returns true");
    check(input->UnregisterCallback(read_token, 0), "UnregisterCallback(read_token) returns true");
    check(input->UnregisterCallback(guide_token, 0), "UnregisterCallback(guide_token) returns true");
    check(input->UnregisterCallback(layout_token, 0), "UnregisterCallback(layout_token) returns true");

    // 4. Device lookup slots (16..19) and IGameInputDevice methods.
    check(input->FindDevice(16, nullptr, nullptr) == E_POINTER, "FindDeviceFromId(null) -> E_POINTER");
    check(input->FindDevice(17, nullptr, nullptr) == E_POINTER, "FindDeviceFromObject(null) -> E_POINTER");
    check(input->FindDevice(18, nullptr, nullptr) == E_POINTER, "FindDeviceFromPlatformHandle(null) -> E_POINTER");
    check(input->FindDevice(19, nullptr, nullptr) == E_POINTER, "FindDeviceFromPlatformString(null) -> E_POINTER");
    if (ro_page) {
        check(input->FindDevice(16, nullptr, static_cast<IGameInputDevice**>(ro_page)) == E_POINTER,
              "FindDeviceFromId(PAGE_READONLY) -> E_POINTER");
        check(input->FindDevice(17, nullptr, static_cast<IGameInputDevice**>(ro_page)) == E_POINTER,
              "FindDeviceFromObject(PAGE_READONLY) -> E_POINTER");
        check(input->FindDevice(18, nullptr, static_cast<IGameInputDevice**>(ro_page)) == E_POINTER,
              "FindDeviceFromPlatformHandle(PAGE_READONLY) -> E_POINTER");
        check(input->FindDevice(19, nullptr, static_cast<IGameInputDevice**>(ro_page)) == E_POINTER,
              "FindDeviceFromPlatformString(PAGE_READONLY) -> E_POINTER");
    }

    IGameInputDevice* by_obj = nullptr;
    IGameInputDevice* by_str = nullptr;
    IGameInputDevice* by_handle = cap.gamepad;
    check(input->FindDevice(17, nullptr, &by_obj) == S_OK && by_obj == cap.gamepad,
          "FindDeviceFromObject returns virtual gamepad");
    check(input->FindDevice(19, L"pad0", &by_str) == S_OK && by_str == cap.gamepad,
          "FindDeviceFromPlatformString returns virtual gamepad");
    check(input->FindDevice(18, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(4)), &by_handle) ==
                  device_not_found &&
              by_handle == nullptr,
          "FindDeviceFromPlatformHandle sets out to null and returns device_not_found");
    if (by_obj) {
        check(by_obj->GetDeviceStatus() == status_connected, "IGameInputDevice::GetDeviceStatus == status_connected");
        const std::uint32_t rumble[4] = {0, 0, 0, 0};
        by_obj->SetRumbleState(rumble);
        void* dev_qi = nullptr;
        check(by_obj->QueryInterface(&dummy_iid, nullptr) == E_POINTER,
              "IGameInputDevice::QueryInterface(null) -> E_POINTER");
        if (ro_page) {
            check(by_obj->QueryInterface(&dummy_iid, static_cast<void**>(ro_page)) == E_POINTER,
                  "IGameInputDevice::QueryInterface(PAGE_READONLY) -> E_POINTER");
        }
        check(by_obj->QueryInterface(&dummy_iid, &dev_qi) == S_OK && dev_qi == by_obj,
              "IGameInputDevice::QueryInterface succeeds");
        if (dev_qi) static_cast<IGameInputDevice*>(dev_qi)->Release();
        by_obj->Release();
    }
    if (by_str) by_str->Release();

    // 5. Reading output pointer checks, GetTimestamp, GetDevice, and cross-kind refusals.
    check(input->GetCurrentReading(kind_gamepad, nullptr, nullptr) == E_POINTER,
          "GetCurrentReading(null) -> E_POINTER");
    check(input->GetNextReading(nullptr, kind_gamepad, nullptr, nullptr) == E_POINTER,
          "GetNextReading(null out) -> E_POINTER");
    check(input->GetPreviousReading(nullptr, kind_gamepad, nullptr, nullptr) == E_POINTER,
          "GetPreviousReading(null out) -> E_POINTER");
    if (ro_page) {
        check(input->GetCurrentReading(kind_gamepad, nullptr, static_cast<IGameInputReading**>(ro_page)) == E_POINTER,
              "GetCurrentReading(PAGE_READONLY) -> E_POINTER");
        check(input->GetNextReading(nullptr, kind_gamepad, nullptr, static_cast<IGameInputReading**>(ro_page)) ==
                  E_POINTER,
              "GetNextReading(PAGE_READONLY) -> E_POINTER");
        check(input->GetPreviousReading(nullptr, kind_gamepad, nullptr, static_cast<IGameInputReading**>(ro_page)) ==
                  E_POINTER,
              "GetPreviousReading(PAGE_READONLY) -> E_POINTER");
    }

    SetEnvironmentVariableA("X2W_INPUT_KEYS", "41,42");
    IGameInputReading* pad_reading = nullptr;
    IGameInputReading* kbd_reading = nullptr;
    check(input->GetCurrentReading(kind_gamepad, cap.gamepad, &pad_reading) == S_OK && pad_reading != nullptr,
          "GetCurrentReading(kind_gamepad) succeeds");
    check(input->GetCurrentReading(kind_keyboard, cap.keyboard, &kbd_reading) == S_OK && kbd_reading != nullptr,
          "GetCurrentReading(kind_keyboard) succeeds");

    if (pad_reading) {
        const std::uint64_t rts = pad_reading->GetTimestamp();
        const std::uint64_t ts1 = input->GetCurrentTimestamp();
        check(rts > 0 && rts <= ts1, "IGameInputReading::GetTimestamp is valid and <= current timestamp");

        pad_reading->GetDevice(nullptr);
        if (ro_page) pad_reading->GetDevice(static_cast<IGameInputDevice**>(ro_page));
        IGameInputDevice* rdev = nullptr;
        pad_reading->GetDevice(&rdev);
        check(rdev == cap.gamepad, "gamepad reading GetDevice returns virtual gamepad");
        if (rdev) rdev->Release();

        void* r_qi = nullptr;
        check(pad_reading->QueryInterface(&dummy_iid, nullptr) == E_POINTER,
              "IGameInputReading::QueryInterface(null) -> E_POINTER");
        if (ro_page) {
            check(pad_reading->QueryInterface(&dummy_iid, static_cast<void**>(ro_page)) == E_POINTER,
                  "IGameInputReading::QueryInterface(PAGE_READONLY) -> E_POINTER");
        }
        check(pad_reading->QueryInterface(&dummy_iid, &r_qi) == S_OK && r_qi == pad_reading,
              "IGameInputReading::QueryInterface succeeds");
        if (r_qi) static_cast<IGameInputReading*>(r_qi)->Release();

        check(pad_reading->GetKeyCount() == 0, "GetKeyCount on gamepad-only reading returns 0");
        KeyState ks[4]{};
        check(pad_reading->GetKeyState(4, ks) == 0, "GetKeyState on gamepad-only reading returns 0");
        check(pad_reading->GetSequenceNumber(kind_keyboard) == 0,
              "GetSequenceNumber(keyboard) on gamepad-only reading returns 0");
        check(!pad_reading->GetGamepadState(nullptr), "GetGamepadState(null) returns false");
        if (ro_page) {
            check(!pad_reading->GetGamepadState(static_cast<GamepadState*>(ro_page)),
                  "GetGamepadState(PAGE_READONLY) returns false");
        }
        if (edge_ptr) {
            check(!pad_reading->GetGamepadState(static_cast<GamepadState*>(edge_ptr)),
                  "GetGamepadState(straddling) returns false");
        }
        GamepadState gp{};
        check(pad_reading->GetGamepadState(&gp), "GetGamepadState(valid) on gamepad reading returns true");
        pad_reading->Release();
    }

    if (kbd_reading) {
        IGameInputDevice* rdev = nullptr;
        kbd_reading->GetDevice(&rdev);
        check(rdev == cap.keyboard, "keyboard reading GetDevice returns virtual keyboard");
        if (rdev) rdev->Release();

        check(kbd_reading->GetKeyCount() == 2, "keyboard reading reports 2 pressed keys");
        KeyState ks[4]{};
        check(kbd_reading->GetKeyState(0, ks) == 0, "GetKeyState(0, buf) returns 0");
        check(kbd_reading->GetKeyState(4, nullptr) == 0, "GetKeyState(4, null) returns 0");
        if (ro_page) {
            check(kbd_reading->GetKeyState(4, static_cast<KeyState*>(ro_page)) == 0,
                  "GetKeyState(4, PAGE_READONLY) returns 0");
        }
        if (edge_ptr) {
            check(kbd_reading->GetKeyState(4, static_cast<KeyState*>(edge_ptr)) == 0,
                  "GetKeyState(4, straddling) returns 0");
        }
        check(kbd_reading->GetKeyState(4, ks) == 2 && ks[0].virtualKey == 0x41 && ks[1].virtualKey == 0x42,
              "GetKeyState(4, valid) populates pressed keys");

        GamepadState gp{};
        check(!kbd_reading->GetGamepadState(&gp), "GetGamepadState on keyboard-only reading returns false");
        check(kbd_reading->GetSequenceNumber(kind_gamepad) == 0,
              "GetSequenceNumber(gamepad) on keyboard-only reading returns 0");
        kbd_reading->Release();
    }

    SetEnvironmentVariableA("X2W_INPUT_KEYS", nullptr);
    if (ro_page) VirtualFree(ro_page, 0, MEM_RELEASE);
    if (split_pages) VirtualFree(split_pages, 0, MEM_RELEASE);
    input->Release();
    FreeLibrary(mod);
    std::printf(failures ? "startup_gameinput_contracts_check: %d failure(s)\n"
                         : "startup_gameinput_contracts_check: ok\n",
                failures);
    return failures ? 1 : 0;
}
