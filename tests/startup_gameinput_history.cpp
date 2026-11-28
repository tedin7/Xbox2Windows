// Verifies GameInput.dll single-kind GetPreviousReading after cross-kind changes and concurrent GameInputCreate.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kind_keyboard = 0x00000010;
constexpr std::uint32_t kind_gamepad = 0x00040000;
constexpr HRESULT reading_not_found = static_cast<HRESULT>(0x838A0003L);

struct GamepadState {
    std::uint32_t buttons;
    float leftTrigger, rightTrigger;
    float leftThumbstickX, leftThumbstickY, rightThumbstickX, rightThumbstickY;
};
static_assert(sizeof(GamepadState) == 28, "GamepadState layout");

struct PadShared {
    std::uint32_t magic, buttons, title_read, reads;
};

struct IGameInputDevice {
    void* const* vtbl;
};

struct IGameInputReading {
    void* const* vtbl;

    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInputReading*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
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
    HRESULT GetPreviousReading(IGameInputReading* reference, std::uint32_t kind, IGameInputDevice* device,
                               IGameInputReading** reading)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, IGameInputReading*, std::uint32_t, IGameInputDevice*,
                                       IGameInputReading**);
        return reinterpret_cast<Fn>(vtbl[6])(this, reference, kind, device, reading);
    }
};

int check(bool ok, const char* msg)
{
    if (!ok) {
        std::printf("FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::printf("usage: %s <GameInput.dll>\n", argv[0]);
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) {
        std::printf("LoadLibraryA(%s) failed: %lu\n", argv[1], GetLastError());
        return 2;
    }
    using CreateFn = HRESULT(__stdcall*)(IGameInput**);
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(dll, "GameInputCreate"));
    if (!create) {
        std::printf("GetProcAddress(GameInputCreate) failed\n");
        return 2;
    }

    int failures = 0;

    // 1. Concurrent GameInputCreate + immediate vtable use.
    {
        constexpr int kThreads = 8;
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::atomic<int> thread_failures{0};
        std::vector<std::thread> threads;
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&] {
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire)) {}
                IGameInput* gi = nullptr;
                if (FAILED(create(&gi)) || !gi) {
                    thread_failures.fetch_add(1);
                    return;
                }
                IGameInputReading* r = nullptr;
                if (FAILED(gi->GetCurrentReading(kind_gamepad, nullptr, &r)) || !r) {
                    thread_failures.fetch_add(1);
                } else {
                    r->Release();
                }
                gi->Release();
            });
        }
        while (ready.load(std::memory_order_acquire) < kThreads) {}
        go.store(true, std::memory_order_release);
        for (auto& t : threads) t.join();
        failures += check(thread_failures.load() == 0, "concurrent GameInputCreate should succeed");
    }

    IGameInput* input = nullptr;
    failures += check(SUCCEEDED(create(&input)) && input != nullptr, "GameInputCreate");
    if (!input) return 1;

    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(PadShared),
                                        L"Local\\x2w_input_pad");
    PadShared* shared = mapping ? static_cast<PadShared*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(PadShared)))
                                : nullptr;
    failures += check(shared != nullptr, "Local\\x2w_input_pad mapping");
    if (!shared) {
        input->Release();
        return 1;
    }

    // 2. Boundary: first gamepad state (buttons=0), then keyboard-only change, then GetPreviousReading(kind_gamepad).
    shared->buttons = 0;
    SetEnvironmentVariableA("X2W_INPUT_KEYS", "41");
    IGameInputReading* r1 = nullptr;
    failures += check(SUCCEEDED(input->GetCurrentReading(kind_gamepad, nullptr, &r1)) && r1, "r1 GetCurrentReading");
    IGameInputDevice* pad_dev = nullptr;
    if (r1) r1->GetDevice(&pad_dev);
    const std::uint64_t seq1 = r1 ? r1->GetSequenceNumber(kind_gamepad) : 0;

    SetEnvironmentVariableA("X2W_INPUT_KEYS", "42");
    IGameInputReading* r1_after_key = nullptr;
    failures += check(SUCCEEDED(input->GetCurrentReading(kind_gamepad, pad_dev, &r1_after_key)) && r1_after_key,
                      "r1_after_key GetCurrentReading");
    if (r1_after_key) {
        failures += check(r1_after_key->GetSequenceNumber(kind_gamepad) == seq1,
                          "r1_after_key keeps same pad_sequence");
        IGameInputReading* before_first = nullptr;
        const HRESULT hr = input->GetPreviousReading(r1_after_key, kind_gamepad, pad_dev, &before_first);
        failures += check(hr == reading_not_found && before_first == nullptr,
                          "GetPreviousReading from first pad state after key change must return reading_not_found");
        if (before_first) before_first->Release();
        r1_after_key->Release();
    }

    // 3. Second gamepad state (buttons=4), then keyboard-only change, then GetPreviousReading(kind_gamepad).
    shared->buttons = 4;
    IGameInputReading* r2 = nullptr;
    failures += check(SUCCEEDED(input->GetCurrentReading(kind_gamepad, pad_dev, &r2)) && r2, "r2 GetCurrentReading");
    const std::uint64_t seq2 = r2 ? r2->GetSequenceNumber(kind_gamepad) : 0;
    failures += check(seq2 == seq1 + 1, "r2 advances pad_sequence");

    SetEnvironmentVariableA("X2W_INPUT_KEYS", "43");
    IGameInputReading* r3 = nullptr;
    failures += check(SUCCEEDED(input->GetCurrentReading(kind_gamepad, pad_dev, &r3)) && r3, "r3 GetCurrentReading");
    if (r3) {
        failures += check(r3->GetSequenceNumber(kind_gamepad) == seq2, "r3 keeps seq2 after key-only change");
        IGameInputReading* prev = nullptr;
        const HRESULT hr = input->GetPreviousReading(r3, kind_gamepad, pad_dev, &prev);
        failures += check(hr == S_OK && prev != nullptr, "GetPreviousReading(r3, kind_gamepad) succeeds");
        if (prev) {
            GamepadState gs{};
            failures += check(prev->GetSequenceNumber(kind_gamepad) == seq1,
                              "GetPreviousReading(r3, kind_gamepad) must return seq1, not duplicate seq2");
            failures += check(prev->GetGamepadState(&gs) && gs.buttons == 0,
                              "GetPreviousReading(r3, kind_gamepad) carries seq1 buttons=0");
            prev->Release();
        }
        // Cross-kind GetPreviousReading(r3, kind_keyboard) must still find the prior keyboard reading.
        IGameInputReading* kprev = nullptr;
        failures += check(SUCCEEDED(input->GetPreviousReading(r3, kind_keyboard, nullptr, &kprev)) && kprev != nullptr,
                          "cross-kind GetPreviousReading(r3, kind_keyboard) succeeds");
        if (kprev) kprev->Release();
        r3->Release();
    }

    // A combined reference captured the later keyboard event. The pad's second
    // reading precedes that event and remains the closest gamepad match.
    IGameInputReading* combined = nullptr;
    failures += check(SUCCEEDED(input->GetCurrentReading(kind_gamepad | kind_keyboard, nullptr, &combined)) && combined,
                      "combined GetCurrentReading after the second keyboard change");
    if (combined) {
        IGameInputReading* prev = nullptr;
        const HRESULT hr = input->GetPreviousReading(combined, kind_gamepad, pad_dev, &prev);
        GamepadState state{};
        failures += check(hr == S_OK && prev && prev->GetSequenceNumber(kind_gamepad) == seq2 &&
                          prev->GetGamepadState(&state) && state.buttons == 4,
                          "combined reference must retain the gamepad reading before its keyboard event");
        if (prev) prev->Release();
        combined->Release();
    }

    if (r2) r2->Release();
    if (r1) r1->Release();
    UnmapViewOfFile(shared);
    CloseHandle(mapping);
    input->Release();
    FreeLibrary(dll);
    return failures ? 1 : 0;
}
