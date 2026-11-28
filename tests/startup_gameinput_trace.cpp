// Checks the opt-in GameInput input trace (X2W_INPUT_TRACE=1). A child process plays one
// X2W_INPUT_SCRIPT through IGameInput::GetCurrentReading with X2W_INPUT_CLOCK_MS; the parent runs
// the same child with the switch on and off, so the control proves that the default path logs no
// x2w-input-trace line and still observes the same gamepad states.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "startup_stubs/present_signature.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr std::uint32_t kind_gamepad = 0x00040000;
constexpr std::uint32_t button_a = 0x4;
constexpr const char* trace_prefix = "x2w-input-trace";

struct GamepadState {
    std::uint32_t buttons;
    float leftTrigger, rightTrigger, leftThumbstickX, leftThumbstickY, rightThumbstickX, rightThumbstickY;
};
static_assert(sizeof(GamepadState) == 28, "GamepadState layout");

struct PadShared {
    std::uint32_t magic, buttons, title_read, reads;
};

struct IGameInputReading {
    const void* const* vtbl;
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
    const void* const* vtbl;
    std::uint32_t Release()
    {
        using Fn = std::uint64_t(__stdcall*)(IGameInput*);
        return static_cast<std::uint32_t>(reinterpret_cast<Fn>(vtbl[2])(this));
    }
    HRESULT GetCurrentReading(std::uint32_t kind, void* device, IGameInputReading** reading)
    {
        using Fn = HRESULT(__stdcall*)(IGameInput*, std::uint32_t, void*, IGameInputReading**);
        return reinterpret_cast<Fn>(vtbl[4])(this, kind, device, reading);
    }
};

int failures = 0;

void check(bool ok, const char* what)
{
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s\n", what);
}

void clear_shared_pad()
{
    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(PadShared), L"Local\\x2w_input_pad");
    if (auto* shared = mapping ? static_cast<PadShared*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(PadShared))) : nullptr)
        shared->buttons = 0;
}

std::string read_log(const std::string& path)
{
    std::string text;
    if (FILE* file = std::fopen(path.c_str(), "rb")) {
        char buffer[4096];
        for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) text.append(buffer, n);
        std::fclose(file);
    }
    return text;
}

std::size_t count(const std::string& text, const char* needle)
{
    std::size_t found = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++found;
    return found;
}

bool same_state(const GamepadState& state, std::uint32_t buttons, float lt, float rt, float lx, float ly, float rx, float ry)
{
    return state.buttons == buttons && state.leftTrigger == lt && state.rightTrigger == rt && state.leftThumbstickX == lx &&
           state.leftThumbstickY == ly && state.rightThumbstickX == rx && state.rightThumbstickY == ry;
}

bool expected_at(int ms, GamepadState& want)
{
    want = {};
    if (ms >= 2000 && ms < 2500) {
        want.buttons = button_a;
    } else if (ms >= 2500 && ms < 3000) {
        want.leftThumbstickX = 1.0f;
        want.leftThumbstickY = 0.5f;
    } else if (ms >= 3000 && ms < 3500) {
        want.leftThumbstickX = 1.0f;
    } else if (ms >= 3600 && ms < 3700) {
        want.buttons = 0x8;  // B press, shorter than the 250 ms trace heartbeat and off its grid
    }
    return true;
}

bool read_at(IGameInput* input, int ms, GamepadState& state)
{
    char text[24];
    std::snprintf(text, sizeof(text), "%d", ms);
    SetEnvironmentVariableA("X2W_INPUT_CLOCK_MS", text);
    IGameInputReading* reading = nullptr;
    const HRESULT hr = input->GetCurrentReading(kind_gamepad, nullptr, &reading);
    bool ok = SUCCEEDED(hr) && reading && reading->GetGamepadState(&state);
    if (reading) reading->Release();
    return ok;
}

bool xinput_pad_connected()
{
    struct State {
        DWORD packet;
        BYTE pad[12];
    };
    HMODULE module = LoadLibraryW(L"xinput1_4.dll");
    if (!module) return false;
    using Fn = DWORD(WINAPI*)(DWORD, State*);
    const auto get = reinterpret_cast<Fn>(GetProcAddress(module, "XInputGetState"));
    State state{};
    return get && get(0, &state) == 0;
}

// The script starts with a mark and a wait, then presses A and moves the left stick. The wait
// completes 1.5 s after the child publishes a new picture at 1.0 s; the entries that follow are
// shifted to 2.0, 2.5, 3.0 and 3.5 s.
constexpr const char* script_text =
    "mark=start@0,wait=0.5/2,4@+0.5+0.5,lx=1@+0.5+1,mark=turn@+0,ly=0.5@+0+0.5,8@+1.1+0.1";

int run_child(const char* dll, bool trace)
{
    clear_shared_pad();
    SetEnvironmentVariableA("X2W_INPUT_PRESS", nullptr);
    SetEnvironmentVariableA("X2W_INPUT_KEYS", nullptr);

    x2w_present_signature::Block* block = x2w_present_signature::create_block(GetCurrentProcessId(), 4);
    if (!block) {
        std::printf("child: create_block failed\n");
        return 2;
    }
    std::uint8_t first[x2w_present_signature::cells], second[x2w_present_signature::cells];
    std::memset(first, 40, sizeof(first));
    std::memset(second, 200, sizeof(second));
    x2w_present_signature::publish(block, first, 10);
    x2w_present_signature::note_present(block, 10);

    HMODULE module = LoadLibraryA(dll);
    if (!module) {
        std::printf("child: LoadLibraryA(%s) failed: %lu\n", dll, GetLastError());
        return 2;
    }
    using CreateFn = HRESULT(__stdcall*)(IGameInput**);
    const auto create = reinterpret_cast<CreateFn>(GetProcAddress(module, "GameInputCreate"));
    IGameInput* input = nullptr;
    if (!create || FAILED(create(&input)) || !input) {
        std::printf("child: GameInputCreate failed\n");
        FreeLibrary(module);
        return 2;
    }

    for (int ms = 0; ms <= 4000; ms += 50) {
        if (ms == 1000) {
            x2w_present_signature::publish(block, second, 11);
            x2w_present_signature::note_present(block, 11);
        }
        GamepadState state{};
        if (!read_at(input, ms, state)) {
            std::printf("child: GetCurrentReading failed at %d ms\n", ms);
            ++failures;
            break;
        }
        GamepadState want{};
        expected_at(ms, want);
        if (!same_state(state, want.buttons, want.leftTrigger, want.rightTrigger, want.leftThumbstickX, want.leftThumbstickY,
                        want.rightThumbstickX, want.rightThumbstickY)) {
            std::printf("child: %s at %d ms: buttons 0x%x lt %g rt %g lx %g ly %g rx %g ry %g\n", trace ? "trace" : "control",
                        ms, state.buttons, state.leftTrigger, state.rightTrigger, state.leftThumbstickX, state.leftThumbstickY,
                        state.rightThumbstickX, state.rightThumbstickY);
            ++failures;
            break;
        }
    }
    input->Release();
    FreeLibrary(module);

    char log_path[MAX_PATH]{};
    const DWORD log_length = GetEnvironmentVariableA("X2W_TRAP_LOG", log_path, MAX_PATH);
    if (!log_length || log_length >= MAX_PATH) {
        std::printf("child: X2W_TRAP_LOG missing\n");
        return 2;
    }
    const std::string log = read_log(log_path);
    if (trace) {
        const char* const state_line =
            "x2w-input-trace state buttons=0x00000004 lt=0.000 rt=0.000 lx=0.000 ly=0.000 rx=0.000 ry=0.000 "
            "script=2.000 present=11";
        const char* const stick_line =
            "x2w-input-trace state buttons=0x00000000 lt=0.000 rt=0.000 lx=1.000 ly=0.500 rx=0.000 ry=0.000 "
            "script=2.500 present=11";
        const char* const b_line =
            "x2w-input-trace state buttons=0x00000008 lt=0.000 rt=0.000 lx=0.000 ly=0.000 rx=0.000 ry=0.000 "
            "script=3.600 present=11";
        check(count(log, state_line) == 1, "trace: A press line with script time 2.000 and present 11");
        check(count(log, stick_line) == 1, "trace: left stick and Y axis line at script time 2.500");
        check(count(log, b_line) == 1, "trace: off-grid B press line at script time 3.600");
        check(count(log, "x2w-input-trace wait event=begun item=2 at=0.000 elapsed=0.000 signal=1") == 1,
              "trace: wait begun line");
        check(count(log, "x2w-input-trace wait event=changed item=2 at=1.000 elapsed=1.000 signal=1") == 1,
              "trace: wait changed line");
        check(count(log, "x2w-input-trace wait event=done item=2 at=1.500 elapsed=1.500 signal=1") == 1,
              "trace: wait done line");
        check(count(log, "x2w-input-trace mark label=start item=1 at=0.000 present=10") == 1, "trace: start mark line");
        check(count(log, "x2w-input-trace mark label=turn item=5 at=2.500 present=11") == 1, "trace: turn mark line");
        check(count(log, "x2w-input-trace wait event=timeout") == 0, "trace: no timeout line");
        const std::size_t states = count(log, "x2w-input-trace state ");
        check(states >= 12 && states <= 21, "trace: bounded state lines (12..21 in four seconds)");
    } else {
        check(count(log, trace_prefix) == 0, "control: no x2w-input-trace line without the switch");
        check(count(log, "x2w-input script mark start at 0.000 s (item 1, present 10)") == 1,
              "control: the script and its ordinary log still run");
    }
    return failures ? 1 : 0;
}

std::string temp_log_path(const char* suffix)
{
    char temp[MAX_PATH];
    GetTempPathA(MAX_PATH, temp);
    return std::string(temp) + "x2w_gameinput_trace_" + suffix + "_" + std::to_string(GetCurrentProcessId()) + ".log";
}

int run_mode(const char* self, const char* dll, bool trace)
{
    const std::string log_path = temp_log_path(trace ? "on" : "off");
    DeleteFileA(log_path.c_str());
    SetEnvironmentVariableA("X2W_TRAP_LOG", log_path.c_str());
    SetEnvironmentVariableA("X2W_INPUT_TRACE", trace ? "1" : "0");
    SetEnvironmentVariableA("X2W_INPUT_SCRIPT", script_text);
    clear_shared_pad();

    const std::string command = std::string("\"") + self + "\" \"" + dll + "\" child " + (trace ? "trace" : "control");
    STARTUPINFOA startup{sizeof(startup)};
    PROCESS_INFORMATION process{};
    if (!CreateProcessA(nullptr, const_cast<char*>(command.c_str()), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup,
                        &process)) {
        std::printf("CreateProcess(%s) failed: %lu\n", trace ? "trace" : "control", GetLastError());
        return 2;
    }
    DWORD code = 1;
    if (WaitForSingleObject(process.hProcess, 60000) != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1);
        std::printf("%s child timed out\n", trace ? "trace" : "control");
    }
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (code != 0) {
        const std::string log = read_log(log_path);
        std::printf("%s child failed (exit %lu), log follows:\n%.4000s\n", trace ? "trace" : "control", code, log.c_str());
        DeleteFileA(log_path.c_str());
        return 1;
    }
    std::printf("PASS: startup_gameinput_trace %s\n", trace ? "trace" : "control");
    DeleteFileA(log_path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::printf("usage: startup_gameinput_trace <GameInput.dll> [child trace|control]\n");
        return 2;
    }
    if (argc >= 4 && std::strcmp(argv[2], "child") == 0) {
        if (std::strcmp(argv[3], "trace") != 0 && std::strcmp(argv[3], "control") != 0) {
            std::printf("unknown child mode %s\n", argv[3]);
            return 2;
        }
        return run_child(argv[1], std::strcmp(argv[3], "trace") == 0);
    }
    if (xinput_pad_connected()) {
        std::printf("SKIP: XInput pad 0 is connected; its values merge into every reading\n");
        return 77;
    }
    char self[MAX_PATH]{};
    if (!GetModuleFileNameA(nullptr, self, MAX_PATH)) {
        std::printf("GetModuleFileNameA failed: %lu\n", GetLastError());
        return 2;
    }
    int status = run_mode(self, argv[1], true);
    if (!status) status = run_mode(self, argv[1], false);
    return status ? status : (failures ? 1 : 0);
}
