// Experimental (docs/startup-runtime.md): GameInput.dll stand-in with one virtual gamepad.
//
// The title imports GameInput.dll by ordinal 1 (GameInputCreate). This stand-in implements the
// IGameInput / IGameInputDevice / IGameInputReading shapes of the public GameInput.h (version 0
// interface) for a single gamepad whose state comes from the keyboard while a window of this
// process has focus, merged with XInput pad 0 if one is connected:
//
//   W A S D left stick     arrows d-pad         I J K L right stick
//   Space / Enter  A       Esc / Backspace  B   X  X      Y  Y
//   Q / E  shoulders       Z / C  triggers      Tab  View   F1 / M  Menu   F / G  stick clicks
//
// Every method logs its first calls ("x2w-input"), so the slots the title really uses are visible.
#define X2W_TRAP_MODULE "GameInput.dll"
#define X2W_TRAP_MODULE_ID 8
#include "startup_stubs/trap.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/qpc_microseconds.h"
#include "startup_stubs/input_script.h"
#include "startup_stubs/present_signature.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {
using u64 = std::uint64_t;
constexpr HRESULT device_not_found = static_cast<HRESULT>(0x838A0002);
constexpr HRESULT reading_not_found = static_cast<HRESULT>(0x838A0003);
constexpr HRESULT reading_too_old = static_cast<HRESULT>(0x838A0004);
constexpr std::uint32_t kind_keyboard = 0x00000010;
constexpr std::uint32_t kind_gamepad = 0x00040000, kind_controller_bits = 0x0000000E;
constexpr std::uint32_t status_connected = 0x47;  // connected, input enabled, output enabled, synchronized
constexpr std::uint32_t family_xbox_one = 1;      // GameInputFamilyXboxOne (GameInputDeviceFamily)
constexpr std::size_t max_keys = 32;
constexpr std::size_t history_capacity = 64;

struct KeyState {
    std::uint32_t scanCode;
    std::uint32_t codePoint;
    std::uint8_t virtualKey;
    bool isDeadKey;
};
static_assert(sizeof(KeyState) == 12, "KeyState layout");

struct GamepadState {
    std::uint32_t buttons;
    float left_trigger, right_trigger, left_x, left_y, right_x, right_y;
};
enum : std::uint32_t {
    b_menu = 1, b_view = 2, b_a = 4, b_b = 8, b_x = 0x10, b_y = 0x20, b_up = 0x40, b_down = 0x80, b_left = 0x100, b_right = 0x200,
    b_lb = 0x400, b_rb = 0x800, b_ls = 0x1000, b_rs = 0x2000};

void note(const char* what, LONG& calls, u64 a = 0, u64 b = 0, u64 c = 0)
{
    if (x2w_local::first_calls(calls, 3)) x2w_local::logf("x2w-input %s(0x%llx, 0x%llx, 0x%llx)", what, a, b, c);
}

u64 now_microseconds()
{
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return x2w_startup::qpc_microseconds(static_cast<u64>(counter.QuadPart), static_cast<u64>(frequency.QuadPart));
}

bool process_has_focus()
{
    const HWND window = GetForegroundWindow();
    DWORD process = 0;
    if (window) GetWindowThreadProcessId(window, &process);
    return process == GetCurrentProcessId();
}

// XInput is optional: loaded on first use, never required.
struct XInputPad { WORD buttons; BYTE lt, rt; SHORT lx, ly, rx, ry; };
struct XInputState { DWORD packet; XInputPad pad; };
using XInputGetStateFn = DWORD(WINAPI*)(DWORD, XInputState*);
XInputGetStateFn xinput_get_state()
{
    static const XInputGetStateFn function = [] {
        const HMODULE module = LoadLibraryW(L"xinput1_4.dll");
        return module ? reinterpret_cast<XInputGetStateFn>(GetProcAddress(module, "XInputGetState")) : nullptr;
    }();
    return function;
}

// External input: a named shared block "Local\x2w_input_pad" whose buttons a separate tool may set
// (OR-ed into every sample); the buttons the title last read are written back, so such a tool can show
// what reached the title. Layout: magic "XPAD", buttons, buttons read, read count (four u32).
struct PadShared { std::uint32_t magic, buttons, title_read, reads; };
PadShared* pad_shared()
{
    static PadShared* const shared = [] {
        HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(PadShared), L"Local\\x2w_input_pad");
        if (!mapping) return static_cast<PadShared*>(nullptr);
        auto* view = static_cast<PadShared*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(PadShared)));
        if (view) view->magic = 0x44415058;  // "XPAD"
        return view;
    }();
    return shared;
}

// Milliseconds since this process started. X2W_INPUT_CLOCK_MS replaces this clock for the input script
// (test seam: tests/startup_gameinput_script_syntax.cpp sets it before each poll).
u64 process_milliseconds()
{
    const DWORD saved = GetLastError();
    char text[24]{};
    const DWORD length = GetEnvironmentVariableA("X2W_INPUT_CLOCK_MS", text, sizeof(text));
    SetLastError(saved);
    if (length && length < sizeof(text)) return std::strtoull(text, nullptr, 10);
    static const ULONGLONG started = [] {
        FILETIME created{}, exited{}, kernel{}, user{}, now{};
        GetSystemTimeAsFileTime(&now);
        const ULONGLONG tick = GetTickCount64();
        if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return tick;
        const auto value = [](const FILETIME& f) { return (static_cast<ULONGLONG>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
        const ULONGLONG age = value(now) > value(created) ? (value(now) - value(created)) / 10000 : 0;
        return age < tick ? tick - age : 0;
    }();
    return GetTickCount64() - started;
}

// X2W_INPUT_SCRIPT (src/startup_stubs/input_script.h), parsed once on GameInputCreate or the first poll.
// At most 4,095 characters (a longer value is ignored) and 256 entries, as before; every rejected item is
// also logged with its index and reason.
const x2w_input_script::Script& input_script()
{
    static const x2w_input_script::Script script = [] {
        x2w_input_script::Script parsed;
        constexpr DWORD max_characters = 4095;
        const DWORD saved = GetLastError();
        const DWORD size = GetEnvironmentVariableA("X2W_INPUT_SCRIPT", nullptr, 0);  // characters + 1, 0 if unset
        std::string text(size ? size : 1, '\0');
        const DWORD length = size && size - 1 <= max_characters ? GetEnvironmentVariableA("X2W_INPUT_SCRIPT", text.data(), size) : 0;
        SetLastError(saved);
        if (size && size - 1 > max_characters) {
            x2w_local::logf("x2w-input X2W_INPUT_SCRIPT ignored: %lu characters, at most %u", static_cast<unsigned long>(size - 1),
                            static_cast<unsigned>(max_characters));
            return parsed;
        }
        if (!length || length >= size) return parsed;
        text.resize(length);
        parsed = x2w_input_script::parse(text.c_str());
        const std::size_t rejected = parsed.rejections.size();
        std::size_t malformed = 0;
        for (const x2w_input_script::Rejection& r : parsed.rejections) malformed += r.reason != "over 256 entries";
        if (rejected)
            x2w_local::logf("x2w-input X2W_INPUT_SCRIPT: %d of %d entries used (%d malformed, at most %d)",
                            static_cast<int>(parsed.items - rejected), static_cast<int>(parsed.items), static_cast<int>(malformed),
                            static_cast<int>(x2w_input_script::max_entries));
        x2w_local::logf("x2w-input script: %zu entries accepted, %zu rejected, time base %s", parsed.entries.size(), rejected,
                        parsed.from_process_start ? "process-start" : "first-poll");
        constexpr std::size_t listed = 32;
        for (std::size_t i = 0; i < rejected && i < listed; ++i) {
            const x2w_input_script::Rejection& r = parsed.rejections[i];
            x2w_local::logf("x2w-input script item %zu rejected (%s): \"%.48s\"", r.index, r.reason.c_str(), r.text.c_str());
        }
        if (rejected > listed) x2w_local::logf("x2w-input script: %zu more rejected items not listed", rejected - listed);
        if (!parsed.waits.empty() || !parsed.marks.empty())
            x2w_local::logf("x2w-input script: %zu waits, %zu marks; picture signal from Local\\x2w_present_signature_%lu",
                            parsed.waits.size(), parsed.marks.size(), static_cast<unsigned long>(GetCurrentProcessId()));
        if (!parsed.groups.empty()) x2w_local::logf("x2w-input script: %zu groups", parsed.groups.size());
        // The presenter samples the picture only for scripts that wait on it.
        if (!parsed.waits.empty())
            x2w_local::logf("x2w-input script: picture signature %s",
                            x2w_present_signature::request_sampling(GetCurrentProcessId()) ? "requested" : "could not be requested");
        // A wait with ;fine reads the 64 x 72 grid; the presenter copies it only while asked.
        if (parsed.wants_fine())
            x2w_local::logf("x2w-input script: fine picture signature (64 x 72) %s",
                            x2w_present_signature::request_fine(GetCurrentProcessId()) ? "requested" : "could not be requested");
        return parsed;
    }();
    return script;
}

// Seconds on the script's time base; the first call fixes the first-poll zero and logs the resolved base.
double script_seconds(const x2w_input_script::Script& script)
{
    static const u64 zero = [&] {
        const u64 first_poll = process_milliseconds();
        const u64 base = script.from_process_start ? 0 : first_poll;
        x2w_local::logf("x2w-input script time base %s: first poll at %.3f s, first entry at %.3f s since process start",
                        script.from_process_start ? "process-start" : "first-poll", first_poll / 1000.0,
                        base / 1000.0 + x2w_input_script::first_start(script));
        return base;
    }();
    return static_cast<double>(static_cast<long long>(process_milliseconds() - zero)) / 1000.0;
}

// The presenter's picture signature (src/startup_stubs/present_signature.h), opened by process id. While
// no presenter has created it, opening is retried at most every 0.5 s.
static_assert(x2w_input_script::signal_cells == x2w_present_signature::cells &&
                  x2w_input_script::fine_cells == x2w_present_signature::fine_cells, "signature grid size");
bool read_signal(x2w_input_script::Signal& out)
{
    static const x2w_present_signature::Block* block;
    static ULONGLONG next_try;
    if (!block) {
        const ULONGLONG tick = GetTickCount64();
        if (tick < next_try) return false;
        next_try = tick + 500;
        const DWORD saved = GetLastError();
        block = x2w_present_signature::open_block(GetCurrentProcessId());
        SetLastError(saved);
        if (!block) return false;
    }
    x2w_present_signature::Snapshot snapshot;
    if (!x2w_present_signature::read(block, snapshot)) return false;
    out.available = true;
    out.sequence = snapshot.sequence;
    out.present = snapshot.presents;
    std::memcpy(out.cells, snapshot.luma, sizeof(out.cells));
    out.fine_available = snapshot.fine_valid;
    if (snapshot.fine_valid) std::memcpy(out.fine, snapshot.fine, sizeof(out.fine));
    return true;
}

void log_event(const x2w_input_script::Event& e)
{
    using x2w_input_script::Event;
    switch (e.type) {
    case Event::wait_begun:
        x2w_local::logf("x2w-input script wait item %zu begun at %.3f s%s", e.item, e.at, e.signal ? "" : " (no presenter signal yet)");
        break;
    case Event::wait_changed:
        x2w_local::logf("x2w-input script wait item %zu: picture changed at %.3f s, %.3f s after the wait began", e.item, e.at, e.elapsed);
        break;
    case Event::wait_done:
        x2w_local::logf("x2w-input script wait item %zu done at %.3f s after %.3f s", e.item, e.at, e.elapsed);
        break;
    case Event::wait_timeout:
        if (!e.label.empty())
            x2w_local::logf("x2w-input script wait item %zu timed out at %.3f s after %.3f s%s; group %s is skipped from here", e.item, e.at,
                            e.elapsed, e.signal ? "" : " (no presenter signal)", e.label.c_str());
        else
            x2w_local::logf("x2w-input script wait item %zu timed out at %.3f s after %.3f s%s; the script %s", e.item, e.at, e.elapsed,
                            e.signal ? "" : " (no presenter signal)", e.stopped ? "stops" : "goes on");
        break;
    case Event::group_skipped:
        x2w_local::logf("x2w-input script group %s (items %zu-%zu) failed at step item %zu: rest of the group skipped, "
                        "the script goes on after item %zu from %.3f s",
                        e.label.c_str(), e.group_item, e.end_item, e.item, e.end_item, e.at);
        break;
    case Event::mark:
        if (e.signal)
            x2w_local::logf("x2w-input script mark %s at %.3f s (item %zu, present %llu)", e.label.c_str(), e.at, e.item,
                            static_cast<unsigned long long>(e.present));
        else
            x2w_local::logf("x2w-input script mark %s at %.3f s (item %zu, no presenter signal)", e.label.c_str(), e.at, e.item);
        break;
    }
}

// Scripts with waits or marks: one timeline for every polling thread.
SRWLOCK timeline_lock = SRWLOCK_INIT;
x2w_input_script::Output play_timeline(const x2w_input_script::Script& script, double t, double& end)
{
    static x2w_input_script::Timeline timeline(script);
    static x2w_input_script::Signal signal;
    std::vector<x2w_input_script::Event> events;
    AcquireSRWLockExclusive(&timeline_lock);
    x2w_input_script::Signal latest;
    if (read_signal(latest)) signal = latest;
    timeline.advance(t, signal, events);
    const x2w_input_script::Output out = timeline.play(t);
    end = timeline.end();
    ReleaseSRWLockExclusive(&timeline_lock);
    for (const x2w_input_script::Event& e : events) log_event(e);
    return out;
}

GamepadState sample()
{
    GamepadState s{};
    if (PadShared* shared = pad_shared()) s.buttons |= shared->buttons;
    if (process_has_focus()) {
        const auto down = [](int key) { return (GetAsyncKeyState(key) & 0x8000) != 0; };
        s.left_x = (down('D') ? 1.0f : 0.0f) - (down('A') ? 1.0f : 0.0f);
        s.left_y = (down('W') ? 1.0f : 0.0f) - (down('S') ? 1.0f : 0.0f);
        s.right_x = (down('L') ? 1.0f : 0.0f) - (down('J') ? 1.0f : 0.0f);
        s.right_y = (down('I') ? 1.0f : 0.0f) - (down('K') ? 1.0f : 0.0f);
        s.left_trigger = down('Z') ? 1.0f : 0.0f;
        s.right_trigger = down('C') ? 1.0f : 0.0f;
        if (down(VK_SPACE) || down(VK_RETURN)) s.buttons |= b_a;
        if (down(VK_ESCAPE) || down(VK_BACK)) s.buttons |= b_b;
        if (down('X')) s.buttons |= b_x;
        if (down('Y')) s.buttons |= b_y;
        if (down(VK_UP)) s.buttons |= b_up;
        if (down(VK_DOWN)) s.buttons |= b_down;
        if (down(VK_LEFT)) s.buttons |= b_left;
        if (down(VK_RIGHT)) s.buttons |= b_right;
        if (down('Q')) s.buttons |= b_lb;
        if (down('E')) s.buttons |= b_rb;
        if (down('F')) s.buttons |= b_ls;
        if (down('G')) s.buttons |= b_rs;
        if (down(VK_TAB)) s.buttons |= b_view;
        if (down(VK_F1) || down('M')) s.buttons |= b_menu;
    }
    XInputState pad{};
    if (const XInputGetStateFn get = xinput_get_state(); get && get(0, &pad) == 0) {
        const WORD w = pad.pad.buttons;
        const std::uint32_t map[][2] = {{0x0010, b_menu}, {0x0020, b_view}, {0x1000, b_a}, {0x2000, b_b}, {0x4000, b_x}, {0x8000, b_y},
                                        {0x0001, b_up}, {0x0002, b_down}, {0x0004, b_left}, {0x0008, b_right}, {0x0100, b_lb},
                                        {0x0200, b_rb}, {0x0040, b_ls}, {0x0080, b_rs}};
        for (const auto& entry : map)
            if (w & entry[0]) s.buttons |= entry[1];
        const auto axis = [](SHORT value) { return std::abs(value) < 7000 ? 0.0f : value / 32767.0f; };
        const auto merge = [](float& into, float value) { if (std::fabs(value) > std::fabs(into)) into = value; };
        merge(s.left_x, axis(pad.pad.lx));
        merge(s.left_y, axis(pad.pad.ly));
        merge(s.right_x, axis(pad.pad.rx));
        merge(s.right_y, axis(pad.pad.ry));
        merge(s.left_trigger, pad.pad.lt / 255.0f);
        merge(s.right_trigger, pad.pad.rt / 255.0f);
    }
    // X2W_INPUT_PRESS=<button mask>,<start s>,<period s>: presses those buttons for 0.2 s at every period
    // (unattended testing, e.g. 0x20,70,5 presses Y every five seconds from second 70).
    static const struct Auto { std::uint32_t mask; double start, period; ULONGLONG zero; } automatic = [] {
        Auto a{0, 0, 0, GetTickCount64()};
        char text[64]{};
        if (GetEnvironmentVariableA("X2W_INPUT_PRESS", text, sizeof(text))) {
            unsigned mask = 0;
            if (std::sscanf(text, "%x,%lf,%lf", &mask, &a.start, &a.period) == 3) a.mask = mask;
        }
        return a;
    }();
    if (automatic.mask && automatic.period > 0) {
        const double t = (GetTickCount64() - automatic.zero) / 1000.0 - automatic.start;
        if (t >= 0 && std::fmod(t, automatic.period) < 0.2) s.buttons |= automatic.mask;
    }
    // X2W_INPUT_SCRIPT (src/startup_stubs/input_script.h): timed presses, holds and axis values, e.g.
    // 4@60,1@70,20@80 presses A, then Menu, then Y for 0.3 s each; 4@60+2 holds A; lx=-1@60+1.5 holds left.
    // Waits and marks (wait=..., mark=..., @+<s>) go through the timeline; other scripts as before.
    const x2w_input_script::Script& script = input_script();
    const bool timeline = !script.waits.empty() || !script.marks.empty();
    if (!script.entries.empty() || timeline) {
        const double t = script_seconds(script);
        double end = -1;
        const x2w_input_script::Output out = timeline ? play_timeline(script, t, end) : x2w_input_script::play(script, t);
        s.buttons |= out.buttons;
        // One line per button entry the first time a poll sees it due (fixed-time scripts; entries in script order): the entry,
        // its mask, its time and how late the first poll came, so a run shows which presses the title could have seen and how
        // often it polls. Waits shift later entries in a timeline script, so those are not logged here.
        if (!timeline) {
            static volatile LONG next_logged;
            for (;;) {
                const LONG index = next_logged;
                if (index < 0 || static_cast<std::size_t>(index) >= script.entries.size()) break;
                const x2w_input_script::Entry& due = script.entries[static_cast<std::size_t>(index)];
                if (due.at > t) break;
                if (InterlockedCompareExchange(&next_logged, index + 1, index) != index) continue;
                if (due.axis < 0)
                    x2w_local::logf("x2w-input script press item %ld mask 0x%x due at %.3f s for %.2f s, first poll at %.3f s (%.3f s late)",
                                    index + 1, static_cast<unsigned>(due.mask), due.at, due.duration, t, t - due.at);
            }
        }
        float* const axes[x2w_input_script::axis_count] = {&s.left_trigger, &s.right_trigger, &s.left_x, &s.left_y, &s.right_x, &s.right_y};
        for (int i = 0; i < x2w_input_script::axis_count; ++i)
            if (out.active[i] && std::fabs(out.axes[i]) > std::fabs(*axes[i])) *axes[i] = out.axes[i];
        static volatile LONG finished;
        static const double last_end = x2w_input_script::last_end(script);
        if (!timeline) end = last_end;
        if (end >= 0 && t >= end && InterlockedExchange(&finished, 1) == 0)
            x2w_local::logf("x2w-input script time passed its last entry (%zu entries, ended at %.3f s on the script time base)",
                            script.entries.size(), end);
    }
    return s;
}

std::uint32_t sample_keys(KeyState* out, std::size_t capacity)
{
    std::uint32_t count = 0;
    auto push_vk = [&](std::uint32_t vk) {
        // VK_SHIFT/VK_CONTROL/VK_MENU are aliases of the left/right keys, never a physical key of their own.
        if (count >= capacity || vk == 0 || vk > 0xFE || vk == VK_SHIFT || vk == VK_CONTROL || vk == VK_MENU) return;
        const auto vkey = static_cast<std::uint8_t>(vk);
        for (std::uint32_t i = 0; i < count; ++i)
            if (out[i].virtualKey == vkey) return;
        KeyState ks{};
        // _EX keeps the 0xE0/0xE1 prefix, so Left Arrow (0xE04B) differs from keypad 4 (0x4B).
        ks.scanCode = static_cast<std::uint32_t>(MapVirtualKeyW(vk, MAPVK_VK_TO_VSC_EX));
        // The navigation cluster shares scan codes with keypad keys and is reported unprefixed by some Windows builds.
        const bool extended_key = (vk >= VK_PRIOR && vk <= VK_DOWN) || vk == VK_INSERT || vk == VK_DELETE ||
                                  vk == VK_DIVIDE || vk == VK_LWIN || vk == VK_RWIN || vk == VK_APPS;
        if (extended_key && ks.scanCode != 0 && (ks.scanCode & 0xFF00u) == 0) ks.scanCode |= 0xE000u;
        const UINT ch = MapVirtualKeyW(vk, MAPVK_VK_TO_CHAR);
        if ((ch & 0x80000000u) == 0 && (ch & 0xFFFFu) != 0) ks.codePoint = ch & 0xFFFFu;
        ks.virtualKey = vkey;
        ks.isDeadKey = false;
        out[count++] = ks;
    };

    char env[256]{};
    if (GetEnvironmentVariableA("X2W_INPUT_KEYS", env, sizeof(env)) > 0) {
        for (char* item = std::strtok(env, ", "); item && count < capacity; item = std::strtok(nullptr, ", ")) {
            unsigned vk = 0;
            if (std::sscanf(item, "%x", &vk) == 1) push_vk(vk);
        }
    }
    if (process_has_focus()) {
        for (std::uint32_t vk = 0x08; vk <= 0xFE && count < capacity; ++vk) {
            if ((GetAsyncKeyState(static_cast<int>(vk)) & 0x8000) != 0) push_vk(vk);
        }
    }
    return count;
}

struct Object {
    void* const* vtable;
    LONG refs;
};
template <std::size_t N>
struct Table { void* slots[N]; };

u64 __stdcall add_ref(Object* self) { return static_cast<u64>(InterlockedIncrement(&self->refs)); }
u64 __stdcall release_static(Object* self) { return static_cast<u64>(InterlockedDecrement(&self->refs)); }
HRESULT __stdcall query_interface(Object* self, const GUID*, void** out)
{
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    InterlockedIncrement(&self->refs);
    *out = self;
    return S_OK;
}
template <int Family, int N>
__declspec(noinline) u64 unknown_slot(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    static LONG calls;
    if (x2w_local::first_calls(calls, 2))
        x2w_local::logf("x2w-input %s::slot%d(0x%llx, 0x%llx, 0x%llx) not implemented -> 0",
                        Family == 0 ? "GameInput" : Family == 1 ? "Device" : "Reading", N, rdx, r8, r9);
    (void)rcx;
    return 0;
}
template <int Family, std::size_t... I>
void fill_unknown(void** slots, std::index_sequence<I...>)
{
    ((slots[I] = reinterpret_cast<void*>(&unknown_slot<Family, static_cast<int>(I)>)), ...);
}

// ---- device ------------------------------------------------------------------------------
struct DeviceObj : Object {
    const unsigned char* info;
};
Table<32> device_table;
unsigned char device_info[0x400];
unsigned char keyboard_device_info[0x400];
DeviceObj the_device{{device_table.slots, 1}, device_info};
DeviceObj the_keyboard_device{{device_table.slots, 1}, keyboard_device_info};

const void* __stdcall device_get_info(DeviceObj* self)
{
    static LONG calls;
    note("Device::GetDeviceInfo", calls);
    return self->info;
}
std::uint32_t __stdcall device_get_status(Object*) { return status_connected; }
void __stdcall device_set_rumble(Object*, const void*) {}

// ---- readings ----------------------------------------------------------------------------
struct Reading : Object {
    std::uint32_t kind;
    GamepadState state;
    std::uint32_t key_count;
    KeyState keys[max_keys];
    u64 timestamp;
    u64 order;
    u64 pad_sequence, key_sequence;  // per-kind sequence numbers, as GameInput documents them
};
constexpr std::uint32_t history_pad = 1, history_keys = 2;
struct HistoryEntry {
    u64 order;  // internal ordering across kinds; never exposed to the title
    u64 pad_sequence, key_sequence;
    u64 timestamp;
    GamepadState pad;
    std::uint32_t key_count;
    KeyState keys[max_keys];
};
// One retention ring per kind, indexed by that kind's own sequence: keyboard traffic never evicts gamepad history.
struct KindRing {
    HistoryEntry entries[history_capacity];
    u64 latest;  // latest sequence of this kind (0 = none yet)
    std::size_t count;
};
Table<32> reading_table;
KindRing pad_ring, key_ring;
u64 order_counter;
HistoryEntry latest_entry;
SRWLOCK state_lock = SRWLOCK_INIT;

bool wants_gamepad(std::uint32_t kind) { return (kind & (kind_gamepad | kind_controller_bits)) != 0; }
bool wants_keyboard(std::uint32_t kind) { return (kind & kind_keyboard) != 0; }
bool wants_any(std::uint32_t kind) { return wants_gamepad(kind) || wants_keyboard(kind); }

u64 __stdcall reading_release(Reading* self)
{
    const LONG left = InterlockedDecrement(&self->refs);
    if (left == 0) delete self;
    return static_cast<u64>(left);
}
std::uint32_t __stdcall reading_kind(Reading* self) { return self->kind; }
u64 __stdcall reading_sequence(Reading* self, std::uint32_t kind)
{
    if ((kind & kind_keyboard) && (self->kind & kind_keyboard)) return self->key_sequence;
    if (wants_gamepad(kind) && wants_gamepad(self->kind)) return self->pad_sequence;
    return 0;
}
u64 __stdcall reading_timestamp(Reading* self) { return self->timestamp; }
void __stdcall reading_device(Reading* self, Object** device)
{
    if (device && x2w_local::writable(device, sizeof(*device))) {
        DeviceObj* d = (self->kind == kind_keyboard) ? &the_keyboard_device : &the_device;
        InterlockedIncrement(&d->refs);
        *device = d;
    }
}
std::uint32_t __stdcall reading_key_count(Reading* self)
{
    static LONG calls;
    note("Reading::GetKeyCount", calls);
    if ((self->kind & kind_keyboard) == 0) return 0;
    return self->key_count;
}
std::uint32_t __stdcall reading_key_state(Reading* self, std::uint32_t state_array_count, KeyState* state_array)
{
    static LONG calls;
    note("Reading::GetKeyState", calls, state_array_count);
    if ((self->kind & kind_keyboard) == 0 || !state_array || state_array_count == 0) return 0;
    const std::uint32_t count = self->key_count < state_array_count ? self->key_count : state_array_count;
    if (count == 0 || !x2w_local::writable(state_array, static_cast<std::size_t>(count) * sizeof(KeyState))) return 0;
    std::memcpy(state_array, self->keys, static_cast<std::size_t>(count) * sizeof(KeyState));
    return count;
}
const ULONGLONG loaded_at = GetTickCount64();
bool __stdcall reading_gamepad(Reading* self, GamepadState* out)
{
    static LONG calls, presses;
    note("Reading::GetGamepadState", calls);
    if (!out || !x2w_local::writable(out, sizeof(*out)) || (self->kind & (kind_gamepad | kind_controller_bits)) == 0)
        return false;
    *out = self->state;
    // Each new button combination the title reads is logged once per change (at most 64 lines).
    static volatile LONG last_logged;
    const LONG buttons = static_cast<LONG>(self->state.buttons);
    if (buttons && InterlockedExchange(&last_logged, buttons) != buttons && x2w_local::first_calls(presses, 64))
        x2w_local::logf("x2w-input gamepad buttons 0x%x read by the title at %.1f s", self->state.buttons,
                        (GetTickCount64() - loaded_at) / 1000.0);
    if (!buttons) InterlockedExchange(&last_logged, 0);
    if (PadShared* shared = pad_shared()) {
        shared->title_read = self->state.buttons;
        InterlockedIncrement(reinterpret_cast<volatile LONG*>(&shared->reads));
    }
    return true;
}


std::uint32_t resolve_reading_kind(std::uint32_t requested, Object* device)
{
    if (device == &the_device) return kind_gamepad;
    if (device == &the_keyboard_device) return kind_keyboard;
    std::uint32_t result = 0;
    if (wants_gamepad(requested)) result |= kind_gamepad;
    if (wants_keyboard(requested)) result |= kind_keyboard;
    return result;
}

HistoryEntry capture_current()
{
    const GamepadState pad = sample();
    KeyState keys[max_keys]{};
    const std::uint32_t key_count = sample_keys(keys, max_keys);

    AcquireSRWLockExclusive(&state_lock);
    std::uint32_t changed = (order_counter == 0) ? (history_pad | history_keys) : 0;
    if (!changed) {
        if (std::memcmp(&pad, &latest_entry.pad, sizeof(pad)) != 0) changed |= history_pad;
        if (key_count != latest_entry.key_count ||
            (key_count > 0 && std::memcmp(keys, latest_entry.keys, key_count * sizeof(KeyState)) != 0))
            changed |= history_keys;
    }
    if (changed) {
        latest_entry.order = ++order_counter;
        latest_entry.timestamp = now_microseconds();
        latest_entry.pad = pad;
        latest_entry.key_count = key_count;
        if (key_count > 0) std::memcpy(latest_entry.keys, keys, key_count * sizeof(KeyState));
        if (changed & history_pad) latest_entry.pad_sequence = ++pad_ring.latest;
        if (changed & history_keys) latest_entry.key_sequence = ++key_ring.latest;
        if (changed & history_pad) {
            pad_ring.entries[pad_ring.latest % history_capacity] = latest_entry;
            if (pad_ring.count < history_capacity) ++pad_ring.count;
        }
        if (changed & history_keys) {
            key_ring.entries[key_ring.latest % history_capacity] = latest_entry;
            if (key_ring.count < history_capacity) ++key_ring.count;
        }
    }
    const HistoryEntry result = latest_entry;
    ReleaseSRWLockExclusive(&state_lock);
    return result;
}

std::uint32_t history_mask(std::uint32_t reading_kind)
{
    return (wants_gamepad(reading_kind) ? history_pad : 0) | ((reading_kind & kind_keyboard) ? history_keys : 0);
}

// Finds the nearest entry before or after the reference position in one kind's ring.
// Returns 0 on hit, -1 if there is none (not found), -2 if the wanted entry was already evicted (too old).
int find_in_ring(const KindRing& ring, u64 ref_sequence, u64 ref_order, int step, bool same_kind_only, HistoryEntry& out)
{
    if (ring.count == 0) return -1;
    const u64 oldest = ring.latest - ring.count + 1;
    u64 target = ref_sequence + 1;
    if (step < 0) {
        if (ref_sequence == 0) return -1;
        if (ref_sequence < oldest) return -2;
        target = ref_sequence;
        // Step past ref_sequence when the reference already carries this kind exclusively,
        // or when this kind changed at ref_order.
        if (same_kind_only || ring.entries[target % history_capacity].order == ref_order) --target;
    }
    if (target == 0 || target > ring.latest) return -1;
    if (target < oldest) return -2;
    out = ring.entries[target % history_capacity];
    return 0;
}

// Nearest entry in `step` direction that changed any wanted kind; ordering across kinds uses the internal order.
int find_history(const Reading& ref, int step, std::uint32_t mask, HistoryEntry& out)
{
    AcquireSRWLockShared(&state_lock);
    const std::uint32_t ref_mask = history_mask(ref.kind);
    int result = -1;
    bool have = false;
    for (int i = 0; i < 2; ++i) {
        const std::uint32_t bit = i == 0 ? history_pad : history_keys;
        if (!(mask & bit)) continue;
        const KindRing& ring = i == 0 ? pad_ring : key_ring;
        const bool same_kind_only = (mask == bit) && (ref_mask == bit);
        HistoryEntry candidate{};
        const int found = find_in_ring(ring, i == 0 ? ref.pad_sequence : ref.key_sequence, ref.order, step,
                                       same_kind_only, candidate);
        if (found == -2) {
            result = -2;
            have = false;
            break;
        }
        if (found == 0 && (!have || (step > 0 ? candidate.order < out.order : candidate.order > out.order))) {
            out = candidate;
            have = true;
            result = 0;
        }
    }
    ReleaseSRWLockShared(&state_lock);
    return result;
}

Reading* make_reading(const HistoryEntry& entry, std::uint32_t kind, Object* device)
{
    auto* reading = new Reading{};
    reading->vtable = reading_table.slots;
    reading->refs = 1;
    reading->kind = resolve_reading_kind(kind, device);
    reading->state = entry.pad;
    reading->key_count = entry.key_count;
    if (entry.key_count > 0) std::memcpy(reading->keys, entry.keys, entry.key_count * sizeof(KeyState));
    reading->pad_sequence = entry.pad_sequence;
    reading->key_sequence = entry.key_sequence;
    reading->timestamp = entry.timestamp;
    reading->order = entry.order;
    return reading;
}

HRESULT validate_kind_and_device(std::uint32_t kind, Object* device)
{
    if (!wants_any(kind)) return reading_not_found;
    if (device) {
        if (device != &the_device && device != &the_keyboard_device) return device_not_found;
        if (device == &the_device && !wants_gamepad(kind)) return reading_not_found;
        if (device == &the_keyboard_device && !wants_keyboard(kind)) return reading_not_found;
    }
    return S_OK;
}

// ---- IGameInput ---------------------------------------------------------------------------
Table<32> input_table;
Object the_input{input_table.slots, 1};

u64 __stdcall input_timestamp(Object*) { return now_microseconds(); }
HRESULT __stdcall input_current_reading(Object*, std::uint32_t kind, Object* device, Reading** out)
{
    static LONG calls;
    note("GameInput::GetCurrentReading", calls, kind, reinterpret_cast<u64>(device));
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    *out = nullptr;
    const HRESULT check_hr = validate_kind_and_device(kind, device);
    if (FAILED(check_hr)) return check_hr;
    *out = make_reading(capture_current(), kind, device);
    return S_OK;
}
HRESULT __stdcall input_next_reading(Object*, Reading* reference, std::uint32_t kind, Object* device, Reading** out)
{
    static LONG calls;
    note("GameInput::GetNextReading", calls, reinterpret_cast<u64>(reference), kind);
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    *out = nullptr;
    if (!reference) return E_INVALIDARG;
    const HRESULT check_hr = validate_kind_and_device(kind, device);
    if (FAILED(check_hr)) return check_hr;
    capture_current();
    HistoryEntry entry{};
    const int found = find_history(*reference, 1, history_mask(resolve_reading_kind(kind, device)), entry);
    if (found == -2) return reading_too_old;
    if (found != 0) return reading_not_found;
    *out = make_reading(entry, kind, device);
    return S_OK;
}
HRESULT __stdcall input_previous_reading(Object*, Reading* reference, std::uint32_t kind, Object* device, Reading** out)
{
    static LONG calls;
    note("GameInput::GetPreviousReading", calls, reinterpret_cast<u64>(reference), kind);
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    *out = nullptr;
    if (!reference) return E_INVALIDARG;
    const HRESULT check_hr = validate_kind_and_device(kind, device);
    if (FAILED(check_hr)) return check_hr;
    HistoryEntry entry{};
    const int found = find_history(*reference, -1, history_mask(resolve_reading_kind(kind, device)), entry);
    if (found == -2) return reading_too_old;
    if (found != 0) return reading_not_found;
    *out = make_reading(entry, kind, device);
    return S_OK;
}

// Microsoft's v0 GetTemporalReading was never implemented and was removed in v1.
HRESULT __stdcall input_temporal_reading(Object*, u64 timestamp, Object* device, Reading** out)
{
    static LONG calls;
    note("GameInput::GetTemporalReading", calls, timestamp, reinterpret_cast<u64>(device));
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    *out = nullptr;
    return E_NOTIMPL;
}
using DeviceCallback = void(__stdcall*)(u64 token, void* context, Object* device, u64 timestamp, std::uint32_t current, std::uint32_t previous);
HRESULT __stdcall input_register_device_callback(Object*, Object* device, std::uint32_t kind, std::uint32_t status_filter, std::uint32_t enumeration,
                                                 void* context, DeviceCallback callback, u64* token)
{
    static LONG calls, tokens;
    note("GameInput::RegisterDeviceCallback", calls, kind, status_filter, enumeration);
    if (token && !x2w_local::writable(token, sizeof(*token))) return E_POINTER;
    const u64 value = static_cast<u64>(InterlockedIncrement(&tokens));
    if (token) *token = value;
    // Any enumeration request reports matching virtual devices at once (connected).
    if (callback && enumeration != 0) {
        const u64 ts = now_microseconds();
        if (wants_gamepad(kind) && (!device || device == &the_device))
            callback(value, context, &the_device, ts, status_connected, 0);
        if (wants_keyboard(kind) && (!device || device == &the_keyboard_device))
            callback(value, context, &the_keyboard_device, ts, status_connected, 0);
    }
    return S_OK;
}
HRESULT __stdcall input_register_callback(Object*, u64 a, u64 b, u64 c, u64 d, u64 e, u64* token)
{
    static LONG calls, tokens;
    note("GameInput::Register*Callback(other)", calls, a, b, c);
    (void)d;
    (void)e;
    if (token && !x2w_local::writable(token, sizeof(*token))) return E_POINTER;
    if (token) *token = 0x1000 + static_cast<u64>(InterlockedIncrement(&tokens));
    return S_OK;
}
// RegisterGuideButtonCallback / RegisterKeyboardLayoutCallback(device, context, callback, token).
HRESULT __stdcall input_register_short_callback(Object*, Object*, void*, void*, u64* token)
{
    static LONG calls, tokens;
    note("GameInput::Register{GuideButton,KeyboardLayout}Callback", calls);
    if (token && !x2w_local::writable(token, sizeof(*token))) return E_POINTER;
    if (token) *token = 0x2000 + static_cast<u64>(InterlockedIncrement(&tokens));
    return S_OK;
}
void __stdcall input_stop_callback(Object*, u64) {}
bool __stdcall input_unregister_callback(Object*, u64, u64) { return true; }
HRESULT __stdcall input_find_device_by_id(Object*, const void* id, Object** device)
{
    static LONG calls;
    note("GameInput::FindDeviceFromId", calls);
    if (!device || !x2w_local::writable(device, sizeof(*device))) return E_POINTER;
    if (id && x2w_local::readable(id, 32) &&
        (std::memcmp(id, keyboard_device_info + 0x20, 32) == 0 ||
         std::memcmp(id, keyboard_device_info + 0x40, 32) == 0)) {
        InterlockedIncrement(&the_keyboard_device.refs);
        *device = &the_keyboard_device;
        return S_OK;
    }
    InterlockedIncrement(&the_device.refs);
    *device = &the_device;
    return S_OK;
}
// FindDeviceFromPlatformHandle: the argument is an opaque OS handle, never a pointer; no device is bound to one.
HRESULT __stdcall input_find_device_by_handle(Object*, void*, Object** device)
{
    static LONG calls;
    note("GameInput::FindDeviceFromPlatformHandle", calls);
    if (!device || !x2w_local::writable(device, sizeof(*device))) return E_POINTER;
    *device = nullptr;
    return device_not_found;
}
// FindDeviceFromObject / FindDeviceFromPlatformString: the argument is not read; the virtual pad answers.
HRESULT __stdcall input_find_device_other(Object*, const void*, Object** device)
{
    static LONG calls;
    note("GameInput::FindDeviceFrom{Object,PlatformString}", calls);
    if (!device || !x2w_local::writable(device, sizeof(*device))) return E_POINTER;
    InterlockedIncrement(&the_device.refs);
    *device = &the_device;
    return S_OK;
}
void __stdcall input_set_focus_policy(Object*, std::uint32_t policy)
{
    static LONG calls;
    note("GameInput::SetFocusPolicy", calls, policy);
}

void build()
{
    static const bool built = [] {
        fill_unknown<0>(input_table.slots, std::make_index_sequence<32>{});
    fill_unknown<1>(device_table.slots, std::make_index_sequence<32>{});
    fill_unknown<2>(reading_table.slots, std::make_index_sequence<32>{});
    for (void** slots : {input_table.slots, device_table.slots, reading_table.slots}) {
        slots[0] = reinterpret_cast<void*>(&query_interface);
        slots[1] = reinterpret_cast<void*>(&add_ref);
        slots[2] = reinterpret_cast<void*>(&release_static);
    }
    // IGameInput (GameInput.h, version 0), after IUnknown.
    input_table.slots[3] = reinterpret_cast<void*>(&input_timestamp);
    input_table.slots[4] = reinterpret_cast<void*>(&input_current_reading);
    input_table.slots[5] = reinterpret_cast<void*>(&input_next_reading);
    input_table.slots[6] = reinterpret_cast<void*>(&input_previous_reading);
    input_table.slots[7] = reinterpret_cast<void*>(&input_temporal_reading);
    input_table.slots[8] = reinterpret_cast<void*>(&input_register_callback);          // RegisterReadingCallback
    input_table.slots[9] = reinterpret_cast<void*>(&input_register_device_callback);
    input_table.slots[10] = reinterpret_cast<void*>(&input_register_short_callback);   // RegisterGuideButtonCallback
    input_table.slots[11] = reinterpret_cast<void*>(&input_register_short_callback);   // RegisterKeyboardLayoutCallback
    input_table.slots[12] = reinterpret_cast<void*>(&input_stop_callback);
    input_table.slots[13] = reinterpret_cast<void*>(&input_unregister_callback);
    input_table.slots[16] = reinterpret_cast<void*>(&input_find_device_by_id);
    input_table.slots[17] = reinterpret_cast<void*>(&input_find_device_other);
    input_table.slots[18] = reinterpret_cast<void*>(&input_find_device_by_handle);
    input_table.slots[19] = reinterpret_cast<void*>(&input_find_device_other);
    input_table.slots[21] = reinterpret_cast<void*>(&input_set_focus_policy);
    // IGameInputDevice.
    device_table.slots[3] = reinterpret_cast<void*>(&device_get_info);
    device_table.slots[4] = reinterpret_cast<void*>(&device_get_status);
    device_table.slots[10] = reinterpret_cast<void*>(&device_set_rumble);
    // IGameInputReading.
    reading_table.slots[2] = reinterpret_cast<void*>(&reading_release);
    reading_table.slots[3] = reinterpret_cast<void*>(&reading_kind);
    reading_table.slots[4] = reinterpret_cast<void*>(&reading_sequence);
    reading_table.slots[5] = reinterpret_cast<void*>(&reading_timestamp);
    reading_table.slots[6] = reinterpret_cast<void*>(&reading_device);
    reading_table.slots[14] = reinterpret_cast<void*>(&reading_key_count);
    reading_table.slots[15] = reinterpret_cast<void*>(&reading_key_state);
    reading_table.slots[22] = reinterpret_cast<void*>(&reading_gamepad);
    // GameInputDeviceInfo: size, vendor/product of a standard pad, a fixed device id, family, capabilities, supported input.
    std::uint32_t size = sizeof(device_info);
    std::memcpy(device_info, &size, 4);
    const std::uint16_t ids[2] = {0x045E, 0x0B12};
    std::memcpy(device_info + 4, ids, 4);
    for (int i = 0; i < 32; ++i) device_info[0x20 + i] = static_cast<unsigned char>(0xA0 + i);  // deviceId
    for (int i = 0; i < 32; ++i) device_info[0x40 + i] = static_cast<unsigned char>(0xA0 + i);  // deviceRootId
    // GameInputDeviceFamily: Virtual -1, XboxOne 1 (XboxGIP / XInputHID), Xbox360 2 (XUSB22), Hid 3, I8042 4. The
    // vendor/product above name an Xbox Series pad, a GIP device, so the family is XboxOne.
    const std::uint32_t tail[4] = {family_xbox_one, 0, kind_gamepad | kind_controller_bits, 0xF /* rumble motors */};
    std::memcpy(device_info + 0x60, tail, sizeof(tail));
    // Virtual keyboard device info.
    std::memcpy(keyboard_device_info, &size, 4);
    const std::uint16_t kbd_ids[2] = {0x045E, 0x07A5};
    std::memcpy(keyboard_device_info + 4, kbd_ids, 4);
    for (int i = 0; i < 32; ++i) keyboard_device_info[0x20 + i] = static_cast<unsigned char>(0xB0 + i);
    for (int i = 0; i < 32; ++i) keyboard_device_info[0x40 + i] = static_cast<unsigned char>(0xB0 + i);
        const std::uint32_t kbd_tail[4] = {0 /* family: unknown/PC */, 0, kind_keyboard, 0};
        std::memcpy(keyboard_device_info + 0x60, kbd_tail, sizeof(kbd_tail));
        return true;
    }();
    (void)built;
}
}  // namespace

// Ordinal 1: HRESULT GameInputCreate(IGameInput** gameInput).
extern "C" __declspec(noinline) HRESULT GameInputCreate(void** out)
{
    build();
    x2w_local::logf("x2w-input GameInputCreate(%p): virtual gamepad stand-in (keyboard + XInput pad 0)", static_cast<void*>(out));
    input_script();
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    InterlockedIncrement(&the_input.refs);
    *out = &the_input;
    return S_OK;
}
