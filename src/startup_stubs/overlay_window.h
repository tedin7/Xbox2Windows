#pragma once

#include <windows.h>

#include <algorithm>
#include <vector>

namespace x2w_startup {
inline constexpr wchar_t presenter_class[] = L"X2WOutputWindow";
inline constexpr wchar_t dialog_class[] = L"#32770";  // message boxes and assert dialogs

// A top-level window the title made: visible, unowned, in this process, not the presenter.
inline bool is_title_window(HWND candidate, wchar_t (&name)[256])
{
    DWORD process = 0;
    GetWindowThreadProcessId(candidate, &process);
    if (process != GetCurrentProcessId() || !IsWindowVisible(candidate) || GetWindow(candidate, GW_OWNER))
        return false;
    return GetClassNameW(candidate, name, 256) && lstrcmpW(name, presenter_class) != 0;
}

inline HWND overlay_window()
{
    HWND window = nullptr;
    EnumWindows([](HWND candidate, LPARAM out) -> BOOL {
        wchar_t name[256]{};
        if (!is_title_window(candidate, name))
            return TRUE;
        *reinterpret_cast<HWND*>(out) = candidate;
        return FALSE;
    }, reinterpret_cast<LPARAM>(&window));
    return window;
}

// X2W_HIDE_TITLE_WINDOW=1 (docs/startup-runtime.md#switches), exactly "1". Read on each call.
inline bool hide_title_window_requested()
{
    char value[8]{};
    return GetEnvironmentVariableA("X2W_HIDE_TITLE_WINDOW", value, sizeof(value)) == 1 && value[0] == '1';
}

inline constexpr UINT_PTR hide_title_timer = 0x5832;  // timer id on the output window

// Hides the title's own top-level windows while the output window shows the picture. It runs on
// the output window's thread: start() arms a 500 ms timer when the switch is on, and every
// WM_TIMER calls step(). Dialogs stay visible so that they can be seen and answered.
struct TitleWindowHider {
    std::vector<HWND> hidden;  // windows this hider hid, shown again when the picture stops

    static bool start(HWND output)
    {
        return hide_title_window_requested() && SetTimer(output, hide_title_timer, 500, nullptr) != 0;
    }

    // `presents`: frames shown in the output window so far; `last_shown`: whether the latest frame
    // was. Returns how many windows were hidden, or minus how many were shown again because the
    // latest frame was not shown (the GDI fallback draws its status into a visible title window).
    // *foreground_refused is set when a hidden window was in front and the output window could not
    // take the foreground; the foreground goes to the output window before anything is hidden.
    int step(HWND output, LONG presents, LONG last_shown, bool* foreground_refused = nullptr)
    {
        if (foreground_refused) *foreground_refused = false;
        if (!(presents > 0 && last_shown != 0)) {
            int shown = 0;
            for (HWND window : hidden) {
                if (!IsWindow(window)) continue;
                ShowWindowAsync(window, SW_SHOWNOACTIVATE);  // its result is the previous visibility
                ++shown;
            }
            hidden.clear();
            return -shown;
        }
        std::vector<HWND> found;
        EnumWindows([](HWND candidate, LPARAM out) -> BOOL {
            wchar_t name[256]{};
            if (is_title_window(candidate, name) && lstrcmpW(name, dialog_class) != 0)
                reinterpret_cast<std::vector<HWND>*>(out)->push_back(candidate);
            return TRUE;
        }, reinterpret_cast<LPARAM>(&found));
        const HWND foreground = GetForegroundWindow();
        for (HWND window : found) {
            if (window != foreground) continue;
            const bool taken = SetForegroundWindow(output) && GetForegroundWindow() == output;
            if (foreground_refused) *foreground_refused = !taken;
        }
        // Asynchronous: the title's window thread may be busy, and this thread must keep pumping.
        for (HWND window : found) {
            ShowWindowAsync(window, SW_HIDE);
            if (std::find(hidden.begin(), hidden.end(), window) == hidden.end()) hidden.push_back(window);
        }
        return static_cast<int>(found.size());
    }
};
}  // namespace x2w_startup
