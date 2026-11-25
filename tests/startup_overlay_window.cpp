#include "startup_stubs/overlay_window.h"

#include <cstdio>

int main()
{
    WNDCLASSW wc{};
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpfnWndProc = DefWindowProcW;
    wc.lpszClassName = x2w_startup::presenter_class;
    if (!RegisterClassW(&wc)) return 1;
    // Offscreen, unactivated windows exercise the real z-order enumerator.
    const auto make = [&](const wchar_t* name, HWND owner, bool visible) {
        HWND hwnd = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, name, L"", WS_POPUP,
                                    -32000, -32000, 8, 8, owner, nullptr, wc.hInstance, nullptr);
        if (hwnd && visible) ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        return hwnd;
    };
    HWND output = make(wc.lpszClassName, nullptr, true);
    HWND hidden = make(L"STATIC", nullptr, false);
    HWND owned = make(L"STATIC", output, true);
    if (!output || !hidden || !owned || x2w_startup::overlay_window()) return 2;
    HWND guest = make(L"STATIC", nullptr, true);
    if (!guest || x2w_startup::overlay_window() != guest) return 3;
    if (!SetWindowPos(output, HWND_TOP, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE) ||
        x2w_startup::overlay_window() != guest) return 4;
    // With two guests the enumeration stops at the first match, so the upper one is selected
    // in either order. Continuing to the end would select the lower one.
    HWND second = make(L"STATIC", nullptr, true);
    const auto raise = [](HWND hwnd) {
        return SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);
    };
    if (!second || !raise(guest) || x2w_startup::overlay_window() != guest) return 6;
    if (!raise(second) || x2w_startup::overlay_window() != second) return 7;
    DestroyWindow(second);
    DestroyWindow(guest);
    if (x2w_startup::overlay_window()) return 5;
    // X2W_HIDE_TITLE_WINDOW: the switch is on only for exactly "1" (negative controls first).
    using Hider = x2w_startup::TitleWindowHider;
    SetEnvironmentVariableA("X2W_HIDE_TITLE_WINDOW", nullptr);
    if (x2w_startup::hide_title_window_requested()) return 8;
    if (Hider::start(output)) return 20;  // no timer without the switch
    SetEnvironmentVariableA("X2W_HIDE_TITLE_WINDOW", "0");
    if (x2w_startup::hide_title_window_requested()) return 9;
    SetEnvironmentVariableA("X2W_HIDE_TITLE_WINDOW", "10");
    if (x2w_startup::hide_title_window_requested()) return 16;
    SetEnvironmentVariableA("X2W_HIDE_TITLE_WINDOW", "1");
    if (!x2w_startup::hide_title_window_requested()) return 10;
    // With the switch, start() arms a timer whose WM_TIMER reaches the output window.
    if (!Hider::start(output)) return 19;
    MSG tick{};
    bool ticked = false;
    for (ULONGLONG until = GetTickCount64() + 3000; !ticked && GetTickCount64() < until; Sleep(10))
        ticked = PeekMessageW(&tick, output, WM_TIMER, WM_TIMER, PM_REMOVE) && tick.wParam == x2w_startup::hide_title_timer;
    KillTimer(output, x2w_startup::hide_title_timer);
    if (!ticked) return 19;
    // Only the guest is hidden: the presenter, a window it owns and a dialog stay visible.
    HWND title = make(L"STATIC", nullptr, true);
    HWND dialog = make(x2w_startup::dialog_class, nullptr, true);
    if (!title || !dialog || !IsWindowVisible(title) || !IsWindowVisible(dialog)) return 11;
    Hider hider;
    // ShowWindowAsync posts work to the window's owning thread. Drain that
    // thread's queue before checking whether the posted request took effect.
    const auto wait_visibility = [](HWND window, bool visible) {
        const ULONGLONG deadline = GetTickCount64() + 3000;
        MSG message{};
        do {
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
                DispatchMessageW(&message);
            if ((IsWindowVisible(window) != FALSE) == visible) return true;
            Sleep(10);
        } while (GetTickCount64() < deadline);
        return false;
    };
    // Before the first frame reaches the output window nothing is hidden.
    if (hider.step(output, 0, 1) != 0 || !IsWindowVisible(title)) return 17;
    if (hider.step(output, 1, 1) != 1 || !wait_visibility(title, false)) return 12;
    if (!IsWindowVisible(output) || !IsWindowVisible(owned) || !IsWindowVisible(dialog)) return 13;
    // Shown again by the title, hidden again by the next step; nothing left to hide after that.
    ShowWindow(title, SW_SHOWNOACTIVATE);
    if (hider.step(output, 2, 1) != 1 || !wait_visibility(title, false)) return 14;
    if (hider.step(output, 3, 1) != 0) return 15;
    // A frame that is not shown brings the title window back for the GDI fallback.
    if (hider.step(output, 4, 0) != -1 || !wait_visibility(title, true)) return 18;
    // Focus: when the hidden window is in front, the output window takes the foreground first.
    // A host that does not let this process set the foreground cannot show this; it is reported.
    HWND front_output = CreateWindowExW(0, x2w_startup::presenter_class, L"", WS_POPUP, 0, 0, 8, 8,
                                        nullptr, nullptr, wc.hInstance, nullptr);
    HWND front = CreateWindowExW(0, L"STATIC", L"", WS_POPUP, 0, 0, 8, 8, nullptr, nullptr, wc.hInstance, nullptr);
    if (!front_output || !front) return 21;
    ShowWindow(front_output, SW_SHOW);
    ShowWindow(front, SW_SHOW);
    const bool focus_checked = SetForegroundWindow(front) && GetForegroundWindow() == front;
    if (focus_checked) {
        bool refused = true;
        if (hider.step(front_output, 5, 1, &refused) < 1 || refused || GetForegroundWindow() != front_output ||
            !wait_visibility(front, false)) return 21;
    }
    DestroyWindow(front);
    DestroyWindow(front_output);
    DestroyWindow(dialog);
    DestroyWindow(title);
    DestroyWindow(owned);
    DestroyWindow(hidden);
    DestroyWindow(output);
    std::puts("GDI overlay selected the guest below or above the presenter and the upper of two guests; "
              "no target after guest destruction; X2W_HIDE_TITLE_WINDOW exactly 1: timer armed and ticked, "
              "guest hidden only after a shown frame (twice), shown again after an unshown frame, "
              "presenter, its owned window and a dialog kept visible");
    std::puts(focus_checked ? "foreground handed to the output window before the hide"
                            : "foreground handoff not checked: this host refused SetForegroundWindow to the test");
    return 0;
}
