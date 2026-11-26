// Input for tests/startup_debug_window_check.py: one visible window whose picture changes at a known time.
//
// A 160x120 popup window at the top left corner of the screen, painted red; after STARTUP_WINDOW_PROBE_FLIP_MS
// milliseconds (default 3500) it is painted blue. The probe keeps running until the debugger ends the run
// (or for one minute), so the debugger's end-of-run window.png shows the last picture.
// STARTUP_WINDOW_PROBE_HANG_MS=<ms>: after that long the window thread stops pumping messages (a title whose window thread
// is stuck): PrintWindow and every other call that needs the window's thread then waits until the process is killed.
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
COLORREF g_color = RGB(200, 30, 30);

LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
    if (message == WM_PAINT) {
        PAINTSTRUCT paint{};
        HDC dc = BeginPaint(hwnd, &paint);
        RECT area{};
        GetClientRect(hwnd, &area);
        HBRUSH brush = CreateSolidBrush(g_color);
        FillRect(dc, &area, brush);
        DeleteObject(brush);
        EndPaint(hwnd, &paint);
        return 0;
    }
    if (message == WM_ERASEBKGND) return 1;
    return DefWindowProcW(hwnd, message, wparam, lparam);
}
}  // namespace

DWORD WINAPI short_thread(void*) { return 0; }

int main()
{
    // STARTUP_WINDOW_PROBE_MESSAGE=1: one 400+ character Unicode debug message with CRLF inside and a unique last word.
    // STARTUP_WINDOW_PROBE_BREAK=1: after one second a __debugbreak() in the main image (a title soft assert).
    char mode[8];
    if (GetEnvironmentVariableA("STARTUP_WINDOW_PROBE_MESSAGE", mode, sizeof(mode)) && mode[0] == '1') {
        std::wstring message = L"probe-msg-start \u00e8\u4e2d\r\nsecond line\r\n";
        while (message.size() < 400) message += L"0123456789 \u00fc ";
        message += L"\r\nprobe-msg-suffix-7f3a";
        OutputDebugStringW(message.c_str());
        // the same from a thread of its own: the debugger must name that thread (a "thread N created" line), not the main one
        if (HANDLE worker = CreateThread(nullptr, 0, [](void*) -> DWORD { OutputDebugStringW(L"probe-worker-message"); return 0; },
                                         nullptr, 0, nullptr)) {
            WaitForSingleObject(worker, 5000);
            CloseHandle(worker);
        }
    }
    // STARTUP_WINDOW_PROBE_CHURN=1: load and free a DLL, start a thread that ends at once (the debugger logs the unload, and
    // the thread's creation and exit with the live thread count).
    char churn[8];
    if (GetEnvironmentVariableA("STARTUP_WINDOW_PROBE_CHURN", churn, sizeof(churn)) && churn[0] == '1') {
        if (HMODULE module = LoadLibraryW(L"msimg32.dll")) FreeLibrary(module);
        if (HANDLE thread = CreateThread(nullptr, 0, short_thread, nullptr, 0, nullptr)) {
            WaitForSingleObject(thread, 5000);
            CloseHandle(thread);
        }
    }
    WNDCLASSW wc{};
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpfnWndProc = window_proc;
    wc.lpszClassName = L"startup_window_probe";
    if (!RegisterClassW(&wc)) return 1;
    HWND hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_NOACTIVATE, wc.lpszClassName, L"window probe", WS_POPUP | WS_VISIBLE,
                                0, 0, 160, 120, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) return 2;
    ULONGLONG flip_ms = 3500;
    char text[32];
    if (GetEnvironmentVariableA("STARTUP_WINDOW_PROBE_FLIP_MS", text, sizeof(text)) && text[0]) flip_ms = strtoull(text, nullptr, 10);
    ULONGLONG hang_ms = 0;
    if (GetEnvironmentVariableA("STARTUP_WINDOW_PROBE_HANG_MS", text, sizeof(text)) && text[0]) hang_ms = strtoull(text, nullptr, 10);
    ULONGLONG debug_ms = 0;
    if (GetEnvironmentVariableA("STARTUP_WINDOW_PROBE_DEBUG_MS", text, sizeof(text)) && text[0]) debug_ms = strtoull(text, nullptr, 10);
    const ULONGLONG begin = GetTickCount64();
    bool flipped = false;
    std::printf("window-probe shown\n");
    std::fflush(stdout);
    while (GetTickCount64() - begin < 60000) {
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&message);
        if (!flipped && GetTickCount64() - begin >= flip_ms) {
            g_color = RGB(30, 30, 200);
            InvalidateRect(hwnd, nullptr, TRUE);
            UpdateWindow(hwnd);
            flipped = true;
            std::printf("window-probe flipped\n");
            std::fflush(stdout);
        }
        // STARTUP_WINDOW_PROBE_DEBUG_MS=<ms>: from then on the window thread raises a debug event (OutputDebugString) every pass;
        // it is frozen until the debugger continues it, which the debugger must keep doing while it ends the run.
        if (debug_ms && GetTickCount64() - begin >= debug_ms) OutputDebugStringA("window-probe debug event\n");
        static bool broke = false;
        if (!broke && GetTickCount64() - begin >= 1000 && GetEnvironmentVariableA("STARTUP_WINDOW_PROBE_BREAK", mode, sizeof(mode)) && mode[0] == '1') {
            broke = true;
            __debugbreak();
            std::printf("window-probe continued after the breakpoint\n");
            std::fflush(stdout);
        }
        if (hang_ms && GetTickCount64() - begin >= hang_ms) {
            std::printf("window-probe hung\n");
            std::fflush(stdout);
            Sleep(120000);  // no more messages: the window thread is stuck
        }
        Sleep(10);
    }
    return 0;
}
