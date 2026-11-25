// Source-built EXE control for package preflight. It imports one documented but
// unimplemented XGameRuntime function. The test package never contains the
// fixture-only xgameruntime.dll, so strict planning must refuse this import
// before any image is mapped or executed.
#include <windows.h>

extern "C" __declspec(dllimport) HRESULT WINAPI XPackageMountWithUiAsync(const char* package_identifier, void* async);

// The Debug preset enables /RTC1. This CRT-free fixture still references the
// two RTC bookkeeping symbols, so provide inert definitions instead of linking
// the runtime whose presence would add unrelated imports to the control.
extern "C" void _RTC_InitBase() {}
extern "C" void _RTC_Shutdown() {}

extern "C" HRESULT WINAPI fixture_entry()
{
    return XPackageMountWithUiAsync(nullptr, nullptr);
}
