// Input for the startup debugger's call-hook check (tests/startup_debug_check.py). The check copies
// this DLL next to a copy of tests/startup_debug_probe.cpp under the movie-player module name that
// tools/startup/debug_launch.py hooks with --trace-calls 1, so the export below is one of its call
// hooks. The export only counts its calls; no other input is involved.
#include <windows.h>

namespace {
volatile LONG calls;
}  // namespace

extern "C" __declspec(dllexport) LONG BinkNextFrame(void*)
{
    return InterlockedIncrement(&calls);
}
