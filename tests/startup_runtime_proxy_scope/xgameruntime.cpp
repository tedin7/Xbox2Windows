// Source-built stand-in for xgameruntime.dll used only by startup_runtime_proxy_scope.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
LONG calls;
}

extern "C" __declspec(dllexport) LONG WINAPI query_calls() { return InterlockedCompareExchange(&calls, 0, 0); }

extern "C" __declspec(dllexport) HRESULT WINAPI QueryApiImpl(const GUID*, const GUID*, void** out)
{
    InterlockedIncrement(&calls);
    if (out) *out = nullptr;
    return E_FAIL;
}
