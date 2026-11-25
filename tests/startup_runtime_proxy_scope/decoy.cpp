// Unrelated DLL that also exports QueryApiImpl; must not become the cached gaming-runtime provider.
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
    return 0xA11C0001;  // distinct from the fake xgameruntime path
}
