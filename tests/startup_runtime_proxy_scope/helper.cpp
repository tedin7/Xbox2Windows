// Package helper: imports GetProcAddress by name so the runtime proxy can rewrite its IAT after load.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

extern "C" __declspec(dllexport) FARPROC WINAPI resolve_query(HMODULE module)
{
    return GetProcAddress(module, "QueryApiImpl");
}
