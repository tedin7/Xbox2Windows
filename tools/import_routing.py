"""Shared candidate module routes for Xbox2Windows PE import reports.

The policy classifies import module names only. It does not resolve exports,
prove a provider, or load the image.
"""

from collections.abc import MutableMapping

POLICY_VERSION = 2
ROUTE_KIND = "candidate"
RESOLUTION = "not_checked"
ROUTES = ("hle", "windows", "stub", "unsupported")

# Each entry is data, not a guess. HLE modules are built by this tree. Stub
# entries are public GDK modules without a shim here. Windows entries are
# explicit system DLLs plus the documented API-set prefixes.
POLICY = {
    "hle": {
        "xmem.dll": "Xbox2Windows builds src/xmem_shim.cpp as xmem.dll",
        "d3d12_xs.dll": "Xbox2Windows builds src/d3d12_xs_shim.cpp as d3d12_xs.dll",
    },
    "stub": {
        "xgameruntime.dll": "GDK runtime module; no implemented Xbox2Windows provider in this tree",
        "xg_xs.dll": "GDK console GPU module; no Xbox2Windows shim in this tree",
    },
    "windows": frozenset({
        "advapi32.dll",
        "bcrypt.dll",
        "cfgmgr32.dll",
        "combase.dll",
        "crypt32.dll",
        "d3d12.dll",
        "d3dcompiler_47.dll",
        "dbghelp.dll",
        "dxgi.dll",
        "gameinput.dll",
        "gdi32.dll",
        "kernel32.dll",
        "kernelbase.dll",
        "mfplat.dll",
        "mfreadwrite.dll",
        "msvcrt.dll",
        "ntdll.dll",
        "ole32.dll",
        "oleaut32.dll",
        "powrprof.dll",
        "psapi.dll",
        "rpcrt4.dll",
        "setupapi.dll",
        "shell32.dll",
        "shlwapi.dll",
        "ucrtbase.dll",
        "user32.dll",
        "userenv.dll",
        "version.dll",
        "winhttp.dll",
        "winmm.dll",
        "ws2_32.dll",
        "xaudio2_8.dll",
        "xaudio2_9.dll",
    }),
    "windows_prefixes": ("api-ms-win-", "ext-ms-"),
}


def _validate_module(module):
    """Reject non-str, empty, whitespace, or control-character module names."""
    if not isinstance(module, str):
        raise ValueError(f"module name must be a str, got {type(module).__name__}")
    if not module or any(ch.isspace() or ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in module):
        raise ValueError(f"invalid module name: {module!r}")


def classify(module):
    """Return (route, reason) for a module name. No route is inferred."""
    _validate_module(module)
    name = module.lower()
    if name in POLICY["hle"]:
        return "hle", POLICY["hle"][name]
    if name in POLICY["stub"]:
        return "stub", POLICY["stub"][name]
    if name in POLICY["windows"]:
        return "windows", "Windows system DLL"
    for prefix in POLICY["windows_prefixes"]:
        if name.startswith(prefix):
            return "windows", "Windows API-set contract"
    return "unsupported", "no matching policy entry"


def annotate(row, module_key="module"):
    """Add the candidate route fields to an import row and return it.

    The caller keeps its own caller and symbol fields. No provider or export
    resolution is recorded.
    """
    if not isinstance(row, MutableMapping):
        raise TypeError(f"row must be a MutableMapping, got {type(row).__name__}")
    if module_key not in row:
        raise KeyError(module_key)
    module = row[module_key]
    route, reason = classify(module)
    row.update({
        "route": route,
        "reason": reason,
        "route_kind": ROUTE_KIND,
        "resolution": RESOLUTION,
        "provider": None,
        "stub": route == "stub",
        "candidate_provider": module if route == "hle" else None,
    })
    return row
