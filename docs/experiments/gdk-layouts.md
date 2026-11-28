# GDK layouts and ABI probes

tools/gdk_layouts.py parses the output of MSVC /d1reportAllClassLayout and keeps
only class names, sizes, member names, offsets and flags. tools/gdk_abi_probe.py
turns that JSON into a C++ probe: static_asserts for a compile check, or a
program that prints sizeof and offsetof values for a GRDK/GXDK comparison.

The logs are produced outside the repository. No GDK header, LIB or DLL enters
the tree. The parser is tested on a synthetic log with a base class, vftable,
bitfield, array and alignment member, so the Python check runs on Linux.

    python -B tests/gdk_layouts_check.py
    python -B tools/gdk_layouts.py parse D:\gdk-oracle\fixtures\layout-xg-one.log --json out\xg.json --module xg_xs_gen8 --platform Gaming.Xbox.XboxOne.x64 --msvc 19.44.35207
    python -B tools/gdk_abi_probe.py generate --layouts out\xg.json --out out\xg-probe.cpp --include windows.h --include xg.h --prefix XG_ --print

Measured on 28 November 2026 with MSVC 19.44.35207 and the 261000 headers:

| Header | Platform | Classes in the log | Probe |
|---|---|---:|---|
| XboxOne/xg.h | Gaming.Xbox.XboxOne.x64 | 2,098 | 16 XG_ types; sizeof probe compiled |
| XboxOne/d3d12_x.h | Gaming.Xbox.XboxOne.x64 | 3,029 | 74 D3D12XBOX_ types; 25 anonymous skipped; compiled |
| GameInput.h | Gaming.Desktop.x64 | 3,344 | 41 GameInput types; 1 anonymous skipped; compiled |
| XGameRuntime.h | Gaming.Desktop.x64 | 3,363 | 30 selected public types; 3 anonymous skipped; compiled with windows.h first |

The Xbox One headers use these macros: _GAMING_XBOX, _GAMING_XBOX_XBOXONE and
WINAPI_FAMILY=WINAPI_FAMILY_GAMES. The GRDK headers use _GAMING_DESKTOP and
WINAPI_FAMILY=WINAPI_FAMILY_DESKTOP_APP. XGameRuntime.h needs windows.h before
it, because XTaskQueue.h uses STDAPI.

The parser treats a single-token member line, such as Width, as an unknown type
with a known name. MSVC prints that form for some typedefs. Anonymous nested
classes appear as NAME::<unnamed-tag>; the probe generator skips them because
C++ cannot name them.

Limits: a layout is a compile-time fact for one compiler and one macro set. It
does not prove runtime behavior, and a mismatch between a header layout and a
live library still needs an ABI probe or a runtime oracle. Private members and
non-standard-layout classes need review before offsetof is enabled; the generator
emits offsetof only for classes without bases or a vftable and for named,
non-bitfield fields.
