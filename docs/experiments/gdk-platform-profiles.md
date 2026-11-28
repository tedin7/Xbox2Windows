# GDK MSBuild platform profiles

tools/gdk_platform_profiles.py reads the GDK platform files under
VS2022/flatDeployment/MSBuild/Microsoft/VC/v170 and writes a JSON profile per
platform. It keeps the MSBuild conditions and resolves the simple variables
defined in the same files. Variables that refer to the installed edition
(DurangoXdkInstallPath, DurangoGRDKInstallPath, XdkEdition) stay visible instead
of being guessed.

The GDK installation and the JSON profiles stay outside the repository.

    python -B tests/gdk_platform_profiles_check.py
    python -B tools/gdk_platform_profiles.py list --root D:\gdk-ref\261000\gxdk\Microsoft GDK\261000\GXDK
    python -B tools/gdk_platform_profiles.py scan --root D:\gdk-ref\261000\gxdk\Microsoft GDK\261000\GXDK --platform Gaming.Xbox.Scarlett.x64 --json out\scarlett.json

The --sdk option selects the versioned subdirectory (261000, 999999, ...). A
platform directory without that subdirectory is rejected before any output is
written, so the reported SDK, the source paths and the extracted values always
refer to the same edition.

Measured on 28 November 2026 from the 261000 payload:

| Platform | Sources | Defines | Includes | Libs | Dependencies |
|---|---:|---:|---:|---:|---:|
| Gaming.Desktop.x64 (GRDK) | 10 | 2 | 1 | 1 | 6 |
| Gaming.Xbox.Scarlett.x64 | 10 | 7 | 3 | 3 | 1 |
| Gaming.Xbox.XboxOne.x64 | 10 | 7 | 3 | 3 | 1 |

The console profiles define _GAMING_XBOX, _GAMING_XBOX_SCARLETT or
_GAMING_XBOX_XBOXONE, WIN32_LEAN_AND_MEAN and
WINAPI_FAMILY=WINAPI_FAMILY_GAMES, plus the ATL/WRL strict-COM macros. The
desktop profile defines _GAMING_DESKTOP and
WINAPI_FAMILY=WINAPI_FAMILY_DESKTOP_APP. Gaming.Desktop.arm64 is not present in
this payload.

The include and library roots are Console_EndpointIncludeRoot and
Console_EndpointLibRoot. The console dependency list in Console_Libs is
pixevt.lib, d3d12_xs.lib, gameinput.lib, xgameplatform.lib, xgameruntime.lib,
xmem.lib and xg_xs.lib. The Scarlett profile also sets
EnableEnhancedInstructionSet=AdvancedVectorExtensions, FloatingPointModel=Fast
and /favor:AMD64 for v142/v143.

The profile is a build recipe, not a runtime fact. A fixture generated from it
still needs the matching compiler, Windows SDK and GDK edition; the unresolved
installation variables are kept exactly because their values differ per host.
