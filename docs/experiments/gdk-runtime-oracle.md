# GDK runtime oracle: GRDK and shim

tools/gdk_runtime_oracle.py compiles the same original fixture twice: once
against the GRDK import library and once against the Xbox2Windows shim. It runs
both executables, masks handles, console ids and per-user temp paths, and writes
a normalized diff and a JSON report.

    python tools/gdk_runtime_oracle.py run --gdk-root "D:\gdk-ref\261000\grdk\Microsoft GDK\261000\GRDK" --shim-dir "out\build\dev\fixture-shims" --fixture "tests\gdk\runtime_families_probe.cpp" --out "out\gdk-runtime" --vcvars "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

On a host without cl.exe the command returns 77. The normalization and diff
subcommands need no SDK:

    python tools/gdk_runtime_oracle.py normalize out\gdk-runtime\candidate.txt
    python tools/gdk_runtime_oracle.py diff out\gdk-runtime\reference.txt out\gdk-runtime\candidate.txt

Measured on 28 November 2026, Windows 11 build 26200, the inbox
xgameruntime.dll 10.0.26100.9656 and the local shim:

| Observation | GRDK reference | Xbox2Windows shim |
|---|---|---|
| XGameRuntimeInitialize | S_OK | not exported |
| XSystemGetDeviceType | 1 (Pc) | 1 (Pc) |
| XSystemGetConsoleId | S_OK, 39 bytes, all-zero id | S_OK, 39 bytes, 58325731...01 |
| XGameGetXboxTitleId | 0x80070490 (ERROR_NOT_FOUND) | S_OK, 0x58325731 |
| XPersistentLocalStorageGetPathSize | 0x80004005 | S_OK, 78 |
| XPersistentLocalStorageGetSpaceInfo | 0x80004005 | S_OK, 64 MiB total, 16 MiB used |
| XPackageMount of a missing identifier | 0x80004001 (E_NOTIMPL) | S_OK, synthetic handle and temp path |
| XTaskQueueCreate(Manual, Manual) | S_OK | S_OK |
| XTaskQueueDispatch(Completion, 0) | false | false |
| XSystemGetXboxLiveSandboxId | S_OK, RETAIL | not exported |

The desktop runtime has no package identity on this host, so the title id and
persistent-local-storage calls fail. The shim synthesizes those values. The
oracle makes the difference explicit; it does not claim the shim matches the
console. Console-only APIs have static oracles only (headers, LIB exports and
vtables), and a successful mount of a missing identifier is an error-path
observation, not a package-format result.
