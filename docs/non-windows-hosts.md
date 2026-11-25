# Contributing from macOS and Linux

Xbox2Windows targets x64 Windows for its C++, HLSL, MASM, D3D12, WARP, and GPU test suite. Contributors on macOS and Linux can still edit documentation, run the platform-independent Python checks, and run the LLVM 15 `gfx1013` shader translation checks locally before handing off Windows verification.

## Platform split

| Work or check | macOS / Linux | x64 Windows | Notes |
|---|---|---|---|
| Documentation edits | Yes | Yes | Markdown, diff hygiene, and link checks (`tests/docs_links_check.py`). |
| Platform-independent Python checks | Yes | Yes | The quick starts below run a subset of checks. See the `python-linux` job in the [Linux CI workflow](../.github/workflows/windows.yml) for the full CI list. |
| LLVM 15 `gfx1013` translation checks | Yes | No (via CI / separate LLVM) | `tests/pixel_aot_check.py binary`, `tests/ffx_dispatch_aot_check.py`, and `tests/rdna2_tools_check.py` (with `LLVM_LIB`, `LLVM_MC`, `LLVM_OBJCOPY`) verify assembling, disassembling, and lowering only. |
| `cmake --preset dev` build (MSVC `cl` + Ninja) | No | Yes | `CMakePresets.json` selects `cl`. |
| MASM fixtures (`tests/cpu/*.asm`) and `cpu_*` tests | No | Yes | Assembled and built by the Windows toolchain. |
| DXC compilation and rendering (`pixel_aot_warp`, `pixel_aot_hardware`) | No | Yes | Requires Windows SDK / DXC (`dxc.exe`, `dxil.dll`) and D3D12. |
| D3D12, WARP, and DXR tests (`*_warp`, `*_hardware`) | No | Yes | Covers software WARP and physical GPU checks. |
| System32 PE export inventory (`tools/pe_inventory.py`) | No | Yes | Uses `ctypes.windll` on Windows. |
| Standalone GPU check packaging (`tools/package-gpu-check.ps1`) | No | Yes | Builds and smoke-tests the Windows x64 ZIP package. |

`tests/pixel_aot_check.py binary`, `tests/ffx_dispatch_aot_check.py`, and `tests/rdna2_tools_check.py` do not invoke DXC or render pixels; desktop HLSL compilation and readback (`pixel_aot_warp`, `pixel_aot_hardware`) remain Windows-only.

## Quick start on macOS

Install Python 3.12 and LLVM 15 with Homebrew (both are needed as explicit paths because `llvm@15` is keg-only):

```bash
brew install python@3.12 llvm@15
```

From the repository root, create a virtual environment under `out/pe-env`, install `pefile` at the version pinned in `requirements-dev.txt`, and run the platform-independent checks below.
The block runs in a subshell and returns the first failure:

```bash
(
python_prefix="$(brew --prefix python@3.12)" || exit $?
"$python_prefix/bin/python3.12" -m venv out/pe-env || exit $?
./out/pe-env/bin/python -m pip install -c requirements-dev.txt pefile || exit $?
for t in \
  tests/script_state_check.py \
  tests/resource_root_check.py \
  tests/bcpack_check.py \
  tests/bcpack_symbols_check.py \
  tests/bc7_blocks_check.py \
  tests/bc7_tiles_check.py \
  tests/resource_ranges_check.py \
  tests/ttpf_batch_export_check.py \
  tests/awc_metadata_check.py \
  tests/sgd5_effects_check.py \
  tests/dxcontainer_part_offsets_check.py \
  tests/layout_chunks_check.py \
  tests/dependency_paths_check.py \
  tests/intake/paths_check.py \
  tests/shader_probe_check.py \
  tests/shader_census_check.py \
  tests/dxil_retarget_check.py \
  tests/rdna2_tools_check.py \
  tests/pso_readiness_check.py \
  tests/docs_links_check.py \
  tests/bidi_controls_check.py \
  tests/hardware_reports_check.py \
  tests/gpu_report_format_check.py; do
  ./out/pe-env/bin/python -B "$t" || exit $?
done
)
```

`tests/form_coverage_check.py` and `tests/xmem_attribute_scan_check.py` import `iced-x86`, which pip builds from source on Apple Silicon, so this list leaves them out. Run them on Linux or Windows.

Then run the three LLVM 15 `gfx1013` checks:

```bash
LLVM_PREFIX="$(brew --prefix llvm@15)"
./out/pe-env/bin/python -B tests/pixel_aot_check.py binary \
  "$LLVM_PREFIX/bin/llvm-mc" "$LLVM_PREFIX/bin/llvm-objcopy" "$LLVM_PREFIX/lib/libLLVM-15.dylib"
./out/pe-env/bin/python -B tests/ffx_dispatch_aot_check.py \
  "$LLVM_PREFIX/bin/llvm-mc" "$LLVM_PREFIX/bin/llvm-objcopy" "$LLVM_PREFIX/lib/libLLVM-15.dylib"
LLVM_LIB="$LLVM_PREFIX/lib/libLLVM-15.dylib" \
  LLVM_MC="$LLVM_PREFIX/bin/llvm-mc" \
  LLVM_OBJCOPY="$LLVM_PREFIX/bin/llvm-objcopy" \
  ./out/pe-env/bin/python -B tests/rdna2_tools_check.py
```

## Quick start on Linux

The [CI workflow](../.github/workflows/windows.yml) runs the platform-independent Python suite on Ubuntu 24.04 with Python 3.12 (`python-linux`) and the `gfx1013` machine-code checks on Ubuntu 22.04 with LLVM 15 (`pixel-binary`).
Both jobs use `tools/run_checks_parallel.py`; the machine-code matrix supplies per-check LLVM arguments and environment overrides.
The runner lists checks that exit 77 under `SKIPPED:` and other nonzero exits under `FAILED:`. Either summary makes the runner exit 1 so CI catches required checks that did not run. Open a skipped check's log group to see its output.

On Debian/Ubuntu, install `python3-venv` first. Without it, `python3 -m venv` fails with `ensurepip is not available`:

```bash
sudo apt-get update -q
sudo apt-get install -y python3-venv
```

From the repository root, create `out/pe-env` (Python 3.11 or newer; CI uses 3.12), install dependencies, and run the platform-independent checks below.
The block runs in a subshell and returns the first failure:

```bash
(
python3 -m venv out/pe-env || exit $?
./out/pe-env/bin/python -m pip install -r requirements-dev.txt || exit $?
for t in \
  tests/script_state_check.py \
  tests/resource_root_check.py \
  tests/bcpack_check.py \
  tests/bcpack_symbols_check.py \
  tests/bc7_blocks_check.py \
  tests/bc7_tiles_check.py \
  tests/form_coverage_check.py \
  tests/resource_ranges_check.py \
  tests/ttpf_batch_export_check.py \
  tests/awc_metadata_check.py \
  tests/sgd5_effects_check.py \
  tests/dxcontainer_part_offsets_check.py \
  tests/layout_chunks_check.py \
  tests/dependency_paths_check.py \
  tests/intake/paths_check.py \
  tests/shader_probe_check.py \
  tests/shader_census_check.py \
  tests/dxil_retarget_check.py \
  tests/rdna2_tools_check.py \
  tests/pso_readiness_check.py \
  tests/xmem_attribute_scan_check.py \
  tests/docs_links_check.py \
  tests/bidi_controls_check.py \
  tests/hardware_reports_check.py \
  tests/gpu_report_format_check.py; do
  ./out/pe-env/bin/python -B "$t" || exit $?
done
)
```

On Debian/Ubuntu, install LLVM 15 and run the three `gfx1013` checks:

```bash
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends llvm-15
./out/pe-env/bin/python -B tests/pixel_aot_check.py binary \
  llvm-mc-15 llvm-objcopy-15 libLLVM-15.so.1
./out/pe-env/bin/python -B tests/ffx_dispatch_aot_check.py \
  llvm-mc-15 llvm-objcopy-15 libLLVM-15.so.1
LLVM_LIB=libLLVM-15.so.1 LLVM_MC=llvm-mc-15 LLVM_OBJCOPY=llvm-objcopy-15 \
  ./out/pe-env/bin/python -B tests/rdna2_tools_check.py
```

On Fedora, the system `python3` creates the virtual environment without an extra package. Install the `llvm15` compatibility package. The package also installs versioned tool names such as `llvm-mc-15` and `llvm-objcopy-15`. The commands below use the package's full paths on x86-64 to select LLVM 15 explicitly:

```bash
sudo dnf install -y llvm15
LLVM15=/usr/lib64/llvm15
./out/pe-env/bin/python -B tests/pixel_aot_check.py binary \
  "$LLVM15/bin/llvm-mc" "$LLVM15/bin/llvm-objcopy" "$LLVM15/lib/libLLVM-15.so"
./out/pe-env/bin/python -B tests/ffx_dispatch_aot_check.py \
  "$LLVM15/bin/llvm-mc" "$LLVM15/bin/llvm-objcopy" "$LLVM15/lib/libLLVM-15.so"
LLVM_LIB="$LLVM15/lib/libLLVM-15.so" \
  LLVM_MC="$LLVM15/bin/llvm-mc" \
  LLVM_OBJCOPY="$LLVM15/bin/llvm-objcopy" \
  ./out/pe-env/bin/python -B tests/rdna2_tools_check.py
```

These commands were checked on Fedora 45 with `llvm15` 15.0.7 and Python 3.15.2. See the [Fedora 45 package file list](https://packages.fedoraproject.org/pkgs/llvm15/llvm15/fedora-45.html) for the versioned tools and their paths. Fedora's default `llvm` package is a newer major version (23 on Fedora 45), which the fixtures do not target.

## Optional Vulkan arithmetic check

With the Linux build of DXC, a Vulkan device, and [VkRunner](https://gitlab.freedesktop.org/mesa/vkrunner), run the generated FP32 exponent helpers against a Python reference.

```bash
python3 -B tests/pixel_fp32_vulkan_check.py /path/to/dxc /path/to/vkrunner
```

The probe checks finite output bits, signed zero, infinities, and NaN class through a compute buffer. A second dispatch uses an intentionally incorrect underflow rounding rule and must fail the same comparison. Repeat `--device <index>` to test specific one-based VkRunner device indices. The fixture exercises the desktop helpers through SPIR-V. Pixel shader lowering and D3D12 readback have their own tests.

## Optional Vulkan subnormal check

With the Linux build of DXC, `spirv-val` and `spirv-dis` from SPIRV-Tools, a C compiler, the Vulkan loader headers and a Vulkan 1.2 device, check whether the driver keeps binary32 subnormals in a source-built pixel shader. [`tests/vk_pixel_draw.c`](../tests/vk_pixel_draw.c) is a small Vulkan host that draws a full-screen triangle with a compiled vertex and pixel shader, 256 bytes of constants and an RGBA32F or RGBA32UI target, and writes the raw pixels to a file. The check builds it when you pass the `.c` file.

```bash
python3 -B tests/pixel_denorm_vulkan_check.py /path/to/dxc tests/vk_pixel_draw.c --device Intel
```

`--device` is a substring of the Vulkan device name (repeat it for several adapters), and `VK_DRIVER_FILES` selects a driver such as lavapipe. To build the host once and reuse it, run `cc -std=c11 -O1 -o out/vk/vk_pixel_draw tests/vk_pixel_draw.c -lvulkan` and pass `out/vk/vk_pixel_draw` instead of the `.c` file.

A `ps_6_2` pixel shader runs seven operations whose inputs or results are binary32 subnormals and writes each result word to its own RGBA32UI pixel. DXC compiles it with `-spirv -fspv-target-env=vulkan1.2 -fvk-b-shift 0 0 -fvk-t-shift 16 0 -fvk-s-shift 32 0 -fvk-u-shift 48 0`. The words read back must equal IEEE binary32 arithmetic computed in Python under `-denorm preserve`. The same shader compiled with `-denorm ftz` is the negative control and must differ, otherwise the check fails because it cannot tell preserve from flush. The log lists the device, driver, DXC and SPIRV-Tools versions, the SPIR-V execution mode, every word read back for both modes and the cases where the control differs.

A device that reports `shaderDenormPreserveFloat32` as false is a capability skip (exit code 77). A module that `spirv-val` rejects or that the driver refuses to build is a validation refusal (exit code 78). Neither counts as a mismatch, and a mismatch exits with 1. A refusal that follows a `preserve` readback that differs from IEEE (for example, `-denorm ftz` refused) is still a mismatch and exits with 1, because 78 means a refusal with no measured mismatch. Run without arguments, `python3 -B tests/pixel_denorm_vulkan_check.py` needs no device, DXC or SPIR-V tools: it checks the classification of the two readbacks (preserve mismatch with an FTZ refusal, an FTZ control that matches preserve, and the nominal cases), and the Linux Python checks job runs it.

## Optional `draw-core` under Vulkan

The `draw-core` mode of `tests/pixel_aot_check.py` draws the core translated pixel shaders and compares every channel bit for bit, with the negative controls of `check_render`. On Windows it uses D3D12. `draw-core-vulkan` runs the same shaders and controls on a Vulkan 1.2 device. It needs the same tools as the subnormal check above (DXC, `spirv-val`, a C compiler and the Vulkan loader headers).

```bash
python3 -B tests/pixel_aot_check.py draw-core-vulkan /path/to/dxc Intel
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  python3 -B tests/pixel_aot_check.py draw-core-vulkan /path/to/dxc llvmpipe --allow-denorm-mismatch
```

The second argument is a substring of the Vulkan device name. The mode builds `tests/vk_pixel_draw.c` under `out/`, compiles each shader to SPIR-V with the DXC options of the subnormal check plus `-fvk-invert-y` for the vertex shader, validates it with `spirv-val` and draws it through [`tests/vk_offscreen.py`](../tests/vk_offscreen.py), a stand-in for the `offscreen_check` operations `--check-pixel`, `--check-pixel-case`, `--check-pixel-batch` and `--check-pixel-quad-lod`. It keeps their command lines, exit codes, `pixel_mismatch x= y=` words and `case=N status=` lines. Cases with title-memory trailers are refused, and there is no debug layer (the log says `debug_layer=unavailable`). Other `offscreen_check` modes are not reproduced.

The shaders are compiled with `-denorm preserve`. A device without `shaderDenormPreserveFloat32` therefore exits 77, a capability skip. `--allow-denorm-mismatch` draws anyway. Such a run is a diagnostic: a pass shows that the device produced the expected bits for these shaders, not that it preserves subnormals, and a failure may come from the missing guarantee instead of from the shader. It is not a conformance result. `X2W_VK_ALLOW_NO_DENORM`, `X2W_VK_HOST`, `X2W_VK_VS`, `X2W_VK_DEVICE` and `X2W_VK_WORK` are the variables the mode sets for `vk_offscreen.py`; you do not need to set them. Run without arguments, `python3 -B tests/vk_offscreen.py` needs no device or DXC: it checks the case parser, the comparison, the batch status lines and the refusals against synthetic readbacks, and the Linux Python checks job runs it.

## Expected CMake failure and no-stub policy

`CMakePresets.json` sets `CMAKE_CXX_COMPILER` to `cl` for the `dev` preset. Running `cmake --preset dev` on macOS or Linux is expected to fail:

```text
-- The CXX compiler identification is unknown
CMake Error at CMakeLists.txt:2 (project):
  The CMAKE_CXX_COMPILER:

    cl

  is not a full path and was not found in the PATH.
```

Do not work around this by stubbing `cl`, MASM, D3D12, or WARP to make CMake configure succeed on non-Windows hosts. Full compilation and runtime tests require x64 Windows.

## WSL and Linux translation layers

- **WSL:** You can run the Linux Python and LLVM 15 checks inside WSL, while the CMake, MSVC, DXC, and D3D12 build still runs from Windows PowerShell as documented in [First build](../README.md#first-build).
- **Linux GPU Check via Proton and Wine:** Contributors have verified the standalone Windows GPU Check under Linux translation layers (Wine or Proton with `vkd3d-proton`). Mesa RADV reported native 64-lane waves and passed forced wave64 on an [AMD Ryzen 5 integrated GPU](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7067289597) and an [RX 9070 XT reported by @barbiscool](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7069870226). On the RX 9070 XT, RADV defaults to 32-lane waves (unlike Windows 64-lane defaults) and passes forced 64. NVIDIA runs on an [RTX 4070](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064884397), [RTX 5060 Ti](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7067186626) and [RTX 3060](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064204388) reported 32-lane maximums and completed all 408 BVH comparisons. Translation-layer caveats: `vkd3d-proton` is required (Wine's built-in D3D12 fails `bvh_waves` and skips differential checks), Wine DXGI reports placeholder driver versions (e.g. `35.0.65535.65535`), and `bvh_root_roundtrip_hardware` fails on `vkd3d-proton` due to its acceptance of undefined root flags (`0x03000000`) that Microsoft D3D12 rejects with `E_INVALIDARG`. If you report a translation-layer run of the standalone GPU Check in `#1`, include the commit or release tag, GPU model, Mesa or NVIDIA driver version, Proton/Wine and `vkd3d-proton` versions, and each subtest's output (including skips).
- A native Linux runtime is outside the scope of this guide and remains a separate request from the x64 Windows target.

### Compile check with msvc-wine

[msvc-wine](https://github.com/mstorsjo/msvc-wine) runs the real MSVC toolchain and Windows SDK under Wine. This is not a stub, so you can use it to confirm that a C++ or HLSL change compiles with `cl` and DXC before Windows CI runs. Review the Visual Studio Build Tools license before using the unattended acceptance option below.

```bash
python3 vsdownload.py --accept-license --dest ~/.local/msvc
./install.sh ~/.local/msvc
```

The SDK's `dxc.exe` is not wrapped. Save this script as `~/.local/msvc/bin/x64/dxc` and make it executable. It maps absolute Linux paths to Wine's `Z:` drive:

```bash
#!/bin/bash
DXC="$HOME/.local/msvc/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"
args=()
for a in "$@"; do
  if [[ -e $a && $a == /* || $a == /*/* && -d ${a%/*} ]]; then args+=("z:${a//\//\\}"); else args+=("$a"); fi
done
WINEDEBUG=-all exec wine "$DXC" "${args[@]}"
```

Then configure with `CMAKE_SYSTEM_NAME=Windows`. Without it, CMake's compiler check fails to link with `LNK1181`.

```bash
export PATH="$HOME/.local/msvc/bin/x64:$PATH"
cmake --preset release -DCMAKE_SYSTEM_NAME=Windows -DDXC_EXECUTABLE="$HOME/.local/msvc/bin/x64/dxc"
cmake --build --preset release
```

The resulting test binaries run under Wine and `vkd3d`, not Windows D3D12 or WARP. Expect `desktop_memory_query`, the `d3d12_xs_shim_*` export and root checks, and `ffx_provider_loop_*` and `bvh_root_roundtrip_*` to fail there, and the DXR and BVH tests to skip. Treat a Wine run as a compile check only. Do not report Wine test results as Windows verification or hardware coverage.

### Cross-compile with clang-cl and an xwin SDK

clang-cl with lld-link and a Windows SDK laid out by [xwin](https://github.com/Jake-Shadle/xwin)
is a second way to build Windows binaries on Linux without Wine in the build. Like msvc-wine it
is a real compiler and the real SDK headers and import libraries, not a stub, and like msvc-wine
it is a compile check: MSVC and Windows CI remain the reference. Review the SDK license before
using xwin's acceptance option.

Measured on one host (Debian 13, clang-cl 19.1.7, `xwin splat` of SDK 10.0.26100): the eight
startup stand-ins (`xmem_shim`, `d3d12_xs_shim` and the six `startup_stub_*` targets) and the
device-free test executables build from a clean checkout. The whole tree and the MASM fixtures
were not attempted.

Save this as a toolchain file:

```cmake
# clang-cl, lld-link and an xwin SDK splat as a Windows x64 toolchain on Linux.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_CXX_COMPILER clang-cl-19)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_LINKER lld-link-19)
set(CMAKE_AR llvm-lib-19)
set(CMAKE_RC_COMPILER llvm-rc-19)
set(CMAKE_MT llvm-mt-19)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES XWIN LIBCASE)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# The paths are quoted inside the flag strings so that a directory with spaces stays one argument.
set(inc "/imsvc \"${XWIN}/crt/include\" /imsvc \"${XWIN}/sdk/include/ucrt\" /imsvc \"${XWIN}/sdk/include/um\" /imsvc \"${XWIN}/sdk/include/shared\" /imsvc \"${XWIN}/sdk/include/winrt\"")
# /EHa last: CMake's own default would add /EHsc, and the stand-ins probe memory inside __try.
set(CMAKE_CXX_FLAGS "${inc} /DWIN32 /D_WINDOWS /clang:-fbracket-depth=2048 /EHa" CACHE STRING "" FORCE)
set(lib "\"/libpath:${XWIN}/crt/lib/x86_64\" \"/libpath:${XWIN}/sdk/lib/um/x86_64\" \"/libpath:${XWIN}/sdk/lib/ucrt/x86_64\" \"/libpath:${LIBCASE}\"")
set(CMAKE_EXE_LINKER_FLAGS "${lib}" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${lib}" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "${lib}" CACHE STRING "" FORCE)
```

```bash
mkdir -p out/libcase
ln -s "$XWIN/sdk/lib/um/x86_64/onecore.lib" out/libcase/OneCore.lib
cmake -S . -B out/build/clang -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/clang-cl-xwin.cmake -DXWIN="$XWIN" -DLIBCASE="$PWD/out/libcase" \
  -DDXC_EXECUTABLE=/path/to/linux/dxc
cmake --build out/build/clang --target xmem_shim d3d12_xs_shim startup_stub_acphal startup_stub_chad \
  startup_stub_xapu startup_stub_xcrdapi startup_stub_xg_xs startup_stub_gameinput
```

The toolchain file quotes the SDK and `LIBCASE` paths inside its flag strings, so those two and
the build directory may contain spaces (checked with all three in directories whose names have a
space; a source directory with spaces was not tried).

Four things differ from an MSVC build:

- **`/EHa` is required, and it has to come after CMake's default `/EHsc`.** The stand-ins probe
  title memory inside `__try` (`src/startup_stubs/local_log.h`). MSVC catches the access violation
  of a load written directly in the `__try` body under `/EHsc`; clang-cl only does so with
  asynchronous exception handling. Built with `/EHsc`, the first probe of an unreadable address
  ends the process with an unhandled access violation inside `d3d12_xs.dll`. The toolchain file
  above sets the flags itself for that reason.
- **`-fbracket-depth`.** `d3d12_xs_device.cpp` instantiates a 512-argument fold expression; clang
  stops at 256 nested expressions by default.
- **`OneCore.lib`.** `CMakeLists.txt` links `OneCore`; the splat has `onecore.lib`, and lld-link on
  a case-sensitive file system does not find it. The symlink above supplies the spelling.
- **DXC.** Configure needs a `dxc`. The Linux build from the
  [DirectXShaderCompiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases)
  satisfies `find_program`; the HLSL fixtures were not built with it here.

### Startup debugger under Wine

The [startup stand-ins and debugger](startup-runtime.md) target Windows. The following was
needed to run `tools/startup/debug_launch.py` on the stand-ins built as above, under Wine with
vkd3d-proton (measured: Proton Experimental's Wine 11.0 and vkd3d-proton 2576c0b4, NVIDIA driver
550.163.01, one console executable, offline). It is a translation-layer run for finding problems,
not Windows verification; report it as such.

- **Python.** The debugger uses the Win32 debug API through `ctypes`, so it runs under the
  Windows build of CPython inside Wine, not the host's Python. The embeddable package works once
  its `python312._pth` file is removed (with it, the script's own directory is not on the import
  path). Unpack the `win_amd64` wheel of `iced-x86` next to `python.exe` for the checks that
  import it.
- **D3D12.** Wine's built-in D3D12 is not enough. Copy vkd3d-proton's `d3d12.dll` and
  `d3d12core.dll`, DXVK's `dxgi.dll`, and the `libvkd3d*.dll` files Wine's `d3dcompiler_47` imports
  into the prefix's `system32`, and set `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n"`.
- **Package drive.** An executable that opens its package by drive letter needs that letter in the
  prefix: remove the auto-assigned device link of that letter in `dosdevices/` and add a symlink
  to the package folder, for example `ln -s /path/to/package "$WINEPREFIX/dosdevices/g:"`.
- **FXC.** `tools/startup/gfx2hlsl.py --pending` compiles with `fxc.exe`. The x64 `fxc.exe` and
  its `d3dcompiler_47.dll` from the Windows SDK run under Wine with
  `WINEDLLOVERRIDES="d3dcompiler_47=n"`, so that FXC uses the SDK's compiler and not Wine's. Set
  `X2W_FXC` to its Windows path and run the translator with the Windows Python.
- **Gaming runtime.** The runtime proxy wraps the system `xgameruntime.dll`, which Wine does not
  have. The x64 `xgameruntime.dll` of Microsoft's `Microsoft.GDK.Windows` NuGet package, placed
  next to the executable, loads under Wine and answers `QueryApiImpl`.
- **Section backing.** Wine backs anonymous sections with files in its server directory. When
  that directory is on a small tmpfs (`/run/user/<uid>` is often a tenth of RAM), unset
  `XDG_RUNTIME_DIR` for the run so that Wine uses `/tmp`.
- **Descriptor handles.** vkd3d-proton keeps tag bits in the low bits of its CPU descriptor
  handles, where native D3D12 returns aligned addresses. An executable that does arithmetic on the
  handles of a sampler or render-target heap can then pass a corrupted handle back (seen once,
  on a sampler heap). This is an open difference; the stand-ins hand out the desktop handle for
  those heap types.
- **Two top-level windows.** The executable's own window can sit above the stand-in's output
  window under an X11 window manager. With `X2W_HIDE_TITLE_WINDOW=1` (`run.sh --hide-title-window`),
  the stand-in hides it while frames are shown in the output window
  ([Switches](startup-runtime.md#switches)). Keyboard input to the title after the hide is not
  measured. Otherwise, minimise it to see the output.

Known cost under Wine: one section per 64 KiB physical page
(`src/xmem_mapped_pages.cpp`) is slow there. A request for about 100,000 pages did not finish
within 30 s in the measured run; with a local change that maps the same pages as a few large
views, the run was past that request within 5 s of starting.

## Troubleshooting

- **Homebrew `llvm@15` is keg-only:** Homebrew does not symlink `llvm@15` binaries into `PATH` by default. Pass `"$(brew --prefix llvm@15)"` explicitly instead of calling bare `llvm-mc`.
- **Shared library filename (`libLLVM-15.dylib` vs `libLLVM-15.so.1`):** Pass `libLLVM-15.dylib` from `"$(brew --prefix llvm@15)/lib"` on macOS, `libLLVM-15.so.1` on Ubuntu/Debian, and `/usr/lib64/llvm15/lib/libLLVM-15.so` on Fedora.
- **LLVM versions newer than 15:** CI and the test fixtures target LLVM 15. Newer LLVM versions may print a different `exp` instruction spelling during AMDGPU disassembly.
- **`tests/shader_probe_check.py` and `tests/dxil_retarget_check.py` with vs without arguments:** Running either script with no arguments runs its platform-independent controls (`shader_metadata_controls` and `dxil_retarget_controls`). Passing compiled `.cso` shaders, `root_probe`, and `dxil.dll` (`shader_metadata_probe`, `dxil_retarget_probe`) requires Windows and DXC.
- **`tests/rdna2_tools_check.py` skips one test without `LLVM_LIB`:** Without `LLVM_LIB`, the script runs its pure-Python disassembler, extractor, and scanner checks and skips `test_llvm_comparison`. Setting `LLVM_LIB`, `LLVM_MC`, and `LLVM_OBJCOPY` runs the LLVM 15 comparison check as well.
- **`tools/pe_inventory.py` fails to import `ctypes.windll`:** `pe_inventory.py` inspects System32 exports on x64 Windows and is not supported on macOS or Linux.
- **Keep virtual environments out of Git:** Create the Python environment at `out/pe-env`, which is already ignored via `/out/` in `.gitignore`.

## Evidence and Windows verification handoff

Follow [Share a reproducible finding](../CONTRIBUTING.md#share-a-reproducible-finding) and [AGENTS.md](../AGENTS.md) when preparing a pull request or issue from a non-Windows host:

- Record the exact commit SHA under test, host OS and architecture, Python and LLVM versions, exact commands run, and observed results.
- State clearly that Windows C++, HLSL, WARP, and hardware checks were not run locally when contributing from macOS or Linux, and rely on the `build-and-test` CI job or a Windows contributor to verify Windows behavior.
- Never include game files, dumps, `.exe`/`.dll` binaries, torrents, distribution links, personal paths, or credentials in issues, pull requests, or Git.
