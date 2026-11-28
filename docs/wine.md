# Startup experiments under Wine and Proton on Linux

This page covers what changes when the [startup stand-ins and debugger](startup-runtime.md) run
through Proton's Wine on a Linux desktop. The one-time setup is in
[Startup debugger under Wine](non-windows-hosts.md#startup-debugger-under-wine): the stand-ins
built with clang-cl and an xwin SDK, the Windows Python, the D3D12 DLLs and the package drive.
This page adds the launch environment, audio, the behaviour seen only under Wine, and how to report a run.

## What a Wine run shows

A Wine run is a translation-layer run of research tooling. It can find problems in the stand-ins
and give counts to compare. It is not Windows verification, it says nothing about hardware coverage,
and it does not establish that an executable starts, renders or plays. Some checks fail or skip under
Wine for reasons that have nothing to do with the change under test. Report those separately from
passes and failures, as [AGENTS.md](../AGENTS.md#evidence) asks for capability skips.

## Environment

| Part | Measured with | Note |
|---|---|---|
| Wine | Proton Experimental from Steam (experimental-11.0, Wine 11.0) | Run `files/bin/wine` directly with its own `WINEPREFIX` and `WINEFSYNC=1`; no Steam runtime |
| D3D12 | vkd3d-proton and DXVK's `dxgi.dll` copied into the prefix | `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11=n"` |
| Stand-ins | clang-cl 19 and an xwin SDK splat, Release | See [Cross-compile with clang-cl](non-windows-hosts.md#cross-compile-with-clang-cl-and-an-xwin-sdk) |
| Host | Debian 13, X11 desktop, NVIDIA 550.163.01, one GPU | Other drivers and Wayland sessions were not tried |

- **Display.** The stand-ins open an output window, so the run needs the desktop's X display. A shell
  without a desktop session (ssh, a service, a scheduler) has neither `DISPLAY` nor `XAUTHORITY`.
  Set both, for example `DISPLAY=:0` and the session's authority file.
- **One run at a time.** All processes in a prefix share one `wineserver`. A second run in the same
  prefix shares it, and heavy load from other processes changes timing. The mapping burst described
  below was sensitive to both.
- **Shader cache.** Set `VKD3D_SHADER_CACHE_PATH` to a directory under `out/`. Without it,
  vkd3d-proton writes its pipeline cache to the current directory, which can be the working tree.
- **Paths.** The stand-ins and the debugger take Windows paths. A host path is `Z:` followed by the
  path with backslashes.

## Launch environment

[`run.sh`](../tools/startup/run.sh) is the Linux counterpart of [`run.ps1`](../tools/startup/run.ps1):
the same switches as GNU long options (`--exe`, `--data`, `--log`, `--render`, `--translate`, ...). It starts
`tools/startup/debug_launch.py` with a Windows Python inside Wine and exits with the debugger's exit code.
Set `WINEPREFIX` to a prepared prefix ([Startup debugger under Wine](non-windows-hosts.md#startup-debugger-under-wine))
and `X2W_WIN_PYTHON` to the host path of a Windows `python.exe`; `WINE` (or `PROTON`) picks the Wine
binary. `--data`, `--log`, `--storage` and `--shader-cache` take a host path or a drive path such as `G:\`.
A drive path is passed on as it is; run.sh creates the log and shader-cache folders through the prefix's drive links. A
title that opens its package by drive letter needs that letter. The title id and storage size come from the package's
`MicrosoftGame.config`, the first tag of each in any case, as `run.ps1` reads them. Unlike `run.ps1`, a value must start
on its tag's line, and a UTF-16 config is not read. With `--render --translate`, the shader watcher runs in
its own process group with `d3dcompiler_47=n` added to its `WINEDLLOVERRIDES`, and the whole group is
stopped when the run ends. `--hide-title-window` sets `X2W_HIDE_TITLE_WINDOW=1`
([Switches](startup-runtime.md#switches)).

These variables matter on Linux. `run.sh` sets them, except where the table says otherwise; a value you
set yourself stays, except `XDG_RUNTIME_DIR`, which it always unsets:

| Variable | Why |
|---|---|
| The `run.ps1` set: `X2W_AUDIO_PERMISSIVE`, `X2W_D3D12_DEVICE_NOOP`, `X2W_D3D12_LIST_NOOP`, `X2W_D3D12_PERMISSIVE`, `X2W_XMEM_PERMISSIVE`, `X2W_RUNTIME_PROXY` (all `=1`) | The same stand-in behaviour as a Windows run. `run.sh` sets the variables behind `--render`, `--fake-user`, `--gpu-mirror` and `--host-cpu` as [Switches](startup-runtime.md#switches) lists them |
| `X2W_XMEM_SHARED_VIEWS=1` | With one section and view per page, a start-up batch of about 100,000 pages took 42.9 s under Wine. With shared views it took 82 ms ([Physical pages in shared sections](startup-runtime.md#physical-pages-in-shared-sections)). Without it, the measured runs did not get past start-up |
| `X2W_XMEM_JOIN_CALLS=1` | Start-up maps nearly every page with a call of its own. With it, such a call joins the shared view that ends at its address instead of adding a view, and under Wine each map costs more the more views the process has ([Behaviour seen only under Wine](#behaviour-seen-only-under-wine)). It needs `X2W_XMEM_SHARED_VIEWS=1` |
| `XDG_RUNTIME_DIR` unset | Wine then backs sections in `/tmp`, not on a small tmpfs ([Startup debugger under Wine](non-windows-hosts.md#startup-debugger-under-wine)). This also breaks audio; see the next section |
| `PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native` | Restores audio while `XDG_RUNTIME_DIR` stays unset. `run.sh` sets it only when that socket exists |
| `WINEDLLOVERRIDES=d3d12,d3d12core,dxgi,d3d11=n`, `WINEFSYNC=1` | Wine's built-in D3D12 is not enough; the prefix holds vkd3d-proton and DXVK |
| `WINEDEBUG` | `-all` keeps the logs small. For audio, add `err+xaudio2,warn+xaudio2,err+mmdevapi,err+pulse,err+alsa` |
| `_NO_DEBUG_HEAP` | The debugger, not `run.sh`, sets it to `1` unless it is already set. It has no effect under Wine (below) |

## Audio

The console audio stand-ins produce no output ([Audio output](startup-runtime.md#audio-output)).
What can be heard comes from the desktop XAudio2 DLL that an executable loads from the system. Under
Proton, that is Wine's `xaudio2_9`, which plays through `mmdevapi` and `winepulse`.

`winepulse` finds the PulseAudio or PipeWire-Pulse socket through `XDG_RUNTIME_DIR`. With that
variable unset, the connection is refused (`env -u XDG_RUNTIME_DIR pactl info` prints
`Connection failure: Connection refused`), and Wine falls back to `winealsa`. ALSA's `default`
device is the PulseAudio plugin only when a server is reachable, so ALSA falls back to the first
hardware card. On the measured host no sound reached the desktop's default output, and:
- the debugger's output had `pcm_dmix ... unable to open slave` followed by underruns, in 86 of 103 runs;
- there were no Pulse lines.

Keep `XDG_RUNTIME_DIR` unset and export `PULSE_SERVER` as in the table. Check it during a run with
`pactl list short sink-inputs`: in the measured run, a stream from Wine appeared on the default sink
and the ALSA lines dropped from 4 to 0. The console audio path stays silent either way.

## Behaviour seen only under Wine

- **Single-page mapping bursts get slower as views accumulate.**
  - With `X2W_XMEM_SHARED_VIEWS=1`, an `XMemMapPhysicalPages` call for one page still makes a view of
    its own: two `MapViewOfFile3` calls, the first refused by the larger placeholder, plus one
    placeholder split.
  - In the measured runs, one console executable made about 50,000-55,000 such calls, starting about
    9 s in and peaking at 7,000-12,700 calls per second.
  - Under Wine the cost of each map grows with the number of views in the process: about 28 µs at
    1,000 views, 0.6 ms at 37,000 and 1.1 ms at 52,000.
  - A source-built caller outside any executable, making 50,000 single-page calls, measured a median of
    40 µs per call at the start and 2,140 µs at the end (53×).
  - During the burst the mapping thread does little else, and presents dropped to about 5 per second.
    They recovered to about 30 per second once the burst ended. In some runs the burst outlasted the
    executable's own timeout of about 40 s, and the process ended itself.
  - How often this happened depended on the time of the run (machine load): 2 of 2 runs in one window,
    0 of 9 over the next half hour. Windows was not measured.
- **Replacing a placeholder is not atomic for other threads.** One experiment extended a mapped run by
  unmapping it, coalescing the placeholders and mapping one larger view. A concurrent reader then saw
  zeros, not an access violation, on 15-23 of 41,000-102,000 reads per run. A fault-retry handler
  cannot cover that window, so a stand-in should not remap pages that other threads may read.
- **No debug heap.** Wine never sets the heap tail-check, free-check and parameter-validation flags
  for a process started under a debugger. `startup_debug_probe` reports `0x0` with and without
  `_NO_DEBUG_HEAP=0`, so the debug-heap control in `startup_debug` fails under Wine. In 3 interleaved
  pairs of runs, `_NO_DEBUG_HEAP=1` made no measurable difference there.
- **Descriptor handles and window order** differ as described in
  [Startup debugger under Wine](non-windows-hosts.md#startup-debugger-under-wine).

## Reporting a run

Summarize the run's log directory with [`triage_run.py`](../tools/startup/triage_run.py) and share
only its output. The report keeps allowlisted counts, codes and slot numbers, and groups any other
text as `other (text omitted)`. Do not attach raw logs, paths, names or captures. Say that it is a
Wine run and give the Proton and vkd3d-proton versions, the GPU and driver, the repository commit,
the switches you set and the observation window. List Wine-only failures and capability skips
separately, as [Evidence and Windows verification handoff](non-windows-hosts.md#evidence-and-windows-verification-handoff) describes.

## Commands

```bash
python3 -B tools/startup/triage_run.py out/startup/run1 --shader-cache out/startup/shader_cache
python3 -B tests/startup_triage_run_check.py
pactl list short sink-inputs
```

On Windows, these checks cover the Wine differences above (`startup_debug` fails its debug-heap control under Wine):

```powershell
ctest --preset dev --no-tests=error -R '^startup_debug' --output-on-failure
ctest --preset release --no-tests=error -R '^xmem_shim_(mapped_pages|page_runs)$' -V
```
