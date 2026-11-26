#!/usr/bin/env bash
# Runs a console executable on the desktop stand-ins under the startup debugger inside Wine
# (docs/wine.md). The Linux counterpart of tools/startup/run.ps1, with the same switches.
#
# Copies the stand-in DLLs from the build next to the executable (a folder that holds the
# executable and the package's other DLLs; prepare it yourself, never commit it), sets the
# permissive switches and starts tools/startup/debug_launch.py with a Windows Python in Wine.
# The run is offline: the debugger refuses sockets, NetBIOS and DNS. Logs go to --log;
# <log>.out holds the debugger's summary. The script exits with the debugger's exit code.
#
# Environment:
#   WINEPREFIX       the prefix to run in (required; set it up as docs/non-windows-hosts.md describes)
#   X2W_WIN_PYTHON   host path of a Windows python.exe that runs inside Wine (required)
#   WINE             the wine binary (default: $PROTON/bin/wine when PROTON is set, else wine)
#   X2W_FXC          the FXC the shader watcher runs under --translate (docs/startup-runtime.md);
#                    a host path is passed in its Z: spelling
#   WINEDLLOVERRIDES, WINEFSYNC, WINEDEBUG  default to the docs/wine.md values when unset
#   X2W_HOST_PYTHON  the host python that compares the shader cache with the translator (default: python3)
# Paths may be host paths or drive paths such as 'G:\'. The debugger and the watcher get them as
# given (host paths in their Z: spelling); this script reads and creates drive paths through the
# prefix's dosdevices links. Needs bash 4.4 or newer, GNU realpath and setsid (Linux).
#
# Example:
#   WINEPREFIX=~/prefix X2W_WIN_PYTHON=~/py/python.exe tools/startup/run.sh \
#       --exe ~/run/title.exe --data 'G:\' --log out/startup/run1 --seconds 180 --render --fake-user
set -uo pipefail

usage() {
    echo "usage: run.sh --exe EXE --data DIR --log DIR [--seconds 120] [--build-dir out/build/release]" >&2
    echo "       [--memory-gib 14] [--continue-asserts 0] [--input-script S] [--storage DIR]" >&2
    echo "       [--shader-cache out/startup/shader_cache] [--console 0|1] [--render] [--fake-user]" >&2
    echo "       [--gpu-mirror] [--host-cpu] [--translate] [--startup-below-40] [--trace-calls]" >&2
    echo "       [--hide-title-window] [--capture-every S] [--fresh-shader-cache]" >&2
    exit 64
}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
here="$repo/tools/startup"
exe="" data="" log="" seconds=120 build_dir=out/build/release memory_gib=14 continue_asserts=0
input_script="" storage="" shader_cache=out/startup/shader_cache console=1 capture_every=0
render="" fake_user="" gpu_mirror="" host_cpu="" translate="" below_40="" trace_calls="" hide_title="" fresh_cache=""
while [ $# -gt 0 ]; do
    case $1 in
        --exe|--data|--log|--seconds|--build-dir|--memory-gib|--continue-asserts|--input-script|--storage|\
        --shader-cache|--console|--capture-every)
            [ $# -ge 2 ] || usage
            case $1 in
                --exe) exe=$2 ;; --data) data=$2 ;; --log) log=$2 ;; --seconds) seconds=$2 ;;
                --build-dir) build_dir=$2 ;; --memory-gib) memory_gib=$2 ;;
                --continue-asserts) continue_asserts=$2 ;;  # main-image int3s to log and step over
                --input-script) input_script=$2 ;;          # X2W_INPUT_SCRIPT, e.g. "4@70" = A at 70 s
                --storage) storage=$2 ;;                    # X2W_STORAGE_PATH: persistent local storage root
                --shader-cache) shader_cache=$2 ;; --console) console=$2 ;;
                --capture-every) capture_every=$2 ;;        # window_<sss>s.png every N seconds (0 = off)
            esac
            shift 2 ;;
        --render) render=1; shift ;;              # draw through translated pipelines
        --fake-user) fake_user=1; shift ;;        # answer the user API for one local user
        --gpu-mirror) gpu_mirror=1; shift ;;      # map CPU-view pages into the paired GPU-view reservation
        --host-cpu) host_cpu=1; shift ;;          # rewrite instructions the host processor lacks
        --translate) translate=1; shift ;;        # with --render: translate new shader requests during the run
        --startup-below-40) below_40=1; shift ;;  # X2W_XMEM_STARTUP_BELOW_40
        --trace-calls) trace_calls=1; shift ;;    # also break on runtime and movie-player calls (slow)
        --hide-title-window) hide_title=1; shift ;;  # X2W_HIDE_TITLE_WINDOW: hide the title's own window
        --fresh-shader-cache) fresh_cache=1; shift ;;  # with --render: move a cache that is not empty aside first
        -h|--help) usage ;;
        *) echo "run.sh: unknown argument '$1'" >&2; usage ;;
    esac
done
[ -n "$exe" ] && [ -n "$data" ] && [ -n "$log" ] || usage
[ -n "${WINEPREFIX:-}" ] || { echo "run.sh: set WINEPREFIX (docs/non-windows-hosts.md)" >&2; exit 64; }
[ -n "${X2W_WIN_PYTHON:-}" ] || { echo "run.sh: set X2W_WIN_PYTHON to a Windows python.exe (docs/wine.md)" >&2; exit 64; }
export WINEPREFIX
wine=${WINE:-}
[ -n "$wine" ] || { [ -n "${PROTON:-}" ] && wine=$PROTON/bin/wine; } || wine=wine

# A drive path ('G:\dir') names a host path through the prefix's dosdevices links.
host_path() {
    if [[ $1 =~ ^([A-Za-z]):[\\/]?(.*)$ ]]; then
        local rest=${BASH_REMATCH[2]//\\//}
        printf '%s\n' "$WINEPREFIX/dosdevices/${BASH_REMATCH[1],,}:/$rest"
    else
        realpath -m -- "$1"
    fi
}
# The Windows spelling of a path for the debugger: drive paths stay, host paths go through Z:.
win_path() {
    if [[ $1 =~ ^[A-Za-z]: ]]; then
        printf '%s\n' "$1"
    else
        local full
        full=$(realpath -m -- "$1")
        printf 'Z:%s\n' "${full//\//\\}"
    fi
}

case $build_dir in /*) ;; *) build_dir=$repo/$build_dir ;; esac
run_dir=$(dirname "$(host_path "$exe")")
for dll in xmem d3d12_xs AcpHal XCRDAPI chad xapu xg_xs GameInput; do
    cp -f "$build_dir/$dll.dll" "$run_dir/" || exit 1
done
export X2W_AUDIO_PERMISSIVE=1 X2W_D3D12_DEVICE_NOOP=1 X2W_D3D12_LIST_NOOP=1
export X2W_D3D12_PERMISSIVE=1 X2W_XMEM_PERMISSIVE=1 X2W_RUNTIME_PROXY=1
# Under Wine, one section and view per page did not get past start-up (docs/wine.md).
export X2W_XMEM_SHARED_VIEWS=${X2W_XMEM_SHARED_VIEWS:-1}
export WINEDEBUG=${WINEDEBUG:--all} WINEFSYNC=${WINEFSYNC:-1}
# Wine's built-in D3D12 is not enough (docs/wine.md); the prefix holds vkd3d-proton and DXVK.
export WINEDLLOVERRIDES=${WINEDLLOVERRIDES:-d3d12,d3d12core,dxgi,d3d11=n}
# Keep vkd3d-proton's pipeline cache under out/ instead of the working directory.
export VKD3D_SHADER_CACHE_PATH=${VKD3D_SHADER_CACHE_PATH:-$repo/out/linux/vkd3d-cache}
mkdir -p "$VKD3D_SHADER_CACHE_PATH"
# Without XDG_RUNTIME_DIR, Wine backs sections in /tmp rather than a small tmpfs; PULSE_SERVER
# keeps audio working (docs/wine.md).
pulse=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/pulse/native
unset XDG_RUNTIME_DIR
[ -n "${PULSE_SERVER:-}" ] || [ ! -S "$pulse" ] || export PULSE_SERVER=unix:$pulse
config=$(host_path "$data")/MicrosoftGame.config
if [ -f "$config" ]; then
    # The first tag of each in the file, any case, as run.ps1's -match reads them; a caller's value
    # stays when a tag is missing. Unlike run.ps1, a value must start on its tag's line, and a
    # UTF-16 config is not read.
    title_id=$(grep -oi '<TitleId>[^<]*' "$config" | head -n 1 | cut -d '>' -f 2- | tr -d '\r')
    size_mb=$(grep -oi '<SizeMB>[^<]*' "$config" | head -n 1 | cut -d '>' -f 2- | tr -d '\r')
    [ -z "$title_id" ] || export X2W_TITLE_ID=$title_id
    [ -z "$size_mb" ] || export X2W_STORAGE_MB=$size_mb
fi
if [ -n "$storage" ]; then
    # The title appends its own names to this path as it is, so it ends in a backslash.
    X2W_STORAGE_PATH=$(win_path "$storage")
    [[ $X2W_STORAGE_PATH == *\\ ]] || X2W_STORAGE_PATH+=\\
    export X2W_STORAGE_PATH
fi
[ -z "$input_script" ] || export X2W_INPUT_SCRIPT=$input_script
[ -z "$fake_user" ] || export X2W_FAKE_USER=1
if [ -z "$fake_user" ] && [ "${X2W_FAKE_USER:-}" != 1 ]; then
    echo "WARNING: Without --fake-user, user API calls go to the host gaming runtime (docs/startup-runtime.md)." >&2
fi
[ -z "$gpu_mirror" ] || export X2W_XMEM_GPU_MIRROR=1
[ -z "$host_cpu" ] || export X2W_HOST_CPU=1
[ -z "$below_40" ] || export X2W_XMEM_STARTUP_BELOW_40=1
[ -z "$hide_title" ] || export X2W_HIDE_TITLE_WINDOW=1
if [ -n "$hide_title" ] && [ -z "$render" ]; then
    echo "WARNING: --hide-title-window needs --render (no output window without it); it has no effect" >&2
fi
host_python=${X2W_HOST_PYTHON:-python3}
if [ -n "$fresh_cache" ] && [ -z "$render" ]; then
    echo "WARNING: --fresh-shader-cache needs --render (no shader cache without it); ignored" >&2
fi
if [ -n "$render" ]; then
    if [ -n "$fresh_cache" ]; then
        "$host_python" -B "$here/shader_cache_revision.py" "$(host_path "$shader_cache")" --fresh >&2
    fi
    mkdir -p "$(host_path "$shader_cache")"
    export X2W_D3D12_RENDER=1 X2W_DRAW_NULL_IDENTITY=1
    X2W_SHADER_CACHE=$(win_path "$shader_cache")
    export X2W_SHADER_CACHE
fi
# A trailing separator would put the summary inside the log folder as ".out".
while [[ ${#log} -gt 1 && $log =~ [^:][/\\]$ ]]; do log=${log%?}; done
log_host=$(host_path "$log")
mkdir -p "$log_host"

watcher=""
# shellcheck disable=SC2329  # called through the EXIT trap
stop_watcher() {
    [ -n "$watcher" ] || return 0
    # The watcher leads its own process group (setsid): its parallel workers and their
    # fxc.exe are separate processes in that group.
    kill -TERM -- "-$watcher" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 -- "-$watcher" 2>/dev/null || return 0
        sleep 0.2
    done
    kill -KILL -- "-$watcher" 2>/dev/null
    return 0
}
trap stop_watcher EXIT
# On Ctrl-C: stop the watcher, then end by SIGINT so that a calling shell loop stops too. bash
# runs the trap once the debugger, which gets the Ctrl-C as well, has returned. A TERM sent to this
# script alone needs no trap: bash waits for the debugger, runs the EXIT trap and ends by SIGTERM.
# shellcheck disable=SC2329  # called through the trap below
on_signal() {
    stop_watcher
    watcher=""
    trap - "$1" EXIT
    kill -s "$1" $$
}
trap 'on_signal INT' INT
# Requests keyed by another revision are all refused, and the translator's log does not say why: warn once,
# before the watcher starts (revision.txt of an earlier run) or after the run (the one this run's stand-ins wrote).
revision_warned=""
check_cache_revision() {
    local note rc
    note=$("$host_python" -B "$here/shader_cache_revision.py" "$(host_path "$shader_cache")" \
        --translator "$here/gfx2hlsl.py" 2>&1)
    rc=$?
    if [ $rc -eq 1 ]; then
        echo "WARNING: $note" >&2
        revision_warned=1
    elif [ $rc -ne 0 ]; then
        echo "WARNING: the shader cache revision was not checked ($host_python exit $rc)" >&2
        revision_warned=1
    fi
}
if [ -n "$translate" ] && [ -z "$render" ]; then
    echo "WARNING: --translate needs --render (no shader cache without it); ignored" >&2
fi
if [ -n "$translate" ] && [ -n "$render" ]; then
    # Only the watcher loads the native d3dcompiler_47, so that FXC uses the SDK's compiler
    # (docs/non-windows-hosts.md); the stand-in presenter compiles through Wine's built-in one.
    # Without job control (set +m, the default in a script) the watcher is not a group leader
    # before setsid, so setsid does not fork and $! is the new group's id.
    # An empty X2W_FXC (manifest only) is passed on as it is.
    if [ -n "${X2W_FXC:-}" ]; then X2W_FXC=$(win_path "$X2W_FXC"); export X2W_FXC; fi
    check_cache_revision
    WINEDLLOVERRIDES="$WINEDLLOVERRIDES;d3dcompiler_47=n" \
        setsid "$wine" "$X2W_WIN_PYTHON" -B "$(win_path "$here/gfx2hlsl.py")" --watch "$X2W_SHADER_CACHE" \
        > "$log_host/translate.txt" 2> "$log_host/translate_err.txt" < /dev/null &
    watcher=$!
fi
trace_args=()
[ -z "$trace_calls" ] || trace_args=(--trace-calls 1)
console_args=()
[ "$console" = 1 ] || console_args=(--console "$console")
capture_args=()
[ "$capture_every" = 0 ] || capture_args=(--capture-every "$capture_every")
"$wine" "$X2W_WIN_PYTHON" -B "$(win_path "$here/debug_launch.py")" "$(win_path "$exe")" "$(win_path "$data")" \
    "$(win_path "$log")" --timeout "$seconds" --memory-gib "$memory_gib" --trace-files 1 \
    --continue-asserts "$continue_asserts" "${trace_args[@]}" "${console_args[@]}" "${capture_args[@]}" > "$log_host.out" 2>&1
debugger_exit=$?
if [ -n "$translate" ] && [ -n "$render" ] && [ -z "$revision_warned" ]; then check_cache_revision; fi
head -n 5 "$log_host.out"
exit $debugger_exit
