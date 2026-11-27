"""Compare slot 15 recording modes and verify the page-probe negative control."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PHASES = re.compile(
    r"CopyBufferRegion phases: calls (\d+), us per call \(max\).*"
    r"; calls waiting over 1 ms on a lock (\d+)"
)


def verify_phases(log, fast):
    matches = PHASES.findall(log)
    if len(matches) != 1:
        raise AssertionError(f"expected one copy phase window, got {matches}")
    calls, slow = map(int, matches[0])
    if calls != 2000 or (fast and slow):
        raise AssertionError(f"fast={fast}: {calls} calls, {slow} slow lock waits")


def run(executable, fast, mutant=False):
    env = dict(os.environ, X2W_COPY15_FAST=str(int(fast)),
               X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1",
               X2W_D3D12_RENDER="1", X2W_D3D12_DEBUG="1",
               X2W_D3D12_LINEAR_BUFFERS="0", X2W_TRANSLATED_INDIRECT_COPY_RANGES="0")
    command = [executable, "warp", "copy15-mutant" if mutant else "copy15"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    print(result.stdout, end="")
    phase = next((line for line in result.stderr.splitlines() if PHASES.search(line)), "")
    print(phase)
    if mutant:
        fault = result.returncode == 92 and "copy15 unreadable source fault" in result.stdout
        missed_refusal = result.returncode == 93 and "copy15 missing unreadable source refusal" in result.stdout
        if not (fault or missed_refusal):
            raise AssertionError(f"page-probe control did not expose the missing protection check: {result}")
        verify_phases(result.stderr, False)
        return None
    if result.returncode:
        raise AssertionError(f"copy15 child failed ({result.returncode}):\n{result.stderr[-8000:]}")
    verify_phases(result.stderr, fast)
    if result.stderr.count("CopyBufferRegion skipped: title_backed_source") != 2:
        raise AssertionError("expected both legacy title_backed_source refusals")
    timing = re.search(r"copy15 record us (\d+\.\d+)", result.stdout)
    if not timing or float(timing[1]) <= 0:
        raise AssertionError("missing copy-thread duration")
    return float(timing[1])


def draw(executable):
    from startup_address_srv_check import draw_texture
    from startup_copied_vertex_srv_check import PIXEL, VERTEX

    for fast in ("0", "1"):
        with tempfile.TemporaryDirectory() as directory:
            pixels = draw_texture(Path(directory), executable, "warp", PIXEL,
                                  {"X2W_TEST_COPIED_VERTEX_SRV": "copy15", "X2W_COPY15_FAST": fast},
                                  binding="srv 0 buf table 0 0", vertex_hlsl=VERTEX)
        full = bytes((73, 149, 0, 255))
        short = bytes((73, 0, 0, 255))
        expected = (full * 2 + short * 4 + full * 2) * 4
        if pixels != expected:
            raise AssertionError(f"fast={fast}: got {pixels!r}, expected {expected.hex()}")
        print(f"fast={fast}: translated WARP draw matched all 32 pixels")


if __name__ == "__main__":
    executable, *mode = sys.argv[1:]
    if mode == ["--mutant"]:
        run(executable, True, mutant=True)
    elif mode == ["--draw"]:
        draw(executable)
    elif not mode:
        baseline = run(executable, False)
        fast = run(executable, True)
        # Both modes avoid draw_lock for this buffer pair. Record the ratio
        # without treating the artificial draw-lock hold as a speedup oracle.
        print(f"copy15 ON/OFF recording time {fast / baseline:.2%}")
    else:
        raise SystemExit(f"unknown mode: {mode}")
