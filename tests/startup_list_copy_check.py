"""Compare slot 15 recording modes and verify the page-probe negative control."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PHASES = re.compile(
    r"CopyBufferRegion phases: calls (\d+), us per call \(max\).*?"
    r" page-probes (\d+\.\d+) \(\d+\.\d+\).*"
    r"; calls waiting over 1 ms on a lock (\d+)(?:; guarded-faults (\d+))?$",
    re.MULTILINE,
)


def verify_phases(log, expected, *, check_speed=False):
    matches = PHASES.findall(log)
    windows = [(int(calls), int(faults) if faults else None)
               for calls, _, _, faults in matches]
    if windows != expected:
        raise AssertionError(f"phase windows {windows}, expected {expected}")
    if check_speed:
        _, probes, slow, _ = matches[0]
        # Loose bounds: the gate runs several builds at once, so a preempted copy can look slow.
        # Taking the draw lock would stall every copy in the ten held batches (about 1000 of 2000).
        if float(probes) >= 50 or int(slow) > 100:
            raise AssertionError(f"page probes averaged {probes} us with {slow} slow lock waits")


def run(executable, fast, mutant=False):
    env = dict(os.environ, X2W_COPY15_FAST=str(int(fast)),
               X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1",
               X2W_D3D12_RENDER="1", X2W_D3D12_DEBUG="1",
               X2W_D3D12_LINEAR_BUFFERS="0", X2W_TRANSLATED_INDIRECT_COPY_RANGES="0")
    command = [executable, "warp", "copy15-mutant" if mutant else "copy15"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    print(result.stdout, end="")
    print("\n".join(line for line in result.stderr.splitlines() if PHASES.search(line)))
    if mutant:
        if result.returncode != 92 or "copy15 destination changed after refused copy" not in result.stdout:
            raise AssertionError(f"partial-move control did not corrupt the destination: {result}")
        verify_phases(result.stderr, [(2000, 0), (1, 1)], check_speed=True)
        return None
    if result.returncode:
        raise AssertionError(f"copy15 child failed ({result.returncode}):\n{result.stderr[-8000:]}")
    verify_phases(result.stderr, [(2000, 0 if fast else None), (1, 1 if fast else None)], check_speed=fast)
    if result.stderr.count("CopyBufferRegion skipped: title_backed_source") != 2:
        raise AssertionError("expected both legacy title_backed_source refusals")
    timing = re.search(r"copy15 record us (\d+\.\d+)", result.stdout)
    if not timing or float(timing[1]) <= 0:
        raise AssertionError("missing copy-thread duration")
    return float(timing[1])


def draw(executable, xmem):
    from startup_address_srv_check import draw_texture
    from startup_copied_vertex_srv_check import PIXEL, VERTEX

    for mode, fast in (("copy15", "0"), ("copy15", "1"), ("copy15-tail", "1"), ("copy15-lazy", "1")):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            log = folder / "trap.log"
            pixels = draw_texture(folder, executable, "warp", PIXEL, {
                "X2W_TEST_COPIED_VERTEX_SRV": mode, "X2W_COPY15_FAST": fast,
                "X2W_TEST_COPY15_XMEM": xmem, "X2W_TRAP_LOG": str(log),
                "X2W_D3D12_LINEAR_BUFFERS": "0", "X2W_XMEM_LAZY_COMMIT": "1",
                "X2W_XMEM_LAZY_STARTUP": "0", "X2W_XMEM_GPU_MIRROR": "0",
                "X2W_XMEM_SHARED_VIEWS": "0", "X2W_XMEM_PERMISSIVE": "0",
            }, binding="srv 0 buf table 0 0", vertex_hlsl=VERTEX)
            full = bytes((73, 149, 0, 255))
            short = bytes((73, 0, 0, 255))
            expected = (full * 2 + short * 4 + full * 2) * 4
            if pixels != expected:
                raise AssertionError(f"{mode} fast={fast}: got {pixels!r}, expected {expected.hex()}")
            lazy = mode == "copy15-lazy"
            expected_windows = [(1 if lazy else 2000, 0 if fast == "1" else None)]
            if mode == "copy15-tail":
                expected_windows.append((1, 1))
            verify_phases(log.read_text(errors="replace"), expected_windows, check_speed=mode == "copy15" and fast == "1")
            print(f"{mode} fast={fast}: WARP draw matched all 32 CPU-reference pixels")


if __name__ == "__main__":
    executable, *mode = sys.argv[1:]
    if mode == ["--mutant"]:
        run(executable, True, mutant=True)
    elif len(mode) == 2 and mode[0] == "--draw":
        draw(executable, mode[1])
    elif not mode:
        baseline = run(executable, False)
        fast = run(executable, True)
        # Both modes avoid draw_lock for this buffer pair. Record the ratio
        # without treating the artificial draw-lock hold as a speedup oracle.
        print(f"copy15 ON/OFF recording time {fast / baseline:.2%}")
    else:
        raise SystemExit(f"unknown mode: {mode}")
