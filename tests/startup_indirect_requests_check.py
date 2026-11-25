"""A console pipeline used only through ExecuteIndirect (list slot 61) still writes its translation request.

usage: startup_indirect_requests_check.py <startup_stream_consumer> <warp|hardware>

The consumer builds the stream pipeline (slot 86, with an input layout so its request key differs), then the
pipeline under test, binds it on a guest list and records refused slot-61 calls (no root or target is bound):

  slot57-draw      a device slot 57 pipeline from the same shaders and root, three Draw calls: one request with
                   the slot-57 shaders and root, from one lookup (the pass remembers the pipeline it asked for)
  slot57-bound     the same pipeline bound with no call: no request (negative control)
  slot57-dispatch  the same pipeline with one Dispatch call (a compute-state refusal): no request
  desktop-draw     a slot-86 pipeline with desktop shaders and one Draw call: no lookup

Every slot-61 lookup logs "x2w-render ExecuteIndirect looks up translation for pipeline" (first 16).
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_draw_check import pixel, vertex

LOOKUP = "x2w-render ExecuteIndirect looks up translation for pipeline"


def run(consumer, backend, directory, mode, runs=1):
    cache = directory / mode
    cache.mkdir()
    vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_INPUT_LAYOUT="1")
    env.setdefault("X2W_D3D12_DEBUG", "0")
    logs = []
    for _ in range(runs):
        result = subprocess.run([consumer, vs, ps, root, backend, directory / "unused", f"indirect-{mode}"], env=env,
                                capture_output=True, text=True)
        if backend == "hardware" and result.returncode == 77:
            print("No D3D12 feature-level 12 hardware adapter")
            sys.exit(77)
        if result.returncode:
            raise RuntimeError(f"{mode}: consumer failed on {backend}: {result.returncode}\n{result.stderr[-2000:]}")
        logs.append(result.stderr)
    found = {}
    for marker in sorted(cache.glob("req_*.a")):
        key = marker.name[4:-2]
        found[key] = {ext: (cache / f"req_{key}{ext}").read_bytes() if (cache / f"req_{key}{ext}").exists() else None
                      for ext in (".a", ".b", ".root", ".ia")}
    leftovers = list(cache.glob("*.tmp"))
    assert not leftovers, f"{mode}: temporary request files left: {leftovers}"
    return found, logs, vs.read_bytes(), ps.read_bytes(), root.read_bytes()


def lookups(log):
    return [line for line in log.splitlines() if LOOKUP in line]


def check(consumer, backend):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        bound, logs, *_ = run(consumer, backend, temp, "slot57-bound")
        assert len(bound) == 1, f"slot57-bound: {len(bound)} requests, expected only the stream pipeline's"
        stream_key, stream = next(iter(bound.items()))
        assert stream[".ia"] is not None, "slot57-bound: the stream request has no input layout"
        assert not lookups(logs[0]), "slot57-bound: a lookup without a slot-61 call"

        dispatch, logs, *_ = run(consumer, backend, temp, "slot57-dispatch")
        assert set(dispatch) == {stream_key}, f"slot57-dispatch: {len(dispatch) - 1} extra requests, expected 0"
        assert not lookups(logs[0]), "slot57-dispatch: a compute-state refusal looked the pipeline up"

        _, logs, *_ = run(consumer, backend, temp, "desktop-draw")
        assert "stream pipeline -> REAL desktop pipeline" in logs[0], "desktop-draw: no desktop pipeline was built"
        assert not lookups(logs[0]), "desktop-draw: a desktop pipeline was looked up"

        # Two processes into one cache: the request is written once, the second process finds it.
        found, logs, vs, ps, root = run(consumer, backend, temp, "slot57-draw", runs=2)
        assert stream_key in found, "slot57-draw: the stream pipeline's request is missing"
        extra = {key: files for key, files in found.items() if key != stream_key}
        assert len(extra) == 1, f"slot57-draw: {len(extra)} indirect-only requests, expected 1"
        files = next(iter(extra.values()))
        # Device slot 57 records the vertex program as a and the pixel program as b.
        assert files[".a"] == vs and files[".b"] == ps and files[".root"] == root and files[".ia"] is None, \
            "slot57-draw: the request does not carry the slot-57 shaders and root"
        for index, log in enumerate(logs):
            seen = lookups(log)
            assert len(seen) == 1, f"slot57-draw run {index + 1}: {len(seen)} lookups for 3 calls, expected 1"
            assert seen[0].endswith("(new)"), f"slot57-draw run {index + 1}: {seen[0]}"
    print(f"startup_indirect_requests: PASS on {backend} (slot57-draw: 1 request, 1 lookup for 3 calls, written "
          f"once over 2 processes; slot57-bound, slot57-dispatch: no request; desktop-draw: no lookup)")
    return 0


if __name__ == "__main__":
    sys.exit(check(*sys.argv[1:]))
