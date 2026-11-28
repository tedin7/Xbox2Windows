"""Native pixel atomics through offline publication and the real draw path."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from native_direct_container_check import patched_pixel, signature
from native_startup_request_cache_check import stream_vertex_container
from startup_pixel_uav_check import run_case, DRAWN_LINE, CLEAR

MASK = 0xffffffff
RED = bytes((255, 0, 0, 255))


def shader(stride, glc=True, threshold=None, offset=0, delta=1, wqm=False, bad=None, dynamic=False):
    vop1 = lambda op, dst, src: 0x7e000000 | dst << 17 | op << 9 | src
    vop2 = lambda op, dst, src0, src1: op << 25 | dst << 17 | src1 << 9 | src0
    code = [0xbe810380, 0xf40c0200, 0xfa000000 | stride]
    if wqm or bad == "wqm":
        code += [0xbe98047e, 0xbefe0a7e]  # Save exact EXEC, enter WQM.
    # v0 = x, v1 = y. Per-pixel output address = 4*(4*y+x+1).
    code += [vop1(7, 0, 256), vop1(7, 1, 257), vop1(1, 4, 255), delta,
             vop1(1, 2, 242), vop1(1, 3, 128)]
    if dynamic:
        # Two contended counters at the last two DWORDs of the view.
        code += [vop2(0x1b, 5, 129, 0), vop2(0x1a, 5, 130, 5)]
    if bad == "branch":
        code += [0xbf820000]  # Unflattened guest CFG edge.
    if threshold is not None:
        code += [0x7d080080 | threshold, 0xbe84246a, 0xbf880002]
    if wqm:
        # Horizontal quad swap of integer SV_Position.x. At x=2 on the
        # odd edge this consumes the helper at x=3, not a constant.
        code += [0x7e0802fa, 0xff0cb100]  # v4 = quadSwapX(v0).
        if glc:
            # Preserve v0/v1 for the quad read; form the private output offset
            # in WQM before restoring exact EXEC. v6 = 4*(4*y+x).
            code += [vop2(0x1a, 1, 130, 1), vop2(0x1c, 6, 257, 0),
                     vop2(0x1a, 6, 130, 6)]
        code += [0xbefe0418]  # Restore exact EXEC.
    op = 0xe0c80000 | (0x4000 if glc else 0) | offset
    if bad == "other":
        op = 0xe0cc0000
    if bad == "indexed":
        op |= 0x2000
    if dynamic:
        op |= 0x1000
    code += [op, 0x80020405 if dynamic else 0x80020400]  # v4 is both addend and GLC destination.
    if threshold is not None:
        code += [0xbefe0404]  # Restore the proven original EXEC.
    if wqm and glc:
        # A second already-supported exact atomic add records the returned old
        # value into a disjoint lane slot; no extra exact-mask ALU/store allowed.
        code += [0xe0c81004, 0x80020406]
    elif not wqm and bad != "wqm":
        # Store returned VGPR (or unchanged addend) after reconvergence.
        code += [vop2(0x1a, 1, 130, 1), vop2(0x1c, 0, 257, 0),
                 vop2(0x1a, 0, 130, 0), 0xe0701004, 0x80020400]
    return patched_pixel(len(code), code, table=True, input_address=3 << 8,
                         input_signature=((b"ISG1", signature(b"SV_Position", 1)),),
                         export=(0xf800180f, 0x02030302))


def main(cli, consumer, sdk, backend):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        vs = folder / "vs"
        vs.write_bytes(stream_vertex_container())
        env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1",
                   X2W_D3D12_RENDER="1", X2W_D3D12_DEVICE_NOOP="1",
                   X2W_TEST_DEBUG_LAYER="1", X2W_D3D12_DEBUG="0")
        # The guest descriptor increment differs between AMD and app-local WARP.
        probe = folder / "probe"
        probe.mkdir()
        ps = probe / "ps"
        root = probe / "root"
        ps.write_bytes(shader(32))
        publish = dict(env, X2W_SHADER_CACHE=str(probe), X2W_TEST_PIXEL_UAV="raw-atomic")
        publish.pop("X2W_TEST_DEBUG_LAYER")
        if backend == "hardware":
            publish["X2W_TEST_DISPATCH_HARDWARE"] = "1"
        r = subprocess.run([consumer, vs, ps, root], env=publish, capture_output=True, text=True)
        if r.returncode == 77:
            return 77
        assert r.returncode == 0, r.stderr
        stride = int(re.search(r"raw descriptor stride: (\d+)", r.stdout)[1])
        print(f"descriptor stride={stride}", flush=True)

        def case(name, *, glc=True, threshold=None, offset=0, delta=1, edge=False, wqm=False, dynamic=False):
            work = folder / name
            work.mkdir()
            cache = work / "cache"
            cache.mkdir()
            ps, root = work / "ps", work / "root"
            ps.write_bytes(shader(stride, glc, threshold, offset, delta, wqm, dynamic=dynamic))
            mode = "raw-atomic-edge" if edge else "raw-atomic"
            p = dict(publish, X2W_SHADER_CACHE=str(cache), X2W_TEST_PIXEL_UAV=mode)
            r = subprocess.run([consumer, vs, ps, root], env=p, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            r = subprocess.run([cli, sdk, folder/"frontend", "gfx1013-container-pair", "--requests", cache, cache],
                               capture_output=True, text=True)
            assert r.returncode == 0, (name, r.stdout, r.stderr)
            stage, = cache.glob("*.ps.cso")
            manifest = stage.with_name(stage.name.replace(".ps.cso", ".txt"))
            assert f"uav 0 raw table 0 {stride}" in manifest.read_text()
            ir = subprocess.check_output([Path(sdk)/"dxc.exe", "-dumpbin", stage], text=True)
            assert "@dx.op.atomicBinOp.i32" in ir and "@dx.op.isHelperLane.i1" in ir, ir
            if wqm:
                assert "@dx.op.quadOp.i32" in ir, ir
            program = SimpleNamespace(directory=work, cache=cache, consumer=consumer, vs=vs, ps=ps, root=root)
            rc, out, log, pixels, words = run_case(program, mode, env, backend)
            if rc == 77:
                sys.exit(77)
            assert rc == 0 and "device removed reason: 0x00000000" in out, (name, rc, out, log)
            assert DRAWN_LINE in log, (name, log)
            if "debug layer: on" in out:
                assert "debug layer: on, errors: 0" in out, out
            coverage = [(x,y) for y in range(3 if edge else 4) for x in range(3 if edge else 4)]
            active = [(x,y) for x,y in coverage if threshold is None or x < threshold]
            want_pixels = b"".join(RED if (x,y) in coverage else CLEAR for y in range(4) for x in range(8))
            assert pixels == want_pixels, (name, pixels.hex(), want_pixels.hex())
            expected = [0xabcd0000 + i for i in range(64)]
            base = 36
            for group in range(2 if dynamic else 1):
                lanes = [(x,y) for x,y in active if not dynamic or x%2 == group]
                target = base + offset//4 + group
                in_view = offset//4 + group < 20
                seed = expected[target] if in_view else 0
                if in_view:
                    increment = sum(x ^ 1 for x,y in lanes) if wqm else delta*len(lanes)
                    expected[target] = (seed + increment) & MASK
                if not wqm or glc:
                    returned = []
                    for x,y in coverage:
                        if dynamic and x%2 != group:
                            continue
                        pos = base + 1 + 4*y+x
                        if glc and (x,y) in lanes:
                            returned.append((words[pos]-expected[pos]) & MASK if wqm else words[pos])
                            expected[pos] = words[pos]  # Compare the full old-value multiset independently.
                        else:
                            expected[pos] = (x ^ 1) if wqm else delta
                    want_old = sorted(((seed + delta*j) & MASK) if in_view else 0 for j in range(len(lanes)))
                    if wqm and glc and in_view:
                        # Validate a serialization, including zero increments, without imposing GPU lane order.
                        from functools import lru_cache
                        from collections import Counter
                        edges = Counter((old, x ^ 1) for old,(x,y) in zip(returned, lanes))
                        pairs = tuple(edges)
                        @lru_cache(None)
                        def chain(value, counts):
                            if not any(counts):
                                return value == expected[target]
                            return any(n and old == value and chain((value+add)&MASK,
                                       counts[:i]+(n-1,)+counts[i+1:])
                                       for i, ((old,add),n) in enumerate(zip(pairs,counts)))
                        assert chain(seed, tuple(edges.values())), (name, edges, seed, expected[target])
                    else:
                        assert sorted(returned) == (want_old if glc else []), (name, returned, want_old)
            assert words == expected, (name, [(i,hex(a),hex(b)) for i,(a,b) in enumerate(zip(words,expected)) if a!=b], len(words))
            print(name, f"{len(coverage)} covered, {len(active)} active; 64 DWORDs and 32 pixels exact",
                  next((l for l in out.splitlines() if l.startswith("adapter=")), ""), flush=True)

        case("contention-return")
        case("contention-no-return", glc=False)
        case("half-exec", threshold=2)
        case("empty-exec", threshold=0)
        case("all-exec", threshold=4)
        case("wrap", delta=0x60000001)
        case("last-dword", offset=76)
        case("out-of-view", offset=80)
        case("dynamic-two-counters", offset=72, dynamic=True)
        case("odd-edge", edge=True)
        case("wqm-restore-edge-helper-read", glc=False, edge=True, wqm=True)
        case("wqm-glc-edge-helper-read", edge=True, wqm=True)
        case("wqm-glc-out-of-view", offset=80, edge=True, wqm=True)
        for bad, reason in (("other", "graphics buffer atomic"), ("indexed", "indexed compute atomic"),
                            ("wqm", "pixel atomic requires exact coverage"), ("branch", "straight-line pixel raw DWORD add")):
            source = folder / (bad+".ps")
            dest = folder / (bad+".cso")
            source.write_bytes(shader(stride, bad=bad))
            r = subprocess.run([cli,sdk,folder/"frontend","gfx1013-container-ps",source,dest],
                               capture_output=True,text=True)
            assert r.returncode != 0 and not dest.exists() and reason in r.stderr, (bad,r.stderr)
            print(bad, r.stderr.strip(), flush=True)
        print(backend, "13 actual draws and 4 refusal controls passed")
        return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
