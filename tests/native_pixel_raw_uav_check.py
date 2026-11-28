"""Source-built native pixel buffer store through the actual graphics cache and draw."""
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from native_direct_container_check import patched_pixel, signature
from native_startup_request_cache_check import stream_vertex_container
from startup_pixel_uav_check import run_case, DRAWN_LINE, CLEAR

VALUE = 0x11223344


def shader(stride):
    vop1 = lambda op, dst, src: 0x7e000000 | dst << 17 | op << 9 | src
    vop2 = lambda op, dst, src0, src1: op << 25 | dst << 17 | src1 << 9 | src0
    # Load the second guest descriptor. Pixel (x,y) writes DWORD 4*y+x
    # relative to that view; the triangle covers x=0..3, y=0..3.
    code = [0xbe810380, 0xf40c0200, 0xfa000000 | stride,
            vop1(7, 0, 256), vop1(7, 1, 257),
            vop2(0x1a, 1, 130, 1), vop2(0x1c, 0, 257, 0),
            vop2(0x1a, 0, 130, 0), vop1(1, 4, 255), VALUE,
            0xe0701000, 0x80020400, vop1(1, 2, 242), vop1(1, 3, 128)]
    return patched_pixel(len(code), code, table=True, input_address=3 << 8,
                         input_signature=((b"ISG1", signature(b"SV_Position", 1)),),
                         export=(0xf800180f, 0x02030302))


def main(cli, consumer, sdk, backend, baseline=False):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cache = folder / "cache"
        cache.mkdir()
        vs, ps, root = (folder / n for n in ("vs", "ps", "root"))
        vs.write_bytes(stream_vertex_container())
        ps.write_bytes(shader(32))
        env = dict(os.environ, X2W_SHADER_CACHE=str(cache), X2W_D3D12_PERMISSIVE="1",
                   X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1", X2W_D3D12_DEVICE_NOOP="1",
                   X2W_TEST_PIXEL_UAV="raw-ok", X2W_TEST_DEBUG_LAYER="1", X2W_D3D12_DEBUG="0")
        publish = dict(env)
        publish.pop("X2W_TEST_DEBUG_LAYER")
        if backend == "hardware":
            publish["X2W_TEST_DISPATCH_HARDWARE"] = "1"
        probe = subprocess.run([consumer, vs, ps, root], env=publish, capture_output=True, text=True)
        if probe.returncode == 77:
            sys.exit(77)
        assert probe.returncode == 0, probe.stderr
        stride = int(re.search(r"raw descriptor stride: (\d+)", probe.stdout)[1])
        assert stride >= 32
        print(f"guest descriptor stride: {stride}")
        if stride != 32:
            for entry in cache.iterdir():
                entry.unlink()
            ps.write_bytes(shader(stride))
            subprocess.run([consumer, vs, ps, root], env=publish, check=True, capture_output=True)
        line = f"uav 0 raw table 0 {stride}"
        result = subprocess.run([cli, sdk, folder/"frontend", "gfx1013-container-pair", "--requests", cache, cache],
                                capture_output=True, text=True)
        print(result.stdout, result.stderr, flush=True)
        assert result.returncode == 0, result.stderr
        stage, = cache.glob("*.ps.cso")
        manifest = stage.with_name(stage.name.replace(".ps.cso", ".txt"))
        if baseline:
            assert not manifest.exists(), "baseline unexpectedly published raw UAV"
            # Pair compilation succeeds on baseline; supply only the missing manifest
            # to isolate the real runtime's independent refusal.
            manifest.write_text("x2w-manifest 2\nud 1\nudsrc 0 rect\n" + line + "\nparams 0\ntargets 0\n")
        text = manifest.read_text()
        assert line in text.splitlines(), text
        program = SimpleNamespace(directory=folder, cache=cache, consumer=consumer, vs=vs, ps=ps, root=root)

        def check(name, start=None, count=16, reason=None, decoy=False):
            rc, out, log, pixels, words = run_case(program, name, env, backend)
            if rc == 77:
                print(out)
                sys.exit(77)
            assert rc == 0, (name, rc, out, log)
            assert "device removed reason: 0x00000000" in out, out
            if "debug layer: on" in out:
                assert "debug layer: on, errors: 0" in out, out
            expected = [0xabcd0000 + i for i in range(64)]
            if start is not None:
                base = (0 if decoy else 32) + start
                expected[base:base+count] = [VALUE]*count
            assert words == expected, (name, [(i, hex(a), hex(b)) for i,(a,b) in enumerate(zip(words,expected)) if a!=b], len(words), out, log)
            assert (DRAWN_LINE in log) == (start is not None), (name, log)
            if reason:
                assert reason in log, (name, log)
            red = bytes((255,0,0,255))
            assert pixels == ((red*4 + CLEAR*4)*4 if start is not None else CLEAR*32), (name, pixels.hex(), log)
            print(name, "64 DWORDs + 32 pixels exact;", next((l for l in out.splitlines() if l.startswith("adapter=")), ""), next((l for l in out.splitlines() if l.startswith("debug layer:")), ""), flush=True)

        if baseline:
            check("raw-ok", reason="buffer view (raw uav) is not bound")
            return
        check("raw-ok", 4)
        check("raw-short", 4, 8)
        check("raw-last", 16)
        for name, reason in (
            ("raw-typed", "incompatible raw buffer UAV"),
            ("raw-structured", "incompatible raw buffer UAV"),
            ("raw-no-ua", "resource lacks ALLOW_UNORDERED_ACCESS"),
            ("raw-missing", "table entry holds no view"),
            ("raw-srv", "table entry is not an unordered-access view"),
            ("raw-no-table", "no descriptor table at the parameter"),
        ):
            check(name, reason=reason)
        # A different live descriptor at offset zero must change the written buffer.
        manifest.write_text(text.replace(line, "uav 0 raw table 0 0"))
        check("raw-ok.decoy", 4, decoy=True)
        manifest.write_text(text.replace(line, "uav 0 raw table 0 4096"))
        check("raw-ok.out-of-table", reason="table entry outside the bound table")
        manifest.write_text(text.replace(line, "uav 0 raw table 0 1"))
        check("raw-ok.unaligned", reason="raw view requires an aligned table record without typedload")
        manifest.write_text(text.replace(line, line + " typedload"))
        check("raw-ok.typedload", reason="raw view requires an aligned table record without typedload")
        if stride > 32:
            manifest.write_text(text.replace(line, "uav 0 raw table 0 32"))
            check("raw-ok.wrong-stride", reason="raw UAV offset is not a descriptor boundary")
        print(backend, f"{13 + (stride > 32)} raw UAV cases passed")


if __name__ == "__main__":
    main(*sys.argv[1:5], baseline=len(sys.argv) == 6 and sys.argv[5] == "baseline")
