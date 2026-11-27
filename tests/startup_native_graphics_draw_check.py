"""Exercise first-use native graphics translation and a fresh-process cache hit."""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_direct_container_check import patched_pixel, vertex
from startup_stream_draw_check import vop1, vop2


def triangle():
    code = [vop2(0x1a, 1, 129, 0), vop2(0x1b, 1, 130, 1), vop2(0x1b, 2, 130, 0)]
    for v in (1, 2):
        code.extend([vop1(6, v, 256 + v), vop2(8, v, 244, v), vop2(3, v, 243, v)])
    code.extend([vop1(1, 3, 128), vop1(1, 4, 242), 0xf80008cf, 0x04030201, 0xbf810000])
    return vertex(program=code)


def check(folder, consumer, sdk, adapter, compiler):
    cache = folder / "cache"
    cache.mkdir()
    vs, ps, root, pixels = [folder / name for name in ("vs", "ps", "root", "pixels")]
    vs.write_bytes(triangle())
    ps.write_bytes(patched_pixel(2, [vop1(1, 0, 128), vop1(1, 1, 242)], export=(0xf800180f, 0x01000001)))
    env = dict(os.environ, X2W_SHADER_CACHE=str(cache), X2W_NATIVE_SHADERS=sdk,
               X2W_D3D12_DEBUG="1", X2W_D3D12_RENDER="1", X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1")
    clear = bytes((0, 0, 255, 255))
    red = bytes((255, 0, 0, 255))
    args = [consumer, vs, ps, root, adapter, pixels]
    for name, hits in (("cold", "0,0,0"), ("hot", "1,1,1")):
        run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
        if run.returncode == 77:
            return 77
        if run.returncode or f"ready=1 cache(frontend,vs,ps)={hits}" not in run.stderr or "-> drawn;" not in run.stderr:
            raise RuntimeError(run.stderr[-8000:])
        for line in run.stderr.splitlines():
            if "x2w-d3d12-debug " in line:
                assert " sev=0 " not in line and " sev=1 " not in line, line
            if "debug layer:" in line:
                print(line)
        print(run.stdout.strip())
        assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
        print(f"{adapter} {name}: cache={hits}, 16 red and 16 clear pixels")
    # Publish a verified disk pair, then hold back its ready marker until after the first draw.
    subprocess.run([compiler, "--watch", sdk, cache, cache, "--once"], check=True, timeout=45)
    marker, = cache.glob("req_*.a")
    manifest = cache / (marker.stem.removeprefix("req_") + ".txt")
    published = {p: p.read_bytes() for p in
                 (manifest, manifest.with_suffix(".vs.cso"), manifest.with_suffix(".ps.cso"))}
    for fault, diagnostic in (("MANIFEST", "unsupported manifest version"), ("ROOT", "failed 0x80004005")):
        manifest.rename(str(manifest) + ".pending")
        first = Path(str(pixels) + ".first")
        first.unlink(missing_ok=True)
        pixels.unlink()
        fault_env = dict(env, X2W_TEST_PUBLISH_MANIFEST=str(manifest), X2W_NATIVE_SHADERS_ONLY="0")
        fault_env[f"X2W_TEST_NATIVE_GRAPHICS_{fault}_FAILURE"] = "1"
        run = subprocess.run(args, env=fault_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0 and diagnostic in run.stderr and "-> drawn;" in run.stderr, run.stderr
        assert first.read_bytes() == clear * 32
        assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
        assert all(path.read_bytes() == expected for path, expected in published.items())
        print(f"{adapter} {fault}: same-process fallback drew 16 red pixels after publication")
    ps.write_bytes(patched_pixel(1, [0xffffffff], export=(0xf800180f, 0x01000001)))
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
    assert run.returncode == 0 and "refused:" in run.stderr and "-> drawn;" not in run.stderr, run.stderr
    assert pixels.read_bytes() == clear * 32
    print("unsupported stage: all 32 pixels retain the clear color")
    # The first DWORD is 0.25 in record zero, 1.0 in record one. Its byte-1 window is 0x003e8000.
    records = bytearray(64)
    struct.pack_into("<f", records, 0, 0.25)
    struct.pack_into("<f", records, 16, 1.0)
    env["X2W_TEST_FORMATTED_VIEW"] = f"0:1688:0:{records.hex()}"
    for index, stride in ((1, 16), (1, 8), (0xffffffff, 16)):
        env["X2W_TEST_INDEXED_STRIDE"] = str(stride)
        program = [0xbe810380, 0xf4080200, 0xfa000000, vop1(1, 0, 255), index,
                   0xe0302000, 0x80020000, vop1(1, 1, 128), vop1(1, 2, 242)]
        ps.write_bytes(patched_pixel(len(program), program, table=True, uses=[(0x300, 0, 0, 16, 4)],
                                     export=(0xf800180f, 0x02010100)))
        run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        for line in run.stderr.splitlines():
            if "x2w-d3d12-debug " in line:
                assert " sev=0 " not in line and " sev=1 " not in line, line
        image = pixels.read_bytes()
        if stride == 8:
            assert "needs stride 16, got 8" in run.stderr and "-> drawn;" not in run.stderr, run.stderr[-7000:]
            assert image == clear * 32
        else:
            assert "-> drawn;" in run.stderr, run.stderr[-7000:]
            color = red if index == 1 else bytes((0, 0, 0, 255))
            assert image == (color * 4 + clear * 4) * 4, (image.hex(), run.stderr[-6000:])
        print(f"{adapter} idxbuf index={index:#x} actual stride={stride}: first RGBA {image[:4].hex()}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        sys.exit(check(Path(directory), *sys.argv[1:]))
