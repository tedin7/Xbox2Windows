"""Exercise first-use native graphics translation and a fresh-process cache hit."""
import os
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


def check(folder, consumer, cli, sdk, adapter):
    cache = folder / "cache"
    cache.mkdir()
    vs, ps, root, pixels = [folder / name for name in ("vs", "ps", "root", "pixels")]
    vs.write_bytes(triangle())
    ps.write_bytes(patched_pixel(2, [vop1(1, 0, 128), vop1(1, 1, 242)], export=(0xf800180f, 0x01000001)))
    env = dict(os.environ, X2W_SHADER_CACHE=str(cache), X2W_NATIVE_SHADERS=sdk,
               X2W_D3D12_DEBUG="1", X2W_D3D12_RENDER="1", X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1")
    env.pop("X2W_TEST_NATIVE_GRAPHICS_BAD_MANIFEST", None)
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
    # A published disk pair must be used without attempting native compilation.
    offline = subprocess.run([cli, "--watch", sdk, folder / "offline", cache, "--once"],
                             capture_output=True, text=True, timeout=45)
    assert offline.returncode == 0 and "compiled graphics" in offline.stdout, (offline.stdout, offline.stderr)
    key = next(cache.glob("req_*.a")).stem.removeprefix("req_")
    published = [cache / f"{key}{suffix}" for suffix in (".txt", ".vs.cso", ".ps.cso")]
    contents = [path.read_bytes() for path in published]
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
    assert run.returncode == 0 and "-> drawn;" in run.stderr, run.stderr
    assert "native graphics " not in run.stderr, run.stderr
    assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
    assert [path.read_bytes() for path in published] == contents
    print(f"{adapter} published disk pair: 16 red and 16 clear pixels, cache files unchanged")

    manifest = published[0]
    retries = (
        ("unavailable DXC", {"X2W_NATIVE_SHADERS": str(folder / "missing-sdk")},
         (f"native graphics {key} refused:",)),
        ("invalid native manifest", {"X2W_TEST_NATIVE_GRAPHICS_BAD_MANIFEST": "1"},
         (f"unsupported manifest version for {key}", f"native graphics {key} ready=0")),
    )
    for failure, override, diagnostics in retries:
        manifest.rename(Path(str(manifest) + ".pending"))
        retry_env = dict(env, X2W_TEST_PUBLISH_MANIFEST=str(manifest), **override)
        run = subprocess.run(args, env=retry_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0 and all(text in run.stderr for text in diagnostics), run.stderr
        assert Path(str(pixels) + ".first").read_bytes() == clear * 32
        assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
        assert [path.read_bytes() for path in published] == contents
        print(f"{adapter} {failure}: disk publication recovers the next draw in the same process")

    ps.write_bytes(patched_pixel(1, [0xffffffff], export=(0xf800180f, 0x01000001)))
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
    assert run.returncode == 0 and "refused:" in run.stderr and "-> drawn;" not in run.stderr, run.stderr
    assert pixels.read_bytes() == clear * 32
    print("unsupported stage: all 32 pixels retain the clear color")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        sys.exit(check(Path(directory), *sys.argv[1:]))
