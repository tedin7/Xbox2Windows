"""Startup request consumption and runtime cache publication for compile_dxil (#2933).

usage:
  native_startup_request_cache_check.py controls <compile_dxil> <sdk>
  native_startup_request_cache_check.py e2e <compile_dxil> <startup_stream_consumer> <offscreen_check> <sdk> <warp|hardware>
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools" / "startup"))

import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402
from native_direct_container_check import patched_pixel, vertex as container_vertex  # noqa: E402
from startup_compute_request_check import DATA, STATE_B, prepare  # noqa: E402
from startup_compute_translate_check import state_a  # noqa: E402
from startup_image_3d_check import root_blob as graphics_const_root_blob  # noqa: E402

TARGET_MASKS = (0x8002, 0x8003)  # PERSP_SAMPLE allocated (v0..v1), PERSP_CENTER (v2..v3), POS_FIXED_PT (v4)
DECOY_MASKS = (0x8002, 0x8002)   # PERSP_CENTER (v0..v1), POS_FIXED_PT (v2)
COMPUTE_GROUP = (2, 3, 4)
COMPUTE_RSRC2 = C.USER7 | C.TGID_X | C.TGID_Y | C.TGID_Z | C.tidig(3)


def fullscreen_vs_words() -> bytes:
    vop1 = lambda op, dst, src: 0x7E000000 | dst << 17 | op << 9 | src
    vop2 = lambda op, dst, src0, vsrc1: op << 25 | dst << 17 | vsrc1 << 9 | src0
    words = [vop2(0x1A, 1, 129, 0), vop2(0x1B, 1, 130, 1), vop2(0x1B, 2, 130, 0)]
    for v in (1, 2):
        words.extend([vop1(6, v, 256 + v), vop2(8, v, 244, v), vop2(3, v, 243, v)])
    words.extend([vop1(1, 3, 128), vop1(1, 4, 242), 0xF80008CF, 0x04030201, 0xBF810000])
    return struct.pack(f"<{len(words)}I", *words)


VS_WORDS = fullscreen_vs_words()


def stream_vertex_container() -> bytes:
    words = list(struct.unpack(f"<{len(VS_WORDS) // 4}I", VS_WORDS))
    return container_vertex(program=words)


def stream_pixel_container(green: bool) -> bytes:
    vop1 = lambda op, dst, src: 0x7E000000 | dst << 17 | op << 9 | src
    prefix = [vop1(1, 0, 128), vop1(1, 1, 242)]
    channels = 0x01000100 if green else 0x01000001
    return patched_pixel(where=len(prefix), prefix=prefix, export=(0xF800180F, channels))


def ps_words(reg: int) -> bytes:
    return struct.pack("<3I", 0xF800180F, reg * 0x01010101, 0xBF810000)


def compile_graphics_table_root(sdk: Path, work: Path) -> bytes:
    hlsl = work / "graphics_table_root.hlsl"
    out = work / "graphics_table_root.bin"
    hlsl.write_text('#define RS "DescriptorTable(SRV(t0))"\n', encoding="ascii")
    subprocess.run([sdk / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS", "-Fo", out, hlsl], check=True)
    return out.read_bytes()


def describe_psin(key: int, masks: tuple[int, int]) -> str:
    return f"psin1 {key:016x} ena=0x{masks[0]:04x} addr=0x{masks[1]:04x}\n"


def put_graphics_request(cache: Path, vs: bytes, ps: bytes, root: bytes,
                         masks: tuple[int, int] | None = None,
                         psin_text: str | None = None,
                         override_key: int | None = None) -> int:
    key = G.pipeline_key(vs, ps, root, ps_inputs=masks) if override_key is None else override_key
    stem = cache / f"req_{key:016x}"
    for ext, blob in ((".b", ps), (".root", root), (".a", vs)):
        stem.with_suffix(ext).write_bytes(blob)
    text = psin_text if psin_text is not None else (describe_psin(key, masks) if masks is not None else None)
    if text is not None:
        stem.with_suffix(".psin").write_bytes(text.encode("ascii"))
    return key


def compute_program_words() -> bytes:
    # Same arithmetic and buffer load/store body as Case A (2x3x4 group, 3x2x2 groups),
    # reading the placed input SRV from table 1 byte 64 and the placed output UAV from table 1 byte 32.
    loads = struct.pack(
        "<6I",
        0xBE8A0306,  # s_mov_b32 s10, s6
        0xBE8B0380,  # s_mov_b32 s11, 0
        0xF4080305, 0xFA000040,  # s_load_dwordx4 s[12:15], s[10:11], 0x40
        0xF4080405, 0xFA000020,  # s_load_dwordx4 s[16:19], s[10:11], 0x20
    )
    return loads + C.WORDS_A[16:]


def compile_compute_root(sdk: Path, work: Path) -> bytes:
    hlsl = work / "compute_root.hlsl"
    out = work / "compute_root.bin"
    hlsl.write_text(
        '#define RS "RootConstants(num32BitConstants=2, b0), '
        'DescriptorTable(SRV(t0, numDescriptors=4), UAV(u0, numDescriptors=4, offset=0))"\n',
        encoding="ascii",
    )
    subprocess.run([sdk / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS", "-Fo", out, hlsl], check=True)
    return out.read_bytes()


def put_compute_request(cache: Path, code: bytes, root: bytes, sa: bytes,
                        override_key: int | None = None) -> int:
    desc = struct.pack("<12Q", 0, 0, len(sa), 0, len(STATE_B), 0, len(code), 0, len(DATA), 0, 0, 11)
    key = G.compute_key(code, DATA, root, desc, sa, STATE_B) if override_key is None else override_key
    for ext, part in ((".desc", desc), (".sa", sa), (".sb", STATE_B), (".data", DATA), (".root", root), (".c67", code)):
        (cache / f"req_{key:016x}{ext}").write_bytes(part)
    return key


def check_controls(cli: Path, sdk: Path) -> int:
    repo = HERE.parent
    reader_text = (repo / "src" / "shader_compute_request.cpp").read_text(encoding="utf-8")
    runtime_text = (repo / "src" / "startup_stubs" / "d3d12_draw.inl").read_text(encoding="utf-8")
    reader_revs = re.findall(r'constexpr char revision\[\] = "(x2w-startup-shader-\d+(?:cs)?)";', reader_text)
    runtime_gfx = re.search(r'constexpr char revision\[\] = "(x2w-startup-shader-\d+)";', runtime_text)
    runtime_cs = re.search(r'constexpr char compute_revision\[\] = "(x2w-startup-shader-\d+cs)";', runtime_text)
    assert runtime_gfx and runtime_cs and reader_revs == [runtime_gfx.group(1), runtime_cs.group(1)], (
        reader_revs, runtime_gfx and runtime_gfx.group(1), runtime_cs and runtime_cs.group(1)
    )

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        frontend = work / "frontend"
        g_root = compile_graphics_table_root(sdk, work)
        g_const_root = graphics_const_root_blob()
        c_root = compile_compute_root(sdk, work)
        c_code = compute_program_words()
        c_sa = state_a(COMPUTE_GROUP, COMPUTE_RSRC2)

        # 1. Valid graphics (.psin) and compute (.c67) startup requests publish runtime cache entries in place.
        cache = work / "cache"
        cache.mkdir()
        target_key = put_graphics_request(cache, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        decoy_key = put_graphics_request(cache, VS_WORDS, ps_words(2), g_const_root, DECOY_MASKS)
        cs_key = put_compute_request(cache, c_code, c_root, c_sa)

        g_run = subprocess.run(
            [cli, sdk, frontend, "--jobs", "2", "gfx1013-container-pair", "--requests", cache, cache],
            capture_output=True, text=True,
        )
        assert g_run.returncode == 0, (g_run.stdout, g_run.stderr)
        assert "requests=2 compiled=2 refused=0" in g_run.stdout, g_run.stdout

        c_run = subprocess.run(
            [cli, sdk, frontend, "gfx1013-direct-cs@0", "--requests", cache, cache],
            capture_output=True, text=True,
        )
        assert c_run.returncode == 0, (c_run.stdout, c_run.stderr)
        assert "requests=1 compiled=1 refused=0" in c_run.stdout, c_run.stdout

        for key, masks, expected_ud in (
            (target_key, TARGET_MASKS, "x2w-manifest 1\nud 1\nudsrc 0 rect\n"),
            (decoy_key, DECOY_MASKS, "x2w-manifest 1\nud 1\nudsrc 0 rootconst 1 0\n"),
        ):
            vs_cso = cache / f"{key:016x}.vs.cso"
            ps_cso = cache / f"{key:016x}.ps.cso"
            psdep = cache / f"{key:016x}.psdep"
            manifest = cache / f"{key:016x}.txt"
            assert vs_cso.read_bytes().startswith(b"DXBC")
            assert ps_cso.read_bytes().startswith(b"DXBC")
            assert psdep.read_text(encoding="ascii") == describe_psin(key, masks)
            assert manifest.read_text(encoding="ascii").startswith(expected_ud), manifest.read_text(encoding="ascii")
            assert manifest.stat().st_mtime_ns >= max(vs_cso.stat().st_mtime_ns, ps_cso.stat().st_mtime_ns)
            assert not (cache / f"req_{key:016x}.vs.cso").exists()
            assert not (cache / f"req_{key:016x}.ps.cso").exists()

        cs_cso = cache / f"{cs_key:016x}.cs.cso"
        cs_manifest = cache / f"{cs_key:016x}.txt"
        assert cs_cso.read_bytes().startswith(b"DXBC")
        assert cs_manifest.stat().st_mtime_ns >= cs_cso.stat().st_mtime_ns
        assert not (cache / f"req_{cs_key:016x}.cso").exists()
        assert cs_manifest.read_text(encoding="ascii") == (
            "x2w-manifest 2\n"
            "stage cs\n"
            "group 2 3 4\n"
            "ud 2\n"
            "udsrc 0 rootconst 0 0\n"
            "udsrc 1 rootconst 0 1\n"
            "srv 0 buf table 1 64\n"
            "uav 0 table 1 32\n"
        )
        assert G.pending_keys(str(cache), skip_failed=False) == [], "published entries not recognised as complete"

        archive = work / "archive"
        archive.mkdir()
        ps = ps_words(4)
        legacy_graphics_key = G.fnv(b"x2w-startup-shader-28")
        for part in (VS_WORDS, ps, g_root, b"psin1" + struct.pack("<II", *TARGET_MASKS)):
            legacy_graphics_key = G.fnv(part, legacy_graphics_key)
        put_graphics_request(archive, VS_WORDS, ps, g_root, TARGET_MASKS, override_key=legacy_graphics_key)
        legacy_desc = struct.pack("<12Q", 0, 0, len(c_sa), 0, len(STATE_B), 0, len(c_code), 0, len(DATA), 0, 0, 11)
        legacy_compute_key = G.fnv(b"x2w-startup-shader-28cs")
        for part in (c_code, DATA, c_root, legacy_desc, c_sa, STATE_B):
            legacy_compute_key = G.fnv(part, G.fnv(struct.pack("<Q", len(part)), legacy_compute_key))
        put_compute_request(archive, c_code, c_root, c_sa, override_key=legacy_compute_key)
        for profile, key, current_key in (
            ("gfx1013-container-pair", legacy_graphics_key, target_key),
            ("gfx1013-direct-cs@0", legacy_compute_key, cs_key),
        ):
            result = subprocess.run(
                [cli, sdk, frontend, profile, "--requests", archive, archive],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, (profile, result.stdout, result.stderr)
            assert "requests=1 compiled=1 refused=0" in result.stdout, result.stdout
            assert (archive / f"{key:016x}.txt").read_bytes() == (cache / f"{current_key:016x}.txt").read_bytes()

        # 2. Stale stage change on an already-published .psin entry refuses with diagnostic and withdraws <key>.txt.
        (cache / f"req_{target_key:016x}.b").write_bytes(ps_words(2))
        stale_run = subprocess.run(
            [cli, sdk, frontend, "gfx1013-container-pair", "--requests", cache, cache],
            capture_output=True, text=True,
        )
        assert stale_run.returncode != 0 and "stale pixel input description" in stale_run.stderr, stale_run.stderr
        assert not (cache / f"{target_key:016x}.txt").exists(), "stale .psin kept a loadable manifest"
        assert (cache / f"{decoy_key:016x}.txt").is_file(), "unrelated valid entry was disturbed"

        # 3. Fresh-directory negative controls: mismatched keys and malformed/stale/contradictory .psin files.
        def expect_graphics_refusal(label: str, setup, expected_stderr: str) -> None:
            case_dir = work / label
            case_dir.mkdir()
            bad_key = setup(case_dir)
            res = subprocess.run(
                [cli, sdk, frontend, "gfx1013-container-pair", "--requests", case_dir, case_dir],
                capture_output=True, text=True,
            )
            assert res.returncode != 0 and expected_stderr in res.stderr, (label, res.returncode, res.stderr)
            assert not list(case_dir.glob("*.txt")), (label, list(case_dir.iterdir()))
            assert not (case_dir / f"{bad_key:016x}.vs.cso").exists(), label
            assert not (case_dir / f"{bad_key:016x}.ps.cso").exists(), label

        expect_graphics_refusal(
            "graphics-key-mismatch",
            lambda d: put_graphics_request(d, VS_WORDS, ps_words(0), g_root, override_key=0x0123456789ABCDEF),
            "graphics request key mismatch",
        )
        expect_graphics_refusal(
            "psin-other-pipeline",
            lambda d: put_graphics_request(
                d, VS_WORDS, ps_words(4), g_root, TARGET_MASKS, psin_text=describe_psin(decoy_key, DECOY_MASKS)
            ),
            "pixel input description names another pipeline",
        )
        expect_graphics_refusal(
            "psin-missing-addr",
            lambda d: (
                lambda k: put_graphics_request(d, VS_WORDS, ps_words(4), g_root, psin_text=f"psin1 {k:016x} ena=0x8002\n", override_key=k)
            )(G.pipeline_key(VS_WORDS, ps_words(4), g_root, ps_inputs=TARGET_MASKS)),
            "pixel input address mask missing from the stream description",
        )
        expect_graphics_refusal(
            "psin-missing-ena",
            lambda d: (
                lambda k: put_graphics_request(d, VS_WORDS, ps_words(4), g_root, psin_text=f"psin1 {k:016x} addr=0x8003\n", override_key=k)
            )(G.pipeline_key(VS_WORDS, ps_words(4), g_root, ps_inputs=TARGET_MASKS)),
            "pixel input enable mask missing from the stream description",
        )
        expect_graphics_refusal(
            "psin-contradictory",
            lambda d: (
                lambda k: put_graphics_request(
                    d, VS_WORDS, ps_words(4), g_root, psin_text=f"psin1 {k:016x} ena=0x8003 addr=0x8002\n", override_key=k
                )
            )(G.pipeline_key(VS_WORDS, ps_words(4), g_root, ps_inputs=(0x8003, 0x8002))),
            "contradictory pixel input enable/address masks",
        )
        expect_graphics_refusal(
            "psin-malformed",
            lambda d: (
                lambda k: put_graphics_request(
                    d, VS_WORDS, ps_words(4), g_root, psin_text=f"psin1 {k:016x} ena=8002 addr=8003\n", override_key=k
                )
            )(G.pipeline_key(VS_WORDS, ps_words(4), g_root, ps_inputs=TARGET_MASKS)),
            "invalid pixel input description",
        )
        expect_graphics_refusal(
            "psin-allocated-disabled",
            lambda d: put_graphics_request(d, VS_WORDS, ps_words(2), g_root, (0x0002, 0x8002)),
            "unsupported pixel entry VGPR 2",
        )

        bad_cs_dir = work / "compute-key-mismatch"
        bad_cs_dir.mkdir()
        bad_cs_key = put_compute_request(bad_cs_dir, c_code, c_root, c_sa, override_key=cs_key ^ 1)
        bad_cs_run = subprocess.run(
            [cli, sdk, frontend, "gfx1013-direct-cs@0", "--requests", bad_cs_dir, bad_cs_dir],
            capture_output=True, text=True,
        )
        assert bad_cs_run.returncode != 0 and "compute request key mismatch" in bad_cs_run.stderr, bad_cs_run.stderr
        assert not (bad_cs_dir / f"{bad_cs_key:016x}.txt").exists()
        assert not (bad_cs_dir / f"{bad_cs_key:016x}.cs.cso").exists()

    print("native_startup_request_cache controls: current and archived graphics (.psin) + compute (.c67) cache entries and 8 refusal controls passed")
    return 0


def check_e2e(cli: Path, consumer: Path, offscreen: Path, sdk: Path, backend: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        frontend = work / "frontend"

        # Part 1: .psin VS/PS requests compiled by compile_dxil --requests and executed on GPU.
        psin_cache = work / "psin_cache"
        psin_cache.mkdir()
        g_root = compile_graphics_table_root(sdk, work)
        target_key = put_graphics_request(psin_cache, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        decoy_key = put_graphics_request(psin_cache, VS_WORDS, ps_words(2), g_root, DECOY_MASKS)
        wrong_key = put_graphics_request(psin_cache, VS_WORDS, ps_words(2), g_root, (0x8003, 0x8003))
        subprocess.run(
            [cli, sdk, frontend, "--jobs", "3", "gfx1013-container-pair", "--requests", psin_cache, psin_cache],
            check=True, capture_output=True, text=True,
        )
        for label, key, should_match in (("target-v4", target_key, True), ("decoy-v2", decoy_key, True), ("wrong-v2", wrong_key, False)):
            vs_cso = psin_cache / f"{key:016x}.vs.cso"
            ps_cso = psin_cache / f"{key:016x}.ps.cso"
            draw_res = subprocess.run(
                [offscreen, backend, "--startup-packed-position", vs_cso, ps_cso],
                capture_output=True, text=True,
            )
            if draw_res.returncode == 77:
                print(draw_res.stdout + draw_res.stderr, end="")
                return 77
            if should_match:
                assert draw_res.returncode == 0 and "all 91 packed positions matched" in draw_res.stdout, (
                    label, draw_res.stdout, draw_res.stderr
                )
            else:
                assert draw_res.returncode != 0, f"{label} negative control unexpectedly matched"

        # Part 2: Runtime-published graphics request -> compile_dxil --requests -> runtime cache loader draw.
        draw_dir = work / "runtime_draw"
        draw_cache = draw_dir / "cache"
        draw_cache.mkdir(parents=True)
        vs_path, ps_path, root_path, pixels_path = (draw_dir / n for n in ("vs", "ps", "root", "pixels"))
        vs_path.write_bytes(stream_vertex_container())
        draw_env = dict(
            os.environ,
            X2W_D3D12_PERMISSIVE="1",
            X2W_D3D12_LIST_NOOP="1",
            X2W_D3D12_RENDER="1",
            X2W_D3D12_DEBUG="1",
            X2W_SHADER_CACHE=str(draw_cache),
        )
        drawn_images = []
        for green in (False, True):
            ps_path.write_bytes(stream_pixel_container(green))
            subprocess.run([consumer, vs_path, ps_path, root_path], env=draw_env, check=True, capture_output=True, text=True)
            subprocess.run(
                [cli, sdk, frontend, "gfx1013-container-pair", "--requests", draw_cache, draw_cache],
                check=True, capture_output=True, text=True,
            )
            res = subprocess.run([consumer, vs_path, ps_path, root_path, backend, pixels_path],
                                 env=draw_env, capture_output=True, text=True)
            if res.returncode == 77:
                print(res.stdout + res.stderr, end="")
                return 77
            assert res.returncode == 0 and "-> drawn;" in res.stderr, (res.returncode, res.stderr[-2000:])
            debug_errors = [ln for ln in res.stderr.splitlines() if "x2w-d3d12-debug " in ln and (" sev=0 " in ln or " sev=1 " in ln)]
            assert not debug_errors, debug_errors[0]
            color = bytes((0, 255, 0, 255) if green else (255, 0, 0, 255))
            clear = bytes((0, 0, 255, 255))
            expected_pixels = (color * 4 + clear * 4) * 4
            actual_pixels = pixels_path.read_bytes()
            assert actual_pixels == expected_pixels, (actual_pixels.hex(), expected_pixels.hex())
            drawn_images.append(actual_pixels)
        assert sum(drawn_images[0][i:i + 4] != drawn_images[1][i:i + 4] for i in range(0, 128, 4)) == 16

        # Part 3: Runtime-published .c67 compute request -> compile_dxil --requests -> runtime cache loader dispatch.
        cs_dir = work / "runtime_compute"
        cs_cache, cs_env = prepare(cs_dir)
        cs_env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1" if backend == "hardware" else "0")
        if backend == "hardware":
            cs_env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
        else:
            cs_env.pop("X2W_TEST_DISPATCH_HARDWARE", None)
        c_root = compile_compute_root(sdk, work)
        (cs_dir / "compute_root").write_bytes(c_root)
        cs_env["X2W_TEST_COMPUTE_ROOT"] = str(cs_dir / "compute_root")
        c_code = compute_program_words()
        c_sa = state_a(COMPUTE_GROUP, COMPUTE_RSRC2)
        (cs_dir / "program").write_bytes(c_code)
        (cs_dir / "data").write_bytes(DATA)
        (cs_dir / "state").write_bytes(c_sa)
        out_bin = cs_dir / "out.bin"

        def run_dispatch() -> tuple[list[int], subprocess.CompletedProcess[str]]:
            out_bin.unlink(missing_ok=True)
            args = [consumer, cs_dir / "vertex", cs_dir / "pixel", cs_dir / "root", "slot67",
                    cs_dir / "program", cs_dir / "data", cs_dir / "state", "dispatch-heap"]
            res = subprocess.run(args, env=dict(cs_env, X2W_TEST_DISPATCH_OUT=str(out_bin)), capture_output=True, text=True)
            if res.returncode == 77:
                print(res.stdout + res.stderr, end="")
                sys.exit(77)
            assert res.returncode == 0, (res.returncode, res.stderr[-2500:])
            return list(struct.unpack(f"<{C.WORDS}I", out_bin.read_bytes())), res

        before_words, before_res = run_dispatch()
        assert before_words == [C.SENTINEL] * C.WORDS, "dispatch ran before compile_dxil published cache entry"
        assert "Dispatch(3, 2, 2) refused: no translation" in before_res.stderr, before_res.stderr[-2000:]

        subprocess.run(
            [cli, sdk, frontend, "gfx1013-direct-cs@0", "--requests", cs_cache, cs_cache],
            check=True, capture_output=True, text=True,
        )
        assert G.pending_keys(str(cs_cache), skip_failed=False) == []

        after_words, after_res = run_dispatch()
        if backend == "hardware" or "Dispatch(3, 2, 2) translated:" in after_res.stderr:
            debug_errors = [
                ln for ln in after_res.stderr.splitlines() if "x2w-d3d12-debug " in ln and (" sev=0 " in ln or " sev=1 " in ln)
            ]
            assert not debug_errors, debug_errors[0]
            assert "Dispatch(3, 2, 2) translated:" in after_res.stderr, after_res.stderr[-2000:]
            assert after_words == C.expected("A"), "compute dispatch output mismatch"
        else:
            # Wave-4 WARP builds validate the manifest, load <key>.cs.cso, build tr.root, and stop at [WaveSize(32)] PSO creation.
            assert "not translated: pipeline creation failed" in after_res.stderr, after_res.stderr[-2000:]

    print(f"native_startup_request_cache e2e: PASS on {backend} (.psin 91-pixel draws, runtime graphics cache draw, runtime .c67 compute dispatch of {C.WORDS} words)")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "controls":
        return check_controls(Path(argv[1]), Path(argv[2]))
    if len(argv) == 6 and argv[0] == "e2e" and argv[5] in ("warp", "hardware"):
        return check_e2e(Path(argv[1]), Path(argv[2]), Path(argv[3]), Path(argv[4]), argv[5])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
