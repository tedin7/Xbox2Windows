"""Diagnostic .fail lifecycle and read-only --refusals summary checks for compile_dxil (#2935).

usage:
  native_compile_dxil_refusals_check.py <compile_dxil> <sdk>
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools" / "startup"))

import gfx2hlsl as G  # noqa: E402
import triage_run  # noqa: E402
from native_rtv_variant_check import FLOAT32, UINT16, UNORM8, pixel_stage, put_variant  # noqa: E402
from native_startup_request_cache_check import (  # noqa: E402
    COMPUTE_GROUP,
    COMPUTE_RSRC2,
    DECOY_MASKS,
    TARGET_MASKS,
    VS_WORDS,
    compile_compute_root,
    compile_graphics_table_root,
    compute_program_words,
    ps_words,
    put_compute_request,
    put_graphics_request,
    stream_vertex_container,
)
from startup_compute_translate_check import state_a  # noqa: E402


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        snap[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return snap


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    cli = Path(argv[0])
    sdk = Path(argv[1])

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        frontend = work / "frontend"
        g_root = compile_graphics_table_root(sdk, work)
        c_root = compile_compute_root(sdk, work)
        c_code = compute_program_words()
        c_sa = state_a(COMPUTE_GROUP, COMPUTE_RSRC2)

        # 1. Batch --requests writes <key>.fail on refusal and removes stale <key>.fail on subsequent success.
        cache = work / "batch_cache"
        cache.mkdir()
        valid_gfx = put_graphics_request(cache, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        bad_vgpr2 = put_graphics_request(cache, VS_WORDS, ps_words(2), g_root, (0x0002, 0x8002))
        bad_decode_gfx = put_graphics_request(
            cache, VS_WORDS, struct.pack("<2I", 0xFFFFFFFF, 0xBF810000), g_root, DECOY_MASKS
        )
        # Pre-seed a stale .fail beside a valid request to verify --requests deletes it upon success.
        stale_fail = cache / f"{valid_gfx:016x}.fail"
        stale_fail.write_text("stale failure\n", encoding="utf-8")

        g_res = subprocess.run(
            [cli, sdk, frontend, "--jobs", "2", "gfx1013-container-pair", "--requests", cache, cache],
            capture_output=True,
            text=True,
        )
        assert g_res.returncode != 0 and "requests=3 compiled=1 refused=2" in g_res.stdout, (
            g_res.returncode, g_res.stdout, g_res.stderr
        )
        assert (cache / f"{valid_gfx:016x}.txt").is_file()
        assert not stale_fail.exists(), "stale .fail survived successful compilation"
        for bad_key in (bad_vgpr2, bad_decode_gfx):
            fail_file = cache / f"{bad_key:016x}.fail"
            assert fail_file.is_file(), f"missing .fail for refused graphics request {bad_key:016x}"
            fail_text = fail_file.read_text(encoding="utf-8")
            assert fail_text.endswith("\n") and len(fail_text.splitlines()) == 1 and fail_text.strip(), repr(fail_text)
            assert not (cache / f"{bad_key:016x}.txt").exists()
            assert not (cache / f"{bad_key:016x}.vs.cso").exists()
            assert not (cache / f"{bad_key:016x}.ps.cso").exists()

        valid_cs = put_compute_request(cache, c_code, c_root, c_sa)
        bad_cs_decode = put_compute_request(
            cache, c_code[:-4] + struct.pack("<2I", 0xFFFFFFFF, 0xBF810000), c_root, c_sa
        )
        stale_cs_fail = cache / f"{valid_cs:016x}.fail"
        stale_cs_fail.write_text("stale compute failure\n", encoding="utf-8")

        c_res = subprocess.run(
            [cli, sdk, frontend, "--jobs", "2", "gfx1013-direct-cs@0", "--requests", cache, cache],
            capture_output=True,
            text=True,
        )
        assert c_res.returncode != 0 and "requests=2 compiled=1 refused=1" in c_res.stdout, (
            c_res.returncode, c_res.stdout, c_res.stderr
        )
        assert (cache / f"{valid_cs:016x}.txt").is_file()
        assert not stale_cs_fail.exists(), "stale compute .fail survived successful compilation"
        cs_fail = cache / f"{bad_cs_decode:016x}.fail"
        assert cs_fail.is_file()
        assert "unknown gfx1013 instruction format" in cs_fail.read_text(encoding="utf-8")
        assert not (cache / f"{bad_cs_decode:016x}.txt").exists()
        assert not (cache / f"{bad_cs_decode:016x}.cs.cso").exists()

        # Verify triage_run.py ingests and classifies the native .fail files cleanly.
        log_dir = work / "logs"
        log_dir.mkdir()
        (log_dir / "stderr.txt").write_text("", encoding="utf-8")
        triage_report = triage_run.triage_run(log_dir, shader_cache=cache)
        assert "`X2W_SHADER_CACHE` `.fail` shaders: `3`" in triage_report, triage_report
        assert "| 1 | `unsupported pixel entry VGPR` |" in triage_report, triage_report
        assert "| 1 | `undecodable word` |" in triage_report, triage_report
        assert "| 1 | `compute: undecodable word` |" in triage_report, triage_report

        # Mutate the valid graphics request into a key mismatch and re-run: old .txt must vanish and .fail appear.
        (cache / f"req_{valid_gfx:016x}.b").write_bytes(ps_words(2))
        stale_run = subprocess.run(
            [cli, sdk, frontend, "gfx1013-container-pair", "--requests", cache, cache],
            capture_output=True,
            text=True,
        )
        assert stale_run.returncode != 0
        assert not (cache / f"{valid_gfx:016x}.txt").exists(), "refused re-run kept stale manifest"
        assert "stale pixel input description" in (cache / f"{valid_gfx:016x}.fail").read_text(encoding="utf-8")

        # 2. Read-only --refusals summary across graphics (.a) and slot-67 compute (.c67) requests.
        req_dir = work / "refusals_dir"
        req_dir.mkdir()
        g_ok = put_graphics_request(req_dir, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        # Two distinct VGPR-disabled failures (VGPR 2 and VGPR 4) should normalize and group together.
        g_vgpr2 = put_graphics_request(req_dir, VS_WORDS, ps_words(2), g_root, (0x0002, 0x8002))
        g_vgpr4 = put_graphics_request(req_dir, VS_WORDS, ps_words(4), g_root, (0x0002, 0x8006))
        # Two undecodable graphics failures at different byte offsets (0 and 4) should group together.
        g_dec0 = put_graphics_request(req_dir, VS_WORDS, struct.pack("<2I", 0xFFFFFFFF, 0xBF810000), g_root, DECOY_MASKS)
        g_dec4 = put_graphics_request(
            req_dir, VS_WORDS, struct.pack("<3I", 0xBF800000, 0xFFFFFFFF, 0xBF810000), g_root, DECOY_MASKS
        )
        # Unsupported instruction with a distinct hex encoding (s_trap 0x1234 -> 0xbf921234).
        g_trap = put_graphics_request(
            req_dir, VS_WORDS, struct.pack("<3I", 0xBF921234, 0xF800180F, 0x04040404), g_root, TARGET_MASKS
        )

        c_ok = put_compute_request(req_dir, c_code, c_root, c_sa)
        c_dec = put_compute_request(
            req_dir, c_code[:-4] + struct.pack("<2I", 0xFFFFFFFF, 0xBF810000), c_root, c_sa
        )
        c_key_bad = put_compute_request(req_dir, c_code, c_root, c_sa, override_key=c_ok ^ 1)

        before_snap = snapshot_tree(req_dir)
        r1 = subprocess.run(
            [cli, "--refusals", req_dir, "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        r2 = subprocess.run(
            [cli, "--refusals", req_dir, "--jobs", "3", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        r_text = subprocess.run(
            [cli, "--refusals", req_dir, "--jobs", "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        after_snap = snapshot_tree(req_dir)
        assert before_snap == after_snap, "read-only --refusals modified the request directory"

        assert r1.stdout == r2.stdout, (r1.stdout, r2.stdout)
        parsed = json.loads(r1.stdout)
        assert parsed["graphics"]["examined"] == 6, parsed
        assert parsed["graphics"]["no_refusal"] == 1, parsed
        assert parsed["graphics"]["refused"] == 5, parsed
        gfx_reasons = {row["reason"]: row["count"] for row in parsed["graphics"]["reasons"]}
        assert gfx_reasons.get("unsupported pixel entry VGPR N") == 2, gfx_reasons
        assert gfx_reasons.get("unknown gfx1013 instruction format at byte N") == 2, gfx_reasons
        assert len(gfx_reasons) == 3 and sum(gfx_reasons.values()) == 5, gfx_reasons

        assert parsed["compute"]["examined"] == 3, parsed
        assert parsed["compute"]["no_refusal"] == 1, parsed
        assert parsed["compute"]["refused"] == 2, parsed
        cs_reasons = {row["reason"]: row["count"] for row in parsed["compute"]["reasons"]}
        assert cs_reasons == {
            "compute request key mismatch": 1,
            "unknown gfx1013 instruction format at byte N": 1,
        }, cs_reasons

        # Negative control: hex words, PCs, and request keys must not appear in the census output.
        for forbidden in (
            f"{g_ok:016x}",
            f"{g_vgpr2:016x}",
            f"{g_vgpr4:016x}",
            f"{g_dec0:016x}",
            f"{g_dec4:016x}",
            f"{g_trap:016x}",
            f"{c_dec:016x}",
            f"{c_key_bad:016x}",
            "bf921234",
            "1234",
            "+0x",
        ):
            assert forbidden not in r1.stdout.lower() and forbidden not in r_text.stdout.lower(), (
                forbidden, r1.stdout, r_text.stdout
            )

        for expected_line in (
            "Graphics requests",
            "Requests examined: 6",
            "No refusal recorded (not compiled): 1",
            "Refused: 5",
            "2\tunsupported pixel entry VGPR N",
            "2\tunknown gfx1013 instruction format at byte N",
            "Compute requests (device slot 67)",
            "Compute requests examined: 3",
            "No refusal recorded (not compiled): 1",
            "Refused: 2",
            "1\tcompute request key mismatch",
        ):
            assert expected_line in r_text.stdout, (expected_line, r_text.stdout)

        # Empty directory is valid and reports 0 examined.
        empty_dir = work / "empty"
        empty_dir.mkdir()
        r_empty = subprocess.run([cli, "--refusals", empty_dir, "--json"], capture_output=True, text=True, check=True)
        empty_json = json.loads(r_empty.stdout)
        assert empty_json == {
            "graphics": {"examined": 0, "no_refusal": 0, "refused": 0, "reasons": []},
            "compute": {"examined": 0, "no_refusal": 0, "refused": 0, "reasons": []},
        }, empty_json

        # RTV variant requests (.rtv): valid + refused variants, read-only guarantee, and same-key dedup.
        rtv_dir = work / "rtv_refusals"
        rtv_dir.mkdir()
        rtv_base = put_graphics_request(rtv_dir, stream_vertex_container(), pixel_stage(), g_root)
        rtv_u16, _ = put_variant(rtv_dir, rtv_base, (UINT16, UNORM8))
        rtv_f32, _ = put_variant(rtv_dir, rtv_base, (FLOAT32,))
        rtv_forged, _ = put_variant(rtv_dir, rtv_base, (UINT16,), name_key=0x1111222233334444)
        rtv_missing, _ = put_variant(rtv_dir, 0x9999888877776666, (UINT16,))
        rtv_bad_fmt, _ = put_variant(rtv_dir, rtv_base, (7,))

        rtv_before = snapshot_tree(rtv_dir)
        r_rtv1 = subprocess.run([cli, "--refusals", rtv_dir, "--json"], capture_output=True, text=True, check=True)
        r_rtv2 = subprocess.run(
            [cli, "--refusals", rtv_dir, "--jobs", "2", "--json"], capture_output=True, text=True, check=True
        )
        assert snapshot_tree(rtv_dir) == rtv_before, "--refusals modified RTV request directory"
        assert r_rtv1.stdout == r_rtv2.stdout, (r_rtv1.stdout, r_rtv2.stdout)
        rtv_json = json.loads(r_rtv1.stdout)
        assert rtv_json["graphics"]["examined"] == 6, rtv_json
        assert rtv_json["graphics"]["no_refusal"] == 3, rtv_json
        assert rtv_json["graphics"]["refused"] == 3, rtv_json
        for variant_key in (rtv_base, rtv_u16, rtv_f32, rtv_forged, rtv_missing, rtv_bad_fmt):
            assert f"{variant_key:016x}" not in r_rtv1.stdout.lower(), (variant_key, r_rtv1.stdout)

        dedupe_dir = work / "rtv_dedupe"
        dedupe_dir.mkdir()
        dedupe_key = put_graphics_request(dedupe_dir, stream_vertex_container(), pixel_stage(), g_root)
        put_variant(dedupe_dir, dedupe_key, (UINT16,), name_key=dedupe_key)
        r_dedupe = subprocess.run([cli, "--refusals", dedupe_dir, "--json"], capture_output=True, text=True, check=True)
        dedupe_json = json.loads(r_dedupe.stdout)
        assert dedupe_json["graphics"] == {
            "examined": 1,
            "no_refusal": 0,
            "refused": 1,
            "reasons": [{"reason": "invalid RTV variant request", "count": 1}],
        }, dedupe_json

        # 3. Negative controls for invalid --refusals CLI invocations.
        bad_file = work / "not_a_dir.txt"
        bad_file.write_text("x", encoding="utf-8")
        for bad_args in (
            ["--refusals"],
            ["--refusals", work / "missing_dir"],
            ["--refusals", bad_file],
            ["--refusals", req_dir, "--jobs", "0"],
            ["--refusals", req_dir, "--jobs", "65"],
            ["--refusals", req_dir, "--bogus"],
        ):
            res = subprocess.run([cli, *bad_args], capture_output=True, text=True)
            assert res.returncode != 0 and res.stderr.strip(), (bad_args, res.returncode, res.stdout, res.stderr)

    print("native_compile_dxil_refusals checks: PASS (.fail lifecycle, triage_run classification, and read-only --refusals summary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
