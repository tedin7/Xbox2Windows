"""Continuous and one-shot --watch mode checks for compile_dxil (#2935).

usage:
  native_compile_dxil_watch_check.py <compile_dxil> <sdk>
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools" / "startup"))

import gfx2hlsl as G  # noqa: E402
from native_rtv_variant_check import FLOAT32, UINT16, UNORM8, put_variant  # noqa: E402
from native_startup_request_cache_check import (  # noqa: E402
    COMPUTE_GROUP,
    COMPUTE_RSRC2,
    DECOY_MASKS,
    TARGET_MASKS,
    VS_WORDS,
    compile_compute_root,
    compile_graphics_table_root,
    compute_program_words,
    describe_psin,
    ps_words,
    put_compute_request,
    put_graphics_request,
)
from startup_compute_translate_check import state_a  # noqa: E402


def touch(path: Path, data: bytes | None = None, mtime: float | None = None) -> None:
    if data is not None:
        path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def run_once(cli: Path, sdk: Path, frontend: Path, req_dir: Path, jobs: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli, "--watch", sdk, frontend, req_dir, "--jobs", str(jobs), "--once"],
        capture_output=True,
        text=True,
        timeout=60,
    )


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

        # Part 1: --once pending, skip, resume, .psin/.psdep recheck, and parallel equivalence.
        d1 = work / "once_serial"
        d1.mkdir()
        # Incomplete marker without .a / .c67 must be ignored.
        (d1 / "req_00000000000000aa.b").write_bytes(ps_words(4))
        (d1 / "req_00000000000000aa.root").write_bytes(g_root)
        (d1 / "req_00000000000000bb.data").write_bytes(b"\0" * 16)

        g_key = put_graphics_request(d1, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        g_bad_key = put_graphics_request(d1, VS_WORDS, ps_words(2), g_root, (0x0002, 0x8002))
        c_key = put_compute_request(d1, c_code, c_root, c_sa)

        r1 = run_once(cli, sdk, frontend, d1, jobs=1)
        assert r1.returncode == 0, (r1.returncode, r1.stdout, r1.stderr)
        assert f"{g_key:016x}: compiled graphics" in r1.stdout, r1.stdout
        assert f"{c_key:016x}: compiled compute" in r1.stdout, r1.stdout
        assert f"{g_bad_key:016x}: refused:" in r1.stdout, r1.stdout
        assert "00000000000000aa" not in r1.stdout and "00000000000000bb" not in r1.stdout, r1.stdout
        assert not (d1 / "00000000000000aa.fail").exists()
        assert (d1 / f"{g_key:016x}.txt").is_file()
        assert (d1 / f"{g_key:016x}.psdep").is_file()
        assert (d1 / f"{c_key:016x}.txt").is_file()
        assert (d1 / f"{g_bad_key:016x}.fail").is_file()

        # Re-running --once retries the .fail (first-pass semantics of --once) and skips complete entries.
        txt_mtime = (d1 / f"{g_key:016x}.txt").stat().st_mtime_ns
        cs_mtime = (d1 / f"{c_key:016x}.txt").stat().st_mtime_ns
        r2 = run_once(cli, sdk, frontend, d1, jobs=1)
        assert r2.returncode == 0, (r2.returncode, r2.stdout, r2.stderr)
        assert f"{g_key:016x}:" not in r2.stdout and f"{c_key:016x}:" not in r2.stdout, r2.stdout
        assert f"{g_bad_key:016x}: refused:" in r2.stdout, r2.stdout
        assert (d1 / f"{g_key:016x}.txt").stat().st_mtime_ns == txt_mtime
        assert (d1 / f"{c_key:016x}.txt").stat().st_mtime_ns == cs_mtime

        # Seed outputs that satisfy the old watcher's completeness check.
        cs_manifest = d1 / f"{c_key:016x}.txt"
        cs_cso = d1 / f"{c_key:016x}.cs.cso"
        cs_hlsl = d1 / f"{c_key:016x}.cs.hlsl"
        current_manifest = cs_manifest.read_bytes()
        old_manifest = b"".join(line for line in current_manifest.splitlines(keepends=True)
                                if not line.startswith(b"native "))
        assert current_manifest.splitlines().count(b"native 1") == 1
        assert not cs_hlsl.exists()
        dummy_cso = b"DXBC" + b"\0" * 20 + struct.pack("<III", 36, 1, 0)
        for stale_manifest in (old_manifest, old_manifest + b"native 0\n"):
            cs_cso.write_bytes(dummy_cso)
            cs_manifest.write_bytes(stale_manifest)
            refreshed = run_once(cli, sdk, frontend, d1)
            assert refreshed.returncode == 0, (refreshed.stdout, refreshed.stderr)
            assert f"{c_key:016x}: compiled compute" in refreshed.stdout, refreshed.stdout
            assert cs_manifest.read_bytes() == current_manifest
            assert cs_cso.read_bytes().startswith(b"DXBC") and cs_cso.read_bytes() != dummy_cso
            assert not cs_hlsl.exists()

        cs_hlsl.write_bytes(b"// Python output\n")
        cs_cso.write_bytes(dummy_cso)
        cs_manifest.write_bytes(old_manifest)
        python_outputs = (cs_hlsl, cs_cso, cs_manifest)
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in python_outputs]
        skipped = run_once(cli, sdk, frontend, d1)
        assert skipped.returncode == 0, (skipped.stdout, skipped.stderr)
        assert f"{c_key:016x}:" not in skipped.stdout, skipped.stdout
        assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in python_outputs] == before

        # A Python output with incomplete bytecode is still pending. The native rebuild removes its sidecar, so
        # the stand-in no longer reads the new unmarked-looking pair as a Python output.
        cs_cso.write_bytes(dummy_cso[:16])
        rebuilt = run_once(cli, sdk, frontend, d1)
        assert rebuilt.returncode == 0, (rebuilt.stdout, rebuilt.stderr)
        assert f"{c_key:016x}: compiled compute" in rebuilt.stdout, rebuilt.stdout
        assert cs_manifest.read_bytes() == current_manifest
        assert not cs_hlsl.exists()

        # Corrupting/truncating a stage .cso triggers resume and restores valid DXBC.
        vs_cso = d1 / f"{g_key:016x}.vs.cso"
        orig_vs = vs_cso.read_bytes()
        vs_cso.write_bytes(orig_vs[:16])
        r3 = run_once(cli, sdk, frontend, d1, jobs=1)
        assert r3.returncode == 0 and f"{g_key:016x}: compiled graphics" in r3.stdout, (r3.stdout, r3.stderr)
        assert vs_cso.read_bytes() == orig_vs

        # Updating .psin mtime newer than .txt (with mismatched .b) rechecks, withdraws .txt, and writes .fail.
        (d1 / f"req_{g_key:016x}.b").write_bytes(ps_words(2))
        future = time.time() + 2.0
        touch(d1 / f"req_{g_key:016x}.psin", mtime=future)
        r4 = run_once(cli, sdk, frontend, d1, jobs=1)
        assert r4.returncode == 0 and f"{g_key:016x}: refused:" in r4.stdout, (r4.stdout, r4.stderr)
        assert not (d1 / f"{g_key:016x}.txt").exists()
        assert (d1 / f"{g_key:016x}.fail").is_file()

        # Removing .psin when .psdep exists ('lost') forces recheck and withdraws .txt.
        d_lost = work / "lost_psin"
        d_lost.mkdir()
        lost_key = put_graphics_request(d_lost, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        assert run_once(cli, sdk, frontend, d_lost, jobs=1).returncode == 0
        assert (d_lost / f"{lost_key:016x}.txt").is_file() and (d_lost / f"{lost_key:016x}.psdep").is_file()
        (d_lost / f"req_{lost_key:016x}.psin").unlink()
        r_lost = run_once(cli, sdk, frontend, d_lost, jobs=1)
        assert r_lost.returncode == 0 and f"{lost_key:016x}: refused:" in r_lost.stdout, (r_lost.stdout, r_lost.stderr)
        assert not (d_lost / f"{lost_key:016x}.txt").exists()
        assert (d_lost / f"{lost_key:016x}.fail").is_file()

        # Parallel --once --jobs 2 matches serial --once --jobs 1.
        d_serial = work / "cmp_serial"
        d_parallel = work / "cmp_parallel"
        for d in (d_serial, d_parallel):
            d.mkdir()
            put_graphics_request(d, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
            put_graphics_request(d, VS_WORDS, ps_words(2), g_root, DECOY_MASKS)
            put_graphics_request(d, VS_WORDS, ps_words(2), g_root, (0x0002, 0x8002))
            put_compute_request(d, c_code, c_root, c_sa)
            put_compute_request(d, c_code, c_root, c_sa, override_key=c_key ^ 1)
        rs = run_once(cli, sdk, frontend, d_serial, jobs=1)
        rp = run_once(cli, sdk, frontend, d_parallel, jobs=2)
        assert rs.returncode == 0 and rp.returncode == 0
        assert rs.stdout == rp.stdout, (rs.stdout, rp.stdout)
        assert G.pending_keys(str(d_parallel), skip_failed=True) == []

        # RTV variants in --once: compile base + variants, publish rtv1 manifest + .psdep, and recheck on base .psin update.
        d_rtv = work / "once_rtv"
        d_rtv.mkdir()
        rtv_base = put_graphics_request(d_rtv, VS_WORDS, ps_words(4), g_root, TARGET_MASKS)
        rtv_u16, rtv_u16_line = put_variant(d_rtv, rtv_base, (UINT16, UNORM8))
        rtv_f32, rtv_f32_line = put_variant(d_rtv, rtv_base, (FLOAT32,))
        rtv_bad, _ = put_variant(d_rtv, rtv_base, (7,))
        r_rtv = run_once(cli, sdk, frontend, d_rtv, jobs=2)
        assert r_rtv.returncode == 0, (r_rtv.returncode, r_rtv.stdout, r_rtv.stderr)
        for vkey, vline in ((rtv_u16, rtv_u16_line), (rtv_f32, rtv_f32_line)):
            assert f"{vkey:016x}: compiled graphics" in r_rtv.stdout, r_rtv.stdout
            vtxt = (d_rtv / f"{vkey:016x}.txt").read_text(encoding="ascii")
            assert vtxt.splitlines()[-1] == vline, vtxt
            assert (d_rtv / f"{vkey:016x}.psdep").is_file()
            assert not (d_rtv / f"{vkey:016x}.fail").exists()
        assert f"{rtv_bad:016x}: refused:" in r_rtv.stdout and (d_rtv / f"{rtv_bad:016x}.fail").is_file()
        assert G.pending_keys(str(d_rtv), skip_failed=True) == []

        # Updating the base request's .psin (with mismatched .b) rechecks both the base and its .rtv variants.
        (d_rtv / f"req_{rtv_base:016x}.b").write_bytes(ps_words(2))
        touch(d_rtv / f"req_{rtv_base:016x}.psin", mtime=time.time() + 2.0)
        r_rtv_recheck = run_once(cli, sdk, frontend, d_rtv, jobs=2)
        assert r_rtv_recheck.returncode == 0, (r_rtv_recheck.returncode, r_rtv_recheck.stdout, r_rtv_recheck.stderr)
        for key in (rtv_base, rtv_u16, rtv_f32):
            assert f"{key:016x}: refused:" in r_rtv_recheck.stdout, r_rtv_recheck.stdout
            assert not (d_rtv / f"{key:016x}.txt").exists()
            assert (d_rtv / f"{key:016x}.fail").is_file()

        # Part 2: Live background --watch process retries pre-existing .fail once, skips unchanged .fail,
        # translates newly written graphics/compute requests, and clears .fail when fixed.
        wdir = work / "live_watch"
        wdir.mkdir()
        now = time.time()
        stale_key = put_graphics_request(wdir, VS_WORDS, ps_words(2), g_root, (0x0002, 0x8002))
        touch(wdir / f"req_{stale_key:016x}.a", mtime=now - 20.0)
        stale_fail = wdir / f"{stale_key:016x}.fail"
        touch(stale_fail, b"stale reason\n", mtime=now - 10.0)

        watch = subprocess.Popen(
            [cli, "--watch", sdk, frontend, wdir, "--jobs", "2", "--interval", "50"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert wait_for(
                lambda: stale_fail.exists() and "unsupported pixel entry VGPR 2" in stale_fail.read_text(encoding="utf-8")
            ), "watcher did not retry pre-existing .fail on first pass"

            # Write a new graphics request (initially tampered .b so it refuses, then fixed to the same key).
            mut_key = G.pipeline_key(VS_WORDS, ps_words(4), g_root, ps_inputs=TARGET_MASKS)
            stem = wdir / f"req_{mut_key:016x}"
            stem.with_suffix(".b").write_bytes(ps_words(2))
            stem.with_suffix(".root").write_bytes(g_root)
            stem.with_suffix(".psin").write_bytes(describe_psin(mut_key, TARGET_MASKS).encode("ascii"))
            stem.with_suffix(".a").write_bytes(VS_WORDS)

            mut_fail = wdir / f"{mut_key:016x}.fail"
            assert wait_for(lambda: mut_fail.exists()), "watcher did not emit .fail for newly written refused request"
            first_fail_ns = mut_fail.stat().st_mtime_ns
            stale_fail_ns = stale_fail.stat().st_mtime_ns
            time.sleep(0.35)
            assert mut_fail.stat().st_mtime_ns == first_fail_ns, "watcher rewrote unchanged .fail on subsequent pass"
            assert stale_fail.stat().st_mtime_ns == stale_fail_ns, "watcher rewrote stale .fail on subsequent pass"

            # Fix the refused request with newer mtime: watcher must compile it and remove .fail.
            future2 = time.time() + 2.0
            touch(stem.with_suffix(".b"), ps_words(4), mtime=future2)
            touch(stem.with_suffix(".a"), VS_WORDS, mtime=future2)
            mut_txt = wdir / f"{mut_key:016x}.txt"
            assert wait_for(lambda: mut_txt.exists() and not mut_fail.exists()), (
                "watcher did not recover fixed request or remove .fail"
            )
            assert (wdir / f"{mut_key:016x}.vs.cso").read_bytes().startswith(b"DXBC")
            assert (wdir / f"{mut_key:016x}.ps.cso").read_bytes().startswith(b"DXBC")

            # Write a valid .rtv variant and an invalid .rtv variant while the watcher is running.
            live_rtv, live_rtv_line = put_variant(wdir, mut_key, (UINT16, UNORM8))
            live_rtv_bad, _ = put_variant(wdir, mut_key, (7,))
            live_rtv_txt = wdir / f"{live_rtv:016x}.txt"
            live_rtv_fail = wdir / f"{live_rtv_bad:016x}.fail"
            assert wait_for(lambda: live_rtv_txt.exists() and live_rtv_fail.exists()), (
                "watcher did not process newly written .rtv variants"
            )
            assert live_rtv_txt.read_text(encoding="ascii").splitlines()[-1] == live_rtv_line
            assert (wdir / f"{live_rtv:016x}.vs.cso").read_bytes().startswith(b"DXBC")
            assert (wdir / f"{live_rtv:016x}.ps.cso").read_bytes().startswith(b"DXBC")

            # Write a new compute request while the watcher is running.
            live_cs = put_compute_request(wdir, c_code, c_root, c_sa)
            live_cs_txt = wdir / f"{live_cs:016x}.txt"
            assert wait_for(lambda: live_cs_txt.exists()), "watcher did not compile newly written compute request"
            assert (wdir / f"{live_cs:016x}.cs.cso").read_bytes().startswith(b"DXBC")
            assert not (wdir / f"{live_cs:016x}.fail").exists()
            assert watch.poll() is None, "watcher exited unexpectedly"
        finally:
            watch.kill()
            out, err = watch.communicate(timeout=20)

        assert not err.strip(), f"unexpected watcher stderr: {err!r}"
        assert f"{stale_key:016x}: refused:" in out, out
        assert f"{mut_key:016x}: refused:" in out and f"{mut_key:016x}: compiled graphics" in out, out
        assert f"{live_cs:016x}: compiled compute" in out, out

        # Part 3: CLI negative controls.
        bad_file = work / "not_a_dir"
        bad_file.write_text("x", encoding="utf-8")
        for bad_args in (
            ["--watch"],
            ["--watch", sdk, frontend],
            ["--watch", work / "missing_sdk", frontend, wdir, "--once"],
            ["--watch", sdk, frontend, work / "missing_req", "--once"],
            ["--watch", sdk, frontend, bad_file, "--once"],
            ["--watch", sdk, frontend, wdir, "--interval", "5", "--once"],
            ["--watch", sdk, frontend, wdir, "--interval", "60001", "--once"],
            ["--watch", sdk, frontend, wdir, "--interval", "abc", "--once"],
            ["--watch", sdk, frontend, wdir, "--jobs", "0", "--once"],
            ["--watch", sdk, frontend, wdir, "--jobs", "65", "--once"],
            ["--watch", sdk, frontend, wdir, "--bogus", "--once"],
        ):
            res = subprocess.run([cli, *bad_args], capture_output=True, text=True)
            assert res.returncode != 0 and res.stderr.strip(), (bad_args, res.returncode, res.stdout, res.stderr)

    print("native_compile_dxil_watch checks: PASS (--once controls, resume/recheck, live --watch retry/skip/recovery, and CLI negative controls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
