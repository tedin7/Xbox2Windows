"""A compute program that compiles but gets no runtime manifest names the reason (#3224).

The program reads a root CBV and a root SRV (the raw-root fixture), so it translates to DXIL but the
compute manifest has no line for a root descriptor. A table-bound control still publishes its manifest.
--requests reports the reason on stdout and leaves no .fail, so watchers keep retrying the key; --watch,
which refuses the key, writes the reason to its .fail.

usage:
  native_compute_manifest_reasons_check.py <compile_dxil> <sdk>
"""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools" / "startup"))

import gfx2hlsl as G  # noqa: E402
import triage_run  # noqa: E402
from native_compute_clear_check import write_request  # noqa: E402
from native_startup_request_cache_check import (  # noqa: E402
    COMPUTE_GROUP,
    COMPUTE_RSRC2,
    compile_compute_root,
    compute_program_words,
    put_compute_request,
)
from startup_compute_translate_check import state_a  # noqa: E402

# s_load_dwordx4 from the root CBV at s[6:7]; copy the root SRV address s[8:9] to s[2:3]; load from it; s_endpgm.
ROOT_CODE = [0xF40C0403, 0xFA000010, 0xBE820408, 0xF40C0601, 0xFA000030, 0xBF810000]
ROOT_SA = struct.pack("<7I", 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 10 << 1)
REASON = "unsupported startup runtime manifest bindings: SRV kind root_buffer_srv"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    cli, sdk = Path(argv[0]).resolve(), Path(argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src = work / "root.hlsl"
        src.write_text('#define RS "RootConstants(num32BitConstants=2,b1),CBV(b0),SRV(t0)"', encoding="ascii")
        subprocess.run([sdk / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS", "-Fo", work / "root.bin", src], check=True)
        requests = work / "requests"
        requests.mkdir()
        key = write_request(requests, ROOT_CODE, (work / "root.bin").read_bytes(), ROOT_SA).stem[4:]
        control = "%016x" % put_compute_request(
            requests, compute_program_words(), compile_compute_root(sdk, work), state_a(COMPUTE_GROUP, COMPUTE_RSRC2))
        watched = work / "watched"
        shutil.copytree(requests, watched)

        # 1. --requests in a live cache: the DXIL is published, the reason goes to stdout, and no .fail is left.
        # A genuine refusal beside it keeps its .fail. The Python watcher's two selections tell them apart.
        live = work / "live"
        shutil.copytree(requests, live)
        broken = compute_program_words()[:-4] + struct.pack("<2I", 0xFFFFFFFF, 0xBF810000)
        refused = "%016x" % put_compute_request(
            live, broken, compile_compute_root(sdk, work), state_a(COMPUTE_GROUP, COMPUTE_RSRC2))
        (live / f"{key}.fail").write_text("stale failure\n", encoding="ascii")
        run = subprocess.run([cli, sdk, work / "frontend", "--jobs", "2", "gfx1013-direct-cs@0", "--requests",
                              live, live], capture_output=True, text=True)
        assert run.returncode == 1 and "requests=3 compiled=2 refused=1" in run.stdout, (run.stdout, run.stderr)
        retry_all = sorted(int(k, 16) for k in (key, refused))
        assert G.pending_keys(str(live), False) == retry_all, G.pending_keys(str(live), False)
        assert G.pending_keys(str(live), True) == [int(key, 16)], G.pending_keys(str(live), True)
        assert f"no_manifest input=req_{key}.c67 {REASON}\n" in run.stdout, run.stdout
        assert run.stdout.count("no_manifest ") == 1, run.stdout
        assert (live / f"{key}.cs.cso").read_bytes().startswith(b"DXBC")
        assert not (live / f"{key}.txt").exists() and not (live / f"{key}.fail").exists()
        assert (live / f"{control}.txt").is_file() and not (live / f"{control}.fail").exists()
        assert (live / f"{refused}.fail").is_file() and not (live / f"{refused}.cs.cso").exists()

        # 2. --refusals counts it with the same text.
        summary = json.loads(subprocess.run([cli, "--refusals", requests, "--json"], capture_output=True,
                                            text=True, check=True).stdout)["compute"]
        assert summary["examined"] == 2 and summary["no_refusal"] == 1 and summary["refused"] == 1, summary
        assert summary["reasons"] == [{"reason": REASON, "count": 1}], summary

        # 3. --watch writes the same line, and triage_run.py reports it as a fixed category without the key.
        watch = subprocess.run([cli, "--watch", sdk, work / "watch_cache", watched, "--once"],
                               capture_output=True, text=True)
        assert watch.returncode == 0 and f"{key}: refused: {REASON}" in watch.stdout, (watch.stdout, watch.stderr)
        assert f"{control}: compiled compute" in watch.stdout, watch.stdout
        assert (watched / f"{key}.fail").read_text(encoding="ascii") == REASON + "\n"
        # A watcher refusal is a genuine failure: skipped by the repeated pass, retried by the first.
        assert G.pending_keys(str(watched), True) == [] and G.pending_keys(str(watched), False) == [int(key, 16)]
        logs = work / "logs"
        logs.mkdir()
        (logs / "stderr.txt").write_text("", encoding="utf-8")
        report = triage_run.triage_run(logs, shader_cache=watched)
        assert "| 1 | `compute: runtime manifest: SRV kind root_buffer_srv` |" in report, report
        assert key not in report and control not in report
    print("compute manifest reasons: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
