"""Compare default and -Od DXIL of a source-built mesh shader through one PSO harness.

Each generated variant is a mesh shader with a known draw: threads 0-2 emit one
triangle that covers the 32x32 target. Before the vertices are written, every
thread runs a generated integer computation on a constant-buffer seed. The
vertex depth stays inside the clip volume only when the result equals the value
this script computes independently in Python, so a variant draws the green
reference only if the compiled computation is correct. The stress families grow
that computation without changing the expected image:

  control     no extra computation
  chain_N     N dependent straight-line multiply/xor/add steps
  branches_N  N divergent if/else regions that depend on the group thread
  loops_N     N loops whose trip count depends on a seed word and the thread

For every variant the script compiles the same HLSL twice with DXC (default
optimization and -Od, target ms_6_5) and records compile time, peak working set
and outcome. Every DXIL blob that compiles is fed unchanged to
`offscreen_check <adapter> --mesh-entry`, which uses the same root signature,
pixel shader, pipeline stream and draw for both blobs and prints the time of
the mesh CreatePipelineState call. The script records that time separately from
the process wall time, peak working set and outcome. Two negative controls run
on each blob: a changed seed and a wrong reference image must both be reported
as a draw mismatch.

Outcomes are kept separate: `pass`, `mismatch`, `unsupported` (no mesh shader
tier, exit 77), `hresult` (a D3D12 call failed; the HRESULT is recorded),
`timeout`, `crash` (NTSTATUS exit) and `error` for anything else. Compile
outcomes come from tools/dxc_compile_time.py.

Limits:
- The families are synthetic. They do not replay archive records and cannot
  say which archive program hits a compile or pipeline limit.
- A passing draw establishes the expected image for this seed, constant
  buffer layout and pixel shader. It does not establish equivalence of the two
  blobs for other inputs.
- Timeouts are local policy: --compile-timeout and --pipeline-timeout have no
  defaults. No watchdog, memory or host stack setting is changed.
- --check mode exits 77 when the adapter has no mesh shader tier; that is a
  capability skip, not a pass.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dxc_compile_time import (  # noqa: E402
    MIB, close_process, dxc_version, open_process, peak_working_set, run_once)

TARGET = "ms_6_5"
PIXEL_TARGET = "ps_6_0"
SIDE = 32
CB_BYTES = 512
TRACE_BYTES = 80
SEED = 0x9E3779B9
LOOP_BASE = 5
MASK = 0xFFFFFFFF
MAX_SIZE = 65536
VARIANT_RE = re.compile(r"(chain|branches|loops)_(\d+)")
OPTIMIZATIONS = ("default", "Od")
FAMILIES = ("control", "chain", "branches", "loops")
GREEN = bytes([0, 255, 0, 255])
BLACK = bytes([0, 0, 0, 255])
DEFAULT_CHECK_VARIANTS = "control,chain_64,branches_64,loops_16"

PIXEL = "float4 main() : SV_Target { return float4(0, 1, 0, 1); }\n"

HEADER = """cbuffer Seed : register(b0) { uint4 seed[32]; };
cbuffer Depth : register(b1) { float4 depth[32]; };
struct Vertex { float4 position : SV_Position; };

[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(uint tid : SV_GroupThreadID, out vertices Vertex verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    if (tid == 0) tris[0] = uint3(0, 1, 2);
    uint h = seed[0].x;
"""

FOOTER = """    if (tid < 3) {
        float2 corner = float2(tid == 1 ? 3.0f : -1.0f, tid == 2 ? 3.0f : -1.0f);
        float z = h == expected[tid] ? depth[0].x : 2.0f;
        verts[tid].position = float4(corner, z, 1.0f);
    }
}
"""


def _chain_step(h, k):
    return ((((h ^ (h >> 15)) * 0x2C1B3C6D) & MASK) + k) & MASK


def _branch_step(h, k, tid):
    if ((tid + k) & ((k % 7) + 1)) != 0:
        return (h + k * 0x01000193) & MASK
    return (h ^ (k * 0x85EBCA6B)) & MASK


def _loop_step(h, k, tid, base):
    for i in range((base + ((tid + k) & 3)) & MASK):
        h = ((h * 0x01000193) & MASK) ^ ((i + k) & MASK)
    return h


def expected_values(family, size, seed=SEED, base=LOOP_BASE):
    """Return the Python reference for h in threads 0, 1 and 2."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {', '.join(FAMILIES)}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"size {size!r} must be a non-negative integer")
    if family == "control" and size != 0:
        raise ValueError(f"control family requires size 0, got {size}")
    values = []
    for tid in range(3):
        h = seed & MASK
        for k in range(size):
            if family == "chain":
                h = _chain_step(h, k)
            elif family == "branches":
                h = _branch_step(h, k, tid)
            elif family == "loops":
                h = _loop_step(h, k, tid, base)
        values.append(h)
    return values


def _body(family, size):
    lines = []
    for k in range(size):
        if family == "chain":
            lines.append(f"    h = (h ^ (h >> 15u)) * 0x2c1b3c6du + {k}u;\n")
        elif family == "branches":
            lines.append(
                f"    if (((tid + {k}u) & {(k % 7) + 1}u) != 0u) h += {k}u * 0x01000193u;\n"
                f"    else h ^= {k}u * 0x85ebca6bu;\n")
        elif family == "loops":
            lines.append(
                f"    for (uint i{k} = 0u; i{k} < seed[1].x + ((tid + {k}u) & 3u); ++i{k})\n"
                f"        h = (h * 0x01000193u) ^ (i{k} + {k}u);\n")
    return "".join(lines)


def variant_info(name, seed=SEED):
    """Return (family, size, source, expected) for a generated variant name."""
    if not isinstance(name, str):
        raise ValueError(
            f"unknown variant {name!r}; use control, chain_N, branches_N or loops_N")
    if name == "control":
        family, size = "control", 0
    else:
        match = VARIANT_RE.fullmatch(name)
        if not match:
            raise ValueError(
                f"unknown variant {name!r}; use control, chain_N, branches_N or loops_N")
        family, size = match.group(1), int(match.group(2))
        if size < 1 or size > MAX_SIZE:
            raise ValueError(f"variant size {size} must be between 1 and {MAX_SIZE}")
    expected = expected_values(family, size, seed)
    table = ", ".join(f"0x{value:08x}u" for value in expected)
    source = (HEADER + f"    static const uint expected[3] = {{ {table} }};\n"
              + _body(family, size) + FOOTER)
    return family, size, source, expected


def fixture(seed=SEED, visible=True):
    """The --mesh-entry input: two 512-byte CBVs, the untouched trace and the reference image."""
    cb0 = bytearray(CB_BYTES)
    cb0[0:4] = (seed & MASK).to_bytes(4, "little")
    cb0[16:20] = LOOP_BASE.to_bytes(4, "little")
    cb1 = bytearray(CB_BYTES)
    cb1[0:4] = bytes.fromhex("0000003f")  # 0.5f
    trace = bytes([0xCD]) * TRACE_BYTES
    return bytes(cb0 + cb1) + trace + (GREEN if visible else BLACK) * (SIDE * SIDE)


HRESULT_RE = re.compile(r"^(.*) failed \(HRESULT (0x[0-9a-fA-F]{8})\)", re.MULTILINE)
PIPELINE_MS_RE = re.compile(r"^mesh_pipeline_ms=([\d.]+)", re.MULTILINE)
MISMATCH_RE = re.compile(r"^pixels=1024 mismatches=(\d+)", re.MULTILINE)


def classify_draw(exit_code, timed_out, text):
    """Return (outcome, detail) for one offscreen_check run."""
    if timed_out:
        return "timeout", ""
    if exit_code is None:
        return "error", "no exit code"
    if exit_code == 77 or "mesh_shader_tier=unsupported" in text:
        return "unsupported", "mesh shader tier"
    if exit_code < 0 or exit_code >= 0xC0000000:
        return "crash", f"0x{exit_code & MASK:08x}"
    hresult = HRESULT_RE.search(text)
    if hresult:
        return "hresult", f"{hresult.group(1)} {hresult.group(2).lower()}"
    if exit_code == 0:
        return "pass", ""
    if "Draw reference mismatch" in text:
        found = MISMATCH_RE.search(text)
        return "mismatch", f"{found.group(1)} pixels" if found else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "error", lines[-1][:200] if lines else f"exit {exit_code}"


def run_draw(renderer, adapter, mesh, pixel, data_path, timeout_s, sample_interval=0.005):
    """Run one --mesh-entry draw and measure the whole process."""
    args = [str(renderer), adapter, "--mesh-entry", str(mesh), str(pixel), str(data_path)]
    start = time.perf_counter()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    handle = open_process(proc.pid) if open_process else None
    peak = 0
    stop = threading.Event()

    def monitor():
        nonlocal peak
        while not stop.is_set():
            value = peak_working_set(handle)
            if value and value > peak:
                peak = value
            stop.wait(sample_interval)

    thread = threading.Thread(target=monitor, daemon=True) if handle else None
    if thread:
        thread.start()
    timed_out = False
    try:
        try:
            output, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            output, _ = proc.communicate()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        if thread:
            thread.join(timeout=1.0)
        if handle:
            value = peak_working_set(handle)
            if value and value > peak:
                peak = value
            close_process(handle)
    text = (output or b"").decode("utf-8", "replace")
    outcome, detail = classify_draw(proc.returncode, timed_out, text)
    pipeline = PIPELINE_MS_RE.search(text)
    if outcome in ("timeout", "crash", "error"):
        # The renderer flushes mesh_pipeline_ms right after CreatePipelineState returns.
        phase = "after" if pipeline else "before or during"
        detail = f"{detail} {phase} pipeline creation".strip()
    return {
        "outcome": outcome,
        "detail": detail,
        "exit_code": proc.returncode,
        "wall_s": round(elapsed, 3),
        "peak_mib": round(peak / MIB, 1) if peak else None,
        "pipeline_ms": float(pipeline.group(1)) if pipeline else None,
        "output": text,
    }


def adapter_identity(text):
    keys = ("backend", "adapter", "driver")
    return [line.strip() for line in text.splitlines() if line.strip().startswith(keys)]


def measure(names, dxc, renderer, adapter, work, repeat, compile_timeout, pipeline_timeout,
            seed=SEED):
    pixel_hlsl = work / "pixel.hlsl"
    pixel_hlsl.write_text(PIXEL, encoding="utf-8")
    pixel = work / "pixel.dxil"
    compiled = run_once(dxc, pixel_hlsl, pixel, ["-T", PIXEL_TARGET, "-E", "main"], compile_timeout)
    if compiled["outcome"] != "dxil":
        raise RuntimeError(f"pixel shader did not compile: {compiled['first_diagnostic']}")
    inputs = {}
    for label, data in (("reference", fixture(seed)), ("wrong_seed", fixture(seed ^ 1)),
                        ("wrong_reference", fixture(seed, visible=False))):
        inputs[label] = work / f"{label}.bin"
        inputs[label].write_bytes(data)
    identity = []
    records = []
    for name in names:
        family, size, source, expected = variant_info(name, seed)
        hlsl = work / f"{name}.hlsl"
        hlsl.write_text(source, encoding="utf-8")
        for optimization in OPTIMIZATIONS:
            flags = ["-T", TARGET, "-E", "main"] + (["-Od"] if optimization == "Od" else [])
            mesh = work / f"{name}.{optimization}.dxil"
            compile_run = run_once(dxc, hlsl, mesh, flags, compile_timeout)
            record = {
                "variant": name, "family": family, "size": size, "optimization": optimization,
                "hlsl_bytes": len(source.encode("utf-8")),
                "expected": [f"0x{value:08x}" for value in expected],
                "compile_outcome": compile_run["outcome"],
                "compile_s": round(compile_run["wall_s"], 3),
                "compile_peak_mib": (round(compile_run["peak_bytes"] / MIB, 1)
                                     if compile_run["peak_bytes"] else None),
                "compile_diagnostic": compile_run["first_diagnostic"],
                "dxil_bytes": compile_run["cso_bytes"],
                "command": " ".join(f'"{arg}"' if " " in str(arg) else str(arg)
                                    for arg in compile_run["args"]),
                "draws": [], "controls": {},
            }
            print(f"{name} {optimization}: compile {record['compile_s']:.3f} s "
                  f"{record['compile_outcome']}", flush=True)
            if compile_run["outcome"] == "dxil":
                for _ in range(repeat):
                    draw = run_draw(renderer, adapter, mesh, pixel, inputs["reference"],
                                    pipeline_timeout)
                    if not identity:
                        identity = adapter_identity(draw["output"])
                    record["draws"].append({k: v for k, v in draw.items() if k != "output"})
                    print(f"  draw: {draw['outcome']} pipeline={draw['pipeline_ms']} ms "
                          f"wall={draw['wall_s']:.3f} s peak={draw['peak_mib']} MiB "
                          f"{draw['detail']}", flush=True)
                    if draw["outcome"] != "pass":
                        break
                if record["draws"][-1]["outcome"] == "pass":
                    for label in ("wrong_seed", "wrong_reference"):
                        control = run_draw(renderer, adapter, mesh, pixel, inputs[label],
                                           pipeline_timeout)
                        record["controls"][label] = control["outcome"]
                        print(f"  control {label}: {control['outcome']} {control['detail']}",
                              flush=True)
            records.append(record)
    return identity, records


def summarize(record):
    draws = record["draws"]
    passed = [draw for draw in draws if draw["pipeline_ms"] is not None]
    return {
        "draw_outcome": draws[-1]["outcome"] if draws else "not_run",
        "draw_detail": draws[-1]["detail"] if draws else "",
        "pipeline_ms_first": draws[0]["pipeline_ms"] if draws else None,
        "pipeline_ms_median": (round(statistics.median(d["pipeline_ms"] for d in passed), 3)
                               if passed else None),
        "draw_wall_s_median": (round(statistics.median(d["wall_s"] for d in draws), 3)
                               if draws else None),
        "draw_peak_mib_max": max((d["peak_mib"] for d in draws if d["peak_mib"]), default=None),
        "controls": ";".join(f"{k}={v}" for k, v in record["controls"].items()),
    }


def check_failures(records):
    """Return the reasons a --check run fails, or an empty list."""
    failures = []
    for record in records:
        key = f"{record['variant']} {record['optimization']}"
        if record["compile_outcome"] != "dxil":
            failures.append(f"{key}: compile {record['compile_outcome']}")
            continue
        draws = record["draws"]
        if not draws or draws[-1]["outcome"] != "pass":
            failures.append(f"{key}: draw {draws[-1]['outcome'] if draws else 'not_run'}")
            continue
        if any(draw["pipeline_ms"] is None for draw in draws):
            failures.append(f"{key}: no mesh_pipeline_ms line from the renderer")
        for label in ("wrong_seed", "wrong_reference"):
            if record["controls"].get(label) != "mismatch":
                failures.append(f"{key}: control {label} gave {record['controls'].get(label)}")
    return failures


CSV_FIELDS = (
    "variant", "family", "size", "optimization", "hlsl_bytes", "dxil_bytes",
    "compile_outcome", "compile_s", "compile_peak_mib", "compile_diagnostic",
    "draw_outcome", "draw_detail", "pipeline_ms_first", "pipeline_ms_median", "draw_wall_s_median",
    "draw_peak_mib_max", "controls",
)


def print_table(records):
    print()
    print(f"{'variant':<16}{'opt':<9}{'DXIL B':>10}{'compile s':>11}  {'compile':<11}"
          f"{'PSO 1st':>10}{'PSO med':>10}{'peak MiB':>10}  draw / controls")
    for record in records:
        summary = summarize(record)
        first = "-" if summary["pipeline_ms_first"] is None else f"{summary['pipeline_ms_first']:.3f}"
        pso = "-" if summary["pipeline_ms_median"] is None else f"{summary['pipeline_ms_median']:.3f}"
        peak = "-" if summary["draw_peak_mib_max"] is None else f"{summary['draw_peak_mib_max']:.1f}"
        print(f"{record['variant']:<16}{record['optimization']:<9}{record['dxil_bytes']:>10,}"
              f"{record['compile_s']:>11.3f}  {record['compile_outcome']:<11}{first:>10}{pso:>10}{peak:>10}"
              f"  {summary['draw_outcome']} {summary['draw_detail']} {summary['controls']}")


def write_report(path, payload, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({**payload, "variants": records}, indent=1) + "\n",
                        encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, **summarize(record)})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dxc", type=Path, help="DXC executable")
    parser.add_argument("--renderer", type=Path, help="offscreen_check executable")
    parser.add_argument("--adapter", choices=("hardware", "warp"),
                        help="offscreen_check backend; XBOX2WINDOWS_ADAPTER selects the hardware adapter")
    parser.add_argument("--variants", default=DEFAULT_CHECK_VARIANTS,
                        help=f"comma-separated variants (default: {DEFAULT_CHECK_VARIANTS})")
    parser.add_argument("--repeat", type=int, default=1, help="reference draws per blob (default: 1)")
    parser.add_argument("--compile-timeout", type=float, metavar="SECONDS",
                        help="wall timeout for one DXC run; local policy, required")
    parser.add_argument("--pipeline-timeout", type=float, metavar="SECONDS",
                        help="wall timeout for one offscreen_check run; local policy, required")
    parser.add_argument("--seed", type=lambda text: int(text, 0), default=SEED,
                        help="32-bit seed word; a new seed varies the DXIL to limit reuse of "
                             "identical shader blobs (default: 0x9e3779b9)")
    parser.add_argument("--check", action="store_true",
                        help="fail unless every blob compiles, draws the reference and fails "
                             "both controls; exit 77 without a mesh shader tier")
    parser.add_argument("--out", type=Path, help="write CSV, or JSON when the name ends in .json")
    parser.add_argument("--work-dir", type=Path, help="keep the generated HLSL, DXIL and inputs here")
    parser.add_argument("--list-variants", action="store_true",
                        help="print variant names, HLSL sizes and expected words without compiling")
    args = parser.parse_args(argv)

    try:
        names = list(dict.fromkeys(n.strip() for n in args.variants.split(",") if n.strip()))
        if not names:
            raise ValueError("--variants selects nothing")
        if not 0 <= args.seed <= MASK:
            raise ValueError("--seed must fit in 32 bits")
        infos = {name: variant_info(name, args.seed) for name in names}
    except ValueError as exc:
        parser.error(str(exc))
    if args.list_variants:
        for name, (family, size, source, expected) in infos.items():
            words = " ".join(f"0x{value:08x}" for value in expected)
            print(f"{name:<16}{family:<10}{size:>7}{len(source.encode('utf-8')):>10,}  {words}")
        return 0
    for option in ("dxc", "renderer", "adapter", "compile_timeout", "pipeline_timeout"):
        if getattr(args, option) is None:
            parser.error(f"--{option.replace('_', '-')} is required unless --list-variants is given")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not all(math.isfinite(value) and value > 0 for value in (args.compile_timeout, args.pipeline_timeout)):
        parser.error("timeouts must be finite and positive")
    for option in ("dxc", "renderer"):
        if not getattr(args, option).is_file():
            parser.error(f"--{option} not found: {getattr(args, option)}")

    temporary = None
    try:
        if args.work_dir:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            work = args.work_dir
        else:
            temporary = tempfile.TemporaryDirectory(prefix="mesh-od-pipeline-")
            work = Path(temporary.name)
        identity, records = measure(names, args.dxc, args.renderer, args.adapter, work,
                                    args.repeat, args.compile_timeout, args.pipeline_timeout,
                                    args.seed)
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary:
            temporary.cleanup()

    for line in identity:
        print(line)
    print_table(records)
    if args.out:
        payload = {
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dxc": str(args.dxc), "dxc_version": dxc_version(args.dxc),
            "target": TARGET, "adapter": args.adapter, "identity": identity,
            "seed": f"0x{args.seed:08x}",
            "repeat": args.repeat, "compile_timeout_s": args.compile_timeout,
            "pipeline_timeout_s": args.pipeline_timeout,
        }
        try:
            write_report(args.out, payload, records)
        except OSError as exc:
            print(f"error: failed to write {args.out}: {exc}", file=sys.stderr)
            return 1
        print(f"\nwrote {args.out}")
    if not args.check:
        return 0
    if records and all(record["compile_outcome"] == "dxil" and record["draws"]
                       and all(draw["outcome"] == "unsupported" for draw in record["draws"])
                       for record in records):
        print("mesh shaders unsupported on this adapter (capability skip)")
        return 77
    failures = check_failures(records)
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
