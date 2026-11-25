"""Depth target binding for translated draws (#1999 item D).

usage: startup_draw_depth_target_check.py <startup_draw_depth_target> <warp|hardware> [check|drop|debug]
Exit codes: 0 pass, 1 unexpected result, 2 broken run, 77 no hardware adapter (and, for "debug" only,
no D3D12 debug layer).

tests/startup_draw_depth_target.cpp draws a near green triangle (z 0.25), then a far red one (z 0.75), with a
depth-tested stream pipeline (LESS, writes on), and reads back the colour target and the D32 depth target.

"check" (the default) runs three cases, each in its own process:
  bound    the depth handle resolves: green in front, depth 0.25 in the drawn half and 1.0 elsewhere,
           one bind and nothing unresolved
  null     the depth handle holds 0, which maps to no descriptor: the stand-in logs it and counts it once
           and binds the colour target without depth, so red overwrites green and the depth target keeps 1.0
  unknown  the depth handle is in no descriptor heap: the stand-in logs it, counts it once and refuses the
           whole bind, so nothing is drawn (the colour target keeps its clear colour, depth keeps 1.0)

"drop" is the negative control: the bound case's draw with the null handle, checked against the
bound case's expectations. The control passes only when the expected mismatch appears (red in front,
depth untouched, and one colour target bind). A broken run or an unexpected picture fails CTest.

"debug" runs the bound case with the D3D12 debug layer (X2W_D3D12_DEBUG=1) and requires the bound picture
and no debug-layer message of any severity after the fence. Without an installed debug layer the message
check cannot run, so this mode exits 77 (a capability skip that CTest reports as such) instead of passing.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_stream_check import gfx2hlsl
from startup_stream_depth_check import vertex
from startup_stream_draw_check import pixel

GREEN, RED, CLEAR = bytes((0, 255, 0, 255)), bytes((255, 0, 0, 255)), bytes((0, 0, 255, 255))


def target(color):
    return (color * 4 + CLEAR * 4) * 4


def depth(drawn):
    return (struct.pack("<4f", *[drawn] * 4) + struct.pack("<4f", *[1.0] * 4)) * 4


BOUND = (target(GREEN), depth(0.25))
UNBOUND = (target(RED), depth(1.0))
REFUSED = (CLEAR * 32, depth(1.0))
NO_DEPTH_LOG = "maps to no descriptor; binding no depth"
REFUSED_LOG = "OMSetRenderTargets refused: depth handle is not a known depth stencil view"


class Broken(Exception):
    pass


def prepare(directory, program, backend):
    cache = directory / "cache"
    cache.mkdir(parents=True)
    paths = [directory / name for name in ("vs_front", "vs_back", "ps_green", "ps_red", "root")]
    paths[0].write_bytes(vertex(0x3e800000))  # z = 0.25f
    paths[1].write_bytes(vertex(0x3f400000))  # z = 0.75f
    paths[2].write_bytes(pixel(True))
    paths[3].write_bytes(pixel(False))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1", X2W_D3D12_DEBUG="1",
               X2W_SHADER_CACHE=str(cache))
    args = [program, *map(str, paths), backend]
    request = subprocess.run(args + ["request", str(directory / "request")], env=env, capture_output=True, text=True)
    if backend == "hardware" and request.returncode == 77:
        return None
    if request.returncode:
        raise Broken(f"request run failed on {backend}: {request.returncode}\n{request.stderr}")
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    return args, env


def run_case(directory, args, env, case):
    out = directory / case
    run = subprocess.run(args + [case, str(out)], env=env, capture_output=True, text=True)
    # Every case but "unknown" must draw. In "unknown" the stand-in refuses the whole OMSetRenderTargets,
    # so no target is bound and no draw may be issued; its "nothing drawn" is checked through the read-back
    # (the clear colour and depth 1.0) and the refusal line, not through a "-> drawn;" line.
    print(f"[{case}] " + run.stdout.strip().replace("\n", f"\n[{case}] "))
    debug = [line for line in run.stderr.splitlines() if "x2w-d3d12-debug " in line]
    for line in run.stderr.splitlines():
        if "x2w-d3d12-debug " in line or "OMSetRenderTargets" in line or "translator error" in line:
            print(f"[{case}] {line}")
    if run.returncode or (case != "unknown" and "-> drawn;" not in run.stderr):
        raise Broken(f"{case} run failed: {run.returncode}\n{run.stderr}")
    counts = re.search(r"binds \+(\d+), unresolved \+(\d+)", run.stdout)
    if not counts:
        raise Broken(f"{case}: no counter line\n{run.stdout}")
    return {
        "colour": Path(f"{out}.rgba").read_bytes(),
        "depth": Path(f"{out}.depth").read_bytes(),
        "binds": int(counts[1]),
        "unresolved": int(counts[2]),
        "logged": NO_DEPTH_LOG in run.stderr,
        "refused": REFUSED_LOG in run.stderr,
        "debug": debug,
        "debug_layer": "x2w-d3d12 debug layer: 0x00000000 enabled" in run.stderr,
    }


def describe(result):
    values = struct.unpack("<32f", result["depth"])
    return f"colour {result['colour'][:16].hex()}..., depth row 0 {values[:8]}"


def main():
    program, backend = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "check"
    if mode not in ("check", "drop", "debug"):
        print(__doc__)
        return 64
    try:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            prepared = prepare(directory, program, backend)
            if prepared is None:
                print("No D3D12 feature-level 12 hardware adapter")
                return 77
            args, env = prepared
            if mode == "drop":
                result = run_case(directory, args, env, "drop")
                failures = []
                if (result["colour"], result["depth"]) != BOUND:
                    failures.append("picture")
                stated = failures == ["picture"] and (result["colour"], result["depth"]) == UNBOUND and result["binds"] == 1
                print(f"control drop: {describe(result)} -> " +
                      ("failed for the stated reason" if stated else "did NOT fail for the stated reason"))
                return 0 if stated else 1
            if mode == "debug":
                bound = run_case(directory, args, env, "bound")
                if not bound["debug_layer"]:
                    print("the D3D12 debug layer is not installed: the debug-message check cannot run (skip)")
                    return 77
                picture = (bound["colour"], bound["depth"]) == BOUND
                print(f"debug: debug layer enabled, {len(bound['debug'])} debug-layer message(s) after the fence; "
                      f"bound picture {'matches' if picture else 'DIFFERS'}")
                return 0 if picture and not bound["debug"] else 1
            failed = []
            bound = run_case(directory, args, env, "bound")
            if (bound["colour"], bound["depth"]) != BOUND:
                failed.append(f"bound: {describe(bound)}")
            if bound["binds"] != 1 or bound["unresolved"] or bound["logged"]:
                failed.append(f"bound: binds +{bound['binds']}, unresolved +{bound['unresolved']}, logged {bound['logged']}")
            null = run_case(directory, args, env, "null")
            if (null["colour"], null["depth"]) != UNBOUND:
                failed.append(f"null: {describe(null)}")
            if null["binds"] != 1 or null["unresolved"] != 1 or not null["logged"]:
                failed.append(f"null: binds +{null['binds']}, unresolved +{null['unresolved']}, logged {null['logged']}")
            unknown = run_case(directory, args, env, "unknown")
            if (unknown["colour"], unknown["depth"]) != REFUSED:
                failed.append(f"unknown: {describe(unknown)}")
            if unknown["binds"] or unknown["unresolved"] != 1 or not unknown["refused"]:
                failed.append(f"unknown: binds +{unknown['binds']}, unresolved +{unknown['unresolved']}, refused {unknown['refused']}")
    except (Broken, OSError, subprocess.CalledProcessError) as error:
        # A broken run cannot count as seeing the expected mismatch.
        print(f"error: {error}")
        return 2
    for line in failed:
        print("FAIL", line)
    if failed:
        return 1
    print(f"{backend}: bound depth target occluded red with depth 0.25; a zero depth "
          "handle bound the colour target without depth; an unknown one refused the bind; each logged and counted once")
    return 0


if __name__ == "__main__":
    sys.exit(main())
