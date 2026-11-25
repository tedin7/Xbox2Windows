"""Check that a Vulkan driver preserves binary32 subnormals in a source-built pixel shader.

The pixel shader (ps_6_2) runs seven operations whose inputs or results are binary32 subnormals and writes the
result words to an RGBA32UI target through tests/vk_pixel_draw.c. The words read back are compared with IEEE
binary32 arithmetic computed in Python. The shader is compiled twice:

* `-denorm preserve` must match IEEE on every case.
* `-denorm ftz` is the negative control and must differ on at least one case, otherwise the probe cannot
  tell preserve from flush and the check fails.

A device without `shaderDenormPreserveFloat32` is reported as a capability skip, and a shader module or pipeline
the driver refuses (or that spirv-val rejects) as a validation refusal. Neither counts as a mismatch.

Exit code: 0 all devices conform, 1 a mismatch or a tool error, 77 every device was a capability skip,
78 a validation refusal and no measured mismatch. A preserve readback that differs from IEEE exits 1 even when
the ftz build is refused.

Without arguments the script runs only the device-free controls of the classification (preserve mismatch with an
ftz refusal, an ftz control that matches preserve, and the nominal cases); the Linux Python checks job runs it so.
"""

import argparse
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Vulkan 1.2 and the bindings used by the Vulkan host: b0 = 0, t0 = 16, s0 = 32, u0 = 48.
DXC_FLAGS = [
    "-spirv", "-fspv-target-env=vulkan1.2",
    "-fvk-b-shift", "0", "0", "-fvk-t-shift", "16", "0", "-fvk-s-shift", "32", "0", "-fvk-u-shift", "48", "0",
]

# (name, a, b, HLSL operation, IEEE result with subnormals preserved)
CASES = [
    ("sub*1", 0x00000001, 0x3F800000, "a * b", 0x00000001),
    ("sub+0", 0x007FFFFF, 0x00000000, "a + b", 0x007FFFFF),
    ("minnorm*0.5", 0x00800000, 0x3F000000, "a * b", 0x00400000),
    ("minnorm-sub", 0x00800000, 0x00000001, "a - b", 0x007FFFFF),
    ("sub+sub", 0x00000001, 0x00000001, "a + b", 0x00000002),
    ("-sub*1", 0x80000003, 0x3F800000, "a * b", 0x80000003),
    ("sub*2^23", 0x00000001, 0x4B000000, "a * b", 0x00800000),
]
WIDTH, HEIGHT = 4, 2  # one pixel per case, row-major

VERTEX = """
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

PIXEL = """
cbuffer Constants : register(b0) { uint4 words[16]; };
uint word(uint i) { return words[i >> 2][i & 3]; }
uint4 main(float4 pos : SV_Position) : SV_Target {
    uint n = uint(pos.y) * %d + uint(pos.x);
    float a = asfloat(word(2 * n));
    float b = asfloat(word(2 * n + 1));
    uint r = 0;
    switch (n) {
%s
    }
    return uint4(r, 0, 0, 0);
}
"""


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ieee(a, b, op):
    # Double arithmetic on binary32 operands, rounded once to binary32, is correctly rounded for + - and *.
    x, y = f32(a), f32(b)
    return bits32({"a * b": x * y, "a + b": x + y, "a - b": x - y}[op])


def classify(preserve, ftz):
    """Combine the two readbacks into ok, mismatch, refused or error.

    Each argument is "error", "refused" or the list of result words. A measured preserve mismatch is reported
    even when the ftz build is refused. Returns (status, preserve conforms, names of the cases where ftz
    differs, or None when ftz has no readback).
    """
    wants = [case[4] for case in CASES]
    preserved = preserve == wants if isinstance(preserve, list) else None
    if isinstance(preserve, list) and not preserved:
        differing = [c[0] for c, got in zip(CASES, ftz) if got != c[4]] if isinstance(ftz, list) else None
        return "mismatch", False, differing
    if "error" in (preserve, ftz):
        return "error", preserved, None
    if "refused" in (preserve, ftz):
        return "refused", preserved, None
    differing = [c[0] for c, got in zip(CASES, ftz) if got != c[4]]
    return ("ok" if differing else "mismatch"), True, differing


def controls():
    """Device-free controls of the classification (mismatch and error exit 1, refused 78, ok 0)."""
    wants = [case[4] for case in CASES]
    bad = [wants[0] ^ 1] + wants[1:]
    flushed = [0] + wants[1:]
    expected = [
        ((wants, flushed), "ok"),
        ((wants, wants), "mismatch"),  # ftz control indistinguishable from preserve
        ((bad, flushed), "mismatch"),
        ((bad, "refused"), "mismatch"),  # a measured preserve mismatch is not hidden by an ftz refusal
        ((bad, "error"), "mismatch"),
        ((wants, "refused"), "refused"),
        (("refused", flushed), "refused"),
        (("refused", "refused"), "refused"),
        ((wants, "error"), "error"),
        (("error", "refused"), "error"),
    ]
    for (preserve, ftz), want in expected:
        got = classify(preserve, ftz)[0]
        if got != want:
            raise ValueError(f"classify({preserve!r}, {ftz!r}) is {got}, expected {want}")
    print(f"classification controls: {len(expected)} cases passed")


def pixel_source():
    arms = "\n".join(f"        case {i}: r = asuint({op}); break;" for i, (_, _, _, op, _) in enumerate(CASES))
    return PIXEL % (WIDTH, arms)


def tool_version(command, pattern):
    out = subprocess.run(command, capture_output=True, text=True)
    match = re.search(pattern, out.stdout + out.stderr)
    return match.group(0) if match else (out.stdout + out.stderr).strip().splitlines()[0]


def build_host(host, work):
    if host.suffix != ".c":
        return host.resolve()
    binary = work / "vk_pixel_draw"
    subprocess.run(["cc", "-std=c11", "-O1", "-o", binary, host, "-lvulkan"], check=True)
    return binary


def run_mode(dxc, spirv_val, host, device, vs, work, mode, constants, texture):
    """Return (status, words, detail). Status is ok, refused or error."""
    hlsl = work / f"{mode}.hlsl"
    spv = work / f"{mode}.spv"
    hlsl.write_text(pixel_source())
    subprocess.run([str(dxc), *DXC_FLAGS, "-T", "ps_6_2", "-E", "main", "-denorm", mode, "-Fo", spv, hlsl],
                   check=True)
    dis = subprocess.run(["spirv-dis", spv], capture_output=True, text=True).stdout
    modes = sorted(set(re.findall(r"OpExecutionMode %\w+ (Denorm\w+) 32", dis)))
    val = subprocess.run([str(spirv_val), "--target-env", "vulkan1.2", spv], capture_output=True, text=True)
    if val.returncode != 0:
        return "refused", None, f"spirv-val: {(val.stdout + val.stderr).strip()}", modes
    out = work / f"{mode}.bin"
    run = subprocess.run([str(host), device, vs, spv, constants, texture, "1", "1", out, str(WIDTH), str(HEIGHT),
                          "uint"], capture_output=True, text=True)
    if run.returncode == 78:
        return "refused", None, run.stderr.strip(), modes
    if run.returncode != 0 or not out.exists():
        return "error", None, f"host exit {run.returncode}: {run.stderr.strip()}", modes
    raw = out.read_bytes()
    pixels = struct.unpack(f"<{WIDTH * HEIGHT * 4}I", raw)
    return "ok", [pixels[4 * i] for i in range(len(CASES))], "", modes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dxc", type=Path, nargs="?", help="DXC binary (omit with host to run only the device-free controls)")
    parser.add_argument("host", type=Path, nargs="?", help="vk_pixel_draw binary, or tests/vk_pixel_draw.c to build it")
    parser.add_argument("--device", action="append", default=None,
                        help="Substring of the Vulkan device name (repeatable; default: first device)")
    args = parser.parse_args()
    controls()
    if args.dxc is None and args.host is None:
        return 0
    if args.dxc is None or args.host is None:
        parser.error("dxc and host go together")
    for _, a, b, op, want in CASES:
        if ieee(a, b, op) != want:
            raise ValueError("The IEEE table disagrees with the Python reference")
    dxc_version = tool_version([str(args.dxc), "--version"], r"\d+\.\d+\.\d+[^\n]*|\(\d+\.\d+\.\d+[^)]*\)")
    spirv_val = "spirv-val"
    spirv_version = tool_version([spirv_val, "--version"], r"SPIRV-Tools v[^\n]*")
    out_root = Path(__file__).resolve().parents[1] / "out"
    out_root.mkdir(exist_ok=True)
    statuses = []
    with tempfile.TemporaryDirectory(prefix="pixel-denorm-", dir=out_root) as d:
        work = Path(d)
        host = build_host(args.host, work)
        vs_hlsl, vs = work / "vs.hlsl", work / "vs.spv"
        vs_hlsl.write_text(VERTEX)
        subprocess.run([str(args.dxc), *DXC_FLAGS, "-fvk-invert-y", "-T", "vs_6_2", "-E", "main", "-Fo", vs, vs_hlsl],
                       check=True)
        constants = work / "constants.bin"
        constants.write_bytes(struct.pack("<64I", *[w for _, a, b, _, _ in CASES for w in (a, b)],
                                          *[0] * (64 - 2 * len(CASES))))
        texture = work / "texture.rgba"
        texture.write_bytes(bytes(4))
        print(f"DXC {dxc_version}; {spirv_version}")
        for device in args.device or [""]:
            caps = subprocess.run([str(host), "--caps", device], capture_output=True, text=True)
            if caps.returncode == 77:
                print(f"[{device or 'default'}] capability skip: no matching Vulkan device\n{caps.stderr.strip()}")
                statuses.append("skip")
                continue
            if caps.returncode != 0:
                raise SystemExit(f"Host failed: {caps.stderr.strip()}")
            print(caps.stdout.strip())
            if "denorm_preserve32=1" not in caps.stdout:
                print("  capability skip: shaderDenormPreserveFloat32 is false, so -denorm preserve cannot be "
                      "requested (not a mismatch)")
                statuses.append("skip")
                continue
            results = {}
            for mode in ("preserve", "ftz"):
                status, words, detail, modes = run_mode(args.dxc, spirv_val, host, device, vs, work, mode,
                                                        constants, texture)
                print(f"  -denorm {mode} (SPIR-V execution modes: {', '.join(modes) or 'none'}), ps_6_2:")
                if status != "ok":
                    print(f"    {'validation refusal' if status == 'refused' else 'error'}: {detail}")
                    results[mode] = status
                    continue
                results[mode] = words
                for (name, a, b, op, want), got in zip(CASES, words):
                    print(f"    {name:12} {a:08x} {op} {b:08x} -> {got:08x} (IEEE {want:08x}) "
                          f"{'match' if got == want else 'DIFFER'}")
            status, preserved, differing = classify(results["preserve"], results["ftz"])
            if status == "refused":
                print("  validation refusal: reported apart from mismatches")
            elif status != "error":
                print(f"  preserve: {'conforms to IEEE on all cases' if preserved else 'MISMATCH'}")
                if differing is not None:
                    print(f"  ftz control: {len(differing)} of {len(CASES)} cases differ "
                          f"({', '.join(differing) or 'none'})")
                    if not differing and preserved:
                        print("  FAIL: the ftz control matched IEEE, so this probe cannot tell preserve from flush")
            statuses.append(status)
    if "mismatch" in statuses or "error" in statuses:
        raise SystemExit(1)
    if "refused" in statuses:
        return 78
    if all(s == "skip" for s in statuses):
        return 77
    print(f"{len(CASES)} cases per device checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
