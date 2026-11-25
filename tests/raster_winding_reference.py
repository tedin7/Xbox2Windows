"""Check D3D12 culling and front-face winding against two mirrored triangles."""
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory


VERTEX = """
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 points[3] = {float2(-1, 1), float2(-0.0625, 1), float2(-1, -1)};
    float2 p = points[id % 3];
    if (id >= 3) p.x = -p.x;
    return float4(p, 0.5, 1);
}
"""
PIXEL = """
float4 main(bool front : SV_IsFrontFace) : SV_Target {
    return front ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
"""
# The left triangle's screen vertices are (0,0), (3.75,0) and (0,4).
# The right triangle is its mirror across x=4. No pixel center lies on a diagonal.
COVERAGE = ("LLL..RRR", "LL....RR", "L......R", "........")
COLORS = {".": (-1.,) * 4, "F": (0., 1., 0., 1.), "B": (1., 0., 0., 1.)}
EXPECTED = {
    ("none", "cw"): "FB", ("front", "cw"): ".B", ("back", "cw"): "F.",
    ("none", "ccw"): "BF", ("front", "ccw"): "B.", ("back", "ccw"): ".F",
}


def reference(pair):
    labels = {".": ".", "L": pair[0], "R": pair[1]}
    return b"".join(struct.pack("<4f", *COLORS[labels[pixel]]) for row in COVERAGE for pixel in row)


def verify(dxc, renderer, adapter, directory):
    compiled = []
    for name, text in (("vs", VERTEX), ("ps", PIXEL)):
        source, shader = directory / f"{name}.hlsl", directory / f"{name}.cso"
        source.write_text(text)
        subprocess.run([dxc, "-E", "main", "-T", f"{name}_6_0", "-Fo", shader, source], check=True)
        compiled.append(shader)

    fixture = directory / "expected.bin"
    command = [renderer, adapter, "--raster-reference", *compiled, "8", "4", fixture]
    def draw(state, pair, mismatch, show_adapter=False):
        fixture.write_bytes(reference(pair))
        result = subprocess.run([*command, *state], capture_output=True, text=True, timeout=20)
        report = result.stdout + result.stderr
        if adapter == "hardware" and result.returncode == 77 and "No D3D12-capable hardware adapter" in report:
            print(report)
            raise SystemExit(77)
        diagnostic = f"{mismatch} of 32 pixels differ" if mismatch else "all 32 pixels matched"
        if result.returncode != int(bool(mismatch)) or diagnostic not in report.splitlines():
            raise AssertionError(report)
        if "debug_layer=enabled" in report and "debug_errors=0 debug_warnings=0" not in report.splitlines():
            raise AssertionError(report)
        if show_adapter:
            print("\n".join(line for line in report.splitlines()
                            if line.startswith(("backend=", "adapter=", "driver=", "debug_errors="))))
            # Some hosted adapters enumerate as hardware but identify themselves as the Basic Render Driver.
            if adapter == "hardware" and " vendor=1414 device=008c" in report:
                print("skip: the hardware selection returned the Microsoft Basic Render Driver")
                raise SystemExit(77)

    for state, pair in EXPECTED.items():
        draw(state, pair, 0, show_adapter=(state == ("none", "cw")))
        print(f"{adapter} {state}: 32 pixels matched")
    for changed, expected in (
        (("none", "cw"), ("front", "cw")),
        (("none", "cw"), ("back", "cw")),
        (("back", "cw"), ("front", "cw")),
        (("back", "cw"), ("back", "ccw")),
    ):
        # Each covered triangle occupies six pixels; each wrong face changes all six.
        mismatch = 6 * sum(a != b for a, b in zip(EXPECTED[changed], EXPECTED[expected]))
        draw(changed, EXPECTED[expected], mismatch)
        print(f"{adapter} {changed} against {expected}: {mismatch} of 32 pixels differ")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        verify(*sys.argv[1:], Path(directory))
