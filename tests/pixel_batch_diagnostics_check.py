"""Exercise per-case D3D12 errors in a batch of three matching pixel draws."""
from pathlib import Path
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory


def verify(dxc, renderer, adapter, control, directory):
    source, shader = directory / "color.hlsl", directory / "color.dxil"
    source.write_text("cbuffer Params : register(b0) { float4 color; };\n"
                      "float4 main() : SV_Target { return color; }\n")
    subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    cases = []
    for index in range(3):
        color = struct.pack("<4f", index, index + 1, 0.5, 1)
        path = directory / f"{index}.case"
        path.write_bytes(color.ljust(256, b"\0") + color * 64)
        cases.append(path)

    def batch(operation):
        return subprocess.run([renderer, adapter, operation, shader, *cases],
                              capture_output=True, text=True, timeout=25)

    def check_status(result, expected, exit_code):
        lines = [line for line in result.stdout.splitlines() if line.startswith("case=")]
        if lines != [f"case={i} status={s}" for i, s in enumerate(expected)] or result.returncode != exit_code:
            raise AssertionError(result.stdout + result.stderr)

    normal = batch("--check-pixel-batch")
    if adapter == "hardware" and normal.returncode == 77:
        print(normal.stdout + normal.stderr)
        return 77
    check_status(normal, ["pass"] * 3, 0)

    result = batch(f"--pixel-batch-{control}-control")
    print(result.stdout + result.stderr)
    skip = {"debug": "the D3D12 debug layer", "removal": "ID3D12Device5"}[control]
    if result.returncode == 77 and f"skip: pixel batch control requires {skip}" in result.stdout.splitlines():
        if control == "debug" and "debug_layer=unavailable" not in result.stdout:
            raise AssertionError(result.stdout)
        return 77
    check_status(result, ["pass", "error", "pass" if control == "debug" else "error"], 1)

    # Keep the setup diagnostics and all three cases separate on stderr.
    sections = re.split(r"^case=\d+ path=.*\n", result.stderr, flags=re.MULTILINE)
    if len(sections) != 4:
        raise AssertionError(result.stderr)
    if control == "debug":
        counts = re.findall(r"^debug_errors=(\d+) debug_warnings=\d+\ncase=\d+ status=", result.stdout, re.MULTILINE)
        if counts != ["0", "1", "0"] or [s.count("pixel batch control error") for s in sections] != [0, 0, 1, 0]:
            raise AssertionError(result.stdout + result.stderr)
    elif "Pixel batch device failed (HRESULT 0x887a0005)" not in sections[3]:
        raise AssertionError(result.stderr)
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        sys.exit(verify(*sys.argv[1:], Path(folder)))
