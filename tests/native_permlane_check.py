"""Direct compute permlane with sparse guest EXEC."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_direct_container_check import compute
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1


SENTINEL = 0x69696969
ACTIVE = 0xFFFFFEFD  # Disable physical lanes 1 and 8 in each wave32.
SELECTORS = (1, 8, 3, 0, 15, 2, 9, 4, 8, 1, 14, 5, 2, 11, 6, 7)

# VOP3A word 0: opcode 25:16 (377/378), FI 11, BOUND_CTRL 12, v1 destination.
# Word 1: S0 8:0 (258 = v2), S1 17:9, S2 26:18. A selector of 255
# references the shared third word. Keep the CPU model separate from the machine words.
CASES = (
    ("cross-literal-bound", True, False, True, SELECTORS,
     (0xD7781001, 0x0035FF02, 0x492F0381)),  # x16 BC, v2/literal/s13
    ("cross-sgprs-fi", True, True, False, SELECTORS,
     (0xD7780801, 0x00341902)),  # x16 FI, v2/s12/s13
    ("cross-inline-zero", True, False, False, (0,) * 16,
     (0xD7780001, 0x02010102)),  # x16, v2/inline 0/inline 0
    ("same-literal-bound", False, False, True, SELECTORS,
     (0xD7771001, 0x0035FF02, 0x492F0381)),  # permlane16 BC, v2/literal/s13
    ("same-sgprs", False, False, False, SELECTORS,
     (0xD7770001, 0x00341902)),  # permlane16, v2/s12/s13
    ("cross-s2-literal", True, False, False, SELECTORS,
     (0xD7780001, 0x03FC1902, 0x76B25E18)),  # x16, v2/s12/literal
    ("cross-shared-literal", True, False, False, (1, 8, 3, 0, 15, 2, 9, 4) * 2,
     (0xD7780001, 0x03FDFF02, 0x492F0381)),  # x16, v2/literal/literal
)


def program(instruction):
    # Each group writes 64 words starting after eight guards.
    # Initialize the destination and a distinct source while EXEC is full.
    return [
        sop1(3, 20, 4), sop1(3, 21, 128), 0xF408080A, 0xFA000020,
        scalar(0x1E, 7, 5, 136), vop2(0x1A, 8, 130, 0), vop2(0x25, 8, 7, 8),
        vop2(0x25, 1, 145, 0), vop2(0x25, 2, 148, 0),
        sop1(3, 12, 255), 0x492F0381, sop1(3, 13, 255), 0x76B25E18,
        sop1(3, 126, 255), ACTIVE, *instruction,
        # Restore EXEC so the store observes preserved destinations too.
        sop1(3, 126, 193), 0xE0701000, 0x80080108, 0xBF810000,
    ]


def reference(cross_row, fi, bound_zero, selectors):
    result = list(range(17, 81))
    enabled = [bool(ACTIVE & (1 << (lane % 32))) for lane in range(64)]
    for half in (0, 32):
        rows = (half, half + 16)
        source_rows = rows[::-1] if cross_row else rows
        for destination, source in zip(rows, source_rows):
            for offset, peer in enumerate(selectors):
                lane = destination + offset
                if enabled[lane]:
                    if fi or enabled[source + peer]:
                        result[lane] = 20 + source + peer
                    elif bound_zero:
                        result[lane] = 0
    return tuple(result)


def request(work, root, instruction, threads=64):
    sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), 1 | (1 << 20), 0, 0x8A)  # bit 20: wave32 (#2999)
    return write_request(work, program(instruction), root, sa)


def controls(cli, sdk, work, root):
    requests = work / "requests"
    requests.mkdir()
    good = request(requests, root, CASES[0][-1])
    refused = {}
    for instruction, threads, error in (
            ((0xD7782001, 0x00341902), 64, "unsupported permlane modifier or op_sel"),  # x16 op_sel[2]
            ((0xD7788001, 0x00341902), 64, "unsupported permlane modifier or op_sel"),  # x16 clamp
            ((0xD7780001, 0x00360702), 64, "permlane selectors must be uniform"),  # v2/v3/s13
            ((0xD7780001, 0x00341902), 16, "thread count divisible by 32"),  # x16 partial pair
            ((0xD7770001, 0x00341902), 8, "thread count divisible by 16"),  # permlane16 partial row
            ((0xD7780001, 0x00341880), 64, "S0 must be a VGPR")):  # inline 0/s12/s13
        refused[request(requests, root, instruction, threads)] = error
    result = subprocess.run(
        [cli, sdk, work / "cache", "gfx1013-direct-cs@0", "--requests", requests, requests],
        capture_output=True, text=True)
    assert result.returncode and "requests=7 compiled=1 refused=6" in result.stdout, (
        result.stdout, result.stderr)
    good_key = good.stem.removeprefix("req_")
    assert (requests / f"{good_key}.cs.cso").is_file()
    for marker, error in refused.items():
        key = marker.stem.removeprefix("req_")
        reason = (requests / f"{key}.fail").read_text(encoding="utf-8")
        assert error in reason, (marker, reason)
        assert not (requests / f"{key}.cs.cso").exists()

    # Compile a container without its wave32 flag to exercise split guest wave64.
    # Give EXEC halves distinct values, then restore both after the permutation.
    shader = work / "wave64.cdx"
    output = shader.with_suffix(".cso")
    words = [vop2(0x25, 1, 145, 0), vop2(0x25, 2, 148, 0),
             sop1(3, 13, 255), 0x76B25E18,
             sop1(3, 126, 255), ACTIVE, sop1(3, 127, 255), 0xFFFFFF7B,
             *CASES[0][-1], sop1(4, 126, 193), 0xBF810000]
    shader.write_bytes(compute(struct.pack(f"<{len(words)}I", *words), 64, wave32=False))
    result = subprocess.run([cli, sdk, work / "cache", "gfx1013-container-cs", shader, output],
                            capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert output.read_bytes().startswith(b"DXBC")
    return 0


def hardware(replay, sdk, work, root):
    for name, cross_row, fi, bound_zero, selectors, instruction in CASES:
        marker = request(work, root, instruction)
        output = marker.with_suffix(".bin")
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        lanes = reference(cross_row, fi, bound_zero, selectors)
        expected = (SENTINEL,) * 8 + lanes * 2 + (SENTINEL,) * (2064 - 136)
        actual = struct.unpack("<2064I", output.read_bytes())
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert got == want, (name, index, hex(got), hex(want))
        print(f"{name}: 128 results and guards matched")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / "root.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
