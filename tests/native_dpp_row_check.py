"""Direct compute DPP16 row shifts and rotates with sparse guest EXEC."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1


SENTINEL = 0x69696969
ACTIVE = 0xFFFFFEFD  # Disable physical lanes 1 and 8 in each wave32.

# VOP1/VOP2 src0=250. DPP word: VGPR 7:0, DPP_CTRL 16:8, FI 18,
# BOUND_CTRL 19, bank mask 27:24, row mask 31:28.
# Keep the CPU model's parameters separate from these hand-encoded words.
CASES = (
    ("mov", "shl", 1, False, False, 0x7E0202FA, 0xFF010101),  # v_mov_b32_dpp row_shl:1
    ("mov", "shl", 1, False, True,  0x7E0202FA, 0xFF090101),  # v_mov_b32_dpp row_shl:1 BC
    ("mov", "shl", 15, False, False, 0x7E0202FA, 0xFF010F01),  # v_mov_b32_dpp row_shl:15
    ("mov", "shl", 15, False, True,  0x7E0202FA, 0xFF090F01),  # v_mov_b32_dpp row_shl:15 BC
    ("mov", "shr", 1, False, False, 0x7E0202FA, 0xFF011101),  # v_mov_b32_dpp row_shr:1
    ("mov", "shr", 1, False, True,  0x7E0202FA, 0xFF091101),  # v_mov_b32_dpp row_shr:1 BC
    ("mov", "shr", 15, False, False, 0x7E0202FA, 0xFF011F01),  # v_mov_b32_dpp row_shr:15
    ("mov", "shr", 15, False, True,  0x7E0202FA, 0xFF091F01),  # v_mov_b32_dpp row_shr:15 BC
    ("mov", "ror", 1, False, False, 0x7E0202FA, 0xFF012101),  # v_mov_b32_dpp row_ror:1
    ("mov", "ror", 1, False, True,  0x7E0202FA, 0xFF092101),  # v_mov_b32_dpp row_ror:1 BC
    ("mov", "ror", 15, False, False, 0x7E0202FA, 0xFF012F01),  # v_mov_b32_dpp row_ror:15
    ("mov", "ror", 15, False, True,  0x7E0202FA, 0xFF092F01),  # v_mov_b32_dpp row_ror:15 BC
    ("add_u", "shr", 1, True, False, 0x4A0204FA, 0xFF051101),  # v_add_nc_u32_dpp row_shr:1 FI
    ("mov", "shr", 1, True, True, 0x7E0202FA, 0xFF0D1101),  # v_mov_b32_dpp row_shr:1 FI BC
    ("add_f", "ror", 8, False, True, 0x060204FA, 0xFF092801),  # v_add_f32_dpp row_ror:8 BC
    ("min_f", "ror", 8, False, True, 0x1E0204FA, 0xFF092801),  # v_min_f32_dpp row_ror:8 BC
)


def program(op, instruction, control):
    # Each group writes 64 words starting after eight guard words.
    # Seed v1 (also the destination) and v2 under full EXEC.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xF408080A, 0xFA000020,
             scalar(0x1E, 7, 5, 136), vop2(0x1A, 8, 130, 0), vop2(0x25, 8, 7, 8),
             vop2(0x25, 1, 145, 0), vop2(0x25, 2, 148, 0)]
    if op in ("add_f", "min_f"):
        words += [0x7E020D01, 0x7E040D02]  # v_cvt_f32_u32 v1, v1 and v2, v2
    # Restore EXEC before the store so suppressed writes can be checked.
    return words + [sop1(3, 126, 255), ACTIVE, instruction, control,
                    sop1(3, 126, 193), 0xE0701000, 0x80080108, 0xBF810000]


def reference(op, direction, amount, fi, bound_zero):
    initial = list(range(17, 81))
    result = initial.copy()
    enabled = [bool(ACTIVE & (1 << (lane % 32))) for lane in range(64)]
    for start in range(0, 64, 16):
        row = initial[start:start + 16]
        if direction == "shl":
            peers = list(range(amount, 16)) + [None] * amount
        elif direction == "shr":
            peers = [None] * amount + list(range(16 - amount))
        else:
            peers = list(range(16 - amount, 16)) + list(range(16 - amount))
        for offset, peer in enumerate(peers):
            lane = start + offset
            if not enabled[lane]:
                continue
            if peer is not None and (fi or enabled[start + peer]):
                source = row[peer]
            elif bound_zero:
                source = 0
            else:
                continue
            own = lane + 20
            if op == "mov":
                result[lane] = source
            elif op == "min_f":
                result[lane] = min(source, own)
            else:
                result[lane] = source + own
    if op in ("add_f", "min_f"):
        return struct.unpack("<64I", struct.pack("<64f", *result))
    return tuple(result)


def request(work, root, op, instruction, control, threads=64):
    sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), 1, 0, 0x8A)
    return write_request(work, program(op, instruction, control), root, sa)


def controls(cli, sdk, work, root):
    requests = work / "requests"
    requests.mkdir()
    good = request(requests, root, "mov", 0x7E0202FA, 0xFF010101)  # row_shl:1
    refused = {}
    for control, threads, error in (
            (0xFF015001, 64, "unsupported DPP lane selection"),  # row_share:0
            (0xFF014201, 64, "unsupported DPP lane selection"),  # control 0x142
            (0x7F010101, 64, "DPP requires a full-row, full-bank"),  # row_shl:1, row_mask:7
            (0xFF010101, 24, "thread count divisible by 16")):  # row_shl:1, partial row
        marker = request(requests, root, "mov", 0x7E0202FA, control, threads)
        refused[marker] = error
    result = subprocess.run(
        [cli, sdk, work / "cache", "gfx1013-direct-cs@0", "--requests", requests, requests],
        capture_output=True, text=True)
    assert result.returncode and "requests=5 compiled=1 refused=4" in result.stdout, (
        result.stdout, result.stderr)
    good_key = good.stem.removeprefix("req_")
    assert (requests / f"{good_key}.cs.cso").is_file()
    for marker, error in refused.items():
        key = marker.stem.removeprefix("req_")
        reason = (requests / f"{key}.fail").read_text(encoding="utf-8")
        assert error in reason, (marker, reason)
        assert not (requests / f"{key}.cs.cso").exists()
    return 0


def hardware(replay, sdk, work, root):
    for op, direction, amount, fi, bound_zero, instruction, control in CASES:
        marker = request(work, root, op, instruction, control)
        output = marker.with_suffix(".bin")
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        lanes = reference(op, direction, amount, fi, bound_zero)
        expected = (SENTINEL,) * 8 + lanes * 2 + (SENTINEL,) * (2064 - 136)
        actual = struct.unpack("<2064I", output.read_bytes())
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert got == want, (op, hex(control), index, hex(got), hex(want))
        print(f"{op} {hex(control)}: 128 results and guards matched")
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
