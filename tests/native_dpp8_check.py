"""Wave32 hardware readback for direct compute DPP8."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


ADD, MIN, MAX = 0x03, 0x0F, 0x10
SWAP = (4, 5, 6, 7, 0, 1, 2, 3)
ASYMMETRIC = (7, 0, 3, 1, 6, 2, 5, 4)
ALL = 0xFFFFFFFF
SPARSE = 0x55555555
SENTINEL = 0x69696969


def program(op, selectors, active, fi):
    # Each workgroup writes 32 words, starting after eight guard words.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xF408080A, 0xFA000020,
             scalar(0x1E, 7, 5, 135), vop2(0x1A, 8, 130, 0), vop2(0x25, 8, 7, 8),
             vop2(0x25, 9, 145, 0), 0x7E000000 | 9 << 17 | 6 << 9 | 265]
    # Seed v9 under full EXEC and alias both inputs with the destination.
    # Restore full EXEC before storing so disabled destinations are visible.
    src0 = 234 if fi else 233
    instruction = vmove(9, src0) if op is None else vop2(op, 9, src0, 9)
    control = 9 | sum(peer << (8 + 3 * lane) for lane, peer in enumerate(selectors))
    return words + [sop1(3, 126, 255), active, instruction, control,
                    sop1(3, 126, 193), 0xE0701000, 0x80080908, 0xBF810000]


def reference(op, selectors, active, fi):
    initial = list(range(17, 49))
    result = initial.copy()
    for start in range(0, 32, 8):
        for offset, selection in enumerate(selectors):
            lane, peer = start + offset, start + selection
            if not active & (1 << lane):
                continue
            source = initial[peer] if fi or active & (1 << peer) else 0
            if op == ADD:
                result[lane] = source + initial[lane]
            elif op == MIN:
                result[lane] = min(source, initial[lane])
            elif op == MAX:
                result[lane] = max(source, initial[lane])
            else:
                result[lane] = source
    return struct.unpack("<32I", struct.pack("<32f", *result))


def hardware(replay, sdk, work, root):
    # Unique values in all four octets expose a wrong group base. In each
    # sparse case lane 0 selects disabled lane 7, exposing ignored FI.
    cases = (
        ("min swap", MIN, SWAP, ALL, False),
        ("add swap", ADD, SWAP, ALL, False),
        ("max swap", MAX, SWAP, ALL, False),
        ("mov asymmetric", None, ASYMMETRIC, ALL, False),
        ("add sparse without FI", ADD, ASYMMETRIC, SPARSE, False),
        # min(0, x) differs from the old destination, so it separates "zero" from "keep".
        ("min sparse without FI", MIN, ASYMMETRIC, SPARSE, False),
        ("add sparse with FI", ADD, ASYMMETRIC, SPARSE, True),
    )
    sa = struct.pack("<7I", 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x8A)
    for name, op, selectors, active, fi in cases:
        marker = write_request(work, program(op, selectors, active, fi), root, sa)
        output = work / "out.bin"
        run = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if run.returncode == 77:
            return 77
        run.check_returncode()
        lanes = reference(op, selectors, active, fi)
        expected = (SENTINEL,) * 8 + lanes * 2 + (SENTINEL,) * (2064 - 72)
        actual = struct.unpack("<2064I", output.read_bytes())
        for index, (observed, wanted) in enumerate(zip(actual, expected)):
            assert observed == wanted, (name, index, hex(observed), hex(wanted))
        print(f"{name}: 64 results and guards matched")
    return 0


def main(replay, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / "root.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        return hardware(replay, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
