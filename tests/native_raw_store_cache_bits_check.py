"""Read back direct raw stores with cache hints and check modifier refusals."""
import struct
import subprocess
import sys

from native_bytewin_store_check import (
    GLC, GUARD_WORDS, IDXEN, LANE_BYTES, LANES, OUTPUT_WORDS, PLANE_BYTES, SENTINEL,
    STORE_OPS, compute_program, main, request, sop1,
)


DLC = 1 << 15
LDS = 1 << 16
SLC = 1 << 22
TFE = 1 << 23
DWORD = STORE_OPS[0]
DWORDX4 = STORE_OPS[3]
REFUSAL = "raw store needs a linear byte offset"


def program(stores):
    code = compute_program(ops=())[:-1]
    # Groups (0,0,0) and (1,0,0) each run one wave32.
    # Enable local lanes 0..7 in each group after initializing the data VGPRs.
    code += [sop1(3, 126, 255), 0xff, sop1(3, 127, 128)]
    for plane, (opcode, word1_flags) in enumerate(stores):
        # One 1024-byte plane per store, 16 bytes per lane, including inactive lanes.
        code += [opcode | plane * PLANE_BYTES, 0x80020208 | word1_flags]
    return code + [0xBF810000]


def controls(cli, sdk, work, root):
    for name, opcode, word1_flags in (("idxen", DWORD | GLC | IDXEN, 0),
                                      ("lds", DWORD | GLC | LDS, 0),
                                      ("tfe", DWORDX4 | GLC, TFE)):
        marker = request(work, root, program(((opcode, word1_flags),)))
        output = marker.with_suffix(".cso")
        result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                capture_output=True, text=True)
        assert result.returncode and REFUSAL in result.stderr and not output.exists(), (name, result.stderr)
        print(f"{name}: refused")
    return 0


def hardware(replay, sdk, work, root):
    stores = ((DWORD | GLC, 0), (DWORDX4 | GLC, 0), (DWORD | DLC, 0), (DWORDX4, SLC))
    marker = request(work, root, program(stores))
    output = work / "stores.bin"
    result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
    if result.returncode == 77:
        return 77
    result.check_returncode()

    expected = [SENTINEL] * OUTPUT_WORDS
    for plane, width in enumerate((1, 4, 1, 4)):
        for lane in range(LANES):
            if lane % 32 < 8:
                start = GUARD_WORDS + (plane * PLANE_BYTES + lane * LANE_BYTES) // 4
                expected[start:start + width] = [LANE_BYTES * lane + j + 1 for j in range(width)]
    actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
    for word, (got, want) in enumerate(zip(actual, expected)):
        assert got == want, (word, hex(got), hex(want))
    print("cache-hinted stores: updated words, untouched words and guards matched")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:], checks={"controls": controls, "hardware": hardware}))
