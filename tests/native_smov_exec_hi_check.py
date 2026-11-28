"""Exercise EXEC_HI scalar moves in the direct compute frontend."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_direct_container_check import compute, words
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


EXEC_LO, EXEC_HI = 126, 127
END = 0xBF810000
THREADS = 32
LANES = 2 * THREADS
LOW_MASK = 0x9
VALUES = (0x550055, 5, 0x80000002)
GUARD = 0x69696969
OUTPUT_WORDS = 2064


def mov(dst, src):
    return sop1(3, dst, src)


def program():
    # Load the output UAV from the table in s4. v8 addresses one dword per lane.
    code = [
        mov(20, 4), mov(21, 128), 0xF408080A, 0xFA000020,
        scalar(0x1E, 7, 5, 135),  # group.x * 128
        vop2(0x1A, 8, 130, 0), vop2(0x25, 8, 7, 8),
        mov(EXEC_LO, 128 + LOW_MASK),
        mov(EXEC_HI, 255), VALUES[0], mov(12, EXEC_HI),
        0xE0701000, 0x80080008,  # store local.x under the narrowed EXEC
        mov(EXEC_HI, 128 + VALUES[1]), mov(13, EXEC_HI),
        mov(14, 255), VALUES[2], mov(EXEC_HI, 14), mov(15, EXEC_HI),
        mov(EXEC_LO, 193),  # restore all wave32 lanes for the readbacks
    ]
    for plane, reg in enumerate((12, 13, 15), start=1):
        code.extend((vmove(4, reg), 0xE0701000 | (4 * LANES * plane), 0x80080408))
    return code + [END]


def reference():
    masked = tuple(lane if LOW_MASK & (1 << lane) else GUARD for lane in range(THREADS))
    readbacks = tuple(value for value in VALUES for _ in range(LANES))
    return (GUARD,) * 8 + masked * 2 + readbacks + (GUARD,) * (OUTPUT_WORDS - 8 - 4 * LANES)


def hardware(replay, sdk, work):
    source = work / "root.hlsl"
    root = work / "root.bin"
    source.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
    subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                    "-Fo", root, source], check=True)
    sa = struct.pack("<7I", 0, 0, 0, THREADS | (1 << 16), 1 | (1 << 20), 0, 0x8A)
    marker = write_request(work, program(), root.read_bytes(), sa)
    output = work / "readback.bin"
    result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
    for index, (got, want) in enumerate(zip(actual, reference())):
        assert got == want, f"word {index}: got {got:#010x}, expected {want:#010x}"
    return 0


def wave64(cli, sdk, work):
    source = work / "wave64"
    output = work / "wave64.cso"
    code = words(mov(EXEC_LO, 129), mov(EXEC_HI, 255), VALUES[0], mov(12, EXEC_HI),
                 mov(EXEC_HI, 128 + VALUES[1]), mov(13, EXEC_HI),
                 mov(14, 255), VALUES[2], mov(EXEC_HI, 14), mov(15, EXEC_HI), vmove(4, 15), END)
    source.write_bytes(compute(code, threads=64, wave32=False))
    subprocess.run([cli, sdk, work / "cache", "gfx1013-container-cs", source, output], check=True)
    assert output.read_bytes().startswith(b"DXBC")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        return {"hardware": hardware, "wave64": wave64}[mode](tool, sdk, Path(temp))


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
