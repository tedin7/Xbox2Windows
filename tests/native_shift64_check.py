"""Direct compute logical 64-bit shift readback and inline-float refusal."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1


AMOUNTS = (0, 1, 31, 32, 33, 63)
SGPR_LOW, SGPR_HIGH = 0x89ABCDEF, 0x01234567
LITERAL = 0x80000001
SOURCES = (
    ("inline -1", 193, -1),
    ("inline -16", 208, -16),
    ("inline 64", 192, 64),
    ("inline 1", 129, 1),
    ("literal", 255, LITERAL),
    ("SGPR pair", 22, (SGPR_HIGH << 32) | SGPR_LOW),
)
SHIFTS = (("right", 0xD7000002), ("left", 0xD6FF0002))
MASK64 = (1 << 64) - 1
SENTINEL = 0x69696969
GUARD_WORDS = 8
LANES = 64
OUTPUT_WORDS = 2064


def compute_program(amount, sources=SOURCES):
    # Use the quad fixture's descriptor setup and group index. v8 holds the
    # byte offset of a lane's two words within a plane of 64 result pairs: 8 bytes per lane, 256 per group.
    program = [sop1(3, 20, 4), sop1(3, 21, 128),
               0xF4080000 | 32 << 6 | 10, 0xFA000020,
               scalar(0x1e, 7, 5, 136), vop2(0x1a, 8, 131, 0), vop2(0x25, 8, 7, 8),
               sop1(3, 22, 255), SGPR_LOW, sop1(3, 23, 255), SGPR_HIGH]
    # Replay's UAV view already starts after the leading guard words.
    offset = 0
    for _, source, _ in sources:
        for _, opcode in SHIFTS:
            program += [opcode, (128 + amount) | (source << 9)]
            if source == 255:
                program.append(LITERAL)
            # buffer_store_dwordx2 v[2:3], v8, s[32:35], 0 offen
            program += [0xE0741000 | offset, 0x80080208]
            offset += LANES * 8
    return program + [0xBF810000]


def controls(cli, sdk, work, root):
    marker = request(work, root, compute_program(1, (("inline float", 242, None),)))
    output = marker.with_suffix(".cso")
    result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                            capture_output=True, text=True)
    error = "direct 64-bit shift operand"
    assert result.returncode and error in result.stderr, (result.stdout, result.stderr)
    assert not output.exists(), output
    print(error)
    return 0


def hardware(replay, sdk, work, root):
    # Manual mutants that must fail: zero-extend the negative inline values,
    # sign-extend the literal, or swap the SGPR pair's words.
    # One source per dispatch: the store's immediate offset is 12 bits. All twelve planes in one dispatch
    # would not fit: the ninth starts at 4096, which is the OFFEN bit.
    for amount in AMOUNTS:
        for source in SOURCES:
            marker = request(work, root, compute_program(amount, (source,)))
            output = work / "out.bin"
            result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS
            start = GUARD_WORDS
            name, _, value = source
            bits = value & MASK64
            for (direction, _), expected in zip(SHIFTS, (bits >> amount, (bits << amount) & MASK64)):
                words = (expected & 0xFFFFFFFF, expected >> 32)
                end = start + LANES * 2
                assert actual[start:end] == words * LANES, (name, direction, amount, words, actual[start:end])
                start = end
            assert actual[start:] == (SENTINEL,) * (OUTPUT_WORDS - start)
        print(f"Shift by {amount}: all 6 sources, both directions and guards matched")
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
