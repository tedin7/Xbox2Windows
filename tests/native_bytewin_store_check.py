"""Read back direct compute byte-window stores and atomics, and check descriptor refusals."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


LANES = 64
LANE_BYTES = 16
PLANE_BYTES = LANES * LANE_BYTES
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969
STORE_OPS = (0xE0701000, 0xE0741000, 0xE07C1000, 0xE0781000)
ATOMIC_ADD = 0xE0C80000
ATOMIC_ADD_X2 = 0xE1480000
ATOMIC_UMIN = 0xE0D80000
ATOMIC_CMPSWAP = 0xE0C40000
OFFEN, IDXEN, GLC = 0x1000, 0x2000, 0x4000


def request(work, root, program):
    sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1 | 1 << 20, 0, 0x8A)
    return write_request(work, program, root, sa)


def compute_program(*, constant_stride=False, foreign_word=False, oob=0x30000000, ops=STORE_OPS):
    # Root table 0 is in s4. Load slot 1 into s[8:11] through s[20:21].
    program = [sop1(3, 20, 4), sop1(3, 21, 128), 0xF4080000 | 8 << 6 | 10, 0xFA000020]
    count_word = 10
    stride_word = 9
    if foreign_word:
        # Give word 2 a valid byte_count proof, but from slot 2.
        program += [0xF4080000 | 12 << 6 | 10, 0xFA000040]
        count_word = 14
        stride_word = 13
    program += [
        scalar(0x27, 18, stride_word, 255), 0xE0010,  # s_bfe_u32: extract the stride
        scalar(0x09, 18, 18, 129),                   # s_max_u32: clamp the stride to 1
        scalar(0x0E, 9, 9, 255), 0xC000FFFF,        # s_and_b32: clear the stride
    ]
    if constant_stride:
        program += [sop1(3, 22, 132)]                # Plain SGPR constant 4.
    program += [
        scalar(0x26, 10, count_word, 22 if constant_stride else 18),  # s_mul_i32
        scalar(0x10, 11, 11, 255), oob,                            # s_or_b32
    ]

    # v8 = 512 * group.x + 16 * lane.x. The replay view begins after its leading guard.
    program += [scalar(0x1E, 7, 5, 137), vop2(0x1A, 8, 132, 0), vop2(0x25, 8, 7, 8)]
    for component in range(4):
        program.append(vop2(0x25, 2 + component, 129 + component, 8))
    # One 1024-byte plane per width. The last immediate is 3072, within 12 bits.
    for plane, opcode in enumerate(ops):
        program += [opcode | plane * PLANE_BYTES, 0x80020208]
    return program + [0xBF810000]


def atomic_program():
    # Each plane uses the same raw UAV for the atomic and its VGPR readback.
    program = compute_program(ops=())[:-1]
    for plane, opcode in enumerate((ATOMIC_ADD | GLC, ATOMIC_ADD, ATOMIC_UMIN | GLC, ATOMIC_CMPSWAP | GLC)):
        offset = plane * PLANE_BYTES
        program += [vop2(0x25, 2, 129, 8)]  # v2 = lane byte offset + 1
        if plane == 3:
            program += [vmove(3, 255), SENTINEL]  # v3 is the compare operand.
        program += [opcode | OFFEN | offset, 0x80020208,
                    0xE0701000 | (offset + 4), 0x80020208]
        if plane == 3:
            # Retry with a different replacement and the same now-stale compare.
            program += [vmove(2, 128), opcode | OFFEN | offset, 0x80020208,
                        0xE0701000 | (offset + 8), 0x80020208]
    return program + [0xBF810000]


def atomic_reference():
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(LANES):
        value = LANE_BYTES * lane + 1
        results = ((SENTINEL + value, SENTINEL),
                   (SENTINEL + value, value),
                   (min(SENTINEL, value), SENTINEL),
                   (value, SENTINEL, value))
        for plane, words in enumerate(results):
            start = GUARD_WORDS + (plane * PLANE_BYTES + lane * LANE_BYTES) // 4
            expected[start:start + len(words)] = words
    return expected


def controls(cli, sdk, work, root):
    # Manual mutants that must fail: dropping the clamped-stride proof must
    # accept the constant; ignoring descriptor identity must accept the foreign word.
    for name, program, errors in (
            ("constant stride", compute_program(constant_stride=True),
             ("unbound numeric SGPR s10", "non-contiguous resource descriptor")),
            ("foreign word 2", compute_program(foreign_word=True), ("non-contiguous resource descriptor",)),
            ("wrong OOB bits", compute_program(oob=0x20000000),
             ("unbound numeric SGPR s11", "non-contiguous resource descriptor")),
            ("indexed atomic through byte window", compute_program(ops=(ATOMIC_ADD | IDXEN,)),
             ("non-contiguous resource descriptor",)),
            ("64-bit atomic through byte window", compute_program(ops=(ATOMIC_ADD_X2 | OFFEN,)),
             ("unsupported buffer atomic width or operation",))):
        marker = request(work, root, program)
        output = marker.with_suffix(".cso")
        result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                capture_output=True, text=True)
        assert result.returncode and any(error in result.stderr for error in errors), (name, result.stderr)
        assert not output.exists(), output
        print(f"{name}: refused")
    return 0


def hardware(replay, sdk, work, root):
    store_expected = [SENTINEL] * OUTPUT_WORDS
    for plane, width in enumerate(range(1, 5)):
        for lane in range(LANES):
            start = GUARD_WORDS + (plane * PLANE_BYTES + lane * LANE_BYTES) // 4
            store_expected[start:start + width] = [LANE_BYTES * lane + j + 1 for j in range(width)]
    for name, program, expected in (("stores", compute_program(), store_expected),
                                    ("atomics", atomic_program(), atomic_reference())):
        marker = request(work, root, program)
        output = work / f"{name}.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        for word, (got, want) in enumerate(zip(actual, expected)):
            assert got == want, (name, word, hex(got), hex(want))
        print(f"{name}: updated words, untouched words and guards matched")
    return 0


def main(mode, tool, sdk, *, checks=None):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / "root.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        if checks is None:
            checks = {"controls": controls, "hardware": hardware}
        return checks[mode](tool, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
