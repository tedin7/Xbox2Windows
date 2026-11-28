"""Wave32 readback of direct compute 64-bit integer comparisons."""
import operator
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


ZERO, LITERAL, NEG_ONE = 128, 255, 193
VCC_LO, EXEC_LO, SDST = 106, 126, 14
LANES, GROUPS, GUARD_WORDS, OUTPUT_WORDS = 32, 2, 8, 2064
MASK32, SENTINEL, OLD_VCC = 0xFFFFFFFF, 0x69696969, 0xAAAAAAAA
PREDICATES = {
    "lt": (1, operator.lt), "eq": (2, operator.eq), "le": (3, operator.le),
    "gt": (4, operator.gt), "ne": (5, operator.ne), "ge": (6, operator.ge),
}

# Every case must fail on main at the measured compare with "direct comparison width".
CASES = (
    ("ne", True, "e32", "pair"),
    ("eq", True, "e32", "pair"),
    ("lt", True, "e32", "sign"),
    ("lt", False, "e32", "sign"),
    ("eq", True, "e32", "inline"),
    ("eq", False, "e64", "inline"),
    ("le", True, "e64", "pair"),
    ("gt", False, "e64", "sign"),
    ("ge", False, "e64", "pair"),
    ("ne", True, "cmpx", "pair"),
)


def vopc(op, src0, vgpr):
    return 0x7C000000 | op << 17 | vgpr << 9 | src0


def program(predicate, signed, form, inputs):
    # group.x is s5 (IDs 0 and 1), and local.x is v0 (IDs 0 through 31).
    # Byte address = 32 + 16 * (32 * group.x + local.x).
    # Readback word index = byte address / 4, including the 8-word guard.
    words = [
        sop1(3, EXEC_LO, NEG_ONE),
        sop1(3, 20, 4), sop1(3, 21, ZERO),
        0xF4080000 | 32 << 6 | 10, 0xFA000020,
        scalar(0x1E, 7, 5, 137), vop2(0x1A, 8, 132, 0), vop2(0x25, 8, 7, 8),
        scalar(0, 10, 5, 135), vmove(6, 256),  # s10 = group.x + 7, v6 = local.x
    ]
    src0, literal = 10, None
    if inputs == "pair":
        words += [scalar(0, 11, 5, 129), vmove(7, 11)]
    elif inputs == "sign":
        words += [sop1(3, 11, ZERO), scalar(0x1E, 12, 5, 159), vmove(7, 12)]
    else:
        src0 = NEG_ONE if inputs == "inline" else LITERAL
        low = MASK32 if inputs == "inline" else 0x80000001
        literal = low if inputs == "literal" else None
        words += [
            vmove(6, LITERAL), low, vmove(9, NEG_ONE),
            vopc(0xC4, 10, 0),  # vcc = group.x + 7 > local.x
            vop2(1, 7, ZERO, 9),  # high word = 0xffffffff on selected lanes
        ]

    words += [sop1(3, VCC_LO, LITERAL), OLD_VCC]
    opcode = (0xA0 if signed else 0xE0) + PREDICATES[predicate][0]
    if form == "e64":
        words += [0xD4000000 | opcode << 16 | SDST, src0 | (256 + 6) << 9]
    else:
        words.append(vopc(opcode + (0x10 if form == "cmpx" else 0), src0, 6))
    if literal is not None:
        words.append(literal)

    if form == "cmpx":
        words += [sop1(3, SDST, EXEC_LO), sop1(3, EXEC_LO, NEG_ONE)]
    dst = VCC_LO if form == "e32" else SDST
    words += [
        vmove(2, dst), vmove(9, 170),
        vop2(1, 4, 145, 9),  # select on VCC before copying an explicit result
    ]
    if dst != VCC_LO:
        words.append(sop1(3, VCC_LO, dst))
    words += [
        vop2(1, 3, 145, 9), vop2(0x16, 5, 132, 8),
        0xE0781000, 0x80080208, 0xBF810000,
    ]
    return words


def reference(predicate, signed, form, inputs):
    def integer(bits):
        return bits - (1 << 64) if signed and bits >> 63 else bits

    expected = [SENTINEL] * OUTPUT_WORDS
    compare = PREDICATES[predicate][1]
    for group in range(GROUPS):
        pivot = group + 7
        conditions = []
        for lane in range(LANES):
            if inputs == "pair":
                high = (group + 1) << 32
                lhs, rhs = high | pivot, high | lane
            elif inputs == "sign":
                lhs, rhs = pivot, (group << 63) | lane
            else:
                low = MASK32 if inputs == "inline" else 0x80000001
                # An inline constant is sign-extended to 64 bits for both i64 and u64.
                lhs = (MASK32 << 32) | low
                rhs = (MASK32 << 32 if lane < pivot else 0) | low
            conditions.append(compare(integer(lhs), integer(rhs)))
        mask = sum(bit << lane for lane, bit in enumerate(conditions))
        for lane, bit in enumerate(conditions):
            item = group * LANES + lane
            index = GUARD_WORDS + 4 * item
            old_bit = bit if form == "e32" else (OLD_VCC >> lane) & 1
            expected[index:index + 4] = [mask, 42 if bit else 17, 42 if old_bit else 17, item]
    return tuple(expected)


def main(replay, sdk):
    # The request selects wave32 with .sa dword 4 bit 20.
    sa = struct.pack("<7I", 0, 0, 0, LANES | 1 << 16, 1 | 1 << 20, 0, 0x8A)
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source, root = work / "root.hlsl", work / "root.bin"
        source.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, source], check=True)
        for case in CASES:
            marker = write_request(work, program(*case), root.read_bytes(), sa)
            output = work / "readback.bin"
            result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                print(result.stdout, result.stderr)
                return 77
            assert result.returncode == 0, (case, result.stdout, result.stderr)
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            expected = reference(*case)
            for index, (got, want) in enumerate(zip(actual, expected)):
                assert got == want, (case, index, hex(got), hex(want))
            print(f"{case}: masks, lane selects, addresses and guards matched")
        # A 32-bit literal on a 64-bit integer compare stays refused.
        for signed in (True, False):
            marker = write_request(work, program("eq", signed, "e32", "literal"), root.read_bytes(), sa)
            result = subprocess.run([replay, sdk, marker, work / "cache", "0", work / "refused.bin", "--direct"],
                                    capture_output=True, text=True)
            assert result.returncode != 0 and "direct 64-bit integer comparison operand" in result.stdout + result.stderr, (
                result.stdout, result.stderr)
        print("literal sources refused")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
