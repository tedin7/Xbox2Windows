"""Direct compute 64-bit SALU readback and EXEC destination refusal."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1, vmove


BFM, CSELECT, CSELECT32, NOT = 0x25, 0x0B, 0x0A, 0x08
ZERO, ONE, NEG_ONE, LITERAL = 128, 129, 193, 255
PAIR = (0x89ABCDEF, 0x01234567)
LITERAL_VALUE = 0x80000001
MASK32 = (1 << 32) - 1
LANES = 64
PLANE_BYTES = LANES * 8
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969


def set_pair(words):
    return [sop1(3, 22, LITERAL), words[0], sop1(3, 23, LITERAL), words[1]]


def set_scc(value):
    # s_cmp_eq_u32 0, (0 for true, 1 for false)
    return 0xBF060080 | (ZERO if value else ONE) << 8


def compute_program(instructions, dst):
    # Two 32-lane groups, with 8 bytes per lane and 256 bytes per group in each plane.
    program = [sop1(3, 20, 4), sop1(3, 21, ZERO),
               0xF4080000 | 32 << 6 | 10, 0xFA000020,
               scalar(0x1E, 7, 5, 136), vop2(0x1A, 8, 131, 0), vop2(0x25, 8, 7, 8)]
    program += instructions
    # One pair per lane in each of two planes. A ninth plane would set OFFEN
    # because the store's immediate offset is only 12 bits.
    program += [vmove(2, dst), vmove(3, dst + 1), 0xE0741000, 0x80080208,
                scalar(CSELECT32, 26, ONE, ZERO), vmove(2, 26), vmove(3, ZERO),
                0xE0741000 | PLANE_BYTES, 0x80080208, 0xBF810000]
    return program


def controls(cli, sdk, work, root):
    for words, error in (([scalar(CSELECT, 126, NEG_ONE, ZERO), 0xBF810000], "invalid s_cselect_b64 destination"),
                         ([sop1(NOT, 24, 106), 0xBF810000], "invalid s_not_b64 source")):
        marker = request(work, root, words)
        output = marker.with_suffix(".cso")
        result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                capture_output=True, text=True)
        assert result.returncode and error in result.stderr, (result.stdout, result.stderr)
        assert not output.exists(), output
        print(error)
    return 0


def cases():
    # Manual mutants that must fail: 32-bit s_bfm_b64 with a zero high word,
    # zero-extending -1 in s_cselect_b64, and s_not_b64 SCC from the low word only.
    for width, offset in ((0, 0), (1, 0), (31, 1), (32, 32), (33, 0), (63, 1), (64, 0)):
        bits = ((1 << (width & 63)) - 1) << (offset & 63)
        scc = int(bits == 0)
        # Use odd SGPR s23 for the width and s22 for the offset in the measured case.
        if (width, offset) == (32, 32):
            code = [sop1(3, 23, ZERO + width), sop1(3, 22, ZERO + offset),
                    set_scc(scc), scalar(BFM, 24, 23, 22)]
        else:
            code = [set_scc(scc), scalar(BFM, 24, ZERO + width, ZERO + offset)]
        yield f"bfm {width},{offset}", code, 24, (bits & MASK32, (bits >> 32) & MASK32), scc

    for scc in (0, 1):
        yield (f"cselect pair,-1 SCC={scc}",
               set_pair(PAIR) + [set_scc(scc), scalar(CSELECT, 24, 22, NEG_ONE)],
               24, PAIR if scc else (MASK32, MASK32), scc)
        yield (f"cselect literal,pair SCC={scc}",
               set_pair(PAIR) + [set_scc(scc), scalar(CSELECT, 106, LITERAL, 22), LITERAL_VALUE],
               106, (LITERAL_VALUE, 0) if scc else PAIR, scc)

    for source in (PAIR, (0, 0), (MASK32, MASK32), (MASK32, 0)):
        words = tuple(word ^ MASK32 for word in source)
        scc = int(any(words))
        yield (f"not {source}", set_pair(source) + [set_scc(1 - scc), sop1(NOT, 106, 22)],
               106, words, scc)


def hardware(replay, sdk, work, root):
    for name, code, dst, words, scc in cases():
        marker = request(work, root, compute_program(code, dst))
        output = work / "out.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        end = GUARD_WORDS + LANES * 4
        assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS, name
        assert actual[GUARD_WORDS:end] == words * LANES + (scc, 0) * LANES, (name, words, scc)
        assert actual[end:] == (SENTINEL,) * (OUTPUT_WORDS - end), name
        print(f"{name}: result words, SCC and guards matched")
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
