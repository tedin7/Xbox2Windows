"""Check slot 67 wave64 WREXEC lane stores, destination words and SCC."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


MOV_B32, ANDN2_WREXEC_B64 = 3, 0x3A
ZERO, ONE, NEG_ONE, LITERAL = 128, 129, 193, 255
EXEC_LO, EXEC_HI = 126, 127
OLD_EXEC = (0x80000012, 0x40000009)
GUARD_WORDS, OUTPUT_WORDS = 8, 2064
GROUP_WORDS, PLANE_WORDS = 128, 256
SENTINEL = 0x69696969
CASES = (
    # X, source words, destination, incoming SCC, expected result, expected SCC
    (64, (0x80000084, 0xC0000021), 24, 0, (0x84, 0x80000020), 1),
    (128, (0x80000084, 0xC0000021), 24, 0, (0x84, 0x80000020), 1),
    (128, (0x80000002, 0xC0000021), 22, 0, (0, 0x80000020), 1),
    (64, OLD_EXEC, 24, 1, (0, 0), 0),
)


def program(source, dst, incoming_scc, src=22):
    # s4 holds the table, s5 holds group.x and v0 holds local.x.
    # v8 = 512 * group.x + 4 * local.x, calculated before narrowing EXEC.
    words = [
        sop1(MOV_B32, 20, 4), sop1(MOV_B32, 21, ZERO),
        0xF408080A, 0xFA000020,
        scalar(0x1E, 7, 5, ZERO + 9),
        vop2(0x1A, 8, ZERO + 2, 0), vop2(0x25, 8, 7, 8),
        0xBF8CC07F,
        sop1(MOV_B32, EXEC_LO, LITERAL), OLD_EXEC[0],
        sop1(MOV_B32, EXEC_HI, LITERAL), OLD_EXEC[1],
        sop1(MOV_B32, 22, LITERAL), source[0],
        sop1(MOV_B32, 23, LITERAL), source[1],
        0xBF060080 | (ZERO if incoming_scc else ONE) << 8,  # s_cmp_eq_u32
        sop1(ANDN2_WREXEC_B64, dst, src),
        0xE0701000, 0x80080008,  # store v0 under the new EXEC
        scalar(0x0A, 26, ONE, ZERO),  # s_cselect_b32 captures SCC
        sop1(MOV_B32, EXEC_LO, NEG_ONE), sop1(MOV_B32, EXEC_HI, NEG_ONE),
    ]
    for plane, reg in enumerate((dst, dst + 1, 26), start=1):
        words += [vmove(4, reg), 0xE0701000 | (plane * PLANE_WORDS * 4), 0x80080408]
    return words + [0xBF810000]


def request(work, root, threads, words):
    # .sa dword 4 bit 20 is clear, selecting the wave64 model.
    sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), 1, 0, 0x8A)
    return write_request(work, words, root, sa)


def expected_output(threads, result, scc):
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for local in range(threads):
            index = GUARD_WORDS + group * GROUP_WORDS + local
            lane = local % 64
            if result[lane // 32] & (1 << (lane % 32)):
                expected[index] = local
            for plane, value in enumerate((*result, scc), start=1):
                expected[index + plane * PLANE_WORDS] = value
    return expected


def main(cli, replay, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader, root_file = work / "root.hlsl", work / "root.bin"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root_file, shader], check=True)
        root = root_file.read_bytes()
        cache = work / "cache"
        compile_cmd = [cli, sdk, cache, "gfx1013-direct-cs@0"]
        replay_cases = []
        for threads, source, dst, incoming_scc, result, scc in CASES:
            marker = request(work, root, threads, program(source, dst, incoming_scc))
            output = marker.with_suffix(".cso")
            compiled = subprocess.run([*compile_cmd, marker, output], capture_output=True, text=True)
            assert compiled.returncode == 0 and output.is_file(), (compiled.stdout, compiled.stderr)
            replay_cases.append((threads, result, scc, marker))

        for src, dst in ((106, 24), (23, 24), (22, 25)):
            marker = request(work, root, 64, program(CASES[0][1], dst, 0, src))
            output = marker.with_suffix(".cso")
            compiled = subprocess.run([*compile_cmd, marker, output], capture_output=True, text=True)
            assert compiled.returncode != 0 and "unsupported wave64 scalar operation" in compiled.stderr, (
                src, dst, compiled.stdout, compiled.stderr)
            assert not output.exists(), output

        for threads, result, scc, marker in replay_cases:
            output = marker.with_suffix(".bin")
            replayed = subprocess.run([replay, sdk, marker, cache, "0", output, "--direct"])
            if replayed.returncode == 77:
                return 77
            replayed.check_returncode()
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            for index, (got, want) in enumerate(zip(actual, expected_output(threads, result, scc))):
                assert got == want, (threads, result, index, hex(got), hex(want))
            print(f"wave64 X={threads}, EXEC={result[1]:08x}:{result[0]:08x}, SCC={scc} matched")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
