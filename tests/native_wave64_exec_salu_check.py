"""Compile and replay slot 67 scalar EXEC operands in both wave models."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


BCNT1_B64, BFM_B64 = 0x10, 0x25
VCC, EXEC, ZERO, NEG_ONE = 106, 126, 128, 193
COUNT_ERROR = "invalid s_bcnt1_i32_b64 source"
BFM_ERROR = "invalid s_bfm_b64 destination"
GROUP_STRIDE_WORDS = 128
GUARD_WORDS, OUTPUT_WORDS = 8, 2064
SENTINEL = 0x69696969
END = 0xBF810000


def request(work, root, wave_size, threads, words):
    z = 1 | (1 << 20 if wave_size == 32 else 0)
    sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), z, 0, 0x8A)
    return write_request(work, words, root, sa)


def program(wave_size, operation):
    # s4 is the UAV table and s5 is group.x. v0 is local.x.
    # Replay dispatches groups 0 and 1. The byte address is 512 * group.x + 4 * local.x.
    if wave_size == 64:
        words = [sop1(4, EXEC, NEG_ONE)]
    else:
        # Explicitly enable all 32 lanes and seed the tracked upper word.
        words = [sop1(3, EXEC, NEG_ONE), sop1(3, EXEC + 1, ZERO + 1)]
    words += [
        sop1(3, 20, 4), sop1(3, 21, ZERO),
        0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
        scalar(0x1E, 7, 5, ZERO + 9),  # s_lshl_b32 s7, s5, 9
        0x34100082,  # v_lshlrev_b32 v8, 2, v0
        0x4A101007,  # v_add_nc_u32 v8, s7, v8
    ]
    if operation == "count":
        words += [
            0x36040083,  # v_and_b32 v2, 3, v0
            0x7DA40480,  # v_cmpx_eq_u32 0, v2
            0x7DAA00A0,  # v_cmpx_ne_u32 32, v0
            sop1(BCNT1_B64, 24, EXEC),
            vmove(1, 24),
        ]
        data = 1
    else:
        width, offset = (16, 0) if operation == "lower" else (32, 32)
        words.append(scalar(BFM_B64, EXEC, ZERO + width, ZERO + offset))
        data = 0
    return words + [0xE0701000, 0x80080008 | data << 8, END]


def expected_words(wave_size, threads, operation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for local in range(threads):
            lane = local % wave_size
            word = GUARD_WORDS + group * GROUP_STRIDE_WORDS + local
            if operation == "count" and local % 4 == 0 and local != 32:
                # Wave64 loses lane 32 in the first guest wave only. Wave32
                # counts eight lower bits and the explicitly set EXEC_HI bit.
                expected[word] = 9 if wave_size == 32 else 15 if local < 64 else 16
            elif operation == "lower" and lane < 16:
                expected[word] = local
            elif operation == "upper" and lane >= 32:
                expected[word] = local
    return expected


def compile_request(cli, sdk, cache, marker):
    output = marker.with_suffix(".cs.cso")
    result = subprocess.run([cli, sdk, cache, "gfx1013-direct-cs@0", marker, output],
                            capture_output=True, text=True)
    return output, result


def main(cli, replay, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader, root_file = work / "root.hlsl", work / "root.bin"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root_file, shader], check=True)
        root = root_file.read_bytes()
        cache = work / "cache"
        cases = [(64, threads, op) for threads in (64, 128) for op in ("count", "lower", "upper")]
        cases.append((32, 32, "count"))
        positives = []
        for wave_size, threads, op in cases:
            marker = request(work, root, wave_size, threads, program(wave_size, op))
            output, result = compile_request(cli, sdk, cache, marker)
            assert result.returncode == 0 and output.is_file(), (
                wave_size, threads, op, result.stdout, result.stderr)
            positives.append((wave_size, threads, op, marker))

        for instruction, error in (
            (sop1(BCNT1_B64, 24, VCC), COUNT_ERROR),
            (scalar(BFM_B64, VCC, ZERO + 16, ZERO), BFM_ERROR),
            (sop1(BCNT1_B64, 24, 23), COUNT_ERROR),
        ):
            marker = request(work, root, 64, 64, [instruction, END])
            output, result = compile_request(cli, sdk, cache, marker)
            assert result.returncode and error in result.stderr, (error, result.stdout, result.stderr)
            assert not output.exists(), output

        for wave_size, threads, op, marker in positives:
            output = work / f"wave{wave_size}_{threads}_{op}.bin"
            result = subprocess.run([replay, sdk, marker, cache, "0", output, "--direct"])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            expected = expected_words(wave_size, threads, op)
            for index, (got, want) in enumerate(zip(actual, expected)):
                assert got == want, (wave_size, threads, op, index, hex(got), hex(want))
            print(f"wave{wave_size}, {threads} threads, {op}: both groups and guards matched")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
