"""Compile and replay 64-bit NOT and CSELECT with the compute mask registers."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_scalar_lds_check import scalar, sop1, vmove
from native_wave64_exec_salu_check import (
    END, EXEC, GROUP_STRIDE_WORDS, GUARD_WORDS, NEG_ONE, OUTPUT_WORDS,
    SENTINEL, VCC, ZERO, compile_request, request,
)


NOT_B64, CSELECT_B64 = 0x08, 0x0B
MOV_B32, MOV_B64, LITERAL = 3, 4, 255
M0, RESULT = 124, 24
MASK32, MASK64 = (1 << 32) - 1, (1 << 64) - 1
EXEC_MASK = 0xA5600FF00F0FF00F
VCC_MASK = 0x3150C4D2FFFFFFFF
PLANE_WORDS = 2 * GROUP_STRIDE_WORDS
CASES = ("not_exec", "not_vcc", "not_vcc_all_ones", "select_scc0", "select_scc1", "not_into_exec")


def move_pair(dst, bits):
    return [sop1(MOV_B32, dst, LITERAL), bits & MASK32,
            sop1(MOV_B32, dst + 1, LITERAL), bits >> 32]


def set_scc(value):
    # s_cmp_eq_u32 0, 0 (SCC = 1) or s_cmp_eq_u32 0, 1 (SCC = 0).
    return 0xBF060000 | (ZERO + (1 - value)) << 8 | ZERO


def store(vgpr, plane):
    return [0xE0701000 | (4 * PLANE_WORDS * plane), 0x80080008 | vgpr << 8]


def program(case):
    # Prepare every lane's address before narrowing EXEC. The shader uses
    # the same two-group UAV layout as the wave64 EXEC SALU fixture.
    words = [
        sop1(MOV_B64, EXEC, NEG_ONE),
        sop1(MOV_B32, 20, 4), sop1(MOV_B32, 21, ZERO),
        0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
        scalar(0x1E, 7, 5, ZERO + 9),  # s_lshl_b32 s7, s5, 9
        0x34100082,  # v_lshlrev_b32 v8, 2, v0
        0x4A101007,  # v_add_nc_u32 v8, s7, v8
    ]
    words += move_pair(EXEC, EXEC_MASK)
    words += move_pair(VCC, MASK64 if case == "not_vcc_all_ones" else VCC_MASK)
    if case.startswith("select"):
        words += [set_scc(int(case[-1])), scalar(CSELECT_B64, RESULT, EXEC, VCC)]
    elif case == "not_into_exec":
        words += [set_scc(0), sop1(NOT_B64, EXEC, EXEC), *store(0, 0)]
        # Save both new EXEC words before restoring all lanes for readback.
        words.append(sop1(MOV_B64, RESULT, EXEC))
    else:
        source = EXEC if case == "not_exec" else VCC
        words += [set_scc(int(case == "not_vcc_all_ones")), sop1(NOT_B64, RESULT, source)]

    words += [sop1(MOV_B64, EXEC, NEG_ONE), scalar(0x0A, RESULT + 2, ZERO + 1, ZERO)]
    for plane, src in enumerate((RESULT, RESULT + 1, RESULT + 2), start=1):
        words += [vmove(1, src), *store(1, plane)]
    return words + [END]


def expected_words(wave_size, case):
    if case.startswith("select"):
        scc = int(case[-1])
        result = EXEC_MASK if scc else VCC_MASK
    else:
        source = {"not_exec": EXEC_MASK, "not_vcc": VCC_MASK,
                  "not_vcc_all_ones": MASK64, "not_into_exec": EXEC_MASK}[case]
        result = source ^ MASK64
        scc = int(result != 0)
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for lane in range(wave_size):
            index = GUARD_WORDS + group * GROUP_STRIDE_WORDS + lane
            if case == "not_into_exec" and result & (1 << lane):
                expected[index] = lane
            for plane, value in enumerate((result & MASK32, result >> 32, scc), start=1):
                expected[index + plane * PLANE_WORDS] = value
    return expected


def controls(cli, sdk, work, root):
    cache = work / "cache"
    failures = []
    for wave_size in (32, 64):
        for case in CASES:
            marker = request(work, root, wave_size, wave_size, program(case))
            output, result = compile_request(cli, sdk, cache, marker)
            if result.returncode or not output.is_file():
                failures.append((wave_size, case, result.stdout, result.stderr))
    assert not failures, failures

    prefix = [sop1(MOV_B32, 23, ZERO), sop1(MOV_B32, M0, ZERO), set_scc(1)]
    refusals = [
        (sop1(NOT_B64, RESULT, src), "invalid s_not_b64 source")
        for src in (23, M0)
    ] + [
        (scalar(CSELECT_B64, RESULT, a, b), "invalid s_cselect_b64 source")
        for src in (23, M0) for a, b in ((src, ZERO), (ZERO, src))
    ] + [(scalar(CSELECT_B64, EXEC, ZERO, ZERO), "invalid s_cselect_b64 destination")]
    for instruction, error in refusals:
        marker = request(work, root, 64, 64, prefix + [instruction, END])
        output, result = compile_request(cli, sdk, cache, marker)
        assert result.returncode and error in result.stderr, (error, result.stdout, result.stderr)
        assert not output.exists(), output
    print("Wave32 and wave64 compile controls and invalid operand refusals passed")
    return 0


def hardware(replay, sdk, work, root):
    for wave_size in (32, 64):
        for case in CASES:
            marker = request(work, root, wave_size, wave_size, program(case))
            output = work / f"wave{wave_size}_{case}.bin"
            result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            for index, (got, want) in enumerate(zip(actual, expected_words(wave_size, case))):
                assert got == want, (wave_size, case, index, hex(got), hex(want))
            print(f"wave{wave_size} {case}: both groups and guards matched")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader, root_file = work / "root.hlsl", work / "root.bin"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root_file, shader], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root_file.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
