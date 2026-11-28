"""Direct compute VOP2 signed byte dot-accumulate lowering and replay."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1, vmove


DOT4C = 0x0D
ALL = 0xFFFFFFFF
EVEN = 0x55555555
SENTINEL = 0x69696969
CASES = (
    # Name, encoded src0, extension words, packed src1, accumulator base, dot, EXEC
    ("literal", 255, (0xFBF201F7,), 0x04030201, 100, -69, ALL),
    ("sgpr", 10, (), 0x80808080, 0x7FFFFFF0, 65536, EVEN),
    ("inline", 193, (), 0x01020304, 64, -10, ALL),
    ("vgpr_dst_alias", 259, (), 0x7F000000, 0x80000000, -16256, ALL),
)


def program(case, *, modifier=None):
    name, src0, extension, b, base, _, active = case
    # Group IDs (0,0,0) and (1,0,0), 32 lanes each. UAV u0 starts with
    # eight guard words, then one consecutive 32-word row per group.
    code = [
        sop1(3, 126, 193),                 # EXEC_LO = all lanes
        sop1(3, 20, 4), sop1(3, 21, 128), # s[20:21] = resource table base
        0xF408080A, 0xFA000020,            # load s[32:35] for u0
        scalar(0x1E, 7, 5, 135),          # s7 = group_id.x * 128 bytes
        vop2(0x1A, 8, 130, 0),           # v8 = local lane * 4 bytes
        vop2(0x25, 8, 7, 8),             # v8 += s7
        vmove(2, 255), b,
        vop2(0x25, 3, 255, 0), base,      # v3 = accumulator base + local lane
        0xBF8CC07F,                      # wait for descriptor load
    ]
    if name == "sgpr":
        code += [sop1(3, 10, 255), 0x80808080]
    code += [sop1(3, 126, 255), active]
    if modifier == "dpp":
        src0, extension = 250, (0xFF00E403,)  # v3, identity quad_perm, full masks
    elif modifier == "sdwa":
        src0, extension = 249, (0x06060603,)  # v3, DWORD selectors
    return code + [
        vop2(DOT4C, 3, src0, 2), *extension,
        sop1(3, 126, 193),                # restore EXEC before readback
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xBF810000,
    ]


def ir(cli, sdk, work, root):
    for case, modifier, error in (
        *((case, None, None) for case in CASES),
        (CASES[-1], "dpp", "unsupported DPP arithmetic"),
        (CASES[-1], "sdwa", "unimplemented direct SDWA operand profile"),
    ):
        marker = request(work, root, program(case, modifier=modifier))
        output = marker.with_suffix(".cso")
        result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                capture_output=True, text=True)
        if error:
            assert result.returncode and error in result.stderr, (modifier, result.stdout, result.stderr)
            assert not output.exists(), output
        else:
            assert result.returncode == 0 and output.is_file(), (case[0], result.stdout, result.stderr)
            dump = subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", output], capture_output=True,
                                  text=True, check=True).stdout
            assert dump.count("@dx.op.dot4AddPacked.i32(i32 163,") == 1, (case[0], dump)
    print("signed VOP2 dot-accumulate DXIL verified for four source encodings; DPP and SDWA refused")
    return 0


def hardware(replay, sdk, work, root):
    for case in CASES:
        name, _, _, _, base, dot, active = case
        marker = request(work, root, program(case))
        output = work / "out.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        row = tuple((base + lane + (dot if (active >> lane) & 1 else 0)) & ALL
                    for lane in range(32))
        expected = (SENTINEL,) * 8 + row * 2 + (SENTINEL,) * (2064 - 72)
        actual = struct.unpack("<2064I", output.read_bytes())
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert got == want, f"{name} word {index}: got {got:08x}, expected {want:08x}"
        print(f"{name}: 64 results and all guards matched")
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / "root.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        return {"ir": ir, "hardware": hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
