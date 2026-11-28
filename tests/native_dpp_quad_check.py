"""Direct compute DPP16 quad_perm readback and compile-only refusals."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


OR, ADD, SUBREV = 0x1C, 0x03, 0x05
SWAP = (1, 0, 3, 2)
ALL = 0xFFFFFFFF
SPARSE = ALL ^ (1 << 1)
SENTINEL = 0x69696969


def compute_program(op, pattern, active=ALL, fi=False, bound_zero=False):
    control = 0xFF000001 | pattern << 8 | fi << 18 | bound_zero << 19
    # Reuse the mirror fixture's table load and addressing ABI. Two groups
    # write 32 consecutive words each, after eight guard words.
    program = [sop1(3, 20, 4), sop1(3, 21, 128),
               0xF4080000 | 32 << 6 | 10, 0xFA000020,
               scalar(0x1e, 7, 5, 135), vop2(0x1a, 8, 130, 0), vop2(0x25, 8, 7, 8),
               vop2(0x25, 1, 145, 0)]
    if op == OR:
        program += [vmove(2, 192)]  # inline integer 64
    else:
        program += [vop2(0x25, 2, 148, 0),
                    0x7E000000 | 1 << 17 | 6 << 9 | 257,  # v_cvt_f32_u32 v1, v1
                    0x7E000000 | 2 << 17 | 6 << 9 | 258]  # v_cvt_f32_u32 v2, v2
    # Seed both registers under full EXEC, alias src0 with dst, and restore
    # EXEC before storing so suppressed writes are visible in the readback.
    return program + [sop1(3, 126, 255), active,
                      vop2(op, 1, 250, 2), control, sop1(3, 126, 193),
                      0xE0701000, 0x80080108, 0xBF810000]


def reference(op, selection, active, fi, bound_zero):
    initial = list(range(17, 49))
    enabled = [(active >> lane) & 1 for lane in range(32)]
    result = initial.copy()
    for base in range(0, 32, 4):
        for offset, source in enumerate(selection):
            lane, peer = base + offset, base + source
            if not enabled[lane]:
                continue
            if fi or enabled[peer]:
                src0 = initial[peer]
            elif bound_zero:
                src0 = 0
            else:
                continue
            if op == OR:
                result[lane] = src0 | 64
            elif op == ADD:
                result[lane] = src0 + (lane + 20)
            else:
                result[lane] = (lane + 20) - src0
    if op != OR:
        return struct.unpack("<32I", struct.pack("<32f", *result))
    return tuple(result)


def request(work, root, program, threads=32):
    sa = struct.pack("<7I", 0, 0, 0, threads | 1 << 16, 1 | (1 << 20), 0, 0x8A)
    return write_request(work, program, root, sa)


def controls(cli, sdk, work, root):
    for pattern, threads, error in (
            (0xB1, 32, None),
            (0x150, 32, "unsupported DPP lane selection"),  # row_share:0
            (0xB1, 30, "DPP quad_perm needs a thread count divisible by four")):
        marker = request(work, root, compute_program(OR, pattern), threads)
        output = marker.with_suffix(".cso")
        result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                capture_output=True, text=True)
        if error is None:
            assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
        else:
            assert result.returncode and error in result.stderr, (result.stdout, result.stderr)
            assert not output.exists(), output
            print(error)
    return 0


def hardware(replay, sdk, work, root):
    # Manual mutants that must fail: read the permutation in reversed lane
    # order (rotation), or ignore BOUND_CTRL (sparse EXEC add with BC=1).
    cases = (
        ("or swap", OR, SWAP, ALL, False, False),
        ("add swap", ADD, SWAP, ALL, False, False),
        ("subrev broadcast", SUBREV, (0, 0, 0, 0), ALL, False, False),
        ("or identity", OR, (0, 1, 2, 3), ALL, False, False),
        ("or rotation", OR, (1, 2, 3, 0), ALL, False, False),
        ("add disabled source, zero", ADD, SWAP, SPARSE, False, True),
        ("add disabled source, preserve", ADD, SWAP, SPARSE, False, False),
        ("add disabled source, fetch", ADD, SWAP, SPARSE, True, False),
    )
    for name, op, selection, active, fi, bound_zero in cases:
        pattern = sum(source << (2 * lane) for lane, source in enumerate(selection))
        marker = request(work, root, compute_program(op, pattern, active, fi, bound_zero))
        output = work / "out.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        lanes = reference(op, selection, active, fi, bound_zero)
        expected = (SENTINEL,) * 8 + lanes * 2 + (SENTINEL,) * (2064 - 72)
        actual = struct.unpack("<2064I", output.read_bytes())
        mismatch = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), None)
        assert mismatch is None, (name, mismatch, actual[mismatch], expected[mismatch])
        print(f"{name}: 64 results and guards matched")
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
