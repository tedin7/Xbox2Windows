"""Direct wave32 SMEM loads with vcc_lo / vcc_hi as the scalar offset and vcc_hi as a destination."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


def vcc_smem_readback(work, gpu, sdk):
    source, root = work / "built.hlsl", work / "built.root"
    source.write_text('#define SIG "SRV(t0),DescriptorTable(UAV(u0,numDescriptors=8))"')
    subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, source], check=True)
    # The raw .sa capture sets bit 20 of dword 4 for wave32.
    sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1 | 1 << 20, 0, 0x80 | 7 << 1)
    words = [sop1(4, 16, 4), sop1(3, 18, 255), 132, sop1(3, 19, 255), 0x31016fac,
             sop1(3, 20, 6), sop1(3, 21, 128), 0xf4080000 | 40 << 6 | 10, 0xfa000020,
             scalar(0x1e, 106, 7, 130), scalar(0xe, 106, 106, 132), sop1(3, 107, 136),
             0xf4200008 | 30 << 6, 106 << 25 | 4,       # s_buffer_load_dword s30, s[16:19], vcc_lo
             0xf4240008 | 32 << 6, 107 << 25 | 20,      # s_buffer_load_dwordx2 s[32:33], s[16:19], vcc_hi
             0xf40c0002 | 48 << 6, 107 << 25 | 80,      # s_load_dwordx8 s[48:55], s[4:5], vcc_hi
             0xf4200008 | 107 << 6, 107 << 25 | 40,     # s_buffer_load_dword vcc_hi, s[16:19], vcc_hi
             0xbf8cc07f, scalar(0x1e, 12, 7, 134), vmove(8, 12), sop1(3, 126, 129)]
    for j, sgpr in enumerate((30, 32, 33, *range(48, 56), 107)):
        words += [vmove(9, sgpr), 0xe0701000 | 4 * j, 0x800a0908]
    marker = write_request(work, words + [0xbf810000], root.read_bytes(), sa)
    dest = work / "vcc-smem-readback.bin"
    result = subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"])
    if result.returncode == 77:
        sys.exit(77)
    result.check_returncode()
    expected = [0x69696969] * 2064
    for group in range(2):
        start = 8 + 16 * group
        indices = (1 + group, 7, 8, *range(22, 30), 12)
        expected[start:start + len(indices)] = [0x51aa0000 + index for index in indices]
    actual = struct.unpack("<2064I", dest.read_bytes())
    assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Wave32 SMEM VCC offsets and destination matched 24 result words and all guards.")


def main(gpu, sdk):
    with tempfile.TemporaryDirectory() as temp:
        vcc_smem_readback(Path(temp), gpu, sdk)


if __name__ == "__main__":
    main(*sys.argv[1:])
