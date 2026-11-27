"""Direct wave32 SDWA VOPC compares that write their lane mask to vcc_hi and read it back as a scalar source."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


def sdwa_vcc_hi_readback(work, gpu, sdk):
    # Each lane owns two dwords. The negated zero class bit is present only in lane zero.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
             scalar(0x1e, 7, 5, 136),                        # group byte offset = group.x << 8
             0x1a << 25 | 8 << 17 | 131,                    # v8 = local.x << 3
             0x25 << 25 | 8 << 17 | 8 << 9 | 7,             # v8 += group byte offset
             vmove(5, 255), 0x20000000,
             0x16 << 25 | 5 << 17 | 5 << 9 | 256,          # v5 >>= local.x
             0x7d100af9, 0x0396eb80,                       # class(-0, BYTE_3(v5)) -> vcc_hi
             sop1(3, 14, 107), vmove(9, 14), 0xe0701000, 0x80080908,
             0x1c << 25 | 6 << 17 | 255, 0x100,            # v6 = local.x | 0x100
             0x7d0a0cf9, 0x0086eb6b,                       # ne(vcc_hi, BYTE_0(v6)) -> vcc_hi
             sop1(3, 14, 107), vmove(9, 14), 0xe0701004, 0x80080908,
             0xbf810000]
    sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1 | 1 << 20, 0, 0x80 | 5 << 1)  # bit 20: wave32 (#2999)
    marker = write_request(work, words, (work / "root.bin").read_bytes(), sa)
    dest = work / "sdwa-vcc-hi.bin"
    result = subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"])
    if result.returncode == 77:
        sys.exit(77)
    result.check_returncode()
    class_mask = sum(1 << lane for lane in range(32) if ((0x20000000 >> lane) >> 24) & (1 << 5))
    ne_mask = sum(1 << lane for lane in range(32) if class_mask != ((lane | 0x100) & 0xff))
    expected = [0x69696969] * 2064
    expected[8:136] = [class_mask, ne_mask] * 64
    actual = struct.unpack("<2064I", dest.read_bytes())
    assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Both wave32 SDWA vcc_hi masks matched in every lane and all guards held.")


def main(gpu, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        src, root = work / "root.hlsl", work / "root.bin"
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, src], check=True)
        sdwa_vcc_hi_readback(work, gpu, sdk)


if __name__ == "__main__":
    main(*sys.argv[1:])
