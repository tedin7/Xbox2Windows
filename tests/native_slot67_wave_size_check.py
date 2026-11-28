"""Compare slot 67 wave64 and wave32 lane counts for the same source-built program."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request


PROGRAM = (
    0xBE940304, 0xBE950380,  # s[20:21] = UAV table address from s4
    0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
    0x8F078805,              # s_lshl_b32 s7, s5, 8 (group byte offset)
    0x34100082,              # v_lshlrev_b32 v8, 2, v0 (local byte offset)
    0x4A101007,              # v_add_nc_u32 v8, s7, v8
    0xD7650001, 0x000100C1,  # v_mbcnt_lo_u32_b32 v1, -1, 0
    0xD7660001, 0x000202C1,  # v_mbcnt_hi_u32_b32 v1, -1, v1
    0xE0701000, 0x80080108,  # buffer_store_dword v1, v8, s[32:35], 0 offen
    0xBF810000,              # s_endpgm
)
THREADS = 64
DISPATCH_THREADS = 2 * THREADS  # native_store_replay dispatches two groups
GUARD = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969


def main(cli, replay, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader, root = work / "root.hlsl", work / "root.bin"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        # (wave size, threads per group): a wave64 request with a 32-thread group has one guest wave of lanes 0..31.
        cases = ((64, 64), (32, 64), (64, 32))
        markers = {}
        for wave_size, threads in cases:
            z = 1 | (1 << 20 if wave_size == 32 else 0)
            sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), z, 0, 0x8A)
            markers[wave_size, threads] = write_request(work, PROGRAM, root.read_bytes(), sa)

        cache, batch = work / "cache", work / "batch"
        subprocess.run([cli, sdk, cache, "gfx1013-direct-cs@0", "--requests", work, batch],
                       check=True)
        for marker in markers.values():
            assert (batch / f"{marker.stem[4:]}.cs.cso").is_file(), marker

        for (wave_size, threads), marker in markers.items():
            output = work / f"wave{wave_size}_{threads}.bin"
            result = subprocess.run([replay, sdk, marker, cache, "0", output, "--direct"])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            expected = [SENTINEL] * OUTPUT_WORDS
            # Each group stores its lane indices at a 256-byte group offset (s_lshl_b32 s7, s5, 8).
            for group in range(2):
                for tid in range(threads):
                    expected[GUARD + group * THREADS + tid] = tid % min(wave_size, threads)
            for index, (got, want) in enumerate(zip(actual, expected)):
                assert got == want, (wave_size, index, got, want)
            print(f"wave{wave_size}, {threads}-thread groups: {2 * threads} lane indices and buffer guards matched")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
