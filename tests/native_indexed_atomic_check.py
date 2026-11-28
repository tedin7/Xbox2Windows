"""Check indexed atomics against recorded structured, raw and typed UAV windows."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_compute_clear_check import write_request
from native_scalar_lds_check import sop1, scalar, vmove


def program(mutant):
    # s4 is the table, s5 is group.x. Only lane zero may perform each atomic.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), sop1(3, 126, 129),
             scalar(0x1e, 7, 5, 132), scalar(0, 7, 7, 132), scalar(0x1e, 8, 5, 135), vmove(8, 8)]
    for j in range(5):
        words += [0xf4080000 | ((32 + 4*j) << 6) | 10, 0xfa000000 | (32 + 32*j)]
    for j in range(4):
        slot, first = (8 + j) << 16, 10 + 3*j
        # Aliased vaddr/vdata. GLC returns the original value. Without GLC
        # the source stays unchanged. The huge unsigned index must never wrap.
        words += [vmove(1, 7), 0xe0c86000, 0x80000101 | slot, vmove(first, 257),
                  vmove(1, 7), vmove(2, 132 if mutant else 131), 0xe0c82000, 0x80000201 | slot,
                  vmove(first+1, 258), vmove(1, 193), 0xe0c86000, 0x80000201 | slot, vmove(first+2, 258)]
    for j in range(12):
        words += [0xe0701000 | (4*j), 0x800c0008 | (10+j) << 8]
    # Only group zero touches each window's last complete dword. Group one
    # keeps a different pre-existing VGPR value while EXEC is empty.
    for j in range(4):
        slot = (8 + j) << 16
        last = 8188 if j == 3 else 2047
        words += [vmove(2, 137), vmove(3, 137), vmove(1, 255), last, vmove(4, 255), last + 1,
                  0xbf068005, scalar(0x0a, 126, 129, 128),
                  0xe0c86000, 0x80000201 | slot, 0xe0c86000, 0x80000304 | slot, sop1(3, 126, 129),
                  0xe0701000 | (48 + 8*j), 0x800c0208, 0xe0701000 | (52 + 8*j), 0x800c0308]
    return words + [0xbf810000]


def check_output(blob, layout, mutant):
    rest = memoryview(blob)
    for _, offset, _, stride, write in layout:
        assert write == 1
        words = (2048 + 16) * stride // 4
        data = struct.unpack_from(f"<{words}I", rest)
        rest = rest[4 * words:]
        expected = [0x69696969] * words
        for group in range(2):
            index = 4 + 16*group
            increment = 4 if mutant else 3
            if offset == 160:
                window = 8 + group*32
                expected[window:window+12] = [0x69696969, increment, 0] * 4
                expected[window+12:window+20] = ([9, 9] if group else [0x69696969, 0]) * 4
            else:
                assert offset in (32, 64, 96, 128)
                start = 8 * stride // 4
                word = index//4 if offset == 128 else index*stride//4
                expected[start + word] += index + increment
                if not group:
                    last = 2047 if offset == 128 else 2047 * stride // 4
                    expected[start + last] += 9
        assert data == tuple(expected), (offset, [(j, a, b) for j,(a,b) in enumerate(zip(data,expected)) if a!=b][:8])
    assert not rest and len(layout) == 5


def main(cli, gpu, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work/"root.hlsl", work/"root.bin"
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk)/"dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, src], check=True)
        sa = struct.pack("<7I",0,0,0,32 | 1<<16,1 | (1 << 20),0,0x8a)
        previous = None
        for mutant in (False, True):
            words = program(mutant)
            marker = write_request(work, words, root.read_bytes(), sa)
            dest = work/"result.bin"
            run = subprocess.run([gpu, sdk, marker, work/"cache", "0", dest, "--direct"], capture_output=True,text=True)
            if run.returncode == 77:
                print(run.stderr); sys.exit(77)
            assert run.returncode == 0, run.stderr
            data = dest.read_bytes()
            check_output(data, [tuple(map(int,line.split())) for line in run.stdout.splitlines()], mutant)
            if previous is not None:
                diff = sum(a != b for a,b in zip(struct.iter_unpack("<I",previous),struct.iter_unpack("<I",data)))
                assert diff == 16, diff
            previous = data
        slot = words.index(0xe0c86000)
        for extra in (4, 0x1000, 0x8000):
            invalid = words.copy()
            invalid[slot] |= extra
            marker = write_request(work,invalid,root.read_bytes(),sa)
            r = subprocess.run([cli,sdk,work/"cache","gfx1013-direct-cs@0",marker,work/"bad.cso"],
                               capture_output=True,text=True)
            assert r.returncode and not (work/"bad.cso").exists() and "atomic" in r.stderr, r.stderr
    print("Four bound UAV shapes, old values, inactive lanes, unsigned OOB, window guards and 16-word mutation matched.")


if __name__ == "__main__":
    main(*sys.argv[1:])
