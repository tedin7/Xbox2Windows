"""Exercise the C++ frontend with a source-assembled two-wave request."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_compute_clear_check import write_request


def sdwa_fixtures(cli_args, gpu, sdk, work, root, sa):
    def xor(dst, lhs, rhs, sel0, sel1, dest=6, unused=0, sext0=False):
        return [(0x1d << 25) | (dst << 17) | (rhs << 9) | 249,
                lhs | (dest << 8) | (unused << 11) | (sel0 << 16) | (int(sext0) << 19) | (sel1 << 24)]

    setup = [
        0xbe940304, 0xbe950380, 0xbe960305, 0xbe970380,
        0xf4080000 | (32 << 6) | 10, 0xfa000040,
        0xf40c0000 | (40 << 6) | 11, 0xfa0000e0,
        0xd7460008, 12 | (133 << 9) | (256 << 18), 0x7e12020d,
        0xe0042000, 0x80080208,  # load v2, v3 from element x
        0x7e000200 | (20 << 17) | 258, 0x7e000200 | (21 << 17) | 259,
    ]
    finish = []
    for reg, low in [(20, 2), (21, 4)]:
        # Store both halves exactly as floats, including NaN-shaped integer results.
        finish += [(0x1b << 25) | (low << 17) | (reg << 9) | 255, 65535,
                   (0x16 << 25) | ((low + 1) << 17) | (reg << 9) | 144]
    for reg in range(2, 6):
        finish.append(0x7e000000 | (reg << 17) | (6 << 9) | (256 + reg))
    finish += [0xf0200f08, (10 << 16) | (2 << 8) | 8, 0xbf810000]
    cases = [
        ("sdwa-alias", xor(20, 20, 20, 5, 6) + xor(21, 3, 2, 2, 1), False),
        ("sdwa-preserve", xor(20, 2, 3, 2, 6, sext0=True) + xor(21, 2, 21, 5, 6, 1, 2), False),
        ("sdwa-sext", xor(20, 2, 3, 2, 6, sext0=True) + xor(21, 2, 21, 5, 6, 1, 1), False),
        ("sdwa-alias", xor(20, 20, 20, 4, 6) + xor(21, 3, 2, 2, 1), True),
        ("sdwa-preserve", xor(20, 2, 3, 2, 6, sext0=True) + xor(21, 2, 21, 5, 6, 1, 0), True),
        ("sdwa-class", [
            0x7e000000 | (20 << 17) | (0x65 << 9) | 277,  # swap v20 and v21
            0x7c000000 | (0x88 << 17) | (3 << 9) | 249, 0x03968e80,  # class(-0, byte3(v3)) -> s14
            0xbeea030e,  # vcc_lo = s14
            (1 << 25) | (21 << 17) | (21 << 9) | 276,  # choose swapped v20 or v21
        ], False),
    ]
    for mode, operations, mutant in cases:
        marker = write_request(work, setup + operations + finish, root, sa)
        result = subprocess.run([gpu, sdk, marker, work / "cache", mode], text=True, capture_output=True)
        print(mode, "mutant" if mutant else "reference", result.stdout.strip(), flush=True)
        assert result.returncode == int(mutant), (result.stdout, result.stderr)
        assert ("differing_components=0 " in result.stdout) != mutant
    invalid = write_request(work, setup + xor(20, 2, 3, 7, 6) + finish, root, sa)
    rejected = subprocess.run([*cli_args, invalid, work / "invalid.cso"], capture_output=True, text=True)
    assert rejected.returncode and "SDWA modifiers" in rejected.stderr and not (work / "invalid.cso").exists()


def raw_offset_fixtures(cli_args, gpu, sdk, work, root, sa):
    def move(dst, src):
        return 0x7e000200 | (dst << 17) | src

    def load(op, dst, addr, offset=0):
        return [0xe0001000 | (op << 18) | offset, 0x80080000 | (dst << 8) | addr]

    setup = [
        0xbe940304, 0xbe950380, 0xbe960305, 0xbe970380,
        0xf4080000 | (32 << 6) | 10, 0xfa000040,
        0xf40c0000 | (40 << 6) | 11, 0xfa0000e0,
        0xd7460008, 12 | (133 << 9) | (256 << 18), move(9, 13),
        0xd746000a, 13 | (134 << 9) | (264 << 18),  # i = group.y * 64 + x
        (0x1a << 25) | (20 << 17) | (10 << 9) | 131,  # byte offset = i << 3
    ]
    finish = [0x7e000000 | (dst << 17) | (6 << 9) | (256 + src)
              for dst, src in zip(range(2, 6), (20, 24, 25, 10))]
    finish += [0xf0200f08, (10 << 16) | (2 << 8) | 8, 0xbf810000]
    cases = [
        ("raw-offset", [move(24, 276)], load(0x0d, 24, 24), 52, False),
        ("raw-offset", [move(25, 276)], load(0x0d, 24, 25), 52, False),
        ("raw-edge", [move(24, 255), (2 * 2048 + 15) * 4], load(0x0d, 24, 24), 52, False),
        # First word overwrites v24; a second instruction now uses the wrong pair's in-bounds address.
        ("raw-offset", [move(24, 276)], load(0x0c, 24, 24) + load(0x0c, 25, 24, 4), 52, True),
        ("raw-offset", [move(24, 276)], load(0x0d, 24, 24), 0, True),
        ("raw-three", [move(26, 276)], load(0x0f, 24, 26), 52, False),
        ("raw-four", [move(27, 276)], load(0x0e, 24, 27), 52, False),
        ("raw-math", [move(24, 276)], load(0x0d, 24, 24), 52, False),
        # Dword accesses ignore the low two bits after adding the immediate.
        ("raw-offset", [(0x25 << 25) | (20 << 17) | (20 << 9) | 131, move(24, 276)],
         load(0x0d, 24, 24), 52, False),
    ]
    for mode, addr, pair, offset, mutant in cases:
        tail = finish
        if mode in ("raw-three", "raw-four"):
            registers = (24, 25, 26, 27 if mode == "raw-four" else 20)
            tail = [0x7e000000 | (dst << 17) | (6 << 9) | (256 + src)
                    for dst, src in zip(range(2, 6), registers)] + finish[4:]
        elif mode == "raw-math":
            tail = finish[:4] + [(8 << 25) | (2 << 17) | (5 << 9) | 255, 0x3e800000,
                                 0x7e000000 | (2 << 17) | (0x36 << 9) | 258,
                                 0x7e000000 | (5 << 17) | (0x0b << 9) | 280] + finish[4:]
        words = setup + addr + load(0x0c, 20, 20, offset) + pair + tail
        marker = write_request(work, words, root, sa)
        run = subprocess.run([gpu, sdk, marker, work / "cache", mode], capture_output=True, text=True)
        print(mode, "mutant" if mutant else "reference", run.stdout.strip(), flush=True)
        assert run.returncode == int(mutant), (run.stdout, run.stderr)
        expected = 2048 if mutant else 0
        assert f"differing_components={expected} " in run.stdout
    # Select a different 32-byte descriptor for each of the two X workgroups.
    # Both vector loads share the descriptor. Clobber its source index before either load.
    def scalar(op, dst, a, b):
        return 0x80000000 | (op << 23) | (dst << 16) | (b << 8) | a

    table_load = 0xf4080000 | (32 << 6) | 10
    for flip in (0, 1):
        selector = [scalar(0, 28, 12, 128 + flip), scalar(0x0e, 28, 28, 129), scalar(0x1e, 28, 28, 133)]
        words = (setup[:4] + selector + [table_load, (28 << 25) | 64, 0xbe9c0380] + setup[6:] +
                 [move(24, 276)] + load(0x0c, 20, 20, 52) + load(0x0d, 24, 24) + finish)
        marker = write_request(work, words, root, sa)
        run = subprocess.run([gpu, sdk, marker, work / "cache", "indexed"], capture_output=True, text=True)
        print("indexed", "flipped" if flip else "reference", run.stdout.strip(), flush=True)
        assert run.returncode == flip and f"differing_components={6144 * flip} " in run.stdout, (run.stdout, run.stderr)
    for change, value in ((0, 0xbe94030a), (8, (28 << 25) | 512)):
        invalid_words = words.copy()
        invalid_words[change] = value
        marker = write_request(work, invalid_words, root, sa)
        run = subprocess.run([*cli_args, marker, work / "invalid-span.cso"], text=True, capture_output=True)
        assert run.returncode and not (work / "invalid-span.cso").exists(), run.stdout
    invalid = write_request(work, setup + load(0x0d, 255, 20) + finish, root, sa)
    result = subprocess.run([*cli_args, invalid, work / "invalid-pair.cso"], capture_output=True, text=True)
    assert result.returncode and "destination overflow" in result.stderr and not (work / "invalid-pair.cso").exists()
    indexed = load(0x0d, 24, 20)
    indexed[0] |= 1 << 13
    invalid = write_request(work, setup + indexed + finish, root, sa)
    result = subprocess.run([*cli_args, invalid, work / "indexed.cso"], capture_output=True, text=True)
    assert result.returncode and "needs offen" in result.stderr and not (work / "indexed.cso").exists()


def global_stride_fixtures(gpu, sdk, work, root, source, sa):
    source.write_text('#define SIG "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED),'
                      'DescriptorTable(SRV(t0,numDescriptors=16)),DescriptorTable(UAV(u0,numDescriptors=16)),'
                      'RootConstants(num32BitConstants=4,b0)"')
    subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, source], check=True)
    # GPU input s[40:47] feeds s43. Select a heap slot, then extract descriptor word 1 bits 16..29.
    # The table index is deliberately unbounded at compile time, and is overwritten before the extract.
    for selector_offset in (0, 4):
        words = [0xbe940304, 0xbe950380, 0xbe960305, 0xbe970380,
                 0xf4080000 | (24 << 6) | 10, 0xfa000040,
                 0xf40c0000 | (56 << 6) | 11, 0xfa0000e0,
                 0x80000000 | (0x1e << 23) | (20 << 16) | (133 << 8) | 12,
                 0xf42c0000 | (40 << 6) | 12, (20 << 25) | selector_offset,
                 0xbe9e030a, 0xbe9f0380, 0x80000000 | (0x1e << 23) | (43 << 16) | (133 << 8) | 43,
                 0xf4080000 | (48 << 6) | 15, 43 << 25, 0xbeab0380,
                 0x93800000 | (28 << 16) | (255 << 8) | 49, 0x000e0010,
                 0xd7460008, 12 | (133 << 9) | (256 << 18), 0x7e12020d]
        words += [0x7e000000 | (v << 17) | (6 << 9) | 28 for v in range(2, 6)]
        words += [0xf0200f08, (14 << 16) | (2 << 8) | 8, 0xbf810000]
        marker = write_request(work, words, root.read_bytes(), sa)
        result = subprocess.run([gpu, sdk, marker, work / "cache", "global-stride"], capture_output=True, text=True)
        print("global-stride", selector_offset, result.stdout.strip(), flush=True)
        assert result.returncode == int(bool(selector_offset)), (result.stdout, result.stderr)
        assert f"differing_components={8192 if selector_offset else 0} " in result.stdout


def main(cli, gpu, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source = work / "root.hlsl"
        source.write_text('#define SIG "DescriptorTable(SRV(t0,numDescriptors=16)),DescriptorTable(UAV(u0,numDescriptors=16))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, source], check=True)
        sa = struct.pack("<7I", 0, 0, 0, 8 | (8 << 16), 1 | (1 << 20), 0, 0x8d8c)
        # s4 and s5 are the two guest tables. The byte offsets deliberately differ
        # from the desktop registers that the translator assigns to them.
        words = [
            0xbe940304, 0xbe950380, 0xbe960305, 0xbe970380,
            0xf4080000 | (32 << 6) | 10, 0xfa000040,  # descriptor s32:35, table 0 +64
            0xf40c0000 | (40 << 6) | 11, 0xfa0000e0,  # descriptor s40:47, table 1 +224
            0xbe8b0384,                              # s11 = byte offset 4
            0xf4240000 | (11 << 6) | 16, 11 << 25,  # s[11:12] = buffer[s11], mask in second word
            0x938dff08, 0x00050014,                 # wave in group from TG_SIZE_EN -> s13
            0x938eff08, 0x00060000,                 # number of waves -> s14
            0xd7650002, (128 << 9) | 193,           # v2 = mbcnt(-1, 0)
            0xd7650003, (128 << 9) | 12,            # v3 = mbcnt(s12, 0)
        ]
        for dst, src in [(2,258),(3,259),(4,13),(5,14)]:
            words.append(0x7e000000 | (dst << 17) | (6 << 9) | src)  # v_cvt_f32_u32
        words.extend([0xf0200f08, (10 << 16) | (2 << 8), 0xbf810000])
        marker = write_request(work, words, root.read_bytes(), sa)
        args = [cli, sdk, work / "cache", "gfx1013-compute-wave32@0"]
        result = subprocess.run([*args, marker, work / "good.cso"], capture_output=True, text=True, check=True)
        assert "t0 root=0 bytes=64 min_size=4" in result.stdout and "u0 root=1 bytes=224" in result.stdout
        rejected = write_request(work, [*words[:-1], 0xffffffff, words[-1]], root.read_bytes(), sa)
        result = subprocess.run([*args, rejected, work / "bad.cso"], capture_output=True, text=True)
        # 0xffffffff is not a gfx1013 encoding, so the decoder refuses it before any lowering.
        assert result.returncode and "unknown gfx1013 instruction format at byte" in result.stderr, result.stderr
        assert not (work / "bad.cso").exists()
        batch_dir = work / "batch-output"
        batch = subprocess.run([cli, sdk, work / "cache", "--jobs", "2", "gfx1013-compute-wave32@0",
                                "--requests", work, batch_dir], capture_output=True, text=True)
        assert batch.returncode == 1 and "requests=2 compiled=1 refused=1" in batch.stdout
        assert (batch_dir / f"{marker.stem[4:]}.cs.cso").read_bytes() == (work / "good.cso").read_bytes()
        assert (batch_dir / f"{marker.stem[4:]}.txt").is_file()
        assert not (batch_dir / f"{rejected.stem[4:]}.cs.cso").exists()
        assert not (batch_dir / f"{rejected.stem[4:]}.txt").exists()
        first = subprocess.run([gpu, sdk, marker, work / "cache"])
        if first.returncode:
            return first.returncode

        source.write_text('#define SIG "DescriptorTable(SRV(t0,numDescriptors=16)),'
                          'DescriptorTable(UAV(u0,numDescriptors=16)),RootConstants(num32BitConstants=4,b0)"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, source], check=True)
        sa = struct.pack("<7I", 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x198)
        # One 32-lane group per tile. s6/s7 are root constants and s12/s13 are the group IDs.
        code = [
            0xbe940304, 0xbe950380, 0xbe960305, 0xbe970380,
            0xf4080000 | (32 << 6) | 10, 0xfa000040,
            0xf40c0000 | (40 << 6) | 11, 0xfa0000e0,
            0x930f060d, 0x800f070f,                    # s15 = group.y * root[0] + root[1]
            0xd7460008, 12 | (133 << 9) | (256 << 18),  # v8 = (group.x << 5) + local.x
            0x80000000 | (0x31 << 23) | (15 << 16) | (129 << 8) | 15,  # (s15 << 4) + 1
            0x7c000000 | (0xc1 << 17) | (8 << 9) | 159,  # wave mask for 31 < x
            0xbe910383, 0xbf860001, 0xbe910389,           # s17 = 3 in tile 0, 9 in tile 1
            0x4a000000 | (10 << 17) | (8 << 9) | 17,     # v10 = x + s17
            0x7e12020d,                                 # v9 = group.y
            0xe0042000, 0x80080208,                     # v[2:3] = formatted[v8].xy
            0x4a06060f,                                 # v3 = v3 + s15
        ]
        for dst, src in [(2, 258), (3, 259), (4, 266), (5, 13)]:
            code.append(0x7e000000 | (dst << 17) | (6 << 9) | src)
        # Convert y/2 with ties-to-even, then round-trip through unsigned integer.
        code.extend([0x10000000 | (5 << 17) | (5 << 9) | 240,
                     0x7e000000 | (5 << 17) | (0x23 << 9) | 261,
                     0x7e000000 | (5 << 17) | (7 << 9) | 261,
                     0x7e000000 | (5 << 17) | (6 << 9) | 261])
        code.extend([0xf0200f08, (10 << 16) | (2 << 8) | 8, 0xbf810000])
        typed = write_request(work, code, root.read_bytes(), sa)
        subprocess.run([*args, typed, work / "typed.cso"], check=True)
        unbound = write_request(work, [*code[:-1], 0x800f000f, code[-1]], root.read_bytes(), sa)
        rejection = subprocess.run([*args, unbound, work / "unbound.cso"], capture_output=True, text=True)
        assert rejection.returncode and "unbound numeric SGPR s0" in rejection.stderr
        assert not (work / "unbound.cso").exists()
        subprocess.run([gpu, sdk, typed, work / "cache", "typed"], check=True)
        # Only the low 16 bits are stored. Distinct high bits expose an accidental high-half selection.
        for half, mode in ((0xc100, "half-value"), (0x4100, "half-mutant")):
            words = [0xbe960305, 0xbe970380, 0xf40c0000 | (40 << 6) | 11, 0xfa0000e0,
                     0xd7460008, 12 | (133 << 9) | (256 << 18), 0x7e12020d,
                     0x7e0402ff, 0x45000000 | half,
                     0xf0200108, 0x80000000 | (10 << 16) | (2 << 8) | 8, 0xbf810000]
            marker = write_request(work, words, root.read_bytes(), sa)
            result = subprocess.run([gpu, sdk, marker, work / "cache", mode], capture_output=True, text=True)
            mutant = mode == "half-mutant"
            assert result.returncode == int(mutant), (result.stdout, result.stderr)
            print(mode, result.stdout.strip(), flush=True)
        sdwa_fixtures(args, gpu, sdk, work, root.read_bytes(), sa)
        raw_offset_fixtures(args, gpu, sdk, work, root.read_bytes(), sa)
        global_stride_fixtures(gpu, sdk, work, root, source, sa)
        return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
