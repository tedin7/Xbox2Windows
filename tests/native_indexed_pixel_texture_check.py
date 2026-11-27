"""Verify bounded eight-DWORD PS texture lookups in the linked native path."""
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_direct_container_check import patched_pixel, publish, vertex, words


def texture_program(original, mask, opposite, base_offset=32):
    # Adapt the existing MRT draw. t0 holds one runtime byte offset and the UV
    # offset. t1 and t2 have different colors and different spatial patterns.
    source = list(struct.unpack(f"<{len(original) // 4}I", original))
    start = source.index(0xf4100000 | (16 << 6) | 10)
    assert source[start + 1] == 0xfa000020 and source[start + 3] == 0xfa000038

    # Move the table base to s[48:49] before reading s[16:23] and s[24:31].
    source[0:2] = [0xbe800300 | (48 << 16) | 8, 0xbe800380 | (49 << 16)]
    source[start - 2] = (source[start - 2] & ~63) | 24
    offset_load = source[start + 2:start + 4]

    # Explicit words of the fixture's point sampler (2 bits set at bit 54).
    sampler = [0xbeac0380, 0xbead03ff, 0x00c00000, 0xbeae0380, 0xbeaf0380]
    # Define untouched halves of the fixture's packed export VGPRs.
    initial = [0x7e000280 | (dst << 17) for dst in (47, 80, 81)]

    def and_mask(dst, src):
        return 0x87000000 | (dst << 16) | ((128 + mask) << 8) | src

    # s40 = s12 & mask; s41 = (s12 ^ toggle) & mask. Select each
    # descriptor from the same base byte offset in the same bound root table.
    toggle = 32 if opposite else 0
    lookups = [*offset_load, and_mask(40, 12),
               0xf40c0000 | (16 << 6) | 24, (40 << 25) | base_offset,
               0x89000000 | (41 << 16) | ((128 + toggle) << 8) | 12,
               and_mask(41, 41), 0xf40c0000 | (24 << 6) | 24, (41 << 25) | base_offset]
    return words(*initial, *source[:start], *lookups, *sampler, *source[start + 7:])


def run(work, writer, compiler, sdk, adapter):
    subprocess.run([writer, "--write-fixture", work], check=True)
    vs, ps, root = [(work / name).read_bytes() for name in ("vertex.bin", "pixel.bin", "root.bin")]
    cache, output = work / "cache", work / "objects"
    for name, mask, offset, refusal in (
        ("distinct", 32, 32, ""), ("same", 32, 32, ""),
        ("unaligned", 31, 32, "32-byte-aligned SRV offset"),
        ("short-table", 32, 96, "exceeds its declared root range"),
    ):
        program = texture_program(ps, mask, name != "same", offset)
        key = publish(work / name, program, vs, root, ps_inputs=(2, 2))
        result = subprocess.run([compiler, sdk, cache, "gfx1013-container-pair",
                                 "--requests", work / name, output], capture_output=True, text=True)
        if refusal:
            assert result.returncode and refusal in result.stderr, result.stderr
            print(name, refusal)
            continue
        assert result.returncode == 0, result.stderr
        manifest = (output / f"{key}.txt").read_text().splitlines()
        assert {"srv 1 tex2d table 0 32", "srv 2 tex2d table 0 64"} <= set(manifest), manifest
        result = subprocess.run([writer, adapter, sdk, cache, output / f"{key}.ps.cso", output / f"{key}.vs.cso"],
                                text=True, capture_output=True)
        print(name, result.returncode, result.stdout.strip(), result.stderr.strip())
        if result.returncode == 77:
            return 77
        if name == "distinct":
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode and "sampled MRT mismatch" in result.stderr, result.stderr

    # Type 5 denotes the global resource heap. A real table at root 0 exists,
    # but a load via this unrelated ABI input must never resolve to that table.
    heap = [0xbe810380, 0x8702a002, 0xf40c0200, 2 << 25, 0x7e000280, 0x7e020280,
            0xf0000108, 0x00020000]
    pixel = patched_pixel(len(heap), heap, root_inputs=((5, 0, 1, 0, 0), (0, 0, 1, 0, 0x102)))
    request = work / "global-heap"
    key = publish(request, vertex(), pixel, root)
    result = subprocess.run([compiler, sdk, cache, "gfx1013-container-pair",
                             "--requests", request, output], capture_output=True, text=True)
    assert result.returncode and not (output / f"{key}.txt").exists(), result.stderr
    assert "direct descriptor address outside bound table" in result.stderr, result.stderr
    print("global-heap", result.stderr.strip())
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        sys.exit(run(Path(folder), *sys.argv[1:]))
