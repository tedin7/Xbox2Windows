"""Compile raw MUBUF loads with cache hints and compare hardware UAV readback."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import pack, words


GLC_DLC = (1 << 14) | (1 << 15)
SLC = 1 << 22
REFUSAL = "raw buffer load needs a linear byte offset and no modifiers"
VIEWS = """x2w-global-srv-views 2
select 0
need raw_buffer
slot 0 raw_buffer 1 256
slot 1 raw_buffer 1 256
slot 2 texture 0 0
slot 3 typed_buffer 4 256
"""


def container(offen, cache_flags, extra_flags):
    # Dispatch(1, 1, 1) runs group (0,0,0), with one wave32 and local IDs in v0.
    # EXEC enables lanes 0..7. Each lane writes the first dword of a 16-byte u0 slot.
    soffset = 128 if offen else 18
    code = words(
        0xbe810380,  # s_mov_b32 s1, 0
        0xbe830380,  # s_mov_b32 s3, 0
        0xf4080200, 0xfa000000,  # s_load_dwordx4 s[8:11], s[0:1], 0 (selector t0)
        0xf4080300, 0xfa000020,  # s_load_dwordx4 s[12:15], s[0:1], 32 (output u0)
        0xf4200404, 0xfa000000,  # s_buffer_load_dword s16, s[8:11], 0
        0xbf8cc07f,  # s_waitcnt lgkmcnt(0)
        0x8f118510,  # s_lshl_b32 s17, s16, 5
        0xf4080501, 0x22000000,  # s_load_dwordx4 s[20:23], s[2:3], s17 (selected heap SRV)
        0xbf8cc07f,  # s_waitcnt lgkmcnt(0)
        0x34020084,  # v_lshlrev_b32_e32 v1, 4, v0
        0xbe92038c,  # s_mov_b32 s18, 12
        0xbefe03ff, 0x000000ff,  # s_mov_b32 exec_lo, 0xff
        0xbeff0380,  # s_mov_b32 exec_hi, 0
        0xe0300000 | (int(offen) << 12) | cache_flags,
        (soffset << 24) | 0x00050200 | int(offen) | extra_flags,  # v2, v1/off, s[20:23]
        0xbf8c3f70,  # s_waitcnt vmcnt(0)
        0xe0701000, 0x80030201,  # buffer_store_dword v2, v1, s[12:15], 0 offen
        0xbf810000,  # s_endpgm
    )

    # Same two root inputs as native_global_srv_heap_replay_check.py.
    body = bytearray(256)
    struct.pack_into("<Q", body, 0, 6)
    for offset in (8, 16, 24, 32, 40, 48, 72):
        struct.pack_into("<Q", body, offset, 0xDEADBEEFDEADBEEF)
    record = 112
    struct.pack_into("<Q", body, 96, record)
    struct.pack_into("<Q", body, record + 40, len(body))
    struct.pack_into("<I", body, record + 56, 2)
    body.extend(struct.pack("<IIIIBBH", 3, 0, 1, 0, 0, 0, 0))
    body.extend(struct.pack("<IIIIBBH", 5, 0, 1, 0, 2, 0, 0))
    struct.pack_into("<QQII", body, record, len(body), len(code), 1 << 20, 32)
    struct.pack_into("<II", body, record + 76, 512, 6)
    struct.pack_into("<IIII", body, record + 120, 0x8400 | 0x1008, 32, 1, 1)
    body.extend(code)
    return pack(body)


def request_key(root, cdx):
    key = 1469598103934665603
    for byte in b"native-container-1" + struct.pack("<Q", len(root)) + root + cdx:
        key = ((key ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{key:016x}"


def run(*command):
    return subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=100)


def main(fixture, compile_dxil, sdk):
    with tempfile.TemporaryDirectory(prefix="x2w_raw_load_cache_") as scratch:
        base = Path(scratch)
        cache = base / "cache"
        cache.mkdir()
        root_path = base / "fixture.root"
        result = run(fixture, "root", root_path)
        if result.returncode != 0:
            raise AssertionError((result.stdout, result.stderr))
        root = root_path.read_bytes()

        replays = []
        for name, offen, cache_flags, extra_flags, refused in (
            ("offen_glc_dlc", True, GLC_DLC, 0, False),
            ("soffset_glc_dlc", False, GLC_DLC, 0, False),
            ("offen_slc", True, 0, SLC, False),
            ("offen_lds", True, GLC_DLC | (1 << 16), 0, True),
            ("soffset_tfe", False, GLC_DLC, 1 << 23, True),
        ):
            directory = base / name
            directory.mkdir()
            cdx = container(offen, cache_flags, extra_flags)
            key = request_key(root, cdx)
            request = directory / f"req_{key}.cdx"
            views = request.with_suffix(".views")
            request.write_bytes(cdx)
            request.with_suffix(".root").write_bytes(root)
            views.write_text(VIEWS, encoding="ascii", newline="\n")
            out_dir = base / f"{name}_out"
            cso = out_dir / f"{key}.cs.cso"
            result = run(compile_dxil, sdk, cache, "gfx1013-container-cs", "--requests", directory, out_dir)
            if refused:
                if result.returncode != 1 or REFUSAL not in result.stderr or cso.exists():
                    raise AssertionError((name, result.stdout, result.stderr))
                continue
            if result.returncode != 0 or "compiled=1 refused=0" not in result.stdout or not cso.exists():
                raise AssertionError((name, result.stdout, result.stderr))
            # Slot 0 starts at 0xA0000000 and increases by one per dword.
            # OFFEN reads source word 4*lane. SOFFSET reads source word 3 in every lane.
            expected = [0x69696969] * 72
            for lane in range(8):
                expected[4 * lane] = 0xA0000000 + (4 * lane if offen else 3)
            reference = directory / "expected.bin"
            reference.write_bytes(words(*expected))
            replays.append((name, request, cso, views, reference))

        for name, request, cso, views, reference in replays:
            result = run(fixture, "replay", sdk, cache, request, cso, views, "hardware", "ok", "0", reference)
            if result.returncode == 77:
                print(result.stdout, result.stderr)
                return 77
            if result.returncode != 0 or "output matches CPU reference (72 words)" not in result.stdout:
                raise AssertionError((name, result.stdout, result.stderr))
            print(f"{name} matched the 72-word hardware readback")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
