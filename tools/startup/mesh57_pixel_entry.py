"""Read a reviewed pixel entry only when a slot 57 creation archive and all six profile spans match."""
import argparse
from pathlib import Path
import struct
import sys

from gfx2hlsl import fnv, mesh57_key


def pixel_entry(request, archive, profile):
    ms, ps, root = (request.with_suffix(ext).read_bytes() for ext in (".m57", ".ps57", ".root"))
    key = mesh57_key(ms, ps, root)
    if request.name != f"req_{key:016x}.ps57":
        raise ValueError("raw MS, PS or creation root disagrees with request key")
    data = archive.read_bytes()
    if len(data) < 20:
        raise ValueError("truncated fixed creation archive")
    recorded, desc_size, a_size, b_size = struct.unpack_from("<Q3I", data)
    if recorded != key or desc_size != 0x300 or max(a_size,b_size) > 0x1000 or len(data) != 20 + desc_size + a_size + b_size:
        raise ValueError("creation archive key or lengths disagree")
    desc = data[20:20 + desc_size]
    a = data[20 + desc_size:20 + desc_size + a_size]
    b = data[20 + desc_size + a_size:]
    fields = struct.unpack_from("<15Q", desc)
    if (any(fields[i] for i in (0,1,3,5,9,10,13)) or
            (fields[2], fields[4], fields[6], fields[14]) != (len(a), len(b), len(ps), len(ms))):
        raise ValueError("normalized mesh descriptor disagrees with request and state lengths")
    variant = key
    for part in (desc, a, b):
        variant = fnv(part, fnv(struct.pack("<Q", len(part)), variant))
    if archive.name != f"req_{variant:016x}.s57" or profile.name != f"{variant:016x}.mesh57-state":
        raise ValueError("fixed-state variant disagrees with archive or profile name")
    data = profile.read_bytes()
    # Mesh57Profile has a fixed Win32 ABI; the static assertion in mesh57_profile.h protects this offset.
    header_size = 532
    if len(data) < header_size + 12 or data[:8] not in (b"X2WM572\0", b"X2WM573\0", b"X2WM574\0", b"X2WM575\0", b"X2WM576\0"):
        raise ValueError("profile lacks a reviewed pixel entry")
    enable, address, wave = struct.unpack_from("<3I", data, header_size)
    if address > 0xffff or enable & ~address or wave not in (32, 64):
        raise ValueError("invalid reviewed pixel entry")
    parts = (desc, a, b, root, ms, ps)
    if struct.unpack_from("<6I", data, 8) != tuple(map(len, parts)):
        raise ValueError("profile span lengths disagree")
    offset = header_size + 12
    version = data[6] - ord("0")
    if version >= 3:
        offset += 12  # MS constant mapping
    if version >= 4:
        if offset + 4 > len(data):
            raise ValueError("truncated view count")
        count, = struct.unpack_from("<I", data, offset)
        if not 1 <= count <= 15:
            raise ValueError("invalid MS view entry count")
        offset += 4 + 12 * count
    if version >= 5:
        offset += 4  # MS resource heap SGPR
    for part in parts:
        if data[offset:offset + len(part)] != part:
            raise ValueError("profile creation or shader bytes differ")
        offset += len(part)
    if offset != len(data):
        raise ValueError("trailing profile data")
    return f"meshps1 {enable} {address} {wave}" + (" guest_root" if version == 6 else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="req_<request>.ps57 with adjacent .m57 and .root")
    parser.add_argument("archive", type=Path, help="req_<variant>.s57 from the same creation")
    parser.add_argument("reviewed_profile", type=Path, help="<variant>.mesh57-state carrying reviewed pixel entry metadata")
    args = parser.parse_args()
    try:
        print(pixel_entry(args.request, args.archive, args.reviewed_profile))
    except (OSError, ValueError) as error:
        sys.exit(str(error))
