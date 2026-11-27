#!/usr/bin/env python3
"""x2w_replay writes a recording's memory pages at their recorded addresses (docs/startup-runtime.md#d3d12-replay-record).

A recording made here (no title, no calls) holds pages for one range at 0x300000000000: a first MAP with pages A, B, A (a
duplicate) and a zero run, then a later MAP that changes page 1 to C. Checked:
  - the replay reserves the range at its address and ends with the image A, C, A, zeros (its hash in replay_summary.json);
  - control: without the second MAP the image is A, B, A, zeros (each MAP is applied, in order);
  - a range that cannot be reserved at its address (kernel space) stops the replay with exit 3 and the address.

A call-only recording checks that a Release for a missing resource cannot use a stale argument as its receiver.

usage: startup_replay_pages_play_check.py <x2w_replay>
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

BASE = 0x300000000000
PAGE = 4096
SPAN = 0x10000
SETTINGS = b"X2W_D3D12_PERMISSIVE=1\0X2W_D3D12_LIST_NOOP=1\0X2W_D3D12_RENDER=1\0"


def chunk(tag, header=b"", data=b""):
    return struct.pack("<4sIQ", tag, len(header), len(data)) + header + data


def page_map(address, pages, seq, runs):
    """runs: (first, [keys]) for content runs, (first, count) for zero runs."""
    body, count = b"", 0
    for first, what in runs:
        if isinstance(what, int):
            body += struct.pack("<III", first, what, 1)
        else:
            body += struct.pack("<III", first, len(what), 0) + b"".join(what)
        count += 1
    return chunk(b"MAP ", struct.pack("<QQQII", address, pages * PAGE, seq, count, 0), body)


def recording(path, address, second_map=True):
    keys = {name: bytes([i + 1]) * 16 for i, name in enumerate("ABC")}
    content = {name: bytes([0x40 + i]) * PAGE for i, name in enumerate("ABC")}
    data = chunk(b"XREC", struct.pack("<IIQ", 2, 104, 10000000)) + chunk(b"XENV", data=SETTINGS)
    data += b"".join(chunk(b"PAGE", keys[n], content[n]) for n in "AB")
    data += page_map(address, 8, 1, [(0, [keys["A"], keys["B"], keys["A"]]), (3, 5)])
    if second_map:
        data += chunk(b"PAGE", keys["C"], content["C"]) + page_map(address, 8, 5, [(1, [keys["C"]])])
    data += chunk(b"XEND", struct.pack("<QQQQ", 0, 0, len(data), 0))
    path.write_bytes(data)
    return content


def fnv(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % value


def play(player, recording_path, out):
    result = subprocess.run([str(player), str(recording_path), str(out), "--adapter", "warp"], capture_output=True, text=True,
                            timeout=120)
    return result.returncode, result.stdout + result.stderr


def check_missing_release(player, directory):
    device, missing_resource = 0x10000000, 0x20000000
    call = struct.Struct("<QIHHQ11QQdii")
    data = chunk(b"XREC", struct.pack("<IIQ", 3, call.size, 10000000)) + chunk(b"XENV", data=SETTINGS)
    # Seed the device mapping with AddRef and balance it with Release. The third call models a Release whose
    # resource creation produced no replay mapping and whose unused RDX still names the device.
    for seq, kind, slot, obj, rdx, ret in (
        (1, 0, 1, device, 0, 2),
        (2, 0, 2, device, 0, 1),
        (3, 7, 2, missing_resource, device, 0),
    ):
        data += chunk(b"CALL", call.pack(seq, 1, kind, slot, obj, rdx, *([0] * 10), ret, 0.0, 0, 0))
    path = directory / "missing_release.x2wr"
    path.write_bytes(data + chunk(b"XEND", struct.pack("<QQQQ", 3, 0, len(data), 0)))
    out = directory / "missing_release"
    code, text = play(player, path, out)
    assert code == 0, "missing resource Release: exit %d\n%s" % (code, text[-2000:])
    summary = json.loads((out / "replay_summary.json").read_text())
    assert (summary["calls"], summary["played"], summary["unknown_object"]) == (3, 2, 1), summary


def main(player):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        cases = {"both maps": ("ACA", True), "first map only": ("ABA", False)}
        for name, (pages, second) in cases.items():
            content = recording(directory / "pages.x2wr", BASE, second)
            out = directory / name.replace(" ", "_")
            code, text = play(player, directory / "pages.x2wr", out)
            if code == 77:
                print("No D3D12 WARP adapter")
                return 77
            assert code == 0, "%s: exit %d\n%s" % (name, code, text[-2000:])
            summary = json.loads((out / "replay_summary.json").read_text())
            expected = b"".join(content[p] for p in pages) + bytes(SPAN - 3 * PAGE)
            ranges = summary["ranges"]
            assert ranges == [{"address": "0x%x" % BASE, "bytes": SPAN, "hash": fnv(expected)}], (name, ranges, fnv(expected))
            assert summary["pages"]["applied"] == (2 if second else 1) and summary["pages"]["missing"] == 0, summary["pages"]

        recording(directory / "taken.x2wr", 0xFFFF800000000000)
        code, text = play(player, directory / "taken.x2wr", directory / "taken")
        assert code == 3 and "0xffff800000000000" in text, (code, text[-2000:])
        check_missing_release(player, directory)
        print("replay pages: image A,C,A,zeros at 0x%x; first map only A,B,A; a range that cannot be had stops with exit 3"
              % BASE)
        print("replay Release: known device calls played; missing resource with a stale RDX skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
