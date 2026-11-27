#!/usr/bin/env python3
"""Replay the mapped slot 133 creations from the source-built resource budget scene."""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


def chunks(data):
    at = 0
    while at < len(data):
        tag, header_size, body_size = struct.unpack_from("<4sIQ", data, at)
        header = at + 16
        end = header + header_size + body_size
        yield tag, data[header:header + header_size], data[at:end]
        at = end


def select_creations(data):
    records = list(chunks(data))
    placements = [struct.unpack("<5Q", header) for tag, header, _ in records if tag == b"HPLC"]
    assert len(placements) == 2, placements
    first, second = placements
    assert first[1] > 0 and first[1:4] == second[1:4], placements
    assert (first[4], second[4]) == (0, 0x10000), placements
    sequences = {first[0], second[0]}
    clean, poisoned, legacy = bytearray(), bytearray(), bytearray()
    decoy_base = 0x310000000000
    assert not first[2] <= decoy_base < first[2] + first[3]
    decoy = struct.pack("<4sIQQQQII", b"MAP ", 32, 0, decoy_base, 0xC0000, 0, 0, 0)
    for tag, header, raw in records:
        if tag in (b"CALL", b"HPLC", b"ARGM", b"OUTP"):
            seq, = struct.unpack_from("<Q", header)
            if seq not in sequences:
                continue
        elif tag not in (b"XREC", b"XENV", b"BLOB"):
            continue
        clean.extend(raw)
        if tag == b"CALL":
            _, _, kind, slot, _, address = struct.unpack_from("<QIHHQQ", header)
            assert (kind, slot) == (0, 133)
            placement = first if seq == first[0] else second
            assert address == placement[2] + placement[4]
        if tag == b"CALL" and seq == second[0]:
            # Feed the address-based call an interior pointer in place of the second buffer's address.
            raw = bytearray(raw)
            struct.pack_into("<Q", raw, 16 + 24, decoy_base + 0xB8700)
        poisoned.extend(raw)
        if tag != b"HPLC" or seq != second[0]:
            legacy.extend(raw)
    expected = [(seq, heap, offset) for seq, heap, _, _, offset in placements]
    return clean, poisoned + decoy, legacy + decoy, expected


def play(player, backend, env, directory, name, data):
    path = directory / (name + ".x2wr")
    path.write_bytes(data)
    output = directory / name
    result = subprocess.run([player, str(path), str(output), "--adapter", backend], env=env,
                            capture_output=True, text=True, timeout=60)
    log = (output / "stderr.txt").read_text(errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr + log
    assert "played 2 of 2 calls (0 on objects never created, 0 without their memory)" in log, log
    matches = re.findall(r"slot 133 seq (\d+) heap id (\d+) offset (0x[0-9a-f]+) hr=0x00000000", log)
    return [(int(seq), int(heap), int(offset, 16)) for seq, heap, offset in matches], log


def main(scene, player, backend):
    env = {key: value for key, value in os.environ.items() if not key.startswith("X2W_")}
    env.update(X2W_D3D12_BUDGET="1", X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        capture = directory / "resource_budget.x2wr"
        recorded = subprocess.run([scene, backend], env=dict(env, X2W_REPLAY_RECORD="all",
                                  X2W_REPLAY_RECORD_FILE=str(capture), X2W_REPLAY_PAGES="0"),
                                  capture_output=True, text=True, timeout=60)
        if recorded.returncode == 77:
            print("Adapter unavailable")
            return 77
        assert recorded.returncode == 0, recorded.stdout + recorded.stderr
        if "mapped-heap subcase unavailable:" in recorded.stdout:
            print("Adapter lacks D3D12_FEATURE_EXISTING_HEAPS")
            return 77
        clean, poisoned, legacy, expected = select_creations(capture.read_bytes())
        actual, log = play(player, backend, env, directory, "clean", clean)
        assert actual == expected, log
        actual, log = play(player, backend, env, directory, "interior_address", poisoned)
        assert actual == expected, log
        actual, log = play(player, backend, env, directory, "legacy_control", legacy)
        assert actual == expected[:1], log
        assert re.search(r"Device::slot133.*offset=0xb8700 .*-> 0x00000000 not aliased", log), log
        print("resource_budget placements replay equal at offsets 0 and 0x10000. "
              "The address-only control uses offset 0xb8700 and committed backing.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
