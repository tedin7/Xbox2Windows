#!/usr/bin/env python3
"""Play synthetic v2/v3 CALLs and reject CALLs with invalid sizes or slots."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_replay_pages_play_check import BASE, PAGE, SPAN, chunk, page_map

CALL = {2: struct.Struct("<QIHHQ7QQdii"), 3: struct.Struct("<QIHHQ11QQdii")}
DEVICE = 0x1234


def recording(version, *, wire_version=None, slot=15, tail=False):
    wire_version = version if wire_version is None else wire_version
    args = [2] + [0] * (6 if wire_version == 2 else 10)  # RTV descriptor stride
    data = chunk(b"XREC", struct.pack("<IIQ", version, CALL[version].size, 10_000_000))
    data += chunk(b"XENV", data=b"X2W_D3D12_PERMISSIVE=1\0")
    data += chunk(b"CALL", CALL[wire_version].pack(1, 1, 0, slot, DEVICE, *args, 32, 0.0, 0, 0))
    if tail:
        # The last unused stack word looks like a pointer to the last byte of a captured range.
        # The player must not read a full word there to discover outputs.
        address = BASE + SPAN - 1
        args[10] = address
        data += chunk(b"CALL", CALL[3].pack(2, 1, 0, 15, DEVICE, *args, 32, 0.0, 0, 0))
        data += chunk(b"ARGM", struct.pack("<QIIQQQ", 2, 10, 0, address, 1, 1))
        data += chunk(b"BLOB", struct.pack("<QQ", 1, 1), b"\0")
        data += page_map(BASE, SPAN // PAGE, 1, [(0, SPAN // PAGE)])
    return data


def pre_install_device():
    """A recording that starts after the device existed: its first device call is the struct-return form (object = the
    title's return buffer, the device second), and a second device-kind object never named by an OUTP appears later."""
    stack, second = 0x7253F9E0, 0x5678
    data = chunk(b"XREC", struct.pack("<IIQ", 3, CALL[3].size, 10_000_000))
    data += chunk(b"XENV", data=b"X2W_D3D12_PERMISSIVE=1\0")
    calls = [(26, stack, [DEVICE, 0, 1]), (15, DEVICE, [2]), (15, DEVICE, [2]), (15, second, [2])]
    for seq, (slot, obj, args) in enumerate(calls, 1):
        data += chunk(b"CALL", CALL[3].pack(seq, 1, 0, slot, obj, *(args + [0] * (11 - len(args))), 0, 0.0, 0, 0))
    return data


def play(player, directory, name, data):
    path = directory / f"{name}.x2wr"
    path.write_bytes(data)
    out = directory / name
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("X2W_")}
    result = subprocess.run([player, str(path), str(out), "--adapter", "warp"], env=env,
                            capture_output=True, text=True, timeout=25)
    log = out / "stderr.txt"
    diagnostic = result.stdout + result.stderr + (log.read_text(errors="replace") if log.exists() else "")
    return result.returncode, diagnostic, out


def main(player):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for version, wire, slot, message in (
            (2, 3, 15, "unexpected CALL size"),
            (3, 2, 15, "unexpected CALL size"),
            (3, 3, 512, "CALL slot out of range"),
        ):
            code, text, _ = play(player, directory, f"bad-{version}-{wire}-{slot}",
                                 recording(version, wire_version=wire, slot=slot))
            assert code == 2 and message in text, (code, text[-2500:])

        for version in (2, 3):
            code, text, out = play(player, directory, f"v{version}", recording(version, tail=version == 3))
            if code == 77:
                print("No D3D12 WARP adapter")
                return 77
            assert code == 0, (version, code, text[-2500:])
            summary = json.loads((out / "replay_summary.json").read_text())
            calls = 1 if version == 2 else 2
            assert summary["version"] == version and summary["calls"] == summary["played"] == calls, summary
            assert summary["unknown_object"] == summary["missing_blob"] == 0, summary
            assert summary["pages"]["title_memory_args"] == calls - 1, summary

        # The device is the object most device calls name, not the return buffer of a first struct-return call; a
        # device-kind object no OUTP named is adopted as the player's device.
        code, text, out = play(player, directory, "pre-install", pre_install_device())
        assert code == 0, (code, text[-2500:])
        summary = json.loads((out / "replay_summary.json").read_text())
        assert summary["calls"] == summary["played"] == 4 and summary["unknown_object"] == 0, summary
        assert "replay: adopted pre-install wrapper 0 0x5678" in text, text[-2500:]
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
