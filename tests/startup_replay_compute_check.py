"""Compare a synthetic compute dispatch's live and replayed title-memory output."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as compute
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl

CALL_PREFIX = struct.Struct("<QIHHQ")
CHUNK = struct.Struct("<4sIQ")
OUTPUT = struct.Struct(f"<{compute.WORDS}I")
SENTINELS = OUTPUT.pack(*([compute.SENTINEL] * compute.WORDS))


def chunks(data):
    at = 0
    while at < len(data):
        tag, header_bytes, payload_bytes = CHUNK.unpack_from(data, at)
        header = at + CHUNK.size
        end = header + header_bytes + payload_bytes
        assert end <= len(data), (tag, end, len(data))
        yield tag, at, header, end
        at = end


def calls(data):
    for tag, _, header, _ in chunks(data):
        if tag == b"CALL":
            seq, _, kind, slot, _ = CALL_PREFIX.unpack_from(data, header)
            yield seq, kind, slot, header + CALL_PREFIX.size


def append_late_output_snapshot(data, output):
    # Repeat the first pre-dispatch mapping of the UAV after all calls. It restores the
    # recorded sentinel if the player takes its dump after applying the trailing maps.
    for tag, start, header, end in chunks(data):
        if tag == b"MAP ":
            address, size, _ = struct.unpack_from("<3Q", data, header)
            if address <= output and output + OUTPUT.size <= address + size:
                late = bytearray(data[start:end])
                struct.pack_into("<Q", late, CHUNK.size + 16, (1 << 64) - 2)
                return data + late
    raise AssertionError("recording has no base snapshot for the placed UAV")


def record(directory, consumer, env):
    recording = directory / "dispatch.x2wr"
    live, _ = dispatch(directory, consumer, dict(
        env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
        X2W_REPLAY_PAGES="1", X2W_REPLAY_TITLE_RANGES="1", X2W_REPLAY_PAGES_AT_SUBMIT="1"), "dispatch")
    assert live == compute.expected("A"), "live dispatch differs from the synthetic CPU reference"
    data = recording.read_bytes()
    recorded_calls = list(calls(data))
    # The slot-67 description is recorded as the stand-in keeps it (0x300 bytes) with its (pointer, size) parts one
    # level below: state blocks A and B, code and data (words 1, 3, 5 and 7).
    [create] = [seq for seq, kind, slot, _ in recorded_calls if (kind, slot) == (0, 67)]
    parts = {}
    for tag, _, header, _ in chunks(data):
        if tag == b"ARGM":
            seq, arg, depth, _, length, _ = struct.unpack_from("<QIIQQQ", data, header)
            if seq == create:
                parts[arg] = (depth, length)
    assert parts.get(0) == (1, 0x300), parts
    assert {0x200, 0x400, 0x600, 0x800} <= set(parts) and all(d == 0 for a, (d, _) in parts.items() if a), parts
    # Case A creates the input and then the UAV with slot 133. Its first argument is the title address.
    placed = [struct.unpack_from("<Q", data, args)[0] for _, kind, slot, args in recorded_calls if (kind, slot) == (0, 133)]
    assert len(placed) == 2 and placed[0] != placed[1], placed
    dump = f"{placed[1]:x}:{OUTPUT.size}"
    data = append_late_output_snapshot(data, placed[1])
    recording.write_bytes(data)
    return recording, data, recorded_calls, dump, OUTPUT.pack(*live)


def replay(player, recording, out, adapter, dump, groups):
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_")}
    result = subprocess.run([player, recording, out, "--adapter", adapter, "--dump", dump],
                            cwd=out.parent, env=env, capture_output=True, text=True, timeout=90)
    log_path = out / "stderr.txt"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr, log[-3000:])
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["played"] == summary["calls"] and summary["fence_timeouts"] == 0, summary
    assert summary["pages"]["applied"] > 0 and summary["pages"]["missing"] == 0 and summary["pages"]["bad_maps"] == 0, summary
    assert f"Dispatch({groups}, 2, 2) translated:" in log, log[-3000:]
    content = (out / "dump.bin").read_bytes()
    assert len(content) == OUTPUT.size == 1280, (len(content), OUTPUT.size)
    return content


def main(consumer, player, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        _, first = dispatch(directory, consumer, env, "dispatch")
        published = lines(first, "x2w-draw compute request")
        assert published and " published " in published[0], first.stderr[-2000:]
        key = int(published[0].split()[3], 16)
        result = subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], check=True,
                                capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
        assert f"{key:016x}: ok" in result.stdout, result.stdout

        recording, data, recorded_calls, dump, expected = record(directory, consumer, env)
        actual = replay(player, recording, directory / "replay", adapter, dump, 3)
        if actual != expected:
            first_byte = next(i for i, (got, want) in enumerate(zip(actual, expected)) if got != want)
            raise AssertionError(f"replay differs at byte {first_byte}: {actual[first_byte]:#04x} vs {expected[first_byte]:#04x}")

        dispatch_args = [args for _, kind, slot, args in recorded_calls if (kind, slot) == (4, 14)]
        assert len(dispatch_args) == 1, dispatch_args
        args, = dispatch_args
        assert struct.unpack_from("<3Q", data, args) == (3, 2, 2)
        mutated = bytearray(data)
        struct.pack_into("<Q", mutated, args, 1)
        mutant = directory / "fewer_groups.x2wr"
        mutant.write_bytes(mutated)
        reduced = replay(player, mutant, directory / "replay_mutant", adapter, dump, 1)
        assert reduced != expected, "reducing the dispatch group count left all output bytes unchanged"
        assert reduced != SENTINELS, "reduced dispatch wrote nothing"
        print(f"{adapter}: replay matched all {OUTPUT.size} bytes. Reducing the dispatch from 3 to 1 X groups changed the output.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
