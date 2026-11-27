"""Compare two fenced compute-queue readbacks with and without the second submit MAP."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


BASE = 0x300000000000
PAGE = 4096
EXPECTED = b"\x2a" * PAGE + b"\x59" * PAGE
STALE = b"\x2a" * (2 * PAGE)


def without_second_submit_map(data):
    submits, maps, pages = [], [], []
    kept = bytearray()
    at = 0
    while at < len(data):
        tag, header_size, payload_size = struct.unpack_from("<4sIQ", data, at)
        header = at + 16
        end = header + header_size + payload_size
        assert end <= len(data), (tag, end, len(data))
        assert tag != b"PRES", "the fixture must never present"
        if tag == b"CALL":
            seq, _, kind, slot, *_ = struct.unpack_from("<QIHHQ11QQdii", data, header)
            assert (kind, slot) != (1, 32), "the fixture must never call Present"
            if (kind, slot) == (1, 10):
                submits.append(seq)
        if tag == b"PAGE":
            assert header_size == 16 and payload_size == PAGE, (header_size, payload_size)
            pages.append(data[header + header_size:end])
        if tag == b"MAP ":
            address, size, stamp, runs, _ = struct.unpack_from("<QQQII", data, header)
            assert address == BASE and size == 0x10000, (address, size)
            maps.append((stamp, runs))
            if len(maps) == 2:
                assert runs == 1 and payload_size == 28, (runs, payload_size)
                assert struct.unpack_from("<III", data, header + header_size) == (0, 1, 0)
                at = end
                continue
        kept.extend(data[at:end])
        at = end
    assert len(submits) == 2 and len(maps) == 2, (submits, maps)
    assert len(pages) == 2 and b"".join(pages) == EXPECTED, f"unexpected PAGE payloads ({len(pages)} chunks)"
    assert [stamp for stamp, _ in maps] == [seq - 1 for seq in submits], (submits, maps)
    return kept


def run(args, env):
    return subprocess.run([str(arg) for arg in args], env=env, capture_output=True, text=True, timeout=90)


def replay(player, recording, out, backend, env):
    result = run([player, recording, out, "--adapter", backend], dict(env, X2W_REPLAY_DUMP_MAP_RANGES="1"))
    log = out / "stderr.txt"
    diagnostics = log.read_text(errors="replace")[-2000:] if log.exists() else ""
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr, diagnostics)
    dumps = list(out.glob("map_*.bin"))
    assert len(dumps) == 1, (dumps, diagnostics)
    return dumps[0].read_bytes()


def compare(actual, expected):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for at, (got, want) in enumerate(zip(actual, expected)):
        assert got == want, f"readback byte {at}: got {got:#04x}, expected {want:#04x}"


def main(scene, player, backend):
    env = {key: value for key, value in os.environ.items() if not key.startswith("X2W_")}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        recording = root / "recording.x2wr"
        readback = root / "live.bin"
        result = run([scene, backend, readback], dict(
            env, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
            X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording), X2W_REPLAY_PAGES="1",
            X2W_REPLAY_PAGES_AT_SUBMIT="1", X2W_REPLAY_SCAN_MS="250", X2W_REPLAY_MAX_MIB="16"))
        if result.returncode == 77:
            print(result.stdout.strip() or "No suitable D3D12 adapter")
            return 77
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)

        live = readback.read_bytes()
        compare(live, EXPECTED)
        mutant = root / "without_second_map.x2wr"
        mutant.write_bytes(without_second_submit_map(recording.read_bytes()))
        compare(replay(player, recording, root / "full", backend, env), live)
        compare(replay(player, mutant, root / "mutant", backend, env), STALE)
        print("Two fenced readback pages match. Removing the second submit MAP leaves stale bytes.")

        # A 2 MiB page budget and a 4 MiB buffer of distinct pages: the base image stops at 3/4 of the budget, so the
        # second submit's changed page still reaches the file and the replay reads the late write.
        cut = root / "cut.x2wr"
        result = run([scene, backend, root / "cut_live.bin"], dict(
            env, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
            X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(cut), X2W_REPLAY_PAGES="1",
            X2W_REPLAY_PAGES_AT_SUBMIT="1", X2W_REPLAY_SCAN_MS="250", X2W_REPLAY_MAX_MIB="16",
            X2W_REPLAY_PAGES_MAX_MIB="2", X2W_TEST_SUBMIT_PAGES_BIG="1"))
        log = result.stdout + result.stderr
        assert result.returncode == 0, (result.returncode, log[-3000:])
        assert "the base image stopped at 1 MiB, 3/4 of the 2 MiB page budget" in log, log[-3000:]
        assert "budget of 2 MiB reached" not in log, log[-3000:]
        compare(replay(player, cut, root / "cut", backend, env), EXPECTED)
        print("With the base image cut at 3/4 of a 2 MiB page budget the late write still replays.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
