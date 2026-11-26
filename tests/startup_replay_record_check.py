#!/usr/bin/env python3
"""X2W_REPLAY_RECORD=all writes every call the title makes into the stand-in to a replay recording
(docs/startup-runtime.md#d3d12-replay-record), and recording changes nothing the title sees.

The stream draw fixture draws a green quad through the stand-in twice: once plain, once recorded. Checked:
  - the recorded run draws the same pixels;
  - the file is well formed: XREC first, XEND last, every chunk complete;
  - XEND's call count is the number of CALL chunks; every call has its own sequence number;
  - device calls (the creations) and command-list calls are in it, and creations name the wrapper they returned (OUTP);
  - every ARGM names a BLOB the file holds.

usage: startup_replay_record_check.py <startup_stream_consumer> <warp|hardware>
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

DEVICE_KIND, LIST_KIND = 0, 4
CALL = struct.Struct("<QIHHQ7QQdii")


def chunks(data):
    """(tag, header, payload) for every chunk; raises on a cut chunk."""
    at = 0
    while at < len(data):
        if at + 16 > len(data):
            raise AssertionError("cut chunk head at %d" % at)
        tag, header_bytes, data_bytes = struct.unpack_from("<4sIQ", data, at)
        end = at + 16 + header_bytes + data_bytes
        if end > len(data):
            raise AssertionError("cut chunk %r at %d" % (tag, at))
        yield tag.decode("ascii"), data[at + 16:at + 16 + header_bytes], data[at + 16 + header_bytes:end]
        at = end


def summarize(path):
    calls, blobs, argm, outputs, kinds = [], set(), [], 0, {}
    tags = []
    for tag, header, payload in chunks(path.read_bytes()):
        tags.append(tag)
        if tag == "CALL":
            seq, _thread, kind, slot, _obj, *rest = CALL.unpack(header)
            calls.append(seq)
            kinds[kind] = kinds.get(kind, 0) + 1
        elif tag == "BLOB":
            blob_hash, size = struct.unpack_from("<QQ", header)
            assert size == len(payload), (size, len(payload))
            blobs.add(blob_hash)
        elif tag == "ARGM":
            argm.append(struct.unpack_from("<QIIQQQ", header)[5])
        elif tag == "OUTP":
            outputs += 1
        elif tag == "XEND":
            end_calls = struct.unpack_from("<QQQQ", header)[0]
    return tags, calls, blobs, argm, outputs, kinds, end_calls if "XEND" in tags else None


def draw(args, env, backend, output):
    result = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    if backend == "hardware" and result.returncode == 77:
        return None
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError("draw failed on %s: %d\n%s" % (backend, result.returncode, result.stderr[-3000:]))
    return output.read_bytes()


def main(consumer, backend):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        cache = directory / "cache"
        cache.mkdir()
        vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
        vs.write_bytes(vertex())
        ps.write_bytes(pixel(True))
        args = [consumer, vs, ps, root]
        env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
                   X2W_SHADER_CACHE=str(cache))
        env.setdefault("X2W_D3D12_DEBUG", "0")
        subprocess.run(args, env=env, check=True)
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
        plain = draw(args, env, backend, output)
        if plain is None:
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        # X2W_REPLAY_DIR (tester_run -Record): the recording is replay.x2wr in that folder, made if missing.
        recording = directory / "record" / "replay.x2wr"
        recorded = draw(args, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_DIR=str(recording.parent)), backend, output)
        assert recorded == plain, "recording changed the picture"
        tags, calls, blobs, argm, outputs, kinds, end_calls = summarize(recording)
        assert tags[0] == "XREC" and tags[-1] == "XEND", (tags[:3], tags[-3:])
        assert end_calls == len(calls), (end_calls, len(calls))
        assert len(set(calls)) == len(calls), "two calls share a sequence number"
        assert kinds.get(DEVICE_KIND) and kinds.get(LIST_KIND), kinds
        assert outputs, "no creation recorded the wrapper it returned"
        assert all(h in blobs for h in argm), "an ARGM names a blob the file does not hold"
        print("replay record: %d calls (%s), %d blobs, %d argument memories, %d outputs, %d bytes" % (
            len(calls), ", ".join("kind %d: %d" % item for item in sorted(kinds.items())), len(blobs), len(argm),
            outputs, recording.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
