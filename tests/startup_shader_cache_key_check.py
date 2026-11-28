"""Validate stored source identity before startup translation or output publication."""
import importlib
import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "startup"))
gfx2hlsl = importlib.import_module("gfx2hlsl")

VERTEX = struct.pack("<3I", 0xf800080f | (12 << 4), 0, 0xbf810000)
PIXEL = struct.pack("<3I", 0xf800080f, 0, 0xbf810000)
ROOT = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 68, 1, 36)
        + b"RTS0" + struct.pack("<I", 24) + struct.pack("<6I", 1, 0, 24, 0, 24, 0))
BASE_MISMATCH = "shader request source key mismatch"


def check_request(blobs, key, diagnostic=None, rtv_request=None, layout=None):
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        base = int(rtv_request.split()[1]) if rtv_request else key
        for suffix, data in zip((".a", ".b", ".root"), blobs):
            (cache / f"req_{base:016x}{suffix}").write_bytes(data)
        if layout is not None:
            (cache / f"req_{base:016x}.ia").write_bytes(layout)
        if rtv_request:
            (cache / f"req_{key:016x}.rtv").write_text(rtv_request, encoding="ascii")
        # Explicit upstream manifest-only mode; real hashing, parsing and HLSL translation.
        # The wrapping spy observes translation; it never replaces its behavior.
        with patch.dict(os.environ, {"X2W_FXC": ""}), patch.object(
                gfx2hlsl, "translate_to", wraps=gfx2hlsl.translate_to) as translate:
            result = gfx2hlsl.translate_request(str(cache), key)
        outputs = {p.name: p.read_bytes() for p in cache.glob(f"{key:016x}.*")}
        if diagnostic is not None:
            assert result == f"unsupported: {diagnostic}", (result, sorted(outputs))
            assert set(outputs) == {f"{key:016x}.fail"}, outputs
            assert (cache / f"{key:016x}.fail").read_text() == diagnostic + "\n"
            translate.assert_not_called()
        else:
            assert result == "ok", result
            assert set(outputs) == {f"{key:016x}{s}" for s in (".vs.hlsl", ".ps.hlsl", ".txt")}, outputs
            translate.assert_called_once()
            if rtv_request:
                assert rtv_request in outputs[f"{key:016x}.txt"].decode().splitlines()
        return outputs


def main():
    # Independently executed C++ runtime fnv_bytes/cached_translation known-answer vector.
    # Revision x2w-startup-shader-29, then the default-on feature extended-buffer-formats1, both without NUL.
    # Stored order is a, b, root, then optional input layout.
    vector = (b"\x00VS\xff", b"PS\x00\x80", b"RTS0\x00")
    assert gfx2hlsl.pipeline_key(*vector) == 0xd98edd63437408d0
    assert gfx2hlsl.pipeline_key(*vector, b"ia\x00\xfe") == 0xb2045d7dbd86acdc
    blobs = (VERTEX, PIXEL, ROOT)
    canonical = gfx2hlsl.pipeline_key(*blobs)
    check_request(blobs, canonical)
    print("canonical raw request: accepted and published")
    check_request(blobs, canonical ^ 1, BASE_MISMATCH)
    print("wrong base key: exact diagnostic, no shader/manifest, no translation")
    for index, suffix in enumerate((".a", ".b", ".root")):
        changed = tuple(data + (b"\x00" if i == index else b"") for i, data in enumerate(blobs))
        check_request(changed, canonical, BASE_MISMATCH)
        print(f"changed {suffix}: exact diagnostic, no shader/manifest, no translation")
    # Identity includes the optional .ia bytes even when this VS needs no vertex fetch.
    layout = b"slot 0"
    layout_key = gfx2hlsl.pipeline_key(*blobs, layout)
    check_request(blobs, layout_key, layout=layout)
    check_request(blobs, layout_key, BASE_MISMATCH, layout=layout + b"\n")
    check_request(blobs, layout_key, BASE_MISMATCH)
    print("input layout: correct accepted; changed or missing .ia rejected before translation")
    # Legacy raw storage order is hashed as stored, then classified for translation.
    reversed_blobs = (PIXEL, VERTEX, ROOT)
    check_request(reversed_blobs, gfx2hlsl.pipeline_key(*reversed_blobs))
    from startup_stream_check import container
    stored = (container(1), container(0), ROOT)
    container_key = gfx2hlsl.pipeline_key(*stored)
    check_request(stored, container_key)
    check_request(stored, container_key ^ 1, BASE_MISMATCH)
    print("container requests: correct accepted; mismatch rejected before translation")
    for correct in (True, False):
        base = canonical if correct else canonical ^ 1
        request = f"rtv1 {base} 28"
        key = gfx2hlsl.fnv(request.encode("ascii"))
        check_request(blobs, key, None if correct else "RTV variant source key mismatch", request)
    print("RTV requests: correct accepted; existing mismatch diagnostic preserved")
    print("startup shader cache key checks: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
