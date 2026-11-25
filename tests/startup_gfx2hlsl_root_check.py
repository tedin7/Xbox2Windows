"""A malformed root signature blob is refused per request by tools/startup/gfx2hlsl.py --pending.

Each bad request gets a .fail file with a reason; the run exits 0 and still translates a valid
request that sorts after a bad one.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/startup"))
import gfx2hlsl


def code(target):
    # One export followed by s_endpgm.
    return struct.pack("<3I", 0xf800080f | target << 4, 0, 0xbf810000)


def container(parts):
    """DXBC header (hash zeroed) followed by (fourcc, body) parts."""
    base = 32 + 4 * len(parts)
    offsets, body = [], b""
    for fourcc, data in parts:
        offsets.append(base + len(body))
        body += fourcc + struct.pack("<I", len(data)) + data
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, base + len(body), len(parts))
            + struct.pack("<%dI" % len(offsets), *offsets) + body)


TRUNCATED = "root signature blob truncated"
VALID_ROOT = container([(b"RTS0", struct.pack("<6I", 1, 0, 24, 0, 24, 0))])
# name -> (root blob, expected .fail text)
BAD = {
    "empty": (b"", "root signature blob missing"),
    "truncated header": (b"DXBC" + bytes(8), TRUNCATED),
    "part count past end": (b"DXBC" + bytes(16) + struct.pack("<III", 1, 36, 1000), TRUNCATED),
    "RTS0 without body": (container([(b"RTS0", b"")]), TRUNCATED),
    "parameter table past end": (container([(b"RTS0", struct.pack("<6I", 1, 4, 24, 0, 24, 0))]), TRUNCATED),
    "descriptor table past end": (container([(b"RTS0", struct.pack("<6I", 1, 1, 24, 0, 36, 0)
                                                        + struct.pack("<3I", 1, 0, 400))]), TRUNCATED),
    "no RTS0": (container([(b"ABCD", bytes(8))]), "no RTS0 part"),
}
# One type-1 parameter (vis 0) whose table, at byte 36 of the part, is (register 5, space 2, count 3).
TABLE_ROOT = container([(b"RTS0", struct.pack("<6I", 1, 1, 24, 0, 36, 0)
                         + struct.pack("<3I", 1, 0, 36) + struct.pack("<3I", 5, 2, 3))])


def main():
    table = gfx2hlsl.parse_root(TABLE_ROOT)["params"]
    assert table == [dict(type=1, vis=0, reg=5, space=2, count=3)], table
    vs, ps = code(12), code(0)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        requests = {name: root for name, (root, _) in BAD.items()}
        requests["valid"] = VALID_ROOT
        keys = {}
        for name, root in requests.items():
            keys[name] = gfx2hlsl.pipeline_key(vs, ps, root)
            for ext, data in ((".a", vs), (".b", ps), (".root", root)):
                (directory / ("req_%016x%s" % (keys[name], ext))).write_bytes(data)
        # --pending visits requests in key order; the valid one must come after a bad one.
        assert keys["valid"] > min(keys[name] for name in BAD), "valid request does not sort after a bad one"
        # The empty X2W_FXC publishes the manifest without compiling, so the valid request is
        # translated the same way on a host without fxc.exe.
        result = subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", str(directory)],
                                capture_output=True, text=True, env=dict(os.environ, X2W_FXC=""))
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        for name, (_, reason) in BAD.items():
            fail = directory / ("%016x.fail" % keys[name])
            assert fail.is_file(), "%s: no .fail file\n%s" % (name, output)
            assert reason in fail.read_text(), (name, fail.read_text())
            assert not (directory / ("%016x.txt" % keys[name])).exists(), name
        assert (directory / ("%016x.txt" % keys["valid"])).is_file(), \
            "valid request after bad ones was not translated\n" + output
        assert not (directory / ("%016x.fail" % keys["valid"])).exists()
    print("malformed root blobs refused per request, valid request still translated")


if __name__ == "__main__":
    main()
