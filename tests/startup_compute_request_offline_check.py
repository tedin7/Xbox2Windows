"""Offline side of device slot 67 compute requests, pure Python (no Windows or D3D12 needed).

- compute_key of fixed synthetic parts equals a key computed here from the framing rule (revision + "cs", then
  each part preceded by its u64 length, in the order code, data, root, desc, state A, state B) and a literal.
- A cache directory with one synthetic request (req_<key>.{desc,sa,sb,data,root,c67}) is found by
  request_keys(compute=True) and pending_keys, but not by the graphics request_keys; gfx2hlsl.py --pending
  records the compute stage's refusal of its 24-byte state block A in <key>.fail and writes no manifest; --refusals --json reports it in
  the compute section only.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

REASON = 'compute state block A is 24 bytes, not 28'  # the compute stage refuses the synthetic block
CODE = struct.pack('<2I', 0x7e020280, 0xbf810000)
DATA = struct.pack('<2Q', 0xfedcba9876543210, 0xffff00001111008c)
ROOT = b'RTS0 synthetic root'
DESC = struct.pack('<12Q', 0, 0, 24, 0, 20, 0, len(CODE), 0, len(DATA), 0, 0, 11)
STATE_A = struct.pack('<3Q', 1, 0x2c, 0xfff0000000000003)
STATE_B = struct.pack('<5I', 0x1234abcd, 0xffff0000, 0, 0, 0)
KEY = 0x96c99985746a35d1  # framed key at revision 29cs


def framed_key():
    h = 1469598103934665603
    for b in b'x2w-startup-shader-29cs':
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    for part in (CODE, DATA, ROOT, DESC, STATE_A, STATE_B):
        for b in struct.pack('<Q', len(part)) + part:
            h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def main():
    key = G.compute_key(CODE, DATA, ROOT, DESC, STATE_A, STATE_B)
    assert key == framed_key(), '%016x != %016x' % (key, framed_key())
    assert key == KEY, '%016x != literal %016x' % (key, KEY)
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        for ext, part in (('.desc', DESC), ('.sa', STATE_A), ('.sb', STATE_B), ('.data', DATA), ('.root', ROOT),
                          ('.c67', CODE)):
            (cache / ('req_%016x%s' % (key, ext))).write_bytes(part)
        assert list(G.request_keys(str(cache), compute=True)) == [key]
        assert list(G.request_keys(str(cache))) == []
        assert G.pending_keys(str(cache), skip_failed=False) == [key]
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', str(cache)], check=True,
                             capture_output=True, text=True, env=dict(os.environ, X2W_TRANSLATE_JOBS='1')).stdout
        assert out == '%016x: unsupported: %s\n' % (key, REASON), out
        assert (cache / ('%016x.fail' % key)).read_text() == REASON + '\n'
        assert not (cache / ('%016x.txt' % key)).exists()
        report = json.loads(subprocess.run([sys.executable, '-B', G.__file__, '--refusals', str(cache), '--json'],
                                           check=True, capture_output=True, text=True).stdout)
        assert report['per_request']['examined'] == 0, report['per_request']
        assert report['compute'] == dict(examined=1, no_refusal=0, reasons=[
            dict(reason=REASON, containing=1, first=1)]), report['compute']
    print('compute request offline: key vector, request_keys/pending_keys, --pending record and --refusals '
          'compute section pass')


if __name__ == '__main__':
    main()
