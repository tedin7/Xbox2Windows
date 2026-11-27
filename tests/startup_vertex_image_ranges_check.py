"""CPU controls for conservative program-image index range facts."""
from fractions import Fraction
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
from vertex_image_ranges import VertexImageRanges
import gfx2hlsl as G
from startup_buffer_address_check import encode_mtbuf, encode_vop1


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def controls():
    r = VertexImageRanges()
    r.unorm('v0')
    r.fma('v1', 'v0', 255.0, 0.5)
    r.cvt_u32('v2', 'v1')
    assert r.facts['v2'] == ('uint', 0, 255)
    for byte in range(256):
        value = int(f32(255.0 * f32(byte / 255.0) + 0.5))
        assert value == byte
        assert r.facts['v2'][1] <= value <= r.facts['v2'][2]
    r.cmpx_ne('v2', 255)
    r.integer('one', 1, 1)
    r.add_scaled('v3', 'v2', 2, 'v2')
    r.add_scaled('v4', 'v3', 1, 'one')
    assert r.facts['v3'] == ('uint', 0, 762)
    assert r.facts['v4'] == ('uint', 1, 763)
    assert r.in_table('v3', 763) and not r.in_table('v3', 762)
    table = [0x40000000 + i * 73 for i in range(765)]
    assert [table[3 * i] for i in range(255)] != [table[3 * i + 1] for i in range(255)]
    r.exec_changed()
    assert not r.in_table('v3', 1000)
    r.integer('wide', 0, 0xffffffff)
    r.add_scaled('overflow', 'wide', 2, 'one')
    assert 'overflow' not in r.facts
    r.unorm('x')
    r.fma('x', 'x', 1.0, -0.5)
    assert 'x' not in r.facts
    r.unorm('x')
    r.fma('x', 'x', 0.1, 0.1)
    assert r.facts['x'][2] >= Fraction.from_float(0.1) * 2
    # Feed hand-encoded instructions through the real RDNA2 decoder.
    mtbuf = encode_mtbuf(vdata=9)
    fma = struct.pack('<III', 0xd54b000f, 255 | (268 << 9) | (240 << 18), 0x437f0000)
    cvt = encode_vop1(7, 15, 271)
    cmpx = struct.pack('<II', 0x7c000000 | (0xd5 << 17) | (15 << 9) | 255, 255)
    triple = struct.pack('<II', 0xd7460012, 271 | (129 << 9) | (271 << 18))
    end = struct.pack('<I', 0xbf810000)
    code = mtbuf + fma + cvt + cmpx + triple + end
    decoded = G.decode(code)
    bounds = G.vertex_image_range_facts(decoded)
    assert bounds[40]['v18'] == ('uint', 0, 762), bounds
    # A restore may activate lanes excluded by CMPX, so all facts are lost.
    restored = G.decode(mtbuf + fma + cvt + cmpx + struct.pack('<I', 0xbeff037e) + triple + end)
    assert 'v18' not in G.vertex_image_range_facts(restored)[44]
    # A forward branch into the indexed block invalidates the incoming facts.
    join = struct.pack('<I', 0xbf880000)  # cbranch_execz to the following instruction
    joined = G.decode(mtbuf + fma + cvt + cmpx + join + triple + end)
    assert 'v18' not in G.vertex_image_range_facts(joined)[44]
    # An unrelated, unmodelled write kills the old integer range.
    overwrite = encode_vop1(1, 15, 256)
    killed = G.decode(mtbuf + fma + cvt + cmpx + overwrite + triple + end)
    assert 'v18' not in G.vertex_image_range_facts(killed)[44]
    # Signed-normalized or unknown inputs cannot prove an unsigned index.
    signed = G.decode(encode_mtbuf(fmt=57, vdata=9) + fma + cvt + cmpx + triple + end)
    assert 'v18' not in G.vertex_image_range_facts(signed)[40]
    unbounded = G.decode(mtbuf + fma + encode_vop1(7, 15, 270) + cmpx + triple + end)
    assert 'v18' not in G.vertex_image_range_facts(unbounded)[40]
    print('vertex image range controls: passed')


if __name__ == '__main__':
    controls()
