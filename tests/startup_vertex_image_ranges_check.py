"""CPU controls for conservative program-image index range facts."""
from fractions import Fraction
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
from vertex_image_ranges import VertexImageRanges


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
    print('vertex image range controls: passed')


if __name__ == '__main__':
    controls()
