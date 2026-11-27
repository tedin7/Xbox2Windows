"""GETPC and vector carry provenance controls using synthetic RDNA2 words."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
import gfx2hlsl as G
from startup_gfx2hlsl_add_sub_co_ci_check import vop3b


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def program(*, extra=b'', delta=None, high=0x82118011, duplicate=False):
    low_add = words(*vop3b(0x30f, 20, 106, 276, 16, 0))
    high_add = words(*vop3b(0x128, 21, 106, 17, 277, 106))
    before = words(0xbe901f00)
    before += words(0x8010ff10, 0) + words(high) + low_add + high_add
    if duplicate:
        before += words(0xbe921f00)
    code = before + extra + words(0xbf810000)
    displacement = len(code) - 4 if delta is None else delta
    code = code[:8] + words(displacement) + code[12:]
    return code + words(0x12345678, 0xabcdef01), G.decode(code)


def reject(label, **kwargs):
    code, ins = program(**kwargs)
    try:
        G.vertex_image_getpc_proof(code, ins)
    except G.Unsupported as error:
        assert label in str(error), (label, error)
    else:
        raise AssertionError('expected refusal: ' + label)


def main():
    code, ins = program()
    start, data, setup, pairs = G.vertex_image_getpc_proof(code, ins)
    assert (start, data, setup, pairs) == (36, (0x12345678, 0xabcdef01), (0, 4, 12), {16: 24})
    reject('expected exactly one GETPC', duplicate=True)
    reject('GETPC requires literal low add and zero high carry', high=0x82118111)
    reject('table start is outside the data tail', delta=0)
    reject('GETPC pair escapes', extra=words(0x7e000210))  # v_mov_b32 v0, s16
    reject('GETPC carry SCC escapes', extra=words(0xbf840000))  # s_cbranch_scc0
    # A low carry without the matching high half cannot be elided.
    code, ins = program()
    bad = bytearray(code)
    bad[24:32] = words(0xbf800000, 0xbf800000)
    try:
        G.vertex_image_getpc_proof(bytes(bad), G.decode(bytes(bad[:36])))
    except G.Unsupported as error:
        assert 'GETPC vector carry is incomplete' in str(error), error
    else:
        raise AssertionError('missing carry accepted')
    print('vertex image GETPC controls: passed')


if __name__ == '__main__':
    main()
