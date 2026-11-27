"""Controls for GETPC-relative per-lane GLOBAL address pattern recognition."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
import gfx2hlsl as G
from startup_buffer_address_check import encode_mtbuf, encode_vop1, encode_vop2
from startup_compute_global_load_check import encode_global, GLOBAL_LOAD_DWORD
from startup_gfx2hlsl_add_sub_co_ci_check import vop3b


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def program(*, nwords=763, unbounded=False, shift=2, high_source=19, after_shift=b''):
    mtbuf = encode_mtbuf(vdata=9)
    fma = words(0xd54b000f, 255 | (268 << 9) | (240 << 18), 0x437f0000)
    cvt = encode_vop1(7, 15, 270 if unbounded else 271)
    cmpx = words(0x7c000000 | (0xd5 << 17) | (15 << 9) | 255, 255)
    triple = words(0xd7460012, 271 | (129 << 9) | (271 << 18))
    ashr = encode_vop2(0x18, 19, 159, 18)
    shift64 = words(0xd6ff0012, (128 + shift) | (274 << 9))
    prefix = mtbuf + fma + cvt + cmpx + triple + ashr + shift64 + after_shift
    pc = len(prefix)
    low = words(*vop3b(0x30f, 20, 106, 274, 16, 0))
    high = words(*vop3b(0x128, 21, 106, 17, 256 + high_source, 106))
    load = encode_global(GLOBAL_LOAD_DWORD, 7, 20, 0x7d)
    rest = words(0xbe901f00, 0x8010ff10, 0, 0x82118011) + low + high + load + words(0xbf810000)
    code = prefix + rest
    code = code[:pc + 8] + words(len(code) - pc - 4) + code[pc + 12:]
    image = code + b''.join(words(0x10000000 + i) for i in range(nwords))
    return image, G.decode(code), len(code) - len(load) - 4


def check_refusal(text, **kwargs):
    code, ins, _ = program(**kwargs)
    try:
        plan = G.vertex_image_getpc_proof(code, ins)
        G.vertex_image_address_proof(ins, plan)
    except G.Unsupported as error:
        assert text in str(error), (text, error)
    else:
        raise AssertionError('expected refusal: ' + text)


def main():
    code, ins, load_pc = program()
    proof = G.vertex_image_getpc_proof(code, ins)
    loads = G.vertex_image_address_proof(ins, proof)
    assert loads == {load_pc: ('v7', 44, 762)}, loads
    check_refusal('program image load index not proven in range', nwords=762)
    check_refusal('program image load index not proven in range', unbounded=True)
    check_refusal('program image address is not a proven index shifted by two', shift=1)
    check_refusal('program image address high word does not match the shifted pair', high_source=18)
    check_refusal('program image address is not a proven index shifted by two',
                  after_shift=encode_vop1(1, 18, 128))
    check_refusal('program image address is not a proven index shifted by two',
                  after_shift=words(0xbeff037e))
    print('vertex image address controls: passed')


if __name__ == '__main__':
    main()
