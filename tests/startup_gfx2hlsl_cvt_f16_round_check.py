"""Binary16 conversion rounding: controls or dispatch <dxc> <offscreen_check> <adapter>."""
import math
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'tools'), str(HERE.parent / 'tools' / 'startup')]
import gfx2hlsl as G
from startup_compute_lds_dual_atomics_check import (PREAMBLE_LINES, SENTINEL, USER7, WORDS, mubuf_store,
    root_blob, sopp, v_mov_imm, vop1, vop2, vopc, run_shader, check_debug_layer, renderer_diagnostics)
from startup_gfx2hlsl_sdwa_cvt_check import encode_vop1_sdwa

VALUES = [1 + 2**-11, 1 + 3*2**-11, 1 + 2**-11 + 2**-23, 1 + 2**-11 - 2**-23,
          2 - 2**-11, 65519.99, 65520., 2**-25, 3*2**-25,
          -(1 + 3*2**-11), -65520., 0., -0., float('inf'), -float('inf'), 1.75]

def bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]

def half(word, rtz=False):
    sign = (word >> 16) & 0x8000
    value = struct.unpack('<f', struct.pack('<I', word & 0x7fffffff))[0]
    if math.isinf(value):
        return sign | 0x7c00
    if rtz:
        # Binary16 positive encodings are ordered; select the greatest value <= input.
        lo, hi = 0, 0x7bff
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if struct.unpack('<e', struct.pack('<H', mid))[0] <= value:
                lo = mid
            else:
                hi = mid - 1
        return sign | lo
    if value >= 65520:
        return sign | 0x7c00
    return sign | struct.unpack('<H', struct.pack('<e', value))[0]

def fixture(rtz, immediate=12, sdwa_modifier=False, dpp=False):
    inputs = [bits(v) for v in VALUES]
    code = bytearray(b''.join(raw for _, raw in PREAMBLE_LINES))
    code += vop2(0x1a, 4, 0x82, 0)
    code += v_mov_imm(1, inputs[0])
    for lane, word in enumerate(inputs[1:], 1):
        code += vopc(0xc2, 0x80 + lane, 0)
        code += v_mov_imm(5, word)
        code += vop2(1, 1, 257, 5)
    if rtz:
        code += sopp(0x24, immediate)
    expected = []
    # Plain, VOP3 -abs, and SDWA high half with preserve.
    for variant in range(3):
        if variant == 0:
            code += (struct.pack('<II', 0x7e000000 | 3 << 17 | 10 << 9 | 250,
                                 1 | 0xb1 << 8 | 0xf << 24 | 0xf << 28)
                     if dpp else vop1(10, 3, 257))
        elif variant == 1:
            code += struct.pack('<II', 0xd58a0103, 257 | (1 << 29))
        else:
            code += v_mov_imm(3, 0xabcd1234)
            code += encode_vop1_sdwa(10, 3, 1, dst_sel='WORD_1', dst_unused='UNUSED_PRESERVE', src0_sel='DWORD', neg=int(sdwa_modifier), abs_=int(sdwa_modifier))
        code += mubuf_store(0x1c, 3, 4, offset=4 * len(expected))
        for word in inputs:
            if variant == 1:
                word = (word & 0x7fffffff) | 0x80000000
            result = half(word, rtz)
            expected.append((result << 16 | 0x1234) if variant == 2 else result)
    code += sopp(1)
    hlsl = G.ComputePipeline(bytes(code), root_blob(), USER7, (len(inputs), 1, 1)).hlsl_cs()
    return hlsl, expected + [SENTINEL] * (WORDS - len(expected))

def controls():
    nearest, zero = fixture(False), fixture(True)
    assert nearest[0].count('f16rn(F(') >= 2
    assert zero[0].count('f16rz(F(') >= 2 and 'uint f16rz(float value)' in zero[0]
    assert 'uint f16rz(float value)' not in nearest[0]
    assert sum(half(bits(v)) != half(bits(v), True) for v in VALUES) >= 5
    assert half(bits(65520.)) == 0x7c00 and half(bits(65520.), True) == 0x7bff
    assert half(bits(2**-25)) == 0 and half(bits(3*2**-25)) == 2
    assert half(bits(1 + 2**-11)) == 0x3c00 and half(bits(1 + 3*2**-11)) == 0x3c02
    for rtz, helper in ((False, 'f16rn'), (True, 'f16rz')):
        dpp_shader, _ = fixture(rtz, dpp=True)
        assert helper + '(F(x2w_quad[' in dpp_shader
    modified, _ = fixture(False, sdwa_modifier=True)
    assert '0x7fffffffu' in modified and '0x80000000u' in modified
    try:
        fixture(True, immediate=4)
    except G.Unsupported as error:
        assert 'directed binary16/binary64 rounding' in str(error)
    else:
        raise AssertionError('round toward +infinity was accepted')
    print('conversion rounding controls passed')
    return nearest, zero

def dispatch(dxc, runner, adapter, fixtures):
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name, (source, expected) in zip(('rne', 'rtz'), fixtures):
            run = run_shader(name, dxc, runner, adapter, Path(tmp), source, expected)
            if run.returncode == 77:
                print(run.stdout, run.stderr)
                return 77
            check_debug_layer(name, run.stdout, run.stderr)
            assert run.returncode == 0 and not renderer_diagnostics(run.stderr, adapter), (run.stdout, run.stderr)
        source, expected = fixtures[0]
        mutant = G.F16RZ + '\n' + source.replace('f16rn(F(', 'f16rz(F(')
        assert mutant != G.F16RZ + '\n' + source
        run = run_shader('wrong_truncation', dxc, runner, adapter, Path(tmp), mutant, expected)
        if run.returncode == 77:
            print(run.stdout, run.stderr)
            return 77
        check_debug_layer('wrong_truncation', run.stdout, run.stderr)
        assert run.returncode == 1 and renderer_diagnostics(run.stderr, adapter), (run.stdout, run.stderr)
        print('wrong truncation rejected')
    print('conversion rounding readback passed')
    return 0

if __name__ == '__main__':
    fixtures = controls()
    if sys.argv[1:2] == ['dispatch']:
        sys.exit(dispatch(*sys.argv[2:], fixtures))
