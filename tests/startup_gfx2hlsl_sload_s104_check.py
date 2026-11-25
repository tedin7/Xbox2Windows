"""Check s_load into s[104:107] in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_sload_s104_check.py                              pure Python controls
       startup_gfx2hlsl_sload_s104_check.py draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                                                      draw the translated pair, compare with a CPU reference

s106 and s107 are vcc. A descriptor load (a table pointer plus an immediate offset) emits no code and only
records which table bytes the destination registers name, so it may end in s[104:107]. A load that must
write the registers (a root constant buffer) or that starts above s105 is still refused.
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import POST_WORDS, PRE_WORDS, vs_words, words_text  # noqa: E402


class Pipe:
    sampler_heap = False

    def resource(self, key, kind):
        return 'r0'


def run(symbols, op, operands):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    try:
        getattr(s, 'op_' + op)(operands.split(', '), [])
    except G.Unsupported as error:
        return str(error)
    return s


def translate(*instructions):
    stage = G.Stage.__new__(G.Stage)
    stage.pipe, stage.sym, stage.lines = Pipe(), dict(TABLE), []
    stage.kind, stage.attrs = 'ps', {}
    stage.used_s, stage.used_v, stage.masks = set(), set(), set()
    stage.exec_known = True
    stage.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    stage.translate()
    return stage


TABLE = {4: ('ptr', ('table', 0))}
ROOT = {4: ('ptr', ('root', 1))}


# One descriptor table at s[8:9] (an SRV/UAV range of two entries, as the buffer store check declares it).
_RTS0 = b''.join((struct.pack('<6I', 2, 1, 24, 0, 0, 0), struct.pack('<3I', 0, 0, 36),
                  struct.pack('<2I', 1, 44), struct.pack('<6I', 0, 2, 0, 0, 0, 0)))
_PART = b'RTS0' + struct.pack('<I', len(_RTS0)) + _RTS0
TABLE_ROOT = b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(_PART), 1, 36) + _PART

# Pixel body, words from llvm-mc gfx1013 with the first instruction's SDST patched from s[100:103] to s[104:107]
# (the assembler does not name s106/s107, the machine code does):
#   s_load_dwordx4 s[104:107], s[8:9], 0x20
#   lane 0..3:  v_cmp_eq_u32 vcc_lo, v<lane>, v<4+lane>;  v_cndmask_b32 v<12+lane>, v<8+lane>, v<lane>, vcc_lo
# The descriptor load overwrites the vcc word; the compare redefines it before the mask is read.
BODY_WORDS = bytes.fromhex('041a08f4200000fa0009847d08011802010b847d09031a02020d847d0a051c02030f847d0b071e02')
DRAW = [(5, 5, 7), (1, 2, 9), (0xffffffff, 0xffffffff, 3), (0x80000000, 0x7fffffff, 0x1234)]
DRAW_B = [(9, 9, 0), (4, 4, 4), (0, 1, 2), (0xdeadbeef, 0xdeadbeef, 1)]


def reference(sets):
    return [a if a == b else c for a, b, c in sets]


def check_draw(dxc, renderer, adapter, fxc=None):
    main()
    ps = PRE_WORDS + BODY_WORDS + POST_WORDS
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for number, sets in enumerate((DRAW, DRAW_B)):
                # Triples are (a, b, c): the load is a no-op, so the pixel is a == b ? a : c on each lane.
                pipe = G.Pipeline(vs_words(sets), ps, TABLE_ROOT)
                vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs%d_%s' % (number, kind), vs_profile, directory)
                px = compile_hlsl(compiler, kind, pipe.hlsl_ps(), 'ps%d_%s' % (number, kind), ps_profile, directory)
                expected = words_text(reference(sets))
                run = draw(renderer, adapter, vs, px, expected)
                if run.returncode:
                    raise AssertionError('%s %s\n%s\n%s' % (kind, expected, run.stdout, run.stderr))
                print('%s s104 draw %d: %s' % (kind, number, expected), flush=True)


def main():
    s = run(TABLE, 's_load_dwordx4', 's[104:107], s[4:5], 0x20')
    assert not isinstance(s, str), s
    assert [s.sym[104 + i] for i in range(4)] == [('desc', ('table', 0), 0x20, i) for i in range(4)], s.sym
    assert s.lines == [], s.lines
    s = run(TABLE, 's_load_dwordx8', 's[100:107], s[4:5], 0x20')
    assert not isinstance(s, str), s
    for bad, symbols in (('s_load_dwordx4 s[104:107], s[4:5], 0x20', ROOT),
                         ('s_load_dwordx2 s[106:107], s[4:5], 0x20', TABLE),
                         ('s_load_dwordx2 s[105:106], s[4:5], 0x20', TABLE),
                         ('s_load_dwordx4 s[105:108], s[4:5], 0x20', TABLE)):
        op, operands = bad.split(' ', 1)
        message = run(symbols, op, operands)
        assert isinstance(message, str) and 's_load destination' in message, (bad, message)
    # The loaded descriptor replaced vcc: reads are refused until vcc is written again.
    s = run(TABLE, 's_load_dwordx4', 's[104:107], s[4:5], 0x20')
    for read in (lambda: s.u('vcc_lo'), lambda: s.emit('v0 = (vcc & 1) ? 1u : 0u;')):
        try:
            read()
        except G.Unsupported as error:
            assert 'vcc read' in str(error), error
        else:
            raise AssertionError('vcc read accepted after s_load into s[104:107]')
    assert s.descriptor('s[104:107]') == (('table', 0), 0x20)
    s.emit('vcc = (v1 == 0u) & 1;')
    assert s.u('vcc_lo') == 'vcc'
    try:
        s.descriptor('s[104:107]')
    except G.Unsupported as error:
        assert 'overlaps a redefined vcc' in str(error), error
    else:
        raise AssertionError('descriptor still accepted after its VCC word was clobbered')
    # The branch jumps over the compare, so its target can read the descriptor word as VCC.
    load = 's_load_dwordx4 s[104:107], s[4:5], 0x20'
    compare = 'v_cmp_eq_u32_e32 0, v1'
    read = 'v_cndmask_b32_e32 v0, 0, 1'
    assert translate(load, compare, read, 's_endpgm').u('vcc_lo') == 'vcc'
    try:
        translate(load, 's_cbranch_scc0 label_000C', compare, read, 's_endpgm')
    except G.Unsupported as error:
        assert 'vcc read after a descriptor load into s106/s107' in str(error), error
    else:
        raise AssertionError('branch skipped the only VCC redefinition')
    translate(load, compare, 's_cbranch_scc0 label_0010', 's_nop 0', read, 's_endpgm')
    # The load may be bypassed on one path; the resource would then have no matching descriptor.
    fetch = 'buffer_load_dword v0, off, s[104:107], 0'
    translate(load, 's_cbranch_scc0 label_000C', 's_nop 0', fetch, 's_endpgm')
    try:
        translate('s_cbranch_scc0 label_000C', load, 's_nop 0', fetch, 's_endpgm')
    except G.Unsupported as error:
        assert ('without a matching load on every path' in str(error)
                or 'joined at a branch target' in str(error)), error
    else:
        raise AssertionError('branch skipped the only descriptor load')
    s = run(TABLE, 's_load_dwordx4', 's[100:103], s[4:5], 0x20')
    assert s.u('vcc_lo') == 'vcc'
    # A second translation pass starts with vcc intact: the load at the end of the first pass must not leak into it.
    again = translate(read, load, 's_endpgm')
    again.translate()
    assert again.lines == translate(read, load, 's_endpgm').lines
    print('ok')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
