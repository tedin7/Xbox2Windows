"""Check descriptor pointer propagation through s_add_u32 / s_addc_u32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_desc_ptr_prop_check.py [draw <dxc> <offscreen_check> <adapter>]

A shader advances a descriptor table or root pointer with `s_add_u32 sD, sP, <literal>` and
`s_addc_u32 sD+1, sP+1, 0`, into the same or another register pair, then loads descriptors through
the result. The load must read at the advanced byte offset. A non-literal addend must be refused,
never silently treated as the unchanged pointer.
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    sampler_heap = False

    def __init__(self):
        self.keys = []
        self.modified = set()

    def ud(self, key):
        return 'ud_%s' % '_'.join(map(str, key))

    def resource(self, key, kind):
        if (key, kind) not in self.keys:
            self.keys.append((key, kind))
        return 'r%d' % self.keys.index((key, kind))


def stage(symbols):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    s.masks, s.varying_s = set(), set()
    s.scc_uniform = True
    return s


def run(symbols, program, branches=False):
    """Symbols after the (op, operands) steps, or the Unsupported message."""
    s = stage(symbols)
    s.has_branches = branches
    s.ins = [(4 * i, 4, op, '') for i, (op, _operands) in enumerate(program)]
    try:
        for i, (op, operands) in enumerate(program):
            s.pc = 4 * i
            if op == 's_load_dwordx4':
                s.op_s_load_dwordx4(operands.split(', '), [])
            else:
                s.lower(op, '%s %s' % (op, operands), operands.split(', '), [])
    except G.Unsupported as error:
        return 'refused: %s' % error
    return s


TABLE = {8: ('ptr', ('table', 0))}   # a table pointer is 32 bits wide: no high word
ROOT = {8: ('ptr', ('root', 1)), 9: ('ptrhi', ('root', 1))}
ADVANCE = [('s_add_u32', 's20, s8, 0x40'), ('s_addc_u32', 's21, s9, 0')]
TABLE_ADVANCE = [('s_add_u32', 's20, s8, 0x40')]
LOAD = [('s_load_dwordx4', 's[4:7], s[20:21], 0x10')]

s = run(TABLE, TABLE_ADVANCE + LOAD)
check('table pointer advanced into another pair, descriptor read at pointer + 0x40 + 0x10',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s if isinstance(s, str) else s.sym))

same = [('s_add_u32', 's8, s8, 0x20')]
s = run(TABLE, same + [('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('table pointer advanced in place',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x20, 0), str(s if isinstance(s, str) else s.sym))

twice = [('s_add_u32', 's20, s8, 0x40'), ('s_add_u32', 's22, s20, 0x10')]
s = run(TABLE, twice + [('s_load_dwordx4', 's[4:7], s[22:23], 0x4')])
check('two advances add up',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x54, 0), str(s if isinstance(s, str) else s.sym))

s = run(ROOT, ADVANCE + LOAD)
check('root constant-buffer pointer advanced, buffer read at the advanced offset',
      not isinstance(s, str) and any('Load(80)' in line for line in s.lines), str(s if isinstance(s, str) else s.lines))

s = run(TABLE, [('s_add_u32', 's8, s8, s2')])
check('a register addend on a pointer is refused', isinstance(s, str) and 'non-literal' in s, str(s))

s = run(ROOT, ADVANCE + [('s_load_dwordx4', 's[4:7], s[20:21], s2')])
check('a register offset through an advanced pointer is still refused', isinstance(s, str), str(s))

# The untracked-offset mark of a branchy program must be refused here too, never a TypeError.
s = stage({8: ('ptr', ('root', 1), 'added'), 9: ('ptrhi', ('root', 1))})
try:
    s.descriptor('s[8:9]')
    check('an untracked-offset pointer used whole as a descriptor is refused', False, 'accepted')
except G.Unsupported:
    check('an untracked-offset pointer used whole as a descriptor is refused', True)

s = run(ROOT, ADVANCE)
try:
    s.descriptor('s[20:21]')
    check('an advanced pointer used whole as a descriptor is refused', False, 'accepted')
except G.Unsupported:
    check('an advanced pointer used whole as a descriptor is refused', True)

s = run(ROOT, ADVANCE, branches=True)
check('a pointer advance in a program with branches is refused', isinstance(s, str), str(s))

# The high-word add is the carry of the low-word add right before it, never of an unrelated carry in SCC.
s = run(ROOT, [('s_cmp_eq_u32', '0, 0'), ('s_addc_u32', 's9, s9, 0')])
check('s_addc_u32 without the low-word add is refused', isinstance(s, str), str(s))

s = run(ROOT, [ADVANCE[0], ('s_cmp_eq_u32', '0, 0'), ADVANCE[1]])
check('s_addc_u32 separated from its low-word add is refused', isinstance(s, str), str(s))

# A fold needs the complete matching pair: the load must not trust the low word alone.
LOW = ADVANCE[0]
s = run(ROOT, [LOW] + LOAD)
check('missing high-word add is refused', isinstance(s, str), str(s))

s = run(ROOT, [('s_add_u32', 's8, s8, 0x20'), ('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('missing high-word add in place is refused', isinstance(s, str), str(s))

s = run(ROOT, ADVANCE + [('s_mov_b32', 's21, 0')] + LOAD)
check('overwritten high word is refused', isinstance(s, str), str(s))

mixed = dict(ROOT)
mixed[10], mixed[11] = ('ptr', ('root', 2)), ('ptrhi', ('root', 2))
s = run(mixed, [LOW, ('s_addc_u32', 's21, s11, 0')] + LOAD)
check('high word from another pointer is refused', isinstance(s, str), str(s))

s = run(mixed, [LOW, ('s_addc_u32', 's21, s10, 0')] + LOAD)
check('high word from a different register pair is refused', isinstance(s, str), str(s))

s = run(ROOT, [('s_add_u32', 's20, s8, 0xfffff'), ('s_addc_u32', 's21, s9, 0')] + LOAD)
check('complete pair with a large low-word add is folded',
      not isinstance(s, str) and any('Load(%d)' % (0xfffff + 0x10) in line for line in s.lines), str(s))

# The high word is bound to the accumulated offset: the original word never stands in for an advanced one.
s = run(ROOT, ADVANCE + [('s_mov_b32', 's21, s9')] + LOAD)
check('completed high word overwritten by the original one is refused', isinstance(s, str), str(s))

SECOND = [('s_add_u32', 's22, s20, 0x20')]
s = run(ROOT, ADVANCE + SECOND + [('s_addc_u32', 's23, s9, 0'), ('s_load_dwordx4', 's[4:7], s[22:23], 0x10')])
check('second advance with the original high word is refused', isinstance(s, str), str(s))

s = run(ROOT, ADVANCE + SECOND + [('s_addc_u32', 's23, s21, 0'), ('s_load_dwordx4', 's[4:7], s[22:23], 0x10')])
check('second advance with the completed high word of the first pair is folded',
      not isinstance(s, str) and any('Load(%d)' % (0x40 + 0x20 + 0x10) in line for line in s.lines), str(s))

s = run(ROOT, [('s_add_u32', 's8, s8, 0x20'), ('s_addc_u32', 's9, s9, 0'), ('s_add_u32', 's8, s8, 0x20'),
               ('s_addc_u32', 's9, s9, 0'), ('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('in-place advance twice with each completed high word is folded',
      not isinstance(s, str) and any('Load(64)' in line for line in s.lines), str(s))

# Real entry state: a table pointer is one 32-bit register with no high word, so it advances with
# the low add alone and its neighbour register stays untouched.
def entry(params, kind='vs'):
    s = stage({})
    s.kind, s.pipe.root = kind, {'params': params, 'flags': 0}
    s.setup_entry()
    return s.sym


REAL = entry([{'type': 0}, {'type': 2}])
s = run(REAL, [('s_add_u32', 's20, s8, 0x40')] + LOAD)
check('real table pointer advanced by the low add alone, descriptor read at +0x50',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s))

s = run(REAL, [('s_add_u32', 's8, s8, 0x40'), ('s_load_dwordx4', 's[4:7], s[8:9], 0x10')])
check('real table pointer advanced in place keeps the next register symbol',
      not isinstance(s, str) and s.sym.get(9) == REAL[9] and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s))

s = run(REAL, [('s_add_u32', 's20, s9, 0x40'), ('s_addc_u32', 's21, s10, 0')] + LOAD)
check('real root pointer pair advanced through the load',
      not isinstance(s, str) and any('Load(80)' in line for line in s.lines), str(s))

s = run(REAL, [('s_add_u32', 's20, s9, 0x40')] + LOAD)
check('real root pointer without its high add is refused', isinstance(s, str), str(s))

copied = dict(ROOT)
copied[30] = ('ptrhi', ('root', 1))
s = run(copied, [LOW, ('s_addc_u32', 's21, s30, 0')] + LOAD)
check('a copied high word completes the pair', not isinstance(s, str), str(s))

viam0 = dict(ROOT)
viam0['m0'] = ('ptr', ('root', 1))
s = run(viam0, [('s_add_u32', 's20, m0, 0x40'), ('s_addc_u32', 's21, s9, 0')] + LOAD)
check('a low word held in m0 completes the pair', not isinstance(s, str), str(s))

# A program with branches does not track the added offset: the load through the advanced pointer is a refusal
# (never a crash), and the same straight-line shape keeps folding.
IN_PLACE = [('s_add_u32', 's8, s8, 0x40'), ('s_load_dwordx4', 's[4:7], s[8:9], 0x10')]
s = run(TABLE, IN_PLACE, branches=True)
check('in-place pointer advance then load in a program with branches is refused, not a crash',
      isinstance(s, str) and 'untracked offset' in s, str(s))

s = run(TABLE, IN_PLACE)
check('the same in-place advance without branches still folds to +0x50',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s if isinstance(s, str) else s.sym))

# The pair is complete only after its s_addc_u32: an in-place pair folds, and a high word that was only
# copied (never advanced) must not stand in for it.
s = run(ROOT, [('s_add_u32', 's8, s8, 0x20'), ('s_addc_u32', 's9, s9, 0'),
               ('s_load_dwordx4', 's[4:7], s[8:9], 0x10')])
check('complete in-place pair folds to +0x30',
      not isinstance(s, str) and any('Load(%d)' % 0x30 in line for line in s.lines), str(s if isinstance(s, str) else s.lines))

s = run(ROOT, [ADVANCE[0], ('s_mov_b32', 's21, s9')] + LOAD)
check('out-of-place low add with a copied, unadvanced high word is refused', isinstance(s, str), str(s))

s = run(ROOT, [ADVANCE[0], ADVANCE[1]] + LOAD)
check('complete out-of-place pair leaves the original high source untouched',
      not isinstance(s, str) and s.sym.get(9) == ('ptrhi', ('root', 1)), str(s if isinstance(s, str) else s.sym))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
def descriptor_number_pipeline(word=2):
    from startup_gfx2hlsl_sload_s104_check import TABLE_ROOT, vs_words
    from startup_gfx2hlsl_scalar_null_check import POST_WORDS, smem, sop1, sop2, vop1

    # Mask NUM_RECORDS to six bits and scale it by five. Keep the result
    # across an overwrite of the source word before converting it to float.
    body = smem(2, 72, 8, imm=0x20) + struct.pack('<I', 0xbf8c0000)
    body += sop2(0x0e, 46, 72 + word, 191)
    body += sop2(0x26, 47, 46, 133)
    body += sop1(3, 72 + word, 128)
    body += vop1(6, 12, 47)
    body += b''.join(vop1(1, reg, 128) for reg in range(13, 16))
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), body + POST_WORDS, TABLE_ROOT)


def check_descriptor_number():
    pipe = descriptor_number_pipeline()
    assert pipe.manifest().startswith('x2w-manifest 3\n')
    assert ('descword', 0, 0x28) in pipe.uds
    assert not pipe.resources
    # A heap record has no graphics entry-word representation.
    s = stage({72: ('desc', ('resheap', 0, 0), 0, 2)})
    s.kind = 'ps'
    s.scalar('s_and_b32', ['s46', 's72', '63'])
    try:
        s.S(46)
    except G.Unsupported as error:
        assert 'cannot materialize descriptor word 2' in str(error), error
    else:
        raise AssertionError('an unavailable descriptor word translated')


def draw_descriptor_number(dxc, renderer, adapter):
    from startup_flat_param_check import compile_hlsl
    from startup_gfx2hlsl_bit_alu_check import words_text
    import gpu_case_runner

    correct, wrong = (descriptor_number_pipeline(word) for word in (2, 1))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', correct.hlsl_vs(), 'vs_descriptor_number', 'vs_6_0', directory)
        for label, pipe in (('record_count', correct), ('wrong_word', wrong)):
            ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), label, 'ps_6_0', directory)
            for descriptor in ((0, 0x30112, 91, 0), (0, 0x40023, 109, 0)):
                values = [descriptor[(source[2] - 0x20) // 4] if source[0] == 'descword' else 0
                          for source in pipe.uds]
                values += [0] * (4 - len(values))
                expected = words_text([G.fbits(float((descriptor[2] & 63) * 5)), 0, 0, 0])
                result = gpu_case_runner.run(renderer, adapter,
                                             ['--startup-flat-param', vs, ps, expected, words_text(values)])
                if result.returncode == 77:
                    raise SystemExit(77)
                if pipe is correct:
                    assert result.returncode == 0, (result.stdout, result.stderr)
                else:
                    actual = words_text([G.fbits(float((descriptor[1] & 63) * 5)), 0, 0, 0])
                    assert result.returncode == 1 and 'actual=' + actual in result.stderr, result.stderr


def descriptor_limits_pipeline(size_word=2, invert=False):
    from startup_gfx2hlsl_sload_s104_check import TABLE_ROOT, vs_words
    from startup_gfx2hlsl_scalar_null_check import POST_WORDS, smem, sop2, sopc, sopp, vop2
    from startup_image_sample_a16_check import DIM_2D, encode_mimg

    body = smem(3, 64, 8, imm=0x40) + sopp(0x0c)
    # Two four-bit size-minus-one fields. Combine each with the low bit of a
    # third word before applying the mip shift from another descriptor field.
    body += sop2(0x0e, 46, 64 + size_word, 143)
    body += sop2(0x20, 47, 64 + size_word, 132)
    body += sop2(0x0e, 47, 47, 143)
    body += sop2(0x0e, 48, 68, 129)
    body += sop2(0x20, 49, 68, 132)
    for limit in (46, 47):
        body += sop2(0x00, limit, limit, 129)
        body += sop2(0x2e, limit, limit, 48)
        body += sop2(0x20, limit, limit, 49)
        body += sop2(0x09, limit, limit, 129)
        body += sop2(0x01, limit, limit, 129)
    body += sopc(0x06, 64, 128)
    selects = b''
    for dst, limit in ((50, 46), (51, 47)):
        yes, no = (limit, 193) if invert else (193, limit)
        selects += sop2(0x0a, dst, yes, no)
    # Both edges reach the vector code with assigned runtime scalars.
    body += selects + sopp(0x05, len(selects) // 4) + selects
    body += vop2(0x1b, 8, 255, 0) + struct.pack('<I', 0xffff)
    body += vop2(0x16, 9, 144, 0)
    for coord, limit in ((8, 50), (9, 51)):
        body += vop2(0x1a, 10, 129, coord)
        body += vop2(0x25, coord, 266, coord)
        body += vop2(0x13, coord, limit, coord)
    body += encode_mimg(0, dim=DIM_2D, vaddr=8, vdata=12, srsrc=64) + sopp(0x0c)
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), body + POST_WORDS, TABLE_ROOT,
                      ps_hw_inputs=(0x8000, 0x8000))


def check_descriptor_limits():
    pipe = descriptor_limits_pipeline()
    expected = {('descword', 0, addr) for addr in (0x40, 0x48, 0x50)}
    assert {source for source in pipe.uds if source[0] == 'descword'} == expected, pipe.uds
    assert pipe.manifest().startswith('x2w-manifest 3\n')
    assert 'r0.Load(' in pipe.hlsl_ps()


def draw_descriptor_limits(dxc, renderer, adapter):
    from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl
    from startup_gfx2hlsl_bit_alu_check import words_text
    import gpu_case_runner

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs_limits', 'vs_6_0', directory)
        correct = descriptor_limits_pipeline()
        wrong_word = descriptor_limits_pipeline(size_word=3)
        inverted = descriptor_limits_pipeline(invert=True)
        descriptor = (1, 0, 0x13, 0x11, 0x10, 0, 0, 0)
        for label, pipe, bound, mismatch in (
            ('bound', correct, True, None),
            ('unbound', correct, False, None),
            ('wrong_word', wrong_word, True, '(1,0)'),
            ('inverted', inverted, True, '(2,0)'),
        ):
            ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), label, 'ps_6_0', directory)
            values = [descriptor[(source[2] - 0x40) // 4] if bound and source[0] == 'descword' else 0
                      for source in pipe.uds]
            expected = []
            for y in range(8):
                for x in range(8):
                    texel = 1 + min(3 * x, 3) + 4 * min(3 * y, 1) if bound else 0
                    expected.extend((G.fbits(float(texel)), 0, 0, 0))
            path = directory / (label + '.bin')
            path.write_bytes(struct.pack('<256I', *expected))
            result = gpu_case_runner.run(renderer, adapter, [
                '--startup-descriptor-limits', vs, ps, path, words_text(values), str(int(bound)),
            ])
            if result.returncode == 77:
                raise SystemExit(77)
            if mismatch is None:
                assert result.returncode == 0, (result.stdout, result.stderr)
            else:
                assert result.returncode == 1 and f'pixel {mismatch} mismatch' in result.stderr, result.stderr


check_descriptor_number()
check_descriptor_limits()
if len(sys.argv) > 1:
    if len(sys.argv) != 5 or sys.argv[1] != 'draw':
        raise SystemExit(__doc__)
    draw_descriptor_number(*sys.argv[2:])
    draw_descriptor_limits(*sys.argv[2:])
print('startup_gfx2hlsl_desc_ptr_prop: ok')
