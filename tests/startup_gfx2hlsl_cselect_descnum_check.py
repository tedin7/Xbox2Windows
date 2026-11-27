"""s_cselect_b32 on descriptor-derived data in the compute stage of tools/startup/gfx2hlsl.py (#2618, C500 rows).

usage: startup_gfx2hlsl_cselect_descnum_check.py [controls]
       startup_gfx2hlsl_cselect_descnum_check.py dispatch <dxc> <offscreen_check> <warp|hardware>

The C500 compute rows stop at `s_cselect_b32 sX, 0, sX` where sX holds a number computed from a buffer descriptor
(descnum). Such a select can zero a buffer's size, which disables the buffer, so a translation that drops the zero
arm would read real data where the console reads zeros. The select now records both arms and its own offset (the
SCC it read) as an opaque symbol, and every later use that would need its value refuses.

The fixture is case A of startup_gfx2hlsl_compute_check.py (a descriptor table load s[12:19], a buffer load
through s[12:15], a buffer store through s[16:19]) with synthetic instructions spliced in. Their bytes come from
the RDNA 2 SOP1/SOP2/SOPC/VOP1/VOPC layouts, and the decoder must print the expected text for each.

Checked (controls):
- accepted: the select with its result dead, in both arm orders, with a literal arm, and on an extracted stride;
  the HLSL differs from case A only by the compare's scc line, and sX holds both arms and the select's offset;
- refused, each with its reason: numeric reads (v_mov, s_cmp, v_cmp, m0), scalar ALU on the result, the
  structured-to-raw conversion with the selected stride, a copy into the load descriptor's word 1, 2 or 3 (also
  through a second copy), a copy into the store descriptor, SCC from an ALU on descriptor data, a read after a
  forward join or at a loop head that a path brings the select to, a load whose word 2 is joined after a select on
  one path or on a loop back edge, and a select in the high word of a 64-bit ALU source (the same joins and loop
  without the select translate);
- still refused as before: s_cselect_b64, m0 and vcc_lo destinations, a raw descriptor word, a pointer or run-time
  register arm, and the vertex and pixel stages (either as a numeric use of descnum or because the descriptor word
  cannot be materialized);
- still numeric: a select of a number from word 3 of a table descriptor, which S materializes, translates as
  `scc ? 0 : <word 3>` and a later v_mov reads it.
Negative controls, each a mutated translator that must fail a named check:
- zero-arm-dropped: the select keeps only its descnum arm, so the conversion with the selected stride translates;
- load-check-removed: Stage.select_word finds nothing, so a load through a copied select translates;
- alu-check-removed: Stage.refuse_select_source accepts everything, so an ALU copy into word 2 reaches the load;
- join-check-removed: Stage.joined_word finds nothing, so a load after a join with a select on one path translates;
- high-word-unchecked: refuse_select_source sees only the low word of each source, so s_or_b64 of a pair whose high
  word holds the select translates;
- materializable-symbolic: Stage.descnum_materializes says no, so the word-3 select turns opaque and its read refuses.
The dispatch mode runs case A with the dead select on D3D12 and compares all 320 words with case A's reference.
"""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402
import startup_gfx2hlsl_scalar_m0_check as M  # noqa: E402

ZERO, LIT = 128, 255


def sop1(op, d, a):
    return struct.pack('<I', 0xbe800000 | d << 16 | op << 8 | a)


def sop2(op, d, a, b, lit=None):
    return struct.pack('<I', 0x80000000 | op << 23 | d << 16 | b << 8 | a) + (b'' if lit is None else struct.pack('<I', lit))


def sopc(op, a, b):
    return struct.pack('<I', 0xbf000000 | op << 16 | b << 8 | a)


def vop1(op, d, a):
    return struct.pack('<I', 0x7e000000 | d << 17 | op << 9 | a)


def vopc(op, a, vb):
    return struct.pack('<I', 0x7c000000 | op << 17 | vb << 9 | a)


MOV, ADD, CSEL, CSEL64, AND, LSHR = 0x03, 0x00, 0x0a, 0x0b, 0x0e, 0x20
WORDS = {
    's_lshr_b32 s21, s14, 2': sop2(LSHR, 21, 14, 130),
    's_lshr_b32 s20, s14, 2': sop2(LSHR, 20, 14, 130),
    's_lshr_b32 s21, s15, 2': sop2(LSHR, 21, 15, 130),
    's_cmp_eq_u32 s4, 0': sopc(0x06, 4, ZERO),
    's_cmp_eq_u32 s21, 0': sopc(0x06, 21, ZERO),
    's_cselect_b32 s21, 0, s21': sop2(CSEL, 21, ZERO, 21),
    's_cselect_b32 s21, s21, 0': sop2(CSEL, 21, 21, ZERO),
    's_cselect_b32 s21, 0x100, s21': sop2(CSEL, 21, LIT, 21, 0x100),
    's_cselect_b32 s21, s4, s21': sop2(CSEL, 21, 4, 21),
    's_cselect_b32 s14, 0, s14': sop2(CSEL, 14, ZERO, 14),
    's_cselect_b32 s22, 0, s10': sop2(CSEL, 22, ZERO, 10),
    's_cselect_b32 m0, 0, s21': sop2(CSEL, 124, ZERO, 21),
    's_cselect_b32 vcc_lo, 0, s21': sop2(CSEL, 106, ZERO, 21),
    's_cselect_b64 s[22:23], 0, s[20:21]': sop2(CSEL64, 22, ZERO, 20),
    's_mov_b32 s13, s21': sop1(MOV, 13, 21),
    's_mov_b32 s14, s21': sop1(MOV, 14, 21),
    's_mov_b32 s15, s21': sop1(MOV, 15, 21),
    's_mov_b32 s18, s21': sop1(MOV, 18, 21),
    's_mov_b32 s22, s21': sop1(MOV, 22, 21),
    's_mov_b32 s14, s22': sop1(MOV, 14, 22),
    's_mov_b32 m0, s21': sop1(MOV, 124, 21),
    's_add_u32 s22, s21, 0': sop2(ADD, 22, 21, ZERO),
    's_add_u32 s22, s22, 1': sop2(ADD, 22, 22, 129),
    's_and_b32 s22, s15, 0x30000000': sop2(AND, 22, 15, LIT, 0x30000000),
    'v_mov_b32_e32 v5, s21': vop1(0x01, 5, 21),
    'v_mov_b32_e32 v5, m0': vop1(0x01, 5, 124),
    'v_cmp_gt_u32_e32 vcc_lo, s21, v0': vopc(0xc4, 21, 0),
    's_add_u32 s21, s4, 1': sop2(ADD, 21, 4, 129),
    's_add_u32 s21, s4, 2': sop2(ADD, 21, 4, 130),
    's_cmp_eq_u32 s5, 0': sopc(0x06, 5, ZERO),
    's_mov_b32 s22, 0': sop1(MOV, 22, ZERO),
    's_add_i32 s22, s22, 1': sop2(0x02, 22, 22, 129),
    's_cmp_lt_u32 s22, 3': sopc(0x0a, 22, 131),
    's_cbranch_scc1 1': struct.pack('<I', 0xbf850001),
    's_cbranch_scc1 3': struct.pack('<I', 0xbf850003),
    's_cbranch_scc1 65529': struct.pack('<I', 0xbf85fff9),  # back 7 dwords
    's_cbranch_scc1 65528': struct.pack('<I', 0xbf85fff8),  # back 8 dwords
    's_cbranch_scc1 2': struct.pack('<I', 0xbf850002),
    's_cselect_b32 s14, 0, s21': sop2(CSEL, 14, ZERO, 21),
    's_mov_b32 s14, s14': sop1(MOV, 14, 14),
    's_or_b64 s[14:15], s[20:21], 0': sop2(0x11, 14, 20, ZERO),
    **{line: bytes.fromhex(C.SPLICE[line]) for line in C.RAW_CONVERSION},
}
for _text, _code in WORDS.items():
    (_, _, _name, _decoded), = G.decode(_code)
    if _name.startswith('s_cbranch'):  # the decoder prints an absolute label; the key spells SIMM16
        _decoded = '%s %d' % (_name, struct.unpack('<I', _code)[0] & 0xffff)
    assert _decoded == _text, (_text, _decoded)

DESCNUM = 's_lshr_b32 s21, s14, 2'            # a number from word 2 (NUM_RECORDS) of the load descriptor s[12:15]
COMPARE = 's_cmp_eq_u32 s4, 0'                # SCC from a root constant: a run-time value
SELECT = 's_cselect_b32 s21, 0, s21'          # the C500 form: zero or keep
SELECTED = (DESCNUM, COMPARE, SELECT)
STRIDE_EXTRACT, STRIDE_CLEAR, SCALE_RECORDS, OOB_CLEAR, OOB_RAW = C.RAW_CONVERSION  # on the store descriptor s[16:19]
LOAD = C.WORDS_A.index(bytes.fromhex('001030e004050380'))  # buffer_load_dword v5, v4, s[12:15], 0 offen
STORE = C.STORE_A                                          # buffer_store_dword v5, v4, s[16:19], 0 offen
assert [t for _, _, _, t in G.decode(C.WORDS_A[LOAD:LOAD + 8])] == ['buffer_load_dword v5, v4, s[12:15], 0 offen']

failures = []


def check(what, ok, detail=''):
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % (detail,)))
    if not ok:
        failures.append(what)


def at(position, *lines):
    return C.WORDS_A[:position] + b''.join(WORDS[line] for line in lines) + C.WORDS_A[position:]


def around_load(before, after):
    """Case A with `before` spliced in front of its buffer load and `after` behind it."""
    return (C.WORDS_A[:LOAD] + b''.join(WORDS[line] for line in before) + C.WORDS_A[LOAD:LOAD + 8]
            + b''.join(WORDS[line] for line in after) + C.WORDS_A[LOAD + 8:])


def translate(code):
    pipe = G.ComputePipeline(code, C.root_blob(), C.CASES['A'][2], (2, 3, 4))
    return pipe, pipe.hlsl_cs()


def accepted(what, code):
    try:
        result = translate(code)
    except G.Unsupported as reason:
        check(what, False, 'refused: %s' % reason)
        return None, None
    check(what, True)
    return result


def refused(what, reason, code):
    """`reason` is one phrase or a tuple of phrases, any of which the refusal may name."""
    reasons = reason if isinstance(reason, tuple) else (reason,)
    try:
        translate(code)
    except G.Unsupported as error:
        check(what, any(r in str(error) for r in reasons), 'expected %r in %r' % (reason, str(error)))
        return
    check(what, False, 'translated, expected a refusal naming %r' % (reason,))


def same_arms(got, want):
    """The select symbol `got` against `want`, where a ('descnum',) arm of `want` matches any descnum symbol (this
    line's descnum carries the expression it came from)."""
    return (got is not None and len(got) == len(want) and got[:3] == want[:3]
            and all(g == w or (w == ('descnum',) and g[:1] == w) for g, w in zip(got[3:], want[3:])))


def offset(position, *lines):
    """Byte offset of the instruction after `lines` spliced at `position`."""
    return position + sum(len(WORDS[line]) for line in lines)


def check_accepted():
    _, base = translate(C.WORDS_A)
    for select, arms in ((SELECT, (('const', 0), ('descnum',))), ('s_cselect_b32 s21, s21, 0', (('descnum',), ('const', 0))),
                         ('s_cselect_b32 s21, 0x100, s21', (('const', 0x100), ('descnum',)))):
        pipe, hlsl = accepted('dead %s translates' % select, at(STORE, DESCNUM, COMPARE, select))
        if pipe is None:
            continue
        extra = sorted(set(hlsl.splitlines()) - set(base.splitlines()))
        check('%s adds only the compare to the HLSL' % select,
              len(hlsl.splitlines()) == len(base.splitlines()) + 1 and len(extra) == 1 and extra[0].strip().startswith('scc = '),
              extra)
        want = ('descnum', 'cselect', offset(STORE, DESCNUM, COMPARE)) + arms
        check('%s keeps both arms and its offset' % select, same_arms(pipe.cs.sym.get(21), want),
              (pipe.cs.sym.get(21), want))
    # The extracted stride of the structured-to-raw conversion is a descnum too; the select keeps its symbol as an arm.
    pipe, _ = accepted('dead select of the extracted stride translates',
                       at(STORE, STRIDE_EXTRACT, COMPARE, SELECT))
    if pipe is not None:
        sym = pipe.cs.sym.get(21)
        check('the stride arm is kept whole', G.descriptor_select(sym) and sym[3] == ('const', 0)
              and sym[4][:2] == ('descnum', 'stride') and sym[4][2][0] == 'desc' and sym[4][2][3] == 5, sym)


def check_later_uses():
    numeric = 'numeric use of s21 holding descnum'
    refused('v_mov_b32 of the result', numeric, at(STORE, *SELECTED, 'v_mov_b32_e32 v5, s21'))
    refused('s_cmp_eq_u32 on the result', numeric, at(STORE, *SELECTED, 's_cmp_eq_u32 s21, 0'))
    refused('v_cmp_gt_u32 on the result', numeric, at(STORE, *SELECTED, 'v_cmp_gt_u32_e32 vcc_lo, s21, v0'))
    refused('the result through m0', 'numeric use of m0 holding descnum',
            at(STORE, *SELECTED, 's_mov_b32 m0, s21', 'v_mov_b32_e32 v5, m0'))
    refused('scalar ALU on the result', 's_add_u32 on a select of descriptor data',
            at(STORE, *SELECTED, 's_add_u32 s22, s21, 0'))
    refused('scalar ALU on a copy of the result', 's_add_u32 on a select of descriptor data',
            at(STORE, *SELECTED, 's_mov_b32 s22, s21', 's_add_u32 s22, s22, 1'))
    # The structured-to-raw conversion scales NUM_RECORDS by the stride; with the selected stride it would scale by 0
    # on one arm. Without the select this conversion translates (case I-raw of the compute check).
    refused('raw conversion with the selected stride', 's_mul_i32 on a select of descriptor data',
            at(STORE, STRIDE_EXTRACT, COMPARE, SELECT, STRIDE_CLEAR, SCALE_RECORDS, OOB_CLEAR, OOB_RAW))
    for word in (13, 14, 15):
        refused('load through a select copied into s%d' % word, 'descriptor operand s[12:15]: s%d holds a select' % word,
                at(LOAD, *SELECTED, 's_mov_b32 s%d, s21' % word))
    refused('load through a second copy of the select', 'descriptor operand s[12:15]: s14 holds a select',
            at(LOAD, *SELECTED, 's_mov_b32 s22, s21', 's_mov_b32 s14, s22'))
    refused('an ALU copy into word 2 before the load', 's_add_u32 on a select of descriptor data',
            at(LOAD, *SELECTED, 's_add_u32 s22, s21, 0', 's_mov_b32 s14, s22'))
    refused('store through a select copied into word 2', 'buffer store through a modified descriptor',
            at(STORE, *SELECTED, 's_mov_b32 s18, s21'))
    # SCC from an ALU on descriptor data is not emitted, so the select's condition is unknown.
    refused('select on SCC from descriptor data', 'scc from descriptor data in s_cselect_b32',
            at(STORE, DESCNUM, 's_and_b32 s22, s15, 0x30000000', SELECT))
    # The select writes no HLSL variable, so a join must not turn sX back into a run-time register. Forward: one path
    # skips the select and brings a run-time s21. The same shape with a run-time write in the skipped block translates.
    entry = ('s_add_u32 s21, s4, 1', 's_cmp_eq_u32 s5, 0')
    refused('read after a join with a skipped select', 'numeric use of s21 holding joined at a branch target',
            at(STORE, *entry, 's_cbranch_scc1 3', *SELECTED, 'v_mov_b32_e32 v5, s21'))
    accepted('the same join with a run-time write translates',
             at(STORE, *entry, 's_cbranch_scc1 1', 's_add_u32 s21, s4, 2', 'v_mov_b32_e32 v5, s21'))
    # Back edge: the loop head reads s21, which is run-time on entry and the select on the next pass.
    loop = ('v_mov_b32_e32 v5, s21', *SELECTED, 's_add_i32 s22, s22, 1', 's_cmp_lt_u32 s22, 3', 's_cbranch_scc1 65529')
    code = at(STORE, 's_add_u32 s21, s4, 1', 's_mov_b32 s22, 0', *loop)
    head = offset(STORE, 's_add_u32 s21, s4, 1', 's_mov_b32 s22, 0')
    assert 's_cbranch_scc1 label_0x%x' % head in [t for _, _, _, t in G.decode(code)], 'the back edge misses the head'
    refused('read at a loop head after a select on the back edge', 'numeric use of s21 holding joined at a branch target',
            code)
    # A select that reaches a descriptor word through a join leaves that word joined. The access names its resource by
    # word 0 only, so a joined word after it refuses (#2903 review, case 1). The same join with the loaded word on
    # both paths keeps the word and translates.
    split = (DESCNUM, 's_cmp_eq_u32 s5, 0', 's_cbranch_scc1 2', COMPARE)
    refused('load after a join where one path selects into word 2', 'descriptor operand s[12:15]: s14 holds a word joined',
            at(LOAD, *split, 's_cselect_b32 s14, 0, s21'))
    accepted('the same join with the loaded word on both paths translates', at(LOAD, *split, 's_mov_b32 s14, s14'))
    # The same through a loop back edge: the select into word 2 at the bottom reaches the load at the head.
    tail = ('s_add_i32 s22, s22, 1', 's_cmp_lt_u32 s22, 3')
    code = around_load(['s_mov_b32 s22, 0'], [DESCNUM, COMPARE, 's_cselect_b32 s14, 0, s21', *tail, 's_cbranch_scc1 65528'])
    assert 's_cbranch_scc1 label_0x%x' % (LOAD + 4) in [t for _, _, _, t in G.decode(code)], 'the back edge misses the load'
    refused('load at a loop head after a select into word 2 on the back edge',
            'descriptor operand s[12:15]: s14 holds a word joined', code)
    code = around_load(['s_mov_b32 s22, 0'], [DESCNUM, COMPARE, *tail, 's_cbranch_scc1 65529'])
    assert 's_cbranch_scc1 label_0x%x' % (LOAD + 4) in [t for _, _, _, t in G.decode(code)], 'the back edge misses the load'
    accepted('the same loop without the select translates', code)
    # A select in the high word of a 64-bit source (#2903 review, case 2): scalar() names the low word only.
    refused('select in the high word of a 64-bit ALU source', 's_or_b64 on a select of descriptor data',
            at(LOAD, 's_lshr_b32 s20, s14, 2', *SELECTED, 's_or_b64 s[14:15], s[20:21], 0'))


def check_still_refused():
    # A descnum S cannot materialize (word 2 of a table descriptor, a vertex-buffer table) refuses where it is read.
    unmaterialized = 'cannot materialize descriptor word'
    refused('s_cselect_b64 on descnum', ('numeric use of s20 holding descnum', unmaterialized),
            at(STORE, 's_lshr_b32 s20, s14, 2', COMPARE, 's_cselect_b64 s[22:23], 0, s[20:21]'))
    refused('m0 destination', ('numeric use of s21 holding descnum', unmaterialized),
            at(STORE, DESCNUM, COMPARE, 's_cselect_b32 m0, 0, s21'))
    refused('vcc_lo destination', ('numeric use of s21 holding descnum', unmaterialized),
            at(STORE, DESCNUM, COMPARE, 's_cselect_b32 vcc_lo, 0, s21'))
    refused('raw descriptor word arm', 'numeric use of s14 holding desc', at(STORE, COMPARE, 's_cselect_b32 s14, 0, s14'))
    refused('pointer arm', 'numeric use of s10 holding ptr', at(STORE, COMPARE, 's_cselect_b32 s22, 0, s10'))
    refused('run-time register arm', ('numeric use of s21 holding descnum', unmaterialized),
            at(STORE, DESCNUM, COMPARE, 's_cselect_b32 s21, s4, s21'))
    # Vertex and pixel stages keep the refusal: the probe stage holds a descriptor in s[16:19].
    for kind in ('vs', 'ps'):
        stage = M.ProbeStage()
        stage.kind = kind
        stage.one('s_add_u32', 's_add_u32 s20, s16, 4')
        try:
            stage.one('s_cselect_b32', 's_cselect_b32 s20, 0, s20')
        except G.Unsupported as error:
            check('%s stage keeps the refusal' % kind, any(r in str(error) for r in (
                'numeric use of s20 holding descnum', unmaterialized)), str(error))
        else:
            check('%s stage keeps the refusal' % kind, False, 'translated')


def check_still_numeric():
    """A select of a number S materializes (word 3 of a table descriptor) keeps its run-time value."""
    try:
        _, hlsl = translate(at(STORE, 's_lshr_b32 s21, s15, 2', COMPARE, SELECT, 'v_mov_b32_e32 v5, s21'))
    except G.Unsupported as error:
        check('select of a word-3 descnum stays numeric', False, 'refused: %s' % error)
        return
    check('select of a word-3 descnum stays numeric',
          any(line.strip().startswith('s21 = scc ? 0x00000000u : ') for line in hlsl.splitlines()), 'no numeric select')


def check_all():
    check_accepted()
    check_later_uses()
    check_still_refused()
    check_still_numeric()


def zero_arm_dropped(original):
    def select(self, ops, pair):
        d, r = G.reg_range(ops[0]), G.reg_range(ops[2])
        sym = self.sym.get(r[1]) if r and r[0] == 's' else None
        if not pair and d and d[0] == 's' and sym and sym[0] == 'descnum' and ops[1] == '0':
            self.sym[d[1]] = sym  # the careless lift: the zero arm is gone
            return
        original(self, ops, pair)
    return select


def low_words_only(original):
    def refuse(self, base, srcs):  # the check before the #2903 review: the low register of each source only
        return original(self, base, ['s%d' % G.reg_range(t)[1] if G.reg_range(t) and G.reg_range(t)[0] == 's' else t
                                     for t in srcs])
    return refuse


MUTATIONS = {
    'zero-arm-dropped': ('scalar_select', zero_arm_dropped, 'raw conversion with the selected stride'),
    'load-check-removed': ('select_word', lambda original: lambda self, tok: None, 'load through a select copied into s14'),
    'alu-check-removed': ('refuse_select_source', lambda original: lambda self, base, srcs: None,
                          'an ALU copy into word 2 before the load'),
    'join-check-removed': ('joined_word', lambda original: lambda self, tok: None,
                           'load after a join where one path selects into word 2'),
    'high-word-unchecked': ('refuse_select_source', low_words_only, 'select in the high word of a 64-bit ALU source'),
    'materializable-symbolic': ('descnum_materializes', lambda original: lambda self, sym: False,
                                'select of a word-3 descnum stays numeric'),
}


def check_mutations():
    global failures
    for name, (method, mutate, expected) in MUTATIONS.items():
        original = getattr(G.Stage, method)
        setattr(G.Stage, method, mutate(original))
        kept, failures = failures, []
        try:
            print('-- control %s (the check must fail)' % name)
            check_all()
        finally:
            setattr(G.Stage, method, original)
            caught, failures = failures, kept
        check('control %s fails "%s"' % (name, expected), expected in caught, caught)


def check_dispatch(dxc, runner, adapter):
    """Case A with the dead select, on D3D12: every word must match case A's reference."""
    C.CASES['A-cselect'] = (C.SOURCE_A, at(STORE, *SELECTED), *C.CASES['A'][2:])
    reference = C.expected
    C.expected = lambda name: reference('A' if name == 'A-cselect' else name)
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        run = C.run_case('A-cselect', dxc, str(Path(runner).resolve()), adapter, Path(tmp))
    print(run.stdout, end='')
    if run.returncode == 77:
        return 77
    diagnostics = C.renderer_diagnostics(run.stderr, adapter)
    if run.returncode or diagnostics:
        print('case A-cselect: %s' % (run.stderr or 'exit %d' % run.returncode), file=sys.stderr)
        return 1
    print('case A-cselect: all %d words match' % C.WORDS)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1:] == ['controls']:
        check_all()
        check_mutations()
        print('%d failure(s)' % len(failures))
        sys.exit(1 if failures else 0)
    if len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    sys.exit(__doc__)
