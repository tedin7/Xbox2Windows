"""Check the NGG vertex pass-through lowering in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_ngg_passthrough_check.py                              pure Python controls
       startup_gfx2hlsl_ngg_passthrough_check.py draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                                                              draw the translated pair, compare with a CPU reference

A pass-through NGG vertex shader asks for its allocation, parks per-lane values in LDS, waits at a barrier, reads
them back and exports position, parameters and the primitive word. The lowering drops the allocation request and the
barrier, keeps the position and parameter exports, and accepts the primitive export only when it hands on the entry
VGPR0 word unchanged.

The LDS round trip becomes a register copy only when every address is proven to be stride * lane + offset built from
the lane id (v_mbcnt_lo/hi), aligned, inside 64 KiB, and the four-byte intervals of all lane pairs and all write
offsets are disjoint. Every other address (the entry v0, a copy of it, a stride that lets lanes meet, an address read
as data) stays refused.
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    def resource(self, key, kind):
        return 'r0'

    def mrt_kind(self, index):
        return 'float'


ALLOC = 's_sendmsg 0x9'
BARRIER = 's_barrier 0x0'
PRIM = 'exp target_20 v0, off, off, off done'
POS = 'exp pos0 v4, v5, v6, v7 done'
LANE = ('v_mbcnt_lo_u32_b32 v1, -1, 0', 'v_mbcnt_hi_u32_b32 v1, -1, v1')   # v1 = lane id 0..63


def run(*instructions, kind='vs', branch=False):
    """Translate disassembly text with the same dispatch and branch scan as a real stage."""
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.lines, s.kind = Pipe(), [], kind
    s.attrs, s.sym, s.exports = {}, {}, {}
    s.used_s, s.used_v, s.masks = set(), set(), set()
    s.exec_known = True
    asm = list(instructions)
    if branch:
        asm = ['s_cbranch_scc0 label_0008'] + asm
    s.ins = [(4 * i, 4, text.split()[0], text) for i, text in enumerate(asm)]
    try:
        s.translate()
    except G.Unsupported as error:
        return str(error)
    return [line for _, lines, _ in s.blocks for line in lines]


def refused(label, got, reason):
    check('refused: ' + label, isinstance(got, str) and reason in got, repr(got))


got = run(ALLOC, BARRIER, POS, PRIM)
check('allocation request and barrier are dropped', got == ['o.pos = float4(F(v4), F(v5), F(v6), F(v7));'], repr(got))

# ---- proven shapes are lowered ------------------------------------------------------------------------------------
# Stride 8, offsets 0 and 4 (R0Y00's "disjoint case"): distinct payloads v2 and v8 come back unchanged in v3 and v9.
DISJOINT = (ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v2', 'ds_write_b32 v10, v8 offset:4', BARRIER,
            'ds_read_b32 v3, v10', 'ds_read_b32 v9, v10 offset:4', POS)
got = run(*DISJOINT)
check('lowered: stride 8, offsets 0 and 4, distinct payloads',
      got == ['v3 = v2;', 'v9 = v8;', 'o.pos = float4(F(v4), F(v5), F(v6), F(v7));'], repr(got))
# One offset at stride 4, an address from add and mad forms, a read through another register with the same form.
got = run(ALLOC, *LANE, 'v_mad_u32_u24 v10, v1, 12, 32', 'v_add_nc_u32 v11, 16, v10', 'ds_write_b32 v11, v2',
          BARRIER, 'ds_read_b32 v3, v10 offset:16', POS)
check('lowered: stride 12 from mad, read through a register of equal form', got[:1] == ['v3 = v2;'], repr(got))
got = run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 2, v1', 'ds_write_b32 v10, v2 offset:16', BARRIER,
          'ds_read_b32 v3, v10 offset:16', POS)
check('lowered: stride 4, single offset', got[:1] == ['v3 = v2;'], repr(got))

got = run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v2', 'ds_write_b32 v10, v8', BARRIER,
          'ds_read_b32 v3, v10', POS)
check('lowered: the same lane rewrites its slot, the read returns the latest payload', got[:1] == ['v3 = v8;'], repr(got))

# ---- controls that must be refused --------------------------------------------------------------------------------
# With a stride of 4, lane A's offset 20 and lane B's offset 16 name the same DWORD (lane 0 + 20 == lane 1 + 16).
ALIAS = (ALLOC, *LANE, 'v_lshlrev_b32 v10, 2, v1', 'ds_write_b32 v10, v2 offset:16', 'ds_write_b32 v10, v8 offset:20',
         BARRIER, 'ds_read_b32 v3, v10 offset:20', 'ds_read_b32 v9, v10 offset:16', PRIM)
refused('aligned offsets 16 and 20 alias across lanes (distinct payloads v2, v8)', run(*ALIAS), 'not disjoint')
refused('stride 0: every lane shares one slot',
        run(ALLOC, *LANE, 'v_mul_u32_u24 v10, 0, v1', 'ds_write_b32 v10, v2', BARRIER, 'ds_read_b32 v3, v10', POS),
        'not disjoint')
refused('stride 2 is not aligned',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 1, v1', 'ds_write_b32 v10, v2', POS), 'aligned stride')
refused('offset 2 is not aligned',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v2 offset:2', POS), 'aligned stride')
refused('lane 63 leaves the 64 KiB allocation',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 10, v1', 'ds_write_b32 v10, v2 offset:4096', POS), 'aligned stride')
# Non-affine and unproven addresses.
refused('entry v0 as the address', run(ALLOC, 'ds_write_b32 v0, v2', BARRIER, 'ds_read_b32 v3, v0', PRIM), 'is proven')
refused('plain copy of the entry v0',
        run(ALLOC, 'v_mov_b32_e32 v1, v0', 'ds_write_b32 v1, v8 offset:16', POS), 'is proven')
refused('low half of the lane count alone (lanes 32..63 share a value)',
        run(ALLOC, LANE[0], 'ds_write_b32 v1, v8', POS), 'is proven')
refused('sum of two lane-varying registers (non-affine)',
        run(ALLOC, *LANE, 'v_add_nc_u32 v10, v1, v2', 'ds_write_b32 v10, v8', POS), 'only valid as an LDS address')
refused('product of the lane id with itself (non-affine)',
        run(ALLOC, *LANE, 'v_mul_u32_u24 v10, v1, v1', 'ds_write_b32 v10, v8', POS), 'only valid as an LDS address')
refused('address register overwritten by an unproven value',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'v_xor_b32 v10, v2, v10', 'ds_write_b32 v10, v8', POS),
        'only valid as an LDS address')
refused('lane address read as data',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'v_add_nc_u32 v6, v2, v10', POS), 'only valid as an LDS address')
refused('lane address exported', run(ALLOC, *LANE, 'exp pos0 v1, v1, v1, v1 done'), 'only valid as an LDS address')
refused('lane address stored as the payload',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v10', POS), 'only valid as an LDS address')
# Slots.
refused('read of a form nothing wrote',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v2', BARRIER, 'ds_read_b32 v3, v10 offset:4'),
        'no write')
refused('read without a write', run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_read_b32 v3, v10'), 'no write')
refused('payload register changed after the write',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v2', 'v_mov_b32 v2, 5', BARRIER,
            'ds_read_b32 v3, v10'), 'no write')
refused('LDS access with a second modifier',
        run(ALLOC, *LANE, 'v_lshlrev_b32 v10, 3, v1', 'ds_write_b32 v10, v2 offset:4 gds'), 'at most one offset')
# Stage controls.
REFUSED = [
    ('primitive word computed from v0', (ALLOC, 'v_add_nc_u32_e32 v0, 1, v0', PRIM), 'non-identity primitive export'),
    ('primitive word from another register', (ALLOC, 'exp target_20 v1, off, off, off done'),
     'non-identity primitive export'),
    ('primitive export with extra channels', (ALLOC, 'exp target_20 v0, v1, off, off done'),
     'non-identity primitive export'),
    ('compressed primitive export', (ALLOC, 'exp target_20 v0, off, off, off compr done'),
     'non-identity primitive export'),
    ('primitive export after a branch', (PRIM,), 'non-identity primitive export', True),
    ('unknown message', ('s_sendmsg 0x2',), 's_sendmsg'),
    ('allocation message with stream bits', ('s_sendmsg 0x109',), 's_sendmsg'),
    ('LDS write in a pixel shader', ('ds_write_b32 v1, v8 offset:16',), 'ds_write_b32',
     False, 'ps'),
    ('LDS after a branch', (*LANE, 'ds_write_b32 v1, v8 offset:16'), 'requires a straight-line vertex shader', True),
]
for label, instructions, reason, *extra in REFUSED:
    branch = bool(extra and extra[0])
    kind = extra[1] if len(extra) > 1 else 'vs'
    refused(label, run(*instructions, branch=branch, kind=kind), reason)


def mutation_control():
    """Without the disjointness check the aliasing case is accepted: the control above fails then."""
    saved = G.Stage.lds_overlap
    G.Stage.lds_overlap = lambda self, a, b: False
    try:
        return run(*ALIAS)
    finally:
        G.Stage.lds_overlap = saved


mutated = mutation_control()
check('mutation: no disjointness check, the aliasing control is lowered (so the control above would fail)',
      isinstance(mutated, list), repr(mutated))


# ---- real D3D12 draw: the lowered vertex shader against a CPU reference ----------------------------------------------
# Vertex words: the recorded full-screen triangle prefix (5 instructions, v0 entry word, v1/v2/v4/v5 position), twelve
# payload literals in v11..v22, then the NGG tail below (llvm-mc gfx1013 from startup_gfx2hlsl_ngg_passthrough_tail.s, checked by the binary mode): allocation request, lane id, address v24 = 48 * lane and v25 = v24 + 64, twelve
# ds_write_b32 through v25 offsets 0..44, barrier, twelve ds_read_b32 through v24 offsets 64..108 into v30..v41, the
# position, the parameters from v30..v41 and the primitive word. The twelve payloads exist only in LDS after the writes:
# the pixel reads v30..v41, so a wrong or dropped forward changes the drawn word.
TAIL_WORDS = bytes.fromhex(
    '090090bf170065d7c1000100170066d7c12e0200b02e3016c030324a000034d8190b0000040034d8190c0000080034d8190d0000'
    '0c0034d8190e0000100034d8190f0000140034d819100000180034d8191100001c0034d819120000200034d819130000240034d8'
    '19140000280034d8191500002c0034d8191600007fc08cbf00008abf4000d8d81800001e4400d8d81800001f4800d8d818000020'
    '4c00d8d8180000215000d8d8180000225400d8d8180000235800d8d8180000245c00d8d8180000256000d8d8180000266400d8d8'
    '180000276800d8d8180000286c00d8d8180000297fc08cbfcf0800f8010205040f0200f81e1f20211f0200f8222324252f0200f8'
    '26272829410900f800000000000081bf')
TAIL_SOURCE = HERE / 'startup_gfx2hlsl_ngg_passthrough_tail.s'


def ngg_vs_words(sets):
    from startup_flat_param_check import VS_WORDS
    from startup_gfx2hlsl_bit_alu_check import mov_literal
    words = [w for s in range(3) for w in (x[s] for x in sets)]
    return VS_WORDS[:40] + b''.join(mov_literal(11 + i, w) for i, w in enumerate(words)) + TAIL_WORDS


def check_binary(llvm_mc):
    from startup_flat_param_check import assemble
    built = assemble(llvm_mc, TAIL_SOURCE.read_text())
    if built != TAIL_WORDS:
        raise AssertionError('NGG tail: assembled %s, recorded %s' % (built.hex(), TAIL_WORDS.hex()))
    print('binary: NGG tail matches llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    from startup_flat_param_check import compile_hlsl, draw, root_blob
    from startup_gfx2hlsl_bit_alu_check import DRAW, OPS, ps_words, words_text
    op = 'v_xor3_b32'
    reference = OPS[op][2]
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            good_ps = compile_hlsl(compiler, kind, G.Pipeline(ngg_vs_words([(0, 0, 0)] * 4), ps_words(op),
                                                                root_blob()).hlsl_ps(),
                                   'ps_%s' % kind, ps_profile, directory)
            for index in range(0, 8, 4):
                sets = DRAW[op][index:index + 4]
                hlsl = G.Pipeline(ngg_vs_words(sets), ps_words(op), root_blob()).hlsl_vs()
                if 'ds_' in hlsl or 'GroupMemory' in hlsl:
                    raise AssertionError('NGG lowering left LDS code in the vertex shader')
                vs = compile_hlsl(compiler, kind, hlsl, 'vs_%d_%s' % (index, kind), vs_profile, directory)
                expected = words_text(reference(*x) for x in sets)
                result = draw(renderer, adapter, vs, good_ps, expected)
                if result.returncode:
                    raise AssertionError('%s %s\n%s\n%s' % (kind, expected, result.stdout, result.stderr))
                print('%s ngg draw %d: %s' % (kind, index // 4, expected), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == 'binary':
        if failures:
            print('FAILED: ' + ', '.join(failures))
            sys.exit(1)
        if len(sys.argv) > 2:
            check_binary(sys.argv[2])
        print('all ok')
    elif sys.argv[1] == 'draw':
        if failures:
            print('FAILED: ' + ', '.join(failures))
            sys.exit(1)
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
