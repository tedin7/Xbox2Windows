"""`s_mov_b32` copy of a plain constant SGPR in tools/startup/gfx2hlsl.py: run-time write and join.

The copy folds the constant into later reads (revision x2w-startup-shader-28), and these two cases pin what must
stay true next to the fold:
- the copy still assigns its register at run time (`s12 = ...;` is emitted), so a reader the fold does not reach
  (a branch target after a join) sees a real value;
- at a join where one arm rewrites the copy at run time, the copy's constant is dropped, so a read after the join
  uses the register and not the stale constant.

The instruction words are built from the SOP1 / SOP2 / SOPC / SOPP field layout that `tools/disasm_rdna2.py` decodes.
"""
import os
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(op, sdst, ssrc0):
    return struct.pack('<I', 0xBE800000 | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) | (ssrc0 & 0xFF))


def sop2(op, sdst, ssrc0, ssrc1):
    return struct.pack('<I', (0b10 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8)
                       | (ssrc0 & 0xFF))


def sopc(op, ssrc0, ssrc1):
    return struct.pack('<I', 0xBF000000 | ((op & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))


def sopp(op, simm=0):
    return struct.pack('<I', 0xBF800000 | ((op & 0x7F) << 16) | (simm & 0xFFFF))


MOV, ADD, CMP_EQ, ENDPGM, CBRANCH_SCC0 = 0x03, 0x00, 0x06, 0x01, 0x04
INLINE_1, INLINE_3, INLINE_5 = 129, 131, 133
S4, S6, S12, S13, S15 = 4, 6, 12, 13, 15
CONST_WORD = '0x00000005u'


class Pipeline:
    """Just enough pipeline state for Stage.translate()."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


def blocks_of(code):
    stage = G.Stage.__new__(G.Stage)
    stage.pipe = Pipeline()
    stage.kind = 'ps'
    stage.lines, stage.sym = [], {}
    stage.used_s, stage.used_v, stage.masks = set(), set(), set()
    stage.attrs = {}
    stage.exec_known = True
    stage.ins = G.decode(code)
    stage.force_joined = {}
    stage.translate()
    return {off: lines for off, lines, _ in stage.blocks}


COPY = [sop1(MOV, S15, INLINE_5), sop1(MOV, S12, S15)]            # s15 = 5; s12 = s15
READ = sop2(ADD, S13, S12, INLINE_1)                              # s13 = s12 + 1

# Straight line: the read folds the constant, and the copy is still written at run time.
straight = blocks_of(b''.join(COPY + [READ, sopp(ENDPGM)]))
text = '\n'.join(l for lines in straight.values() for l in lines)
check('straight line: the copy is still assigned at run time', 's12 = ' in text, text)
check('straight line: the read folds the copied constant', 's12 + ' not in text and CONST_WORD in text, text)

# Skip: s_cmp_eq_u32 s4, 3; s_cbranch_scc0 JOIN; s_mov_b32 s12, s6 (run-time rewrite); JOIN: read.
join_at = 4 * (len(COPY) + 3)
skip = (COPY + [sopc(CMP_EQ, S4, INLINE_3), sopp(CBRANCH_SCC0, 1), sop1(MOV, S12, S6), READ, sopp(ENDPGM)])
joined = blocks_of(b''.join(skip))
after = '\n'.join(joined.get(join_at, []))
check('join: a read after a run-time rewrite on one arm uses the register', 's12' in after, repr(joined))
check('join: the stale constant is not folded into the read', CONST_WORD not in after, after)

if failures:
    print('FAIL: ' + ', '.join(failures))
    sys.exit(1)
print('startup_gfx2hlsl_const_copy_join_check: ok')
