"""DPP spellings of VOP2 operations in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_dpp_check.py

The translator models one lane. A DPP instruction whose control reads the lane's own value
(quad_perm:[0,1,2,3] or the identity dpp8, row_mask and bank_mask 0xf) is the plain operation, so it must translate
to exactly the statements of the plain spelling. Every control that reads a neighbouring lane, masks the lane or
needs bound_ctrl must be refused, because lowering it as the plain operation would silently use the wrong value.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


def stage():
    class Probe(G.Stage):
        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.exec_known = True
    return Probe()


def translate(line):
    s = stage()
    s.one(line.split(' ', 1)[0], line)
    return s.lines


IDENTITY = 'quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf'
PAIRS = [
    ('v_add_f32', 'v1, v2, v3'), ('v_sub_f32', 'v1, v2, v3'),
    ('v_or_b32', 'v1, v2, v3'), ('v_add_nc_u32', 'v1, v2, v3'),
]
for op, operands in PAIRS:
    plain = translate('%s %s' % (op, operands))
    dpp = translate('%s_dpp %s %s' % (op, operands, IDENTITY))
    check(op + ' dpp == plain', plain and dpp == plain, '%s vs %s' % (dpp, plain))
    dpp8 = translate('%s_dpp %s dpp8:[0,1,2,3,4,5,6,7]' % (op, operands))
    check(op + ' dpp8 identity == plain', dpp8 == plain, '%s vs %s' % (dpp8, plain))

# source modifiers and the output modifier travel with the plain operation
check('add_f32 dpp keeps clamp', translate('v_add_f32_dpp v1, -v2, |v3| %s clamp' % IDENTITY) ==
      translate('v_add_f32 v1, -v2, |v3| clamp'))

REFUSED = [
    'v_add_f32_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf',
    'v_add_f32_dpp v1, v2, v3 row_shr:1 row_mask:0xf bank_mask:0xf',
    'v_or_b32_dpp v1, v2, v3 row_mirror row_mask:0xf bank_mask:0xf',
    'v_add_nc_u32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xa bank_mask:0xf',
    'v_sub_f32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0x5',
    'v_add_f32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf bound_ctrl:0',
    'v_add_f32_dpp v1, v2, v3 dpp8:[1,0,2,3,4,5,6,7]',
    'v_add_f32_dpp v1, v2, v3 row_mask:0xf bank_mask:0xf',
    'v_swap_b32_dpp v1, v2 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf',
]
for line in REFUSED:
    try:
        translate(line)
        check('refuse ' + line, False, 'translated')
    except G.Unsupported as err:
        check('refuse ' + line, True, str(err))

if failures:
    print('FAIL: %s' % ', '.join(failures))
    sys.exit(1)
print('PASS')
