"""A VOPC compare whose destination is not an SGPR, vcc or exec is refused by name in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_compare_dst_check.py

The e64 (VOP3) form of a VOPC compare names its scalar destination. The generic compare path wrote exec and vcc
itself and took `reg_range(dst)[1]` for anything else; for vcc_hi, exec_hi or m0, reg_range returns None, so the
TypeError ended the whole translation instead of refusing the instruction. The destination is now checked as the
class-compare path and the carry-out of v_add_co_u32 already do, and refused as "<op> into <dst>".

Synthetic instruction text only. Checked: integer and float compares into vcc_hi, exec_hi and m0 raise
gfx2hlsl.Unsupported naming the destination (not another exception). Controls: the same compares into s20 are
lowered and leave s20 marked as a lane mask; into vcc_lo they write vcc; into exec_lo they write exec.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = 0


def check(what, ok, detail=''):
    global failures
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % (detail,)))
    if not ok:
        failures += 1


class Pipe:
    sampler_heap = False
    root = {'params': [{'type': 2}], 'flags': 0}

    def resource(self, key, kind):
        return 'r0'

    def ud(self, source):
        return 0


def stage():
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.pc = 0
    return s


OPS = ['v_cmp_lt_u32_e64', 'v_cmp_eq_i32_e64', 'v_cmp_gt_f32_e64', 'v_cmp_neq_f32_e64']


def main():
    for op in OPS:
        for dst in ('vcc_hi', 'exec_hi', 'm0'):
            s = stage()
            try:
                s.one(op, '%s %s, v0, v1' % (op, dst))
                check('%s into %s refused' % (op, dst), False, 'accepted: %s' % s.lines)
            except G.Unsupported as e:
                check('%s into %s refused' % (op, dst), dst in str(e), str(e))
            except Exception as e:  # the crash this change removes
                check('%s into %s refused' % (op, dst), False, '%s: %s' % (type(e).__name__, e))
        s = stage()
        s.one(op, '%s s20, v0, v1' % op)
        check('%s into s20 lowered as a lane mask' % op, any(l.startswith('s20 = ') for l in s.lines) and 20 in s.masks,
              s.lines)
        s = stage()
        s.one(op, '%s vcc_lo, v0, v1' % op)
        check('%s into vcc_lo writes vcc' % op, any(l.startswith('vcc = ') for l in s.lines), s.lines)
        s = stage()
        s.one(op, '%s exec_lo, v0, v1' % op)
        check('%s into exec_lo writes exec' % op, any(l.startswith('exec = ') for l in s.lines), s.lines)
    print('compare destination checks: %d failed' % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
