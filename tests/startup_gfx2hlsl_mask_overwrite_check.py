"""A scalar write ends a register's life as a lane mask in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_mask_overwrite_check.py

The translator keeps the SGPRs that hold a lane mask (a v_cmp destination, a saveexec copy) in `Stage.masks`, and a
scalar logic op with a mask source is lowered as a mask op: its result is reduced to the lane's bit. The generic
scalar path cleared the mark when it overwrote a register with ordinary data, but the other writers did not:
after `v_cmp_eq_u32_e64 s6, v0, v1` and `s_buffer_load_dword s6, s[16:19], 0x0`, s6 holds a buffer word, yet
`s_or_b32 s8, s6, s7` was still lowered as a mask OR and cut to bit 0. Every scalar write through `set_s` now clears
the mark (the writers of a mask mark the register again after it), and `s_load` clears it on its destination
registers, also in the forms that only record a descriptor symbol.

Synthetic instruction text and register states only. Checked, each with the s_or_b32 that follows evaluated on a
buffer word with the restricted evaluator of startup_gfx2hlsl_bit_alu_check: an s_buffer_load, an s_load through a
root pointer, an s_load of a descriptor, and an s_cselect_b32 into the former mask register. Controls: without the
overwrite the OR stays a mask OR, and a compare or a mask logic op written after a load marks the register again.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

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


DESC = {16: ('desc', ('table', 0), 0x20, 0), 17: ('desc', ('table', 0), 0x20, 1),
        18: ('desc', ('table', 0), 0x20, 2), 19: ('desc', ('table', 0), 0x20, 3),
        2: ('ptr', ('root', 0)), 3: ('ptrhi', ('root', 0)),
        12: ('ptr', ('table', 1)), 13: ('ptrhi', ('table', 1))}


def stage():
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.sym.update(DESC)
    s.pc = 0
    return s


def run(s, *texts):
    for text in texts:
        s.one(text.split(' ')[0], text)


def or_value(s, s6, s7):
    """Value of s8 after s_or_b32 s8, s6, s7, from the emitted line."""
    s.lines = []
    run(s, 's_or_b32 s8, s6, s7')
    line = next(l for l in s.lines if l.startswith('s8 = '))
    return Expr(line[5:-1]).run({'s6': s6, 's7': s7})[0], line


COMPARE = 'v_cmp_eq_u32_e64 s6, v0, v1'
WORD = 0x12345678


def main():
    # Control: a compare marks s6, and an OR of it is a mask OR (the lane's bit).
    s = stage()
    run(s, COMPARE)
    check('control: v_cmp marks s6 as a lane mask', 6 in s.masks, s.masks)
    value, line = or_value(s, 1, 0)
    check('control: an OR of the mask is reduced to the lane bit', '& 1' in line and 8 in s.masks, line)

    overwrites = (
        ('s_buffer_load_dword s6, s[16:19], 0x0', True),
        ('s_load_dword s6, s[2:3], 0x8', True),     # through a root pointer: set_s with a buffer load
        ('s_cselect_b32 s6, s20, s21', True),
    )
    for text, evaluate in overwrites:
        s = stage()
        run(s, COMPARE, text)
        check('%s clears the mark on s6' % text, 6 not in s.masks, s.masks)
        if evaluate:
            value, line = or_value(s, WORD, 0x100)
            check('%s: the following OR keeps the whole word (0x%08x)' % (text, WORD | 0x100),
                  value == WORD | 0x100 and 8 not in s.masks, line)

    # An s_load of a descriptor only records symbols; the mark must go there as well.
    s = stage()
    run(s, 'v_cmp_eq_u32_e64 s5, v0, v1', 's_load_dwordx4 s[4:7], s[12:13], 0x20')
    check('s_load_dwordx4 of a descriptor clears the mark on s5', 5 not in s.masks, s.masks)

    # A mask written after a load marks the register again.
    s = stage()
    run(s, 's_buffer_load_dword s6, s[16:19], 0x0', COMPARE)
    check('control: a compare after the load marks s6 again', 6 in s.masks, s.masks)
    s = stage()
    run(s, 's_buffer_load_dword s6, s[16:19], 0x0', 's_and_b32 s6, vcc_lo, s7')
    check('control: a mask logic op after the load marks s6 again', 6 in s.masks, (s.masks, s.lines))
    # Every other writer of a mask marks the register after its write as well.
    for writer in ('s_and_saveexec_b32 s6, vcc_lo', 's_andn2_wrexec_b32 s6, vcc_lo', 's_mov_b32 s6, exec_lo',
                   's_cselect_b64 s[6:7], -1, 0', 'v_cmp_class_f32_e64 s6, v0, v1'):
        s = stage()
        try:
            run(s, 's_buffer_load_dword s6, s[16:19], 0x0', writer)
        except G.Unsupported as error:
            check('control: %s after the load translates' % writer, False, error)
            continue
        check('control: %s after the load marks s6 again' % writer, 6 in s.masks, (s.masks, s.lines))

    if failures:
        print('FAILED: %d check(s)' % failures)
        return 1
    print('mask overwrite: loads and selects clear the mask mark; mask writers mark again')
    return 0


if __name__ == '__main__':
    sys.exit(main())
