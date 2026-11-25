"""s_cselect_b32 into vcc_lo and exec_lo in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_cselect_mask_check.py

Synthetic SOP2 words only (op [29:23] = 10, sdst [22:16], ssrc1 [15:8], ssrc0 [7:0]; the names are asserted from the
translator's decoder). In wave32, vcc_lo and exec_lo hold the whole lane mask, and the generic scalar path already
writes them as `vcc` and `exec`. The select path assumed an SGPR destination, so `s_cselect_b32 vcc_lo, -1, 0`
raised a TypeError instead of a translation or an Unsupported refusal.

Checked: vcc_lo and exec_lo receive the lane's bit of the selected word for both SCC values (evaluated with the
restricted evaluator of startup_gfx2hlsl_bit_alu_check), exec_lo clears exec_known, an SGPR and an m0 destination
lower as before, and a destination that is neither (the null SGPR) is refused with an Unsupported message.
Negative control: the same evaluation with the selected word not reduced to the lane bit gives a different vcc.
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
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % detail))
    if not ok:
        failures += 1


class Pipe:
    root = {'params': [], 'flags': 0}

    def ud(self, source):
        return 0


SOP2_CSELECT_B32 = 0x80000000 | 10 << 23
INLINE_ZERO, INLINE_ONE, INLINE_MINUS_ONE = 128, 129, 193
VCC_LO, EXEC_LO, M0, NULL = 106, 126, 124, 125


def cselect(sdst, ssrc0, ssrc1):
    return SOP2_CSELECT_B32 | sdst << 16 | ssrc1 << 8 | ssrc0


def one(word):
    s = G.Stage(Pipe(), 'ps', b'')
    s.lines = []
    _, _, name, text = G.decode(struct.pack('<I', word))[0]
    s.pc = 0
    s.one(name, text)
    return s, text, s.lines


def value(line, target, scc):
    assert line.startswith(target + ' = ') and line.endswith(';'), line
    return Expr(line[len(target) + 3:-1].replace('scc ?', '(scc != 0u) ?')).run({'scc': scc})[0]


def main():
    for sdst, target in ((VCC_LO, 'vcc'), (EXEC_LO, 'exec')):
        for ssrc0, ssrc1, words in ((INLINE_MINUS_ONE, INLINE_ZERO, (0xffffffff, 0)),
                                    (INLINE_ZERO, INLINE_ONE, (0, 1))):
            try:
                s, text, lines = one(cselect(sdst, ssrc0, ssrc1))
            except Exception as error:  # the old select path raised TypeError here
                check('s_cselect_b32 into %s translates' % target, False, '%s: %s' % (type(error).__name__, error))
                continue
            check('%s decodes as s_cselect_b32 %s_lo' % (target, target), text.startswith('s_cselect_b32 %s_lo, ' % target), text)
            check('%s: one line writes the whole-mask variable' % text, len(lines) == 1 and lines[0].startswith(target + ' = '), lines)
            got = [value(lines[0], target, scc) for scc in (1, 0)]
            check('%s: lane bit of the selected word for scc 1 and 0' % text, got == [w & 1 for w in words], got)
            if target == 'exec':
                check('%s clears exec_known' % text, s.exec_known is False)

    # Negative control: without the lane-bit reduction the vcc word would be the whole selected word.
    _, _, lines = one(cselect(VCC_LO, INLINE_MINUS_ONE, INLINE_ZERO))
    mutated = lines[0].replace(') & 1;', ');')
    check('control: the unreduced select gives 0xffffffff, not the lane bit',
          mutated != lines[0] and value(mutated, 'vcc', 1) == 0xffffffff and value(lines[0], 'vcc', 1) == 1, mutated)

    _, text, lines = one(cselect(5, INLINE_ZERO, INLINE_MINUS_ONE))
    check('SGPR destination unchanged', text == 's_cselect_b32 s5, 0, -1' and lines == ['s5 = scc ? 0x00000000u : 0xffffffffu;'], lines)
    _, text, lines = one(cselect(M0, INLINE_MINUS_ONE, INLINE_ZERO))
    check('m0 destination unchanged', lines == ['m0 = scc ? 0xffffffffu : 0x00000000u;'], (text, lines))
    try:
        _, text, lines = one(cselect(NULL, INLINE_MINUS_ONE, INLINE_ZERO))
    except G.Unsupported as error:
        check('null destination refused', 's_cselect into' in str(error), str(error))
    except Exception as error:
        check('null destination refused', False, '%s: %s' % (type(error).__name__, error))
    else:
        check('null destination refused', False, lines)

    if failures:
        print('FAILED: %d check(s)' % failures)
        return 1
    print('cselect mask: vcc_lo and exec_lo are the lane mask; SGPR and m0 unchanged; null refused')
    return 0


if __name__ == '__main__':
    sys.exit(main())
