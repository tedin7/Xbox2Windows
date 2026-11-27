"""Pixel DPP quad reads after an EXEC window (tools/startup/gfx2hlsl.py, Stage.whole_quad_exec_sites), pure Python.

A pixel program saves its exact entry mask, turns whole-quad EXEC on and saves that mask too. It then runs a short
window with the exact mask (the shape compilers emit around a sample that must not run on helper lanes) and restores
the saved whole-quad mask:

  s_mov_b64 s[6:7], exec ; s_wqm_b64 exec, exec ; s_mov_b64 s[4:5], exec
  v1 = x + 16 y (float) ; v2 = v1 + 0.5 ; s0 = 0 ; if (s0 == 1) skip the window   (uniform, never taken)
  s_mov_b64 exec, s[6:7] ; v_mov_b32 v3, 1.0 ; s_mov_b64 exec, s[4:5]
  v_add_f32_dpp v4, v2, v2 quad_perm:[1,0,3,2]          the pixel at x ^ 1 plus this pixel
  exp mrt0 v4, v4, v4, v4

The host runs every lane of a quad as a thread; inside the window the guest leaves its helper lanes off, while the
host thread of a helper lane still writes (the host starts every thread with EXEC 1). A quad read after the restore therefore gives the guest's value only
for a register that no window wrote and that was not computed from one. The read of v2 is accepted. The controls
change one instruction each and must be refused: the read takes v3 (written in the window), copies v3 into v2 after the
restore, derives v2 through a VCC compare of v3, overwrites the saved whole-quad mask before the restore, or restores
the exact mask instead. Rewriting v3 from v2 after the restore makes it clean again and is accepted.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import VS_WORDS, root_blob  # noqa: E402

PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0

SAVE_EXACT = '7e0486be'      # s_mov_b64 s[6:7], exec
WQM = '7e0afebe'             # s_wqm_b64 exec, exec
SAVE_WHOLE = '7e0484be'      # s_mov_b64 s[4:5], exec
POSITION = '8f000236' '9000042c' '84040434' '01050238' '010d027e' 'f0020406'  # v1 = x + 16 y, v2 = v1 + 0.5
EXACT = '0604febe'           # s_mov_b64 exec, s[6:7]
WINDOW_WRITE = 'f202067e'    # v_mov_b32_e32 v3, 1.0
RESTORE = '0404febe'         # s_mov_b64 exec, s[4:5]
READ_V2 = 'fa040806' '02b100ff'  # v_add_f32_dpp v4, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
READ_V3 = 'fa040806' '03b100ff'  # v_add_f32_dpp v4, v3, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
EXPORT = '0f1800f8' '04040404' '000081bf'  # exp mrt0 v4, v4, v4, v4 done vm ; s_endpgm
COPY_V3_TO_V2 = '0303047e'   # v_mov_b32_e32 v2, v3
COPY_V2_TO_V3 = '0203067e'   # v_mov_b32_e32 v3, v2
VCC_FROM_V3 = '8806887d' '02030402'  # v_cmp_gt_u32_e32 vcc_lo, 8, v3 ; v_cndmask_b32_e32 v2, v2, v1, vcc_lo
CLOBBER_SAVED = '800384be'   # s_mov_b32 s4, 0
COMPARE_INTO_SAVED = '0400c4d4' '88020200'  # v_cmp_gt_u32_e64 s[4:5], 8, v1
SWAP_V2_V3 = '03cb047e'      # v_swap_b32 v2, v3
SELECT_BY_EXACT = '020001d5' '01051a00'  # v_cndmask_b32_e64 v2, v1, v2, s[6:7]
SELECT_BY_WHOLE = '020001d5' '01051200'  # v_cndmask_b32_e64 v2, v1, v2, s[4:5]
# A uniform scalar branch over the window (never taken: s0 is 0). The read then follows a join of a path that kept the
# whole-quad mask and a path through the window, so EXEC there is not symbolically known and the CFG proof decides.
UNIFORM_TEST = '800380be' '008106bf'  # s_mov_b32 s0, 0 ; s_cmp_eq_u32 s0, 1


def skip(words):
    return '%02x0085bf' % (len(words) // 8)  # s_cbranch_scc1 over that many dwords

failures = []


def program(*parts):
    return bytes.fromhex(''.join(parts))


def base(read=READ_V2, after_restore='', before_restore='', restore=RESTORE, branch=True):
    window = EXACT + WINDOW_WRITE + before_restore + restore
    guard = UNIFORM_TEST + skip(window) if branch else ''
    return program(SAVE_EXACT, WQM, SAVE_WHOLE, POSITION, guard, window, after_restore, read, EXPORT)


def translate(code):
    try:
        return G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail[:2000]))
    if not ok:
        failures.append(label)


def main():
    text, reason = translate(base())
    check('a quad read of a register the window did not write translates', text is not None, str(reason))
    if text is not None:
        check('the read is a quad swap of v2', 'QuadReadAcrossX(v2)' in text, text)
    text, reason = translate(base(after_restore=COPY_V2_TO_V3, read=READ_V3))
    check('a window register rewritten after the restore from a clean one translates', text is not None, str(reason))
    if text is not None:
        check('the read is a quad swap of v3', 'QuadReadAcrossX(v3)' in text, text)
    text, reason = translate(base(after_restore=SELECT_BY_WHOLE))
    check('a select by the saved whole-quad mask translates', text is not None, str(reason))
    text, reason = translate(base(branch=False))
    check('straight-line window and restore (EXEC known, s_wqm flag lost) translates', text is not None, str(reason))
    if text is not None:
        check('straight-line: the read is a quad swap of v2', 'v256 = QuadReadAcrossX(v2);' in text, text)
    refused = (
        ('the read takes a register written in the window', base(read=READ_V3)),
        ('a window register copied after the restore', base(after_restore=COPY_V3_TO_V2)),
        ('a VCC compare of a window register selects the source', base(after_restore=VCC_FROM_V3)),
        ('the saved whole-quad mask is overwritten before the restore', base(before_restore=CLOBBER_SAVED)),
        ('the restore takes the exact entry mask', base(restore=EXACT)),
        ('a VALU compare overwrites the saved whole-quad mask', base(before_restore=COMPARE_INTO_SAVED)),
        ('a swap moves a window register into the read register', base(after_restore=SWAP_V2_V3)),
        ('the exact entry mask selects the source (helper lanes differ)', base(after_restore=SELECT_BY_EXACT)),
        ('straight-line: the read takes a register written in the window', base(read=READ_V3, branch=False)),
    )
    for label, code in refused:
        text, reason = translate(code)
        check('refused: %s' % label, text is None and reason is not None and 'v_add_f32_dpp' in reason, str(reason))
    if failures:
        print('FAIL')
        for label in failures:
            print('  ' + label)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
