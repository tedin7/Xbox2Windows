"""Pixel DPP quad reads after an EXEC window (tools/startup/gfx2hlsl.py, Stage.whole_quad_exec_sites).

  (no arguments)                          translation controls and refusals, pure Python
  draw <dxc> <offscreen_check> <warp|hardware>
                                          also draw the accepted program over a triangle that cuts 2 x 2 quads (so
                                          helper lanes exist) and compare every pixel with the guest's words; the
                                          dirty-source mutant must fail there and pass a full-screen draw

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
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, root_blob  # noqa: E402

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


# ---- draw -------------------------------------------------------------------------------------------------------------
# The accepted program is drawn as a triangle whose long edge cuts 2 x 2 quads, so helper lanes really exist: pixel (x, y)
# of the 16 x 16 target is covered when x + y <= 15 (the edge is the line X + Y = 16.25 in pixel units, which passes
# through no pixel centre). A pixel outside keeps the clear value -1.0 in all four channels.
WIDTH = HEIGHT = 16
CLEAR = 0xbf800000
PARTIAL_VS = """\
float4 main(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 1.03125 : -1.0, id == 2 ? -1.03125 : 1.0, 0.0, 1.0);
}
"""
WINDOW_V3 = 1.0  # v3 written in the window; a guest helper lane never writes it and keeps its initial 0


def covered(x, y):
    return x + y <= 15


def reference(read_v3=False):
    """The guest's words: v2 = 16 y + x + 0.5 on every lane of a quad (helper lanes too); the read takes the x ^ 1
    neighbour's v2 (or, for the mutant, its v3: 1.0 on a covered neighbour, 0 on a helper neighbour)."""
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if not covered(x, y):
                out.extend([CLEAR] * 4)
                continue
            own = 16 * y + x + 0.5
            if read_v3:
                neighbour = WINDOW_V3 if covered(x ^ 1, y) else 0.0
            else:
                neighbour = 16 * y + (x ^ 1) + 0.5
            out.extend(struct.unpack('<I', struct.pack('<f', neighbour + own)) * 4)
    return out


def check_draw(dxc, renderer, adapter):
    from startup_image_3d_check import compile_hlsl  # noqa: E402
    text, reason = translate(base())
    if text is None:
        check('the accepted program translates for the draw', False, str(reason))
        return
    # The read a translator that accepted a dirty source would emit: the same program reading v3 instead of v2.
    dirty = text.replace('QuadReadAcrossX(v2)', 'QuadReadAcrossX(v3)')
    check('the dirty-source mutant changes the HLSL', dirty != text)
    check('some covered pixel has a helper neighbour (the triangle cuts quads)',
          any(covered(x, y) and not covered(x ^ 1, y) for y in range(HEIGHT) for x in range(WIDTH)))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        partial = compile_hlsl(dxc, 'dxc', PARTIAL_VS, 'vs_partial', 'vs_6_0', directory)
        full = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs_full', 'vs_6_0', directory)
        ps = compile_hlsl(dxc, 'dxc', text, 'ps_clean', 'ps_6_0', directory)
        bad = compile_hlsl(dxc, 'dxc', dirty, 'ps_dirty', 'ps_6_0', directory)
        clean_ref = words_file(directory, 'clean', reference())
        dirty_ref = words_file(directory, 'dirty', reference(read_v3=True))
        run = draw(renderer, adapter, partial, ps, clean_ref)
        print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                              if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
        check('%s: the accepted read matches the guest on all %d pixels, helper neighbours included'
              % (adapter, WIDTH * HEIGHT), run.returncode == 0 and 'pixels matched' in run.stdout,
              run.stdout + run.stderr)
        run = draw(renderer, adapter, partial, bad, dirty_ref)
        lines = [ln for ln in run.stderr.splitlines() if ln.startswith('pixel mismatch')]
        check('%s: the dirty-source mutant differs from the guest where a neighbour is a helper lane: %s'
              % (adapter, ' / '.join(lines)), run.returncode != 0 and bool(lines), run.stdout + run.stderr)
        # Power control: with every pixel covered there is no helper lane and the same mutant agrees with the guest,
        # so only the cut quads above detect it.
        full_ref = words_file(directory, 'full', [w for y in range(HEIGHT) for x in range(WIDTH) for w in
                                                  struct.unpack('<I', struct.pack('<f', WINDOW_V3 + 16 * y + x + 0.5))
                                                  * 4])
        run = draw(renderer, adapter, full, bad, full_ref)
        check('%s: without helper lanes the mutant is not detected (power control)' % adapter,
              run.returncode == 0 and 'pixels matched' in run.stdout, run.stdout + run.stderr)


def words_file(directory, name, values):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(values), *values))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def controls():
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


def main():
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'draw' and len(sys.argv) >= 5:
        controls()
        check_draw(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
    if failures:
        print('FAIL')
        for label in failures:
            print('  ' + label)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
