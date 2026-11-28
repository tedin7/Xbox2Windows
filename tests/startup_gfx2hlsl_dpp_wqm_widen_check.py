"""Pixel DPP quad reads after s_wqm_b64 exec, sN (tools/startup/gfx2hlsl.py, Stage.wqm_widen_site).

  (no arguments)                          translation controls and refusals, pure Python
  draw <dxc> <offscreen_check> <warp|hardware>
                                          also draw the accepted program over a triangle that cuts 2 x 2 quads (so
                                          helper lanes exist) and compare every pixel with a CPU reference; the mutants
                                          (no helper participation, a wrong quad lane) must fail there

The shape compilers emit for a pixel kill followed by derivatives (RDNA 2 ISA, S_WQM_B64: a destination bit is set when
any bit of its quad in the source is set):

  s_mov_b64 s[6:7], exec ; s_wqm_b64 exec, exec          save the exact entry mask, helper lanes on
  v5 = v0 & 1 ; vcc = (x is even)                        a per-lane kill test
  s_andn2_b64 s[6:7], s[6:7], vcc                        live lanes: covered pixels with an odd x
  s_wqm_b64 exec, s[6:7]                                 EXEC = every lane of every quad that has a live lane
  s_mov_b64 s[4:5], exec ; exact window ; s_mov_b64 exec, s[4:5]      the whole-quad mask is saved and restored
  v2 = x + 16 y + 0.5 ; v_add_f32_dpp v4, v2, v2 quad_perm:[2,3,0,1]    this pixel plus the one at y ^ 1
  s_mov_b64 exec, s[6:7] ; exp mrt0 v4, v4, v4, v4       only live lanes write

The vertical neighbour of a live pixel can be a helper lane (the triangle edge cuts the quad) or a killed lane. Both are
on in the widened EXEC, so the read gives their v2. The translation sets exec = QuadAny(mask bit) there, with the exact
entry mask (IsHelperLane) as the source. The controls change one thing each and must be refused: an export inside the
widened region, and the same program behind a branch.
"""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_dpp_whole_quad_window_check as W  # noqa: E402

failures = []
SAVE_EXACT, WQM, POSITION, EXACT = W.SAVE_EXACT, W.WQM, W.POSITION, W.EXACT
EVEN_X = W.EVEN_X                    # v5 = v0 & 1 ; vcc = (1 > v5): an even x
ANDN2_VCC = '066a868a'               # s_andn2_b64 s[6:7], s[6:7], vcc
WQM_MASK = '060afebe'                # s_wqm_b64 exec, s[6:7]
READ_Y = 'fa040806' '024e00ff'       # v_add_f32_dpp v4, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
EXPORT_ONLY = '0f1800f8' '04040404'  # exp mrt0 v4, v4, v4, v4 done vm
END = '000081bf'                     # s_endpgm
AND_VCC_EXEC = '7e6afe87'            # s_and_b64 exec, exec, vcc
WIDEN = 'exec = x2w_quad_any(s6) ? 1u : 0u;'
WIDTH = HEIGHT = W.WIDTH


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail[:2000]))
    if not ok:
        failures.append(label)


def code(inside='', before=''):
    window = W.program(EXACT, W.WINDOW_WRITE, W.RESTORE)
    return W.program(SAVE_EXACT, WQM, EVEN_X, before, ANDN2_VCC, WQM_MASK, inside, W.SAVE_WHOLE, window.hex(),
                     POSITION, READ_Y, EXACT, EXPORT_ONLY, END)


def proofs(program):
    """The verdicts wqm_widen_site_proof gives while translating `program`."""
    stage = next(m.Stage for m in sys.modules.values() if hasattr(getattr(m, 'Stage', None), 'wqm_widen_site_proof'))
    original, seen = stage.wqm_widen_site_proof, []

    def record(self, off):
        seen.append(original(self, off))
        return seen[-1]
    stage.wqm_widen_site_proof = record
    try:
        translate(program)
    finally:
        stage.wqm_widen_site_proof = original
    return seen


def narrow_effect():
    """s_and_b64 exec, exec, vcc keeps the helper lanes of the even pixels that the widening added: the scan of the
    widened region goes through that narrowing and sees the export behind it (it must not stop at the EXEC write)."""
    def body(*mid):
        window = W.program(EXACT, W.WINDOW_WRITE, W.RESTORE)
        return W.program(SAVE_EXACT, WQM, EVEN_X, ANDN2_VCC, WQM_MASK, W.SAVE_WHOLE, *mid, W.SAVE_WHOLE, window.hex(),
                         POSITION, READ_Y, EXACT, EXPORT_ONLY, END)
    check('a narrowing of the widened EXEC with no effect behind it keeps the widening',
          proofs(body(AND_VCC_EXEC, W.RESTORE)) == [True])
    check('an export behind a narrowing of the widened EXEC refuses the widening',
          proofs(body(AND_VCC_EXEC, EXPORT_ONLY, W.RESTORE)) == [False])
    check('an export after the exact mask is restored keeps the widening',
          proofs(body(EXACT, EXPORT_ONLY, W.RESTORE)) == [True])


def live(x, y):
    return W.covered(x, y) and x % 2 == 1


def reference():
    """Live pixels (covered, odd x) write v2 of this pixel plus v2 of the pixel at y ^ 1, whatever its coverage."""
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if live(x, y):
                value = (16 * y + x + 0.5) + (16 * (y ^ 1) + x + 0.5)
                out.extend(struct.unpack('<I', struct.pack('<f', value)) * 4)
            else:
                out.extend([W.CLEAR] * 4)
    return out


def translate(program):
    pipe, reason = W.pipeline(program)
    return (pipe.hlsl_ps() if pipe else None), (pipe.ps_profile() if pipe else None), reason


def controls():
    text, profile, reason = translate(code())
    check('a quad read after s_wqm_b64 exec, sN of a lane mask translates', text is not None, str(reason))
    if text is None:
        return
    check('EXEC is the quad-wide OR of the lane mask', text.count(WIDEN) == 1, text)
    check('the read is a vertical quad swap', 'QuadReadAcrossY(v2)' in text, text)
    check('the mask copy is the exact entry mask and the stage needs shader model 6.6',
          'IsHelperLane() ? 0u : 1u' in text and profile == 'ps_6_6', '%s %s' % (profile, text))
    text, _, reason = translate(code(inside=EXPORT_ONLY))
    check('an export inside the widened region is refused', text is None, str(reason))
    text, _, reason = translate(code(before=W.UNIFORM_TEST + W.skip(W.EVEN_X)))
    check('the widening behind a branch is refused', text is None, str(reason))
    narrow_effect()



def check_draw(dxc, renderer, adapter):
    from startup_image_3d_check import compile_hlsl  # noqa: E402
    text, profile, reason = translate(code())
    if text is None:
        check('the accepted program translates for the draw', False, str(reason))
        return
    mutants = {
        'a helper lane that stays off': text.replace(WIDEN, 'exec = IsHelperLane() ? 0u : (x2w_quad_any(s6) ? 1u : 0u);'),
        'no widening (each lane keeps its own bit)': text.replace(WIDEN, 'exec = s6 & 1u;'),
        'the horizontal neighbour': text.replace('QuadReadAcrossY(v2)', 'QuadReadAcrossX(v2)'),
        'its own lane': text.replace('QuadReadAcrossY(v2)', 'v2'),
    }
    check('every mutant changes the HLSL', all(m != text for m in mutants.values()))
    check('some live pixel has a helper neighbour (the triangle cuts quads)',
          any(live(x, y) and not W.covered(x, y ^ 1) for y in range(HEIGHT) for x in range(WIDTH)))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        partial = compile_hlsl(dxc, 'dxc', W.PARTIAL_VS, 'vs_partial', 'vs_6_0', directory)
        ref = W.words_file(directory, 'ref', reference())
        ps = compile_hlsl(dxc, 'dxc', text, 'ps_widen', profile, directory)
        run = W.draw(renderer, adapter, partial, ps, ref)
        print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                              if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
        check('%s: the widened read matches the guest on all %d pixels, helper neighbours included'
              % (adapter, WIDTH * HEIGHT), run.returncode == 0 and 'pixels matched' in run.stdout,
              run.stdout + run.stderr)
        for label, mutant in mutants.items():
            bad = compile_hlsl(dxc, 'dxc', mutant, 'ps_mutant', profile, directory)
            run = W.draw(renderer, adapter, partial, bad, ref)
            check('%s: the mutant with %s differs from the guest' % (adapter, label),
                  run.returncode != 0 and any(ln.startswith('pixel mismatch') for ln in run.stderr.splitlines()),
                  run.stdout + run.stderr)


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
