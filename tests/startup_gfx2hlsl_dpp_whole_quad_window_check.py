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
MIN_SWAP_V2 = 'fa040826' '02b100ff'   # v_min_u32_dpp v4, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
MOV_BROADCAST_V2 = 'fa02087e' '02aa00ff'  # v_mov_b32_dpp v4, v2 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
COPY_V3_TO_V2 = '0303047e'   # v_mov_b32_e32 v2, v3
COPY_V2_TO_V3 = '0203067e'   # v_mov_b32_e32 v3, v2
VCC_FROM_V3 = '8806887d' '02030402'  # v_cmp_gt_u32_e32 vcc_lo, 8, v3 ; v_cndmask_b32_e32 v2, v2, v1, vcc_lo
CLOBBER_SAVED = '800384be'   # s_mov_b32 s4, 0
AND_VCC_INTO_S6 = '7e6a8687'  # s_and_b64 s[6:7], exec, vcc (before s_wqm)
COMPARE_INTO_SAVED = '0400c4d4' '88020200'  # v_cmp_gt_u32_e64 s[4:5], 8, v1
SWAP_V2_V3 = '03cb047e'      # v_swap_b32 v2, v3
SELECT_BY_EXACT = '020001d5' '01051a00'  # v_cndmask_b32_e64 v2, v1, v2, s[6:7]
SELECT_BY_WHOLE = '020001d5' '01051200'  # v_cndmask_b32_e64 v2, v1, v2, s[4:5]
# A uniform scalar branch over the window (never taken: s0 is 0). The read then follows a join of a path that kept the
# whole-quad mask and a path through the window, so EXEC there is not symbolically known and the CFG proof decides.
UNIFORM_TEST = '800380be' '008106bf'  # s_mov_b32 s0, 0 ; s_cmp_eq_u32 s0, 1


# Narrowing instead of a window: a v_cmpx (a kill test) on a clean compare keeps EXEC lane-exact, so a register written
# while narrowed is still the guest's on every lane after the restore. Narrowing with the exact entry mask is a window.
COPY_V2_TO_V3_WHOLE = '0203067e'              # v_mov_b32_e32 v3, v2 (whole quad)
KILL_TEST = '8802a87d'                        # v_cmpx_gt_u32_e32 8, v1
AND_EXACT = '7e06fe87'                        # s_and_b64 exec, exec, s[6:7]
NARROWED_WRITE = '0103067e'                   # v_mov_b32_e32 v3, v1 (narrowed)


def narrowed(narrow=KILL_TEST):
    body = COPY_V2_TO_V3_WHOLE + narrow + NARROWED_WRITE + RESTORE
    return program(SAVE_EXACT, WQM, SAVE_WHOLE, POSITION, UNIFORM_TEST, skip(body), body, READ_V3, EXPORT)


# A v_cndmask_b32 quad broadcast: v4 = VCC ? v1 : (v2 of quad lane 1), with VCC set on even x. Only src0 is exchanged.
EVEN_X = '81000a36' '810a887d'               # v_and_b32_e32 v5, 1, v0 ; v_cmp_gt_u32_e32 vcc_lo, 1, v5
SELECT_BROADCAST = 'fa020802' '025500ff'     # v_cndmask_b32_dpp v4, v2, v1, vcc_lo quad_perm:[1,1,1,1]
SELECT_PROGRAM = (SAVE_EXACT + WQM + SAVE_WHOLE + POSITION + EVEN_X + SELECT_BROADCAST
                  + '0f1800f8' '04040404' '000081bf')


def select_reference():
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            value = float(16 * y + x) if x % 2 == 0 else 16 * (y & ~1) + (x | 1) + 0.5
            out.extend(struct.unpack('<I', struct.pack('<f', value)) * 4)
    return out


def skip(words):
    return '%02x0085bf' % (len(words) // 8)  # s_cbranch_scc1 over that many dwords

failures = []


def program(*parts):
    return bytes.fromhex(''.join(parts))


def base(read=READ_V2, after_restore='', before_restore='', restore=RESTORE, branch=True):
    window = EXACT + WINDOW_WRITE + before_restore + restore
    guard = UNIFORM_TEST + skip(window) if branch else ''
    return program(SAVE_EXACT, WQM, SAVE_WHOLE, POSITION, guard, window, after_restore, read, EXPORT)


def pipeline(code):
    try:
        pipe = G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=PACKED)
        pipe.hlsl_ps()
        return pipe, None
    except G.Unsupported as reason:
        return None, str(reason)


def translate(code, helper=True):
    """The pixel HLSL, or None and the refusal. helper=False disables the exact-entry-EXEC retry (Stage.retry_helper)."""
    saved = G.Stage.retry_helper
    if not helper:
        G.Stage.retry_helper = lambda self, reason: False
    try:
        pipe, reason = pipeline(code)
        return (pipe.hlsl_ps() if pipe else None), reason
    finally:
        G.Stage.retry_helper = saved


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
ONE = 0x3f800000
# Hand-written, not translated: each lane's IsHelperLane and its X and Y neighbours' (SM 6.6).
HELPER_PROBE_PS = """\
float4 main(float4 pos : SV_Position) : SV_Target0
{
    uint own = IsHelperLane() ? 1u : 0u;
    return float4(own, QuadReadAcrossX(own), QuadReadAcrossY(own), 0);
}
"""
skips = []


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
        # The helper retry: the read of v3, written in the exact-entry window, compiled with IsHelperLane (ps_6_6). A guest
        # helper lane never writes v3 there, and with the exact entry mask the host's helper thread does not either.
        helper_pipe, reason = pipeline(base(read=READ_V3))
        # Capability: the adapter's IsHelperLane must agree with coverage, read on each lane and across the quad.
        probe = compile_hlsl(dxc, 'dxc', HELPER_PROBE_PS, 'ps_helper_probe', 'ps_6_6', directory)
        probe_ref = words_file(directory, 'helper_probe', [
            w for y in range(HEIGHT) for x in range(WIDTH) for w in (
                [0, ONE if not covered(x ^ 1, y) else 0, ONE if not covered(x, y ^ 1) else 0, 0] if covered(x, y)
                else [CLEAR] * 4)])
        run = draw(renderer, adapter, partial, probe, probe_ref)
        if run.returncode != 0:
            lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
            print('%s: SKIP the exact-entry-mask draw: the adapter\'s IsHelperLane disagrees with coverage (%s)'
                  % (adapter, ' / '.join(lines)))
            skips.append(adapter)
            helper_pipe = False
        if helper_pipe is False:
            pass
        elif helper_pipe is None:
            check('the helper retry translates the window read for the draw', False, str(reason))
        else:
            exact = helper_pipe.hlsl_ps()
            hook = 'IsHelperLane() ? 0u : 1u'
            check('the helper-retry HLSL holds the exact entry mask once', exact.count(hook) == 1, exact)
            ps = compile_hlsl(dxc, 'dxc', exact, 'ps_exact', 'ps_6_6', directory)
            run = draw(renderer, adapter, partial, ps, dirty_ref)
            check('%s: with the exact entry mask the window read matches the guest on the cut quads' % adapter,
                  run.returncode == 0 and 'pixels matched' in run.stdout, run.stdout + run.stderr)
            # Mutant: the hook without IsHelperLane (every thread's copy 1, as before the retry).
            hookless = compile_hlsl(dxc, 'dxc', exact.replace(hook, '1u'), 'ps_hookless', 'ps_6_6', directory)
            run = draw(renderer, adapter, partial, hookless, dirty_ref)
            check('%s: the hook without IsHelperLane differs from the guest where a neighbour is a helper lane'
                  % adapter, run.returncode != 0 and any(ln.startswith('pixel mismatch') for ln in run.stderr.splitlines()),
                  run.stdout + run.stderr)
            run = draw(renderer, adapter, full, hookless, full_ref)
            check('%s: without helper lanes the hookless mutant is not detected (power control)' % adapter,
                  run.returncode == 0 and 'pixels matched' in run.stdout, run.stdout + run.stderr)
        # The other forms the proof now admits after a window, over the cut quads: an integer quad swap and a broadcast.
        def bits(value):
            return struct.unpack('<I', struct.pack('<f', value))[0]

        def v2(x, y):
            return bits(16 * y + x + 0.5)

        def v3_narrowed(x, y):
            # v3 = v2 on the whole quad, then v1 where the kill test 8 > bits(v1) holds (only x + 16 y == 0).
            return bits(16 * y + x) if bits(16 * y + x) < 8 else v2(x, y)

        for form, code, value, own_text, quad_text in (
                ('v_min_u32 swap', base(read=MIN_SWAP_V2), lambda x, y: min(v2(x ^ 1, y), v2(x, y)),
                 'QuadReadAcrossX(v2)', 'QuadReadAcrossX(v2)'),
                ('v_mov_b32 broadcast', base(read=MOV_BROADCAST_V2), lambda x, y: v2(x & ~1, y | 1),
                 'QuadReadLaneAt(v2, 2u)', 'QuadReadLaneAt(v2, 2u)'),
                ('narrowed write', narrowed(), lambda x, y: bits(struct.unpack('<f', struct.pack('<I', v3_narrowed(x ^ 1, y)))[0]
                                                                 + 16 * y + x + 0.5),
                 'QuadReadAcrossX(v3)', 'QuadReadAcrossX(v3)')):
            text, reason = translate(code)
            if text is None:
                check('%s after the window translates for the draw' % form, False, str(reason))
                continue
            ref = words_file(directory, form.replace(' ', '_'),
                             [w for y in range(HEIGHT) for x in range(WIDTH)
                              for w in ([value(x, y)] * 4 if covered(x, y) else [CLEAR] * 4)])
            ps = compile_hlsl(dxc, 'dxc', text, 'ps_' + form.replace(' ', '_'), 'ps_6_0', directory)
            run = draw(renderer, adapter, partial, ps, ref)
            check('%s: %s after the window matches the guest on all pixels of the cut quads' % (adapter, form),
                  run.returncode == 0 and 'pixels matched' in run.stdout, run.stdout + run.stderr)
            check('%s own-lane mutant changes the HLSL' % form, quad_text in text)
            own = compile_hlsl(dxc, 'dxc', text.replace(own_text, own_text[own_text.index('(') + 1:own_text.index(',') if ',' in own_text else -1], 1), 'ps_own_' + form.replace(' ', '_'),
                               'ps_6_0', directory)
            run = draw(renderer, adapter, partial, own, ref)
            check('%s: %s reading its own lane fails' % (adapter, form),
                  run.returncode != 0 and any(ln.startswith('pixel mismatch') for ln in run.stderr.splitlines()),
                  run.stdout + run.stderr)
        # The v_cndmask_b32 broadcast, full screen: every pixel against the CPU reference; reading src0 from the own
        # lane instead of quad lane 1 must fail.
        text, reason = translate(bytes.fromhex(SELECT_PROGRAM))
        if text is None:
            check('the v_cndmask_b32 broadcast translates for the draw', False, str(reason))
            return
        select_ref = words_file(directory, 'select', select_reference())
        ps = compile_hlsl(dxc, 'dxc', text, 'ps_select', 'ps_6_0', directory)
        run = draw(renderer, adapter, full, ps, select_ref)
        check('%s: v_cndmask_b32 broadcast matches on all %d pixels' % (adapter, WIDTH * HEIGHT),
              run.returncode == 0 and 'pixels matched' in run.stdout, run.stdout + run.stderr)
        own = compile_hlsl(dxc, 'dxc', text.replace('QuadReadLaneAt(v2, 1u)', 'v2'), 'ps_select_own', 'ps_6_0',
                           directory)
        run = draw(renderer, adapter, full, own, select_ref)
        check('%s: v_cndmask_b32 broadcast reading its own lane fails' % adapter,
              run.returncode != 0 and any(ln.startswith('pixel mismatch') for ln in run.stderr.splitlines()),
              run.stdout + run.stderr)


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
    text, reason = translate(bytes.fromhex(SELECT_PROGRAM))
    check('a v_cndmask_b32 quad broadcast translates', text is not None, str(reason))
    if text is not None:
        check('the broadcast reads quad lane 1 of v2 and the own lane of v1 and VCC',
              'QuadReadLaneAt(v2, 1u)' in text and text.replace('QuadReadLaneAt(v2, 1u)', 'v2') != text, text)
    text, reason = translate(narrowed())
    check('a register written under a v_cmpx narrowing (lane-exact) is read after the restore', text is not None,
          str(reason))
    # A register the exact-entry window wrote differs on helper lanes while the host starts every thread with EXEC 1:
    # the proof refuses these reads. The helper retry gives the copy of EXEC taken before s_wqm the guest's bit
    # (IsHelperLane, shader model 6.6), after which the same reads are lane-exact and translate.
    helper_reads = (
        ('a register written under a narrowing by the exact entry mask', narrowed(narrow=AND_EXACT)),
        ('the read takes a register written in the window', base(read=READ_V3)),
        ('a window register copied after the restore', base(after_restore=COPY_V3_TO_V2)),
        ('a VCC compare of a window register selects the source', base(after_restore=VCC_FROM_V3)),
        ('a swap moves a window register into the read register', base(after_restore=SWAP_V2_V3)),
        ('the exact entry mask selects the source', base(after_restore=SELECT_BY_EXACT)),
        ('straight-line: the read takes a register written in the window', base(read=READ_V3, branch=False)),
    )
    # Still wrong with the exact entry mask: EXEC at the read is not whole-quad, or the window mask is not a plain copy
    # of the entry EXEC (only that copy becomes IsHelperLane), or a later s_wqm (a no-op on the host) re-enables the
    # guest's helper lanes after the window.
    entry_and = program(AND_VCC_INTO_S6, WQM, SAVE_WHOLE, POSITION, UNIFORM_TEST,
                        skip(EXACT + WINDOW_WRITE + RESTORE), EXACT + WINDOW_WRITE + RESTORE, READ_V3, EXPORT)
    second_wqm = program(SAVE_EXACT, WQM, SAVE_WHOLE, POSITION, UNIFORM_TEST, skip(EXACT + WQM + WINDOW_WRITE),
                         EXACT + WQM + WINDOW_WRITE, READ_V3, EXPORT)
    refused = (
        ('the window mask is EXEC and VCC taken before s_wqm, not a plain copy', entry_and),
        ('a second s_wqm inside the window', second_wqm),
        ('the saved whole-quad mask is overwritten before the restore', base(before_restore=CLOBBER_SAVED)),
        ('the restore takes the exact entry mask', base(restore=EXACT)),
        ('a VALU compare overwrites the saved whole-quad mask', base(before_restore=COMPARE_INTO_SAVED)),
    )
    for label, code in helper_reads + refused:
        text, reason = translate(code, helper=False)
        check('refused without the helper retry: %s' % label,
              text is None and reason is not None and 'v_add_f32_dpp' in reason, str(reason))
    for label, code in helper_reads:
        pipe, reason = pipeline(code)
        text = pipe.hlsl_ps() if pipe else None
        check('the helper retry translates: %s' % label, text is not None, str(reason))
        if text is not None:
            check('  ... with the exact entry mask and shader model 6.6',
                  'IsHelperLane() ? 0u : 1u' in text and pipe.ps_profile() == 'ps_6_6', pipe.ps_profile())
    for label, code in refused:
        text, reason = translate(code)
        check('refused with the helper retry too: %s' % label,
              text is None and reason is not None and 'v_add_f32_dpp' in reason, str(reason))
    # A program the proof accepts without the retry is translated exactly as before: same text, shader model 6.0.
    for label, code in (('the clean read', base()), ('the narrowed write', narrowed()),
                        ('the straight-line read', base(branch=False)), ('the select broadcast',
                                                                        bytes.fromhex(SELECT_PROGRAM))):
        pipe, reason = pipeline(code)
        plain, _ = translate(code, helper=False)
        check('no helper retry for %s: identical HLSL, ps_6_0' % label,
              pipe is not None and pipe.hlsl_ps() == plain and pipe.ps_profile() == 'ps_6_0'
              and 'IsHelperLane' not in plain, str(reason))


def main():
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'draw' and len(sys.argv) >= 5:
        controls()
        check_draw(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
    if skips:
        print('SKIPPED (capability): the exact-entry-mask draw on %s' % ', '.join(skips))
    if failures:
        print('FAIL')
        for label in failures:
            print('  ' + label)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
