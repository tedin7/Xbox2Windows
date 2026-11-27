"""DPP16 quad reads in a pixel program with branches and loops (tools/startup/gfx2hlsl.py), read back from a real
D3D12 draw.

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     assemble the fixture sources with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel stage with DXC ps_6_0 (quad reads need shader
                                       model 6.0; the optional fxc argument is accepted and ignored), draw 16 x 16
                                       pixels and compare every pixel with the CPU reference below

The translator runs each guest lane as its own HLSL thread and turns branches into a pc dispatch loop. A quad read
(QuadReadAcross*, QuadReadLaneAt) gives the neighbour's word only when the four threads of the quad reach it together
with the words the four guest lanes hold. The guest wave runs as one, so this holds when every branch on a path to
the read is wave-uniform and EXEC is still the whole-quad mask of the entry s_wqm exec, exec. The translator proves
that over the whole program after translation (Stage.quad_flow_refusals) and refuses the read otherwise: a vccz/vccnz
loop, an SCC taken from a lane mask, an EXEC write that can reach the read (also around a loop back edge), and an
s_wqm that does not run before every branch. No wave size is assumed: the quad operations need only the 2 x 2 quad.

The draw fixture LOOP is the shape of a uniform loop over a scalar counter with an if inside it:
  s_mov_b64 s[4:5], exec ; s_wqm_b64 exec, exec      entry: save the exact mask, whole quads on
  v1 = (x | y << 16) * 0x9b5d ; v2 = v3 = v5 = 0 ; s0 = 0
  loop:                                              k = s0 = 0 .. 4
    v2 += quad_perm:[1,0,3,2] v1                     the pixel at x ^ 1
    v3 += quad_perm:[2,2,2,2] v1                     lane 2 of the quad, pixel (x & ~1, y | 1)
    if (s0 != 2) { v4 = quad_perm:[0,0,2,2] v1 ; v5 ^= v4 }   pixel (x & ~1, y), skipped on pass 2
    v1 += 0x101 ; s0 += 1 ; loop while s0 < 5
  s_and_b64 exec, exec, s[4:5] ; exp mrt0 v2, v3, v5, v1
So the neighbour word read on pass k is w(neighbour) + k * 0x101 and the reference below sums or XORs it with plain
integers. The EXEC restore after the loop does not reach any read and is allowed. Mutations of the translated text
(each of the three reads taking the wrong lane) must fail the same draw.

The SUBREV fixtures branch on the zero root constant supplied by the renderer.
The untaken arm narrows EXEC and exits. The taken arm reads a whole quad and
subtracts the selected neighbour from its own value. Comparing all four output
words catches both a wrong neighbour and a reversed subtraction. Each own-lane
mutant must fail GPU readback. A narrowed incoming edge and a VCC branch are
translation refusal controls.
"""
import re
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
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
WIDTH = HEIGHT = 16
MULTIPLIER = 0x9b5d
STEP, PASSES, SKIPPED = 0x101, 5, 2
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
failures = []

# name -> (source with numeric branch offsets, words assembled by llvm-mc 19 for gfx1013)
FIXTURES = {
    'LOOP': ('''s_mov_b64 s[4:5], exec
s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v5, 0
s_mov_b32 s0, 0
v_add_nc_u32_dpp v2, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_nc_u32_dpp v3, v1, v3 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
s_cmp_eq_u32 s0, 2
s_cbranch_scc1 3
v_mov_b32_dpp v4, v1 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_e32 v5, v4, v5
v_add_nc_u32_e32 v1, 0x101, v1
s_add_u32 s0, s0, 1
s_cmp_lt_u32 s0, 5
s_cbranch_scc1 -14
s_and_b64 exec, exec, s[4:5]
exp mrt0 v2, v3, v5, v1 done vm
s_endpgm
''', '7e0484be7e0afebeff0002165d9b00008002047e8002067e80020a7e800380befa04044a01b100fffa06064a01aa00ff008206bf030085bf'
         'fa02087e01a000ff040b0a3aff02024a010100000081008000850abff2ff85bf7e04fe870f1800f802030501000081bf'),
    # The fall-through clears and restores EXEC, then joins a path that kept the WQM mask.
    'ADD_AFTER_EXEC': ('''s_wqm_b64 exec, exec
v_and_b32_e32 v1, 15, v0
v_lshrrev_b32_e32 v2, 16, v0
v_lshlrev_b32_e32 v2, 4, v2
v_or_b32_e32 v1, v1, v2
v_cvt_f32_u32_e32 v1, v1
v_add_f32_e32 v2, 0.5, v1
s_mov_b32 s0, 0
s_cmp_eq_u32 s0, 1
s_cbranch_scc1 5
s_and_saveexec_b64 s[4:5], 0
v_add_f32_dpp v1, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_mov_b64 exec, s[4:5]
s_nop 0
v_add_f32_dpp v3, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, -v1, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf
exp mrt0 v3, v4, v5, v2 done vm
s_endpgm
''', '7e0afebe8f0002369000042c8404043401050238010d027ef0020406800380be008106bf050085bf802484be'
         'fa04020602b100ff0404febe000080bffa04060601b100fffa040806014e10fffa040a06011b00ff0f1800f803040502000081bf'),
    'VCC_LOOP': ('''s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0
v_add_nc_u32_dpp v2, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_nc_u32_e32 v1, 3, v1
v_cmp_gt_u32_e32 vcc_lo, 0x100, v1
s_cbranch_vccnz -6
exp mrt0 v2, v1, v0, v0 done vm
s_endpgm
''', '7e0afebeff0002165d9b00008002047efa04044a01b100ff8302024aff02887d00010000faff87bf0f1800f802010000000081bf'),
    'MASK_SCC_LOOP': ('''s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0
v_add_nc_u32_dpp v2, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_nc_u32_e32 v1, 3, v1
v_cmp_gt_u32_e32 vcc_lo, 0x100, v1
s_and_b64 s[2:3], vcc, exec
s_cbranch_scc1 -7
exp mrt0 v2, v1, v0, v0 done vm
s_endpgm
''', '7e0afebeff0002165d9b00008002047efa04044a01b100ff8302024aff02887d000100006a7e8287f9ff85bf0f1800f802010000000081bf'),
    'EXEC_IN_LOOP': ('''s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0
s_mov_b32 s0, 0
v_add_nc_u32_dpp v2, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_cmp_gt_u32_e32 vcc_lo, 0x100, v1
s_and_saveexec_b64 s[4:5], vcc
v_add_nc_u32_e32 v1, 3, v1
s_or_b64 exec, exec, s[4:5]
s_add_u32 s0, s0, 1
s_cmp_lt_u32 s0, 4
s_cbranch_scc1 -10
exp mrt0 v2, v1, v0, v0 done vm
s_endpgm
''', '7e0afebeff0002165d9b00008002047e800380befa04044a01b100ffff02887d000100006a2484be8302024a7e04fe880081008000840abf'
     'f6ff85bf0f1800f802010000000081bf'),
    'WQM_AFTER_BRANCH': ('''s_mov_b32 s0, 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc1 1
v_mov_b32_e32 v0, 1
s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_dpp v2, v1 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
exp mrt0 v2, v1, v0, v0 done vm
s_endpgm
''', '800380be008006bf010085bf8102007e7e0afebeff0002165d9b0000fa02047e010000ff0f1800f802010000000081bf'),
    # A quad min inside an if: the read meets the EXEC the saveexec narrowed (refused; the reason names the write).
    'MIN_IN_IF': ('''s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0
v_cmp_gt_u32_e32 vcc_lo, 0x100, v1
s_and_saveexec_b64 s[4:5], vcc
v_min_u32_dpp v2, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_or_b64 exec, exec, s[4:5]
exp mrt0 v2, v1, v0, v0 done vm
s_endpgm
''', '7e0afebeff0002165d9b00008002047eff02887d000100006a2484befa02042601b100ff7e04fe880f1800f802010000000081bf'),
}
REFUSALS = {
    'MIN_IN_IF': 'v_min_u32_dpp requires a straight-line pixel shader with entry EXEC: EXEC written at +0x18 by s_and_saveexec_b64',
    'VCC_LOOP': 's_cbranch_vccnz at +0x24 is not proven wave-uniform',
    'MASK_SCC_LOOP': 's_cbranch_scc1 at +0x28 is not proven wave-uniform',
    'EXEC_IN_LOOP': 'the EXEC write at +0x24 reaches it',
    'WQM_AFTER_BRANCH': 'the s_wqm exec, exec does not run before every branch',
}


# One arm narrows EXEC and exits. Only the whole-quad arm reaches the subtraction.
# The shared encoding differs only in the DPP16 control byte.
SUBREV_SOURCE = """s_wqm_b64 exec, exec
v_cvt_f32_u32_e32 v1, v0
s_cmp_eq_u32 s9, 0
s_cbranch_scc1 4
v_cmp_gt_u32_e32 vcc_lo, 8, v0
s_and_saveexec_b64 s[4:5], vcc
v_mov_b32_e32 v1, 0
s_endpgm
v_subrev_f32_dpp v2, v1, v1 quad_perm:[{perm}] row_mask:0xf bank_mask:0xf bound_ctrl:1
exp mrt0 v2, v1, v1, v1 done vm
s_endpgm
"""
SUBREV = {
    'pair': ('0,0,2,2', 0xa0, lambda x, y: (x & ~1, y)),
    'broadcast': ('2,2,2,2', 0xaa, lambda x, y: (x & ~1, y | 1)),
    'swap_x': ('1,0,3,2', 0xb1, lambda x, y: (x ^ 1, y)),
    'swap_y': ('2,3,0,1', 0x4e, lambda x, y: (x, y ^ 1)),
}


def subrev_words(control, alternate_exit='000081bf'):
    return bytes.fromhex('7e0afebe000d027e098006bf040085bf8800887d6a2484be8002027e'
                         f'{alternate_exit}fa02040a01{control:02x}08ff0f1800f802010101000081bf')


def translate_words(code):
    try:
        return G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


def subrev_reference(neighbour):
    def packed(x, y):
        return x | (y << 16)

    values = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            local = packed(x, y)
            values.extend((local - packed(*neighbour(x, y)), local, local, local))
    return struct.unpack('<%dI' % len(values), struct.pack('<%df' % len(values), *values))


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail[:2000]))
    if not ok:
        failures.append(label)


def words(name):
    return bytes.fromhex(FIXTURES[name][1])


def translate_code(code):
    try:
        return G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)

def translate(name):
    return translate_words(words(name))


def translate(name):
    return translate_code(words(name))


def add_reference(local_read=None):
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            local = 16 * y + x + 0.5
            for channel, (dx, dy, sign) in enumerate(((1, 0, 1), (0, 1, -1), (1, 1, 1))):
                neighbour = 16 * (y ^ dy) + (x ^ dx) if channel != local_read else 16 * y + x
                out.append(local + sign * neighbour)
            out.append(local)
    return struct.unpack('<%dI' % len(out), struct.pack('<%df' % len(out), *out))


ADD_INTRINSICS = ('QuadReadAcrossX', 'QuadReadAcrossY', 'QuadReadAcrossDiagonal')


def add_bypass():
    # Take the other CFG edge by comparing s0 with zero.
    code = words('ADD_AFTER_EXEC')
    return bytes.fromhex(replace_once(code.hex(), '008106bf', '008006bf'))


def check_add_controls():
    for edge, code in (('restore', words('ADD_AFTER_EXEC')), ('bypass', add_bypass())):
        text, reason = translate_code(code)
        check('DPP add with the %s edge translates' % edge, text is not None, str(reason))
        if text is not None:
            for index, intrinsic in enumerate(ADD_INTRINSICS):
                check('%s is outside the EXEC guard' % intrinsic,
                      ('v256 = %s(v1);\n        if (exec & 1) v%d =' % (intrinsic, index + 3)) in text)
            check('the add in the disabled quad leaves v1 unchanged',
                  'v256 = QuadReadAcrossX(v2);\n        if (exec & 1) v1 =' in text)
    reference = add_reference()
    for index, intrinsic in enumerate(ADD_INTRINSICS):
        check('%s CPU reference detects a lane-local read' % intrinsic, reference != add_reference(local_read=index))
    original = words('ADD_AFTER_EXEC').hex()
    controls = (
        # Restore a literal selecting only lane 0.
        ('0404febe', '8104febe', 'EXEC written at +0x34 by s_mov_b64'),
        # Overwrite the saved mask before restoring it.
        ('0404febe000080bf', '800384be0404febe', 'EXEC written at +0x38 by s_mov_b64'),
        # Replace the scalar compare that drives the branch with a vector compare.
        ('008106bf', '0000827d', 'is not proven wave-uniform'),
        ('01b100ff', '01b100ef', 'with row_mask:0xe'),
    )
    for old, new, reason in controls:
        mutated = bytes.fromhex(replace_once(original, old, new))
        text, actual = translate_code(mutated)
        check('DPP add control refused: %s' % reason, text is None and reason in str(actual), str(actual))


# ---- CPU reference ----------------------------------------------------------------------------------------------------
def word(x, y):
    return ((x | (y << 16)) * MULTIPLIER) & MASK


def image():
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            swap = sum(word(x ^ 1, y) + k * STEP for k in range(PASSES)) & MASK
            broadcast = sum(word(x & ~1, y | 1) + k * STEP for k in range(PASSES)) & MASK
            pair = 0
            for k in range(PASSES):
                if k != SKIPPED:
                    pair ^= (word(x & ~1, y) + k * STEP) & MASK
            out += [swap, broadcast, pair, (word(x, y) + PASSES * STEP) & MASK]
    return out


# ---- mutations of the translated text ---------------------------------------------------------------------------------
def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def swap_own(text):
    return replace_once(text, 'QuadReadAcrossX(v1)', 'v1')


def broadcast_lane1(text):
    return replace_once(text, 'QuadReadLaneAt(v1, 2u)', 'QuadReadLaneAt(v1, 1u)')


def pair_own(text):
    chain = re.compile(r'x2w_lane == 0u \? x2w_q\d : x2w_lane == 1u \? x2w_q\d : x2w_lane == 2u \? x2w_q\d : x2w_q\d;')
    assert len(chain.findall(text)) == 1
    return chain.sub('x2w_lane == 0u ? x2w_q0 : x2w_lane == 1u ? x2w_q1 : x2w_lane == 2u ? x2w_q2 : x2w_q3;', text)


MUTATIONS = {'swap reads its own lane': swap_own, 'broadcast reads lane 1': broadcast_lane1,
             'pair keeps every lane': pair_own}


SUBREV_MUTANTS = {
    'pair': pair_own,
    'broadcast': lambda text: replace_once(text, 'QuadReadLaneAt(v1, 2u)', 'v1'),
    'swap_x': swap_own,
    'swap_y': lambda text: replace_once(text, 'QuadReadAcrossY(v1)', 'v1'),
}


def subrev_controls():
    own = subrev_reference(lambda x, y: (x, y))
    for name, (_, control, neighbour) in SUBREV.items():
        code = subrev_words(control)
        text, reason = translate_words(code)
        check('SUBREV %s translates after an EXEC-writing exit arm' % name, text is not None, str(reason))
        if text is not None:
            check('SUBREV %s branch reads the root constant' % name, 'scc = ud1 == ' in text)
            check('SUBREV %s own-lane mutant changes HLSL' % name, SUBREV_MUTANTS[name](text) != text)
        check('SUBREV %s CPU reference rejects the own lane' % name, subrev_reference(neighbour) != own)
        # Replace the exit with a branch to the read. EXEC on that edge is narrowed.
        text, reason = translate_words(subrev_words(control, alternate_exit='000082bf'))
        check('SUBREV %s refuses a join with narrowed EXEC' % name, text is None and 'v_subrev_f32_dpp' in reason,
              str(reason))
        # Replace the SCC compare and branch with a lane comparison and a VCC branch.
        divergent = code[:8] + bytes.fromhex('8800887d040087bf') + code[16:]
        text, reason = translate_words(divergent)
        check('SUBREV %s refuses divergent control flow' % name,
              text is None and 'not proven wave-uniform' in reason, str(reason))


# ---- controls ---------------------------------------------------------------------------------------------------------
def controls():
    check_add_controls()
    text, reason = translate('LOOP')
    check('LOOP translates (uniform scalar loop and if, quad reads inside)', text is not None, str(reason))
    if text is not None:
        check('LOOP keeps the pc dispatch loop', '[loop] for' in text, text)
        check('LOOP reads the quad: X swap, lane-2 broadcast, four-lane pair read',
              'QuadReadAcrossX(v1)' in text and 'QuadReadLaneAt(v1, 2u)' in text and text.count('QuadReadLaneAt(v1, ') == 5,
              text)
        for how, mutate in MUTATIONS.items():
            check('LOOP %s mutation changes the text' % how, mutate(text) != text)
    for name, expected in REFUSALS.items():
        text, reason = translate(name)
        check('%s refused: %s' % (name, expected), text is None and reason is not None and expected in reason,
              str(reason))
    reference = image()
    own = [v for y in range(HEIGHT) for x in range(WIDTH)
           for v in (sum(word(x, y) + k * STEP for k in range(PASSES)) & MASK,) * 2 + (0, 0)]
    check('reference differs from a lane-local reading', reference[0::4] != own[0::4] and reference[1::4] != own[1::4])
    # A dead quad read behind an early s_endpgm in a branch-free program has no branch to prove: no crash.
    dead = bytes.fromhex('7e0afebeff0002165d9b00008002047e000081bffa04044a01b100fffa02087e01a000ff000081bf')
    try:
        G.Pipeline(VS_WORDS, dead, root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
        check('dead quad read after s_endpgm translates without a crash', True)
    except G.Unsupported as reason:
        check('dead quad read after s_endpgm translates without a crash', True, str(reason))
    except Exception as error:  # noqa: BLE001 - the regression is an unexpected exception type
        check('dead quad read after s_endpgm translates without a crash', False, repr(error))
    # A straight-line program with the same reads still translates as before.
    straight = bytes.fromhex('7e0afebeff0002165d9b0000fa02047e01b100ff0f1800f802010000000081bf')
    try:
        G.Pipeline(VS_WORDS, straight, root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
        check('straight-line quad swap still translates', True)
    except G.Unsupported as reason:
        check('straight-line quad swap still translates', False, str(reason))


# ---- draw -------------------------------------------------------------------------------------------------------------
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


def check_draw(dxc, renderer, adapter, fxc=None):
    text, reason = translate('LOOP')
    if text is None:
        check('LOOP translates for the draw', False, str(reason))
        return
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs_dxc', 'vs_6_0', directory)
        reference = words_file(directory, 'reference', image())
        ps = compile_hlsl(dxc, 'dxc', text, 'ps_loop', 'ps_6_0', directory)
        run = draw(renderer, adapter, vs, ps, reference)
        print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                              if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
        check('dxc %s LOOP: all %d pixels match the reference' % (adapter, WIDTH * HEIGHT),
              run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout, run.stdout + run.stderr)
        for how, mutate in MUTATIONS.items():
            bad = compile_hlsl(dxc, 'dxc', mutate(text), 'ps_' + how.replace(' ', '_'), 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, bad, reference)
            lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
            check('dxc %s LOOP %s mutation fails against the reference: %s' % (adapter, how, ' / '.join(lines)),
                  run.returncode != 0 and bool(lines) and lines[0].startswith('pixel mismatch'), run.stdout + run.stderr)



def check_add_draw(dxc, renderer, adapter):
    reference_words = add_reference()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs_dxc', 'vs_6_0', directory)
        reference = words_file(directory, 'add_reference', reference_words)
        for edge, code in (('restore', words('ADD_AFTER_EXEC')), ('bypass', add_bypass())):
            text, reason = translate_code(code)
            if text is None:
                check('DPP add translates for the draw', False, str(reason))
                continue
            ps = compile_hlsl(dxc, 'dxc', text, 'ps_add_' + edge, 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, ps, reference)
            print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                  if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
            check('dxc %s DPP add %s: all %d pixels match' % (adapter, edge, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            if edge == 'restore':
                mutants = [(intrinsic, intrinsic + '(v1)', 'v1') for intrinsic in ADD_INTRINSICS]
                mutants.append(('inactive_write', 'if (exec & 1) v1 = U(F(v256)', 'v1 = U(F(v256)'))
                for name, old, new in mutants:
                    mutant = replace_once(text, old, new)
                    ps = compile_hlsl(dxc, 'dxc', mutant, 'ps_add_' + name, 'ps_6_0', directory)
                    run = draw(renderer, adapter, vs, ps, reference)
                    check('dxc %s DPP add %s mutant mismatches' % (adapter, name),
                          run.returncode != 0 and any(ln.startswith('pixel mismatch') for ln in run.stderr.splitlines()),
                          run.stdout + run.stderr)

def check_subrev_draw(dxc, renderer, adapter, fxc=None):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs_subrev', 'vs_6_0', directory)
        for name, (_, control, neighbour) in SUBREV.items():
            text, reason = translate_words(subrev_words(control))
            if text is None:
                check('SUBREV %s translates for the draw' % name, False, str(reason))
                continue
            reference = words_file(directory, name, subrev_reference(neighbour))
            ps = compile_hlsl(dxc, 'dxc', text, 'ps_' + name, 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, ps, reference)
            print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                  if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
            check('SUBREV %s %s: all %d pixels match' % (name, adapter, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            mutant = compile_hlsl(dxc, 'dxc', SUBREV_MUTANTS[name](text), 'mutant_' + name, 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, mutant, reference)
            check('SUBREV %s %s own-lane mutant fails readback' % (name, adapter),
                  run.returncode != 0 and any(line.startswith('pixel mismatch') for line in run.stderr.splitlines()),
                  run.stdout + run.stderr)


def check_binary(llvm_mc):
    for name, (perm, control, _) in SUBREV.items():
        check('llvm-mc assembles the SUBREV %s fixture to the same words' % name,
              assemble(llvm_mc, SUBREV_SOURCE.format(perm=perm)) == subrev_words(control))
    for name, (source, hexwords) in FIXTURES.items():
        check('llvm-mc assembles the %s fixture to the same words' % name,
              assemble(llvm_mc, source) == bytes.fromhex(hexwords))


def main():
    if len(sys.argv) == 1:
        controls()
        subrev_controls()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'draw':
            controls()
            subrev_controls()
            check_draw(*args)
            check_add_draw(*args)

            check_subrev_draw(*args)
        else:
            raise SystemExit(__doc__)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
