"""The first active lane of a pixel stage in tools/startup/gfx2hlsl.py (#2618, the F rows), read back from a D3D12 draw.

  (no arguments)                       a 32-lane model of the source pair, translation controls and refusals
  binary <llvm-mc>                     assemble every fixture with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel stage with DXC ps_6_0 (the lowering uses a shader
                                       model 6.0 wave intrinsic, so FXC is not used), draw a triangle over 32 x 32
                                       pixels and compare every covered pixel with a CPU reference for its desktop
                                       wave. The optional fxc argument is accepted and ignored.

The source pair (RDNA 2 ISA, S_FF1_I32_B32 and V_READLANE_B32):
  s_ff1_i32_b32 sD, exec_lo        the index of the lowest set bit of EXEC: the first active lane
  v_readlane_b32 sX, vY, sD        vY of lane sD (bits 4:0 in wave32), whatever EXEC says
so sX is vY of the first active lane. The fixtures are written in LLVM 15 spelling and assembled from source.

The model below runs the pair on 32 lanes with plain integers, from the ISA text and not from the lowering, for random
EXEC masks, and shows that sX is the input of the lowest active lane.

The desktop contract is the one of the #2835 wave reductions: the console packs pixels into waves in its own way, so a
correct program cannot depend on which pixel is first, and the pair becomes WaveReadLaneFirst(vY) over the active lanes
of the desktop wave. The lane number has no desktop meaning, so every other read of sD is refused, as are a program
with branches, an EXEC that is not the entry mask at the s_ff1 or at the readlane, s_wqm (helper lanes in EXEC on the
console), a readlane source that is a v_writelane_b32 spill, and the other stages.

The draw cannot know the desktop packing in advance, so the test appends one observation to the translated pixel
stage: the fourth channel is replaced by the lane index, the lane count (WaveGetLaneCount) and WaveActiveMin of the
pixel id, which names the wave without using the intrinsic under test. The readback (offscreen_check
--startup-pixel-dump) is grouped by that name, and every covered pixel's result is compared with the input of the
covered pixel with the lowest lane index in its group. The triangle leaves quads partly covered: their helper lanes run
the stage, are not in the console EXEC (no s_wqm), and must not be picked. Planted errors in the translated text (the
lane's own value, the wave's largest value, the first lane of the quad, the lane after the first) must fail.
"""
import collections
import random
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
from startup_image_3d_check import compile_hlsl  # noqa: E402
from startup_gfx2hlsl_s_ff1_check import Probe  # noqa: E402
from startup_gfx2hlsl_wave_reduce_check import (  # noqa: E402
    CLEAR, CMP, MULTIPLIER, SIZE, START_MUL, TRIANGLE_VS, V_READLANE_B32, WQM, assemble_object, dump, export, move,
    program, sop1, sopp, translate, vop3)

MASK = 0xffffffff
EXEC_LO, EXEC_HI, VCC_LO = 126, 127, 106
S_MOV_B32, S_FF1_I32_B32, S_AND_SAVEEXEC_B32, S_LSHL_B32 = 0x03, 0x13, 0x3c, 0x1e
V_WRITELANE_B32 = 0x361
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail))
    if not ok:
        failures.append(label)


def sop2(op, sdst, ssrc0, ssrc1):
    return struct.pack('<I', 0x80000000 | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0)


def ff1(sd=5):
    return [('s_ff1_i32_b32 s%d, exec_lo' % sd, sop1(S_FF1_I32_B32, sd, EXEC_LO))]


def readlane(sx=6, vy=1, sd=5):
    return [('v_readlane_b32 s%d, v%d, s%d' % (sx, vy, sd), vop3(V_READLANE_B32, sx, 256 + vy, sd))]


def s_mov(sd, src, text):
    return ('s_mov_b32 s%d, %s' % (sd, text), sop1(S_MOV_B32, sd, src))


SAVEEXEC = [('s_and_saveexec_b32 s0, vcc_lo', sop1(S_AND_SAVEEXEC_B32, 0, VCC_LO))]
RESTORE = [('s_mov_b32 exec_lo, s0', sop1(S_MOV_B32, EXEC_LO, 0))]
SHIFT = [('s_lshl_b32 s7, s5, 2', sop2(S_LSHL_B32, 7, 5, 130))]
SPILL = [('v_writelane_b32 v3, s0, 0', vop3(V_WRITELANE_B32, 3, 0, 128))]   # v3: a table of scalar slots
END = [move(4, 6)] + export('v4', 'v1', 'v1')            # first lane's input, own input, own input, position
FIRST = program(START_MUL, ff1(), readlane(), END)
# Two reads of one first lane: the input and the packed position (x | y << 16) of the same pixel.
TWO = program(START_MUL, ff1(), readlane(), readlane(7, 0), [move(4, 6), move(2, 7)], export('v4', 'v2', 'v1'))
# The lane number overwritten after its readlane, then read as a number: the new value, not the lane.
REUSED = program(START_MUL, ff1(), readlane(), [s_mov(5, 131, '3'), move(2, 5)], [move(4, 6)],
                 export('v4', 'v2', 'v1'))


def branch_program():
    """s_cbranch_execz at the top, landing on the export after the pair."""
    body = START_MUL + ff1() + readlane() + [move(4, 6)]
    words_before = sum(len(w) for _t, w in body) // 4
    label = [('s_cbranch_execz label_%04X' % (4 * (1 + words_before)), sopp(0x08, words_before))]
    return program(label, body, export('v4', 'v1', 'v1'))


def refusals():
    """(label, program, expected reason fragment): each is refused, the fragment names the reason."""
    return [
        ('after s_wqm exec, exec', program(START_MUL, WQM, ff1(), readlane(), END), 'after s_wqm exec, exec'),
        ('EXEC changed before the s_ff1', program(START_MUL, CMP, SAVEEXEC, ff1(), readlane(), END),
         'EXEC is not the entry mask'),
        ('EXEC changed between the s_ff1 and the readlane', program(START_MUL, CMP, ff1(), SAVEEXEC, readlane(), END),
         'EXEC is no longer the entry mask'),
        # Conservative: the restore brings the entry mask back, but the one-lane model no longer knows it.
        ('EXEC changed and restored between (conservative)',
         program(START_MUL, CMP, ff1(), SAVEEXEC, RESTORE, readlane(), END), 'EXEC is no longer the entry mask'),
        ('the lane number shifted as data', program(START_MUL, ff1(), SHIFT, readlane(), END),
         's_lshl_b32 s7, s5, 2 reads the lane number as data'),
        ('the lane number moved to a VGPR', program(START_MUL, ff1(), readlane(), [move(4, 5)],
                                                    export('v4', 'v1', 'v1')),
         'v_mov_b32_e32 v4, s5 reads the lane number as data'),
        ('the lane number copied to another SGPR', program(START_MUL, ff1(), [s_mov(7, 5, 's5')], readlane(6, 1, 7),
                                                           END),
         's_mov_b32 s7, s5 reads the lane number as data'),
        ('a lane spill VGPR as the readlane source', program(START_MUL, SPILL, ff1(), readlane(6, 3), END),
         'is a lane spill written by v_writelane_b32'),
        ('a program with branches', branch_program(), 'with branches'),
        # The translator refuses exec_hi as an operand before the pair is reached; the pair's own wave64 guard is
        # behind it.
        ('a program that names exec_hi', program(START_MUL, [s_mov(8, EXEC_HI, 'exec_hi')], ff1(), readlane(), END),
         'exec_hi'),
    ]


# ---- 32-lane model of the source pair (the ISA, not the lowering) -----------------------------------------------------
def model_pair(inputs, exec_mask):
    """sX after s_ff1_i32_b32 sD, exec_lo / v_readlane_b32 sX, vY, sD on 32 lanes (vY = inputs)."""
    sd = MASK if exec_mask == 0 else (exec_mask & -exec_mask).bit_length() - 1
    return inputs[sd & 31]   # wave32: the lane select's bits 4:0


def controls():
    rng = random.Random(2618)
    agree = True
    for _ in range(500):
        inputs = [rng.getrandbits(32) for _ in range(32)]
        mask = rng.getrandbits(32) or 1
        first = min(k for k in range(32) if mask >> k & 1)
        agree &= model_pair(inputs, mask) == inputs[first]
    check('32-lane model: the pair reads the input of the lowest active lane for 500 random EXEC masks', agree)

    for label, (_source, words), want in (
            ('the pair', FIRST, ['s6 = WaveReadLaneFirst(v1);']),
            ('two reads of one first lane', TWO, ['s6 = WaveReadLaneFirst(v1);', 's7 = WaveReadLaneFirst(v0);']),
            ('the lane number overwritten after its readlane', REUSED,
             ['s6 = WaveReadLaneFirst(v1);', 's5 = 0x00000003u;', 'v2 = 0x00000003u;'])):
        text, reason = translate(words)
        lines = [ln.strip() for ln in (text or '').split('\n')]
        check('%s translates' % label, text is not None and all(w in lines for w in want),
              reason or '\n'.join(ln for ln in lines if re.search(r'\bs[5-7]\b', ln)))
        if text is not None and label != 'the lane number overwritten after its readlane':
            check('%s: the lane number has no run-time value' % label,
                  not any(re.search(r'\bs5\b', ln) for ln in lines), '\n'.join(lines))

    for label, (_source, words), want in refusals():
        text, reason = translate(words)
        check('refused: %s (%s)' % (label, reason), text is None and want in (reason or ''), str(reason))

    for kind in ('vs', 'ms'):
        stage = Probe(kind=kind)
        try:
            stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s5, exec_lo')
            check('s_ff1_i32_b32 of exec_lo in a %s stage is refused' % kind, False, str(stage.lines))
        except G.Unsupported as error:
            check('s_ff1_i32_b32 of exec_lo in a %s stage is refused (%s)' % (kind, error),
                  'outside compute' in str(error), str(error))


# ---- the draw ---------------------------------------------------------------------------------------------------------
OBSERVE = ('    o.mrt0.w = asfloat(WaveGetLaneIndex() | (WaveGetLaneCount() << 8) |'
           ' (WaveActiveMin(uint(i.pos.x) | (uint(i.pos.y) << 5)) << 16));\n    return o;')


def observed(text):
    assert text.count('    return o;') == 1
    return text.replace('    return o;', OBSERVE)


def input_word(x, y):
    """v_mul_u32_u24_e32 v1, 0x9b5d, v0 with v0 = x | y << 16 (both below 2^24 here)."""
    return ((x | (y << 16)) * MULTIPLIER) & MASK


def position(x, y):
    return x | (y << 16)


# Planted errors in the translated text, each a replacement of the lowered readlane.
MUTATIONS = {
    'own lane': lambda v: v,
    'largest value of the wave': lambda v: 'WaveActiveMax(%s)' % v,
    'first lane of the quad': lambda v: 'QuadReadLaneAt(%s, 0)' % v,
    'lane after the first': lambda v: 'WaveReadLaneAt(%s, WaveReadLaneFirst(WaveGetLaneIndex()) + 1)' % v,
}


def mutate(text, how):
    return re.sub(r'WaveReadLaneFirst\((v\d+)\)', lambda m: MUTATIONS[how](m.group(1)), text)


# (label, program, channels): each of the first three output channels is 'first' (the first lane's input), 'first
# position' (the first lane's packed position) or 'input' (the pixel's own input).
DRAWS = [('the pair', FIRST, ('first', 'input', 'input')),
         ('two reads of one first lane', TWO, ('first', 'first position', 'input'))]


def verify(channels, words):
    """Group the readback by desktop wave and compare with the CPU reference. Returns (errors, facts)."""
    pixels = struct.unpack('<%dI' % (len(words) // 4), words)
    groups, errors = collections.defaultdict(list), []
    for y in range(SIZE):
        for x in range(SIZE):
            p = pixels[(y * SIZE + x) * 4:(y * SIZE + x) * 4 + 4]
            if p == (CLEAR,) * 4:
                continue
            groups[p[3] >> 16].append((x, y, p))
    counts = set()
    for key, members in groups.items():
        lanes = [p[3] & 0xff for _x, _y, p in members]
        counts.update((p[3] >> 8) & 0xff for _x, _y, p in members)
        if len(set(lanes)) != len(lanes) or (key & 31, key >> 5) not in {(x, y) for x, y, _p in members}:
            errors.append('wave %d: lanes %s do not name one wave' % (key, lanes))
            continue
        fx, fy, _fp = min(members, key=lambda m: m[2][3] & 0xff)
        for x, y, p in members:
            expected = [input_word(fx, fy) if kind == 'first' else position(fx, fy) if kind == 'first position'
                        else input_word(x, y) for kind in channels]
            if list(p[:3]) != expected:
                errors.append('pixel (%d,%d): %s, reference %s (first lane (%d,%d) of %d pixels)'
                              % (x, y, ' / '.join('%08x' % w for w in p[:3]), ' / '.join('%08x' % w for w in expected),
                                 fx, fy, len(members)))
    covered = {(x, y) for members in groups.values() for x, y, _p in members}
    quads = {(x // 2, y // 2) for x, y in covered}
    partial_quads = sum(1 for qx, qy in quads
                        if sum((2 * qx + i, 2 * qy + j) in covered for i in (0, 1) for j in (0, 1)) < 4)
    if len(counts) != 1:
        errors.append('lane counts %s' % sorted(counts))
    lane_count = min(counts) if counts else 0
    shared = sum(1 for members in groups.values() if len(members) > 1)
    facts = ('covered=%d waves=%d lane_count=%s partial_waves=%d partial_quads=%d waves_of_several=%d'
             % (len(covered), len(groups), '/'.join(map(str, sorted(counts))),
                sum(1 for members in groups.values() if len(members) < lane_count), partial_quads, shared))
    if partial_quads == 0 or len(covered) < 100 or shared == 0:
        errors.append('the draw does not exercise partly covered quads and shared waves: ' + facts)
    return errors, facts


def check_draw(dxc, renderer, adapter, fxc=None):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', TRIANGLE_VS, 'vs_dxc', 'vs_6_0', directory)
        for label, (_source, words), channels in DRAWS:
            text, reason = translate(words)
            if text is None:
                check('%s translates for the draw' % label, False, str(reason))
                continue
            for how in [None] + list(MUTATIONS):
                source = observed(text if how is None else mutate(text, how))
                if how is not None and source == observed(text):
                    check('%s %s %s: the planted error applies' % (adapter, label, how), False)
                    continue
                tag = re.sub(r'\W+', '_', label + ('' if how is None else '_' + how))
                ps = compile_hlsl(dxc, 'dxc', source, 'ps_' + tag, 'ps_6_0', directory)
                out = directory / ('%s.bin' % tag)
                run = dump(renderer, adapter, vs, ps, out)
                if run.returncode != 0 or not out.exists():
                    check('%s %s %s: draw' % (adapter, label, how or 'lowering'), False, run.stdout + run.stderr)
                    continue
                errors, facts = verify(channels, out.read_bytes())
                clean = 'debug_layer=enabled' not in run.stdout or 'debug_errors=0 debug_warnings=0' in run.stdout
                if how is None:
                    print('  ' + ' '.join(ln for ln in run.stdout.splitlines() if ln.startswith(
                        ('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                    print('  %s: %s' % (label, facts))
                    check('%s %s: every covered pixel matches the first lane of its desktop wave' % (adapter, label),
                          not errors and clean, '; '.join(errors[:3]) + ' ' + run.stdout)
                else:
                    check('%s %s %s mutation fails against the reference: %s (%d pixels)' % (
                        adapter, label, how, errors[0] if errors else '', len(errors)), bool(errors))


def check_binary(llvm_mc):
    programs = [(label, prog) for label, prog, _c in DRAWS] + [('the lane number overwritten', REUSED)]
    programs += [(label, prog) for label, prog, _w in refusals()]
    for label, (source, words) in programs:
        if 'label_' not in source:
            built = assemble(llvm_mc, source)
        else:  # a branch: llvm-mc needs the label defined, at the word the fixture names
            target = int(re.search(r'label_([0-9A-F]{4})', source).group(1), 16)
            out, offset = [], 0
            for text, size in zip(source.splitlines(), [n for _o, n, _name, _t in G.decode(words)]):
                if offset == target:
                    out.append('L%04X:' % target)
                out.append(text.replace('label_', 'L'))
                offset += size
            built = assemble_object(llvm_mc, '\n'.join(out) + '\n')
        check('llvm-mc assembles the program "%s" to the same words' % label, built == words,
              built.hex() + ' / ' + words.hex())


def main():
    if len(sys.argv) == 1:
        controls()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'draw':
            controls()
            check_draw(*args)
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
