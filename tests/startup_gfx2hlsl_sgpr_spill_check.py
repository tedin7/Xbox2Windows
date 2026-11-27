"""SGPR spill to VGPR lanes in the pixel stage of tools/startup/gfx2hlsl.py (#2125, item E).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture source and compare it with the words built here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair, draw 16 x 16 pixels and compare every pixel
                                       with the reference below

A compiler parks a scalar in one lane of a VGPR: `v_writelane_b32 vN, sM, k` writes SGPR sM into lane k of vN and
`v_readlane_b32 sD, vN, k` reads lane k back into an SGPR (AMD RDNA 2 ISA reference, VOP3 lane operations; LLVM 15
AMDGPU spill lowering). Both ignore EXEC and only lane k of vN changes. When k is a constant and no other
instruction names vN, vN is a table of scalar slots and the model needs no cross-lane read: one scalar per
(vN, k). The translator refuses every other use of those registers (a lane chosen at run time, vN also used as a
vector, a read of a slot nothing wrote, branches, lane masks as the value).

The fixture writes four slots in three VGPRs, clobbers the source SGPRs, reloads them and ORs each reloaded word
into the packed screen position in v0. One reload follows a second write to the same slot, so a lowering that
keeps the first value, mixes lanes of one VGPR, or mixes VGPRs at one lane reads a different image. The reference
is written here with integers from the slot rule: the last value written to (VGPR, lane).
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

WIDTH = HEIGHT = 16
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
A, B, C = 0x40200000, 0x40400000, 0x40800000  # binary32 words with no bit below 21, clear of x | y << 16
FIXTURE_TEXT = '''s_mov_b32 s0, 0x40200000
s_mov_b32 s1, 0x40400000
s_mov_b32 s2, 4.0
v_writelane_b32 v10, s0, 0
v_writelane_b32 v10, s1, 31
v_writelane_b32 v11, s2, 0
s_or_b32 s8, s0, s1
v_writelane_b32 v12, s8, 7
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s8, 0
v_readlane_b32 s3, v10, 0
v_readlane_b32 s4, v10, 31
v_readlane_b32 s5, v11, 0
v_writelane_b32 v10, s5, 0
v_readlane_b32 s6, v10, 0
v_readlane_b32 s9, v12, 7
v_or_b32_e32 v1, s3, v0
v_or_b32_e32 v2, s4, v0
v_or_b32_e32 v3, s6, v0
v_or_b32_e32 v4, s9, v0
exp mrt0 v1, v2, v3, v4 done vm
s_endpgm
'''
FIXTURE_WORDS = struct.pack('<37I', *(
    0xbe8003ff, 0x40200000, 0xbe8103ff, 0x40400000, 0xbe8203f6, 0xd761000a, 0x00010000, 0xd761000a,
    0x00013e01, 0xd761000b, 0x00010002, 0x88080100, 0xd761000c, 0x00010e08, 0xbe800380, 0xbe810380,
    0xbe820380, 0xbe880380, 0xd7600003, 0x0001010a, 0xd7600004, 0x00013f0a, 0xd7600005, 0x0001010b,
    0xd761000a, 0x00010005, 0xd7600006, 0x0001010a, 0xd7600009, 0x00010f0c, 0x38020003, 0x38040004,
    0x38060006, 0x38080009, 0xf800180f, 0x04030201, 0xbf810000))
# The same program for a wave64 stage: it copies the 64-bit exec pair first (only a wave64 compiler emits that form), and
# v10 holds lanes 8 and 40, which are one slot in wave32 and two in wave64.
W64_TEXT = 's_mov_b64 s[20:21], exec\n' + FIXTURE_TEXT.replace(
    'v_writelane_b32 v10, s0, 0', 'v_writelane_b32 v10, s0, 8').replace(
    'v_writelane_b32 v10, s1, 31', 'v_writelane_b32 v10, s1, 40').replace(
    'v_writelane_b32 v10, s5, 0', 'v_writelane_b32 v10, s5, 8').replace(
    'v_readlane_b32 s3, v10, 0', 'v_readlane_b32 s3, v10, 8').replace(
    'v_readlane_b32 s4, v10, 31', 'v_readlane_b32 s4, v10, 40').replace(
    'v_readlane_b32 s6, v10, 0', 'v_readlane_b32 s6, v10, 8')
_LANE_WORDS = {(0xd761000a, 0x00010000): 0x00011000, (0xd761000a, 0x00013e01): 0x00015001,
               (0xd761000a, 0x00010005): 0x00011005, (0xd7600003, 0x0001010a): 0x0001110a,
               (0xd7600004, 0x00013f0a): 0x0001510a, (0xd7600006, 0x0001010a): 0x0001110a}


def _wave64_words():
    words = list(struct.unpack('<37I', FIXTURE_WORDS))
    for i in range(len(words) - 1):
        words[i + 1] = _LANE_WORDS.get((words[i], words[i + 1]), words[i + 1])
    return struct.pack('<38I', 0xbe94047e, *words)  # s_mov_b64 s[20:21], exec


W64_WORDS = _wave64_words()
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def reference_slots(text=FIXTURE_TEXT):
    """Execute the fixture's lane writes and reads on a model of the VGPRs: eight-by-thirty-two lanes of words."""
    lanes, scalars = {}, {}
    for line in text.splitlines():
        op, _, rest = line.partition(' ')
        args = [a.strip() for a in rest.split(',')]
        if op == 's_mov_b32':
            scalars[args[0]] = 0x40800000 if args[1] == '4.0' else int(args[1], 0)
        elif op == 's_or_b32':
            scalars[args[0]] = scalars[args[1]] | scalars[args[2]]
        elif op == 's_mov_b64':
            pass  # the exec copy of the wave64 fixture: no value the image reads
        elif op == 'v_writelane_b32':
            lanes[(args[0], int(args[2]))] = scalars[args[1]]
        elif op == 'v_readlane_b32':
            scalars[args[0]] = lanes[(args[1], int(args[2]))]
    return [scalars[r] for r in ('s3', 's4', 's6', 's9')]


def pixel_words(x, y, text=FIXTURE_TEXT):
    return [(x | y << 16) | value for value in reference_slots(text)]


def image(text=FIXTURE_TEXT):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            words += pixel_words(x, y, text)
    return words


def pipeline(words=FIXTURE_WORDS):
    return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for lane operations to be translated one at a time."""

    def __init__(self, program=(), kind='ps'):
        self.kind = kind
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v, self.varying_s = set(), set(), set(), set()
        self.scc_uniform, self.exec_known = True, True
        self.has_branches, self.blocks, self.masks = False, [], set()
        self.ins = [(4 * i, 1, G.re.sub(r'_e(32|64)$', '', a.split(' ', 1)[0]), a) for i, a in enumerate(program)]


def refused(program, kind='ps', branches=False):
    """-> the refusal reason of the first refused instruction of `program`, or None."""
    stage = Probe(program, kind)
    stage.has_branches = branches
    for _, _, name, asm in stage.ins:
        try:
            stage.one(name, asm)
        except G.Unsupported as reason:
            return str(reason)
    return None


def controls():
    # 1. The reference follows the slot rule: the last write to a (VGPR, lane) wins; other lanes and VGPRs are untouched.
    check('reference slots: v10[0] reloaded after the overwrite, v10[31], v11[0] and the derived word',
          reference_slots() == [A, B, C, A | B], str(reference_slots()))

    # 2. The fixture words decode to the assembly text they were built from (the decoder is the reference for the bits).
    got = [asm for _, _, _, asm in G.decode(FIXTURE_WORDS)]
    normal = lambda s: re.sub(r'_e(32|64)\b', '', s)  # noqa: E731
    want = FIXTURE_TEXT.splitlines()
    check('fixture decodes to its source (%d instructions)' % len(want),
          [normal(a) for a in got] == [normal(w) for w in want], '%s\n%s' % (got, want))

    # 3. Translation: one scalar slot per (VGPR, lane); the spill VGPRs never become vector registers.
    ps = pipeline().hlsl_ps()
    slots = ('sp10_0', 'sp10_31', 'sp11_0', 'sp12_7')
    check('four slot variables are declared and initialized: %s' % re.findall(r'sp\d+_\d+ = 0', ps),
          all(re.search(r'\b%s = 0' % s, ps) for s in slots))
    check('no vector register is declared for the spill VGPRs', not re.search(r'\bv1[012]\b', ps))
    check('the second write to v10 lane 0 is emitted after the first reload',
          ps.index('sp10_0 = s5') > ps.index('s3 = sp10_0'))

    # 4. Accepted and refused shapes, each with a reason.
    ok = ['v_writelane_b32 v10, s0, 3', 'v_readlane_b32 s1, v10, 3']
    check('write then read of one slot is accepted', refused(ok) is None)
    check('an inline constant value is accepted', refused(['v_writelane_b32 v10, 7, 3', 'v_readlane_b32 s1, v10, 3']) is None)
    cases = [
        ('read before any write', ['v_readlane_b32 s1, v10, 3']),
        ('read of a lane nothing wrote', ['v_writelane_b32 v10, s0, 3', 'v_readlane_b32 s1, v10, 4']),
        ('read of another VGPR at the written lane', ['v_writelane_b32 v10, s0, 3', 'v_readlane_b32 s1, v11, 3']),
        ('lane chosen by an SGPR', ['v_writelane_b32 v10, s0, s2', 'v_readlane_b32 s1, v10, 3']),
        ('lane 64', ['v_writelane_b32 v10, s0, 64']),
        ('lanes 8 and 40 of one VGPR (one slot in wave32, two in wave64)',
         ['v_writelane_b32 v10, s0, 8', 'v_writelane_b32 v10, s1, 40', 'v_readlane_b32 s2, v10, 8']),
        ('lane mask value (vcc_lo)', ['v_writelane_b32 v10, vcc_lo, 3']),
        ('lane mask value (exec_lo)', ['v_writelane_b32 v10, exec_lo, 3']),
        ('destination outside the plain SGPRs', ['v_writelane_b32 v10, s0, 3', 'v_readlane_b32 vcc_lo, v10, 3']),
        ('spill VGPR also used as a vector (after)',
         ['v_writelane_b32 v10, s0, 3', 'v_readlane_b32 s1, v10, 3', 'v_add_f32 v1, v10, v2']),
        ('modifier', ['v_writelane_b32 v10, s0, 3 clamp']),
    ]
    for label, program in cases:
        reason = refused(program)
        check('refused: %s (%s)' % (label, reason), reason is not None)
    for vector_op in ('v_mov_b32 v10, 1.0', 'v_mov_b32 v9, v[9:10]'):
        reason = refused([vector_op, 'v_writelane_b32 v10, s0, 3'], kind='ms')
        check('mesh spill VGPR used by %s is refused' % vector_op, 'also used as a vector' in (reason or ''), str(reason))
    check('a vertex stage is accepted', refused(ok, kind='vs') is None, str(refused(ok, kind='vs')))
    check('a compute stage is accepted', refused(ok, kind='cs') is None, str(refused(ok, kind='cs')))
    check('refused: a kind outside pixel, vertex, mesh and compute (%s)' % refused(ok, kind='gs'), refused(ok, kind='gs') is not None)
    wave64 = ['v_writelane_b32 v10, s0, 40', 'v_writelane_b32 v10, s1, 63', 'v_readlane_b32 s2, v10, 40']
    check('wave64 lanes 40 and 63 are accepted', refused(wave64) is None, str(refused(wave64)))
    check('refused: a shader with branches (%s)' % refused(ok, branches=True), refused(ok, branches=True) is not None)

    # 4b. Lanes 8 and 40 of one VGPR are two slots only when the program proves wave64 by its own exec forms.
    pair = ['v_writelane_b32 v10, s0, 8', 'v_writelane_b32 v10, s1, 40', 'v_readlane_b32 s2, v10, 8']
    check('lanes 8 and 40 with a 64-bit exec form are accepted', refused(['s_mov_b64 s[20:21], exec'] + pair) is None,
          str(refused(['s_mov_b64 s[20:21], exec'] + pair)))
    for label, exec_form in (('no exec form', []), ('a wave32 exec form', ['s_and_saveexec_b32 s20, s21']),
                             ('both exec forms', ['s_mov_b64 s[20:21], exec', 's_and_saveexec_b32 s22, s23'])):
        reason = refused(exec_form + pair)
        check('lanes 8 and 40 with %s stay refused (%s)' % (label, reason), 'one slot in wave32' in (reason or ''), str(reason))
    check('the wave64 fixture decodes to its source',
          [G.re.sub(r'_e(32|64)\b', '', a) for _, _, _, a in G.decode(W64_WORDS)] ==
          [G.re.sub(r'_e(32|64)\b', '', w) for w in W64_TEXT.splitlines()])
    check('wave64 reference: lanes 8 and 40 of v10 are separate slots', reference_slots(W64_TEXT) == [A, B, C, A | B],
          str(reference_slots(W64_TEXT)))
    check('wave64 program: proven by its code and translated',
          'sp10_8 = 0x40200000u' in pipeline(W64_WORDS).hlsl_ps() and 'sp10_40 = 0x40400000u' in pipeline(W64_WORDS).hlsl_ps())
    check('wave32-safe fixture still has no exec form (decided by lanes 0 and 31 only)',
          not G.wave64_exec_proof(G.decode(FIXTURE_WORDS)))
    check('wave64 fixture is proven wave64', G.wave64_exec_proof(G.decode(W64_WORDS)))

    # 5. The reload keeps the constant the write stored, so later scalar folding sees the same word.
    stage = Probe(['v_writelane_b32 v10, 7, 3', 'v_readlane_b32 s1, v10, 3'])
    for _, _, name, asm in stage.ins:
        stage.one(name, asm)
    check('a constant slot reloads as a constant: %s' % (stage.sym.get(1),), stage.sym.get(1) == ('const', 7))

    # 6. A word loaded from a constant buffer at an immediate offset is a plain number: it may be spilled.
    stage = Probe(['v_writelane_b32 v10, s0, 3', 'v_readlane_b32 s1, v10, 3'])
    stage.sym[0] = ('fixed_buffer_word',)  # what s_buffer_load_dword s0, ..., <immediate> records
    reason = None
    try:
        for _, _, name, asm in stage.ins:
            stage.one(name, asm)
    except G.Unsupported as err:
        reason = str(err)
    check('a buffer word loaded at an immediate offset is accepted as a spill source (%s)' % reason, reason is None)


def check_binary(llvm_mc):
    check('llvm-mc assembles the fixture to the same words', assemble(llvm_mc, FIXTURE_TEXT) == FIXTURE_WORDS)
    check('llvm-mc assembles the wave64 fixture to the same words', assemble(llvm_mc, W64_TEXT) == W64_WORDS)


def words_file(directory, name, words):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(words), *words))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def mutate(text, old, new):
    mutated = re.sub(old, new, text)
    assert mutated != text, (old, text)
    return mutated


def check_draw(dxc, renderer, adapter, fxc=None):
    text = pipeline().hlsl_ps()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        good = words_file(directory, 'good', image())
        for stage_name, compiler, profile_vs, profile_ps in (
                [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])):
            vs = compile_hlsl(compiler, stage_name, FULLSCREEN_VS, 'vs_' + stage_name, profile_vs, directory)
            ps = compile_hlsl(compiler, stage_name, text, 'ps_' + stage_name, profile_ps, directory)
            run = draw(renderer, adapter, vs, ps, good)
            check('%s %s: all %d pixels equal the slot reference' % (adapter, stage_name, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            # Negative controls: each wrong lowering must read a different image than the reference.
            wrong = {
                'the overwritten slot keeps its first value': mutate(text, r'sp10_0 = s5;', 'sp10_0 = sp10_0;'),
                # A constant slot folds into the export, so the wrong lane shows up as lane 0's word there.
                'lane 31 reads lane 0 of the same VGPR': mutate(text, r'v2 = 0x40400000u', 'v2 = 0x40200000u'),
                'another VGPR at lane 0 reads v10': mutate(text, r'= sp11_0;', '= sp10_0;'),
                'no reload (the clobbered SGPR stays)': mutate(text, r'(s9) = sp12_7;', r'\1 = 0;'),
            }
            for label, source in wrong.items():
                bad = compile_hlsl(compiler, stage_name, source, 'bad_%s_%d' % (stage_name, abs(hash(label))),
                                   profile_ps, directory)
                run = draw(renderer, adapter, vs, bad, good)
                check('%s %s mutation "%s" fails against the reference' % (adapter, stage_name, label),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            # The wave64 program: lanes 8 and 40 of v10 are two slots, proven by its 64-bit exec copy.
            text64 = pipeline(W64_WORDS).hlsl_ps()
            good64 = words_file(directory, 'good64', image(W64_TEXT))
            ps64 = compile_hlsl(compiler, stage_name, text64, 'ps64_' + stage_name, profile_ps, directory)
            run = draw(renderer, adapter, vs, ps64, good64)
            check('%s %s wave64 (lanes 8 and 40): all %d pixels equal the slot reference' % (adapter, stage_name, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout, run.stdout + run.stderr)
            wrong64 = {
                'the overwritten slot keeps its first value': mutate(text64, r'sp10_8 = s5;', 'sp10_8 = sp10_8;'),
                'lane 40 reads lane 8 of the same VGPR': mutate(text64, r'v2 = 0x40400000u', 'v2 = 0x40200000u'),
                'no reload (the clobbered SGPR stays)': mutate(text64, r'(s9) = sp12_7;', r'\1 = 0;'),
            }
            for label, source in wrong64.items():
                bad = compile_hlsl(compiler, stage_name, source, 'bad64_%s_%d' % (stage_name, abs(hash(label))),
                                   profile_ps, directory)
                run = draw(renderer, adapter, vs, bad, good64)
                check('%s %s wave64 mutation "%s" fails against the reference' % (adapter, stage_name, label),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)


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
