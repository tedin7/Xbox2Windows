"""`src_scc` at a branch target in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_src_scc_join_check.py [dispatch <dxc> <offscreen_check> <adapter>]
       startup_gfx2hlsl_src_scc_join_check.py draw <dxc> <offscreen_check>
       startup_gfx2hlsl_src_scc_join_check.py binary <llvm-mc>

`dispatch` also compiles the accepted if/else and skipped-block programs with DXC and compares a D3D12 UAV
readback (WARP or hardware) with the compare each read must see.

A compiler materializes SCC into an SGPR on both arms of an if/else: `s_bitcmp1_b32 s4, 3`,
`s_cbranch_scc0 ELSE`, then `s_mov_b32 s10, src_scc` in each arm. The arm at ELSE starts at a branch
target, where `src_scc` was refused as unverified. The translator now carries, on every forward branch
edge, the offset of the emitted SCC write that reaches it, and only when that write was uniform. A branch
target keeps SCC proven when the fall-through and every edge into it carry the same write and no back edge
enters it. Every path into the target then ran that one write last, both for the lane being modelled and
for the wave, whichever path each took (a per-lane `s_cbranch_execz` skip included). So the model's SCC is
the hardware value.

The pixel draw covers compare and logic writers, both arms of a uniform join, and a counted loop.
The vertex draw and compute dispatch also exercise the reaching-path fallback with two uniform SCC diamonds.
Negative controls cover a missing writer, an SCC write skipped by a lane-local EXEC branch, an
unproved back edge, lane-mask SCC, and an unmodeled writer. One producer polarity mutation
must fail D3D12 readback.

The instruction words are built here from the field layout that `tools/disasm_rdna2.py` decodes:
- SOP1 (`[31:23] == 0b101111101`, sdst in `w0[22:16]`, op in `w0[15:8]`, src0 in `w0[7:0]`)
- SOP2 (`[31:30] == 0b10`, op in `w0[29:23]`, sdst in `w0[22:16]`, src1 in `w0[15:8]`, src0 in `w0[7:0]`)
- SOPC (`[31:23] == 0b101111110`, op in `w0[22:16]`, src1 in `w0[15:8]`, src0 in `w0[7:0]`)
- SOPP (`[31:23] == 0b101111111`, op in `w0[22:16]`, simm16 in `w0[15:0]`, branch target
  `pc + 4 + 4 * simm16`)
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
import startup_firstlane_pixel_fixture as firstlane_pixel  # noqa: E402
from startup_flat_param_check import PS_WORDS, assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


SOP1_MOV_B32, SOP1_WQM_B64 = 0x03, 0x0A
SOP2_AND_B32 = 0x0E
SOPC_CMP_EQ_U32 = 0x06
SOPC_BITCMP1_B32 = 0x0D
SOPP_NOP, SOPP_ENDPGM, SOPP_BRANCH, SOPP_CBRANCH_SCC0, SOPP_CBRANCH_EXECZ = 0x00, 0x01, 0x02, 0x04, 0x08

S_SCC = 253
S_THREE = 131      # inline integer 3
VCC_LO, EXEC_LO = 106, 126
S4, S5, S6, S7, S10 = 4, 5, 6, 7, 10
UNVERIFIED = 'src_scc with unverified SCC'


def sop1(op, sdst, ssrc0):
    return struct.pack('<I', 0xBE800000 | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) | (ssrc0 & 0xFF))


def sop2(op, sdst, ssrc0, ssrc1):
    return struct.pack('<I', (0b10 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8)
                       | (ssrc0 & 0xFF))


def sopc(op, ssrc0, ssrc1):
    return struct.pack('<I', 0xBF000000 | ((op & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))


def sopp(op, simm=0):
    return struct.pack('<I', 0xBF800000 | ((op & 0x7F) << 16) | (simm & 0xFFFF))


def branch(op, at, target):
    """SOPP branch at byte offset `at` to byte offset `target`."""
    return sopp(op, (target - at - 4) // 4)


class Pipeline:
    """Just enough pipeline state for Stage.translate()."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


def translate_program(code):
    decoded = G.decode(code)
    stage = G.Stage.__new__(G.Stage)
    stage.pipe = Pipeline()
    stage.kind = 'ps'
    stage.lines, stage.sym = [], {}
    stage.used_s, stage.used_v, stage.masks = set(), set(), set()
    stage.attrs = {}
    stage.exec_known = True
    stage.ins = decoded
    stage.force_joined = {}
    stage.translate()
    return [asm for _o, _s, _n, asm in decoded], [lines for _, lines, _ in stage.blocks]


def refused_reason(code):
    try:
        translate_program(code)
    except G.Unsupported as error:
        return str(error)
    return None


READ = sop1(SOP1_MOV_B32, S10, S_SCC)          # s_mov_b32 s10, src_scc
TEST = sopc(SOPC_BITCMP1_B32, S4, S_THREE)     # s_bitcmp1_b32 s4, 3

# 1. The if/else shape: the else arm is entered only by the taken s_cbranch_scc0 edge. The then arm writes SCC
#    after its read, which reaches neither arm's read.
if_else = (TEST                                         # 0x00
           + branch(SOPP_CBRANCH_SCC0, 0x04, 0x14)      # 0x04 -> 0x14
           + READ                                       # 0x08
           + sopc(SOPC_CMP_EQ_U32, S4, S5)              # 0x0c
           + branch(SOPP_BRANCH, 0x10, 0x18)            # 0x10 -> 0x18
           + READ                                       # 0x14 (target of the taken edge only)
           + sopp(SOPP_ENDPGM))                         # 0x18 (join)
asm, blocks = translate_program(if_else)
check('if/else words decode as written',
      asm == ['s_bitcmp1_b32 s4, 3', 's_cbranch_scc0 label_0x14', 's_mov_b32 s10, src_scc', 's_cmp_eq_u32 s4, s5',
              's_branch label_0x18', 's_mov_b32 s10, src_scc', 's_endpgm'], repr(asm))
check('if/else: both arms read the SCC of s_bitcmp1_b32',
      blocks == [['scc = ((s4 >> (0x00000003u & 31)) & 1u) == 1u;'],
                 ['s10 = (scc ? 1u : 0u);', 'scc = s4 == s5;'],
                 ['s10 = (scc ? 1u : 0u);'], []], repr(blocks))

# 2. A join that a skipped block without an SCC write reaches: both edges carry the compare at 0x00.
skip_nop = (sopc(SOPC_CMP_EQ_U32, S4, S5)               # 0x00
            + branch(SOPP_CBRANCH_SCC0, 0x04, 0x0c)     # 0x04 -> 0x0c
            + sopp(SOPP_NOP)                            # 0x08
            + READ                                      # 0x0c (join)
            + sopp(SOPP_ENDPGM))
asm, blocks = translate_program(skip_nop)
check('join after a block without SCC writes reads the compare',
      blocks == [['scc = s4 == s5;'], [], ['s10 = (scc ? 1u : 0u);']], repr(blocks))

# 3. The same over s_cbranch_execz: the lane can skip a block the wave runs, which is harmless only because
#    the block writes no SCC.
skip_execz = (sopc(SOPC_CMP_EQ_U32, S4, S5)             # 0x00
              + branch(SOPP_CBRANCH_EXECZ, 0x04, 0x0c)  # 0x04 -> 0x0c
              + sopp(SOPP_NOP)                          # 0x08
              + READ                                    # 0x0c (join)
              + sopp(SOPP_ENDPGM))
asm, blocks = translate_program(skip_execz)
check('join after an execz-skipped block without SCC writes reads the compare',
      blocks == [['scc = s4 == s5;'], [], ['s10 = (scc ? 1u : 0u);']], repr(blocks))

# An SCC write inside a block s_cbranch_execz skips: the wave may run it while the lane skips it.
check('SCC write in an execz-skipped block refuses the read at the join',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)
                     + branch(SOPP_CBRANCH_EXECZ, 0x04, 0x0c)
                     + sopc(SOPC_CMP_EQ_U32, S6, S7)
                     + READ + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))
# A back edge into the target: the loop body's SCC write is not known when the target is reached.
check('back-edge target refuses the read',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)                       # 0x00
                     + READ                                              # 0x04 (loop head)
                     + sopc(SOPC_CMP_EQ_U32, S6, S7)                     # 0x08
                     + branch(SOPP_CBRANCH_SCC0, 0x0c, 0x04)             # 0x0c -> 0x04
                     + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))
# A lane-mask SCC: s_and_b32 with vcc_lo sets SCC from every lane's bit, which one lane does not see.
check('lane-mask SCC across a branch refuses the read',
      refused_reason(sop2(SOP2_AND_B32, S6, S4, VCC_LO)                  # 0x00
                     + branch(SOPP_CBRANCH_EXECZ, 0x04, 0x0c)            # 0x04 -> 0x0c
                     + sopp(SOPP_NOP)                                    # 0x08
                     + READ + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))          # 0x0c
# 8. An unmodeled writer on one path: s_wqm_b64 leaves the HLSL SCC unchanged.
check('unmodelled writer on one path refuses the read',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)
                     + branch(SOPP_CBRANCH_SCC0, 0x04, 0x0c)
                     + sop1(SOP1_WQM_B64, EXEC_LO, EXEC_LO)
                     + READ + sopp(SOPP_ENDPGM)).endswith(
                         'SCC producer s_wqm_b64 at +0x8 yields an SCC write that is not modelled on edge +0x8 -> +0xc'))


MASK = 0xFFFFFFFF
DISPATCH_CASES = ((0x00000000, 0x00000000), (0x00000008, 0x00000008), (0x00000008, 0x00000001),
                  (0xFFFFFFF7, 0xFFFFFFF7), (0xFFFFFFFF, 0x12345678), (0x12345678, 0x12345678))


def dispatch_hlsl(cases, fill):
    """One thread, one block per case: structured if/else, then the skipped-block join, on the translated lines."""
    _, if_blocks = translate_program(if_else)
    _, skip_blocks = translate_program(skip_nop)
    lines = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, (a, b) in enumerate(cases):
        out = 4 * (len(fill) + 2 * i)
        lines += ['    {',
                  f'        uint s4 = data.Load({8 * i}) ^ {fill[2 * i] ^ a:#010x}u;',
                  f'        uint s5 = data.Load({8 * i + 4}) ^ {fill[2 * i + 1] ^ b:#010x}u;',
                  '        uint s10 = 0u;', '        bool scc = true;']
        lines += ['        ' + t for t in if_blocks[0]]
        lines += ['        if (scc) {'] + ['            ' + t for t in if_blocks[1]]
        lines += ['        } else {'] + ['            ' + t for t in if_blocks[2]] + ['        }']
        lines.append(f'        data.Store({out}, s10);')
        lines += ['        s10 = 0u;', '        scc = true;']
        lines += ['        ' + t for t in skip_blocks[0] + skip_blocks[2]]
        lines += [f'        data.Store({out + 4}, s10);', '    }']
    return '\n'.join(lines + ['}']) + '\n'


def check_dispatch(dxc, runner, adapter):
    runner = str(Path(runner).resolve())
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(2 * len(DISPATCH_CASES)))
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        src, cso, dump = (Path(tmp) / ('src_scc_join' + e) for e in ('.hlsl', '.cso', '.bin'))
        src.write_text(dispatch_hlsl(DISPATCH_CASES, fill), encoding='ascii')
        compiled = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                  stderr=subprocess.PIPE, text=True)
        if compiled.returncode or compiled.stderr:
            raise AssertionError(compiled.stderr or f'DXC exited with {compiled.returncode}')
        words = len(fill) + 2 * len(DISPATCH_CASES)
        padded = (words + 3) // 4 * 4
        run = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * padded), str(dump)],
                             capture_output=True, text=True)
        print(run.stdout, end='')
        if run.returncode == 77:
            sys.exit(77)
        if run.returncode or run.stderr:
            raise AssertionError(run.stderr or f'raw UAV dump exited with {run.returncode}')
        got = struct.unpack(f'<{padded}I', dump.read_bytes())
    check('operand region unchanged', got[:len(fill)] == fill)
    for i, (a, b) in enumerate(DISPATCH_CASES):
        # CPU reference: the if/else reads the s_bitcmp1_b32 s4, 3 result on both arms; the join reads s4 == s5.
        want = ((a >> 3) & 1, int(a == b))
        have = got[len(fill) + 2 * i:len(fill) + 2 * i + 2]
        check(f'readback s4={a:#010x} s5={b:#010x}: got {have}, want {want}', have == want)
    if failures:
        sys.exit(1)


PIXEL_TAIL = """s_cselect_b32 s12, 1.0, 0.5
v_cvt_f32_u32_e32 v3, v0
v_add_f32_e32 v2, 1.0, v3
v_cvt_f32_u32_e32 v4, s10
v_mul_f32_e32 v2, v4, v2
v_mov_b32_e32 v5, s12
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
"""
PIXEL_TAIL_WORDS = 'f2f00c85000d067ef20604060a0c087e040504100c020a7e0f1800f802030405000081bf'
PIXEL_COMPARE = """s_cmp_eq_u32 7, {value}
s_mov_b32 s10, src_scc
"""
PIXEL_JOIN = """s_cmp_eq_u32 7, {value}
s_cbranch_scc0 2
s_cmp_eq_u32 3, 3
s_branch 1
s_xor_b32 s8, 13, 13
s_mov_b32 s10, src_scc
"""
PIXEL_LOGIC = """s_and_b32 s8, 2, {value}
s_mov_b32 s10, src_scc
"""
PIXEL_CARRY = """s_cmp_eq_u32 9, {value}
s_cbranch_scc0 2
s_bitcmp1_b32 4, 2
s_branch 1
s_lshr_b32 s8, 1, 1
s_add_u32 s11, -1, src_scc
s_mov_b32 s10, src_scc
"""
PIXEL_LOOP = """s_mov_b32 s10, 0
s_cmp_eq_u32 0, 1
s_mov_b32 s8, 0
s_mov_b32 s9, src_scc
s_add_u32 s10, s10, s9
s_add_u32 s8, s8, 1
s_cmp_lt_u32 s8, 3
s_cbranch_scc1 -5
"""
PIXEL_MISSING = """v_cmp_eq_u32_e32 vcc_lo, 0, v0
s_cbranch_vccz 1
s_cmp_eq_u32 3, 3
s_mov_b32 s10, src_scc
"""

# Each scalar branch is uniform. Separate draws exercise both arms, and each pixel
# has a distinct colour input from its packed position.
PIXEL_CASES = {
    'compare_equal': (PIXEL_COMPARE.format(value=7), '878706bffd038abe', int(7 == 7)),
    'compare_unequal': (PIXEL_COMPARE.format(value=8), '878806bffd038abe', int(7 == 8)),
    'join_compare': (PIXEL_JOIN.format(value=7), '878706bf020084bf838306bf010082bf8d8d0889fd038abe', int(3 == 3)),
    'join_logic': (PIXEL_JOIN.format(value=8), '878806bf020084bf838306bf010082bf8d8d0889fd038abe', int((13 ^ 13) != 0)),
    'logic_nonzero': (PIXEL_LOGIC.format(value=3), '82830887fd038abe', int((2 & 3) != 0)),
    'logic_zero': (PIXEL_LOGIC.format(value=1), '82810887fd038abe', int((2 & 1) != 0)),
    'carry_set': (PIXEL_CARRY.format(value=9), '898906bf020084bf84820dbf010082bf81810890c1fd0b80fd038abe',
                  (0xffffffff + ((4 >> 2) & 1)) >> 32),
    'carry_clear': (PIXEL_CARRY.format(value=8), '898806bf020084bf84820dbf010082bf81810890c1fd0b80fd038abe',
                    (0xffffffff + int((1 >> 1) != 0)) >> 32),
    'counted_loop': (PIXEL_LOOP, '80038abe808106bf800388befd0389be0a090a800881088008830abffbff85bf',
                     int(0 == 1) + sum(i < 3 for i in range(1, 3))),
    'missing_writer': (PIXEL_MISSING, '8000847d010086bf838306bffd038abe', None),
}


def pixel_words(prefix):
    return bytes.fromhex(prefix + PIXEL_TAIL_WORDS)


def translate_pixel(words):
    pipe = G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=(0x8000, 0x8000))
    return pipe.hlsl_ps()


MISSING_WRITER_REASON = ('src_scc with unverified SCC [operand read: s_mov_b32, destination sgpr] at +0xc: '
                         'entry path has no SCC producer on edge +0x4 -> +0xc')


def pixel_controls():
    accepted = {}
    for name, (_, prefix, scale) in PIXEL_CASES.items():
        try:
            hlsl = translate_pixel(pixel_words(prefix))
        except G.Unsupported as error:
            check(name, scale is None and str(error) == MISSING_WRITER_REASON, str(error))
        else:
            check(name, scale is not None, 'translated a path without an SCC writer')
            if scale is not None:
                accepted[name] = hlsl
    return accepted


def pixel_reference(scale):
    result = bytearray()
    # The loop's final latch is false even though two earlier bits sum to 2.
    alpha = 1.0 if scale == 1 else 0.5
    for y in range(4):
        for x in range(4):
            packed = x | (y << 16)
            result.extend(struct.pack('<4f', (packed + 1) * scale, packed, scale, alpha))
    return result


def check_pixel_draw(dxc, runner, shaders):
    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'scc_vs', 'vs_6_0', directory)

        def draw(name, hlsl, scale):
            ps = compile_hlsl(dxc, 'dxc', hlsl, name, 'ps_6_0', directory)
            expected = directory / (name + '.bin')
            expected.write_bytes(pixel_reference(scale))
            run = subprocess.run([runner, 'warp', '--startup-pixel-words', str(vs), str(ps),
                                  '4', '4', str(expected)], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(1 if failures else 77)
            return run

        for name, hlsl in shaders.items():
            run = draw(name, hlsl, PIXEL_CASES[name][2])
            check(name + ' WARP readback', run.returncode == 0 and 'all 16 pixels matched' in run.stdout,
                  run.stdout + run.stderr)

        hlsl = shaders['join_logic']
        producer = 'scc = s8 != 0;'
        if hlsl.count(producer) != 1:
            raise AssertionError('expected one logic SCC producer in the join fixture')
        mutant = hlsl.replace(producer, 'scc = s8 == 0;')
        run = draw('mutant_polarity', mutant, PIXEL_CASES['join_logic'][2])
        check('logic writer polarity mutation fails WARP readback',
              run.returncode != 0 and 'pixel mismatch (' in run.stderr and '16 of 16 pixels differ' in run.stderr,
              run.stdout + run.stderr)


def check_firstlane_draw(dxc, runner, hlsl, bindings, whole_quad):
    def float_word(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]

    def hex_words(words):
        return ','.join(f'{word:08x}' for word in words)

    # Include enough backing for the widest format case in the generated switch.
    values = (97.0, 19.25, 33.0, 41.0, 59.0, 71.0)
    fallback = 5.5
    buffer = hex_words(float_word(value) for value in values)
    word3 = (G.DESC_FORMAT_CODES['BUF_FMT_32_FLOAT'] << G.DESC_FORMAT_SHIFT) | 4
    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'firstlane_vs', 'vs_6_0', directory)

        def draw(name, shader, flag, bindings, reference, mismatch=False):
            expected = directory / (name + '.bin')
            expected.write_bytes(b''.join(
                struct.pack('<4f', *reference(x | (y << 16), y, flag))
                for y in range(8) for x in range(8)))
            constants = {('rootconst', 1, 0): flag, ('rootconst', 1, 1): float_word(fallback),
                         ('descword3', 0): word3}
            entry = hex_words(constants.get(key, 0) for key in bindings)
            ps = compile_hlsl(dxc, 'dxc', shader, name, 'ps_6_0', directory)
            run = subprocess.run([runner, 'warp', '--startup-pixel-buffer-words', str(vs), str(ps),
                                  '8', '8', str(expected), entry, buffer], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(1 if failures else 77)
            if mismatch:
                expected_result = run.returncode != 0 and '64 of 64 pixels differ' in run.stderr
            else:
                expected_result = run.returncode == 0 and 'all 64 pixels matched' in run.stdout
            expected_count = '64 differing pixels' if mismatch else '64 matching pixels'
            check(f'{name} WARP readback ({expected_count})', expected_result, run.stdout + run.stderr)

        def entry_reference(pos, _y, flag):
            picked = values[1] if flag else fallback
            return pos + picked, pos, 2.0 if flag else 4.0, 2.0 if flag else 0.5

        def whole_quad_reference(pos, y, flag):
            picked = values[2] if flag else fallback
            return pos + picked, pos, float(y >= 4), 1.0

        whole_quad_hlsl, swapped, whole_quad_bindings = whole_quad
        for flag in (0, 1):
            draw(f'firstlane_flag_{flag}', hlsl, flag, bindings, entry_reference)
            draw(f'whole_quad_flag_{flag}', whole_quad_hlsl, flag, whole_quad_bindings, whole_quad_reference)
        draw('mutant_whole_quad_swapped_arms', swapped, 1, whole_quad_bindings, whole_quad_reference, mismatch=True)

        for name, original, mutant in (
            ('mutant_join_scc', 's14 = scc ?', 's14 = !scc ?'),
            ('mutant_join_value', 's16 = s11;', 's16 = ud2;'),
            ('mutant_load_offset', '        a = 4;', '        a = 0;'),
        ):
            assert hlsl.count(original) == 1, (name, original)
            draw(name, hlsl.replace(original, mutant), 1, bindings, entry_reference, mismatch=True)



# The two diamonds run with opposite, wave-uniform compare results. Each false
# arm has two incoming comparisons, so it needs the reaching-path proof.
VERTEX_SOURCE = """\
s_cmp_eq_u32 17, 23
s_cbranch_scc0 4
s_cmp_lg_u32 2, 5
s_cbranch_scc0 2
s_mov_b32 s20, src_scc
s_branch 1
s_mov_b32 s20, src_scc
s_cmp_eq_u32 17, 17
s_cbranch_scc0 4
s_cmp_lg_u32 2, 5
s_cbranch_scc0 2
s_mov_b32 s21, src_scc
s_branch 1
s_mov_b32 s21, src_scc
v_mov_b32 v1, 0
v_cmp_eq_u32 vcc_lo, 0, v0
v_cndmask_b32 v1, v1, -4.0, vcc_lo
v_cmp_eq_u32 vcc_lo, 4, v0
v_cndmask_b32 v1, v1, 4.0, vcc_lo
v_cmp_eq_u32 vcc_lo, 2, v0
v_cndmask_b32 v2, -4.0, 4.0, vcc_lo
v_cmp_eq_u32 vcc_lo, 5, v0
v_cndmask_b32 v2, v2, 4.0, vcc_lo
v_cmp_le_u32 vcc_lo, 3, v0
v_mov_b32 v6, s20
v_cndmask_b32 v6, v6, s21, vcc_lo
v_cvt_f32_u32 v6, v6
v_cvt_f32_u32 v7, s20
v_mov_b32 v3, 0
v_mov_b32 v4, 1.0
exp pos0 v1, v2, v3, v4 done
exp param0 v6, v7, v3, v4
s_endpgm
"""
SCALAR_WORDS = bytes.fromhex(
    '919706bf040084bf828507bf020084bffd0394be010082bffd0394be'
    '919106bf040084bf828507bf020084bffd0395be010082bffd0395be'
)
VERTEX_TAIL_WORDS = bytes.fromhex(
    '8002027e8000847d010001d501efa9018400847d010001d501eda9018200847d'
    '020001d5f7eca9018500847d020001d502eda9018300867d14020c7e060001d5'
    '062ba801060d0c7e140c0e7e8002067ef202087ecf0800f8010203040f0200f806070304000081bf'
)
MISSING_WRITER_WORDS = bytes.fromhex(PIXEL_CASES['missing_writer'][1])
FALSE_ARM_SAVE = 's20 = (scc ? 1u : 0u);'
TRUE_ARM_SAVE = 's21 = (scc ? 1u : 0u);'


def check_missing_writer(stage, translate):
    try:
        translate(MISSING_WRITER_WORDS)
    except G.Unsupported as error:
        check(stage + ' missing writer', str(error) == MISSING_WRITER_REASON, str(error))
    else:
        check(stage + ' missing writer', False, 'translated a path without an SCC writer')


def check_saves(stage, hlsl):
    check(stage + ' SCC arms', hlsl.count(FALSE_ARM_SAVE) == 2 and hlsl.count(TRUE_ARM_SAVE) == 2)
    return hlsl


def flip_false_save(hlsl):
    # The second s20 assignment is the taken false arm of the first diamond.
    before, save, after = hlsl.rpartition(FALSE_ARM_SAVE)
    return before + save.replace('1u : 0u', '0u : 1u') + after


def translate_compute(prefix):
    rsrc2 = 6 << 1  # six user SGPRs, including the table pointer at s4:s5
    pipe = G.ComputePipeline(prefix + sopp(SOPP_ENDPGM), root_blob(), rsrc2, (1, 1, 1))
    return pipe.hlsl_cs()


def compute_controls():
    check_missing_writer('compute', translate_compute)
    try:
        hlsl = translate_compute(SCALAR_WORDS)
    except G.Unsupported as error:
        check('compute SCC arms', False, str(error))
        return None
    return check_saves('compute', hlsl)


def check_compute_join_dispatch(dxc, runner, adapter, hlsl):
    # Observe the scalar results at the end of the generated compute entry point.
    body, end = hlsl.rsplit('}', 1)
    assert end == '\n'
    shader = 'RWByteAddressBuffer result : register(u0);\n' + body + '    result.Store2(0, uint2(s20, s21));\n}\n'
    out = Path(here).parent / 'out'
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for name, source, expected_differences in (
                ('compute_scc', shader, ()), ('compute_scc_mutant', flip_false_save(shader), (0,))):
            cso = compile_hlsl(dxc, 'dxc', source, name, 'cs_6_0', directory)
            dump = directory / (name + '.bin')
            run = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', '16', str(dump)],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(1 if failures else 77)
            if run.returncode or run.stderr:
                raise AssertionError(run.stderr or f'raw UAV dump exited with {run.returncode}')
            got = struct.unpack_from('<2I', dump.read_bytes())
            want = (int(17 == 23), int(17 == 17))
            differing = tuple(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            check(name + ' GPU readback', differing == expected_differences, f'got {got}, want {want}')


def translate_vertex(prefix):
    pipe = G.Pipeline(prefix + VERTEX_TAIL_WORDS, PS_WORDS, root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def vertex_controls():
    check_missing_writer('vertex', translate_vertex)
    try:
        vs, ps = translate_vertex(SCALAR_WORDS)
    except G.Unsupported as error:
        check('vertex SCC arms', False, str(error))
        return None
    return check_saves('vertex', vs), ps


def vertex_reference():
    # Two oversized triangles cover the left and right halves. Red selects
    # the corresponding saved bit; green carries the left bit across the target.
    left, right = int(17 == 23), int(17 == 17)
    return b''.join(struct.pack('<4f', left if x < 2 else right, left, 0, 1)
                    for _ in range(4) for x in range(4))


# Hand-encoded scalar sequence: save two SCC values, traverse nested SCC
# diamonds, compare the joined copy, then read SCC after a second diamond.
# An earlier EXEC branch exercises restoration of a common reaching definition.
def nested_words(mutation=None, stale_bad=False):
    branch_to = lambda op, at, target: branch(op, at * 4, target * 4)
    words = [
        bytes.fromhex('8000847d'), branch_to(SOPP_CBRANCH_EXECZ, 1, 3), sopp(SOPP_NOP),
        sopc(SOPC_BITCMP1_B32, 9, 157), sop1(3, 15, S_SCC),
        sop2(SOP2_AND_B32, 16, 9, 129), sopc(6, S_SCC, 128), sop1(3, 18, S_SCC),
        sopc(1, 15, 128), branch_to(SOPP_CBRANCH_SCC0, 9, 17),
        sop1(3, 14, S_SCC), sopc(1, 18, 128), branch_to(SOPP_CBRANCH_SCC0, 12, 15),
        sop1(3, 15, S_SCC), branch_to(SOPP_BRANCH, 14, 18), sop1(3, 15, S_SCC),
        branch_to(SOPP_BRANCH, 16, 18), sop1(3, 15, S_SCC),
        sopc(1, 15, 128), branch_to(SOPP_CBRANCH_SCC0, 19, 22),
        sop1(3, 16, 129), branch_to(SOPP_BRANCH, 21, 23), sop1(3, 16, 130),
        sop1(3, 19, S_SCC)]
    # Fixed descriptor-table loads preserve SCC on both arms.
    def load(offset):
        return struct.pack('<II', 0xf4000000 | (2 << 18) | (20 << 6) | 4,
                           (125 << 25) | offset)
    insertions = {13: load(0x10), 15: load(0x30), 20: load(0x10), 22: load(0x30)}
    if stale_bad:
        # A nonuniform write is killed by the later SCC save on every path.
        insertions[3] = sop2(SOP2_AND_B32, 15, 9, VCC_LO)
    offsets, cursor = {}, 0
    for i, word in enumerate(words):
        offsets[i] = cursor
        cursor += len(insertions.get(i, b'')) + len(word)
    branches = {1: (SOPP_CBRANCH_EXECZ, 3), 9: (SOPP_CBRANCH_SCC0, 17),
                12: (SOPP_CBRANCH_SCC0, 15), 14: (SOPP_BRANCH, 18),
                16: (SOPP_BRANCH, 18), 19: (SOPP_CBRANCH_SCC0, 22), 21: (SOPP_BRANCH, 23)}
    for i, (op, target) in branches.items():
        words[i] = branch(op, offsets[i], offsets[target])
    if mutation:
        i, word = mutation
        words[i] = branch(SOPP_BRANCH, offsets[i], offsets[11]) if word == 'back' else word
    result = b''.join(insertions.get(i, b'') + word for i, word in enumerate(words))
    # This later write forces another analysis pass; it cannot reach the SCC read.
    return result + (sop2(SOP2_AND_B32, 15, 9, VCC_LO) + sop1(3, 15, 128) if stale_bad else b'')


NESTED_TAIL = bytes.fromhex(
    'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402'
    'f202087e80020a7e0f020c7e060d0c7e13020e7e070d0e7e8002107ef202127e'
    'cf0800f8010205040f0200f806070809000081bf')


def nested_translate(code):
    pipe = G.Pipeline(code + NESTED_TAIL, PS_WORDS, root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest(), pipe.uds


def nested_reference(value, wrong_scc=False):
    outer = (value >> 29) & 1
    inner = int((value & 1) == 0)
    saved = inner if outer else 0
    result = int(saved != 0) ^ int(wrong_scc)
    return b''.join(struct.pack('<4f', float(saved), float(result), 0.0, 1.0) for _ in range(16))


def nested_controls():
    code = nested_words()
    retry = G.Stage.retry_scc_edge_proof
    G.Stage.retry_scc_edge_proof = lambda self, reason: False
    try:
        try:
            nested_translate(code)
        except G.Unsupported as error:
            check('nested base refusal', 'src_scc with unverified SCC' in str(error), str(error))
        else:
            check('nested base refusal', False, 'previous proof accepted the fixture')
        old = translate_vertex(SCALAR_WORDS)
    finally:
        G.Stage.retry_scc_edge_proof = retry
    check('unchanged vertex output', old == translate_vertex(SCALAR_WORDS))
    try:
        shaders = nested_translate(code)
    except G.Unsupported as error:
        check('nested SCC edge', False, str(error))
        return None
    check('nested SCC edge', True)
    check('nested root binding', ('rootconst', 1, 0) in shaders[3], shaders[2])
    try:
        nested_translate(nested_words(stale_bad=True))
    except G.Unsupported as error:
        check('overwritten nonuniform write', False, str(error))
    else:
        check('overwritten nonuniform write', True)
    for index, word, diagnostic in (
            (3, sop2(SOP2_AND_B32, 6, 9, VCC_LO), 'non-uniform'),
            (22, sop2(SOP2_AND_B32, 6, 9, VCC_LO), 'non-uniform'),
            (16, 'back', 'non-uniform')):
        mutant = nested_words((index, word))
        try:
            nested_translate(mutant)
        except G.Unsupported as error:
            check('nested refusal ' + str(index), diagnostic in str(error), str(error))
        else:
            check('nested refusal ' + str(index), False, 'translated unsafe path')
    for value, expected in ((0, (0.0, 0.0)), (1 << 29, (1.0, 1.0)), ((1 << 29) | 1, (0.0, 0.0))):
        check('nested CPU path %x' % value, struct.unpack_from('<2f', nested_reference(value)) == expected)
    check('mutant discriminated', nested_reference(1 << 29) != nested_reference(1 << 29, True))
    return shaders


def check_nested_draw(dxc, runner, shaders):
    vs, ps, manifest, bindings = shaders
    out = Path(here).parent / 'out'
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vertex = compile_hlsl(dxc, 'dxc', vs, 'nested_vs', 'vs_6_0', directory)
        pixel = compile_hlsl(dxc, 'dxc', ps, 'nested_ps', 'ps_6_0', directory)
        check('root binding', ('rootconst', 1, 0) in bindings, manifest)
        for value in (0, 1 << 29, (1 << 29) | 1):
            expected = directory / ('nested_%x.bin' % value)
            expected.write_bytes(nested_reference(value))
            constants = [value if source == ('rootconst', 1, 0) else 0 for source in bindings]
            entry = ','.join('%08x' % word for word in constants)
            run = subprocess.run([runner, 'warp', '--startup-pixel-buffer-words', str(vertex), str(pixel),
                                  '4', '4', str(expected), entry, '00000000'], capture_output=True, text=True)
            if run.returncode == 77:
                sys.exit(1 if failures else 77)
            report = run.stdout + run.stderr
            check('nested WARP %x' % value, run.returncode == 0 and 'all 16 pixels matched' in report, report)


def check_vertex_draw(dxc, runner, shaders):
    vs, ps = shaders
    # Flipping only the false-arm save changes green at every pixel.
    mutant = flip_false_save(vs)
    out = Path(here).parent / 'out'
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        pixel = compile_hlsl(dxc, 'dxc', ps, 'vertex_scc_ps', 'ps_6_0', directory)
        expected = directory / 'vertex_scc.bin'
        expected.write_bytes(vertex_reference())
        for name, source, mismatches in (('vertex_scc', vs, 0), ('vertex_scc_mutant', mutant, 16)):
            vertex = compile_hlsl(dxc, 'dxc', source, name, 'vs_6_0', directory)
            run = subprocess.run([runner, 'warp', '--raster-reference', str(vertex), str(pixel),
                                  '4', '4', str(expected), 'none', 'cw'], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(1 if failures else 77)
            report = run.stdout + run.stderr
            expected_result = f'{mismatches} of 16 pixels differ' if mismatches else 'all 16 pixels matched'
            check(name + ' WARP readback', run.returncode == int(bool(mismatches))
                  and expected_result in report.splitlines(), report)


if len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
    compute = compute_controls()
    if not failures:
        check_dispatch(*sys.argv[2:])
        check_compute_join_dispatch(*sys.argv[2:], compute)
elif len(sys.argv) == 4 and sys.argv[1] == 'draw':
    shaders = pixel_controls()
    firstlane_hlsl, bindings = firstlane_pixel.check_translation()
    vertex = vertex_controls()
    nested = nested_controls()
    whole_quad = firstlane_pixel.check_whole_quad_translation()
    if not failures:
        check_pixel_draw(*sys.argv[2:], shaders)
        check_firstlane_draw(*sys.argv[2:], firstlane_hlsl, bindings, whole_quad)
        check_vertex_draw(*sys.argv[2:], vertex)
        check_nested_draw(*sys.argv[2:], nested)
elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
    for name, (source, prefix, _) in PIXEL_CASES.items():
        check(name + ' encoding', assemble(sys.argv[2], source + PIXEL_TAIL) == pixel_words(prefix))
    check('vertex encoding', assemble(sys.argv[2], VERTEX_SOURCE) == SCALAR_WORDS + VERTEX_TAIL_WORDS)
elif len(sys.argv) == 1:
    pixel_controls()
    vertex_controls()
    nested_controls()
    compute_controls()
else:
    sys.exit(__doc__)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
