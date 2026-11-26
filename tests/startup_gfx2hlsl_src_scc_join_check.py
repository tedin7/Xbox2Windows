"""`src_scc` at a branch target in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_src_scc_join_check.py [dispatch <dxc> <offscreen_check> <adapter>]

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

Negative controls keep the refusal: an SCC write in one arm, an SCC write in a block `s_cbranch_execz`
skips, a back edge into the target, a lane-mask SCC (`s_and_b32` with `vcc_lo`), and an unmodelled
writer (`s_add_u32`) on one path.

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

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


SOP1_MOV_B32 = 0x03
SOP2_ADD_U32 = 0x00
SOP2_AND_B32 = 0x0E
SOPC_CMP_EQ_U32 = 0x06
SOPC_BITCMP1_B32 = 0x0D
SOPP_NOP, SOPP_ENDPGM, SOPP_BRANCH, SOPP_CBRANCH_SCC0, SOPP_CBRANCH_EXECZ = 0x00, 0x01, 0x02, 0x04, 0x08

S_SCC = 253
S_THREE = 131      # inline integer 3
VCC_LO = 106
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

# Negative controls: each keeps `src_scc with unverified SCC`.
# 4. An SCC write in the skipped arm: the two edges carry different writes.
check('SCC write in one arm refuses the read at the join',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)                       # 0x00
                     + branch(SOPP_CBRANCH_SCC0, 0x04, 0x0c)             # 0x04 -> 0x0c
                     + sopc(SOPC_CMP_EQ_U32, S6, S7)                     # 0x08
                     + READ + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))          # 0x0c
# 5. An SCC write inside a block s_cbranch_execz skips: the wave may run it while the lane skips it.
check('SCC write in an execz-skipped block refuses the read at the join',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)
                     + branch(SOPP_CBRANCH_EXECZ, 0x04, 0x0c)
                     + sopc(SOPC_CMP_EQ_U32, S6, S7)
                     + READ + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))
# 6. A back edge into the target: the loop body's SCC write is not known when the target is reached.
check('back-edge target refuses the read',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)                       # 0x00
                     + READ                                              # 0x04 (loop head)
                     + sopc(SOPC_CMP_EQ_U32, S6, S7)                     # 0x08
                     + branch(SOPP_CBRANCH_SCC0, 0x0c, 0x04)             # 0x0c -> 0x04
                     + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))
# 7. A lane-mask SCC: s_and_b32 with vcc_lo sets SCC from every lane's bit, which one lane does not see.
check('lane-mask SCC across a branch refuses the read',
      refused_reason(sop2(SOP2_AND_B32, S6, S4, VCC_LO)                  # 0x00
                     + branch(SOPP_CBRANCH_EXECZ, 0x04, 0x0c)            # 0x04 -> 0x0c
                     + sopp(SOPP_NOP)                                    # 0x08
                     + READ + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))          # 0x0c
# 8. An unmodelled SCC writer on one path: s_add_u32 sets the carry the model does not emit.
check('unmodelled writer on one path refuses the read',
      refused_reason(sopc(SOPC_CMP_EQ_U32, S4, S5)
                     + branch(SOPP_CBRANCH_SCC0, 0x04, 0x0c)
                     + sop2(SOP2_ADD_U32, S6, S4, S5)
                     + READ + sopp(SOPP_ENDPGM)).startswith(UNVERIFIED))


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


if len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
    check_dispatch(*sys.argv[2:])
    print('PASS')
    sys.exit(0)
elif len(sys.argv) != 1:
    sys.exit(__doc__)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
