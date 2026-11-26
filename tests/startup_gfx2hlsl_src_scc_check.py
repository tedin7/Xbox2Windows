"""The `src_scc` scalar source in tools/startup/gfx2hlsl.py (issue #1820).

usage: startup_gfx2hlsl_src_scc_check.py
       startup_gfx2hlsl_src_scc_check.py dispatch <dxc> <offscreen_check> <warp|hardware>
       startup_gfx2hlsl_src_scc_check.py alias_dispatch <dxc> <offscreen_check> warp

The census contained `s_mul_i32 s28, 1.0, src_scc`, refused as `operand 'src_scc'` (#870, issue #1820).
Source code 253 reads the SCC bit as the integer value 0 or 1; the generated shader carries SCC as a
`bool scc`, so `Stage.u()` lowers `src_scc` to `(scc ? 1u : 0u)` when a modeled writer has set SCC on
every reaching path in the current block, and refuses when the incoming SCC is unverified (at shader entry,
after an unmodeled SCC writer such as `s_add_u32`, after descriptor-elided SCC, or across control-flow
joins).

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`):
- SOP2 (`[31:30] == 0b10`, op in `w0[29:23]`, sdst in `w0[22:16]`, src1 in `w0[15:8]`, src0 in `w0[7:0]`,
  optional literal dword when a source is 255)
- SOPC (`[31:23] == 0b101111110`, op in `w0[22:16]`, src1 in `w0[15:8]`, src0 in `w0[7:0]`)
- SOPP (`[31:23] == 0b101111111`, op in `w0[22:16]`, simm16 in `w0[15:0]`)
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


SOP2_ADD_U32 = 0x00
SOP2_AND_B32 = 0x0E
SOP2_MUL_I32 = 0x26
SOPC_CMP_EQ_U32 = 0x06
SOPP_NOP = 0x00
SOPP_ENDPGM = 0x01
SOPP_CBRANCH_SCC0 = 0x04

S_ZERO = 128        # inline integer 0
S_ONE = 129         # inline integer 1
S_SCC = 253         # the src_scc scalar source code
S_1_0 = 242         # the inline float constant 1.0 (its raw bits are 0x3f800000)
LITERAL = 255
S1, S4, S5, S27, S28, S29 = 1, 4, 5, 27, 28, 29


def encode_sop2(op, sdst, ssrc0, ssrc1, literal=None):
    w0 = ((0b10 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16)
          | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))
    code = struct.pack('<I', w0)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


def encode_sopc(op, ssrc0, ssrc1, literal=None):
    w0 = (0xBF000000 | ((op & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))
    code = struct.pack('<I', w0)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


def encode_sopp(op, simm=0):
    w0 = 0xBF800000 | ((op & 0x7F) << 16) | (simm & 0xFFFF)
    return struct.pack('<I', w0)


class Pipeline:
    """Just enough pipeline state for Stage.translate()."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


def translate_program(code):
    """Decode and run the full Stage.translate() CFG/block pass on synthetic bytes."""
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
    body = ' '.join(line for _, lines, _ in stage.blocks for line in lines)
    return [asm for _o, _s, _n, asm in decoded], body, stage


def refused_reason(code):
    try:
        translate_program(code)
    except G.Unsupported as error:
        return str(error)
    return None


MASK = 0xFFFFFFFF
SOP1_MOV_B32 = 0x03
# (a, b) pairs: s_cmp_eq_u32 sets SCC = (a == b), then s_mov_b32 s3, src_scc stores it.
DISPATCH_CASES = ((0, 0), (1, 1), (0xFFFFFFFF, 0xFFFFFFFF), (0x12345678, 0x12345678),
                  (0, 1), (1, 0), (0x80000000, 0x00000001), (0xFFFFFFFF, 0xFFFFFFFE))


def encode_sop1(op, sdst, ssrc):
    return struct.pack('<I', (0b101111101 << 23) | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) | (ssrc & 0xFF))


class Probe(G.Stage):
    """Minimal Stage for one-instruction-at-a-time lowering (same shape as the s_ff1 check)."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True
        self.kind = 'vs'
        self.past_entry = False


def lower_cmp_then_read_scc():
    """Lower `s_cmp_eq_u32 s1, s2` then `s_mov_b32 s3, src_scc` with the real handlers."""
    code = encode_sopc(SOPC_CMP_EQ_U32, S1, 2) + encode_sop1(SOP1_MOV_B32, 3, S_SCC)
    stage = Probe()
    for _o, _n, ins, _asm in G.decode(code):
        stage.one(ins, _asm)
    return stage.lines


def dispatch(dxc, runner, adapter, lines, directory, name):
    n = len(DISPATCH_CASES)
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(2 * n))
    hlsl = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, (a, b) in enumerate(DISPATCH_CASES):
        hlsl.extend((
            '    {',
            '        uint s1 = data.Load(%d) ^ %#010xu;' % (8 * i, fill[2 * i] ^ a),
            '        uint s2 = data.Load(%d) ^ %#010xu;' % (8 * i + 4, fill[2 * i + 1] ^ b),
            '        uint s3;',
            '        bool scc;',
        ))
        hlsl.extend('        ' + line for line in lines)
        hlsl.append('        data.Store(%d, s3);' % (4 * (2 * n + i)))
        hlsl.append('    }')
    hlsl.append('}')
    src, cso, output = (directory / (name + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(hlsl) + '\n', encoding='ascii')
    compiled = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                              stderr=subprocess.PIPE, text=True)
    if compiled.returncode or compiled.stderr:
        raise AssertionError(compiled.stderr or 'DXC exited with %d' % compiled.returncode)
    words = 3 * n
    dump_words = (words + 3) // 4 * 4
    ran = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * dump_words), str(output)],
                         capture_output=True, text=True)
    print(ran.stdout, end='')
    if ran.returncode == 77:
        sys.exit(77)
    if ran.returncode or ran.stderr:
        raise AssertionError(ran.stderr or 'raw UAV dump exited with %d' % ran.returncode)
    actual = struct.unpack('<%dI' % dump_words, output.read_bytes())
    assert actual[:2 * n] == fill, 'operand region changed: %s' % (actual[:2 * n],)
    return actual[2 * n:words]


def check_dispatch(dxc, runner, adapter):
    lines = lower_cmp_then_read_scc()
    check('dispatch lowering is the compare then the SCC read',
          lines == ['scc = s1 == s2;', 's3 = (scc ? 1u : 0u);'], repr(lines))
    runner = str(Path(runner).resolve())
    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        got = dispatch(dxc, runner, adapter, lines, directory, 'src_scc')
        for i, (a, b) in enumerate(DISPATCH_CASES):
            want = 1 if a == b else 0  # CPU reference: src_scc reads the SCC bit as 0 or 1
            check('dispatch src_scc after s_cmp_eq_u32 %#x, %#x' % (a, b), got[i] == want,
                  'got %#x want %#x' % (got[i], want))
        check('dispatch covers both SCC=0 and SCC=1', {0, 1} <= set(got), repr(got))
        # Negative control: the same harness with the lowering mutated to (scc ? 0u : 1u) must read back
        # the wrong value on every case, so a wrong src_scc lowering cannot pass the checks above.
        mutated = [line.replace('(scc ? 1u : 0u)', '(scc ? 0u : 1u)') for line in lines]
        check('negative control mutates the lowering', mutated != lines, repr(mutated))
        bad = dispatch(dxc, runner, adapter, mutated, directory, 'src_scc_mutated')
        for i, (a, b) in enumerate(DISPATCH_CASES):
            check('negative control reads the wrong value for %#x, %#x' % (a, b),
                  bad[i] == (0 if a == b else 1), 'got %#x' % bad[i])
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('dispatch ok')


def alias_shader(op):
    from startup_gfx2hlsl_descriptor_scc_gpu import ENTRY_ROOT, program

    # One low dword updates SCC, then two in-place dwords consume and update it.
    guest = f"""\
s_mov_b32 s31, s5
s_mov_b32 s32, 0
s_load_dwordx4 s[36:39], s[31:32], 0
s_load_dwordx4 s[44:47], s[31:32], 0x30
s_waitcnt lgkmcnt(0)
s_buffer_load_dwordx4 s[48:51], s[36:39], 0
s_waitcnt lgkmcnt(0)
{op} s48, s48, 1
{op} s49, s49, src_scc
{op} s50, s50, src_scc
s_mov_b32 s51, src_scc
v_mov_b32 v7, s48
v_mov_b32 v8, s49
v_mov_b32 v9, s50
v_mov_b32 v10, s51
buffer_store_dwordx4 v[7:10], v0, s[44:47], 0 offen
s_endpgm
"""
    refusals = []
    pipe = G.ComputePipeline(b'', ENTRY_ROOT, 6 << 1, (1, 1, 1), decoded=program(guest), refusals=refusals)
    source = pipe.hlsl_cs()
    assert not refusals, refusals
    return source


def late_snapshot(source, op):
    """Write the result from live inputs, then wrongly snapshot the clobbered input for SCC."""
    for dst in ('s49', 's50'):
        snapshot = f'    {{ uint scc_src0 = {dst}, scc_src1 = (scc ? 1u : 0u);'
        assignment = f'    {dst} = scc_src0 {op} scc_src1;'
        old = snapshot + '\n' + assignment
        new = f'    {{ {dst} = {dst} {op} (scc ? 1u : 0u);\n' + snapshot.replace('{ ', '  ', 1)
        assert source.count(old) == 1, source
        source = source.replace(old, new, 1)
    return source


def check_alias_dispatch(dxc, runner, adapter):
    from startup_gfx2hlsl_descriptor_scc_gpu import compile_shader, compute_case, verify_readback

    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for name, op, step, cases in (
            ('s_add_u32', '+', 1, ((MASK, MASK, 7), (MASK, MASK, MASK), (41, MASK, 6))),
            ('s_sub_u32', '-', -1, ((0, 0, 7), (0, 0, 0), (41, 0, 6))),
        ):
            source = alias_shader(name)
            good = compile_shader(dxc, directory, name, 'cs_6_0', source)
            bad = compile_shader(dxc, directory, name + '_late', 'cs_6_0', late_snapshot(source, op))
            for i, words in enumerate(cases):
                total = sum(word << (32 * j) for j, word in enumerate(words)) + step
                expected = tuple((total >> (32 * j)) & MASK for j in range(3))
                expected += (int(total < 0 or total >= 1 << 96),)
                case = directory / 'alias.case'
                compute_case(case, (), expected, words)
                command = [runner, adapter, '--compute-store-sized-case']
                verify_readback([*command, str(good), str(case)])
                # In the first two cases the middle word consumes carry or borrow and passes it on.
                # The late snapshot loses it, so the first mismatch is the high word.
                verify_readback([*command, str(bad), str(case)], 'word=2 expected=' if i < 2 else None)


if len(sys.argv) == 5 and sys.argv[1] in ('dispatch', 'alias_dispatch'):
    run = check_dispatch if sys.argv[1] == 'dispatch' else check_alias_dispatch
    run(*sys.argv[2:])
    sys.exit(0)
elif len(sys.argv) != 1:
    sys.exit(__doc__)


# 1. At shader entry, SCC is unverified: reading src_scc before any modeled writer must be refused.
check('entry src_scc in s_mul_i32 is refused',
      refused_reason(encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC) + encode_sopp(SOPP_ENDPGM))
      .startswith('src_scc with unverified SCC'))
check('the refusal names its path, opcode and destination class',
      refused_reason(encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC) + encode_sopp(SOPP_ENDPGM))
      == 'src_scc with unverified SCC [operand read: s_mul_i32, destination sgpr]')
check('entry src_scc in s_and_b32 is refused',
      refused_reason(encode_sop2(SOP2_AND_B32, S28, S_SCC, S5) + encode_sopp(SOPP_ENDPGM))
      .startswith('src_scc with unverified SCC'))

# 2. Modeled compare -> census s_mul_i32 s28, 1.0, src_scc: decodes and lowers using the SCC bit.
cmp_census = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
              + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)
              + encode_sopp(SOPP_ENDPGM))
asm, emitted, stage = translate_program(cmp_census)
check('encodes and decodes compare + census form',
      asm[:2] == ['s_cmp_eq_u32 0, 0', 's_mul_i32 s28, 1.0, src_scc'], repr(asm))
check('compare + census form lowers to the SCC bit',
      'scc = 0x00000000u == 0x00000000u;' in emitted and 's28 = 0x3f800000u * (scc ? 1u : 0u);' in emitted,
      emitted)
check('low16_only=True returns the same 0/1 expression',
      stage.u('src_scc', low16_only=True) == '(scc ? 1u : 0u)')

# 3. Modeled bitwise writer -> src_scc in the first source position -> next instruction reads updated SCC.
bitwise_seq = (encode_sop2(SOP2_AND_B32, S27, S4, S5)
               + encode_sop2(SOP2_AND_B32, S28, S_SCC, S5)
               + encode_sop2(SOP2_MUL_I32, S29, S_1_0, S_SCC)
               + encode_sopp(SOPP_ENDPGM))
asm, emitted, _stage = translate_program(bitwise_seq)
check('encodes and decodes bitwise sequence with src_scc in first position',
      asm[:3] == ['s_and_b32 s27, s4, s5', 's_and_b32 s28, src_scc, s5', 's_mul_i32 s29, 1.0, src_scc'],
      repr(asm))
check('bitwise sequence reads and updates SCC cleanly',
      's28 = (scc ? 1u : 0u) & s5; scc = s28 != 0;' in emitted
      and 's29 = 0x3f800000u * (scc ? 1u : 0u);' in emitted,
      emitted)

# 4. Distinguishing carry case: compare sets SCC=0, then s_add_u32 0xffffffff + 1 wraps and sets hardware
#    carry SCC=1. The add emits that carry, so the src_scc read afterwards sees it (not the stale compare result).
carry_seq = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ONE)
             + encode_sop2(SOP2_ADD_U32, S1, LITERAL, S_ONE, literal=0xFFFFFFFF)
             + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)
             + encode_sopp(SOPP_ENDPGM))
_asm, emitted, _stage = translate_program(carry_seq)
carry_line = 'scc = (0xffffffffu + 0x00000001u) < (0xffffffffu);'
check('s_add_u32 carry after compare is emitted ahead of the src_scc read',
      carry_line in emitted and emitted.index(carry_line) < emitted.index('0x3f800000u * (scc ? 1u : 0u)'), emitted)

# 5. s_add_u32 reading src_scc as an input before overwriting SCC is accepted (input read precedes write),
#    while a second src_scc after s_add_u32 is refused.
add_reads_scc = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
                 + encode_sop2(SOP2_ADD_U32, S1, S_SCC, S_ONE)
                 + encode_sopp(SOPP_ENDPGM))
asm, emitted, _stage = translate_program(add_reads_scc)
read_line = 's1 = (scc ? 1u : 0u) + 0x00000001u;'
check('s_add_u32 reading incoming verified src_scc is accepted',
      read_line in emitted, emitted)
# The destination reads the incoming SCC before this add's own carry replaces it (the carry is emitted after the write).
check('s_add_u32 src_scc operand is read before the add writes SCC',
      emitted.index(read_line) < emitted.index('scc = ((scc ? 1u : 0u) + 0x00000001u) < ((scc ? 1u : 0u));'), emitted)
for opcode, name in ((SOP2_ADD_U32, 's_add_u32'), (0x01, 's_sub_u32')):
    for dst in (106, 124, 125, 126):
        reason = refused_reason(encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
                                + encode_sop2(opcode, dst, S1, S_SCC)
                                + encode_sopp(SOPP_ENDPGM))
        check(f'{name} with special destination {dst} keeps its refusal',
              reason == f'{name} reads src_scc with a destination that is also a source or a special register', reason)
    source = alias_shader(name)
    check(f'{name} snapshots both aliased instructions', source.count('{ uint scc_src0 =') == 2)
    late_snapshot(source, '+' if name == 's_add_u32' else '-')

_asm, emitted, _stage = translate_program(
    encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
    + encode_sop2(SOP2_ADD_U32, S1, S_SCC, S_ONE)
    + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)
    + encode_sopp(SOPP_ENDPGM))
carry_line = 'scc = ((scc ? 1u : 0u) + 0x00000001u) < ((scc ? 1u : 0u));'
check('reading src_scc after s_add_u32 sees the add\'s carry',
      carry_line in emitted and emitted.index(carry_line) < emitted.index('0x3f800000u * (scc ? 1u : 0u)'), emitted)

# 6. Across a control-flow join: branch skips an SCC write to target 0x0c, so the two paths into the join carry
#    different SCC writes and src_scc there is refused unless a modeled writer runs inside the joined block first.
#    (A skipped block without an SCC write keeps SCC proven: startup_gfx2hlsl_src_scc_join_check.py.)
join_refused = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)     # 0x00
                + encode_sopp(SOPP_CBRANCH_SCC0, simm=1)          # 0x04 -> label_0x0c (skip 1 dword at 0x08)
                + encode_sopc(SOPC_CMP_EQ_U32, S4, S5)            # 0x08
                + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)    # 0x0c (branch target / join)
                + encode_sopp(SOPP_ENDPGM))                       # 0x10
check('src_scc across a control-flow join is refused',
      refused_reason(join_refused).startswith('src_scc with unverified SCC'),
      repr(refused_reason(join_refused)))

join_rewritten = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)   # 0x00
                  + encode_sopp(SOPP_CBRANCH_SCC0, simm=1)        # 0x04 -> 0x0c
                  + encode_sopp(SOPP_NOP, simm=0)                 # 0x08
                  + encode_sopc(SOPC_CMP_EQ_U32, S4, S5)          # 0x0c (modeled writer after join)
                  + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)  # 0x10
                  + encode_sopp(SOPP_ENDPGM))
asm, emitted, _stage = translate_program(join_rewritten)
check('src_scc after a modeled writer in the joined block is accepted',
      's28 = 0x3f800000u * (scc ? 1u : 0u);' in emitted, emitted)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
