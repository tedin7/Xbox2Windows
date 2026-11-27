"""A vertex stage loads a resource heap record inside a branch (tools/startup/gfx2hlsl.py, dynamic_heap_record).

  (no arguments)   translation controls, pure Python
  dxc <dxc.exe>    also compile the translated vertex stage as vs_6_6 with that DXC

The register-offset `s_load_dwordx4 sD, s[ptr:ptr+1], sOFF` was refused in any program that had a branch target, so a
vertex stage that loads the record in a branch arm could not be translated. The fixture is the flat-parameter vertex
stage with a record load inside an arm, entered when a root constant is non-zero. A second uniform branch splits the
arm, so the buffer read that uses the descriptor is in a later block than the load. The index temporary must
therefore be declared once at the top of the function and assigned in the block of the load.

Controls, each of which must stay refused:
- a pixel stage with the same load in a branch (only the vertex stage declares the index outside its block);
- a record index that is a lane mask, not a wave-uniform value;
- a record loaded in one arm and read after the join, where the other path holds no descriptor.

The words are encoded from the SOP1 / SOP2 / SOPC / SOPP, SMEM and MUBUF field layouts that
`tools/disasm_rdna2.py` decodes. The first test below checks them against the decoded text.
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

from startup_gfx2hlsl_dynamic_heap_check import G, root_blob, compile_shader
from startup_flat_param_check import PS_WORDS, VS_WORDS


def sop1(op, dst, src):
    return struct.pack('<I', 0xBE800000 | (dst << 16) | (op << 8) | src)


def sopc(op, a, b):
    return struct.pack('<I', 0xBF000000 | (op << 16) | (b << 8) | a)


def sopp(op, imm=0):
    return struct.pack('<I', 0xBF800000 | (op << 16) | (imm & 0xFFFF))


def smem(op, sdata, sbase, soffset):
    return struct.pack('<II', 0xF4000000 | (op << 18) | (sdata << 6) | (sbase >> 1), soffset << 25)


def mubuf(op, vdata, vaddr, srsrc):
    """MUBUF with offen: the vector address register holds the byte offset."""
    return struct.pack('<II', 0xE0000000 | (op << 18) | (1 << 12),
                       0x80000000 | ((srsrc >> 2) << 16) | (vdata << 8) | vaddr)


S_MOV_B32, S_ENDPGM, S_WAITCNT, S_CBRANCH_SCC1, S_CMP_EQ_U32 = 0x03, 0x01, 0x0C, 0x05, 0x06
S_LOAD_DWORDX4, BUFFER_LOAD_DWORD = 0x02, 0x0C
LGKM0, VM0, VCC_LO, ZERO, ONE = 0xC07F, 0x3F70, 106, 128, 129
V_MOV_V11_0 = bytes.fromhex('8002167e')
S_HEAP, S_OFFSET, S_COND, S_RECORD = 10, 8, 9, 16  # heap pointer, byte offset and a root constant, s[16:19]

HEAD, EXPORTS = VS_WORDS[:-20], VS_WORDS[-20:]  # everything before the exports; the two exports and s_endpgm


def assemble(items):
    """items: bytes, or 'join' (a branch to the end of the arm), or ('back', n) (a branch back to item n)."""
    sizes = [4 if not isinstance(i, bytes) else len(i) for i in items]
    at = [sum(sizes[:n]) for n in range(len(items) + 1)]
    out = []
    for n, item in enumerate(items):
        if item == 'join':
            item = sopp(S_CBRANCH_SCC1, (at[-1] - (at[n] + 4)) // 4)
        elif isinstance(item, tuple):
            item = sopp(S_CBRANCH_SCC1, (at[item[1]] - (at[n] + 4)) // 4)
        out.append(item)
    return b''.join(out)


COND = sopc(S_CMP_EQ_U32, S_COND, ZERO)
COND_ONE = sopc(S_CMP_EQ_U32, S_COND, ONE)
LOAD = smem(S_LOAD_DWORDX4, S_RECORD, S_HEAP, S_OFFSET)
READ = mubuf(BUFFER_LOAD_DWORD, 8, 11, S_RECORD)
SETUP = [sop1(S_MOV_B32, S_HEAP + 1, ZERO), V_MOV_V11_0]


def vertex(arm):
    return HEAD + assemble(SETUP + [COND, 'join'] + arm) + EXPORTS


# The load in the arm, a second branch, then the read in the next block.
ARM = [LOAD, sopp(S_WAITCNT, LGKM0), COND_ONE, 'join', READ, sopp(S_WAITCNT, VM0)]
# The load in the arm; the read after the join sees one path with a descriptor and one without.
AFTER_JOIN = [LOAD, sopp(S_WAITCNT, LGKM0)]
AFTER_JOIN_VS = HEAD + assemble(SETUP + [COND, 'join'] + AFTER_JOIN) + READ + sopp(S_WAITCNT, VM0) + EXPORTS
# The record index is a lane mask copied into an SGPR.
MASK_INDEX_VS = HEAD + assemble(SETUP + [sop1(S_MOV_B32, 12, VCC_LO), COND, 'join'] + [
    smem(S_LOAD_DWORDX4, S_RECORD, S_HEAP, 12), sopp(S_WAITCNT, LGKM0), READ, sopp(S_WAITCNT, VM0)]) + EXPORTS
# The pixel stage: the same load and read in an arm.
PS_BRANCHED = assemble([sop1(S_MOV_B32, S_HEAP + 1, ZERO), COND, 'join', LOAD, sopp(S_WAITCNT, LGKM0),
                        bytes.fromhex('8002087e'), mubuf(BUFFER_LOAD_DWORD, 5, 4, S_RECORD),
                        sopp(S_WAITCNT, VM0)]) + bytes.fromhex('0f1800f805050505') + sopp(S_ENDPGM)

# The load and the read inside a loop: every pass reloads the record, so the index is assigned again each pass.
LOOP_VS = HEAD + assemble(SETUP + [LOAD, sopp(S_WAITCNT, LGKM0), READ, sopp(S_WAITCNT, VM0), COND_ONE, ('back', 2)]) + EXPORTS

REFUSAL = "s_load with a register offset through ('resheap',)"


def translate(vs, ps=PS_WORDS):
    pipe = G.Pipeline(vs, ps, root_blob())
    return pipe, pipe.hlsl_vs(), pipe.hlsl_ps()


def refusal(vs, ps=PS_WORDS):
    try:
        translate(vs, ps)
    except G.Unsupported as reason:
        return str(reason)
    return None


def expect(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH'))
    if not condition:
        sys.exit('%s %s' % (label, detail))


def main(dxc=None):
    vs = vertex(ARM)
    decoded = [asm for _, _, _, asm in G.decode(vs)]
    tail = ['s_mov_b32 s11, 0', 'v_mov_b32_e32 v11, 0', 's_cmp_eq_u32 s9, 0', 's_cbranch_scc1 label_0x%x',
            's_load_dwordx4 s[16:19], s[10:11], s8', 's_waitcnt lgkmcnt(0)', 's_cmp_eq_u32 s9, 1',
            's_cbranch_scc1 label_0x%x', 'buffer_load_dword v8, v11, s[16:19], 0 offen', 's_waitcnt vmcnt(0)']
    start = len(G.decode(HEAD))
    expect('fixture decodes', [re.sub(r'label_0x[0-9a-f]+', 'label_0x%x', asm) for asm in decoded[start:start + 10]] == tail,
           str(decoded[start:start + 10]))
    pipe, vs_source, ps_source = translate(vs)
    expect('manifest names the dynamic heap', 'dynamic_heap\n' in pipe.manifest())
    lines = vs_source.splitlines()
    names = sorted(set(re.findall(r'heap_index_[0-9a-f]+', vs_source)))
    expect('one index temporary', len(names) == 1, str(names))
    name = names[0]
    declared = [i for i, line in enumerate(lines) if re.fullmatch(r'\s*uint %s = 0;' % name, line)]
    assigned = [i for i, line in enumerate(lines) if line.strip().startswith(name + ' = (')]
    used = [i for i, line in enumerate(lines) if 'x2w_heap_load(' + name in line]
    blocks = [i for i, line in enumerate(lines) if re.match(r'\s*(\[branch\] )?(else )?if \(pc == \d+\) \{', line)]
    expect('declared once before the first block', len(declared) == 1 and declared[0] < blocks[0])
    expect('assigned once, read later', len(assigned) == 1 and used and min(used) > assigned[0])
    expect('the read is in another block than the load', any(assigned[0] < b < min(used) for b in blocks))
    if dxc:
        with tempfile.TemporaryDirectory() as tmp:
            compile_shader(dxc, Path(tmp), 'vs_6_6', vs_source)
        print('dxc vs_6_6 compile: ok')

    reason = refusal(VS_WORDS, PS_BRANCHED)
    expect('pixel stage branch refused', reason == REFUSAL, str(reason))
    reason = refusal(MASK_INDEX_VS)
    expect('lane-mask index refused', reason == REFUSAL, str(reason))
    reason = refusal(AFTER_JOIN_VS)
    expect('record read after the join refused', reason is not None, str(reason))
    loop_source = translate(LOOP_VS)[1]
    loop_lines = loop_source.splitlines()
    set_at = [i for i, line in enumerate(loop_lines) if re.match(r'\s*heap_index_[0-9a-f]+ = \(', line)]
    read_at = [i for i, line in enumerate(loop_lines) if 'x2w_heap_load(heap_index_' in line]
    expect('loop: assigned and read in the same pass', len(set_at) == 1 and len(read_at) == 1 and read_at[0] == set_at[0] + 2
           and '[loop] for' in loop_source)
    print('heap record in a branch: controls matched')


if __name__ == '__main__':
    main(*sys.argv[2:3] if sys.argv[1:2] == ['dxc'] else ())
