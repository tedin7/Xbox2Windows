"""A vertex stage loads a resource heap record inside a branch (tools/startup/gfx2hlsl.py, dynamic_heap_record).

  (no arguments)   translation controls, pure Python
  dxc <dxc.exe>    also compile the translated vertex stage as vs_6_6 with that DXC
  warp <offscreen_check> <dxc.exe>   draw both vertex fixtures on WARP and compare the target (with a mutant)

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
import hashlib
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from startup_gfx2hlsl_dynamic_heap_check import G, IMAGE_HEAD, IMAGE_LOAD, IMAGE_TAIL, root_blob, compile_shader
from startup_gfx2hlsl_dynamic_heap_check import draw as draw_pixel
from startup_gfx2hlsl_getpc_refusal_check import pipeline as vector_table_pipeline, vertex as vector_table_vertex
from startup_gfx2hlsl_heap_record_check import root_blob as table_root_blob
from startup_flat_param_check import PS_WORDS, VS_WORDS


def sop1(op, dst, src):
    return struct.pack('<I', 0xBE800000 | (dst << 16) | (op << 8) | src)


def sop2(op, dst, a, b):
    return struct.pack('<I', 0x80000000 | (op << 23) | (dst << 16) | (b << 8) | a)


def sop2_literal(op, dst, src, literal):
    return sop2(op, dst, src, 255) + struct.pack('<I', literal)


def sopc(op, a, b):
    return struct.pack('<I', 0xBF000000 | (op << 16) | (b << 8) | a)


def sopp(op, imm=0):
    return struct.pack('<I', 0xBF800000 | (op << 16) | (imm & 0xFFFF))


def smem(op, sdata, sbase, soffset, imm=0):
    return struct.pack('<II', 0xF4000000 | (op << 18) | (sdata << 6) | (sbase >> 1), (soffset << 25) | imm)


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
IMAGE_RECORD = smem(0x03, S_RECORD, S_HEAP, S_OFFSET)
IMAGE_COORDS = bytes.fromhex('8002187e80021a7e')  # v_mov_b32 v12, 0; v_mov_b32 v13, 0
IMAGE_READ = bytes.fromhex('080100f00c080400')  # image_load v8, v[12:13], s[16:23] dmask:0x1


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

# Part 2: the loaded record is rewritten into a raw buffer descriptor before the read (the structured-to-raw
# conversion): stride in bits 16..29 of word 1 becomes the multiplier of NUM_RECORDS, the stride field is cleared and
# the out-of-range select in word 3 is set. Encoded with llvm-mc (gfx1030, wave32) and checked by the decode test.
S_BFE_STRIDE = bytes.fromhex('11ff949310000e00')      # s_bfe_u32 s20, s17, 0xe0010
S_MAX_STRIDE = bytes.fromhex('14819484')              # s_max_u32 s20, s20, 1
S_MUL_RECORDS = bytes.fromhex('12141293')             # s_mul_i32 s18, s18, s20
S_CLEAR_STRIDE = bytes.fromhex('11ff1187ffff00c0')    # s_and_b32 s17, s17, 0xc000ffff
S_CLEAR_OOB = bytes.fromhex('13ff1387ffffffcf')       # s_and_b32 s19, s19, 0xcfffffff
S_SET_OOB = bytes.fromhex('13ff138800000030')         # s_or_b32 s19, s19, 0x30000000
REWRITE = [S_BFE_STRIDE, S_MAX_STRIDE, S_MUL_RECORDS, S_CLEAR_STRIDE, S_CLEAR_OOB, S_SET_OOB]
# Every word of the same quadruple is needed: each control drops or changes one step.
REWRITE_CONTROLS = {
    'no out-of-range select': REWRITE[:5],
    'stride field kept': [S_BFE_STRIDE, S_MAX_STRIDE, S_MUL_RECORDS, S_CLEAR_OOB, S_SET_OOB],
    'stride mask too wide': [S_BFE_STRIDE, S_MAX_STRIDE, S_MUL_RECORDS, S_CLEAR_STRIDE.replace(b'\xff\xff\x00\xc0', b'\xff\xff\x01\xc0'),
                             S_CLEAR_OOB, S_SET_OOB],
    'records not scaled': [S_BFE_STRIDE, S_MAX_STRIDE, S_CLEAR_STRIDE, S_CLEAR_OOB, S_SET_OOB],
    'select value changed': [S_BFE_STRIDE, S_MAX_STRIDE, S_MUL_RECORDS, S_CLEAR_STRIDE, S_CLEAR_OOB,
                             S_SET_OOB.replace(b'\x00\x00\x00\x30', b'\x00\x00\x00\x20')],
}
# The same read, with the rewritten descriptor.
REWRITE_ARM = [LOAD, sopp(S_WAITCNT, LGKM0)] + REWRITE + [READ, sopp(S_WAITCNT, VM0)]
# The result of the read copied into all four parameter words, so the target holds one value four times (a plain
# copy: no denormal or NaN words to flush).
COPY_V8 = bytes.fromhex('08030c7e08030e7e0803127e')   # v_mov_b32 v6, v8 / v7, v8 / v9, v8


def vertex_copy(arm):
    return HEAD + assemble(SETUP + [COND, 'join'] + arm) + COPY_V8 + EXPORTS


def vertex_image(record=IMAGE_RECORD, read=IMAGE_READ):
    return vertex_copy([record, sopp(S_WAITCNT, LGKM0), COND_ONE, 'join', IMAGE_COORDS, read, sopp(S_WAITCNT, VM0)])


def vertex_image_loop(back=4, next_counter=b''):
    # Load a fixed root-table descriptor, then narrow EXEC using the vertex ID. The tested triangle's IDs all pass.
    setup = (sop1(S_MOV_B32, 14, 11) + sop1(S_MOV_B32, 15, ZERO) + smem(0x02, 24, 14, 125)
             + sopp(S_WAITCNT, LGKM0) + sop1(S_MOV_B32, 13, ZERO) + sop1(S_MOV_B32, 45, ZERO)
             + IMAGE_COORDS + bytes.fromhex('8002107e7e039ebe8000267d'))  # sum=0, save EXEC, v_cmpx_le_i32 0, v0
    body = (sop2(0x1e, 51, 45, 130) + smem(0x08, 23, 24, 51) + sopp(S_WAITCNT, LGKM0)
            + sop2(0x1e, 23, 23, 133) + smem(0x03, 16, 12, 23) + sopp(S_WAITCNT, LGKM0)
            + bytes.fromhex('080100f00c0e0400') + sopp(S_WAITCNT, VM0)  # image_load v14
            + struct.pack('<I', (0x03 << 25) | (8 << 17) | (8 << 9) | (256 + 14))
            + sop2(0x00, 45, 45, ONE) + sopc(S_CMP_EQ_U32, 45, 9) + next_counter)
    return (HEAD + setup + sopp(8, len(body) // 4 + 1) + body + sopp(back, -(len(body) // 4 + 1))
            + sop1(S_MOV_B32, 126, 30) + COPY_V8 + EXPORTS)


def translate_image_loop(words):
    pipe = G.Pipeline(words, PS_WORDS, table_root_blob(True))
    return pipe, pipe.hlsl_vs(), pipe.hlsl_ps()


def descriptor_word_vertex(load):
    # v8 = float(s25 + 3), where s25 is a copy of the third descriptor word.
    return (HEAD + load + sopp(S_WAITCNT, LGKM0) + sop1(S_MOV_B32, 25, 18)
            + bytes.fromhex('8302107e1910104a080d107e') + COPY_V8 + EXPORTS)


def descriptor_word_control():
    load = sop1(S_MOV_B32, 14, 11) + sop1(S_MOV_B32, 15, ZERO) + smem(0x02, 16, 14, 125)
    pipe, vs, ps = translate_image_loop(descriptor_word_vertex(load))
    index = pipe.uds.index(('descword', 1, 8))
    assert f'udsrc {index} descword 1 8' in pipe.manifest()
    assert f'v8 = ud{index} + v8;' in vs
    reason = refusal(descriptor_word_vertex(SETUP[0] + LOAD))
    assert reason and 'cannot materialize descriptor word 2' in reason, reason
    return pipe, vs, ps, index


DIM_BASE = sop2_literal(0x27, 25, 19, 0x4000c)
DIM_WIDTH = (sop2_literal(0x27, 26, 17, 0x2001e) + sop2(0x00, 26, 26, ONE)
             + sop2_literal(0x0e, 27, 18, 0x3fff) + sop2(0x2f, 27, 27, 26))
DIM_HEIGHT = sop2_literal(0x27, 28, 18, 0x10000e) + sop2(0x00, 28, 28, ONE)
DIM_CLAMP = (sop2(0x20, 27, 27, 25) + sop2(0x09, 27, 27, ONE)
             + sop2(0x20, 28, 28, 25) + sop2(0x09, 28, 28, ONE))
DIM_EXPORT = bytes.fromhex('1b0c087e1c0c0a7e') + bytes.fromhex(IMAGE_TAIL)  # convert s27 and s28 to float


def image_dimensions_null_shader(compare, on_null=ZERO, on_bound=27):
    # Give the null path a nonzero candidate width so dropping the select changes the readback.
    width_min = sop2(0x09, 27, 27, ONE)  # s_max_u32 s27, s27, 1
    select = sop2(0x0a, 27, on_null, on_bound)  # s_cselect_b32 s27, 0, s27
    return (bytes.fromhex(IMAGE_HEAD) + DIM_BASE + DIM_WIDTH + DIM_HEIGHT + DIM_CLAMP
            + width_min + compare + select + bytes.fromhex(IMAGE_LOAD) + DIM_EXPORT)


def image_dimensions_null_control():
    eq_zero = sopc(0x00, 17, ZERO)  # s_cmp_eq_i32 s17, 0
    pipe, vs, ps = translate(VS_WORDS, image_dimensions_null_shader(eq_zero))
    assert pipe.ps.retry_heap_dimensions and ps.count('.GetDimensions(0u,') == 3
    assert re.search(r'scc = heap_w_\w+ == 0u;', ps)
    assert re.search(r's27 = scc \? 0x00000000u : \(max\(heap_w_\w+, 0x00000001u\)\);', ps)
    lg_zero = sopc(0x07, ZERO, 17)  # s_cmp_lg_u32 0, s17
    lg = translate(VS_WORDS, image_dimensions_null_shader(lg_zero, 27, ZERO))[2]
    assert re.search(r'scc = heap_w_\w+ != 0u;', lg)
    for compare in (sopc(0x00, 17, ONE),  # s_cmp_eq_i32 s17, 1
                    sopc(0x02, 17, ZERO)):  # s_cmp_gt_i32 s17, 0
        reason = refusal(VS_WORDS, image_dimensions_null_shader(compare))
        assert reason and 'cannot materialize descriptor word 1' in reason, reason
    return pipe, vs, ps


def image_dimensions_control():
    head, image = bytes.fromhex(IMAGE_HEAD), bytes.fromhex(IMAGE_LOAD)
    body = DIM_BASE + DIM_WIDTH + DIM_HEIGHT + DIM_CLAMP
    pipe, vs, ps = translate(VS_WORDS, head + body + image + DIM_EXPORT)
    assert 'dynamic_heap\ndynamic_heap_images\n' in pipe.manifest()
    assert pipe.ps.retry_heap_dimensions and ps.count('.GetDimensions(0u,') == 2
    assert re.search(r'v4 = U\(float\(heap_w_\w+\)\);', ps)
    assert re.search(r'v5 = U\(float\(heap_h_\w+\)\);', ps)
    split_width = DIM_WIDTH[:-4] + sop2(0x1e, 27, 27, 130) + sop2(0x00, 27, 26, 27)
    short_height = sop2_literal(0x27, 28, 18, 0xe000e) + sop2(0x00, 28, 28, ONE)
    split = translate(VS_WORDS, head + DIM_BASE + split_width + short_height + DIM_CLAMP + image + DIM_EXPORT)[2]
    assert split.count('.GetDimensions(0u,') == 2
    for bad in (sop2_literal(0x27, 27, 17, 0x90014) + image,  # format
                DIM_WIDTH + image,  # raw mip-zero width when the view may start at a later mip
                body,  # no MIMG to determine the type
                body + sop1(S_MOV_B32, 16, ZERO) + image):  # a write to the record precedes the first MIMG
        reason = refusal(VS_WORDS, head + bad + DIM_EXPORT)
        assert reason and 'descriptor' in reason, reason
    other_dim = bytes.fromhex('100100f0080e0400')  # image_load with a 3D resource operand
    reason = refusal(VS_WORDS, head + body + image + DIM_EXPORT[:-12] + other_dim + bytes.fromhex(IMAGE_TAIL))
    assert reason and 'image dimension disagrees' in reason, reason
    return pipe, vs, ps


EMBEDDED_WORDS = (0x3fa00000, 0x40200000, 0x40800000, 0x41000000, 0x41800000)
EMBEDDED_MASK_WRITE = bytes.fromhex('1c0206d4f0020200')  # v_cmp_ge_f32_e64 s28, 0.5, |v0|


def embedded_scalar_vertex(*, carry=0, flags=0x31016fac, bound=20, displacement=0, escape=b'', before_copy=b''):
    flags_word = sop1(S_MOV_B32, 31, 255) + struct.pack('<I', flags)
    prefix = (sop1(0x1f, 20, 0) + struct.pack('<I', 0xb0160000 | bound)
              + sop1(S_MOV_B32, 23, 255) + struct.pack('<I', flags)
              + sop2(0x1e, 12, 8, 130) + sop1(S_MOV_B32, 15, 12))
    high = b'' if carry is None else sop2(0x04, 21, 21, ZERO + carry)
    reads = (smem(0x09, 12, 28, 12, 1) + smem(0x08, 25, 20, 15, 8) + smem(0x08, 26, 20, 15, 12))
    tail = (high + before_copy + struct.pack('<I', 0xb01e000c) + flags_word + sop1(0x04, 28, 20)
            + reads + sopp(S_WAITCNT, LGKM0) + escape
            + bytes.fromhex('0c020c7e0d020e7e1902107e1a02127e') + EXPORTS)
    delta = len(prefix) - 4 + 8 + len(tail) + displacement
    return HEAD + prefix + sop2_literal(0x00, 20, 20, delta) + tail


def embedded_scalar_control():
    data = struct.pack('<5I', *EMBEDDED_WORDS)
    code = embedded_scalar_vertex()
    _, vs, _ = translate(code + data + bytes.fromhex('000000f8'))  # invalid if decoded instead of treated as data
    assert 'static const uint embedded_vs_scalar[5]' in vs
    assert 'embedded_vs_sload(a, 1u, 12u)' in vs and 'embedded_vs_sload(a, 5u, 12u)' in vs
    assert 'embedded_vs_sload(s15, 8u, 20u)' in vs and 'embedded_vs_sload(s15, 12u, 20u)' in vs
    assert '(scalar_offset + imm) & ~3u' in vs and 'a > bound - 4u' in vs
    assert 'embedded_vs_scalar[5]' in translate(embedded_scalar_vertex(carry=None) + data)[1]
    cmp_code = embedded_scalar_vertex(escape=EMBEDDED_MASK_WRITE)
    cmp_pipe, cmp_vs, cmp_ps = translate(cmp_code + data)
    assert 'embedded_vs_scalar[5]' in cmp_vs and re.search(r'\bs28 = .*abs\(', cmp_vs)

    mov_scc = sop1(S_MOV_B32, 21, 253)  # s_mov_b32 s21, src_scc
    read_s21 = sopc(0x01, 21, ZERO)  # s_cmp_lg_i32 s21, 0
    both_arms = COND + sopp(S_CBRANCH_SCC1, 2) + mov_scc + sopp(2, 1) + mov_scc
    joined = embedded_scalar_vertex(escape=both_arms + read_s21)
    assert 'embedded_vs_scalar[5]' in translate(joined + data)[1]

    cmov = sop1(0x05, 21, ONE)  # s_cmov_b32 s21, 1
    read_s28 = sopc(0x01, 28, ZERO)
    for label, escape in (
        ('unwritten copy', read_s28),
        ('conditional move before read', COND + cmov + read_s21),
        ('mask write before read', EMBEDDED_MASK_WRITE + read_s28),
        ('ordinary rewrite in one arm', COND + sopp(S_CBRANCH_SCC1, 1) + mov_scc + read_s21),
    ):
        bad = embedded_scalar_vertex(escape=escape)
        assert G.embedded_scalar_buffer_proof(bad + data, G.decode(bad)) is None, label
        reason = refusal(bad + data)
        assert reason and 'shader load address' in reason, (label, reason)

    for bad in (embedded_scalar_vertex(flags=0x21016fac), embedded_scalar_vertex(bound=28),
                embedded_scalar_vertex(displacement=-4), embedded_scalar_vertex(carry=1),
                embedded_scalar_vertex(escape=bytes.fromhex('1402107e')),  # v_mov_b32 v8, s20
                embedded_scalar_vertex(before_copy=sopp(2, 0))):  # branch separates setup and access
        assert G.embedded_scalar_buffer_proof(bad + data, G.decode(bad)) is None
    with patch.object(G, 'embedded_scalar_buffer_proof', return_value=None):
        reason = refusal(code + data)
    assert reason and 'shader load address' in reason, reason

    old = vector_table_pipeline(vector_table_vertex())
    assert [hashlib.sha256(x.encode()).hexdigest() for x in (old.hlsl_vs(), old.hlsl_ps(), old.manifest())] == [
        'c316062e876cfa7970566e2f472296f04290961ae518555fee43b8b760ec8a0a',
        '6ae9d0b469c599e2cbbe58a2bac0c74cf9d9536ce9a105228dc8792c21fcdd52',
        '9a6837358af0301f1dc4f3c4b36cdf8cfbf81be2f4f11c269c8027b4efd50839',
    ]
    return cmp_pipe, cmp_vs, cmp_ps


# A divergent block (v_cmpx, s_cbranch_execz) first, so the program is not wave-uniform throughout; then a root constant
# picks the arm. The first arm writes the index register from a register nothing defines, the else arm shifts the
# index it had before the branch and loads the record. Encoded with llvm-mc (gfx1030, wave32):
#   s_mov_b32 s11, 0 / v_mov_b32 v11, 0 / s_mov_b32 s30, exec_lo / v_cmpx_eq_i32 0, v0 / s_cbranch_execz skip /
#   s_mov_b32 s12, 1 / skip: s_mov_b32 exec_lo, s30 / s_and_b32 s12, s8, 0x7fffffff / s_cmp_eq_u32 s9, 0 /
#   s_cbranch_scc1 other / s_mov_b32 s12, s14 / s_branch join / other: s_lshl_b32 s12, s12, 5 /
#   s_load_dwordx4 s[16:19], s[10:11], s12 / s_waitcnt lgkmcnt(0) / buffer_load_dword v8, v11, s[16:19], 0 offen /
#   s_waitcnt vmcnt(0) / join: v_mov_b32 v6, v8 / v7, v8 / v9, v8
ELSE_ARM = bytes.fromhex(
    '80038bbe8002167e7e039ebe8000247d010088bf81038cbe1e03febe08ff0c87ffffff7f098006bf020085bf0e038cbe070082bf'
    '0c850c8f050408f4000000187fc08cbf001030e00b080480703f8cbf08030c7e08030e7e0803127e')
ELSE_VS = HEAD + ELSE_ARM + EXPORTS
# The same, with the index register written from the undefined register before the branch (s_and replaced).
ELSE_VARYING_VS = HEAD + bytes.fromhex(
    '80038bbe8002167e7e039ebe8000247d010088bf81038cbe1e03febe0e038cbe098006bf020085bf0e038cbe070082bf'
    '0c850c8f050408f4000000187fc08cbf001030e00b080480703f8cbf08030c7e08030e7e0803127e') + EXPORTS
# The first arm falls into the else block (no s_branch, s_cbranch_scc1 skips one word): the block is entered by two
# paths, one of which wrote the index from the undefined register, so the load stays refused.
ELSE_FALLS_VS = HEAD + ELSE_ARM.replace(bytes.fromhex('020085bf0e038cbe070082bf'),
                                        bytes.fromhex('010085bf0e038cbe')) + EXPORTS

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
    vertex_image_control()
    vertex_image_loop_control()
    descriptor_word_control()
    image_dimensions_control()
    image_dimensions_null_control()
    embedded_scalar_control()
    vs = vertex(ARM)
    decoded = [asm for _, _, _, asm in G.decode(vs)]
    tail = ['s_mov_b32 s11, 0', 'v_mov_b32_e32 v11, 0', 's_cmp_eq_u32 s9, 0', 's_cbranch_scc1 label_0x%x',
            's_load_dwordx4 s[16:19], s[10:11], s8', 's_waitcnt lgkmcnt(0)', 's_cmp_eq_u32 s9, 1',
            's_cbranch_scc1 label_0x%x', 'buffer_load_dword v8, v11, s[16:19], 0 offen', 's_waitcnt vmcnt(0)']
    start = len(G.decode(HEAD))
    expect('fixture decodes', [re.sub(r'label_0x[0-9a-f]+', 'label_0x%x', asm) for asm in decoded[start:start + 10]] == tail,
           str(decoded[start:start + 10]))
    pipe, vs_source, ps_source = translate(vs)
    manifest = pipe.manifest()
    expect('manifest names the buffer heap', 'dynamic_heap\n' in manifest and 'dynamic_heap_images\n' not in manifest)
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
    rewrite_controls()
    else_arm_controls()


def rewrite_controls():
    decoded = [asm for _, _, _, asm in G.decode(assemble(REWRITE))]
    expect('rewrite decodes', decoded == [
        's_bfe_u32 s20, s17, 0xe0010', 's_max_u32 s20, s20, 1', 's_mul_i32 s18, s18, s20',
        's_and_b32 s17, s17, 0xc000ffff', 's_and_b32 s19, s19, 0xcfffffff', 's_or_b32 s19, s19, 0x30000000'], str(decoded))
    pipe, vs_source, _ = translate(vertex_copy(REWRITE_ARM))
    expect('rewritten record translates', 'dynamic_heap\n' in pipe.manifest() and 'x2w_heap_load(' in vs_source)
    for label, steps in REWRITE_CONTROLS.items():
        reason = refusal(vertex_copy([LOAD, sopp(S_WAITCNT, LGKM0)] + steps + [READ, sopp(S_WAITCNT, VM0)]))
        expect('rewrite refused: ' + label, reason == 'dynamic resource heap descriptor requires four intact words', str(reason))
    print('heap record rewrite: controls matched')


def else_arm_controls():
    pipe, vs_source, _ = translate(ELSE_VS)
    expect('else arm translates', 'dynamic_heap\n' in pipe.manifest() and 'x2w_heap_load(' in vs_source)
    reason = refusal(ELSE_VARYING_VS)
    expect('else arm: index varying before the branch refused', reason == REFUSAL, str(reason))
    expect('else arm: fall-through fixture changed', len(ELSE_FALLS_VS) == len(ELSE_VS) - 4)
    reason = refusal(ELSE_FALLS_VS)
    expect('else arm: first arm falls into it, refused', reason == REFUSAL, str(reason))
    print('heap record in an else arm: controls matched')


def draw(offscreen, vs, ps, offset, reference, count, negative=False, cond=2, heap_count=2, extra=None):
    """One draw of the vertex stage that reads the heap record named by `offset` (ud1) when ud2 is `cond`."""
    entry = [0] * count + [heap_count]
    entry[1], entry[2] = offset, cond
    for index, word in (extra or {}).items():
        entry[index] = word
    as_hex = lambda words: ','.join('%08x' % word for word in words)
    expected = reference if isinstance(reference, list) else [reference] * 4
    result = subprocess.run([offscreen, 'warp', '--startup-dynamic-heap', vs, ps, as_hex(entry),
                             as_hex(expected)], capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    expect('draw offset %d%s' % (offset, ' (mutant)' if negative else ''), result.returncode == int(negative),
           result.stdout + result.stderr)
    if negative:
        expect('mutant reads another record', 'startup_entry_words mismatch' in result.stderr, result.stderr)


def mutate(source, old, new):
    expect('mutation site ' + old, source.count(old) == 1, source)
    return source.replace(old, new)


def vertex_image_control():
    pipe, vs, ps = translate(vertex_image())
    assert 'dynamic_heap\ndynamic_heap_images\n' in pipe.manifest()
    assert re.search(r'uint [^;]*\bheap_index_\w+\b[^;]*;', vs)
    assert 'Texture2D<float4> heap_tex_' in vs and '.Load(int3(' in vs
    bad = vertex_image(record=sop1(S_MOV_B32, 12, VCC_LO) + smem(0x03, S_RECORD, S_HEAP, 12))
    reason = refusal(bad)
    assert reason and 'register offset through' in reason, reason
    sample = bytes.fromhex('080180f00c08040c')  # image_sample with implicit derivatives
    reason = refusal(vertex_image(read=sample))
    assert reason and 'dynamic resource heap views require a raw buffer load' in reason, reason
    return pipe, vs, ps


def warp_image(offscreen, dxc):
    pipe, source, ps_source = vertex_image_control()
    count = len(pipe.uds)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        draw(offscreen, vs, ps, 96, 0x3fa00000, count, heap_count=5)
        draw(offscreen, vs, ps, 128, 0x41800000, count, heap_count=5)
        wrong = compile_shader(dxc, directory, 'vs_6_6', mutate(source, '(ud1 + 0u) >> 5;', '(ud1 + 32u) >> 5;'))
        draw(offscreen, wrong, ps, 96, 0x3fa00000, count, negative=True, heap_count=5)


def vertex_image_loop_control():
    words = vertex_image_loop()
    pipe, vs, ps = translate_image_loop(words)
    assert pipe.vs.loop_uniform is not None
    assert 'dynamic_heap\n' in pipe.manifest() and 'srv 0' in pipe.manifest()
    assert '[loop] for' in vs and '.Load(' in vs and re.search(r'\(s23 \+ 0u\) >> 5;', vs)
    with patch.object(G.Stage, 'retry_loop_uniform', return_value=False):
        try:
            translate_image_loop(words)
        except G.Unsupported as reason:
            assert 'register offset through' in str(reason), str(reason)
        else:
            raise AssertionError('the loop translated without the retry')
    for bad in (vertex_image_loop(back=6), vertex_image_loop(back=8),
                vertex_image_loop(next_counter=sop1(S_MOV_B32, 45, VCC_LO))):
        try:
            translate_image_loop(bad)
        except G.Unsupported as reason:
            assert 'register offset through' in str(reason), str(reason)
        else:
            raise AssertionError('the loop with an unproved edge translated')
    old, old_vs, old_ps = translate(LOOP_VS)
    assert old.vs.loop_uniform is None
    assert [hashlib.sha256(x.encode()).hexdigest() for x in (old_vs, old_ps, old.manifest())] == [
        'b6729f75ee319bec8177de1d2f3e68496bfc3b9548bbc483dcef8c684e4297cd',
        '834e3ce1d54b3ffbaae9da19647ae684f64a31b20e2fff7a12c4634664fc16f9',
        'b9eb3cb0811ceb855942b96467c0dea4a9b5c6e040da2fdc2d2de49a738d7b25',
    ]
    return pipe, vs, ps


def warp_image_loop(offscreen, dxc):
    pipe, source, ps_source = vertex_image_loop_control()
    count = len(pipe.uds)
    expected, = struct.unpack('<I', struct.pack('<f', sum((1.25, 16.0))))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        draw(offscreen, vs, ps, 0, expected, count, heap_count=8)
        wrong = compile_shader(dxc, directory, 'vs_6_6', mutate(source, '(s23 + 0u) >> 5;', '(s23 + 0u) >> 6;'))
        draw(offscreen, wrong, ps, 0, expected, count, negative=True, heap_count=8)


def warp_descriptor_word(offscreen, dxc):
    pipe, source, ps_source, index = descriptor_word_control()
    num_records = 37
    expected, = struct.unpack('<I', struct.pack('<f', num_records + 3))
    count = len(pipe.uds)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        draw(offscreen, vs, ps, 0, expected, count, extra={index: num_records})
        wrong = compile_shader(dxc, directory, 'vs_6_6', mutate(source, f'ud{index} + v8', f'ud{index} + v8 + 1u'))
        draw(offscreen, wrong, ps, 0, expected, count, negative=True, extra={index: num_records})


def warp_image_dimensions(offscreen, dxc):
    pipe, vs_source, source = image_dimensions_control()
    count = len(pipe.uds)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', vs_source)
        ps = compile_shader(dxc, directory, 'ps_6_6', source)
        for slot, base, full_width, full_height, blue, alpha in ((3, 0, 37, 11, 4, 8), (4, 1, 43, 15, 64, 128)):
            expected = list(struct.unpack('<4I', struct.pack('<4f', max(full_width >> base, 1),
                                                             max(full_height >> base, 1), blue, alpha)))
            draw_pixel(offscreen, vs, ps, 32 * slot, expected, count, heap_count=5)
        wrong = compile_shader(dxc, directory, 'ps_6_6', source.replace('.GetDimensions(0u,', '.GetDimensions(1u,'))
        draw_pixel(offscreen, vs, wrong, 32 * slot, expected, count, negative=True, heap_count=5)
        null_pipe, null_vs, null_source = image_dimensions_null_control()
        vs = compile_shader(dxc, directory, 'vs_6_6', null_vs)
        ps = compile_shader(dxc, directory, 'ps_6_6', null_source)
        count = len(null_pipe.uds)
        cases = [(slot, list(struct.unpack('<4I', struct.pack('<4f', *reference))))
                 for slot, reference in ((3, (37, 11, 4, 8)), (5, (0, 0, 0, 0)))]
        for slot, expected in cases:
            draw_pixel(offscreen, vs, ps, 32 * slot, expected, count, heap_count=6)
        compare = re.search(r'heap_w_\w+ == 0u', null_source).group()
        wrong = compile_shader(dxc, directory, 'ps_6_6', mutate(null_source, compare, compare.replace('==', '!=')))
        for slot, expected in cases:
            draw_pixel(offscreen, vs, wrong, 32 * slot, expected, count, negative=True, heap_count=6)


def embedded_reference(index):
    def read(offset, size):
        word = (4 * index + offset) // 4
        return EMBEDDED_WORDS[word] if 4 * (word + 1) <= size else 0

    return [read(1, 12), read(5, 12), read(8, 20), read(12, 20)]


def warp_embedded_scalar(offscreen, dxc):
    pipe, source, ps_source = embedded_scalar_control()
    count = len(pipe.uds)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        for index in (0, 1, 2, 3):
            draw(offscreen, vs, ps, index, embedded_reference(index), count)
        wrong = compile_shader(dxc, directory, 'vs_6_6', mutate(source, '[a >> 2u]', '[(a >> 2u) + 1u]'))
        draw(offscreen, wrong, ps, 0, embedded_reference(0), count, negative=True)


def warp(offscreen, dxc):
    """The fixture heap holds three raw buffers whose first words are 1.25, 2.5 and 4.0. Both vertex stages read a
    record chosen by a root constant inside a branch; the target must hold the word the record names."""
    # second: the root constant that names record 1 (a byte offset in the first two, an index shifted by 5 in the third)
    for label, words, cond, second in (('branch', vertex_copy(ARM), 2, 32),
                                       ('rewrite', vertex_copy(REWRITE_ARM), 2, 32), ('else arm', ELSE_VS, 0, 1)):
        pipe, vs_source, ps_source = translate(words)
        count = len(pipe.uds)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            vs = compile_shader(dxc, directory, 'vs_6_6', vs_source)
            ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
            draw(offscreen, vs, ps, 0, 0x3fa00000, count, cond=cond)
            draw(offscreen, vs, ps, second, 0x40200000, count, cond=cond)
            wrong = compile_shader(dxc, directory, 'vs_6_6', mutate(vs_source, ') >> 5;', ') >> 6;'))
            draw(offscreen, wrong, ps, second, 0x40200000, count, negative=True, cond=cond)
        print('heap record %s: readback and mutant matched' % label)


if __name__ == '__main__':
    if sys.argv[1:2] == ['warp']:
        warp(*sys.argv[2:4])
    elif sys.argv[1:2] == ['warp-image']:
        warp_image(*sys.argv[2:4])
    elif sys.argv[1:2] == ['warp-image-loop']:
        warp_image_loop(*sys.argv[2:4])
    elif sys.argv[1:2] == ['warp-descriptor-word']:
        warp_descriptor_word(*sys.argv[2:4])
    elif sys.argv[1:2] == ['warp-image-dimensions']:
        warp_image_dimensions(*sys.argv[2:4])
    elif sys.argv[1:2] == ['warp-embedded-scalar']:
        warp_embedded_scalar(*sys.argv[2:4])
    else:
        main(*sys.argv[2:3] if sys.argv[1:2] == ['dxc'] else ())
