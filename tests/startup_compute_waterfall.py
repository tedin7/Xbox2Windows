"""Source-built numeric, raw-buffer and integer ALU waterfalls for compute readfirstlane."""
import struct
import subprocess
from unittest.mock import patch

from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, MASK, SENTINEL, USER7, WORDS, assemble, refused, root_blob


# The input buffer contains the keys. Root constant s4 selects how many lanes enter the loop.
# Every lane writes its result after the original EXEC is restored, including the initial 13 for inactive lanes.
PREFIX = (
    ('s_mov_b32 s10, s6', '06038abe'), ('s_mov_b32 s11, 0', '80038bbe'),
    ('s_load_dwordx8 s[12:19], s[10:11], 0', '05030cf4000000fa'), ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
    ('v_lshlrev_b32_e32 v4, 2, v0', '82000834'), ('buffer_load_dword v5, v4, s[12:15], 0 offen', '001030e004050380'),
    ('s_waitcnt vmcnt(0)', '703f8cbf'), ('v_mov_b32_e32 v7, 13', '8d020e7e'),
    ('s_mov_b32 s26, exec_lo', '7e039abe'), ('v_cmpx_gt_u32_e32 s4, v0', '0400a87d'),
)
SAVE = ('s_mov_b32 s21, exec_lo', '7e0395be')
HEAD = ('v_readfirstlane_b32 s22, v5', '05052c7e')
COMPARE = ('v_cmpx_eq_i32_e32 s22, v5', '160a247d')
ADD = ('s_add_u32 s23, s22, 17', '16911780')
CONST = ('s_mov_b32 s23, 17', '910397be')
USE_CONST = ('v_add_nc_u32 v7, s23, v7', '170e0e4a')
XOR = ('v_xor_b32_e32 v7, s23, v0', '17000e3a')
CLOSE = ('s_andn2_wrexec_b32 s21, s21', '154795be')
RESTORE = ('s_mov_b32 exec_lo, s26', '1a03febe')
STORE = ('buffer_store_dword v7, v4, s[16:19], 0 offen', '001070e004070480')
END = ('s_endpgm', '000081bf')
FIXED_LOAD = (('s_load_dwordx4 s[28:31], s[10:11], 0', '050708f4000000fa'), ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
              ('v_mov_b32_e32 v8, s22', '1602107e'), ('buffer_load_dword v7, v8, s[28:31], 0 offen', '001030e008070780'),
              ('s_waitcnt vmcnt(0)', '703f8cbf'))


WIDE_PREFIX = (
    ('s_mov_b32 s32, s6', '0603a0be'),
    ('s_mov_b32 s33, 0', '8003a1be'),
    ('s_load_dwordx8 s[36:43], s[32:33], 0x0', '10090cf4000000fa'),
    ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
    ('v_lshlrev_b32_e32 v10, 2, v0', '82001434'),
    ('buffer_load_dword v11, v10, s[36:39], 0 offen', '001030e00a0b0980'),
    ('s_waitcnt vmcnt(0)', '703f8cbf'),
    ('v_and_b32_e32 v12, 3, v0', '83001836'),
    ('v_lshlrev_b32_e32 v12, 4, v12', '84181834'),
    ('s_mov_b32 s46, exec_lo', '7e03aebe'),
    ('s_mov_b32 s47, exec_lo', '7e03afbe'),
)
WIDE_HEAD = (
    ('v_readfirstlane_b32 s48, v11', '0b05607e'),
    ('v_cmpx_eq_i32_e32 s48, v11', '3016247d'),
)
WIDE_ADDRESS = (
    ('s_lshl_b32 s49, s48, 7', '3087318f'),
    ('v_add_nc_u32_e32 v12, s49, v12', '3118184a'),
)
WIDE_LOADS = (
    ('buffer_load_dwordx4 v[14:17], v12, s[36:39], 0 offen offset:128', '801038e00c0e0980'),
    ('buffer_load_dwordx2 v[18:19], v12, s[36:39], 0 offen offset:196', 'c41034e00c120980'),
)
THREE_WORD_LOADS = (
    ('buffer_load_dwordx3 v[14:16], v12, s[36:39], 0 offen offset:128', '80103ce00c0e0980'),
    ('buffer_load_dword v17, v12, s[36:39], 0 offen offset:140', '8c1030e00c110980'),
    WIDE_LOADS[1],
)
REFUSED_LOADS = (
    ('buffer_load_dwordx4 v[14:17], v12, s[36:39], 0 idxen', '002038e00c0e0980'),
    ('buffer_load_dwordx4 v[14:17], v12, s[36:39], 0 offen glc', '005038e00c0e0980'),
)
WIDE_CLOSE = (
    ('s_waitcnt vmcnt(0)', '703f8cbf'),
    ('s_andn2_wrexec_b32 s47, s47', '2f47afbe'),
)
WIDE_STORE = (
    ('s_mov_b32 exec_lo, s46', '2e03febe'),
    ('v_mul_lo_u32 v13, v0, 24', '0d0069d500310100'),
    ('buffer_store_dwordx4 v[14:17], v13, s[40:43], 0 offen', '001078e00d0e0a80'),
    ('buffer_store_dwordx2 v[18:19], v13, s[40:43], 0 offen offset:16', '101074e00d120a80'),
    ('s_endpgm', '000081bf'),
)
ALU_BODY = (
    ('s_or_b32 s49, s48, 0x80000000', '30ff318800000080'),
    ('s_bfe_i32 s50, s49, 0x30000', '31ff329400000300'),
    ('s_min_i32 s51, s49, 3', '31833383'),
    ('s_max_i32 s52, s49, 3', '31833484'),
    ('s_min_u32 s53, s49, 3', '3183b583'),
    ('v_mov_b32_e32 v14, s50', '32021c7e'),
    ('v_mov_b32_e32 v15, s51', '33021e7e'),
    ('v_mov_b32_e32 v16, s52', '3402207e'),
    ('v_mov_b32_e32 v17, s53', '3502227e'),
    ('v_and_b32_e32 v18, s49, v0', '31002436'),
    ('v_or_b32_e32 v19, s49, v0', '31002638'),
    ('v_not_b32_e32 v20, v11', '0b6f287e'),
    ('v_xnor_b32_e32 v21, s49, v0', '31002a3c'),
)
ALU_STORE = (
    WIDE_STORE[0],
    ('v_lshlrev_b32_e32 v13, 5, v0', '85001a34'),
    WIDE_STORE[2],
    ('buffer_store_dwordx4 v[18:21], v13, s[40:43], 0 offen offset:16', '101078e00d120a80'),
    END,
)

def program(*, before_save=(), after_save=(), compare=COMPARE, body=(ADD, XOR), after_restore=()):
    loop = (HEAD, compare, *body, CLOSE)
    offset = -(sum(len(bytes.fromhex(code)) for _, code in loop) // 4 + 1)
    branch = ('s_cbranch_scc1 %d' % offset, struct.pack('<I', 0xbf850000 | (offset & 0xffff)).hex())
    return (*PREFIX, *before_save, SAVE, *after_save, *loop, branch, RESTORE, *after_restore, STORE, END)


def translate(rows, group=(4, 1, 1)):
    code = bytes.fromhex(''.join(word for _, word in rows))
    return G.ComputePipeline(code, root_blob(), USER7, group).hlsl_cs()


def wide_program(body, stores=WIDE_STORE):
    loop = (*WIDE_HEAD, *body, *WIDE_CLOSE)
    offset = -(sum(len(code) // 8 for _, code in loop) + 1)
    branch = ('s_cbranch_scc1 %d' % offset, struct.pack('<I', 0xbf850000 | (offset & 0xffff)).hex())
    return (*WIDE_PREFIX, *loop, branch, *stores)


def wide_fixtures():
    yield 'wide', wide_program((*WIDE_ADDRESS, *WIDE_LOADS)), 6
    yield 'three-word', wide_program((*WIDE_ADDRESS, *THREE_WORD_LOADS)), 6
    yield 'integer-alu', wide_program(ALU_BODY, ALU_STORE), 8


def wide_controls(llvm_mc):
    mods = 'compute readfirstlane waterfall: body instruction buffer_load_dwordx4 has unproved modifiers'
    for load in REFUSED_LOADS:
        # Keep the 32 input keys and test the legacy waterfall modifier proof in isolation.
        with patch.object(G.ComputePipeline, 'retry_group_vote_for', return_value=False):
            refused(mods, lambda: translate(wide_program((*WIDE_ADDRESS, load)), (32, 1, 1)))
    for _, rows, _ in wide_fixtures():
        if llvm_mc:
            assert assemble(llvm_mc, '\n'.join(asm for asm, _ in rows) + '\n') == bytes.fromhex(
                ''.join(code for _, code in rows))
        translate(rows, (32, 1, 1))


def wide_reference(label, keys, source):
    words = []
    for lane, key in enumerate(keys):
        if label == 'integer-alu':
            signed = key - (1 << 31)
            bits = signed & MASK
            words.extend((key if key < 4 else MASK - 3, min(signed, 3) & MASK, max(signed, 3),
                          min(bits, 3), bits & lane, bits | lane, MASK ^ key, MASK ^ bits ^ lane))
        else:
            index = 32 * (1 + key) + 4 * (lane % 4)
            words.extend((*source[index:index + 4], *source[index + 17:index + 19]))
    return words


def wide_case(directory, label, batch, keys, width):
    source = [0xa5310041 ^ (i * 0x10307) for i in range(WORDS)]
    source[:len(keys)] = keys
    expected = wide_reference(label, keys, source)
    assert len(expected) == width * len(keys)
    case = directory / ('%s-%d.case' % (label, batch))
    words = [1, *expected, *([SENTINEL] * (WORDS - len(expected))), *([0] * 16), *source]
    case.write_bytes(struct.pack('<%dI' % len(words), *words))
    return case, expected, source


def first_destination_mutant(rows):
    def load_first(stage, ops, mods):
        stage.op_buffer_load_dword(ops, mods, 1)

    with patch.object(G.Stage, 'op_buffer_load_dwordx4', load_first):
        return translate(rows, (32, 1, 1))


def dropped_offset_mutant():
    # Clear the low byte of the 12-bit MUBUF immediate in the four-word instruction.
    asm, code = WIDE_LOADS[0]
    load = (asm.removesuffix(' offset:128'), '00' + code[2:])
    return translate(wide_program((*WIDE_ADDRESS, load, WIDE_LOADS[1])), (32, 1, 1))


def wide_dispatch(directory, dxc, runner, adapter):
    # The runner compares 320 words per case. Two dispatches cover 64 threads without truncating any lane.
    batches = ([2, 0, 1, 0, 3, 0, 2, 1] * 4, [1, 3, 0, 1, 2, 0, 1, 3] * 3 + [1, 4, 1, 0, 2, 0, 1, 1])
    src, shader = directory / 'wide-waterfall.hlsl', directory / 'wide-waterfall.cso'

    def compile_shader(hlsl):
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)

    for label, rows, width in wide_fixtures():
        compile_shader(translate(rows, (32, 1, 1)))
        cases = [wide_case(directory, label, batch, keys, width) for batch, keys in enumerate(batches)]
        check_references(label, runner, adapter, shader, [(case, 'batch %d' % i, None)
                         for i, (case, _, _) in enumerate(cases)])
        case, expected, source = cases[0]
        if label == 'wide':
            for mutation, mutant, word, actual in (
                    ('first destination only', first_destination_mutant(rows), 1, 0),
                    ('immediate offset dropped', dropped_offset_mutant(), 0, source[64])):
                compile_shader(mutant)
                check_references(label, runner, adapter, shader, [
                    (case, mutation, 'word=%d expected=%08x actual=%08x' % (word, expected[word], actual))])
        else:
            # Perturb every newly covered result in the readback oracle, independently.
            slots = (1,) if label == 'three-word' else range(width)
            controls = []
            for slot in slots:
                wrong = directory / ('%s-wrong-%d.case' % (label, slot))
                data = bytearray(case.read_bytes())
                struct.pack_into('<I', data, 4 * (slot + 1), expected[slot] ^ 1)
                wrong.write_bytes(data)
                controls.append((wrong, 'word %d flipped' % slot,
                                 'word=%d expected=%08x actual=%08x' % (slot, expected[slot] ^ 1, expected[slot])))
            check_references(label, runner, adapter, shader, controls)

def controls(llvm_mc=None):
    live = ('v_mov_b32_e32 v7, s23', '17020e7e')
    copy = ('s_mov_b32 s24, s21', '150398be')
    reset = ('s_mov_b32 s23, 0', '800397be')
    conditional = ('s_cbranch_scc0 1', '010084bf')
    cases = (
        (program(), None),
        (program(after_save=(('s_nop 0', '000080bf'),)), None),
        (program(body=FIXED_LOAD), None),
        (program(after_restore=(reset, live)), None),
        (program(body=(('s_mov_b32 s23, s22', '160397be'), ('s_lshl_b32 s23, s23, 1', '1781178f'),
                       ('s_or_b32 s23, s23, 1', '17811788'), ('s_and_b32 s23, s23, 31', '179f1787'),
                       ('v_add_nc_u32_e32 v7, s23, v0', '17000e4a'))), None),
        # A body scalar that does not depend on the key holds one value in every iteration: read after the loop.
        (program(body=(CONST, XOR), after_restore=(USE_CONST,)), None),
        (program(body=(CONST, ('s_add_u32 s23, s23, s22', '17161780'), XOR), after_restore=(USE_CONST,)),
         'body scalar is live'),
        (program(body=(CONST, XOR), after_restore=(('v_mov_b32_e32 v7, s22', '16020e7e'),)), 'body scalar is live'),
        (program(body=(STORE,)), 'body instruction buffer_store_dword'),
        (program(after_restore=(live,)), 'body scalar is live'),
        (program(after_restore=(conditional, reset, live)), 'body scalar is live'),
        (program(body=(('s_add_u32 s23, s23, 1', '17811780'), XOR)), 'loop-carried scalar'),
        (program(body=(copy, XOR)), 'reads or changes a lane mask'),
        (program(compare=('v_cmpx_eq_i32_e32 s22, v4', '1608247d')), 'same key'),
    )
    for rows, reason in cases:
        if llvm_mc:
            source = '\n'.join(asm for asm, _ in rows) + '\n'
            assert assemble(llvm_mc, source) == bytes.fromhex(''.join(word for _, word in rows)), source
        if reason:
            # Keep the four key lanes and isolate the legacy per-lane waterfall proof.
            with patch.object(G.ComputePipeline, 'retry_group_vote_for', return_value=False):
                refused(reason, lambda: translate(rows))
        else:
            assert 's22 = v5;' in translate(rows)
    for rows, reason in (
        (program(body=(('v_mbcnt_lo_u32_b32 v7, exec_lo, 0', '070065d77e000100'),)), 'requires a known mask'),
        (program(body=(('s_add_u32 s23, 0, src_scc', '80fd1780'), XOR)), 'entry path has no SCC producer'),
        (program(before_save=(conditional,)), 'incompatible lane mask definitions for s21'),
        (program(body=(('s_load_dwordx4 s[28:31], s[22:23], 0', '0b0708f4000000fa'), *FIXED_LOAD[1:])),
         's_load through s22'),
    ):
        if llvm_mc:
            assert assemble(llvm_mc, '\n'.join(asm for asm, _ in rows) + '\n') == bytes.fromhex(''.join(word for _, word in rows))
        refused(reason, lambda: translate(rows))
    refused('at most 32 threads', lambda: translate(program(), (33, 1, 1)))
    wide_controls(llvm_mc)


def dispatch(directory, dxc, runner, adapter):
    for label, body in (('arithmetic', (ADD, XOR)), ('fixed-buffer', FIXED_LOAD), ('invariant', (CONST, XOR))):
        src, shader = directory / 'waterfall.hlsl', directory / 'waterfall.cso'
        after = (USE_CONST,) if label == 'invariant' else ()
        src.write_text(translate(program(body=body, after_restore=after)), encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        cases = []
        inputs = [(n, (64, 32, 64, 96)) for n in range(5)]
        if label == 'arithmetic':
            inputs.append((4, (MASK, MASK, 0, 0)))
        for active, keys in inputs:
            source = [17 * i + 9 for i in range(WORDS)]
            source[:4] = keys
            output = [13] * 4
            for lane, key in enumerate(keys[:active]):
                output[lane] = (((key + 17) & MASK) ^ lane) if label == 'arithmetic' else source[key // 4]
                if label == 'invariant':  # the body scalar is 17 in every pass: v7 = (17 ^ lane) + 17
                    output[lane] = (17 ^ lane) + 17
            if label == 'invariant':
                output = [o if i < active else 13 + 17 for i, o in enumerate(output)]  # the add runs for every lane
            case = directory / ('waterfall-%d.case' % len(cases))
            words = [1, *output, *([SENTINEL] * (WORDS - 4)), active, *([0] * 15), *source]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, 'keys %s, %d active' % (keys, active), None))
            if active == 4 and keys[0] != keys[1]:
                wrong = (output[0] ^ 1) if label != 'fixed-buffer' else output[0]
                words[2] = wrong  # lane 1 using lane 0's key
                case = directory / 'waterfall-wrong.case'
                case.write_bytes(struct.pack('<%dI' % len(words), *words))
                cases.append((case, 'uniform key', 'word=1 expected=%08x actual=%08x' % (wrong, output[1])))
        check_references(label + ' waterfall', runner, adapter, shader, cases)
    wide_dispatch(directory, dxc, runner, adapter)
    return 0
