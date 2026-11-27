"""Source-built four-lane numeric waterfall for the compute readfirstlane fixture (#2644)."""
import struct
import subprocess

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


def program(*, before_save=(), after_save=(), compare=COMPARE, body=(ADD, XOR), after_restore=()):
    loop = (HEAD, compare, *body, CLOSE)
    offset = -(sum(len(bytes.fromhex(code)) for _, code in loop) // 4 + 1)
    branch = ('s_cbranch_scc1 %d' % offset, struct.pack('<I', 0xbf850000 | (offset & 0xffff)).hex())
    return (*PREFIX, *before_save, SAVE, *after_save, *loop, branch, RESTORE, *after_restore, STORE, END)


def translate(rows, group=(4, 1, 1)):
    code = bytes.fromhex(''.join(word for _, word in rows))
    return G.ComputePipeline(code, root_blob(), USER7, group).hlsl_cs()


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
        (program(body=(('v_mbcnt_lo_u32_b32 v7, exec_lo, 0', '070065d77e000100'),)), 'body instruction v_mbcnt'),
        (program(after_restore=(live,)), 'body scalar is live'),
        (program(after_restore=(conditional, reset, live)), 'body scalar is live'),
        (program(after_save=(('s_mov_b32 s21, 1', '810395be'),)), 'unchanged copy of EXEC'),
        (program(body=(('s_add_u32 s23, s23, 1', '17811780'), XOR)), 'loop-carried scalar'),
        (program(body=(('s_add_u32 s23, 0, src_scc', '80fd1780'), XOR)), 'loop-carried scalar'),
        (program(body=(copy, XOR)), 'reads or changes a lane mask'),
        (program(compare=('v_cmpx_eq_i32_e32 s22, v4', '1608247d')), 'same key'),
        (program(before_save=(conditional,)), 'another branch enters'),
        (program(body=(('s_load_dwordx4 s[28:31], s[22:23], 0', '0b0708f4000000fa'), *FIXED_LOAD[1:])),
         'per-lane descriptor selection'),
    )
    for rows, reason in cases:
        if llvm_mc:
            source = '\n'.join(asm for asm, _ in rows) + '\n'
            assert assemble(llvm_mc, source) == bytes.fromhex(''.join(word for _, word in rows)), source
        if reason:
            refused(reason, lambda: translate(rows))
        else:
            assert 's22 = v5;' in translate(rows)
    refused('at most 32 threads', lambda: translate(program(), (33, 1, 1)))


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
    return 0
