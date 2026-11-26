"""Constant-lane SGPR spills of descriptor and pointer words in compute stages of tools/startup/gfx2hlsl.py (#1723).

  (no arguments)                                  translation controls and translator mutations, pure Python
  binary <llvm-mc>                                assemble the fixture sources (gfx1013) and compare with the words here
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                                  translate, compile with DXC cs_6_0, dispatch and compare every word
  control <p-store-dropped|s-arm-swapped|s-store-dropped> <dxc> <offscreen_check> <warp|hardware>
                                                  the same dispatch with a mutated translation, which must mismatch

`v_writelane_b32 vN, sM, k` parks SGPR sM in lane k of vN and `v_readlane_b32 sD, vN, k` reads it back; both ignore
EXEC (AMD RDNA 2 ISA reference, VOP3 lane operations). A descriptor or pointer word has no run-time value in the
translation: it is a symbol (`('desc', table, offset, dword)`, `('ptr', ...)`), so the slot carries the symbol and the
reload restores the same one. The write-before-read proof of tests/startup_gfx2hlsl_spill_branches_check.py (a write
on every path, regions the wave runs while a thread skips them, loop-carried reads) decides whether a reload sees the
symbol, and a reload keeps the provenance of the word it restores: a formatted buffer load past the entry block
accepts a reloaded descriptor only when every write that can supply it was itself the word of an entry-block or
local table load.

Fixtures (source-built gfx1013, root constants key and stride in s4-s5, one descriptor table in s6):
- S spills the table's read descriptor s[12:15] (formatted load) in the entry block and its store descriptor s[16:19]
  in both arms of a uniform branch on the key, clobbers both, and inside an s_cbranch_execz region reloads them and
  loads RG16_UINT words through one and stores through the other. 16 lanes, lanes below 12 store four words each.
- P spills the table pointer pair s[10:11] (the low word a table pointer, the high word the constant zero), clobbers
  it, reloads it inside an s_cbranch_execz region and loads the store descriptor through it (shape of case J of
  tests/startup_gfx2hlsl_compute_check.py). Four groups of eight lanes, lanes below 3 store one word.
"""
import re
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / 'tools'
sys.path.insert(0, str(TOOLS / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + str(detail)[:1500])


SOURCE_S = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_writelane_b32 v40, s12, 0
v_writelane_b32 v40, s13, 1
v_writelane_b32 v40, s14, 2
v_writelane_b32 v40, s15, 3
s_and_b32 s23, s4, 1
s_cmp_lg_u32 s23, 0
s_cbranch_scc0 Lelse
v_writelane_b32 v40, s16, 4
v_writelane_b32 v40, s17, 5
v_writelane_b32 v40, s18, 6
v_writelane_b32 v40, s19, 7
v_mov_b32 v1, 1
s_branch Ljoin
Lelse:
v_writelane_b32 v40, s16, 4
v_writelane_b32 v40, s17, 5
v_writelane_b32 v40, s18, 6
v_writelane_b32 v40, s19, 7
v_mov_b32 v1, 0
Ljoin:
s_mov_b32 s12, 0
s_mov_b32 s13, 0
s_mov_b32 s14, 0
s_mov_b32 s15, 0
s_mov_b32 s16, 0
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_lshlrev_b32 v4, 4, v0
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 12, v0
s_mov_b32 exec_lo, vcc_lo
s_cbranch_execz Lend
v_readlane_b32 s12, v40, 0
v_readlane_b32 s13, v40, 1
v_readlane_b32 s14, v40, 2
v_readlane_b32 s15, v40, 3
v_readlane_b32 s16, v40, 4
v_readlane_b32 s17, v40, 5
v_readlane_b32 s18, v40, 6
v_readlane_b32 s19, v40, 7
buffer_load_format_xyzw v[5:8], v[0:1], s[12:15], 0 idxen offen offset:2
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[5:8], v4, s[16:19], 0 offen
Lend:
s_mov_b32 exec_lo, s21
s_endpgm
"""
SOURCE_P = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[12:15], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_writelane_b32 v41, s10, 0
v_writelane_b32 v41, s11, 1
s_mov_b32 s10, 0
s_mov_b32 s11, 0
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 2, v3
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 3, v0
s_mov_b32 exec_lo, vcc_lo
s_cbranch_execz Lend
v_readlane_b32 s10, v41, 0
v_readlane_b32 s11, v41, 1
s_load_dwordx4 s[16:19], s[10:11], 0x10
s_mul_i32 s22, s7, s5
s_add_u32 s22, s22, s4
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0) lgkmcnt(0)
v_xor_b32 v5, s22, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
Lend:
s_mov_b32 exec_lo, s21
s_endpgm
"""
S_ARM = ('v_writelane_b32 v40, s16, 4\nv_writelane_b32 v40, s17, 5\nv_writelane_b32 v40, s18, 6\n'
         'v_writelane_b32 v40, s19, 7\n')
S_ENTRY = 'v_writelane_b32 v40, s12, 0\nv_writelane_b32 v40, s13, 1\nv_writelane_b32 v40, s14, 2\nv_writelane_b32 v40, s15, 3\n'

# name -> (base source, [(old, new)], what the refusal must say, why it is refused). Each variant is a fixture with one
# change; it is assembled in binary mode like the fixtures and must be refused for that reason and no other.
VARIANTS = {
    'reload-before-write': (SOURCE_S, [(S_ENTRY, '')], 'no v_writelane_b32 of it on some path',
                            'the read descriptor is never written to its lanes'),
    'two-descriptors-joined': (SOURCE_S, [(S_ARM + 'v_mov_b32 v1, 0', S_ARM.replace('s16', 's12').replace('s17', 's13')
                                           .replace('s18', 's14').replace('s19', 's15') + 'v_mov_b32 v1, 0')],
                               'joins different descriptor or pointer words',
                               'one arm spills the read descriptor and the other the store descriptor into the same lanes'),
    'modified-word': (SOURCE_S, [('s_and_b32 s23, s4, 1', 's_and_b32 s17, s17, 0xc000ffff\ns_and_b32 s23, s4, 1')],
                      'modified descriptor or pointer word', 'a stride word rewritten before the spill'),
    'lane-mask': (SOURCE_S, [('v_writelane_b32 v40, s16, 4\nv_writelane_b32 v40, s17, 5\nv_writelane_b32 v40, s18, 6\n'
                              'v_writelane_b32 v40, s19, 7\nv_mov_b32 v1, 1',
                              'v_cmp_gt_u32 s20, 12, v0\nv_writelane_b32 v40, s20, 4\nv_mov_b32 v1, 1')],
                  'one of them a lane mask', 'a compare result spilled in one arm where the other arm spills a descriptor '
                  '(lane masks have their own fixtures in tests/startup_gfx2hlsl_compute_mask_spill_check.py)'),
    'wrong-lane': (SOURCE_S, [('v_readlane_b32 s13, v40, 1', 'v_readlane_b32 s13, v40, 5')],
                   'descriptor whose words were rewritten', 'word 1 of the descriptor reloaded from the lane of word 5'),
    'load-skipped-on-one-path': (SOURCE_S, [('s_load_dwordx8 s[12:19], s[10:11], 0x0\ns_waitcnt lgkmcnt(0)\n',
                                             's_cmp_lg_u32 s4, 0\ns_cbranch_scc0 Lnoload\n'
                                             's_load_dwordx8 s[12:19], s[10:11], 0x0\ns_waitcnt lgkmcnt(0)\nLnoload:\n')],
                                 'holds a lane mask or descriptor', 'the descriptor is loaded on one path only, so a spill of it has no single load to prove'),
    'written-in-region-read-outside': (SOURCE_S, [
        (S_ARM + 'v_mov_b32 v1, 1', 'v_mov_b32 v1, 1'), (S_ARM + 'v_mov_b32 v1, 0', 'v_mov_b32 v1, 0'),
        (''.join('s_mov_b32 s%d, 0\n' % r for r in range(16, 20)), ''),
        (''.join('v_readlane_b32 s%d, v40, %d\n' % (r, r - 12) for r in range(16, 20)), ''),
        ('s_cbranch_execz Lend\n', 's_cbranch_execz Lend\n' + S_ARM), ('Lend:\n', 'Lend:\nv_readlane_b32 s16, v40, 4\n')],
                                       'effect outside scalar registers', 'the store descriptor is written inside the '
                                       'execz region (the lane write is a cross-lane effect there) and reloaded after it'),
    'loop-carried': (SOURCE_S, [('Ljoin:\n', 'Ljoin:\ns_mov_b32 s22, 0\nLloop:\nv_readlane_b32 s16, v40, 4\n'
                                 'v_writelane_b32 v40, s16, 4\ns_add_u32 s22, s22, 1\ns_cmp_lt_u32 s22, 2\n'
                                 's_cbranch_scc1 Lloop\n')],
                     'written later in a loop', 'a reload whose lane is written again by the loop body'),
}
# The read descriptor is loaded once above the branch and spilled inside the arms: its one load reaches every use on every
# path, so the formatted load proof accepts it although the load is not in the block of the spill.
SOURCE_S_ARMS = (SOURCE_S.replace(S_ENTRY, '').replace('v_mov_b32 v1, 1', S_ENTRY + 'v_mov_b32 v1, 1')
                 .replace('v_mov_b32 v1, 0', S_ENTRY + 'v_mov_b32 v1, 0'))
assert SOURCE_S_ARMS.count('v_writelane_b32 v40, s12, 0') == 2
EXPECTED_ACCEPTED = {'S': SOURCE_S, 'P': SOURCE_P, 'S-load-above-arms': SOURCE_S_ARMS}


def variant_source(name):
    base, edits = VARIANTS[name][:2]
    for old, new in edits:
        assert base.count(old) == 1, (name, old)
        base = base.replace(old, new)
    return base


def all_sources():
    return {**EXPECTED_ACCEPTED, **{name: variant_source(name) for name in VARIANTS}}


def assemble_labels(llvm_mc, source):
    """llvm-mc leaves a branch label to the linker, so the words come from a first pass for the instruction sizes."""
    from startup_flat_param_check import assemble
    lines = source.splitlines()
    plain = [line for line in lines if not line.endswith(':')]
    sizes = []
    for line in plain:
        sizes.append(len(assemble(llvm_mc, line + '\n')))
    labels, position, at = {}, 0, 0
    for line in lines:
        if line.endswith(':'):
            labels[line[:-1]] = position
        else:
            position += sizes[at]
            at += 1
    out, position = [], 0
    for line, size in zip(plain, sizes):
        m = re.fullmatch(r'(s_c?branch\w*) (L\w+)', line)
        if m:
            line = '%s %d' % (m[1], (labels[m[2]] - (position + 4)) // 4)
        out.append(line)
        position += size
    words = bytearray()
    for line in out:
        built = bytearray(assemble(llvm_mc, line + '\n'))
        if re.match(r'v_(read|write)lane_b32\b|v_cmp\w*\s', line) and len(built) == 8:
            # src2 of the lane operations and of the VOP3 compares is unused: LLVM 15 encodes it as 0, newer LLVM as the
            # inline constant 0 (128).
            # Clear it so the recorded words hold for both.
            dword = int.from_bytes(built[4:8], 'little') & ~(0x1ff << 18)
            built[4:8] = dword.to_bytes(4, 'little')
        words += built
    return bytes(words)


# CASE: (source, words as hex, rsrc2, group size, group counts)
USER7, TGID_X, tidig = C.USER7, C.TGID_X, C.tidig
CASES_ENCODING = {
    'S': (USER7, (16, 1, 1), (1, 1, 1)),
    'P': (USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
}
WORDS = {
    'S': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010004811787178007bf0a0084bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '8102027e090082bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e01008002027e80038cbe'
        '80038dbe80038ebe80038fbe800390be800391be800392be800393be840008347e0395be8c00887d6a03febe150088bf'
        '0c0060d7280101000d0060d7280301000e0060d7280501000f0060d728070100100060d728090100110060d7280b0100'
        '120060d7280d0100130060d7280f010002300ce000050380703f8cbf001078e0040504801503febe000081bf'),
    'P': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa7fc08cbf290061d70a000100290061d70b02010080038abe80038bbe0783148f'
        '1400064a820608347e0395be8300887d6a03febe0e0088bf0a0060d7290101000b0060d729030100050408f4100000fa'
        '0705169316041680001030e00405038070008cbf160a0a3a001070e0040504801503febe000081bf'),
    'reload-before-write': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf04811787178007bf0a0084bf280061d710080100280061d7110a0100'
        '280061d7120c0100280061d7130e01008102027e090082bf280061d710080100280061d7110a0100280061d7120c0100'
        '280061d7130e01008002027e80038cbe80038dbe80038ebe80038fbe800390be800391be800392be800393be84000834'
        '7e0395be8c00887d6a03febe150088bf0c0060d7280101000d0060d7280301000e0060d7280501000f0060d728070100'
        '100060d728090100110060d7280b0100120060d7280d0100130060d7280f010002300ce000050380703f8cbf001078e0'
        '040504801503febe000081bf'),
    'two-descriptors-joined': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010004811787178007bf0a0084bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '8102027e090082bf280061d70c080100280061d70d0a0100280061d70e0c0100280061d70f0e01008002027e80038cbe'
        '80038dbe80038ebe80038fbe800390be800391be800392be800393be840008347e0395be8c00887d6a03febe150088bf'
        '0c0060d7280101000d0060d7280301000e0060d7280501000f0060d728070100100060d728090100110060d7280b0100'
        '120060d7280d0100130060d7280f010002300ce000050380703f8cbf001078e0040504801503febe000081bf'),
    'modified-word': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010011ff1187ffff00c004811787178007bf0a0084bf280061d710080100280061d7110a0100280061d7120c0100'
        '280061d7130e01008102027e090082bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '8002027e80038cbe80038dbe80038ebe80038fbe800390be800391be800392be800393be840008347e0395be8c00887d'
        '6a03febe150088bf0c0060d7280101000d0060d7280301000e0060d7280501000f0060d728070100100060d728090100'
        '110060d7280b0100120060d7280d0100130060d7280f010002300ce000050380703f8cbf001078e0040504801503febe'
        '000081bf'),
    'lane-mask': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010004811787178007bf060084bf1400c4d48c000200280061d7140801008102027e090082bf280061d710080100'
        '280061d7110a0100280061d7120c0100280061d7130e01008002027e80038cbe80038dbe80038ebe80038fbe800390be'
        '800391be800392be800393be840008347e0395be8c00887d6a03febe150088bf0c0060d7280101000d0060d728030100'
        '0e0060d7280501000f0060d728070100100060d728090100110060d7280b0100120060d7280d0100130060d7280f0100'
        '02300ce000050380703f8cbf001078e0040504801503febe000081bf'),
    'wrong-lane': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010004811787178007bf0a0084bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '8102027e090082bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e01008002027e80038cbe'
        '80038dbe80038ebe80038fbe800390be800391be800392be800393be840008347e0395be8c00887d6a03febe150088bf'
        '0c0060d7280101000d0060d7280b01000e0060d7280501000f0060d728070100100060d728090100110060d7280b0100'
        '120060d7280d0100130060d7280f010002300ce000050380703f8cbf001078e0040504801503febe000081bf'),
    'load-skipped-on-one-path': bytes.fromhex(
        '06038abe80038bbe048007bf030084bf05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d7'
        '0e040100280061d70f06010004811787178007bf0a0084bf280061d710080100280061d7110a0100280061d7120c0100'
        '280061d7130e01008102027e090082bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '8002027e80038cbe80038dbe80038ebe80038fbe800390be800391be800392be800393be840008347e0395be8c00887d'
        '6a03febe150088bf0c0060d7280101000d0060d7280301000e0060d7280501000f0060d728070100100060d728090100'
        '110060d7280b0100120060d7280d0100130060d7280f010002300ce000050380703f8cbf001078e0040504801503febe'
        '000081bf'),
    'S-load-above-arms': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf04811787178007bf120084bf280061d710080100280061d7110a0100'
        '280061d7120c0100280061d7130e0100280061d70c000100280061d70d020100280061d70e040100280061d70f060100'
        '8102027e110082bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100280061d70c000100'
        '280061d70d020100280061d70e040100280061d70f0601008002027e80038cbe80038dbe80038ebe80038fbe800390be'
        '800391be800392be800393be840008347e0395be8c00887d6a03febe150088bf0c0060d7280101000d0060d728030100'
        '0e0060d7280501000f0060d728070100100060d728090100110060d7280b0100120060d7280d0100130060d7280f0100'
        '02300ce000050380703f8cbf001078e0040504801503febe000081bf'),
    'written-in-region-read-outside': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010004811787178007bf020084bf8102027e010082bf8002027e80038cbe80038dbe80038ebe80038fbe84000834'
        '7e0395be8c00887d6a03febe150088bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '0c0060d7280101000d0060d7280301000e0060d7280501000f0060d72807010002300ce000050380703f8cbf001078e0'
        '04050480100060d7280901001503febe000081bf'),
    'loop-carried': bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf280061d70c000100280061d70d020100280061d70e040100280061d7'
        '0f06010004811787178007bf0a0084bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e0100'
        '8102027e090082bf280061d710080100280061d7110a0100280061d7120c0100280061d7130e01008002027e800396be'
        '100060d728090100280061d7100801001681168016820abff9ff85bf80038cbe80038dbe80038ebe80038fbe800390be'
        '800391be800392be800393be840008347e0395be8c00887d6a03febe150088bf0c0060d7280101000d0060d728030100'
        '0e0060d7280501000f0060d728070100100060d728090100110060d7280b0100120060d7280d0100130060d7280f0100'
        '02300ce000050380703f8cbf001078e0040504801503febe000081bf'),
}


KEY, STRIDE, SENTINEL, WORDS_OUT = C.KEY, C.STRIDE, C.SENTINEL, C.WORDS
LIVE_LANES = 12


def expected(name):
    out = [SENTINEL] * WORDS_OUT
    if name == 'S':
        data = C.formatted_input()
        for lane in range(LIVE_LANES):  # the same RG16_UINT words as case L-uint, the lanes the compare leaves active
            low, high = struct.unpack_from('<HH', data, 6 * lane + 3)
            out[4 * lane:4 * lane + 4] = (high, low, 1, 0)
    else:  # P: the words of case J
        for gx in range(4):
            for tx in range(3):
                i = gx * 8 + tx
                out[i] = C.input_word(i) ^ ((gx * STRIDE + KEY) & C.MASK)
    return out


def translate(name, code=None, module=G):
    rsrc2, local, _ = CASES_ENCODING['P' if name == 'P' else 'S']
    return module.ComputePipeline(code or WORDS[name], C.root_blob(), rsrc2, local)


def refusal(name, module=G):
    try:
        translate(name, module=module).hlsl_cs()
    except module.Unsupported as reason:
        return str(reason)
    return None


def controls(module=G, quiet=False):
    """-> the labels of the controls that failed (all printed unless quiet)."""
    red = []

    def verdict(label, ok, detail=''):
        if not ok:
            red.append(label)
        if not quiet:
            check(label, ok, detail)

    for name in ('S', 'P', 'S-load-above-arms'):
        reason = refusal(name, module)
        verdict('fixture %s translates (%s)' % (name, reason), reason is None)
    for name, (_, _, fragment, why) in VARIANTS.items():
        reason = refusal(name, module)
        verdict('refused: %s (%s)' % (why, reason), reason is not None and fragment in reason, reason)
    return red


def probe_refusals():
    """Unit probes on a compute Stage that has symbols a table program cannot give it: ptrhi, a pointer with an offset."""
    def reason(sym_text, source_sym, kind='cs'):
        stage = G.Stage.__new__(G.Stage)
        stage.kind, stage.lines, stage.sym = kind, [], {20: source_sym} if source_sym else {}
        stage.used_s, stage.used_v, stage.uniform_v, stage.varying_s = set(), set(), set(), set()
        stage.scc_uniform, stage.exec_known, stage.has_branches, stage.blocks, stage.masks = True, True, False, [], set()
        stage.ins = [(4 * i, 1, a.split(' ', 1)[0], a) for i, a in enumerate(sym_text)]
        for _, _, name, asm in stage.ins:
            try:
                stage.one(name, asm)
            except G.Unsupported as why:
                return str(why)
        return stage

    spill = ['v_writelane_b32 v10, s20, 3', 'v_readlane_b32 s21, v10, 3']
    for label, sym in (('a table pointer', ('ptr', ('table', 0))), ('the high word of a root pointer', ('ptrhi', ('root', 2))),
                       ('a descriptor word', ('desc', ('table', 0), 0, 1))):
        stage = reason(spill, sym)
        check('compute stage spills %s and reloads the same symbol' % label,
              not isinstance(stage, str) and stage.sym.get(21) == sym, stage)
    for label, sym in (('a pointer with an offset', ('ptr', ('table', 0), 'added')),
                       ('a descriptor word with its stride cleared', ('desc', ('table', 0), 0, 1, 'stride0'))):
        text = reason(spill, sym)
        check('refused: %s (%s)' % (label, text), isinstance(text, str) and 'modified descriptor or pointer word' in text)
    for label, sym in (('a table pointer', ('ptr', ('table', 0))), ('a descriptor word', ('desc', ('table', 0), 0, 1))):
        text = reason(spill, sym, kind='ps')
        check('refused: %s in a pixel stage (%s)' % (label, text),
              isinstance(text, str) and 'holds a lane mask or descriptor' in text)


# Each translator mutation must turn at least one control red.
MUTATIONS = {
    'the reload restores no symbol': ('        self.set_s(dst[1], \'sp%d_%d\' % (n, lane), sym=sym, uniform=uniform)',
                                      '        self.set_s(dst[1], \'sp%d_%d\' % (n, lane), sym=None, uniform=uniform)'),
    'the reload drops the provenance': ("            return all(\n                getattr(self, 'reload_proven', {}).get(reloads[w], False)",
                                        "            return all(\n                False and getattr(self, 'reload_proven', {}).get(reloads[w], False)"),
    'every reload counts as proven': ("            return all(\n                getattr(self, 'reload_proven', {}).get(reloads[w], False)",
                                      "            return all(\n                True or getattr(self, 'reload_proven', {}).get(reloads[w], False)"),
    'a write counts as proven wherever the descriptor came from': (
        "        if sym[0] != 'desc' or not getattr(self, 'past_entry', False):\n            return True\n        return self.formatted",
        "        if True:\n            return True\n        return self.formatted"),
    'the reload reads the neighbouring lane': ('                            reach[i] = state.get((vreg, lane))',
                                               '                            reach[i] = state.get((vreg, lane ^ 1))'),
    'different symbols join into none': ("        if sym is None and any(s[0] is not None and s[0][0] in self.SPILL_SYMBOLS for s in stored):",
                                         "        if False:"),
    'a modified word is carried': ("                if not names_one_value(held):", "                if False:"),
    'no branches in a compute stage': ("(kind not in ('ps', 'cs') or getattr(self, 'spill_reach', None) is None)):",
                                       "(kind != 'ps' or getattr(self, 'spill_reach', None) is None)):"),
    'compute stages refused': ("SPILL_STAGES = ('ps', 'vs', 'ms', 'cs')", "SPILL_STAGES = ('ps', 'vs', 'ms')"),
    'no loop-carried refusal': ("        if max(writes) > i:", "        if False:"),
    'no write-before-read refusal': ("        if writes is None:\n            raise Unsupported('v_readlane_b32 reads v%d lane %d with no",
                                     "        if writes is None and False:\n            raise Unsupported('v_readlane_b32 reads v%d lane %d with no"),
}


def mutated(old, new):
    source = (TOOLS / 'startup' / 'gfx2hlsl.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    assert source.count(old) == 1, old
    module = types.ModuleType('gfx2hlsl_mutant')
    module.__file__ = str(TOOLS / 'startup' / 'gfx2hlsl.py')
    exec(compile(source.replace(old, new), module.__file__, 'exec'), module.__dict__)
    return module


def check_mutations():
    for label, (old, new) in MUTATIONS.items():
        try:
            red = controls(mutated(old, new), quiet=True)
        except Exception as error:  # a mutant that crashes the translator is caught as well
            red = ['crash: %s' % error]
        check('mutation "%s" turns %d control(s) red: %s' % (label, len(red), red[:3]), bool(red))


def check_text():
    s, p = translate('S').hlsl_cs(), translate('P').hlsl_cs()
    check('the descriptor slots are declared and never assigned', [l for l in s.splitlines() if 'sp40_' in l] ==
          [l for l in s.splitlines() if l.lstrip().startswith('uint sp40_')], s)
    check('the formatted load reads word 3 of the table descriptor', 'switch ((ud3 >> 12) & 127u)' in s)
    check('the pointer slot is never assigned, the constant high word is', '= sp41_0' not in p and 's11 = sp41_1;' in p, p)
    check('the pointer fixture stores through the reloaded table', p.count('u0.Store(') == 1)


# ---- binary and dispatch ---------------------------------------------------------------------------------------------
def check_binary(llvm_mc):
    for name, source in all_sources().items():
        built = assemble_labels(llvm_mc, source)
        check('llvm-mc assembles %s to the recorded words' % name, built == WORDS[name], built.hex())


# control -> (case, text the translation must contain once, replacement)
CONTROLS = {
    's-arm-swapped': ('S', 'pc = (!scc) ? 2 : 1;', 'pc = (!scc) ? 1 : 2;'),
    's-store-dropped': ('S', 'u0.Store4(a, uint4(v5, v6, v7, v8));', ''),
    'p-store-dropped': ('P', 'u0.Store(a, v5);', ''),
}


def run_case(name, dxc, runner, adapter, directory, control=None):
    hlsl = translate(name).hlsl_cs()
    if control:
        _, old, new = CONTROLS[control]
        assert hlsl.count(old) == 1, (control, hlsl)
        hlsl = hlsl.replace(old, new)
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    groups = CASES_ENCODING[name][2]
    constants = [KEY, STRIDE] + [0] * 14
    source = b''
    if name == 'S':
        constants[2:4] = [C.L_STRIDE, C.L_WORD3['L-uint']]
        source = C.formatted_input()
    case.write_bytes(struct.pack('<I', groups[0] | groups[1] << 8 | groups[2] << 16)
                     + struct.pack('<%dI' % WORDS_OUT, *expected(name)) + struct.pack('<16I', *constants) + source)
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        runs = [(CONTROLS[control][0], control)] if control else [(n, None) for n in ('S', 'P')]
        if not control:
            runs += [(case, mutation) for mutation, (case, _, _) in CONTROLS.items()]
        for name, mutation in runs:
            run = run_case(name, dxc, runner, adapter, Path(tmp), mutation)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = [line for line in run.stderr.splitlines()
                           if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
            if mutation:
                if run.returncode != 1 or not any('expected=' in line for line in diagnostics):
                    raise AssertionError('control %s: unexpected result (exit %d) %s' % (mutation, run.returncode, run.stderr))
                print('control %s: rejected at %s' % (mutation, diagnostics[0]))
            elif run.returncode or diagnostics:
                raise AssertionError('case %s: %s' % (name, run.stderr or 'exit %d' % run.returncode))
            else:
                print('case %s: all %d words match' % (name, WORDS_OUT))
    return 0


def main():
    args = sys.argv[1:]
    if not args:
        controls()
        probe_refusals()
        check_text()
        check_mutations()
    elif len(args) == 2 and args[0] == 'binary':
        check_binary(args[1])
    elif len(args) == 4 and args[0] == 'dispatch':
        controls()
        sys.exit(check_dispatch(*args[1:]) or (1 if failures else 0))
    elif len(args) == 5 and args[0] == 'control':
        sys.exit(check_dispatch(*args[2:], control=args[1]))
    else:
        sys.exit(__doc__)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
