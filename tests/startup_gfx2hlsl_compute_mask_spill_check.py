"""Constant-lane SGPR spills of lane masks in compute stages of tools/startup/gfx2hlsl.py (#1723).

  (no arguments)                                  translation controls and translator mutations, pure Python
  binary <llvm-mc>                                assemble the fixture sources (gfx1013) and compare with the words here
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                                  translate, compile with DXC cs_6_0, dispatch and compare every word
  control <m-saveexec-dropped|m-mask-inverted|v-branch-inverted> <dxc> <offscreen_check> <warp|hardware>
                                                  the same dispatch with a mutated translation, which must mismatch

`v_writelane_b32 vN, M, k` copies the whole wave mask M (vcc_lo, or an SGPR holding a lane mask) into lane k of vN
whatever EXEC is, and `v_readlane_b32 D, vN, k` copies it back to every lane (AMD RDNA 2 ISA reference, VOP3 lane
operations). The translator models a lane mask per invocation as that invocation's own bit (a v_cmp stores `w`, or
`exec & w` once EXEC is not known, so an inactive lane's bit is 0). A reload that is only used as a mask afterwards
sees, in every invocation, the bit that invocation had at the write: the spill is the identity on that bit, so the slot
variable takes the mask's own variable at the write and the reload marks the destination a mask again.

Fixtures (source-built gfx1013, 4 groups of 8 lanes, root constants key and stride in s4-s5, one descriptor table in s6):
- M spills the compare result vcc_lo (lanes below 5, lane 0 of v40) and an SGPR compare result s26 (lanes below 3,
  lane 1), branches on the key, clobbers both, reloads them into SGPRs, selects with v_cndmask_b32_e64 and runs a
  masked xor through s_and_saveexec_b32.
- V spills a vcc_lo that is the same in every lane of a group (2 > group index), clobbers vcc_lo, reloads it into
  vcc_lo itself, selects with v_cndmask_b32_e32 and skips an add with s_cbranch_vccz.
"""
import re
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / 'tools'
sys.path.insert(0, str(TOOLS / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402
import startup_gfx2hlsl_compute_spill_check as S  # noqa: E402

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + str(detail)[:1500])


PROLOGUE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[12:15], s[10:11], 0x0
s_load_dwordx4 s[16:19], s[10:11], 0x10
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 2, v3
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
"""
M_SPILL = """\
v_cmp_gt_u32 vcc_lo, 5, v0
v_writelane_b32 v40, vcc_lo, 0
v_cmp_gt_u32 s26, 3, v0
v_writelane_b32 v40, s26, 1
"""
M_BRANCH = """\
s_and_b32 s22, s4, 1
s_cmp_lg_u32 s22, 0
s_cbranch_scc0 Lelse
v_mov_b32 v7, 100
s_branch Ljoin
Lelse:
v_mov_b32 v7, 200
Ljoin:
"""
M_CLOBBER = """\
v_cmp_gt_u32 vcc_lo, 1, v0
s_mov_b32 s26, 0
"""
M_RELOAD = """\
v_readlane_b32 s25, v40, 0
v_readlane_b32 s27, v40, 1
"""
M_USE = """\
v_cndmask_b32_e64 v6, v5, v7, s25
v_cndmask_b32_e64 v6, v6, 7, s27
s_and_saveexec_b32 s23, s25
v_xor_b32 v6, 0x55, v6
s_mov_b32 exec_lo, s23
buffer_store_dword v6, v4, s[16:19], 0 offen
s_endpgm
"""
SOURCE_M = PROLOGUE + M_SPILL + M_BRANCH + M_CLOBBER + M_RELOAD + M_USE
SOURCE_V = PROLOGUE + """\
v_mov_b32 v8, s7
v_cmp_gt_u32 vcc_lo, 2, v8
v_writelane_b32 v40, vcc_lo, 0
v_cmp_gt_u32 vcc_lo, 1, v0
v_readlane_b32 vcc_lo, v40, 0
v_cndmask_b32_e32 v6, v5, v8, vcc_lo
s_cbranch_vccz Lskip
v_add_nc_u32 v6, 1000, v6
Lskip:
buffer_store_dword v6, v4, s[16:19], 0 offen
s_endpgm
"""
EXPECTED_ACCEPTED = {'M': SOURCE_M, 'V': SOURCE_V}

# name -> (base source, [(old, new)], what the refusal must say, why it is refused). Each variant is a fixture with one
# change; it is assembled in binary mode like the fixtures and must be refused for that reason and no other.
VARIANTS = {
    'm0-source': (SOURCE_M, [('v_writelane_b32 v40, s26, 1\n', 's_mov_b32 m0, 5\nv_writelane_b32 v40, m0, 1\n')],
                  'source m0 is a lane mask or m0', 'm0 is not a mask: it is refused as a spill source'),
    'exec-source': (SOURCE_M, [('v_writelane_b32 v40, s26, 1\n', 'v_writelane_b32 v40, exec_lo, 1\n')],
                    'source exec_lo is a lane mask or m0', 'exec is refused as a spill source'),
    'vcc-hi-source': (SOURCE_M, [('v_writelane_b32 v40, vcc_lo, 0\n', 'v_writelane_b32 v40, vcc_hi, 0\n')],
                      'source vcc_hi is a lane mask or m0', 'vcc_hi is not the wave64 mask half a caller states (wave_size=64 skips the wave32 rename)'),
    'numeric-use': (SOURCE_M, [(M_RELOAD, M_RELOAD + 'v_add_nc_u32 v9, s25, v5\n')], 'reads as data',
                    'the reloaded mask is read as numeric data'),
    'numeric-use-vcc': (SOURCE_V, [('Lskip:\n', 'Lskip:\nv_add_nc_u32 v9, vcc_lo, v5\n')], 'reads as data',
                        'the reloaded vcc_lo is read as numeric data'),
    'masks-joined': (SOURCE_M, [(M_BRANCH, M_BRANCH.replace('v_mov_b32 v7, 100\n', 'v_mov_b32 v7, 100\n'
                                                            'v_writelane_b32 v40, s26, 0\n'))],
                     'one of them a lane mask', 'one arm overwrites lane 0 with a different mask'),
    'mask-and-number-joined': (SOURCE_M, [(M_BRANCH, M_BRANCH.replace('v_mov_b32 v7, 100\n', 'v_mov_b32 v7, 100\n'
                                                                      's_mov_b32 s28, 9\nv_writelane_b32 v40, s28, 0\n'))],
                               'one of them a lane mask', 'one arm overwrites lane 0 with a number'),
    'reload-before-write': (SOURCE_M, [('v_writelane_b32 v40, s26, 1\n', '')], 'no v_writelane_b32 of it on some path',
                            'lane 1 is never written'),
    'mask-reloaded-into-vcc-from-number': (SOURCE_V, [('v_writelane_b32 v40, vcc_lo, 0\n', 's_mov_b32 s28, 1\n'
                                                       'v_writelane_b32 v40, s28, 0\n')],
                                           'requires an SGPR destination', 'a number is reloaded into vcc_lo'),
    'mask-from-one-arm': (SOURCE_M, [('v_cmp_gt_u32 s26, 3, v0\nv_writelane_b32 v40, s26, 1\n', 's_mov_b32 s26, 0\n'),
                                     ('v_mov_b32 v7, 100\n', 'v_mov_b32 v7, 100\nv_cmp_gt_u32 s26, 3, v0\n'),
                                     ('Ljoin:\n', 'Ljoin:\nv_writelane_b32 v40, s26, 1\n')],
                      'lane mask not proven on every path', 'the mask is computed in one arm only'),
    'write-in-execz': (SOURCE_M, [(M_CLOBBER, M_CLOBBER + 's_mov_b32 s21, exec_lo\nv_cmp_gt_u32 vcc_lo, 5, v0\n'
                                   's_mov_b32 exec_lo, vcc_lo\ns_cbranch_execz Lz\nv_writelane_b32 v40, vcc_lo, 2\n'
                                   'Lz:\ns_mov_b32 exec_lo, s21\n')],
                       'inside a block s_cbranch_execz skips per lane', 'a mask written inside an execz region'),
    'loop-carried': (SOURCE_M, [('Ljoin:\n', 'Ljoin:\ns_mov_b32 s22, 0\nLloop:\nv_readlane_b32 s25, v40, 0\n'
                                 'v_writelane_b32 v40, s25, 0\ns_add_u32 s22, s22, 1\ns_cmp_lt_u32 s22, 2\n'
                                 's_cbranch_scc1 Lloop\n')],
                     'written later in a loop', 'a mask reloaded by a loop body that writes it again'),
}


def variant_source(name):
    base, edits = VARIANTS[name][:2]
    for old, new in edits:
        assert base.count(old) == 1, (name, old)
        base = base.replace(old, new)
    return base


def all_sources():
    return {**EXPECTED_ACCEPTED, **{name: variant_source(name) for name in VARIANTS}}


ENCODING = (C.USER7 | C.TGID_X | C.tidig(1), (8, 1, 1), (4, 1, 1))
WORDS = {
    'M': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d71a02010004811687168007bf030084bfff020e7e'
        '64000000020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100060001d5050f6600'
        '060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'V': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf0702107e8210887d280061d76a0001008100887d6a0060d72801010005110c02020086bfff0c0c4ae8030000'
        '001070e004060480000081bf'),
    'm0-source': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d4830002008503fcbe280061d77c02010004811687168007bf030084bf'
        'ff020e7e64000000020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100060001d5'
        '050f6600060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'exec-source': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d77e02010004811687168007bf030084bfff020e7e'
        '64000000020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100060001d5050f6600'
        '060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'vcc-hi-source': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76b0001001a00c4d483000200280061d71a02010004811687168007bf030084bfff020e7e'
        '64000000020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100060001d5050f6600'
        '060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'numeric-use': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d71a02010004811687168007bf030084bfff020e7e'
        '64000000020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100190a124a060001d5'
        '050f6600060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'numeric-use-vcc': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf0702107e8210887d280061d76a0001008100887d6a0060d72801010005110c02020086bfff0c0c4ae8030000'
        '6a0a124a001070e004060480000081bf'),
    'masks-joined': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d71a02010004811687168007bf050084bfff020e7e'
        '64000000280061d71a000100020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100'
        '060001d5050f6600060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'mask-and-number-joined': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d71a02010004811687168007bf060084bfff020e7e'
        '6400000089039cbe280061d71c000100020082bfff020e7ec80000008100887d80039abe190060d7280101001b0060d7'
        '28030100060001d5050f6600060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'reload-before-write': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d48300020004811687168007bf030084bfff020e7e64000000020082bf'
        'ff020e7ec80000008100887d80039abe190060d7280101001b0060d728030100060001d5050f6600060001d5060f6d00'
        '193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'mask-reloaded-into-vcc-from-number': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf0702107e8210887d81039cbe280061d71c0001008100887d6a0060d72801010005110c02020086bfff0c0c4a'
        'e8030000001070e004060480000081bf'),
    'mask-from-one-arm': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a00010080039abe04811687168007bf050084bfff020e7e640000001a00c4d483000200'
        '020082bfff020e7ec8000000280061d71a0201008100887d80039abe190060d7280101001b0060d728030100060001d5'
        '050f6600060001d5060f6d00193c97beff0c0c3a550000001703febe001070e004060480000081bf'),
    'write-in-execz': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d71a02010004811687168007bf030084bfff020e7e'
        '64000000020082bfff020e7ec80000008100887d80039abe7e0395be8500887d6a03febe020088bf280061d76a040100'
        '1503febe190060d7280101001b0060d728030100060001d5050f6600060001d5060f6d00193c97beff0c0c3a55000000'
        '1703febe001070e004060480000081bf'),
    'loop-carried': bytes.fromhex(
        '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380'
        '703f8cbf8500887d280061d76a0001001a00c4d483000200280061d71a02010004811687168007bf030084bfff020e7e'
        '64000000020082bfff020e7ec8000000800396be190060d728010100280061d7190001001681168016820abff9ff85bf'
        '8100887d80039abe190060d7280101001b0060d728030100060001d5050f6600060001d5060f6d00193c97beff0c0c3a'
        '550000001703febe001070e004060480000081bf'),
}
WORDS_OUT = C.WORDS
SENTINEL = C.SENTINEL
KEY = C.KEY


WAVE64_VARIANTS = {'vcc-hi-source'}   # the caller states wave64, so vcc_hi is the high half of VCC, not a renamed SGPR


def translate(name, code=None, module=G):
    return module.ComputePipeline(code or WORDS[name], C.root_blob(), ENCODING[0], ENCODING[1],
                                  wave_size=64 if name in WAVE64_VARIANTS else None)


def refusal(name, module=G):
    try:
        translate(name, module=module).hlsl_cs()
    except module.Unsupported as reason:
        return str(reason)
    return None


def expected(name):
    out = [SENTINEL] * WORDS_OUT
    for gx in range(4):
        for tx in range(8):
            i, word = gx * 8 + tx, C.input_word(gx * 8 + tx)
            if name == 'M':
                v7 = 100 if KEY & 1 else 200
                a, b = tx < 5, tx < 3
                word = v7 if a else word
                word = 7 if b else word
                out[i] = word ^ 0x55 if a else word
            else:
                word = gx if gx < 2 else word
                out[i] = word + 1000 if gx < 2 else word
    return out


def controls(module=G, quiet=False):
    """-> the labels of the controls that failed (all printed unless quiet)."""
    red = []

    def verdict(label, ok, detail=''):
        if not ok:
            red.append(label)
        if not quiet:
            check(label, ok, detail)

    for name in EXPECTED_ACCEPTED:
        reason = refusal(name, module)
        verdict('fixture %s translates (%s)' % (name, reason), reason is None)
    for name, (_, _, fragment, why) in VARIANTS.items():
        if name == 'write-in-execz':
            # Exercise the per-lane EXECZ check before group voting changes the spill rules.
            with patch.object(module.ComputePipeline, 'retry_group_vote_for', return_value=False):
                reason = refusal(name, module)
        else:
            reason = refusal(name, module)
        verdict('refused: %s (%s)' % (why, reason), reason is not None and fragment in reason, reason)
    verdict('mask spills stay refused in a pixel stage (%s)' % probe_pixel(module), 'holds a lane mask or descriptor' in
            probe_pixel(module))
    return red


def probe_pixel(module):
    """A pixel stage that spills a lane mask keeps the refusal of step 1: only compute stages model the mask this way."""
    stage = module.Stage.__new__(module.Stage)
    stage.kind, stage.lines, stage.sym = 'ps', [], {}
    stage.used_s, stage.used_v, stage.uniform_v, stage.varying_s = set(), set(), set(), set()
    stage.scc_uniform, stage.exec_known, stage.has_branches, stage.blocks, stage.masks = True, True, False, [], {20}
    stage.join_masks = set()
    stage.ins = [(0, 1, 'v_writelane_b32', 'v_writelane_b32 v10, s20, 3')]
    try:
        stage.one('v_writelane_b32', 'v_writelane_b32 v10, s20, 3')
    except module.Unsupported as why:
        return str(why)
    return 'accepted'


# Each translator mutation must turn at least one control red.
MUTATIONS = {
    'a mask is not marked again after the reload': ('        if mask:\n            self.masks.add(dst[1])',
                                                    '        if False:\n            self.masks.add(dst[1])'),
    'a reloaded mask is read as data': ("                if not (self.per_lane_read(j, key) or copy or", "                if not (True or copy or"),
    'a lane select is not a mask read': ("                if not (self.per_lane_read(j, key) or copy or", "                if not (copy or"),
    'two writes with a mask join': ('        if any(s[3] for s in stored) and len(stored) > 1:', '        if False:'),
    'm0 is a mask source': ("            elif ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0'):",
                            "            elif ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi'):"),
    'exec is a mask source': ("            elif ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0'):",
                              "            elif ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'm0'):"),
    'vcc_hi is a mask source': ("            if ops[1] == 'vcc_lo' and self.spills_lane_masks():",
                                "            if ops[1] in ('vcc_hi', 'vcc_lo') and self.spills_lane_masks():"),
    'a mask not proven on every path is spilled': ('                if src[1] in self.join_masks:', '                if False:'),
    'the write stores the inverted bit': ("value, sym, uniform, proven, mask = 'vcc', None",
                                         "value, sym, uniform, proven, mask = '(vcc ^ 1u)', None"),
    'the reload restores the neighbouring slot': ("        self.set_s(dst[1], 'sp%d_%d' % (n, lane), sym=None if mask else sym,",
                                                  "        self.set_s(dst[1], 'sp%d_%d' % (n, lane ^ int(mask)), sym=None if mask else sym,"),
    'every stage spills masks': ("        return getattr(self, 'kind', None) == 'cs'", '        return True'),
    'a number is reloaded into vcc_lo': ("            if not (mask and ops[0] == 'vcc_lo'):", "            if ops[0] != 'vcc_lo':"),
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
            module = mutated(old, new)
            red = controls(module, quiet=True) + check_text(module, quiet=True)
        except Exception as error:  # a mutant that crashes the translator is caught as well
            red = ['crash: %s' % error]
        check('mutation "%s" turns %d control(s) red: %s' % (label, len(red), red[:3]), bool(red))


def check_text(module=G, quiet=False):
    """-> the labels of the text checks that failed. The slot takes the mask's variable at the write and the reload copies
    it back, which is what makes the spill the identity on each invocation's bit."""
    red = []

    def verdict(label, ok, detail=''):
        if not ok:
            red.append(label)
        if not quiet:
            check(label, ok, detail)

    m, v = translate('M', module=module).hlsl_cs(), translate('V', module=module).hlsl_cs()
    lines = [line.strip() for line in m.splitlines()]
    verdict('M: the slots take the mask variables, the reloads copy them back',
            'sp40_0 = vcc;' in lines and 'sp40_1 = s26;' in lines and 's25 = sp40_0;' in lines and 's27 = sp40_1;' in lines, m)
    verdict('M: the compare writes the bit of its own invocation', 'vcc = ((0x00000005u > v0) ? 1u : 0u);' in lines, m)
    verdict('M: the reloaded masks select and narrow EXEC', 'v6 = (s25 & 1) ? v7 : v5;' in lines and
            'exec = (exec & s25) & 1; scc = exec != 0;' in lines, m)
    vl = [line.strip() for line in v.splitlines()]
    verdict('V: vcc_lo is reloaded into vcc and drives the branch',
            'sp40_0 = vcc;' in vl and 'vcc = sp40_0;' in vl and 'pc = ((vcc & 1) == 0) ? 2 : 1;' in vl, v)
    return red


# ---- binary and dispatch ---------------------------------------------------------------------------------------------
def check_binary(llvm_mc):
    for name, source in all_sources().items():
        built = S.assemble_labels(llvm_mc, source)
        check('llvm-mc assembles %s to the recorded words' % name, built == WORDS[name], built.hex())


# control -> (case, text the translation must contain once, replacement)
CONTROLS = {
    'm-saveexec-dropped': ('M', 'exec = (exec & s25) & 1; scc = exec != 0;', 'scc = exec != 0;'),
    'm-mask-inverted': ('M', 'sp40_0 = vcc;', 'sp40_0 = vcc ^ 1u;'),
    'v-branch-inverted': ('V', 'pc = ((vcc & 1) == 0) ? 2 : 1;', 'pc = ((vcc & 1) != 0) ? 2 : 1;'),
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
    groups = ENCODING[2]
    constants = [KEY, C.STRIDE] + [0] * 14
    case.write_bytes(struct.pack('<I', groups[0] | groups[1] << 8 | groups[2] << 16)
                     + struct.pack('<%dI' % WORDS_OUT, *expected(name)) + struct.pack('<16I', *constants))
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        runs = [(CONTROLS[control][0], control)] if control else [(n, None) for n in ('M', 'V')]
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
