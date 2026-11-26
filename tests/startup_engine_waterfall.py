"""Source-built four-lane numeric waterfall for the compute readfirstlane fixture (#2644)."""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
import struct
import subprocess

from compute_load_format_check import check_references
from startup_engine_compute_check import G, MASK, SENTINEL, USER7, WORDS, assemble, refused, root_blob


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
# The compilers' pick: v_cmp_eq of the key into VCC, then s_and_saveexec keeps the remaining lanes in s21 (no copy of EXEC before the loop).
COMPARE_EQ = ('v_cmp_eq_u32_e32 vcc_lo, s22, v5', '160a847d')
COMPARE_EQ_WRONG = ('v_cmp_eq_u32_e32 vcc_lo, s22, v4', '1608847d')
SAVEEXEC = ('s_and_saveexec_b32 s21, vcc_lo', '6a3c95be')
SITE = ('v_readfirstlane_b32 s24, v7', '0705307e')
RESTORE = ('s_mov_b32 exec_lo, s26', '1a03febe')
STORE = ('buffer_store_dword v7, v4, s[16:19], 0 offen', '001070e004070480')
END = ('s_endpgm', '000081bf')
FIXED_LOAD = (('s_load_dwordx4 s[28:31], s[10:11], 0', '050708f4000000fa'), ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
              ('v_mov_b32_e32 v8, s22', '1602107e'), ('buffer_load_dword v7, v8, s[28:31], 0 offen', '001030e008070780'),
              ('s_waitcnt vmcnt(0)', '703f8cbf'))


def program(*, before_save=(), after_save=(), compare=COMPARE, body=(ADD, XOR), after_restore=(), saveexec=False):
    loop = (HEAD, *compare_rows(compare, saveexec), *body, CLOSE)
    offset = -(sum(len(bytes.fromhex(code)) for _, code in loop) // 4 + 1)
    branch = ('s_cbranch_scc1 %d' % offset, struct.pack('<I', 0xbf850000 | (offset & 0xffff)).hex())
    save = () if saveexec else (SAVE,)
    return (*PREFIX, *before_save, *save, *after_save, *loop, branch, RESTORE, *after_restore, STORE, END)


def compare_rows(compare, saveexec):
    """The pick of the loop: v_cmpx of the key (the default), or with saveexec a v_cmp_eq into VCC and s_and_saveexec."""
    if not saveexec:
        return (compare,)
    return (COMPARE_EQ if compare == COMPARE else compare, SAVEEXEC)


# First-lane readfirstlane (RFL3): the lanes above the root constant s4 stay enabled, s22 reads v5 of the lowest of them
# (lane 0 when none is enabled) and every lane adds it to 13 after EXEC is restored.
ABOVE = ('v_cmpx_lt_u32_e32 s4, v0', '0400a27d')
FIRST_READ = ('v_readfirstlane_b32 s22, v5', '05052c7e')
ADD_FIRST = ('v_add_nc_u32_e32 v7, s22, v7', '160e0e4a')


def first_lane_program():
    return (*PREFIX[:-1], ABOVE, FIRST_READ, RESTORE, ADD_FIRST, STORE, END)


def first_lane_expected(lanes, active, source, wave=32):
    """Per thread: 13 plus the v5 word of the lowest enabled thread of its own guest wave (its lane 0 when none)."""
    out = []
    for lane in range(lanes):
        base = lane // wave * wave
        enabled = [t for t in range(base, min(base + wave, lanes)) if t > active]
        out.append((13 + source[enabled[0] if enabled else base]) & MASK)
    return out


def translate(rows, group=(4, 1, 1), wave=None):
    code = bytes.fromhex(''.join(word for _, word in rows))
    return G.ComputePipeline(code, root_blob(), USER7, group, wave=wave).hlsl_cs()


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
    # A b32 close covers only the low 32 lanes of a wave64: more than 32 threads are one wave per 32 when the guest wave is 32
    # (each runs its own pass, per lane), and a wave-wide matter for the tick model when it is 64.
    assert plain_translation(program(), (33, 1, 1)) and plain_translation(program(), (33, 1, 1), 32)
    assert not plain_translation(program(), (33, 1, 1), 64)
    saveexec_controls(llvm_mc)
    # A source that differs between lanes is read from the lowest enabled lane of its guest wave (WaveSync gather).
    if llvm_mc:
        rows = first_lane_program()
        source = '\n'.join(asm for asm, _ in rows) + '\n'
        assert assemble(llvm_mc, source) == bytes.fromhex(''.join(word for _, word in rows)), source
        wave64 = assemble(llvm_mc, 's_and_saveexec_b64 s[0:1], vcc\nv_mov_b32 v0, 1\n')
        assert G.guest_wave_size(wave64) == 64 and G.guest_wave_size(bytes.fromhex(''.join(w for _, w in rows))) == 32
    for lanes, wave in ((4, 32), (32, 32), (33, 32), (64, 32), (64, 64)):
        text = translate(first_lane_program(), (lanes, 1, 1), wave)
        assert 'if ((exec & 1) != 0u) InterlockedMin(x2w_first[' in text and 'x2w_value_' in text, text
        assert 'groupshared uint x2w_first[3][%d];' % -(-lanes // wave) in text, text
    refused('guest wave size', lambda: translate(first_lane_program(), (4, 1, 1), 128))


def plain_translation(rows, group=(4, 1, 1), wave=None):
    """True when the program translates in the per-lane model (each lane runs its own pass of the waterfall)."""
    code = bytes.fromhex(''.join(word for _, word in rows))
    try:
        pipe = G.ComputePipeline(code, root_blob(), USER7, group, wave=wave)
        pipe.hlsl_cs()
    except G.Unsupported:
        return False
    return not getattr(pipe.cs, 'uses_tick', False)


def saveexec_controls(llvm_mc=None):
    """The compilers' waterfall (v_cmp_eq + s_and_saveexec pick, s_andn2_wrexec close) takes the per-lane path exactly where
    the v_cmpx form does: a body scalar that is live after the loop, loop-carried scalar state, a mask or a wave-wide read in
    the body, or a pick on another register keep it off the per-lane path (refused, or the wave-synchronized tick model)."""
    live = ('v_mov_b32_e32 v7, s23', '17020e7e')
    copy = ('s_mov_b32 s24, s21', '150398be')
    positives = (
        program(saveexec=True),
        program(saveexec=True, body=FIXED_LOAD),
        program(saveexec=True, body=(CONST, XOR), after_restore=(USE_CONST,)),
    )
    negatives = (
        program(saveexec=True, body=(CONST, ('s_add_u32 s23, s23, s22', '17161780'), XOR), after_restore=(USE_CONST,)),
        program(saveexec=True, after_restore=(live,)),
        program(saveexec=True, body=(('s_add_u32 s23, s23, 1', '17811780'), XOR)),
        program(saveexec=True, body=(copy, XOR)),
        program(saveexec=True, body=(SITE, ADD, XOR)),
        program(saveexec=True, compare=COMPARE_EQ_WRONG),
    )
    for rows in (*positives, *negatives):
        if llvm_mc:
            source = '\n'.join(asm for asm, _ in rows) + '\n'
            assert assemble(llvm_mc, source) == bytes.fromhex(''.join(word for _, word in rows)), source
    for rows in positives:
        assert plain_translation(rows), 'a waterfall the per-lane proof covers went to the tick model'
        assert 's22 = v5;' in translate(rows)
    for rows in negatives:
        assert not plain_translation(rows), 'a waterfall the per-lane proof does not cover was translated per lane'
    # b32 close with more than 32 threads: fine when the guest wave is known to be 32 (each guest wave runs its own pass)
    assert plain_translation(program(saveexec=True), (64, 1, 1))


def first_lane_dispatch(directory, dxc, runner, adapter):
    """Groups of 32 and 64 threads (one and two guest waves) are many host waves on WARP (4 lanes) and some GPUs: the
    lowest enabled lane must not depend on how the host splits the group."""
    cases = []
    shaders = {}
    for lanes in (32, 64):
        src, shader = directory / ('first_lane%d.hlsl' % lanes), directory / ('first_lane%d.cso' % lanes)
        src.write_text(translate(first_lane_program(), (lanes, 1, 1), 32), encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        shaders[lanes] = shader
    source = [(0x9e3779b1 * (i + 1) + 7) & MASK for i in range(WORDS)]
    for lanes, shader in shaders.items():
        cases = []
        for active in (0, 5, 16, 30, 31, 40, 63):
            output = first_lane_expected(lanes, active, source)
            words = [1, *output, *([SENTINEL] * (WORDS - lanes)), active, *([0] * 15), *source]
            case = directory / ('first-lane-%d-%d.case' % (lanes, active))
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, '%d threads, lanes above %d enabled' % (lanes, active), None))
            if active == 16:
                wrong = list(output)
                lane = 5  # lane 0 of its wave instead of its first enabled lane (17)
                wrong[lane] = (13 + source[0]) & MASK
                words = [1, *wrong, *([SENTINEL] * (WORDS - lanes)), active, *([0] * 15), *source]
                case = directory / ('first-lane-%d-wrong.case' % lanes)
                case.write_bytes(struct.pack('<%dI' % len(words), *words))
                cases.append((case, 'lane 0 instead of the first enabled lane',
                              'word=%d expected=%08x actual=%08x' % (lane, wrong[lane], output[lane])))
        status = check_references('first-lane readfirstlane %d' % lanes, runner, adapter, shader, cases)
        if status:
            return status
    return 0


def dispatch(directory, dxc, runner, adapter):
    if first_lane_dispatch(directory, dxc, runner, adapter):
        return 77
    for label, body, via_saveexec in (('arithmetic', (ADD, XOR), False), ('fixed-buffer', FIXED_LOAD, False),
                                      ('invariant', (CONST, XOR), False), ('saveexec arithmetic', (ADD, XOR), True),
                                      ('saveexec invariant', (CONST, XOR), True)):
        src, shader = directory / 'waterfall.hlsl', directory / 'waterfall.cso'
        after = (USE_CONST,) if label.endswith('invariant') else ()
        src.write_text(translate(program(body=body, after_restore=after, saveexec=via_saveexec)), encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        cases = []
        inputs = [(n, (64, 32, 64, 96)) for n in range(5)]
        if label.endswith('arithmetic'):
            inputs.append((4, (MASK, MASK, 0, 0)))
        for active, keys in inputs:
            source = [17 * i + 9 for i in range(WORDS)]
            source[:4] = keys
            output = [13] * 4
            for lane, key in enumerate(keys[:active]):
                output[lane] = (((key + 17) & MASK) ^ lane) if label.endswith('arithmetic') else source[key // 4]
                if label.endswith('invariant'):  # the body scalar is 17 in every pass: v7 = (17 ^ lane) + 17
                    output[lane] = (17 ^ lane) + 17
            if label.endswith('invariant'):
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
