"""Synthetic gfx1013 pixel program for the forward lane-0 proof checks."""
import struct

from startup_image_3d_check import G, VS_WORDS, root_blob

PROGRAM = bytes.fromhex(
    '098006bf 090085bf'                         # if s9 (entry flag) is nonzero
    '040508f4200000fa 7fc08cbf'                 # s_load_dwordx4 s[20:23], s[8:9], 32
    '040000e000070580 703f8cbf 0705167e'        # formatted load at byte 4, read lane 0 into s11
    '81038cbe 020082bf'                         # save 1 in s12, jump to join
    '0a038bbe 80038cbe'                         # else s11 = s10 (entry fallback), s12 = 0
    '0c8106bf 020084bf'                         # if s12 == 1 (produces the SCC used below)
    'f4038dbe 010082bf f6038dbe'                # s13 = 2.0, else s13 = 4.0
    '0df00e85'                                 # s_cselect_b32 s14, s13, 0.5
    '0e02107e 08051e7e 0b02127e 0905207e'      # read lane 0 from copies of s14 and s11
    '000d027e 10020406 0d02067e 0f02087e'      # float(packed pos), add selected value, copy scales
    '0f0800f802010304 000081bf'                 # export (sum, packed pos, branch scale, select scale)
)

REFUSALS = {
    'per-pixel buffer index': PROGRAM.replace(bytes.fromhex('040000e000070580'),
                                              bytes.fromhex('042000e000070580')),  # idxen v0
    'per-pixel SCC branch': bytes.fromhex('8000847d 6a810487') + PROGRAM[4:],  # cmp v0, SCC from VCC
    'SCC written on just one arm': PROGRAM.replace(bytes.fromhex('f4038dbe'),
                                                    bytes.fromhex('f4800d88')),  # s_or_b32 writes SCC
    'saved flag written on just one arm': PROGRAM.replace(bytes.fromhex('80038cbe'),
                                                          bytes.fromhex('000080bf')),  # s_nop
}


def pipeline(code=PROGRAM, refusals=None):
    # Expand the fixture's b0 root constants from one word to flag and fallback.
    root = root_blob().replace(struct.pack('<3I', 0, 0, 1) + bytes(20),
                               struct.pack('<3I', 0, 0, 2) + bytes(20))
    decoded = {'vs': G.decode(VS_WORDS), 'ps': G.decode(code)} if refusals is not None else None
    return G.Pipeline(VS_WORDS, code, root, ps_hw_inputs=(0x8000, 0x8000),
                      refusals=refusals, decoded=decoded)


def check_translation():
    accepted = pipeline()
    hlsl = accepted.hlsl_ps()
    collected = []
    assert pipeline(refusals=collected).hlsl_ps() == hlsl and not collected, collected
    assert 's11 = v7;' in hlsl and 's14 = scc ? s13 : 0x3f000000u;' in hlsl
    assert 'Wave' not in hlsl
    for name, code in REFUSALS.items():
        assert code != PROGRAM, name
        try:
            pipeline(code)
        except G.Unsupported as reason:
            assert str(reason) == G.READFIRSTLANE_UNIFORM, (name, str(reason))
        else:
            raise AssertionError(f'{name} translated')
        collected = []
        pipeline(code, refusals=collected)
        assert (G.READFIRSTLANE_UNIFORM, 'ps') in collected, (name, collected)
        print(f'{name}: {G.READFIRSTLANE_UNIFORM}')
    return hlsl, accepted.uds


def whole_quad_program(control=None):
    """Build a pixel program with a masked colour write and a wave-wide join."""
    def word(value):
        return struct.pack('<I', value)

    def scalar_move(dst, src, opcode=4):
        return word(0xbe800000 | dst << 16 | opcode << 8 | src)

    def vector_unary(opcode, dst, src):
        return word(0x7e000000 | dst << 17 | opcode << 9 | src)

    def branch(opcode, skipped):
        return word(0xbf800000 | opcode << 16 | ((skipped // 4) & 0xffff))

    exec_reg = 126
    flag_compare = word(0xbf068009)  # s_cmp_eq_u32 s9, 0
    # 0x3ffff < packed pixel position exactly when y >= 4 on this target.
    narrow = word(0x7c000000 | 0xd1 << 17 | 255) + word(0x3ffff)
    restore = scalar_move(exec_reg, 34)
    overwrite = scalar_move(35, 128, opcode=3)
    descriptor = word(0xf4080000 | 28 << 6 | 4) + word(0xfa000030)
    load = word(0xe0000000 | 8) + word(0x80000000 | 7 << 16 | 13 << 8)
    if control == 'indexed load':
        load = word(0xe0000000 | 0x2000 | 8) + load[4:]
    wait_descriptor = word(0xbf8cc07f)
    wait_load = word(0xbf8c3f70)
    read = vector_unary(2, 17, 256 + 13)
    then = descriptor + wait_descriptor
    if control == 'narrowed read':
        then += narrow
    then += load + wait_load
    if control == 'EXEC write between load and read':
        then += restore
    if control == 'split load and read':
        then += branch(2, 0)
    if control == 'overwritten load':
        then += vector_unary(1, 13, 256)
    then += read
    if control == 'narrowed read':
        then += restore
    if control == 'mask-changing loop':
        then += narrow + flag_compare
        then += branch(4, -len(then) - 4)
    otherwise = scalar_move(17, 10, opcode=3)
    if control == 'swapped arms':
        then, otherwise = otherwise, then

    code = scalar_move(32, exec_reg) + scalar_move(exec_reg, exec_reg, opcode=10)
    code += scalar_move(34, exec_reg)
    code += vector_unary(1, 5, 128) + narrow + vector_unary(1, 5, 242)  # blue = 0 or 1
    if control == 'saved pair overwritten on one path':
        code += flag_compare + branch(5, len(overwrite)) + overwrite
    code += scalar_move(exec_reg, 36) if control == 'unknown restore' else restore
    if control in ('AND on one path', 'AND-NOT on one path'):
        opcode, rhs = (0x0f, 128) if control == 'AND on one path' else (0x15, 126)
        code += flag_compare + branch(5, 4)
        code += word(0x80000000 | opcode << 23 | exec_reg << 16 | rhs << 8 | exec_reg)
    join = branch(2, len(otherwise))
    code += flag_compare + branch(5, len(then + join)) + then + join + otherwise
    code += vector_unary(6, 6, 256)  # green = float(packed position)
    code += word(3 << 25 | 4 << 17 | 6 << 9 | 17)  # red = selected scalar + green
    code += vector_unary(1, 7, 242)  # alpha = 1
    return code + word(0xf800080f) + word(0x07050604) + word(0xbf810000)


def check_whole_quad_translation():
    accepted = pipeline(whole_quad_program())
    hlsl = accepted.hlsl_ps()
    collected = []
    assert pipeline(whole_quad_program(), refusals=collected).hlsl_ps() == hlsl and not collected, collected
    assert 's17 = v13;' in hlsl and 'Wave' not in hlsl
    assert accepted.ps.firstlane_ps_exec == set()  # exact entry EXEC still excludes WQM
    for control in ('narrowed read', 'saved pair overwritten on one path',
                    'EXEC write between load and read', 'indexed load', 'unknown restore',
                    'split load and read', 'overwritten load', 'mask-changing loop',
                    'AND on one path', 'AND-NOT on one path'):
        code = whole_quad_program(control)
        try:
            pipeline(code)
        except G.Unsupported as reason:
            assert str(reason) == G.READFIRSTLANE_UNIFORM, (control, str(reason))
        else:
            raise AssertionError(f'{control} translated')
        collected = []
        pipeline(code, refusals=collected)
        assert (G.READFIRSTLANE_UNIFORM, 'ps') in collected, (control, collected)
        print(f'{control}: {G.READFIRSTLANE_UNIFORM}')
    swapped = pipeline(whole_quad_program('swapped arms'))
    assert swapped.uds == accepted.uds
    return hlsl, swapped.hlsl_ps(), accepted.uds
