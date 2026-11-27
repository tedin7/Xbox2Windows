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
