"""Synthetic pixel joins with an overwritten SGPR, and a pixel AOT WARP draw."""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_image_3d_check import G, VS_WORDS, root_blob


def scalar_load(op, dst, base, offset):
    return (0xf4000000 | op << 18 | dst << 6 | base // 2, 125 << 25 | offset)


def compare(op, src0, src1):
    return 0xbf000000 | op << 16 | src1 << 8 | src0


def move(dst, src):
    return 0x7e000200 | dst << 17 | src


def encode_program(words, labels, branches):
    for at, (op, target) in branches.items():
        words[at] = 0xbf800000 | op << 16 | (labels[target] - at - 1)
    return struct.pack(f'<{len(words)}I', *words), {name: at * 4 for name, at in labels.items()}


def program(overwrite=None, *, firstlane=False):
    words, labels, branches = [], {}, {}

    def label(name):
        labels[name] = len(words)

    def branch(op, target):
        branches[len(words)] = (op, target)
        words.append(0)

    nop = 0xbf800000
    # A wave vote on the varying packed pixel position guards the old definitions.
    words.extend((0x7d8200ff, 0x0003ffff))  # v_cmp_lt_u32_e32 vcc_lo, 0x3ffff, v0 (y >= 4)
    branch(0x06, 'rewrite')  # s_cbranch_vccz
    words.append(0xbea40480)  # s_mov_b64 s[36:37], 0
    words.append(0xbea4047e)  # s_mov_b64 s[36:37], exec
    label('rewrite')
    words.extend(scalar_load(2, 32, 8, 0))  # table descriptor in s[32:35]
    words.append(0xbf8cc07f)  # s_waitcnt lgkmcnt(0)
    words.extend(scalar_load(9, 36, 32, 0x10))
    words.extend(scalar_load(9, 40, 32, 0x20))
    words.extend(scalar_load(9, 42, 32, 0x30))
    words.append(0xbf8cc07f)

    for diamond, scalar in enumerate((40, 42)):
        words.append(compare(0x13, scalar, 128))  # s_cmp_lg_u64 s[scalar:scalar+1], 0
        branch(0x04, f'else{diamond}')  # s_cbranch_scc0
        words.append(move(6 + diamond, 242))  # 1.0
        words.append(0xbea40380 if overwrite == diamond else nop)  # s_mov_b32 s36, 0
        branch(0x02, f'join{diamond}')
        label(f'else{diamond}')
        words.append(move(6 + diamond, 240))  # 0.5
        label(f'join{diamond}')
        if diamond == 0 and firstlane:
            words.append(0x7e280506)  # v_readfirstlane_b32 s20, v6

    words.append(compare(0x0d, 36, 144))  # s_bitcmp1_b32 s36, 16
    words.append(move(8, 240))
    label('bit_branch')
    branch(0x04, 'export')
    words.append(move(8, 242))
    label('export')
    words.extend((move(9, 20 if firstlane else 256 + 6), move(10, 242),
                  0xf800180f, 0x0a070908, 0xbf810000))
    return encode_program(words, labels, branches)


def single_edge_program(*, fallthrough=False, shared=False, per_lane=False):
    words = []
    branches = {}

    def branch(op, target):
        branches[len(words)] = (op, target)
        words.append(0)

    if not per_lane:
        # A skipped scalar write read later asks the existing translator for wave-wide branch lowering.
        words.extend((0x7d8200ff, 0x0003ffff))  # v_cmp_lt_u32_e32 vcc_lo, 0x3ffff, v0
        branch(0x06, 'load')
        words.extend((0xbea50380, 0xbea4047e))  # s_mov_b32 s37, 0; s_mov_b64 s[36:37], exec
    labels = {'load': len(words)}
    words.extend(scalar_load(2, 32, 8, 0))
    words.append(0xbf8cc07f)
    words.extend(scalar_load(8, 37, 32, 0x10))
    words.append(0xbf8cc07f)
    words.extend((0x8029c125, compare(0x0a, 41, 143)))  # s41 = s37 - 1; SCC = s41 < 15
    if per_lane:
        words.extend((0x7d8200ff, 0x0003ffff))  # position-dependent VCC
    else:
        words.extend((0xbeaf03ff, 0x3f800000))  # s_mov_b32 s47, 0x3f800000
    labels['enter'] = len(words)
    branch(0x06 if per_lane else 0x04, 'else')
    if per_lane:
        # A mask write does not activate the scalar-data wave vote heuristic.
        words.append(0xbea4047e)  # s_mov_b64 s[36:37], exec
    else:
        words.append(0x80258125)  # s_add_u32 s37, s37, 1
        words.extend((0xbeaf03ff, 0x3f000000, move(9, 47)))  # s47 = 0.5; green = s47
    words.append(move(8, 240))  # first arm: red = 0.5
    if not fallthrough:
        branch(0x02, 'else' if shared else 'export')
    labels['else'] = len(words)
    if not per_lane:
        words.append(move(9, 47))  # green = the constant carried by the taken edge
    words.extend((0x800bd025, compare(0x0a, 11, 143)))  # s11 = s37 - 16; SCC = s11 < 15
    labels['test'] = len(words)
    branch(0x04, 'outside')
    words.append(move(8, 242))  # inner range: red = 1.0
    branch(0x02, 'export')
    labels['outside'] = len(words)
    words.append(move(8, 128))  # outside both ranges: red = 0.0
    labels['export'] = len(words)
    words.extend((move(10, 242), move(11, 128), 0xf800180f, 0x0a0b0908, 0xbf810000))
    return encode_program(words, labels, branches)


def pipeline(code, refusals=None):
    decoded = {'vs': G.decode(VS_WORDS), 'ps': G.decode(code)} if refusals is not None else None
    return G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=(0x8000, 0x8000),
                      refusals=refusals, decoded=decoded)


def joins(ins, refine=True):
    stage = G.Stage.__new__(G.Stage)
    stage.ins = ins
    return stage.joined_registers(with_cfg=True, refine=refine)[:2]


def require_refusal(code, reason):
    try:
        pipeline(code)
    except G.Unsupported as error:
        assert str(error) == reason, str(error)
    else:
        raise AssertionError(f'expected refusal: {reason}')


def single_edge_controls(wave_refusal):
    code, labels = single_edge_program()
    accepted = pipeline(code)
    assert accepted.ps.refine_joins and accepted.ps.refine_edges
    assert {labels['enter'], labels['test']} <= accepted.ps.uniform_branches
    assert 37 in joins(G.decode(code))[0][labels['else']], 'the sibling write must remain in the CFG join set'
    else_lines = next(lines for off, lines, _ in accepted.ps.blocks if off == labels['else'])
    assert 'v9 = 0x3f800000u;' in else_lines, else_lines
    hlsl = accepted.hlsl_ps()
    notes = []
    assert pipeline(code, notes).hlsl_ps() == hlsl and not notes, notes
    print('sole wave-uniform edge: original scalar and symbol proven in the other arm')

    for mode in ('fallthrough', 'shared'):
        changed, _ = single_edge_program(**{mode: True})
        require_refusal(changed, wave_refusal)
        print(f'{mode} edge: branch after the join remains refused')

    changed, changed_labels = single_edge_program(per_lane=True)
    lane_refusal = 's_add_u32 reads lane mask s37 as numeric data'
    # With forward-edge lane-mask marks (psnext-maskload) the sibling arm's mask write no longer reaches the
    # branch-only block, so s37 is the loaded word there and the program translates; the refined pass must
    # still not treat the per-lane entering branch as wave-uniform.
    try:
        lane_pruned = 's37 = r0.Load(' in pipeline(changed).hlsl_ps()
        assert lane_pruned, 'per-lane edge translated without the loaded s37'
    except G.Unsupported as error:
        assert str(error) == lane_refusal, str(error)
        lane_pruned = False
    # Exercise the guard inside the retry itself. With no skipped scalar-data write the VCC branch is per-lane.
    notes = []
    probe = G.Stage(pipeline(None), 'ps', changed, refusals=notes, decoded=G.decode(changed), refine_joins=True)
    assert not probe.wave_exec and changed_labels['enter'] not in probe.uniform_branches
    assert notes == ([] if lane_pruned else [(lane_refusal, 'ps')]), ('per-lane edge', notes)
    print('per-lane VCC edge: scalar mask refusal retained in the refined pass')

    predicate = 'scc = s11 < 0x0000000fu;'
    return warp_sources(hlsl, 1, predicate, predicate.replace(' < ', ' >= '))


def controls():
    code, labels = program()
    accepted = pipeline(code)
    assert accepted.ps.refine_joins and not accepted.ps.refine_edges
    assert labels['bit_branch'] in accepted.ps.uniform_branches
    hlsl = accepted.hlsl_ps()
    notes = []
    assert pipeline(code, notes).hlsl_ps() == hlsl and not notes, notes
    print('dominating rewrite: bit-test branch is wave-uniform')

    reason = ('wave-wide read in a program with a branch (s_cbranch_scc0) that is '
              'neither wave-uniform nor an EXEC / VCC vote')
    for arm in (1, 0):
        changed, changed_labels = program(arm)
        joined, forward = joins(G.decode(changed))
        target = changed_labels[f'join{arm}']
        assert 36 in joined[target] and 36 in forward[target], (arm, joined, forward)
        require_refusal(changed, reason)
        print(f'diamond {arm + 1}: both reaching definitions remain joined and the branch is refused')

    ins = G.decode(code)
    joined, forward = joins(ins)
    assert {36, 37} <= joined[labels['rewrite']] and {36, 37} <= forward[labels['rewrite']]
    assert {36, 37} <= joins(ins, refine=False)[0][labels['join1']]
    for at in ('else0', 'join0', 'else1', 'join1'):
        assert not ({36, 37} & (joined[labels[at]] | forward[labels[at]])), at

    # A textual decoy, a conditional writer, and a back edge must not supply a single reaching definition.
    for label, asm, target in (
            ('skipped textual writer', ('s_mov_b32 s36, 1', 's_branch label_0xc',
                                        's_mov_b32 s36, 2', 's_nop 0'), 0xc),
            ('conditional writer', ('s_cbranch_scc0 label_0x8', 's_mov_b32 s36, 1', 's_cmov_b32 s36, 2',
                                    's_branch label_0x10', 's_nop 0'), 0x10)):
        joined, _ = joins([(i * 4, 4, line.split()[0], line) for i, line in enumerate(asm)])
        assert 36 in joined[target], label
    # A dead predecessor's write is in the former forward split but cannot reach the join from entry.
    asm = ('s_branch label_0xc', 's_mov_b32 s36, 1', 's_branch label_0x10', 's_mov_b32 s36, 2', 's_nop 0')
    dead_pred = [(i * 4, 4, line.split()[0], line) for i, line in enumerate(asm)]
    assert all(36 in sets[0x10] for sets in joins(dead_pred, refine=False))
    assert all(36 not in sets[0x10] for sets in joins(dead_pred))
    # A wave64 compare can overwrite the adjacent high word even when the decoder prints only s36.
    asm = ('s_cbranch_scc0 label_0x8', 's_mov_b32 s37, 1', 's_mov_b32 s37, 2', 's_cbranch_scc0 label_0x14',
           'v_cmp_lt_u32_e64 s36, 0, v0', 's_nop 0')
    assert 37 in joins([(i * 4, 4, line.split()[0], line) for i, line in enumerate(asm)])[0][0x14]
    loop = ins[:-1] + [(ins[-1][0], 4, 's_branch', 's_branch label_0x0')]
    assert joins(loop) == joins(loop, refine=False)

    # This variant already passes through the legacy first-lane retry and keeps that lowering.
    firstlane_code, _ = program(firstlane=True)
    legacy = pipeline(firstlane_code)
    assert legacy.retry_firstlane and not legacy.ps.refine_joins and 's20 = v6;' in legacy.hlsl_ps()
    print('entry, skipped and conditional writers, mask high word, back edge and legacy first-lane controls passed')
    bit_test = 'scc = ((s36 >> (0x00000010u & 31)) & 1u) == 1u;'
    return warp_sources(hlsl, 6, bit_test, bit_test.replace('== 1u', '== 0u')), single_edge_controls(reason)


def reference(constants):
    word, = struct.unpack_from('<I', constants, 0x10)
    diamond0, = struct.unpack_from('<Q', constants, 0x20)
    diamond1, = struct.unpack_from('<Q', constants, 0x30)
    return tuple(1.0 if value else 0.5 for value in (word & (1 << 16), diamond0, diamond1)) + (1.0,)


def range_reference(constants):
    word, = struct.unpack_from('<I', constants, 0x10)
    if 1 <= word <= 15:
        return (0.5, 0.5, 0.0, 1.0)
    return (float(16 <= word <= 30), 1.0, 0.0, 1.0)


def warp_sources(hlsl, loads, predicate, inverted):
    # Bind the translated scalar loads to the pixel AOT renderer's fixture CBV at b1.
    resource = 'ByteAddressBuffer r0 : register(t0);'
    assert hlsl.count(resource) == 1 and hlsl.count('r0.Load(') == loads
    hlsl = hlsl.replace(resource, 'cbuffer FixtureConstants : register(b1) { uint4 words[16]; };\n'
                        'uint load_word(uint offset) { return words[offset >> 4][(offset >> 2) & 3u]; }')
    hlsl = hlsl.replace('r0.Load(', 'load_word(')
    assert hlsl.count(predicate) == 1
    return hlsl, hlsl.replace(predicate, inverted)


def draw(sources, dxc, renderer):
    from pixel_aot_check import compile_shader, run_pixel_batch, write_case

    (hlsl, mutant), (edge_hlsl, edge_mutant) = sources
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        cases = []
        for bit in (0, 1):
            constants = bytearray(256)
            struct.pack_into('<II', constants, 0x10, bit << 16, (1 - bit) << 16)
            struct.pack_into('<Q', constants, 0x20, bit << 32)
            struct.pack_into('<Q', constants, 0x30, 1 - bit)
            cases.append(write_case(directory / f'bit-{bit}.case', bytes(256), reference(constants),
                                    extra_constants=bytes(constants) + bytes(256)))
        ranges = []
        for value in (15, 16, 31):
            constants = bytearray(256)
            struct.pack_into('<I', constants, 0x10, value)
            ranges.append(write_case(directory / f'range-{value}.case', bytes(256), range_reference(constants),
                                      extra_constants=bytes(constants) + bytes(256)))
        for name, source, batch in (
                ('reaching-defs', hlsl, [(case, 'pass') for case in cases]),
                ('bit-polarity', mutant, [(case, 'mismatch') for case in cases]),
                ('single-edge', edge_hlsl, [(case, 'pass') for case in ranges]),
                ('range-polarity', edge_mutant, list(zip(ranges, ('pass', 'mismatch', 'mismatch'))))):
            path = directory / f'{name}.hlsl'
            shader = directory / f'{name}.cso'
            path.write_text(source, encoding='ascii')
            compile_shader(dxc, path, shader)
            result = run_pixel_batch(renderer, 'warp', shader, batch)
            print(f'{name} WARP readback\n{result.stdout}', end='')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 4 and sys.argv[1] == 'draw':
        try:
            draw(controls(), *sys.argv[2:])
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                sys.exit(77)
            raise
    else:
        sys.exit(f'usage: {sys.argv[0]} [draw <dxc> <offscreen_check>]')
