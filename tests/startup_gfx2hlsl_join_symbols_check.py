"""Scalar symbols at branch targets in tools/startup/gfx2hlsl.py.

  (no arguments)                          translation controls, pure Python
  binary <llvm-mc>                        reassemble the fixture sources and compare them with the words below
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                          run the translated pixel bodies in a compute wrapper and compare the words
  control <dxc> <offscreen_check> <warp|hardware>
                                          the same with the old folded statement, which must give the folded value

Follow-up to #2279, which drops scalar constants at a join that another path writes, over the textual range between
a branch and its target. This check covers what that leaves: symbols that are not constants (descriptor words,
pointers, pairs, unplaced entry registers), and a path that leaves the loop's textual range and comes back.

At a branch target, a register whose program-order symbol may not hold on every path (worked out over the basic blocks
and their dominators) keeps a symbol only when every path into the target brings an equal one that names one value.
Otherwise it is read at run time: every constant is assigned when set, and an entry root constant is assigned at
entry. A register that holds, on some path into the target, a symbol with no run-time value (a descriptor word, a
pointer, an unplaced compute user SGPR) refuses any read before its next write, and so does one that such a symbol
reaches while another edge never defined it (DESC_ONE_PATH).

The fixtures are source-built gfx1013 pixel stages behind a fixed vertex stage:
- PAIR_*, DESC_*: a 64-bit copy and descriptor loads around a join (pure Python); OUTSIDE: a loop whose path leaves the
  loop's textual range and comes back; CS_UNPLACED: a compute program writing an unplaced user SGPR on one path.
- DESC_ONE_PATH, PTR_OFFSETS: refusal controls: the branch skips the only load of the descriptor, and two paths add
  different offsets to one pointer.
- JOIN_SAME, DESC_SAME: the same constant (read at run time since #2279), or the same descriptor, on both paths of a forward join (kept).
- CS_RELOAD: a compute loop that stores through its descriptor at the header and reloads it from the same table slot
  before the back edge (kept, and dispatched); CS_RELOAD_OTHER reloads another slot (refused).
- CALC: s23 = 1, a proven s_setpc_b64 skips s23 = 2 and lands on the read (a calculated jump is a branch; 1).
- LOOP_RFL, JOIN_RFL: v_readfirstlane_b32 of a loop counter, and of a register written on one path of a join (both
  refused as unproven, never recomputed with a folded constant).
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_flat_param_check import G, VS_WORDS, assemble, root_blob

EXPORT = 'exp mrt0 v0, v0, v0, v0 done vm\ns_endpgm\n'
EXPORT_WORDS = '0f1800f800000000' '000081bf'
SOURCES = {
    # A proven s_setpc_b64 jumps over s23 = 2 to the read: the target is a branch target, and the decoy never runs.
    'CALC': ('s_mov_b32 s23, 1\ns_getpc_b64 s[20:21]\ns_add_u32 s20, s20, 16\ns_addc_u32 s21, s21, 0\n'
             's_setpc_b64 s[20:21]\ns_mov_b32 s23, 2\nv_mov_b32 v0, s23\n' + EXPORT,
             '810397be' '001f94be' '14901480' '15801582' '142080be' '820397be' '1702007e' + EXPORT_WORDS),
    # The same through a calculated jump: the decoy's s16 = 3 sits between the s_setpc_b64 and its target, so the
    # loaded descriptor word is joined there and refused, not read as the decoy's constant.
    'DESC_CALC': ('s_load_dwordx8 s[16:23], s[8:9], 0x0\ns_getpc_b64 s[12:13]\ns_add_u32 s12, s12, 16\n'
                  's_addc_u32 s13, s13, 0\ns_setpc_b64 s[12:13]\ns_mov_b32 s16, 3\ns_waitcnt lgkmcnt(0)\n'
                  'v_mov_b32 v0, s16\n' + EXPORT,
                  '04040cf4000000fa' '001f8cbe' '0c900c80' '0d800d82' '0c2080be' '830390be' '7fc08cbf' '1002007e'
                  + EXPORT_WORDS),
    'PAIR_READ': ('s_mov_b32 s30, 0\ns_mov_b32 s9, 5\ns_mov_b64 s[20:21], s[8:9]\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 1\n'
                  's_add_u32 s31, s21, 1\nv_mov_b32 v0, s21\n' + EXPORT,
                  '80039ebe850389be080494be1e8006bf010085bf15811f801502007e0f1800f800000000000081bf'),
    'PAIR_ROOT': ('s_mov_b32 s30, 0\ns_mov_b32 s21, 7\ns_mov_b64 s[20:21], s[8:9]\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 1\n'
                  's_add_u32 s31, s21, 1\nv_mov_b32 v0, s21\n' + EXPORT,
                  '80039ebe870395be080494be1e8006bf010085bf15811f801502007e0f1800f800000000000081bf'),
    'PAIR_WRITE': ('s_mov_b32 s30, 0\ns_mov_b32 s9, 5\ns_mov_b64 s[20:21], s[8:9]\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 1\n'
                   's_mov_b32 s21, 7\nv_mov_b32 v0, s21\n' + EXPORT,
                   '80039ebe850389be080494be1e8006bf010085bf870395be1502007e0f1800f800000000000081bf'),
    'DESC_LOAD': ('s_load_dwordx8 s[16:23], s[8:9], 0x0\ns_mov_b32 s30, 0\nv_mov_b32 v0, 0\nv_mov_b32 v1, 0\n'
                  's_cmp_eq_u32 s30, 0\ns_cbranch_scc1 2\ns_load_dwordx8 s[16:23], s[8:9], 0x20\ns_waitcnt lgkmcnt(0)\n'
                  'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D\ns_waitcnt vmcnt(0)\n'
                  'exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n',
                  '04040cf4000000fa80039ebe8002007e8002027e1e8006bf020085bf04040cf4200000fa7fc08cbf080f00f000040400'
                  '703f8cbf0f1800f804050607000081bf'),
    'DESC_WORD': ('s_load_dwordx8 s[16:23], s[8:9], 0x0\ns_mov_b32 s30, 0\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 1\n'
                  's_mov_b32 s16, 3\ns_waitcnt lgkmcnt(0)\nv_mov_b32 v0, s16\n' + EXPORT,
                  '04040cf4000000fa80039ebe1e8006bf010085bf830390be7fc08cbf1002007e0f1800f800000000000081bf'),
    'OUTSIDE': ('s_mov_b32 s23, 1\ns_mov_b32 s24, 0\ns_mov_b32 s25, 0\ns_add_u32 s25, s25, s23\ns_add_u32 s24, s24, 1\n'
                's_cmp_eq_u32 s24, 1\ns_cbranch_scc1 3\ns_cmp_lt_u32 s24, 3\ns_cbranch_scc1 65530\ns_branch 2\n'
                's_mov_b32 s23, 5\ns_branch 65531\nv_mov_b32 v0, s25\n' + EXPORT,
                '810397be800398be800399be1917198018811880188106bf030085bf18830abffaff85bf020082bf850397befbff82bf'
                '1902007e0f1800f800000000000081bf'),
    'CS_UNPLACED': ('s_mov_b32 s30, 0\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 1\ns_mov_b32 s0, 5\nv_mov_b32 v1, s0\ns_endpgm\n',
                    '80039ebe1e8006bf010085bf850380be0002027e000081bf'),
    'JOIN_SAME': ('s_mov_b32 s23, 1\ns_cmp_eq_u32 s8, 0\ns_cbranch_scc1 1\ns_mov_b32 s23, 1\nv_mov_b32 v0, s23\n' + EXPORT,
                  '810397be' '088006bf' '010085bf' '810397be' '1702007e' + EXPORT_WORDS),
    'DESC_SAME': ('s_load_dwordx8 s[16:23], s[8:9], 0x0\ns_mov_b32 s30, 0\nv_mov_b32 v0, 0\nv_mov_b32 v1, 0\n'
                  's_cmp_eq_u32 s30, 0\ns_cbranch_scc1 2\ns_load_dwordx8 s[16:23], s[8:9], 0x0\ns_waitcnt lgkmcnt(0)\n'
                  'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D\ns_waitcnt vmcnt(0)\n'
                  'exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n',
                  '04040cf4000000fa80039ebe8002007e8002027e1e8006bf020085bf04040cf4000000fa7fc08cbf080f00f000040400'
                  '703f8cbf0f1800f804050607000081bf'),
    # Compute, with the root of startup_gfx2hlsl_compute_check.py: the table descriptor is loaded before the loop, the
    # loop stores through it at its header and reloads it from the same place before the back edge. Each of 8 lanes
    # stores the counter (1, then 2) to two consecutive words.
    'CS_RELOAD': ('s_mov_b32 s10, s6\ns_mov_b32 s11, 0\ns_load_dwordx8 s[12:19], s[10:11], 0x0\ns_mov_b32 s22, 0\n'
                  'v_lshlrev_b32 v4, 3, v0\ns_waitcnt lgkmcnt(0)\ns_add_u32 s22, s22, 1\nv_mov_b32 v5, s22\n'
                  'buffer_store_dword v5, v4, s[16:19], 0 offen\nv_add_nc_u32 v4, 4, v4\n'
                  's_load_dwordx8 s[12:19], s[10:11], 0x0\ns_cmp_lt_u32 s22, 2\ns_cbranch_scc1 65526\ns_endpgm\n',
                  '06038abe80038bbe05030cf4000000fa800396be83000834' '7fc08cbf1681168016020a7e001070e004050480'
                  '8408084a05030cf4000000fa16820abff6ff85bf000081bf'),
    # The same loop reloading from byte 32 of the table: the back edge brings another descriptor.
    'CS_RELOAD_OTHER': ('s_mov_b32 s10, s6\ns_mov_b32 s11, 0\ns_load_dwordx8 s[12:19], s[10:11], 0x0\ns_mov_b32 s22, 0\n'
                        'v_lshlrev_b32 v4, 3, v0\ns_waitcnt lgkmcnt(0)\ns_add_u32 s22, s22, 1\nv_mov_b32 v5, s22\n'
                        'buffer_store_dword v5, v4, s[16:19], 0 offen\nv_add_nc_u32 v4, 4, v4\n'
                        's_load_dwordx8 s[12:19], s[10:11], 0x20\ns_cmp_lt_u32 s22, 2\ns_cbranch_scc1 65526\ns_endpgm\n',
                        '06038abe80038bbe05030cf4000000fa800396be83000834' '7fc08cbf1681168016020a7e001070e004050480'
                        '8408084a05030cf4200000fa16820abff6ff85bf000081bf'),
    # v_readfirstlane_b32 of a VGPR copied from a loop counter (LOOP_RFL) or from a register written on one path of a
    # forward join (JOIN_RFL): the recomputed uniform expression must not carry a constant folded across the join.
    'LOOP_RFL': ('s_mov_b32 s23, 0\ns_add_u32 s23, s23, 1\nv_mov_b32 v7, s23\nv_readfirstlane_b32 s24, v7\n'
                 'v_mov_b32 v0, s24\ns_cmp_lt_u32 s23, 2\ns_cbranch_scc1 65530\n' + EXPORT,
                 '800397be' '17811780' '17020e7e' '0705307e' '1802007e' '17820abf' 'faff85bf' + EXPORT_WORDS),
    'JOIN_RFL': ('s_mov_b32 s23, 1\ns_cmp_eq_u32 s8, 0\ns_cbranch_scc1 1\ns_mov_b32 s23, 2\nv_mov_b32 v7, s23\n'
                 'v_readfirstlane_b32 s24, v7\nv_mov_b32 v0, s24\n' + EXPORT,
                 '810397be' '088006bf' '010085bf' '820397be' '17020e7e' '0705307e' '1802007e' + EXPORT_WORDS),
    # A descriptor loaded on only one path has no definition on the other edge: unproven at the join, refused when read
    # (DESC_ONE_PATH: the branch skips the only load). A graphics user SGPR (s0-s15) counts as written at entry, so a
    # constant written on one path into it is read at run time, not folded (USER_ONE_PATH).
    'DESC_ONE_PATH': ('s_mov_b32 s30, 0\nv_mov_b32 v0, 0\nv_mov_b32 v1, 0\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 2\n'
                      's_load_dwordx8 s[16:23], s[8:9], 0x0\ns_waitcnt lgkmcnt(0)\n'
                      'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D\ns_waitcnt vmcnt(0)\n'
                      'exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n',
                      '80039ebe8002007e8002027e1e8006bf020085bf04040cf4000000fa7fc08cbf080f00f000040400703f8cbf'
                      '0f1800f804050607000081bf'),
    # Two paths add different offsets to one pointer: the symbol keeps ('ptr', source) plus an 'added' mark, so the
    # equal tuples must not be kept as one pointer (PTR_OFFSETS).
    'PTR_OFFSETS': ('s_mov_b32 s30, 0\nv_mov_b32 v0, 0\nv_mov_b32 v1, 0\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 2\n'
                    's_add_u32 s8, s8, 4\ns_branch 1\ns_add_u32 s8, s8, 8\n'
                    's_load_dwordx8 s[16:23], s[8:9], 0x0\ns_waitcnt lgkmcnt(0)\n'
                    'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D\ns_waitcnt vmcnt(0)\n'
                    'exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n',
                    '80039ebe8002007e8002027e1e8006bf020085bf08840880010082bf08880880'
                    '04040cf4000000fa7fc08cbf080f00f000040400703f8cbf0f1800f804050607000081bf'),
    'USER_ONE_PATH': ('s_mov_b32 s30, 0\ns_cmp_eq_u32 s30, 0\ns_cbranch_scc1 1\ns_mov_b32 s12, 2\nv_mov_b32 v0, s12\n'
                      + EXPORT, '80039ebe1e8006bf010085bf82038cbe0c02007e' + EXPORT_WORDS),
}
CS_RSRC2, CS_GROUP = 7 << 1 | 0 << 11, (8, 1, 1)  # s0-s3, two root constants, one table; v0 = tid.x
# The results by instruction semantics (COUNT with a root constant of 3; OUTSIDE adds 1, then 5 twice).
EXPECTED = {'OUTSIDE': 11}
# The folded statements main emitted, and what they give: the folded loops run to the guard and export nothing.
OLD = {'OUTSIDE': ('        s25 = s25 + s23;\n', '        s25 = s25 + 0x00000001u;\n', 3)}


def root(*params):
    """RTS0 with the given parameters: 't' a descriptor table, an int n that many root constants (b0)."""
    head = 24 + 12 * len(params)
    blob, data = b'', b''
    for p in params:
        if p == 't':
            blob += struct.pack('<3I', 0, 0, head + len(data))
            data += struct.pack('<2I', 0, head + len(data) + 8)
        else:
            blob += struct.pack('<3I', 1, 0, head + len(data))
            data += struct.pack('<3I', 0, 0, p)
    rts0 = struct.pack('<6I', 2, len(params), 24, 0, head + len(data), 0) + blob + data
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part


ROOTS = {'DESC_CALC': ('t', 1), 'PAIR_READ': ('t', 1), 'PAIR_ROOT': ('t', 1), 'PAIR_WRITE': ('t', 1), 'DESC_LOAD': ('t', 1),
         'DESC_WORD': ('t', 1), 'DESC_SAME': ('t', 1), 'DESC_ONE_PATH': ('t', 1),
         'PTR_OFFSETS': ('t', 1)}


def translate(name):
    return G.Pipeline(VS_WORDS, bytes.fromhex(SOURCES[name][1]), root(*ROOTS[name]) if name in ROOTS else root_blob())


def translate_cs(name):
    import startup_gfx2hlsl_compute_check as C
    return G.ComputePipeline(bytes.fromhex(SOURCES[name][1]), C.root_blob(), CS_RSRC2, CS_GROUP)


def refused(name, fragment):
    try:
        translate(name).hlsl_ps()
    except G.Unsupported as reason:
        assert fragment in str(reason), (name, str(reason))
        return
    raise AssertionError('%s accepted, expected a refusal naming %r' % (name, fragment))


def lines(name):
    return translate(name).hlsl_ps().splitlines()


def check_controls():
    # A calculated jump is a branch: the decoy's s23 = 2 sits between the jump and its target, so s23 is joined there.
    assert '        v0 = s23;' in lines('CALC') and not any('v0 = 0x00000002u' in l for l in lines('CALC'))
    # The same for a descriptor word: refused, not read as the decoy's constant.
    refused('DESC_CALC', 'numeric use of s16 holding joined at a branch target')
    # Straight-line code keeps folding: a constant read with no join between keeps its literal.
    straight = G.Pipeline(VS_WORDS, bytes.fromhex('810397be' '1702007e' + EXPORT_WORDS), root_blob()).hlsl_ps()
    assert '    v0 = 0x00000001u;' in straight.splitlines(), straight
    # A register only read on the skipped path is not joined: the copied constant still folds (main did too).
    assert '        v0 = 0x00000005u;' in lines('PAIR_READ')
    # The copied high word of a root constant still reads the root constant, not an earlier run-time value.
    pair = translate('PAIR_ROOT')
    assert '        v0 = ud%d;' % pair.uds.index(('rootconst', 1, 0)) in pair.hlsl_ps().splitlines(), pair.hlsl_ps()
    # Written on the skipped path: read at run time, and the 64-bit copy assigns the high word at run time too.
    write = lines('PAIR_WRITE')
    assert '        s21 = 0x00000005u;' in write and '        v0 = s21;' in write, write
    # A symbol with no run-time value on one path refuses the read: a descriptor reloaded on one path, and a
    # descriptor word overwritten by a constant on one path.
    refused('DESC_LOAD', "holds ('joined at a branch target',)")
    refused('DESC_WORD', 'numeric use of s16 holding joined at a branch target')
    # A path that leaves the loop's textual range and comes back (T -> X -> Y -> T) writes s23 = 5: s23 is joined at
    # the loop header T (+0xc), worked out over the blocks rather than the textual range, and read at run time there.
    stage = G.Stage.__new__(G.Stage)
    stage.ins = G.decode(bytes.fromhex(SOURCES['OUTSIDE'][1]))
    assert 23 in stage.joined_registers()[0xc], stage.joined_registers()
    assert '        s25 = s25 + s23;' in lines('OUTSIDE')
    # Compute: an unplaced user SGPR (refused in straight-line code) written on one path only is refused at the join.
    import startup_gfx2hlsl_compute_check as C
    try:
        G.ComputePipeline(bytes.fromhex(SOURCES['CS_UNPLACED'][1]), C.root_blob(), C.CASES['A'][2], (1, 1, 1)).hlsl_cs()
    except G.Unsupported as reason:
        assert 'numeric use of s0 holding joined at a branch target' in str(reason), str(reason)
    else:
        raise AssertionError('CS_UNPLACED accepted')
    # Every path brings an equal symbol: kept. A constant equal on both paths still folds, and a descriptor reloaded
    # from the same table slot on one path is still bound, in a forward join and across a loop's back edge.
    assert '        v0 = s23;' in lines('JOIN_SAME')  # #2279 drops the constant at any join another path writes
    same = translate('DESC_SAME')
    assert '        t = r0.Load(int3(v0, v1, 0));' in same.hlsl_ps().splitlines(), same.hlsl_ps()
    reload = translate_cs('CS_RELOAD')
    cs = reload.hlsl_cs().splitlines()
    assert '        u0.Store(a, v5);' in cs and '        s22 = s22 + 0x00000001u;' in cs, cs
    assert 'uav 0 table 1 16' in reload.manifest().splitlines(), reload.manifest()
    # Another descriptor on the back edge: refused where the loop header reads it.
    try:
        translate_cs('CS_RELOAD_OTHER').hlsl_cs()
    except G.Unsupported as reason:
        assert "buffer store through s[16:19] holding ('joined at a branch target',)" in str(reason), str(reason)
    else:
        raise AssertionError('CS_RELOAD_OTHER accepted')
    # Refusal controls: no definition on the skipping edge, and different offsets added to one pointer.
    refused('DESC_ONE_PATH', "descriptor operand s[16:23] holds ('joined at a branch target',)")
    refused('PTR_OFFSETS', "s_load through s8 holding ('joined at a branch target',)")
    user = lines('USER_ONE_PATH')
    assert '        v0 = s12;' in user and not any('v0 = 0x00000002u' in l for l in user), user
    # LOOP_RFL reads a uniform loop counter via v_readfirstlane_b32 dynamically (not constant-folded);
    # JOIN_RFL writes both paths before the forward join and stays refused.
    loop_rfl = lines('LOOP_RFL')
    assert '        s23 = s23 + 0x00000001u;' in loop_rfl and '        s24 = s23;' in loop_rfl, loop_rfl
    refused('JOIN_RFL', 'v_readfirstlane_b32 requires a proven uniform source')
    # Equal tuples that do not name one value are not kept: a word modified on both paths (the written value is not
    # in the symbol) and descnum.
    assert not G.names_one_value(('desc', ('table', 0), 0, 0, 'modified')) and not G.names_one_value(('descnum',))
    assert G.names_one_value(('desc', ('table', 0), 0, 0)) and G.names_one_value(('const', 2))
    assert G.names_one_value(('ptr', ('table', 0))) and not G.names_one_value(('ptr', ('table', 0), 'added'))
    print('loop counters, forward joins, a root-constant loop, 64-bit copies and descriptor joins pass; equal symbols '
          'on every path are kept; straight-line code still folds')


def check_binary(llvm_mc):
    for name, (source, words) in SOURCES.items():
        built = assemble(llvm_mc, source).hex()
        assert built == words, '%s: assembled %s, recorded %s' % (name, built, words)
    print('fixture words match llvm-mc')


def run(dxc, runner, adapter, hlsl, directory, name):
    source = hlsl.replace('PSOut main(PSIn i)', 'PSOut translated(PSIn i)') + '''
RWByteAddressBuffer data : register(u0);
[numthreads(1, 1, 1)]
void main() {
    PSIn i = (PSIn)0;
    data.Store4(0, asuint(translated(i).mrt0));
}
'''
    src, cso, output = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'bin'))
    src.write_text(source, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    result = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', '64', str(output)],
                            capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode == 77:
        sys.exit(77)
    if result.returncode or result.stderr:
        raise AssertionError(result.stderr or 'raw UAV dump exited with %d' % result.returncode)
    return struct.unpack('<4I', output.read_bytes()[:16])


def check_dispatch(dxc, runner, adapter, control=False):
    runner = str(Path(runner).resolve())
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name, want in EXPECTED.items():
            hlsl = translate(name).hlsl_ps()
            if control:
                old, new, folded = OLD[name]
                assert hlsl.count(old) == 1, (name, hlsl)
                hlsl = hlsl.replace(old, new)
                want = folded
            got = run(dxc, runner, adapter, hlsl, Path(tmp), name)
            assert got == (want,) * 4, (name, got, want)
            print('%s%s: %d in all four channels' % ('control ' if control else '', name, want))
        if not control:
            run_reload(dxc, runner, adapter, Path(tmp))


def run_reload(dxc, runner, adapter, directory):
    """CS_RELOAD through the compute-store runner of startup_gfx2hlsl_compute_check.py: root UAV u0 is the stored
    buffer. Lane t writes 1 and 2 to words 2t and 2t + 1; every other word keeps the sentinel."""
    import startup_gfx2hlsl_compute_check as C
    want = [C.SENTINEL] * C.WORDS
    for t in range(CS_GROUP[0]):
        want[2 * t:2 * t + 2] = [1, 2]
    src, cso, case = (directory / ('CS_RELOAD.%s' % ext) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(translate_cs('CS_RELOAD').hlsl_cs(), encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    case.write_bytes(struct.pack('<I', 1 | 1 << 8 | 1 << 16) + struct.pack('<%dI' % C.WORDS, *want)
                     + struct.pack('<16I', *[0] * 16))
    result = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode == 77:
        sys.exit(77)
    if result.returncode or result.stderr:
        raise AssertionError('CS_RELOAD: %s' % (result.stderr or 'exit %d' % result.returncode))
    print('CS_RELOAD: the descriptor kept across the back edge stores 1 and 2 for each of %d lanes' % CS_GROUP[0])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] in ('dispatch', 'control'):
        check_dispatch(*sys.argv[2:], control=sys.argv[1] == 'control')
    else:
        sys.exit(__doc__)
