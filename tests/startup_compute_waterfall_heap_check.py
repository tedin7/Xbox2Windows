"""Per-lane resource-heap record index inside a compute v_readfirstlane_b32 waterfall loop (#2644 T3).

usage:
  startup_compute_waterfall_heap_check.py [--llvm-mc <llvm-mc>]
  startup_compute_waterfall_heap_check.py --dxil <dxc>
  startup_compute_waterfall_heap_check.py --oracle <dxc> <offscreen_check> <warp|hardware>
  startup_compute_waterfall_heap_check.py --dispatch <startup_stream_consumer> <dxc> <warp|hardware>

Covers:
1. gfx2hlsl translation and manifest generation for bounded per-lane resource-heap indices in compute
   waterfall loops, including both direct heap offsets (s_lshl_b32 + s_load_dwordx4 s[...], s[8:9], sI)
   and the adjacent 64-bit pointer-add carry form (s_add_u32 + s_add_u32 0, src_scc / s_addc_u32, with
   either in-body s_lshl_b32 or pre-loop v_lshlrev_b32 ..., 5), plus v_min_u32 and v_bfe_u32 bounds.
2. Soundness negative controls: unbounded keys, oversized masks, one-arm diamond bounds, bounded writes under
   narrowed EXEC, missing/wrong shift, missing resheap root flag, non-zero high word, broken carry, descriptor
   mutation, exit-live scalars, and forbidden store/formatted/idxen uses.
3. Independent DXIL handle metadata assertions via DXC cs_6_0: confirms the resource handle selects SRV space 2
   [0..3] with nonUniformIndex=true, and that stripping NonUniformResourceIndex changes it to false.
4. Four-record GPU oracle (--oracle): reference, static switch, no-marker, and first-active (asserting the
   exact 12/25 first-active mismatches that distinguish lane-local selection from a wave-uniform shortcut).
5. End-to-end stand-in Dispatch fixture (--dispatch): translates both the direct-offset and 64-bit carry
   waterfall programs via gfx2hlsl.py --pending, dispatches all five oracle key patterns plus active-lane
   subsets through d3d12_xs.dll with the debug layer enabled, and verifies a short-range view heap that omits
   the highest record (3) refuses cleanly before Dispatch.
"""
import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import array_atomics_dxil_check as dxil  # noqa: E402
from startup_compute_dispatch_check import dispatch, lines  # noqa: E402
from startup_compute_request_check import DATA, prepare  # noqa: E402
from startup_compute_translate_check import state_a  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402
import startup_gfx2hlsl_heap_record_check as H  # noqa: E402
from startup_gfx2hlsl_compute_check import G, assemble, refused  # noqa: E402

GROUP = (4, 1, 1)
NO_DATA = bytes(C.WORDS * 4)

# Bounded four-record waterfall program matching the reference fixture (`bounded-waterfall.s`).
BOUNDED_PREFIX = (
    ('s_mov_b32 s12, s7', '07038cbe'),
    ('s_mov_b32 s13, 0', '80038dbe'),
    ('s_load_dwordx8 s[40:47], s[12:13], 0x0', '060a0cf4000000fa'),
    ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
    ('s_mov_b32 s9, 0', '800389be'),
    ('v_lshlrev_b32_e32 v4, 2, v0', '82000834'),
    ('buffer_load_dword v5, v4, s[40:43], 0 offen', '001030e004050a80'),
    ('s_waitcnt vmcnt(0)', '703f8cbf'),
)
MASK_AND3 = ('v_and_b32_e32 v5, 3, v5', '830a0a36')
INIT_AND_GUARD = (
    ('v_mov_b32_e32 v7, 13', '8d020e7e'),
    ('s_mov_b32 s26, exec_lo', '7e039abe'),
    ('v_cmpx_gt_u32_e32 s4, v0', '0400a87d'),
    ('s_mov_b32 s21, exec_lo', '7e0395be'),
)
HEAD = ('v_readfirstlane_b32 s22, v5', '05052c7e')
COMPARE = ('v_cmpx_eq_i32_e32 s22, v5', '160a247d')
SHIFT5 = ('s_lshl_b32 s23, s22, 5', '1685178f')
HEAP_LOAD = ('s_load_dwordx4 s[28:31], s[8:9], s23', '040708f40000002e')
CARRY_ADD_S23 = (
    ('s_add_u32 s24, s8, s23', '08171880'),
    ('s_add_u32 s25, 0, src_scc', '80fd1980'),
    ('s_load_dwordx4 s[28:31], s[24:25], 0x0', '0c0708f4000000fa'),
)
CARRY_ADD_S22 = (
    ('s_add_u32 s24, s8, s22', '08161880'),
    ('s_add_u32 s25, 0, src_scc', '80fd1980'),
    ('s_load_dwordx4 s[28:31], s[24:25], 0x0', '0c0708f4000000fa'),
)
WAIT_LGKM = ('s_waitcnt lgkmcnt(0)', '7fc08cbf')
BODY_LOAD = ('buffer_load_dword v7, v4, s[28:31], 0 offen', '001030e004070780')
WAIT_VM = ('s_waitcnt vmcnt(0)', '703f8cbf')
CLOSE = ('s_andn2_wrexec_b32 s21, s21', '154795be')
RESTORE = ('s_mov_b32 exec_lo, s26', '1a03febe')
FINAL_STORE = ('buffer_store_dword v7, v4, s[44:47], 0 offen', '001070e004070b80')
END = ('s_endpgm', '000081bf')

ENCODINGS = {
    'v_min_u32_e32 v5, 3, v5': '830a0a26',
    'v_bfe_u32 v5, v5, 0, 2': '050048d505010a02',
    'v_lshlrev_b32_e32 v5, 5, v5': '850a0a34',
    'v_lshlrev_b32_e32 v5, 4, v5': '840a0a34',
    'v_and_b32_e32 v5, 0x7ff, v5': 'ff0a0a36ff070000',
    's_lshl_b32 s22, s22, 5': '1685168f',
    's_lshl_b32 s23, s22, 4': '1684178f',
    's_mov_b32 s24, s23': '170398be',
    's_mov_b32 s24, s22': '160398be',
    's_load_dwordx4 s[28:31], s[8:9], s24': '040708f400000030',
    's_load_dwordx4 s[28:31], s[8:9], s22': '040708f40000002c',
    's_load_dwordx4 s[28:31], s[22:23], 0x0': '0b0708f4000000fa',
    's_addc_u32 s25, s9, 0': '09801982',
    's_mov_b32 s9, 1': '810389be',
    's_cmp_eq_u32 s4, 0': '048006bf',
    's_cbranch_scc0 1': '010084bf',
    's_cbranch_scc0 2': '020084bf',
    's_branch 1': '010082bf',
    's_nop 0': '000080bf',
    's_or_b32 s31, s31, 0': '1f801f88',
    's_mov_b32 s22, 0': '800396be',
    'v_mov_b32_e32 v7, s22': '16020e7e',
    'v_mov_b32_e32 v7, s28': '1c020e7e',
    'buffer_load_dword v7, v4, s[28:31], 0 idxen': '002030e004070780',
    'buffer_load_format_x v7, v4, s[28:31], 0 offen': '001000e004070780',
    'buffer_store_dword v7, v4, s[28:31], 0 offen': '001070e004070780',
}

BOUNDED_ORACLE_HLSL = """\
ByteAddressBuffer keys : register(t0);
#if defined(STATIC_SWITCH)
ByteAddressBuffer r0 : register(t1), r1 : register(t2), r2 : register(t3), r3 : register(t4);
#else
ByteAddressBuffer records[4] : register(t1);
#endif
RWByteAddressBuffer result : register(u0);
cbuffer config : register(b0) { uint active; };

[RootSignature("SRV(t0), DescriptorTable(SRV(t1, numDescriptors=4)), UAV(u0), RootConstants(num32BitConstants=1,b0)")]
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    uint lane = tid.x;
    uint value = 13;
    if (lane < active) {
#if defined(FIRST_ACTIVE)
        uint index = keys.Load(0) & 3;
#else
        uint index = keys.Load(4 * lane) & 3;
#endif
#if defined(STATIC_SWITCH)
        switch (index) {
        case 0: value = r0.Load(4 * lane); break;
        case 1: value = r1.Load(4 * lane); break;
        case 2: value = r2.Load(4 * lane); break;
        default: value = r3.Load(4 * lane); break;
        }
#elif defined(NO_MARKER)
        value = records[index].Load(4 * lane);
#else
        value = records[NonUniformResourceIndex(index)].Load(4 * lane);
#endif
    }
    result.Store(4 * lane, value);
}
"""

ORACLE_ROWS = ((0, 1, 2, 3), (3, 2, 1, 0), (2, 1, 2, 3), (4, 9, 14, 19), (7, 7, 7, 7))


def row_pair(asm):
    return (asm, ENCODINGS[asm])


def build_program(*, bound=(MASK_AND3,), init=INIT_AND_GUARD, body=(SHIFT5, HEAP_LOAD, WAIT_LGKM, BODY_LOAD, WAIT_VM),
                  after_restore=()):
    loop = (HEAD, COMPARE, *body, CLOSE)
    offset = -(sum(len(bytes.fromhex(code)) for _, code in loop) // 4 + 1)
    branch = ('s_cbranch_scc1 %d' % offset, struct.pack('<I', 0xbf850000 | (offset & 0xffff)).hex())
    return (*BOUNDED_PREFIX, *bound, *init, *loop, branch, RESTORE, *after_restore, FINAL_STORE, END)


def assemble_rows(rows):
    return b''.join(bytes.fromhex(code) for _, code in rows)


def translate(rows, *, root=None):
    pipe = G.ComputePipeline(assemble_rows(rows), root or H.root_blob(), H.RSRC2, GROUP)
    return pipe.hlsl_cs(), pipe.manifest()


def trace_srv_handle(ir_text, *, expected_binding):
    """Follow buffer RawBufferLoad handle back to createHandle / createHandleFromBinding in DXIL IR."""
    definitions, metadata, named = dxil.parse_module(ir_text)
    roots = named.get('!dx.resources', [])
    assert len(roots) == 1, 'module missing !dx.resources'
    srvs = metadata[roots[0]][0]
    srv_table = {}
    for entry in ([] if srvs == 'null' else metadata[srvs]):
        fields = metadata[entry]
        rid = dxil.integers(fields[0])[0]
        srv_table[rid] = {'space': dxil.integers(fields[3])[0], 'lower': dxil.integers(fields[4])[0],
                          'size': dxil.integers(fields[5])[0]}

    loads = [ops for op, ops in definitions.values() if op in ('rawBufferLoad', 'bufferLoad')]
    found = []
    for ops in loads:
        handle = dxil.last_value(ops[1])
        while definitions.get(handle, (None,))[0] == 'annotateHandle':
            handle = dxil.last_value(definitions[handle][1][1])
        op, operands = definitions.get(handle, (None, None))
        if op == 'createHandle':
            rclass = dxil.integers(operands[1])[0]
            binding = srv_table.get(dxil.integers(operands[2])[0]) if rclass == 0 else None
            idx, flag = operands[3], operands[4]
        elif op == 'createHandleFromBinding':
            lower, upper, space, rclass = dxil.integers(operands[1])
            binding = {'space': space, 'lower': lower, 'size': upper - lower + 1} if rclass == 0 else None
            idx, flag = operands[2], operands[3]
        else:
            continue
        if binding == expected_binding:
            found.append({
                'creation': op,
                'binding': binding,
                'index_var': dxil.last_value(idx),
                'non_uniform': dxil.last_value(flag) == 'true',
                'raw': idx,
            })
    assert len(found) == 1, f'expected one load on {expected_binding}, found {found}\n{ir_text}'
    return found[0]


def verify_dxil(dxc, hlsl_source, *, expected_binding, check_stripped=True):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'shader.hlsl'
        cso = Path(tmp) / 'shader.cso'
        src.write_text(hlsl_source, encoding='ascii')
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, src], check=True)
        ir = subprocess.run([dxc, '-dumpbin', cso], check=True, capture_output=True, text=True).stdout
        traced = trace_srv_handle(ir, expected_binding=expected_binding)
        assert traced['non_uniform'], f'DXIL lost nonUniformIndex on {expected_binding}: {traced}'
        if check_stripped:
            assert 'NonUniformResourceIndex(' in hlsl_source
            stripped = Path(tmp) / 'stripped.hlsl'
            cso_no = Path(tmp) / 'stripped.cso'
            stripped.write_text(hlsl_source.replace('NonUniformResourceIndex(', '('), encoding='ascii')
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', cso_no, stripped], check=True)
            ir_no = subprocess.run([dxc, '-dumpbin', cso_no], check=True, capture_output=True, text=True).stdout
            traced_no = trace_srv_handle(ir_no, expected_binding=expected_binding)
            assert not traced_no['non_uniform'], f'failed to detect omitted NonUniformResourceIndex: {traced_no}'


def check_translator(llvm_mc=None):
    # Positive forms.
    hlsl_a, man_a = translate(build_program())
    assert 'ByteAddressBuffer rh[4] : register(t0, space2);' in hlsl_a
    assert 'rh[NonUniformResourceIndex(h0)].Load(a + 0)' in hlsl_a
    assert man_a.endswith('uav 0 table 1 16\nresheap 4 2\n'), man_a

    # 64-bit pointer add + src_scc carry form (post-loop shift and pre-loop v_lshlrev_b32).
    hlsl_b, man_b = translate(build_program(body=(SHIFT5, *CARRY_ADD_S23, WAIT_LGKM, BODY_LOAD, WAIT_VM)))
    assert 'rh[NonUniformResourceIndex(h0)].Load(a + 0)' in hlsl_b and 'resheap 4 2\n' in man_b
    hlsl_pre, man_pre = translate(build_program(bound=(MASK_AND3, row_pair('v_lshlrev_b32_e32 v5, 5, v5')),
                                                body=(*CARRY_ADD_S22, WAIT_LGKM, BODY_LOAD, WAIT_VM)))
    assert 'h0 = (s22 >> 5);' in hlsl_pre and 'rh[NonUniformResourceIndex(h0)].Load(a + 0)' in hlsl_pre
    assert 'resheap 4 2\n' in man_pre

    # s_addc_u32 high-word carry, v_min_u32, v_bfe_u32, in-place s_lshl_b32, and clobber of s22 after load.
    translate(build_program(body=(SHIFT5, ('s_add_u32 s24, s8, s23', '08171880'), row_pair('s_addc_u32 s25, s9, 0'),
                                  ('s_load_dwordx4 s[28:31], s[24:25], 0x0', '0c0708f4000000fa'),
                                  WAIT_LGKM, BODY_LOAD, WAIT_VM)))
    translate(build_program(bound=(row_pair('v_min_u32_e32 v5, 3, v5'),)))
    hlsl_bfe, man_bfe = translate(build_program(bound=(row_pair('v_bfe_u32 v5, v5, 0, 2'),)))
    assert 'resheap 4 2\n' in man_bfe and 'rh[NonUniformResourceIndex(h0)].Load' in hlsl_bfe
    hlsl_inp, _ = translate(build_program(body=(row_pair('s_lshl_b32 s22, s22, 5'),
                                                row_pair('s_load_dwordx4 s[28:31], s[8:9], s22'),
                                                WAIT_LGKM, BODY_LOAD, WAIT_VM)))
    assert 'h0 = (s22 >> 5);' in hlsl_inp
    hlsl_reused, _ = translate(build_program(body=(SHIFT5, HEAP_LOAD, WAIT_LGKM, row_pair('s_mov_b32 s22, 0'),
                                                   BODY_LOAD, WAIT_VM)))
    assert 'h0 = s22;' in hlsl_reused and 'rh[NonUniformResourceIndex(h0)].Load' in hlsl_reused
    hlsl_ocpy, _ = translate(build_program(body=(SHIFT5, row_pair('s_mov_b32 s24, s23'),
                                                 row_pair('s_load_dwordx4 s[28:31], s[8:9], s24'),
                                                 WAIT_LGKM, BODY_LOAD, WAIT_VM)))
    assert 'h0 = s22;' in hlsl_ocpy

    # Negative controls: every missing proof or forbidden extension stays refused.
    controls = (
        ('unbounded key', build_program(bound=()), 'per-lane descriptor selection requires a proven bound heap range'),
        ('oversized mask', build_program(bound=(row_pair('v_and_b32_e32 v5, 0x7ff, v5'),)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('only one diamond arm bounds key',
         build_program(bound=(row_pair('s_cmp_eq_u32 s4, 0'), row_pair('s_cbranch_scc0 2'), MASK_AND3,
                              row_pair('s_branch 1'), row_pair('s_nop 0'))),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('bounded after EXEC narrowed without prior bound',
         build_program(bound=(), init=(('v_mov_b32_e32 v7, 13', '8d020e7e'), ('s_mov_b32 s26, exec_lo', '7e039abe'),
                                       ('v_cmpx_gt_u32_e32 s4, v0', '0400a87d'), MASK_AND3,
                                       ('s_mov_b32 s21, exec_lo', '7e0395be'))),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('missing shift', build_program(body=(row_pair('s_load_dwordx4 s[28:31], s[8:9], s22'),
                                              WAIT_LGKM, BODY_LOAD, WAIT_VM)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('wrong shift', build_program(body=(row_pair('s_lshl_b32 s23, s22, 4'), HEAP_LOAD,
                                            WAIT_LGKM, BODY_LOAD, WAIT_VM)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('wrong pre-loop shift',
         build_program(bound=(MASK_AND3, row_pair('v_lshlrev_b32_e32 v5, 4, v5')),
                       body=(*CARRY_ADD_S22, WAIT_LGKM, BODY_LOAD, WAIT_VM)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('non-resheap base', build_program(body=(row_pair('s_load_dwordx4 s[28:31], s[22:23], 0x0'),
                                                 WAIT_LGKM, BODY_LOAD, WAIT_VM)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('modified descriptor word', build_program(body=(SHIFT5, HEAP_LOAD, WAIT_LGKM,
                                                         row_pair('s_or_b32 s31, s31, 0'), BODY_LOAD, WAIT_VM)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('non-zero resheap high word',
         build_program(bound=(row_pair('s_mov_b32 s9, 1'), MASK_AND3)),
         'whose high word is not a constant zero'),
        ('missing carry on s_add_u32 pointer add',
         build_program(body=(SHIFT5, ('s_add_u32 s24, s8, s23', '08171880'),
                             ('s_load_dwordx4 s[28:31], s[24:25], 0x0', '0c0708f4000000fa'),
                             WAIT_LGKM, BODY_LOAD, WAIT_VM)),
         'per-lane descriptor selection requires a proven bound heap range'),
        ('live key at loop exit',
         build_program(after_restore=(row_pair('v_mov_b32_e32 v7, s22'),)), 'body scalar is live at the loop exit'),
        ('live descriptor word at loop exit',
         build_program(after_restore=(row_pair('v_mov_b32_e32 v7, s28'),)), 'body scalar is live at the loop exit'),
        ('idxen load through per-lane descriptor',
         build_program(body=(SHIFT5, HEAP_LOAD, WAIT_LGKM,
                             row_pair('buffer_load_dword v7, v4, s[28:31], 0 idxen'), WAIT_VM)),
         'has unproved modifiers'),
        ('formatted load through per-lane descriptor',
         build_program(body=(SHIFT5, HEAP_LOAD, WAIT_LGKM,
                             row_pair('buffer_load_format_x v7, v4, s[28:31], 0 offen'), WAIT_VM)),
         'is not proven lane-local'),
        ('store through per-lane descriptor',
         build_program(body=(SHIFT5, HEAP_LOAD, WAIT_LGKM,
                             row_pair('buffer_store_dword v7, v4, s[28:31], 0 offen'), WAIT_VM)),
         'is not proven lane-local'),
    )
    for label, rows, expected in controls:
        try:
            refused(expected, lambda: translate(rows))
        except AssertionError as err:
            raise AssertionError(f'{label}: {err}') from None
    refused('per-lane descriptor selection requires a proven bound heap range',
            lambda: translate(build_program(), root=H.root_blob(flags=0)))

    # Check binary reassembly against llvm-mc if provided.
    checked_asm = 0
    if llvm_mc:
        seen = set()
        for asm, hexcode in (*BOUNDED_PREFIX, MASK_AND3, *INIT_AND_GUARD, HEAD, COMPARE, SHIFT5, HEAP_LOAD,
                             *CARRY_ADD_S23, *CARRY_ADD_S22, WAIT_LGKM, BODY_LOAD, WAIT_VM, CLOSE, RESTORE,
                             FINAL_STORE, END, *ENCODINGS.items()):
            if asm not in seen and 's_cbranch' not in asm and 's_branch' not in asm:
                seen.add(asm)
                got = assemble(llvm_mc, asm + '\n')
                assert got == bytes.fromhex(hexcode), f'{asm}: {got.hex()} != {hexcode}'
                checked_asm += 1
    print(f'startup_compute_waterfall_heap: PASS (translator + {len(controls) + 1} refused controls'
          + (f' + {checked_asm} llvm-mc encodings' if checked_asm else '') + ')')


def check_oracle(dxc, offscreen_check, adapter):
    expected_binding = {'space': 0, 'lower': 1, 'size': 4}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        hlsl = d / 'bounded.hlsl'
        hlsl.write_text(BOUNDED_ORACLE_HLSL, encoding='ascii')
        results = {}
        variants = (
            ('reference', [], True, ((25, 0),)),
            ('static-switch', ['-D', 'STATIC_SWITCH'], None, ((25, 0),)),
            ('first-active', ['-D', 'FIRST_ACTIVE'], None, ((13, 12),)),
            # On WARP, dropping NonUniformResourceIndex still matches 25/25; on GPUs that scalarize dynamic
            # descriptor loads (such as RX 7900 XTX), it reads lane 0 and mismatches 12/25, like first-active.
            ('no-marker', ['-D', 'NO_MARKER'], False, ((25, 0), (13, 12))),
        )
        for name, defs, want_nu, valid_counts in variants:
            cso = d / f'{name}.cso'
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', *defs, '-Fo', cso, hlsl], check=True)
            if want_nu is not None:
                ir = subprocess.run([dxc, '-dumpbin', cso], check=True, capture_output=True, text=True).stdout
                traced = trace_srv_handle(ir, expected_binding=expected_binding)
                assert traced['non_uniform'] is want_nu, (name, traced)
            res = subprocess.run([offscreen_check, adapter, '--compute-waterfall-oracle', cso],
                                 capture_output=True, text=True)
            if res.returncode == 77:
                print(res.stdout + res.stderr)
                return 77
            assert res.returncode == 0, f'{name} failed:\n{res.stdout}{res.stderr}'
            m = re.search(r'dispatches=25 matches=(\d+) mismatches=(\d+)', res.stdout)
            assert m, res.stdout + res.stderr
            got_m, got_mis = int(m[1]), int(m[2])
            assert (got_m, got_mis) in valid_counts, f'{name}: got {(got_m, got_mis)}'
            results[name] = (got_m, got_mis)
    print(f"startup_compute_waterfall_heap_oracle ({adapter}): PASS {results}")
    return 0


def check_dispatch(consumer, dxc, adapter):
    shapes = (
        ('direct', assemble_rows(build_program())),
        ('carry64', assemble_rows(build_program(body=(SHIFT5, *CARRY_ADD_S23, WAIT_LGKM, BODY_LOAD, WAIT_VM)))),
    )
    for label, program_bytes in shapes:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache, env = prepare(directory)
            env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
            if adapter == 'hardware':
                env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
            (directory / 'compute_root').write_bytes(H.root_blob(table_range=True))
            env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
            (directory / 'program').write_bytes(program_bytes)
            (directory / 'data').write_bytes(NO_DATA)
            (directory / 'state').write_bytes(state_a(GROUP, H.RSRC2))
            untouched = [C.SENTINEL] * C.WORDS

            # 1. Before translation: refused with "no translation", output untouched.
            got, result = dispatch(directory, consumer, env, 'dispatch-heap-waterfall')
            if result.returncode == 77:
                print(result.stdout + result.stderr)
                return 77
            assert got == untouched, f'{label}: untranslated dispatch wrote output'
            assert len(lines(result, 'Dispatch(1, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
            published = lines(result, 'x2w-draw compute request')
            assert published and ' published ' in published[0], result.stderr[-3000:]
            key = int(published[0].split()[3], 16)

            # 2. Offline translate and validate DXIL nonUniformIndex metadata.
            out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                 capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
            assert f'{key:016x}: ok' in out, out
            assert 'resheap 4 2\n' in (cache / f'{key:016x}.txt').read_text(encoding='ascii')
            verify_dxil(dxc, (cache / f'{key:016x}.cs.hlsl').read_text(encoding='ascii'),
                        expected_binding={'space': 2, 'lower': 0, 'size': 4})

            # 3. Positive readback across the oracle key sets and active lane counts (0, 2, 4).
            cases = [(keys, 4) for keys in ORACLE_ROWS] + [((0, 1, 2, 3), 2), ((0, 1, 2, 3), 0)]
            for idx, (keys, active) in enumerate(cases):
                in_path = directory / f'in-{idx}.bin'
                in_words = list(keys) + [0] * (C.WORDS - 4)
                in_path.write_bytes(struct.pack(f'<{C.WORDS}I', *in_words))
                case_env = dict(env, X2W_TEST_DISPATCH_IN=str(in_path), X2W_TEST_DISPATCH_ACTIVE=str(active))
                got, result = dispatch(directory, consumer, case_env, 'dispatch-heap-waterfall')
                assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1 and not lines(result, 'refused'), \
                    result.stderr[-3000:]
                errors = [ln for ln in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in ln or ' sev=1 ' in ln]
                assert not errors, errors[0]
                expected = [1000 * ((keys[lane] & 3) + 1) + 17 * lane if lane < active else 13 for lane in range(4)] + untouched[4:]
                wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
                assert not wrong, f'{label} case {idx} keys={keys} active={active} word={wrong[0]} got={got[wrong[0]]:08x} expected={expected[wrong[0]]:08x}'

            # 4. Short-range negative control: 3-descriptor heap omits record 3 and must refuse before Dispatch.
            in_path = directory / 'in-short.bin'
            in_path.write_bytes(struct.pack(f'<{C.WORDS}I', 0, 1, 2, 3, *([0] * (C.WORDS - 4))))
            short_env = dict(env, X2W_TEST_DISPATCH_IN=str(in_path), X2W_TEST_DISPATCH_ACTIVE='4')
            got, result = dispatch(directory, consumer, short_env, 'dispatch-heap-waterfall-short')
            assert got == untouched, f'{label}: short-range control modified output'
            ref = lines(result, 'refused: ')
            census = lines(result, 'x2w-render compute census')
            want_reason = "a heap record is outside the list's view heap"
            assert len(ref) == 1 and want_reason in ref[0], result.stderr[-3000:]
            assert len(census) == 3 and census[1].endswith(f'reason "{want_reason}": 1 keys, 1 dispatches'), census
            errors = [ln for ln in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in ln or ' sev=1 ' in ln]
            assert not errors, errors[0]
    print(f'startup_compute_waterfall_heap_dispatch ({adapter}): PASS (direct + carry64 translated, '
          f'DXIL nonUniformIndex verified, all {len(cases)} key/active readbacks matched, short-range control refused)')
    return 0


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dxil')
    parser.add_argument('--oracle', nargs=3)
    parser.add_argument('--dispatch', nargs=3)
    args, unknown = parser.parse_known_args()
    if unknown:
        sys.exit(__doc__)
    if args.dxil:
        hlsl, _ = translate(build_program())
        verify_dxil(args.dxil, hlsl, expected_binding={'space': 2, 'lower': 0, 'size': 4})
        print('startup_compute_waterfall_heap_dxil: PASS (nonUniformIndex true with intrinsic, false without)')
        return 0
    if args.oracle:
        return check_oracle(*args.oracle)
    if args.dispatch:
        return check_dispatch(*args.dispatch)
    check_translator(args.llvm_mc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
