"""A compute program that reads a buffer descriptor from the resource heap record a root constant indexes.

usage: startup_gfx2hlsl_heap_record_check.py [--llvm-mc <llvm-mc>]

The root signature has three root constants (s4-s6), one descriptor table (s7) and the directly indexed
resource heap (root flag 0x400, pointer in s8). The program is case A of startup_gfx2hlsl_compute_check.py
with its input descriptor loaded as

    s_lshl_b32 s6, s6, 5
    s_load_dwordx4 s[16:19], s[8:9], s6

after s9 is set to zero. gfx2hlsl.py names that input `srv 0 buf heap rootconst 0 2` in the manifest: the
dispatch reads the 32-byte heap record that root constant word 2 indexes. The HLSL and the manifest are
compared whole. Every other form keeps a refusal: another shift, another index, an index register written
again, a branch target between the shift and the load, a high word that is not a constant zero, an offset
modifier, eight dwords, an index read from a buffer at an offset that differs between groups (named as such;
startup_gfx2hlsl_heap_word_check.py covers the buffer forms that are accepted), a formatted load or a store through
the record, and a nonzero graphics pointer high word. --llvm-mc reassembles every instruction used here and compares the words.
"""
import argparse
import struct

from startup_gfx2hlsl_compute_check import G, TGID_X, TGID_Y, TGID_Z, assemble, refused, tidig

GROUP, GROUPS = (2, 3, 4), (3, 2, 2)
RSRC2 = (9 << 1) | TGID_X | TGID_Y | TGID_Z | tidig(3)  # s0-s3, three constants, the table, the heap pointer
INDEX_WORD = 2  # the root constant word the program shifts

SHIFT, CLEAR, LOAD = 's_lshl_b32 s6, s6, 5', 's_mov_b32 s9, 0', 's_load_dwordx4 s[16:19], s[8:9], s6'
READ = 'buffer_load_dword v5, v4, s[16:19], 0 offen'
WRITE = 'buffer_store_dword v5, v4, s[20:23], 0 offen'
PROGRAM = (
    's_mov_b32 s12, s7', 's_mov_b32 s13, 0', 's_load_dwordx4 s[20:23], s[12:13], 0x10',
    's_mov_b32 s14, s9', CLEAR, SHIFT, LOAD, 's_waitcnt lgkmcnt(0)',
    'v_mad_u32_u24 v3, v1, 2, v0', 'v_mad_u32_u24 v3, v2, 6, v3', 's_mul_i32 s24, s10, 3', 's_add_u32 s24, s24, s14',
    's_mul_i32 s25, s11, 6', 's_add_u32 s24, s24, s25', 's_mul_i32 s24, s24, 24', 'v_add_nc_u32 v3, s24, v3',
    'v_lshlrev_b32 v4, 2, v3', READ, 'v_mul_lo_u32 v6, v3, s5', 'v_add_nc_u32 v6, s4, v6', 's_waitcnt vmcnt(0)',
    'v_mul_lo_u32 v5, v5, 33', 'v_xor_b32 v5, v5, v6', WRITE, 's_endpgm')
# Every instruction of the program and of the controls, recorded from llvm-mc 15.
ENCODED = {
    's_mov_b32 s12, s7': '07038cbe', 's_mov_b32 s13, 0': '80038dbe',
    's_load_dwordx4 s[20:23], s[12:13], 0x10': '060508f4100000fa', 's_mov_b32 s14, s9': '09038ebe',
    CLEAR: '800389be', SHIFT: '0685068f', LOAD: '040408f40000000c', 's_waitcnt lgkmcnt(0)': '7fc08cbf',
    'v_mad_u32_u24 v3, v1, 2, v0': '030043d501050104', 'v_mad_u32_u24 v3, v2, 6, v3': '030043d5020d0d04',
    's_mul_i32 s24, s10, 3': '0a831893', 's_add_u32 s24, s24, s14': '180e1880', 's_mul_i32 s25, s11, 6': '0b861993',
    's_add_u32 s24, s24, s25': '18191880', 's_mul_i32 s24, s24, 24': '18981893', 'v_add_nc_u32 v3, s24, v3': '1806064a',
    'v_lshlrev_b32 v4, 2, v3': '82060834', READ: '001030e004050480', 'v_mul_lo_u32 v6, v3, s5': '060069d5030b0000',
    'v_add_nc_u32 v6, s4, v6': '040c0c4a', 's_waitcnt vmcnt(0)': '703f8cbf', 'v_mul_lo_u32 v5, v5, 33': '050069d505430100',
    'v_xor_b32 v5, v5, v6': '050d0a3a', WRITE: '001070e004050580', 's_endpgm': '000081bf',
    's_lshl_b32 s6, s6, 4': '0684068f', 's_lshl_b32 s6, s6, 6': '0686068f', 's_lshl_b32 s6, s6, s5': '0605068f',
    's_lshl_b32 s6, s14, 5': '0e85068f', 's_mov_b32 s9, 1': '810389be', 's_mov_b32 s9, s5': '050389be',
    's_load_dwordx4 s[16:19], s[8:9], s6 offset:0x20': '040408f42000000c',
    's_load_dwordx8 s[16:23], s[8:9], s6': '04040cf40000000c', 's_load_dwordx4 s[16:19], s[8:9], s5': '040408f40000000a',
    's_load_dwordx4 s[16:19], s[8:9], m0': '040408f4000000f8', 's_add_u32 s6, s6, 0': '06800680',
    's_load_dwordx4 s[28:31], s[12:13], 0x20': '060708f4200000fa',
    's_buffer_load_dword s6, s[28:31], 0x0': '8e0120f4000000fa', 's_buffer_load_dword s6, s[28:31], s14': '8e0120f40000001c',
    's_buffer_load_dword s26, s[28:31], s14': '8e0620f40000001c',
    's_lshl_b32 s6, s26, 5': '1a85068f', 's_cmp_eq_u32 s4, 0': '048006bf', 's_cbranch_scc0 1': '010084bf', 's_nop 0': '000080bf',
    'buffer_load_format_xyzw v[5:8], v4, s[16:19], 0 offen': '00100ce004050480',
    'buffer_store_dword v5, v4, s[16:19], 0 offen': '001070e004050480',
}
TABLE_LOAD = 's_load_dwordx4 s[28:31], s[12:13], 0x20'
JUMP = ('s_cmp_eq_u32 s4, 0', 's_cbranch_scc0 1', 's_nop 0')  # the instruction after the s_nop is a branch target

MANIFEST = """\
x2w-manifest 2
stage cs
group 2 3 4
ud 3
udsrc 0 rootconst 0 0
udsrc 1 rootconst 0 1
udsrc 2 rootconst 0 2
srv 0 buf heap rootconst 0 2
uav 0 table 1 16
"""
BINDINGS = """\
cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; };
ByteAddressBuffer r0 : register(t0);
RWByteAddressBuffer u0 : register(u0);
"""
MAIN = """\
[numthreads(2, 3, 4)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint s6 = 0, s9 = 0, s10 = 0, s11 = 0, s13 = 0, s14 = 0, s24 = 0, s25 = 0;
    uint v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;
    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;
    s9 = gid.x;
    s10 = gid.y;
    s11 = gid.z;
    v0 = tid.x;
    v1 = tid.y;
    v2 = tid.z;
    s13 = 0x00000000u;
    s14 = s9;
    s9 = 0x00000000u;
    s6 = ud2 << (0x00000005u & 31);
    scc = s6 != 0;
    v3 = (v1 & 0xffffffu) * (0x00000002u & 0xffffffu) + v0;
    v3 = (v2 & 0xffffffu) * (0x00000006u & 0xffffffu) + v3;
    s24 = s10 * 0x00000003u;
    scc = (s24 + s14) < (s24);
    s24 = s24 + s14;
    s25 = s11 * 0x00000006u;
    scc = (s24 + s25) < (s24);
    s24 = s24 + s25;
    s24 = s24 * 0x00000018u;
    v3 = s24 + v3;
    v4 = v3 << (0x00000002u & 31);
    a = v4 + 0;
    v5 = r0.Load(a + 0);
    v6 = v3 * ud1;
    v6 = ud0 + v6;
    v5 = v5 * 0x00000021u;
    v5 = v5 ^ v6;
    a = (v4 + 0) & 0xfffffffcu;
    u0.Store(a, v5);
}
"""


def root_blob(table_range=False, flags=0x400):
    """RTS0: param 0 = three root constants (b0), param 1 = a descriptor table, the resource heap directly indexed.
    table_range gives the table one SRV range, as startup_gfx2hlsl_compute_check.root_blob does for a device."""
    params_at, constants_at = 24, 24 + 2 * 12
    table_at = constants_at + 12
    ranges = struct.pack('<6I', 0, 1, 0, 0, 0, 0xffffffff) if table_range else b''
    rts0 = (struct.pack('<6I', 2, 2, params_at, 0, table_at + 8 + len(ranges), flags)
            + struct.pack('<3I', 1, 0, constants_at) + struct.pack('<3I', 0, 0, table_at)
            + struct.pack('<3I', 0, 0, 3) + struct.pack('<2I', len(ranges) // 24, table_at + 8) + ranges)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part


def words(edits=None):
    """The program's code; edits maps one of its lines to the lines that take its place."""
    edits = edits or {}
    assert set(edits) <= set(PROGRAM), edits
    return b''.join(bytes.fromhex(ENCODED[line]) for original in PROGRAM for line in edits.get(original, (original,)))


WORDS = words()


def pipeline(edits=None, root=None):
    return G.ComputePipeline(words(edits), root or root_blob(), RSRC2, GROUP)


def check_lowering():
    pipe = pipeline()
    hlsl = pipe.hlsl_cs()
    assert pipe.manifest() == MANIFEST, pipe.manifest()
    assert hlsl.endswith(BINDINGS + MAIN), hlsl[hlsl.index('cbuffer'):]
    # A branch target before the shift changes nothing: the form is not tied to the entry block.
    assert pipeline({SHIFT: JUMP + (SHIFT,)}).manifest() == MANIFEST


def check_refusals():
    not_shifted = 'whose offset is not a root constant shifted left by 5 since the last branch target'
    from_buffer = 's_load of a resource heap record whose index is read from a buffer'
    high = 'through a pointer whose high word is not a constant zero'
    controls = (
        ('shift by 4', {SHIFT: ('s_lshl_b32 s6, s6, 4',)}, not_shifted),
        ('shift by 6', {SHIFT: ('s_lshl_b32 s6, s6, 6',)}, not_shifted),
        ('shift by a register', {SHIFT: ('s_lshl_b32 s6, s6, s5',)}, not_shifted),
        ('no shift', {SHIFT: ()}, not_shifted),
        ('a group ID shifted', {SHIFT: ('s_lshl_b32 s6, s14, 5',)}, not_shifted),
        ('another register as the offset', {LOAD: ('s_load_dwordx4 s[16:19], s[8:9], s5',)}, not_shifted),
        ('m0 as the offset', {LOAD: ('s_load_dwordx4 s[16:19], s[8:9], m0',)}, not_shifted),
        ('index register written again', {SHIFT: (SHIFT, 's_add_u32 s6, s6, 0')}, not_shifted),
        ('index register loaded from a buffer after the shift',
         {SHIFT: (SHIFT, TABLE_LOAD, 's_buffer_load_dword s6, s[28:31], 0x0')}, not_shifted),
        ('branch target between the shift and the load', {SHIFT: (SHIFT,) + JUMP}, not_shifted),
        ('index read from a buffer at a group-ID offset',
         {SHIFT: (TABLE_LOAD, 's_buffer_load_dword s26, s[28:31], s14', 's_lshl_b32 s6, s26, 5')}, from_buffer),
        ('index read from a buffer at a group-ID offset into the shifted register',
         {SHIFT: (TABLE_LOAD, 's_buffer_load_dword s6, s[28:31], s14', SHIFT)}, from_buffer),
        ('high word 1', {CLEAR: ('s_mov_b32 s9, 1',)}, high),
        ('high word a root constant', {CLEAR: ('s_mov_b32 s9, s5',)}, high),
        ('high word left as the group ID', {CLEAR: ()}, high),
        ('offset modifier', {LOAD: ('s_load_dwordx4 s[16:19], s[8:9], s6 offset:0x20',)}, 'with offset:0x20'),
        ('eight dwords', {LOAD: ('s_load_dwordx8 s[16:23], s[8:9], s6',)}, 's_load of a resource heap record of 8 dwords'),
        ('formatted load', {READ: ('buffer_load_format_xyzw v[5:8], v4, s[16:19], 0 offen',)},
         'formatted buffer load requires a hardware table descriptor'),
        ('store through the record', {WRITE: ('buffer_store_dword v5, v4, s[16:19], 0 offen',)}, "('resheap', 0, 2)"),
    )
    for name, edits, reason in controls:
        try:
            refused(reason, lambda: pipeline(edits).manifest())
        except AssertionError as error:
            raise AssertionError('%s: %s' % (name, error)) from None
    # A root signature without the directly indexed heap has no heap pointer: s8 is then a plain user SGPR.
    refused('s_load through s8 holding', lambda: pipeline(root=root_blob(flags=0)))
    # A graphics descriptor still requires a zero high word.
    stage = pipeline().cs
    stage.kind = 'ps'
    stage.sym[9] = ('const', 1)
    refused("s_load with a register offset through ('resheap',)",
            lambda: stage.op_s_load_dwordx4(['s[16:19]', 's[8:9]', 's6'], []))
    # Only a buffer read is named in the manifest; a store or another kind through a record is refused there too.
    pipe = pipeline()
    key = (('resheap', 0, INDEX_WORD), 0)
    assert pipe.resources[key]['kind'] == 'buf'
    pipe.resources[key]['kind'] = 'tex2d'
    refused("resource through ('resheap', 0, 2)", pipe.manifest)
    pipe = pipeline()
    pipe.uavs[key] = dict(index=1, kind='raw')
    refused("store through ('resheap', 0, 2)", pipe.manifest)
    return len(controls)


def check_mutations():
    """Translator mutation a control above must catch: without the write tracking the refused control translates."""
    again = {SHIFT: (SHIFT, 's_add_u32 s6, s6, 0')}
    saved = G.scalar_destinations
    G.scalar_destinations = lambda name, asm: set() if name.startswith('s_add_u32') else saved(name, asm)
    try:
        assert pipeline(again).manifest() == MANIFEST, 'a write no longer ends the mark, and the control did not change'
    finally:
        G.scalar_destinations = saved
    refused('not a root constant', lambda: pipeline(again))


def check_binary(llvm_mc):
    for line, encoded in ENCODED.items():
        assert assemble(llvm_mc, line + '\n') == bytes.fromhex(encoded), line
    assert assemble(llvm_mc, '\n'.join(PROGRAM) + '\n') == WORDS
    print('heap record fixture: %d instructions reassembled to the recorded words' % len(ENCODED))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    args = parser.parse_args()
    if args.llvm_mc:
        check_binary(args.llvm_mc)
    check_lowering()
    count = check_refusals()
    check_mutations()
    print('startup_gfx2hlsl_heap_record: PASS (manifest and HLSL exact; %d refused controls; graphics, manifest and '
          'translator-mutation checks)' % count)
