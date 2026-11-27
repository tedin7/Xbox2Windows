"""A compute program whose resource heap record index is a word it reads from a table buffer.

usage: startup_gfx2hlsl_heap_word_check.py [--llvm-mc <llvm-mc>]

The program of startup_gfx2hlsl_heap_record_check.py with its index no longer the root constant itself: root constant
word 2 picks the byte offset of a word in the buffer of table byte 96, and that word is the index:

    s_load_dwordx4 s[28:31], s[12:13], 0x60
    s_mulk_i32 s6, 0x150
    s_add_u32 s6, s6, 0x120
    s_buffer_load_dword s6, s[28:31], s6
    s_lshl_b32 s6, s6, 5
    s_load_dwordx4 s[16:19], s[8:9], s6

gfx2hlsl.py names that input `srv 1 buf heap word table 1 96 at 288 rootconst 0 2 times 336`: the dispatch reads the
word at byte 288 + 336 * (root constant word 2) of that buffer, then the 32-byte heap record it indexes. The byte
offset is the same for every thread of the dispatch. A literal byte offset, an offset modifier, s_mul_i32 and
s_addk_i32 and a later word of a wider load give the same line with their own numbers; the HLSL and the manifest of
the first form are compared whole. Every other form keeps a refusal: an offset from the group ID, an offset read
from a buffer, a scale or a byte offset that is not a multiple of 4, a register added after the arithmetic, a
modified descriptor, and a branch target between the arithmetic and the load. --llvm-mc reassembles every
instruction used here and compares the words.
"""
import argparse

import startup_gfx2hlsl_heap_record_check as H
from startup_gfx2hlsl_compute_check import G, assemble, refused, root_blob

TABLE = 's_load_dwordx4 s[28:31], s[12:13], 0x60'
SCALE, ADD = 's_mulk_i32 s6, 0x150', 's_add_u32 s6, s6, 0x120'
READ_INDEX = 's_buffer_load_dword s6, s[28:31], s6'
WAIT = 's_waitcnt lgkmcnt(0)'
INDEX = (TABLE, SCALE, ADD, WAIT, READ_INDEX, WAIT, H.SHIFT)
# Every instruction added here, recorded from llvm-mc 15.
ENCODED = {
    TABLE: '060708f4600000fa', SCALE: '500106b8', ADD: '06ff068020010000', READ_INDEX: '8e0120f40000000c',
    's_buffer_load_dwordx2 s[26:27], s[28:31], s6': '8e0624f40000000c', 's_lshl_b32 s6, s27, 5': '1b85068f',
    's_buffer_load_dword s6, s[28:31], 0x40': '8e0120f4400000fa', 's_buffer_load_dword s6, s[28:31], s14': '8e0120f40000001c',
    's_mulk_i32 s6, 0x152': '520106b8', 's_add_u32 s6, s6, 0x122': '06ff068022010000',
    's_and_b32 s29, s29, 0xffff': '1dff1d87ffff0000', 's_add_u32 s6, s6, s14': '060e0680',
    's_mul_i32 s6, s6, s5': '06050693', 's_mul_i32 s6, s6, 0x150': '06ff069350010000', 's_addk_i32 s6, 0x120': '200186b7',
    's_buffer_load_dword s6, s[28:31], s6 offset:0x10': '8e0120f41000000c',
}
ALL = {**H.ENCODED, **ENCODED}

MANIFEST = """\
x2w-manifest 2
stage cs
group 2 3 4
ud 3
udsrc 0 rootconst 0 0
udsrc 1 rootconst 0 1
udsrc 2 rootconst 0 2
srv 0 buf table 1 96
srv 1 buf heap word table 1 96 at %s
uav 0 table 1 16
"""
FORM = '288 rootconst 0 2 times 336'
BINDINGS = """\
cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; };
ByteAddressBuffer r0 : register(t0);
ByteAddressBuffer r1 : register(t1);
RWByteAddressBuffer u0 : register(u0);
"""
# The heap record program's body with the index read through r0 (the table buffer) and the input through r1.
MAIN = H.MAIN.replace('    s6 = ud2 << (0x00000005u & 31);\n', """\
    s6 = ud2 * 0x00000150u;
    scc = (s6 + 0x00000120u) < (s6);
    s6 = s6 + 0x00000120u;
    s6 = r0.Load(s6 + 0);
    s6 = s6 << (0x00000005u & 31);
""").replace('v5 = r0.Load(a + 0);', 'v5 = r1.Load(a + 0);')
assert MAIN.count('r0.Load') == 1 and MAIN.count('r1.Load') == 1


def words(index=INDEX):
    """The heap record program with the index lines in place of its shift."""
    return b''.join(bytes.fromhex(ALL[line]) for original in H.PROGRAM
                    for line in (index if original == H.SHIFT else (original,)))


WORDS = words()


def pipeline(index=INDEX):
    return G.ComputePipeline(words(index), H.root_blob(), H.RSRC2, H.GROUP)


def replace(old, *new):
    assert old in INDEX
    return tuple(line for original in INDEX for line in (new if original == old else (original,)))


def check_lowering():
    pipe = pipeline()
    hlsl = pipe.hlsl_cs()
    assert pipe.manifest() == MANIFEST % FORM, pipe.manifest()
    assert hlsl.endswith(BINDINGS + MAIN), hlsl[hlsl.index('cbuffer'):]
    forms = (
        ('literal byte offset', (TABLE, WAIT, 's_buffer_load_dword s6, s[28:31], 0x40', WAIT, H.SHIFT), '64'),
        ('offset modifier', replace(READ_INDEX, 's_buffer_load_dword s6, s[28:31], s6 offset:0x10'),
         '304 rootconst 0 2 times 336'),
        ('s_mul_i32 and s_addk_i32', replace(ADD, 's_addk_i32 s6, 0x120'), FORM),
        ('s_mul_i32 by a literal', replace(SCALE, 's_mul_i32 s6, s6, 0x150'), FORM),
        ('second word of a wider load', replace(H.SHIFT, 's_lshl_b32 s6, s27, 5')[:4] + (
            's_buffer_load_dwordx2 s[26:27], s[28:31], s6', WAIT, 's_lshl_b32 s6, s27, 5'), '292 rootconst 0 2 times 336'),
        # A branch target before the arithmetic changes nothing.
        ('branch target before the arithmetic', (TABLE,) + H.JUMP + INDEX[1:], FORM),
    )
    for name, index, form in forms:
        got = pipeline(index).manifest()
        assert got == MANIFEST % form, '%s:\n%s' % (name, got)
    return len(forms)


def check_refusals():
    other = ('s_load of a resource heap record whose index is read from a buffer other than a table buffer at a '
             'dispatch-constant offset')
    controls = (
        ('offset from the group ID', replace(READ_INDEX, 's_buffer_load_dword s6, s[28:31], s14')),
        ('offset read from a buffer', replace(READ_INDEX, 's_buffer_load_dword s6, s[28:31], 0x40', WAIT, READ_INDEX)),
        ('scale not a multiple of 4', replace(SCALE, 's_mulk_i32 s6, 0x152')),
        ('byte offset not a multiple of 4', replace(ADD, 's_add_u32 s6, s6, 0x122')),
        ('a register added after the arithmetic', replace(ADD, ADD, 's_add_u32 s6, s6, s14')),
        ('scaled by a register', replace(SCALE, 's_mul_i32 s6, s6, s5')),
        ('modified descriptor', replace(TABLE, TABLE, WAIT, 's_and_b32 s29, s29, 0xffff')),
        ('branch target between the arithmetic and the load', replace(ADD, ADD, *H.JUMP)),
    )
    for name, index in controls:
        try:
            refused(other, lambda: pipeline(index).manifest())
        except AssertionError as error:
            raise AssertionError('%s: %s' % (name, error)) from None
    # A branch target between the load and the shift ends the word's mark, as for a root constant.
    refused('whose offset is not a root constant shifted left by 5',
            lambda: pipeline(replace(READ_INDEX, READ_INDEX, *H.JUMP)).manifest())
    # Only a buffer read is named in the manifest; another kind through such a record is refused there.
    pipe = pipeline()
    key = next(k for k in pipe.resources if k[0][:2] == ('resheap', 'word'))
    pipe.resources[key]['kind'] = 'tex2d'
    refused("resource through ('resheap', 'word'", pipe.manifest)
    return len(controls) + 2


def check_binary(llvm_mc):
    for line, encoded in ENCODED.items():
        assert assemble(llvm_mc, line + '\n') == bytes.fromhex(encoded), line
    text = [line for original in H.PROGRAM for line in (INDEX if original == H.SHIFT else (original,))]
    assert assemble(llvm_mc, '\n'.join(text) + '\n') == WORDS
    print('heap word fixture: %d instructions reassembled to the recorded words' % len(ENCODED))


RAW_WORDS = bytes.fromhex(
    '800387be '  # s_mov_b32 s7, 0
    '030308f4000000fa '  # s_load_dwordx4 s[12:15], s[6:7], 0
    '030608f4100000fa '  # s_load_dwordx4 s[24:27], s[6:7], 16
    '7fc08cbf '  # s_waitcnt lgkmcnt(0)
    '0e891e93 '  # s_mul_i32 s30, s14, 9
    '0c8012bf '  # s_cmp_eq_u64 s[12:13], 0
    '81801f85 '  # s_cselect_b32 s31, 1, 0
    '1e020e7e '  # v_mov_b32 v7, s30
    '1f02107e '  # v_mov_b32 v8, s31
    '83000c34 '  # v_lshlrev_b32 v6, 3, v0
    '001074e006070680 '  # buffer_store_dwordx2 v[7:8], v6, s[24:27], 0 offen
    '000081bf '  # s_endpgm
)


PAIR_WORDS = bytes.fromhex(
    '070394be '  # s_mov_b32 s20, s7
    '800395be '  # s_mov_b32 s21, 0
    '800389be '  # s_mov_b32 s9, 0
    '06851c8f '  # s_lshl_b32 s28, s6, 5
    '040810f400000038 '  # s_load_dwordx16 s[32:47], s[8:9], s28
    '0a0608f4100000fa '  # s_load_dwordx4 s[24:27], s[20:21], 16
    '7fc08cbf '  # s_waitcnt lgkmcnt(0)
    '800390be '  # s_mov_b32 s16, 0
    '800391be '  # s_mov_b32 s17, 0
    '800392be '  # s_mov_b32 s18, 0
    '800393be '  # s_mov_b32 s19, 0
    'f002187e '  # v_mov_b32 v12, 0.5
    'f0021a7e '  # v_mov_b32 v13, 0.5
    '08019cf00c078800 '  # image_sample_lz v7, v[12:13], s[32:39], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
    '08019cf00c088a00 '  # image_sample_lz v8, v[12:13], s[40:47], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
    '703f8cbf '  # s_waitcnt vmcnt(0)
    '83000c34 '  # v_lshlrev_b32 v6, 3, v0
    '001074e006070680 '  # buffer_store_dwordx2 v[7:8], v6, s[24:27], 0 offen
    '000081bf '  # s_endpgm
)


# The second sample uses the sampler immediately after the image. The remainder of
# the wide read includes another sampler to serve as the wrong-offset control.
MIXED_WORDS = bytes.fromhex(
    '070394be '  # s_mov_b32 s20, s7
    '800395be '  # s_mov_b32 s21, 0
    '800389be '  # s_mov_b32 s9, 0
    '06851c8f '  # s_lshl_b32 s28, s6, 5
    '040810f400000038 '  # s_load_dwordx16 s[32:47], s[8:9], s28
    '0a0608f4100000fa '  # s_load_dwordx4 s[24:27], s[20:21], 16
    '7fc08cbf '  # s_waitcnt lgkmcnt(0)
    '800390be '  # s_mov_b32 s16, 0
    '800391be '  # s_mov_b32 s17, 0
    '800392be '  # s_mov_b32 s18, 0
    '800393be '  # s_mov_b32 s19, 0
    'ff02187e0000c03f '  # v_mov_b32 v12, 0x3fc00000
    'f0021a7e '  # v_mov_b32 v13, 0.5
    '08019cf00c078800 '  # image_sample_lz v7, v[12:13], s[32:39], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
    '08019cf00c084801 '  # image_sample_lz v8, v[12:13], s[32:39], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D
    '703f8cbf '  # s_waitcnt vmcnt(0)
    '83000c34 '  # v_lshlrev_b32 v6, 3, v0
    '001074e006070680 '  # buffer_store_dwordx2 v[7:8], v6, s[24:27], 0 offen
    '000081bf '  # s_endpgm
)


def check_raw_descriptor_values():
    def lower(code):
        return G.ComputePipeline(code, root_blob(), 7 << 1, (8, 1, 1))

    pipe = lower(RAW_WORDS)
    assert 'udsrc 2 desc_word table 1 0 2' in pipe.manifest()
    assert 'udsrc 3 desc_null table 1 0 0' in pipe.manifest()
    assert 'u0.Store2' in pipe.hlsl_cs()
    address_read = RAW_WORDS.replace(bytes.fromhex('0e891e93'), bytes.fromhex('0c891e93'))
    assert address_read != RAW_WORDS
    refused('numeric use of a descriptor address word', lambda: lower(address_read))

    pipe = G.ComputePipeline(PAIR_WORDS, H.root_blob(), 9 << 1, (8, 1, 1))
    assert 'srv 0 tex2d heap record 0 2 -1 0 0 0 0' in pipe.manifest()
    assert 'srv 1 tex2d heap record 0 2 -1 0 0 0 32' in pipe.manifest()
    assert 'r0.SampleLevel' in pipe.hlsl_cs() and 'r1.SampleLevel' in pipe.hlsl_cs()

    pipe = G.ComputePipeline(MIXED_WORDS, H.root_blob(), 9 << 1, (8, 1, 1))
    assert 'smprecord 0 heap record 0 2 -1 0 0 0 32' in pipe.manifest()
    assert 'r0.SampleLevel(qr[0]' in pipe.hlsl_cs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    args = parser.parse_args()
    if args.llvm_mc:
        check_binary(args.llvm_mc)
    forms = check_lowering()
    count = check_refusals()
    check_raw_descriptor_values()
    print('startup_gfx2hlsl_heap_word: PASS (manifest and HLSL exact; %d more forms; %d refused controls)' % (forms, count))
