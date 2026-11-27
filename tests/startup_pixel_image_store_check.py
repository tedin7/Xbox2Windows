"""Check gfx2hlsl's pixel-stage image_store on a hand-assembled gfx1013 program (#1996 item F).

The pixel program loads an image descriptor from descriptor table 0, sets the texel address (1, 2) and
four distinct binary32 values, stores them with image_store ... dmask:0xf dim:SQ_RSRC_IMG_2D, and exports
the same values to mrt0. The words were assembled by hand from the MIMG layout and are decoded back by
tools/disasm_rdna2.py here; --llvm-mc reassembles the source and compares the words.

Without arguments: translation controls and refusals. With draw: also compile the translation with DXC and
draw it through offscreen_check --startup-image-store, which reads every texel of a 4x4 image back. Negative
controls: the translator refuses the program when the lowering is removed, and the translated shader with
its store deleted, or with x and y exchanged, leaves texel (1, 2) at its starting value.

The store runs under the lane's exec bit. Two variants put v_cmpx_eq_u32 before the store, so exec is not
known at translation time: with 1 == v0 the lane stays on and the texel is written, with 2 == v0 every lane
is off and the texel keeps its starting value. A control draws the exec-off variant with the guard removed
from the store and must see the texel written.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools'))
from startup_image_3d_check import FULLSCREEN_VS, G, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402
import disasm_rdna2 as D  # noqa: E402

VALUES = (1.0, 2.0, 3.0, 4.0)
BITS = ['%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in VALUES]

SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, 1
v_mov_b32 v1, 2
v_mov_b32 v4, 1.0
v_mov_b32 v5, 2.0
v_mov_b32 v6, 0x40400000
v_mov_b32 v7, 4.0
s_waitcnt lgkmcnt(0)
image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
STORE = bytes.fromhex('080f20f000040400')
ARRAY_STORE = bytes.fromhex('280f20f000040400')
PREFIX = bytes.fromhex('04040cf4000000fa' '8102007e' '8202027e' 'f202087e' 'f4020a7e' 'ff020c7e00004040'
                       'f6020e7e' '7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf' '0f1800f804050607' '000081bf')
WORDS = PREFIX + STORE + SUFFIX
# a16: x and y share v0 (x low half, y high half). v0 = 1 | (2 << 16) addresses the same texel (1, 2); v1 holds a
# sentinel that must not be read as part of the address.
A16_STORE = bytes.fromhex('080f20f000040440')
A16_PREFIX = PREFIX.replace(bytes.fromhex('8102007e' '8202027e'), bytes.fromhex('ff02007e' '01000200' '8202027e'))
A16_WORDS = A16_PREFIX + A16_STORE + SUFFIX
A16_LINE = 'w0[uint2(v0 & 0xffffu, v0 >> 16)] = asfloat(uint4(v4, v5, v6, v7));'
STORE_LINE = 'w0[uint2(v0, v1)] = asfloat(uint4(v4, v5, v6, v7));'
GUARDED_LINE = 'if (exec & 1) ' + STORE_LINE
ARRAY_SOURCE = SOURCE.replace(
    'image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D',
    'v_mov_b32 v2, 1\nimage_store v[4:7], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY',
)
ARRAY_STORE_LINE = 'w0[uint3(v0, v1, v2)] = asfloat(uint4(v4, v5, v6, v7));'
ARRAY_GUARDED_LINE = 'if (exec & 1) ' + ARRAY_STORE_LINE


def mov_v2_inline(constant):
    """v_mov_b32_e32 v2, <constant>: VOP1 opcode 0x01, vdst=2, inline integer 0..64 (0x80 + n)."""
    return struct.pack('<I', 0x7e040280 + constant)


def array_program(layer):
    return PREFIX + mov_v2_inline(layer) + ARRAY_STORE + SUFFIX


ARRAY_WORDS = array_program(1)
ARRAY_WORDS_LAST = array_program(3)


def cmpx_eq(constant):
    """v_cmpx_eq_u32_e32 <constant>, v0: VOPC (bits 31:25 = 0x3e), opcode 0xd2, VSRC1 v0, SRC0 an inline
    integer 1..64 (0x80 + n). In gfx10 the VOPC v_cmpx form writes only exec."""
    return struct.pack('<I', 0x7c000000 | (0xd2 << 17) | (0 << 9) | (0x80 + constant))


EXEC_ON = PREFIX + cmpx_eq(1) + STORE + SUFFIX   # v0 is 1: the lane stays active
EXEC_OFF = PREFIX + cmpx_eq(2) + STORE + SUFFIX  # v0 is 1: every lane is turned off before the store
ARRAY_EXEC_ON = PREFIX + mov_v2_inline(1) + cmpx_eq(1) + ARRAY_STORE + SUFFIX
ARRAY_EXEC_OFF = PREFIX + mov_v2_inline(1) + cmpx_eq(2) + ARRAY_STORE + SUFFIX


# Each active lane updates its own pixel. s9..s12 supply the four entry cbuffer words.
RW_DESCRIPTOR = bytes.fromhex('04060cf4000000fa')  # s_load_dwordx8 s[24:31], s[8:9], 0
RW_POSITION = bytes.fromhex(
    '7fc08cbf'          # s_waitcnt lgkmcnt(0)
    'ff041836ffff0000'  # v_and_b32 v12, 0xffff, v2
    '90041a2c'          # v_lshrrev_b32 v13, 16, v2
    '81021c7e'          # v_mov_b32 v14, 1 (array view slice)
)
RW_NARROW = bytes.fromhex('8818a87d')  # v_cmpx_gt_u32 8, v12
RW_LOAD = bytes.fromhex('080f00f00c200600')  # image_load v[32:35], v[12:13], s[24:31]
RW_STORE = bytes.fromhex('080f20f00c200600')
RW_ADD = bytes.fromhex('703f8cbf' '09404006' '0a424206' '0b444406' '0c464606')
RW_EXPORT = bytes.fromhex('0f1800f820212223' '000081bf')


def rw_program(*, array=False, stale=False, stride=32):
    descriptors = RW_DESCRIPTOR
    load, store = RW_LOAD, RW_STORE
    if stale:
        # s[36:43], a distinct table entry one descriptor stride in (32 bytes on the console, the adapter's
        # increment on the desktop: 64 on WARP)
        descriptors += bytes.fromhex('04090cf4') + struct.pack('<I', 0xfa000000 | stride)
        load = load[:6] + bytes.fromhex('0900')
    if array:
        load, store = b'\x28' + load[1:], b'\x28' + store[1:]
    return descriptors + RW_POSITION + RW_NARROW + load + RW_ADD + store + RW_EXPORT


def rw_root():
    root = bytearray(root_blob())
    struct.pack_into('<I', root, 100, 4)  # RTS0 parameter 1: four root constants
    return bytes(root)


def read_write_controls():
    for array in (False, True):
        kind = 'tex2da' if array else 'tex2d'
        coord = 'uint3(v12, v13, v14)' if array else 'uint2(v12, v13)'
        words = rw_program(array=array)
        pipe = G.Pipeline(VS_WORDS, words, rw_root())
        hlsl, manifest = pipe.hlsl_ps(), pipe.manifest()
        load = 'if (exec & 1) t = w0[%s];' % coord
        store = 'if (exec & 1) w0[%s] =' % coord
        assert hlsl.index(load) < hlsl.index(store), hlsl
        assert 'uav 0 %s table 0 0\nuavload 0\n' % kind in manifest, manifest
        assert not any(line.startswith('srv ') for line in manifest.splitlines()), manifest
        census = []
        decoded_pipe = G.Pipeline(VS_WORDS, words, rw_root(), refusals=census)
        assert not census and decoded_pipe.hlsl_ps() == hlsl and decoded_pipe.manifest() == manifest

        full_store = (b'\x28' + RW_STORE[1:]) if array else RW_STORE
        partial_store = full_store[:1] + b'\x09' + full_store[2:]
        partial_words = words.replace(full_store, partial_store)
        partial = G.Pipeline(VS_WORDS, partial_words, rw_root())
        partial_hlsl = partial.hlsl_ps()
        assert store + ' asfloat(uint4(v32, 0u, 0u, v33));' in partial_hlsl, partial_hlsl
        assert partial_hlsl.count(f'w0[{coord}]') == 2, partial_hlsl
        # The explicit image_load still needs typed UAV loads on the shared binding.
        assert 'uavload 0' in partial.manifest().splitlines()

        stale = G.Pipeline(VS_WORDS, rw_program(array=array, stale=True), rw_root())
        assert 'srv 0 %s table 0 32' % kind in stale.manifest() and 'uavload' not in stale.manifest()

    # This sampler has linear XY filtering. Prove sample/LOD can translate on their own.
    sampler = bytes.fromhex('9203a8be' '8003a9be' 'ff03aabe00005004' '8003abbe')
    before = RW_DESCRIPTOR + RW_POSITION
    after = RW_ADD + RW_STORE + RW_EXPORT
    conflict = 'both read and stored to'
    sample = bytes.fromhex('080f80f00c204601')
    modified_descriptor = bytes.fromhex('80039fbe')  # s_mov_b32 s31, 0
    for words in (
        before + sampler + RW_LOAD + after[:-len(RW_EXPORT)] + sample + RW_EXPORT,
        before + RW_LOAD + modified_descriptor + after,
    ):
        with_refusal = []
        G.Pipeline(VS_WORDS, words, rw_root(), refusals=with_refusal)
        assert any(conflict in reason for reason, _ in with_refusal), with_refusal

    assert 'uavload 0' in G.Pipeline(
        VS_WORDS, before + RW_STORE + RW_LOAD + RW_ADD + RW_EXPORT, rw_root()).manifest()


    for label, load in (
        ('mip 1', '080f04f00c200600'),
        ('a16', '080f00f002200640'),
        ('d16', '080f00f00c200680'),
        ('mixed dimension', '280f00f00c200600'),
        ('filtered sample', '080f80f00c204601'),
        ('LOD query', '080380f10c204601'),
        ('gather', '08011cf10c204601'),
    ):
        head, read = before + sampler, bytes.fromhex(load)
        G.Pipeline(VS_WORDS, head + read + RW_ADD + RW_EXPORT, rw_root())
        try:
            G.Pipeline(VS_WORDS, head + read + after, rw_root())
        except G.Unsupported as error:
            assert conflict in str(error), (label, error)
        else:
            raise AssertionError('accepted shared image with ' + label)

    with patch.dict('os.environ', X2W_INTEGER_TEXTURE_SRVS='1'):
        try:
            G.Pipeline(VS_WORDS, rw_program(), rw_root())
        except G.Unsupported as error:
            assert conflict in str(error), error
        else:
            raise AssertionError('accepted shared image through integer SRV path')
    print('shared 2D and array UAV loads and stores; incompatible reads refused')


def translate(words):
    return G.Pipeline(VS_WORDS, words, root_blob())


def refused(words, *needles):
    try:
        translate(words)
    except G.Unsupported as error:
        text = str(error)
        assert all(n in text for n in needles), (needles, text)
        return text
    raise AssertionError('accepted: %s' % (needles,))


def with_store(store):
    return PREFIX + store + SUFFIX


def patched(byte, mask, value=None):
    store = bytearray(STORE)
    store[byte] = value if value is not None else store[byte] | mask
    return bytes(store)


# The packed 16-bit form (a16, x in the low half and y in the high half of one VGPR) of the same read-modify-write:
# the load and the store name v2. PACKED_OTHER stores at v3 and PACKED_MOVED changes v2 between the two.
PACKED_SOURCE = """s_load_dwordx8 s[24:31], s[8:9], 0x0
s_waitcnt lgkmcnt(0)
{pre}image_load v[32:35], {load_at}, s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16
s_waitcnt vmcnt(0)
v_add_f32_e32 v32, s9, v32
v_add_f32_e32 v33, s10, v33
v_add_f32_e32 v34, s11, v34
v_add_f32_e32 v35, s12, v35
{between}image_store v[32:35], {store_at}, s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16
exp mrt0 v32, v33, v34, v35 done vm
s_endpgm
"""
PACKED_FIELDS = {
    'packed': dict(pre='', load_at='v2', between='', store_at='v2'),
    'other texel': dict(pre='', load_at='v2', between='', store_at='v3'),
    'moved coordinate': dict(pre='', load_at='v2', between='v_add_nc_u32_e32 v2, 1, v2\n', store_at='v2'),
}
PACKED_WORDS = {
    'packed': bytes.fromhex('04060cf4000000fa7fc08cbf081f00f002200640703f8cbf094040060a4242060b4444060c464606081f20f0022006400f1800f820212223000081bf'),
    'other texel': bytes.fromhex('04060cf4000000fa7fc08cbf081f00f002200640703f8cbf094040060a4242060b4444060c464606081f20f0032006400f1800f820212223000081bf'),
    'moved coordinate': bytes.fromhex('04060cf4000000fa7fc08cbf081f00f002200640703f8cbf094040060a4242060b4444060c4646068104044a081f20f0022006400f1800f820212223000081bf'),
}


def packed_controls():
    pipe = G.Pipeline(VS_WORDS, PACKED_WORDS['packed'], rw_root())
    hlsl, manifest = pipe.hlsl_ps(), pipe.manifest()
    unpacked = 'uint2(v2 & 0xffffu, v2 >> 16)'
    assert 't = w0[uint2(v2 & 0xffffu, v2 >> 16)];' in hlsl.replace('if (exec & 1) ', ''), hlsl
    assert 'w0[%s] = asfloat(uint4(' % unpacked in hlsl, hlsl
    assert 'uav 0 tex2d table 0 0\nuavload 0\n' in manifest, manifest
    assert not any(line.startswith('srv ') for line in manifest.splitlines()), manifest
    for label in ('other texel', 'moved coordinate'):
        try:
            G.Pipeline(VS_WORDS, PACKED_WORDS[label], rw_root())
        except G.Unsupported as error:
            assert 'both read and stored to' in str(error), (label, error)
        else:
            raise AssertionError('accepted a read-write image whose store is not at the texel it read: ' + label)
    print('packed 16-bit read-write image shared; a store at another texel or after the coordinate changed is refused')


def controls():
    text = D.decode_instruction(STORE, 0)[2]
    assert text == 'image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D', text
    pipe = translate(WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'RWTexture2D<float4> w0 : register(u0);' in ps, ps
    assert '    ' + STORE_LINE in ps, ps
    assert 'register(t' not in ps, ps
    assert 'uav 0 tex2d table 0 0' in manifest and 'srv ' not in manifest and 'uavload' not in manifest, manifest
    assert manifest.startswith('x2w-manifest 2\n'), manifest
    print('store: %s manifest uav 0 tex2d table 0 0' % STORE_LINE)

    # Cache-policy bits and unorm (integer texel coordinates) do not change the written texel.
    for name, store in (('glc', patched(1, 0x20)), ('slc', patched(3, 0x02)),
                        ('dlc', patched(0, 0x80)), ('unorm', patched(1, 0x10))):
        assert D.decode_instruction(store, 0)[2].endswith(' ' + name), D.decode_instruction(store, 0)
        assert STORE_LINE in translate(with_store(store)).hlsl_ps(), name
    print('glc, slc, dlc and unorm accepted')

    # 2D-array pixel store (#2742): three address registers, RWTexture2DArray<float4>, manifest tex2da.
    array_text = D.decode_instruction(ARRAY_STORE, 0)[2]
    assert array_text == 'image_store v[4:7], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY', array_text
    array_pipe = translate(ARRAY_WORDS)
    array_ps, array_manifest = array_pipe.hlsl_ps(), array_pipe.manifest()
    assert 'RWTexture2DArray<float4> w0 : register(u0);' in array_ps, array_ps
    assert '    ' + ARRAY_STORE_LINE in array_ps, array_ps
    assert 'register(t' not in array_ps, array_ps
    assert 'uav 0 tex2da table 0 0' in array_manifest and 'srv ' not in array_manifest, array_manifest
    assert array_manifest.startswith('x2w-manifest 2\n'), array_manifest
    print('array store: %s manifest uav 0 tex2da table 0 0' % ARRAY_STORE_LINE)

    # Refusals, each by name: packed data or addresses, fault-status returns, a 128-bit descriptor,
    # a partial channel mask and other image dimensions.
    # a16: x and y packed in v0 is lowered; other dimensions would pack a second register and stay refused.
    decoded = D.decode_instruction(A16_STORE, 0)[2]
    assert decoded == 'image_store v[4:7], v0, s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D a16', decoded
    a16_ps = translate(A16_WORDS).hlsl_ps()
    assert '    ' + A16_LINE in a16_ps and STORE_LINE not in a16_ps, a16_ps
    print('a16 store: %s' % A16_LINE)
    for name, store in (('a16 3D', bytes.fromhex('100f20f000040440')), ('a16 2D array', bytes.fromhex('280f20f000040440')),
                        ('a16 1D', bytes.fromhex('000f20f000040440'))):
        print('refused %-20s %s' % (name + ':', refused(with_store(store), 'a16 only on a 2D image')))
    try:
        pipe_a16 = translate(A16_WORDS)
        pipe_a16.ps.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D', 'a16'])
    except G.Unsupported as error:
        assert 'one packed register' in str(error), error
    else:
        raise AssertionError('a16 with two address registers was accepted')
    cases = (('d16', patched(7, 0x80)), ('tfe', patched(2, 0x01)),
             ('lwe', patched(2, 0x02)), ('r128', patched(1, 0x80)), ('reserved', patched(1, 0x40)),
             ('SQ_RSRC_IMG_3D', bytes.fromhex('100f20f000040400')),
             ('SQ_RSRC_IMG_CUBE', bytes.fromhex('180f20f000040400')),
             ('SQ_RSRC_IMG_1D', bytes.fromhex('000f20f000040400')))
    for name, store in cases:
        decoded = D.decode_instruction(store, 0)[2]
        message = refused(with_store(store), 'image_store')
        print('refused %-20s %s' % (name + ':', message))
        assert decoded.startswith('image_store'), decoded
    # 2D-array requires three address registers and still refuses pixel d16.
    try:
        array_pipe.ps.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D_ARRAY'])
    except G.Unsupported as error:
        assert 'x, y and layer expected' in str(error), error
    else:
        raise AssertionError('2D-array store with two address registers was accepted')
    try:
        array_pipe.ps.op_image_store(['v[4:5]', 'v[0:2]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D_ARRAY', 'd16'])
    except G.Unsupported as error:
        assert 'modifier d16' in str(error), error
    else:
        raise AssertionError('pixel 2D-array d16 store was accepted')
    # The decoder sizes VDATA from DMASK. The typed store supplies zeros for unselected channels.
    for mask, data, values in (
            (0x7, 'v[4:6]', 'v4, v5, v6, 0u'),
            (0x1, 'v4', 'v4, 0u, 0u, 0u'),
            (0x5, 'v[4:5]', 'v4, 0u, v5, 0u')):
        store = patched(1, 0, mask)
        decoded = D.decode_instruction(store, 0)[2]
        assert decoded.startswith('image_store %s, ' % data) and 'dmask:%#x' % mask in decoded, decoded
        partial_pipe = translate(with_store(store))
        ps = partial_pipe.hlsl_ps()
        line = f'w0[uint2(v0, v1)] = asfloat(uint4({values}));'
        assert ps.count(line) == 1 and ps.count('w0[uint2(v0, v1)]') == 1, ps
        manifest = partial_pipe.manifest().splitlines()
        assert 'uav 0 tex2d table 0 0' in manifest and 'uavload 0' not in manifest, manifest
        print('partial dmask %#x: %s' % (mask, line))
    # The mask must match the data registers. An empty mask stays refused.
    for mask, data, needle in ((0x7, 'v[4:7]', 'three VGPRs expected'), (0x3, 'v4', 'two VGPRs expected'),
                               (0x0, 'v4', 'no channel selected')):
        try:
            pipe.ps.op_image_store([data, 'v[0:1]', 's[16:23]'], ['dmask:%#x' % mask, 'dim:SQ_RSRC_IMG_2D'])
        except G.Unsupported as error:
            assert needle in str(error), error
        else:
            raise AssertionError('dmask %#x with %s was accepted' % (mask, data))
        print('refused dmask %#x with %s (%s)' % (mask, data, needle))
    # Four data registers are required for the full mask: three are refused for the data, not the mask.
    try:
        pipe.ps.op_image_store(['v[4:6]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'four VGPRs expected' in str(error), error
    else:
        raise AssertionError('dmask 0xf with three data registers was accepted')
    print('refused dmask 0xf with three data registers (v[4:6])')
    # Mask text that is not an integer is a named refusal, not a ValueError.
    try:
        pipe.ps.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xz', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert str(error) == 'image_store with dmask 0xz: not an integer', error
    else:
        raise AssertionError('a malformed dmask was accepted')
    print('refused dmask 0xz (not an integer)')
    refused(with_store(bytes.fromhex('080f24f000040400')), 'image_store_mip')
    print('refused image_store_mip (not lowered)')
    try:
        pipe.vs.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'outside a pixel stage' in str(error), error
    else:
        raise AssertionError('vertex-stage image_store accepted')
    print('refused image_store outside a pixel stage')

    read_write_controls()

    # Exec not known at translation time: the store, and only the store line, is guarded by the lane's bit.
    for constant, words in ((1, EXEC_ON), (2, EXEC_OFF)):
        decoded = D.decode_instruction(cmpx_eq(constant), 0)[2]
        assert decoded == 'v_cmpx_eq_u32_e32 %d, v0' % constant, decoded
        ps = translate(words).hlsl_ps()
        assert '    exec = ((0x%08xu == v0) ? 1u : 0u);' % constant in ps, ps
        assert '    ' + GUARDED_LINE in ps, ps
        assert ps.count(STORE_LINE) == 1, ps
    # With exec known (the entry mask) the store is not guarded.
    assert GUARDED_LINE not in translate(WORDS).hlsl_ps()
    for constant, words in ((1, ARRAY_EXEC_ON), (2, ARRAY_EXEC_OFF)):
        ps = translate(words).hlsl_ps()
        assert '    exec = ((0x%08xu == v0) ? 1u : 0u);' % constant in ps, ps
        assert '    ' + ARRAY_GUARDED_LINE in ps, ps
        assert ps.count(ARRAY_STORE_LINE) == 1, ps
    assert ARRAY_GUARDED_LINE not in array_ps
    print('exec after v_cmpx: %s' % GUARDED_LINE)

    # Negative control: without the lowering the program is refused as before.
    lowering = G.Stage.op_image_store
    del G.Stage.op_image_store
    try:
        refused(WORDS, 'image_store')
    finally:
        G.Stage.op_image_store = lowering
    print('control: with op_image_store removed the program is refused')


def check_binary(llvm_mc):
    for label, src, expected in (('2d', SOURCE, WORDS), ('2d_array', ARRAY_SOURCE, ARRAY_WORDS)):
        built = assemble(llvm_mc, src)
        if built != expected:
            raise AssertionError('%s: assembled %s, recorded %s' % (label, built.hex(), expected.hex()))
    for label, fields in PACKED_FIELDS.items():
        built = assemble(llvm_mc, PACKED_SOURCE.format(**fields))
        if built != PACKED_WORDS[label]:
            raise AssertionError('packed %s: assembled %s, recorded %s' % (label, built.hex(), PACKED_WORDS[label].hex()))
    print('fixture words match llvm-mc')


def mismatch_line(stderr):
    """The texel_mismatch report: on a machine with several adapters stderr starts with adapter notices."""
    return next(line for line in stderr.splitlines() if 'texel_mismatch ' in line)


def check_draw(dxc, renderer, adapter):
    source = translate(WORDS).hlsl_ps()
    expected = ','.join(BITS)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)

        def draw(text, label, want=expected):
            ps = compile_hlsl(dxc, 'dxc', text, label, 'ps_6_0', directory)
            return subprocess.run([renderer, adapter, '--startup-image-store', str(vs), str(ps), '1,2', want],
                                  capture_output=True, text=True)

        run = draw(source, 'store')
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            sys.exit(77)
        if run.returncode:
            raise AssertionError('store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw: texel (1, 2) = %s, all other texels untouched' % expected)
        untouched = 'actual=' + ','.join(['bf800000'] * 4)
        for label, text, needle in (
                ('store deleted', source.replace(STORE_LINE, ''), 'x=1 y=2 expected=%s %s' % (expected, untouched)),
                ('x and y exchanged', source.replace('w0[uint2(v0, v1)]', 'w0[uint2(v1, v0)]'), 'x=2 y=1')):
            assert text != source, label
            run = draw(text, label.replace(' ', '_'))
            if run.returncode == 0 or 'texel_mismatch ' + needle not in run.stderr:
                raise AssertionError('control %s was not detected\n%s\n%s' % (label, run.stdout, run.stderr))
            print('control %s: fails as expected (%s)' % (label, mismatch_line(run.stderr)))

        # a16: v0 = 1 | (2 << 16) must reach the same texel (1, 2). Controls: ignoring the high half writes (1, 0);
        # exchanging the halves writes (2, 1). The readback scans row by row, so the first wrong texel is the one reported.
        a16_source = translate(A16_WORDS).hlsl_ps()
        run = draw(a16_source, 'a16_store')
        if run.returncode:
            raise AssertionError('a16 store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw a16: texel (1, 2) = %s, all other texels untouched' % expected)
        for label, text, needle in (
                ('a16 high half ignored', a16_source.replace('v0 >> 16', '0u'), 'x=1 y=0'),
                ('a16 halves exchanged', a16_source.replace('uint2(v0 & 0xffffu, v0 >> 16)', 'uint2(v0 >> 16, v0 & 0xffffu)'),
                 'x=2 y=1')):
            assert text != a16_source, label
            run = draw(text, label.replace(' ', '_'))
            if run.returncode == 0 or 'texel_mismatch ' + needle not in run.stderr:
                raise AssertionError('control %s was not detected\n%s\n%s' % (label, run.stdout, run.stderr))
            print('control %s: fails as expected (%s)' % (label, mismatch_line(run.stderr)))

        # Preserving the old channels with read-modify-write would leave -1.0 and fail the zero comparison.
        partial_source = translate(with_store(patched(1, 0, 0x5))).hlsl_ps()
        partial_want = ','.join((BITS[0], '00000000', BITS[1], '00000000'))
        run = draw(partial_source, 'partial_store', partial_want)
        if run.returncode:
            raise AssertionError('partial store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw dmask 0x5: texel (1, 2) = %s, other texels untouched' % partial_want)
        values = 'uint4(v4, 0u, v5, 0u)'
        for label, replacement in (('partial channels preserved', 'uint4(v4, 0xbf800000u, v5, 0xbf800000u)'),
                                   ('partial channels swapped', 'uint4(v5, 0u, v4, 0u)')):
            text = partial_source.replace(values, replacement)
            assert text != partial_source, label
            run = draw(text, label.replace(' ', '_'), partial_want)
            if run.returncode == 0 or 'texel_mismatch x=1 y=2' not in run.stderr:
                raise AssertionError('control %s was not detected\n%s\n%s' % (label, run.stdout, run.stderr))
            print('control %s: fails as expected (%s)' % (label, mismatch_line(run.stderr)))

        for mask in (0x1, 0x2, 0x4, 0x7, 0x9):
            channels = [c for c in range(4) if mask >> c & 1]
            want = ','.join(BITS[channels.index(c)] if c in channels else '00000000' for c in range(4))
            source_m = translate(with_store(patched(1, 0, mask))).hlsl_ps()
            label = 'dmask_%#x' % mask
            run = draw(source_m, label, want)
            if run.returncode:
                raise AssertionError('partial store %#x draw failed\n%s\n%s' % (mask, run.stdout, run.stderr))
            print('draw dmask %#x: texel (1, 2) = %s, other texels untouched' % (mask, want))
            store = next(line for line in source_m.splitlines() if 'w0[uint2(v0, v1)] = asfloat(uint4(' in line)
            full = source_m.replace(store, '    w0[uint2(v0, v1)] = asfloat(uint4(v4, v5, v6, v7));')
            run = draw(full, label + '_all_four', want)
            if run.returncode == 0 or 'texel_mismatch x=1 y=2' not in run.stderr:
                raise AssertionError('control all-four-channels %#x was not detected\n%s\n%s' % (mask, run.stdout, run.stderr))
            print('control dmask %#x written as all four channels: fails as expected (%s)' % (mask, mismatch_line(run.stderr)))

        # Exec after v_cmpx: an active lane writes the texel, an inactive one leaves it at its starting value.
        untouched_bits = ','.join(['bf800000'] * 4)
        on, off = translate(EXEC_ON).hlsl_ps(), translate(EXEC_OFF).hlsl_ps()
        for label, text, want in (('exec on', on, expected), ('exec off', off, untouched_bits)):
            run = draw(text, label.replace(' ', '_'), want)
            if run.returncode:
                raise AssertionError('%s draw failed\n%s\n%s' % (label, run.stdout, run.stderr))
            print('draw %s: texel (1, 2) = %s, all other texels untouched' % (label, want))
        # Control: the exec-off shader with the guard removed from the store writes the texel.
        unguarded = off.replace(GUARDED_LINE, STORE_LINE)
        assert unguarded != off
        run = draw(unguarded, 'exec_off_unguarded', untouched_bits)
        needle = 'texel_mismatch x=1 y=2 expected=%s actual=%s' % (untouched_bits, expected)
        if run.returncode == 0 or needle not in run.stderr:
            raise AssertionError('control exec guard removed was not detected\n%s\n%s' % (run.stdout, run.stderr))
        print('control exec guard removed: fails as expected (%s)' % mismatch_line(run.stderr))


def main():
    controls()
    packed_controls()
    if len(sys.argv) == 1:
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
