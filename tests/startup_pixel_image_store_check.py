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


# d16: the data is four binary16 values packed in two VGPRs (R, G in the first, B, A in the second, low half first), stored
# with image_store v[4:5] ... d16. v4 = 0xc0003c00 (1.0, -2.0), v5 = 0x7bff3800 (0.5, 65504.0). Every half is exact in
# binary32 and the four values differ, so a swapped half, a binary32 reading of the words or a missing conversion all change
# the texel.
D16_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, 1
v_mov_b32 v1, 2
v_mov_b32 v4, 0xc0003c00
v_mov_b32 v5, 0x7bff3800
s_waitcnt lgkmcnt(0)
image_store v[4:5], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v4, v5 done vm
s_endpgm
"""
D16_PREFIX = bytes.fromhex('04040cf4000000fa' '8102007e' '8202027e' 'ff02087e003c00c0' 'ff020a7e0038ff7b' '7fc08cbf')
D16_STORE = bytes.fromhex('080f20f000040480')
D16_SUFFIX = bytes.fromhex('703f8cbf' '0f1800f804050405' '000081bf')
D16_WORDS = D16_PREFIX + D16_STORE + D16_SUFFIX
D16_VALUES = (1.0, -2.0, 0.5, 65504.0)
D16_BITS = ['%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in D16_VALUES]
D16_LINE = ('w0[uint2(v0, v1)] = float4(f16tof32(v4 & 0xffffu), f16tof32(v4 >> 16), '
            'f16tof32(v5 & 0xffffu), f16tof32(v5 >> 16));')


def cmpx_eq(constant):
    """v_cmpx_eq_u32_e32 <constant>, v0: VOPC (bits 31:25 = 0x3e), opcode 0xd2, VSRC1 v0, SRC0 an inline
    integer 1..64 (0x80 + n). In gfx10 the VOPC v_cmpx form writes only exec."""
    return struct.pack('<I', 0x7c000000 | (0xd2 << 17) | (0 << 9) | (0x80 + constant))


EXEC_ON = PREFIX + cmpx_eq(1) + STORE + SUFFIX   # v0 is 1: the lane stays active
EXEC_OFF = PREFIX + cmpx_eq(2) + STORE + SUFFIX  # v0 is 1: every lane is turned off before the store
ARRAY_EXEC_ON = PREFIX + mov_v2_inline(1) + cmpx_eq(1) + ARRAY_STORE + SUFFIX
ARRAY_EXEC_OFF = PREFIX + mov_v2_inline(1) + cmpx_eq(2) + ARRAY_STORE + SUFFIX


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

    # d16: the two data registers hold four binary16 values, widened to binary32 for the float UAV.
    decoded = D.decode_instruction(D16_STORE, 0)[2]
    assert decoded == 'image_store v[4:5], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D d16', decoded
    pipe16 = translate(D16_WORDS)
    ps16 = pipe16.hlsl_ps()
    assert '    ' + D16_LINE in ps16 and STORE_LINE not in ps16, ps16
    assert 'uav 0 tex2d table 0 0' in pipe16.manifest(), pipe16.manifest()
    print('d16 store: %s' % D16_LINE)
    # d16 needs exactly two data registers (a binary32 store needs four), and the other flags stay refused with it.
    for data, needle in (('v[4:7]', 'two VGPRs expected'), ('v4', 'two VGPRs expected')):
        try:
            pipe16.ps.op_image_store([data, 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D', 'd16'])
        except G.Unsupported as error:
            assert needle in str(error), error
        else:
            raise AssertionError('d16 with data %s was accepted' % data)
    try:
        pipe.ps.op_image_store(['v[4:5]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'four VGPRs expected' in str(error), error
    else:
        raise AssertionError('two data registers without d16 were accepted')
    refused(D16_PREFIX + D16_STORE[:7] + bytes([D16_STORE[7] | 0x40]) + D16_SUFFIX, 'image_store')
    print('d16 needs two data registers; d16 with a16 refused')

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
    cases = (('tfe', patched(2, 0x01)),
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
    # A partial channel mask, encoded: the decoder sizes VDATA from the mask (v[4:6], v4). The store reads the texel
    # back, replaces only the selected channels in order from consecutive VDATA registers, and writes it again.
    for mask, data, line in (
            (0x7, 'v[4:6]', 't = w0[uint2(v0, v1)]; t.x = asfloat(v4); t.y = asfloat(v5); t.z = asfloat(v6); '
                            'w0[uint2(v0, v1)] = t;'),
            (0x1, 'v4', 't = w0[uint2(v0, v1)]; t.x = asfloat(v4); w0[uint2(v0, v1)] = t;'),
            (0x5, 'v[4:5]', 't = w0[uint2(v0, v1)]; t.x = asfloat(v4); t.z = asfloat(v5); w0[uint2(v0, v1)] = t;')):
        store = patched(1, 0, mask)
        decoded = D.decode_instruction(store, 0)[2]
        assert decoded.startswith('image_store %s, ' % data) and 'dmask:%#x' % mask in decoded, decoded
        partial_pipe = translate(with_store(store))
        ps = partial_pipe.hlsl_ps()
        assert '    { %s }' % line in ps and STORE_LINE not in ps, ps
        # The read-back needs typed UAV load, so the manifest says so right after the uav line.
        assert 'uav 0 tex2d table 0 0\nuavload 0\n' in partial_pipe.manifest(), partial_pipe.manifest()
        print('partial dmask %#x: { %s }' % (mask, line))
    # The mask must match the data registers, and an empty mask or a 16-bit partial store stays refused.
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

    # Reading and storing the same image in one program would need a read-write view: refused.
    load = bytes.fromhex('080f00f000080400')  # image_load v[8:11], v[0:1], s[16:23] dmask:0xf dim:2D
    assert D.decode_instruction(load, 0)[2].startswith('image_load v[8:11]'), D.decode_instruction(load, 0)
    refused(PREFIX + load + STORE + SUFFIX, 'both read and stored')
    assert translate(PREFIX + load + SUFFIX).manifest().startswith('x2w-manifest 1\n')
    print('refused a load and a store of the same image')

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
    for label, src, expected in (('2d', SOURCE, WORDS), ('2d_array', ARRAY_SOURCE, ARRAY_WORDS),
                                 ('d16', D16_SOURCE, D16_WORDS)):
        built = assemble(llvm_mc, src)
        if built != expected:
            raise AssertionError('%s: assembled %s, recorded %s' % (label, built.hex(), expected.hex()))
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

        # d16: four binary16 values from two VGPRs reach the float texel; controls with a wrong half order, a binary32
        # reading of the words and the store deleted must fail.
        d16 = translate(D16_WORDS).hlsl_ps()
        d16_expected = ','.join(D16_BITS)
        run = draw(d16, 'd16_store', d16_expected)
        sys.stdout.write(run.stdout)
        if run.returncode:
            raise AssertionError('d16 store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw d16: texel (1, 2) = %s, all other texels untouched' % d16_expected)
        swapped = ('w0[uint2(v0, v1)] = float4(f16tof32(v4 >> 16), f16tof32(v4 & 0xffffu), '
                   'f16tof32(v5 >> 16), f16tof32(v5 & 0xffffu));')
        for label, text in (('d16 halves swapped', d16.replace(D16_LINE, swapped)),
                            ('d16 read as binary32', d16.replace(D16_LINE, 'w0[uint2(v0, v1)] = asfloat(uint4(v4, v5, v4, v5));')),
                            ('d16 store deleted', d16.replace(D16_LINE, ''))):
            assert text != d16, label
            run = draw(text, label.replace(' ', '_'), d16_expected)
            if run.returncode == 0 or 'texel_mismatch x=1 y=2' not in run.stderr:
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

        # Partial dmask 0x5 stores v4 (1.0) to x and v5 (2.0) to z; y and w keep the starting value -1.0. Controls:
        # without the read-back the other channels become 0; with the data in the wrong channels x and z differ.
        partial_source = translate(with_store(patched(1, 0, 0x5))).hlsl_ps()
        partial_want = ','.join((BITS[0], 'bf800000', BITS[1], 'bf800000'))
        run = draw(partial_source, 'partial_store', partial_want)
        if run.returncode:
            raise AssertionError('partial store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw dmask 0x5: texel (1, 2) = %s, other channels and texels untouched' % partial_want)
        readback = 't = w0[uint2(v0, v1)];'
        for label, text in (('partial no read-back', partial_source.replace(readback, 't = 0;')),
                            ('partial channels swapped', partial_source.replace('t.x = asfloat(v4); t.z = asfloat(v5);',
                                                                               't.z = asfloat(v4); t.x = asfloat(v5);'))):
            assert text != partial_source, label
            run = draw(text, label.replace(' ', '_'), partial_want)
            if run.returncode == 0 or 'texel_mismatch x=1 y=2' not in run.stderr:
                raise AssertionError('control %s was not detected\n%s\n%s' % (label, run.stdout, run.stderr))
            print('control %s: fails as expected (%s)' % (label, mismatch_line(run.stderr)))

        # Every partial mask the lowering accepts: the k-th selected channel takes the k-th data register (v4..), the
        # others keep the starting value -1.0. Control: a store of all four channels from v4..v7 overwrites them.
        for mask in (0x1, 0x2, 0x4, 0x7, 0x9):
            channels = [c for c in range(4) if mask >> c & 1]
            want = ','.join(BITS[channels.index(c)] if c in channels else 'bf800000' for c in range(4))
            source_m = translate(with_store(patched(1, 0, mask))).hlsl_ps()
            label = 'dmask_%#x' % mask
            run = draw(source_m, label, want)
            if run.returncode:
                raise AssertionError('partial store %#x draw failed\n%s\n%s' % (mask, run.stdout, run.stderr))
            print('draw dmask %#x: texel (1, 2) = %s, other channels and texels untouched' % (mask, want))
            start = source_m.index('{ t = w0[')
            end = source_m.index('= t; }', start) + len('= t; }')
            full = source_m[:start] + 'w0[uint2(v0, v1)] = asfloat(uint4(v4, v5, v6, v7));' + source_m[end:]
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
