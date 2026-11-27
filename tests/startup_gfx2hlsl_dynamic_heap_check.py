"""Translate and draw pixel shaders that take a heap record byte offset in s8: a raw buffer (x4) and an image view (x8)."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_flat_param_check import G, VS_WORDS


def root_blob(compute=False, *, cbv=False):
    """Two graphics root constants, or three compute constants, a table and a root buffer."""
    params = 3 if compute else 1
    constants = 24 + params * 12
    data = struct.pack('<6I', 2, params, 24, 0, 0, 0x400)
    data += struct.pack('<3I', 1, 0, constants)
    if compute:
        table = constants + 12
        data += struct.pack('<6I', 0, 0, table, 2 if cbv else 3, 0, table + 32)
    data += struct.pack('<3I', 0, 0, 3 if compute else 2)
    if compute:
        data += struct.pack('<8I', 1, table + 8, 0, 1, 0, 0, 0, 0xffffffff)
        data += struct.pack('<3I', 1, 0, 0)
    data = data[:16] + struct.pack('<I', len(data)) + data[20:]
    part = b'RTS0' + struct.pack('<I', len(data)) + data
    return (b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(part), 1, 36) + part)


PS = bytes.fromhex(
    '80038bbe'          # s_mov_b32 s11, 0
    '050408f400000010'  # s_load_dwordx4 s[16:19], s[10:11], s8
    '7fc08cbf'          # s_waitcnt lgkmcnt(0)
    '8002087e'          # v_mov_b32 v4, 0
    '001030e004050480'  # buffer_load_dword v5, v4, s[16:19], 0 offen
    '703f8cbf'          # s_waitcnt vmcnt(0)
    '0f1800f805050505'  # exp mrt0 v5, v5, v5, v5 done vm
    '000081bf')


# s_load_dwordx8 of the 32-byte heap record at byte offset s8, then image_load of texel (0, 0) from that view.
# (The words are assembled from the listing in the comments; the vertex stage and the root fixture are shared.)
IMAGE_HEAD = ('80038bbe'          # s_mov_b32 s11, 0
              '05040cf400000010'  # s_load_dwordx8 s[16:23], s[10:11], s8
              '7fc08cbf')         # s_waitcnt lgkmcnt(0)
IMAGE_LOAD = ('8002107e'          # v_mov_b32 v8, 0
              '8002127e'          # v_mov_b32 v9, 0
              '080f00f008040400'  # image_load v[4:7], v[8:9], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D
              '703f8cbf')         # s_waitcnt vmcnt(0)
IMAGE_TAIL = '0f1800f804050607' '000081bf'  # exp mrt0 v4, v5, v6, v7 done vm; s_endpgm
IMAGE_PS = bytes.fromhex(IMAGE_HEAD + IMAGE_LOAD + IMAGE_TAIL)
# The load comes first and a uniform branch on root constant s9 skips the view's use (the record is read in one arm).
IMAGE_ARM_PS = bytes.fromhex(IMAGE_HEAD + '098006bf' '040085bf' + IMAGE_LOAD + IMAGE_TAIL)
# The load sits in the skipped arm only, so after the join the record is not defined on every path.
IMAGE_JOINED_PS = bytes.fromhex('80038bbe' '098006bf' '020085bf' '05040cf400000010' '7fc08cbf'
                                + IMAGE_LOAD + IMAGE_TAIL)
IMAGE_UNALIGNED_PS = bytes.fromhex('80038bbe' '05040cf410000010' '7fc08cbf' + IMAGE_LOAD + IMAGE_TAIL)  # offset:0x10
# An s_buffer_load through the first four words of the image record (a record read as a buffer is refused).
IMAGE_AS_BUFFER_PS = bytes.fromhex(IMAGE_HEAD + '480120f4000000fa' '7fc08cbf'  # s_buffer_load_dword s5, s[16:19], 0
                                   '0502087e' '0f1800f804040404' '000081bf')  # v_mov_b32 v4, s5; exp mrt0 v4 x4; s_endpgm

# The index register s12 is first written without a uniformity proof (it is a word of a heap record loaded in an arm
# that root constant s9 skips), then rewritten from s8 in the block after that join, which every path runs. A second
# uniform branch follows, and the record is loaded through s12 at its join: only the rewrite can define s12 there.
S9_SKIP = lambda words: '098106bf' '%02x0085bf' % words  # s_cmp_eq_u32 s9, 1; s_cbranch_scc1 +words
ARM_RECORD = '05030cf400000010' '7fc08cbf'  # s_load_dwordx8 s[12:19], s[10:11], s8; s_waitcnt lgkmcnt(0)
S12_FROM_S8 = '08038cbe'                   # s_mov_b32 s12, s8
S12_FROM_MASK = '8000847d' '6a038cbe'      # v_cmp_eq_u32 vcc_lo, 0, v0; s_mov_b32 s12, vcc_lo (a lane mask)
LOAD_S12 = '05040cf400000018' '7fc08cbf'   # s_load_dwordx8 s[16:23], s[10:11], s12; s_waitcnt lgkmcnt(0)
V8_ZERO = '8002107e'                       # v_mov_b32 v8, 0
REWRITE_PS = bytes.fromhex('80038bbe' + S9_SKIP(3) + ARM_RECORD + S12_FROM_S8 + S9_SKIP(1) + V8_ZERO + LOAD_S12
                           + IMAGE_LOAD + IMAGE_TAIL)
# Still refused: the rewrite comes before the unproven write; the rewrite is itself unproven; the rewrite sits in the
# second arm only, so the other path into the load's join still carries the unproven write.
REWRITE_REFUSED = {
    'rewrite before the write': '80038bbe' + S12_FROM_S8 + S9_SKIP(3) + ARM_RECORD + S9_SKIP(1) + V8_ZERO,
    'rewrite unproven': '80038bbe' + S9_SKIP(3) + ARM_RECORD + S12_FROM_MASK + S9_SKIP(1) + V8_ZERO,
    'rewrite in one arm': '80038bbe' + S9_SKIP(3) + ARM_RECORD + S9_SKIP(2) + S12_FROM_S8 + V8_ZERO,
}


def translate_image():
    pipe = G.Pipeline(VS_WORDS, IMAGE_PS, root_blob())
    ps = pipe.hlsl_ps()
    assert 'dynamic_heap\n' in pipe.manifest() and pipe.ps_profile() == 'ps_6_6'
    assert 'Texture2D<float4> heap_tex_' in ps and 'ResourceDescriptorHeap[NonUniformResourceIndex(' in ps
    assert '(ud1 + 0u) >> 5;' in ps and 'heap_count ?' in ps
    arm = G.Pipeline(VS_WORDS, IMAGE_ARM_PS, root_blob()).hlsl_ps()
    assert 'Texture2D<float4> heap_tex_' in arm
    for code, fragment in ((IMAGE_JOINED_PS, 'joined at a branch target'),
                           (IMAGE_UNALIGNED_PS, 'register offset through'),
                           (IMAGE_AS_BUFFER_PS, 'eight intact words')):
        try:
            G.Pipeline(VS_WORDS, code, root_blob())
        except G.Unsupported as reason:
            assert fragment in str(reason), (fragment, str(reason))
        else:
            raise AssertionError('accepted, expected a refusal naming %r' % fragment)
    return pipe, ps


def translate_rewrite():
    """The index rewritten by a proven write that every path runs after the unproven one; the refused orders."""
    pipe = G.Pipeline(VS_WORDS, REWRITE_PS, root_blob())
    ps = pipe.hlsl_ps()
    assert 'Texture2D<float4> heap_tex_' in ps and 's12 = ud1;' in ps, ps
    for label, head in REWRITE_REFUSED.items():
        try:
            G.Pipeline(VS_WORDS, bytes.fromhex(head + LOAD_S12 + IMAGE_LOAD + IMAGE_TAIL), root_blob())
        except G.Unsupported as reason:
            assert 'register offset through' in str(reason), (label, str(reason))
        else:
            raise AssertionError('accepted: ' + label)
    return pipe, ps


def translate():
    pipe = G.Pipeline(VS_WORDS, PS, root_blob())
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    assert 'dynamic_heap\n' in pipe.manifest() and pipe.ps_profile() == 'ps_6_6'
    assert '>> 5;' in ps and 'ResourceDescriptorHeap[' in ps and 'index >= heap_count' in ps
    return pipe, vs, ps


def compile_shader(dxc, directory, profile, source):
    hlsl = directory / (profile + '.hlsl')
    cso = hlsl.with_suffix('.cso')
    hlsl.write_text(source)
    subprocess.run([dxc, '-Gis', '-T', profile, '-E', 'main', '-Fo', cso, hlsl], check=True)
    return cso


def mutate(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def draw(offscreen, vs, ps, offset, reference, count, negative=False, heap_count=2):
    # Physical descriptor 2 is a poison view outside the buffer heap of the first cases (heap_count 2); the image cases
    # open the whole five-slot heap.
    entry = [0] * count + [heap_count]
    entry[1] = offset
    as_hex = lambda words: ','.join(f'{word:08x}' for word in words)
    expected = reference if isinstance(reference, list) else [reference] * 4
    result = subprocess.run([offscreen, 'warp', '--startup-dynamic-heap', vs, ps, as_hex(entry),
                             as_hex(expected)], capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == int(negative), (result.stdout, result.stderr)
    if negative:
        assert 'startup_entry_words mismatch' in result.stderr, result.stderr


def main(offscreen, dxc):
    pipe, vs_source, ps_source = translate()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', vs_source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        count = len(pipe.uds)
        # The three fixture buffers contain 1.25, 2.5 and 4.0 respectively.
        draw(offscreen, vs, ps, 0, 0x3fa00000, count)
        draw(offscreen, vs, ps, 32, 0x40200000, count)
        draw(offscreen, vs, ps, 64, 0, count)
        wrong_index = compile_shader(dxc, directory, 'ps_6_6', mutate(ps_source, '>> 5;', '>> 6;'))
        draw(offscreen, vs, wrong_index, 32, 0x40200000, count, negative=True)
        unbounded = compile_shader(dxc, directory, 'ps_6_6',
                                   mutate(ps_source, 'index >= heap_count', 'index > heap_count'))
        draw(offscreen, vs, unbounded, 64, 0, count, negative=True)

        # Slots 3 and 4 hold 1x1 float textures: (1.25, 2.5, 4, 8) and (16, 32, 64, 128).
        _, image_source = translate_image()
        first, second = [0x3fa00000, 0x40200000, 0x40800000, 0x41000000], [0x41800000, 0x42000000, 0x42800000, 0x43000000]
        image = compile_shader(dxc, directory, 'ps_6_6', image_source)
        draw(offscreen, vs, image, 96, first, count, heap_count=5)
        draw(offscreen, vs, image, 128, second, count, heap_count=5)
        shifted = compile_shader(dxc, directory, 'ps_6_6', mutate(image_source, '(ud1 + 0u) >> 5;', '(ud1 + 32u) >> 5;'))
        draw(offscreen, vs, shifted, 96, first, count, negative=True, heap_count=5)

        # The index rewritten after the arm's record load (s9 is 0, so the arm runs): the view follows the rewrite.
        rewrite_pipe, rewrite_source = translate_rewrite()
        rewrite_count = len(rewrite_pipe.uds)
        rewrite = compile_shader(dxc, directory, 'ps_6_6', rewrite_source)
        draw(offscreen, vs, rewrite, 96, first, rewrite_count, heap_count=5)
        draw(offscreen, vs, rewrite, 128, second, rewrite_count, heap_count=5)
        stale = compile_shader(dxc, directory, 'ps_6_6', mutate(rewrite_source, 's12 = ud1;', 's12 = ud1 + 32u;'))
        draw(offscreen, vs, stale, 96, first, rewrite_count, negative=True, heap_count=5)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        translate()
        translate_image()
        translate_rewrite()
    else:
        main(*sys.argv[1:])
    print('dynamic graphics heap: readback and controls matched' if len(sys.argv) > 1
          else 'dynamic graphics heap: register offset translated')
