"""Translate and draw raw-buffer and 2D texture resource-heap fixtures.

Without arguments, run the Python translation checks. With the offscreen test
executable and DXC, also compare WARP readback and run the GPU mutants.
"""
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


# Source-built fixtures. The binary mode assembles these strings for gfx1013 and checks the embedded words.
TEXTURE_U_SOURCE = """\
s_mov_b32 s11, 0
s_mov_b32 s36, 0
s_mov_b32 s37, 0
s_mov_b32 s38, 0
s_mov_b32 s39, 0
v_mov_b32 v6, 0.5
v_mov_b32 v7, 0.5
s_cmp_lt_u32 s8, 2
s_cbranch_scc1 low
s_and_b32 s14, s8, 7
s_lshl_b32 s14, s14, 5
s_load_dwordx8 s[24:31], s[10:11], s14
s_waitcnt lgkmcnt(0)
image_sample_lz v[12:15], v[6:7], s[24:31], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D
s_branch done
low:
s_add_u32 s14, s8, 0
s_lshl_b32 s14, s14, 5
s_load_dwordx8 s[24:31], s[10:11], s14
s_waitcnt lgkmcnt(0)
image_sample_lz v[12:15], v[6:7], s[24:31], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D
done:
s_waitcnt vmcnt(0)
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
"""
TEXTURE_U = bytes.fromhex(
    '80038bbe8003a4be8003a5be8003a6be8003a7bef0020c7ef0020e7e08820abf'
    '080085bf08870e870e850e8f05060cf40000001c7fc08cbf080f9cf0060c2601'
    '070082bf08800e800e850e8f05060cf40000001c7fc08cbf080f9cf0060c2601'
    '703f8cbf0f1800f80c0d0e0f000081bf'
)

TEXTURE_L_SOURCE = """\
s_mov_b32 s11, 0
s_mov_b32 s36, 0
s_mov_b32 s37, 0
s_mov_b32 s38, 0
s_mov_b32 s39, 0
v_and_b32 v5, 3, v0
v_mov_b32 v6, 0.5
v_mov_b32 v7, 0.5
s_mov_b64 s[16:17], exec
s_mov_b64 s[18:19], exec
again:
v_readfirstlane_b32 s14, v5
v_cmpx_eq_u32 s14, v5
s_lshl_b32 s14, s14, 5
s_load_dwordx8 s[24:31], s[10:11], s14
s_waitcnt lgkmcnt(0)
image_sample_lz v[12:15], v[6:7], s[24:31], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
s_andn2_wrexec_b64 s[18:19], s[18:19]
s_cbranch_scc1 again
s_mov_b64 exec, s[16:17]
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
"""
TEXTURE_L = bytes.fromhex(
    '80038bbe8003a4be8003a5be8003a6be8003a7be83000a36f0020c7ef0020e7e'
    '7e0490be7e0492be05051c7e0e0aa47d0e850e8f05060cf40000001c7fc08cbf'
    '080f9cf0060c2601703f8cbf123a92bef5ff85bf1004febe0f1800f80c0d0e0f'
    '000081bf'
)

def texture_pipe(code, edits=(), *, root=None, vertex=False):
    decoded = []
    for off, size, name, asm in G.decode(code):
        for old, new in edits:
            if asm == old or asm.startswith(old + ' '):
                asm = new
                name = new.split()[0]
                break
        decoded.append((off, size, name, asm))
    if root is None:
        root = root_blob()
    if not edits and not vertex:
        return G.Pipeline(VS_WORDS, code, root, ps_hw_inputs=(0x8000, 0x8000))
    refusals = []
    if vertex:
        pipe = G.Pipeline(code, None, root, refusals=refusals, decoded={'vs': decoded})
    else:
        pipe = G.Pipeline(VS_WORDS, code, root, refusals=refusals, decoded={'ps': decoded},
                          ps_hw_inputs=(0x8000, 0x8000))
    if refusals:
        raise G.Unsupported(refusals[0][0])
    return pipe


def translate_textures():
    uniform, lane = (texture_pipe(code) for code in (TEXTURE_U, TEXTURE_L))
    for pipe in (uniform, lane):
        assert 'dynamic_heap texture2d\n' in pipe.manifest() and pipe.ps_profile() == 'ps_6_6'
        assert 'ResourceDescriptorHeap[NonUniformResourceIndex(min(' in pipe.hlsl_ps()
        assert '.SampleLevel(' in pipe.hlsl_ps()
    assert uniform.hlsl_ps().count(' >> 5;') == 2
    assert 'uint heap_index_34 = v5;' in lane.hlsl_ps()
    assert 'while (' not in lane.hlsl_ps()
    return uniform, lane


def check_texture_index_sources():
    # Root CBV in s[8:9] in place of the two root constants. A scalar read from it
    # supplies both wave-uniform arms with the index.
    cbv_root = bytearray(root_blob())
    struct.pack_into('<I', cbv_root, 44 + 24, 2)
    cbv = texture_pipe(TEXTURE_U, [
        ('v_mov_b32_e32 v6, 0.5', 's_load_dword s12, s[8:9], 0x8'),
        ('s_cmp_lt_u32', 's_cmp_lt_u32 s12, 2'),
        ('s_and_b32', 's_and_b32 s14, s12, 7'),
        ('s_add_u32', 's_add_u32 s14, s12, 0'),
    ], root=cbv_root)
    assert 'srv 0 buf root 0' in cbv.manifest()
    assert 'r0.Load(8)' in cbv.hlsl_ps()
    assert cbv.hlsl_ps().count('ResourceDescriptorHeap[') == 2

    vertex = texture_pipe(TEXTURE_L, [
        ('exp mrt0 v12, v13, v14, v15 done vm', 'exp pos0 v12, v13, v14, v15 done'),
    ], vertex=True)
    assert 'dynamic_heap texture2d' in vertex.manifest()
    assert '.SampleLevel(' in vertex.hlsl_vs()


def check_binary(llvm_mc, llvm_objcopy):
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / 'fixture.o'
        binary = obj.with_suffix('.bin')
        for source, expected in ((TEXTURE_U_SOURCE, TEXTURE_U), (TEXTURE_L_SOURCE, TEXTURE_L)):
            subprocess.run([llvm_mc, '--triple=amdgcn-amd-amdhsa', '--mcpu=gfx1013', '-filetype=obj',
                            '-o', obj], input=source, text=True, check=True)
            subprocess.run([llvm_objcopy, '-O', 'binary', '--only-section=.text', obj, binary], check=True)
            assert binary.read_bytes() == expected


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


def draw(offscreen, vs, ps, offset, reference, count, negative=False):
    entry = [0] * count + [2]  # physical descriptor 2 is a poison view outside the logical heap
    entry[1] = offset
    as_hex = lambda words: ','.join(f'{word:08x}' for word in words)
    result = subprocess.run([offscreen, 'warp', '--startup-dynamic-heap', vs, ps, as_hex(entry),
                             as_hex([reference] * 4)], capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == int(negative), (result.stdout, result.stderr)
    if negative:
        assert 'startup_entry_words mismatch' in result.stderr, result.stderr


def draw_textures(offscreen, vs, ps, pipe, index, colors, *, negative=False):
    entry = [0] * len(pipe.uds) + [4]
    entry[pipe.uds.index(('rootconst', 0, 0))] = index

    def as_hex(words):
        return ','.join(f'{word:08x}' for word in words)

    result = subprocess.run([offscreen, 'warp', '--startup-dynamic-textures', vs, ps, as_hex(entry),
                             as_hex(G.fbits(c) for color in colors for c in color)],
                            capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == int(negative), (result.stdout, result.stderr)
    if negative:
        assert 'startup_entry_words mismatch' in result.stderr, result.stderr


def test_texture_draws(offscreen, dxc, directory):
    uniform, lane = translate_textures()
    vs = compile_shader(dxc, directory, 'vs_6_6', uniform.hlsl_vs())
    colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
    zero = [(0, 0, 0, 0)]
    ps = compile_shader(dxc, directory, 'ps_6_6', uniform.hlsl_ps())
    for index, color in enumerate(colors):
        draw_textures(offscreen, vs, ps, uniform, index, [color])
    for index in (4, 5):
        draw_textures(offscreen, vs, ps, uniform, index, zero)
    bad_bounds = compile_shader(dxc, directory, 'ps_6_6',
                                uniform.hlsl_ps().replace(', heap_count))', ', heap_count - 1u))'))
    draw_textures(offscreen, vs, bad_bounds, uniform, 5, zero, negative=True)
    wrong_uniform = compile_shader(dxc, directory, 'ps_6_6',
                                   uniform.hlsl_ps().replace('s14 >> 5;', '(s14 >> 5) + 1u;'))
    draw_textures(offscreen, vs, wrong_uniform, uniform, 1, [colors[1]], negative=True)
    ps = compile_shader(dxc, directory, 'ps_6_6', lane.hlsl_ps())
    draw_textures(offscreen, vs, ps, lane, 0, colors)
    wrong_lane = compile_shader(dxc, directory, 'ps_6_6', mutate(lane.hlsl_ps(), '= v5;', '= v5 + 1u;'))
    draw_textures(offscreen, vs, wrong_lane, lane, 0, colors, negative=True)
    implicit = texture_pipe(TEXTURE_L, [('image_sample_lz',
        'image_sample v[12:15], v[6:7], s[24:31], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D')])
    ps = compile_shader(dxc, directory, 'ps_6_6', implicit.hlsl_ps())
    draw_textures(offscreen, vs, ps, implicit, 0, colors)
    wrong_implicit = compile_shader(dxc, directory, 'ps_6_6', mutate(implicit.hlsl_ps(), '= v5;', '= v5 + 1u;'))
    draw_textures(offscreen, vs, wrong_implicit, implicit, 0, colors, negative=True)
    load = texture_pipe(TEXTURE_L, [('image_sample_lz',
        'image_load v[12:15], v[6:7], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D'),
        ('v_mov_b32_e32 v6, 0.5', 'v_mov_b32_e32 v6, 0'),
        ('v_mov_b32_e32 v7, 0.5', 'v_mov_b32_e32 v7, 0')])
    ps = compile_shader(dxc, directory, 'ps_6_6', load.hlsl_ps())
    draw_textures(offscreen, vs, ps, load, 0, colors)
    wrong_load = compile_shader(dxc, directory, 'ps_6_6', mutate(load.hlsl_ps(), '= v5;', '= v5 + 1u;'))
    draw_textures(offscreen, vs, wrong_load, load, 0, colors, negative=True)


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
        test_texture_draws(offscreen, dxc, directory)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        translate()
        translate_textures()
        check_texture_index_sources()
    elif sys.argv[1] == 'binary':
        check_binary(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
        print('dynamic graphics heap: readback and controls matched')
    print('dynamic graphics heap: PASS')
