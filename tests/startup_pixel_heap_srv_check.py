"""A translated pixel program that samples the resource-heap record a root constant indexes, drawn through d3d12_xs.dll.

  (no arguments)     translation controls, pure Python
  binary <llvm-mc>   reassemble the fixtures and compare them with the recorded words
  <startup_stream_consumer> <warp|hardware> <fxc|dxc> <dxc> <fxc>   the draws

The program loads an image record with s_load_dwordx8 from the resource heap at index << 5 (root flag 0x400), the index
being root constant word 0 of parameter 1, and writes texel (0, 0) to the target with the key in green. The consumer
(X2W_TEST_JOINED_TABLE + X2W_TEST_HEAP_SRV, tests/startup_address_srv.cpp) publishes two views of two different textures
at heap records 0 and 1 and draws a one-pixel-wide strip per key, twice with the two textures swapped. Keys 0 and 1 must
show the texture at their record, so the second pass must show the other texture; key 2 names a record that is not in the
title's view heap and key 3 a second record past it: those draws are refused by name and leave the cleared target. A draw that
(Word case: key 3 is root constant 0x40000001, whose byte offset x 4 wraps to 4 at 32 bits as the shader computes it, so it is drawn with record 0 where a 64-bit offset would be refused.) ignored the index (always record 0), did not re-read it, or guessed a texture for a missing record fails one of these.
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
from startup_flat_param_check import G, VS_WORDS, assemble  # noqa: E402
from startup_stream_draw_check import container, vertex  # noqa: E402
from startup_mrt_check import compile_shader  # noqa: E402

HEAP_FLAG = 0x400
CLEARED = bytes((0, 0, 255, 255))  # the target the consumer clears to, as R8G8B8A8
WORDS = {
    's_mov_b32 s11, 0': '80038bbe',
    's_lshl_b32 s24, s9, 5': '0985188f',
    's_lshl_b32 s24, s9, 6': '0986188f',
    's_load_dwordx8 s[16:23], s[10:11], s24': '05040cf400000030',
    's_load_dwordx4 s[16:19], s[10:11], s24': '050408f400000030',
    'v_mov_b32 v0, 0': '8002007e',
    'v_mov_b32 v1, 0': '8002027e',
    's_waitcnt lgkmcnt(0)': '7fc08cbf',
    'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D': '080f00f000040400',
    's_waitcnt vmcnt(0)': '703f8cbf',
    'v_cvt_f32_u32_e32 v5, s9': '090c0a7e',
    'v_mul_f32_e32 v5, 0x3e000000, v5': 'ff0a0a100000003e',
    'exp mrt0 v4, v5, v6, v7 done vm': '0f1800f804050607',
    's_endpgm': '000081bf',
    's_mov_b32 s25, 0': '800399be',
    's_mov_b32 s24, s8': '080398be',
    's_load_dwordx4 s[12:15], s[24:25], 0x40': '0c0308f4400000fa',
    's_mul_i32 s28, s9, 4': '09841c93',
    's_buffer_load_dword s26, s[12:15], s28': '860620f400000038',
    's_lshl_b32 s27, s26, 5': '1a851b8f',
    's_load_dwordx8 s[16:23], s[10:11], s27': '05040cf400000036',
}
# s8 is the (unused) descriptor table, s9 the root constant, s10 the 32-bit heap pointer, s11 its high word.
PROGRAM = """s_mov_b32 s11, 0
s_lshl_b32 s24, s9, 5
s_load_dwordx8 s[16:23], s[10:11], s24
v_mov_b32 v0, 0
v_mov_b32 v1, 0
s_waitcnt lgkmcnt(0)
image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_cvt_f32_u32_e32 v5, s9
v_mul_f32_e32 v5, 0x3e000000, v5
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
# The same program with the index read from a word of a buffer: the descriptor at table byte 64 (s[12:15]), the word at
# byte key * 4 (root constant word * 4). The consumer's buffer holds records 1, 0, 5, 7 for keys 0..3.
PROGRAM_WORD = PROGRAM.replace('''s_lshl_b32 s24, s9, 5
s_load_dwordx8 s[16:23], s[10:11], s24
''', '''s_mov_b32 s25, 0
s_mov_b32 s24, s8
s_load_dwordx4 s[12:15], s[24:25], 0x40
s_mul_i32 s28, s9, 4
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s26, s[12:15], s28
s_waitcnt lgkmcnt(0)
s_lshl_b32 s27, s26, 5
s_load_dwordx8 s[16:23], s[10:11], s27
''')
# case: (program, manifest line, consumer mode, record each key names)
CASES = {
    'rootconst': (PROGRAM, 'srv 0 tex2d heap rootconst 1 0', '1', (0, 1), False),
    'word': (PROGRAM_WORD, 'srv 1 tex2d heap word table 0 64 at 0 rootconst 1 0 times 4', 'word', (1, 0), True),
}
REFUSED = {
    'shift by 6': (PROGRAM.replace('s9, 5', 's9, 6'), 'not a root constant shifted left by 5'),
    'a buffer record in a pixel stage': (PROGRAM.replace('s_load_dwordx8 s[16:23]', 's_load_dwordx4 s[16:19]'), 'of 4 dwords'),
}


def root(flags=HEAP_FLAG):
    """RTS0: parameter 0 a descriptor table, parameter 1 one root constant, with the given root flags."""
    head = 24 + 12 * 2
    blob = struct.pack('<3I', 0, 0, head) + struct.pack('<3I', 1, 0, head + 8)
    data = struct.pack('<2I', 0, head + 8) + struct.pack('<3I', 0, 0, 1)
    rts0 = struct.pack('<6I', 2, 2, 24, 0, head + len(data), flags) + blob + data
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part


def encode(source):
    return bytes.fromhex(''.join(WORDS[line] for line in source.splitlines()))


def manifest(source):
    return G.Pipeline(VS_WORDS, encode(source), root()).manifest().splitlines()


def check_host():
    for name, (program, line, _, _, _) in CASES.items():
        lines = manifest(program)
        assert line in lines, (name, lines)
    for name, (source, reason) in REFUSED.items():
        try:
            manifest(source)
        except G.Unsupported as error:
            assert reason in str(error), (name, str(error))
        else:
            raise AssertionError(name + ' was accepted')
    try:
        G.Pipeline(VS_WORDS, encode(PROGRAM), root(0)).manifest()
    except G.Unsupported:
        pass
    else:
        raise AssertionError('a root without the heap flag translated the heap load')
    print('startup_pixel_heap_srv: host controls ok (1 binding, %d refusals, no-flag refusal)' % len(REFUSED))


def expected(swap, records, wrap=False):
    """One pass: four strips (keys 0..3), the texture at the record a key names (^ swap) for keys 0 and 1, else the cleared target."""
    column = []
    for key in range(4):
        if key < 2:
            column.append(bytes((16 + 128 * (records[key] ^ swap), (0, 32)[key], 0, 255)))
        else:
            column.append(bytes((16 + 128 * (records[1] ^ swap), 255, 0, 255)) if wrap and key == 3 else CLEARED)
    return column


def draw(folder, consumer, backend, kind, compiler, case='rootconst'):
    program, manifest_line, mode, records, wrap = CASES[case]
    cache = folder / 'cache'
    cache.mkdir()
    vs, ps, signature, pixels = (folder / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(container(0, [word for word, in struct.iter_unpack('<I', encode(program))]))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_TEST_JOINED_TABLE='1', X2W_TEST_HEAP_SRV=mode, X2W_SHADER_CACHE=str(cache),
               X2W_FXC=str(compiler) if kind == 'fxc' else '')
    env['PATH'] = str(Path(compiler).resolve().parent) + os.pathsep + env['PATH']
    cmd = [consumer, vs, ps, signature]
    first = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if first.returncode == 77:
        raise SystemExit(77)
    assert first.returncode == 0, first.stdout + first.stderr
    subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], env=env, check=True)
    # revision.txt is the cache revision the runtime writes with its first request (#2815), not a translation.
    text_file, = (path for path in cache.glob('*.txt') if path.name != 'revision.txt')
    text = text_file.read_text(encoding='ascii')
    assert text.splitlines().count(manifest_line) == 1, text
    for stage in ('vs', 'ps'):
        compile_shader(compiler, kind, stage, text_file.with_suffix(f'.{stage}.hlsl'),
                       text_file.with_suffix(f'.{stage}.cso'), folder / f'{stage}.asm')
    result = subprocess.run(cmd + [backend, pixels], env=env, capture_output=True, text=True)
    if result.returncode == 77:
        raise SystemExit(77)
    assert result.returncode == 0, result.stdout + result.stderr
    data = pixels.read_bytes()
    want = b''.join(bytes(pixel) for swap in (0, 1) for pixel in expected(swap, records, wrap))
    columns = [data[4 * x:4 * x + 4] for x in range(8)]
    rows = [data[32 * y:32 * y + 32] for y in range(4)]
    assert len(set(rows)) == 1, 'rows differ: ' + data.hex()
    assert b''.join(columns) == want, (b''.join(columns).hex(), want.hex())
    # Two passes, two refused keys per pass, each refused by name and never drawn with a guessed texture.
    refused = result.stderr.count('heap record refused (record is outside the list')
    other = result.stderr.count('heap record refused (record is not a shader-resource view')
    assert refused + other == (2 if wrap else 4), (refused, other, result.stderr[-2000:])
    print(f'{kind} {backend} {case}: heap records 0 and 1 drawn, swapped views followed, records refused by name' + (', a wrapped byte offset read as the shader does' if wrap else '') + '')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_host()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        for line in PROGRAM_WORD.splitlines() + PROGRAM.splitlines() + ['s_lshl_b32 s24, s9, 6', 's_load_dwordx4 s[16:19], s[10:11], s24']:
            assert assemble(sys.argv[2], line + '\n') == bytes.fromhex(WORDS[line]), line
        print('heap srv fixture instructions match gfx1013 assembly')
    elif len(sys.argv) in (6, 7):
        consumer, backend, kind, dxc, fxc = sys.argv[1:6]
        with tempfile.TemporaryDirectory() as temporary:
            draw(Path(temporary), consumer, backend, kind, fxc if kind == 'fxc' else dxc,
                 sys.argv[6] if len(sys.argv) == 7 else 'rootconst')
    else:
        raise SystemExit(__doc__)
