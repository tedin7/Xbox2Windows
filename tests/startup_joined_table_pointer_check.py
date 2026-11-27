"""Check 32-bit graphics table pointers at forward joins and read both bound textures (#2463).

Run with no arguments for the host controls, or with binary <llvm-mc> to reassemble every source.
The draw arguments are <startup_stream_consumer> <warp|hardware> <fxc|dxc> <dxc> <fxc> [choice].
"""
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_flat_param_check import G, PS_WORDS, VS_WORDS, assemble
from startup_gfx2hlsl_join_symbols_check import root
from startup_stream_draw_check import container, vertex
from startup_mrt_check import compile_shader


# LLVM 15 gfx1013 encodings. Each program is also assembled in full by the binary check.
WORDS = {
    'buffer_load_dword v6, off, s[16:19], 0': '000030e000060480',
    'exp mrt0 v4, v5, v6, v7 done vm': '0f1800f804050607',
    'exp param0 v6, v6, v6, v6': '0f0200f806060606',
    'exp pos0 v6, v6, v6, v6 done': 'cf0800f806060606',
    'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D': '080f00f000040400',
    's_add_u32 s24, s24, 32': '18a01880',
    's_add_u32 s24, s0, 16': '00901880',
    's_add_u32 s24, s8, 32': '08a01880',
    's_add_u32 s24, s8, 64': '08c01880',
    's_add_u32 s26, s26, 0xfffff': '1aff1a80ffff0f00',
    's_add_u32 s26, s26, 2': '1a821a80',
    's_add_u32 s26, s26, 32': '1aa01a80',
    's_add_u32 s26, s26, s9': '1a091a80',
    's_cbranch_scc1 1': '010085bf',
    's_cbranch_scc1 65531': 'fbff85bf',
    's_cbranch_scc1 65533': 'fdff85bf',
    's_cmp_eq_u32 s9, 0': '098006bf',
    's_cmp_eq_u32 s9, 1': '098106bf',
    's_cmp_lg_u32 s30, 0': '1e8007bf',
    's_endpgm': '000081bf',
    's_load_dwordx4 s[16:19], s[0:1], 0x20': '000408f4200000fa',
    's_load_dwordx4 s[16:19], s[24:25], 0x10': '0c0408f4100000fa',
    's_load_dwordx8 s[16:23], s[24:25], 0x0': '0c040cf4000000fa',
    's_mov_b32 s24, 5': '850398be',
    's_mov_b32 s24, s26': '1a0398be',
    's_mov_b32 s24, s8': '080398be',
    's_mov_b32 s25, 0': '800399be',
    's_mov_b32 s25, s1': '010399be',
    's_mov_b32 s26, 13': '8d039abe',
    's_mov_b32 s26, s8': '08039abe',
    's_mov_b32 s30, 17': '91039ebe',
    's_mov_b32 s30, 2': '82039ebe',
    's_sub_u32 s30, s30, 1': '1e819e80',
    's_waitcnt lgkmcnt(0)': '7fc08cbf',
    's_waitcnt vmcnt(0)': '703f8cbf',
    's_xor_b32 s26, s26, 16': '1a901a89',
    'v_cvt_f32_u32_e32 v5, s9': '090c0a7e',
    'v_mov_b32 v0, 0': '8002007e',
    'v_mov_b32 v1, 0': '8002027e',
    'v_mul_f32_e32 v5, 0x3e000000, v5': 'ff0a0a100000003e',
}

PREFIX = """s_cmp_eq_u32 s9, 0
s_cbranch_scc1 1
s_mov_b32 s24, 5
"""
ADVANCE = """s_mov_b32 s26, s8
s_add_u32 s26, s26, 32
s_mov_b32 s24, s26
"""
BRANCH = """s_cmp_eq_u32 s9, 1
s_cbranch_scc1 1
"""
HIGH = 's_mov_b32 s25, 0\n'
DRAW = """v_mov_b32 v0, 0
v_mov_b32 v1, 0
s_load_dwordx8 s[16:23], s[24:25], 0x0
s_waitcnt lgkmcnt(0)
image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_cvt_f32_u32_e32 v5, s9
v_mul_f32_e32 v5, 0x3e000000, v5
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
HELD = PREFIX + ADVANCE + HIGH + BRANCH
# Each accepted program loads one descriptor. The old arm writes s24 before the copy.
# Once s24 holds the pointer, a write to its source s26 cannot change it.
ACCEPT = {
    'base': ('s_mov_b32 s24, s8\n' + HIGH, 0),
    'straight': (ADVANCE + HIGH, 32),
    'joined': (HELD + 's_mov_b32 s30, 17\n', 32),
    'source changed': (HELD + 's_mov_b32 s26, 13\n', 32),
    'equal addresses': (HELD + 's_add_u32 s24, s8, 32\n', 32),
}
REJECT = {
    'missing on an arm': (BRANCH + 's_add_u32 s24, s8, 32\n' + HIGH, 'joined at a branch target'),
    'unproven address': (HELD.replace('s_add_u32 s26, s26, 32', 's_add_u32 s26, s26, s9')
                         + 's_mov_b32 s30, 17\n', 'joined at a branch target'),
    'unsupported arithmetic': (ADVANCE.replace('s_mov_b32 s24, s26', 's_xor_b32 s26, s26, 16\ns_mov_b32 s24, s26')
                               + HIGH + BRANCH + 's_mov_b32 s30, 17\n', 'untracked offset'),
    'loop-carried address': (ADVANCE + HIGH + 's_add_u32 s24, s24, 32\ns_cmp_eq_u32 s9, 0\ns_cbranch_scc1 65533\n',
                             'joined at a branch target'),
    # Reset the low word inside each of two iterations. The join alone cannot reject the load:
    # it is the restriction to forward edges that prevents the advance from being folded.
    'advance inside a backward loop': (
        's_mov_b32 s30, 2\ns_mov_b32 s24, s8\ns_add_u32 s24, s24, 32\n'
        's_sub_u32 s30, s30, 1\ns_cmp_lg_u32 s30, 0\ns_cbranch_scc1 65531\n' + HIGH, 'untracked offset'),
    'offset overflow': (HELD.replace('s_add_u32 s26, s26, 32', 's_add_u32 s26, s26, 0xfffff\ns_add_u32 s26, s26, 2')
                        + 's_mov_b32 s30, 17\n', 'accumulated table pointer offset'),
}

# The direct load reads VB slot 2. The low add in the second program may carry, so copying
# the original high word is insufficient to prove the same address.
VB_DRAW = """s_waitcnt lgkmcnt(0)
buffer_load_dword v6, off, s[16:19], 0
s_waitcnt vmcnt(0)
exp pos0 v6, v6, v6, v6 done
exp param0 v6, v6, v6, v6
s_endpgm
"""
VB_DIRECT = 's_load_dwordx4 s[16:19], s[0:1], 0x20\n' + VB_DRAW
VB_COPIED_HIGH = ('s_add_u32 s24, s0, 16\ns_mov_b32 s25, s1\n'
                  's_load_dwordx4 s[16:19], s[24:25], 0x10\n' + VB_DRAW)


CHOICE_DRAW = """s_load_dword s42, s[40:41], 0x20
s_load_dwordx8 s[48:55], s[40:41], 0x0
s_waitcnt lgkmcnt(0)
v_mov_b32 v2, 0
v_mov_b32 v3, 0
image_load v[8:11], v[2:3], s[48:55] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32 v9, s42
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
"""
CHOICE_WORDS = {
    's_load_dword s42, s[40:41], 0x20': '940a00f4200000fa',
    's_load_dwordx8 s[48:55], s[40:41], 0x0': '140c0cf4000000fa',
    'v_mov_b32 v2, 0': '8002047e',
    'v_mov_b32 v3, 0': '8002067e',
    'v_mov_b32 v9, s42': '2a02127e',
    'image_load v[8:11], v[2:3], s[48:55] dmask:0xf dim:SQ_RSRC_IMG_2D': '080f00f002080c00',
    'exp mrt0 v8, v9, v10, v11 done vm': '0f1800f808090a0b',
    's_add_u32 s40, s40, 32': '28a02880',
}
JOIN_REFUSAL = "s_load through s40 holding ('joined at a branch target',)"


def choice_source(count, *, loop=False, bad_high=False):
    """Dispatch on a root constant, covering taken, fall-through, and unconditional edges."""
    if count == 2 and not loop and not bad_high:
        return ("s_mov_b32 s40, s10\ns_mov_b32 s41, 0\n"
                "s_cmp_eq_u32 s9, 1\ns_cbranch_scc1 1\ns_mov_b32 s40, s8\n" + CHOICE_DRAW)
    dispatch = []
    for i in range(1, count):
        # Remaining compare/branch pairs, followed by i three-instruction arms.
        dispatch += [f's_cmp_eq_u32 s9, {i}', f's_cbranch_scc1 {2 * (count - i - 1) + 3 * i}']
    for i in range(count):
        src = 0 if i == 0 else i + 1
        high = int(bad_high and i == 1)
        dispatch += [f's_mov_b32 s40, s{8 + src}', f's_mov_b32 s41, {high}', f's_branch {3 * (count - i - 1)}']
    draw = CHOICE_DRAW
    if loop:
        first, rest = draw.split('\n', 1)
        draw = first + '\ns_add_u32 s40, s40, 32\ns_cmp_eq_u32 s9, 0\ns_cbranch_scc1 65531\n' + rest
    return '\n'.join(dispatch) + '\n' + draw


def encode_choice(source):
    machine = bytearray()
    for line in source.splitlines():
        if line in CHOICE_WORDS or line in WORDS:
            machine.extend(bytes.fromhex(CHOICE_WORDS.get(line) or WORDS[line]))
            continue
        if line.startswith('s_cmp_eq_u32 s9, '):
            word = 0xbf068009 | (int(line.rsplit(' ', 1)[-1]) << 8)
        elif line.startswith(('s_cbranch_scc1 ', 's_branch ')):
            word = (0xbf850000 if line.startswith('s_cbranch') else 0xbf820000) | int(line.split()[1])
        elif line.startswith('s_mov_b32 s40, s'):
            word = 0xbea80300 | int(line.rsplit('s', 1)[-1])
        elif line.startswith('s_mov_b32 s41, '):
            word = 0xbea90380 | int(line.rsplit(' ', 1)[-1])
        else:
            raise AssertionError(line)
        machine.extend(struct.pack('<I', word))
    return bytes(machine)


def choice_root(count):
    return root('t', 1, *('t',) * (count - 1))


CHOICE_SOURCES = {count: choice_source(count) for count in (2, 3, 4, 5)}
CHOICE_CONTROLS = {'five tables': (5, CHOICE_SOURCES[5]),
                   'loop advance': (2, choice_source(2, loop=True)),
                   'mismatched high word': (2, choice_source(2, bad_high=True))}


def encode(source):
    return bytes.fromhex(''.join(WORDS[line] for line in source.splitlines()))


def bindings(vs, ps):
    pipe = G.Pipeline(vs, ps, root('t', 1))
    return [line for line in pipe.manifest().splitlines() if line.startswith('srv ')]


def require_refusal(name, vs, ps, reason):
    try:
        bindings(vs, ps)
    except G.Unsupported as error:
        assert reason in str(error), (name, str(error))
    else:
        raise AssertionError(f'{name} was accepted')


def check_host():
    for name, (prefix, offset) in ACCEPT.items():
        assert bindings(VS_WORDS, encode(prefix + DRAW)) == [f'srv 0 tex2d table 0 {offset}'], name
    for name, (prefix, reason) in REJECT.items():
        require_refusal(name, VS_WORDS, encode(prefix + DRAW), reason)
    assert bindings(encode(VB_DIRECT), PS_WORDS) == ['srv 0 buf vb 2']
    require_refusal('copied vertex buffer table high word', encode(VB_COPIED_HIGH), PS_WORDS,
                    'its high word is not the matching advanced pair')
    assert bindings(VS_WORDS, encode(HELD + 's_add_u32 s24, s8, 64\n' + DRAW)) == [
        'srv 0 tex2d table 0 64', 'srv 1 tex2d table 0 32']
    for count in (2, 3, 4):
        pipe = G.Pipeline(VS_WORDS, encode_choice(CHOICE_SOURCES[count]), choice_root(count))
        tables = [0, *range(2, count + 1)]
        lines = pipe.manifest().splitlines()
        assert [line for line in lines if line.startswith('srv ')] == [
            f'srv {i} tex2d table {table} 0' for i, table in enumerate(tables)]
        assert lines[0] == 'x2w-manifest 3'
        assert [line for line in lines if line.startswith('srvchoice ')] == [f'srvchoice {i}' for i in range(count)]
        assert [line for line in lines if 'tableword' in line] == [
            f'udsrc {i + 4} tableword {table} 32' for i, table in enumerate(tables)]
        assert 'switch (ptrsel_' in pipe.hlsl_ps()
    for name, (count, source) in CHOICE_CONTROLS.items():
        try:
            G.Pipeline(VS_WORDS, encode_choice(source), choice_root(count))
        except G.Unsupported as error:
            assert str(error) == JOIN_REFUSAL, (name, str(error))
        else:
            raise AssertionError(f'{name} was accepted')
    assert G.lifted_family(JOIN_REFUSAL) == 'joined-table-pointer'
    print('Graphics table pointer joins: single bindings, 2–4 choices, and refusal controls matched')


def expected(offset):
    # Each group of four columns sees opposite table contents. G records the root constant
    # used by the branches. Multiplication by 1/8 avoids ambiguous half-integer UNORM results.
    return bytes(channel for _ in range(4) for x in range(8) for channel in
                 (16 + 128 * ((offset // 32) ^ (x // 4)), (0, 32, 64, 96)[x % 4], 0, 255))


def draw(folder, consumer, backend, kind, compiler, prefix, offset):
    cache = folder / 'cache'
    cache.mkdir()
    vs, ps, signature, pixels = (folder / file for file in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(container(0, [word for word, in struct.iter_unpack('<I', encode(prefix + DRAW))]))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_TEST_JOINED_TABLE='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=str(compiler) if kind == 'fxc' else '')
    env['PATH'] = str(Path(compiler).resolve().parent) + os.pathsep + env['PATH']
    cmd = [consumer, vs, ps, signature]
    subprocess.run(cmd, env=env, check=True)
    subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], env=env, check=True)
    manifest, = (path for path in cache.glob('*.txt') if path.name != 'revision.txt')
    text = manifest.read_text(encoding='ascii')
    line = f'srv 0 tex2d table 0 {offset}'
    assert text.splitlines().count(line) == 1
    for stage in ('vs', 'ps'):
        compile_shader(compiler, kind, stage, manifest.with_suffix(f'.{stage}.hlsl'),
                       manifest.with_suffix(f'.{stage}.cso'), folder / f'{stage}.asm')
    baseline = expected(offset)
    for wanted in (offset, 32 - offset):
        manifest.write_text(text.replace(line, f'srv 0 tex2d table 0 {wanted}'), encoding='ascii')
        result = subprocess.run(cmd + [backend, pixels], env=env, capture_output=True, text=True)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0 and '-> drawn;' in result.stderr, result.stdout + result.stderr
        data = pixels.read_bytes()
        assert data == expected(wanted), (data.hex(), expected(wanted).hex())
        changes = sum(data[i:i+4] != baseline[i:i+4] for i in range(0, len(data), 4))
        assert changes == (0 if wanted == offset else 32), changes
        print(f'{kind} {backend} table byte {wanted}, {changes} pixels changed from the source expectation')


def choice_pixels(count, mutant=False):
    data = bytearray()
    for i in range(count):
        table = 0 if mutant else i
        width = (i + 1) * 8 // count - i * 8 // count
        data.extend(bytes((16 + 64 * table, 32 * (table + 1), 0, 255)) * width)
    return bytes(data) * 4


def draw_choice(folder, consumer, backend, kind, compiler, count):
    cache = folder / 'cache'
    cache.mkdir()
    vs, ps, signature, pixels = (folder / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    code = encode_choice(CHOICE_SOURCES[count])
    ps.write_bytes(container(0, [word for word, in struct.iter_unpack('<I', code)]))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_TEST_JOINED_TABLE=str(count), X2W_SHADER_CACHE=str(cache),
               X2W_FXC=str(compiler) if kind == 'fxc' else '')
    env['PATH'] = str(Path(compiler).resolve().parent) + os.pathsep + env['PATH']
    cmd = [consumer, vs, ps, signature]
    subprocess.run(cmd, env=env, check=True)
    subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], env=env, check=True)
    manifest, = (path for path in cache.glob('*.txt') if path.name != 'revision.txt')
    for stage in ('vs', 'ps'):
        compile_shader(compiler, kind, stage, manifest.with_suffix(f'.{stage}.hlsl'),
                       manifest.with_suffix(f'.{stage}.cso'), folder / f'{stage}.asm')

    def readback():
        pixels.unlink(missing_ok=True)
        result = subprocess.run(cmd + [backend, pixels], env=env, capture_output=True, text=True)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0 and '-> drawn;' in result.stderr, result.stdout + result.stderr
        return pixels.read_bytes()

    expected = choice_pixels(count)
    actual = readback()
    assert actual == expected, (count, actual.hex(), expected.hex())

    shader = manifest.with_suffix('.ps.hlsl')
    mutant, mutations = re.subn(r'(ptrsel_[0-9a-f]+_s[0-9]+ = )[1-3]u;', r'\g<1>0u;',
                                shader.read_text(encoding='ascii'))
    assert mutations == count - 1, mutations
    shader.write_text(mutant, encoding='ascii')
    compile_shader(compiler, kind, 'ps', shader, manifest.with_suffix('.ps.cso'), folder / 'mutant.asm')
    actual = readback()
    assert actual != expected and actual == choice_pixels(count, mutant=True), (count, actual.hex())
    print(f'{kind} {backend}: {count} tables matched the CPU reference and the zero selector failed it')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_host()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        sources = [prefix + DRAW for prefix, _ in (*ACCEPT.values(), *REJECT.values())]
        for source in [*sources, VB_DIRECT, VB_COPIED_HIGH]:
            assert assemble(sys.argv[2], source) == encode(source)
        for source in {*CHOICE_SOURCES.values(), *(source for _, source in CHOICE_CONTROLS.values())}:
            assert assemble(sys.argv[2], source) == encode_choice(source)
        print('Graphics table fixtures match gfx1013 assembly')
    elif len(sys.argv) == 7 and sys.argv[6] == 'choice':
        consumer, backend, kind, dxc, fxc = sys.argv[1:6]
        for count in (2, 3, 4):
            with tempfile.TemporaryDirectory() as temporary:
                draw_choice(Path(temporary), consumer, backend, kind, fxc if kind == 'fxc' else dxc, count)
    elif len(sys.argv) == 6:
        consumer, backend, kind, dxc, fxc = sys.argv[1:]
        for prefix, offset in ACCEPT.values():
            with tempfile.TemporaryDirectory() as temporary:
                draw(Path(temporary), consumer, backend, kind, fxc if kind == 'fxc' else dxc, prefix, offset)
    else:
        raise SystemExit(__doc__)
