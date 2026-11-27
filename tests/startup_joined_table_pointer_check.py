"""Check 32-bit graphics table pointers at forward joins and read both bound textures (#2463).

Run with no arguments for the host controls, or with binary <llvm-mc> to reassemble every source.
The draw arguments are <startup_stream_consumer> <warp|hardware> <fxc|dxc> <dxc> <fxc>.
"""
import os
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
    'different addresses': (HELD + 's_add_u32 s24, s8, 64\n', 'joined at a branch target'),
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
    print('Graphics table pointer joins: six resource bindings and eight refusals matched')


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
    manifest, = (path for path in cache.glob('*.txt') if path.name not in ('revision.txt', 'x2w_key_seed.txt'))
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


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_host()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        sources = [prefix + DRAW for prefix, _ in (*ACCEPT.values(), *REJECT.values())]
        for source in [*sources, VB_DIRECT, VB_COPIED_HIGH]:
            assert assemble(sys.argv[2], source) == encode(source)
        print('All fourteen graphics table fixtures match gfx1013 assembly')
    elif len(sys.argv) == 6:
        consumer, backend, kind, dxc, fxc = sys.argv[1:]
        for prefix, offset in ACCEPT.values():
            with tempfile.TemporaryDirectory() as temporary:
                draw(Path(temporary), consumer, backend, kind, fxc if kind == 'fxc' else dxc, prefix, offset)
    else:
        raise SystemExit(__doc__)
