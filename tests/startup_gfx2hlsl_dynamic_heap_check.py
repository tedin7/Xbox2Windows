"""Translate and draw a pixel shader that takes a heap record byte offset in s8."""
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


if __name__ == '__main__':
    if len(sys.argv) == 1:
        translate()
    else:
        main(*sys.argv[1:])
    print('dynamic graphics heap: readback and controls matched' if len(sys.argv) > 1
          else 'dynamic graphics heap: register offset translated')
