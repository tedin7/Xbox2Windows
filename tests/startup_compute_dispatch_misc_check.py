"""Exercise opt-in compute bindings through slot 67 and WARP."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

from startup_compute_dispatch_texture_check import clean, run
from startup_compute_dispatch_check import lines
from startup_compute_request_check import prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import root_blob


CASES = ('zero_input', 'zero_output', 'texture_alias', 'integer_output')
SENTINEL = 0xc5c5c5c5
HEADER = 'x2w-manifest 2\nstage cs\ngroup 1 1 1\nud 0\n'


def fields(directory, input_records, output_records):
    path = directory / 'fields'
    path.write_bytes(struct.pack('<6I', 4, input_records, 0, 4, output_records, 0))
    return str(path)


def fixture(directory, case, signed=False):
    if case in ('zero_input', 'zero_output'):
        before = [SENTINEL] * 320
        reference = before.copy()
        if case == 'zero_input':
            source = directory / 'input'
            source.write_bytes(struct.pack('<320I', 39, *([0] * 319)))
            env = {'X2W_TEST_DISPATCH_IN': str(source), 'X2W_TEST_DISPATCH_FIELDS': fields(directory, 0, 1)}
            reference[0] = 17
            return ('dispatch-misc', '<320I', env, before, reference,
                    'an input buffer descriptor has zero records', 'srv 0 buf table 1 0\nuav 0 table 1 16\n',
                    'ByteAddressBuffer src : register(t0); RWByteAddressBuffer dst : register(u0);',
                    'dst.Store(0, src.Load(0) + src.Load(16) + 17);')
        reference[4] = 83
        return ('dispatch-misc', '<320I', {'X2W_TEST_DISPATCH_FIELDS': fields(directory, 320, 0)}, before, reference,
                'an output range has zero bytes', 'uav 0 table 1 16\nuav 1 table 1 32\n',
                'RWByteAddressBuffer empty : register(u0); RWByteAddressBuffer probe : register(u1);',
                'empty.Store(0, 41); probe.Store(0, 83);')
    if case == 'texture_alias':
        before = list(range(11, 11 + 64 * 8))
        reference = before.copy()
        reference[0] = before[7] + 5
        return ('texture-misc-self', '<512f', {}, before, reference,
                'an output texture is also an input of the dispatch',
                'srv 0 tex2d table 1 0 via_uav 0\nuav 0 tex2d float table 1 64 channels 1\n',
                'RWTexture2D<float> image : register(u0);', 'image[uint2(0, 0)] = image[uint2(7, 0)] + 5;')

    before = list(range(0x5000, 0x5000 + 32 * 8 * 4))
    reference = before.copy()
    values = (-39, 261, -8193, 53) if signed else (65013, 261, 8193, 53)
    reference[:4] = [value & 0xffff for value in values]
    element = 'int' if signed else 'uint'
    vector = f'{element}4'
    components = ', '.join(map(str, values))
    return ('texture-misc-sint' if signed else 'texture-store-uint', '<1024H', {}, before, reference,
            'an output texture format is not a float or normalized one',
            f'uav 0 tex2d {element} table 1 64 channels 4\n',
            f'RWTexture2D<{vector}> image : register(u0);',
            f'image[uint2(0, 0)] = {vector}({components});')


def verify(directory, consumer, dxc, case, signed=False):
    cache, env = prepare(directory)
    root = directory / 'compute_root'
    root.write_bytes(root_blob(3, table_range=True))
    # s_endpgm serves only as a source-built request marker. The fixture supplies the desktop program.
    (directory / 'program').write_bytes(struct.pack('<I', 0xbf810000))
    (directory / 'data').write_bytes(bytes(16))
    (directory / 'state').write_bytes(state_a((1, 1, 1), 0))
    mode, layout, overrides, before, reference, reason, binding, declarations, body = fixture(directory, case, signed)
    env = dict(env, X2W_TEST_COMPUTE_ROOT=str(root), X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1',
               X2W_CS_MISC='0', **overrides)
    env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    for variable in ('X2W_TEST_DISPATCH_FIELDS', 'X2W_TEST_DISPATCH_IN'):
        if variable not in overrides:
            env.pop(variable, None)
    got, result = run(directory, consumer, env, mode, layout)
    requests = lines(result, 'x2w-draw compute request ')
    assert got == before and len(requests) == 1, result.stderr[-3000:]
    key = int(requests[0].split()[3], 16)
    shader, binary = directory / 'misc.hlsl', cache / f'{key:016x}.cs.cso'
    manifest = cache / f'{key:016x}.txt'

    def compile_shader(code):
        shader.write_text(f'{declarations}\n[numthreads(1,1,1)]\n'
                          f'void main(uint3 id : SV_DispatchThreadID) {{ if (all(id == 0)) {{ {code} }} }}\n')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(binary), str(shader)], check=True)

    compile_shader(body)
    # Publish the legacy binding syntax with the switch off so the refusal comes
    # from descriptor resolution, after the manifest and PSO have loaded.
    legacy_binding = (binding.replace(' via_uav 0', '').replace('channels 1', 'channels 4')
                      .replace('tex2d uint', 'tex2d float').replace('tex2d int', 'tex2d float'))
    manifest.write_text(HEADER + legacy_binding)
    got, result = run(directory, consumer, env, mode, layout)
    refusals = lines(result, 'refused: ')
    assert got == before and len(refusals) == 1 and refusals[0].endswith(reason), result.stderr[-3000:]
    env['X2W_CS_MISC'] = '1'
    manifest.write_text(HEADER + binding)
    got, result = run(directory, consumer, env, mode, layout)
    assert len(lines(result, ') translated')) == 1, result.stderr[-3000:]
    clean(result)
    assert got == reference, next((i, g, e) for i, (g, e) in enumerate(zip(got, reference)) if g != e)

    # Each mutant still dispatches. Its GPU readback must differ at the word that exercises the binding.
    mutant_reference = reference.copy()
    if case == 'zero_input':
        env['X2W_TEST_DISPATCH_FIELDS'] = fields(directory, 1, 1)
        mutant_reference[0] = 56
    elif case == 'zero_output':
        env['X2W_TEST_DISPATCH_FIELDS'] = fields(directory, 320, 1)
        mutant_reference[0] = 41
    elif case == 'texture_alias':
        compile_shader(body.replace('uint2(7, 0)', 'uint2(9, 0)'))
        mutant_reference[0] = before[9] + 5
    else:
        compile_shader(body.replace('261', '262'))
        mutant_reference[1] = 262
    got, result = run(directory, consumer, env, mode, layout)
    assert len(lines(result, ') translated')) == 1, result.stderr[-3000:]
    clean(result)
    assert got == mutant_reference and got != reference, 'mutant did not change the expected readback'
    print(f'{mode}: {len(reference)} values match CPU, switch off refused, mutant rejected by CPU comparison')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('consumer')
    parser.add_argument('dxc')
    parser.add_argument('case', choices=CASES)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        verify(directory / 'main', args.consumer, args.dxc, args.case)
        if args.case == 'integer_output':
            verify(directory / 'signed', args.consumer, args.dxc, args.case, signed=True)
