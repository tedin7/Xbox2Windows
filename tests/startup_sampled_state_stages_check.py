"""Copy and sample on one translated direct list, then compare every WARP pixel."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl

TEXTURE = 'Texture2D<float4> image : register(t%d); SamplerState sample_point : register(s0);\n'
INTERFACE = 'struct Vertex { float4 position : SV_Position; nointerpolation float4 color : COLOR0; };\n'
SAMPLE = 'image.SampleLevel(sample_point, float2(0.75, 0.5), 0)'
REFERENCE = bytes((201, 119, 57, 255)) * (8 * 4)
# INVALID_SUBRESOURCE_STATE, GPU_BASED_VALIDATION_INCOMPATIBLE_RESOURCE_STATE, and 1358, the same draw-time check on
# a runtime that tracks texture layouts ("Incompatible texture barrier layout", seen on the Win2025 WARP).
STATE_ERROR = re.compile(r'sev=[01] id=(?:538|942|1358)\b')


def shaders(vertex_reads, pixel_reads, pixel_slot):
    vs = (TEXTURE % 0 if vertex_reads else '') + INTERFACE + '''
Vertex main(uint index : SV_VertexID) {
    Vertex v;
    float2 xy = index == 0 ? float2(-1, 1) : index == 1 ? float2(3, 1) : float2(-1, -3);
    v.position = float4(xy, 0, 1);
    v.color = %s;
    return v;
}
''' % (SAMPLE if vertex_reads else 'float4(1, 1, 1, 1)')
    ps = (TEXTURE % pixel_slot if pixel_reads else '') + INTERFACE + '''
float4 main(Vertex v) : SV_Target0 {
    return v.color%s;
}
''' % ((' * float4(1, 1, ' + SAMPLE + '.b, 1)') if vertex_reads and pixel_reads
       else ' * ' + SAMPLE if pixel_reads else '')
    return vs, ps


def check(directory, compiler, consumer):
    cache = directory / 'cache'
    cache.mkdir()
    vertex, pixel, root, output = (directory / name for name in ('vertex', 'pixel', 'root', 'rgba'))
    vertex.write_bytes(container(1))
    pixel.write_bytes(container(0))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_SHADER_CACHE=str(cache), X2W_TEST_VERTEX_SAMPLE='1', X2W_D3D12_DEBUG='1')
    env.pop('X2W_TEST_DEBUG_LAYER', None)
    args = [consumer, vertex, pixel, root]
    subprocess.run(args, env=env, check=True, timeout=45)
    key = '%016x' % gfx2hlsl.pipeline_key(vertex.read_bytes(), pixel.read_bytes(), root.read_bytes())

    def prepare(stages, vertex_reads, pixel_reads, cube=False):
        lines = ['x2w-manifest 1', 'ud 1', 'udsrc 0 rect', 'smp 0 0 0 0 0', 'targets 0']
        for slot, stage in enumerate(stages):
            # The alias case binds t0 and t1 to the same guest view.
            lines.append(f'srv {slot} {"texcube" if cube else "tex2d"} table 0 0')
            if stage:
                lines += [f'srvstage {slot} {stage}']
        (cache / f'{key}.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
        sources = shaders(vertex_reads, pixel_reads, len(stages) - 1)
        for stage, source in zip(('vs', 'ps'), sources):
            if cube and stage == 'vs':
                source = source.replace('Texture2D', 'TextureCube').replace('float2(0.75, 0.5)', 'float3(-1, -0.5, 0.5)')
            path = directory / f'{stage}.hlsl'
            path.write_text(source, encoding='ascii')
            subprocess.run([compiler, '-T', stage + '_6_0', '-E', 'main', '-Fo',
                            cache / f'{key}.{stage}.cso', path], check=True, timeout=30)

    def draw(name, switch, expected, mutant=False, cube=False):
        output.unlink(missing_ok=True)
        run = subprocess.run([*args, 'warp', output],
                             env=dict(env, X2W_SAMPLED_STATE_STAGES=switch, X2W_TEST_VERTEX_SAMPLE='cube' if cube else '1'),
                             capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, (name, run.returncode, run.stdout, run.stderr)
        assert '-> drawn;' in run.stderr and 'CopyTextureRegion refused' not in run.stderr, run.stderr
        actual = output.read_bytes()
        assert actual == expected, (name, actual[:32].hex(), expected[:32].hex(), len(actual))
        debug = 'GPU validation enabled' in run.stdout
        assert debug or 'debug layer unavailable' in run.stdout, run.stdout
        errors = STATE_ERROR.findall(run.stderr)
        assert bool(errors) == (debug and mutant), (name, run.stdout, run.stderr)
        print(f'{name}: 32 pixels match, WARP, {run.stdout.strip()}, state errors={errors}')

    prepare(['nonpixel'], True, False)
    draw('vertex', '1', REFERENCE)
    draw('pixel-only state mutant', '0', REFERENCE, mutant=True)
    prepare(['pixel'], False, True)
    draw('pixel', '1', REFERENCE)
    # Both stages consume the texture. Multiplying the blue channel is observable in the CPU reference.
    both_reference = bytes((201, 119, round(57 * 57 / 255), 255)) * (8 * 4)
    prepare(['both'], True, True)
    draw('shared slot', '1', both_reference)
    prepare(['nonpixel', 'pixel'], True, True)
    draw('aliased slots', '1', both_reference)
    prepare([None], True, True)
    draw('legacy manifest', '1', both_reference)
    prepare(['nonpixel'], True, False, cube=True)
    # Direction (-1, -0.5, 0.5) selects -X with (u, v) = (0.75, 0.75).
    draw('vertex cube', '1', bytes((181, 149, 57, 255)) * (8 * 4), cube=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as temp:
        check(Path(temp), *sys.argv[1:])
