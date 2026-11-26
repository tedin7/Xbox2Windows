"""Observe the raw record count and the copied placement through GPU readbacks."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache
from startup_compute_dispatch_check import C, G, dispatch, lines, prepare_case


def request(directory, consumer, env):
    _, result = dispatch(directory, consumer, env, 'dispatch')
    published, = lines(result, 'x2w-draw compute request')
    assert ' published ' in published, published
    return published.split()[3]


def raw_tail(directory, consumer, dxc):
    cache, env = prepare_case(directory, 'warp')
    key = request(directory, consumer, env)
    source = cache / f'{key}.cs.hlsl'
    source.write_text("""\
ByteAddressBuffer input_buffer : register(t0);
RWByteAddressBuffer output_buffer : register(u0);
cbuffer entry : register(b0) { uint records; };

[numthreads(2, 3, 4)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
    uint word = (group.x + 3 * group.y + 6 * group.z) * 24
              + lane.x + 2 * lane.y + 6 * lane.z;
    uint address = word * 4;
    if (address + 4 <= records)
        output_buffer.Store(address, input_buffer.Load(address) ^ 0xa5a5a5a5u);
}
""", encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cache / f'{key}.cs.cso', source], check=True)
    # The first dispatch consumes the raw token's byte count. The second writes 72
    # words into a structured descriptor with 320 records, so its bound admits every store.
    (cache / f'{key}.txt').write_text(
        'x2w-manifest 2\nstage cs\ngroup 2 3 4\nud 1\nudsrc 0 uav_numrecords 0\n'
        'srv 0 buf table 1 0\nuav 0 table 1 16\n', encoding='ascii')

    got, result = dispatch(directory, consumer, dict(env, X2W_VIEW_TOKEN_UAV='1'), 'dispatch-view-token-raw-tail')
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    assert len(lines(result, 'Dispatch(3, 1, 1) translated')) == 1, result.stderr[-3000:]
    # Both passes XOR the same constant. The mutant suppresses the tail stores,
    # so the second dispatch reads the initial zeros and returns 0xa5a5a5a5.
    quarter = 72
    expected = [C.input_word(i) for i in range(3 * quarter, 4 * quarter)] + [C.SENTINEL] * (C.WORDS - quarter)
    for i, (actual, want) in enumerate(zip(got, expected)):
        assert actual == want, f'raw token tail word {i}: got {actual:08x}, expected {want:08x}'
    print(f'WARP: all {quarter} last-quarter words and {C.WORDS - quarter} untouched words match')


def placed_draw(directory, consumer, dxc):
    compute = directory / 'compute'
    cache, env = prepare_case(compute, 'warp')
    key = request(compute, consumer, env)
    translated = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
    assert f'{key}: ok' in translated.stdout, translated.stdout

    hlsl = """\
ByteAddressBuffer b : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {
    return float4(float(b.Load(0) & 255u) / 255.0, 0, 0, 1);
}
"""
    draw_env, args, output = prepare_draw_cache(
        directory, consumer, hlsl, {'X2W_TEST_PLACED_TOKEN_DRAW': '1'}, 'srv 0 buf table 0 0')
    for suffix in ('.txt', '.cs.cso'):
        name = key + suffix
        shutil.copyfile(cache / name, directory / 'cache' / name)
    draw_env.update(X2W_TEST_COMPUTE_ROOT=env['X2W_TEST_COMPUTE_ROOT'], X2W_TEST_DISPATCH_OUT=str(output),
                    X2W_D3D12_LINEAR_BUFFERS='0', X2W_D3D12_DEBUG='1')
    args += ['slot67', compute / 'program', compute / 'data', compute / 'state', 'dispatch-view-token-placed-draw']

    gpu_red = C.expected('A')[0] & 255
    assert gpu_red != 0xdd
    for enabled, red in ((False, 0xdd), (True, gpu_red)):
        output.unlink(missing_ok=True)
        result = subprocess.run(args, env=dict(draw_env, X2W_VIEW_TOKEN_UAV=str(int(enabled))),
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 77:
            sys.exit(77)
        context = f'switch={int(enabled)}\n{result.stderr[-4000:]}'
        assert result.returncode == 0, f'consumer exit {result.returncode}\n{context}'
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == int(enabled), context
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors
        actual = output.read_bytes()
        pixel = bytes((red, 0, 0, 255))
        assert len(actual) == 8 * 4 * 4, context
        for i in range(0, len(actual), 4):
            assert actual[i:i + 4] == pixel, (
                f'placed token pixel {i // 4}: got {actual[i:i + 4].hex()}, expected {pixel.hex()}\n{context}')
        assert bool(lines(result, "constants bound from the placed buffer's own copy")) == enabled, context
    print(f'WARP: 32 pixels match title memory (red=221) with the switch off and GPU output (red={gpu_red}) with it on')


if __name__ == '__main__':
    mode, consumer, dxc = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temporary:
        {'raw-tail': raw_tail, 'placed-draw': placed_draw}[mode](Path(temporary), consumer, dxc)
