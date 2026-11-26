"""Exercise compute and graphics access to GPU-only view-token buffers on WARP."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_compute_dispatch_check import C, G, dispatch, lines, prepare_case
from startup_stream_check import container
from startup_stream_draw_check import vertex


WRITTEN = 3 * 2 * 2 * 2 * 3 * 4


def transformed(value, i):
    return ((value * 33) & C.MASK) ^ ((C.KEY + i * C.STRIDE) & C.MASK)


def records_guard(pipe):
    # Case I translates a raw store that normally ignores NUM_RECORDS. Exercise the runtime's
    # manifest constants by guarding that store exactly when the output descriptor is raw.
    pipe.hlsl_cs()
    stride = pipe.ud(('uav_stride', 0))
    records = pipe.ud(('uav_numrecords', 0))
    hlsl = pipe.hlsl_cs()
    store = '    u0.Store(a, v5);'
    assert hlsl.count(store) == 1
    guard = f'if (ud{stride} != 0u || (ud{records} >= 4u && a <= ud{records} - 4u))'
    return hlsl.replace(store, f'    {guard} u0.Store(a, v5);'), pipe.manifest()


def check_raw_token(directory, consumer, dxc, expected):
    directory.mkdir()
    cache, env = prepare_case(directory, 'warp', 'I')
    env.pop('X2W_VIEW_TOKEN_UAV', None)
    _, request = dispatch(directory, consumer, env, 'dispatch')
    key = int(lines(request, 'x2w-draw compute request')[0].split()[3], 16)
    hlsl, manifest = records_guard(G.load_compute_request(cache, key))
    shader = cache / f'{key:016x}.cs.hlsl'
    shader.write_text(hlsl)
    subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', shader.with_suffix('.cso'), shader],
                   check=True)
    (cache / f'{key:016x}.txt').write_text(manifest)
    got, result = dispatch(directory, consumer, dict(env, X2W_VIEW_TOKEN_UAV='1', X2W_D3D12_GBV='1'),
                           'dispatch-view-token-raw')
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 2, result.stderr[-3000:]
    # The producer covers the entire 288-dword view. A byte count divided by four cuts off
    # stores after 72 dwords, so the later dispatch reads zero instead of the last quarter.
    last_quarter = 3 * WRITTEN // 4
    assert expected[last_quarter:WRITTEN] != [transformed(0, i) for i in range(last_quarter, WRITTEN)]
    mismatch = next((i for i in range(last_quarter, WRITTEN) if got[i] != expected[i]), None)
    assert mismatch is None, f'raw token output, last quarter: first differing word {mismatch}\n{result.stderr[-3000:]}'
    assert got == expected, 'raw token output differs outside its last quarter'
    print(f'WARP: raw token writes {WRITTEN * 4} bytes, including all {(WRITTEN - last_quarter) * 4} bytes of the last quarter')


def draw_pixel_shader():
    return container(0, [
        0xf4080304, 0xfa000000,  # s_load_dwordx4 s[12:15], s[8:9], 0x0
        0xbf8cc07f,              # s_waitcnt lgkmcnt(0)
        # buffer_load_dwordx4 v[0:3], off, s[12:15], 0 offset:<last 16 bytes>
        0xe0380000 | (WRITTEN - 4) * 4, 0x80030000,
        0xbf8c3f70,              # s_waitcnt vmcnt(0)
        0xf800180f, 0x03020100,  # exp mrt0 v0, v1, v2, v3 done vm
        0xbf810000,
    ])


def check_draw(consumer, dxc, fxc):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, 'warp')
        env.pop('X2W_VIEW_TOKEN_UAV', None)
        env.pop('X2W_TEST_TRUNCATED_ONLY', None)
        env.pop('X2W_DRAW_NULL_IDENTITY', None)
        vs, ps, root = (directory / name for name in ('vertex', 'pixel', 'root'))
        vs.write_bytes(vertex())
        ps.write_bytes(draw_pixel_shader())
        # Pick input words that the compute shader transforms into exact UNORM endpoints.
        # The destination initially holds (1,1,1,1); the producer must change green.
        first = WRITTEN - 4
        rgba = (255, 0, 255, 255)
        inverse = pow(33, -1, 1 << 32)
        inputs = [C.input_word(i) for i in range(C.WORDS)]
        for i, channel in enumerate(rgba, first):
            bits = 0x3f800000 if channel else 0
            inputs[i] = ((bits ^ ((C.KEY + i * C.STRIDE) & C.MASK)) * inverse) & C.MASK
        computed = [transformed(inputs[i], i) for i in range(first, WRITTEN)]
        rgba_reference = bytes(round(x * 255) for x in struct.unpack('<4f', struct.pack('<4I', *computed)))
        assert rgba_reference == bytes(rgba)
        (directory / 'input').write_bytes(struct.pack(f'<{C.WORDS}I', *inputs))
        output = directory / 'pixels'
        env.update(X2W_TEST_DISPATCH_IN=str(directory / 'input'), X2W_TEST_TOKEN_DRAW_OUT=str(output))
        args = [consumer, vs, ps, root, 'slot67',
                directory / 'program', directory / 'data', directory / 'state', 'dispatch-view-token-draw']

        def run(run_env):
            output.unlink(missing_ok=True)
            result = subprocess.run(args, env=run_env, capture_output=True, text=True, timeout=60)
            if result.returncode == 77:
                raise SystemExit(77)
            assert result.returncode == 0, (result.returncode, result.stderr[-3000:])
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors
            return output.read_bytes(), result

        _, request = run(env)
        compute_key = int(lines(request, 'x2w-draw compute request')[0].split()[3], 16)
        graphics_key = G.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
        translated = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                    capture_output=True, text=True,
                                    env=dict(os.environ, X2W_DXC=dxc, X2W_FXC=fxc)).stdout
        for key in (compute_key, graphics_key):
            assert f'{key:016x}: ok' in translated, translated

        clear = bytes((0, 0, 255, 255))
        for enabled in (False, True):
            run_env = dict(env, X2W_VIEW_TOKEN_UAV='1', X2W_D3D12_GBV='1') if enabled else env
            pixels, result = run(run_env)
            assert lines(result, '-> drawn;'), result.stderr[-3000:]
            assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == int(enabled), result.stderr[-3000:]
            color = rgba_reference if enabled else bytes(4)
            expected = (color * 4 + clear * 4) * 4
            assert pixels == expected, f'switch={int(enabled)}: got {pixels.hex()}, expected {expected.hex()}'
        print('WARP: 32 RGBA pixels match with the switch on and off')


def main(consumer, dxc):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, 'warp')
        env.pop('X2W_VIEW_TOKEN_UAV', None)
        srv_mode = 'dispatch-view-token-srv'
        uav_mode = 'dispatch-view-token-uav'

        _, request = dispatch(directory, consumer, env, 'dispatch')
        key = int(lines(request, 'x2w-draw compute request')[0].split()[3], 16)
        translation = subprocess.run(
            [sys.executable, '-B', G.__file__, '--pending', cache],
            check=True, capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc),
        )
        assert f'{key:016x}: ok' in translation.stdout, translation.stdout

        written = WRITTEN
        untouched = [C.SENTINEL] * C.WORDS
        tail = untouched[written:]
        expected = [transformed(transformed(C.input_word(i), i), i) for i in range(written)] + tail
        zero_input = [transformed(0, i) for i in range(written)] + tail
        assert expected != zero_input

        validated = False
        for enabled, mode, reference, dispatches in (
            (False, srv_mode, zero_input, 1),
            (False, uav_mode, untouched, 0),
            (True, srv_mode, expected, 2),
            (True, uav_mode, expected, 2),
        ):
            run_env = dict(env, X2W_VIEW_TOKEN_UAV='1', X2W_D3D12_GBV='1') if enabled else env
            got, result = dispatch(directory, consumer, run_env, mode)
            context = f'{mode}, switch={int(enabled)}\n{result.stderr[-3000:]}'
            mismatch = next((i for i, (a, b) in enumerate(zip(got, reference)) if a != b), None)
            assert mismatch is None, f'first differing word {mismatch}\n{context}'
            assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == dispatches, context
            refusals = lines(result, 'Dispatch(3, 2, 2) refused:')
            assert len(refusals) == 2 - dispatches, context
            if not enabled:
                assert refusals[0].endswith('an output is a view token, not a buffer descriptor'), context
                if mode == uav_mode:
                    assert refusals[1].endswith('an input view token is not a buffer view of a desktop resource'), context
            layer = lines(result, 'x2w-d3d12 debug layer: ')
            if enabled and layer and '0x00000000' in layer[0]:  # a machine without the graphics tools has no validation to ask for
                assert lines(result, 'GPU-based validation enabled'), context
                validated = True
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors
        # A view destination inside a descriptor slot is refused (a tagged-handle driver reads wild memory); the dispatch is unchanged.
        got, result = dispatch(directory, consumer, env, 'dispatch-view-misaligned')
        assert got == C.expected('A'), 'a misaligned view destination changed the dispatch\n' + result.stderr[-3000:]
        assert len(lines(result, 'is not on a descriptor slot boundary, refused')) == 1, result.stderr[-3000:]
        plain = lines(dispatch(directory, consumer, env, 'dispatch')[1], 'is not on a descriptor slot boundary')
        assert not plain, plain
        # Switch X2W_VIEW_SLOT_ALIGN=round (default off = the refusal above): the destination is rounded down to its slot, counted.
        # The dispatch result is not asserted: the view now lands on slot 0.
        _, rounded = dispatch(directory, consumer, dict(env, X2W_VIEW_SLOT_ALIGN='round'), 'dispatch-view-misaligned')
        assert len(lines(rounded, 'slot boundary, rounded to 0x')) == 1, rounded.stderr[-3000:]
        assert '(count 1)' in lines(rounded, 'slot boundary, rounded to 0x')[0], rounded.stderr[-3000:]
        assert not lines(rounded, 'slot boundary, refused'), rounded.stderr[-3000:]

        check_raw_token(directory / 'raw', consumer, dxc, expected)

        # The token's buffer is placed over title memory the desktop cannot alias. A later list copy of it must carry the GPU's
        # bytes, never the stale title memory: the stand-in refuses the copy. Without note_placed_gpu_write it copies the stale
        # 0xdddddddd words in title memory instead.
        got, result = dispatch(directory, consumer, dict(env, X2W_VIEW_TOKEN_UAV='1'), 'dispatch-view-token-placed')
        copies = lines(result, 'CopyBufferRegion from a placed buffer without title backing')
        assert copies and copies[0].endswith('1296 bytes -> refused'), f'copy of a GPU-written placed buffer: {copies}\n{result.stderr[-3000:]}'
        assert not lines(result, 'copied in title memory'), 'stale title memory was copied'
        assert 0xdddddddd not in got, 'stale title words reached the readback'
        print(f'WARP: {C.WORDS} words match for SRV and slot-19 reads, including switch-off controls; '
              f'GPU-based validation {"on" if validated else "unavailable (no graphics tools)"}')


if __name__ == '__main__':
    if sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
