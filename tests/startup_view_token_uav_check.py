"""Exercise a GPU-only buffer written through a UAV token on WARP."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_compute_dispatch_check import C, G, dispatch, lines, prepare_case


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

        def transformed(value, i):
            return ((value * 33) & C.MASK) ^ ((C.KEY + i * C.STRIDE) & C.MASK)

        written = 3 * 2 * 2 * 2 * 3 * 4
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

        # Case I clears the stride before it stores. The raw-tail regression separately exercises the record count.
        raw_directory = directory / 'raw'
        raw_directory.mkdir()
        raw_cache, raw_env = prepare_case(raw_directory, 'warp', 'I')
        dispatch(raw_directory, consumer, raw_env, 'dispatch')
        subprocess.run([sys.executable, '-B', G.__file__, '--pending', raw_cache], check=True, capture_output=True,
                       text=True, env=dict(os.environ, X2W_DXC=dxc))
        got, result = dispatch(raw_directory, consumer, dict(raw_env, X2W_VIEW_TOKEN_UAV='1'), uav_mode)
        mismatch = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        assert mismatch is None, f'raw token output: first differing word {mismatch}\n{result.stderr[-3000:]}'

        # Retain the copy refusal check. The placed-draw regression observes the GPU-written bytes.
        got, result = dispatch(directory, consumer, dict(env, X2W_VIEW_TOKEN_UAV='1'), 'dispatch-view-token-placed')
        copies = lines(result, 'CopyBufferRegion from a placed buffer without title backing')
        assert copies and copies[0].endswith('1296 bytes -> refused'), f'copy of a GPU-written placed buffer: {copies}\n{result.stderr[-3000:]}'
        assert not lines(result, 'copied in title memory'), 'stale title memory was copied'
        assert 0xdddddddd not in got, 'stale title words reached the readback'
        print(f'WARP: {C.WORDS} words match for SRV and slot-19 reads, including switch-off controls; '
              f'GPU-based validation {"on" if validated else "unavailable (no graphics tools)"}')


if __name__ == '__main__':
    main(*sys.argv[1:])
