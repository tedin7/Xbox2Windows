"""Named reasons for a translated compute output that is not an aliased placed buffer range (list slot 14).

usage: startup_compute_dispatch_output_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer records case A of startup_gfx2hlsl_compute_check.py as startup_compute_dispatch_check.py does. After
gfx2hlsl.py --pending with DXC, the unchanged recording still matches all 320 words. Then the output descriptor or
its buffer is changed one condition at a time, and each dispatch is refused with its own reason in the log and the
census and writes no output word:

- dispatch-output-texture-token: the token of a texture view where the output descriptor was;
- zero records, a range over 16 MiB (stride 4, 4 Mi + 1 records) and a range 16 bytes past the placed output
  buffer, through X2W_TEST_DISPATCH_FIELDS (synthetic values);
- dispatch-output-unplaced: memory between the two placed buffers;
- dispatch-output-copied: a buffer placed there at an offset the desktop cannot alias, so it is a separate copy;
- dispatch-output-state: the output buffer placed (and aliased) in the unordered access state;
- dispatch-output-unaligned: the output starts 8 bytes into its buffer;
- dispatch-swapped: the output names the input buffer, which has no unordered access.

A resource with an aliased placement and no recorded information also reports the common state reason; no
fixture reaches it. Exit 77 when the adapter is not available.
"""
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl as G


def fields(records, stride=4):
    return struct.pack('<6I', 4, C.WORDS, 0, stride, records, 0)


CONTROLS = (
    ('dispatch-output-texture-token', None, 'an output buffer descriptor is a view token'),
    # fields(0), an output descriptor with no records, is a null view here: its writes are dropped, as D3D12 drops them.
    ('dispatch', fields((16 << 20) // 4 + 1), 'an output buffer range is over 16 MiB or not whole dwords'),
    ('dispatch', fields(C.WORDS + 4), 'an output range crosses the end of a placed buffer'),
    ('dispatch-output-unplaced', None, 'an output is in title memory outside any placed buffer'),
    ('dispatch-output-copied', None, 'an output is in a placed buffer that does not alias title memory'),
    ('dispatch-output-state', None, "an output buffer's resource was not created in the common state"),
    ('dispatch-output-unaligned', None, 'an output buffer range is not 16-byte aligned in its resource'),
    ('dispatch-swapped', None, 'an output buffer has no unordered access'),
)


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')

        got, result = dispatch(directory, consumer, env, 'dispatch', probe=True)
        print(result.stdout, end='')
        assert got == untouched and len(lines(result, 'refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out

        got, result = dispatch(directory, consumer, env, 'dispatch')
        assert got == expected and not lines(result, 'refused'), result.stderr[-3000:]

        for mode, words, reason in CONTROLS:
            case_env = dict(env)
            if words:
                (directory / 'fields').write_bytes(words)
                case_env['X2W_TEST_DISPATCH_FIELDS'] = str(directory / 'fields')
            got, result = dispatch(directory, consumer, case_env, mode)
            name = mode if not words else f'{mode} fields {struct.unpack("<6I", words)[3:5]}'
            assert got == untouched, f'{name}: a refused Dispatch wrote the output'
            refused = lines(result, 'refused: ')
            census = lines(result, 'x2w-render compute census')
            assert len(refused) == 1 and refused[0].endswith('refused: ' + reason), f'{name}:\n{result.stderr[-3000:]}'
            assert len(census) == 3 and census[1].endswith(f'reason "{reason}": 1 keys, 1 dispatches'), (name, census)
            if mode == 'dispatch-output-state':  # the buffer is aliased, so only its state is refused
                placed = [line for line in lines(result, 'Device::slot133') if ' state=0x8 ' in line]
                assert len(placed) == 1 and '-> 0x00000000 aliased ' in placed[0], (name, placed)
        print(f'startup_compute_dispatch_output: PASS on {adapter} (all {C.WORDS} words match; {len(CONTROLS)} outputs '
              'refused, each with its own reason and an untouched output)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
