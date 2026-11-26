"""A translated compute program dispatched through the stand-in's command list (list slot 14).

usage: startup_compute_dispatch_check.py <startup_stream_consumer> <dxc> <warp|hardware> [--output-view-tokens]

The consumer creates case A of startup_gfx2hlsl_compute_check.py as a device slot 67 pipeline and records, on a
stand-in list, the descriptor heap, the pipeline, its root signature, two root constants (word 0 by slot 37, word 1
by slot 39, with a slot 40 call of other words in between), one table of two
hardware buffer descriptors and Dispatch(3, 2, 2). Input and output are placed buffers that alias the consumer's
memory; the output words are read from that memory after the queue's fence. Four runs share one cache:

1. before translation: the request is published, the Dispatch is refused with "no translation" and every output
   word keeps the sentinel (the check fails when nothing translated is dispatched);
2. after gfx2hlsl.py --pending with DXC: all 320 words equal the case's expectation;
3. dispatch-none (no Dispatch recorded): every word keeps the sentinel;
4. dispatch-swapped (the two descriptors exchanged, so the output names the buffer without unordered access):
   refused with one logged reason, every word keeps the sentinel and the input is unchanged;
5. dispatch-same-root (the same compute root set again after the constants): all 320 words still match;
6. five more refusals, each with its one logged reason and an untouched output: dispatch-missing-word (no slot 37
   call), dispatch-root-changed (another compute root and back after the constants), dispatch-reset (list closed
   and reset after the constants), dispatch-wrong-root (recorded under a root the pipeline was not created with)
   and dispatch-same-buffer (the output buffer is also the input).
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G


DISPATCH_TIMEOUT_SECONDS = 60


def dispatch(directory, consumer, env, mode):
    out = directory / 'out.bin'
    out.unlink(missing_ok=True)
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', mode]
    try:
        result = subprocess.run(args, env=dict(env, X2W_TEST_DISPATCH_OUT=str(out)), capture_output=True, text=True,
                                timeout=DISPATCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as hung:
        tail = hung.stderr.decode(errors='replace') if isinstance(hung.stderr, bytes) else (hung.stderr or '')
        raise AssertionError(f'{mode}: the consumer did not finish in {DISPATCH_TIMEOUT_SECONDS} s\n{tail[-3000:]}') from None
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    if env.get('X2W_D3D12_DEBUG') == '1':  # every mode, not only the first: the debug layer reports a bad call at once
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, f'{mode}: debug layer error: {errors[0]}'
    return list(struct.unpack('<%dI' % C.WORDS, out.read_bytes())), result


def lines(result, text):
    return [line for line in result.stderr.splitlines() if text in line]


def prepare_case(directory, adapter, case='A'):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    else:
        env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, words, rsrc2, group, groups = C.CASES[case]
    assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
    (directory / 'program').write_bytes(words)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    return cache, env


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        assert expected != untouched

        got, result = dispatch(directory, consumer, env, 'dispatch')
        print(result.stdout, end='')
        assert got == untouched, 'a Dispatch without a translation wrote the output'
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)
        census = lines(result, 'x2w-render compute census')
        assert census == ['x2w-render compute census: 1 keys dispatched',
                          'x2w-render compute census reason "no translation": 1 keys, 1 dispatches',
                          f'x2w-render compute census key {key:016x}: 1 dispatches, 0 translated, first refusal: '
                          'no translation [00000000 00000000 00000000 00000000]'], census

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out

        got, result = dispatch(directory, consumer, env, 'dispatch')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
        census = lines(result, 'x2w-render compute census')
        assert len(census) == 2 and census[1].endswith(': 1 dispatches, 1 translated, first refusal: none '
                                                       '[00000000 00000000 00000000 00000000]'), census
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        got, result = dispatch(directory, consumer, env, 'dispatch-none')
        assert got == untouched and not lines(result, 'Dispatch('), 'the output changed without a Dispatch'
        assert not lines(result, 'x2w-render compute census'), 'a census without a Dispatch'

        got, result = dispatch(directory, consumer, env, 'dispatch-swapped')
        assert got == untouched, 'a refused Dispatch wrote the output'
        refused = lines(result, 'refused: ')
        census = lines(result, 'x2w-render compute census')
        assert len(census) == 3 and census[1].endswith(
            'reason "an output buffer was created without unordered access": 1 keys, 1 dispatches') \
            and '[00000000 00000000 ' not in census[2], census
        assert len(refused) == 1 and 'an output buffer was created without unordered access' in refused[0], \
            result.stderr[-3000:]
        got, result = dispatch(directory, consumer, env, 'dispatch-same-root')
        assert got == expected and not lines(result, 'refused'), 'setting the same compute root again lost the constants'

        unset = 'an entry value was not set since the compute root'
        for mode, reason in (('dispatch-missing-word', unset), ('dispatch-root-changed', unset), ('dispatch-reset', unset),
                             ('dispatch-wrong-root', 'the compute root set is not the one the pipeline was created with'),
                             ('dispatch-same-buffer', 'an output buffer is also an input')):
            got, result = dispatch(directory, consumer, env, mode)
            assert got == untouched, f'{mode}: a refused Dispatch wrote the output'
            refused = lines(result, 'refused: ')
            assert len(refused) == 1 and refused[0].endswith('refused: ' + reason), f'{mode}: {refused}\n{result.stderr[-3000:]}'
        print(f'startup_compute_dispatch: PASS on {adapter} (all {C.WORDS} words match after translation and with the '
              'same root set again; untouched before translation, without a Dispatch and for 6 refusals)')


TOKEN_SHADER = """\
RWByteAddressBuffer output_buffer : register(u0);
cbuffer entry : register(b0) { uint word3; uint stride; uint records; }
[numthreads(8, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint size;
    output_buffer.GetDimensions(size);
    if (size == 32 && tid.x * 4 < size)
        output_buffer.Store(tid.x * 4, 0x71243000u ^ word3 ^ (stride << 8) ^ records ^ tid.x);
}
"""
TOKEN_MANIFEST = """\
x2w-manifest 2
stage cs
group 8 1 1
ud 3
udsrc 0 uav_descword3 0
udsrc 1 uav_stride 0
udsrc 2 uav_numrecords 0
uav 0 table 1 32
"""


def token_reference(first=None, word3=0x30000fac, stride=0, records=32):
    output = [0x63140000 + i for i in range(C.WORDS)]
    if first is not None:
        for lane in range(8):
            output[first + lane] = 0x71243000 ^ word3 ^ (stride << 8) ^ records ^ lane
    return output


def output_view_tokens(consumer, dxc):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
        env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
        env.pop('X2W_OUTPUT_VIEW_TOKENS', None)
        root = directory / 'compute_root'
        root.write_bytes(C.root_blob(3, table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(root)
        # The slot 67 code identifies this synthetic translation. The test supplies the
        # compiled HLSL directly so it exercises the runtime binder independently of lowering.
        (directory / 'program').write_bytes(struct.pack('<2I', 0xbf800025, 0xbf810000))  # s_nop 37; s_endpgm
        (directory / 'data').write_bytes(b'output-token-fixture')
        (directory / 'state').write_bytes(state_a((8, 1, 1), 0))
        mode = 'dispatch-output-token-raw'
        got, result = dispatch(directory, consumer, env, mode)
        print(result.stdout, end='')
        assert got == token_reference()
        assert len(lines(result, 'Dispatch(1, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
        [request] = lines(result, 'x2w-draw compute request')
        key = request.split()[3]
        source = directory / 'token.hlsl'
        source.write_text(TOKEN_SHADER, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cache / (key + '.cs.cso'), source], check=True)
        (cache / (key + '.txt')).write_text(TOKEN_MANIFEST, encoding='ascii')

        got, result = dispatch(directory, consumer, env, mode)
        assert got == token_reference()
        assert len(lines(result, 'Dispatch(1, 1, 1) refused: an output is a view token, not a buffer descriptor')) == 1, result.stderr[-3000:]
        assert not lines(result, 'Dispatch(1, 1, 1) translated')

        env['X2W_OUTPUT_VIEW_TOKENS'] = '1'
        for variant, first, word3, stride, records in (
            ('raw', 8, 0x30000fac, 0, 32),
            ('structured', 8, 0xfac, 8, 4),
            ('typed', 8, 0x01016fac, 4, 8),
            ('shifted', 12, 0x30000fac, 0, 32),
        ):
            got, result = dispatch(directory, consumer, env, 'dispatch-output-token-' + variant)
            assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1 and not lines(result, 'Dispatch(1, 1, 1) refused'), result.stderr[-3000:]
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors[0]
            expected = token_reference(first, word3, stride, records)
            wrong = [i for i, (actual, want) in enumerate(zip(got, expected, strict=True)) if actual != want]
            assert not wrong, f'{variant}: first mismatch at word {wrong[0]}, {len(wrong)} total'
            if variant == 'shifted':
                mismatches = sum(a != b for a, b in zip(got, token_reference(8), strict=True))
                assert mismatches == 12, f'wrong-range control had {mismatches} mismatches'
                print(f'WARP wrong-range control: {mismatches} mismatches against the correct CPU reference')
            else:
                print(f'WARP {variant} UAV token: {len(got)} dwords, zero mismatches')
    print('WARP switch-off control: original refusal and untouched output')


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[3:] == ['warp', '--output-view-tokens']:
        output_view_tokens(*sys.argv[1:3])
        sys.exit(0)
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
