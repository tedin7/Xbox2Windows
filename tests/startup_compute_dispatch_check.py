"""A translated compute program dispatched through the stand-in's command list (list slot 14).

usage: startup_compute_dispatch_check.py <startup_stream_consumer> <dxc> <warp|hardware>
       startup_compute_dispatch_check.py controls

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


def dispatch(directory, consumer, env, mode):
    out = directory / 'out.bin'
    out.unlink(missing_ok=True)
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', mode]
    result = subprocess.run(args, env=dict(env, X2W_TEST_DISPATCH_OUT=str(out)), capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    words = 2 * C.WORDS if mode.startswith('dispatch-root-views') else C.WORDS
    return list(struct.unpack('<%dI' % words, out.read_bytes())), result


def lines(result, text):
    return [line for line in result.stderr.splitlines() if text in line]


def prepare_case(directory, adapter):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    else:
        env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, words, rsrc2, group, groups = C.CASES['A']
    assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
    (directory / 'program').write_bytes(words)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    return cache, env


def root_view_signature():
    types = (3, 4, 2)  # SRV, UAV, CBV
    params_at = 24
    desc_at = params_at + 12 * len(types)
    rts = struct.pack('<6I', 1, len(types), params_at, 0, desc_at + 8 * len(types), 0)
    rts += b''.join(struct.pack('<3I', kind, 0, desc_at + 8 * i) for i, kind in enumerate(types))
    rts += struct.pack('<6I', 0, 0, 0, 0, 0, 0)
    part = b'RTS0' + struct.pack('<I', len(rts)) + rts
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(part), 1, 36) + part


def check_root_translation():
    # A five-lane program with a register offset and separate constant offsets on its load and store.
    words = (
        0x34180083,              # v_lshlrev_b32 v12, 3, v0
        0xdc308004, 0x0d04000c,  # global_load_dword v13, v12, s[4:5] offset:4
        0xf4000384, 0xfa000000,  # s_load_dword s14, s[8:9], 0
        0xdc70800c, 0x00060d0c,  # global_store_dword v12, v13, s[6:7] offset:12
        0xbf810000,              # s_endpgm
    )

    def program(code):
        return G.ComputePipeline(struct.pack('<8I', *code), root_view_signature(), 10 << 1, (5, 1, 1))

    translated = program(words)
    manifest = translated.manifest()
    assert 'srv 0 buf root 0\n' in manifest and 'srv 1 buf root 2\n' in manifest, manifest
    assert 'uav 0 root 1\n' in manifest, manifest
    hlsl = translated.hlsl_cs()
    assert 'r0.Load(a + 4u)' in hlsl and 'r1.Load(0)' in hlsl and 'u0.Store(a + 12u, v13)' in hlsl, hlsl
    invalid = (*words[:6], words[6] ^ (2 << 16), words[7])  # redirect the scalar base to the root SRV
    try:
        program(invalid)
    except G.Unsupported as exc:
        assert 'global store requires a root UAV' in str(exc), exc
    else:
        raise AssertionError('a GLOBAL store through a root SRV translated')
    print('startup_compute_dispatch root translation: PASS')


def check_root_views(directory, consumer, dxc):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    env['X2W_NOLOCK_REFUSE'] = '0'
    env.pop('X2W_COMPUTE_ROOT_VIEWS', None)
    root = directory / 'compute_root'
    root.write_bytes(root_view_signature())
    env['X2W_TEST_COMPUTE_ROOT'] = str(root)
    (directory / 'program').write_bytes(struct.pack('<I', 0xbf810000))  # s_endpgm, for a synthetic request key
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a((5, 1, 1), 10 << 1))
    source = [0x610fe3 + 29 * i for i in range(C.WORDS)]
    source_path = directory / 'input.bin'
    source_path.write_bytes(struct.pack('<%dI' % C.WORDS, *source))
    env['X2W_TEST_DISPATCH_IN'] = str(source_path)
    env.pop('X2W_TEST_DISPATCH_FIELDS', None)
    untouched = [C.SENTINEL] * C.WORDS

    def run(mode='dispatch-root-views', enabled=True):
        run_env = dict(env, X2W_COMPUTE_ROOT_VIEWS='1') if enabled else env
        got, result = dispatch(directory, consumer, run_env, mode)
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors
        for slot, calls, fitting in ((41, 1, 1), (43, 1, 1), (45, 2, 1), (46, 1, 1)):
            expected = f'x2w-render root view slot {slot}: calls={calls} fitting={fitting} placed={calls} upload=0 unknown=0'
            assert expected in lines(result, f'root view slot {slot}:'), result.stderr[-4000:]
        return got, result

    got, result = run(enabled=False)
    assert got == source + untouched
    assert len(lines(result, 'Dispatch(1, 1, 1) refused: no translation')) == 1, result.stderr[-4000:]
    requests = lines(result, 'x2w-draw compute request')
    assert requests and ' published ' in requests[0], result.stderr[-4000:]
    key = requests[0].split()[3]
    manifest = cache / f'{key}.txt'
    shader = directory / 'root_views.hlsl'

    def publish(mutant):
        declarations = ('ByteAddressBuffer bias : register(t0);' if mutant else
                        'ByteAddressBuffer records : register(t0); ByteAddressBuffer bias : register(t1);')
        # In the misbound variant load the same records through the UAV to avoid simultaneous SRV/UAV states.
        reader = 'result' if mutant else 'records'
        shader.write_text(f"""{declarations}
RWByteAddressBuffer result : register(u0);
[numthreads(5, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    uint at = 8 * id.x;
    result.Store(at, {reader}.Load(at) + bias.Load(0));
}}
""", encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cache / f'{key}.cs.cso'), str(shader)],
                       check=True, capture_output=True, text=True)
        inputs = 'srv 0 buf root 2\n' if mutant else 'srv 0 buf root 0\nsrv 1 buf root 2\n'
        manifest.write_text('x2w-manifest 2\nstage cs\ngroup 5 1 1\nud 0\n' + inputs + 'uav 0 root 1\n',
                            encoding='ascii')

    publish(False)
    expected_output = untouched.copy()
    written_input = source.copy()
    for lane in range(5):
        value = source[8 + 2 * lane] + 37
        expected_output[12 + 2 * lane] = value
        written_input[8 + 2 * lane] = value
    expected = source + expected_output

    def compare(got):
        assert got == expected, next((i, actual, reference) for i, (actual, reference)
                                     in enumerate(zip(got, expected)) if actual != reference)

    got, result = run()
    compare(got)
    assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-4000:]
    assert lines(result, 'compute root view dispatches:') == ['x2w-render compute root view dispatches: input=1 output=1']
    for mode, enabled, reason in (
        ('dispatch-root-views', False, 'a resource that is not a raw table buffer, a 2D texture or a cube map'),
        ('dispatch-root-views-changed', True, 'a compute root view is not bound'),
    ):
        got, result = run(mode, enabled)
        assert got == source + untouched
        assert len(lines(result, 'Dispatch(1, 1, 1) refused: ' + reason)) == 1, result.stderr[-4000:]
        assert lines(result, 'compute root view dispatches:') == ['x2w-render compute root view dispatches: input=0 output=0']

    publish(True)
    got, result = run('dispatch-root-views-mutant')
    assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-4000:]
    assert got == written_input + untouched, 'misbound UAV did not write exactly the five input words'
    try:
        compare(got)
    except AssertionError as exc:
        assert exc.args == ((8, written_input[8], source[8]),), exc
    else:
        raise AssertionError('the wrong-address mutant passed the CPU oracle')
    print('startup_compute_dispatch root views: PASS on WARP (GPU readback, two refusals and wrong-address mutant)')


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


if __name__ == '__main__':
    if sys.argv[1:] == ['controls']:
        check_root_translation()
        sys.exit(0)
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
    if sys.argv[3] == 'warp':
        check_root_translation()
        with tempfile.TemporaryDirectory() as temporary:
            check_root_views(Path(temporary), *sys.argv[1:3])
