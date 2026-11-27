"""Named reasons for a translated compute input that is not a complete GPU-backed buffer range (list slot 14).

usage: startup_compute_dispatch_input_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer records case A of startup_gfx2hlsl_compute_check.py as startup_compute_dispatch_check.py does. After
gfx2hlsl.py --pending with DXC, the unchanged recording still matches all 320 words. Then one input descriptor at a
time is replaced, and each dispatch is refused with its own reason, logs which input and descriptor once, writes
no output word and leaves the input as written:

- dispatch-input-empty: an unfilled heap slot, submitted twice to check one diagnostic;
- dispatch-input-past: 16 bytes past the end of the placed input buffer;
- dispatch-input-unplaced: memory between the two placed buffers;
- dispatch-input-copied: a buffer placed at an offset the desktop cannot alias, so it is a separate copy;
- dispatch-input-token: the token of a texture view;
- dispatch-input-null-token: a buffer SRV token with no resource. It runs only with
  X2W_NULL_TOKEN_ZERO=1, when all output words must match the CPU reference for zero input.

Last, dispatch-input-memory-view replaces the input descriptor with the token of a buffer view (device slot 70) of
memory no buffer is placed over, 16 bytes in, and writes the input words there after the view exists: all 320
words match. The stand-in places that buffer in the consumer's own heap, outside its placement index.
The title-input switch cases use a second program and a D3D12 readback heap. The CPU sum covers three records,
including one in the final 16 bytes. A first-chunk-only copy must fail the same comparison.
The view-token case reads a selected texture mip and writes a selected raw UAV range.
Exit 77 when the adapter is not available.
"""
from math import prod
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

CONTROLS = (
    ('dispatch-input-empty', 'an input buffer descriptor has zero records'),
    ('dispatch-input-past', 'an input range crosses the end of a placed buffer'),
    ('dispatch-input-unplaced', 'an input is in title memory outside any placed buffer'),
    ('dispatch-input-copied', 'an input is in a placed buffer that does not alias title memory'),
    ('dispatch-input-token', 'an input view token is not a buffer view of a desktop resource'),
    ('dispatch-input-null-token', 'an input view token is not a buffer view of a desktop resource'),
)


# One thread sums three four-dword records. The fixture supplies the middle and tail byte offsets as root constants.
SUM_SOURCE = """\
s_mov_b32 s20, s6
s_mov_b32 s21, 0
s_load_dwordx8 s[24:31], s[20:21], 0
s_waitcnt lgkmcnt(0)
v_mov_b32 v8, 0
v_mov_b32 v9, s4
v_mov_b32 v10, s5
buffer_load_dwordx4 v[12:15], v8, s[24:27], 0 offen
buffer_load_dwordx4 v[16:19], v9, s[24:27], 0 offen
buffer_load_dwordx4 v[20:23], v10, s[24:27], 0 offen
s_waitcnt vmcnt(0)
""" + ''.join(f'v_add_nc_u32 v12, v12, v{i}\n' for i in range(13, 24)) + """\
buffer_store_dword v12, v8, s[28:31], 0 offen
s_endpgm
"""
SUM_CODE = bytes.fromhex(
    '060394be 800395be 0a060cf4 000000fa 7fc08cbf 8002107e 0402127e 0502147e '
    '001038e0 080c0680 001038e0 09100680 001038e0 0a140680 703f8cbf '
    '0c1b184a 0c1d184a 0c1f184a 0c21184a 0c23184a 0c25184a '
    '0c27184a 0c29184a 0c2b184a 0c2d184a 0c2f184a 001070e0 080c0780 000081bf')


def check_title_inputs(directory, cache, consumer, dxc, env, untouched):
    env = dict(env, X2W_TEST_UPLOAD_RING_MAX_KIB='64', X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT='1')
    enabled = dict(env, X2W_COMPUTE_TITLE_INPUTS='1')
    (directory / 'program').write_bytes(SUM_CODE)
    (directory / 'state').write_bytes(state_a((1, 1, 1), C.USER7))

    def run(mode, settings):
        got, result = dispatch(directory, consumer, settings, 'dispatch-input-title-' + mode)
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors
        return got, result

    def refused(mode, settings, reason):
        got, result = run(mode, settings)
        messages = lines(result, 'Dispatch(1, 1, 1) refused: ')
        assert got == untouched and len(messages) == 1 and messages[0].endswith('refused: ' + reason), (
            mode, got[0], result.stderr[-3000:])
        return result

    cold = refused('plain', env, 'no translation')
    key, = [line.split()[3] for line in lines(cold, 'x2w-draw compute request') if ' published ' in line]
    compiler = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                              capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
    assert f'{key}: ok' in compiler.stdout, compiler.stdout

    expected = [sum((r + 1) * 37 + i * 9 for r in range(3) for i in range(4))] + untouched[1:]

    def compare(got, mode):
        assert got == expected, f'{mode}: readback sum {got[0]}, CPU sum {expected[0]}'

    for mode, reason in (
        ('plain', 'an input is in title memory outside any placed buffer'),
        ('large', 'an input buffer range is over 16 MiB'),
        ('ring', 'an input range could not be copied from title memory'),
    ):
        refused(mode, env, reason)

    for mode, gpu, placed, kib, plain, retries in (
        ('plain', 0, 0, 0, 1, 0),
        ('large', 0, 1, 40960, 0, 0),
        ('gpu', 1, 0, 0, 0, 0),
        ('alias', 0, 0, 0, 0, 0),
        ('ring', 0, 1, 12, 0, 1),
    ):
        got, result = run(mode, enabled)
        assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-3000:]
        totals, = lines(result, 'x2w-render compute dispatches:')
        assert totals.endswith(f'from a GPU copy {gpu}, from a title snapshot {placed} ({kib} KiB), '
                               f'from plain title memory {plain}, ring retries {retries}, ring full 0'), totals
        compare(got, mode)

    refused('protected', enabled, 'an input range in title memory is not readable')
    full = refused('ring-full', enabled, 'an input range could not be copied from title memory (ring full)')
    totals, = lines(full, 'x2w-render compute dispatches:')
    assert totals.endswith('from plain title memory 0, ring retries 0, ring full 1'), totals

    got, result = run('large', dict(enabled, X2W_TEST_COMPUTE_INPUT_FIRST_CHUNK_ONLY='1'))
    assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-3000:]
    try:
        compare(got, 'first-chunk-only mutant')
    except AssertionError:
        assert got == [sum(37 + i * 9 for i in range(4))] + untouched[1:], got
    else:
        raise AssertionError('first-chunk-only mutant unexpectedly passed the CPU comparison')
    print('title inputs: five CPU matches, three switch-off refusals, protected page and ring-full refusals, mutant caught')


def check_view_tokens(directory, cache, consumer, dxc, env, untouched):
    mode = 'dispatch-input-view-tokens'
    # The title program identifies the synthetic request. The fixture supplies its desktop translation directly.
    (directory / 'program').write_bytes(bytes.fromhex('000081bf'))  # s_endpgm
    (directory / 'state').write_bytes(state_a((1, 1, 1), C.USER7))
    got, result = dispatch(directory, consumer, env, mode)
    assert got == untouched and len(lines(result, 'refused: no translation')) == 1, result.stderr[-3000:]
    key, = [line.split()[3] for line in lines(result, 'x2w-draw compute request') if ' published ' in line]
    shader = directory / 'view_tokens.hlsl'
    shader.write_text("""\
Texture2D<float4> source : register(t0);
RWByteAddressBuffer target : register(u0);
[numthreads(1, 1, 1)]
void main() {
    uint4 rgba = uint4(round(source.Load(int3(0, 0, 0)) * 255.0));
    target.Store2(0, uint2(rgba.x + 257 * rgba.y, rgba.z + 257 * rgba.w));
}
""")
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cache / f'{key}.cs.cso', shader], check=True)
    (cache / f'{key}.txt').write_text(
        'x2w-manifest 2\nstage cs\nud 0\nsrv 0 tex2d table 1 64\nuav 0 table 1 0\n')

    def run(suffix='', *, enabled=True, refusal=None):
        settings = dict(env, X2W_COMPUTE_VIEW_TOKENS='1') if enabled else env
        readback, log = dispatch(directory, consumer, settings, mode + suffix)
        assert not [line for line in lines(log, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line], log.stderr[-3000:]
        refused = lines(log, 'Dispatch(1, 1, 1) refused: ')
        if refusal:
            assert readback == untouched and len(refused) == 1 and refused[0].endswith(refusal), log.stderr[-3000:]
        else:
            assert not refused and len(lines(log, 'Dispatch(1, 1, 1) translated')) == 1, log.stderr[-3000:]
        return readback

    run(enabled=False, refusal='an output is a view token, not a buffer descriptor')
    run('-wrong-kind', refusal='an output shader resource view token cannot supply unordered access')
    rgba = (17, 93, 201, 45)
    value = [rgba[0] + 257 * rgba[1], rgba[2] + 257 * rgba[3]]

    def expected_at(index):
        expected = untouched.copy()
        expected[index:index + len(value)] = value
        return expected

    def compare(readback):
        expected = expected_at(8)
        for i, (actual, reference) in enumerate(zip(readback, expected, strict=True)):
            assert actual == reference, f'dword {i}: read {actual:#x}, expected {reference:#x}'

    compare(run())
    shifted = run('-shifted')
    assert shifted == expected_at(12), 'the shifted UAV must write the same values at its own offset'
    try:
        compare(shifted)
    except AssertionError as mismatch:
        print(f'view tokens: CPU readback matches and wrong-offset control fails at {mismatch}')
    else:
        raise AssertionError('wrong-offset UAV unexpectedly passed the CPU comparison')


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        for variable in ('X2W_COMPUTE_VIEW_TOKENS', 'X2W_NULL_TOKEN_ZERO', 'X2W_COMPUTE_TITLE_INPUTS', 'X2W_TEST_COMPUTE_INPUT_FIRST_CHUNK_ONLY',
                         'X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT', 'X2W_TEST_UPLOAD_RING_MAX_KIB',
                         'X2W_TRANSLATED_INDIRECT_ADDRESS_ALIAS', 'X2W_TRANSLATED_INDIRECT_COPY_RANGES',
                         'X2W_D3D12_LINEAR_BUFFERS', 'X2W_TEST_DISPATCH_HARDWARE'):
            env.pop(variable, None)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')  # guest lists; debug-layer messages go to the log
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        _, words, rsrc2, group, groups = C.CASES['A']
        assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
        (directory / 'program').write_bytes(words)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(group, rsrc2))
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')

        got, result = dispatch(directory, consumer, env, 'dispatch')
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
        used = lines(result, f'x2w-render key use compute {key:016x} first ')
        assert len(used) == 1 and used[0].endswith(' uses 1') and ' last ' in used[0], (used, result.stderr[-3000:])
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        for mode, reason in CONTROLS:
            got, result = dispatch(directory, consumer, env, mode)
            assert got == untouched, '%s: a refused Dispatch wrote the output' % mode
            refused = lines(result, 'refused: ')
            census = lines(result, 'x2w-render compute census')
            where = lines(result, f'x2w-draw compute {key:016x} first refused input 0 (table ')
            calls = 2 if mode == 'dispatch-input-empty' else 1
            assert len(refused) == calls and all(line.endswith('refused: ' + reason) for line in refused), (mode, refused)
            assert len(census) == 3 and census[1].endswith(f'reason "{reason}": 1 keys, {calls} dispatches'), (mode, census)
            assert len(where) == 1, (mode, result.stderr[-3000:])
            slot_lines = lines(result, 'unfilled heap slot ')
            assert slot_lines == ([f'x2w-draw unfilled heap slot {0x32c - 1} key {key:016x}']
                                  if mode == 'dispatch-input-empty' else []), (mode, slot_lines)

        null_env = dict(env, X2W_NULL_TOKEN_ZERO='1')
        got, result = dispatch(directory, consumer, null_env, 'dispatch-input-empty')
        assert got == untouched and not lines(result, 'Dispatch(3, 2, 2) translated'), result.stderr[-3000:]
        assert lines(result, 'unfilled heap slot ') == [f'x2w-draw unfilled heap slot {0x32c - 1} key {key:016x}'], result.stderr[-3000:]

        got, result = dispatch(directory, consumer, dict(env, X2W_COMPUTE_VIEW_TOKENS='1'), 'dispatch-input-token')
        refusal, = lines(result, 'refused: ')
        assert got == untouched and refusal.endswith(
            'an input shader resource view token describes a texture, but the shader needs a buffer'), result.stderr[-3000:]

        got, result = dispatch(directory, consumer, null_env, 'dispatch-input-null-token')
        written = prod(group) * prod(groups)
        zero_expected = [(C.KEY + i * C.STRIDE) & C.MASK for i in range(written)]
        zero_expected += untouched[written:]
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert got == zero_expected, 'null-token output differs from the CPU reference for zero input'
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        got, result = dispatch(directory, consumer, env, 'dispatch-input-memory-view')
        assert got == expected and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and             not lines(result, 'refused: '), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        # Opt-in X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT: an input in a placed buffer that does not alias title memory holds
        # its bytes only in title memory (the desktop copy is zeros). Off, it is refused; on, the bytes are snapshotted
        # into the upload ring and all 320 words match; with title memory zero (dispatch-input-copied) the dispatch runs
        # on zeros: not refused, not the expected words, not untouched.
        snapshot_env = dict(env, X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT='1')
        got, result = dispatch(directory, consumer, env, 'dispatch-input-snapshot')
        assert got == untouched and lines(result, 'refused: ')[0].endswith(CONTROLS[3][1]), result.stderr[-3000:]
        got, result = dispatch(directory, consumer, snapshot_env, 'dispatch-input-snapshot')
        assert got == expected and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert len(lines(result, 'comes from a snapshot of title memory (1280 bytes)')) == 1, result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        got, result = dispatch(directory, consumer, snapshot_env, 'dispatch-input-copied')
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert got != expected and got != untouched, 'a snapshot of zero title memory must not give the expected words'
        # An all-zero descriptor is an unbound buffer: the dispatch runs on zeros instead of being refused (the empty
        # control above has a nonzero base and stays refused).
        got, result = dispatch(directory, consumer, env, 'dispatch-input-null')
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert len(lines(result, 'comes from zeros for a null descriptor (16 bytes)')) == 1, result.stderr[-3000:]
        assert got == zero_expected, 'all-zero descriptor output differs from the CPU reference for zero input'
        # A manifest whose first input is out of order is not bound, and the log names the line: the pipeline key, the
        # line number, the reason and the line itself.
        manifest = cache / f'{key:016x}.txt'
        original = manifest.read_text()
        assert 'srv 0 ' in original, original
        manifest.write_text(original.replace('srv 0 ', 'srv 3 ', 1))
        got, result = dispatch(directory, consumer, env, 'dispatch')
        manifest.write_text(original)
        first = next(l for l in original.splitlines() if l.startswith('srv 0 ')).replace('srv 0 ', 'srv 3 ', 1)
        rejected = lines(result, f'x2w-draw compute pipeline {key:016x} manifest line ')
        assert got == untouched and len(rejected) == 1 and ' rejected (a resource that is not a raw table buffer' in rejected[0] \
            and rejected[0].endswith(': ' + first[:120]), result.stderr[-3000:]
        print(f'{adapter}: title-input snapshot: off refused, on all {C.WORDS} words match, zero title memory runs on zeros')
        print(f'startup_compute_dispatch_input: baseline PASS on {adapter} (all {C.WORDS} words match, also with the input behind '
              f'a memory view token; {len(CONTROLS)} inputs refused, each with its own reason and an untouched output)')
        check_title_inputs(directory, cache, consumer, dxc, env, untouched)
        check_view_tokens(directory, cache, consumer, dxc, env, untouched)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        assert C.assemble(sys.argv[2], SUM_SOURCE) == SUM_CODE
        pipe = G.ComputePipeline(SUM_CODE, C.root_blob(table_range=True), C.USER7, (1, 1, 1))
        assert 'srv 0 buf table 1 0\nuav 0 table 1 16' in pipe.manifest()
        assert pipe.hlsl_cs().count('= r0.Load(') == 12
        print('title input sum fixture assembles and translates')
        sys.exit(0)
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
