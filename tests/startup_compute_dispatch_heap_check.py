"""A translated compute program whose input is the resource heap record a root constant indexes (list slot 14).

usage: startup_compute_dispatch_heap_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer creates the program of startup_gfx2hlsl_heap_record_check.py as a device slot 67 pipeline, with a
root signature that indexes the resource heap directly, and records the descriptor heap, the pipeline, the root
signature, three root constants, one table and Dispatch(3, 2, 2). The output descriptor is at table byte 16 as in
startup_compute_dispatch_check.py. The input descriptor is only in heap record 2 (heap byte 64), which the third
root constant indexes; heap bytes 0 and 32 hold the output's descriptor, so a dispatch that ignores the index or
scales it by 16 is refused instead of passing. Three runs share one cache:

1. before translation: the request is published and the Dispatch is refused with "no translation";
2. after gfx2hlsl.py --pending with DXC: the manifest names the heap record and all 320 words equal the
   expectation of case A of startup_gfx2hlsl_compute_check.py (the program computes the same words);
3. dispatch-heap-outside (the constant is 0x07ffffff, a record past the heap): refused with one
   logged reason, every word keeps the sentinel and the input is unchanged.
A second program reads four typed elements from the heap. One cached translation is used for three desktop
views and two native descriptor records. Extended formats also run as both native records and typed views,
with an opt-out control. Unsupported desktop formats leave the output untouched. Mutated DXIL
must dispatch successfully and produce a readback mismatch against the same CPU oracle.
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
import startup_gfx2hlsl_heap_record_check as H
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

OUTSIDE = "a heap record is outside the list's view heap"


FORMATTED_ASM = """\
s_mov_b32 s11, 0
s_mov_b32 s10, s7
s_load_dwordx4 s[24:27], s[10:11], 0x10
s_mov_b32 s9, 0
s_lshl_b32 s15, s6, 5
s_load_dwordx4 s[32:35], s[8:9], s15
s_waitcnt lgkmcnt(0)
buffer_load_format_xyzw v[12:15], v0, s[32:35], 0 idxen
v_lshlrev_b32 v7, 4, v0
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[12:15], v7, s[24:27], 0 offen
s_endpgm
"""
FORMATTED_CODE = bytes.fromhex(
    '80038bbe 07038abe 050608f4100000fa 800389be 06850f8f 040808f40000001e 7fc08cbf '
    '00200ce0000c0880 84000e34 703f8cbf 001078e0070c0680 000081bf'
)
FORMATTED_GROUP = (4, 1, 1)
FORMATTED_RSRC2 = 9 << 1
FORMATTED_BINDINGS = (
    'udsrc 3 stride 0\nudsrc 4 descword3 0\nsrv 0 buf heap rootconst 0 2\nuav 0 table 1 16\n'
)
FORMATTED_INDEXED_ASM = FORMATTED_ASM.replace('s_lshl_b32 s15, s6, 5', """\
s_load_dwordx4 s[40:43], s[10:11], 0x60
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s19, s[40:43], 0x34
s_waitcnt lgkmcnt(0)
s_lshl_b32 s15, s19, 5""")
FORMATTED_INDEXED_CODE = FORMATTED_CODE.replace(
    bytes.fromhex('06850f8f'), bytes.fromhex('050a08f4600000fa 7fc08cbf d40420f4340000fa 7fc08cbf 13850f8f')
)
FORMATTED_INDEXED_BINDINGS = (
    'udsrc 3 stride 1\nudsrc 4 descword3 1\nsrv 0 buf table 1 96\n'
    'srv 1 buf heap word table 1 96 at 52\nuav 0 table 1 16\n'
)
INPUT = struct.pack('<16e', -1, 1.5, 2, -2.5, 3, -3.5, 4, -4.5, 0.125, 0.25, 128, 16, 21, 31, -41, 51)
EXTENDED_INPUT = struct.pack('<4I', 0x816f01ff, 0x3c180f00, 0x401e0380, 0x7f008000)
EXTENDED_CODES = {'r32-uint': 20, 'snorm': 57, 'ufloat': 36}
EXTENDED_SELECT = 6 | 4 << 3 | 7 << 6 | 5 << 9  # z, x, w, y, as in the view tokens
NO_FORMAT = 'a formatted input view has no buffer format code'


def f32_word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def formatted_expected(kind):
    if kind == 'half':
        return [f32_word(value) for value in struct.unpack('<16e', INPUT)]
    if kind == 'packed':
        return [f32_word((word >> shift) % (1 << width) / ((1 << width) - 1))
                for word in struct.unpack_from('<4I', INPUT)
                for shift, width in ((0, 10), (10, 10), (20, 10), (30, 2))]
    if kind == 'uint':
        return [INPUT[4 * element + channel] for element in range(4) for channel in (2, 0, 3, 1)]
    if kind == 'r32-uint':
        return [channel for word in struct.unpack('<4I', EXTENDED_INPUT) for channel in (0, word, 1, 0)]
    if kind == 'snorm':
        return [f32_word(max(-1.0, value / 127))
                for element in struct.iter_unpack('<4b', EXTENDED_INPUT)
                for value in (element[2], element[0], element[3], element[1])]
    if kind == 'ufloat':
        # Expand each unsigned float to binary16, whose exponent has the same width and bias.
        values = []
        for (word,) in struct.iter_unpack('<I', EXTENDED_INPUT):
            x = struct.unpack('<e', struct.pack('<H', (word & 0x7ff) << 4))[0]
            y = struct.unpack('<e', struct.pack('<H', ((word >> 11) & 0x7ff) << 4))[0]
            z = struct.unpack('<e', struct.pack('<H', (word >> 22) << 5))[0]
            values.extend(map(f32_word, (z, x, 1.0, y)))
        return values
    assert kind == 'unknown'
    return [0] * 16


def check_formatted_dispatch(directory, consumer, dxc, adapter, indexed=False, extended=True):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    if extended:
        env.pop('X2W_EXTENDED_BUFFER_FORMATS', None)
    else:
        env['X2W_EXTENDED_BUFFER_FORMATS'] = '0'
    root = directory / 'compute_root'
    root.write_bytes(H.root_blob(table_range=True))
    (directory / 'program').write_bytes(FORMATTED_INDEXED_CODE if indexed else FORMATTED_CODE)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(FORMATTED_GROUP, FORMATTED_RSRC2))
    source = directory / 'input.bin'
    env.update(X2W_TEST_COMPUTE_ROOT=str(root), X2W_TEST_DISPATCH_IN=str(source))
    untouched = [C.SENTINEL] * C.WORDS

    def run(kind, hardware_word=None):
        payload = EXTENDED_INPUT if kind in EXTENDED_CODES else INPUT
        source.write_bytes(payload.ljust(C.WORDS * 4, b'\0'))
        binding = dict(env)
        if hardware_word is not None:
            fields = directory / 'fields.bin'
            stride = 4 if kind in EXTENDED_CODES else 8
            fields.write_bytes(struct.pack('<6I', stride, 4, hardware_word, 4, C.WORDS, 0))
            binding['X2W_TEST_DISPATCH_FIELDS'] = str(fields)
        mode = 'indexed' if indexed else 'plain' if hardware_word is not None else kind
        return dispatch(directory, consumer, binding, 'dispatch-heap-format-' + mode)

    def mismatches(kind, hardware_word=None, expected_kind=None):
        got, result = run(kind, hardware_word)
        assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-3000:]
        assert not lines(result, 'refused: '), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors
        assert got[16:] == untouched[16:], f'{kind}: wrote past the four output elements'
        # DXC can lower division by a constant to reciprocal multiplication.
        tolerance = 1 if kind == 'packed' else 0
        return [(i, actual, expected) for i, (actual, expected) in enumerate(zip(got, formatted_expected(expected_kind or kind)))
                if abs(actual - expected) > tolerance]

    got, result = run('half')
    assert got == untouched and len(lines(result, 'Dispatch(1, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
    key = lines(result, 'x2w-draw compute request')[0].split()[3]
    translated = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                capture_output=True, text=True, env=dict(env, X2W_DXC=dxc)).stdout
    assert f'{key}: ok' in translated, translated
    manifest = (cache / f'{key}.txt').read_text()
    assert manifest.endswith(FORMATTED_INDEXED_BINDINGS if indexed else FORMATTED_BINDINGS), manifest
    if indexed:
        wrong = mismatches('half')
        assert not wrong, wrong
        print(f'formatted heap on {adapter}: table-word index matched the half-float reference')
        return

    for kind, code in EXTENDED_CODES.items():
        # Restoring kind != 'cs' yields zeros and fails readback. Dropping extended
        # at the runtime token call site refuses the typed-view leg.
        native = code << 12 | EXTENDED_SELECT
        if extended:
            for binding, word in (('native', native), ('typed view', None)):
                wrong = mismatches(kind, word)
                assert not wrong, (kind, binding, 'first mismatch (index, actual, expected)', wrong[0])
        else:
            assert not mismatches(kind, native, expected_kind='unknown'), (kind, 'opt-out native readback mismatch')
            got, result = run(kind)
            assert got == untouched, (kind, 'opt-out view wrote the output')
            refused = lines(result, 'refused: ')
            assert len(refused) == 1 and refused[0].endswith(NO_FORMAT), result.stderr[-3000:]
    if not extended:
        print(f'formatted heap on {adapter}: opt-out native zeros and typed-view refusals passed')
        return key

    for kind in ('half', 'packed', 'uint'):
        wrong = mismatches(kind)
        assert not wrong, (kind, wrong)
    got, result = run('unsupported')
    assert got == untouched, 'the unsupported view wrote the output'
    refused = lines(result, 'refused: ')
    assert len(refused) == 1 and refused[0].endswith(NO_FORMAT), result.stderr[-3000:]

    identity = 4 | 5 << 3 | 6 << 6 | 7 << 9
    assert not mismatches('half', 71 << 12 | identity)
    # An unknown native code reaches the HLSL default even with a forced-one selector.
    assert not mismatches('unknown', 127 << 12 | (identity & ~7) | 1)

    hlsl = (cache / f'{key}.cs.hlsl').read_text()
    bytecode = cache / f'{key}.cs.cso'
    mutant = directory / 'mutant.hlsl'
    for old, new, cases in (
        ('switch ((ud4 >> 12) & 127u)', 'switch (71u)', ('packed', 'uint')),
        ('v0 * ud3', 'v0 * 4u', ('half',)),
    ):
        assert hlsl.count(old) == 1, old
        mutant.write_text(hlsl.replace(old, new))
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', bytecode, mutant], check=True)
        if cases == ('packed', 'uint'):
            wrong = mismatches('half')
            assert not wrong, ('frozen format failed on the original binding', wrong)
        for kind in cases:
            wrong = mismatches(kind)
            assert wrong, (new, kind, 'mutant matched the CPU reference')
            print(f'{kind}: mutant {new!r} differs at word {wrong[0][0]}')

    print(f'formatted heap on {adapter}: six typed views, five native records and the named refusal passed')
    return key


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')  # guest lists; debug-layer messages go to the log
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        (directory / 'compute_root').write_bytes(H.root_blob(table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        assert H.GROUPS == (3, 2, 2) and H.INDEX_WORD == 2, 'the consumer hard-codes the dispatch and the index word'
        (directory / 'program').write_bytes(H.WORDS)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(H.GROUP, H.RSRC2))
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        assert expected != untouched

        got, result = dispatch(directory, consumer, env, 'dispatch-heap')
        print(result.stdout, end='')
        assert got == untouched, 'a Dispatch without a translation wrote the output'
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out
        assert (cache / f'{key:016x}.txt').read_text() == H.MANIFEST

        got, result = dispatch(directory, consumer, env, 'dispatch-heap')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        got, result = dispatch(directory, consumer, env, 'dispatch-heap-outside')
        assert got == untouched, 'a refused Dispatch wrote the output'
        refused = lines(result, 'refused: ')
        census = lines(result, 'x2w-render compute census')
        assert len(refused) == 1 and OUTSIDE in refused[0], result.stderr[-3000:]
        assert len(census) == 3 and census[1].endswith(f'reason "{OUTSIDE}": 1 keys, 1 dispatches'), census
        print(f'startup_compute_dispatch_heap: PASS on {adapter} (all {C.WORDS} words match through heap record 2; '
              'untouched before translation and for a record outside the heap)')
        enabled_key = check_formatted_dispatch(directory / 'formatted', consumer, dxc, adapter)
        disabled_key = check_formatted_dispatch(directory / 'formatted-off', consumer, dxc, adapter, extended=False)
        assert enabled_key != disabled_key, 'compute cache key ignored the extended format setting'
        check_formatted_dispatch(directory / 'formatted-indexed', consumer, dxc, adapter, indexed=True)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
