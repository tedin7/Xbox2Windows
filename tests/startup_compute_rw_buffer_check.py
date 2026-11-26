"""Source-built gfx1013 compute fixtures: one buffer descriptor that the program both loads and stores through.

  (no arguments)                                decode and translate, check the HLSL and the manifest (pure Python)
  --llvm-mc <llvm-mc>                           the same, with the fixtures assembled by llvm-mc
  --dispatch <dxc> <offscreen_check> <adapter>  run raw GPU readback against a CPU reference
  --runtime-dispatch <startup_stream_consumer> <dxc> <adapter>
                                               run formatted GPU readback through the startup runtime

The old translation refused such a program ("used for a load and a store") because a read-only view and a UAV of one
buffer cannot be bound in one dispatch. The loads now read the same RWByteAddressBuffer the stores write.

  load_then_store: out[i] = out[i] + 7        (the runner fills the output with the sentinel word)
  store_then_load: out[i] = i; out[i] = out[i] + 7   (a store followed by a load of the word it wrote)
  formatted: add four binary16 increments from a separate buffer to each selected output record
"""
import argparse
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import startup_compute_readfirstlane_check as rf  # noqa: E402
from startup_gfx2hlsl_compute_check import G, KEY, MASK, SENTINEL, STRIDE, WORDS, assemble, refused, root_blob  # noqa: E402

PROLOGUE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s27, s7, 2
v_add_nc_u32_e32 v4, s27, v0
v_mov_b32_e32 v6, v4
v_lshlrev_b32_e32 v4, 2, v4
"""
LOAD_THEN_STORE = PROLOGUE + """\
buffer_load_dword v5, v4, s[16:19], 0 offen
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v5, 7, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
STORE_THEN_LOAD = PROLOGUE + """\
buffer_store_dword v6, v4, s[16:19], 0 offen
buffer_load_dword v5, v4, s[16:19], 0 offen
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v5, 7, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
IDXEN_LOAD = LOAD_THEN_STORE.replace('buffer_load_dword v5, v4, s[16:19], 0 offen', 'buffer_load_dword v5, v6, s[16:19], 0 idxen')
FORMATTED = """\
s_mov_b64 s[20:21], s[6:7]
s_mov_b32 s21, 0
s_load_dwordx8 s[24:31], s[20:21], 0x0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v1, 1, v0
v_lshlrev_b32_e32 v2, 3, v0
buffer_load_format_xyzw v[8:11], v1, s[28:31], 0 idxen
buffer_load_dwordx2 v[14:15], v2, s[24:27], 0 offen
s_waitcnt vmcnt(0)
v_cvt_pkrtz_f16_f32 v12, v8, v9
v_cvt_pkrtz_f16_f32 v13, v10, v11
v_pk_add_f16 v16, v12, v14
v_pk_add_f16 v17, v13, v15
v_lshrrev_b32_e32 v18, 16, v16
v_lshrrev_b32_e32 v19, 16, v17
v_cvt_f32_f16_e32 v8, v16
v_cvt_f32_f16_e32 v9, v18
v_cvt_f32_f16_e32 v10, v17
v_cvt_f32_f16_e32 v11, v19
buffer_store_format_xyzw v[8:11], v1, s[28:31], 0 idxen
s_endpgm
"""
THREADS = rf.LOCAL[0] * rf.GROUPS[0]
FIXTURES = {'load_then_store': LOAD_THEN_STORE, 'store_then_load': STORE_THEN_LOAD, 'idxen_load': IDXEN_LOAD, 'formatted': FORMATTED}
# The assembled words, so the test needs no assembler; --llvm-mc reassembles and compares.
WORDS_HEX = {
    'load_then_store': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001030e004050480'
                       '703f8cbf870a0a4a001070e004050480000081bf',
    'store_then_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001070e004060480'
                       '001030e004050480703f8cbf870a0a4a001070e004050480000081bf',
    'formatted': '060494be800395be0a060cf4000000fa7fc08cbf810002348300043400200ce001080780001034e0020e0680'
                 '703f8cbf0813185e0a171a5e10400fcc0c1d021811400fcc0d1f02189020242c9022262c1017107e1217127e'
                 '1117147e1317167e00201ce001080780000081bf',
    'idxen_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834002030e006050480'
                  '703f8cbf870a0a4a001070e004050480000081bf',
}


def translate(name):
    return G.ComputePipeline(bytes.fromhex(WORDS_HEX[name]), root_blob(), rf.RSRC2, rf.LOCAL)


def expected(name):
    out = [SENTINEL] * WORDS
    for i in range(THREADS):
        out[i] = ((SENTINEL if name == 'load_then_store' else i) + 7) & MASK
    return out


def controls(llvm_mc=None):
    if llvm_mc:
        for name, source in FIXTURES.items():
            built = assemble(llvm_mc, source).hex()
            assert built == WORDS_HEX[name], '%s: assembled %s' % (name, built)
        print('fixture words match %s' % llvm_mc)
    for name in ('load_then_store', 'store_then_load'):
        pipe = translate(name)
        hlsl = pipe.hlsl_cs()
        assert hlsl.count('RWByteAddressBuffer u0 : register(u0);') == 1, hlsl
        assert 'u0.Load(' in hlsl and 'u0.Store(' in hlsl, hlsl
        assert 'ByteAddressBuffer r0' not in hlsl and ' r0.' not in hlsl, hlsl
        manifest = pipe.manifest().splitlines()
        assert 'uav 0 table 1 16' in manifest, manifest
        assert not [line for line in manifest if line.startswith('srv ')], manifest
        print('%s: one RW view, loads and stores through u0' % name)
    # The descriptor stays refused where a read-only view is needed: a buffer index needs the SRV stride.
    refused('read-write descriptor', lambda: translate('idxen_load').hlsl_cs())
    pipe = translate('formatted')
    hlsl = pipe.hlsl_cs()
    assert hlsl.count('RWByteAddressBuffer u0 : register(u0);') == 1
    assert 'a = v1 * ud2 + 0;' in hlsl and 'ldu(u0, a)' in hlsl
    assert 'store_format_xyzw(u0, ud3, ud2, ud4, v1, 0u, 0u, 0u,' in hlsl
    assert pipe.manifest().splitlines()[6:] == [
        'udsrc 2 uav_stride 0', 'udsrc 3 uav_rw_word3 0', 'udsrc 4 uav_numrecords 0',
        'srv 0 buf table 1 0', 'uav 0 table 1 16',
    ], pipe.manifest()

    load = '00200ce001080780'
    store = '00201ce001080780'
    original = WORDS_HEX['formatted']
    store_first = bytes.fromhex(original.replace(store, '').replace(load, store + load))
    other_order = G.ComputePipeline(store_first, root_blob(), rf.RSRC2, rf.LOCAL)
    assert 'ldu(u0, a)' in other_order.hlsl_cs() and 'uav_rw_word3 0' in other_order.manifest()
    rewritten = bytes.fromhex(original.replace(store, '80039fbe' + store))  # s_mov_b32 s31, 0
    refused('modified descriptor', lambda: G.ComputePipeline(rewritten, root_blob(), rf.RSRC2, rf.LOCAL).hlsl_cs())
    # From the corresponding synthetic instructions with the same registers: raw load, inline
    # MTBUF format, plain store, and buffer_atomic_add. Each touches the shared descriptor.
    for target, replacement in ((load, '002038e001080780' + load),
                                (load, '00206bea01080780' + load),
                                (store, '001078e001080780'),
                                (store, store + '0010c8e001080780')):
        assert original.count(target) == 1
        code = bytes.fromhex(original.replace(target, replacement))
        refused('used for a load and a store',
                lambda: G.ComputePipeline(code, root_blob(), rf.RSRC2, rf.LOCAL).hlsl_cs())
    print('formatted: one UAV, one unrelated SRV, mixed accesses refused')
    print('rw buffer checks passed')


def dispatch(dxc, runner, adapter):
    controls()
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        for name in ('load_then_store', 'store_then_load'):
            hlsl = translate(name).hlsl_cs()
            case = td / (name + '.case')
            case.write_bytes(struct.pack('<I', rf.GROUPS[0] | (rf.GROUPS[1] << 8) | (rf.GROUPS[2] << 16))
                             + struct.pack('<%dI' % WORDS, *expected(name))
                             + struct.pack('<16I', KEY, STRIDE, *([0] * 14)))
            # The control reads zero instead of the buffer: the readback must then differ from the reference.
            mutant = re.sub(r'u0\.Load\(([^;]*)\);', r'(0u & u0.Load(\1));', hlsl, count=1)
            assert mutant != hlsl and mutant.count('(0u & u0.Load(') == 1, hlsl
            for label, text in (('baseline', hlsl), ('dropped-load', mutant)):
                src, cso = td / f'{name}_{label}.hlsl', td / f'{name}_{label}.cso'
                src.write_text(text, encoding='ascii')
                subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
                proc = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)],
                                      capture_output=True, text=True)
                print(proc.stdout, end='')
                if proc.returncode == 77:
                    return 77
                failed = proc.returncode != 0
                if label == 'baseline' and (failed or proc.stderr.strip()):
                    raise AssertionError('%s baseline (%s): %s' % (name, adapter, proc.stderr or 'exit %d' % proc.returncode))
                if label != 'baseline' and not failed:
                    raise AssertionError('%s %s (%s) still matched the reference' % (name, label, adapter))
                print('%s %s (%s): %s' % (name, label, adapter, 'all %d words match' % WORDS if label == 'baseline' else 'rejected'))
    return 0


FORMATTED_STRIDE = 16
FORMATTED_RECORDS = 8
STALE_BASE = 256


def formatted_case(dispatches, *, stale=False, shift=0, records=FORMATTED_RECORDS):
    size = WORDS * 4
    initial = bytearray(struct.pack('<I', SENTINEL) * WORDS)
    source = bytearray(size)
    for record in range(FORMATTED_RECORDS):
        struct.pack_into('<4e', initial, record * FORMATTED_STRIDE, *(
            (record + 1) * (channel + 1) / 4 for channel in range(4)))
    for lane in range(rf.LOCAL[0]):
        # Include different signs and one half-ULP tie at the starting magnitude of lane 0.
        struct.pack_into('<4e', source, lane * 8, 2 ** -13, (lane + 1) / 8, -0.125, 0.5)
    size_of_records = FORMATTED_STRIDE * FORMATTED_RECORDS
    source[STALE_BASE:STALE_BASE + size_of_records] = initial[:size_of_records]
    result = initial.copy()
    for _ in range(dispatches):
        prior = initial if stale else result.copy()
        for lane in range(rf.LOCAL[0]):
            index = lane * 2
            if index >= records:
                continue
            values = struct.unpack_from('<4e', prior, (index + shift) * FORMATTED_STRIDE)
            increments = struct.unpack_from('<4e', source, lane * 8)
            struct.pack_into('<4e', result, index * FORMATTED_STRIDE,
                             *(a + b for a, b in zip(values, increments)))
    return initial, source, list(struct.unpack('<%dI' % WORDS, result))


def runtime_dispatch(consumer, dxc, adapter):
    from startup_compute_dispatch_check import dispatch as run_dispatch, lines
    from startup_compute_request_check import DATA, prepare
    from startup_compute_translate_check import state_a

    controls()
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
        env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
        env.pop('X2W_COMPUTE_RW_FORMATTED', None)
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        for name, data in (('compute_root', root_blob(table_range=True)),
                           ('program', bytes.fromhex(WORDS_HEX['formatted'])),
                           ('data', DATA), ('state', state_a(rf.LOCAL, rf.RSRC2))):
            (directory / name).write_bytes(data)
        initial, source, untouched = formatted_case(0)
        for name, data in (('IN', source), ('INITIAL', initial)):
            path = directory / (name.lower() + '.bin')
            path.write_bytes(data)
            env['X2W_TEST_DISPATCH_' + name] = str(path)
        fields = directory / 'fields.bin'
        env.update(X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'), X2W_TEST_DISPATCH_FIELDS=str(fields))
        unified = (1 << 24) | (71 << 12) | 0xfac
        legacy = (103 << 12) | 0xfac

        def run(count, word3=unified, records=FORMATTED_RECORDS, enabled=True):
            fields.write_bytes(struct.pack('<6I', 4, WORDS, 0, FORMATTED_STRIDE, records, word3))
            mode = 'dispatch-rw-once' if count == 1 else 'dispatch-rw-twice'
            run_env = dict(env, X2W_COMPUTE_RW_FORMATTED='1') if enabled else env
            return run_dispatch(directory, consumer, run_env, mode)

        def compare(label, count, expected, *, reason=None, **kwargs):
            got, result = run(count, **kwargs)
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors
            status = 'refused: ' + reason if reason else 'translated'
            assert len(lines(result, 'Dispatch(1, 1, 1) ' + status)) == count, result.stderr[-3000:]
            first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
            assert first is None, f'{label}: word {first} got {got[first]:08x}, expected {expected[first]:08x}'
            print(f'{label}: all {WORDS} words match')

        got, result = run(1)
        assert got == untouched and lines(result, 'Dispatch(1, 1, 1) refused: no translation'), result.stderr[-3000:]
        key = lines(result, 'x2w-draw compute request')[0].split()[3]
        pending = subprocess.run([sys.executable, '-B', G.__file__, '--pending', str(cache)],
                                 check=True, capture_output=True, text=True, env=dict(env, X2W_DXC=dxc, X2W_FXC='1'))
        assert f'{key}: ok' in pending.stdout, pending.stdout + pending.stderr
        pipe = translate('formatted')
        hlsl = pipe.hlsl_cs()
        assert (cache / f'{key}.txt').read_text() == pipe.manifest()

        compare('default off', 1, untouched, enabled=False,
                reason='shared formatted compute buffers require X2W_COMPUTE_RW_FORMATTED=1')
        compare('unsupported shared format', 1, untouched, word3=(1 << 24) | (75 << 12) | 0xfac,
                reason='a shared formatted output has an unsupported descriptor layout')
        compare('first dispatch', 1, formatted_case(1)[2])
        compare('two dispatches', 2, formatted_case(2)[2])
        compare('legacy descriptor format', 2, formatted_case(2)[2], word3=legacy)
        compare('bounded records', 2, formatted_case(2, records=6)[2], records=6)

        address = 'a = v1 * ud2 + 0;'
        begin = hlsl.index(address)
        end = hlsl.index('a = v2 + 0;', begin)
        load = hlsl[begin:end]
        stale_load = load.replace(address, f'a = v1 * ud2 + {STALE_BASE};').replace('u0', 'r0')
        shifted_load = load.replace(address, 'a = (v1 + 1u) * ud2 + 0;')
        assert 'ldu(r0, a)' in stale_load and shifted_load != load
        for label, mutant, count, wrong in (
                ('stale-read', stale_load, 2, formatted_case(2, stale=True)[2]),
                ('one-stride', shifted_load, 1, formatted_case(1, shift=1)[2])):
            assert wrong != formatted_case(count)[2]
            path = directory / (label + '.hlsl')
            path.write_text(hlsl[:begin] + mutant + hlsl[end:], encoding='ascii')
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cache / f'{key}.cs.cso'), str(path)],
                           check=True)
            if label == 'stale-read':
                compare('stale-read first dispatch', 1, formatted_case(1)[2])
            compare(label + ' mutant readback', count, wrong)
            print(f'{label}: rejected by the unmutated CPU reference')
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    parser.add_argument('--runtime-dispatch', nargs=3, metavar=('CONSUMER', 'DXC', 'ADAPTER'))
    args = parser.parse_args(argv)
    if args.runtime_dispatch:
        return runtime_dispatch(*args.runtime_dispatch)
    if args.dispatch:
        return dispatch(*args.dispatch)
    controls(args.llvm_mc)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
