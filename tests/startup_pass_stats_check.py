"""Pass thermometer (X2W_PASS_STATS=1, docs/startup-runtime.md#pass-thermometer) on a three-pass scene.

usage: startup_pass_stats_check.py <startup_stream_consumer> <dxc> <warp|hardware> [--llvm-mc LLVM_MC]

The consumer's pass-stats scene (tests/startup_pass_stats_scene.h) clears texture A to (0.25, 0.5, 0.75, 1) (pass A),
dispatches the exposure program below, which stores A times root constant 0 into texture T (pass B), copies T to
the presented surface S (pass C) and presents S. The recording also copies A, T and S to a readback buffer, so
every reduction is compared with CPU statistics of the same texels read back from the GPU.

1. before translation, switch on: the Dispatch is refused, so there is no compute pass; the copy pass after it
   carries the refused call, and the collapse line names it (nothing wrote T);
2. exposure 0: passes.jsonl holds the clear, compute and copy passes with the CPU statistics of A, T and S; A has
   signal, B and C are all zero, and the collapse line names B;
3. exposure 1: B and C carry A's statistics and the line reports no collapse;
4. switch off: no passes.jsonl, no thermometer line, and the same readback as run 2.
Exit 77 when the adapter is not available.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_texture_check import translate
from startup_compute_dispatch_check import lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a

GROUP = (4, 2, 1)  # the scene dispatches (8, 4, 1) groups over 32x8 texels
RSRC2 = (10 << 1) | C.TGID_X | C.TGID_Y | C.tidig(2)  # three root constants put the descriptor table at s7
SOURCE = """\
s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[24:31], s[12:13], 0x40
s_load_dwordx8 s[32:39], s[12:13], 0x80
v_lshl_add_u32 v8, s10, 2, v0
v_lshl_add_u32 v9, s11, 1, v1
v_mov_b32_e32 v10, 0
s_waitcnt lgkmcnt(0)
image_load_mip v[14:17], v[8:10], s[32:39] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v14, s4, v14
v_mul_f32_e32 v15, s4, v15
v_mul_f32_e32 v16, s4, v16
v_mul_f32_e32 v17, s4, v17
image_store v[14:17], v[8:9], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D
s_endpgm
"""
WORDS = bytes.fromhex(
    '07038cbe 80038dbe 06060cf4 400000fa 06080cf4 800000fa 080046d7 0a040104 090046d7 0b020504 8002147e 7fc08cbf'
    ' 080f04f0 080e0800 703f8cbf 041c1c10 041e1e10 04202010 04222210 080f20f0 080e0600 000081bf')
TEXELS = 32 * 8
IMAGE = TEXELS * 4
COLLAPSE = 'x2w-replay pass collapse: present 1: '


def cpu_stats(image):
    """Per-channel statistics of 32x8 RGBA8_UNORM texels, as the reduction reads them (float32 of byte / 255)."""
    values = [[struct.unpack('<f', struct.pack('<f', image[i * 4 + c] / 255.0))[0] for i in range(TEXELS)] for c in range(4)]
    return {'texels': TEXELS, 'channels': 4, 'min': [min(v) for v in values], 'max': [max(v) for v in values],
            'mean': [sum(v) / TEXELS for v in values], 'zero': [v.count(0.0) for v in values],
            'nan': [0] * 4, 'inf': [0] * 4}


def clean(result):
    """No refusal and no debug-layer error (the collapse line's 'refused calls' count is not a refusal)."""
    refused = [line for line in lines(result, 'refused') if 'pass collapse' not in line]
    assert not refused, refused[:3]
    errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    assert not errors, errors[0]
    layer = lines(result, 'x2w-d3d12 debug layer: ')
    return 'enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'


def same_stats(got, want, what):
    assert got is not None, f'{what}: no stats'
    what = f'{what} (got {json.dumps(got)})'
    for name in ('texels', 'channels', 'zero', 'nan', 'inf'):
        assert got[name] == want[name], f'{what}: {name} {got[name]} != {want[name]}'
    for name, tolerance in (('min', 1e-7), ('max', 1e-7), ('mean', 1e-6)):
        for c in range(4):
            assert abs(got[name][c] - want[name][c]) <= tolerance, f'{what}: {name}[{c}] {got[name][c]} != {want[name][c]}'


def run(directory, consumer, env, mode, out):
    out.mkdir()
    readback = out / 'readback.bin'
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', mode]
    result = subprocess.run(args, env=dict(env, X2W_TEST_DISPATCH_OUT=str(readback), X2W_PASS_STATS_DIR=str(out)),
                            capture_output=True, text=True, timeout=120)
    if result.returncode == 77:
        raise SystemExit(77)
    assert result.returncode == 0, (result.returncode, result.stderr[-4000:])
    print('\n'.join(lines(result, 'x2w-pass stats') + lines(result, 'pass collapse')))
    images = readback.read_bytes()
    assert len(images) == 3 * IMAGE, len(images)
    passes = out / 'passes.jsonl'
    records = [json.loads(line) for line in passes.read_text().splitlines()] if passes.exists() else None
    return result, [images[i * IMAGE:(i + 1) * IMAGE] for i in range(3)], records


def collapse(result):
    found = lines(result, 'x2w-replay pass collapse: ')
    assert len(found) == 1, (found, result.stderr[-3000:])
    return found[0][found[0].index(COLLAPSE) + len(COLLAPSE):]


def writes(record):
    return [(w['resource'], w['format'], w['width'], w['height']) for w in record['writes']]


def check_request(directory, consumer, env):
    result, images, records = run(directory, consumer, dict(env, X2W_PASS_STATS='1'), 'pass-stats', directory / 'request')
    assert len(lines(result, 'refused: no translation')) == 1, result.stderr[-3000:]
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    assert [r['kind'] for r in records] == ['clear', 'copy'], records
    clear_pass, copy_pass = records
    assert copy_pass['refused'] == 1 and clear_pass['refused'] == 0, records
    same_stats(copy_pass['writes'][0]['stats'], cpu_stats(images[2]), 'request run, S')
    assert collapse(result) == ('last pass with signal = none, first all-zero/NaN pass after it = '
                                '#1 (key -, copy, refused calls 1)'), result.stderr[-3000:]
    return key


def check_dark(directory, consumer, env, key):
    result, images, records = run(directory, consumer, dict(env, X2W_PASS_STATS='1'), 'pass-stats', directory / 'dark')
    layer = clean(result)
    assert any(images[0]) and not any(images[1]) and not any(images[2]), 'A is not drawn, or T or S is not black'
    assert [r['kind'] for r in records] == ['clear', 'compute', 'copy'], records
    a, b, c = records
    assert all(r['present'] == 1 and r['refused'] == 0 for r in records), records
    ids = [a['writes'][0]['resource'], b['writes'][0]['resource'], c['writes'][0]['resource']]
    assert len(set(ids)) == 3 and all(ids), ids
    assert [writes(r) for r in records] == [[(i, 28, 32, 8)] for i in ids], records
    assert b['keys'] == ['%016x' % key] and b['inputs'] == [ids[0]] and c['inputs'] == [ids[1]], records
    assert a['inputs'] == [] and c['keys'] == [] and a['keys'] == [], records
    for record, image, name in zip(records, images, 'ABC'):
        same_stats(record['writes'][0]['stats'], cpu_stats(image), f'pass {name}')
    assert [r['signal'] for r in records] == [True, False, False], records
    assert collapse(result) == ('last pass with signal = #0 (key -, clear), first all-zero/NaN pass after it = '
                                '#1 (key %016x, compute, refused calls 0)' % key), result.stderr[-3000:]
    return images, layer


def check_bright(directory, consumer, env):
    result, images, records = run(directory, consumer, dict(env, X2W_PASS_STATS='1'), 'pass-stats-bright', directory / 'bright')
    clean(result)
    assert images[0] == images[1] == images[2] and any(images[0]), 'exposure 1 does not copy A to T and S'
    for record, image, name in zip(records, images, 'ABC'):
        same_stats(record['writes'][0]['stats'], cpu_stats(image), f'bright pass {name}')
    assert [r['signal'] for r in records] == [True, True, True], records
    assert collapse(result) == ('no collapse: the last pass that wrote the presented surface, '
                                '#2 (key -, copy), has signal'), result.stderr[-3000:]


def check_off(directory, consumer, env, dark):
    result, images, records = run(directory, consumer, env, 'pass-stats', directory / 'off')
    clean(result)
    assert records is None, 'passes.jsonl written with the switch off'
    assert not lines(result, 'x2w-pass stats') and not lines(result, 'pass collapse'), result.stderr[-3000:]
    assert images == dark, 'the switch changed the readback'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('consumer')
    parser.add_argument('dxc')
    parser.add_argument('adapter', choices=('warp', 'hardware'))
    parser.add_argument('--llvm-mc')
    options = parser.parse_args()
    consumer = options.consumer
    if options.llvm_mc:
        assert C.assemble(options.llvm_mc, SOURCE) == WORDS, 'the exposure program words do not match llvm-mc'
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        cache, base = prepare(directory)
        env = dict(base, X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
        env.pop('X2W_PASS_STATS', None)
        if options.adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        (directory / 'compute_root').write_bytes(C.root_blob(3, table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        (directory / 'program').write_bytes(WORDS)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(GROUP, RSRC2))
        key = check_request(directory, consumer, env)
        translate(cache, options.dxc, key)
        dark, layer = check_dark(directory, consumer, env, key)
        check_bright(directory, consumer, env)
        check_off(directory, consumer, env, dark)
    print(f'pass thermometer: A signal, B and C all zero, collapse at B; exposure 1 has no collapse; '
          f'switch off writes nothing ({options.adapter}, debug layer {layer})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
