"""WaveSync (gfx2hlsl.py): the wave-wide gathers a translated compute thread needs, independent of the host wave size.

usage: startup_engine_wavesync_check.py                       the text of the helper and its refusals
       startup_engine_wavesync_check.py --emit <dir>          also write probe_<group>[_wrong].hlsl, in_<group>.bin and
                                                                want_<group>.bin for a GPU readback (see below)
       startup_engine_wavesync_check.py dispatch <dxc> <offscreen_check> <warp|hardware>
                                                                GPU readback of 8 groups through --compute-store-case
       startup_engine_wavesync_check.py controls <dxc> <offscreen_check> <warp|hardware>
                                                                the three wrong-value controls must be rejected (see the end)

The probe shader loops 3 times; each pass runs three gather sites (first active lane with a value, any lane, one lane's
value) over a per-thread word read from t0 and writes 4 words per pass to u0. want_<group>.bin is the CPU reference
(first active lane and its value, any, the value of lane 5 * pass + 1 of the guest wave). A GPU run of probe_<group>.hlsl
must equal it for guest waves of 32 and 64 and for a group that is not a multiple of the wave; probe_<group>_wrong.hlsl
reads the lane next to the right one and must differ, so a run that cannot tell lanes apart fails the control.
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G

PASSES = 3
GROUPS = ((32, 32), (64, 64), (128, 32), (128, 64), (96, 64), (40, 32))   # (threads, guest wave size)


def lane_of(i):
    return 5 * i + 1


def shader(threads, wave, wrong=False):
    ws = G.WaveSync((threads, 1, 1), wave)
    out = ['ByteAddressBuffer t0 : register(t0);', 'RWByteAddressBuffer u0 : register(u0);'] + ws.declarations()
    out += ['[numthreads(%d, 1, 1)]' % threads, 'void main(uint3 tid : SV_GroupThreadID) {']
    out += ws.prologue()
    out.append('uint d = t0.Load(x2w_lin * 4u);')
    out.append('for (uint i = 0u; i < %du; ++i) {' % PASSES)
    f, fv, a = ws.gather_first('((d >> i) & 1u) != 0u', 'd * 7u + i')
    b, c = ws.gather_any('((d >> (i + 3u)) & 1u) != 0u'), None
    c, cl = ws.gather_lane('d + i', '5u * i + %du' % (2 if wrong else 1))
    out += a + b[1] + cl
    out.append('u0.Store4((x2w_lin * %du + i * 4u) * 4u, uint4(%s, %s, %s, %s));' % (PASSES * 4, f, fv, b[0], c))
    out += ['}', '}']
    return '\n'.join(out) + '\n'


def reference(threads, wave, data):
    want = []
    for lin in range(threads):
        base, row = lin // wave * wave, []
        lanes = [base + n for n in range(wave) if base + n < threads]
        for i in range(PASSES):
            act = [n for n in lanes if (data[n] >> i) & 1]
            first = act[0] - base if act else 0xffffffff
            row += [first, (data[act[0]] * 7 + i) & 0xffffffff if act else 0,
                    int(any((data[n] >> (i + 3)) & 1 for n in lanes)),
                    (data[base + lane_of(i)] + i) & 0xffffffff if lane_of(i) < wave and base + lane_of(i) < threads else 0]
        want.append(row)
    return want


def check_text():
    ws = G.WaveSync((64, 1, 1), 32)
    assert ws.waves == 2 and ws.declarations()[0] == 'groupshared uint x2w_any[3][2];'
    var, lines = ws.gather_any('b')
    assert var == 'x2w_any_0' and sum(line.count('GroupMemoryBarrierWithGroupSync') for line in lines) == 1, lines
    assert any('InterlockedOr' in line for line in lines)
    _, _, lines = ws.gather_first('e', 'v')
    assert any('InterlockedMin' in line for line in lines)
    for call in (lambda: ws.gather_any('b', uniform=False), lambda: ws.gather_first('e', 'v', uniform=False),
                 lambda: ws.gather_lane('v', '0u', uniform=False), lambda: G.WaveSync((64, 1, 1), 16),
                 lambda: G.WaveSync((32, 32, 2), 32)):
        try:
            call()
        except G.Unsupported:
            continue
        raise AssertionError('not refused')
    # The GPU probe: its CPU reference sees the data and the controls change the shader text (a control that leaves the
    # text alone could not be rejected), and the sentinel appears wherever no thread folds into a slot.
    plain = fold_shader(96, 64)
    for control in CONTROLS:
        assert fold_shader(96, 64, control) != plain, control
    assert gpu_expected(40, 32)[40:] == [SENTINEL] * (RUNNER_WORDS - 40)
    assert len(set(gpu_expected(96, 64)[:96])) > 1          # the empty-mask wave folds differently from the full one
    return 'text and refusals'


def emit(directory):
    os.makedirs(directory, exist_ok=True)
    for threads, wave in GROUPS:
        data = [(n * 2654435761 + 12345) & 0x3f if n % 7 else 0 for n in range(threads)]
        name = '%d_%d' % (threads, wave)
        for wrong in (False, True):
            with open(os.path.join(directory, 'probe_%s%s.hlsl' % (name, '_wrong' if wrong else '')), 'w') as f:
                f.write(shader(threads, wave, wrong))
        with open(os.path.join(directory, 'in_%s.bin' % name), 'wb') as f:
            f.write(struct.pack('<%dI' % threads, *data))
        with open(os.path.join(directory, 'want_%s.bin' % name), 'wb') as f:
            for row in reference(threads, wave, data):
                f.write(struct.pack('<%dI' % len(row), *row))


# ---- GPU readback through the shared compute store runner (offscreen_check --compute-store-case) -------------------------
# The runner binds t0 (320 input words, here the case's own source words), u0 (320 output words that start at the sentinel)
# and compares all 320 output words with the case file. One output word per thread is too little for the three passes x
# four values of the probe above, so this variant folds the twelve values of a thread into one word (FNV-1a over the words)
# and XORs it into slot lin % 320 (a group of 1024 threads puts three or four threads on a slot; XOR is order independent).
# The CPU reference folds the same values the same way, so a wrong first lane, any flag or lane value changes the word.
RUNNER_WORDS = 320
SENTINEL = 0xc5c5c5c5
GPU_GROUPS = ((32, 32), (64, 64), (128, 32), (128, 64), (96, 64), (40, 32), (1024, 32), (1024, 64))   # (threads, guest wave)
CONTROLS = ('wrong-lane', 'wrong-first', 'no-any')
FNV_BASIS, FNV_PRIME = 0x811c9dc5, 16777619


def gpu_data(threads, wave):
    """Per-thread words: bits 0-2 pick the lanes of the first-lane gather, bits 3-5 those of the any gather. Every seventh
    lane has no bit set, and the whole second guest wave has none (an empty mask in every pass); the words repeat after 320."""
    data = [(n * 2654435761 + 12345) & 0x3f if n % 7 else 0 for n in range(RUNNER_WORDS)]
    if threads > wave:
        data = [0 if n // wave == 1 else v for n, v in enumerate(data)]
    return data


def fold_shader(threads, wave, control=None):
    ws = G.WaveSync((threads, 1, 1), wave)
    out = ['cbuffer entry : register(b0) { uint ud0; uint ud1; };',
           'ByteAddressBuffer t0 : register(t0);', 'RWByteAddressBuffer u0 : register(u0);'] + ws.declarations()
    out += ['[numthreads(%d, 1, 1)]' % threads, 'void main(uint3 tid : SV_GroupThreadID) {']
    out += ws.prologue()
    out.append('uint slot = x2w_lin %% %du;' % RUNNER_WORDS)
    out.append('uint d = t0.Load(slot * 4u);')
    out.append('uint h = %du;' % FNV_BASIS)
    out.append('for (uint i = 0u; i < %du; ++i) {' % PASSES)
    f, fv, a = ws.gather_first('((d >> i) & 1u) != 0u', 'd * 7u + i' + (' + 1u' if control == 'wrong-first' else ''))
    b, bl = ws.gather_any('((d >> (i + 3u)) & 1u) != 0u')
    c, cl = ws.gather_lane('d + i', '5u * i + %du' % (2 if control == 'wrong-lane' else 1))
    out += a + bl + cl
    any_value = '0u' if control == 'no-any' else b
    for value in (f, fv, any_value, c):
        out.append('h = (h ^ (%s)) * %du;' % (value, FNV_PRIME))
    out += ['}', 'u0.InterlockedXor(slot * 4u, h);', '}']
    return '\n'.join(out) + '\n'


def fold(values):
    h = FNV_BASIS
    for value in values:
        h = ((h ^ value) * FNV_PRIME) & 0xffffffff
    return h


def gpu_expected(threads, wave):
    """The 320 words the runner must read back: the sentinel XOR the folds of every thread that maps to the slot."""
    data = gpu_data(threads, wave)
    words = [SENTINEL] * RUNNER_WORDS
    for lin, row in enumerate(reference(threads, wave, [data[n % RUNNER_WORDS] for n in range(threads)])):
        words[lin % RUNNER_WORDS] ^= fold(row)
    return words


def run_gpu_case(threads, wave, dxc, runner, adapter, directory, control=None):
    name = 'wavesync_%d_%d%s' % (threads, wave, '_' + control if control else '')
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(fold_shader(threads, wave, control), encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    case.write_bytes(struct.pack('<I', 1) + struct.pack('<%dI' % RUNNER_WORDS, *gpu_expected(threads, wave))
                     + struct.pack('<16I', *([0] * 16)) + struct.pack('<%dI' % RUNNER_WORDS, *gpu_data(threads, wave)))
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_gpu(dxc, runner, adapter, controls=False):
    runner = str(Path(runner).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        if controls:
            runs = [(t, w, c) for c in CONTROLS for t, w in ((96, 64), (40, 32), (128, 64), (1024, 32))]
        else:
            runs = [(t, w, None) for t, w in GPU_GROUPS]
        for threads, wave, control in runs:
            run = run_gpu_case(threads, wave, dxc, runner, adapter, Path(tmp), control)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = [line for line in run.stderr.splitlines() if not line.startswith('XBOX2WINDOWS_ADAPTER: skipping ')]
            if control:
                if run.returncode != 1 or not any('expected=' in line for line in diagnostics):
                    raise AssertionError('control %s %d/%d: not rejected (exit %d) %s' % (control, threads, wave, run.returncode, run.stderr))
                print('control %s %d threads, guest wave %d: rejected' % (control, threads, wave))
            elif run.returncode or diagnostics:
                raise AssertionError('group %d/%d: %s' % (threads, wave, run.stderr or 'exit %d' % run.returncode))
            else:
                print('group %d threads, guest wave %d: all %d words match' % (threads, wave, RUNNER_WORDS))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit')
    parser.add_argument('mode', nargs='*', help='dispatch|controls <dxc> <offscreen_check> <warp|hardware>')
    args = parser.parse_args()
    what = check_text()
    if args.emit:
        emit(args.emit)
    if args.mode and args.mode[0] in ('dispatch', 'controls') and len(args.mode) == 4:
        sys.exit(check_gpu(*args.mode[1:], controls=args.mode[0] == 'controls'))
    print('startup_gfx2hlsl_wavesync: PASS (%s)' % what)
