"""WaveSync (gfx2hlsl.py): the wave-wide gathers a translated compute thread needs, independent of the host wave size.

usage: startup_gfx2hlsl_wavesync_check.py                       the text of the helper and its refusals
       startup_gfx2hlsl_wavesync_check.py --emit <dir>          also write probe_<group>[_wrong].hlsl, in_<group>.bin and
                                                                want_<group>.bin for a GPU readback (see below)

The probe shader loops 3 times; each pass runs three gather sites (first active lane with a value, any lane, one lane's
value) over a per-thread word read from t0 and writes 4 words per pass to u0. want_<group>.bin is the CPU reference
(first active lane and its value, any, the value of lane 5 * pass + 1 of the guest wave). A GPU run of probe_<group>.hlsl
must equal it for guest waves of 32 and 64 and for a group that is not a multiple of the wave; probe_<group>_wrong.hlsl
reads the lane next to the right one and must differ, so a run that cannot tell lanes apart fails the control.
"""
import argparse
import os
import struct
import sys

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit')
    args = parser.parse_args()
    what = check_text()
    if args.emit:
        emit(args.emit)
    print('startup_gfx2hlsl_wavesync: PASS (%s)' % what)
