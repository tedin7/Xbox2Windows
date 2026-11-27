#!/usr/bin/env python3
"""Compute LDS b64 ordered-pair controls and D3D12 readback."""
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_compute_lds_b64_check as sibling
from startup_gfx2hlsl_compute_check import patch, refused, renderer_diagnostics

WORDS = sibling.WORDS
SENTINEL = sibling.SENTINEL
BASE = patch(sibling.WORDS_P, '00008abf', '000080bf')
OOB = patch(BASE, '1000d8d908000008', '0004d8d908000008')
SAME_WAVE = patch(BASE, 'ff08103a00010000', 'ff08103a08000000')


def reference(out_of_bounds=False, wrong_peer=False, same_wave=False):
    # AMD RDNA 2 ISA guide, 10.4.3 and 12.13: byte-addressed, little-endian LDS.
    result = [SENTINEL] * WORDS
    for group in range(2):
        lds = bytearray(1024)
        for wave in range(2):
            for lane in range(wave * 32, wave * 32 + 32):
                value = (group << 8) + lane
                struct.pack_into('<II', lds, lane * 8 + 16, value, value ^ 0xa5a50000)
        for lane in range(64):
            peer = lane if wrong_peer else lane ^ (1 if same_wave else 32)
            address = 1024 if out_of_bounds else peer * 8 + 16
            for word in range(2):
                offset = address + word * 4
                value = struct.unpack_from('<I', lds, offset)[0] if offset + 4 <= len(lds) else 0
                result[(group * 64 + lane) * 2 + word] = value
    return result


def shader(code=BASE):
    return sibling.translate_case('P', code=code).hlsl_cs()


def controls():
    legacy = sibling.translate_case('P').hlsl_cs()
    assert hashlib.sha256(legacy.encode()).hexdigest() == '6f1c8c0cae027dc199d84a4d74774a68c243f205f740fac098524e15109eac70'
    text = shader()
    assert text.count('GroupMemoryBarrierWithGroupSync();') == 1
    assert text.index('lds[(a + 16u) >> 2] =') < text.index('GroupMemoryBarrierWithGroupSync();') < text.index('w = lds[(a + 16u) >> 2]')
    assert 'if (a < 1004u)' in text
    assert reference() != reference(wrong_peer=True)
    assert reference(same_wave=True) != reference(wrong_peer=True, same_wave=True)
    assert shader(SAME_WAVE).count('GroupMemoryBarrierWithGroupSync();') == 1
    assert reference(out_of_bounds=True)[:256] == [0] * 256
    assert shader(OOB).count('GroupMemoryBarrierWithGroupSync();') == 1
    unaligned = patch(BASE, '1000d8d908000008', '1200d8d908000008')
    refused('proven 4-byte aligned', lambda: shader(unaligned))
    hidden_field = patch(BASE, '100034d904050000', '100034d904050100')
    refused('reserved or unused', lambda: shader(hidden_field))
    branched = patch(BASE, '100034d904050000', '020084bf100034d904050000')
    refused('straight-line', lambda: shader(branched))
    print('general LDS ordered b64 pair: controls passed')


def dispatch(dxc, runner, adapter):
    root = Path(__file__).resolve().parent.parent
    out = root / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for name, code, oob, same_wave in (('exchange', BASE, False, False),
                                            ('same_wave', SAME_WAVE, False, True),
                                            ('out_of_bounds', OOB, True, False)):
            src, cso, case = (directory / (name + ext) for ext in ('.hlsl', '.cso', '.case'))
            src.write_text(shader(code), encoding='ascii')
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            case.write_bytes(struct.pack('<I', 2) + struct.pack('<%dI' % WORDS, *reference(oob, same_wave=same_wave)) + struct.pack('<16I', *([0] * 16)))
            run = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if run.returncode or diagnostics:
                raise AssertionError('%s: exit %d %s' % (name, run.returncode, run.stderr))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(dispatch(*sys.argv[2:]))
    else:
        raise SystemExit('usage: startup_compute_lds_general_check.py [dispatch DXC RUNNER ADAPTER]')
