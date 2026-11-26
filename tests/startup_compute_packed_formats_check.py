"""Synthetic formatted MUBUF programs checked against CPU references on D3D12 WARP."""

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_compute_store_d16_x_check import ROOT, encode_mubuf
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, root_blob
from startup_gfx2hlsl_heap_record_check import RSRC2 as HEAP_RSRC2, root_blob as heap_root

IDENTITY = 0xfac
END = 0xbf810000
WAIT_LGKM = 0xbf8cc07f
WAIT_VM = 0xbf8c3f70


def dwords(*words):
    return struct.pack('<%dI' % len(words), *words)


def move(dst, src):
    return 0xbe800300 | dst << 16 | src


def scalar_load(op, dst, base, offset):
    return dwords(0xf4000000 | op << 18 | dst << 6 | (base // 2), 0xfa000000 | offset)


def vector_shift(dst, shift, src):
    return 0x34000000 | dst << 17 | src << 9 | (128 + shift)


def table_load(dst, offset, entry):
    return dwords(move(30, entry), move(31, 128)) + scalar_load(2, dst, 30, offset)


def program(kind):
    heap = kind == 'heap'
    if heap:
        # Select the SRV via the heap index in the third root constant.
        code = table_load(20, 16, 7)
        code += dwords(move(9, 128), 0x8f068506, 0xf4080604, 0x0c000000)
    else:
        code = table_load(24, 0, 6) + scalar_load(2, 20, 30, 16)
    code += dwords(WAIT_LGKM)
    if kind == 'store':
        code += dwords(vector_shift(3, 3, 0))
        code += encode_mubuf(0x0d, 10, 3, 24, 128, offen=True)
        code += dwords(WAIT_VM)
        code += encode_mubuf(0x87, 10, 0, 20, 128, idxen=True, offset=4)
    else:
        if kind == 'memory':
            # Read a replacement FORMAT/DST_SEL word from the input itself.
            code += scalar_load(8, 27, 24, 0) + dwords(WAIT_LGKM)
        elif kind == 'register':
            # Assemble the descriptor with a format/select word from user data.
            code += dwords(move(27, 4))
        code += dwords(vector_shift(3, 4, 0))
        code += encode_mubuf(3, 10, 3, 24, 128, offen=True, offset=16)
        code += dwords(WAIT_VM)
        code += encode_mubuf(0x1e, 10, 3, 20, 128, offen=True)
    code += dwords(END)
    root, rsrc2 = (heap_root(), HEAP_RSRC2) if heap else (root_blob(), USER7)
    return code, G.ComputePipeline(code, root, rsrc2, (4, 1, 1))


def store_case(fmt, legacy, stride):
    values = (0.25, -0.5, 0.75, 1.5)
    source = b''.join(struct.pack('<4e', *values[lane:], *values[:lane]) for lane in range(4))
    if fmt == 71:
        packed = source
    else:
        scale = 65535 if fmt == 65 else 32767
        low = 0 if fmt == 65 else -1

        def convert(x):
            x = min(1, max(low, x))
            return int(math.copysign(math.floor(abs(x) * scale + 0.5), x)) & 0xffff

        packed = struct.pack('<16H', *(convert(x) for (x,) in struct.iter_unpack('<e', source)))
    field = {65: 96, 66: 97, 71: 103}[fmt] if legacy else fmt
    word3 = (0 if legacy else 1 << 24) | field << 12 | IDENTITY
    expected = bytearray(dwords(*([SENTINEL] * WORDS)))
    for lane in range(3):
        addr = lane * stride + 4
        expected[addr:addr + 8] = packed[lane * 8:lane * 8 + 8]
    return source, (0, 0, word3, stride, 3), expected


def load_case(kind, fmt):
    # Z, X, forced one, Y. Integer formats must return an integer constant one.
    word3 = (1 << 24) | (fmt << 12) | (5 << 9) | (1 << 6) | (4 << 3) | 6
    source = bytearray(dwords(word3, 0, 0, 0))
    expected = bytearray(dwords(*([SENTINEL] * WORDS)))
    for lane in range(4):
        if fmt == 60:
            raw = bytes((19 + lane, 91 - lane, 203 + lane, 75))
            x, y, z, _ = raw
            result = dwords(z, x, 1, y)
        elif fmt == 24:
            raw = struct.pack('<2h', (-32768, -32767, 0, 32767)[lane], (0, 32767, -32767, -32768)[lane])
            x, y = struct.unpack('<2h', raw)
            result = struct.pack('<4f', 0, max(-1, x / 32767), 1, y / 32767)
        else:
            raw = struct.pack('<4e', 1.5 + lane, -0.875, 9.25 - lane, 0.375)
            x, y, z, _ = struct.unpack('<4e', raw)
            result = struct.pack('<4f', z, x, 1, y)
        source += raw.ljust(16, b'\0')
        expected[lane * 16:lane * 16 + 16] = result
    constants = {'heap': (0, 0, 2, word3), 'register': (word3,), 'memory': ()}
    return source, constants[kind], expected


def lowering(kind):
    code, pipe = program(kind)
    ops = [op for _, _, op, _ in G.decode(code)]
    hlsl = pipe.hlsl_cs()
    if kind == 'store':
        assert 'buffer_store_format_d16_xyzw' in ops
        assert 'uint2(v10, v11)' in hlsl and 'buf.Store2(addr, packed)' in hlsl
        assert 'udsrc 2 uav_descword3 0\nudsrc 3 uav_stride 0\nudsrc 4 uav_numrecords 0' in pipe.manifest()
    else:
        assert 'buffer_load_format_xyzw' in ops
        word3 = 'ud3' if kind == 'heap' else 's27'
        assert 'switch ((%s >> 12) & 127u)' % word3 in hlsl
        if kind == 'heap':
            assert 'srv 0 buf heap rootconst 0 2' in pipe.manifest()
            assert 'udsrc 3 descword3 0' in pipe.manifest()
        else:
            assert 'descword3' not in pipe.manifest()
            if kind == 'register':
                assert 's27 = ud0;' in hlsl
            else:
                assert 's27 = r0.Load(' in hlsl
    return hlsl


CONTROLS = {
    'store': ('store_d16_pair(val.y, fmt)', 'store_d16_pair(val.x, fmt)'),
    'register': ('switch ((s27 >> 12) & 127u)', 'switch (22u)'),
    'memory': ('switch ((s27 >> 12) & 127u)', 'switch (22u)'),
    'heap': ('switch ((ud3 >> 12) & 127u)', 'switch (22u)'),
}


def check_dispatch(dxc, runner, directory, control=None):
    for kind in ([control] if control else CONTROLS):
        hlsl = lowering(kind)
        if control:
            old, new = CONTROLS[kind]
            assert hlsl.count(old) == 1
            hlsl = hlsl.replace(old, new)
        src, cso, case = (directory / (kind + ext) for ext in ('.hlsl', '.cso', '.case'))
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        inputs = ([store_case(fmt, legacy, 14 if legacy else 16)
                   for fmt in (65, 66, 71) for legacy in (False, True)]
                  if kind == 'store' else [load_case(kind, fmt) for fmt in (60, 24, 71)])
        for source, constants, expected in inputs:
            case.write_bytes(dwords(0x010101) + expected + dwords(*constants).ljust(64, b'\0')
                             + source.ljust(WORDS * 4, b'\0'))
            run = subprocess.run([runner, 'warp', '--compute-store-sized-case', str(cso), str(case)],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            if control:
                assert run.returncode == 1 and 'expected=' in run.stderr, (kind, run.returncode, run.stderr)
                print('control %s rejected at %s' % (kind, run.stderr.strip()))
            else:
                assert run.returncode == 0, (kind, constants, run.stderr)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        for kind, (old, _) in CONTROLS.items():
            assert lowering(kind).count(old) == 1
        print('packed format synthetic translations passed')
    else:
        ROOT.joinpath('out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'out') as tmp:
            sys.exit(check_dispatch(*sys.argv[1:3], Path(tmp), *sys.argv[3:]))
