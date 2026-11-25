"""Inspect slot 86 mesh requests or publish a separately validated desktop pair."""

import argparse
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_aot import container_image


def fnv(blob, value=1469598103934665603):
    for byte in blob:
        value = (value ^ byte) * 1099511628211 & ((1 << 64) - 1)
    return value


def request_key(*parts):
    key = fnv(b'x2w-startup-mesh-86-1')
    for part in parts:
        key = fnv(part, fnv(struct.pack('<Q', len(part)), key))
    return key


def dxil_part(data):
    if len(data) < 32 or data[:4] != b'DXBC':
        raise ValueError('expected a DXBC container')
    size, count = struct.unpack_from('<II', data, 24)
    if size != len(data) or count > (size - 32) // 4:
        raise ValueError('invalid DXBC part table')
    header = 32 + count * 4
    program = None
    for offset, in struct.iter_unpack('<I', data[32:header]):
        if not header <= offset <= size - 8:
            raise ValueError('invalid DXBC part offset')
        kind, length = struct.unpack_from('<4sI', data, offset)
        if length > size - offset - 8:
            raise ValueError('truncated DXBC part')
        if kind == b'DXIL':
            if program is not None:
                raise ValueError('duplicate DXIL part')
            program = data[offset + 8:offset + 8 + length]
    if not program:
        raise ValueError('missing DXIL program')
    return program


def desktop_stage(blob, expected):
    program = dxil_part(blob)
    if len(program) < 24:
        raise ValueError('truncated desktop DXIL program')
    version, words, magic, _, bitcode_offset, bitcode_size = struct.unpack_from('<II4sIII', program)
    start = 8 + bitcode_offset
    if (version >> 16 != expected or (version >> 4) & 15 != 6 or
            magic != b'DXIL' or words * 4 != len(program) or
            start < 24 or bitcode_size < 4 or bitcode_size > len(program) - start or
            program[start:start + 4] != b'BC\xc0\xde'):
        raise ValueError('unexpected desktop DXIL stage or bitcode')


def load_request(cache, key):
    source = cache / f'req_{key:016x}'
    ms, ps, root, stream = (source.with_suffix(suffix).read_bytes()
                            for suffix in ('.m86', '.ps86', '.root', '.stream86'))
    if request_key(ms, ps, root, stream) != key:
        raise ValueError('request key or source identity mismatch')
    container_image(dxil_part(ms), stage=13)
    if ps:
        container_image(dxil_part(ps), stage=0)
    return ms, ps, root, stream


_MESH_SRV_HANDLE = re.compile(
    r'(%\d+) = call %dx\.types\.Handle @dx\.op\.createHandle'
    r'\(i32 57, i8 0, i32 0, i32 0, i1 false\)')
_MESH_SRV_LOAD = re.compile(
    r'%\d+ = call %dx\.types\.ResRet\.i32 @dx\.op\.rawBufferLoad\.i32'
    r'\(i32 139, %dx\.types\.Handle (%\d+), i32 0, i32 undef, i8 1, i32 4\)')
_MESH_OUTPUT_OPS = frozenset(('groupId.i32', 'setMeshOutputCounts', 'storeVertexOutput.f32',
                              'storeVertexOutput.i32', 'emitIndices'))
_PIXEL_IO_OPS = frozenset(('loadInput.i32', 'loadInput.f32', 'storeOutput.i32', 'storeOutput.f32'))


def _dxil_calls(dump):
    for line in dump.splitlines():
        instruction = line.partition(';')[0].strip()
        if re.search(r'\b(call|invoke|callbr)\b', instruction):
            match = re.fullmatch(r'(?:(?:%\d+ = )?call \S+ @dx\.op\.([\w.]+)\(.*\))', instruction)
            if not match:
                raise ValueError('unexpected call in the single-DWORD mesh SRV profile')
            yield match[1], instruction


def prove_mesh_root_srv(mesh_dump, pixel_dump):
    # Require a single raw SRV, range ID 0, register t0, space 0 and count 1.
    # The byte width comes from the actual constant-offset DXIL load, not the
    # root signature or a user-supplied span.
    table = re.search(r'^; Resource Bindings:\n;\n; Name[^\n]*\n; [- ]+\n((?:; [^\n]+\n)*)',
                      mesh_dump, re.MULTILINE)
    rows = [line[1:].split()[1:] for line in table[1].splitlines() if line[1:].strip()] if table else []
    if rows != [['texture', 'byte', 'r/o', 'T0', 't0', '1']]:
        raise ValueError('expected exactly one non-array raw SRV at t0 space0')
    handles, loads = [], []
    for op, instruction in _dxil_calls(mesh_dump):
        if op == 'createHandle' and (match := _MESH_SRV_HANDLE.fullmatch(instruction)):
            handles.append(match[1])
        elif op == 'rawBufferLoad.i32' and (match := _MESH_SRV_LOAD.fullmatch(instruction)):
            loads.append(match[1])
        elif op not in _MESH_OUTPUT_OPS:
            raise ValueError(f'unsupported DXIL operation for the single-DWORD mesh SRV profile: {op}')
    if len(handles) != 1 or loads != handles:
        raise ValueError('expected one DWORD load from the verified SRV handle')
    if any(op not in _PIXEL_IO_OPS for op, _ in _dxil_calls(pixel_dump)):
        raise ValueError('pixel stage must not access resources in the mesh SRV profile')
    return 4


def publish(cache, key, ms_path, ps_path, *, srv_dxc=None):
    _, ps_source, _, _ = load_request(cache, key)
    if bool(ps_source) != bool(ps_path):
        raise ValueError('desktop pixel-stage presence must match the input stream')
    ms = ms_path.read_bytes()
    ps = ps_path.read_bytes() if ps_path else b''
    desktop_stage(ms, 13)
    if ps:
        desktop_stage(ps, 0)
    if max(len(ms), len(ps)) > 1 << 20:
        raise ValueError('desktop shader exceeds the runtime size bound')
    stem = cache / f'{key:016x}'
    ready = stem.with_suffix('.mesh86.txt')
    hashes = f'{key:016x} {fnv(ms):016x} {(fnv(ps) if ps else 0):016x}'
    manifest = f'x2w-mesh86 1 {hashes}\n'
    with tempfile.TemporaryDirectory(prefix='.mesh86-', dir=cache) as temp:
        work = Path(temp)
        (work / 'ms').write_bytes(ms)
        if ps:
            (work / 'ps').write_bytes(ps)
        if srv_dxc:
            if not ps:
                raise ValueError('the mesh SRV profile requires a pixel stage')
            # Dump the very bytes that will be published, including when the
            # original shader paths change concurrently.
            dumps = [subprocess.run([srv_dxc, '-dumpbin', work / stage], check=True,
                                    capture_output=True, text=True, encoding='utf-8').stdout
                     for stage in ('ms', 'ps')]
            extent = prove_mesh_root_srv(*dumps)
            manifest = f'x2w-mesh86 2 {hashes} srv 0 0 {extent}\n'
        (work / 'marker').write_bytes(manifest.encode('ascii'))
        ready.unlink(missing_ok=True)
        (work / 'ms').replace(stem.with_suffix('.ms.cso'))
        if ps:
            (work / 'ps').replace(stem.with_suffix('.ps.cso'))
        (work / 'marker').replace(ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('key', type=lambda text: int(text, 16))
    parser.add_argument('--mesh', type=Path, help='desktop MS DXIL verified against the captured input')
    parser.add_argument('--pixel', type=Path, help='desktop PS DXIL when the captured input has a pixel stage')
    parser.add_argument('--prove-root-srv', type=Path, metavar='DXC',
                        help='use DXC to verify the bounded single-DWORD mesh root SRV profile')
    args = parser.parse_args()
    try:
        if args.mesh:
            publish(args.cache, args.key, args.mesh, args.pixel, srv_dxc=args.prove_root_srv)
            print(f'{args.key:016x}: published desktop stages')
        else:
            if args.pixel or args.prove_root_srv:
                raise ValueError('--pixel and --prove-root-srv require --mesh')
            ms, ps, root, stream = load_request(args.cache, args.key)
            print(f'{args.key:016x}: MS {len(ms)}, PS {len(ps)}, root {len(root)}, stream {len(stream)} bytes')
    except (ValueError, OSError, struct.error, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
