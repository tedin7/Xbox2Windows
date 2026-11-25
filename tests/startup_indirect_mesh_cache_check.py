"""Publish a synthetic Xbox envelope, then load the source-built desktop stages."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import mesh86_cache


def source_envelope(stage):
    # The envelope exercises cache transport only. Desktop rendering uses
    # the separate shader sources below.
    image = b'FSSM' + bytes(192)
    compressed = zlib.compress(image)
    native = b'XBCH' + struct.pack('<IQQ', 3, len(compressed), len(image)) + compressed
    program = (struct.pack('<I', (stage << 16) | 0x65) + bytes(20) +
               b'XDAT' + struct.pack('<IIIII', 1, 36, 13, 48, len(native)) + native)
    part = b'DXIL' + struct.pack('<I', len(program)) + program
    return b'DXBC' + bytes(16) + struct.pack('<IIII', 1, 36 + len(part), 1, 36) + part


def run(dxc, fixture, backend, work, depth_only, lease_xmem=None, mode=None):
    for suffix, stage in (('ms', 13), ('ps', 0)):
        (work / f'source.{suffix}.dxbc').write_bytes(source_envelope(stage))
    environment = dict(os.environ, X2W_MESH86_FIXTURE_DIR=str(work), X2W_SHADER_CACHE=str(work))
    environment.pop('X2W_MESH86_FIXTURE_READY', None)
    environment.pop('X2W_MESH86_FIXTURE_DEPTH', None)
    if depth_only:
        environment['X2W_MESH86_FIXTURE_DEPTH'] = '1'
    if lease_xmem:
        environment['X2W_MESH86_FIXTURE_XMEM'] = str(lease_xmem)
        for flag in ('X2W_TEST_MESH_XMEM_SKIP_WAIT', 'X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT',
                     'X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT', 'X2W_MESH86_FIXTURE_DUAL',
                     'X2W_MESH86_FIXTURE_INIT_ONLY', 'X2W_MESH_INDIRECT_XMEM'):
            environment.pop(flag, None)
        if mode == 'skip-wait':
            environment['X2W_TEST_MESH_XMEM_SKIP_WAIT'] = '1'
        elif mode in ('dual', 'dual-skip-srv', 'dual-skip-argument', 'dual-init-only'):
            environment.update(X2W_MESH86_FIXTURE_DUAL='1', X2W_MESH_INDIRECT_XMEM='1')
            if mode == 'dual-skip-srv':
                environment['X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT'] = '1'
            elif mode == 'dual-skip-argument':
                environment['X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT'] = '1'
            elif mode == 'dual-init-only':
                environment['X2W_MESH86_FIXTURE_INIT_ONLY'] = '1'
    cmd = [fixture, backend, 'mesh']
    capture = subprocess.run(cmd, env=environment)
    if capture.returncode:
        return capture.returncode
    marker, = work.glob('req_*.m86')
    key = int(marker.stem.removeprefix('req_'), 16)
    source_ms, source_ps, root, stream = mesh86_cache.load_request(work, key)
    assert source_ms == source_envelope(13) and source_ps == (b'' if depth_only else source_envelope(0))
    assert root.startswith(b'DXBC') and stream[:16] == bytes(16)
    ms, ps = (work / f'desktop.{suffix}.cso' for suffix in ('ms', 'ps'))
    mesh_source = HERE / ('xmem_mesh_root_srv.hlsl' if lease_xmem else 'startup_indirect_mesh.hlsl')
    defines = ['-D', 'X2W_MESH_DUAL=1'] if environment.get('X2W_MESH86_FIXTURE_DUAL') else []
    subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'mesh', *defines, '-Fo', ms, mesh_source], check=True)
    if lease_xmem:
        subprocess.run([dxc, '-T', 'ps_6_0', '-E', 'pixel', '-D', 'X2W_MESH_LEASE_COLOR=1',
                        '-Fo', ps, mesh_source], check=True)
    elif not depth_only:
        subprocess.run([dxc, '-T', 'ps_6_0', '-E', 'main', '-Fo', ps, HERE / 'startup_indirect_mesh_cache.hlsl'], check=True)
    mesh86_cache.publish(work, key, ms, None if depth_only else ps, srv_dxc=dxc if lease_xmem else None)
    assert (work / f'{key:016x}.ps.cso').is_file() != depth_only
    ready = work / f'{key:016x}.mesh86.txt'
    correct = ready.read_bytes()
    ready.write_bytes(correct.replace(b'mesh86 2' if lease_xmem else b'mesh86 1', b'mesh86 0', 1))
    stale = subprocess.run(cmd, env=dict(environment, X2W_MESH86_FIXTURE_REJECT_MARKER='1') if lease_xmem else environment)
    if stale.returncode:
        return stale.returncode
    ready.write_bytes(correct)
    # WARP removes the device on this depth-only workload despite accepting
    # PSO creation. Confirm the ready stage is guarded on software adapters.
    if not depth_only or backend == 'hardware':
        environment['X2W_MESH86_FIXTURE_READY'] = '1'
    return subprocess.run(cmd, env=environment).returncode


if __name__ == '__main__':
    dxc, fixture, backend, *lease = sys.argv[1:]
    for depth_only in ((False,) if lease else (False, True)):
        with tempfile.TemporaryDirectory(prefix='mesh86-', dir=Path.cwd()) as directory:
            result = run(dxc, fixture, backend, Path(directory), depth_only, Path(lease[0]) if lease else None,
                         lease[1] if len(lease) > 1 else None)
            if result:
                sys.exit(result)
