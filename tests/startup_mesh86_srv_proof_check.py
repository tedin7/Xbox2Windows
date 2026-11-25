"""Derive the bounded slot 86 SRV extent from source-built DXIL."""

from pathlib import Path
import subprocess
import sys
import tempfile

from startup_indirect_mesh_cache_check import mesh86_cache, source_envelope


def check_load_count():
    mesh_dump = """; Resource Bindings:
;
; Name Type Format Dim ID HLSL Bind Count
; ---------------------------------------
; source texture byte r/o T0 t0 1

%1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
"""
    # Both loads individually match the one-DWORD form and use the same handle.
    load = ('%{result} = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32'
            '(i32 139, %dx.types.Handle %1, i32 0, i32 undef, i8 1, i32 4)\n')
    one_load = mesh_dump + load.format(result=2)
    assert mesh86_cache.prove_mesh_root_srv(one_load, '') == 4
    try:
        mesh86_cache.prove_mesh_root_srv(one_load + load.format(result=3), '')
    except ValueError as error:
        assert str(error) == 'expected one DWORD load from the verified SRV handle', str(error)
    else:
        raise AssertionError('accepted two DWORD loads from the verified SRV handle')
    print('accepted one DWORD load and rejected two loads from the same SRV handle')


def main(dxc, work):
    original = Path(__file__).with_name('xmem_mesh_root_srv.hlsl').read_text(encoding='utf-8')

    def compile_stage(stage, text):
        source = work / f'{stage}.hlsl'
        shader = work / f'{stage}.cso'
        source.write_text(text, encoding='utf-8')
        profile, entry = ('ms_6_5', 'mesh') if stage == 'ms' else ('ps_6_0', 'pixel')
        subprocess.run([dxc, '-T', profile, '-E', entry, '-Fo', shader, source], check=True)
        return shader

    payloads = (source_envelope(13), source_envelope(0), b'fixture root', b'fixture stream')
    key = mesh86_cache.request_key(*payloads)
    for suffix, data in zip(('m86', 'ps86', 'root', 'stream86'), payloads):
        (work / f'req_{key:016x}.{suffix}').write_bytes(data)
    pixel = compile_stage('ps', original)
    mesh = compile_stage('ms', original)
    ready = work / f'{key:016x}.mesh86.txt'
    mesh86_cache.publish(work, key, mesh, pixel)
    assert ready.read_text().startswith('x2w-mesh86 1 ')
    mesh86_cache.publish(work, key, mesh, pixel, srv_dxc=dxc)
    verified = ready.read_bytes()
    assert verified.startswith(b'x2w-mesh86 2 ') and verified.endswith(b' srv 0 0 4\n')

    def reject(stage, text, diagnostic):
        mutant = compile_stage(stage, text)
        try:
            mesh86_cache.publish(work, key, mutant if stage == 'ms' else mesh,
                                 mutant if stage == 'ps' else pixel, srv_dxc=dxc)
        except ValueError as error:
            assert diagnostic in str(error), str(error)
        else:
            raise AssertionError('published a shader outside the one-DWORD SRV profile')
        assert ready.read_bytes() == verified

    reject('ps', original.replace('return vertex.word;', 'return source.Load(0);'), 'pixel stage')
    pixel = compile_stage('ps', original)
    for load in ('source.Load(4)', 'source.Load2(0).x ^ source.Load2(0).y'):
        reject('ms', original.replace('source.Load(0)', load, 1), 'unsupported DXIL operation')
    dynamic = original.replace('void mesh(out', 'void mesh(uint3 group : SV_GroupID, out')
    reject('ms', dynamic.replace('source.Load(0)', 'source.Load(group.x * 4)', 1), 'unsupported DXIL operation')
    reject('ms', original.replace('register(t0)', 'register(t0, space1)'), 't0 space0')
    print('verified one DWORD at t0 space0 and refused PS access, shifted, dynamic, wide and wrong-space loads')


if __name__ == '__main__':
    check_load_count()
    if len(sys.argv) > 1:
        with tempfile.TemporaryDirectory(prefix='mesh86-srv-proof-', dir=sys.argv[2]) as directory:
            main(sys.argv[1], Path(directory))
