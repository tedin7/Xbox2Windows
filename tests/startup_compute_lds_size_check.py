"""Check LDS sizing through slot 67 requests and D3D12 WARP readback."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_compute_lds_b64_check import G, ROOT, SENTINEL, USER7, WORDS, assemble
from startup_gfx2hlsl_compute_check import refused, renderer_diagnostics, root_blob


SOURCE = """\
s_mov_b32 s22, s6
s_mov_b32 s23, 0
s_load_dwordx4 s[24:27], s[22:23], 0x10
s_waitcnt lgkmcnt(0)
v_add_nc_u32 v11, 17, v0
v_lshlrev_b32 v12, 2, v0
ds_write_b32 v12, v11 offset:{offset}
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v13, 0x10, v12
v_and_b32 v13, 0x1c, v13
ds_read_b32 v14, v13 offset:{offset}
s_waitcnt lgkmcnt(0)
buffer_store_dword v14, v12, s[24:27], 0 offen
s_endpgm
"""

CASES = {
    'dispatch': (704, 0, 1024),
    '24k': (20992, 40, 24576),
}


def shader_code(offset):
    disp = struct.pack('<H', offset).hex()
    return bytes.fromhex(
        '060396be800397be0b0608f4100000fa7fc08cbf9100164a82001834'
        f'{disp}34d80c0b00007fc08cbf00008abf90181a3a9c1a1a36'
        f'{disp}d8d80d00000e7fc08cbf001070e00c0e0680000081bf'
    )


def request(directory, offset, program_kib):
    code = shader_code(offset)
    state = struct.pack('<7I', 0, 0, 0, (1 << 16) | 8, 1, 0, USER7 | ((program_kib * 2) << 15))
    data = bytes(4)
    root = root_blob()
    desc = struct.pack('<12Q', 0, 0, len(state), 0, 0, 0, len(code), 0, len(data), 0, 0, 0)
    key = G.compute_key(code, data, root, desc, state, b'')
    prefix = directory / ('req_%016x' % key)
    for ext, value in (('.desc', desc), ('.sa', state), ('.sb', b''), ('.data', data), ('.root', root), ('.c67', code)):
        prefix.with_suffix(ext).write_bytes(value)
    return key, prefix.with_suffix('.lds')


def check_controls(directory):
    key, sidecar = request(directory, 704, 0)

    def load():
        return G.load_compute_request(directory, key)

    refused('COMPUTE_PGM_RSRC2 allocates 0 bytes', load)
    failure = directory / ('%016x.fail' % key)
    failure.touch()
    assert key not in G.pending_keys(directory, skip_failed=True)
    sidecar.write_bytes(struct.pack('<I', 1024))
    later = failure.stat().st_mtime + 1
    os.utime(sidecar, (later, later))
    assert key in G.pending_keys(directory, skip_failed=True)
    pipe = load()
    assert 'groupshared uint lds[256];' in pipe.hlsl_cs()
    subprocess.run([sys.executable, '-B', G.__file__, '--pending', str(directory)],
                   env=dict(os.environ, X2W_FXC='', X2W_TRANSLATE_JOBS='1'), check=True)
    assert (directory / ('%016x.cs.hlsl' % key)).read_text() == pipe.hlsl_cs()
    assert (directory / ('%016x.txt' % key)).read_text() == pipe.manifest()
    assert not failure.exists()
    for size, reason in ((0, 'positive whole number'), (1023, 'positive whole number'),
                         (512, 'one bounded dword'), (32772, 'exceeds 32 KiB')):
        sidecar.write_bytes(struct.pack('<I', size))
        refused(reason, load)
    sidecar.write_bytes(b'\x00' * 5)
    refused('little-endian uint32', load)
    sidecar.write_bytes(struct.pack('<I', 32768))
    assert 'groupshared uint lds[8192];' in load().hlsl_cs()

    key, sidecar = request(directory, 20992, 24)
    static = load()
    assert 'groupshared uint lds[6144];' in static.hlsl_cs()
    sidecar.write_bytes(struct.pack('<I', 32768))
    assert (static.hlsl_cs(), static.manifest()) == (load().hlsl_cs(), load().manifest())

    key, sidecar = request(directory, 20992, 40)
    refused('COMPUTE_PGM_RSRC2 allocates 40 KiB', load)
    sidecar.write_bytes(struct.pack('<I', 24576))
    assert (static.hlsl_cs(), static.manifest()) == (load().hlsl_cs(), load().manifest())
    print('LDS request sizing, retry, 24 and 32 KiB, and static HLSL/manifest stability pass')


def check_dispatch(directory, name, mode, dxc, runner):
    offset, program_kib, dispatch_bytes = CASES[name]
    key, sidecar = request(directory, offset, program_kib)
    sidecar.write_bytes(struct.pack('<I', dispatch_bytes))
    hlsl = G.load_compute_request(directory, key).hlsl_cs()
    assert 'groupshared uint lds[%d];' % (dispatch_bytes // 4) in hlsl
    if mode == 'control':
        peer = '    v13 = 0x00000010u ^ v12;\n'
        assert hlsl.count(peer) == 1
        hlsl = hlsl.replace(peer, '    v13 = v12;\n')

    source, bytecode, case = (directory / ('lds.' + ext) for ext in ('hlsl', 'cso', 'case'))
    source.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(bytecode), str(source)], check=True)
    expected = [17 + (lane ^ 4) for lane in range(8)] + [SENTINEL] * (WORDS - 8)
    case.write_bytes(struct.pack('<I', 0x010101) + struct.pack('<%dI' % WORDS, *expected) + bytes(64))
    run = subprocess.run([runner, 'warp', '--compute-store-case', str(bytecode), str(case)],
                         capture_output=True, text=True)
    print(run.stdout, end='')
    if run.returncode == 77:
        return 77
    diagnostics = renderer_diagnostics(run.stderr, 'warp')
    if mode == 'control':
        first = f'word=0 expected={expected[0]:08x} actual=00000011'
        assert run.returncode == 1 and diagnostics == [first], (run.returncode, run.stderr)
        print(f'{name} same-lane control rejected at {diagnostics[0]}')
    else:
        assert run.returncode == 0 and not diagnostics, (run.returncode, run.stderr)
        print(f'{name}: all {WORDS} words match')
    return 0


if __name__ == '__main__':
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        directory = Path(temporary)
        if len(sys.argv) == 1:
            check_controls(directory)
        elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
            for offset, *_ in CASES.values():
                assert assemble(sys.argv[2], SOURCE.format(offset=offset)) == shader_code(offset)
            print('Both synthetic programs match assembled bytes')
        elif len(sys.argv) == 5 and sys.argv[1] in ('readback', 'control'):
            mode, name, dxc, runner = sys.argv[1:]
            sys.exit(check_dispatch(directory, name, mode, dxc, str(Path(runner).resolve())))
        else:
            sys.exit('expected no arguments, binary <llvm-mc>, or readback/control <dispatch/24k> <dxc> <runner>')
