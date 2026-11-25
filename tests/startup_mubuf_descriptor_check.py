"""Run accepted and refused hardware buffer descriptors through the translated draw path."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_buffer_address_check import encode_mubuf
from startup_stream_draw_check import container, gfx2hlsl, vertex


def check(directory, consumer, backend):
    cache = directory / 'cache'
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ('vs', 'ps', 'root', 'pixels'))
    trap_log = directory / 'trap.log'
    vs.write_bytes(vertex())
    code = (bytes.fromhex('040408f4000000fa7fc08cbf')  # s_load_dwordx4 s[16:19], s[8:9], 0; wait
            + encode_mubuf(op=0, vdata=4)
            + bytes.fromhex('703f8cbf8002007ef202027e0f1800f804000001000081bf'))
    ps.write_bytes(container(0, struct.unpack('<%dI' % (len(code) // 4), code)))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1',
               X2W_D3D12_RENDER='1', X2W_D3D12_DEBUG='1', X2W_D3D12_DEVICE_NOOP='1',
               X2W_TEST_FORMATTED_BUFFER='1', X2W_DRAW_NULL_IDENTITY='1', X2W_SHADER_CACHE=str(cache), X2W_TRAP_LOG=str(trap_log))
    args = [consumer, vs, ps, root]
    subprocess.run(args, env=env, check=True)
    subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    manifest, = cache.glob('*.txt')
    text = manifest.read_text(encoding='ascii')
    assert 'descword3 0' in text and 'srv 0 buf table 0 0' in text, text
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode == 77:
        return 77
    if result.returncode:
        raise RuntimeError(f'consumer failed ({result.returncode})\n{result.stderr[-5000:]}')
    clear, black, red = bytes((0, 0, 255, 255)), bytes((0, 0, 0, 255)), bytes((255, 0, 0, 255))
    want = (clear + clear + black + clear + red + clear * 3) * 4
    actual = output.read_bytes()
    assert actual == want, (actual.hex(), want.hex())
    log = trap_log.read_text(errors='replace')
    for diagnostic in ('missing hardware descriptor 0', 'unsupported FORMAT 2', 'unresolved buffer'):
        assert f'formatted buffer refused ({diagnostic})' in log, (diagnostic, log)
    print(f'{backend}: missing metadata, unsupported FORMAT and unresolved storage stayed blue; valid zero and one drew black and red')
    return 0


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        sys.exit(check(Path(directory), *sys.argv[1:]))
