"""Exercise descriptor heap rollover through the slot 67 translated compute harness."""
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_compute_request_check import prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import USER7, root_blob
from startup_compute_dispatch_check import lines


def run(args, env, mode):
    result = subprocess.run([*args, mode], env=env, capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode == 77:
        sys.exit(77)
    if result.returncode:
        sys.exit(f'{mode}: consumer exit {result.returncode}\n{result.stderr[-5000:]}')
    return result


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env['X2W_COMPUTE_VIEW_TOKENS'] = '1'
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        else:
            env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
        (directory / 'compute_root').write_bytes(root_blob(constants=1, table_range=True))
        (directory / 'program').write_bytes(bytes.fromhex('000081bf'))  # s_endpgm
        (directory / 'data').write_bytes(bytes(range(16)))
        (directory / 'state').write_bytes(state_a((1, 1, 1), USER7))
        args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
                directory / 'program', directory / 'data', directory / 'state']
        request = run(args, env, 'descriptor-ring-request')
        key, = [line.split()[3] for line in lines(request, 'x2w-draw compute request') if ' published ' in line]

        shader = directory / 'descriptor_ring.hlsl'
        shader.write_text("""\
cbuffer Entry : register(b0) { uint index; };
ByteAddressBuffer input_bank : register(t0);
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)]
void main() {
    uint offset = 4 * index;
    result.Store(offset, input_bank.Load(offset));
}
""")
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cache / f'{key}.cs.cso', shader], check=True)
        # Unused SRVs are valid null descriptors. A 32-entry table packs exactly into each 512-entry segment.
        manifest = ['x2w-manifest 2', 'stage cs', 'group 1 1 1', 'ud 1', 'udsrc 0 rootconst 0 0',
                    'srv 0 buf table 1 0']
        manifest.extend(f'srv {i} buf table 1 96' for i in range(1, 31))
        manifest.append('uav 0 table 1 64')
        (cache / f'{key}.txt').write_text('\n'.join(manifest) + '\n')

        result = run(args, env, 'dispatch-descriptor-ring')
        assert not lines(result, 'refused: '), result.stderr[-5000:]


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit('usage: startup_descriptor_ring_check.py <startup_stream_consumer> <dxc> <warp|hardware>')
    main(*sys.argv[1:])
