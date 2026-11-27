"""Compile a source-built slot 67 request through the LDS replay CLI."""
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_compute_translate_check import state_a, write_request
from startup_gfx2hlsl_compute_check import CASES


def main(compiler, sdk):
    script = Path(__file__).resolve().parents[1] / 'tools/startup/compile_lds_request.py'
    _, code, rsrc2, group, _ = CASES['F']
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        key = write_request(work, code, state_a(group, rsrc2))
        marker = work / f'req_{key:016x}.c67'
        command = [sys.executable, '-B', script, marker, work / 'output',
                   '--compiler', compiler, '--sdk', sdk, '--cache', work / 'cache',
                   '--rw-image', '0', '0', '--wave-size', '32']
        subprocess.run(command, check=True)
        bundle, = (work / 'output').iterdir()
        source = (bundle / f'{key:016x}.cs.hlsl').read_text()
        assert '#pragma x2w_lds ' in source and '#pragma x2w_vop2 ' in source
        assert 'GroupMemoryBarrierWithGroupSync();' in source
        assert (bundle / f'{key:016x}.cs.cso').read_bytes().startswith(b'DXBC')
        assert 'group 64 1 1' in (bundle / f'{key:016x}.txt').read_text().splitlines()

        data = marker.with_suffix('.data')
        data.write_bytes(bytes(b ^ 1 for b in data.read_bytes()))
        refused = subprocess.run(command, capture_output=True, text=True)
        assert refused.returncode and 'do not match the key' in refused.stderr, refused.stderr
        assert list((work / 'output').iterdir()) == [bundle]


if __name__ == '__main__':
    main(*sys.argv[1:])
