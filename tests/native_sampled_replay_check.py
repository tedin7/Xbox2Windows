"""Replay a source-built VS/PS pair, then damage each published identity in turn."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
from compile_sampled_pixel_request import build
from gfx2hlsl import fnv, pipeline_key


def main(writer, compiler, sdk):
    with TemporaryDirectory() as folder:
        work = Path(folder)
        subprocess.run([writer, '--write-fixture', work], check=True)
        # Device slot order can differ from shader stage order.
        stages = [(work / name).read_bytes() for name in ('pixel.bin', 'vertex.bin', 'root.bin')]
        base = pipeline_key(*stages)
        for ext, data in zip(('.a', '.b', '.root'), stages):
            (work / f'req_{base:016x}{ext}').write_bytes(data)
        request = f'rtv1 {base} 26 54'.encode()
        marker = work / f'req_{fnv(request):016x}.rtv'
        marker.write_bytes(request)
        args = (marker, work / 'bundles', compiler, sdk, work / 'cache')
        output = build(*args)
        assert build(*args) == output
        meta = json.loads((output / 'replay.json').read_bytes())
        assert meta['pixel_bytes'] == len(stages[0]) and meta['vertex_bytes'] == len(stages[1])
        for name, digest in meta['sha256'].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
        def refuse(message):
            try:
                build(*args)
            except ValueError as exc:
                assert message in str(exc), exc
            else:
                raise AssertionError('accepted damaged replay input')
        next(output.glob('*.vs.cso')).write_bytes(b'damaged')
        refuse('damaged replay bundle')
        (work / f'req_{base:016x}.b').write_bytes(stages[0])
        refuse('key mismatch')
        marker.write_bytes(f'rtv1 {base} 54 26'.encode())
        refuse('matching R11G11B10_FLOAT')
    print('Matched two-stage cold/warm replay and three refusal controls passed')


if __name__ == '__main__':
    main(*sys.argv[1:])
