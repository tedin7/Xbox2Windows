"""The mesh replay oracle (X2W_REPLAY_ORACLE=mesh, src/startup_stubs/d3d12_mesh_oracle.inl) on a real D3D12 draw.

usage: startup_mesh57_oracle_check.py <dxc> <offscreen_check> <warp|hardware> <startup_indirect_draw_indexed>

The slot 57 group fixture of startup_mesh57_skeleton_check.py is published through the stand-in, translated with
X2W_MESH_WAVELOOP=oracle (the proof translation plus its serial wave-loop sibling) and drawn by the stand-in (submitted
through its queue slot 10, X2W_TEST_MESH57_GUEST_QUEUE=1, where the oracle compares), which
reads the target back:
  1. oracle on, both translations right: the picture is the CPU reference and oracle.jsonl says equal;
  2. the published proof translation is made wrong (each lane reads its neighbour's vertex slot) and recompiled:
     without the oracle the picture is the wrong one; with the oracle the picture is still that same wrong picture
     (the oracle never changes what is presented) and oracle.jsonl says differs, with the differing texel count of
     the two pictures;
  3. the serial sibling is missing: the draw is unchanged and the verdict is oracle_unavailable with the reason;
  4. a key that is not selected (mesh:<other key>) is drawn as without the oracle and writes nothing.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_mesh57_skeleton_check as S  # noqa: E402

G = S.G
LANE = 'min(tid & 63u, 32u)'
SWAPPED_LANE = 'min((tid ^ 1u) & 63u, 32u)'


def oracle_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verdicts(path):
    return [line for line in oracle_lines(path) if 'verdict' in line]


def run(dxc, renderer, adapter, publisher):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = S.geometry()
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    correct = b''.join(map(bytes, S.rasterise(points, colours, indices, len(points), len(indices))))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        status, published = S.publish_request(directory, publisher, adapter, S.GROUP_MESH)
        if status:
            return status
        cache, key, root_blob = published
        os.environ['X2W_MESH_WAVELOOP'] = 'oracle'
        if G.translate_request(str(cache), key) != 'ok':
            raise AssertionError('mesh request translation failed')
        name = cache / ('%016x' % key)
        assert Path(str(name) + '.ms57w.txt').read_text() == 'ok\n', 'no serial sibling was published'
        data, arguments, pixels, log, oracle = (directory / n for n in
                                               ('data.bin', 'args.bin', 'pixels.bin', 'runtime.log', 'oracle.jsonl'))
        data.write_bytes(S.buffer(points, colours, words))
        arguments.write_bytes(b'\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00')
        base = dict(os.environ, X2W_MESH_REQUEST_FIXTURE='1', X2W_TEST_MESH57_EXCHANGE=str(directory),
                    X2W_TEST_MESH57_RUNTIME='1', X2W_TEST_MESH57_DATA=str(data), X2W_TEST_MESH57_ARGS=str(arguments),
                    X2W_TEST_MESH57_PIXELS=str(pixels), X2W_TRAP_LOG=str(log), X2W_REPLAY_ORACLE_FILE=str(oracle),
                    X2W_TEST_MESH57_GUEST_QUEUE='1')
        for variable in ('X2W_REPLAY_ORACLE', 'X2W_TEST_MESH57_NO_CONSTANTS', 'X2W_TEST_MESH57_HEAP'):
            base.pop(variable, None)
        last_output = ['']

        def draw(selection=None):
            for path in (pixels, log, oracle):
                path.unlink(missing_ok=True)
            env = dict(base)
            if selection is not None:
                env['X2W_REPLAY_ORACLE'] = selection
            result = subprocess.run([publisher, adapter, 'mesh'], env=env, capture_output=True, text=True)
            if result.returncode == 77:
                raise SystemExit(77)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            last_output[0] = (result.stdout + result.stderr)[-3000:]
            return pixels.read_bytes(), verdicts(oracle)

        def explain(lines):
            text = log.read_text(errors='replace').splitlines() if log.exists() else []
            return '%s\nlog tail:\n%s\noutput tail:\n%s' % (lines, '\n'.join(line[:240] for line in text[-60:]), last_output[0])

        # 1. both translations right
        image, lines = draw('mesh')
        assert image == correct, 'oracle on: picture differs from the CPU reference\n' + explain(lines)
        assert [line['verdict'] for line in lines] == ['equal'], explain(lines)
        assert lines[0]['key'] == '%016x' % key and lines[0]['draw'] == 0, lines
        assert lines[0]['targets'][0]['target'] == 'rt0' and lines[0]['targets'][0]['texels'] == S.SIDE * S.SIDE, lines
        header = oracle_lines(oracle)[0]
        assert header == {'oracle': 'mesh', 'f32_ulp': 4, 'f16_ulp': 2, 'other': 'exact'}, header

        # 2. a wrong proof translation: each lane reads its neighbour's slot
        hlsl = Path(str(name) + '.ms57.hlsl').read_text()
        assert hlsl.count(LANE) == 1, 'the lane expression of the fixture moved'
        pipe = G.MeshPipeline(S.code(S.GROUP_MESH), S.code(S.PIXEL), root_blob)
        pipe.hlsl_ms()
        G.publish_mesh57_stages(str(name), [('ms', pipe.mesh_profile(), hlsl.replace(LANE, SWAPPED_LANE))],
                                str(cache), True)
        wrong, lines = draw()
        assert wrong != correct and not lines, 'the swapped lane must change the picture'
        differing = sum(wrong[i:i + 4] != correct[i:i + 4] for i in range(0, len(correct), 4))
        image, lines = draw('mesh')
        assert image == wrong, 'oracle on: the presented picture changed\n' + explain(lines)
        assert [line['verdict'] for line in lines] == ['differs'], explain(lines)
        target = lines[0]['targets'][0]
        assert target['target'] == 'rt0' and target['differing'] == differing, (differing, lines)
        first = target['first']
        x, y = first['x'], first['y']
        assert wrong[4 * (y * S.SIDE + x):4 * (y * S.SIDE + x) + 4] != correct[4 * (y * S.SIDE + x):4 * (y * S.SIDE + x) + 4]

        # 3. no serial sibling
        Path(str(name) + '.ms57w.txt').unlink()
        image, lines = draw('mesh')
        assert image == wrong, 'oracle unavailable: the presented picture changed'
        assert [(line['verdict'], line['reason'][:17]) for line in lines] == [('oracle_unavailable', 'no serial sibling')], lines

        # 4. another key selected
        image, lines = draw('mesh:%016x' % (key ^ 1))
        assert image == wrong and not lines and not oracle.exists(), lines
    return 0


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    try:
        status = run(*argv[1:])
    except SystemExit as stop:
        status = stop.code
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
