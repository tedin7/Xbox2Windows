"""s_nor_b32 and s_nand_b32 of lane masks into exec_lo or an SGPR in tools/startup/gfx2hlsl.py (Part of #1723).

With every source a lane mask, the lane's bit of ~(a op b) is exact. The emitted lines are run on the integer
evaluator of the sibling check, and with `dispatch <dxc> <offscreen_check> <adapter>` also as a D3D12 compute
shader whose UAV readback is compared with an integer reference (bit = ~(a | b) & 1 for NOR, ~(a & b) & 1 for
NAND, over every pair of source bits). Forms with SCC readers, mixed mask and word sources, vcc_lo, exec_hi and
XNOR into exec_lo stay refused; the sibling check (startup_gfx2hlsl_sop2_complement_check.py) pins those.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_sop2_complement_check as C  # noqa: E402

REFERENCES = {
    's_nor_b32': lambda a, b: ~(a | b) & 1,
    's_nand_b32': lambda a, b: ~(a & b) & 1,
}
DESTINATIONS = ('exec_lo', 's3')
BIT_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


def lowered(name, dest):
    stage = C.Probe()
    stage.masks.update((1, 2))
    stage.one(name, '%s %s, s1, s2' % (name, dest))
    return stage


def cpu_checks():
    for name, reference in REFERENCES.items():
        for dest in DESTINATIONS:
            stage = lowered(name, dest)
            key = 'exec' if dest == 'exec_lo' else dest
            wrong = []
            for a, b in BIT_PAIRS:
                out = C.run_lines(stage.lines, {key: 0, 's1': a, 's2': b, 'scc': C.SCC_UNSET})
                if out[key] != reference(a, b):
                    wrong.append((a, b, out[key]))
            C.check('%s into %s matches the reference on every bit pair' % (name, dest), not wrong, str(wrong))
    C.check('NOR and NAND references differ', REFERENCES['s_nor_b32'](0, 1) != REFERENCES['s_nand_b32'](0, 1))


def accepts(name, dest, kind, group):
    stage = C.Probe()
    stage.kind, stage.pipe = kind, SimpleNamespace(group=group)
    stage.masks.update((1, 2))
    try:
        stage.one(name, '%s %s, s1, s2' % (name, dest))
    except C.G.Unsupported:
        return False
    return True


def stage_checks():
    """exec_lo leaves lanes 32..63 alone, so a wave64 stage must keep refusing it (a wave64 wave would keep stale
    upper lanes active while the low lanes are rewritten). Only compute with at most 32 threads per group is proven
    wave32; the wave size of a pixel or vertex stage is unknown. An SGPR destination is exact per lane in every stage."""
    for name in REFERENCES:
        for kind, group, want in (('ps', None, False), ('cs', (32, 1, 1), True), ('cs', (8, 4, 1), True),
                                  ('cs', (64, 1, 1), False), ('cs', (8, 8, 1), False),
                                  ('ms', None, False), ('vs', None, False)):
            C.check('%s into exec_lo in a %s stage%s is %s' % (name, kind, ' of %s threads' % (group,) if group else '',
                                                              'accepted' if want else 'refused'),
                    accepts(name, 'exec_lo', kind, group) == want)
        C.check('%s into s3 stays accepted in a wave64 mesh stage' % name, accepts(name, 's3', 'ms', None))


def dispatch(dxc, runner, adapter, name, dest, directory):
    stage = lowered(name, dest)
    key = 'exec' if dest == 'exec_lo' else dest
    fill = tuple(((i + 1) * 0x9e3779b1 & 0xffffffff) | 1 for i in range(8))
    lines = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, (a, b) in enumerate(BIT_PAIRS):
        lines.extend((
            '    {',
            f'        uint s1 = data.Load({4 * 2 * i}) ^ {fill[2 * i] ^ a:#010x}u;',
            f'        uint s2 = data.Load({4 * (2 * i + 1)}) ^ {fill[2 * i + 1] ^ b:#010x}u;',
            '        uint s3 = 0; uint exec = 0; bool scc = true;',
        ))
        lines.extend('        ' + line for line in stage.lines)
        lines.append(f'        data.Store({4 * (8 + i)}, {key});')
        lines.append('    }')
    lines.append('}')
    stem = '%s_%s' % (name, dest)
    src, cso, output = (directory / (stem + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(lines) + '\n', encoding='ascii')
    compiled = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                              stderr=subprocess.PIPE, text=True)
    if compiled.returncode or compiled.stderr:
        raise AssertionError(compiled.stderr or 'DXC exited with %d' % compiled.returncode)
    run = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * 16), str(output)],
                         capture_output=True, text=True)
    print(run.stdout, end='')
    if run.returncode == 77:
        sys.exit(77)
    if run.returncode or run.stderr:
        raise AssertionError(run.stderr or 'raw UAV dump exited with %d' % run.returncode)
    actual = struct.unpack('<16I', output.read_bytes())
    assert actual[:8] == fill, '%s: operand region changed' % stem
    return actual[8:12]


def check_dispatch(dxc, runner, adapter):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name, reference in REFERENCES.items():
            for dest in DESTINATIONS:
                got = dispatch(dxc, runner, adapter, name, dest, Path(tmp))
                want = tuple(reference(a, b) for a, b in BIT_PAIRS)
                C.check('%s into %s on the GPU: got %s, want %s' % (name, dest, got, want), got == want)
    if C.failures:
        sys.exit(1)


def main():
    cpu_checks()
    stage_checks()
    if C.failures:
        print('FAILED: %s' % ', '.join(C.failures))
        sys.exit(1)
    print('startup_gfx2hlsl_s_nor_nand_check: ok')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        check_dispatch(*sys.argv[2:])
    else:
        sys.exit(__doc__)
