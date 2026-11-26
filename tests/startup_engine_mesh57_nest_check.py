"""Slot 57 mesh: scalar state written inside an s_cbranch_execz block, read after the join (lane-exact EXEC model, rule 3b).

usage: startup_engine_mesh57_nest_check.py
       startup_engine_mesh57_nest_check.py binary <llvm-mc>
       startup_engine_mesh57_nest_check.py draw <dxc> <offscreen_check> <warp|hardware>

The wave block of startup_mesh57_skeleton_check.py: the first six threads of wave 0 enter the block, wave 1 has no active
lane. The guest decides s_cbranch_execz for the WHOLE wave, so wave 0 runs the block (also threads 6..63, whose own exec
bit is clear) and wave 1 skips it. A scalar the block writes reads afterwards as its block result in wave 0 and as its
pre-block value in wave 1 (rule 3b: per wave, not per thread and not per group). The vertex red is that scalar (s40, float
bits), the green is v7, written only by lanes whose exec bit is set inside the block.

  nest_plain   block: s44 = 9; v7 = 1.0; s40 = 0.5                                     red wave0 128, wave1 64; green 255 for tid < 6
  nest_scc     the same with an inner s_cmp_eq_u32 s44, 9 / s_cbranch_scc1 over v7 = 1.0   green 0 everywhere (the branch is taken)
  nest_scc_not the inner branch not taken (s_cmp_eq_u32 s44, 0)                           green 255 for tid < 6
  nest_execz   an inner s_cbranch_execz range (v_cmpx 3 > v0, saved exec restored)       green 255 for tid < 3

Each case is drawn and compared with a CPU reference computed here from the rule. Three wrong lowerings are drawn against
the same reference and must differ: the decision taken per thread, per group (both waves), and the range run by every wave
(the pre-block value not kept). Without arguments the check is skipped (the engine needs dxc for a mesh stage).
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
from pathlib import Path
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_mesh57_skeleton_check as T  # noqa: E402

BOUND = 6
OLD_RED, NEW_RED = 64, 128      # float bits 0x3e808081 (about 0.251) and 0x3f000000 (0.5), times 255, rounded
INNER_EXECZ_BOUND = 3

HEAD = 's_mov_b32 s44, 9\n'
TAIL = 'v_mov_b32 v7, 1.0\ns_mov_b32 s40, 0x3f000000'
CASES = {
    'nest_plain': (HEAD + TAIL, lambda tid: 255 if tid < BOUND else 0),
    'nest_scc': (HEAD + 's_cmp_eq_u32 s44, 9\ns_cbranch_scc1 1\n' + TAIL, lambda tid: 0),
    'nest_scc_not': (HEAD + 's_cmp_eq_u32 s44, 0\ns_cbranch_scc1 1\n' + TAIL, lambda tid: 255 if tid < BOUND else 0),
    'nest_execz': (HEAD + 's_mov_b64 s[46:47], exec\nv_cmpx_gt_u32_e32 3, v0\ns_cbranch_execz 2\n' + TAIL
                   + '\ns_mov_b64 exec, s[46:47]', lambda tid: 255 if tid < INNER_EXECZ_BOUND else 0),
}


# The words of each body as llvm-mc assembles it (checked by `binary`), so the draw run needs no assembler.
WORDS = {
    'nest_plain': 'beac0389 7e0e02f2 bea803f0',
    'nest_scc': 'beac0389 bf06892c bf850001 7e0e02f2 bea803f0',
    'nest_scc_not': 'beac0389 bf06802c bf850001 7e0e02f2 bea803f0',
    'nest_execz': 'beac0389 beae047e 7da80083 bf880002 7e0e02f2 bea803f0 befe042e',
}


def case_mesh(label):
    return T.wave_load_mesh(T.WAVE_BLOCK_PREFIX, WORDS[label])


def reference(points, colours, triangles, green, wave_red):
    rendered = list(colours)
    for tri in triangles:
        for tid in tri:
            slot = tid & 15
            rendered[slot] = (wave_red(tid & ~63), green(tid), *colours[slot][2:])
    return T.rasterise(points, rendered, triangles, 128, len(triangles))


def mutants(source):
    """HLSL text of each wrong lowering of the wave decision (the gather sites and the outer skip)."""
    gather = re.compile(r'(x2w_any_\d+) = x2w_any\[b\]\[x2w_wave\];')
    assert gather.search(source), 'no wave gather in the translation'
    skips = re.compile(r'\{ skip = true; skip_to = \d+; \}')
    assert skips.search(source), 'no wave skip in the translation'
    return {
        'per-thread decision': gather.sub(r'\1 = (exec & 1u);', source),
        'per-group decision': gather.sub(r'\1 = x2w_any[b][0] | x2w_any[b][1];', source),
        'range run by an empty wave': skips.sub('{ }', source),
    }


def draw(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    points, colours, indices = T.geometry()
    # Provoking threads 39 and 5 share the mixed wave 0; 67 and 70 sit in the empty wave 1.
    triangles = [tuple(slot + wave for slot in tri) for tri, wave in zip(indices, (32, 0, 64, 64))]
    words = [a | b << 10 | c << 20 for a, b, c in triangles]
    data = T.buffer(points, colours, words)
    wave_red = {0: NEW_RED, 64: OLD_RED}.__getitem__
    with tempfile.TemporaryDirectory() as tmp:
        for label, (asm, green) in CASES.items():
            mesh = case_mesh(label)
            key, result = T.request(tmp, mesh)
            T.expect(label + ' translates and compiles', result, 'ok')
            if result != 'ok':
                continue
            base = Path(tmp) / ('%016x' % key)
            constants = bytearray(1024)
            uds = T.G.MeshPipeline(T.code(mesh), T.code(T.PIXEL), T.root()).uds
            for slot, count in enumerate((len(triangles), 128)):
                struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)
            image = reference(points, colours, triangles, green, wave_red)
            swapped = reference(points, colours, triangles, green, {0: OLD_RED, 64: NEW_RED}.__getitem__)
            T.expect(label + ' reference differs from the swapped one', image != swapped, True)
            source = base.with_suffix('.ms57.hlsl').read_text()
            shaders = [(label, base.with_suffix('.ms57.cso'), 0)]
            for what, text in mutants(source).items():
                hlsl = Path(tmp) / 'mutant.hlsl'
                hlsl.write_text(text)
                cso = Path(tmp) / (label + '-' + what.replace(' ', '-') + '.cso')
                subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', cso, hlsl], check=True, capture_output=True)
                shaders.append((label + ' with ' + what, cso, 1))
            path = Path(tmp) / 'nest.bin'
            for what, shader, status in shaders:
                path.write_bytes(bytes(constants) + data + b'\xcd' * 80 + b''.join(map(bytes, image)))
                run = subprocess.run([renderer, adapter, '--mesh-indexed', shader, base.with_suffix('.ps57.cso'), path],
                                     capture_output=True, text=True)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    return 77
                matched = 'pixels=1024 mismatches=0' in run.stdout
                T.expect(what + (' matches the reference' if not status else ' differs from the reference'),
                         (run.returncode, matched), (status, not status))
                print('%s: %s' % (what, run.stdout.strip().splitlines()[-1:]))
    return 0


def binary(mc):
    for label, (asm, _) in CASES.items():
        T.expect(label + ' words match llvm-mc', ' '.join(T.assemble(mc, asm)), WORDS[label])


def main(argv):
    status = 0
    if argv[1:2] == ['binary'] and len(argv) == 3:
        binary(argv[2])
    elif argv[1:2] == ['draw'] and len(argv) == 5:
        status = draw(*argv[2:])
    elif len(argv) == 1:
        print('skip: the engine mesh stage needs dxc; run `draw`')
        return 77
    else:
        print(__doc__)
        return 2
    if T.FAILED:
        return 1
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
