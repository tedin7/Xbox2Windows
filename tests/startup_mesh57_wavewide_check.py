"""Slot 57 mesh programs that decide for the whole guest wave: EXEC skips, first-lane reads, exports under a narrowed EXEC.

usage: startup_mesh57_wavewide_check.py
       startup_mesh57_wavewide_check.py binary <llvm-mc>
       startup_mesh57_wavewide_check.py draw <dxc> <offscreen_check> <warp|hardware>

Every fixture is a short run of instructions placed after the count exchange of the accepted slot 57 shape (the same
splice as startup_mesh57_skeleton_check.py: vertex_red, the run leaves the vertex red in v7). The guest wave has 64 lanes
and v0 is the flat thread id, so the first guest wave holds threads 0..63 and the second 64..127. The fixtures use tests
of the form `thread < N`, which make lanes of the FIRST wave active and none of the second.

The guest rule, from the public ISA text: s_cbranch_execz is taken when NO lane of the wave is active, a scalar write in
the skipped block lands for every lane of the wave when the branch is not taken, a vector write only for the lanes whose
EXEC bit is set, v_readfirstlane_b32 reads the lowest ACTIVE lane (lane 0 when none is active).

  case                            what the program does                                          red of thread t (wave w = t // 64)
  M01 active wave scalar write    s40 = 0; lanes t < 6 stay active; execz skips `s40 = 1.0`      255 in wave 0 (it has active lanes), 0 in wave 1
  M02 empty wave skips            no lane stays active: the wave skips the scalar write           0 everywhere
  M03 guarded vector write        v8 = 0; lanes t < 6 write v8 = 1.0 inside the block            255 for t < 6, 0 elsewhere (the other lanes of an
                                                                                                  active wave keep their own old value)
  M06 first active lane           lanes t > 2 stay active, s41 = v_readfirstlane(float(t) / 255) 3 in wave 0 (first active lane 3), 64 in wave 1
  M07 no active lane              no lane stays active, v_readfirstlane reads lane 0             7 in wave 0, 71 in wave 1 (lane 0 holds t + 7)

Each case is drawn twice, with the provoking vertices in the first wave only (4 primitives, 12 vertices) and spread over both
waves (the 4 primitives start at thread 32, 0, 64 and 64 of a group of 128). The correct reference is drawn through the
request path's translation and must match; a wrong model (the control, one per case) is drawn against the same shader and
must differ in exactly the number of pixels the two CPU references differ in: M01 a skip decided per lane, M02 a wave that
always runs the block, M03 a vector write by every lane of an active wave, M06 the lane's own value or the highest active lane,
M07 a zero. M08 (readfirstlane of a register rewritten inside the skipped block) and M10 (a read of a lane-0-only register)
must be refused by name. Without arguments the fixtures translate (or are refused) as stated; `binary` compares the words with
llvm-mc; `draw` needs a mesh shader tier (exit 77 without one).
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_mesh57_skeleton_check as T  # noqa: E402

G = T.G
SIDE = T.SIDE
# One red per case and layout: the guest rule above. ASM is the source of WORDS (checked by `binary`).
ASM = {
    'M01': 's_mov_b32 s40, 0\nv_cmp_gt_u32 vcc, 6, v0\ns_and_saveexec_b64 s[20:21], vcc\ns_cbranch_execz .L\ns_mov_b32 s40, 1.0\n.L:\n'
           's_mov_b64 exec, -1\nv_mov_b32 v7, s40\n',
    'M02': 's_mov_b32 s40, 0\nv_cmp_gt_u32 vcc, 0, v0\ns_and_saveexec_b64 s[20:21], vcc\ns_cbranch_execz .L\ns_mov_b32 s40, 1.0\n.L:\n'
           's_mov_b64 exec, -1\nv_mov_b32 v7, s40\n',
    'M03': 'v_mov_b32 v8, 0\nv_cmp_gt_u32 vcc, 6, v0\ns_and_saveexec_b64 s[20:21], vcc\ns_cbranch_execz .L\nv_mov_b32 v8, 1.0\n.L:\n'
           's_mov_b64 exec, -1\nv_mov_b32 v7, v8\n',
    'M06': 'v_cvt_f32_u32 v9, v0\nv_mul_f32 v9, 0x3b808081, v9\nv_cmp_lt_u32 vcc, 2, v0\ns_and_saveexec_b64 s[20:21], vcc\n'
           'v_readfirstlane_b32 s41, v9\ns_mov_b64 exec, -1\nv_mov_b32 v7, s41\n',
    'M07': 'v_add_nc_u32 v9, 7, v0\nv_cvt_f32_u32 v9, v9\nv_mul_f32 v9, 0x3b808081, v9\nv_cmp_gt_u32 vcc, 0, v0\n'
           's_and_saveexec_b64 s[20:21], vcc\nv_readfirstlane_b32 s41, v9\ns_mov_b64 exec, -1\nv_mov_b32 v7, s41\n',
    'M08': 'v_cvt_f32_u32 v9, v0\nv_cmp_gt_u32 vcc, 6, v0\ns_and_saveexec_b64 s[20:21], vcc\ns_cbranch_execz .L\nv_mov_b32 v9, 1.0\n'
           'v_readfirstlane_b32 s41, v9\n.L:\ns_mov_b64 exec, -1\nv_mov_b32 v7, s41\n',
    'M10': 'v_mov_b32 v7, v5\n',
}
WORDS = {
    'M01': 'bea80380 7d880086 be94246a bf880001 bea803f2 befe04c1 7e0e0228',
    'M02': 'bea80380 7d880080 be94246a bf880001 bea803f2 befe04c1 7e0e0228',
    'M03': '7e100280 7d880086 be94246a bf880001 7e1002f2 befe04c1 7e0e0308',
    'M06': '7e120d00 101212ff 3b808081 7d820082 be94246a 7e520509 befe04c1 7e0e0229',
    'M07': '4a120087 7e120d09 101212ff 3b808081 7d880080 be94246a 7e520509 befe04c1 7e0e0229',
    'M08': '7e120d00 7d880086 be94246a bf880002 7e1202f2 7e520509 befe04c1 7e0e0229',
    'M10': '7e0e0305',
}
# red of thread t under the guest rule, and the wrong models (control label -> red of thread t).
CASES = {
    'M01': (lambda t: 255 if t < 64 else 0, {'skip decided per lane': lambda t: 255 if t < 6 else 0}),
    'M02': (lambda t: 0, {'a wave that always runs the block': lambda t: 255}),
    'M03': (lambda t: 255 if t < 6 else 0, {'vector write by every lane of an active wave': lambda t: 255 if t < 64 else 0}),
    'M06': (lambda t: 3 if t < 64 else 64, {"the lane's own value": lambda t: t,
                                             'the highest active lane': lambda t: 63 if t < 64 else 127}),
    'M07': (lambda t: 7 if t < 64 else 71, {'a zero when no lane is active': lambda t: 0}),
}
REFUSED = {'M08': None, 'M10': 'is read where it is not defined for every lane'}
LAYOUTS = (('first wave', (0, 0, 0, 0), (4, 12)), ('both waves', (32, 0, 64, 64), (4, 128)))
FAILED = T.FAILED


def mesh(name):
    return T.vertex_red(WORDS[name])


def request(directory, name):
    return T.request(directory, mesh(name))


def image(points, colours, triangles, count, red):
    cols = list(colours)
    for tri in triangles:
        for tid in tri:
            slot = tid % 16
            cols[slot] = (red(tid), *colours[slot][1:])
    return T.rasterise(points, cols, triangles, count, len(triangles))


def controls():
    os.environ['X2W_FXC'] = ''          # translate only: no shader compiler is needed to check the request path
    for name in WORDS:
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, name)
        if name in REFUSED:
            T.expect('%s is refused' % name, result != 'ok', True)
            if REFUSED[name]:
                T.expect('%s refusal names the cause' % name, REFUSED[name] in str(result), True)
            print('%s: refused (%s)' % (name, str(result).splitlines()[0][:100]))
        else:
            T.expect('%s translates' % name, result, 'ok')
            print('%s: translates' % name)
    points, colours, indices = T.geometry()
    for name, (rule, wrong) in CASES.items():
        for label, offsets, _counts in LAYOUTS:
            triangles = [tuple(slot + offset for slot in tri) for tri, offset in zip(indices, offsets)]
            count = 128 if offsets[2] else 12
            good = image(points, colours, triangles, count, rule)
            for model, red in wrong.items():
                other = image(points, colours, triangles, count, red)
                T.expect('%s (%s): the control "%s" differs from the rule' % (name, label, model), good != other, True)
    return FAILED


def binary(llvm_mc):
    for name, text in ASM.items():
        with tempfile.TemporaryDirectory() as tmp:
            source, obj, raw = (str(Path(tmp) / ('p' + ext)) for ext in ('.s', '.o', '.bin'))
            Path(source).write_text(text)
            subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64', '-filetype=obj', '-o', obj, source],
                           check=True)
            subprocess.run([llvm_mc.replace('llvm-mc', 'llvm-objcopy'), '-O', 'binary', '--only-section=.text', obj, raw], check=True)
            data = Path(raw).read_bytes()
        built = ' '.join('%08x' % int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4))
        T.expect('%s words match llvm-mc' % name, built, WORDS[name])
    print('fixture words match llvm-mc')


def draw(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = T.geometry()
    done = 0
    for name, (rule, wrong) in CASES.items():
        for label, offsets, counts in LAYOUTS:
            triangles = [tuple(slot + offset for slot in tri) for tri, offset in zip(indices, offsets)]
            words = [a | b << 10 | c << 20 for a, b, c in triangles]
            count = 128 if offsets[2] else 12
            with tempfile.TemporaryDirectory() as tmp:
                key, result = request(tmp, name)
                if result != 'ok':
                    raise AssertionError('%s: %s' % (name, result))
                base = Path(tmp) / ('%016x' % key)
                pipe = G.MeshPipeline(T.code(mesh(name)), T.code(T.PIXEL), T.root())
                constants = bytearray(1024)
                for slot, value in enumerate(counts):
                    struct.pack_into('<I', constants, 4 * pipe.uds.index(('rootconst', 0, slot)), value)
                head = bytes(constants) + T.buffer(points, colours, words) + b'\xcd' * 80
                good = image(points, colours, triangles, count, rule)
                runs = [('%s (%s): the guest rule' % (name, label), good, 0)]
                for model, red in wrong.items():
                    other = image(points, colours, triangles, count, red)
                    runs.append(('%s (%s): control "%s"' % (name, label, model), other, sum(a != b for a, b in zip(good, other))))
                case = Path(tmp) / 'case.bin'
                for what, picture, differ in runs:
                    if differ:
                        assert differ > 0, what
                    case.write_bytes(head + b''.join(map(bytes, picture)))
                    run = subprocess.run([renderer, adapter, '--mesh-indexed', base.with_suffix('.ms57.cso'),
                                          base.with_suffix('.ps57.cso'), case], capture_output=True, text=True)
                    if run.returncode == 77:
                        print(run.stdout + run.stderr)
                        return 77
                    want_mismatch = differ
                    ok = False
                    if 'guest rule' in what:
                        ok = run.returncode == 0 and re.search(rf'\bpixels={SIDE * SIDE} mismatches=0\b', run.stdout)
                    else:
                        ok = run.returncode == 1 and re.search(rf'\bpixels={SIDE * SIDE} mismatches={want_mismatch}\b', run.stdout)
                    if not ok:
                        raise AssertionError('%s: unexpected result (exit %d)\n%s%s' % (what, run.returncode, run.stdout, run.stderr))
                    print('%s: %s' % (what, 'all pixels match' if not differ else '%d pixels differ' % differ))
                    done += 1
    print('%d draws as expected' % done)
    return 0


def main(argv):
    status = 0
    if argv[1:2] == ['binary'] and len(argv) == 3:
        binary(argv[2])
    elif argv[1:2] == ['draw'] and len(argv) == 5:
        status = draw(*argv[2:])
    elif len(argv) == 1:
        controls()
    else:
        print(__doc__)
        return 2
    if FAILED:
        return 1
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
