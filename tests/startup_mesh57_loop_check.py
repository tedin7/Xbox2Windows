"""Slot 57 mesh constant-trip loop lowering outside the accepted group-memory exchanges (#2413 T6).

usage: startup_mesh57_loop_check.py
       startup_mesh57_loop_check.py binary <llvm-mc>
       startup_mesh57_loop_check.py draw <dxc> <offscreen_check> <warp|hardware>

The source-built fixture (tests/startup_mesh57_loop.s) keeps the skeleton exchanges and inserts one
constant-trip scalar loop after the count exchange. translate_mesh57 unrolls it; the request translates
and draws with vertex red = 1.0. Unsupported shapes still refuse with the named loop message: an
unconditional back-edge (the #2622 refusal program), a data-dependent trip count, and a loop whose
body writes EXEC. The once-through mutant replaces the back-edge with s_nop so s40 stays 1 and red
stays 0; a draw reference that expects the constant-trip image must fail on that mutant.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

LOOP_REASON = 'slot 57 mesh: a loop outside the group-memory exchanges is not covered'

# Constant-trip fixture: four iterations of s_add_u32 s40, s40, 1; s_cmp_lt_u32 s40, 4; then red from s40 == 4.
MESH = (
    'bfa00003 befe0481 be84047e 8706ff03 0f000000 bf850007 2c0a0a87 7e0c0280 d8340008 00000506'
    ' bfa3ffe3 bf8a0000 bf820004 bf8a0000 7e0c0280 d8d80008 05000006 befe04c1 9386ff03 00040018'
    ' 8f078606 d7650000 00000ec1 d7660000 000200c1 bf8cc07f 7e280505 f4080209 fa000000 be950310'
    ' be960311 bf8a0000 bf008006 bf840010 befe0404 bf880007 7e020215 7e040280 7e060216 d8340000'
    ' 00000102 d8340004 00000302 befe04c1 bfa3ffe3 bf8a0000 8f048c15 887c1604 bf900009 bf82000d'
    ' bf8a0000 befe0404 bf880009 7e040280 d8d80000 01000002 d8d80004 03000002 bf8cc17f 7e2a0501'
    ' bf8cc07f 7e2c0503 befe04c1 7e0e0280 bea80380 80288128 bf0a8428 bf85fffd bf068428 bf840001'
    ' 7e0e02f2 7d280016 3608008f 34080884 e0381000 80021004 e03810c0 80021404 bf8c3f70 7e280307'
    ' f80008cf 13121110 f800020f 17161514 befe04c1 7d280015 3608008f 34080882 e0301180 80021804'
    ' bf8c3f70 f8000941 00000018 bf810000').split()
# Once-through mutant: back-edge replaced by s_nop (body runs once; s40 == 1; red stays 0).
ONCE = MESH[: MESH.index('bf85fffd')] + ['bf800000'] + MESH[MESH.index('bf85fffd') + 1:]
# Unconditional back-edge after the count exchange (#2622 refusal shape).
INFINITE = (
    'bfa00003 befe0481 be84047e 8706ff03 0f000000 bf850007 2c0a0a87 7e0c0280 d8340008 00000506'
    ' bfa3ffe3 bf8a0000 bf820004 bf8a0000 7e0c0280 d8d80008 05000006 befe04c1 9386ff03 00040018'
    ' 8f078606 d7650000 00000ec1 d7660000 000200c1 bf8cc07f 7e280505 f4080209 fa000000 be950310'
    ' be960311 bf8a0000 bf008006 bf840010 befe0404 bf880007 7e020215 7e040280 7e060216 d8340000'
    ' 00000102 d8340004 00000302 befe04c1 bfa3ffe3 bf8a0000 8f048c15 887c1604 bf900009 bf82000d'
    ' bf8a0000 befe0404 bf880009 7e040280 d8d80000 01000002 d8d80004 03000002 bf8cc17f 7e2a0501'
    ' bf8cc07f 7e2c0503 befe04c1 bf800000 bf82fffe 7d280016 3608008f 34080884 e0381000 80021004'
    ' e03810c0 80021404 bf8c3f70 f80008cf 13121110 f800020f 17161514 befe04c1 7d280015 3608008f'
    ' 34080882 e0301180 80021804 bf8c3f70 f8000941 00000018 bf810000').split()
# Data-dependent trip: s_cmp_lt_u32 s40, s16 (root constant) instead of an immediate.
DATA_DEP = MESH[: MESH.index('bf0a8428')] + ['bf0a1028'] + MESH[MESH.index('bf0a8428') + 1:]
# EXEC write in the loop body: v_cmpx_gt_u32_e32 0, v0 between the add and the compare (branch -4).
_EXEC_AT = MESH.index('80288128')
EXEC_BODY = MESH[: _EXEC_AT + 1] + ['7da80080', 'bf0a8428', 'bf85fffc'] + MESH[_EXEC_AT + 3:]

PIXEL = 'c8020002 c8060102 c80a0202 c80e0302 f800180f 03020100 bf810000'.split()

FAILED = []
SIDE = 32
TRIANGLES = (((2, 2), (14, 2), (2, 14)), ((18, 3), (30, 5), (21, 14)),
             ((3, 18), (13, 20), (6, 30)), ((18, 18), (30, 18), (30, 30)))
ORDER = (7, 2, 10, 5, 0, 8, 3, 11, 1, 6, 9, 4)


def expect(what, got, want):
    if got != want:
        FAILED.append(what)
        print('FAIL %s: got %r, want %r' % (what, got, want))


def code(words):
    return b''.join(struct.pack('<I', int(word, 16)) for word in words)


def root():
    rts0 = (struct.pack('<6I', 2, 2, 24, 0, 0, 0) + struct.pack('<3I', 1, 0, 48) + struct.pack('<3I', 0, 0, 60)
            + struct.pack('<3I', 0, 0, 2) + struct.pack('<2I', 0, 0))
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 44 + len(rts0), 1, 36) + b'RTS0' + struct.pack('<I', len(rts0)) + rts0


def request(directory, mesh):
    ms, ps, blob = code(mesh), code(PIXEL), root()
    key = G.mesh57_key(ms, ps, blob)
    for ext, data in (('.root', blob), ('.ps57', ps), ('.m57', ms)):
        (Path(directory) / ('req_%016x%s' % (key, ext))).write_bytes(data)
    return key, G.translate_request(str(directory), key)


def ignores_refusal_string(got, _want):
    """Stubbed comparator: any unsupported result matches, regardless of the named reason."""
    return isinstance(got, str) and got.startswith('unsupported:')


def assemble(mc, text):
    r = subprocess.run([mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32',
                        '-show-encoding'], input=text, capture_output=True, text=True, check=True)
    data = b''.join(bytes(int(b, 16) for b in m.split(',')) for m in re.findall(r'encoding: \[([^\]]*)\]', r.stdout))
    return ['%08x' % word for word in struct.unpack('<%dI' % (len(data) // 4), data)]


def geometry():
    points, colours, indices = [None] * 12, [None] * 12, []
    for t, triangle in enumerate(TRIANGLES):
        for c, point in enumerate(triangle):
            slot = ORDER[3 * t + c]
            points[slot] = point
            colours[slot] = (40 + 17 * slot, 250 - 19 * slot, (slot * 53) % 256, 255)
        indices.append(tuple(ORDER[3 * t:3 * t + 3]))
    return points, colours, indices


def buffer(points, colours, words):
    data = bytearray(512)
    for slot, (x, y) in enumerate(points):
        struct.pack_into('<4f', data, 16 * slot, x / 16 - 1, 1 - y / 16, 0.5, 1)
        struct.pack_into('<4f', data, 192 + 16 * slot, *(channel / 255 for channel in colours[slot]))
    for t, word in enumerate(words):
        struct.pack_into('<I', data, 384 + 4 * t, word)
    return bytes(data)


def rasterise(points, colours, indices, verts, prims):
    verts, prims = min(verts, 128), min(prims, 128)
    image = [(0, 0, 0, 255)] * (SIDE * SIDE)
    if not verts:
        return image
    for a, b, c in indices[:prims]:
        a, b, c = (min(i, verts - 1) for i in (a, b, c))
        colour = colours[a % 16]
        tri = [(2 * points[i % 16][0], 2 * points[i % 16][1]) for i in (a, b, c)]
        area = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0])
        if not area:
            continue
        if area < 0:
            tri[1], tri[2] = tri[2], tri[1]
        for y in range(SIDE):
            for x in range(SIDE):
                inside = True
                for k in range(3):
                    (ax, ay), (bx, by) = tri[k], tri[(k + 1) % 3]
                    dx, dy = bx - ax, by - ay
                    edge = dx * (2 * y + 1 - ay) - dy * (2 * x + 1 - ax)
                    inside &= edge > 0 or (edge == 0 and (dy < 0 or (dy == 0 and dx > 0)))
                if inside:
                    image[y * SIDE + x] = colour
    return image


def binary(mc):
    text = (HERE / 'startup_mesh57_loop.s').read_text()
    expect('constant-trip fixture words match llvm-mc', assemble(mc, text), MESH)
    mutant_text = text.replace('s_cbranch_scc1 -3\n', 's_nop 0\n')
    expect('once-through mutant words match llvm-mc', assemble(mc, mutant_text), ONCE)


def host():
    os.environ['X2W_FXC'] = ''  # sources and manifest only; draw mode compiles
    want = 'unsupported: ' + LOOP_REASON

    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('constant-trip loop translates', result, 'ok')
        if result == 'ok':
            names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
            expect('constant-trip publishes mesh and pixel sources', names,
                   ['.ms57.hlsl', '.ms57.txt', '.ps57.hlsl'])
            hlsl = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('constant-trip unrolls four counter updates',
                   hlsl.count('0x00000001u') >= 4 and 's40' in hlsl, True)

    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, ONCE)
        expect('once-through mutant translates', result, 'ok')

    for what, mesh in (
            ('unconditional back-edge', INFINITE),
            ('data-dependent trip count', DATA_DEP),
            ('EXEC write in the loop body', EXEC_BODY),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, mesh)
            expect(what + ' refuses by name', result, want)
            if result == want:
                expect(what + ' .fail text', (Path(tmp) / ('%016x.fail' % key)).read_text(), LOOP_REASON + '\n')

    other = 'unsupported: slot 57 mesh: a branch enters or crosses a group-memory exchange'
    expect('stubbed comparator would accept a wrong reason', ignores_refusal_string(other, want), True)
    expect('exact oracle rejects that wrong reason', other == want, False)
    expect('exact oracle matches the named loop refusal', want == 'unsupported: ' + LOOP_REASON, True)

    # Signedness comes from the compare alone: signed update + unsigned compare across bit 31 is not
    # a finite unroll within MESH_TRIP_MAX (review on #2630).
    expect('signed update / unsigned compare across bit 31 is not a finite trip count',
           G.mesh_trip_count(0x7fffffff, 's_add_i32', 1, 's_cmp_gt_u32', 4, 's_cbranch_scc1', True),
           None)
    expect('unsigned compare after signed update from -3 is one trip',
           G.mesh_trip_count((-3) & 0xffffffff, 's_add_i32', 1, 's_cmp_lt_u32', 0, 's_cbranch_scc1', True),
           1)

    if FAILED:
        print('startup_mesh57_loop: %d failed' % len(FAILED))
        return 1
    print('startup_mesh57_loop: ok')
    return 0


def draw(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    # Constant-trip: every vertex red channel is 255. Once-through mutant keeps the buffer reds.
    trip_colours = [(255, *colour[1:]) for colour in colours]
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('constant-trip translates and compiles', result, 'ok')
        if result != 'ok':
            return 1 if FAILED else 0
        name = str(Path(tmp) / ('%016x' % key))
        uds = G.MeshPipeline(code(MESH), code(PIXEL), root()).uds
        constants = bytearray(1024)
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 0)), 4)
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 1)), 12)
        image = rasterise(points, trip_colours, indices, 12, 4)
        path = Path(tmp) / 'draw.bin'
        path.write_bytes(bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
                         + b''.join(bytes(pixel) for pixel in image))
        r = subprocess.run([renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)],
                           capture_output=True, text=True)
        if r.returncode == 77:
            print(r.stdout + r.stderr)
            return 77
        print('constant-trip draw: %s' % (r.stdout.strip().splitlines()[-1:] or [r.stderr.strip()]))
        expect('constant-trip draw matches CPU oracle',
               (r.returncode, 'pixels=1024 mismatches=0' in r.stdout), (0, True))

        # Negative control: same reference against the once-through mutant (red stays buffer red ≠ 255).
        key2, result2 = request(tmp, ONCE)
        expect('once-through mutant translates and compiles', result2, 'ok')
        if result2 == 'ok':
            name2 = str(Path(tmp) / ('%016x' % key2))
            r = subprocess.run(
                [renderer, adapter, '--mesh-indexed', name2 + '.ms57.cso', name2 + '.ps57.cso', str(path)],
                capture_output=True, text=True)
            if r.returncode == 77:
                print(r.stdout + r.stderr)
                return 77
            # Require a real draw-reference mismatch (not an unrelated PSO/command failure).
            expect('once-through mutant fails the constant-trip reference',
                   (r.returncode != 0
                    and 'Draw reference mismatch' in r.stderr
                    and 'mismatches=' in r.stdout
                    and 'mismatches=0' not in r.stdout), True)

        wrong = list(image)
        at = next(i for i, pixel in enumerate(image) if pixel != (0, 0, 0, 255))
        wrong[at] = (0, 0, 0, 255)
        path.write_bytes(bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
                         + b''.join(bytes(pixel) for pixel in wrong))
        r = subprocess.run([renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)],
                           capture_output=True, text=True)
        expect('a reference with one wrong pixel fails',
               (r.returncode, 'pixels=1024 mismatches=1' in r.stdout, 'Draw reference mismatch' in r.stderr),
               (1, True, True))

    if FAILED:
        print('startup_mesh57_loop draw: %d failed' % len(FAILED))
        return 1
    print('startup_mesh57_loop draw: ok')
    return 0


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 3 and argv[1] == 'binary':
        binary(argv[2])
        raise SystemExit(1 if FAILED else 0)
    if argv[1:2] == ['draw'] and len(argv) == 5:
        raise SystemExit(draw(*argv[2:]))
    if len(argv) != 1:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(host())
