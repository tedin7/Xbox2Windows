"""Slot 57 mesh loop refusal outside the accepted group-memory exchanges (#2413 T6).

usage: startup_mesh57_loop_refusal_check.py
       startup_mesh57_loop_refusal_check.py binary <llvm-mc>

The source-built fixture (tests/startup_mesh57_loop_refusal.s) keeps the skeleton exchanges and inserts one
back-edge after the count exchange. Tip translate_mesh57 must refuse it with the named loop message. The mutant
replaces that back-edge with a second s_nop so the request no longer stops on the loop refusal (it translates
through the source-only path). A stubbed comparator that ignores the refusal string would accept a different
unsupported reason; this check uses an exact match so that stub fails.
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
# Skeleton words through the count-exchange join (s_mov_b64 exec, -1), then s_nop; s_branch -2; then the body.
MESH = (
    'bfa00003 befe0481 be84047e 8706ff03 0f000000 bf850007 2c0a0a87 7e0c0280 d8340008 00000506'
    ' bfa3ffe3 bf8a0000 bf820004 bf8a0000 7e0c0280 d8d80008 05000006 befe04c1 9386ff03 00040018'
    ' 8f078606 d7650000 00000ec1 d7660000 000200c1 bf8cc07f 7e280505 f4080209 fa000000 be950310'
    ' be960311 bf8a0000 bf008006 bf840010 befe0404 bf880007 7e020215 7e040280 7e060216 d8340000'
    ' 00000102 d8340004 00000302 befe04c1 bfa3ffe3 bf8a0000 8f048c15 887c1604 bf900009 bf82000d'
    ' bf8a0000 befe0404 bf880009 7e040280 d8d80000 01000002 d8d80004 03000002 bf8cc17f 7e2a0501'
    ' bf8cc07f 7e2c0503 befe04c1 bf800000 bf82fffe 7d280016 3608008f 34080884 e0381000 80021004'
    ' e03810c0 80021404 bf8c3f70 f80008cf 13121110 f800020f 17161514 befe04c1 7d280015 3608008f'
    ' 34080882 e0301180 80021804 bf8c3f70 f8000941 00000018 bf810000').split()
# Mutant: the same program with the back-edge replaced by a second s_nop.
MUTANT = MESH[: MESH.index('bf82fffe')] + ['bf800000'] + MESH[MESH.index('bf82fffe') + 1:]
PIXEL = 'c8020002 c8060102 c80a0202 c80e0302 f800180f 03020100 bf810000'.split()

FAILED = []


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
    r = subprocess.run([mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32', '-show-encoding'],
                       input=text, capture_output=True, text=True, check=True)
    data = b''.join(bytes(int(b, 16) for b in m.split(',')) for m in re.findall(r'encoding: \[([^\]]*)\]', r.stdout))
    return ['%08x' % word for word in struct.unpack('<%dI' % (len(data) // 4), data)]


def binary(mc):
    text = (HERE / 'startup_mesh57_loop_refusal.s').read_text()
    expect('loop fixture words match llvm-mc', assemble(mc, text), MESH)
    mutant_text = text.replace('s_nop 0\ns_branch -2\n', 's_nop 0\ns_nop 0\n')
    expect('mutant words match llvm-mc', assemble(mc, mutant_text), MUTANT)


def main():
    os.environ['X2W_FXC'] = ''  # sources and manifest only; the refusal is raised before DXC
    want = 'unsupported: ' + LOOP_REASON
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('loop fixture refuses by name', result, want)
        expect('loop fixture publishes only the .fail',
               sorted(p.name for p in Path(tmp).iterdir() if not p.name.startswith('req_')),
               ['%016x.fail' % key])
        if result == want:
            expect('loop .fail text', (Path(tmp) / ('%016x.fail' % key)).read_text(), LOOP_REASON + '\n')

    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MUTANT)
        expect('mutant does not report the loop refusal', result.startswith('unsupported: ' + LOOP_REASON), False)
        expect('mutant translates through the source-only path', result, 'ok')
        if result == 'ok':
            names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
            expect('mutant publishes mesh and pixel sources', names,
                   ['.ms57.hlsl', '.ms57.txt', '.ps57.hlsl'])

    # A stub that ignores the refusal string would accept a different unsupported reason for this oracle.
    other = 'unsupported: slot 57 mesh: a branch enters or crosses a group-memory exchange'
    expect('stubbed comparator would accept a wrong reason', ignores_refusal_string(other, want), True)
    expect('exact oracle rejects that wrong reason', other == want, False)
    expect('exact oracle matches the named loop refusal', want == 'unsupported: ' + LOOP_REASON, True)

    if FAILED:
        print('startup_mesh57_loop_refusal: %d failed' % len(FAILED))
        return 1
    print('startup_mesh57_loop_refusal: ok')
    return 0


if __name__ == '__main__':
    os.environ['X2W_MESH_WAVELOOP'] = '0'  # Exercise the proof-path contract.
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        binary(sys.argv[2])
        raise SystemExit(1 if FAILED else 0)
    if len(sys.argv) != 1:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
