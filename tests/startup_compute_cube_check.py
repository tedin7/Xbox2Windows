"""Compute image_sample_lz / image_sample_l on a cube map: translation controls, then a D3D12 readback of the six faces.

usage: startup_compute_cube_check.py [--llvm-mc LLVM_MC | --dispatch DXC EXE ADAPTER]

The ISA address of a cube sample is S, T and the face index (v_cubesc / v_cubetc divided by |v_cubema| plus 1.5, and
v_cubeid). The translation samples a TextureCube at the direction those three values stand for. The program reads 24
(S, T, face) records, samples the cube bound at t1 and stores the float4 of each. The cube has six 4x4 faces whose
every texel is distinct (face, x, y and a running tag), so a wrong face, a wrong orientation or a swapped axis changes
the readback. The CPU reference is the face-local texel at (S - 1, T - 1), which does not use the direction table.
Off-centre samples (the texel's 0.3 and 0.7 fractions, not its centre) are part of the set. Controls: the face order
of the direction table swapped for two faces, and S and T swapped.
"""
import argparse
import math
import struct
import subprocess
import tempfile
from pathlib import Path

from startup_gfx2hlsl_compute_check import G, TGID_X, USER7, refused, root_blob, tidig
from startup_flat_param_check import assemble

SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_load_dwordx8 s[24:31], s[10:11], 0x40
s_mov_b64 s[20:21], 0
s_mov_b64 s[22:23], 0
v_lshlrev_b32 v4, 4, v0
s_waitcnt lgkmcnt(0)
buffer_load_dwordx3 v[1:3], v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
image_sample_lz v[8:11], v[1:3], s[24:31], s[20:23] dmask:0xf dim:SQ_RSRC_IMG_CUBE
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[8:11], v4, s[16:19], 0 offen
s_endpgm
"""
WORDS = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '05060cf4400000fa' '800494be' '800496be' '84000834' '7fc08cbf'
    '00103ce004010380' '703f8cbf' '180f9cf00108a600' '703f8cbf' '001078e004080480' '000081bf')
LZ, LOD = bytes.fromhex('180f9cf00108a600'), bytes.fromhex('180f90f00108a600')  # image_sample_lz, image_sample_l
IMPLICIT = bytes.fromhex('180f80f00108a600')  # image_sample: implicit LOD
CUBE_LINE = 'image_sample_lz v[8:11], v[1:3], s[24:31], s[20:23] dmask:0xf dim:SQ_RSRC_IMG_CUBE'
RSRC2 = USER7 | TGID_X | tidig(1)
THREADS = 24
SIZE = 4  # texels per face edge
CALL = 'x2w_cube_dir(F(v1), F(v2), F(v3))'
# Face texels on each face as (x, y) and the fractions inside the texel: not the centre, two different corners.
TEXELS = ((0, 0), (3, 1), (1, 3), (2, 2))
FRACTIONS = ((0.3, 0.7), (0.7, 0.3), (0.3, 0.3), (0.7, 0.7))


def pipeline(words=WORDS):
    return G.ComputePipeline(words, root_blob(2, table_range=True), RSRC2, (THREADS, 1, 1))


def texel(face, x, y):
    """Face ID and two gradients with different slopes on each face."""
    return (float(face + 1), float(7 * face + (face + 2) * x),
            float(11 * face + (face + 3) * y), 1.0)


def samples():
    """24 records (S, T, face). The first four are on face 0, the next four on face 1, and so on."""
    out = []
    for face in range(6):
        for (x, y), (fx, fy) in zip(TEXELS, FRACTIONS):
            out.append((1.0 + (x + fx) / SIZE, 1.0 + (y + fy) / SIZE, float(face)))
    return out


def reference(records):
    """The face-local texel for (S - 1, T - 1) of the face index: the cube address without any direction."""
    out = []
    for s, t, face in records:
        x = min(max(int(math.floor((s - 1.0) * SIZE)), 0), SIZE - 1)
        y = min(max(int(math.floor((t - 1.0) * SIZE)), 0), SIZE - 1)
        out.append(texel(int(face), x, y))
    return out


def faces_blob():
    return b''.join(struct.pack('<4f', *texel(f, x, y)) for f in range(6) for y in range(SIZE) for x in range(SIZE))


def samples_blob(records):
    return b''.join(struct.pack('<4f', s, t, face, 0.0) for s, t, face in records)


def controls(llvm_mc):
    if llvm_mc:
        assert assemble(llvm_mc, SOURCE) == WORDS, 'the fixture words no longer match llvm-mc'
        assert assemble(llvm_mc, CUBE_LINE.replace('image_sample_lz', 'image_sample_l').replace('v[1:3]', 'v[1:4]') + '\n') == LOD
        assert assemble(llvm_mc, CUBE_LINE.replace('image_sample_lz', 'image_sample') + '\n') == IMPLICIT
        print('fixture words match llvm-mc')
    assert WORDS.count(LZ) == 1
    decoded = [text for _o, _s, _n, text in G.decode(WORDS)]
    assert [t for t in decoded if t.startswith('image_sample')] == [CUBE_LINE], decoded
    pipe = pipeline()
    source = pipe.hlsl_cs()
    assert pipe.manifest().endswith('srv 0 buf table 1 0\nsrv 1 texcube table 1 64\nuav 0 table 1 16\nsmp 0 0x0 0x0 0x0 0x0\n'), \
        pipe.manifest()
    assert 'TextureCube<float4> r1 : register(t1);' in source
    assert source.count('float3 x2w_cube_dir(float s, float t, float face) {') == 1
    assert source.count('t = r1.SampleLevel(q0, %s, 0);' % CALL) == 1, source[source.index('t = r1'):][:200]
    # image_sample_l takes the same three address registers and one level after them.
    lod = pipeline(WORDS.replace(LZ, LOD)).hlsl_cs()
    assert lod.count('r1.SampleLevel(q0, %s, F(v4))' % CALL) == 1
    # A program with no cube sample carries no direction helper and no TextureCube.
    plain = G.ComputePipeline(bytes.fromhex(
        '06038abe80038bbe05030cf4000000fa7fc08cbf00103ce004010380703f8cbf000081bf'),
        root_blob(2, table_range=True), RSRC2, (THREADS, 1, 1)).hlsl_cs()
    assert 'x2w_cube_dir' not in plain and 'TextureCube' not in plain
    # The implicit-LOD form (a compute lane has no quad) stays refused.
    refused('image dimension SQ_RSRC_IMG_CUBE', lambda: pipeline(WORDS.replace(LZ, IMPLICIT)))
    records = samples()
    assert len(records) == THREADS == 6 * len(TEXELS)
    assert len({tuple(texel_) for texel_ in reference(records)}) == THREADS, 'two samples read the same texel'
    print('cube translation controls ok')


def compile_and_run(directory, compiler, exe, adapter, source, name):
    hlsl, cso, out = (directory / (name + ext) for ext in ('.hlsl', '.cso', '.out'))
    hlsl.write_text(source, encoding='ascii')
    subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, hlsl], check=True)
    run = subprocess.run([exe, adapter, cso, directory / 'faces.bin', directory / 'samples.bin', out],
                         capture_output=True, text=True, timeout=60)
    print('%s:\n%s%s' % (name, run.stdout, run.stderr), end='', flush=True)
    if run.returncode == 77:
        return None
    assert run.returncode == 0, (name, run.returncode)
    raw = out.read_bytes()
    return [struct.unpack_from('<4f', raw, 16 * i) for i in range(THREADS)]


def dispatch(directory, compiler, exe, adapter):
    records = samples()
    (directory / 'faces.bin').write_bytes(faces_blob())
    (directory / 'samples.bin').write_bytes(samples_blob(records))
    expected = reference(records)
    source = pipeline().hlsl_cs()
    assert source.count(CALL) == 1

    def mismatches(got):
        return [i for i in range(THREADS) if tuple(got[i]) != expected[i]]

    got = compile_and_run(directory, compiler, exe, adapter, source, 'cube')
    if got is None:
        return 77
    wrong = mismatches(got)
    assert not wrong, 'sample %d: got %s expected %s (%d of %d samples differ)' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), THREADS)
    print('all %d samples match (6 faces, 4 off-centre samples each)' % THREADS)
    # Each U or V reversal must change every sample, including all four reads on each face.
    swapped = source.replace('f == 0u', 'f == @u').replace('f == 1u', 'f == 0u').replace('f == @u', 'f == 1u')
    transposed = source.replace(CALL, 'x2w_cube_dir(F(v2), F(v1), F(v3))')
    reverse_u = source.replace(CALL, 'x2w_cube_dir(3.0 - F(v1), F(v2), F(v3))')
    reverse_v = source.replace(CALL, 'x2w_cube_dir(F(v1), 3.0 - F(v2), F(v3))')
    for name, text, count in (('wrong face order', swapped, 8), ('S and T swapped', transposed, 12),
                              ('U reversed', reverse_u, THREADS), ('V reversed', reverse_v, THREADS)):
        assert text != source, name
        got = compile_and_run(directory, compiler, exe, adapter, text, name.replace(' ', '_'))
        wrong = mismatches(got)
        assert len(wrong) == count, '%s: %d samples differ, expected %d' % (name, len(wrong), count)
        print('mutant %s: %d samples differ, as expected' % (name, len(wrong)))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--dispatch', nargs=3, metavar=('DXC', 'EXE', 'ADAPTER'))
    args = parser.parse_args()
    if args.dispatch:
        with tempfile.TemporaryDirectory() as tmp:
            raise SystemExit(dispatch(Path(tmp), *args.dispatch))
    controls(args.llvm_mc)
