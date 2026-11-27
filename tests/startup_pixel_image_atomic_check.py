"""Synthetic gfx1013 pixel atomics. No arguments checks translation. 'draw <dxc> <offscreen_check>' uses WARP."""
import operator
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_pixel_image_store_check import FULLSCREEN_VS, G, compile_hlsl, refused, translate

OPS = {
    'add': (17, 'Add', operator.add),
    'sub': (18, 'Add', operator.sub),
    'umin': (21, 'Min', min),
    'umax': (23, 'Max', max),
    'and': (24, 'And', operator.and_),
    'or': (25, 'Or', operator.or_),
    'xor': (26, 'Xor', operator.xor),
}
SIDE = 4
U32 = 0xffffffff


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def vop2(op, dst, src0, src1):
    return op << 25 | dst << 17 | src1 << 9 | src0


def mov(dst, imm):
    return 0x7e000200 | dst << 17 | (128 + imm)


def descriptor(first, offset):
    return words(0xf40c0004 | (first // 4) << 8, 0xfa000000 | offset)


def atomic(op, data=5, addr=6, desc=24, *, glc=False, dmask=1, dim=1):
    return words(0xf0001000 | OPS[op][0] << 18 | glc << 13 | dmask << 8 | dim << 3,
                 (desc // 4) << 16 | data << 8 | addr)


WAIT = words(0xbf8cc07f)  # s_waitcnt lgkmcnt(0)
EXPORT = words(0xf800180f, 0x0f0f0f0f, 0xbf810000)  # exp mrt0 v15, v15, v15, v15 done vm
WHITE = words(0x7e1e02f2)  # v_mov_b32 v15, 1.0
SAVE_EXEC = words(0xbe8c047e)  # s_mov_b64 s[12:13], exec
WQM = words(0xbefe0a7e)  # s_wqm_b64 exec, exec
RESTORE_EXEC = words(0xbefe040c)  # s_mov_b64 exec, s[12:13]
NARROW_EXEC = words(0x87fe7e0a)  # s_and_b64 exec, s[10:11], exec
SKIP_ATOMIC = words(0xbf880002)  # s_cbranch_execz +2 dwords
WAIT_VMEM = words(0xbf8c3f70)


def requested_shape():
    # Packed SV_Position in v2. Compare x < 2 into s[10:11] and keep v3 = y for both halves.
    setup = words(vop2(0x1b, 4, 255, 2), 0xffff, vop2(0x16, 3, 144, 2),
                  0xd4c1000a, (256 + 4) | (130 << 9))
    coords = words(vop2(0x16, 6, 249, 2), 0x04860681, vop2(0x16, 7, 129, 3))
    # v_add_f32_e64 v5, 1.0, 0.5 clamp. VDATA carries the resulting 1.0 as uint bits.
    clamp = words(0xd5038005, 242 | (240 << 9))
    minimum = atomic('umin')
    # The second texel address starts at v2, now holding unpacked x followed by y.
    return (descriptor(16, 32) + descriptor(24, 0) + WAIT + setup + SAVE_EXEC + WQM + coords + clamp
            + RESTORE_EXEC + NARROW_EXEC + SKIP_ATOMIC + minimum
            + RESTORE_EXEC + words(vop2(0x1b, 2, 255, 2), 0xffff)
            + NARROW_EXEC + SKIP_ATOMIC + atomic('or', addr=2, desc=16)
            + RESTORE_EXEC + WHITE + EXPORT)


def draw_words(op, *, narrow=False):
    values = words(
        vop2(0x1b, 8, 255, 2), 0xffff,           # x = packed & 0xffff
        vop2(0x16, 9, 144, 2),                   # y = packed >> 16
        vop2(0x1b, 6, 129, 8), vop2(0x1b, 7, 129, 9),  # shared address (x&1, y&1)
        vop2(0x1a, 12, 130, 9),                  # 4*y
        vop2(0x25, 12, 256 + 8, 12),             # linear = x + 4*y
        vop2(0x25, 5, 160, 12),                  # base data = linear + 32
        vop2(0x1a, 13, 158, 8),                  # x bit 1 becomes sign bit
        vop2(0x1b, 13, 255, 13), 0x80000000,
        vop2(0x1c, 5, 256 + 13, 5),
        mov(10, 0), mov(11, 0), mov(14, 1),
        vop2(0x1a, 14, 256 + 12, 14),            # one bit per pixel
    )
    compare = words(0x7c000000 | 0xd3 << 17 | 8 << 9 | 130)  # 2 <= x
    return (descriptor(24, 0) + descriptor(16, 32) + WAIT + values + (SAVE_EXEC + compare if narrow else b'')
            + atomic(op) + atomic('or', data=14, addr=10, desc=16) + WAIT_VMEM
            + (RESTORE_EXEC if narrow else b'') + WHITE + EXPORT)


def runtime_words():
    # One unsigned min at (1, 2) through table 0. White export exercises the same draw as the float fixture.
    return (descriptor(24, 0) + WAIT + words(mov(6, 1), mov(7, 2), 0x7e0a02ff, 0x80000010)
            + atomic('umin') + WAIT_VMEM + WHITE + EXPORT)


def controls():
    shape = requested_shape()
    decoded = [asm for _, _, _, asm in G.decode(shape)]
    for instruction in (
        's_mov_b64 s[12:13], exec', 's_wqm_b64 exec, exec', 's_and_b64 exec, s[10:11], exec',
        'image_atomic_umin v5, v[6:7], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm',
        'image_atomic_or v5, v[2:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm',
    ):
        assert instruction in decoded, decoded
    assert any(text.startswith('v_lshrrev_b32_sdwa v6,') and text.endswith('src1_sel:WORD_0') for text in decoded)
    assert any(text.startswith('v_add_f32_e64 v5,') and text.endswith('clamp') for text in decoded)
    ps = translate(shape).hlsl_ps()
    assert 'if (exec & 1) InterlockedMin(w0[uint2(v6, v7)], v5);' in ps, ps
    assert 'if (exec & 1) InterlockedOr(w1[uint2(v2, v3)], v5);' in ps, ps

    for name, (_, intrinsic, _) in OPS.items():
        code = draw_words(name)
        pipe = translate(code)
        ps, manifest = pipe.hlsl_ps(), pipe.manifest()
        value = '0u - v5' if name == 'sub' else 'v5'
        assert 'Interlocked%s(w0[uint2(v6, v7)], %s);' % (intrinsic, value) in ps, ps
        assert 'InterlockedOr(w1[uint2(v10, v11)], v14);' in ps, ps
        for slot in range(2):
            assert 'RWTexture2D<uint> w%d : register(u%d);' % (slot, slot) in ps, ps
        assert [line for line in manifest.splitlines() if line.startswith(('uav', 'srv '))] == [
            'uav 0 tex2du table 0 0', 'uav 1 tex2du table 0 32'], manifest

    base = runtime_words()
    for encoded, reason in (
        (atomic('umin', glc=True), 'with glc: returning the previous texel is unsupported'),
        (atomic('umin', dmask=3), 'requires dmask:0x1 for one 32-bit data word'),
        (atomic('umin', dim=0), 'requires dim:SQ_RSRC_IMG_2D'),
        (atomic('umin', dim=2), 'requires dim:SQ_RSRC_IMG_2D'),
    ):
        refused(base.replace(atomic('umin'), encoded), reason)
    helper = 'image_atomic_umin: EXEC may include helper lanes'
    refused(base.replace(atomic('umin'), SAVE_EXEC + WQM + atomic('umin')), helper)
    # A taken branch skips the only restore. Program order alone would mark the join as safe.
    skipped_restore = words(0xbf840001) + RESTORE_EXEC  # s_cbranch_scc0 +1
    join = SAVE_EXEC + WQM + skipped_restore
    refused(base.replace(atomic('umin'), join + atomic('umin')), helper)
    load = words(0xf0000108, 0x00061006)  # image_load v16, v[6:7], s[24:31]
    store = words(0xf0200108, 0x00060506)  # image_store v5, v[6:7], s[24:31]
    for other, reason in ((load, 'both read and stored to'), (store, 'used as tex2d and tex2du')):
        refused(base.replace(atomic('umin'), other + WAIT_VMEM + atomic('umin')), reason)
    refused(base.replace(atomic('umin'), atomic('umin') + load), 'both read and stored to')
    refused(base.replace(atomic('umin'), atomic('umin') + store), 'used as tex2du and tex2d')

    # The scan and liveness must read VDATA, with no kill of a value still live after the atomic.
    _, _, name, asm = next(ins for ins in G.decode(base) if ins[2] == 'image_atomic_umin')
    _, ops, mods = G.parse(asm)
    assert G.effect_destinations(name, ops, mods) == ([], ops)
    assert G.destinations(name, ops, mods) == [] and G.lds_vgpr_writes(name, ops, mods) == set()
    assert G.effect_destinations(name, ops, [*mods, 'glc']) == (ops[:1], ops[1:])
    # An atomic can consume an entry barycentric without clobbering it. The omitted partial
    # interpolant in v12 is different: the scan must reject an atomic that consumes it.
    interp = words(0xd742000c, 256 << 9, 0xd75a0014, (257 << 9) | (268 << 18))
    stage = G.Stage.__new__(G.Stage)
    stage.ins = G.decode(atomic('umin', data=0) + interp + EXPORT)
    stage.interp_f16_scan()
    stage = G.Stage.__new__(G.Stage)
    stage.ins = G.decode(interp + atomic('umin', data=12) + EXPORT)
    try:
        stage.interp_f16_scan()
    except G.Unsupported as exc:
        assert 'image_atomic_umin reads it after' in str(exc), exc
    else:
        raise AssertionError('atomic VDATA must be read by the interpolation scan')

    assert reference('umin')[1][0] == 0x4000ffff
    assert reference('umin', narrow=True)[1][0] == 0x4000cccc
    for op, mutant in WRONG_OP.items():
        assert reference(op)[0] != reference(mutant)[0], (op, mutant)
    print('7 pixel image atomics, no VDATA write, 2 unsigned views, WQM restore and refusal controls')


def initial_texels():
    primary = [0xa0000000 | i for i in range(SIDE * SIDE)]
    for index in (0, 1, SIDE):
        primary[index] = U32
    primary[SIDE + 1] = 0x80000010
    bits = [0x40000000 | (i << 16) for i in range(SIDE * SIDE)]
    return primary, bits


def reference(op, *, narrow=False, shape=False):
    primary, bits = initial_texels()
    for y in range(SIDE):
        for x in range(SIDE):
            if shape:
                if x < 2:
                    data = 0x3f800000  # binary32 bits of saturate(1.0 + 0.5)
                    address = (x // 2) + SIDE * (y // 2)
                    primary[address] = min(primary[address], data)
                    bits[x + SIDE * y] |= data
            elif not narrow or x >= 2:
                lane = x + SIDE * y
                data = ((x & 2) << 30) | (32 + lane)
                address = (x & 1) + SIDE * (y & 1)
                primary[address] = OPS[op][2](primary[address], data) & U32
                bits[0] |= 1 << lane
    return primary, bits


WRONG_OP = {'add': 'sub', 'sub': 'add', 'umin': 'umax', 'umax': 'umin',
            'and': 'or', 'or': 'and', 'xor': 'or'}


def check_draw(dxc, offscreen):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs', 'vs_6_0', directory)

        def draw(label, source, expected, *, signed=False, mismatch=None):
            ps = compile_hlsl(dxc, 'dxc', source, label, 'ps_6_0', directory)
            fixture = directory / (label + '.bin')
            fixture.write_bytes(words(*(word for image in (*initial_texels(), *expected) for word in image)))
            run = subprocess.run([offscreen, 'warp', '--startup-image-atomic', vs, ps, fixture,
                                  'sint' if signed else 'uint'], capture_output=True, text=True)
            if run.returncode == 77 and mismatch is None:
                print(run.stdout, run.stderr)
                raise SystemExit(77)
            if mismatch is None:
                assert run.returncode == 0, (label, run.stdout, run.stderr)
            else:
                assert run.returncode == 1 and ('texel_mismatch uav=%d ' % mismatch) in run.stderr, (
                    label, run.stdout, run.stderr)
                print(label, next(line for line in run.stderr.splitlines() if 'texel_mismatch' in line))

        def mutate(label, shader, old, new, expected, *, signed=False, mismatch=0):
            assert shader.count(old) == 1, (label, old)
            draw(label, shader.replace(old, new), expected, signed=signed, mismatch=mismatch)

        for op, (_, intrinsic, _) in OPS.items():
            op_line = 'Interlocked%s(w0[uint2(v6, v7)], %sv5);' % (intrinsic, '0u - ' if op == 'sub' else '')
            mutant = WRONG_OP[op]
            mutant_line = 'Interlocked%s(w0[uint2(v6, v7)], %sv5);' % (
                OPS[mutant][1], '0u - ' if mutant == 'sub' else '')
            expected = reference(op)
            shader = translate(draw_words(op)).hlsl_ps()
            draw(op, shader, expected)
            mutate(op + '_wrong_op', shader, op_line, mutant_line, expected)
            if op == 'umin':
                sint = shader.replace('RWTexture2D<uint> w0', 'RWTexture2D<int> w0')
                mutate('umin_signed', sint, op_line, 'InterlockedMin(w0[uint2(v6, v7)], asint(v5));',
                       expected, signed=True)
                or_line = 'InterlockedOr(w1[uint2(v10, v11)], v14);'
                # Rendezvous after the plain load so the control loses bits even if separate waves serialize.
                # The predicate is always true because OR preserves the sentinel bit.
                mutate('or_non_atomic', shader, or_line,
                       '{ uint before_or = w1[uint2(v10, v11)]; '
                       'if (WaveActiveAllTrue((before_or & 0x40000000u) != 0)) '
                       'w1[uint2(v10, v11)] = before_or | v14; }', expected, mismatch=1)

        narrow = translate(draw_words('umin', narrow=True)).hlsl_ps()
        expected = reference('umin', narrow=True)
        draw('narrow_exec', narrow, expected)
        # Restore EXEC after the atomics so a late discard cannot mask this control.
        guard = 'if (exec & 1) Interlocked'
        assert narrow.count(guard) == 2 and expected != reference('umin')
        draw('narrow_exec_unguarded', narrow.replace(guard, 'Interlocked'), expected, mismatch=0)

        shape = translate(requested_shape()).hlsl_ps()
        draw('wqm_restore', shape, reference('umin', shape=True))
        print('WARP readback: nine fixtures and ten controls')


if __name__ == '__main__':
    controls()
    if len(sys.argv) == 4 and sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    elif len(sys.argv) != 1:
        raise SystemExit(__doc__)
