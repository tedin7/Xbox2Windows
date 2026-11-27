"""Synthetic pixel texture-to-formatted-buffer path, with CPU and WARP readback controls."""
from pathlib import Path
import struct
import hashlib
import subprocess
import sys
import tempfile

from startup_image_3d_check import VS_WORDS, FULLSCREEN_VS, compile_hlsl
from startup_compute_store_format_xyzw_check import encode_mubuf, desc_word3, G
from startup_compute_readfirstlane_check import sop1, sopc, sopp, vop1, vop2
from startup_image_load_d16_check import encode_mimg

CONSTANTS = (2.0, -0.5, 0.75, 4.0)
IMAGE = (0.3125, 0.125, 0.75, 0.5)
RAMP = (0.125, 0.25, 0.5, 0.875)
RECORDS = 4
STORE = 'buffer_store_format_xyzw'


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def root():
    # Descriptor table followed by four root constant words.
    rts = words(2, 2, 24, 0, 0, 0, 0, 0, 48, 1, 0, 80,
                1, 56, 0, 3, 0, 0, 0, 0, 0, 0, 4)
    return b'DXBC' + bytes(16) + words(1, 44 + len(rts), 1, 36) + b'RTS0' + words(len(rts)) + rts


def load_desc(dest, offset, opcode):
    return words(0xf4000000 | opcode << 18 | dest << 6 | 4, 0xfa000000 | offset)


def program(indices=(0, 1), *, sample=0x27, varying=False, texel_x=0, offen=False, fixed=True, firstlane=True, real_shape=False, nonuniform=False, nonconstant=False, stores=True):
    def mov(dest, src):
        return words(*vop1('v_mov_b32_e32', f'v{dest}', src))

    def arithmetic(op, dst, src, vector):
        return words(*vop2(op + '_e32', f'v{dst}', src, f'v{vector}'))

    code = load_desc(40, 0, 3) + load_desc(48, 32, 3)
    if fixed and not real_shape:
        code += load_desc(56, 64, 2)
    code += words(*sopp('s_waitcnt', 0xc07f))
    for reg in range(60, 64):
        code += words(*sop1('s_mov_b32', f's{reg}', 0))
    code += mov(8, texel_x) + mov(9, 0)
    if real_shape:
        code += encode_mimg(0, dmask=0xa, dim=1, vaddr=8, vdata=2, srsrc=40, unorm=1, a16=1)
        code += encode_mimg(0, dmask=2, dim=1, vaddr=8, vdata=4, srsrc=40, unorm=1, a16=1)
    code += encode_mimg(0, dmask=1, dim=1, vaddr=8, vdata=10, srsrc=40)
    code += words(*sopp('s_waitcnt', 0x3f70))
    code += arithmetic('v_mul_f32', 11, 's9', 10)
    sample_code = encode_mimg(sample, dmask=1, dim=0, vaddr=0 if varying else 11,
                              vdata=12, srsrc=48, ssamp=60) + words(*sopp('s_waitcnt', 0x3f70))
    arm = words(*vop2('v_min_f32_e32', 'v13', '2.0', 'v12'))
    arm += (words(*vop1('v_readfirstlane_b32', 's64', 'v13')) if firstlane else
            words(*sop1('s_mov_b32', 's64', 1)))
    if real_shape:
        arm = sample_code + arm
    else:
        code += sample_code
    code += words(*sop1('s_mov_b32', 's64', 0), *sopc('s_cmp_eq_u32', 's12', 0),
                  *sopp('s_cbranch_scc1', len(arm) // 4)) + arm
    if fixed and real_shape:
        code += load_desc(56, 64, 2) + words(*sopp('s_waitcnt', 0xc07f))
    code += mov(20, 's64')
    code += arithmetic('v_mul_f32', 21, '2.0', 20)
    code += arithmetic('v_add_f32', 22, 'v20', 21)
    code += words(0x7e000000 | (23 << 17) | (1 << 9) | 242)  # v23 = 1.0
    for reg in range(4):
        code += mov(24 + reg, f's{9 + reg}')
    if nonuniform:
        code += mov(20, 'v0')
    for index, data in zip(indices, (20, 24) if stores else ()):
        code += mov(28, 'v0' if nonconstant else index)
        code += encode_mubuf(7, data, 28, 56, 128, idxen=True, offen=offen)
    v20_all = int.from_bytes(bytes([20] * 4), 'little')
    return code + words(0xf800180f, v20_all, *sopp('s_endpgm'))


def translate(**options):
    # Pixel input v0 is POSITION.x, used only by the varying negative control.
    return G.Pipeline(VS_WORDS, program(**options), root(), ps_hw_inputs=(1 << 8, 1 << 8))


def controls():
    pipeline = translate()
    assert translate(real_shape=True).hlsl_ps().count('store_format_xyzw(u0,') == 2
    unrelated = translate(stores=False, firstlane=False)
    assert hashlib.sha256((unrelated.hlsl_ps() + unrelated.manifest()).encode()).hexdigest() == 'ae24909241d38053647b5733d0a6b233779cf6c09da9f4a0cb57aa2c675738e1'
    # With indices 4 and 5, dropping NUM_RECORDS writes the otherwise untouched backing records.
    right = bytearray(fixture(pipeline, untouched=True)[164:212])
    wrong = bytearray(right)
    wrong[32:48] = struct.pack('<8e', 0.25, 0.5, 0.75, 1.0, *CONSTANTS)
    assert right != wrong
    for bad in (desc_word3(1), desc_word3(71) ^ 1, desc_word3(71) | (1 << 21)):
        try:
            G.validate_store_format_xyzw_word3(bad)
        except G.Unsupported:
            pass
        else:
            raise AssertionError('unsupported V# word 3 accepted')
    hlsl = pipeline.hlsl_ps()
    manifest = pipeline.manifest().splitlines()
    assert 'if (exec & 1) store_format_xyzw(u0,' in translate(firstlane=False).hlsl_ps()
    assert 'r0.Load(int3(v8, v9, 0))' in hlsl and '.SampleLevel(' in hlsl
    assert 's64 = v13;' in hlsl
    assert hlsl.count('if (exec & 1) store_format_xyzw(u0,') == 2
    assert 'uav 0 raw table 0 64' in manifest
    for name in ('uav_descword3', 'uav_stride', 'uav_numrecords'):
        assert f'udsrc {pipeline.uds.index((name, 0))} {name} 0' in manifest
    for options, fragment in (
        ({'indices': (0, 0)}, STORE + ' pixel stores have overlapping indices'),
        ({'nonuniform': True}, STORE + ' data not proven wave-uniform'),
        ({'nonconstant': True}, STORE + ' index is not an immediately preceding constant move'),
        ({'offen': True}, STORE + ' offen address form'),
        ({'varying': True}, G.READFIRSTLANE_UNIFORM),
        ({'sample': 0x20}, G.READFIRSTLANE_UNIFORM),
        ({'fixed': False}, STORE + ' through'),
    ):
        try:
            translate(**options)
        except G.Unsupported as error:
            assert fragment in str(error), (options, error)
        else:
            raise AssertionError((options, 'unexpected translation'))
    print('pixel image first-lane and formatted store controls passed')


def fixture(pipeline, *, untouched=False):
    def float_word(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]

    source_values = {
        ('uav_descword3', 0): desc_word3(71),
        ('uav_stride', 0): 8,
        ('uav_numrecords', 0): RECORDS,
    }
    source_values.update({('rootconst', 1, i): float_word(word) for i, word in enumerate(CONSTANTS)})
    entries = [source_values.get(source, 0) for source in pipeline.uds]
    # Two extra initialized records behind NUM_RECORDS make the shader's index guard observable
    # independently of the native raw view's end-of-buffer bounds.
    initial = struct.pack('<24e', *(-8.0 - n / 4 for n in range(24)))
    coordinate = IMAGE[0] * CONSTANTS[0]
    value = min(2.0, RAMP[int((coordinate % 1.0) * len(RAMP))])
    expected = initial if untouched else struct.pack('<8e', value, 2 * value, 3 * value, 1.0, *CONSTANTS) + initial[16:]
    return (words(len(entries), *entries, *([0] * (32 - len(entries))))
            + struct.pack('<8f', *IMAGE, *RAMP) + initial + expected + struct.pack('<f', value))


def draw(dxc, runner, adapter):
    baseline = translate()
    swapped = translate(indices=(1, 0))
    wrong_texel = translate(texel_x=1)
    oob = translate(indices=(RECORDS, RECORDS + 1))
    oob_source = oob.hlsl_ps()
    unchecked = oob_source.replace('idx >= num_records || ', '')
    assert unchecked != oob_source
    cases = (
        ('baseline', baseline, baseline.hlsl_ps(), False, ()),
        ('swapped', swapped, swapped.hlsl_ps(), False, ('record_mismatch record=0 ',)),
        ('wrong_texel', wrong_texel, wrong_texel.hlsl_ps(), False, ('record_mismatch record=0 ', 'pixel_mismatch ')),
        ('out_of_bounds', oob, oob_source, True, ()),
        ('unchecked', oob, unchecked, True, ('record_mismatch record=4 ',)),
    )
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pixel-formatted-store-', dir=out) as temp:
        directory = Path(temp)
        vertex = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vertex', 'vs_6_0', directory)
        for name, pipeline, source, untouched, diagnostics in cases:
            pixel = compile_hlsl(dxc, 'dxc', source, name, 'ps_6_0', directory)
            data = directory / f'{name}.bin'
            data.write_bytes(fixture(pipeline, untouched=untouched))
            sampler, = pipeline.sampler_lines()
            run = subprocess.run([runner, adapter, '--startup-pixel-formatted-store',
                                  str(vertex), str(pixel), sampler, str(data)], capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stderr, file=sys.stderr)
                return 77
            expected_exit = 1 if diagnostics else 0
            if run.returncode != expected_exit or any(text not in run.stderr for text in diagnostics):
                raise AssertionError(f'{name}: unexpected exit {run.returncode}\n{run.stdout}\n{run.stderr}')
            print(f'{name}: {"detected mutant" if diagnostics else "buffer and 16 pixels match CPU reference"}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 4:
        sys.exit(draw(*sys.argv[1:]))
    else:
        sys.exit('usage: startup_pixel_store_format_xyzw_check.py [dxc offscreen_check adapter]')
