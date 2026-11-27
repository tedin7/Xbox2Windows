"""Controls for GETPC-relative per-lane GLOBAL address pattern recognition."""
import hashlib
import os
import struct
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
import gfx2hlsl as G
from startup_buffer_address_check import encode_mtbuf, encode_vop1, encode_vop2, root_with_one_constant
from startup_gfx2hlsl_getpc_refusal_check import export
import startup_stream_two_exit_check as streams
import startup_stream_interface_check as stream_interface
import gpu_case_runner
from startup_flat_param_check import compile_hlsl
from startup_compute_global_load_check import encode_global, GLOBAL_LOAD_DWORD
from startup_gfx2hlsl_add_sub_co_ci_check import vop3b


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def program(*, nwords=763, unbounded=False, shift=2, high_source=19, after_shift=b'', extra=b'', initial=b'', addressed=False):
    mtbuf = encode_mtbuf(vdata=9, offen=int(addressed), vaddr=0)
    fma = words(0xd54b000f, 255 | (268 << 9) | (240 << 18), 0x437f0000)
    cvt = encode_vop1(7, 15, 270 if unbounded else 271)
    cmpx = words(0x7c000000 | (0xd5 << 17) | (15 << 9) | 255, 255)
    triple = words(0xd7460012, 271 | (129 << 9) | (271 << 18))
    ashr = encode_vop2(0x18, 19, 159, 18)
    shift64 = words(0xd6ff0012, (128 + shift) | (274 << 9))
    prefix = initial + mtbuf + fma + cvt + cmpx + triple + ashr + shift64 + after_shift
    pc = len(prefix)
    low = words(*vop3b(0x30f, 20, 106, 274, 16, 0))
    high = words(*vop3b(0x128, 21, 106, 17, 256 + high_source, 106))
    load = encode_global(GLOBAL_LOAD_DWORD, 7, 20, 0x7d)
    rest = words(0xbe901f00, 0x8010ff10, 0, 0x82118011) + low + high + load + extra + words(0xbf810000)
    code = prefix + rest
    code = code[:pc + 8] + words(len(code) - pc - 4) + code[pc + 12:]
    image = code + b''.join(words(0x10000000 + i) for i in range(nwords))
    return image, G.decode(code), len(code) - len(extra) - len(load) - 4


def check_refusal(text, **kwargs):
    code, ins, _ = program(**kwargs)
    try:
        plan = G.vertex_image_getpc_proof(code, ins)
        G.vertex_image_address_proof(ins, plan)
    except G.Unsupported as error:
        assert text in str(error), (text, error)
    else:
        raise AssertionError('expected refusal: ' + text)



def graphics_program(*, before_exports=b'', **kwargs):
    initial = words(0xf4080400, 0xfa000010)  # s_load_dwordx4 s[16:19], s[0:1], 16
    extra = (before_exports + encode_vop1(1, 8, 242) + export(12, (8, 8, 8, 8))
             + export(32, (7, 7, 7, 7)) + export(33, (7, 7, 7, 7), done=True))
    return program(initial=initial, extra=extra, addressed=True, **kwargs)[0]


def pipeline(code):
    pixel = export(0, (0, 1, 2, 3), done=True) + words(0xbf810000)
    return G.Pipeline(code, pixel, root_with_one_constant())


def translator_controls():
    image = graphics_program()
    pipe = pipeline(image)
    source = pipe.hlsl_vs()
    assert source.count('static const uint x2w_image_table[') == 1
    assert source.count('x2w_image_table[x2w_image_index_') == 1
    assert 'program image' not in pipe.manifest()
    for value in range(255):
        assert 3 * value < 763
    table = [0x10000000 + i for i in range(763)]
    assert all(table[3 * i] != table[3 * i + 1] for i in (0, 1, 7, 126, 253))
    for kwargs in ({'nwords': 762}, {'unbounded': True}):
        try:
            pipeline(graphics_program(**kwargs))
        except G.Unsupported as error:
            assert 'program image load index not proven in range' in str(error), error
        else:
            raise AssertionError('unbounded vertex image load accepted')
    for addition, reason in ((encode_vop1(1, 0, 276), 'address word escapes'),
                             (encode_vop1(1, 0, 16), 'GETPC pair escapes')):
        try:
            pipeline(graphics_program(before_exports=addition))
        except G.Unsupported as error:
            assert reason in str(error), (reason, error)
        else:
            raise AssertionError('escaping address accepted')
    # Exercise the DXBC stream request and compare an unrelated fixture with
    # the pre-change HLSL and manifest hashes.
    saved = os.environ.get('X2W_FXC')
    os.environ['X2W_FXC'] = ''
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            word_image = list(struct.unpack('<%dI' % (len(image) // 4), image))
            status, _ = stream_interface.translate(path / 'new', streams.vertex(word_image),
                                                   stream_interface.pixel(), root_with_one_constant())
            assert status == 'ok', status
            status, _ = stream_interface.translate(path / 'old', stream_interface.vertex(), stream_interface.pixel())
            assert status == 'ok', status
            want = {
                '.ps.hlsl': '7b7b2b7161c8df8dc4c526e45618208f29ce0edb99ad46ae0baa839ddaf93173',
                '.vs.hlsl': '3742286636121f9816ec552bbd0761f4dd2a90356b49dceef3fc30ab3fbbe633',
                '.txt': '5cb778176c0dd11018a6919890a39f6aacaefe6218b728397faf4deca8eaa120',
            }
            for suffix, digest in want.items():
                matches = list((path / 'old').glob('*' + suffix))
                assert len(matches) == 1 and hashlib.sha256(matches[0].read_bytes()).hexdigest() == digest, suffix
    finally:
        if saved is None:
            os.environ.pop('X2W_FXC', None)
        else:
            os.environ['X2W_FXC'] = saved
    print('vertex image translator controls: passed')


def main():
    code, ins, load_pc = program()
    proof = G.vertex_image_getpc_proof(code, ins)
    loads = G.vertex_image_address_proof(ins, proof)
    assert loads == {load_pc: ('v7', 44, 762)}, loads
    G.vertex_image_elision_proof(ins, proof, loads)
    check_refusal('program image load index not proven in range', nwords=762)
    check_refusal('program image load index not proven in range', unbounded=True)
    check_refusal('program image address is not a proven index shifted by two', shift=1)
    check_refusal('program image address high word does not match the shifted pair', high_source=18)
    check_refusal('program image address is not a proven index shifted by two',
                  after_shift=encode_vop1(1, 18, 128))
    check_refusal('program image address is not a proven index shifted by two',
                  after_shift=words(0xbeff037e))
    for extra, reason in (
            (encode_vop1(1, 0, 276), 'address word escapes'),
            (words(0xbf860000), 'vector carry escapes'),
            (words(0xbeff037e) + encode_vop1(1, 20, 128) + encode_vop1(1, 0, 276),
             'address word escapes')):
        code, ins, _ = program(extra=extra)
        plan = G.vertex_image_getpc_proof(code, ins)
        loads = G.vertex_image_address_proof(ins, plan)
        try:
            G.vertex_image_elision_proof(ins, plan, loads)
        except G.Unsupported as error:
            assert reason in str(error), (reason, error)
        else:
            raise AssertionError('expected refusal: ' + reason)
    code, ins, _ = program(extra=encode_vop1(1, 20, 128) + encode_vop1(1, 0, 276))
    plan = G.vertex_image_getpc_proof(code, ins)
    G.vertex_image_elision_proof(ins, plan, G.vertex_image_address_proof(ins, plan))
    print('vertex image address controls: passed')
    translator_controls()


def draw(compiler, renderer, adapter):
    samples = (0, 1, 2, 7, 31, 83, 126, 253)
    table = [0x10000000 + i for i in range(763)]
    expected_words = [table[3 * value] for value in samples]
    wrong_words = [table[3 * value + 1] for value in samples]
    assert all(a != b for a, b in zip(expected_words, wrong_words))
    source = pipeline(graphics_program()).hlsl_vs().replace('VSOut main(', 'VSOut guest_main(', 1)
    source += """
struct DrawOut { float4 pos : SV_Position; nointerpolation float4 color : COLOR; };
DrawOut main(uint corner : SV_VertexID, uint row : SV_InstanceID) {
    VSOut translated = body(4u * row);
    DrawOut output;
    output.pos = float4(-1.0 + 0.25 * row + (corner == 1u ? 0.5 : 0.0),
                        corner == 2u ? 1.0 : -1.0, 0.0, 1.0);
    output.color = translated.param0;
    return output;
}
"""
    pixel = ('float4 main(float4 pos : SV_Position, nointerpolation float4 color : COLOR) '
             ': SV_Target { return color; }')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(compiler, 'dxc', source, 'vertex_image', 'vs_6_0', directory)
        ps = compile_hlsl(compiler, 'dxc', pixel, 'vertex_image_pixel', 'ps_6_0', directory)
        expected = directory / 'expected.bin'
        inputs = ','.join('%08x' % (sample << 24) for sample in samples)

        def run(values, expect_success):
            expected.write_bytes(b''.join(struct.pack('<4I', *([value] * 4)) for value in values))
            result = gpu_case_runner.run(renderer, adapter, [
                '--startup-instanced-vertex-buffer-words', vs, ps, '8', '1', expected,
                '0,0,0,0', inputs])
            output = result.stdout + result.stderr
            print(output, end='')
            if result.returncode == 77:
                raise SystemExit(77)
            if expect_success:
                assert result.returncode == 0, output
            else:
                assert result.returncode != 0 and '8 of 8 pixels differ' in output, output

        run(expected_words, True)
        run(wrong_words, False)
    print('vertex image %s: every pixel matches; off-by-one reference differs' % adapter)


if __name__ == '__main__':
    main()
    if len(sys.argv) > 1:
        if len(sys.argv) != 5 or sys.argv[1] != 'draw':
            raise SystemExit('usage: startup_vertex_image_address_check.py draw <dxc> <offscreen_check> <adapter>')
        draw(*sys.argv[2:])
