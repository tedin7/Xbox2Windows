"""Buffer reads of a translated slot 57 mesh program follow the AMD MUBUF address rules (R11).

usage: startup_engine_mesh57_buffer_read_check.py
       startup_engine_mesh57_buffer_read_check.py binary <llvm-mc>
       startup_engine_mesh57_buffer_read_check.py draw <dxc> <offscreen_check> <warp|hardware>

The mesh fixture of startup_mesh57_skeleton_check.py reads its vertex data with raw `offen` loads. Here the reads change, and the picture the
translated program draws is compared with a CPU model of the buffer address rules of the AMD RDNA 2 ISA guide (MUBUF: byte address =
soffset + voffset (offen) + index * stride (idxen) + instruction offset; a read outside the bound view returns 0):

  soffset      the colour load takes its byte offset from an SGPR (`s_mov_b32 s40, 192`) instead of the instruction offset: same picture.
  structured   position and colour loads use `idxen` (index = the vertex slot) with the stride from the descriptor (16): same picture.
  out of range the colour load starts past the end of the 512-byte view (`s_mov_b32 s40, 1024`): every colour channel reads 0.
Each wrong variant is also drawn against the reference of the right address and must fail with exactly the number of pixels the model gives:
a colour read one vertex too far (soffset 208), a read past the view, and the structured load with stride 0 (every vertex reads slot 0).
Without arguments only the translation is checked (the variants translate; only the structured one asks the runtime for the stride constant).
`binary` compares the embedded words with llvm-mc; `draw` compiles the request path's output (ms_6_5, ps_6_0) and draws it.
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_mesh57_skeleton_check as T  # noqa: E402

MESH = T.MESH
COLOUR_LOAD = 'e03810c0'      # buffer_load_dwordx4 v[20:23], v4, s[16:19], 0 offen offset:192 (first word)
COLOUR_LOAD_SECOND = '80041404'
POSITION_SHIFT = '34080884'   # v_lshlrev_b32 v4, 4, v4: the byte offset of the vertex slot
TABLE_LOAD_LITERAL = 'fa000000'   # second word of s_load_dwordx4 s[16:19], s[10:11], 0


def soffset_variant(value):
    """The colour load takes its offset from s40 (= value) with no instruction offset: `... s[16:19], s40 offen`."""
    at = MESH.index(TABLE_LOAD_LITERAL) + 1
    words = MESH[:at] + ['bea803ff', '%08x' % value] + MESH[at:]   # s_mov_b32 s40, <literal>
    first = words.index(COLOUR_LOAD)
    return words[:first] + ['e0381000', '28041404'] + words[first + 2:]


def structured_variant():
    """Position and colour loads index the buffer by vertex slot (`idxen`): v4 = slot, byte address = slot * stride + offset."""
    words = [w for i, w in enumerate(MESH) if not (w == POSITION_SHIFT and i == MESH.index(POSITION_SHIFT))]
    words[words.index('e0381000')] = 'e0382000'
    words[words.index(COLOUR_LOAD)] = 'e03820c0'
    return words


SOFFSET_COLOUR = soffset_variant(192)
SOFFSET_WRONG = soffset_variant(208)
SOFFSET_OUT_OF_RANGE = soffset_variant(1024)
STRUCTURED = structured_variant()

LOAD = 's_load_dwordx4 s[16:19], s[10:11], 0\n'
COLOUR_ASM = 'buffer_load_dwordx4 v[20:23], v4, s[16:19], 0 offen offset:192\n'
POSITION_ASM = 'buffer_load_dwordx4 v[16:19], v4, s[16:19], 0 offen\n'
SHIFT_ASM = 'v_lshlrev_b32 v4, 4, v4\n'


def soffset_asm(text, value):
    return (text.replace(LOAD, LOAD + 's_mov_b32 s40, %d\n' % value)
            .replace(COLOUR_ASM, 'buffer_load_dwordx4 v[20:23], v4, s[16:19], s40 offen\n'))


def structured_asm(text):
    return (text.replace(SHIFT_ASM, '').replace(POSITION_ASM, 'buffer_load_dwordx4 v[16:19], v4, s[16:19], 0 idxen\n')
            .replace(COLOUR_ASM, 'buffer_load_dwordx4 v[20:23], v4, s[16:19], 0 idxen offset:192\n'))


def binary(mc):
    text = (HERE / 'startup_mesh57_skeleton.s').read_text()
    T.expect('soffset words match llvm-mc', T.assemble(mc, soffset_asm(text, 192)), SOFFSET_COLOUR)
    T.expect('wrong soffset words match llvm-mc', T.assemble(mc, soffset_asm(text, 208)), SOFFSET_WRONG)
    T.expect('out-of-range soffset words match llvm-mc', T.assemble(mc, soffset_asm(text, 1024)), SOFFSET_OUT_OF_RANGE)
    T.expect('structured words match llvm-mc', T.assemble(mc, structured_asm(text)), STRUCTURED)


POINTS, COLOURS, INDICES = T.geometry()
WORDS = [a | b << 10 | c << 20 for a, b, c in INDICES]
VIEW = T.buffer(POINTS, COLOURS, WORDS)   # the 512 bytes the draw harness binds as the vertex buffer view


def read_u32(address):
    """The ISA rule for a raw read: outside the bound view it returns 0."""
    return struct.unpack_from('<I', VIEW, address)[0] if 0 <= address and address + 4 <= len(VIEW) else 0


def read_f32(address):
    return struct.unpack('<f', struct.pack('<I', read_u32(address)))[0]


def unorm8(value):
    return 0 if value != value else int(max(0.0, min(1.0, value)) * 255 + 0.5)


def model_image(position_at, colour_at):
    """What the program draws when vertex slot i is read at byte `position_at(i)` (position) and `colour_at(i)` (colour)."""
    points, colours = [], []
    for slot in range(len(POINTS)):
        x, y = read_f32(position_at(slot)), read_f32(position_at(slot) + 4)
        points.append((round((x + 1) * 16), round((1 - y) * 16)))
        colours.append(tuple(unorm8(read_f32(colour_at(slot) + 4 * c)) for c in range(4)))
    return T.rasterise(points, colours, INDICES, len(POINTS), len(INDICES))


RIGHT = model_image(lambda slot: 16 * slot, lambda slot: 192 + 16 * slot)
# (name, words, stride constant or None, the picture the words must draw, the picture the draw is compared with)
CASES = (
    ('soffset', SOFFSET_COLOUR, None, RIGHT, RIGHT),
    ('soffset one vertex too far', SOFFSET_WRONG, None,
     model_image(lambda slot: 16 * slot, lambda slot: 208 + 16 * slot), RIGHT),
    ('out of range', SOFFSET_OUT_OF_RANGE, None, model_image(lambda slot: 16 * slot, lambda slot: 1024 + 16 * slot), RIGHT),
    ('structured, stride 16', STRUCTURED, 16, RIGHT, RIGHT),
    ('structured, stride 0', STRUCTURED, 0, model_image(lambda slot: 0, lambda slot: 192), RIGHT),
)


def controls():
    import os
    os.environ['X2W_FXC'] = ''   # sources and manifest only; the draw mode compiles
    T.expect('the right picture is the standard draw of the fixture',
             RIGHT, T.rasterise(POINTS, COLOURS, INDICES, len(POINTS), len(INDICES)))
    for name, words, stride, drawn, compared in CASES:
        T.expect(name + ': a wrong read draws another picture than the right one', drawn != RIGHT, words in (SOFFSET_WRONG, SOFFSET_OUT_OF_RANGE) or stride == 0)
        with tempfile.TemporaryDirectory() as tmp:
            key, result = T.request(tmp, words)
            T.expect(name + ' translates', result, 'ok')
            manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
            T.expect(name + ': stride constant requested only by idxen loads',
                     [line for line in manifest if re.fullmatch(r'udsrc \d+ stride \d+', line)] != [], words is STRUCTURED)
            T.expect(name + ': one vertex SRV', [line for line in manifest if line.startswith('srv ')], ['srv 0 buf table 1 0'])


def draw(dxc, renderer, adapter):
    import os
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    for name, words, stride, drawn, compared in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            key, result = T.request(tmp, words)
            if result != 'ok':
                raise AssertionError('%s: %s' % (name, result))
            base = str(Path(tmp) / ('%016x' % key))
            manifest = Path(base + '.ms57.txt').read_text().splitlines()
            uds = {tuple(fields[2:]): int(fields[1]) for fields in map(str.split, manifest) if fields and fields[0] == 'udsrc'}
            constants = bytearray(1024)
            struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '0'], len(INDICES))
            struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '1'], len(POINTS))
            if stride is not None:
                struct.pack_into('<I', constants, 4 * uds['stride', '0'], stride)
            expected = sum(a != b for a, b in zip(drawn, compared))
            path = Path(tmp) / 'case.bin'
            path.write_bytes(bytes(constants) + VIEW + b'\xcd' * 80 + b''.join(map(bytes, compared)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed', base + '.ms57.cso', base + '.ps57.cso', str(path)],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if (run.returncode != int(bool(expected))
                    or not re.search(rf'\bpixels={T.SIDE * T.SIDE} mismatches={expected}\b', run.stdout)):
                raise AssertionError(f'{name}: want {expected} differing pixels: {run.stdout}{run.stderr}')
            print(f'{name}: {expected} differing pixels from the right picture')
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
    if T.FAILED:
        return 1
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
