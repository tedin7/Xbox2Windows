"""Check new direct DXIL integer atomics on the typed 2D uint UAV."""
from pathlib import Path
import operator
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_image_atomic_check import (
    OPERAND, TEXELS, UINT_MASK, WIDTH, image_atomic, program_words, reference, render,
)


OPS = {
    'swap': (0x0f, 8, 1 << 28, lambda old, operand: operand),
    'umin': (0x15, 6, 1 << 27, min),
    'and': (0x18, 1, 1 << 25, operator.and_),
    'or': (0x19, 2, 1 << 25, operator.or_),
    'xor': (0x1a, 3, 1 << 25, operator.xor),
}
REFUSED = {'sub': 0x12, 'smin': 0x14, 'smax': 0x16, 'inc': 0x1b, 'dec': 0x1c,
           'fcmpswap': 0x1d, 'fmin': 0x1e, 'fmax': 0x1f}
OPERAND_DIAGNOSTIC = 'unsupported typed 2D image atomic operands'


def compile_shader(cli, sdk, work, opcode, *, glc=True, sparse=False, contend=False, **encoding):
    # Retain the fixture's coordinates, EXEC mask, wait, and old-value probe.
    # With GLC clear the probe observes the original data VGPR instead.
    code = program_words('add', returning=True, drop_glc=not glc, sparse=sparse,
                         contend=contend, **encoding)
    flags = encoding.get('flags', 0) | (0x2000 if glc else 0)
    atomic_word, _ = image_atomic('add', flags=flags, mask=encoding.get('mask'), dim=encoding.get('dim', 1))
    index = code.index(atomic_word)
    code[index] = (atomic_word & ~(0x7f << 18)) | (opcode << 18)
    if contend:
        # The original fixture uses 1 for contended add. Exercise the same
        # high-bit operand as the distinct-texel case here.
        code[code.index(0x7e1c02ff) + 1] = OPERAND
    source = work / 'binop.cs'
    shader = source.with_suffix('.cso')
    shader.unlink(missing_ok=True)
    source.write_bytes(compute(words(*code), table=True))
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                            capture_output=True, text=True, timeout=30)
    return shader, result


def controls(cli, sdk, work):
    for name, (opcode, dxil, _, _) in OPS.items():
        shader, run = compile_shader(cli, sdk, work, opcode)
        assert run.returncode == 0, (
            f'{name}: valid typed image atomic refused (operand diagnostic: {OPERAND_DIAGNOSTIC})',
            run.stdout, run.stderr,
        )
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader],
                              capture_output=True, text=True, timeout=30, check=True).stdout
        # The second atomic is the existing add probe and must have code 0.
        codes = re.findall(r'@dx\.op\.atomicBinOp\.i32\(i32 78, %dx\.types\.Handle [^,]+, i32 (\d+),', dump)
        assert codes == [str(dxil), '0'] and '2d' in dump, (name, dump)
        for encoding in ({'dim': 5}, {'mask': 15}, {'coord': 255}, {'flags': 0x80},
                         {'addr_flags': 0x40000000}, {'addr_flags': 0x200000}):
            _, run = compile_shader(cli, sdk, work, opcode, **encoding)
            assert run.returncode and OPERAND_DIAGNOSTIC in run.stderr, (name, encoding, run.stdout, run.stderr)
    for name, opcode in REFUSED.items():
        _, run = compile_shader(cli, sdk, work, opcode)
        assert run.returncode and 'direct image opcode or descriptor modifiers' in run.stderr, (
            name, run.stdout, run.stderr,
        )
    print('Typed image binary atomic DXIL codes and refusal diagnostics matched')


def expected_texels(update, *, glc, contend):
    initial, _ = reference('add')
    expected = initial.copy()
    for lane in range(0, 32, 2):
        x, y = (5, 6) if contend else (2 + lane % 8, 1 + lane // 8)
        index = y * WIDTH + x
        previous = expected[index]
        expected[index] = update(previous, OPERAND)
        # Contending lanes use the same operand, so the aggregate of returned
        # values depends only on the number of updates, regardless of order.
        probe = index + 12
        expected[probe] = (expected[probe] + (previous if glc else OPERAND)) & UINT_MASK
    return initial, expected


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    print(wave.stdout, end='')
    for name, (opcode, _, format_selector, update) in OPS.items():
        for contend in (False, True):
            for glc in (False, True):
                shader, run = compile_shader(cli, sdk, work, opcode, glc=glc, sparse=True, contend=contend)
                assert run.returncode == 0, (name, glc, contend, run.stdout, run.stderr)
                initial, expected = expected_texels(update, glc=glc, contend=contend)
                payload = work / 'texture.bin'
                payload.write_bytes(words(1 | format_selector, 0, *initial, *expected))
                run = render(renderer, adapter, '--compute-texture-store-case', shader, payload)
                assert run.returncode == 0, (name, glc, contend, run.stdout, run.stderr)
                if 'debug_layer=enabled' in run.stdout:
                    assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
                print(f'{name} GLC={glc} contended={contend}: {TEXELS} texture words matched')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
