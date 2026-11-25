"""Translation controls for three-source integer min, max and median in gfx2hlsl.

  (no arguments)                       translation controls, pure Python
"""
import ast
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402


OPS = {
    'v_min3_i32': ('uint(min(min(int(v2), int(v3)), int(v4)))', 0, True),
    'v_med3_i32': ('uint(max(min(int(v2), int(v3)), min(max(int(v2), int(v3)), int(v4))))', 1, True),
    'v_max3_i32': ('uint(max(max(int(v2), int(v3)), int(v4)))', 2, True),
    'v_min3_u32': ('min(min(v2, v3), v4)', 0, False),
    'v_med3_u32': ('max(min(v2, v3), min(max(v2, v3), v4))', 1, False),
    'v_max3_u32': ('max(max(v2, v3), v4)', 2, False),
}
MASK = 0xffffffff
# Both distinct triples have different signed and unsigned medians.
INPUTS = [(0, 0, 0), (MASK, MASK, MASK), (0x80000000, 0x80000000, 0x80000000)]
for triple in ((MASK, 0, 1), (0x80000000, 0, 0x7fffffff), (MASK, MASK, 1), (0, 1, 1)):
    INPUTS.extend(permutations(triple))


def signed(word):
    return word if word < 0x80000000 else word - 0x100000000


def translate(asm):
    stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)
    stage.lines = []
    stage.used_v = set()
    stage.uniform_v = set()
    stage.exec_known = True
    stage.one(asm.split()[0], asm)
    return stage.lines


def refused(asm, reason):
    try:
        translate(asm)
    except gfx2hlsl.Unsupported as exc:
        assert str(exc) == reason, (asm, str(exc))
    else:
        raise AssertionError('accepted %s' % asm)


FUNCTIONS = {'int': signed, 'uint': lambda value: value & MASK, 'min': min, 'max': max}


def evaluate(expr, values):
    """Evaluate an emitted min/max/int/uint expression over v2..v4, accepting nothing else."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
                and not node.keywords):
            return FUNCTIONS[node.func.id](*[walk(arg) for arg in node.args])
        raise AssertionError('unexpected expression node %s in %s' % (type(node).__name__, expr))
    return walk(ast.parse(expr, mode='eval'))


def main():
    for op, (expected, rank, is_signed) in OPS.items():
        for suffix in ('', '_e64'):
            asm = '%s%s v1, v2, v3, v4' % (op, suffix)
            lines = translate(asm)
            assert lines == ['v1 = %s;' % expected], (asm, lines)
            expr = lines[0][len('v1 = '):-1]
            for a, b, c in INPUTS:
                result = evaluate(expr, {'v2': a, 'v3': b, 'v4': c})
                ordered = sorted((a, b, c), key=signed if is_signed else None)
                assert result == ordered[rank], (asm, (a, b, c), result, ordered[rank])

    # Inline-constant and literal sources go through the shared u() path as 32-bit words.
    lines = translate('v_med3_i32 v1, v2, -1, 0x80000000')
    expected = ('v1 = uint(max(min(int(v2), int(0xffffffffu)), min(max(int(v2), int(0xffffffffu)), '
                'int(0x80000000u))));')
    assert lines == [expected], lines
    refused('v_med3_i32_e64 v1, v2, v3, v4 clamp', "v_med3_i32 with ['clamp']")
    refused('v_med3_i32_e64 v1, -v2, v3, v4', "operand '-v2'")
    entries = {op: gfx2hlsl.VEC_U.pop(op) for op in OPS}
    try:
        refused('v_med3_i32 v1, v2, v3, v4', 'v_med3_i32')
    finally:
        gfx2hlsl.VEC_U.update(entries)
    print('PASS: startup_gfx2hlsl_int_med3_check')


if __name__ == '__main__':
    main()
