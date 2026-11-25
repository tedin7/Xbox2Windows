"""Source-built gfx1013 shader CFG: a proven PC-relative jump and ambiguous/competing-entry refusals.

Run with no arguments for translation controls; `binary <llvm-mc>` independently
assembles the recorded instructions before running those same controls.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402


class Pipe:
    root = {'params': [{'type': 1, 'count': 1}], 'flags': 0}

    def ud(self, key):
        return 0

    def mrt_kind(self, index):
        assert index == 0
        return 'uint'


# The successor of GETPC is byte 8. The jump must go to byte 24, skipping
# the overwrite of v0 at byte 20. Both the decoy and the target are real
# instruction boundaries. The final value is 3 + 5, not 9 + 5.
START = ('v_mov_b32_e32 v0, 3', '8302007e')
GET = ('s_getpc_b64 s[20:21]', '001f94be')
ADD = ('s_add_u32 s20, s20, 16', '14901480')
CARRY = ('s_addc_u32 s21, s21, 0', '15801582')
SET = ('s_setpc_b64 s[20:21]', '142080be')
DECOY = ('v_mov_b32_e32 v0, 9', '8902007e')
TARGET = ('v_add_nc_u32_e32 v0, 5, v0', '8500004a')
EXPORT = ('exp mrt0 v0, v0, v0, v0 done vm', '0f1800f800000000')
END = ('s_endpgm', '000081bf')
POSITIVE = (START, GET, ADD, CARRY, SET, DECOY, TARGET, EXPORT, END)

# A direct branch into SETPC bypasses the otherwise valid address chain.
# The condition comes from an unconstrained root constant, not a speculative
# entry SCC value. Textual adjacency of GET/ADD/ADDC/SET is not sufficient.
COMPARE = ('s_cmp_eq_u32 s8, 0', '088006bf')
COMPETING = (START, COMPARE, ('s_cbranch_scc1 3', '030085bf'), GET, ADD, CARRY, SET,
             DECOY, TARGET, EXPORT, END)

# Both arms define the pair, but one selects the decoy and the other the
# target. Both destinations are actual instruction starts and both paths
# complete the same export with different arithmetic results.
AMBIGUOUS = (START, COMPARE, ('s_cbranch_scc1 4', '040085bf'), GET,
             ('s_add_u32 s20, s20, 32', '14a01480'), CARRY, ('s_branch 4', '040082bf'),
             GET, ('s_add_u32 s20, s20, 20', '14941480'), CARRY,
             ('s_branch 0', '000082bf'), SET, DECOY, TARGET, EXPORT, END)

# Additional controls exercise address-pair clobbering and an aligned offset
# that lands in the literal word of a longer instruction, not at its start.
CLOBBER = (START, GET, ADD, CARRY, ('s_mov_b32 s21, 0', '800395be'), SET,
           DECOY, TARGET, EXPORT, END)
LITERAL_TARGET = (START, GET, ADD, CARRY, SET,
                  ('v_mov_b32_e32 v0, 0x12345678', 'ff02007e78563412'), TARGET, EXPORT, END)
OUTSIDE = (START, GET, ('s_add_u32 s20, s20, 64', '14c01480'), CARRY, SET,
           DECOY, TARGET, EXPORT, END)
DIRECT = (START, ('s_branch 1', '010082bf'), DECOY, TARGET, EXPORT, END)

READ_LANE = ('v_readfirstlane_b32 s2, v1', '0105047e')
CENSUS = (START, SET, READ_LANE, EXPORT, END)
CENSUS_CONTROL = (START, READ_LANE, EXPORT, END)

# When ud0 == 1, SCC0 takes the direct branch around the entire PC chain.
# The original root word s8 reaches the bypass read at byte 24 untouched.
# On the other path, the calculated jump targets byte 32 and exports 3.
# Neither path executes a numeric read of the PC words.
ROOT_READ = ('v_mov_b32_e32 v0, s8', '0802007e')
ROOT_BYPASS = (
    COMPARE, ('s_cbranch_scc0 4', '040084bf'),
    ('s_getpc_b64 s[8:9]', '001f88be'),
    ('s_add_u32 s8, s8, 20', '08940880'),
    ('s_addc_u32 s9, s9, 0', '09800982'),
    ('s_setpc_b64 s[8:9]', '082080be'),
    ROOT_READ, ('s_branch 1', '010082bf'), START, EXPORT, END,
)
ROOT_DIRECT = (COMPARE, ('s_cbranch_scc0 2', '020084bf'), START,
               ('s_branch 1', '010082bf'), ROOT_READ, EXPORT, END)
FIXTURES = {'positive': POSITIVE, 'competing entry': COMPETING, 'ambiguous': AMBIGUOUS,
            'clobber': CLOBBER, 'literal target': LITERAL_TARGET, 'outside': OUTSIDE, 'direct': DIRECT,
            'root bypass': ROOT_BYPASS, 'root direct': ROOT_DIRECT,
            'refusal census': CENSUS, 'refusal census ordinary control': CENSUS_CONTROL}


def words(fixture):
    return b''.join(bytes.fromhex(encoding) for _, encoding in fixture)


def translate(fixture):
    return G.Stage(Pipe(), 'ps', words(fixture))


def assert_result(stage, start_offsets, jump_target):
    assert [(off, term) for off, _, term in stage.blocks] == [
        (start_offsets[0], ('jump', jump_target)), (start_offsets[1], None),
        (start_offsets[2], ('end',))], stage.blocks
    assert stage.has_branches
    body = stage.body()
    assert body == [
        '    uint pc = 0;',
        '    if (pc == 0) {',
        '        v0 = 0x00000003u;',
        '        pc = 2;',
        '    }',
        '    if (pc == 1) {',
        '        v0 = 0x00000009u;',
        '        pc = 2;',
        '    }',
        '    if (pc == 2) {',
        '        v0 = 0x00000005u + v0;',
        '        o.mrt0 = uint4(v0, v0, v0, v0);',
        '        pc = 0xffffu;',
        '    }',
    ], body

    # Evaluate the lowered arithmetic along the actual emitted CFG path.
    # Independently check the skipped path produces a distinguishable value.
    def eval_path(first):
        values = {'v0': 0}
        for off in (first, jump_target):
            lines = next(lines for start, lines, _ in stage.blocks if start == off)
            for line in lines:
                if line.startswith('v0 = '):
                    values['v0'] = Expr(line[5:-1]).run(values)[0]
                else:
                    assert line == 'o.mrt0 = uint4(v0, v0, v0, v0);', line
        return (values['v0'],) * 4

    assert eval_path(start_offsets[0]) == (8,) * 4
    assert eval_path(start_offsets[1]) == (14,) * 4
    assert stage.write_masks == {'mrt0': 15}


def check_root_bypass():
    control = translate(ROOT_DIRECT)
    assert [(off, term) for off, _, term in control.blocks] == [
        (0, ('branch', '!scc', 16)), (8, ('jump', 20)), (16, None), (20, ('end',)),
    ], control.blocks
    read = control.blocks[2][1]
    assert read == ['v0 = ud0;'], read
    assert Expr(read[0][5:-1]).run({'ud0': 1})[0] == 1
    assert control.blocks[3][1] == ['o.mrt0 = uint4(v0, v0, v0, v0);'], control.blocks

    # The current proof conservatively refuses this case: removing the
    # address chain would otherwise destroy the live root binding in the
    # linear translator even on an edge that does not execute that chain.
    try:
        translate(ROOT_BYPASS)
    except G.Unsupported as reason:
        assert 's_setpc_b64:' in str(reason), str(reason)
        assert 'elided PC register s8 has no matching ordinary definition' in str(reason), str(reason)
        print('root bypass: refused (%s)' % reason)
    else:
        raise AssertionError('elided PC write can destroy a bypassed root binding')


def check_refusal_census():
    calculated = 's_setpc_b64: unknown or ambiguous calculated target from s[20:21] at +0x4'
    ordinary = 'v_readfirstlane_b32 requires a proven uniform source'
    for fixture, expected in ((CENSUS_CONTROL, ordinary), (CENSUS, calculated)):
        try:
            translate(fixture)
        except G.Unsupported as reason:
            assert str(reason) == expected, str(reason)
        else:
            raise AssertionError('normal translation accepted a refused fixture')

    refusals = []
    diagnostic = G.Stage(Pipe(), 'ps', words(CENSUS), refusals=refusals)
    assert refusals == [(calculated, 'ps'), ('s_setpc_b64', 'ps'), (ordinary, 'ps')], refusals
    assert diagnostic.calculated_pc == {}, diagnostic.calculated_pc
    assert [(off, term) for off, _, term in diagnostic.blocks] == [(0, ('end',))], diagnostic.blocks
    assert diagnostic.blocks[0][1] == ['v0 = 0x00000003u;', 'o.mrt0 = uint4(v0, v0, v0, v0);']

    refusals = []
    proven = G.Stage(Pipe(), 'ps', words(POSITIVE), refusals=refusals)
    assert refusals == [], refusals
    assert_result(proven, (0, 20, 24), 24)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        for name, fixture in FIXTURES.items():
            actual = assemble(sys.argv[2], '\n'.join(asm for asm, _ in fixture) + '\n')
            assert actual == words(fixture), (name, actual.hex(), words(fixture).hex())
    elif len(sys.argv) != 1:
        raise SystemExit(__doc__)

    assert_result(translate(POSITIVE), (0, 20, 24), 24)
    assert_result(translate(DIRECT), (0, 8, 12), 12)
    check_root_bypass()
    check_refusal_census()
    for name in ('competing entry', 'ambiguous', 'clobber', 'literal target', 'outside'):
        try:
            translate(FIXTURES[name])
        except G.Unsupported as reason:
            assert 's_setpc_b64' in str(reason), (name, str(reason))
            print(f'{name}: refused ({reason})')
        else:
            raise AssertionError(f'{name}: unresolved calculated target accepted')
    print('Calculated-target CFG, selected arithmetic result and refusals pass')


if __name__ == '__main__':
    main()
