"""Image instructions refuse a short operand list instead of indexing it.

`image()` reads vdata, an address and a descriptor. A list shorter than three used to reach `ops[1]` or
`ops[2]`, which `Stage.translate` records as `translator crash in <op>: IndexError` -- its marker for a
translator defect rather than for an instruction it does not lower -- and which the single-shader path does
not catch at all. The accepted forms need a bound descriptor, so this check pins the refusal side only; the
accepted shapes are covered by the per-form fixtures.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

FORMS = ('image_sample', 'image_sample_l', 'image_sample_lz', 'image_sample_b',
         'image_get_lod', 'image_load', 'image_load_mip')


def translate(name, operands):
    pipe = SimpleNamespace(root={'params': [], 'flags': 0})
    text = ' '.join((name, ', '.join(operands))).strip()
    refusals = []
    stage = G.Stage(pipe, 'ps', b'', refusals=refusals, decoded=[(0, 4, name, text)])
    return refusals, stage.blocks


def refuse(name, operands, expected):
    refusals, blocks = translate(name, operands)
    assert len(refusals) == 1, (name, operands, refusals)
    assert not blocks, (name, operands, blocks)
    assert 'translator crash' not in refusals[0][0], (name, operands, refusals)
    assert expected in refusals[0][0], (name, operands, refusals)


def main():
    for form in FORMS:
        refuse(form, ('v[8:11]',), 'image instruction with 1 operands, expected at least 3')
        refuse(form, ('v[8:11]', 'v0'), 'image instruction with 2 operands, expected at least 3')

    print('startup gfx2hlsl image arity: %d forms refused at 1 and 2 operands' % len(FORMS))


if __name__ == '__main__':
    main()
