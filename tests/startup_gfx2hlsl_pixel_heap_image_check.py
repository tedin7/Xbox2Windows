"""A pixel program's image record through the resource heap: s_load_dwordx8 D, <heap pointer>, R with R = index << 5.

The index is a root constant word, or that word with an alignment mask (s_and_b32 word, -16) before it becomes a byte offset
of an s_buffer_load from a table buffer. Both name a record the draw knows when it is bound. The manifest line is
`heap word table T B at A rootconst P W times S mask M`. Every other shape stays refused by name.

Controls drive the stage's own marking and load handlers on the compute fixture's pipeline (see
startup_gfx2hlsl_heap_record_check.py); no GPU is involved, the runtime binding is a separate slice.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in (HERE, HERE.parent / 'tools', HERE.parent / 'tools' / 'startup'):
    sys.path.insert(0, str(sub))
import startup_gfx2hlsl_heap_record_check as H  # noqa: E402

G = H.G
REFUSED = H.refused


def stage(kind='ps'):
    st = H.pipeline().cs
    st.kind = kind
    return st


def rootconst_sym(st, word=H.INDEX_WORD):
    return next(('ud', i) for i, u in enumerate(st.pipe.uds) if u == ('rootconst', 0, word))


def main():
    # 1. root constant word << 5 -> the image record (8 dwords)
    st = stage()
    st.heap_index[6] = ('rootconst', 0, H.INDEX_WORD)
    st.op_s_load_dwordx8(['s[16:23]', 's[8:9]', 's6'], [])
    assert [st.sym[16 + i] for i in range(8)] == [('desc', ('resheap', 0, H.INDEX_WORD), 0, i) for i in range(8)], st.sym
    # 2. the widths that do not match the stage are refused
    REFUSED('of 4 dwords', lambda: stage('ps').op_s_load_dwordx4(['s[16:19]', 's[8:9]', 's6'], []))
    st = stage('cs')
    st.heap_index[6] = ('rootconst', 0, H.INDEX_WORD)
    REFUSED('of 8 dwords', lambda: st.op_s_load_dwordx8(['s[16:23]', 's[8:9]', 's6'], []))
    # 3. the mask: (word & -16) is a root constant word with an alignment mask, only in a pixel program
    st = stage()
    sym = rootconst_sym(st)
    st.mark_heap_index('s_and_b32', 7, ['s5', '-16'], [sym, None])
    assert st.heap_index[7] == ('rc', 0, H.INDEX_WORD, 1, 0, 0xfffffff0), st.heap_index
    st.mark_heap_index('s_add_u32', 8, ['s7', '0x10'], [None, None])
    assert st.heap_index[8] == ('rc', 0, H.INDEX_WORD, 1, 0x10, 0xfffffff0), st.heap_index
    st.mark_heap_index('s_mul_i32', 9, ['s7', '4'], [None, None])
    assert 9 not in st.heap_index, 'a multiply of a masked word must not be named'
    # The SOPK forms read the same marks on every pixel shader: s_addk_i32 keeps the mask, s_mulk_i32 does not name it,
    # and neither may fail on a masked mark (a five-value mark once raised ValueError in a program with no heap load).
    st.sym.pop(7, None)  # the real flow leaves the s_and result untracked
    st.heap_index_was = {7: st.heap_index.pop(7)}
    st.op_s_addk_i32(['s7', '0x10'], [])
    assert st.heap_index[7] == ('rc', 0, H.INDEX_WORD, 1, 0x10, 0xfffffff0), st.heap_index
    st = stage()
    st.mark_heap_index('s_and_b32', 7, ['s5', '-16'], [rootconst_sym(st), None])
    st.sym.pop(7, None)
    st.heap_index_was = {7: st.heap_index.pop(7)}  # a write ends the mark; the handler reads the one it replaced
    st.op_s_mulk_i32(['s7', '0x4'], [])
    assert 7 not in st.heap_index, st.heap_index
    st = stage('cs')
    st.mark_heap_index('s_and_b32', 7, ['s5', '-16'], [rootconst_sym(st), None])
    assert 7 not in st.heap_index, 'compute does not read a mask: not named'
    # 4. the mask is written in the manifest word and read back by heap_where
    where = G.heap_where(('resheap', 'word', 0, 16, 0, 2, 1, 4, 0xfffffff0))
    assert where == 'heap word table 0 16 at 4 rootconst 0 2 times 1 mask 4294967280', where
    assert G.heap_where(('resheap', 0, 2)) == 'heap rootconst 0 2'
    print('startup_gfx2hlsl_pixel_heap_image: PASS (x8 record, x4 and compute refusals, mask marks, manifest words)')


if __name__ == '__main__':
    main()
