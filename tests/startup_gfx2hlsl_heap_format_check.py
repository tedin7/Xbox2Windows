"""Python controls for formatted reads from a dynamically indexed heap record."""
from startup_gfx2hlsl_resheap_branch_check import (
    ELSE_VS, READ, REWRITE, LOAD, S_RECORD, S_WAITCNT, LGKM0, VM0,
    assemble, mubuf, sopp, translate, refusal, vertex_copy, expect,
)
from startup_compute_dynamic_heap_check import CS, G, GROUP, RSRC2, root_blob


def main():
    # MUBUF opcode zero is buffer_load_format_x; bit 13 selects idxen.
    formatted = bytearray(mubuf(0, 8, 0, S_RECORD))
    formatted[1] = (formatted[1] & ~0x10) | 0x20
    fixture = ELSE_VS.replace(READ, bytes(formatted))
    expect('fixture changes one instruction', fixture != ELSE_VS)
    pipe, source, _ = translate(fixture)
    expect('dynamic heap manifest', 'dynamic_heap\n' in pipe.manifest())
    expect('format manifest', 'dynamic_heap_formats\n' in pipe.manifest())
    expect('format read', 'x2w_heap_info(' in source and 'switch ((info.y >> 12) & 127u)' in source)
    raw_pipe, _, _ = translate(ELSE_VS)
    expect('raw manifest unchanged', 'dynamic_heap_formats\n' not in raw_pipe.manifest())
    rewritten = vertex_copy([LOAD, sopp(S_WAITCNT, LGKM0)] + REWRITE + [bytes(formatted), sopp(S_WAITCNT, VM0)])
    expect('rewritten descriptor refused', refusal(rewritten) is not None)
    # Replace the raw read with a formatted four-component indexed read.
    compute_load = bytearray(mubuf(3, 5, 4, S_RECORD))
    compute_load[1] = (compute_load[1] & ~0x10) | 0x20
    compute = CS.replace(bytes.fromhex('001030e004050480'), bytes(compute_load))
    compute = compute.replace(bytes.fromhex('8002087e'), bytes.fromhex('8102087e'))
    compute_pipe = G.ComputePipeline(compute, root_blob(compute=True), RSRC2, GROUP)
    compute_pipe.hlsl_cs()
    expect('compute format manifest', 'dynamic_heap\n' in compute_pipe.manifest()
           and 'dynamic_heap_formats\n' in compute_pipe.manifest())


if __name__ == '__main__':
    main()
