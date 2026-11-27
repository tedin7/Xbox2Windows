"""Formatted dynamic heap translation controls and optional WARP readback."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from startup_gfx2hlsl_resheap_branch_check import (
    ELSE_VS, READ, REWRITE, LOAD, S_RECORD, S_WAITCNT, LGKM0, VM0,
    mubuf, sopp, translate, refusal, vertex_copy, expect, PS_BRANCHED, VS_WORDS, REFUSAL, COPY_V8,
)
from startup_compute_dynamic_heap_check import CS, G, GROUP, RSRC2, root_blob
from startup_gfx2hlsl_dynamic_heap_check import IMAGE_HEAD, compile_shader
from startup_formatted_view_draw_check import cpu_decode, pack_element


def main():
    # MUBUF opcode zero is buffer_load_format_x; bit 13 selects idxen.
    formatted = bytearray(mubuf(0, 8, 0, S_RECORD))
    formatted[1] = (formatted[1] & ~0x10) | 0x20
    fixture = ELSE_VS.replace(READ, bytes(formatted))
    expect('fixture changes one instruction', fixture != ELSE_VS)
    pipe, _, _ = translate(fixture)
    expect('dynamic heap manifest', 'dynamic_heap\n' in pipe.manifest())
    expect('format manifest', 'dynamic_heap_formats\n' in pipe.manifest())
    raw_pipe, _, _ = translate(ELSE_VS)
    expect('raw manifest unchanged', 'dynamic_heap_formats\n' not in raw_pipe.manifest())
    rewritten = vertex_copy([LOAD, sopp(S_WAITCNT, LGKM0)] + REWRITE + [bytes(formatted), sopp(S_WAITCNT, VM0)])
    expect('rewritten descriptor refused', refusal(rewritten) == 'indexed load through a modified descriptor')
    image_ps = bytes.fromhex(IMAGE_HEAD + '8002087e') + bytes(formatted) + bytes.fromhex('703f8cbf0f1800f804040404000081bf')
    expect('image record refused', refusal(VS_WORDS, image_ps) == 'dynamic resource heap descriptor requires eight intact words')
    pixel_load = bytearray(mubuf(0, 5, 4, S_RECORD))
    pixel_load[1] = (pixel_load[1] & ~0x10) | 0x20
    pixel_branch = PS_BRANCHED.replace(mubuf(0x0c, 5, 4, S_RECORD), bytes(pixel_load))
    expect('pixel branch refused', refusal(VS_WORDS, pixel_branch) == REFUSAL)
    # Replace the raw read with a formatted four-component indexed read.
    compute_load = bytearray(mubuf(3, 5, 4, S_RECORD))
    compute_load[1] = (compute_load[1] & ~0x10) | 0x20
    compute = CS.replace(bytes.fromhex('001030e004050480'), bytes(compute_load))
    compute = compute.replace(bytes.fromhex('8002087e'), bytes.fromhex('8102087e'))
    compute_pipe = G.ComputePipeline(compute, root_blob(compute=True), RSRC2, GROUP)
    compute_pipe.hlsl_cs()
    expect('compute format manifest', 'dynamic_heap\n' in compute_pipe.manifest()
           and 'dynamic_heap_formats\n' in compute_pipe.manifest())


def warp(offscreen, dxc):
    load = bytearray(mubuf(3, 6, 11, S_RECORD))
    load[1] = (load[1] & ~0x10) | 0x20
    fixture = ELSE_VS.replace(READ, bytes(load)).replace(COPY_V8, b'').replace(
        bytes.fromhex('8002167e'), bytes.fromhex('8102167e'))
    pipe, vs_source, ps_source = translate(fixture)
    cases = [
        (20, [32], 'UINT', [0x12345678], (4, 5, 6, 7), 4),
        (11, [16], 'UINT', [0x9321], (4, 5, 6, 7), 2),
        (29, [16, 16], 'FLOAT', [0x3c00, 0x4000], (4, 5, 6, 7), 4),
        (56, [8, 8, 8, 8], 'UNORM', [255, 0, 255, 0], (4, 5, 6, 7), 4),
        (36, [11, 11, 10], 'FLOAT', [0x2e9, 0x031, 0x225], (4, 5, 6, 7), 4),
        (20, [32], 'UINT', [0x12345678], (7, 0, 4, 1), 12),
    ]
    def words(blob):
        blob += bytes((-len(blob)) % 4)
        return list(struct.unpack('<%dI' % (len(blob) // 4), blob))
    def run(vs, ps, record, info, expected, index=0, mutant=False):
        entry = [0] * len(pipe.uds) + [2]
        entry[1], entry[2] = index, 0
        records = [record, [0] * 8, info]
        text = lambda seq: ','.join('%08x' % n for n in seq)
        result = subprocess.run([offscreen, 'warp', '--startup-dynamic-heap-records', str(vs), str(ps), text(entry),
                                 text(expected), ';'.join(text(r) for r in records)], capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout, end='')
            sys.exit(77)
        assert result.returncode == int(mutant), result.stdout + result.stderr
        if mutant:
            assert 'startup_entry_words mismatch' in result.stderr
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', vs_source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        for code, widths, kind, fields, selectors, stride in cases:
            element = pack_element(fields, widths)
            decoded = cpu_decode(element, widths, kind)
            expected = [decoded[s - 4] if s >= 4 else decoded[4] if s == 0 else decoded[5] for s in selectors]
            word3 = (code << 12) | sum(s << (3 * i) for i, s in enumerate(selectors))
            record = words(bytes(stride) + element + bytes(16))
            info = [stride, word3, stride, 0]
            run(vs, ps, record, info, expected)
        run(vs, ps, record, info, [0] * 4, index=2)
        run(vs, ps, record, [stride, 0, stride, 0], [0] * 4)
        assert vs_source.count('index * 8u') == 1
        mutant = compile_shader(dxc, directory, 'vs_6_6', vs_source.replace('index * 8u', 'index * 8u + 8u'))
        run(mutant, ps, record, info, expected, mutant=True)
    print('formatted dynamic heap: WARP cases and mutant matched')


if __name__ == '__main__':
    if sys.argv[1:2] == ['warp']:
        warp(*sys.argv[2:])
    else:
        main()
