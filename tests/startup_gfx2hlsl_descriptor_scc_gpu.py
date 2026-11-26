"""Shared pixel and compute readback fixtures for descriptor-word SALU operations."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_gfx2hlsl_compute_check import G, SENTINEL, WORDS, root_blob
from startup_gfx2hlsl_spill_branches_check import program

MASK = 0xffffffff
CASES = (0, 1, 0x3fffffff, 0x40000000, MASK)
BOUND_WORD = 59
ENTRY_ROOT = root_blob(constants=1, table_range=True)
FULLSCREEN_VS = """
float4 main(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0, 0.0, 1.0);
}
"""


def shader(kind, op, rhs, field='descword3'):
    # Root constant zero supplies the initial SCC; the second root parameter is the table.
    entry = 4 if kind == 'cs' else 8
    prepare = {
        'descword3': '',
        'stride': 's_bfe_u32 s39, s37, 0xe0010\n',
        'stride_min1': 's_bfe_u32 s39, s37, 0xe0010\ns_max_u32 s39, s39, 1\n',
    }[field]
    guest = f"""\
s_mov_b32 s31, s{entry + 1}
s_mov_b32 s32, 0
s_load_dwordx4 s[36:39], s[31:32], 0
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s40, s[36:39], 0
s_waitcnt lgkmcnt(0)
{prepare}s_cmp_eq_u32 s{entry}, 0
{op} s39, s39, {rhs}
v_mov_b32 v7, s39
v_mov_b32 v8, 17
s_cbranch_scc0 Lout
v_mov_b32 v8, 29
Lout:
v_mov_b32 v9, s40
"""
    refusals = []
    if kind == 'ps':
        guest += """\
v_and_b32 v10, 0xffff, v7
v_lshrrev_b32 v11, 16, v7
v_cvt_f32_u32 v10, v10
v_cvt_f32_u32 v11, v11
v_cvt_f32_u32 v8, v8
v_cvt_f32_u32 v9, v9
exp mrt0 v10, v11, v8, v9 done vm
s_endpgm
"""
        pipe = G.MeshPixelStage((), b'', ENTRY_ROOT, decoded=program(guest), refusals=refusals)
        source = pipe.hlsl_ps()
    else:
        guest += """\
s_load_dwordx4 s[44:47], s[31:32], 0x30
s_waitcnt lgkmcnt(0)
buffer_store_dwordx3 v[7:9], v0, s[44:47], 0 offen
s_endpgm
"""
        pipe = G.ComputePipeline(b'', ENTRY_ROOT, 6 << 1, (1, 1, 1), decoded=program(guest), refusals=refusals)
        source = pipe.hlsl_cs()
    assert not refusals, refusals
    binding = 'descword3' if field == 'descword3' else 'stride'
    assert pipe.uds == [('rootconst', 0, 0), (binding, 0)], pipe.manifest()
    assert 'srv 0 buf table 1 0' in pipe.manifest(), pipe.manifest()
    assert sum(line.strip().startswith('scc =') for line in source.splitlines()) == 2, source
    return source


def compile_shader(dxc, directory, name, profile, source):
    hlsl, binary = (directory / (name + suffix) for suffix in ('.hlsl', '.cso'))
    hlsl.write_text(source, encoding='ascii')
    subprocess.run([dxc, '-T', profile, '-E', 'main', '-Fo', str(binary), str(hlsl)], check=True)
    return binary


def verify_readback(command, mismatch=None):
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout + result.stderr, end='')
    if result.returncode == 77:
        sys.exit(77)
    if mismatch is None:
        assert result.returncode == 0, command
    else:
        assert result.returncode == 1 and mismatch in result.stderr, (command, result)


def hex_words(words):
    return ','.join(f'{word:08x}' for word in words)


def compute_case(path, entry, expected, source):
    output = [*expected, *([SENTINEL] * (WORDS - len(expected)))]
    constants = [*entry, *([0] * (16 - len(entry)))]
    contents = [*source, *([0] * (WORDS - len(source)))]
    path.write_bytes(struct.pack('<%dI' % (1 + 2 * WORDS + 16), 1, *output, *constants, *contents))


def check_gpu(dxc, runner, adapter, operations, *, field='descword3', cases=CASES):
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        stages = ('ps', 'cs') if field == 'descword3' else ('cs',)
        if 'ps' in stages:
            vs = compile_shader(dxc, directory, 'fullscreen', 'vs_6_0', FULLSCREEN_VS)
        for kind in stages:
            for op, rhs, reference in operations:
                source = shader(kind, op, rhs, field)
                # Leave the initial compare intact and remove the tested instruction's SCC write.
                lines = source.splitlines(keepends=True)
                del lines[[i for i, line in enumerate(lines) if line.strip().startswith('scc =')][1]]
                good = compile_shader(dxc, directory, 'correct', kind + '_6_0', source)
                bad = compile_shader(dxc, directory, 'stale', kind + '_6_0', ''.join(lines))
                for word in cases:
                    result, scc = reference(word)
                    # Reproduce the runtime ABI: SCC seed and table metadata in b0,
                    # with the same table's buffer contents separately bound at t0.
                    entry = int(scc), word  # the compare sets the opposite of the expected SCC
                    tag = 29 if scc else 17
                    print(f'{kind} {op} {field} input {word:#010x}, SCC {int(scc)}', flush=True)
                    if kind == 'ps':
                        outputs = (result & 0xffff, result >> 16, tag, BOUND_WORD)
                        bits = struct.unpack('<4I', struct.pack('<4f', *outputs))
                        args = ('--startup-entry-words', str(vs), None, hex_words(entry), hex_words(bits),
                                hex_words((BOUND_WORD,)))
                        mismatch = 'startup_entry_words mismatch'
                    else:
                        case = directory / 'input.case'
                        compute_case(case, entry, (result, tag, BOUND_WORD), (BOUND_WORD,))
                        args = ('--compute-store-sized-case', None, str(case))
                        mismatch = 'word=1 expected='
                    for binary, diagnostic in ((good, None), (bad, mismatch)):
                        verify_readback([runner, adapter, *(str(binary) if a is None else a for a in args)],
                                        diagnostic)
