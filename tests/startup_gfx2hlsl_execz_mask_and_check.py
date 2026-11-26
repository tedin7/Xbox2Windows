"""A live s_and_b64 across EXECZ, checked with synthetic compute dispatches."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_compute_mask_spill_check as spill

C, G = spill.C, spill.G
SOURCE = spill.PROLOGUE + """\
s_mov_b64 s[24:25], exec
s_mov_b64 s[26:27], exec
s_lshl_b32 s22, s7, 2
v_cmpx_gt_u32 s22, v0
s_cbranch_execz Ljoin
s_and_b64 s[24:25], s[24:25], exec
v_xor_b32 v5, 1, v5
Ljoin:
s_mov_b64 exec, s[26:27]
v_xor_b32 v7, 64, v5
v_cndmask_b32_e64 v6, v5, v7, s24
buffer_store_dword v6, v4, s[16:19], 0 offen
s_endpgm
"""
CODE = bytes.fromhex(
    '06038abe80038bbe050308f4000000fa050408f4100000fa7fc08cbf0783148f1400064a82060834001030e004050380703f8cbf'
    '7e0498be7e049abe0782168f1600a87d020088bf187e9887810a0a3a1a04febec00a0e3a060001d5050f6200001070e0'
    '04060480000081bf')
VOTE = 'x2w_exec_entry[x2w_wave_start] == 0u'


def translate(threads=8, wave_size=None, module=G):
    # s20 = group.x * threads. Only the inline shift immediate changes.
    shift = bytes.fromhex('0783148f')
    assert CODE.count(shift) == 1
    code = CODE.replace(shift, bytes((7, 0x80 + threads.bit_length() - 1, 0x14, 0x8f)))
    return module.ComputePipeline(code, C.root_blob(), spill.ENCODING[0], (threads, 1, 1),
                                  wave_size=wave_size).hlsl_cs()


def expected(threads, width):
    words = [C.SENTINEL] * C.WORDS
    for group in range(4):
        for start in range(0, threads, width):
            full_mask = (1 << min(width, threads - start)) - 1
            exec_at_entry = (1 << max(0, min(width, group * 4 - start))) - 1
            mask = (full_mask & exec_at_entry) if exec_at_entry else full_mask
            for lane in range(min(width, threads - start)):
                pos = group * threads + start + lane
                vector_result = C.input_word(pos) ^ ((exec_at_entry >> lane) & 1)
                words[pos] = vector_result ^ (((mask >> lane) & 1) * 64)
    return words


def dispatch(dxc, runner, mutant):
    output = Path(__file__).resolve().parent.parent / 'out'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        src, cso, case = (Path(tmp) / ('execz.' + ext) for ext in ('hlsl', 'cso', 'case'))
        # The 64-thread group tests that a nonempty wave does not update the empty wave next to it.
        for threads, wave_size in ((8, None), (64, 32), (64, 64)):
            hlsl = translate(threads, wave_size)
            assert hlsl.count(VOTE) == 1
            if mutant:
                hlsl = hlsl.replace(VOTE, '(exec & 1u) == 0u')
            src.write_text(hlsl, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            case.write_bytes(struct.pack('<I', 0x010104)
                             + struct.pack('<%dI' % C.WORDS, *expected(threads, wave_size or 32))
                             + struct.pack('<16I', C.KEY, C.STRIDE, *([0] * 14)))
            result = subprocess.run([str(Path(runner).resolve()), 'warp', '--compute-store-case', str(cso), str(case)],
                                    capture_output=True, text=True)
            print(result.stdout, end='')
            if result.returncode == 77:
                return 77
            if mutant:
                mismatch = threads + 4
                assert result.returncode == 1 and f'word={mismatch} expected=' in result.stderr, result.stderr
                print(f'per-lane branch rejected at word {mismatch}')
            else:
                assert result.returncode == 0 and not result.stderr, result.stderr
    return 0


def controls():
    for threads, wave_size in ((8, None), (64, 32), (64, 64)):
        hlsl = translate(threads, wave_size)
        assert hlsl.count(VOTE) == 1 and 's24 = (s24 & exec) & 1;' in hlsl, hlsl
    # A non-mask initializer, a numeric mask read, and an SCC read keep the refusal.
    variants = (('7e0498be', '900498be'), ('c00a0e3a', '18020e7e'), ('c00a0e3a', '80811585'))
    for old, new in variants:
        original = bytes.fromhex(old)
        assert CODE.count(original) == 1
        code = CODE.replace(original, bytes.fromhex(new))
        try:
            G.ComputePipeline(code, C.root_blob(), spill.ENCODING[0], spill.ENCODING[1]).hlsl_cs()
        except G.Unsupported as reason:
            assert str(reason).startswith('s_and_b64 inside a block s_cbranch_execz skips per lane:'), reason
        else:
            raise AssertionError(f'unsafe variant {new} translated')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        controls()
        print('PASS')
    elif len(args) == 2 and args[0] == 'binary':
        assert spill.S.assemble_labels(args[1], SOURCE) == CODE
        print('PASS')
    elif len(args) == 3 and args[0] in ('dispatch', 'mutant'):
        sys.exit(dispatch(*args[1:], mutant=args[0] == 'mutant'))
    else:
        sys.exit('expected no arguments, binary <llvm-mc>, or dispatch|mutant <dxc> <offscreen_check>')
