"""Plain image_sample keeps its implicit LOD past a uniform join in tools/startup/gfx2hlsl.py.

A plain image_sample takes its LOD from the coordinate derivatives of the pixel quad. After a block boundary the
translation lowered it to SampleLevel(.., 0), so the guest's mip was lost. Where the rule image_sample_b and
image_get_lod already use holds (pixel stage, entry EXEC mask, wave-uniform control flow, no divergent back edge over
the call) it is now the implicit-LOD Sample; everywhere else it stays level 0 (no new refusal).

Fixture (synthetic): the recorded c_lz stage of startup_image_sample_c_check with the MIMG opcode set to 0x20
(image_sample, a 2-register address), the same edits as startup_gfx2hlsl_image_get_lod_check.py:
  straight    nothing before the sample                        -> Sample (as before)
  branch      s_cbranch_scc0 to the next instruction            -> Sample
  join        a uniform forward join (scc), entry EXEC          -> Sample
  divergent   s_mov_b32 exec_lo, 0 before the sample            -> SampleLevel 0 (as before)
  cf_<op>     s_cbranch_vccz / vccnz / execz / execnz before it -> SampleLevel 0 (as before)
  back_<op>   a divergent back edge over the sample             -> SampleLevel 0 (as before)
  exit_<op>   a divergent forward exit, loop closed by s_branch -> SampleLevel 0 (as before)

  (no arguments)   translation controls, pure Python
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_image_sample_c_check as C  # noqa: E402
from startup_gfx2hlsl_image_get_lod_check import AT, MIP_POINT, W2_AT, sopp  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

IMAGE_SAMPLE = 0x20


def stage(branch=False, divergent=False, join=False, cf_op=None, back_op=None, exit_op=None):
    w = bytearray(C.WORDS['lz'])
    w[W2_AT:W2_AT + 4] = struct.pack('<II', 0xbe9a03ff, MIP_POINT)  # s_mov_b32 s26, literal (mip filter point)
    w0, w1 = struct.unpack_from('<II', w, AT)
    w0 = w0 & ~(0x7f << 18) & ~(0xf << 8) & ~(7 << 3) | IMAGE_SAMPLE << 18 | 1 << 8 | 1 << 3
    struct.pack_into('<II', w, AT, w0, w1)
    code = bytes(w)
    if back_op is not None:  # a backward branch right after the sample, to it
        code = code[:AT + 8] + sopp(back_op, -3) + code[AT + 8:]
    if exit_op is not None:  # a forward exit right after the sample, then an unconditional branch back to it
        code = code[:AT + 8] + sopp(exit_op, 1) + sopp(2, -4) + code[AT + 8:]
    insert = b''
    if branch:
        insert += sopp(4, 0)
    if join:
        insert += bytes.fromhex('800380be') + bytes.fromhex('008006bf') + sopp(5, 1) + bytes.fromhex('810381be')
    if divergent:
        insert += bytes.fromhex('8003febe')
    if cf_op is not None:
        insert += bytes.fromhex('800380be') + sopp(cf_op, 1) + bytes.fromhex('810381be')
    return code[:AT] + insert + code[AT:] if insert else code


def calls(**kwargs):
    text = G.Pipeline(VS_WORDS, stage(**kwargs), root_blob(flags=0)).hlsl_ps()
    return [l.strip() for l in text.splitlines() if '.Sample' in l and 'SamplerState' not in l]


def main():
    for name, kwargs in (('straight', {}), ('branch', {'branch': True}), ('join', {'join': True})):
        got = calls(**kwargs)
        assert len(got) == 1 and '.Sample(' in got[0] and 'SampleLevel' not in got[0], (name, got)
    cases = [('divergent', {'divergent': True})]
    cases += [('cf_%d' % op, {'cf_op': op}) for op in (6, 7, 8, 9)]   # vccz, vccnz, execz, execnz
    cases += [('back_%d' % op, {'back_op': op}) for op in (7, 9)]      # vccnz, execnz back edge over the sample
    cases += [('exit_%d' % op, {'exit_op': op}) for op in (7, 9)]      # a divergent forward exit of a loop closed by s_branch
    for name, kwargs in cases:
        got = calls(**kwargs)
        assert len(got) == 1 and 'SampleLevel(' in got[0] and got[0].rstrip(';').endswith(', 0)'), (name, got)
    print('plain image_sample: implicit LOD past a uniform branch/join, level 0 with divergent EXEC or control flow')


if __name__ == '__main__':
    main()
