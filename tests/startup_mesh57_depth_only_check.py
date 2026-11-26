"""Slot 57 mesh requests with a depth-only pixel stage translate in tools/startup/gfx2hlsl.py (#2413, T5).

usage: startup_mesh57_depth_only_check.py

A pixel stage without a colour output ends in `exp null off, off, off, off done vm` (LLVM SILateBranchLowering,
Mesa ACO create_fs_null_export). Before an alpha test it narrows EXEC, so the `vm` export kills the lanes it left.
The mesh stage is the accepted fixture of startup_mesh57_skeleton_check.py. The pixel words are built here:
v_interp_mov_f32 v0, p0, attr0.x; v_cmp_gt_f32 vcc, 0.5, v0; s_andn2_b64 exec, exec, vcc; exp null done vm;
s_endpgm. The pair must translate to a pixel stage with no SV_Target that keeps the discard, and a manifest with
an empty `targets` line. Controls: without the EXEC narrowing there is no discard, a null export followed by
a position export is still refused by the role check, and a null export before or after an mrt0, mrtz or param0
export is refused with no source or manifest written.

A colour stage whose null export is only the early exit of a kill (after the main s_endpgm, reached only by
s_cbranch_execz, or by s_cbranch_scc0 directly after s_cselect exec, exec, 0 in the demote form) translates with its
colour target and discard; five near misses of each form stay refused.
"""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import startup_mesh57_skeleton_check as S  # noqa: E402

INTERP, COMPARE, NARROW = 'c8020002', '7c0800f0', '8afe6a7e'
NULL_EXPORT, ENDPGM = ['f8001890', '00000000'], 'bf810000'
ALPHA_TEST = [INTERP, COMPARE, NARROW] + NULL_EXPORT + [ENDPGM]


def translate(pixel):
    with tempfile.TemporaryDirectory() as tmp:
        key, result = S.request(tmp, S.MESH, pixel)
        files = [Path(tmp) / ('%016x.%s' % (key, ext)) for ext in ('ps57.hlsl', 'ms57.txt')]
        return (result,) + tuple(f.read_text() if f.exists() else '' for f in files)


MIXES = 'unsupported: slot 57 pixel stage mixes the null export with another export'


def kill_exit():
    """A colour stage whose null export is only the early exit of a kill translates; near misses stay refused."""
    words = S.KILL_EXIT_PIXEL
    exit_at = len(words) - 3  # the null export's first word; it and s_endpgm follow the main s_endpgm
    S.expect('kill-exit words decode', [x[3] for x in S.G.decode(S.code(words))],
             ['v_interp_mov_f32_e32 v0, p0, attr0.x', 'v_interp_mov_f32_e32 v1, p0, attr0.y',
              'v_interp_mov_f32_e32 v2, p0, attr0.z', 'v_interp_mov_f32_e32 v3, p0, attr0.w',
              'v_cmpx_lt_f32_e32 0.5, v0', 's_cbranch_execz label_0x24', 'exp mrt0 v0, v1, v2, v3 done vm', 's_endpgm',
              'exp null off, off, off, off done vm', 's_endpgm'])
    result, hlsl, manifest = translate(words)
    S.expect('kill-exit colour stage translates', result, 'ok')
    S.expect('kill-exit stage writes its colour', 'SV_Target' in hlsl, True)
    S.expect('kill-exit stage keeps the vm discard', 'if (!(exec & 1)) discard;' in hlsl, True)
    _, _, colour_manifest = translate(S.PIXEL)
    S.expect('kill-exit manifest names the colour target', manifest.splitlines()[-1:], colour_manifest.splitlines()[-1:])
    branch = words.index('bf880003')
    for label, mutant in (
            # The main s_endpgm removed: the colour export falls through into the null export.
            ('a null export the colour export falls into',
             words[:branch] + [S.branch(0xbf88, branch, exit_at - 1)] + words[branch + 1:exit_at - 1] + words[exit_at:]),
            # No branch reaches the block.
            ('an exit block no branch reaches', words[:branch] + ['bf800000'] + words[branch + 1:]),
            # s_branch enters with lanes still alive.
            ('an exit block entered by s_branch', words[:branch] + ['bf820003'] + words[branch + 1:]),
            # The exit export without vm kills nothing.
            ('an exit export without vm', words[:exit_at] + ['f8000890'] + words[exit_at + 1:]),
            # Anything else in the block: v_mov_b32 v0, 0 before the null export.
            ('an exit block with another instruction',
             words[:branch] + ['bf880004'] + words[branch + 1:exit_at] + ['7e000280'] + words[exit_at:])):
        S.expect(label + ' is refused', translate(mutant)[0], MIXES)
    demote_exit()


def demote_exit():
    """The demote form enters the exit with s_cbranch_scc0 right after s_cselect exec, exec, 0; near misses stay refused."""
    words = S.DEMOTE_EXIT_PIXEL
    result, hlsl, manifest = translate(words)
    S.expect('demote-exit colour stage translates', result, 'ok')
    S.expect('demote-exit stage writes its colour', 'SV_Target' in hlsl, True)
    S.expect('demote-exit stage keeps the vm discard', 'if (!(exec & 1)) discard;' in hlsl, True)
    select, branch = words.index('85fe807e'), words.index('bf840004')
    for label, mutant in (
            # SCC still comes from the live mask, but EXEC is not cleared before the branch.
            ('s_cbranch_scc0 without the s_cselect', words[:select] + ['bf800000'] + words[select + 1:]),
            ('an s_cselect that keeps EXEC', words[:select] + ['85fec17e'] + words[select + 1:]),
            ('an s_cselect into another register', words[:select] + ['858e807e'] + words[select + 1:]),
            # An instruction between s_cselect and the branch (s_mov_b32 s20, 0) breaks the adjacency.
            ('an instruction between s_cselect and s_cbranch_scc0',
             words[:branch] + ['be940380', 'bf840004'] + words[branch + 1:]),
            # s_cbranch_scc1 enters the exit when lanes are still live.
            ('s_cbranch_scc1 into the exit', words[:branch] + ['bf850004'] + words[branch + 1:])):
        S.expect(label + ' is refused', translate(mutant)[0], MIXES)


def main():
    os.environ['X2W_FXC'] = ''  # source and manifest only; no compiler needed
    S.expect('alpha-test words decode', [x[3] for x in S.G.decode(S.code(ALPHA_TEST))],
             ['v_interp_mov_f32_e32 v0, p0, attr0.x', 'v_cmp_gt_f32_e32 vcc_lo, 0.5, v0', 's_andn2_b64 exec, exec, vcc',
              'exp null off, off, off, off done vm', 's_endpgm'])
    result, hlsl, manifest = translate(ALPHA_TEST)
    S.expect('depth-only pair translates', result, 'ok')
    S.expect('pixel stage has no colour output', ('SV_Target' in hlsl, 'void main(PSIn i)' in hlsl), (False, True))
    S.expect('pixel stage keeps the vm discard', 'if (!(exec & 1)) discard;' in hlsl, True)
    S.expect('manifest names no target', manifest.splitlines()[-1:], ['targets '])
    S.expect('manifest is not the mesh-only form', 'ps none' in manifest.splitlines(), False)
    # Control: the same stage with all lanes alive exports under the entry EXEC, so nothing is discarded.
    result, hlsl, _ = translate([INTERP] + NULL_EXPORT + [ENDPGM])
    S.expect('null export alone translates', result, 'ok')
    S.expect('no narrowing, no discard', 'discard' in hlsl, False)
    # Control: a null export followed by a position export is not a depth-only pixel stage.
    position = ['f80008cf', '03020100']
    S.expect('null plus position is refused', translate(NULL_EXPORT + position + [ENDPGM])[0],
             'unsupported: slot 57 pixel stage has an incompatible export or termination')
    # Control: the slot 57 profile is the null export alone. A colour, depth or parameter export before or after
    # it is refused before any source or manifest is written.
    null_first = ['f8000090', '00000000']  # exp null off, off, off, off
    for target, before, after in (('mrt0', 'f800000f', 'f800180f'), ('mrtz', 'f8000081', 'f8001881'),
                                  ('param0', 'f800020f', 'f8001a0f')):
        for label, words, order in (
                ('%s + null' % target, [INTERP, before, '00000000'] + NULL_EXPORT + [ENDPGM], [target, 'null']),
                ('null + %s' % target, [INTERP] + null_first + [after, '00000000', ENDPGM], ['null', target])):
            decoded = [x[3] for x in S.G.decode(S.code(words))]
            S.expect(label + ' decodes', ([a.split()[1] for a in decoded[1:3]], decoded[2].endswith('done vm')),
                     (order, True))
            with tempfile.TemporaryDirectory() as tmp:
                key, result = S.request(tmp, S.MESH, words)
                S.expect(label + ' is refused', result,
                         'unsupported: slot 57 pixel stage mixes the null export with another export')
                S.expect(label + ' leaves only the failure', sorted(p.name for p in Path(tmp).iterdir()
                                                                    if not p.name.startswith('req_')),
                         ['%016x.fail' % key])
    kill_exit()
    if S.FAILED:
        return 1
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
