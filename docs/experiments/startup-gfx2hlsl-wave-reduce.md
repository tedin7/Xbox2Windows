# Whole-wave reductions in gfx2hlsl pixel stages (#2125 item D)

The mirror steps of a reduction read lanes outside the pixel quad, which the one-lane model cannot prove, so
`tools/startup/gfx2hlsl.py` recognises the whole wave32 chain instead of its instructions, in the three shapes the
two censuses in [#2125](https://github.com/R0Y00/Xbox2Windows/issues/2125) report for every wave32 chain.

## The shapes

Each chain is `s_or_saveexec_b32 sS, -1`, the identity fill `v_cndmask_b32_e64 vT, <identity>, vX, sS` (the input
where the saved mask has the lane, the identity elsewhere), the DPP16 steps `quad_perm:[1,0,3,2]`,
`quad_perm:[2,3,0,1]`, `row_half_mirror` and `row_mirror` on `vT, vT, vT` (full row and bank masks, `fi:1`),
`v_permlanex16_b32 vP, vT, 0, 0`, the combine of `vT` and `vP`, `s_mov_b32 exec_lo, sS` and
`v_readlane_b32 sD, vT, 31`. The operations are `v_or_b32`, `v_min_u32` and `v_max_u32`, with identities `0`, `-1`
and `0`.

- The or and max chains read lane 31 right after the restore.
- The min chain reads it later: after its restore come one `v_cndmask_b32_sdwa`, the `s_or_saveexec_b32 sS2, -1` of
  a max chain, and only then its readlane, which the max chain carries before its own fill. Readlane ignores EXEC, so
  that readlane still sees the min result as long as nothing in between writes `vT` (refused if it does) and the
  readlane does not overwrite `sS2` (refused too). The `v_cndmask_b32_sdwa` keeps its own handler under the restored
  EXEC.
- A readlane right after the restore is also accepted for a min chain alone.

With every lane enabled each lane ends with the operation over the lanes that were active, the others adding the
identity (so `fi` and `bound_ctrl` have nothing to decide), and `vT` and `sD` become one `WaveActiveBitOr`,
`WaveActiveMin` or `WaveActiveMax` of `vX`, emitted at the restore, with `sD` copied from `vT` at the readlane.

## Contract and helper lanes

The console packs pixels into waves in its own way, so a correct program cannot depend on which pixels share a wave;
the contract is the active lanes of the desktop wave. The chain is accepted only in a straight-line pixel stage whose
EXEC is still the entry mask and was not widened by `s_wqm`, so on the console the helper lanes hold the identity,
and on the desktop the wave operations leave them out (the draws check it).

## Refusals

The fill is one table, `Stage.WAVE_REDUCE_FILLS`, holding that form only. Everything else that starts like a chain
(an `s_or_saveexec` with -1 followed within 16 instructions by a mirror step or `v_permlanex16_b32`) is refused with
its reason:

- wave64; another fill (another constant, another saved mask, a copy, none); a wrong identity;
- missing or reordered steps, another DPP control, partial row or bank masks, other or mixed operations;
- lane selects other than 0, 0 or `op_sel` on the permlane; a readlane lane other than 31 or before the restore; EXEC
  not restored from `sS`;
- for the later readlane: another instruction in place of the `v_cndmask_b32_sdwa`, `vT` rewritten before it, a
  missing readlane, a readlane that overwrites the max chain's saved EXEC, a later readlane in an or or max chain, a
  chain other than max carrying it, an `s_or_saveexec_b32` after it that starts no chain;
- a branch target inside the chain, a program with branches, a vertex stage, `s_wqm` before the chain, and `vP`
  (which holds the other row) read after the restore before a VALU rewrites it.

Eight of these are conservative: reordered steps, other permlane selects, `op_sel`, lane 30, the readlane before the
restore, another instruction in place of the `v_cndmask_b32_sdwa`, a later readlane in an or or max chain (the
fixture uses max), and an or chain carrying it. The lane model shows they would still reduce; they are refused
because the censuses do not show them.

## Checks

`startup_gfx2hlsl_wave_reduce_controls` checks that the fill table holds the census form only. The fixtures are the
three shapes: or, min and max alone, min then max interleaved with `vT` shared by the two chains and with a `vT` of
its own, min then max each read at once, and an or followed by the interleaved pair. The source chains run on a
32-lane model written from the ISA text (DPP16 sources with `fi`, the permlane row exchange, readlane, the SDWA
select), not from the lowering: for random EXEC masks the readlane result is the reduction of the active lanes, and
each refused mutation either is not (wrong identity, no fill, the input copied without the identity, missing step,
`row_shr:1`, mixed operations, partial row mask, missing later readlane, `vT` rewritten before it, the max chain's
saved EXEC overwritten) or is one of the conservative refusals. A planted error on the model, the min readlane moved
after the max fill with `vT` shared, must not be the min. The controls check the translation of every fixture, with
one wave operation per chain in program order and the `v_cndmask_b32_sdwa` between them, four accepted variants
(`bound_ctrl`, no `fi`, the combine sources the other way round, a VOP3 combine) and every refusal by name. The same
controls run in the Linux `python-linux` job (`tests/checks/python-linux/startup_gfx2hlsl_wave_reduce.txt`), and the
`binary` mode in the `pixel-binary` job (`tests/checks/pixel-binary/startup_gfx2hlsl_wave_reduce.txt`).

`startup_gfx2hlsl_wave_reduce_{warp,hardware}` draw a triangle over 32 x 32 pixels for each operation and for the
interleaved pair; the input is `1 << x` for or and a different word per pixel otherwise, and the pair also writes the
`v_cndmask_b32_sdwa` result. The desktop packing cannot be known in advance, so the test replaces the fourth channel
of the translated stage with the lane index, the lane count and `WaveReadLaneFirst` of the pixel id, reads the target
back through `offscreen_check --startup-pixel-dump`, groups the covered pixels by wave and compares each result with
the operation over the inputs of the covered pixels of its wave. The triangle leaves 30 quads partly covered. Planted
errors in the translated text must fail: another operation, signed for unsigned, the lane's own value and the first
lane's value for each operation; for the pair, the min readlane lowered after the max, min and max swapped, and the
`v_cndmask_b32_sdwa` dropped.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_wave_reduce' -V
python -B tests/startup_gfx2hlsl_wave_reduce_check.py binary <llvm-mc>
```

The `binary` mode assembles every fixture and compares the words; the programs with a branch are assembled to an
object (`-filetype=obj`) and compared with its `.text`, because `-show-encoding` prints a branch to a label with its
offset still unresolved. WARP runs 4-lane waves (one quad per wave), so only
a hardware adapter reduces across quads. This does not measure console behavior or a title, and wave64 chains are
the next slice.
