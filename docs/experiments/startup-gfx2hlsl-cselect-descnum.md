# Scalar select on descriptor-derived data in gfx2hlsl compute stages (#2618, the C500 rows)

In 250 compute rows of the C500-01 and C500-02 batches, the recorded refusal is `numeric use of sX holding descnum`
at `s_cselect_b32 sX, 0, sX`. Before this change, `Stage.scalar_select` in `tools/startup/gfx2hlsl.py` read both
sources as run-time values. A number computed from a buffer descriptor (`descnum`) has none, so the select refused.

## Contract

A select between descriptor data and zero can disable a buffer by zeroing its size. On the console such a buffer
returns zeros. A lowering that keeps only the descriptor arm would read real data instead. So in a compute stage,
`s_cselect_b32 sD, a, b` into an SGPR, where one source holds `descnum` and each other source is a constant, emits
no HLSL. sD holds `('descnum', 'cselect', offset, arm a, arm b)`: both arms and the offset of the select, whose
SCC picks one.

Nothing reads that symbol as a value:

- A numeric read still refuses with `numeric use of sD holding descnum`. This covers VALU and SALU sources, compares,
  m0, store data and lane writes.
- `s_mov_b32` and `s_mov_b64` copy the symbol. Any other scalar ALU instruction with it in any register of a source
  (the high word of a 64-bit pair too) refuses with `<instruction> on a select of descriptor data`
  (`Stage.refuse_select_source`). It would otherwise become a plain `descnum` or an in-place edit of a descriptor
  word, which loses the arm. This also covers the structured-to-raw conversion when its stride comes from such a
  select.
- An access whose descriptor operand has the symbol in any word refuses with
  `descriptor operand s[N:M]: sK holds a select of descriptor data` (`Stage.select_word`, called by
  `Stage.descriptor`). On main b505c32b a buffer load ignores edits to words 1–3 of its descriptor (#2901), so
  without this check a copied select would reach the load with the zero arm dropped.
- In compute, an access whose descriptor operand holds a word joined at a branch target after its first word
  refuses with `descriptor operand s[N:M]: sK holds a word joined at a branch target` (`Stage.joined_word`). A join
  of different symbols leaves only that mark, so a select that reaches a descriptor word on one path, or through a
  loop back edge, would otherwise be dropped. This also refuses compute joins that bring two different non-select
  words there. Main translates those, ignoring the difference, as in #2901.
- The select still refuses with `scc from descriptor data in s_cselect_b32` when its SCC came from an ALU
  instruction on descriptor data, because that SCC is not emitted.

Unchanged: `s_cselect_b64`, an m0, exec or vcc destination, a raw descriptor word or a pointer as a source, a
run-time register next to the `descnum` source, and every non-compute stage.

A row only gets past the select if sX is dead afterwards or is only copied. Where sX is read later, the refusal moves
to that read. The packets record only the first refusal, so how often each case occurs is not known yet.

## Checks

`tests/startup_gfx2hlsl_cselect_descnum_check.py` splices synthetic instructions into case A of
`tests/startup_gfx2hlsl_compute_check.py`. The decoder must print the expected text for each one.

- Accepted: the select with its result dead, in both source orders, with a literal source and on an extracted
  stride. The HLSL may differ from case A only by the compare's `scc` line, and sD must hold both arms and the offset.
- Refused, each checked by its reason: every later use listed above, a copy into word 1, 2 or 3 of the load
  descriptor (also through a second copy), a copy into the store descriptor, and SCC from descriptor data.
- Branches: the select writes no HLSL variable, so a join must not turn sD back into a run-time register. A read
  after a forward join where one path skips the select, and a read at a loop head reached by a back edge after the
  select, both refuse with `numeric use of s21 holding joined at a branch target`. The same forward join with a
  run-time write in place of the select translates.
- Descriptor words through a join, from the #2903 review: a load after a join where one path selects into word 2
  of its descriptor, and the same select at the bottom of a loop whose head is the load, refuse as joined words.
  The same join with the loaded word on both paths, and the same loop without the select, translate. A select in
  the high word of `s_or_b64 s[14:15], s[20:21], 0` refuses.
- Only buffer load and store operands are exercised for the descriptor-operand refusal. Image, sampler and
  `s_buffer_load` operands go through the same `Stage.descriptor` check but have no control here.
- Still refused as before: the unchanged forms above, in the compute stage and in the vertex and pixel probe stages.
- Negative controls, each a mutated translator that must fail a named check:
  - `zero-arm-dropped`: the select keeps only its `descnum` source. The structured-to-raw conversion with the
    selected stride then translates.
  - `load-check-removed`: `select_word` finds nothing. A load through a copied select then translates.
  - `alu-check-removed`: `refuse_select_source` accepts everything. An ALU copy into word 2 then reaches the load.
  - `join-check-removed`: `joined_word` finds nothing. The load after the join, and the load at the loop head, then
    translate.
  - `high-word-unchecked`: `refuse_select_source` sees only the low word of each source. The `s_or_b64` case then
    translates.

```powershell
python -B tests/startup_gfx2hlsl_cselect_descnum_check.py
ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_cselect_descnum_(controls|warp)$" -V
```

The `warp` and `hardware` tests translate case A with the dead select, compile it with DXC `cs_6_0`, dispatch it
and compare all 320 words with case A's reference.

Measured on Windows 11 build 26300 with Python 3.14 at the commit that adds this note (base b505c32b): the controls
pass (41 first-pass checks, and each of the five controls fails its named check). On b505c32b the same file fails
20 of its 34 first-pass checks. The four accepted cases are refused at the select, and 16 later-use, branch and
review checks get that refusal instead of their own reason. The mutation phase cannot run there, because the helpers
don't exist yet.

Still a hypothesis: which later use of sX matters in the C500 rows. The packets don't show the instructions that
produce sX, write SCC or read sX afterwards.
