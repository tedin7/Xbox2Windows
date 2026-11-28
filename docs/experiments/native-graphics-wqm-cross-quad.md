# Cross-quad WQM/EXECZ in a native pixel draw (#2907)

[The fixture](../../tests/native_graphics_wqm_check.py) draws source-built gfx1013 pixel stages through the
direct DXIL frontend. In each one, an `s_cbranch_execz` under WQM decides a scalar that every surviving pixel
exports. The branch is skipped only when EXEC is zero in the whole guest wave. A quad with no EXEC bit at the
branch therefore still sees whether another quad of its wave kept the wave from skipping.

## The accepted form

```
s_mov sE, exec ; s_wqm exec, exec
v_cmp_* vcc, ... ; s_andn2 sE, sE, vcc          one or more coverage filters
v_cmp_* vcc, ... ; s_andn2 sM, sE|exec, vcc     the branch mask, from the survivors or the whole-quad EXEC
v_cmp_* vcc, ... ; s_andn2 sM, sM, vcc          optional further terms
s_wqm exec, sM ; s_cbranch_execz J              the region writes only scalars
J: s_mov exec, sE                               then a terminal colour export
```

Following the maintainers' decision on the issue, the branch condition is `WaveActiveAnyTrue` of the
quad-complete EXEC bit:
- the modelled mask predicate is OR-ed with its `QuadReadAcrossX`, `QuadReadAcrossY` and
  `QuadReadAcrossDiagonal`;
- quad reads see helper lanes, so a quad whose EXEC bits sit only on helpers or on covered pixels that did
  not survive still votes, as it does on the guest;
- all of this uses Shader Model 6.0 operations, with no `[WaveOpsIncludeHelperLanes]`.

The calls are emitted before the first conditional branch, where the whole quad still runs. The IR test checks
this, and checks that their declarations carry the attributes DXC gives the HLSL intrinsics.

These still keep a refusal by name:
- inside the region: a vector or DPP instruction, an EXEC or VCC read, a hardware register access
  (`s_setreg`/`s_getreg`), a branch, or an EXEC write other than the restore;
- a restore that is not the branch target;
- a second branch;
- anything after the restore other than lane-local ALU and colour exports.

`raster_program(2.5, cross=True)` keeps its existing refusal.

## Contract and what this shows

The lowering decides the branch once per **desktop** wave. It relies on the packing contract of #2835 and
#2854: a correct program cannot depend on which quads the console puts in one wave. A program can only see the
branch outcome through an effect outside EXEC, here an SGPR. This fixture builds such a dependency on purpose,
so its reference comes from the measured desktop layout.

**The draws show that the lowering votes over the desktop wave. They do not show equivalence with the console
for an output that depends on another quad.**

## Measured layout

Two HLSL pixel stages on the same draw report, for every covered pixel:
- its wave: the smallest covered id in it;
- the ids of its three quad peers;
- whether each peer is a helper (`IsHelperLane`, SM 6.6).

Wave operations leave helpers out, so the helper flags come from quad reads. The fixture checks that every
desktop quad is the aligned 2x2 block, and that all covered lanes of a quad report the same quad. Quads are
per primitive. On WARP the two middle quads of the 8x2 target are each shaded twice, once with each row covered
and the other row as helpers, as if the full-screen triangle were drawn as two primitives. The layout is
measured in separate draws with the same vertex stage and target. The reference follows the measured quads.

| Adapter | Lanes | Full draw | Diagonal draw |
|---|---|---|---|
| NVIDIA GeForce RTX 3070, driver 32.0.16.2101 | 32 | 4 quads in 1 wave, no helpers | 3 quads in 1 wave, 3 helpers |
| Microsoft Basic Render Driver, 10.0.26100.9818 | 4 | 6 quads in 6 waves, 8 helpers | 5 quads in 5 waves, 11 helpers |

## Cases and controls

| Draw | Mask | Shows |
|---|---|---|
| full | survivors with x <= 1 (14 survivors) | another quad's survivors keep the wave from skipping |
| full | survivors with x <= 4.5 (all 16) | as above, one quad without bits |
| full | x <= -1 | empty wave: the branch is taken, every vote agrees |
| full | whole-quad EXEC with x >= 4.5, survivors x <= 3.5 | bits only on covered pixels that did not survive |
| full | survivors with x >= 4.5 | no bits: WQM never makes a zero mask non-zero |
| diagonal | whole-quad EXEC, x >= 5, y >= 1 | the bit only on the diagonal helper of (4,0) |
| diagonal | whole-quad EXEC, x >= 5, y <= 1 | the bit only on the horizontal helper |
| diagonal | whole-quad EXEC, x = 4..5, y >= 1 | the bit only on the vertical helper |

Each case is drawn against the measured-layout reference. Each control redraws the same program against a
different reference and must differ:
- a quad-local vote;
- a wave vote without the quad OR;
- a vote over surviving pixels only.

Pixels outside exact coverage keep the clear value -1.0.

## Command and result

```powershell
cmake --build --preset dev --target compile_dxil offscreen_check native_store_replay native_pixel_ir_dump
ctest --preset dev --no-tests=error -V -R '^native_graphics_wqm_(hardware|warp|ir)$'
```

See the pull request for the measured counts.
- **On the RTX 3070** every control differs.
- **On WARP** (one quad per wave), the quad-local control cannot differ, and the test prints
  `SKIPPED cross-quad quad-local control`. The three helper cases still differ from the no-OR and survivor
  votes, at pixel (4,0).
- On a physical adapter with no wave holding two quads, the hardware entry exits 77 after the other checks.

## Limits

- **Helper lanes after the vote:** wave intrinsics are undefined on helper lanes, so the branch may diverge for
  them. That is safe only while nothing after the branch reads another lane.
- **Adapters:** a 64-lane adapter has not been run. The RX 9070 XT is blocked by #2915. AMD and Intel have
  not been run.
- **Programs:** only the form above is accepted. Real programs and captures have not been run.
- **Clipping:** a console that clips or splits primitives differently from the desktop forms different quads.
  This holds for the existing quad-local path too.
