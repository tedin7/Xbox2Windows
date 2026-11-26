# First active lane in gfx2hlsl pixel stages (#2618, the F rows)

`s_ff1_i32_b32 sD, exec_lo` gives the index of the lowest lane in EXEC, and `v_readlane_b32 sX, vY, sD` copies `vY`
from that lane whatever EXEC says (RDNA 2 ISA, S_FF1_I32_B32 and V_READLANE_B32; in wave32 the lane select uses bits
4:0). Together they read `vY` from the first active lane. `tools/startup/gfx2hlsl.py` folded the `s_ff1` only at
compute dispatch entry (#2389), and refused it in a pixel stage with "s_ff1_i32_b32 of exec_lo outside compute".

## Lowering and contract

The contract is the one of the wave reductions ([the note](startup-gfx2hlsl-wave-reduce.md)): the console packs
pixels into waves in its own way, so a correct program cannot depend on which pixel is first. The pair becomes
`WaveReadLaneFirst(vY)` over the active lanes of the desktop wave. The lane number itself has no desktop meaning, so
`sD` is a symbol with no run-time value. Each read of it up to its next write must be the lane select of a
`v_readlane_b32`, and every other read is refused, for example a shift, a copy to another SGPR or a move to a VGPR.

It is accepted only where the one-lane model knows the whole wave's EXEC:

- a straight-line pixel stage (no branches);
- EXEC still the entry mask at the `s_ff1` and at each readlane, with no `s_wqm` before them. On the console the
  helper lanes are then out of EXEC, and on the desktop wave operations leave them out;
- a readlane source that is an ordinary VGPR, not a `v_writelane_b32` spill slot;
- no `exec_hi` or 64-bit `exec` operand in the program. The decoder prints wave32 syntax, and in wave64 `exec_lo` is
  half the wave: an empty low half gives -1, which a readlane reads as lane 63.

An EXEC that is changed and then restored before the readlane is refused although it is the entry mask again, because
the one-lane model no longer knows that. That refusal is conservative.

## Checks

`startup_gfx2hlsl_pixel_first_lane_controls` runs the source pair on a 32-lane model written from the ISA text, for
500 random EXEC masks: the readlane result is the input of the lowest active lane. The controls cover three cases
that translate:

- the pair;
- two reads of one first lane;
- a lane number overwritten after its readlane and then read as a number.

They also check every refusal by its reason, and that vertex and mesh stages keep the "outside compute" refusal. The
controls run in the Linux `python-linux` job (`tests/checks/python-linux/startup_gfx2hlsl_pixel_first_lane.txt`), and
the `binary` mode in the `pixel-binary` job (`tests/checks/pixel-binary/startup_gfx2hlsl_pixel_first_lane.txt`).

`startup_gfx2hlsl_pixel_first_lane_{warp,hardware}` draw the triangle of the wave reduction test over 32 x 32 pixels.
The test replaces the fourth channel of the translated stage with the lane index, the lane count and
`WaveActiveMin` of the pixel id. That names the wave without using the intrinsic under test. The test then reads the
target back through `offscreen_check --startup-pixel-dump`. Each covered pixel is compared with the input, or the
packed position, of the covered pixel with the lowest lane index in its wave. Planted errors in the translated text
must fail:

- the lane's own value;
- the largest value of the wave;
- the first lane of the quad, which is a helper lane where the quad is partly covered;
- the lane after the first.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_pixel_first_lane' -V
python -B tests/startup_gfx2hlsl_pixel_first_lane_check.py binary <llvm-mc>
```

WARP runs 4-lane waves (one quad per wave), so on WARP the reference only separates covered lanes from helper lanes
within a quad. Only a hardware adapter picks a first lane across quads. This does not measure console behavior or a
title. Whether the F rows of #2618 stop at this pair needs a replay against the captured requests.
