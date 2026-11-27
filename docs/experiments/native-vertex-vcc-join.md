# Per-vertex VCC across a vertex-stage join in the direct DXIL frontend (#2906)

[The fixture](../../tests/native_graphics_vertex_vcc_join_check.py) draws two source-built gfx1013 vertex
stages through the direct DXIL frontend and compares the readback with a per-vertex CPU model.

## The two programs

1. A root word that the harness supplies at draw time picks the arm, so the branch is uniform but not known
   to the compiler. Each arm runs its own `SV_VertexID` compare into VCC and writes its own constant. After
   the join one CNDMASK selects that constant or 0.5 per vertex. Both outcomes are drawn. The emitted IR
   joins the two compares in a `phi` that feeds the select, and the test requires that.
2. One compare writes VCC and also splits the vertices between a THEN and an ELSE arm under saved EXEC.
   After EXEC is restored a CNDMASK on the same VCC keeps the arm's constant or its negation, so one draw
   holds both VCC values and both arm values. The frontend lowers these guest arms to per-vertex selects in
   one block, so this program has a guest join and no IR join.

## Readback and controls

A desktop pixel stage reads the three vertex values with `GetAttributeAtVertex`, so each of the 8 pixels
reports every vertex without interpolation. The comparison is bit-exact.

Each control redraws a changed program against the unchanged reference and must differ at all 8 pixels:

- arm B of the first program loses its compare, so that edge carries the stale mask of an earlier compare.
  Arm A still matches with the same change;
- a uniform `v0 == v0` compare overwrites the joined mask, in both arms of the first program and in the
  second.

## Command and result

```powershell
cmake --build --preset dev --target compile_dxil offscreen_check
ctest --preset dev --no-tests=error -V -R '^native_graphics_vertex_vcc_join_(warp|hardware)$'
```

On Windows 11 build 26300, Debug build, both tests passed with the debug layer enabled and no debug errors
or warnings: Microsoft Basic Render Driver (driver 10.0.26100.9818) and NVIDIA GeForce RTX 4090
(driver 32.0.16.1088). Neither reported a capability skip.

## Limits

- Branches that need a whole-wave VCC vote are not drawn. Neither is a compare inside a per-vertex masked
  arm, whose VCC on the other lanes depends on the rest of the guest wave.
- An adapter without per-vertex attribute support fails pipeline creation instead of reporting a skip. That
  path has not been observed.
- The pixel stage is desktop HLSL. A translated pixel stage linked to this vertex stage is not covered.
