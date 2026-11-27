# Translated PSO recipes

The shader cache stores the translated vertex and pixel bytecode, the root signature and the
sampler descriptions. It does not store the draw state that produced a desktop graphics pipeline
state object: the blend state, depth-stencil state, rasterizer state, target formats and DSV format.
A run therefore creates each translated PSO on its first draw.

`X2W_PSO_RECIPE_DIR=<dir>` records that state after a successful
`CreateGraphicsPipelineState` and replays matching records before creating a new PSO on a later
run. It is off by default.

## Format

One text file per translation key: `<dir>/pso_<translation-key>.txt`. The file holds up to 64
records and 64 KiB. Each record starts with `x2w-pso-recipe 1` and ends with `end`. It carries the
translation key, the target/depth/raster state key, the target count and formats, the DSV format,
the `indexed_rect` flag, the effective blend state after export write masks, the depth-stencil
state, the rasterizer state, the sample mask and count, and the topology type. No pointers or
process addresses are written.

The parser recomputes the state key from the recorded fields. A record whose recomputed key does
not match the key it names, whose target formats or DSV do not match the current pass, or whose
version is unknown is ignored and the unchanged creation path runs. A malformed or oversized file
is counted and ignored.

The writer applies the same parser before it appends. It refuses to add a record when the file
already holds 64 complete records, when any existing line is rejected, or when the new record
would cross 64 KiB. A 65th record would never be read back, so refusing it keeps the cache and the
parser at the same boundary. It also refuses when a non-empty file does not end in LF: the parser
accepts such a final record, but appending would join its `end` to the next header on one line. The
file is left unchanged.

The read, count/size check and append share one exclusive file handle. A second process that
reaches the append path while the file is owned keeps the PSO it already created and skips the
cache write; a reader denied by that handle also falls back to the unchanged creation path.
Neither side can seek to an end offset that another writer selected.

## Scope and limits

This is state reconstruction, not a D3D12 pipeline library. A replayed record still calls
`CreateGraphicsPipelineState`; the driver may compile or find the PSO in its own cache. The change
moves the state reconstruction and the first-use creation earlier, and makes the exact draw state
available without a title run. It does not claim to eliminate driver compilation.

The device-free test `startup_pso_recipe` covers the serializer/parser round trip, two appended
records, a corrupt state key, an unknown version, an oversized file, target-format and DSV
mismatches, and the 63/64/65 record boundary (63 accepts one more, 64 refuses a 65th, and the
parser rejects a 65th). `startup_stream_draw_warp` runs the two-process replay: the first
translated stream draw writes a recipe, a second process with the same directory loads and
precreates it, and all 32 RGBA bytes match. It also holds the recipe file exclusively during a
draw and checks that the draw still renders with exact pixels while the file size stays unchanged.
It then seeds 63 valid records for another translation key, checks that the next draw appends the
matching record as number 64 and the following process logs `precreated 1`, and seeds 64 records to
check that the draw still renders while the file stays at 64. It removes the final LF from that
parseable file, runs another draw and checks that the file is byte-for-byte unchanged and still
readable. That test covers the unused depth-stencil normalisation as well.

```powershell
ctest --preset dev --no-tests=error -R '^startup_pso_recipe$' -V
ctest --preset dev --no-tests=error -R '^startup_stream_draw_warp$' -V
```
