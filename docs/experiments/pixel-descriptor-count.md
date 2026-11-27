# Pixel shader reads a buffer descriptor count

A translated pixel stage can use hardware buffer descriptor word 2
(`NUM_RECORDS`) as a numeric operand when the table parameter and byte offset
are proven. The graphics draw binder reads the record from the live table and
supplies the count as a draw constant. A missing table, an unreadable record,
a view token, or a non-buffer descriptor at that offset causes the draw to be
refused. The count uses the descriptor's units, which can differ from the size
of the corresponding desktop buffer view.

On Linux x86-64, the synthetic request reported
`numeric use of s54 holding desc` on base revision `82f5cf5c0`. The check below
now accepts the count and retains the refusal for words 0, 1 and 3 and for an
unaligned record offset. Six previously accepted synthetic pipelines produced
identical HLSL and manifests in a comparison with the same base revision.

```sh
python3 -B tests/startup_gfx2hlsl_desc_count_check.py
```

The Windows WARP check binds hardware buffer records with counts 3 and 11,
then compares one RGBA32F pixel per draw with a CPU color calculation. The
control redirects the count binding to the neighboring record, whose count is
one. Both pixel comparisons must detect the wrong binding. Windows execution
is pending.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_desc_count(_warp)?$' -V
```
