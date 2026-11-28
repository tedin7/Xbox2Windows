# Missing graphics shader fallback

With `X2W_DRAW_MISSING_SHADER=1`, a direct triangle-list draw can use a
diagnostic PSO while its translation manifest is absent. It uses the bound
attachment formats and clips all geometry without writing colour, depth or
UAV data. Downstream passes still depend on the missing shader's actual
outputs. The runtime publishes and polls the normal translation request.

On Windows, build the fixture and run the two WARP cases from the repo root.

```powershell
cmake --build --preset dev --target startup_indirect_draw_indexed
ctest --preset dev --no-tests=error -R '^startup_missing_shader_fallback(_off)?_warp$' -V
```

The fixture checks pipeline statistics and reads back an RTV and DSV. The
enabled case expects three IA vertices with both attachments still at their
clear values. The disabled control expects zero IA vertices and the same
clear values. Both then publish the source-built translation and expect the
same guest pipeline to produce a red render target after the manifest retry.
