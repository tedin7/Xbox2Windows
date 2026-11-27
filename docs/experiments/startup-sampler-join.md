# Two inline samplers at a uniform forward join

For 2D explicit-level `image_sample_l` and `image_sample_lz`, a single forward SCC diamond may select two inline four-word sampler definitions. Both arms must finish by writing all four words as literals, with proven uniform control flow and unchanged EXEC. Only normalized point wrap/clamp XY addressing and point mip filtering are admitted, with validated LOD limits. Two `smpl` static bindings preserve the descriptors; the selected runtime word chooses between complete `SampleLevel` calls. Any intervening overlap write invalidates the join proof. Implicit derivatives, partial definitions and more complex control flow retain their existing refusals.

The source-built fixture evaluates both root-constant selector outcomes for both sample forms with an independent point-texel oracle. A third variant holds word 0 unchanged and varies only MAX_LOD in word 1; the explicit level selects mip 1 or mip 0. Swapping sampler bindings changes the result in every variant. A decoded negative diamond keeps every field valid but sets bit 24 of word 1, above the whole-word-1 mask, so the guard refuses it as unsupported joined sampler words. Seven refusal controls and ten binary assembly variants are included.

Both portable checks passed on Apple Silicon; the expanded ten-variant fixture was assembled with LLVM 15.0.7.

```bash
python3 -B tests/startup_image_sample_l_join_check.py
python3 -B tests/startup_image_sample_l_join_check.py binary llvm-mc
```

The registered Windows draw consumes the actual translated shader, both manifest lines and the runtime selector.

```powershell
cmake --build --preset dev --target offscreen_check
ctest --preset dev --no-tests=error -R '^startup_image_sample_l_join_' -V
```

DXC/WARP/hardware draw execution requires Windows and remains unmeasured locally.
