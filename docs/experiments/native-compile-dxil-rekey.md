# Native cache re-keying (`compile_dxil --rekey`)

Added `--rekey` to `tools/startup/compile_dxil.cpp` as the native C++ equivalent of
`tools/startup/shader_cache_rekey.py` (`Refs #2935`).

## Usage

```powershell
compile_dxil --rekey OLD_CACHE_DIR NEW_REQUEST_DIR [--jobs N]
```

This mode runs without `SDK_DIRECTORY` or DXC:

- Scans `OLD_CACHE_DIR` (non-recursively) for `req_<16-hex>` markers (`.a`, `.c67`,
  `.m57`, `.rtv`), counts `.m86` markers, and ignores result/diagnostic files
  (`.txt`, `.fail`, `.cso`, `.psdep`).
- Validates and recomputes keys in parallel using the current output revision
  strings (`x2w-startup-shader-29`, `x2w-startup-shader-29cs`,
  `x2w-startup-mesh-57-3` via shared helpers in `src/shader_compute_request.cpp`),
  while the readers retain `x2w-startup-shader-28` and `x2w-startup-shader-28cs`
  input compatibility.
- Rewrites `.psin` and `.rtv` payloads under their updated keys while leaving all
  shader bytecode/root/layout parts intact.
- Fails with exit code `1` before creating or writing `NEW_REQUEST_DIR` if any
  request is incomplete, unreadable, or malformed; exits `2` on invalid CLI usage
  or when `OLD_CACHE_DIR` and `NEW_REQUEST_DIR` refer to the same directory.
- Publishes files atomically (ready markers `.a`, `.c67`, `.m57` written last)
  and preserves already-present files in `NEW_REQUEST_DIR`.

## Verification

Build `compile_dxil` and run the native rekey parity and control check:

```powershell
.\tools\dev.ps1
cmake --preset release
cmake --build --preset release --target compile_dxil
ctest --preset release --no-tests=error -V -R '^native_compile_dxil_rekey$'
```

Expected: `native_compile_dxil_rekey` exits `0` and verifies byte-for-byte parity
with `tools/startup/shader_cache_rekey.py`, `gfx2hlsl.pending_keys` compatibility,
idempotency, read-only preservation of `OLD_CACHE_DIR`, and all incomplete-request
and CLI negative controls.
