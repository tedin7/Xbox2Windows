# `.fail` diagnostics and read-only `--refusals` summary in `compile_dxil`

[The test fixture](../../tests/native_compile_dxil_refusals_check.py) verifies two diagnostics in
`tools/startup/compile_dxil.cpp` and their ingestion by `tools/startup/triage_run.py` on `cpp-migration`
(part of #2935):

1. **Per-request `.fail` lifecycle in `--requests` batch mode:**
   - Refused graphics (`req_<key>.a`) and slot-67 compute (`req_<key>.c67`) requests atomically publish a
     single-line `<key>.fail` file alongside withdrawing any prior `<key>.txt`, while publishing no stage
     `.cso` bytecode.
   - Re-running `--requests` after fixing a refused request removes the stale `<key>.fail` once `<key>.txt`
     is published.
   - `tools/startup/triage_run.py` classifies native `.fail` entries under allowlisted categories in
     `SHADER_REASONS` without echoing private strings or raw instruction bytes.
2. **Read-only `--refusals REQUEST_DIRECTORY [--jobs N] [--json]`:**
   - Summarizes first-refusal counts across graphics (`req_<key>.a`) and slot-67 compute (`req_<key>.c67`)
     requests without invoking DXC or writing any cache or output files.
   - Strips PC offsets and normalizes hex words, byte offsets, and register indices so repeated failures
     across different instructions or requests aggregate into one row.

3. **Compute programs without a runtime manifest (#3224):**
   - A slot-67 compute program that translates but that `format_compute_runtime_manifest`
     (`src/direct_compute_cache.cpp`) cannot describe gets one fixed reason, for example
     `unsupported startup runtime manifest bindings: SRV kind root_buffer_srv`. The reasons name the
     exit: a zero group dimension, root constants above parameter 15 or more than 8 in one parameter,
     more than 32 user data words, a resource or sampler heap view, a view above root parameter 15,
     non-contiguous SRV or UAV registers, an SRV or UAV kind with no manifest line (the kind, and for
     images the dimension and `array`; a texture SRV of dimension 1D, 2D, 3D or 2D array has a line, a
     cube or multisampled one does not), more than 32 SRVs or 8 UAVs, a comparison sampler, and stride
     words from a view that is not one raw vector SRV. No register numbers, offsets or keys.
   - `--requests` still publishes `<key>.cs.cso` and no `<key>.txt`, still counts the request as
     compiled and still removes `<key>.fail`. It prints the reason in that job's report as
     `no_manifest input=req_<key>.c67 <reason line>`. It writes no `.fail`, because `.fail` is the retry
     marker of both watchers: with `X2W_SKIP_FAILED=1` / `--skip-failed` the Python watcher keeps
     selecting such a key in a live cache.
   - `--watch`, which refuses the key, writes the same line to `<key>.fail` instead of the former
     reasonless one. `--refusals` counts the request under it instead of under no refusal.
     `triage_run.py` reports the `.fail` line as `runtime manifest: <reason>`.

## Verification

```powershell
.\tools\dev.ps1
cmake --preset release
cmake --build --preset release --target compile_dxil native_compute_manifest_reasons
ctest --preset release --no-tests=error -V -R "^(native_compile_dxil_refusals|native_compute_manifest_reasons(_requests)?)$"
.\out\pe-env\Scripts\python.exe -B tests/startup_triage_run_check.py
```

## Limits

- `--refusals` reports the first refusal reached per request by `read_graphics_request` /
  `prepare_direct_graphics(..., /*write_cache=*/false)` and `read_compute_request` / `translate_compute`; it
  does not walk past the first refusal or invoke DXC.
- Graphics requests use `req_<key>.a` and `req_<key>.rtv` ([RTV variant note](native-rtv-variant-requests.md));
  when both markers share a key, the variant takes precedence. Slot-57 mesh requests remain outside the
  native path.
