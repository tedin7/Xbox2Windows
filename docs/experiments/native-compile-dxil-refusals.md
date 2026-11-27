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

## Verification

```powershell
.\tools\dev.ps1
cmake --preset release
cmake --build --preset release --target compile_dxil
ctest --preset release --no-tests=error -V -R "^native_compile_dxil_refusals$"
.\out\pe-env\Scripts\python.exe -B tests/startup_triage_run_check.py
```

## Limits

- `--refusals` reports the first refusal reached per request by `read_graphics_request` /
  `prepare_direct_graphics(..., /*write_cache=*/false)` and `read_compute_request` / `translate_compute`; it
  does not walk past the first refusal or invoke DXC.
- Graphics requests use `req_<key>.a` and `req_<key>.rtv` ([RTV variant note](native-rtv-variant-requests.md));
  when both markers share a key, the variant takes precedence. Slot-57 mesh requests remain outside the
  native path.
