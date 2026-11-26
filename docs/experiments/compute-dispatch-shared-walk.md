# Shared descriptor resolution for translated compute

Set `X2W_DISPATCH_WALK_SHARED=1` to resolve cached dispatch bindings under
the shared draw lock. The runtime serializes first-use translation and texture
materialization, upload reservations, and dispatch recording. The dispatch
phases report includes average exclusive hold in microseconds per call when
the switch is enabled.

Run the WARP check on the Windows test machine.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_dispatch_input_warp$'
```

The check compares all 16,000 output words with a CPU reference for serial
recording and for four concurrent compute lists with the switch off and on.
Each list records 500 dispatches. The concurrent runs must each report 2,000
translations with zero refusals. The enabled run must report less than 30%
of the disabled run's exclusive hold per dispatch.

The lifetime case enables the debug layer. A second thread releases a bound
input after the recording thread resolves it and before that thread
reacquires the exclusive lock. The first dispatch must produce eight reference
words and the second must refuse, leaving its eight output words untouched.
Enable full page heap for `startup_stream_consumer.exe` on the test machine if
available and repeat the same check.

For the mutant, remove the post-acquisition loop that checks each pinned
wrapper's reference count and backing in `record_translated_dispatch` in a
local test build. Keep the references and the release hook intact. The same
CTest must fail the lifetime case because the second dispatch is counted as
translated. Restore the loop for the normal build.
