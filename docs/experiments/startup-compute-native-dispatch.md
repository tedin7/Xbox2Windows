# In-process compute translation in the stand-in (#2934)

[The fixture](../../tests/startup_compute_dispatch_native_check.py) dispatches a source-built gfx1013 compute
program through the stand-in's device slot 67 and Dispatch path with `X2W_NATIVE_SHADERS` set, an empty shader
cache and no offline translator, and compares all 320 output words with a CPU reference.

## Route

1. Slot 67 creation publishes the request as before.
2. On the first Dispatch of the pipeline, `translate_compute_natively`
   (`src/startup_stubs/d3d12_compute_dispatch.inl`) reads the request back with `x2w::read_compute_request`,
   which checks the six parts against the key in the file name. It then runs the C++ frontend
   (`x2w::translate_compute`, entry 0) and compiles the module with `ShaderDxilCompiler::runtime_worker`.
3. The entry values and views fill the same `ComputeTranslation` a published `<key>.txt` fills. The existing
   dispatch path then resolves the table descriptors, sets the resource states and records the Dispatch.

Accepted: root constants, a raw table buffer loaded per lane and a raw table buffer stored to. These are the
bindings the fixture runs. Refused: every other view kind (a scalar buffer load included), a sampler, more than
32 entry values or inputs or 8 outputs, an adapter without 32-lane waves, and whatever the frontend refuses. The
stand-in logs the first 16 refusals of a process with their reason. A refused key is left to a published
translation. The in-process translation is tried once per key and before a published `<key>.txt`.

`NativeRecording` and `NativeDispatchStates` are not used. The dispatch path already allocates the descriptors
and tracks the states of these buffers.

## Program and controls

The program is case A of [`startup_gfx2hlsl_compute_check.py`](../../tests/startup_gfx2hlsl_compute_check.py):
two root constants, group size 2x3x4, 3x2x2 groups, one input word read and one word stored per thread. Its
`s_load_dwordx8` is replaced by two `s_load_dwordx4`, at table bytes 0 and 32. The C++ frontend reads a table as
32-byte records. It refuses case A as written (`unaligned descriptor table entry`), because that program takes
its second descriptor from table byte 16. The fixture checks the two replaced instructions with the Python
decoder, and the consumer's `dispatch-records32` mode writes the table with the output's descriptor at byte 32.

1. Native: all 320 words match the case's CPU expectation (288 written, 32 left at the sentinel). The log names
   the native translation. Afterwards the cache holds the request files and the compiler's own cache, and no
   `<key>.txt`, `<key>.cs.cso` or `<key>.fail`.
2. The same program over the plain table layout has no descriptor at byte 32. Its Dispatch is refused when it is
   bound and every word keeps the sentinel, so the record at byte 32 is what the first run bound.
3. Unsupported workload: case A as written. The stand-in logs the frontend's reason, the Dispatch is refused with
   `no translation` and every word keeps the sentinel.
4. Fallback: `gfx2hlsl.py --pending` then publishes both requests. With native translation still on, the next
   run dispatches case A through the published translation and all 320 words match.
5. Order: the first program runs again with its published translation in the cache. The log still names the
   native translation, and all 320 words match.

Without 32-lane waves, step 1 instead checks that the native translation is refused by that name with every word
untouched. Step 2 is left out, steps 3 and 4 run, step 5 runs the first program through its published
translation, and the check exits 77.

## Command and result

```powershell
cmake --build --preset dev --target d3d12_xs_shim startup_stream_consumer compile_dxil native_wave_request_gpu offscreen_check
ctest --preset dev --no-tests=error -V -R '^startup_compute_dispatch_(native_)?(warp|hardware)$'
ctest --preset dev --no-tests=error -V -R '^native_wave_request_hardware$'
```

Commit `a58e08f9` on `cpp-migration` at `e3670100`, Windows 11 build 26300.9974, Debug build, D3D12 debug layer
enabled.

| Test | Adapter, driver | Result |
|---|---|---|
| `startup_compute_dispatch_native_hardware` | NVIDIA GeForce RTX 4090, 32.0.16.1088 | PASS. All 320 words match after the in-process translation from an empty cache and again with a published translation in the cache. The plain table layout and case A are refused with every word untouched. No debug-layer error message. |
| `startup_compute_dispatch_hardware` | NVIDIA GeForce RTX 4090, 32.0.16.1088 | PASS (the published-translation route, unchanged). |
| `native_wave_request_hardware` | the same machine (the test prints no adapter line) | PASS. |
| `startup_compute_dispatch_warp` | Microsoft Basic Render Driver, 10.0.26100.9818 | PASS. |

Capability skip, not a pass: `startup_compute_dispatch_native_warp` exits 77 on the Microsoft Basic Render
Driver 10.0.26100.9818, which reports no 32-lane waves. Before it exits, the check confirms that the native
translation is refused by that name with every word untouched, that case A is refused and then runs through a
published translation, and that the first program matches all 320 words through its published translation.

The other WARP dispatch tests were run as a regression check and pass (9 tests):

```powershell
ctest --preset dev --no-tests=error -V -R '^(startup_compute_dispatch(_.*)?|startup_slot67_indirect_dispatch|startup_compute_translate_e2e)_warp$'
```

## Limits

- Compute only. Graphics pipelines are not translated in-process.
- A scalar buffer load, a texture, a sampler and the typed, half and indexed views are refused.
- One descriptor shape was dispatched: stride 4 and 320 records, for the input and for the output.
- An indirect Dispatch of a natively translated program was not run.
- A captured request starts with a 36-byte prefix of `v_nop` words before `s_inst_prefetch`. The route passes
  entry 0. With `compile_dxil`, a synthetic prefix translates to the same bindings from entry 0 and from entry
  36. The fixture does not dispatch a prefixed program.
- Only the two adapters above were run, in a Debug build. `run.ps1` takes its stand-ins from the release build
  directory by default, and no start-up run used the switch. The launcher switch was checked with stubs.
- An adapter whose lane range excludes 32 refuses every program.
- The result is not published, so each process translates a key again. The compiler's DXIL cache under
  `<shader cache>\native` is reused.
- The GPU-check package ships `offscreen_check.exe`, which compiles the stand-in source without
  `X2W_NATIVE_SHADERS`. That variant builds and `offscreen_warp` passes. The package script (`tools/package-gpu-check.ps1`) was run at
  `89a75318` (the same code): the extracted package passes its WARP run on the Microsoft Basic Render Driver
  10.0.26100.9818 with 11 tests passing and 68 of 68 recorded shader cases. `bvh_wave64_warp` reports a capability
  skip there (no 64-lane waves).
