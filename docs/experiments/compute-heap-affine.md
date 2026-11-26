# Affine resource heap offsets in compute

A compute scalar load can name a raw buffer through a 32-byte heap record. The
translator now proves offsets of the form `rootWord * scale + add` with
32-bit wraparound and 32-byte alignment. A value survives a control-flow join
when every reachable predecessor proves the same expression. Group IDs and
buffer loads do not seed this proof.

The translated dispatch checks the complete guest record span, copies the
resolved view into the desktop descriptor heap, and passes the remapping bias
and desktop heap size to the shader. The Shader Model 6.6 shader divides the
byte offset by 32, applies the bias, and checks the desktop index before
accessing `ResourceDescriptorHeap`. The existing same-block heap record and
heap word paths retain their manifest and HLSL.

On Linux, run the synthetic proof and byte-encoding checks with LLVM:

```sh
python3 -B tests/startup_compute_heap_affine_check.py --llvm-mc llvm-mc-19
```

The check passed on Linux x86-64 with Python 3.14.8 and LLVM 19.1.7.
Both WARP input programs hit the targeted refusal at baseline `e8015da7a`.
All 33 existing compute fixtures in `startup_gfx2hlsl_compute_check.py`
emitted byte-identical HLSL and manifests against that baseline. The two
legacy heap check suites passed their exact-output and refusal assertions.

On Windows, use the configured test tree and the `startup_stream_consumer`
target:

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_heap_affine(_warp)?$'
```

The WARP fixture dispatches one program that uses an offset across a branch
target and another that computes `(rootWord + 2) * 32`. It waits for the queue
fence and compares all 320 output words with a CPU reference. Each program also
tries an out-of-range guest record and runs a compiled shader mutant that makes
the desktop index out of range. Both controls must leave all output words at
their initial sentinel. The WARP test reports a capability skip if Shader
Model 6.6 or resource binding tier 3 is unavailable. Windows execution is pending.
