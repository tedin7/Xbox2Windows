# Approximate shader translation for replay diagnosis

Set `X2W_GFX_APPROX=1` in the environments of both the offline translator and the
replay process. The default uses the normal translation proofs. Replay requires
an explicit local opt-in even if the recording used approximate translation.

The offline translator tries the normal lowering first. If it cannot prove a
`v_readfirstlane_b32` source uniform, the diagnostic lowering takes the current
lane's source. If the entry EXEC proof fails on a recognized wave reduction,
the diagnostic lowering applies the normal wave intrinsic to the active host
lanes. Both choices can change shader results.

The original request key still locates the shader inputs. An approximate result
uses the hash of `approx1` seeded with that key and publishes a
`<request-key>.approx.txt` link after its manifest. Replay logs each loaded
approximate key once with its stage and the counts of lifted instructions.
The render totals count loaded approximate keys. The tester report includes
counts for cached approximate results and keys logged by the runtime.

The Python fixture checks strict refusal text, pins the accepted translation
keys and output hashes from revision `7472a1ecc`, and checks that only lifted
translations get a diagnostic key. A default-on mutant fails the unset-switch
control. The WARP fixture supplies eight equal input words and compares all
320 output words with a CPU reference. A lowering that zeros the scalar must
produce a readback mismatch.

```sh
python3 -B tests/startup_gfx2hlsl_approx_check.py
```

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_approx_(controls|warp)$'
```

## Level 2 (permissive)

Set `X2W_GFX_APPROX=2` explicitly in the translator and replay environments
to diagnose refused shaders. Level 2 is diagnostic only and is never enabled
by default. It lives only on the branch replay-permissive; never merge it
into the tester-loop or replay lines.

The translator keeps accepted shaders byte-for-byte. For a refusal or a
placeholder pixel stage it tries the level 1 lowerings, then collects
refusals while continuing to lower the shader. A skipped instruction writes
zero to its register destinations. A skipped store, atomic or export drops
its side effect. A skipped atomic with a return value writes zero to that
destination. Proof failures can leave code in place under weaker
assumptions. For example, a recognized reduction can use the HLSL wave
intrinsic over the active desktop lanes. Shader results can be wrong.
If a retry cannot produce a complete shader and manifest, the translator
keeps the strict refusal or the configured placeholder pixel stage. A failed
permissive pass logs the request key, stage and first line of the exception
on stderr. The batch summary counts those failures, including requests that
fall back to a pixel placeholder.

The permissive pass lowers the instructions in EXEC-zero blocks for the active
lanes and can take the current lane's value for an unproven first-lane read.
At an ambiguous scalar join it can use the symbol from the first incoming
edge. For compute it can append missing termination, infer the wave size for
`TG_SIZE_EN`, share a raw UAV between loads and writes, and skip LDS operations
when the exchange proof fails. An unproven resource-heap offset can use the
shader-side descriptor lookup. Mesh entry scalars that the model cannot define
read as zero. [AMD RDNA 2 ISA reference](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)

The marker records the affected stages and counts the observed sites in
`readfirstlane`, `reduction`, `uniform` (including lane masks and SCC),
`execz`, `resource`, `mesh`, `store_dropped`, `unknown_op` and `other`
order. It appears on translator stderr and in the derived manifest. The
derived key hashes `approx2` with the original request key as the seed. The
translator writes `<request-key>.approx2.txt` after the manifest. Replay
prefers an exact result and can use the alias when the exact result is
absent or a placeholder. Slot 57 mesh results are offline only.

The controls above also check level 2 key derivation, cache publication,
failure reporting and recovery on synthetic compute and mesh programs. The
new recovery cases assert the exact refusal at levels 0 and 1 and inspect
the level 2 marker and emitted HLSL. The WARP test
compares all 320 words against a CPU reference, including pairs of a
supported value and the zero result in each of eight threads.
