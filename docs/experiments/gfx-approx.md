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
