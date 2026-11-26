# Translated pipeline creation

Set `X2W_PIPELINE_CREATE_UNLOCKED=1` before starting the process to release the
exclusive draw lock during desktop graphics and compute pipeline creation. A
recording thread that hits the same in-progress entry sleeps until publication.
A failed desktop creation is cached. With the switch unset or set to 0, each
desktop creation runs with the exclusive draw lock held.

The heartbeat reports cumulative successes, failures, total and maximum
creation times in milliseconds, and attempts outside the lock. Creation times
measure the device call. The existing exclusive-hold and lookup-phase timings
exclude the pipeline-specific gaps.

## WARP reproduction

```powershell
ctest --preset dev --no-tests=error -j 1 -V -R '^startup_pipeline_state_creation(_(locked|duplicate_control|translated_draw|translated_dispatch))?_warp$'
```

The contention fixture uses the production reservation and publication
primitive with WARP. Four recording threads issue two draws per key across two
target formats and eight depth comparisons. Every pixel is compared with both
a CPU reference and a single-thread execution. It expects 16 successful
creations and 16 outside the lock with the switch on, and 16 and 0 with it
off, before probing the invalid key. The first creator waits for the second
lookup and injects a short delay before calling the driver so the hold-time
comparison can detect a lock held across creation even with a warm driver
cache. The fixture also checks one failed attempt across 32 lookups and reads
back four words from a compute dispatch. The duplicate control suppresses the
in-progress marker and must report more than 16 successful graphics creations
while still matching the CPU reference.

The translated draw and dispatch tests exercise the respective shim entry
points with the switch on and compare their readbacks with CPU references.
The four-thread fixture measures contention at the shared creation primitive.
Windows results and independent verification remain pending.
