# Shared descriptor walk for translated draws

Set `X2W_WALK_SHARED=1` to snapshot the SRV and UAV table entries of a ready
graphics translation. The default is off. A cold or pending translation uses
the existing exclusive path. The exclusive path checks that it selected the
same ready translation before consuming the snapshot.

| Data used by the walk | Protection |
| --- | --- |
| Pipeline-to-translation map, target-format variant map, ready state and SRV/UAV layouts | Shared `draw_lock` for lookup and snapshot. The existing exclusive path publishes translations and mutates the maps. Ready binding layouts stay fixed. |
| Command list's `Pass`, pipeline identity, target formats and table addresses | Owned by the sole recording thread for that list. Each worker needs its own command list and allocator. |
| Title-owned descriptor bytes | The title keeps the table backing alive and synchronizes writes to the entries it binds. The shim bounds-checks the entry and probes readability before copying it. Neither lock synchronizes external writes to title memory. |
| `view_records()` and its backing storage | Shared `note_lock` while copying the indexed record, nested inside `draw_lock` in the existing order. The capture path publishes records under exclusive `note_lock`. |
| `note_lock` wait counters | The existing interlocked counters record each acquisition. |
| Descriptor and view snapshots | Local to the recording call. Resource identities copied from records are resolved during the exclusive phase. |

The exclusive phase also performs texture materialization, resource state and
lifetime bookkeeping, PSO creation, binding cache inserts, descriptor writes,
upload ring allocation, and draw emission. Wrapper lookup and resource
resolution retain their existing synchronization.

The fixture in `startup_stream_consumer` records the same 16 draws on one, two
and four recording threads. Four targets carry distinct CBVs and every measured
draw covers its own column. The check compares all 128 pixels to the single
thread reference and to the expected colors. Two empty Presents start the
report interval and emit the existing `draw_lock` heartbeat. The fixture
expects 17 shared draw walk holds with the switch on, including the warm-up,
and zero with the switch off. Each successful draw still takes the exclusive
lock for mutation and emission. The shared hold count includes lookups that
miss the ready cache.

```powershell
ctest --preset dev --no-tests=error -R "^startup_draw_walk_threads_(warp|hardware)$" -V
```

For the exclusive-walk mutant, replace the `SharedDrawWalkLock` guard at the
snapshot site with a guard that calls `draw_lock_acquire_exclusive()` and
`draw_lock_release_exclusive()`. Leave the snapshot and emission code in place.
The enabled case should still match the pixels and fail at zero shared holds.
Windows execution is pending.
