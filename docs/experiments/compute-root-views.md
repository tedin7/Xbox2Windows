# Compute root views

Command-list slots 41, 43, 45 and 46 report validation against the current root
signature. Each census line gives the cumulative calls, calls matching the
declared parameter type, and addresses in a placed buffer, list upload memory,
or neither. Slot 46 records a graphics UAV separately from the graphics shader
inputs. Slot 43 continues to forward aligned list upload SRVs to the desktop
list. Measurements from the title are needed to validate the proposed slot
types.

Set `X2W_COMPUTE_ROOT_VIEWS=1` before running the stand-in to resolve root views
in translated dispatches. It defaults to off. A version 2 compute
manifest can use `srv <index> buf root <parameter>` for a raw SRV or CBV input
and `uav <index> root <parameter>` for a raw UAV output. The buffer range runs
from the recorded address to the end of its placement, subject to the raw
buffer validation of the table path. The root CBV requires 256-byte alignment.
The root-view dispatch census counts successful dispatches that used at least
one such input or output.

The baseline translator refused the synthetic root-buffer store with
`global_store_dword`. It already emitted a root source for the loads, which
the baseline runtime rejected with `a resource that is not a raw table buffer,
a 2D texture or a cube map`. The worktree contained zero cached requests or
refusal files. The number of affected keys in an external cache is unknown.

On Linux, `python -B tests/startup_compute_dispatch_check.py controls` passes
with the change. The same synthetic program fails with the translator at
`e8015da7a` with the refusal above. On Windows, the following command exercises
the root storage and WARP dispatch fixtures:

```powershell
ctest --preset dev --no-tests=error -R '^(startup_compute_root|startup_compute_dispatch_warp)$'
```

The WARP fixture checks signature match counts and both buffers using
`CopyBufferRegion` and a CPU reference. It checks the default-off refusal and
a refusal after a root change. Its mutant binds the input address as the UAV
and reads the records through that UAV, so the GPU changes five input words.
The normal CPU comparison must reject the first changed input word.
Windows verification is pending.
