# Shared translated draw binding walk

Set `X2W_DRAW_WALK_SHARED=1` to let recording threads walk the bindings of
ready direct translated draws concurrently. The switch defaults off. The
heartbeat line `translated draw under draw_lock` adds the average time each draw
held the global lock exclusively. Indirect draws and draw capture keep the
exclusive path.

On Windows, with the startup stream consumer built and the Python checks
registered, run:

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_frame_copy(_once|_shared_walk)?_warp$'
```

The shared-walk fixture runs four command lists and draws 1,000 distinct pixels
on each list in each of two frames. It compares the WARP readback against the
CPU colors, the serial recording run, and the switch-off recording run. It
requires exactly one copy for each range in each frame and checks that the
second frame reads the new shared bytes. The synchronized missing-recheck
control must produce four copies of the first shared 4 KiB range. The sum of
global exclusive hold ticks must be below 20% of the switch-off sum over the
same draw count. The output reports ticks and the ratio for the Windows run.

The Python reference and log checks were exercised on Linux with synthetic
results, including duplicate copies. Windows readback, the runtime control,
and the exclusive-hold ratio await the independent WARP run.
