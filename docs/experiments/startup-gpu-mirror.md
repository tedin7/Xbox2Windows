# Persistent GPU mirror for large graphics bindings

Set `X2W_BIGBIND_ONCE=1` and `X2W_GPU_MIRROR=1` to mirror readable raw
graphics bindings of 16 MiB or more. The first binding in a frame hashes the
readable span in 64 KiB blocks after the existing extent and zero-tail checks.
The first upload initializes the default-heap buffer. Later uploads use
`CopyBufferRegion` for changed blocks. The recording reuses its snapshot for
later draws in that frame.

The cache keeps up to 256 MiB of declared buffer ranges. When a previous
recording can still replay or its GPU work is in flight, the new binding uses
the upload-ring path. The first use after an unsubmitted recording reuploads
the readable span. The usual 16 MiB graphics copy limit applies with the
mirror switch off. Formatted loads keep their existing limit.

On the Windows test machine, build the existing `startup_stream_consumer`
target and run:

```powershell
ctest --preset release --no-tests=error -R '^startup_gpu_mirror_warp$' -V
```

The fixture submits three separate frames on D3D12 WARP. It verifies a marker
in every block of a synthetic 32 MiB buffer and compares each frame's RGBA
readback with the CPU reference. Expected upload bytes are 33,554,432,
65,536, and 65,536. A fixture-only switch bypasses the 16 MiB limit for the
mirror-off control, which expects 33,554,432 bytes each frame. The stale
mutant skips dirty-block copies after initialization. Its last two frames
must differ from the CPU reference.

## Shared uploads

Add `X2W_GPU_MIRROR_SHARED=1` to let all direct command lists in a frame use
the same mirror. The switch defaults off. Dirty blocks are packed into staging
storage and recorded on a device-owned upload list. The next direct submission
or Present on that device executes the copies in a separate queue call.
Subsequent bindings of the same range in the frame reuse its snapshot. A range
first bound after a submission can start a new batch. Each upload returns its
buffer to shader-read state. At the execution boundary buffers decay to COMMON
and promote for shader reads in the title lists.

Each mirror remembers its upload fence and the latest read fence from each
queue. A reader waits on a different queue's upload. An overwrite waits on
all other queues that used the previous snapshot. Same-queue ordering covers
the work on that queue. Resetting a title recording leaves the pending upload
intact. A previous frame's recording that can still be submitted pins the
snapshot and forces a later binding to use the ring until that recording is
reset or released. The 256 MiB declared-range budget and 16 MiB fallback limit
apply. The frame log reports executed upload lists, blocks, bytes, binds and
fallbacks.

```powershell
ctest --preset release --no-tests=error -j 1 -R '^startup_gpu_mirror(_shared)?_warp$' -V
```

The shared WARP fixture checks all 1,536 blocks of a synthetic 96 MiB range.
Three recording threads make lists that are submitted in reverse order, in
two queue calls per frame. Expected upload bytes are 100,663,296 and 131,072.
The fixture also resets the first binding before submission and separately
verifies that Present can execute that orphaned upload. A GPU fence gate
holds two readers in frame 1 until frame 2 has been submitted on a different
queue. The upload-after control must corrupt the first submitted list's
readback. The skipped reader-wait control must expose frame 2's data to the
two gated frame 1 readers.
