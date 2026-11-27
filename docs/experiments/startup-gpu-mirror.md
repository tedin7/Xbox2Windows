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
