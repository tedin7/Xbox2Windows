# Dedicated buffers for large uploads

Set `X2W_BIG_UPLOAD_POOL=1` to use a dedicated upload buffer for each
reservation over 2 MiB. Reservations of 2 MiB or less use the upload ring. The switch
defaults to off.

The pool holds up to eight persistently mapped buffers and 128 MiB of capacity,
charged in 64 KiB units. It reuses the smallest idle buffer that fits. It can
release idle buffers to make room for a different size. A closed command list
retains its buffers for resubmission. Once the recording and any frame-cache
holds end, the pool can reuse a buffer after all queues that used it complete
their tracked work. A failed submission fence leaves the buffer pinned.

The synthetic frame fixture interleaves four 16 MiB bindings with 28 small
updates on an 8×4 render target. It compares every pixel with the CPU reference
and checks for upload refusals and missed draws. The control limits the ring to
64 MiB and disables the pool, so at least one large binding must fail and the
corresponding pixel must retain the clear color. The enabled case also runs
with the once-per-frame binding cache. The reservation test exercises the pool
slot mask and fence checks with two synthetic queues.

Run from an existing Windows dev build:

```powershell
ctest --preset dev --no-tests=error -R '^(startup_frame_copy(_once|_pool)?_(warp|hardware)|upload_reservation)$'
```

The Windows results are pending. The fixture measures synthetic upload and draw
behavior.
