# Startup query slot census and result modes

The D3D12 stand-in counts calls to command-list slots 28, 29 and 30 and
reports their argument fits with the render totals. A fit identifies a query
heap from device slot 39, a compatible type and a range within the heap.
Resolve candidates also need a buffer wrapper. The census measures support
for the proposed call layout. Slot identities and semantics remain a hypothesis.

`X2W_QUERY_RESULTS` defaults to off. Set it to `forward` to enable the
original occlusion-only forwarding in render mode. The diagnostic
`X2W_D3D12_QUERY_SLOTS=1` still selects forwarding when a new mode is absent.

Set `X2W_QUERY_RESULTS=visible` to allow synthetic results in writable title
memory of placed buffers. Binary occlusion returns 1. Sample counts use the
bound target area when available and 65,536 otherwise. Statistics fields
are zero. Timestamp results use increasing performance-counter ticks scaled
to the last successful queue frequency reported by slot 18. Until a frequency
has been reported, they use the performance-counter frequency. The recording
does not identify its submission queue, so this mode assumes one timestamp
frequency across the queues that use the results.

On Windows with the D3D12 debug layer installed, run:

```powershell
ctest --preset dev --no-tests=error -R '^startup_list_queries_.*warp(_control)?$' -V
```

The WARP fixture expects 16 visible samples from a synthetic quad and zero
from a fully clipped quad in forward mode. Visible mode checks title memory
and reads the same bytes back through a GPU copy. It also checks timestamp
scaling, statistics, census counts and destination guards. The default
control expects untouched buffers and no debug-layer errors. The mutant
control injects zero samples into visible mode and requires the first
readback comparison to reject that exact value. No Windows result is
recorded here.
