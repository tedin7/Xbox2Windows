# XMem page runs: placeholder splits, joins and run breaks

Refs [#2911](https://github.com/R0Y00/Xbox2Windows/pull/2911). The fixtures below cover the shim's mapping of
physical pages into XMem reservations with shared views (`X2W_XMEM_SHARED_VIEWS=1`). The results quoted here were
measured under Wine (Proton Experimental, November 2026) on Linux x64, with a clang-cl cross build and each CTest
entry's command run through Wine in one prefix. They are not Windows results. The D3D12 fixtures at the end have no
result there.

## Placeholder split for queue slot 31

`xmem_placeholder_split` maps two pool pages into the middle of an XMem reservation through `X2WMapPageBatch`, which needs a placeholder split. Before splitting, xmem checks that the target is a placeholder, because on Windows `VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` can release a reservation that is not one. The check reads bit 0x100 of the `MemoryRegionInfo` flags. Wine (Proton Experimental, November 2026) leaves that bit 0 for placeholders but refuses the same release on a plain reservation with `ERROR_INVALID_ADDRESS` and keeps it whole. xmem measures both once. Where the bit is missing and the plain release is refused, it offers any reserved region to the split, and a plain one fails the split instead. Where the bit is reported, nothing changes. Without this, every slot 31 batch whose destination was not exactly one placeholder failed with `ERROR_INVALID_ADDRESS` under Wine. The fixture also maps onto a committed page in the same reservation, which must fail and keep its bytes (Windows returns `ERROR_INVALID_ADDRESS`, Wine `ERROR_INVALID_PARAMETER`). `xmem_placeholder_split_control` runs the same checks in the test-controls build (`xmem_shim_test_controls`) with `X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK=1` and must fail. `xmem_placeholder_split_deployed` sets the same switch for the deployed `xmem_shim`, which ignores it, and must pass. `xmem_placeholder_split_partial` (test-controls build, shared views) maps a two-page view over a private allocation whose first page is reserved and whose second page is committed. The fallback checks every region the view covers, so it refuses the view and the committed page keeps its bytes. A check of the first region alone accepted the view, and Wine then released the allocation and the view replaced the committed page. Limit: the fallback cannot tell a reserved page of a private allocation made with `VirtualAlloc2(MEM_REPLACE_PLACEHOLDER)` from a placeholder, because Wine does not refuse the release for that kind of allocation. After the refused view, the batch retries each page on its own and releases the reserved first page, which cuts it out of the caller's allocation: under Wine the committed page's allocation then starts one page later. The fixture prints where that allocation starts and asserts only that the first page is still one reserved page at its own address. Where the OS reports the bit, the fallback is not used. The control, `xmem_placeholder_split_deployed` and `xmem_placeholder_split_partial` skip (77) where the OS reports the bit. Run `ctest --preset dev --no-tests=error -R '^xmem_placeholder_split' -V`.

## Split before map

`xmem_map_split_first` maps eight single pages with `XMemMapPhysicalPages`, each into the middle of the
reservation's placeholder. xmem splits first and maps once, so the eight pages take eight `MapViewOfFile3` attempts
and eight splits, and an exact placeholder left by an unmap is mapped without a split. `xmem_map_split_first_control`
runs the old map-first order in the test-controls build (`X2W_XMEM_TEST_MAP_FIRST=1`) and must measure 16 attempts.
`xmem_map_split_first_deployed` sets the same switch for the deployed `xmem_shim`, which ignores it, and must measure
8. Measured under Wine: 8, 16 and 8. Run `ctest --preset dev --no-tests=error -R '^xmem_map_split_first' -V`.

## Joins across mapping calls

With shared views, the pages of one call that are adjacent slots of one section share a view. A call joins the run
that ends at its address only with `X2W_XMEM_JOIN_CALLS=1`, which counts only with shared views on. It is off by
default; `tools/startup/run.sh` sets it, and `tools/startup/run.ps1 -LazyMemory` does not. `xmem_cross_call_merge`
checks the joined layout (32 single-page calls on consecutive slots give one view of 32 pages) and the cases around
it. `xmem_cross_call_merge_no_join` checks that the deployed shim with shared views and no join switch keeps a view
per call (32 views). `xmem_cross_call_merge_control` turns the merge off in the test-controls build
(`X2W_XMEM_TEST_NO_MERGE=1`) and must also show 32 views; `xmem_cross_call_merge_deployed` sets that switch for the
deployed shim, which ignores it and joins. All passed under Wine. Run
`ctest --preset dev --no-tests=error -R '^xmem_cross_call_merge' -V`.

## Run breaks and the fault handler

Unmapping a page of a shared view unmaps the whole view and maps its other pages again; a join and a slot 31 batch do
the same for the runs they change. `xmem_run_break_readers` holds that window open (the test-controls build's
`X2W_XMEM_TEST_HOLD_WINDOW`) and checks that each reader either waits for it or, for a pipe `ReadFile`, fails inside
it without being run again (under Wine with `ERROR_NOACCESS`, both queued messages kept). The access-violation
handler retries a fault on a page whose view was mapped inside a break, once per break and thread.
`xmem_run_break_readers_stamps` holds a fault's handler while 63, 64 and 200 later breaks run on another page; each
fault is retried (under Wine the later breaks took 3, 4 and 10 ms). In the same mode, a page made no-access faults
once on a run no break has mapped (passed on at once) and twice after a break mapped it (one retry), counted by a
first-chance handler of the test. Run `ctest --preset dev --no-tests=error -R '^xmem_run_break_readers' -V`.

## DirectStorage copies into runs

`startup_dstorage_break_window` submits a memory-source request into a page of a run while an unmap holds the run
apart: the request must not complete during the 1000 ms hold, then lands; its control, an XCRDAPI that judges the
destination with its own query, refuses inside the window. The copy takes xmem's mapping mutex only when the
destination or source overlaps a run: `startup_dstorage_break_window_plain` measures a `VirtualAlloc` destination
copied without it and a run destination with it, and `startup_dstorage_break_window_views_off` an XMem page
destination copied without it when shared views are off (`x2w_xmem_copy_counts`, test-controls build).
`startup_dstorage_break_window_outcomes` forces each copy error (`X2W_XMEM_TEST_COPY_ERROR`) and checks the outcome
XCRDAPI logs for it. All passed under Wine. Run
`ctest --preset dev --no-tests=error -R '^startup_dstorage_break_window' -V`.

## D3D12 fixtures (not run here)

`xmem_page_runs_device_warp`, `xmem_page_runs_device_join_warp` and `linear_buffer_backing_share_warp` check view
boundaries through D3D12 placed resources. They need D3D12 with WARP, which the Wine prefix above does not provide:
they fail there (`linear_buffer_backing_share_warp`: "WARP device not created"), so this note has no result for
them.
