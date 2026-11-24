# Startup runtime stand-ins (experimental)

This page describes the experimental desktop stand-ins under `src/startup_stubs/` and the tools
under `tools/startup/`. They let a console executable start on desktop Windows under a debugger,
so that its startup path can be studied one missing piece at a time.

This is not a port and not an emulator. No executable is playable. Every stand-in is opt-in, and
most answers are shortcuts that are logged as such. The open problems are tracked in the issues
whose titles start with `startup-runtime:`.

## Derived pipeline blend state

Device slot 55 accepts up to 64 fixed-size derived pipeline patches. The translated graphics
path reads the blend descriptor from part 2 of a 332-byte patch. Within one derivation the last
blend patch wins. A further derivation inherits that state until it supplies another blend patch.
Other patch parts are outside this blend implementation. The record layout can also be compared
with [XCL's implementation at 64b3f35](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/d3d12x/device_x.rs#L501-L515).

Run `ctest --preset dev --no-tests=error -R "^startup_stream_draw_(hardware|warp)$" -V` after
building `startup_stream_consumer`. The source-built shader writes to half of an 8×4 target.
On A8 the parent preserves the zero clear and the child writes alpha 255. R8 records red 0 or
255 according to the input shader. The untouched half stays zero in all four cases.

The patch scan itself also has a device-free check. Build `startup_derived_blend_check` and run
`ctest --preset dev --no-tests=error -R "^startup_derived_blend(_first_wins_control)?$" -V`.
It covers what the draw cannot observe: the logged descriptor offset (4 bytes into the record),
a byte-for-byte copy of the descriptor, and refusal of a null list or one that is not whole
records. The control expects the first blend patch to win and must fail.

## Stencil reference

Command-list slot 24 (`OMSetStencilRef`, the desktop slot) passes its reference to the desktop list, as slots 21 to 23 do for viewports, scissors and the blend factor, so a translated draw whose pipeline enables the stencil test compares against it. A closed list ignores the call. The slot number is the desktop one and has not been seen in a title log; `startup_stream_depth_{warp,hardware}` checks the references 0x5A, 0x5B, 0 and 0xFF against a stencil cleared to the matching value.

## Default viewport and scissor

A desktop command list that never sets a viewport or scissor rectangle draws no pixel. A translated
draw therefore covers the whole first render target (its width and height, depth 0 to 1) when the title
set no viewport, and the same rectangle as scissor when it set no scissor. Each default applies on its own,
and a viewport or scissor the title did set is never replaced. A list reset starts without either. A call
with no viewports or no rectangles changes nothing. A draw with no render target bound, for example
depth only, gets no default. The default uses the size of mip 0 of the resource; the render target view's mip slice is not tracked.

Run `ctest --preset dev --no-tests=error -R "^startup_stream_viewport_(hardware|warp)$" -V` after building
`startup_stream_consumer`. On an 8×4 target the cases draw with neither call, inside a 4×2 rectangle (the
rest keeps the clear color), and on a reset list that follows the rectangle (all pixels drawn again).

## Stream vertex fetch from the input layout

A vertex shader in a slot-86 stream can take its inputs from a fetch routine that the console driver builds
from the stream's input layout. Such a shader starts with `s_swappc_b64 s[4:5], s[4:5]` and only then reads
the registers the routine loaded. The routine's code is not in the shader container.

The stand-in reads the `INPUT_LAYOUT` subobject and publishes it next to the request as `req_<key>.ia`, before
`.a`. The file holds `ia1 <count>` and then one line per element: semantic, index, format, slot, offset, class
and step rate. A stream with a non-empty layout includes that text in its cache key; other keys are unchanged.
One log line per stream gives the element count and, for each element, format, slot, offset and class:
`x2w-draw stream input layout: 2 elements (format/slot/offset/class) 6/0/0/0 6/0/4294967295/0`.

`gfx2hlsl.py` translates the call as the loads it stands for, through the vertex-buffer path the other
translations use (`srv buf vb <slot>` with a `stride` entry value). The registers come from a table in the
container's native header. From +0x138 it holds one 5-byte record per fetched input: kind (6), input-signature
register, component mask, first VGPR and component count minus one. Each record is matched to a layout element
through the semantic name and index that the input signature gives for its register.

The table was checked offline against the input signatures of every readable vertex program in one title's shader
archive (local files only, not in the repository). 628 programs start with the call. All of them have a header of
0x2f8 bytes and records of kind 6. Each of their 1,417 records names a signature register, and its mask equals
that input's read mask. Records follow register order. The first record starts at v4, and each next one starts 4
VGPRs later. The count reaches the highest component read: five records read only z or only w (mask 0x4 with
count 3, mask 0x8 with count 4), and all components up to that one are loaded. None of the 1,355 programs without
the call has a record. In all 1,983 programs the header's u32 at +0x1d8 equals the number of records.

Refused: a header of any other size, a record of another kind, a mask whose highest bit is not its count,
overlapping registers, a record reaching past v255, and a record count that disagrees with +0x1d8. A fetched element is also refused when it is
per-instance, sits in a vertex-buffer slot above 7 (the stand-in binds slots 0 to 7), has an unknown format, or is
appended (`D3D12_APPEND_ALIGNED_ELEMENT`) after an element of unknown size or one that ends off a 4-byte boundary,
because the packing there is not measured. Elements that no record fetches are not checked. An unreadable layout
is logged, and the translation is requested without it. Components a format lacks read
as 0, 0, 0 and 1, as in the existing vertex-fetch table; that default is not measured for this path.

A shader that starts with the call and has no layout is refused.

After the call, every vertex register the shader reads must be defined on all paths. The translator proves this only
for the entry block: the straight-line code from the call to the first branch or branch target. A register counts as
defined when it is v0, a fetched register, or written in the entry block before the read by an instruction that
runs with the entry exec mask. A write after exec was changed, or in any block after the entry block, does not
count, even when it would run on every path. A read of any other register is refused. Programs whose later
blocks only read registers defined this way still translate. After the call, any exec write ends the
unconditional region, whatever instruction makes it. `s_wqm` of exec into itself only adds lanes and stays a
no-op. Other `s_wqm` forms, writes to exec_hi, and exec writes by an instruction the translator does not model are
refused.

Before this change the call was cut off with the header and every input read as zero. The draw then had zero area
and wrote no pixel. Programs without the call translate as before.

Run `ctest --preset release --no-tests=error -R "^startup_stream_fetch_(warp|hardware)$" -V` after building
`startup_stream_consumer`. The source-built shader declares POSITION and TEXCOORD. The stream lists them in the
other order, TEXCOORD in slot 0 and POSITION in slot 1 at `D3D12_APPEND_ALIGNED_ELEMENT`. Two draws go to an 8×4
target: one from start vertex 3, and one indexed with indices 4 to 9 and base vertex 5. Exactly the two position
rectangles are drawn. Two mutations of the generated fetch must fail the exact check and draw the TEXCOORD
rectangles instead: the records' registers exchanged, and both elements read from slot 0. Leaving the base vertex
out of the fetch index also fails it. A shader without its layout is refused and not drawn.

The translator cases need no device: `python -B tests/startup_stream_fetch_check.py`, or
`ctest --preset release --no-tests=error -R "^startup_stream_fetch_translate$"`. They check the generated loads and
manifest entries for both slots, and that a fetch behind `s_wqm_b64 exec, exec` still translates. They also check
these refusals:
- no layout;
- an export of v12 that nothing defines;
- v12 written only after exec was set to 0;
- v12 written only in code a branch jumps over;
- v12 written after `s_wqm_b32 exec_lo, 0`, or after a write to exec_hi;
- a z-only record whose count stops below z (a count of 3 translates);
- an element in slot 8, a per-instance element, or an element of format 10;
- a record of kind 7, or one that reaches past v255;
- a header record count that disagrees with the table.

The pixel stage still reads attribute N from vertex output N. It does not yet pair the two by semantic, so a
pixel shader that reads only some of the vertex outputs can receive the wrong one. That is a separate problem.

## Output window and fallback diagnostics

The desktop swap chain has its own top-level window, class `X2WOutputWindow`.
The presenter only samples non-empty single-sample 2D surfaces without
`D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE` and with a supported view format
(`src/startup_stubs/present_view_format.h`), and marks itself failed if
closing its command list fails.
When a present cannot show its surface, the GDI fallback selects a visible,
unowned guest window in the same process. If only the presenter is visible,
the fallback leaves it alone. With two visible guests it draws into the upper
one in z-order. To exercise both z-orders with source-built
windows, build `startup_overlay_window` and run
`ctest --preset release --no-tests=error -R '^startup_overlay_window$' -V`.
The same run raises each of two guests in turn and checks that the raised one is
selected.

## Null Texture2D descriptors

When the adapter lacks typed UAV support for B5G5R5A1_UNORM or render-target support for
B4G4R4A4_UNORM, the wrapper uses an R8G8B8A8_UNORM null descriptor if that format supports the
requested view. This applies only to Texture2D views with no resource or counter. Run
`ctest --preset dev --no-tests=error -R '^startup_null_views_' -V` against your build
directory to exercise both slots on hardware and WARP. On Windows 11 build 26300.9974 two adapters on one machine split. An RTX 5080 Laptop (32.0.15.8142) lacks the typed UAV and render-target support the rule keys on and reports `slot 19 format 86 -> 28; slot 20 format 115 -> 28`; an Intel Arc 140T (32.0.101.8132) has it, like in-box WARP 10.0.26100.9818, and both report `slot 19 format 86 -> 86; slot 20 format 115 -> 115`. Physical versus WARP is therefore not the discriminator; select the adapter with `XBOX2WINDOWS_ADAPTER=intel` or `XBOX2WINDOWS_ADAPTER=nvidia` and read the two lines.
Because WARP keeps both formats, `startup_null_texture_view` checks the rule without a GPU: fake
adapters answer the format queries, and the check covers a packed format with or without its use,
one with only the other slot's use, R8G8B8A8 with one use or none, failed queries, and the other
dimensions, formats, resources and counters that are left alone. A rewrite must keep every other
field of the description. `startup_null_texture_view_unchecked_fallback_control` rewrites without
asking about R8G8B8A8 and must fail.

## 16-bit packed targets

A description that asks for a B5G5R5A1_UNORM, B4G4R4A4_UNORM or B5G6R5_UNORM render target is created as
R8G8B8A8_UNORM when the adapter lacks that use for the packed format and has it for
R8G8B8A8_UNORM (`src/startup_stubs/packed_target_format.h`). Without this the description is
refused, its allocation-info answer is all ones and the executable gets no texture. The first
four stand-ins are logged.

The rule covers a single-sample 2D texture or 2D array (2 to 2048 slices; a cube map is six) that is
a render target, alone or with unordered access. Only the format changes: slice count, mip count
and the view dimensions are kept. Other dimensions, zero or more than 2048 slices, multisampled
textures and textures without the render-target flag keep their format.

A stand-in texture carries the format it was asked for as private data. A view that names that
format on that texture is created as R8G8B8A8_UNORM, both when the executable creates the view
and when the draw path builds its own view from the recorded description. A packed view of any
other texture of a different format is refused and logged: of a plain R8G8B8A8_UNORM texture, of
the stand-in for the other packed format, or of a texture in the other packed format.

Limits:

- Quantization. The stand-in keeps 8 bits a channel where the packed formats keep 5, 4 or 1.
  Rendered values are not rounded to the packed precision, so a read of the target can differ
  from the console's in the low bits: a channel cleared to 0.45 reads 115 from the stand-in and
  119 from B4G4R4A4.
- CPU-authored texels. Data the executable writes or reads itself in the 16-bit layout (uploads,
  copies to or from buffers, mapped reads) is not converted. Only textures that are rendered to
  and then sampled are covered.
- B5G6R5_UNORM has no alpha; its stand-in does, so the alpha of a sampled stand-in is whatever was
  rendered, not 1.
- Unordered-access writes to a stand-in and array or cube stand-ins through the wrapper are not
  exercised by a fixture; the selection rule and the view verdicts for them are checked without a device.

Checks:

- `ctest --preset release --no-tests=error -R '^startup_packed_target$' -V`: the selection rule
  against adapters described by their support bits (B5G6R5_UNORM, arrays of 2, 6, 16 and 2048
  slices, and the refused 0, 2049, multisampled and 3D cases), and the view verdicts, without a
  device. It also tags fake resources: an R8G8B8A8_UNORM stand-in must read back the packed format it was
  asked for, and a texture created in the packed format must carry no tag. The five
  `startup_packed_target_*_control` runs perturb the helper's answer (wrong stand-in format, arrays
  left alone, B5G6R5_UNORM left alone, view kept packed, tag read back as none) and must fail
  (`ctest --preset release --no-tests=error -R '^startup_packed_target_.*_control$'`). On the WARP
  build tested, all three packed formats were created as described, so the draw check below did not read a tag.
- `ctest --preset release --no-tests=error -R '^startup_packed_target_draw_' -V`: through the
  wrapper, a texture of each packed format is created (slot 133), cleared through a render-target
  view, sampled in a draw and read back. The expected pixel follows the format this adapter
  gets, so the check holds with and without the stand-in. R8G8B8A8_UNORM is drawn as a control
  and never substituted; a packed view of a plain R8G8B8A8_UNORM texture and a
  B5G5R5A1 view of a B4G4R4A4 target must be refused.

## Descriptor heap starts that are not addresses

A console executable treats the CPU start of a descriptor heap as an address. A desktop handle is
opaque, and a translation layer may keep tag bits in its low bits: through vkd3d-proton a sampler
heap starts at `...007`. With such a start for a heap created without memory of its own, the
source handle the executable later passed to `CopyDescriptorsSimple` for its shader-visible
sampler heap had `0xfd` in its top byte, and the copy faulted inside the layer 1.4 s into a run.

A heap whose desktop start is not pointer-aligned therefore gets plain memory of ours as the
start the executable sees, the way view heaps already do
(`src/startup_stubs/title_heap_memory.h`); handles inside it are translated to the desktop heap
on each call. Heaps whose desktop start is an address are unchanged, so nothing changes on an
adapter whose handles are addresses.

- `ctest --preset release --no-tests=error -R '^startup_title_heap(_tags_control)?$' -V`: the
  rule, without a device. `ctest --preset release --no-tests=error -R '^startup_title_heap$' -V`
  also checks that for an aligned view heap and a tagged sampler heap without memory of their own,
  the allocator is asked for the whole heap's size and its address becomes the heap's memory.
- `ctest --preset release --no-tests=error -R '^startup_title_heap_device_' -V`: through the
  wrapper, the starts of two sampler heaps, a render-target heap and a depth heap must be aligned
  addresses, a heap whose desktop start is tagged must have memory of its own, and three samplers
  are created and copied through handles computed from the starts.

How the executable arrives at the damaged handle is not known; only the handle and the fault were
observed.

## Copies of title memory within a frame

A translated draw that reads title memory by address (a constant buffer, a vertex or index range, a root pointer)
gets a copy in the upload ring. Within one frame the copy is found again by address and size, but only while the
title's memory still holds the bytes it was made from: each entry keeps them, and a draw that finds different bytes
makes a new copy and logs one `frame copy of ... refreshed` line (the first four). Without that, a title that rewrites
one address between two draws of a frame, such as a dynamic constant buffer, gave the second draw the first draw's
data. The comparison uses the kept bytes because the ring is write-combined memory that is slow to read.
`startup_frame_copy_{warp,hardware}` rewrites a constant buffer between two draws and checks both halves.

## Root signature changes and dropped root constants

Command-list slot 34 (`SetGraphicsRootSignature`) clears the translated draw's cached descriptor tables, root
addresses and root constants when the signature changes. Setting the same signature again preserves these arguments.
An unset constant reads as zero in the translated draw. Slot 40
(`SetGraphicsRoot32BitConstants`) keeps 16 parameters of 8 words. A call with a parameter of 16 or more or an offset of
8 or more is dropped, and one whose count passes the eighth word is truncated; each logs one line
(`root constants dropped` or `root constants truncated`, with the parameter, offset and count) for the first four such
calls in the process. `startup_root_reset_{warp,hardware}` draws a 1D texture through a bound table and writes a
root constant into the blue channel. It checks the original bindings, the same signature, a switched signature and
six out-of-range calls. The switched case clears both inputs to the shader.

## Null bindings in translated draws

The `x2w-render totals` line reports `null bindings N`, then the same count split by cause as
`(entry, buffer, token, wrapper, dimension)`. Each null binding falls in exactly one cause, so
the five parts add up to N:

- entry: the descriptor table entry is null, empty or unreadable.
- buffer: the buffer address or size is missing or unreadable, or the frame copy failed.
- token: the descriptor memory names a view id the wrapper never issued.
- wrapper: the texture's resource is missing or is not a resource wrapper.
- dimension: the view dimension does not match the shader, so a null texture is bound.

The classification lives in `src/startup_stubs/null_binding_cause.h`. The source-built
`startup_null_views_check` fixture checks every cause, that the parts sum to the total, and that
valid bindings add nothing. It does not draw, so the counters in the totals line itself are checked
by reading a render log. The device prints the counts by cause value, so
`ctest --preset dev --no-tests=error -R "^startup_null_cause_labels(_buffer_token_control)?$" -V`
checks without a device that each cause lands in the slot printed under its label, and that the
header's names follow the same order. The control swaps the buffer and token slots and must fail.

## Root constant buffer views

Command-list slot 42 (desktop `SetGraphicsRootConstantBufferView` plus four) records the title address for a root
parameter that the root signature declares as a CBV. A translated shader that reads through its root pointer sees that
address. The call is ignored when the parameter is out of range or is not a CBV (one log line, `root CBV N ignored`);
a null or unreadable address clears the slot to zero so a previous binding is not left stale. It is not forwarded to the
desktop list and does not set the vertex-fetch bit of `srv_bound`. The slot number is derived from the desktop order and
the console shift, not yet seen in a title log; `startup_root_cbv_{warp,hardware}` checks the handler with a bound, an
unbound, a null-overwrite and a wrong-type call. Rebinding the same root signature keeps
the address. Switching to a different signature invalidates the stored root addresses.

## Heap-indexed root signatures

Device slot 16 records whether a title root signature was created with
`D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` (`0x400`), `SAMPLER_HEAP_DIRECTLY_INDEXED` (`0x800`)
or both. Command-list slot 32 only tracks the heaps a title sets and never forwards them, so setting such a root
signature on the desktop list made the debug layer report message ID 1419 for each flag. Slot 34 now keeps
tracking the title's parameters, root addresses and constants for every root signature, but sets only an unflagged
one on the desktop list, as before; a flagged one is deferred (`root signature ... with heap flags 0x... not set on
the desktop list`, first three). Root SRVs (slot 44) are not forwarded under a deferred root. Translated draws use
their own root signature (no flags) and are unaffected.

A real draw under a deferred root is decided by `src/startup_stubs/heap_indexed_root.h`: it may bind only when the
list holds a mirror of every heap the root indexes, covering the indices the draw reads, and every root constant,
descriptor table and root descriptor can be set again. A one-entry sampler heap covers only index 0 and never stands
in for the view heap. The stand-in has no mirrored heap and cannot restore that state, so it refuses every real draw
under a flagged root, counts it by reason and logs `real draw refused: root signature heap flags 0x..., <reason>`
(first three per reason, then every 1000th). The `x2w-render totals` line reports `heap-indexed roots deferred N
(real draws refused: view heap, sampler heap, both heaps, state)`. On current main no source-built path reaches a
real draw (`compile_translation` returns no shader), so the refusal itself is checked only by the device-free
`startup_heap_indexed_root` matrix. `startup_root_heap_{warp,hardware}` runs the root CBV draw with the debug layer
on, for each flag alone, both and neither: a plain desktop list with the same root signature and no heaps must
produce one ID 1419 per flag first (otherwise the run skips with 77), and through the stand-in no such message may
appear while the translated draw keeps its colour.

## Render target binds and clears

Command-list slot 50 (desktop `OMSetRenderTargets` plus four) takes the render target handles either as an array or,
with `RTsSingleHandleToDescriptorRange` true, as the first handle only. In the single-range form the other handles follow
the first at the title's own descriptor stride (`title_descriptor_stride`, the same stride the descriptor copies use), for
a count of one as well as for several. The desktop call always receives the resolved array. Before this change a true
single-range flag dropped the bind without a message, so every later draw on the list was skipped as not drawn.

A bind is refused, as before, when a handle holds a view of the wrong kind, a depth view has one read-only plane, a depth
handle resolves to nothing, or the count is above eight. What changed is that the refusals are visible: each handle
that holds no view (render target or depth), and a count above eight, adds to `target binds N (unresolved M)` in
the `x2w-render totals` line and logs one line (the first six of each kind), for example
`OMSetRenderTargets refused: render target 1 (0x...) holds no view`. A depth pointer whose handle maps to no
descriptor still binds no depth, now with a count and a log line.

Slot 52 (`ClearRenderTargetView`) resolves its view the way the bind does: through the descriptor's shadow record,
so a render target view the title copied with `CopyDescriptors` or `CopyDescriptorsSimple` clears the resource it
now holds, with the creation-time table as the fallback for a handle with no record. A handle that resolves to nothing
adds to the unresolved count and logs one line (`ClearRenderTargetView skipped: view 0x... is not a known render target
view`); a depth view passed there is reported as not a render target view.

`startup_rtv_bind_{warp,hardware}` records the bind forms (array, single range with one and two handles, an empty view
inside the range, an empty view alone) and reads the counters through the `x2w_d3d12_render_target_stats` export,
then copies a view into an empty slot, clears the copy red and reads the target back. It does not draw. The export reads
the same counters the totals line prints.

## Depth stencil clears

Command-list slot 51 is `ClearDepthStencilView(view, flags, depth, stencil, rect count, rects)`. The slot is DERIVED, not
seen in a title log: it is desktop slot 47 plus four, between slots 50 (`OMSetRenderTargets`) and 52
(`ClearRenderTargetView`), which are both observed at desktop plus four. Before this change slot 51 fell through to the
`x2w-noop` stand-in, so a depth buffer that the title cleared was never cleared on the desktop list and a depth test
ran against whatever the new texture held.

The depth value is the fourth argument, so a real call passes it in `xmm3`. The generic noop entry takes integer
registers only and would lose it, so with `X2W_D3D12_LIST_NOOP=1` and render mode the list table points slot 51 at the
handler itself. The handler keeps the per-slot call counts the generic entry would have kept, but a frame trace
and a capture do not see it.

The handler reads the view through its shadow record as slot 50 does, moves the resource to `DEPTH_WRITE` and calls the
desktop `ClearDepthStencilView`. The decision is in `src/startup_stubs/list_clear_depth.h`: unknown flag bits and
the stencil bit of a format without a stencil plane are dropped; no flag left means nothing to clear (not counted); a
cleared depth that is not a number in [0, 1], a cleared plane that the view marks read-only, or rectangles that cannot be
read are refused, because the desktop list would report an error for them. A view that holds nothing adds to the
unresolved count of `x2w_d3d12_render_target_stats`; the other refusals add to the second counter of
`x2w_d3d12_depth_clear_stats(forwarded, refused)`. Each refusal logs one line (the first six); the first four calls log
their arguments. A forwarded clear also adds to `clears` in the `x2w-render totals` line. A closed list ignores the call.

One limit: a resource that an earlier bind in the same recording holds as `DEPTH_READ` (a read-only view) is moved to
`DEPTH_WRITE` by a clear through another view of it, and a later draw with the read-only view still bound would then
find the wrong state. No title call shows that pattern yet.

`list_clear_depth_check` runs the decision without a device; its two mutated runs (a stencil clear kept for a format
without stencil, a read-only plane cleared) must fail. `startup_depth_clear_{warp,hardware}` clears a 4x4
`D32_FLOAT_S8X24_UINT` texture through the stand-in and reads both planes back: both planes, depth only (the stencil
argument ignored), stencil only (the depth argument ignored), one rectangle, then the refusals and the closed list. It
does not draw. Before the first clear the planes are read once, and the check fails if they already hold what the first
clear writes.

## Composite provenance fixture

Issue #1628 asks whether a uniform colour left in a composite's destination came from the composite or was
already there. `tests/startup_composite_provenance.cpp` is a source-built model of that question, not a trace
of the title. One command list draws a 64×64 RGBA32F source:
- a blue base;
- a green centre block, `[16, 48)²`;
- a red corner block, `[0, 8)²`.

It then clears the destination to magenta and runs the composite, which `Load`s the exact source texel at
`floor(position) × scale + offset`. The source, the destination immediately before the composite draw and the
destination immediately after it are copied to readback buffers on the same list, behind one fence. A CPU
model built from the first two readbacks predicts every pixel of the third.

The three cases leave three different destination centres:

| case | scissor | UV | centre after |
|---|---|---|---|
| drawn centre | full | identity | green |
| clipped centre | top 16 rows | identity | magenta (retained) |
| UV selects the corner | full | constant texel (2, 2) | red |

Each case reports viewport, scissor, blend (off) and write mask. Two mutants run against the unchanged model
and must mismatch: the corner case with its UV moved to the source centre, and the drawn case with a write
mask of red and alpha only.

On an AMD Radeon RX 9070 XT (driver 32.0.31044.16) and WARP 10.0.26100.9818, Windows 11 build 26300.9974,
debug layer enabled, all 4,096 pixels of each case matched the model. The UV mutant mismatched 4,096 pixels
and the mask mutant 1,088, with zero debug errors or warnings
(`ctest --preset dev --no-tests=error -R '^startup_composite_provenance_(hardware|warp)$' -V`).

## What is measured

With the switches below, an x64 console executable on an AMD RDNA4 GPU (Windows 11, release
build of this branch):

- presents frames at about 29 fps through the D3D12 stand-in, and its translated UI draws reach the
  output window;
- reads the virtual gamepad and answers the first-run user prompt for one local user;
- streams data through the DirectStorage stand-in (about 20 000 requests in the first two minutes,
  zlib-inflated on the CPU);
- plays its intro video through the GPU-view mirror (colours are still wrong, see the issues);
- then stops in its texture-streaming path: either at a soft assert (an `int3` the title steps
  over when no debugger is attached), or, past that, on a black screen.

## Build

The stand-ins are part of the normal build. From a developer prompt:

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release --no-tests=error -R '^startup_stubs$' -V
```

The build produces `xmem.dll`, `d3d12_xs.dll`, `AcpHal.dll`, `XCRDAPI.dll`, `chad.dll`, `xapu.dll`,
`xg_xs.dll` and `GameInput.dll`. Without the switches, every export that has no implementation
traps: it logs `x2w-trap <dll>!<export>` with its argument registers and ends the process with a
code that names the export (see `src/startup_stubs/trap.h`).

## Run

Prepare a folder with the executable and its package DLLs yourself. Never commit or upload it.
Then:

```powershell
.\tools\startup\run.ps1 -Exe <folder>\<title>.exe -DataDir <package root> -LogDir out\startup\run1 `
    -Seconds 180 -Render -FakeUser -GpuMirror -ContinueAsserts 1000 -InputScript "4@70"
```

`run.ps1` copies the stand-ins next to the executable and starts `tools/startup/debug_launch.py`.
It runs the debugger with `out/pe-env/Scripts/python.exe` when that environment exists, because
`-HostCpu` needs `iced-x86` from `requirements-dev.txt`; otherwise it falls back to `python` from
`PATH`.
The debugger runs the process in a kill-on-close job with a memory cap, **offline**: file opens of
socket, NetBIOS and DNS devices are refused (`--offline 0` turns this off; do not).

A process that Windows creates under a debugger normally gets the debug heap: the loader sets the
heap tail-check, free-check and parameter-validation flags and the low-fragmentation heap is not
used, so every allocation and free costs more. The debugger starts the process with
`_NO_DEBUG_HEAP=1` unless the environment already sets `_NO_DEBUG_HEAP`; set `_NO_DEBUG_HEAP=0` to
keep the debug heap, for example to look for heap corruption. In one local startup run on Windows 11
the debug heap spent about 8.9 s freeing the nodes of a 99,776-entry map built by a single
`XMemMapPhysicalPages` call; without it the whole call took 0.46 s and the first DirectStorage batch
completed about 20 s earlier. `startup_debug` checks both settings on its probe.

A second local comparison on Windows 11 build 26200.9974 used a custom debugger
with the same runtime DLL, saved state and warm cache in both runs. Times are seconds
from process creation. The ordinary-heap run also recorded process audio levels.

| Startup milestone | Debugger default (variable unset) | Ordinary heap (`_NO_DEBUG_HEAP=1`) |
|---|---:|---:|
| First captured window frame, including black | 30.14 | 7.40 |
| First opening-movie open request | 132.98 | 86.72 |
| First language selection screen matched | 152.16 | 105.24 |

There was one run per setting. Measure the window and the first visible content
separately when repeating this comparison. Time from the first recorded button
release to language selection was about 17 seconds in both runs. For the public
heap-flag regression, run `ctest --preset dev --no-tests=error -R '^startup_debug$'`.

It writes:

| File | Content |
|---|---|
| `<log>.out` | why the run stopped: exception, registers, stack, likely return addresses, raw stack |
| `<log>/events.txt` | module loads and every exception with its time |
| `<log>/stderr.txt` | the stand-ins' logs (`x2w-d3d12`, `x2w-dstorage`, `x2w-xmem`, `x2w-runtime`, ...) |
| `<log>/files.txt` | every file the process opened |

`run.ps1` prints the first five lines of `<log>.out` and exits with the debugger's exit code: 2 when
the process could not be created, 0 when the debugger finished its report, whatever the process
did. Check with `python3 -B tests/startup_run_ps1_check.py` (ctest `startup_run_ps1`): on Windows it
runs the script on a stub debugger under Windows PowerShell and `pwsh`, elsewhere it checks the
script text only.

`tools/startup/live_view.py` follows a run in a second console. `X2W_TRAP_DUMP="exe+RVA:len,reg+off:len"`
adds memory dumps to the stop report.

`tools/startup/triage_run.py` summarizes a log directory (and optional `--shader-cache <dir>`)
into an allowlist-only Markdown report: known reason and outcome tokens, decimal IDs and counts,
hex HRESULT/format/flag codes, slot numbers and known shader-failure categories (`Device::slot70`
refusals, unhandled `x2w-d3d12` / trap slots, `x2w-d3d12-debug` severity/ID counts, `x2w-dstorage`
failures and status HRESULTs, `x2w-runtime` failing slots, and `.fail` shader categories). Any other
text is grouped as `other (text omitted)` and never copied; raw snippets stay in the local logs.
`Device::slot133` (the console placed-resource call) logs each handled placement with a trailing
`aliased` or `not aliased` field. Those lines are not counted as unhandled; they are counted by
outcome (`aliased`, `not_aliased`, `failed`) in a separate table, and a line with an HRESULT other than 0
(or an uncommitted address) still counts as `failed` and as an unhandled `Device::slot133` event.
Check with `python3 -B tests/startup_triage_run_check.py` (ctest
`startup_triage_run`).

## Launcher check

`tests/startup_launch_check.py` (ctest `startup_launch`) drives `tools/startup/launch.py` on the
source-built `tests/startup_launch_probe.cpp`. A finished run reports the child's exit code and saves
its stdout, stderr and trap log; a child that outlives `--timeout` is terminated; `--memory-gib` and
`--max-processes` stop an over-limit allocation and a helper process, and the same requests succeed
under larger limits; a missing executable is reported instead of looking like a finished run. When
`X2W_STORAGE_PATH` is unset and `S:\` does not exist, `launch.py` creates `<logdir>\storage` as a
per-run fallback (preserving `X2W_STORAGE_PATH` when set); it warns without deleting when
`X2W_SHADER_CACHE` points to a non-empty directory, and prints a fast-fail diagnostic line pointing
to `trap.log`, `X2W_STORAGE_PATH` and `X2W_SHADER_CACHE` on a sub-second non-zero exit. No GPU
and no game files are needed, and the `pr` preset builds the probe and runs the test:

```powershell
ctest --preset release --no-tests=error -R '^startup_launch$' -V
```

## zlib decoder check

`tests/startup_inflate_check.py` (ctest `startup_inflate`) runs the DirectStorage stand-in's decoder
(`src/startup_stubs/inflate.h`) through the source-built `tests/startup_inflate_probe.cpp`, and
`startup_inflate_status` (`tests/startup_inflate_status.cpp`) verifies `InflateStatus` classification
in `inflate_zlib_ex` (`ok`, `insufficient_output`, `corrupt_stream`, `bad_checksum`). CPython's
zlib encodes seeded synthetic data as stored, fixed and dynamic Huffman blocks, with matches up to
the 32 KiB window and a sync flush between blocks; each result must equal the input. A stream with
one byte too little capacity and a truncated stream must be rejected without writes past the
destination. On bare deflate data, a match that starts at the first output byte decodes and one that
starts a byte earlier is rejected before it copies anything. A hand-built dynamic block whose only
distance code is one bit long, the one incomplete code set zlib accepts, must decode with and without
the lookup table. Both also hold with input and capacity to spare.

The decoder looks Huffman codes of up to 10 bits up in a table filled when the code is built; longer
codes and the last byte of input are decoded one bit at a time. A differential case decodes 900
seeded random streams (random, repetitive, word and geometric-byte data; levels 0-9; all five zlib
strategies; some with a sync or full flush), of which 92 have codes longer than 10 bits. Each runs at
exact capacity, one byte short, truncated, and with one to three bits flipped: 3597 cases in all.
Status, size and bytes must equal those of the same probe built with `X2W_INFLATE_FAST_BITS=0`
(bit-at-a-time only), and those of CPython's zlib. The one exception allowed is a stream where
`inflate.h` decodes past an incomplete code set that zlib rejects; with this seed, no case needs it.
The negative control is the probe built with
`X2W_INFLATE_FAULTY_TABLE`, where codes exactly 10 bits long decode as the next symbol. It must
differ from the reference in at least one case; it differs in 265.

A boundary case decodes 57 small streams at every output capacity up to one byte past their size, and
truncated to every length at full and at a random capacity: 63994 cases. The streams hold matches at
every distance from 1 to 16 with lengths around 8, 16 and 32 bytes and up to 258, some after a sync
flush; stored blocks of 0, 1 and 300 bytes at each of the eight bit positions after a fixed block; and
one stored block of 65535 bytes. Status, size and bytes must equal the bit-at-a-time build and
CPython's zlib. The exception is a truncated stored block that would not fit either: the decoder
checks the block's length against the input first and reports `corrupt_stream` where zlib reports
insufficient output (809 cases). A window case decodes 5 streams of 96 to 160 KiB whose matches
reach the 32 KiB window: zlib's encoder keeps 262 bytes of lookahead out of its window and never reaches distance 32768, so three are fixed blocks written by
hand with matches at distance 32768 and lengths up to 258, the last match ending at the end of the
output. Each runs at exact capacity, one byte short, at the end of inner 32768-distance matches and
truncated: 52 cases, all equal to the bit-at-a-time build and to zlib. In every case the probe ends both the stream and the destination
at a no-access page, so a read past the input or a write past the capacity faults.

While at least 8 input bytes and 258 output bytes (the longest match) remain, the decoder takes whole
symbols from a 64-bit bit buffer that one 8-byte load refills, with no per-symbol bounds check. It
leaves anything else (end of block, codes longer than 10 bits, invalid symbols and distances) to the
bit-at-a-time loop, and gives back the whole bytes it loaded but did not use, so that loop, stored
blocks and the Adler-32 trailer start at the same bit position as before. The bit-at-a-time build
has neither this loop nor the 8-byte match copy and the one-memcpy stored block; it copies a byte at
a time. Three negative controls must each be caught: `X2W_INFLATE_FAULTY_MARGIN` enters the loop
one output byte short of the longest match and writes past a destination in the boundary case;
`X2W_INFLATE_FAULTY_INPUT_MARGIN` enters it with 7 input bytes left and reads past an input in the
random case; `X2W_INFLATE_FAULTY_DISTANCE` drops the loop's distance check and does not reject the
match that starts before the output when input and capacity are to spare. No GPU and no game files
are needed:

```powershell
ctest --preset release --no-tests=error -R '^startup_inflate$' -V
```

The same script measures throughput; it is not a ctest. It decodes 16 MiB seeded streams (stored,
fixed Huffman, dynamic Huffman word text, dynamic Huffman bytes of about 7.3 bits of entropy), three
per run, through the probe's batch mode and prints decoded MiB/s next to CPython's zlib on the same
data. Probes given together run interleaved; the first of six rounds is dropped and the best of the
other five is kept. Each run prints the sum of the Adler-32 values of its outputs, which must match
the input's, so the decoding cannot be skipped. Only time inside the decoder counts; the zlib column
includes CPython's call overhead. To compare two commits, build the probe of each and pass both:

```powershell
cmake --build --preset release --target startup_inflate_probe
.\out\pe-env\Scripts\python.exe -B tests/startup_inflate_check.py --bench out/build/release/startup_inflate_probe.exe
```

## DirectStorage destinations that are not writable

A request whose destination (a plain pointer, not 64 KiB page tokens) is not writable up to its declared capacity fails before any byte is written: the stand-in walks the destination with `VirtualQuery`, and a page that is uncommitted, read-only, `PAGE_GUARD` or `PAGE_NOACCESS` refuses the whole request. The log line ends `destination not writable`, `requests_failed` counts it, the next status entry reports `E_FAIL` and the queue goes on with later requests. Before this check, a destination shorter in memory than its declared capacity faulted inside the stand-in's copy (#870). The check covers the whole declared capacity, not only the bytes the request would write. Cost: one `VirtualQuery` per region the destination spans, per request that reaches the copy.

`tests/startup_dstorage_destination_check.cpp` (ctest `startup_dstorage_destination`) sends memory-source requests through the real queue: a read-only, a guard and a no-access middle page, and a capacity one page past the committed pages. Each must fail with the process alive and the bytes of the pages before the bad one still `0xAA`. Afterwards, requests into a fully writable (`PAGE_READWRITE`) buffer and a copy-on-write (`PAGE_WRITECOPY`) mapped view on the same queue must write every byte:

```powershell
ctest --preset release --no-tests=error -R '^startup_dstorage_destination$' -V
```

## Local switches, logging, and memory probing

`tests/startup_local_log_check.cpp` (device-free ctest `startup_local_log`, with negative
control `startup_local_log_guard_drop_control`) verifies `src/startup_stubs/local_log.h`:
`env_on` (a leading `1` in values up to three characters, preserving `GetLastError`), atomic
`first_calls` countdowns, `logf` (stderr and `X2W_TRAP_LOG` output, CRLF and 1533-byte payload
truncation, `MAX_PATH` boundary behavior, and `GetLastError` preservation on file-open failure),
`probe_readable` (invalid-pointer and zero-length refusals, multi-page reads, `PAGE_NOACCESS` /
`MEM_RESERVE` prefix truncation, and `PAGE_GUARD` re-arming verified by `VirtualQuery` and a
second probe), and `readable`/`writable` boundaries across `PAGE_READONLY`, `PAGE_READWRITE`,
`PAGE_WRITECOPY`, `PAGE_NOACCESS`, `PAGE_GUARD`, and `MEM_RESERVE`. No GPU is required:

```powershell
cmake --build --preset dev --target startup_local_log_check
ctest --preset dev --no-tests=error -R '^startup_local_log(_guard_drop_control)?$' -V
```

## Trap-by-default header check

`tests/startup_trap_header_check.cpp` (device-free ctest `startup_trap_header`, 41 cases, with
`WILL_FAIL` negative controls `startup_trap_header_prefix_control` and
`startup_trap_header_precedence_control`, Refs #1149) verifies `src/startup_stubs/trap.h`:
`exit_code` 8-bit module/index bit-masking (`0xE2570000 | ...`), `listed` comma-separated,
exact-match, wildcard `*`, and 2047/2048-byte buffer boundaries, `X2W_TRAP_EXPORT` behavior under
`X2W_TRAP_CONTINUE` (returns `0`, logs `x2w-continue`), `X2W_TRAP_FAIL` (returns `0x80004005`, logs
`x2w-fail`), continue-over-fail precedence, and child-process default `fire` termination (`exit_code`
and matching `x2w-trap` lines on `stderr` and `X2W_TRAP_LOG`). No GPU is required:

```powershell
cmake --build --preset dev --target startup_trap_header_check
ctest --preset dev --no-tests=error -R '^startup_trap_header(_prefix_control|_precedence_control)?$' -V
```

## Switches

| Variable | Effect |
|---|---|
| `X2W_XMEM_PERMISSIVE=1` | accepts the reservation, mapping and allocation shapes a startup run uses |
| `X2W_XMEM_STARTUP_BELOW_40=1` (`run.ps1 -StartupBelow40`) | startup compatibility configuration ([#1758](https://github.com/R0Y00/Xbox2Windows/issues/1758)): the large-page startup blocks with attribute `0x283` (null address, type `0x20003000`, protection `0x2004` or `0x2404`) are placed so that the whole block, last byte included, lies below 2^40. A block that does not fit there returns NULL with the Windows error and logs `x2w-xmem startup block below 2^40 refused:`; nothing is truncated or placed higher. Other shapes and explicit addresses keep their placement. See [Startup blocks below 2^40](experiments.md#startup-blocks-below-240) |
| `X2W_XMEM_GPU_MIRROR=1` | pairs a no-access graphics reservation with the read-write reservation of the same size made just before it, and maps every physical page into both (same section) |
| `X2W_D3D12_PERMISSIVE=1` | wraps the device: recording command lists, console present, view tracking |
| `X2W_D3D12_RENDER=1` | draws through translated pipelines from `X2W_SHADER_CACHE` and forwards [single-record indirect Dispatch](experiments.md#gpu-written-indirect-dispatch-arguments), [DrawIndexed](experiments.md#gpu-written-indirect-indexed-draw-arguments), [Draw](experiments.md#gpu-written-indirect-draw-arguments), and [DispatchMesh](experiments.md#gpu-written-indirect-mesh-dispatch-arguments) for supported desktop pipelines and tracked placed buffers |
| `X2W_SHADER_CACHE=<dir>` | translation cache: the runtime writes source requests `req_<key>.{a,b,root}` and, when needed, RTV variants `req_<key>.rtv`. Run `tools/startup/gfx2hlsl.py --pending <dir>` while the process is running. Active draws poll for an absent manifest at most once per second per pipeline and use the published translation on a subsequent draw. The translator compiles `.vs.cso` and `.ps.cso` with `fxc.exe`: `X2W_FXC`, else the Windows SDK 10.0.26100.0 copy, else the newest `Windows Kits\10\bin\<version>\x64`, else `PATH`. When none exists it writes `<key>.fail` naming `X2W_FXC` and no manifest; `X2W_FXC=` (empty) skips compilation and publishes the manifest alone. |
| `X2W_DRAW_NULL_IDENTITY=1` | null descriptors read as identity values instead of zero |
| `X2W_D3D12_DUMP_SURFACE=<bmp>` | writes the last presented surface |
| `X2W_DRAW_DUMP_FRAME=<n>`, `X2W_DRAW_DUMP_DIR` | Full-resolution RGB snapshots of single-sample RGBA8/BGRA8 UNORM or sRGB targets after translated draws. A later present writes completed queue-fenced copies. At most 48 copies are pending or written; a copy whose fence reports device removal (`UINT64_MAX`) is released unwritten and frees its slot (`src/startup_stubs/draw_dump_tracking.h`, `ctest --preset dev --no-tests=error -R '^startup_dump_fence_(removed_control_)?warp$' -V`; the control must fail). |
| `X2W_AUDIO_PERMISSIVE=1` | enables the audio stand-ins. Without a DSP provider, `XDspActivate` clears valid outputs and reports unavailable resources, and failed `XDspSubmitCommand` calls leave the sequence unchanged. |
| `X2W_RUNTIME_PROXY=1` | proxies the gaming runtime and logs every call |
| `X2W_FAKE_USER=1` | answers the user, gamertag and save APIs for one local user, and creates one offline store context for that user |
| `X2W_STORAGE_PATH`, `X2W_STORAGE_MB`, `X2W_TITLE_ID` | persistent local storage and the manifest values the runtime would return. The path is UTF-8, with a trailing separator added when missing. If the path is invalid or exceeds 259 bytes including the separator, the proxy logs the refusal and uses `S:\` (`src/startup_stubs/storage_root_utf8.h`, `ctest --preset release --no-tests=error -R '^startup_storage_root'`). |
| `X2W_INPUT_SCRIPT=mask@sec,...` | timed virtual gamepad presses (0x4 = A); keyboard and XInput pad 0 also work. Up to 256 entries and 4,095 characters; a longer value is ignored and logged, and dropped or malformed entries are logged (`ctest --preset dev --no-tests=error -R '^startup_gameinput_script_(long|entries)$'`) |
| `X2W_SAMPLER_HEAP=1` | `gfx2hlsl.py` translates a sampler-heap load with a run-time offset into `SamplerState qh[2048] : register(s0, space1)` and writes `smpheap 2048 1 fallback 2047`. The stand-in uses entries 0 to 2046 for the bound heap and reserves entry 2047 for unaligned or out-of-range offsets. It also accepts the original `smpheap 2048 1` manifest with all 2,048 entries available. [#1143](https://github.com/R0Y00/Xbox2Windows/issues/1143) |
| `X2W_XG_VERBOSE=1` | logs every texture layout query with its mip sizes |
| `X2W_XG_BC7_4K_S_LAYOUT=1` | enables the BC7 tiled layout, copy and draw upload when `X2W_D3D12_PERMISSIVE=1` |
| `X2W_TRAP_CONTINUE=<exports>` | lets listed trap exports return 0 instead of ending the process |
| `X2W_STOP_AT=<rva>[,...]` | the startup debugger stops at a main-image breakpoint at one of these addresses even while `-ContinueAsserts` steps over the others, and writes every thread's stack to `threads_stop.txt` |
| `X2W_DUMP_AT=<seconds>[,...]` | the startup debugger writes every thread's stack to `threads_<seconds>.txt` at those times, for a run that stalls before a watchdog or the timeout ends it |
| `X2W_VALLOC_MIN=<hex bytes>` | the startup debugger logs each virtual-memory allocation of at least that size to `files.txt` (`VALLOC`), with the requested address (0 for any, unreadable for an incomplete read), size, type, protection and the main-image and stand-in return addresses on the caller's stack |
| `X2W_HOST_CPU=1` (`run.ps1 -HostCpu`) | the startup debugger rewrites main-image instructions the host processor lacks at their first fault: see [Host processors without the console's instructions](#host-processors-without-the-consoles-instructions) |

With `X2W_RUNTIME_PROXY=1` and `X2W_FAKE_USER=1`, store-interface method 3
(`XStoreCreateContext` on provider `0dd112ac`) writes one context handle for the local user and
returns success. Other store methods are not answered: a call that still carries the local user
handle returns `0x80004005`.
[#1147](https://github.com/R0Y00/Xbox2Windows/issues/1147)

The startup shader cache key includes the translator revision. After updating from
an earlier translation revision, run the recording build once to emit
requests, run `python tools/startup/gfx2hlsl.py --pending <cache-directory>`,
then start the build again. The new key keeps the former bytecode and manifest
together under their original name. A 16-dword table load can contain two
independent image descriptors at offsets 0 and 32 from the load address.

## Game saves

With `X2W_FAKE_USER=1`, the game-save stand-in keeps containers and blobs below
`X2W_STORAGE_PATH`, in `x2w-gamesave`: one directory per container, one file per blob
([#1200](https://github.com/R0Y00/Xbox2Windows/issues/1200)). Opening a container whose directory
exists succeeds. Names must be one plain path component of at most 128 bytes (no separators, `:`,
`..`, control characters, device names or the reserved suffixes `.x2w-part` and `.x2w-old`, in any
case) and a blob is at most 64 MiB. The storage root itself is trusted; a container directory or blob
that is a symbolic link or, on Windows, a junction is refused by every operation.

Blob writes and deletes are kept with the update and reach the disk only when the update is
submitted (the public API makes the whole update atomic at submit). An update closed without a
submit changes nothing. At submit every write goes to a staging file and every delete is checked
first; then each old file is moved aside, the new one put in place and, on any failure, everything
is put back, so a failing action publishes none of the update. This holds for failures inside this
process.

Readers and publication share one cross-process lock per container, a file named
`<container>.x2w-lock` beside the container directory (the suffix is reserved like the other two).
The replacement of the blobs of a submit, a blob or container delete, and every read take it
(`LockFileEx` on Windows, `flock` elsewhere, in `src/startup_stubs/file_lock.h`). Publication takes
it exclusively and releases it only after the last blob is in place; `read_blobs` reads several
blobs under one shared hold, so a reader in another process sees the whole previous generation or the
whole new one, never a missing old blob or a mix. Staging files are written before the lock is taken,
so readers wait only for the renames. The wait is bounded (5 seconds by default, a `Store`
parameter): on timeout the operation returns `busy` and changes nothing, and a holder that dies is
harmless because the system drops its lock when its handle closes. This was preferred over a
manifest or generation directory published by one rename because it keeps the layout the earlier
commits of this work already write (one directory per container, one file per blob). A reader that
does not use this store (another program reading the files directly) is not covered.

Separate limitation, crash recovery: if the writing process is killed in the middle of the
replacement, the lock is released but nothing rolls the update back or completes it. The container
can then hold some blobs of the new generation and some of the old one, with `.x2w-old` and
`.x2w-part` files left behind, and the next reader sees that state. There is no recovery pass yet.

The slot numbers and argument layouts follow the public header order and stay a hypothesis.
Persisted: create container (16), delete container (10, 11), create and close update (23, 24), blob
write and delete staged in an update (25, 26) and published by submit (27, 28 with its result in 29).
Asynchronous read (21, with its result in 22) returns committed blobs. Slot 20 still reports no blob.
The enumerations and queries (12 to 15, 18, 19) still report nothing.
`ctest --preset dev --no-tests=error -R '^startup_gamesave_persist$' -V` checks the storage code the
slots call, not the slots: it writes in one process and reads in a second, including a close without
submit, a failing second action, a second process that sees only committed state, and a redirected
container directory.
`ctest --preset dev --no-tests=error -R '^startup_gamesave_publish_'` pauses a writer, in a second
process, at each step of the publication of a two-blob update (an old blob moved aside, a new one in
place) through a handshake of files. While it is paused another process must be refused with `busy`
(never shown a mix or a missing blob) and one that waits must see the whole new generation. The
existing failing-second-action control checks that the previous generation stays whole, and a lock
holder that ends without unlocking checks the bounded wait and the recovery. The `_unlocked` test runs
the same checks on a generated copy of the headers whose lock does nothing and must be caught (a reader
sees a mixed or incomplete generation). The pause points and that copy exist only in the test build.

Slot 28 first asks the Windows gaming runtime to accept the caller's async block. On rejection it returns before taking staged actions. Once accepted it commits the files and calls `XAsyncComplete` with the status for that block. Slot 29 retrieves the status with the matching operation identity. File publication is synchronous within slot 28. The caller's completion routine can run inline or through its selected completion queue. [XAsyncBegin](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasyncbegin) · [XAsyncComplete](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasynccomplete) · [PC GDK async interface slots](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/xgameruntime/gdk_table.rs#L57-L64)

`ctest --preset dev --no-tests=error -R '^startup_gamesave_submit_'` checks denied starts and retries, two overlapping blocks and completion that reads the file and slot 29 before slot 28 returns. The system variant uses the installed Windows gaming runtime with an immediate work/completion queue. It reports a capability skip when the runtime or queue is unavailable.

Slot 21 reserves the caller's async block on that same XAsync service, with its own operation identity (`local XGameSaveReadBlobData`). If the service rejects the block, slot 21 returns before it reads. Once the block is accepted, it snapshots the named blobs through `read_blobs` (one shared file lock for the set), packs them, and completes with the status and the number of bytes slot 22 needs. It does not use the user service, and no save lock is held across `XAsyncComplete`. The packed bytes are an array of 24-byte records, payload pointer at offset 16, then each name, a NUL and the payload. Slot 22 copies that into the caller's buffer and reports the count. A short buffer returns `0x80830007` and a missing blob returns `0x80830008`; neither writes the caller's count or bytes. A busy snapshot, and any other storage failure, uses the same status as submit (`0x80004005`); a bad name is `0x80070057`. [XGameSaveReadBlobDataResult](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xgamesave/functions/xgamesavereadblobdataresult)

`ctest --preset dev --no-tests=error -R '^startup_gamesave_read_async_'` commits two blobs in one process and reads them in a second through slots 21 and 22. It checks the completion callback, the required size, the count and the exact bytes, a missing blob, a short buffer that leaves the output unchanged, and two overlapping reads. The controlled fixture always runs. The system variant uses an immediate Windows gaming runtime queue and skips only when that runtime is unavailable. `ctest --preset dev --no-tests=error -R '^startup_save_blob_pack'` checks `src/startup_stubs/save_blob_pack.h` directly without a runtime or disk store, including layout, `uint32_t` payload-size and `size_t` overflow refusals, and a negative control for the earlier size check.

## Findings

These are measured on one executable; treat them as hypotheses for others.

- **User API, second interface.** The user provider hands out a second interface for the
  gamertag (method 3: handle, component, size, buffer, used). It must be answered separately from
  the main user object, whose method 3 is the handle duplicate. Without a gamertag, the first-run
  prompt offers no player and never advances.
- **GPU-view reservations.** The title reserves the same large range twice in a row: read-write,
  then no-access with the graphics flag. It maps physical pages only into the first, yet places GPU
  resources (for example the video frame planes) at the same offset in the second. On the console
  the second is a GPU view of the first. Mapping each page's section into both views makes those
  resources readable; without it, a YUV video shows a solid green screen (all-zero planes).
- **DirectStorage status arrays (#1843).** A new status array's slots are complete with `S_OK`, so
  `IsComplete` is true and `GetHResult` is `S_OK` for a slot nothing has been enqueued into.
  `EnqueueStatus` makes its slot pending (`E_PENDING`) until the entries before it have run, and leaves
  the other slots as they are. This follows Microsoft's Win32 implementation of the API in
  Xbox-GDK-Samples (MIT), `SimpleDirectStorageCombo/DirectStorageWin32/DirectStorageWin32WrapperStatusArray.cpp`.
  The `startup_stubs` check covers a fresh array, a slot held pending behind a fence signal while the
  others stay complete, and that slot's completion.
- **DirectStorage ordering.** Requests must run after `Submit`, in order with status and fence
  entries. Titles fill destination page-array entries after `EnqueueRequest`, and also replace
  entries filled earlier, so the whole page array is read again when the request runs. The
  heartbeat counts entries replaced after enqueue. The caller must keep the page array readable,
  and stop changing it, once the request is submitted: the worker copies it once when the request
  runs. If the array cannot be read then, the request fails before any destination page is
  written; a token captured at enqueue is never used.
- **Sub-dword addressing (SDWA).** The translator lowers the byte and word selects of the 32-bit
  integer VOP1/VOP2 SDWA forms: a source selection is zero-extended, or sign-extended under `sext()`,
  and the low bits of the result are placed in the destination field, with `UNUSED_PAD` zeroing the
  others and `UNUSED_PRESERVE` keeping them. The selection helpers match `tools/pixel_aot.py`, which
  already lowers the same forms. Float SDWA forms and a `dst_unused:UNUSED_SEXT` destination still
  stop translation rather than translating to something inexact. [#1141](https://github.com/R0Y00/Xbox2Windows/issues/1141)
  The VOPC SDWA compares (`v_cmp_*_sdwa`, `v_cmpx_*_sdwa`) of the existing 32-bit compare table lower
  too: i32 and u32 sources take their `src0_sel`/`src1_sel` field, sign-extended under `sext()`, and
  the compare reads the extended value as its type. f32 SDWA compares stay refused (#1920). The
  result goes to `vcc_lo`, the printed SGPR, or EXEC for `v_cmpx`, as in the non-SDWA forms. Class
  compares, f32 compares and any other SDWA modifier are refused. Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_vopc_sdwa"`: the controls evaluate every
  integer form against an integer reference, and the draws on WARP and hardware check 16 x 16 pixels
  that a `v_cmpx_le_i32_sdwa` on a sign-extended byte keeps or discards, with translations that drop
  the select or `sext()` as failing controls. [#1906](https://github.com/R0Y00/Xbox2Windows/issues/1906)
- **Soft asserts.** Retail code keeps `int3` instructions followed by a continuation. Without a
  debugger the title's handler steps over them; under the startup debugger use
  `-ContinueAsserts N`. One of them fires about ten times a second in the texture-streaming path
  (a chunked read whose length ends inside a compressed chunk); its cause is open.
- **Struct-returning methods.** When both register arguments of a wrapped method look like objects,
  the one on the calling thread's stack is the return buffer.
- **Queue Wait.** Slots 18 (`GetTimestampFrequency`) and 32 (present) put the console queue's
  Signal and Wait two slots after the desktop's, at 16 and 17. Slot 17 with a fence becomes a desktop
  `ID3D12CommandQueue::Wait`. After one, Signals on that queue complete when the GPU reaches them,
  so a fence never completes before the wait ends. Other queue slots that are given a fence are
  still treated as Signals. The slot numbers are inferred from that layout; no title call to slot
  17 has been logged yet. Run
  `ctest --preset dev --no-tests=error -R "^startup_queue_wait_"`. On Windows 11 build 26300.9974 both backends separate the two slots the way this mapping assumes: on the RTX 5080 Laptop (32.0.15.8142) and on the Microsoft Basic Render Driver, routing the wait through slot 15 leaves `G=1` before `F` is signalled because that slot acts as a Signal, while routing it through slot 17 leaves `G=0` until `F` reaches 1 from the CPU.
- **Resubmitted lists.** In render mode a closed list can be submitted again until its own Reset.
  Each title allocator stands for a pool of desktop allocators. The title's Reset resets the current
  one only if no list holds a recording on it and every queue has passed its last submit; otherwise
  the title continues on an idle allocator from the pool, or a new one. A recording therefore
  survives its allocator's Reset, as on the console. If the fence that tracks a submit cannot be
  created or signalled, that submit's allocator is never reset or reused again. Run
  `ctest --preset dev --no-tests=error -R "^startup_list_resubmit_"`.
- **Failed command-list Close.** A desktop list that fails Close cannot be reopened by desktop
  Reset. The renderer reports the failed HRESULT and excludes the failed recording from queue
  submission. The next Reset creates a fresh list of the same type on the requested allocator
  while keeping the console-facing handle. If creation fails, another Reset can retry. The
  resubmission fixture issues an invalid viewport command on a compute list while an older
  valid submit is waiting on a fence. It verifies the failed Close and wrong-type Reset, then
  closes and submits a recovered recording and lets the queue finish both valid submissions.
- **Upload arena fixture.** The standalone arena allocates 640 writable bytes with 16-byte CPU
  and GPU alignment, preserves them when a later allocation needs a new page, and translates
  only byte ranges inside one live allocation. The idle reset retires those ranges. An eventual
  command-list caller must wait for its recordings and GPU submissions to retire before reset.
  The fixture exercises allocation and retirement without queue submission or guest slot binding.
  Run `cmake --build --preset release --target startup_upload_arena` and
  `ctest --preset release --no-tests=error -R '^startup_upload_arena_' -V`.
- **Inline samplers.** Pixel shaders build sampler descriptors from constants (`s_movk_i32`,
  `s_bfm_b32`, 64-bit moves); the translator folds those into static samplers. A source-built
  pixel stage checks all four manifest words, rejects a mask width from a root constant, and
  decodes the folded words with the stand-in's static sampler path. The clamp sampler reads
  column 1 at x=1.25, while the wrap control reads column 0. Run
  `ctest --preset release --no-tests=error -R "^startup_inline_sampler_"`.
- **3D images.** `image_sample` with `dim:SQ_RSRC_IMG_3D` becomes a `Texture3D<float4>` sampled with
  a `float3` of the three address registers in order; 1D images are refused. A source-built pixel
  stage draws one texel of a 2x2x2 texture on WARP and hardware with DXC and FXC
  (`ctest --preset release --no-tests=error -R "^startup_image_3d_"`).
- **Explicit mip loads.** For 2D textures, 2D arrays and 3D textures, `image_load_mip` passes the final
  address register to `Texture.Load` as the mip level (`int4(x, y, slice, mip)` for an array, whose four
  registers are x, y, slice and mip). Ordinary `image_load` still reads mip zero.
  The source-built 2D draw reads distinct colors from mips zero and two, and a wrong-mip
  reference must fail at pixel (0, 0). The 3D case checks nonconsecutive address registers
  during translation. Run `ctest --preset release --no-tests=error -R '^startup_image_load_mip_' -V`.
  The array case is checked by `startup_image_2d_array_`: translation controls (also run on Linux) and a
  draw of a 4x4 texture of two slices with three mips whose cells all differ, with mutants that exchange
  slice and mip, ignore the mip, or shift it by one. The draw was written without a Windows run. Cube maps
  are still refused for `image_load_mip`.
- **Packed 2D load coordinates.** Plain 2D `image_load` with `a16` reads unsigned x and y from the low and
  high halves of one VGPR and reads mip zero. The packed form requires one address VGPR without an
  NSA extension. Plain loads refuse an empty `dmask`, `r128`, `lwe`, `d16`, `tfe` and reserved bits.
  The source-built startup mip-load check covers this translation on Linux.
- **2D array images.** `image_sample`, `image_sample_lz`, `image_sample_l` and `image_load` with
  `dim:SQ_RSRC_IMG_2D_ARRAY` use a `Texture2DArray<float4>` (manifest kind `tex2da`). The slice is
  the third address component, as `float3(u, v, slice)` for a sample and `int4(x, y, slice, 0)`
  for a load. Before, the stage was refused and every draw through its pipeline was skipped.
  Two source-built pixel stages draw one texel of a 2x2 texture with two slices on WARP and
  hardware with DXC and FXC; a slice fixed to 0 and exchanged coordinates fail it
  (`ctest --preset release --no-tests=error -R "^startup_image_2d_array_"`). Cube, multisampled
  and 1D images are still refused. Not covered: binding a single-slice 2D view to an array declaration
  in the draw path.
- **2D array depth comparison.** `image_sample_c_lz` and `image_sample_c` (sampler MAX_LOD 0) with
  `dim:SQ_RSRC_IMG_2D_ARRAY` take the reference, u, v and the slice as four address registers and lower to
  `Texture2DArray.SampleCmpLevelZero(smp, float3(u, v, slice), reference)` on the same constant comparison
  sampler as the 2D form. Before, they were refused with `image_sample_c[_lz] on SQ_RSRC_IMG_2D_ARRAY`.
  `python -B tests/startup_image_sample_c_check.py` translates the fixtures (words checked against `llvm-mc`) and
  fails seven edited array lowerings. `tests/startup_image_2d_array_check.py` also compiles and draws both
  comparison forms through `offscreen_check --startup-image-2d-array` against a two-slice `R32_FLOAT` 2D array
  (depths `0.25` and `0.75` at texel `(1, 0)`, reference `0.5`, `LESS` via the manifest `smpc` line) and requires
  always-slice-zero and swapped-slice controls to differ (`ctest --preset release --no-tests=error -R "^startup_image_2d_array_"`).
  `image_sample_c_l` stays refused (shader model 6.7).
- **Flat parameters.** `v_interp_mov_f32` reads a `nointerpolation uint4` copy written from the
  exported registers, so integer words reach the pixel stage unchanged. FXC folds `asfloat()` of a
  constant and flushes binary32 denormal patterns such as 1 or 2 to 0, which a float copy lost
  ([#1259](https://github.com/R0Y00/Xbox2Windows/issues/1259)). Colour exports of constant integer
  words still go through `float4` and have the same loss. Reproduce with
  `ctest --preset release --no-tests=error -R "^startup_flat_param_"`.
- **Non-fused multiply-add.** The gfx1013 `v_mad_f32`, `v_mac_f32`, `v_madmk_f32`,
  `v_madak_f32` and `v_mad_legacy_f32` forms round the product before the sum and flush
  binary32 denormals to signed zero at each step. The legacy form keeps its zero
  multiplication rule. The existing HLSL for `v_fma_f32` and `v_fmac_f32` does not
  guarantee a fused result. Run `ctest --preset release --no-tests=error -R '^startup_gfx2hlsl_mad_'`.
  [#1728](https://github.com/R0Y00/Xbox2Windows/issues/1728)
- **Stream-cache identity.** A cached zlib stream is served only to the exact queued
  continuation entries in the same DirectStorage queue. The lookup checks the enqueued request
  id, the source, the compressed size and the destination chunk size. A later independent
  request at a matching offset still decodes its own bytes or fails without touching its
  destination.
- **DirectStorage file ranges.** A nonempty file request checks its byte offset and length against
  the current file size before allocating the staging buffer. A range ending at EOF is valid.
  An overrun or short read fails the queue request and preserves the destination.
  `ctest --preset dev --no-tests=error -R '^startup_dstorage_file_read' -V` reads
  `src/startup_stubs/dstorage_file_read.h` at offsets above 2 GiB, across 4 GiB and at EOF above
  4 GiB in a sparse temporary file, and skips (77) where the temporary folder has no sparse files.
  The `startup_storage_directory_batch` fixture submits ten requests across two synthetic RDS5/ETPF
  files on one queue. In each file, moving the first absolute source offset by 896 KiB selects a
  different valid 28-byte zlib stream. Both return success and produce 4096 bytes. Comparing the
  entire page detects the wrong stream. Moving the second request by the same distance reaches
  EOF and leaves the destination untouched. A subsequent request using its directory offset succeeds.
  Reproduce with `ctest --preset release --no-tests=error -R '^startup_storage_directory_batch$' -V`.
- **DirectStorage refusal order (#1586).** One Submit carries, for a raw file read and again for a
  zlib stream, a valid request, a request at an offset equal to the file size, a request with a source
  size of `0x98cabbec` at a valid offset, and a valid request. The stand-in refuses the two bad ones with
  `read failed`, their whole 4 KiB destinations keep the poison pattern, their status entries report
  `E_FAIL`, and the valid requests before and after complete with exact bytes. Each status entry is
  followed by an `EnqueueSignal` fence, which runs after its own entry and before the next. The inputs
  are generated by the test. It does not add a per-file offset or a size-based lookup, and it does not show
  what the title expects for a refused request.
  Reproduce with `ctest --preset release --no-tests=error -R '^startup_storage_refusal_order$' -V`.
- **DirectStorage queue errors (#1843).** Queue slot 9 (`GetErrorEvent`) returns a
  caller-owned handle to the queue's error event. The first failed read signals the event.
  Slot 10 (`RetrieveErrorRecord`) copies the 656-byte x64 record with the failure count
  and the first failed request in queue order, in the layout the Xbox GDK reference gives
  for `DSTORAGE_ERROR_RECORD`: the result, then `CommandType`
  (`DSTORAGE_COMMAND_TYPE_REQUEST`, 0, for a failed read), then the request arm with its
  options, destination, source, cancellation tag and name. For a file request it records
  the opened filename and read offset. Only failed read requests are recorded. Retrieving
  clears the record and resets the event together, so a later failure starts a new
  notification. An empty record, on a new queue, on a second retrieve and for a
  success-only queue, has a zero count and `CommandType` `DSTORAGE_COMMAND_TYPE_NONE` (-1). The heartbeat continues to
  count failures by cause across all queues.
  [Microsoft's public sample header](https://github.com/microsoft/Xbox-GDK-Samples/blob/1f44bdabed6e340170e2c7d1007500e4dff897bb/Samples/System/SimpleDirectStorageCombo/DirectStorageWin32/dstorage_win32.h)
  · [Microsoft DirectStorage overview](https://learn.microsoft.com/en-us/xbox/gdk/docs/features/console/storage/directstorage/directstorage-overview#geterroreventretrieveerrorrecord)
  · [DSTORAGE_ERROR_RECORD](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/structs/dstorage_error_record?view=gdk-2604)
  · [DSTORAGE_COMMAND_TYPE](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/enums/dstorage_command_type?view=gdk-2604)
- **DirectStorage failure causes (#1198).** After its totals line, the heartbeat prints
  `x2w-dstorage heartbeat <n> s: failed requests by cause` with one count for each of
  `read-at-or-past-eof`, `read-crossing-eof`, `read-other`, `destination-capacity`,
  `destination-not-writable`, `zlib`, `bcpack` and `other`. The causes add up to `failed=` on the
  totals line. `other` holds unknown option bits, page grant and page write failures, a page array
  unreadable at run, an entry that threw and an entry never started. A refused file read is split
  by comparing its range with the file size: at or past EOF when the offset is not below the size,
  crossing EOF when the length is larger than the size minus the offset. No end offset is computed,
  so nothing wraps. The per-request text stays `read failed`, and statuses and ordering are unchanged.
  The `startup_storage_failure_counts` fixture compiles the stand-in into itself, submits two valid
  requests and eleven failing ones covering every cause, and checks the exact count of each cause,
  the heartbeat line text, the `read failed` text and edge ranges near 2^64. Its `one_bucket_control`
  expects every failure under one cause and must fail. The counts cover the run as a whole; they do
  not say which failure came first or what the title expects for a refused request.
  Reproduce with `ctest --preset release --no-tests=error -R '^startup_storage_failure_counts' -V`.
- **DirectStorage page-request ends (#1198).** After the failure causes, the heartbeat prints
  `x2w-dstorage heartbeat <n> s: page requests by end against the first 64 KiB page` with
  `inside-completed`, `inside-failed`, `at-page-end-completed`, `at-page-end-failed`,
  `past-completed` and `past-failed`. It covers requests with a page destination (option bit 21)
  and classifies each by `page_offset + destination_size`, formed in 64 bits so it cannot wrap:
  below 0x10000 is inside the first page, 0x10000 is at its end, above is past it. A destination
  size of 0 ends at its offset, so inside. A request is counted once, when it runs: completed when
  it is written, failed under the same rule that sets `failed=` on the totals line, including a page
  array unreadable at run and an entry that threw. An `empty` request (source size 0) is neither, and
  an entry that never started is not counted. Outcomes, statuses and the per-request text are
  unchanged. The `startup_storage_page_end_counts` fixture compiles the stand-in into itself, loads
  the xmem shim for page writes, submits page requests of every end that complete and that fail at
  each failure site, plus an `empty` and two linear-destination requests, and checks each count, that
  the page counts match the totals less the linear requests, the logged outcome of every failure, the
  heartbeat line text and edge ends where a 32-bit sum would wrap. Its `one_bucket_control` expects
  every page request at the page end and must fail. The counts describe a run's request shapes; they
  do not say why a request of one shape fails or what the title expects for it.
  Reproduce with `ctest --preset release --no-tests=error -R '^startup_storage_page_end_counts' -V`.
- **DirectStorage refused page request, consumer view (#1198).** The `startup_storage_consumer_pair`
  fixture follows one refused page-destination request from enqueue to the release of its pages,
  next to a successful request of the same shape. It loads the stand-in DLL and the xmem shim under
  `X2W_XMEM_PERMISSIVE`, takes two page tokens from `XMemAllocatePhysicalPages`, reserves a view
  with `XMemVirtualAlloc` and maps the pages into it with `XMemMapPhysicalPages`; every read goes
  through that view. Each page gets its own sentinel pattern. One Submit carries two zlib requests
  from memory with `page_offset` 0xF000 and `destination_size` 0x1000, so each ends at its page's
  end, each followed by its status entry and an `EnqueueSignal` fence. S inflates to exactly 0x1000
  bytes and R to 0x1800. Checkpoints: S's status is `S_OK` and R's `E_FAIL`; each fence runs after
  its own status entry and before the next request's; at each fence (read in the fence callback)
  and again after both, S's page shows its sentinel below 0xF000 and the exact inflated bytes from
  there, while every byte of R's page is still R's sentinel and no byte of R's destination equals
  R's stream prefix; the log names `inflated` for S and `destination capacity exceeded` for R. The
  pages are then unmapped and freed (the shim's live page backings drop by two), one page is
  allocated again and mapped at the same address, reads zero-filled, and a follow-up request of S's
  shape on the same queue completes with the exact bytes. Its `accept_refused_control` takes R's
  destination as data, expects R's stream prefix there, and must print that check's failure. A
  manual mutant that writes R's first 0x1000 inflated bytes before the capacity refusal fails the
  fixture on R's sentinel. The file-source refusals and their fence order are covered by
  `startup_storage_refusal_order`, non-writable linear destinations by `startup_dstorage_destination`,
  and a single short or missing-grant page request read through a mapped view, with one fill byte and
  no fence or release, by the `dstorage` case of `startup_stubs` (`tests/startup_stub_consumer.cpp`).
  The fixture shows what a consumer of the stand-in reads; it does not show what the title does with a
  refused request.
  Reproduce with `ctest --preset dev --no-tests=error -R '^startup_storage_consumer_pair' -V`.
- **DirectStorage allocation failure (#1199, item 3).** A queue worker runs each entry inside
  a handler that turns any exception, including `std::bad_alloc` from a buffer sized by a request
  field, into a failed request: the next status entry reports `E_FAIL`, the destination is
  untouched and the worker serves the following entries. The handler only copies a short reason
  into a fixed buffer; the log line is written after the handler has left and after the request's
  buffers were freed, so no formatting or allocation runs while the exception is in flight. The
  worker is a `CreateThread` thread whose stack is committed at creation (1 MiB, `worker_stack_commit`)
  instead of growing on demand, because when the system is out of commit an untouched stack page
  cannot be committed either. The original `0xc00000fd` (stack overflow) was not reproduced here,
  and the stack-commit cause is a hypothesis from reading the code, not a measurement. The
  `startup_storage_alloc_failure` check puts its own process under a job commit limit, submits
  a request that declares a 1.5 GiB inflated size, expects a failure status within a timeout and
  then a correct result for a normal request on the same queue. It returns 77 where the limit is
  not enforced.
  Reproduce with `ctest --preset release --no-tests=error -R '^startup_storage_alloc_failure$' -V`.
  A queue builds its work (three `std::deque`s, whose constructors allocate and can throw
  `std::bad_alloc` even under `new (std::nothrow)`) and starts its worker separately. Enqueue
  only records an entry, in the work when it exists, otherwise in a small record that needs no
  queue storage; it never completes a status entry or signals a fence. Submit retries the
  construction and the worker start (a committed-stack attempt, then a default-stack one). If the
  worker runs, the entries proceed as usual. If it cannot start, every recorded entry fails there:
  its status entry takes the pending failure (`E_OUTOFMEMORY` when nothing worse is pending) and
  its fence is signaled, so nothing is left at `E_PENDING`. An entry that could not even be
  recorded loses its request bytes, so it fails at Submit even if the worker starts then; only when
  the small record itself cannot be allocated is the entry failed inside Enqueue. No exception
  leaves Enqueue or Submit: a request whose page tokens cannot be copied is recorded the same way,
  and if Submit cannot grow the worker's list the entries still waiting fail there.
  Entry order is kept: once an entry has been deferred, every later entry until the next Submit is
  deferred too, so the stored entries are an in-order prefix and the deferred ones the in-order
  suffix. Submit handles the prefix first (worker or failure), then fails the suffix, so a failure
  reaches the next status entry after it and never stays pending behind a status entry that
  completed. Limits: when the prefix runs on the worker, a suffix status entry fails synchronously
  inside Submit and cannot see a failure the worker finds later in the prefix (a later status entry
  will); after the first allocation failure the rest of the batch fails with `E_OUTOFMEMORY` even if
  memory recovered, which is preferable to reordering.
  `startup_storage_worker_start` compiles the stand-in into the test, redirects `CreateThread`
  and replaces `operator new` with one that fails the N-th allocation once and then behaves
  normally. It sweeps N over Enqueue and Submit, and checks pending status and an unsignaled fence
  before Submit, then a request that completed with correct data or a failure, a fence signaled
  once, and a correct request on the same queue afterwards. A second control fails the N-th allocation
  of Enqueue and the M-th of Submit for 1 to 9 requests with the status entry before or after them
  and checks that a failure is reported by one status entry only and that a later clean request and
  status entry complete correctly. A crash names the stage it was in.

## Shortcuts that are known to be wrong

- A shader-visible CBV/SRV/UAV heap larger than the adapter accepts is created at the limit the
  adapter does take (measured once, then bisected) and the title's full-size memory stays the heap's
  CPU side. Desktop descriptor writes and copies use a CPU-only heap for the remainder. Translated
  graphics draws resolve their root tables against the bound logical heap and publish the requested
  records into the draw's private shader-visible heap. Changing the bound heap invalidates its
  tables, and a record extending beyond the logical end supplies a null view. This covers translated
  graphics draws; other consumers need their own binding path. Drivers refuse the request
  with different codes for the same condition (`E_INVALIDARG` on a GTX 1660 SUPER, `E_OUTOFMEMORY` on
  an RTX 3060), so any failure of an oversized shader-visible view heap takes this path. The measured
  limits so far are 1,000,000 descriptors on both adapters. On Windows 11 build 26300.9974 an RTX 5080 Laptop (32.0.15.8142) also reports 1,000,000, so the fixtures' 1,245,184-descriptor request takes the shortcut above and the remaining 245,184 descriptors stay in title memory. Reproduce with
  `ctest --preset release --no-tests=error -R '^startup_(view_heap|logical_view)_' -V`, which skips (77) on an
  adapter that accepts the full request. WARP 10.0.26100.9818 and an RX 6600 (driver 32.0.21045.29003)
  accept it, so those draws do not reach the table bounds there. `ctest --preset dev --no-tests=error -R '^startup_logical_view_table$' -V` checks
  `src/startup_stubs/logical_view_table.h` without a device. It covers the first and last records, one
  past the end, handles inside a record or below the heap, missing heap fields, memory that wraps past
  the top of the address space, and records crossing the end of a window.
- BCPack DirectStorage requests are decoded for BC7 and BC1 streams into destination memory as single atomic texture payloads; each request decompresses a complete multi-lane rANS chunk directly into its destination buffer and is excluded from chunk continuation caching. Unhandled formats fall back to zero-filled destinations. At process exit, one `x2w-dstorage BCPack summary` line reports request and zero-filled counts for every format 1..15 that was requested. The consumer fixture sends a format 6 request and checks that it succeeds, leaves a zero-filled destination, and appears in both counts; BC1 and BC7 requests must not be counted as zero-filled.
- The XG stand-in normally uses desktop copyable footprints. Its opt-in BC7 diagnostic
  reports standard GFX10 4 KiB tiled geometry for a single-slice, single-mip, non-MSAA
  `BC7_UNORM` 2D texture with an unknown layout, default alignment, and no resource flags.
  The diagnostic selects the profile explicitly. Native XG selector values still need
  independent evidence. `CopyIntoSubresource` writes the visible BC7 blocks from pitched source rows
  and preserves tile padding. In render mode, a successful copy marks the backing so the D3D12
  stand-in can detile BC7 blocks into the desktop upload footprint. The companion D3D12
  stand-in must be loaded for render-mode copies. The reported pitch describes the padded
  block-row extent of the surface. Synthetic copy and draw checks run with
  `ctest --preset dev --no-tests=error -R '^startup_xg_bc7_' -V`.
  Those checks use only accepted descriptions at 81x69 and 64x64 (square 2x2 and 1x1 tile grids),
  and the draw check tiles and detiles with the same block offsets. `tests/startup_bc7_4k_s_check.cpp`
  covers `src/startup_stubs/bc7_4k_s.h` without a device. It compares every block offset with the
  documented bit order on partial and non-square tile grids (200x72 is 4x2 tiles). It checks the
  padded sizes from 1x1 to 16384x16384 and detile at three row pitches with guard bytes. It also
  refuses each description with one field changed, such as two mips or 4 KiB alignment. Run
  `ctest --preset dev --no-tests=error -R '^startup_bc7_4k_s' -V`; the `_x_first_control` run
  checks an x-first order and must fail.
- The default translator uses a static default sampler for a dynamically indexed sampler-heap load.
  With the sampler-heap opt-in, translated draws snapshot the bound heap's 16-byte sampler entries and
  bind a private desktop table beside the static samplers. Missing, out-of-range, or unreadable entries
  use a transparent-black border sampler and have separate counters. The runtime keeps recorded tables
  alive through queue completion. Constant sampler-heap offsets still report an unsupported operation.
  Wrapped sampler heaps keep the 16-byte title stride separate from the desktop descriptor increment
  for CPU handles and descriptor copies.
  The source-built readback uses `ctest --preset dev --no-tests=error -R '^startup_sampler_heap_' -V`.
  The readback samples only wrap, clamp and border samplers along U. A device-free check covers the
  rest of the sampler word decoding (each address mode on each axis, filters, anisotropy and border
  colour), the `smpheap` manifest lines that are refused, snapshot counts and unreadable records, and
  snapshot equality: build `startup_translated_sampler_bindings_check` and run
  `ctest --preset dev --no-tests=error -R "^startup_translated_sampler_bindings(_shared_axis_control)?$" -V`.
  The control expects all three axes to use the U field and must fail.
- The startup shader translator lowers `image_sample_l` (explicit level, 2D and 3D, any `dmask`) to
  `SampleLevel(sampler, uv, level)`, taking the level from the last address register. Offset, compare,
  gather, clamp, cube, 1D and `unorm` forms are refused. Only the translation is checked
  (`python -B tests/startup_image_sample_l_check.py`); a WARP readback of distinct mip levels is still open.
- Slot 86 graphics streams can submit VS/PS requests to the offline cache. The fixture sends
  complete shader containers and a wrapped root through WARP, runs `gfx2hlsl.py --pending`, and
  reloads the compiled stages. A second fixture draws a clipped triangle through the translated
  pipeline, submits the wrapped list, waits for its fence and checks all 32 RGBA pixels on WARP and
  hardware. Swapping the red/green shader changes the 16 covered pixels. Reproduce with
  `ctest --preset release --no-tests=error -R "^startup_stream_(request|draw)_"`.
  Other stage combinations and full stream rasterizer/depth state remain open.
- Translated draws use the bound typed RTV formats to request colour-export variants. The
  manifest's `targets` line lists exported slots. A separate `rtv1` line identifies the base
  shader and the ordered RTV formats, and the draw rejects a variant with a mismatched line.
  Uncompressed `R32_UINT` and `R32_SINT` outputs use integer shader signatures.
  Uncompressed binary32 FLOAT outputs bitcast the export words after XOR with a private
  zero-valued entry constant so optimized FXC keeps denormal bit patterns. UNORM outputs still
  undergo float-to-UNORM conversion. Packed integer exports and other integer formats remain
  unsupported by this path.
- Colour-blend registers reach the draw path for pipelines created through device slot 57 only
  (#1216): state block A holds the eight `CB_BLENDn_CONTROL` words at +0xD0, and
  `src/startup_stubs/state_block_blend.h` decodes them into the translated pipeline's blend
  state. `CB_TARGET_MASK` is not located, so all four channels are written; no fixture covers a
  write mask on this path. A word the decoder refuses for any of the eight targets leaves the
  registers unused: the pipeline keeps the path it had before and the refusal is logged. No target
  is drawn unblended in place of a refused word
  (`ctest --preset release --no-tests=error -R "^startup_state_block_blend$"`).
  `tests/startup_slot57_blend_draw_check.py` draws through a source-built slot 57 pipeline: red
  with alpha 0.5 over opaque blue reads (128, 0, 128, 191) within one step with SRC_ALPHA /
  INV_SRC_ALPHA in target 0's word and (255, 0, 0, 128) with ENABLE clear or with the word four
  bytes before or after +0xD0. With two targets bound each follows its own word, and a refused
  word for the second leaves both unblended
  (`ctest --preset release --no-tests=error -R "^startup_slot57_blend_draw_" -V`).
  A graphics stream can supply a desktop `D3D12_BLEND_DESC` directly. `src/startup_stubs/cb_blend.h`
  decodes `CB_BLENDn_CONTROL` and a `CB_TARGET_MASK` nibble into `D3D12_RENDER_TARGET_BLEND_DESC`. It
  refuses forms D3D12 can't express (`BOTH_SRC_ALPHA`, constant alpha without alpha blend factor
  support, MIN/MAX with factors other than ONE). Its draws match a CPU model of the register fields
  (`ctest --preset release --no-tests=error -R "^startup_cb_blend"`). A device-free check decodes
  each 5-bit factor value in each of the four factor slots, each pair of colour and alpha combine
  functions, the MIN/MAX rule on each slot, the write mask and `control_word`, and checks the
  reason of each refusal. Its `_dest_alpha_control` run, which expects a destination alpha factor
  in its colour form, is the negative control
  (`ctest --preset dev --no-tests=error -R "^startup_cb_blend_decode" -V`).
- Audio produces no output from the console audio path. A separate desktop render sink exists
  (see "Audio output"); nothing connects the console path to it yet.

## Audio output

While a permissive XApu connection is open, the desktop shim requests 1 ms Windows
timer resolution for the mix worker's short refill waits. The last disconnect
releases the request. This affects scheduling and preserves the existing audio
routing and sample format.

With `X2W_AUDIO_PERMISSIVE=1`, `xapu.dll` tracks active handles across `XApuConnect`
and `XApuDisconnect`, refuses null outputs (`E_POINTER`) and unknown or disconnected
handles (`E_HANDLE`), and queues one completion entry per `XApuEnqueueCommand` call so
`XApuGetResultCount` and `XApuDequeueResult` retire commands cleanly (zeroing the
`0x58`-byte result buffer without fabricating unmeasured fields, and returning `E_FAIL`
when the queue is empty). The all-zero result is a placeholder: its type field (`+0`) is not a measured
completion value, and no caller has been observed consuming it. Entries are counters only, so memory does
not grow per command. Handles are never reused. Default non-permissive mode still traps on every export
(`ctest --preset dev --no-tests=error -R '^startup_xapu_queue$' -V`).

`src/startup_stubs/audio_sink.{h,cpp}` is a desktop WASAPI shared-mode render sink. It opens the
default render endpoint, takes the mix format from `GetMixFormat` (32-bit float or 16-bit PCM only),
converts each submitted block (format, frame count, bytes) to it, and writes it through
`IAudioRenderClient`. `convert_block` supports mono, stereo, 5.1, and 7.1 inputs (using ITU-R BS.775-1
fold-down coefficients with clamping when downmixing to stereo or mono) and linear sample-rate
resampling when the submitted rate differs from the endpoint mix rate (rejecting invalid rates,
unsupported channel counts, or empty blocks as `Submit::Unsupported`). Consumed frames are the submitted frames minus `GetCurrentPadding`; the
`IAudioClock` position is reported next to them. It records the first accepted block, the
submitted format, per-block frame counts and peak amplitude, the drained interval (queue empty
while the producer withholds data) and the recovery (resume until consumption advances). The
render API reports no discontinuity flag, so the glitch count is "not available" and is kept
apart from both intervals.

```powershell
cmake --build --preset release --target startup_audio_output
ctest --preset release --no-tests=error -R '^startup_audio_output$' -V
```

The fixture runs three cases: no submission (nothing accepted or consumed); a stream of eight
generated 40 ms tone blocks with distinct amplitudes, one of them all zeros (accepted, counted,
peak 0); and starvation (drain to padding 0, hold 300 ms, resume, consumption recovers). Exit
code 77 means no usable render endpoint (or an unsupported mix format). It prints the endpoint,
format and Windows build. A skip is neither a pass nor a failure. Every wait is bounded.

Not established: the layout of the console audio command queue (`xapu.cpp` is not decoded
here), what is audible (no capture-path check), and any hardware or driver glitch count.

## APU heap stand-in

With `X2W_AUDIO_PERMISSIVE=1`, `AcpHal.dll` serves `ApuAlloc` from the heap that `ApuCreateHeap` commits. The
heap is a first-fit free list in a static table of 4096 extents, guarded by one lock. `ApuFree` gives a
block back and merges it with free neighbours, so streaming code that allocates and frees repeatedly no
longer exhausts the heap. While nothing has been freed the offsets are the ones a bump allocator hands out.

- A block is aligned in address (a power of two, 16 at least; 0, anything below 16 and a value that is not a power of two
  all mean 16). Size and alignment are 32 bits. `ApuCreateHeap` creates and publishes the heap under one lock, so
  competing first calls cannot reset a heap that already has live blocks.
- Null output is `E_POINTER`; size 0 is `E_INVALIDARG`; no
  room, or a full extent table, is `E_OUTOFMEMORY`. A refused call leaves the heap and the optional
  APU-address output as they were, and writes null to the virtual-address output when non-null.
- A block that is reused is zeroed, as fresh heap memory is.
- `ApuFree(NULL)` is `S_OK`. An address that does not start a live block (never allocated, inside a block,
  already freed) is `E_INVALIDARG` and changes nothing.
- `ApuIsVirtualAddressValid` and the two map calls still test the whole heap range, not live blocks.

The shape-context allocation (`AcpHalAllocateShapeContexts`, types other than 4) returns `E_NOTIMPL`,
rolling back any already allocated XMA block and zeroing its output pointer and APU address. The source-built
check loads the stub, checks non-permissive traps, cycles 64 MiB through a 2 MiB heap, checks reuse, alignment,
coalescing, every refused call, shape-context allocation and partial-failure rollback, `AcpHalCreate`, and the
address validity/mapping exports, then runs 4 threads of 1000 rounds against an interval map of live blocks:

```
ctest --preset dev --no-tests=error -R '^startup_acphal_apu_free$' -V
```

## Uniform first-lane reads

The startup shader translator accepts `v_readfirstlane_b32` in straight-line code with the
entry mask when its active VGPR holds an integer conversion of a fixed-offset scalar buffer
value. Run `python -B tests/startup_uniform_lane_check.py` to check the producer and the
varying-source, register-clobber, mask and branch controls.

## Frame capture reader

`tools/startup/capture.py` reads the `capture.bin` the frame capture writes. A truncated file, a chunk
with a short header, or an unknown tag raises a `ValueError` that names the tag and offset.
`ctest --preset dev --no-tests=error -R '^startup_capture$' -V` checks every chunk layout against a
synthetic file.

## NV12 plane upload

For a single-mip, single-sample, even-sized 2D NV12 image the desktop stand-in
reports two linear footprints. XG copies luma and interleaved chroma separately,
and the draw uploader addresses both D3D12 subresources. NV12 arrays remain
unsupported. The source-built draw compares a synthetic image and its U/V-swapped
control against the same integer colour conversion on the CPU. This verifies
the stand-in path; the texture format used in the reported video remains to be confirmed.

```powershell
cmake --build --preset release --target startup_nv12_draw d3d12_xs_shim
ctest --preset release --no-tests=error -R '^startup_nv12_draw_' -V
```

`ctest --preset dev --no-tests=error -R '^startup_slice_footprints' -V` checks
`src/startup_stubs/slice_footprints.h` without a GPU. It covers NV12 descriptions refused for odd or zero
sizes, mips, layers, samples or dimension; the one-frame R8 and A8 plane and 15-frame re-read intervals and the
inclusive 32 MiB cap; the footprint request for 15 and 16 mips and without a device; and the bytes up to
the end of the last row. The padded-last-row control must fail.

## Host processors without the console's instructions

A console executable is built for one processor. On a host that lacks CLWB, CLZERO, MONITORX,
MWAITX or SSE4a, the first such instruction ends a startup run with an illegal-instruction
exception. With `X2W_HOST_CPU=1` the startup debugger rewrites the faulting instruction in the
running process and continues; the executable on disk is not changed, and each site costs one
debugger round trip.

| Instruction | Rewritten as |
|---|---|
| `CLWB m8` | `CLFLUSHOPT m8` (one ModRM bit); one `NOP` of the same length once a `CLFLUSHOPT` has faulted on this host |
| `MONITORX`, `MWAITX` | a 3-byte `NOP`; `PAUSE` and `NOP` (the wait may end at any time) |
| `CLZERO` | a jump to a stub that stores 64 zero bytes at `rax & ~63` |
| `INSERTQ xmm, xmm, len, idx`, `EXTRQ xmm, len, idx` (xmm0-7) | a jump to a stub that does the field move in integer registers |
| `MOVNTSS m32, xmm`, `MOVNTSD m64, xmm` | `MOVSS`, `MOVSD` stores (one opcode byte; the non-temporal hint is dropped) |
| `RDPRU` | `RDTSC`; `NOP` (the time-stamp counter stands in for MPERF and APERF, so their ratio reads 1) |
| `RDPID r64` | a jump to a stub that runs `RDTSCP` and moves its `ecx` to the register |
| `SHA1RNDS4`, `SHA1NEXTE`, `SHA1MSG1`, `SHA1MSG2`, `SHA256RNDS2`, `SHA256MSG1`, `SHA256MSG2` | not rewritten: performed by the debugger on the stopped thread at every hit, counted as `sha (per hit)` |

The stubs keep the other registers and the flags. Instructions that the 5-byte jump overwrites
are re-encoded behind the stub with iced-x86. `events.txt` names the first 40 rewritten sites and
the summary counts them by instruction. `tools/startup/host_cpu.py` lists the forms that are
refused (register-count `INSERTQ`/`EXTRQ`, xmm8-15, fields past bit 63).

`python -B tests/startup_host_cpu_check.py` assembles source-built functions, patches them the same
way and calls them on the host: 288 `INSERTQ`/`EXTRQ` results against a model, `CLZERO` with a
moved `lea` and a moved rip-relative load, and decoded `CLWB` and wait rewrites, including the
`NOP` form for every addressing mode. The fall-back from `CLFLUSHOPT` to `NOP` in the debugger is
not exercised here: it needs a host without `CLFLUSHOPT`. Stubs that zero 56
or 72 bytes, and a stub for another bit offset, fail it. On a processor that has SSE4a or CLZERO
the unpatched functions are compared as well. It exits 77 on a host that is not x86-64.

The SHA instructions cost a debugger round trip each, which is acceptable for an occasional
digest and too slow for bulk hashing. `python -B tests/startup_host_cpu_check.py` computes SHA-1
and SHA-256 of four messages through the seven operations and compares them with `hashlib`, and
`ctest --preset release --no-tests=error -R '^startup_host_cpu_debug$' -V` runs the debugger on a
source-built probe that executes SHA-256 and `RDPID` instructions.

Measured so far on one host without these instructions; the rewrites of `EXTRQ` have only run in
the check. A thread that is executing inside the bytes of a jump site while another thread faults
on it is not handled.

## Startup memory accounting

The DirectStorage heartbeat (`x2w-dstorage heartbeat <n> s:` in `stderr.txt`) reports this
process's peak and current commit charge in MiB and the number of live 64 KiB physical pages
(the heartbeat still calls them sections), so runs that stop at different points can be compared by
memory spent rather than by elapsed time. A physical page holds a slot from its first use (a mapping
or a DirectStorage write) until its grant is released; `xmem.dll` publishes that count through the
experimental `x2w_xmem_page_stats` export.

The slots are 64 KiB pieces of shared 256 MiB slabs (4096 slots each, `src/xmem_slab_slots.h`). A slab is
created with the default `CreateFileMappingW` commit behaviour, so all of it is charged to commit when it is
created, and it is closed only when the process ends: freeing pages returns slots, not commit. A released
slot is reused lowest first. The experimental `x2w_xmem_slab_stats` export reports the slabs open, the slots
in use and the slots free in the open slabs, which are the units that matter for commit now. This only
reports; it does not change when commit is charged. `xmem_slab_slots` checks the bookkeeping on any platform
without Windows calls. The heartbeat also prints while the request counters are unchanged
when either memory value changed, so the cost of a stalled load is visible. Each report is followed
by a line of failed requests by cause and a line of page-destination requests by end (see
DirectStorage failure causes and DirectStorage page-request ends under Findings).

```powershell
cmake --build --preset release --target xmem_shim startup_stub_xcrdapi xmem_slab_slots_check
ctest --preset release --no-tests=error -R '^(xmem_slab_slots|xmem_page_stats|startup_heartbeat_commit)$' -V
```

The `startup_heartbeat_commit` check starts the heartbeat before loading the page provider, then
maps two synthetic sections and compares the reported values with `GetProcessMemoryInfo` in the
same process. It releases a separate commit allocation to check that a current-only change
produces another report.

## Allocation sizes for depth with stencil and multisampled textures

A console executable reserves memory of its own for every placed texture, sized by the device's
allocation-info answer. The stand-in forwarded the desktop answer. For two families that answer
is larger than the console's: depth with stencil (the R32G8X24 family takes 8 bytes a texel on the
desktop adapters measured, where the console keeps a 4-byte depth plane and a 1-byte stencil
plane) and multisampled textures (4 MiB placement alignment on desktop, 64 KiB on the console).

For a single description of one of those families the stand-in now answers the console size
(`src/startup_stubs/console_plane_size.h`): each plane padded to whole 64 KiB tiles, where a tile
holds 64 KiB of samples (128 x 128 texels at 4 bytes, 256 x 128 at 2, 256 x 256 at 1, 64 x 64 at
4 bytes with four samples). The first eight uses are logged as `x2w-d3d12 allocation info:
console size ...`. The desktop texture behind the description is unchanged; only the amount of
memory the executable sets aside for it changes.

Measured on one console executable under the startup debugger, with an NVIDIA adapter through
vkd3d-proton: the executable commits one fixed block at start-up and takes its texture memory
from it. With desktop sizes the block was exhausted after 352 texture allocations, the next
allocation returned null, and the run stopped on a null object about 18 s in. Over the 1,561
distinct descriptions queried up to that point the desktop answers were 154 MiB larger in total
than the sizes above. With this change the same run passes that point. Whether native Windows
drivers return the same larger sizes is not measured here; an adapter that already answers 5
bytes a texel for the R32G8X24 family is left alone, because the console size is then not smaller.

The tile rule reproduces the nine reference sizes that `tests/startup_plane_size_check.cpp`
lists first. Those are console allocation sizes for descriptions the executable queries, taken
from the log of an independent implementation of the same device interface; no code was taken
from it.

The rule covers a 2D texture with exactly one mip, dimensions within the D3D12 limits, 1, 2, 4 or
8 samples at quality 0, the default layout and the default or 64 KiB alignment. Any other
description keeps the desktop answer. A desktop failure is kept for desktop formats; only the
console-only formats 118-120, which desktop cannot answer, are answered over it.

Covered formats: depth with stencil in the R32G8X24 family (formats 19-22), the R24G8 family
(44-47, a 4-byte depth plane plus a 1-byte stencil plane) and the console 16-bit depth with
stencil (118-120); multisampled colour in the R32G32B32A32 (1-4), R16G16B16A16 (9-14), R10G10B10A2,
R11G11B10 and R8G8B8A8 (23-32), R16G16 (33-38), R32 (39-43), R16 (53-59), R8 and A8 (60-65) and
B8G8R8A8/B8G8R8X8 (87-93) families. Both allocation-info entries (slots 25 and 108) give the same
answer for them.

Verification status: the device-independent reference sizes in `tests/startup_plane_size_check.cpp`
(`startup_plane_size`) cover R24G8 and the added colour families. They include the first and last
format of each multisampled colour family, B8G8R8A8 at 1920 x 1080, format 118 with four samples,
format 120, a description at 64 KiB alignment, a desktop answer of the same size at 4 MiB alignment,
and the zero-width, too-tall and too-many-slices shapes the rule refuses. `startup_plane_size_device_`
checks that the wrapper answers the helper's result for the adapter's own desktop answer; it passed
on Windows Server 2025 (WARP and the hardware case) in Debug and Release. It does not show that the
console layout fits the answered size for these formats; no console reference sizes are listed for
R24G8 or the added colour families.

- `ctest --preset release --no-tests=error -R '^startup_plane_size(_packed_control)?$' -V` checks
  the rule without a device, including the shapes it refuses.
- `ctest --preset release --no-tests=error -R '^startup_plane_size_device_' -V` goes through the
  wrapper: slots 25 and 108 must both give the helper's result for this adapter's own desktop
  answer, a placed texture (slot 133) is created over memory of the answered size and alignment,
  and descriptions outside the rule must come back unchanged. The placed texture is a committed
  desktop texture that does not alias the address, so this shows the answered size is accepted by
  the creation path, not that the console layout fits in it.

## Physical pages in shared sections

`XMemAllocatePhysicalPages` pages are 64 KiB slots of 256 MiB pagefile-backed sections. With
`X2W_XMEM_SHARED_VIEWS=1`, adjacent pages of one mapping call that are adjacent slots share one
view (`src/xmem_mapped_pages.cpp`). Before, every page had a section and a view of its own, which is
slow for a start-up batch of about 100,000 pages (measured under Wine: 42.9 s for 100,000 pages,
82 ms with shared sections and views).

Unmapping a page inside a shared view unmaps that view and maps its other pages again. A thread
that touches one of those pages in between takes an access violation. A vectored handler retries
it only when the address is inside a view that the unmap in progress, or the last finished one,
took apart, the page is recorded and covered by a view once that unmap is over, and the thread
has not already retried that page for that unmap. Any other fault is passed on, after at most one
retry.

Shared views are opt-in. The caller chooses before the first mapping call by setting
`X2W_XMEM_SHARED_VIEWS=1` in the process environment; the shim reads it once, on that call.

| | Default | `X2W_XMEM_SHARED_VIEWS=1` |
|---|---|---|
| Views | one per page | one per run of adjacent pages |
| Fault handler | none | registered by the first mapping call, never removed |
| `FreeLibrary` | unloads `xmem.dll` | module is pinned and stays until the process ends |

The handler is process-wide, and Windows has no removal that also waits for a dispatch already
under way, so the opt-in keeps the module, the handler and the shim's state in place for the life
of the process, also across a host's unload and reload. Nothing is torn down or waited for at DLL
detach. A host that needs to unload the shim keeps the default. If the handler cannot be
registered or the module cannot be pinned, the opt-in is refused, logged, and pages get a view
each.

What the shim records always matches what Windows shows. If an unmap is refused part of the way,
the views already unmapped are mapped again whole, nothing is removed and the call fails with
that error. If a surviving page cannot be mapped again, it is forgotten (its bytes stay in the
section), the batch is unmapped and the call fails. A freed slot is zeroed when it is reused, and
freed slots are reused in address order so that pages reused together share a view again.

`VirtualQuery` on a page of a shared view reports the view's base as the allocation base and the
bytes from the queried page to the end of the view, not one page. A placed buffer on such memory
is not affected: section-backed pages already took the committed, not-aliased path.

- `ctest --preset release --no-tests=error -R '^xmem_shim_(mapped_pages|page_runs)$' -V` runs the
  earlier contract check and `tests/xmem_page_runs_check.py`: a batch across two sections with
  holes, moved and reused pages; `VirtualQuery` before and after a view is split; two views of
  the same pages with writes seen both ways; an unmap refused at each of three steps
  (`X2W_XMEM_TEST_FAULT`); and a reader on a page while its neighbour is unmapped and mapped
  again. Two cases must end in an access violation, the reader without the handler
  (`X2W_XMEM_NO_FAULT_RETRY=1`, with `X2W_XMEM_TEST_HOLD_WINDOW` holding the unmap until the
  reader has read the unmapped page) and a read of a forgotten page; each runs in a child process
  whose exit code the parent checks. Those controls and `X2W_XMEM_TEST_FAULT` exist only in a
  second build of the shim (`xmem_shim_test_controls`, `X2W_XMEM_TEST_CONTROLS`); the check shows
  that the deployed `xmem.dll` ignores these switches. Four more children cover the lifetime on
  the deployed shim: without the opt-in, pages mapped with a view each and the shim unloading
  after that mapping, twice; with it, an unused shim unloads, a used one stays loaded through `FreeLibrary` with an
  unrelated handled access violation still delivered and the handler still working; `FreeLibrary`
  is called repeatedly while one thread reads a page and another toggles its neighbour; and the
  process exits with those two threads still running.
- `ctest --preset release --no-tests=error -R '^xmem_page_runs_device_' -V` places a buffer
  (slot 133) inside one view and across three kinds of boundary: two mapping calls, two backing
  sections, and single-page views.

## Buffer memory SRVs (device slot 70)

Device slot 70 (`src/startup_stubs/d3d12_memory_srv.inl`, validation in
`src/startup_stubs/memory_srv_buffer.h`) also accepts `D3D12_RESOURCE_DIMENSION_BUFFER`
descriptions with:

- `D3D12_SRV_DIMENSION_BUFFER` (raw `R32_TYPELESS`, structured `Format == UNKNOWN` with a stride that is a multiple of 4 up to
  2048, or typed buffers). It resolves an existing live placed (slot 133) or committed (slot 127)
  buffer covering `[address, address + (FirstElement + NumElements) * element_bytes)`, checks
  alignment and bounds, adjusts `FirstElement` by `(address - base) / element_bytes`, and forwards
  to `CreateShaderResourceView` (slot 18).

Acceleration-structure views (`D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`) are not
supported and are refused. A view needs an actual built structure with the placed AS role and
lifecycle checks, which has no device test yet.

Typed element bytes come from the format (1 to 16 bytes). Formats that cannot back a typed buffer SRV
(sRGB, BGRA, block-compressed, depth/stencil, typeless, planar) are refused with
`unsupported_description` and leave the destination descriptor unchanged. Translated draws bind typed
views with the format's element size and stride. The refusal reasons are listed in
`docs/experiments.md` ("Name the cause of a refused slot 70 call").
`ctest --preset dev --no-tests=error -R '^startup_memory_srv_report$' -V` checks the refusal line
without a device, with a distinct value in each 2D and 2D-array view field, the raw words of a
cube-array view, and widths, first elements, addresses and destinations above 4 GiB.

Checks:

`startup_memory_srv_buffer` is device-free and owns every rejection (bounds, alignment, format,
stride). It changes each buffer description and view field alone, checks the element bytes of every
typed format, views that end exactly at the end of the buffer, raw alignment of the adjusted first
element, and widths, addresses and first elements above 4 GiB. Its `_control` runs (`height`,
`exact-fit`, `first-32`) expect wrong answers and must fail. `startup_address_buffer_srv_*` exercises
only the resolve path with valid addresses on a device, so it does not cover the rejections.

- `ctest --preset release --no-tests=error -R '^startup_memory_srv_(report|buffer)$' -V`
- `ctest --preset release --no-tests=error -R '^startup_memory_srv_buffer_.*_control$' -V`
- `ctest --preset release --no-tests=error -R '^startup_address_(srv|buffer_srv)_' -V`
