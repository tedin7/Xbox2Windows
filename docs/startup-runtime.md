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

## Command-list UAV barriers

Render-mode slot 26 forwards unflagged UAV barriers on an open direct or compute list. A null
resource orders UAV accesses across resources. A named resource must be a wrapper around a local
desktop resource with UAV access. Transition and aliasing records are skipped because desktop
resource states and heap aliases are managed by the shim. The slot returns zero console actions.
The console slot and record layout are also documented by the
[ztrx command-list implementation](https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/list.rs).

Build `startup_present_producer` and run
`ctest --preset release --no-tests=error -R '^startup_list_uav_barriers_(warp|hardware)$' -V`.
The source-built caller submits both supported forms and checks the forwarding log, with refusal
controls for other barrier types, flagged barriers, invalid resources and arrays, and closed lists.

## Occlusion queries

Which console methods sit at command-list slots 28 to 30 is not established by any fixture here. The repository
places `ResourceBarrier` at 26 and four additional console methods in the 26 to 31 range
(32 is `SetDescriptorHeaps`, desktop 28 plus four). It also shows that
the plus-four run has ended by 61 (`ExecuteIndirect`, desktop 59 plus two). As a hypothesis, slots 28 to 30 are
implemented as `BeginQuery(heap, type, index)`, `EndQuery(heap, type, index)` and
`ResolveQueryData(heap, type, start, count, buffer, offset)`. Because that is a hypothesis, the bindings are a
diagnostic opt-in: only with `X2W_D3D12_QUERY_SLOTS=1` (in render mode) are the three slots bound. Unset, they stay
no-ops as before. The fixture below calls those slots directly, so it checks the forwarding, the refusals and the
end at Close, not the numbering.

The query heap comes from device slot 39 with the desktop desc, which the stand-in keeps. Only occlusion and
binary-occlusion queries on an occlusion heap, recorded on a DIRECT list, are forwarded to the desktop list. A
resolve is forwarded only when every index in its range was ended, with the same query type, on the current
recording; a new `BeginQuery` on an index clears its completion, and Close clears all of them. These calls are
refused and logged (`x2w-render ... refused`, the first 16 of each method): another command-list type, an index
outside the heap, an `EndQuery` without a `BeginQuery` on the same list, a resolve over an index not so ended, and
a resolve to anything but a committed readback buffer without title memory, or past that buffer's end. A query
still open at Close is ended there, since the desktop Close would otherwise fail and drop the list. The readback
buffer stays in `COPY_DEST` and is not moved through the list's state tracking.

Known limit: a resolve recorded on a later list than the `EndQuery` calls it covers is refused. Whether a caller
resolves on the recording that ended its queries has not been observed.

`startup_list_queries_{warp,hardware}` sets `X2W_D3D12_QUERY_SLOTS=1`. It resolves a query around a draw over an
8x8 target (64 samples) and an empty one (0) into a readback buffer filled with a sentinel, and reads it through
the console `Map`. It then checks that the refused calls, among them resolves over a never-begun index, a range
with a hole and an index ended with the other occlusion type, leave the sentinel untouched with no debug-layer
error, and it leaves one query open at Close. `startup_list_queries_default_{warp,hardware}_control` runs it
without the switch: the slots stay no-ops and it must report the untouched sentinel in the first word.
`command_list_slot_names.h` still names 56 to 58 as the query methods by the plus-four rule.

## Depth target on translated draws

List slot 50 (`OMSetRenderTargets`) passes a bound depth-stencil view to the desktop list in the same call
as the colour targets, and the translated pipeline takes its `DSVFormat` from that view. A depth handle that
holds 0 maps to no descriptor: the colour targets are bound without depth, and the stand-in logs it and
counts it in `x2w_d3d12_render_target_stats`. A depth handle in no descriptor heap, or one that is not a
depth-stencil view, refuses the whole bind (also logged and counted), so nothing is drawn onto targets the
title did not ask for. `ctest --preset release --no-tests=error -R "^startup_draw_depth_target_" -V` draws a
near and a far triangle with a depth-tested stream pipeline and reads back the colour and the D32 depth
values after the fence. With the view bound, the near triangle wins and depth is 0.25 where drawn. The
`_debug` test also requires no debug-layer message after the fence, and reports a skip (77) when the
debug layer is not installed. With a zero handle, the far triangle overwrites and depth keeps its clear
value. With an unknown handle, nothing is drawn. Each unresolved handle is counted once. The `_drop_control`
draws the bound case with a zero handle and passes only when the intended colour and depth mismatch appears.

Mesh pipelines reach the stand-in only as native desktop-bytecode streams, which must have a pixel stage
and no depth format. A depth-only mesh pass (no colour target, no pixel stage) is therefore refused when
the pipeline is created, and no fixture binds a depth target for one.

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
and step rate. A stream with a non-empty layout includes that text in its cache key; other keys are unchanged. A layout declared
with zero elements publishes nothing by default, so the fetch is refused and not drawn, like an absent or unreadable
layout. With the experimental opt-in `X2W_STREAM_ZERO_FILL=1` it is published as the single line `ia1 0`, and the
translation reads every fetched register as zero. That zero value is an assumption, not a measurement: no independent
reference shows what a fetched input holds for a zero-element layout, so no startup claim rests on it.
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

## Stream stage interface by semantic

In a slot-86 stream pipeline, the pixel shader's attribute N is not vertex parameter N. Each container states its
side of the interface in its native header:
- **Vertex shader:** from +0x1dc, one 6-byte record per exported parameter: kind 6, output-signature register,
  parameter index, then three zero bytes. The count is at +0x29c.
- **Pixel shader** (a header of 0xef8 bytes): from +0x138, one 8-byte record per attribute: kind 6, input-signature
  register, component mask, attribute index, two zero bytes, the register again, and interpolation 1 or 2. The
  count is at +0x238.

`gfx2hlsl.py` links each attribute to the vertex parameter whose output register carries the same semantics with the
same components as the attribute's input register. Raw programs without signatures still pair by number. A pixel shader that declares no inputs reads no tables, and any attribute read it makes is refused.

The translation is refused when:
- a record has any other values (these forms pack two signature registers into one parameter or attribute), or the
  pixel table lists one register twice;
- a pixel record's component mask differs from the input signature's mask for its register;
- a semantic of the attribute's register has no vertex output with the same semantic index and the same components.
  So a four-component vertex output that feeds a two-component pixel input is refused;
- the attribute's semantics come from more than one vertex register;
- either header has another size;
- the code reads an attribute the table does not list.

Graphics and compute use cache revisions `x2w-startup-shader-28` and `x2w-startup-shader-28cs`.
Revision 28 lowers `v_fma_mix*` through a `precise` `mad`, so a zero product keeps its sign.
Revision 25 preserves the sign of zero in binary16 multiply, including the packed and SDWA forms.
Revision 24 invalidates translations that revision 23 wrongly accepted: `vcc_hi_scalar_mask_read` now also
treats a whole-word lane-read writer (`v_readfirstlane`/`v_readlane`) of vcc_hi, in branch order too, as
making a later VOP3 lane-mask read of vcc_hi refuse instead of lowering to bit 0. The two cache keys advance together.
Revision 22 invalidates compute shaders that assumed the entry EXEC mask at a load reached on a later
loop iteration after an EXEC write. The two cache keys advance together.
Revision 21 refuses a compute lane-mask form that revision 20 translated (`s_and_b32 M, X, exec_lo` with X data).
Revision 20 decodes MTBUF/MUBUF format 50 (`R10G10B10A2_UNORM`) in the run-time FORMAT switch.
Revision 19 folds `s_mov_b32` from a scalar register that holds a plain constant, so later reads of the copy use the constant word. The fixed slot 57 mesh
(`x2w-startup-mesh-57-3`) key stays, because that shader is not translated from guest code.
Revisions before 19 read a register copied from a constant as a run-time value.
Revision 18 preserves the sign of zero in floating-point min, max, min3, max3 and med3.
Revisions before 17 lower `v_med3_f32` through a branchless median, which returns the smaller of the
other two operands when a source is NaN.
Revisions before 16 lower `v_fract_f32` through a bare `frac` instead of the clamped `fract32` helper.
Revisions before 15 lower `v_cvt_pk_u8_f32` with a float clamp and accept `s_setreg_imm32_b32` writes of the
rounding mode.
Revisions before 14 evaluate constant `s_brev_b32` at run time instead of folding it into a known word.
Revisions before 12 can overwrite the source mask when a save-EXEC instruction uses the
same source and destination register.
Earlier revisions can also carry a stale scalar symbol across a join, read a stale SCC after
an operation on a descriptor word, or lack the vertex buffer-store lowering.

The record layouts were checked offline against the signatures of one title's shader archive (local files only, not
in the repository). The counts below apply the translator's own record checks.
- **Vertex shaders:** the population is the 2,041 programs whose native image inflates. 1,536 have only simple
  records. Of the 1,983 whose code also decodes, 1,496 have simple records that match the output signature register
  for register, with parameters in register order. 505 use packed records and are refused when a pixel stage with
  inputs needs them.
- **Pixel shaders:** the population is the 10,862 programs whose image inflates and whose input signature declares
  a non-system input. 5,975 have only simple records, and each record's mask equals the signature mask. 4,887 use
  packed or other records and are refused. The 1,239 without inputs need no table.

The translator-only cases need no device: `python -B tests/startup_stream_interface_check.py`, or
`ctest --preset release --no-tests=error -R "^startup_stream_interface_translate$"`. They check that attribute 0
reads the TEXCOORD parameter, and they check each refusal. The draw is
`ctest --preset release --no-tests=error -R "^startup_stream_interface_(warp|hardware)$" -V`, after building
`startup_stream_consumer`. The vertex shader exports a constant COLOR as parameter 0 and a varying TEXCOORD as
parameter 1. The pixel shader reads only TEXCOORD, as attribute 0. On an 8×4 target, red follows TEXCOORD column by
column. The same pipeline translated with pairing by number reads COLOR and fails the check.

## Bounded stream shader exits

The slot-86 vertex and pixel readers decode the bounded instruction span before
counting exits. They accept one through eight reachable exits and validate
direct branch targets against instruction boundaries. In a multi-exit pixel shader with outputs, a
path that ends without writing any output discards the pixel. A vertex
exit without a position export is refused during lowering.

Run `python -B tests/startup_stream_two_exit_check.py` for the synthetic
translation and refusal checks. On Windows, run
`ctest --preset dev --no-tests=error -R '^startup_stream_two_exit_(translate|warp)$' -V`
with the stream consumer built. The WARP test compares all 32 pixels per draw
with a CPU reference. One shader takes three exits across four root-constant
inputs, another uses an end-program word as literal colour data, and a vertex
shader takes two exits. The literal-counting mutant must refuse extraction.
The fall-through mutant must change all 32 pixels from green to yellow.

## Output window and fallback diagnostics

The desktop swap chain has its own top-level window, class `X2WOutputWindow`.
Its title starts with `Xbox2Windows <rev>`, where `<rev>` is `git rev-parse --short=8 HEAD` of the
source tree the shim was built from, with `-dirty` for uncommitted changes to tracked files, or `unknown`
without git. The `D3D12XboxCreateDevice` log line ends with the same `build=<rev>`. The build refreshes
the revision on every build (`cmake/x2w_build_rev.cmake`). `ctest --preset dev --no-tests=error -R
'^startup_build_rev_' -V` checks the generator and the built `d3d12_xs.dll`.
Set `X2W_PRESENT_RECENT_FPS=1` to show FPS for the latest 30-frame block first and run-average
FPS second. The caption still updates every 30 frames when the window shows the frame. The first
update uses the 29 intervals from frame 1 to frame 30. The switch defaults to off.
The WARP fixture presents 30 unpaced frames followed by 30 frames paced at 250 ms each.
It checks the frame 60 caption and compares the presented source surface with a CPU RGBA8 reference.
With the switch off, the pixels must still match and the legacy caption must fail the two-rate check.

```powershell
ctest --preset release --no-tests=error -j 1 -R '^startup_present_recent_fps_warp$' -V
```

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
With `X2W_HIDE_TITLE_WINDOW=1` (exactly `1`), a 500 ms timer on the output window's thread hides the
title's own visible, unowned windows while presents reach the output window, and hides them again when
the title shows them. The hide is asynchronous (`ShowWindowAsync`), so a busy title thread does not
stall the output window. Dialogs (class `#32770`) stay visible. When the latest present did not reach
the output window, the windows are shown again, so that the GDI fallback above has a window to draw its
status into. When a hidden window was in front, the output window is asked to take the foreground
before the hide, and the log says so once when that is refused. Whether keyboard input still reaches
the title after the hide has not been measured. The switch is opt-in and off by default; without it no
timer is created. `startup_overlay_window` checks the helper (`TitleWindowHider` in
`src/startup_stubs/overlay_window.h`) with source-built windows: the switch is off when unset, `0` or
`10`; the timer is armed only with the switch and its `WM_TIMER` arrives; nothing is hidden before the
first shown present; the guest is hidden twice and shown again after a present that was not shown; the
presenter, its owned window and a dialog stay visible. Where the host lets the test set the foreground,
it also checks that the foreground moves to the output window. A run under Wine did; the fixture prints
which case applied.

### Presented planes

A console present (CommandQueue slot 32) carries a plane count and an array of plane records
(`src/startup_stubs/present_planes.h`). Measured from the first presents: one record is 0x48 bytes,
with +0x00 a frame token, +0x08 the resource count and +0x10 a pointer to the resource array. On the
Series S profile a title presented two planes: plane 0 at 1920x1080 stayed black and plane 1 at
3840x2160 held the picture, with alpha 255 wherever it drew. Showing plane 0 alone gave a black
window. Why plane 0 stays black is not known.

By default the output window still shows plane 0 alone, as before the planes were decoded. The
experimental `X2W_PRESENT_COMPOSITE=1` makes it composite the planes instead
(`src/startup_stubs/plane_compositor.h`, shared by the window presenter and its fixture). It stays
opt-in until a matched comparison on the same executable, with this change alone, shows what it does
to a real present. The contract under the switch:

- Order: plane 0 first, each later plane over it, in plane order (`composite_steps`). At most four
  planes are read.
- The lowest drawn plane is opaque: its colour replaces the window and its alpha is ignored.
- Each later plane is blended by its own alpha, not premultiplied:
  `rgb = src.rgb * src.a + dst.rgb * (1 - src.a)`, `a = src.a + dst.a * (1 - src.a)` (D3D12
  `SRC_ALPHA` / `INV_SRC_ALPHA`, alpha `ONE` / `INV_SRC_ALPHA`). Alpha 0 leaves what is below,
  alpha 255 replaces it, and the window alpha stays 255.
- Left out: a plane whose record or resource array is unreadable or empty, a surface that is not a
  known resource or that the presenter cannot draw (`present_view_format.h`), and a repeat of a
  surface already drawn. If plane 0 is left out, the next drawn plane is the opaque base.
- Each plane is scaled to the whole window with a linear, clamped sampler. The planes' source and
  destination rectangles in the record (after +0x10) are not decoded; a plane that covers only part
  of the screen on the console is stretched here.
- The baseline capture is unchanged with or without the switch: the frame capture, the presented
  signature, `X2W_D3D12_DUMP_SURFACE` and the `x2w-render presented` readback all read plane 0's
  first resource, as before. Under the switch the highest plane above plane 0 with a known resource
  is also sampled on the same frames, before compositing, and logged as a raw plane:

  ```text
  x2w-render raw plane <i> (not composited): surface <w>x<h> format <f>, sample <k>, mean level <l>/255, mean alpha <a>/255
  ```

  Neither line is the composited window picture.

The first presents and every 300th log each plane:

```text
x2w-present frame <n> plane <i>/<count>: <width>x<height> format <f> flags=0x<flags>
```

`ctest --preset dev --no-tests=error -R '^startup_present_planes(_control)?$'` checks the record layout
and the composite order on synthetic records; the control reads 0x40-byte records and composites
plane 0 alone, and must print `FAIL:`.
`ctest --preset dev --no-tests=error -R '^startup_present_composite_(warp|overlay_removed_control_warp)$'`
draws source-built planes through the same records, `composite_steps` and compositor (directly, without
the switch) into a 64x32 RGBA8 target on WARP and reads every texel back: a single plane is shown as is (including a cell with alpha 0); a black
plane 0 under a plane with alpha 0, 128, 5 and 255 cells and a third plane with opaque and
half-transparent cells gives 0, 128, 5 and 255 levels and the plane-order result 128,0,127 for half
red over blue (within 1 for blended cells, exact otherwise); a repeated surface and an empty record
are left out. The control runs the same cases, then removes the overlays and composites the black
plane 0 alone. It passes only on `control ok: baseline matched, ...`, printed when the baseline and
stacked cases matched, every fence completed without device removal, and the overlay-free picture is
black and differs from the stacked expectation in exactly the six cells the overlays draw. Any `FAIL:`
line fails it, including queue, compositor, target or plane setup, a fence that does not complete and
device removal. Both exit 77 without a WARP adapter.

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

## Unknown and mismatched view dimensions

A view description with `ViewDimension` 0 (UNKNOWN) makes desktop D3D12 remove the device instead of
failing the call, and so does a dimension that does not suit the resource (debug layer: "The
ViewDimension in the View Desc is incompatible with the type of the Resource"). Measured: a console
package module made such SRVs on 2D RGBA8 textures. Views of device slots 18 to 21 (shader resource,
unordered access, render target, depth stencil) are now checked before they are forwarded:

- an SRV (slot 18) with dimension 0 gets the resource's default view (a null description) where
  desktop allows one: a texture with a typed format that is not a depth format. The default view
  replaces the whole description, including its format and component mapping. Only this SRV case is
  measured. With dimension 0 every other view is skipped;
- a dimension whose resource dimension differs from the resource's is skipped.

The default view for UAV, RTV and DSV descriptions with dimension 0 is an explicit experiment,
`X2W_D3D12_VIEW_DEFAULTS=all`: it is unmeasured, and it drops the requested format the same way.
`X2W_D3D12_VIEW_DEFAULTS=off` skips the SRV case too. Any other value, or none, keeps the SRV
default.

A refused SRV, for any reason, writes a null SRV (R8G8B8A8_UNORM, the requested dimension where it
is a texture dimension, else 2D) into its destination descriptor. Otherwise the slot would keep the
view it held before. That view would still be bindable and would be copied by `CopyDescriptors`,
although the slot's record no longer keeps its resource alive. Translated draws already resolve the
slot to nothing.

A refused UAV, RTV or DSV leaves the previous descriptor and its tracking record in place.

The first of each are logged (`x2w-d3d12 Device::slot<n> view dimension ...`), and when a view still
removes the device, its forwarded description and the resource's shape are logged
(`removing view desc`). Run
`ctest --preset dev --no-tests=error -R '^startup_view_dimension(_control)?$'`
(`src/startup_stubs/view_dimension.h`); the control swaps the resource dimension of the mismatched
cases for the matching one and must print `FAIL:`.

`startup_view_dimension_warp` goes through `d3d12_xs.dll` on WARP. A compute shader samples each
descriptor of a shader-visible heap through the wrapper's desktop heap. The fixture checks three
cases. An unknown-dimension SRV of a texture filled with one distinct colour must read back that
colour. An explicit TEXTURE3D SRV of a 2D texture must be refused with the device not removed.
A descriptor slot that held a valid view of a second colour, then receives a refused TEXTURE2D
SRV of a buffer, must read zero, and so must a `CopyDescriptorsSimple` copy of it. The slot gate
for UAV, RTV and DSV is checked without a device by `startup_view_dimension`.
`startup_view_dimension_off_control_warp` runs the same fixture with
`X2W_D3D12_VIEW_DEFAULTS=off`. It must print `FAIL: (a) default view` with `device not removed`,
so it fails on the missing default view and not on setup or a removed device. Both return 77
without WARP.

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

## Shared-exponent render and unordered-access targets

The packed-target mapping also covers single-sample 2D textures and arrays in
`R9G9B9E5_SHAREDEXP` (67) that request a render target, a typed UAV, or both.
When the adapter lacks one of the requested uses and supports all of them for
`R11G11B10_FLOAT` (26), the stand-in creates format 26. Sampled-only resources,
depth-stencil, multisampled textures and other dimensions keep format 67.
The format query also preserves native format 67 where it supports the requested
uses. [Microsoft Direct3D format specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12R9G9B9E5Format.html)

The stand-in carries a format-67 tag. RTVs, UAVs and SRVs naming 67 on that
resource use format 26, including SRVs reconstructed by the translated draw.
An optimized clear value naming 67 follows the replacement format. A format-67
view on a native format-67 texture is passed through. A format-67 view on an
incompatible texture is refused.

Both formats use 32 bits per texel and clamp negative values to zero.
Format 26 has 6/6/5 mantissa bits with a separate exponent per channel.
Format 67 has three 9-bit mantissas with a shared exponent. Rendered and
UAV-written values can round differently. Uploads and copies in the format-67
bit layout are not converted.

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

With `X2W_BIGBIND_ONCE=1`, translated large raw descriptor bindings keep their
validated length and readable verdict until Present, including when an upload
reservation fails. The next draw retries the reservation. The memory range must
remain valid through the frame. The first successful copy supplies the snapshot
for later draws in that frame.

`X2W_BIGBIND_POOL=1` enables dedicated upload buffers for those snapshots.
Each new buffer is sized to the trimmed length rounded up to 1 MiB. The pool
holds up to 32 buffers and defaults to a 512 MiB cap, set by
`X2W_BIGBIND_POOL_MIB`. A buffer can be reused after Present, after every
referencing command-list recording is discarded, and after the last-use fence
on every submitting queue completes. If the pool cannot supply a buffer, the
draw tries the upload ring. The render totals at every 300th Present include
cumulative trims, copies, hits, copied bytes, and failed reservations for this
cache.

`startup_frame_copy_once_{warp,hardware}` exercises the pool with four draws
per frame and a rewritten range in the second frame. Its readback checks the
pool, ring fallback, saturation, and pool-off paths. A fixture-only mutant drops
the failed cache entry and must report four trims per frame instead of one.

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

The `x2w-render totals` line reports `null bindings N` and its six causes: `entry`,
`buffer`, `token`, `wrapper`, `dimension`, and `ring_full`. A ring-full event
contributes to these totals when the upload failure refuses a translated draw.

The always-on `x2w-render null input` lines show cumulative calls and first/last
frames by pipeline key, shader input index and cause. The 512-row census
keeps the first event's input kind, render-target-0 and depth-view formats,
and cause details. Each totals or 10-second report emits up to 24 changed
input rows ranked by calls. A separate 64-row census counts recorded translated
draws by render-target-0 resource since the previous census report. Each
`x2w-render target` line gives the frame span, total draws and draws that
bound at least one null input. A refused draw contributes no target draw.
The overflow line gives cumulative input events and window target draws
that could not fit in their respective tables. The tester archive includes
these rows and the percentage of draws with null inputs per target window
in `null-inputs.txt`.

The WARP fixture checks synthetic pixels with valid inputs, an unknown
texture token and an unreadable buffer. It also fills the upload ring and
checks that a clear survives the refused draw. A second unknown token on
a different input exercises the key's input-index field. The control
requires the readbacks and baseline census to pass before checking for
the expected index failure.

```powershell
ctest --preset dev --no-tests=error -R '^startup_(null_cause_labels(_buffer_token_control)?|null_views_census(_omit_index_control)?_warp|tester_pack)$'
```

### Refused slot 70 calls and token nulls in the five-second report

Three census lines follow the five-second report whenever their counts changed
(`src/startup_stubs/memory_srv_census.h`):

- `x2w-d3d12 memory SRV refusals (slot 70): total N; <reason> n, ...; distinct shapes K` counts every
  refused slot 70 call by reason. It then prints one `x2w-d3d12 memory SRV refused shape i:` line per
  distinct `reason`, dimension, format, size, mips, layout and flags, for the reasons that depend on the
  description (all but `null_destination`, `unreadable_resource`, `unreadable_view`, `null_address`,
  `destination_outside_view_heaps` and the acceleration-structure holds `as_unaligned`, `as_not_placed` and
  `as_wrong_state`). The reasons include the partial-tile holds (`partial_tile_*`) and the
  console-layout holds (`unsupported_desktop_format`, `console_layout_*`, `overlapping_resource`).
  Up to 128 shapes are kept; later shapes are only counted. The periodic report lists the shapes counted
  since it last ran.
- `x2w-d3d12 memory SRV writes (slot 70): calls N, distinct entries D` counts successful slot 70
  calls and the distinct destination descriptors they wrote.
- `x2w-draw token texture nulls: total N in P pipelines; <pipeline key>=n ...` counts the texture SRVs
  bound null with cause token, per translated pipeline (up to 16, then `other`).

The first refusal of each of the first eight distinct shapes logs its `Device::slot70` line even after
the first eight refusal lines are spent. The first write to each of the first eight distinct entries
logs `x2w-d3d12 memory SRV written i: dest=<handle> address=<address> res{...} view{...}`. The first
token null of each of the first four pipelines logs
`x2w-draw token texture null: pipeline <key> srv <i> entry <address> words <w0> ... <w7>`. All-zero
words mean nothing wrote the entry, for example a refused slot 70 call. Other words may be a
descriptor the title wrote itself. These lines only count and print; a refusal still leaves the
destination descriptor as it was.

`ctest --preset dev --no-tests=error -R '^startup_memory_srv_(census|diag_warp|diag_supported_control_warp)$' -V`
checks this. `startup_memory_srv_census` covers the limits without a device. `startup_memory_srv_diag_warp`
spends the line limit with eight zero-address refusals, has a 2048x2048 BC1 10-mip texture refused, and
draws from the untouched entry. It then publishes a supported 1-mip R8 texture and draws from it. It
expects one counted and named BC1 shape, its line past the limit, one write line for the R8 entry, one
token null with zero words, and the R8 texels. The control publishes the supported R8 texture in place of the BC1 one and must fail.

### Per-reason refusal census per pipeline key

[`PerReasonRefusalCensus`](../src/startup_stubs/per_reason_refusal_census.h) tracks every distinct
`(key, reason)` refusal pair separately instead of overwriting earlier reasons on the same pipeline key
with the last refusal:

- Up to 128 distinct `(key, reason)` pairs are stored with sanitized 64-byte reason strings, independent
  call counts, and `detail_entries = 8` first-seen detail budget hints; entries past the cap advance
  `total` and `overflow calls` without unbounded allocation.
- `report(emit, true)` skips emission when nothing changed since the previous periodic report and emits
  the summary line plus only the `(key, reason)` entries whose counts grew; `report(emit, false)`
  prints the full table without hiding the next periodic delta.
- [`d3d12_xs_device.cpp`](../src/startup_stubs/d3d12_xs_device.cpp) and
  [`d3d12_draw.inl`](../src/startup_stubs/d3d12_draw.inl) wire an `indirect-draw` census instance into
  `draw_translated_indirect`: each refused translated indirect draw with a resolved shader key records
  `(shader_key, refusal_code)` (for example `missing_guest_root`, `sampled_texture_mismatch`,
  `unsupported_resource_or_entry`, `incomplete_table_buffer`), keeps logging the first occurrence of
  each of the first 8 distinct `(key, reason)` pairs even after the 32-call refusal log limit is spent,
  emits deltas in the five-second report, and appends the full `x2w-d3d12 indirect-draw refusals:` lines
  to `x2w_d3d12_census_report` whenever at least one refusal occurred (leaving zero-refusal runs at the
  existing four memory-SRV lines). Refusals without a resolved translated pipeline (`shader_key == 0`)
  are not counted in this census (matching the per-call log gate), so `total` counts only refusals
  attributed to a resolved translated shader key rather than all `ExecuteIndirect` refusals.
- [`d3d12_xs_device.cpp`](../src/startup_stubs/d3d12_xs_device.cpp) also wires an `indirect-other`
  census instance into `list_execute_indirect`: every `ExecuteIndirect` pre-check refusal counted under
  `other` records `(key, reason_code)` (using the bound pipeline's known shader key when available, or
  `0000000000000000` when no console pipeline key is resolved), distinguishes render-target and state
  checks (`no_render_target`, `multiple_render_targets`, `depth_target_bound`,
  `missing_viewport_or_scissor`, `target_format_mismatch`, `graphics_multi_command`,
  `argument_outside_placed_buffer`, etc.), keeps logging the first occurrence of each of the first 8
  distinct `(key, reason)` pairs even after the 16-call global refusal log limit is spent, emits deltas
  in the five-second report, and appends the full `x2w-d3d12 indirect-other refusals:` lines to
  `x2w_d3d12_census_report` whenever at least one `other` refusal occurred.

Check without a device via
`ctest --preset dev --no-tests=error -R '^startup_per_reason_refusal_census(_keep-(last|first)_control)?$' -V`:
the pass run verifies interleaved reasons across two keys, detail limits, overflow accounting and delta
reports; the `keep-last` and `keep-first` controls compile `PerReasonRefusalCensus` with
`X2W_CENSUS_KEEP_LAST` and `X2W_CENSUS_KEEP_FIRST`, print both expected reasons with counts in their
mismatch output, and are marked `WILL_FAIL`.

Check the runtime wiring on WARP and hardware via
`ctest --preset dev --no-tests=error -R '^startup_indirect_(translated|other)_refusal_census_(warp|hardware|control_warp)$' -V`:
the `translated` pass runs interleave `missing_guest_root` and `sampled_texture_mismatch` on one translated pipeline
key alongside `unsupported_resource_or_entry` and `incomplete_table_buffer` on two other keys plus a
succeeding draw, then verify both `x2w_d3d12_census_report` (6 refusals, 3 keys, 4 pairs, 0 overflow)
and both per-call log lines for the shared key; its control replaces the second refusal on key 1 with a
valid draw and must report the real counts (`total 5; distinct keys 3, distinct pairs 3`) as a mismatch.
The `other` pass runs warm up one translated pipeline key, interleave `missing_viewport_or_scissor` and
`argument_outside_placed_buffer` on that key alongside `graphics_multi_command` on key `0` (plus
non-`other` placeholder, graphics-pipeline and compute-pipeline refusals that must not be counted), and
verify `x2w_d3d12_census_report` (4 refusals, 2 keys, 3 pairs, 0 overflow, no `indirect-draw` lines)
and the three per-call log lines; its control replaces the second refusal on key 1 with a valid draw and
must report `total 3; distinct keys 2, distinct pairs 2` as a mismatch.

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

Command-list slots 15 (`CopyBufferRegion`), 17 (`CopyResource`) and 19 (`ResolveSubresource`), unshifted like slots 12 to 25,
copy whole resources in render mode (`src/startup_stubs/list_copy.inl`; the checks are in `list_copy.h`). Each argument is
mapped to its desktop resource, the pair is moved to the copy or resolve states for the call and back to COMMON at Close
(an upload buffer stays readable and a readback buffer stays a copy destination), and the call is forwarded. A call the
desktop would reject is counted, logged (first six) and not forwarded, with the reason: `unresolved`, `same_resource`,
`not_buffer`, `not_texture`, `dimension_mismatch`, `size_mismatch`, `format_mismatch`, `sample_mismatch`,
`bad_subresource`, `source_not_multisampled`, `destination_multisampled`, `bad_format`, `zero_size` or
`range_overflow`, `heap_direction` (an upload buffer as a destination, a readback buffer as a source),
`initial_state` (a placed buffer that does not start in COMMON) or `title_backed_source` (a title-backed default-heap
buffer or an upload buffer, which have no desktop copy the list can refresh without rewriting bytes an earlier submission may
still read). A source the title maps at its own memory must be current: a texture takes the draw path's upload and is refused
when that upload cannot stage it (unreadable or unsupported memory), and an aliased placed buffer is already current. A copy or resolve destination with title memory is marked GPU-written when its list is submitted (on its own list at once), so a later draw does not
upload the stale title memory over the result; a list discarded by Reset or a failed Close marks nothing. CopyResource accepts formats of one DXGI family (colour, depth, 8/16/32-bit
and block-compressed). A resolve format is exact against a fully typed endpoint and only family-compatible with a
typeless one; a typeless resolve format (or an unknown one over typeless endpoints) resolves as the family's typed
choice: half float for `R16G16B16A16`, normalised for the 8-bit and 10-bit families. Calls on a closed list are ignored.
These slots have not been seen in a title log yet. Slot 16 is described in [Command list CopyTextureRegion](#command-list-copytextureregion-list-slot-16). Slot 18 (`CopyTiles`) is not handled.
`ctest -R '^startup_list_copy_validation'` runs the pure checks on any host; `startup_list_copy_{warp,hardware}` copies,
resolves and reads back through the `x2w_d3d12_list_copy_stats` export.

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
`xg_xs.dll` and `GameInput.dll`. To run a title you need only these eight and the build revision stamp, not the test
executables: `cmake --preset release` once, then

```powershell
cmake --build --preset runtime
```

builds the `startup_runtime` target, which depends on exactly the files `run.ps1` copies. Without the switches, every export that has no implementation
traps: it logs `x2w-trap <dll>!<export>` with its argument registers and ends the process with a
code that names the export (see `src/startup_stubs/trap.h`).

## Run

Prepare a folder with the executable and its package DLLs yourself. Never commit or upload it.
Alternatively, stage one executable and its adjacent DLLs with:

```powershell
.\tools\startup\stage_package.ps1 -PackageDir <package root> -Exe <title>.exe -RunDir <fresh run folder>
```

The destination must be absent or empty and separate from the package. Archives and subdirectories
are not copied. Keep using the package root as `-DataDir`, then run the staged executable:

```powershell
.\tools\startup\run.ps1 -Exe <folder>\<title>.exe -DataDir <package root> -LogDir out\startup\run1 `
    -Seconds 180 -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000 -InputScript "4@70"
```

`run.ps1` copies the stand-ins next to the executable (build them with `cmake --build --preset runtime`) and starts
`tools/startup/debug_launch.py`.
Use `-BuildDir` to select their build directory; relative paths are relative to the repository root,
and absolute paths are used as supplied. The default is `out/build/release`.
It runs the debugger with `out/pe-env/Scripts/python.exe` when that environment exists, because
`-HostCpu` needs `iced-x86` from `requirements-dev.txt`; otherwise it falls back to `python` from
`PATH`.
The debugger runs the process in a kill-on-close job with a memory cap, **offline**: file opens of
socket, NetBIOS and DNS devices are refused (`--offline 0` turns this off; do not). Offline mode
applies to the title's process only. Runtime calls can still be serviced by Gaming Services
components outside that process, which can use a signed-in Xbox, Store or Gaming Services account.
Prefer a Windows user or machine with none of those accounts signed in, or the isolated VM in
[Iterating on start-up runs](#iterating-on-start-up-runs). The live log window already notes that it
does not establish that all network activity is blocked; see
[Startup memory accounting](#startup-memory-accounting).

The file hooks that offline mode needs are always armed. The gaming-runtime call hooks
(`QueryApiImpl`, `InitializeApiImpl*`, `XErrorReport`) and the movie-player call hooks (open,
close, next frame, frame buffers, pause, skip) are armed only with `--trace-calls 1`
(`run.ps1 -TraceCalls`), because every hooked call stops the whole process under the debugger.
Without it those calls are not written to `files.txt`, and the movie-player report at the end of
`<log>.out` (call counts, movie header, frame buffers and the luma-plane PNG) is absent; with it both
are as before. The call hooks are armed together with the file hooks, so with file tracing off
(`--offline 0` without `--trace-files 1`) `--trace-calls 1` arms nothing. The summary states
`call tracing: on`, `off`, or, in that case, that `--trace-calls 1` needs file tracing. The hooked
exports are `RUNTIME_CALL_HOOKS` and `MOVIE_CALL_HOOKS` in `debug_launch.py`. `startup_debug` checks
that its fixture's module and export are in that list, that a source-built DLL loaded under the
movie-player module name is not hooked by default, that each of its 7 export calls is counted with
`--trace-calls 1`, and that `--trace-calls 1` with file tracing off counts nothing and says why.

**Window pictures.** At the end of a run that is still going, `debug_launch.py` saves the client area of the
title's window as `window.png`. `--capture-every <seconds>` (`run.ps1 -CaptureEvery`, `run.sh --capture-every`)
also saves it every N seconds (N is at least 1; 0, the default, is off) as `window_<sss>s.png`, with the seconds
since the process started, so a run that passes several screens leaves one picture per screen. Every
tick is saved, also a frame whose pixels equal the previous one (a dark loading screen is still a picture); at
most 120 files are written (later frames are counted, not saved), and the summary prints one line such as
`window captures: 12 saved (7 identical to the one before), first change at 74 s`. A thread of its own takes the pictures, so a capture never delays the debug events, and the
conversion to PNG uses whole-buffer slices instead of a per-byte loop. The end-of-run `window.png` is
unchanged. `tests/startup_window_capture_check.py` checks the file names, the repeats, the cap, the summary line
and the pixels of the PNG against a per-pixel reference. `tests/startup_debug_window_check.py` runs the
source-built `tests/startup_window_probe.cpp` (a window that is red, then blue after 3.5 s) with
`--capture-every 1`: one red and one blue frame are saved, the others are skipped, the reported first change
falls in the second the probe changes it, and without the option no `window_<sss>s.png` exists.

`run.ps1 -TraceApi` sets `X2W_TRACE_API=1` for the child process and enables
`x2w-api` lines for individual runtime provider methods. Use `-TraceCalls` for
the debugger breakpoints described above. The switches work independently and
can be combined. A line containing `x2w-runtime provider` and `(proxied as #…)`
confirms that the runtime wrapped that provider even when method tracing is off.

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
heap-flag regression, run `ctest --preset dev --no-tests=error -R '^startup_debug'`.

It writes:

| File | Content |
|---|---|
| `<log>.out` | why the run stopped: exception, registers, stack, likely return addresses, raw stack |
| `<log>/events.txt` | module loads and every exception with its time |
| `<log>/stderr.txt` | the stand-ins' logs (`x2w-d3d12`, `x2w-dstorage`, `x2w-xmem`, `x2w-runtime`, ...) |
| `<log>/files.txt` | every file the process opened |
| `<log>/translate.txt` | with `-Translate`: one line per shader request translated during the run |

### Translating shaders during a run

With `-Render -Translate`, `run.ps1` starts `tools/startup/gfx2hlsl.py --watch <shader cache>` beside
the debugger and stops it, with its worker processes, when the debugger returns (`-Translate`
without `-Render` is ignored with a warning). The watcher repeats the `--pending` scan every
2 seconds (`--watch <dir> <seconds>` sets another interval), so a pipeline the title creates during
the run is translated within seconds, and the runtime picks it up on a later draw instead of on the
next run. Each pass serves the requests in the order the title published them, those published since the
watcher started first, then the older backlog; during a pass the watcher looks for new requests about every
second between results, so a request the title publishes while a long backlog is being translated starts next
instead of after the backlog. The cache persists: a later run reads every published `<key>.txt` and translates only new
requests. The watcher's first pass and the one-shot `--pending` scan retry a failure only when it is older than the
translator's own files (`gfx2hlsl.py` and the modules it loads from its folder), so a translator update reaches
requests that failed with an earlier build, while a refusal this build already wrote is not translated again on every
run; `X2W_RETRY_FAILED=1` retries every failure. After the first pass, a request whose `<key>.fail` is newer than
its ready marker is not retried during the run (the runtime writes each request once and never rewrites it). An unexpected translator error is written
to `<key>.fail` as `translator error: ...`, ending with the function, file name and line that raised
it, skipping standard-library frames (`(in scalar_select, gfx2hlsl.py line 1234)`), instead of
stopping the scan. An operand refusal
in `<key>.fail` names the instruction that refused it (`operand '-v30' in v_and_b32_e64`); the
refusal's own message, which `--refusals` counts, stays `operand '-v30'`. A worker killed from outside
(for example out of memory) ends that watcher pass only; its requests are retried on the next pass.
FXC is located as described for `X2W_SHADER_CACHE` under [Switches](#switches). A scan with more than
one pending request translates them in parallel processes, one per logical processor (at most 61 on
Windows, the `ProcessPoolExecutor` limit) unless `X2W_TRANSLATE_JOBS` sets the count; each request
writes only its own files. On a 16-thread host,
3,513 requests (132 translated, 3,381 refused) took 56.9 s with one process and 7.8 s in parallel,
with byte-identical results.

Check with `python3 -B tests/startup_gfx2hlsl_watch_check.py` (ctest `startup_gfx2hlsl_watch`): a
request without its ready marker is not consumed, the watcher retries an earlier failure once and
then leaves failures alone, it translates a request written while it runs, four parallel workers
print the same results and write the same failures as one, an unexpected translator error becomes
`<key>.fail`, and an invalid `X2W_TRANSLATE_JOBS` falls back to the processor count.
`startup_run_ps1` checks that `-Translate` starts the watcher on the cache and stops it together
with a child process.

### Compute pipeline requests

Device slot 67 creates a pipeline from a console compute program. The stand-in reads 12 64-bit words of the
description:

| Words | Content |
|---|---|
| 0 | root signature |
| 1-2, 3-4 | two state blocks and their sizes (a size of 0 means the block is absent) |
| 5-6 | program code and its size |
| 7-8 | a second blob and its size |
| 9, 10 | pointers whose targets are not decoded |
| 11 | a value |

It interprets none of these parts and still creates a placeholder. With `X2W_D3D12_RENDER=1` and
`X2W_SHADER_CACHE` set, it writes a request to the cache directory the graphics requests use:

| File | Content |
|---|---|
| `req_<key>.desc` | the 12 words with the pointer words 0, 1, 3, 5, 7, 9 and 10 always set to zero, so no process address reaches a file or the key |
| `req_<key>.sa`, `req_<key>.sb` | state blocks A and B (at most 4 KiB each; empty when the size is 0) |
| `req_<key>.data` | the words 7-8 blob (at most 1 MiB; required) |
| `req_<key>.root` | the serialized root signature |
| `req_<key>.c67` | the program code (at most 1 MiB); written last, as the ready marker |

The `.c67` extension keeps these requests apart from draft #1954, which uses `req_<key>.cs` for desktop DXIL
containers created at device slot 62.

**Reading the parts.** Each part is checked and then copied once, with a fault-safe copy, before anything is
hashed, scanned or written, so a later write by the title cannot leave a file that does not match its key.
If any declared part cannot be taken, no file is written and the creation counts as failed. The logged reason
names the part and the cause:
- a code or data size of 0 (a program without the words 7-8 blob is refused);
- a size over its limit;
- a part that is not readable to its declared end;
- a description that is not readable;
- a root signature without a recorded blob.

A cache directory that is set but does not exist makes every creation fail. Publishing is off only when
render mode is off or `X2W_SHADER_CACHE` is unset; such creations are counted nowhere.

**Key.** The key hashes the revision string followed by `cs`, then each part preceded by its length as a
little-endian u64. The order is: code, words 7-8 blob, root, `.desc`, `.sa`, `.sb` (`compute_key` in
`gfx2hlsl.py`), so moving bytes from one part to the next changes the key. A compute key and a graphics key
can still coincide only through a 64-bit hash collision, or through a graphics program crafted to start with
the compute framing.

What words 9 and 10 point to is not part of the request, so two descriptions that differ only there share a
key. If those targets are decoded and added later, every key changes and each request is published once more.

**Pointers inside blobs.** An aligned qword in `.data`, `.sa` or `.sb` that names committed memory of the
process (one `VirtualQuery` per candidate, so no fault is taken) would make its key run-dependent. These
qwords are counted and published, not refused, because hash-like values can name memory by chance. The code
is not scanned.

**Dedupe.** Each key is written once per cache and never rewritten. Parts left by an interrupted write, which
has no `.c67` yet, are kept, and the next creation completes the request. A key already seen in the process
costs no file access, and the writes take their own lock, not the draw lock.

**Log and totals.** Each creation logs `x2w-draw compute request <key> published`, `already present` or
`not written`, with the reason when refused, up to 12 lines per run. The render totals line ends with
`compute requests published <n> (already present <n>, failed <n>, pointer-like blob qwords <n>)`.

**Offline.** `gfx2hlsl.py --pending` and `--watch` take `.c67` as a ready marker and translate the request
(see [Compute pipeline translation](#compute-pipeline-translation)). A refused request stays in the cache, and
a later `--pending` retries it. `--refusals` reports compute requests in a separate section, so its graphics
counts do not change, and `triage_run.py` puts their `.fail` reasons in a `compute:` class.

**Local observation** (a local build of this change ported to an older base; one title, local files only, not
in the repository):
- One run published 2,689 requests with 0 failed. It wrote 2,689 files of each of the six kinds, 120.3 MB in
  all (98.8 MB of it `.c67`), and counted 729 pointer-like blob qwords.
- A second run with the same cache published 0 (2,689 already present, 0 failed), wrote no files and changed
  no modification time.
- An earlier build with the same layout found that in every request the code length equals word 6 and the
  blob length equals word 8, and that state block A is 28 bytes and state block B 20 bytes.
- In the first 40 descriptions, words 9 and 10 point at a readable block of at least 4 KiB.

**Checks.** Run `ctest --preset dev --no-tests=error -R "^startup_compute_request_(warp|disabled_control)$" -V`
after building `startup_stream_consumer`. On WARP:
- The same synthetic description is created twice in each of two processes, with words 9 and 10 pointing
  into the consumer and a 20-byte state block B, and the files are written once.
- Each refusal writes no file. These include reads next to a `PAGE_NOACCESS` page and sizes 1 byte over each
  limit, while code, data and both state blocks at exactly their limits are published.
- A valid, refused, valid sequence publishes the two valid keys.
- A request whose first four parts already exist is completed without rewriting them.
- The control runs without render mode and must print `FAIL: no compute request`.


### Compute binding evidence at Dispatch

With render mode and `X2W_D3D12_COMPUTE_BINDINGS=1`, the shim logs selected slot 62
pipeline creations and samples the root arguments seen when list slot 14 consumes
those pipelines. Supply comma-separated 16-digit lowercase container hashes in
place of `1` to select specific pipelines. The log prefix is `x2w-compute-bindings`.
The trace records the root match, desktop PSO and root availability, up to 16
root parameters, and a 64-bit mask of DWORDs covered by constant setters.
Binding entries report the setter slot and address category, including aligned
or unaligned upload, aliased or copied placement, and unresolved addresses.

The address category describes ownership at the first byte. Full range, current
GPU contents, descriptor table contents and resource state need separate
validation. The trace leaves command recording unchanged and captures no
constant data or buffer contents. It logs up to 32 creation records and 16
dispatch samples per pipeline, with a 256-dispatch process limit.

```powershell
cmake --build --preset release --target startup_compute_binding_trace
ctest --preset release --no-tests=error -R '^startup_compute_binding_trace_warp$' --output-on-failure
```

### Compute pipeline translation

`gfx2hlsl.py --pending` and `--watch` translate a `.c67` request with the compute stage instead of recording
`compute stage not supported`. The request is checked first, and it is refused into `<key>.fail` (with no
cache path in the text) when:
- a part is missing;
- `.desc` is not 96 bytes;
- a part's length disagrees with its size word in `.desc` (code word 6, data word 8, state blocks words 2 and 4);
- the files do not hash to the key in their name;
- the code is empty, is not whole dwords, or has no `s_endpgm`.

The translation is `ComputePipeline(code, root, COMPUTE_PGM_RSRC2, group size)`. Both register values come
from state block A, which must be 28 bytes. The stage writes `<key>.cs.hlsl`, then `<key>.cs.cso` (DXC
`cs_6_0`), then the `stage cs` manifest `<key>.txt` last; a stage refusal goes to `<key>.fail` with the
stage's own reason. A request is done once `<key>.txt` and `<key>.cs.cso` exist. `X2W_FXC=` (empty)
publishes the manifest alone, as for graphics. `<key>.cs.cso` has the same form as the slot 62 results of
#1441 in the same directory. The keys come from different hashes, so the names clash only on a 64-bit
collision. The stand-in does not load the result yet and still creates a placeholder.

The state-block reading is a **hypothesis** from a local observation of 2,689 requests (an earlier local build
of the request change; one title, local files only):
- dword 3 holds the thread-group size X in its low 16 bits and Y in its high 16 bits, and the low 16 bits of
  dword 4 hold Z. All 2,689 sizes are within the `numthreads` limits, with 43 distinct sizes.
- dword 6 has the `COMPUTE_PGM_RSRC2` layout. Bits 31:24 are zero, and the thread-ID component count never
  exceeds what the group size needs; it covers a Y size above 1 in 2,671 of 2,689. `TG_SIZE_EN` is set in 91,
  which the stage refuses.

The stand-in does not decode these values.

Census: one title, local files only.
- Requests: 2,689, published by a local build of the request change that was ported to an older base.
- Re-keying: the requests were re-keyed offline with this `compute_key`. The files are unchanged; only their
  names change.
- Result: 8 of 2,689 translate and compile, and `--pending` took 11 s. No request is refused by the request
  checks above.
- First refusal per request:
  - `buffer_load_format` from the descriptor format: 1,280;
  - scalar writes under `s_cbranch_execz`: 541;
  - `s_load` with a register offset through the resource heap: 158;
  - `image_store`: 110;
  - `TG_SIZE_EN`: 91.

Run `python -B tests/startup_compute_translate_check.py` (ctest `startup_compute_translate_requests`, pure
Python) to check the refusals above and the case A manifest. `startup_compute_translate_e2e_<warp|hardware>`
goes end to end on the GPU:
1. the stand-in publishes case A of `startup_gfx2hlsl_compute_check.py`;
2. `--pending` translates it and DXC compiles it;
3. `offscreen_check` dispatches `<key>.cs.cso` and compares all 320 words.

Its `wrong_group` control swaps the group size in the state block and must mismatch. This proves that the
request, translation, bytecode and dispatch fit together. It does not prove the state-block reading, because
the fixture builds the block at the offsets the translator reads.

### Shader requests from the archive, before a run

`tools/startup/archive_requests.py <archive> <shader cache> [--roots <dir or req_*.root> ...]` writes
requests without starting the title, then `gfx2hlsl.py --pending <shader cache>` translates them:

```powershell
python -B tools/startup/archive_requests.py <shader archive> out/startup/shader_cache
python -B tools/startup/gfx2hlsl.py --pending out/startup/shader_cache
```

A startup run that traced every DirectStorage read showed where the native programs come from:
zlib streams in the shader archive that start on 64-byte boundaries and inflate to 64 KiB. Inflated
offline, those streams hold the exact bytes the runtime later records. Each program has a 36-byte
header followed by `s_inst_prefetch`, is padded in 64-byte blocks and starts on a 256-byte step,
and in every measured pipeline the vertex program directly follows the pixel program. The tool
carves every program that ends in `s_endpgm` (an all-zero block inside a program would cut it
short), pairs each pixel program with a vertex program that starts on the next 256-byte step
and writes `req_<key>.{a,b,root}` with the runtime's key.

What it cannot take from the archive:

- **Root signatures.** None of the recorded root signatures occurs in the archive or the
  executable; the title appears to serialize them at run time. The tool reads `req_*.root` files
  from earlier runs (by default those in the cache) and writes each pair once per distinct root.
  A key is only used when the title binds that pair with that root, so the extra requests cost
  translation time and nothing else.
- **Pipelines the title builds from other sources.** Programs embedded in the executable as DXBC
  containers are not in the archive.
- **Other pairings.** A program shared by several pipelines, or a pair that is not adjacent, is
  only found by a run (`run.ps1 -Translate`).
- **Programs cut off at the end of a stream.** A zlib stream that starts on the 64-byte boundary
  right after the previous one continues it, and the two are carved as one, so a program split
  between them is kept. A program that reaches the end of such a joined run is skipped.

A stream that would inflate past 1 MiB is counted as `oversized` and skipped without inflating it in
full, and a joined run stops growing at 64 MiB. A request is written as `.b`, `.root`, then `.a`;
if one part cannot be written, no `.a` and no temporary file are left and the run exits with an
error, as it does when the archive yields no pair at all.

In one local measurement the tool joined 1,079 zlib streams into 800 runs, carved 4,982 programs
(2,730 pixel, 2,115 vertex, 137 other) and wrote 3,774 requests for 1,258 pairs and three recorded
root signatures in 22 s. Eight of the eleven pipelines that a cold run recorded were among them with
identical keys, and their translations (`<key>.txt`, both `.hlsl` files and both `.cso` files) were
byte-identical to those from the run. The other three were two pipelines whose programs are DXBC
containers in the executable and one pipeline whose programs were not carved.

Check with `python3 -B tests/startup_archive_requests_check.py` (ctest `startup_archive_requests`).
On a synthetic archive built from documented instruction words, each adjacent pixel/vertex pair
becomes one request per root, and the test computes the expected keys with its own copy of the
runtime's hash. Controls that write nothing:

- a pixel program followed by a pixel program, or a vertex program without a pixel program before it;
- a vertex program one empty step after its pixel program, or after a pixel program without `s_endpgm`;
- a program without the header tag, or a pair at offsets that are not a multiple of 4;
- a pair that reaches the end of its stream, or a pair in a stream off the 64-byte grid;
- an empty or duplicate root file, an oversized stream, or a run without root signatures.

A pair split between two consecutive zlib streams is carved whole. A blocked `.root` write leaves no
`.a` and no temporary file and fails the run, as does an archive without pairs. A second run writes
nothing new.

`run.ps1` prints the first five lines of `<log>.out` and exits with the debugger's exit code: 2 when
the process could not be created, 0 when the debugger finished its report, whatever the process
did. Check with `python3 -B tests/startup_run_ps1_check.py` (ctest `startup_run_ps1`): on Windows it
runs the script on a stub debugger under Windows PowerShell and `pwsh`, elsewhere it checks the
script text only.
`tools/startup/run.sh` is the Linux counterpart ([Launch environment](wine.md#launch-environment)).
`python3 -B tests/startup_run_sh_check.py` (ctest `startup_run_sh`, a skip on Windows) runs it under bash
with a stub Wine, debugger and watcher. It checks the exit code, the debugger's arguments and environment,
and that the watcher and its child run during the debugger call and are gone after the script exits. The
watcher stub ignores SIGTERM, so the SIGKILL fallback is exercised. After a Ctrl-C, run.sh ends by
SIGINT; a TERM sent to run.sh alone waits for the debugger and ends by SIGTERM. Mutated copies of the
script must fail. run.sh and its check need bash 4.4 or newer, GNU `realpath` and `setsid`, so they run
on Linux, not on a stock macOS.

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

The `ExecuteIndirect` table reports the highest sampled cumulative count for each outcome and
command-signature kind in the `x2w-render indirect:` census. A repeated snapshot does not increase
any count. Each cell can come from a different sample, so avoid summing the cells as one run's total
when scanning multiple runs. The section appears only after a complete census line and can show zero
forwarded `dispatchmesh` calls alongside a graphics-pipeline refusal count. Forwarded counts describe
the runtime's forwarding path. GPU completion and image readback require separate evidence.

The GPU Memory Budget section uses complete `x2w-budget` heartbeat lines from
`X2W_D3D12_BUDGET=1`. It counts samples above the local DXGI budget and prints
the sample with the highest usage-to-budget ratio, plus the sample with the most
retained texture bytes. Each row keeps the usage, budget and texture counters
from that same heartbeat. Samples with an unavailable or zero local budget
still contribute texture counts. The over-budget count describes observations
in the supplied logs.

The `Run End-Point` section says how far a run got and how it ended, so that reports of runs with different
memory caps can be told apart. It gives the largest `commit peak=` value of the `x2w-dstorage heartbeat` lines (not
the last one), the highest heartbeat second, the counts of the last `failed requests by cause` heartbeat line (the
eight fixed cause names, each with its decimal count, and the second of that line). It reports the first and last observed
exception and separately reports a confirmed debugger stop. Common first-chance debugger events, such as the loader
breakpoint, are excluded from the observed exception count. The debugger emits an explicit stop record even for a
first-chance breakpoint. For older logs, include the debug launcher's `.out` summary in the input directory to confirm
the stop. The report copies only the 32-bit exception code and whether it was first- or second-chance.
Check with `python3 -B tests/startup_triage_run_check.py` (ctest
`startup_triage_run`).

## Iterating on start-up runs

This is one way to turn repeated start-up runs into a list of translator work that others can pick up. It combines the tools above (`run.ps1`, `gfx2hlsl.py`, `triage_run.py`) with Windows Sandbox and a logon script that you write yourself; the script is not in this repository. Nothing from a run (executable, package, logs, shaders, captures) is committed or posted. Only instruction names, refusal reasons and counts leave the machine.

### 1. Run the executable in a disposable VM

Treat a console executable as untrusted input. Windows Sandbox gives a throw-away VM and can share the host GPU. It requires Windows 10 or 11 Pro, Enterprise or Education, with virtualization enabled and the optional feature "Windows Sandbox" turned on. Its state is discarded when the window closes.

The configuration file (`.wsb`) below turns off what the run does not need. vGPU is a deliberate exception, needed for hardware D3D12 rendering; Microsoft documents that it increases the sandbox's attack surface. Only one folder is writable. The paths in capitals are placeholders. Map the Python install and the Windows SDK folder that contains `fxc.exe` at the same paths as on the host, so `out/pe-env` resolves inside the VM.

```xml
<Configuration>
  <Networking>Disable</Networking>
  <vGPU>Enable</vGPU>
  <MemoryInMB>MEMORY_MB</MemoryInMB>
  <ClipboardRedirection>Disable</ClipboardRedirection>
  <PrinterRedirection>Disable</PrinterRedirection>
  <AudioInput>Disable</AudioInput>
  <VideoInput>Disable</VideoInput>
  <ProtectedClient>Enable</ProtectedClient>
  <MappedFolders>
    <MappedFolder><HostFolder>PACKAGE</HostFolder><SandboxFolder>C:\pkg</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>REPO</HostFolder><SandboxFolder>C:\x2w\repo</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>STAGE</HostFolder><SandboxFolder>C:\x2w\stage</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>PYTHON_INSTALL</HostFolder><SandboxFolder>PYTHON_INSTALL</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>SDK_BIN_X64</HostFolder><SandboxFolder>SDK_BIN_X64</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>OUT</HostFolder><SandboxFolder>C:\x2w\out</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
  </MappedFolders>
  <LogonCommand><Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\x2w\stage\run_in_sandbox.ps1</Command></LogonCommand>
</Configuration>
```

`REPO` holds a release build and `out/pe-env`. `STAGE` holds the logon script and anything it copies into the VM. `MEMORY_MB` must leave room for `run.ps1 -MemoryGiB` plus the VM itself.

The logon script runs only inside the VM, so `-ExecutionPolicy Bypass` and writes to the VM's `System32` affect nothing on the host. It does the following:

1. If the title uses the system gaming runtime, copies the host's `xgameruntime.dll` from `STAGE` into the VM's `System32`.
2. Maps the package to the console's package drive with `subst G: C:\pkg`. Copies the package's top-level `.exe` and `.dll` files into a run folder; the data stays read-only.
3. Sets `X2W_FXC` to `fxc.exe` in the mapped SDK folder. `run.ps1` does not set it, and the VM has no SDK of its own.
4. Starts a background loop of `gfx2hlsl.py --pending C:\x2w\out\shader_cache`, so new pipeline requests are translated while the title runs.
5. Runs `tools/startup/run.ps1` with `-Render`, `-ShaderCache C:\x2w\out\shader_cache`, `-StoragePath` under `C:\x2w\out` and the usual switches (see [Run](#run)). Its offline debugger stays on; the VM is a second barrier around it.
6. Stops the loop. Writes the output of `triage_run.py <log> --shader-cache <cache>` to `C:\x2w\out` with `Out-File -Encoding utf8`; Windows PowerShell's `>` writes UTF-16.

Every file in `OUT` is written by untrusted code. Read it; do not execute anything from it on the host. Do not rebuild or check out another revision of the mapped repository while a run is in progress: the VM reads `tools/` and the build from it.

**Cost of the loop.** Every pass retries each request without a complete result, including those that already have a `.fail`, and re-runs `fxc` for fxc failures. A result is complete when `<key>.txt` exists and, unless `X2W_FXC` is empty (manifest only), both `.cso` files hold bytecode. That load falls on the run being measured, so choose the interval accordingly.

### 2. Keep the shader cache between runs

Archive each run's logs together with a copy of `shader_cache/`, and leave the cache in place for the next run. With a cold cache, every draw whose pipeline has not been translated yet is counted as "not drawn" and is missing from the frame.

When the translator handles more, the next `--pending` pass retries the waiting requests from the same cache. This holds only while the pipeline key is unchanged. The key hashes a translation revision string (`x2w-startup-shader-N`, in `pipeline_key` in `gfx2hlsl.py` and in the runtime). After a revision change the runtime asks for new keys, and the old requests stay in the cache.

Set `X2W_PRE_RUN_FAMILY_RETRY=1` before running `tester_run.ps1` to enable the versioned pre-run retry.
Refusals retain their reason text and record the reason family and translator content fingerprint.
The pre-run pass samples up to eight stale refusals per family, interleaves families, and continues a
family when a sample translates. The cache records where probing stopped so a capped pass can resume
with the next family. The newest run with both render census and per-key use records supplies the
weights. The switch defaults to off.

**Re-key the old requests after a revision change.** The new cache is cold, so a run would have to request every pipeline again. `tools/startup/shader_cache_rekey.py` copies graphics, RTV variant, slot 67 compute and slot 57 mesh requests under their current keys. It uses `pipeline_key`, `compute_key` and `mesh57_key` from the adjacent `gfx2hlsl.py` and updates each RTV variant with its new graphics key. Stop the title and any `--pending` loop first; nothing else may write the cache. Then:

```powershell
.\out\pe-env\Scripts\python.exe -B tools\startup\shader_cache_rekey.py out\startup\shader_cache out\startup\shader_cache_rekeyed
.\out\pe-env\Scripts\python.exe -B tools\startup\gfx2hlsl.py --pending out\startup\shader_cache_rekeyed
robocopy out\startup\shader_cache_rekeyed out\startup\shader_cache /XC /XN /XO
```

1. The re-key step reads the old cache and writes only request files (`req_*`). A file already in the new directory is left as it is, so a second run writes nothing. It reads every request before it writes. If a part is missing or unreadable, or an RTV variant is malformed or names a base that is not in the cache, it prints how many, writes nothing and exits with 1. A slot 57 request requires `.ps57`, `.root` and the `.m57` ready marker. The `.ps57` file is empty when there is no pixel stage. Slot 86 mesh requests are only counted. `X2W_TRANSLATE_JOBS` sets the number of hashing processes.
2. `--pending` translates the re-keyed requests offline, as it does for a running title's cache.
3. The `robocopy` switches skip every file that already exists in the cache, so results the runtime or a later pass wrote are kept. A `robocopy` exit code below 8 means success: 1 is set when files were copied, and 2 is set because the cache holds files that the new directory does not. A code of 8 or more is a failure. On Linux, run the same steps with `./out/pe-env/bin/python` and merge with `cp -rn out/startup/shader_cache_rekeyed/. out/startup/shader_cache/`, which also keeps existing files.

`tests/startup_shader_cache_rekey_check.py` checks the tool on synthetic requests only: the names match the current keys, a newer graphics and compute revision gives other names for those requests, a second run and a current cache change nothing, and an incomplete request or an RTV variant without its base writes nothing. Re-keying recovers the requests, not the translations: some re-keyed pipelines can still fail to translate, which is separate from the cache key. Nothing here shows that a title runs.

When `fxc` is available, a request that later translates loses its `.fail`. In manifest-only mode (`X2W_FXC` empty) the old `.fail` stays, so on a kept cache the `.fail` categories in `triage.md` can include pipelines that now have a `.txt`. That table counts category tokens over all `.fail` files, up to `--max-items`.

### 3. Measure a run against the previous one

With `-Render`, `d3d12_xs` prints a running total to `stderr.txt` every 300 frames:

```
x2w-render totals: ... translated draws N (not drawn M), pipelines translated P / missing Q, ...
```

Compare the last line of each run. More translated draws and fewer "not drawn" mean a translator change reached real draws. "missing" is a live count of pipelines still waiting for a manifest. It can grow while things improve, if the run gets further and requests new pipelines.

Beside the totals line, one line counts every `ExecuteIndirect` call (command-list slot 61) since the start, forwarded or by refusal reason, then by signature kind (`draw`, `drawindexed`, `dispatch`, `dispatchmesh`, or `other` for a placeholder or unknown signature); a reason with no calls reads `0`:

```
x2w-render indirect: forwarded: drawindexed 3; refused graphics-pipeline: drawindexed 1; compute-pipeline: dispatch 1; placeholder-signature: other 1; other: drawindexed 3
```

`graphics-pipeline` is a draw without a matching desktop pipeline and root, `compute-pipeline` a dispatch without a desktop compute pipeline and root, and `other` every remaining refusal, counted under the signature's kind whenever the signature is known. The `x2w-render ExecuteIndirect refused` lines still stop after 16. `startup_indirect_draw_indexed_warp` checks this line; `startup_indirect_counts_control_warp` expects one forwarded call too many and must report the mismatch.

### Optional: Tracy zones

For a timeline instead of the counters, configure a separate build tree with `-DX2W_TRACY=ON`. This is a developer-only build, off by default and in every preset and workflow. Do not use it for a run that must stay offline: the Tracy client opens a TCP listener (from port 8086 up, on 127.0.0.1 only) in each instrumented process.

- **Source.** Configure fetches [Tracy](https://github.com/wolfpld/tracy) v0.15.0, pinned to commit `f2a4f2529e7e71844c9e434b877770a21ecf7587`, with `FetchContent` (network needed at configure time). `-DFETCHCONTENT_SOURCE_DIR_TRACY=<checkout>` uses an existing checkout instead, and `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` forbids any download. Nothing from Tracy is committed here. Tracy is copyright Bartosz Taudul and others, 3-clause BSD license; its `LICENSE` ships in the fetched source, and a redistributed instrumented build must carry it.
- **Client settings.** On-demand (nothing is recorded until a profiler connects), localhost only, no broadcast, no call stacks, no system tracing and no crash handler.
- **Output.** One shared `TracyClient.dll` is linked into `d3d12_xs.dll`, `XCRDAPI.dll` and `xmem.dll`; copy it beside them.
- **Zones.** `ExecuteCommandLists`, `Present` (with a frame mark), the frame-wait sleep, window present, desktop pipeline creation, `device_memory_srv` with its resource scans, each DirectStorage request, and the xmem page map and physical write. `src/startup_stubs/profile_zone.h` defines the macros.
- **Memory.** Three kinds of data for Tracy's Memory view, by name: the pool `XMemAlloc` (every block `XMemAlloc` returns until its `XMemFree`), one pool per instrumented DLL for its C++ heap (`<target> C++ heap`, `operator new`/`delete` replaced in `src/startup_stubs/tracy_new.cpp`), and the plot `XMemVirtualAlloc commits (bytes, cumulative)`. That last one is a plot because the title releases those ranges with Win32 calls the stand-in does not see. Memory the title takes straight from Win32 (`VirtualAlloc`, its own heaps) and what the graphics driver commits are not tracked; compare the pools with the `commit peak=` heartbeat to size that remainder. A profiler that connects late misses earlier allocations (on-demand client).
- **Off.** Without the option nothing is fetched, defined or linked, and the macros expand to nothing or a no-op.

To capture without a GUI, build Tracy's `capture` tool from the same checkout (`cmake -S <checkout>/capture -B <dir>`; that project fetches its own dependencies), start `capture -o run.tracy -a 127.0.0.1 -s <seconds>` and then the instrumented run. Open the file with the Tracy profiler, or export zone statistics with `csvexport`.

### 4. Census, then rank

The first refusal hides the rest. Run the [refusal census](experiments.md#startup-translator-refusal-census) on a copy of the cache:

```powershell
python -B tools/startup/gfx2hlsl.py --refusals <copy of shader_cache> --json | Out-File -Encoding utf8 census.json
```

There are two views:

- `per_code_pair.reasons` counts vertex/pixel code pairs, without RTV variants. Each row has `reason`, `scope` (`decode`, `vs`, `ps` or `pipeline`), `under_every_root` and `under_some_roots`. The sum of the last two is the number of pairs where the reason appears.
- `per_request.reasons` counts request keys, RTV variants included. Each row has `containing`, `first` and `only`.

Counts after the first refusal are approximate, and a pipeline-scope `only` is a lower bound (the `caveat` field).

Rank by the number of pairs a reason appears in, not by first refusals: when most pairs contain several reasons, fixing the most frequent first refusal alone draws few new pipelines. Reasons that share a design, such as descriptor tracking or resource-heap loads, are better handled together.

### 5. Pick work that nobody holds

Before taking a reason:

- search open pull requests and the `help wanted` issues for the instruction name;
- prefer self-contained instructions with clear semantics (scalar ALU, operand modifiers, inline constants) over design topics;
- check the semantics against a primary source, such as the AMD RDNA 2 ISA reference or LLVM's AMDGPU backend for assembler field meanings (state the version), not against another translator in this repository;
- add a negative control that fails when the lowering is removed or its semantics change.

After the change is merged, step 3 on the next run shows whether draws moved.

## Launcher check

`tools/startup/launch.py` validates timeout and job limits before creating the log directory or
starting a process. The timeout is at least one millisecond and stays below the Win32 infinite
wait sentinel. A nonzero memory limit must fit in `size_t` as bytes. The process limit must fit
in a DWORD. A zero memory or process limit disables that limit. Run
`ctest --preset release --no-tests=error -R '^startup_launch_cli_check$' -V` for the input checks.

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

A request whose destination (a plain pointer, not 64 KiB page tokens) is not writable up to its declared capacity fails before any byte is written: the stand-in reads the attributes of the destination's pages without touching them, and a page that is uncommitted, read-only, `PAGE_GUARD` or `PAGE_NOACCESS` refuses the whole request. When every page is resident and plain read-write (`QueryWorkingSetEx`: valid, `PAGE_READWRITE` or `PAGE_EXECUTE_READWRITE`, no guard bit), the request goes ahead; otherwise the destination is walked with `VirtualQuery`, which also accepts copy-on-write pages. The log line ends `destination not writable`, `requests_failed` counts it, the next status entry reports `E_FAIL` and the queue goes on with later requests. Before this check, a destination shorter in memory than its declared capacity faulted inside the stand-in's copy (#870). The check covers the whole declared capacity, not only the bytes the request would write. Cost: one `QueryWorkingSetEx` per 512 pages of the declared capacity, per request that reaches the copy, plus the `VirtualQuery` walk when a page is not resident and plain read-write. On the title's multi-gigabyte regions each `VirtualQuery` scans to the end of the region, about 10 ms of kernel time per request (#2684).

Failed requests report distinct status HRESULTs where the cause is known (#1843). A destination smaller than the data (`destination capacity exceeded`) reports `0x8007007A` (`ERROR_INSUFFICIENT_BUFFER`); a file read at or past the end, or crossing it, reports `0x80070026` (`ERROR_HANDLE_EOF`); every other cause stays `E_FAIL`. The same code is the `HRESULT` of the first-failure record. `OpenFile` with a null path returns `E_INVALIDARG`, `CreateQueue` with a null descriptor returns `E_INVALIDARG`, and a null output pointer returns `E_POINTER`; the output is set to null on every refusal. `tests/startup_dstorage_destination_check.cpp` pins each code on the real stand-in.

`tests/startup_dstorage_destination_check.cpp` (ctest `startup_dstorage_destination`) sends memory-source requests through the real queue: a read-only, a guard and a no-access middle page, and a capacity one page past the committed pages. Each must fail with the process alive and the bytes of the pages before the bad one still `0xAA`; the guard page must still be `PAGE_GUARD` afterwards, and a vectored exception handler must see no guard-page exception. A copy-on-write page followed by a no-access page must be refused with the first page still shared (`QueryWorkingSetEx`) and unchanged. A writable destination whose capacity is three pages but whose data is one page must succeed with the two unwritten pages still `0xAA`, and a committed page that was never touched (not resident) must be accepted. Afterwards, requests into a fully writable (`PAGE_READWRITE`) buffer and a copy-on-write (`PAGE_WRITECOPY`) mapped view on the same queue must write every byte:

```powershell
ctest --preset release --no-tests=error -R '^startup_dstorage_destination$' -V
```

## Local switches, logging, and memory probing

`tests/startup_local_log_check.cpp` (device-free ctest `startup_local_log`, with negative
controls `startup_local_log_guard_drop_control`, `startup_local_log_reopen_control`,
`startup_local_log_cache_failure_control` and `startup_local_log_wrap_control`) verifies
`src/startup_stubs/local_log.h`: `env_on` (a leading `1` in values up to three characters,
preserving `GetLastError`), the atomic `first_calls` gate (true for the first N calls; the
counter stops at N, so a counter that a consumer polling in a loop drives past 2^31 cannot turn
negative and let every call log again; the wrap control shows the old counting gate doing that
from a counter set near the wrap point), `logf` (stderr and `X2W_TRAP_LOG` output, CRLF and 1533-byte payload
truncation, `MAX_PATH` boundary behavior, and `GetLastError` preservation on file-open failure),
the shared `x2w_trap::append_log` (one `X2W_TRAP_LOG` open per module: in a child process, 8
threads write 2,000 lines each, every line must arrive intact and the open count must be 1; the
reopen control respells the path after the first line so each line reopens the file, and passes
only when the open count is its sole failure; a failed open is not kept: while no handle is kept,
ordinary lines retry the open at most once per 250 ms and are dropped in between, and trap lines
always retry. Child processes check that a first open into a missing directory, or of a file held
without sharing, recovers once the path is usable, that an unusable path costs at most one open
attempt per 250 ms, and that a trap line opens the log during the wait; the cache-failure control
is built from a generated copy of `trap.h` that keeps a failed open forever and passes only when
the two recovery checks are its sole failures),
`probe_readable` (invalid-pointer and zero-length refusals, multi-page reads, `PAGE_NOACCESS` /
`MEM_RESERVE` prefix truncation, and `PAGE_GUARD` re-arming verified by `VirtualQuery` and a
second probe), and `readable`/`writable` boundaries across `PAGE_READONLY`, `PAGE_READWRITE`,
`PAGE_WRITECOPY`, `PAGE_NOACCESS`, `PAGE_GUARD`, and `MEM_RESERVE`.
A first-chance exception stops the whole process while a debugger is attached, so `probe_readable` and the fault-safe
copies (`guarded_memcpy`: `copy_title_bytes`, the linear-buffer `guarded_copy`) first ask `VirtualQuery` (one entry per
region, cached per thread) whether the source is in free address space (`MEM_FREE`) and refuse it with no exception.
Reserved, committed, guard and no-access pages still go through the touch / `__try` path, because a reserved page may be
a lazily committed xmem page that the fault handler commits on first touch; `__try` stays the last guard. The render
totals line prints `reads refused before a fault <n>` after the textures count. The check adds an unmapped range
that must return false and 0 with no exception seen by a vectored handler, for the probe and the copy, and counts the
avoided faults. No GPU is required:

```powershell
cmake --build --preset dev --target startup_local_log_check startup_local_log_cache_failure_check
ctest --preset dev --no-tests=error -R '^startup_local_log(_guard_drop_control|_reopen_control|_cache_failure_control)?$' -V
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

## Concurrent calls through a hooked function

The debugger's file, socket and allocation hooks are one patched byte per function. To run the original
instruction it restores the byte, single-steps the thread that hit it and patches the byte again. A second thread that
entered the function inside that window ran untraced, so `files.txt` could miss opens and the
offline mode could miss a refused socket. Now every other thread is held for the step and resumes when the stepping
thread is done (`tools/startup/hook_window.py`). A thread that had already trapped when it was held goes back to the
hook address and traps again afterwards, so its call is traced once, not twice. Each window has a 2 s deadline, so a
thread is never held forever. Per-thread state of a call in flight (the return address and stack pointer it will come back to) is
keyed by thread id, not shared.

Every run ends with `hook windows: entries=N windows=W unprotected=U deferred=D timeouts=T protection=on|off` in the
console output, `finished.txt` and `files.txt`, also when everything is 0. `unprotected` (`U`) counts the exposed step windows in which another thread could
not be held (a thread that has already exited is not counted: it runs no code), or the deadline passed, or protection was off with other threads alive (one exposed window can miss multiple calls). `--suspend-others 0` turns the holding off, to measure what it costs or to see the loss.

`tests/startup_hook_check.py` has two parts. `startup_hook_window` checks the bookkeeping with fake threads and needs no
Windows. `startup_hook_overlap` runs the source-built `startup_launch_probe` in its `overlap` mode under the debugger: four threads
enter `CreateFileW` together 16 times each on one file, and all 64 opens must show in `files.txt` with their result and
`unprotected=0`. The control, `--suspend-others 0`, must lose opens and report `unprotected` above 0.

```powershell
cmake --build --preset dev --target startup_launch_probe
ctest --preset dev --no-tests=error -R '^startup_hook_(window|overlap)$' -V
```

## Switches

| Variable | Effect |
|---|---|
| `X2W_XMEM_PERMISSIVE=1` | accepts the reservation, mapping and allocation shapes a startup run uses |
| `X2W_XMEM_STARTUP_BELOW_40=1` (`run.ps1 -StartupBelow40`) | startup compatibility configuration ([#1758](https://github.com/R0Y00/Xbox2Windows/issues/1758)): the large-page startup blocks with attribute `0x283` (null address, type `0x20003000`, protection `0x2004` or `0x2404`) are placed so that the whole block, last byte included, lies below 2^40. A block that does not fit there returns NULL with the Windows error and logs `x2w-xmem startup block below 2^40 refused:`; nothing is truncated or placed higher. Other shapes and explicit addresses keep their placement. See [Startup blocks below 2^40](experiments.md#startup-blocks-below-240) |
| `X2W_LOW_ADDRESSES=1` (`run.ps1 -LowAddresses`; `tester_run.ps1` turns it on, `-NoLowAddresses` off) | [#1758](https://github.com/R0Y00/Xbox2Windows/issues/1758): `tools/startup/launch.py` and `debug_launch.py` start the title with bottom-up ASLR always off (`PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY`, `PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_OFF`), so the title's own `VirtualAlloc` blocks come from the low end of the address space instead of a random start above 2^40. The package acceleration-structure builder refuses a destination address with bits set above bit 39 and receives the title's CPU address, which the shim cannot move. `startup_launch` runs a probe four times with the switch (every block ends below 2^40) and eight times without (at least one block ends above 2^40). |
| `X2W_XMEM_LAZY_COMMIT=1` | experimental ([#1199](https://github.com/R0Y00/Xbox2Windows/issues/1199)): a fresh physical-page slot is committed on its first CPU access instead of when its page is mapped. A vectored exception handler commits the slot. Writers that cannot fault one in (the main module's `ReadFile`, slot 133 placement, stand-in writes behind a commit check) call `x2w_xmem_commit_range` first. See [Mapped pages committed on first touch](experiments.md#mapped-pages-committed-on-first-touch) |
| `X2W_XMEM_LAZY_STARTUP=1` (`run.ps1 -LazyMemory`, which also sets `X2W_XMEM_LAZY_COMMIT=1` and `X2W_XMEM_SHARED_VIEWS=1`) | experimental, with `X2W_XMEM_LAZY_COMMIT=1` ([#1199](https://github.com/R0Y00/Xbox2Windows/issues/1199)): the [startup commit profile](experiments.md#reserve-and-commit-startup-blocks) only reserves blocks of 32 MiB and up, and the same handler commits them 256 KiB at a time on first touch. Slot 133 commits a whole block (`x2w_xmem_commit_block`) before a GPU resource is placed in it. The main module's `VirtualFree` import drops released blocks. Blocks of 32 MiB and up placed below 2^40 by `X2W_XMEM_STARTUP_BELOW_40` are also reserved lazily. See [Startup blocks committed on first touch](experiments.md#startup-blocks-committed-on-first-touch) |
| `X2W_XMEM_GPU_MIRROR=1` | pairs a no-access graphics reservation with the read-write reservation of the same size made just before it, and maps every physical page into both (same section) |
| `X2W_D3D12_PERMISSIVE=1` | wraps the device: recording command lists, console present, view tracking; on a runtime without shader model 6.6 (in-box Windows 10), root signatures with the heap-indexing flags (`0xC00`) are serialized again without them, and the copy kept for shader translation gets that call's bits back when `CreateRootSignature` receives the returned blob's own buffer (a copy of the bytes keeps the desktop flags; `d3d12_xs_shim_heap_indexing`). The retry runs only on runtimes that refuse the flags (before shader model 6.6); where the runtime accepts them, `d3d12_xs_shim_heap_indexing` reports a capability skip (77) |
| `X2W_D3D12_RENDER=1` | draws through translated pipelines from `X2W_SHADER_CACHE` and forwards [single-record indirect Dispatch](experiments.md#gpu-written-indirect-dispatch-arguments), [DrawIndexed](experiments.md#gpu-written-indirect-indexed-draw-arguments), [Draw](experiments.md#gpu-written-indirect-draw-arguments), and [DispatchMesh](experiments.md#gpu-written-indirect-mesh-dispatch-arguments) for supported desktop pipelines and tracked placed buffers |
| `X2W_SHADER_CACHE=<dir>` | translation cache: the runtime writes source requests `req_<key>.{a,b,root}`, when needed RTV variants `req_<key>.rtv`, and for device slot 67 compute pipelines `req_<key>.{desc,sa,sb,data,root,c67}` (see [Compute pipeline requests](#compute-pipeline-requests)). A console pipeline that an `ExecuteIndirect` draw (list slot 61) refuses writes the same request as a direct draw, once per pipeline, and draws nothing (`startup_indirect_requests_warp`). Only direct draws poll for its manifest, so such a pipeline stays in the `missing` count until a direct draw uses it; the `x2w-render totals` line counts `pipelines first looked up by ExecuteIndirect N`. This is a per-process lookup count. Check for the ready marker `req_<key>.a` to confirm a complete request is available. Run `tools/startup/gfx2hlsl.py --pending <dir>` while the process is running, or start the run with `run.ps1 -Translate` (see [Translating shaders during a run](#translating-shaders-during-a-run)). Active draws poll for an absent manifest at most once per second per pipeline and use the published translation on a subsequent draw. The translator compiles `.vs.cso` and `.ps.cso` with `fxc.exe`: `X2W_FXC`, else the Windows SDK 10.0.26100.0 copy, else the newest `Windows Kits\10\bin\<version>\x64`, else `PATH`. When none exists it writes `<key>.fail` naming `X2W_FXC` without publishing a new manifest. `X2W_FXC=` (empty) publishes the manifest alone. Rerun `--pending` with compilation enabled to fill in missing bytecode for an existing manifest. |
| `X2W_D3D12_LINEAR_BUFFERS=1` or `=copy` | opt-in backing for linear buffers placed at a title address (device slot 133, initial state `COMMON`): shared where the adapter, extent and memory allow it (private allocations larger than 256 MiB are skipped only with `X2W_D3D12_CAP_EXISTING_HEAP=1`), otherwise copied from title memory at each submission, otherwise the default backing is kept and the refusal recorded; `copy` never shares. Unset or `0`, slot 133 keeps its default backing. See [Linear buffer backing (device slot 133, opt-in)](#linear-buffer-backing-device-slot-133-opt-in) |
| `X2W_D3D12_CAP_EXISTING_HEAP=1` | opt-in 256 MiB (`unplaced_existing_heap_allowed`) allocation-size cap on `OpenExistingHeapFromAddress` in device slot 133 (`startup_slot133_heap_cap_*`). Unset (default), private allocations larger than 256 MiB keep the formerly aliased path: a capped copied placement is refused by `CopyBufferRegion` (list slot 15) as `title_backed_source`, losing a late guest write made after recording and before submit. Set, allocations larger than 256 MiB skip `OpenExistingHeapFromAddress` so multi-GiB startup blocks are not pinned in `WorkingSetSize` and `NonLocal` GPU memory |
| `X2W_D3D12_QUERY_SLOTS=1` | diagnostic, in render mode: binds command-list slots 28 to 30 to the hypothesised occlusion query methods (see [Occlusion queries](#occlusion-queries)). Unset, the three slots stay no-ops |
| `X2W_D3D12_BUDGET=1` | adds [video memory budget and allocation counts](#d3d12-resource-budget) to the D3D12 heartbeat |
| `X2W_D3D12_RELEASE_RETIRED=1` | opt-in, effective with `X2W_D3D12_RENDER=1` only (the list handlers that record holds exist only there): frees the desktop object of a released committed 2D texture of one narrow class once no recorded list, submission or release-time fence refers to it ([details](#d3d12-resource-budget)). Unset, nothing changes |
| `X2W_D3D12_LAZY_TEXTURES=1` | opt-in ([#2764](https://github.com/R0Y00/Xbox2Windows/issues/2764), [#2379](https://github.com/R0Y00/Xbox2Windows/issues/2379)): defers desktop `CreateCommittedResource` for read-only textures placed through device slot 133 (`CreatePlacedResourceX`, without `ALLOW_RENDER_TARGET`, `ALLOW_DEPTH_STENCIL` or `ALLOW_UNORDERED_ACCESS`) until first real copy, CPU-real access or translated draw/compute bind; `slot 18` SRVs receive a null placeholder descriptor and refresh in place when the texture materializes ([details](#lazy-slot-133-textures-opt-in)). Unset or `0`, textures commit immediately |
| `X2W_D3D12_RELEASE_RETIRED_VIEWS=1` | legacy diagnostic. Logs that clearing live slots is disabled and enables the existing retirement breakdown when base retirement and budgets are on ([details](#d3d12-resource-budget)) |
| `X2W_D3D12_RETIRE_REASONS=1` | diagnostic only, effective with `X2W_D3D12_RELEASE_RETIRED=1` and `X2W_D3D12_BUDGET=1`: adds one `x2w-retire held:` line to the heartbeat that counts final-released resources whose desktop object is still alive, by reason ([details](#d3d12-resource-budget)). Unset, nothing is recorded |
| `X2W_D3D12_VIEW_DEFAULTS=all` or `=off` | `all` is experimental: UAV, RTV and DSV descriptions with `ViewDimension` 0 also get the resource's default view, which drops their requested format and component mapping. `off` skips the measured SRV case too. Unset, only SRVs of typed non-depth textures get the default view ([details](#unknown-and-mismatched-view-dimensions)) |
| `X2W_D3D12_VIEW_CENSUS=1` | diagnostic only: counts new distinct view records by device slot (a heartbeat line), and logs the record vector capacity and hash bucket count before either grows ([details](#d3d12-view-census)). Unset, nothing is counted or logged |
| `X2W_DRAW_NULL_IDENTITY=1` | null descriptors read as identity values instead of zero |
| `X2W_DEPTH_STATE=1` | translated draws of slot 57 pipelines take their depth-stencil state and stencil reference from state block A instead of the desktop default (depth and stencil off); a refused block keeps the default. Off by default, then nothing changes: see [Shortcuts that are known to be wrong](#shortcuts-that-are-known-to-be-wrong) |
| `X2W_PRESENT_SIGNATURE_EVERY=<n>` | in render mode, samples a coarse signature of the presented picture every `n` presents from the start, without a script request (`0` off). Unset, sampling starts, every 4 presents, only once a script with a `wait` asks for it: see [Waiting for the presented picture](#waiting-for-the-presented-picture) |
| `X2W_D3D12_DUMP_SURFACE=<bmp>` | writes the last presented surface |
| `X2W_PRESENT_COMPOSITE=1` | experimental, in render mode: the output window composites every presented plane, bottom to top, instead of showing plane 0 alone, and the highest plane above plane 0 is sampled as `x2w-render raw plane <i> (not composited)`. Off by default; the baseline capture and `x2w-render presented` read plane 0 either way. See [Presented planes](#presented-planes) |
| `X2W_PRESENT_PLANE=<n>` | diagnostic, in render mode: the output window shows plane `n` of each present alone (plane 0 when `n` is past the count), with or without `X2W_PRESENT_COMPOSITE`. See [Presented planes](#presented-planes) |
| `X2W_HIDE_TITLE_WINDOW=1` (`run.sh --hide-title-window`) | once a present has reached the output window, hides the title's own window; dialogs stay visible, and the window is shown again when a present misses the output window, until the next present reaches it ([Output window and fallback diagnostics](#output-window-and-fallback-diagnostics)). Keyboard input to the title after the hide is not measured. Off by default on every host |
| `X2W_LIVE_LOG` (read by `tools/startup/debug_launch.py`) | the live debug-log window of a run: unset or `minimized` starts it minimized and without taking the focus (so it does not cover the game or appear in window screenshots); `normal` opens it in front as before; `off` opens none. Any other value is the default. |
| `X2W_DRAW_DUMP_FRAME=<n>`, `X2W_DRAW_DUMP_DIR`, `X2W_DRAW_DUMP_PIXEL=<x>,<y>` | Full-resolution RGB snapshots of single-sample RGBA8/BGRA8 UNORM or sRGB targets after translated draws. A later present writes completed queue-fenced copies. At most 48 copies are pending or written; a copy whose fence reports device removal (`UINT64_MAX`) is released unwritten and frees its slot (`src/startup_stubs/draw_dump_tracking.h`, `ctest --preset dev --no-tests=error -R '^startup_dump_fence_(removed_control_)?warp$' -V`; the control must fail). With `X2W_DRAW_DUMP_PIXEL`, each copy covers one texel instead and the present writes `pixel_history.txt` in `X2W_DRAW_DUMP_DIR`: one line per translated draw with `draw=`, `frame=`, `x=`, `y=`, `key=`, `count=`, `format=`, `raw=` and, for an 8-bit four-channel target, `rgba=`. The bounded history names the last draw that changed the texel (`src/startup_stubs/pixel_history.h`, `ctest --preset dev --no-tests=error -R '^startup_pixel_history_' -V`). |
| `X2W_TEARDOWN_REPORT=1`, `X2W_TEARDOWN_NO_GUARD=<pages, readbacks or all>` | Test accounting for the caches whose destructors run during the module's CRT teardown (`src/startup_stubs/d3d12_teardown.h`, #2572). With `X2W_TEARDOWN_REPORT`, each destroyed cache writes one `x2w-teardown <cache> detached=<n> released=<n>` line to stderr with `WriteFile`, because a stdio buffer would not be flushed from a CRT teardown. `X2W_TEARDOWN_NO_GUARD` names a cache whose detach at process exit becomes an ordinary release again: the negative control `startup_teardown_unload` runs against itself, where the termination check has to fail. Unset, both change nothing and no line is written. |
| `X2W_AUDIO_PERMISSIVE=1` | enables the audio stand-ins. Without a DSP provider, `XDspActivate` clears valid outputs and reports unavailable resources, and failed `XDspSubmitCommand` calls leave the sequence unchanged. `XDspDeactivate` and `XDspDisconnect` refuse null and unknown handles with `E_INVALIDARG`. |
| `X2W_RUNTIME_PROXY=1` | proxies the gaming runtime and logs every call. Rewrites regular-IAT `GetProcAddress` and a direct `xgameruntime.dll!QueryApiImpl` import on the main EXE and on DLLs that share its directory (package helpers). Dynamic `QueryApiImpl` / `InitializeApiImpl*` are wrapped only when the provider module's basename is `xgameruntime.dll`. `LdrRegisterDllNotification` only flags that another scan is needed. A worker holds each matching module live while rewriting after linking. Loading a helper and immediately calling it can precede the scan. The enabled proxy stays loaded until process exit. Delay-import IAT slots are not rewritten. Skips this stand-in's own `GetProcAddress` IAT so the hook does not recurse (`src/startup_stubs/runtime_gpa_patch.h`, `ctest --preset dev --no-tests=error -R '^startup_runtime_gpa_patch(_name_only_control)?$'`, and the control must print `FAIL:`; Windows also runs `startup_runtime_proxy_scope` for a late-loaded helper through the real loader) |
| `X2W_FAKE_USER=1` | answers the user, gamertag and save APIs for one local user, and creates one offline store context for that user |
| `X2W_RUNTIME_LOCAL=1` | with `X2W_RUNTIME_PROXY` and `X2W_FAKE_USER`, for a host where the system gaming runtime cannot initialize (no Gaming Services). The first failed `InitializeApiImpl*` is reported as `S_OK`, and every later initialization call returns `S_OK` without calling the real export. Once initialization failure is latched, a failed real `QueryApiImpl` call can fall back only with an exact match of both GUIDs to a supported provider/interface pair for user (including gamertag), game save, package, title, connectivity or store. Persistent local storage uses its existing fake-user gate and requires its exact provider GUID and IID. Unmatched failed queries keep the real HRESULT. The proxy clears a writable output on failure only when fake user is enabled and the initialization failure is latched. A local object's `QueryInterface` accepts only `IID_IUnknown` and the IID it was created for. The local answers above apply, calls that carry the local user or store handle fail as before, and other unanswered methods return `E_NOTIMPL`. `XUserAddAsync` and async operations completed on behalf of a local answer require the real runtime's `QueryApiImpl` to return the XAsync provider (`073b7dcb`). Those operations complete with `S_OK` through XAsync, or return the query failure if XAsync is unavailable (`ctest --preset dev --no-tests=error -R '^startup_runtime_local(_control)?$'`, and the control must print `FAIL:`). |
| `X2W_SKIP_MOVIES=<mode>` | with `X2W_RUNTIME_PROXY=1`, opt-in: the title's imports of its movie player (`bink2scarlett.dll`: open, close, next frame, goto, do frame, wait, should skip, pause) are forwarded to the real functions with a change per mode. `1` or `goto_end`: after a playback's first frame the real goto jumps to the last frame, so the title sees the movie end at once; `frames1`: the header's frame count is set to 1 after the open; `open_null`: the open returns null; `log`: nothing changes, the first calls and the movie header are logged. Unset or `0`: nothing is rewritten. Delay-load imports and pointers fetched with `GetProcAddress` are not rewritten. Tested with a stand-in DLL and player (`startup_movie_skip_*`). |
| `X2W_DEVICE_TYPE=<n>` | with `X2W_RUNTIME_PROXY=1`, answers the title's reads of the OEM `ConsoleMode` registry value: `6` = 0x60 (Series S), `7` = 0x64 (Series X), `1` = value missing (PC); unset, the real registry answers. See [Console profile](#console-profile) |
| `X2W_STORAGE_PATH`, `X2W_STORAGE_MB`, `X2W_TITLE_ID` | persistent local storage and the manifest values the runtime would return. The path is the root of the drive the title sees, not its save folder: the title appends `x2w-gamesave` itself. It is UTF-8, with a trailing separator added when missing. If the path is invalid or exceeds 259 bytes including the separator, the proxy logs the refusal and uses `S:\` (`src/startup_stubs/storage_root_utf8.h`, `ctest --preset release --no-tests=error -R '^startup_storage_root'`). |
| `X2W_INPUT_SCRIPT=mask@sec,...` | timed virtual gamepad presses, holds, stick or trigger values and waits for the presented picture: see [Input script](#input-script); keyboard and XInput pad 0 also work. Up to 256 entries and 4,095 characters; a longer value is ignored and logged, and dropped or malformed entries are logged (`ctest --preset dev --no-tests=error -R '^startup_gameinput_script_(long|entries|syntax)$'`) |
| `X2W_SAMPLER_HEAP=1` | `gfx2hlsl.py` translates a sampler-heap load with a run-time offset into `SamplerState qh[2048] : register(s0, space1)` and writes `smpheap 2048 1 fallback 2047`. The stand-in uses entries 0 to 2046 for the bound heap and reserves entry 2047 for unaligned or out-of-range offsets. It also accepts the original `smpheap 2048 1` manifest with all 2,048 entries available. [#1143](https://github.com/R0Y00/Xbox2Windows/issues/1143) |
| `X2W_XG_VERBOSE=1` | logs every texture layout query with its mip sizes |
| `X2W_XG_BC7_4K_S_LAYOUT=1` | enables the BC7 tiled layout, copy and draw upload when `X2W_D3D12_PERMISSIVE=1` |
| `X2W_TRAP_CONTINUE=<exports>` | lets listed trap exports return 0 instead of ending the process |
| `X2W_STOP_AT=<rva>[,...]` | the startup debugger stops at a main-image breakpoint at one of these addresses even while `-ContinueAsserts` steps over the others, and writes every thread's stack to `threads_stop.txt` |
| `X2W_LOG_AT=<rva>[@<max hits>][:<item>[;<item>...]][,...]` | logs the elapsed time, thread ID, main-image instruction RVA, hit number and selected values to `files.txt`. RVAs and offsets are hexadecimal. The hit limit is decimal and defaults to 16 per RVA. Items are `rax`, `rcx`, `rdx`, `rbx`, `rsp`, `rbp`, `rsi`, `rdi`, `r8` through `r15`, or `[reg+hex]` / `[reg-hex]` for a qword read (`unreadable` on failure). Omit the item list to log all 16 registers. At the limit the debugger restores the original instruction unless another hook shares the breakpoint. INT3 events that other threads raised before the restore are resumed at the restored instruction, not logged, and counted on a separate `LOG_AT late hits resumed after the limit` summary line. Malformed entries (including a repeated item, compared by register and offset) and unusable breakpoints are reported in `events.txt`. The end summary lists logged hits per RVA. Fixture: `ctest --preset dev --no-tests=error -R '^startup_debug' -V`. |
| `X2W_NOP_SOFT_ASSERTS=1` | after `-ContinueAsserts` steps over a main-image `int3` for the first time, the startup debugger replaces its `cc` byte with a `NOP`, so a routine that traps once per chunk runs at native speed. The patch lasts for the life of the process and only targets RVAs of the main image; `X2W_STOP_AT` sites are never patched. `DBG_CONTINUE` already bypasses the title's handler, so the visible behavior is unchanged. Once a site is patched it no longer raises an exception and `asserts_seen` stops counting it, so the `--continue-asserts N` budget no longer measures real hits at that site |
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
then start the build again.
The first request of a process also writes the runtime's revisions to `<cache-directory>/revision.txt`
(`graphics`, `compute` and `mesh57` lines). With `-Render -Translate` (`--render --translate`),
`run.ps1` and `run.sh` compare that file with the `gfx2hlsl.py` they start
(`tools/startup/shader_cache_revision.py`) and print one warning naming both revisions when they differ,
or when the cache holds requests but no `revision.txt`: every cached request would be refused as a key
mismatch. The warning comes before the watcher starts, or after the run when the run's own stand-ins
wrote the first `revision.txt`. Rebuild the stand-ins from the same checkout, or start with an empty cache:
`-FreshShaderCache` (`--fresh-shader-cache`) moves a cache that is not empty to `<cache>.old-<date>-<time>` first
([#2815](https://github.com/R0Y00/Xbox2Windows/issues/2815)). The new key keeps the former bytecode and manifest
together under their original name. A 16-dword table load can contain two
independent image descriptors at offsets 0 and 32 from the load address.
For a read-only census of cached shader refusals, use `python -B tools/startup/gfx2hlsl.py --refusals <cache-directory> [--json]` with a translator that supports the flag. Earlier versions treat it as `--pending` and write into the directory. See [Startup translator refusal census](experiments.md#startup-translator-refusal-census).

## Console profile

`X2W_DEVICE_TYPE` answers the two registry reads a title makes of
`HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM\ConsoleMode`
(`src/startup_stubs/console_mode.h`, `registry_console_mode.inl`): with `X2W_RUNTIME_PROXY=1` the
stand-in rewrites the registry imports of the EXE and package helpers. `6` answers 0x60 (Series S),
`7` answers 0x64 (Series X) and `1` reports the value missing (PC); unset, every call goes to the real
registry. Only registry imports that the Windows registry provides are rewritten: an IAT entry must
name `advapi32.dll`, `kernelbase.dll`, `kernel32.dll` or an `api-ms-win-core-registry-*` API set and be
bound inside the system-directory copy of the first three, and `GetProcAddress` must be called on one
of those modules. A package DLL that exports the same names keeps its own exports and is never kept as
the real registry. Opening the key and reading the value are logged a few times either way, and a
seventeenth OEM key handle open at once is logged once and not tracked. What the values mean on the
console is taken from public notes, not measured here.
`ctest --preset dev --no-tests=error -R '^startup_console_mode(_control|_provider_control)?$'` runs it
against fake registry functions and a source-built decoy DLL with the same export names, with the switch
unset and set. The control leaves the switch unset for the answered cases and the provider control
matches registry imports by name alone; both must print `FAIL:`. `startup_runtime_proxy_scope` checks the
same decoy rule through the real loader against `advapi32.dll`.

The switch changes only what the registry reads answer. A title can take a different start-up path
for each profile: on the Series S profile, one loaded an upscaler package module that the other
profiles do not, and that module met desktop refusals the other profiles did not reach
(`Device::slot76`, views with `ViewDimension` 0, and `R9G9B9E5_SHAREDEXP` textures with unordered
access, the last covered by the
[shared-exponent stand-in](#shared-exponent-render-and-unordered-access-targets)). Choosing a profile
is not evidence that the path it leads to runs.

A title on the Series S profile may present its picture on a plane above plane 0 and leave plane 0
black (see [Presented planes](#presented-planes) for a measured case). The output window shows plane 0 alone by
default, so it then stays black. `X2W_PRESENT_COMPOSITE=1` composites the planes, and
`X2W_PRESENT_PLANE=<n>` shows one plane alone. Whether other titles, or other profiles, present
this way is not known. Under either switch, `X2W_D3D12_DUMP_SURFACE` and the `x2w-render presented`
readback still read plane 0. With `X2W_PRESENT_COMPOSITE=1`, the `x2w-render raw plane <n>` line
shows whether the higher plane holds a picture.

## D3D12 resource budget

Set `X2W_D3D12_BUDGET=1` before startup to add one `x2w-budget` line to each
D3D12 present heartbeat. `local` and `nonlocal` show DXGI current usage and
budget in bytes for node 0 of the adapter matching the desktop device LUID.
An unavailable segment reads `n/a`.

The line ends with the system commit and the physical-page pool (#1199):
`commit=<used>/<limit> avail=<free>` in MiB (all processes: RAM plus page files, what a new commit can still
get) and `sections=<n> slabs=<n> slots=<in use>/<free>` (live 64 KiB physical-page sections, open slabs, slots in
the open slabs), then `job_peak=<MiB> job_limit=<MiB>`: the peak commit of the job the title runs in and its memory
limit (`run.ps1 -MemoryGiB`; `none` without a limit). Each reads `n/a` when it cannot be read. A run that ends on error 1455 shows from these whether
the pool (a flat `sections`) or the resources below (committed texture bytes) grew. The first three commit failures
(error 1455 or 8) of the page pool also
log `x2w-xmem commit exhausted at <call> (error N): job memory limit N GiB reached (run.ps1 -MemoryGiB)` when the job peak is within 256 MiB of its limit (otherwise `job memory limit not reached`), then the job numbers, `system commit=... avail=...` and `pool sections=... slabs=...
slots=.../...` with the hint that `X2W_XMEM_LAZY_COMMIT=1` (`run.ps1 -LazyMemory`) keeps untouched pages out of
the commit. `tests/startup_commit_state_check.cpp` checks the text, `tests/startup_triage_run_check.py` that
`triage_run.py` still reads the line with and without these fields.

Each allocation category reads `live-bytes/live-objects+retained-bytes/retained-objects`.
The categories are desktop placed resources, committed textures, committed
buffers, upload buffers, readback buffers, explicit heaps, and `sysmem` (textures
created in system memory by the fallback below). Resource sizes
come from the desktop allocation info for the created description. A slot 133
texture or buffer with committed backing counts in its committed category.
Each opened or created heap uses its desktop heap size once, even when several
placements share it. Placed resource sizes describe the objects occupying
heaps, so adding both columns would count the backing twice.

Final guest release moves a wrapped allocation to the retained column when
the shim still owns the desktop reference. An explicit heap stays live while
its wrapper or a live placement uses it. A cached opened heap moves to
retained when its last placement retires. Internal resources such as the draw
upload ring, slot 148 pages, tiled SRV materializations, and readback buffers
stay live until their desktop references are released. The diagnostic leaves
those lifetimes intact. `unmeasured` counts tracking failures. DXGI usage can
also include allocations outside these categories, such as swap chain and
descriptor heap backing.

### System-memory fallback for textures

`X2W_D3D12_SYSMEM_FALLBACK=1` (opt in): when the desktop `CreateCommittedResource` of a slot 133 texture in a default
heap fails with `E_OUTOFMEMORY` (the eager path and the lazy first-use path alike), the shim retries once with a custom heap
in the system pool (`MEMORY_POOL_L0`). The CPU page property depends on the adapter (`D3D12_FEATURE_ARCHITECTURE1`): a
shared-memory adapter tries `NOT_AVAILABLE`, then `WRITE_COMBINE`, then `WRITE_BACK`; a discrete adapter tries
`WRITE_COMBINE`, then `WRITE_BACK`, because `NOT_AVAILABLE` belongs to the video-memory pool there. If every retry fails the
caller gets the original `E_OUTOFMEMORY`, as without the switch. Any other failure, and any other heap type, is not retried.

Such a texture is counted in the `sysmem` category (its ticket is released with the resource, like every category), and
the first 32 fallbacks log one `x2w-d3d12 sysmem fallback` line each. The `x2w-budget` line ends with
`sysmem-fallback=N bytes=M`, the total made so far (released or not). `tools/startup/tester_run.ps1` turns the switch on;
`-NoSysmemFallback` turns it off. Source: Microsoft D3D12 documentation for `D3D12_HEAP_PROPERTIES`, `D3D12_MEMORY_POOL` and
`D3D12_CPU_PAGE_PROPERTY`. The rules are in `src/startup_stubs/sysmem_policy.h` (`sysmem_policy` test) and the retry in
`src/startup_stubs/sysmem_create.h` (`startup_sysmem_fallback_{warp,hardware}`: a forced out-of-memory, upload, a sampled
read-back against a CPU reference, and the controls).

`X2W_D3D12_RELEASE_RETIRED=1` ends the retained lifetime for committed
2D textures that have one sample, no render-target, depth-stencil or
unordered-access flag, and live in a default heap without heap flags:
- **Slice 1**: slot 27 and slot 127 textures without title memory (if the title
  exports `__XMemAllocationHooks__`, every slot 127 resource gets title memory
  and stays out of the class).
- **Slice 2**: slot 133 (`CreatePlacedResourceX`) committed 2D textures with
  title memory. Before their desktop `ID3D12Resource` is destroyed, they are
  erased from `resource_address_index` and `texture_uploads`, and stale
  `written_textures` entries whose generation has not changed are cleared so
  reusing the same title address or desktop pointer does not skip uploads or
  resolve a retired wrapper.

A slot-18 descriptor and each tracked copy hold the resource until the slot is explicitly overwritten
or its heap is destroyed. A later binding of that slot still reads the resource after the last guest
Release and the release-time fences. A list or submission holding the resource keeps it alive after
the slot is overwritten, until the recording is discarded or the submission retires.

The legacy `X2W_D3D12_RELEASE_RETIRED_VIEWS=1` setting only enables the retirement breakdown below and
logs that clearing live slots is disabled. The memory-pressure A/B in
[#2379](https://github.com/R0Y00/Xbox2Windows/issues/2379#issuecomment-7179801296) used the earlier
experimental rewrite. `startup_resource_retire_views_*` checks GPU readback from the original and
copied descriptors, and from a recorded copy behind a gated queue. It also checks retirement after
explicit overwrite and normal completion, with the legacy setting on and off.

`X2W_D3D12_RETIRE_REASONS=1` (with `X2W_D3D12_RELEASE_RETIRED=1` and
`X2W_D3D12_BUDGET=1`) adds a diagnostic line to the heartbeat. It counts every
resource whose final guest Release happened and whose desktop object is still
alive, once, by the first cause that applies. Format: `reason=count/bytes(oldest Nf)`,
bytes from the budget ticket, `oldest` in frames since the final Release:
- `view-refs`: a tracked descriptor slot still names it (a slot 18 SRV, and since
  Slice 3a a slot 19 UAV, 20 RTV or 21 DSV); `view-refs-in-visible-heap` is the
  slot-18 part with a descriptor in a shader-visible heap.
- `list-hold`, `flight`: a closed or recording list, or a submission, holds a
  desktop reference.
- `queued-waiting-fence`: queued, the release-time fence of a queue has not passed;
  `drain-pending`: ready, the drain has not run yet; `unqueued`: no queue could take its marker.
- `left-class:<view|view-copy|residency|present|hold-missed>`: it was in the class and the first use named left it.
- `never-class:<buffer|dimension|msaa|rt-ds-uav|heap-type|heap-flags|title-memory|aliased|reserved|slot>`: it was
  never in the class (`slot`: created by a path that does not classify).
- `other`: none of the above.
`total` is the sum; `unattributed` is the budget's retained count and bytes minus
`total` (backing heaps, upload and readback objects, and anything created outside the wrappers).
Wrappers without a desktop object (tiled address textures) are not counted. A
released resource that only a list or submission keeps alive after its wrapper
let go is matched by its desktop pointer. Without the switches nothing is
recorded and no line is built.

Every other resource (buffers, placed buffers, reserved resources, render/depth/UAV
targets, and slot 127 `__XMemAllocationHooks__` textures) keeps the lifetime
described above. A descriptor slot the shim shadows (`HeapInfo::shadow`) holds one
reference for each materialized view it names: a slot 18 SRV, and since Slice 3a
also a slot 19 UAV (one for the view resource and one for its counter resource),
a slot 20 RTV and a slot 21 DSV. `note_view` adds the references when a view is
written into a tracked slot (writing the same view again adds nothing), and the
same functions release exactly the references they added: `note_view` on an
overwrite, `note_copy` when a copy moves another view over the slot or into
another tracked slot, and the heap's `Release` through `retire_drop_heap_views`
when the slot's heap is destroyed. A view the desktop did not create (a removed
device) is recorded unmaterialized: it releases the slot's previous references
and adds none. A resource leaves the class for good when a view call targets a
descriptor slot no heap shadow tracks, or a copy moves a tracked view into an
untracked heap (`retire_leave_class`, slots 18-21, a slot 19 counter too), when
it is passed to MakeResident or Evict (slots 34 and 35), or when it is presented.
Tracked views (including slot 70 `device_memory_srv`) keep the resource retained
while any descriptor slot in `HeapInfo::shadow` names it (updated across
`note_view`, `CopyDescriptors*`, and heap `Release`), and translated draws hold
bound textures via `retire_hold`. Only wrappers can be in the class; a raw
desktop pointer never is. Sampler feedback is not checked: no device slot of the
stand-in reaches it.

The 19-21 references free nothing on their own yet: the class still excludes a
texture created with the render-target, depth-stencil or unordered-access flag
(`retire_classify`), and desktop creates a UAV/RTV/DSV only on a resource that
carries the matching flag. A class texture's 19-21 view call is therefore the
console-only case the shim refuses (a UAV of a depth texture, an RTV of a texture
without render-target); it records no view, adds no reference, and the texture
still retires once it is released and its fences have passed. The accounting is
the prerequisite for a later slice that admits those textures. The slots 19-21
fallback is unchanged: in a heap the shim does not shadow the call keeps
`retire_leave_class`, so nothing is freed under a descriptor the shim cannot
follow. `X2W_D3D12_RELEASE_RETIRED_VIEWS` no longer clears live descriptors; it is a legacy
diagnostic that logs that clearing is disabled and enables the retirement
breakdown. A tracked descriptor reference in any of slots 18-21 keeps its resource
until the slot is overwritten, copied over, or its heap is destroyed (#2679).

The final guest release signals a private desktop fence on every queue the
stand-in created. The desktop object is released when all of those fences have
passed, all tracked descriptor references (slots 18-21) have dropped to zero, and
no recorded list or submission still holds it. A copy (list slot 17), texture copy (16),
resolve destination (19), or translated draw SRV bind that the stand-in forwards
makes the list hold a desktop reference to each resource of the class it names,
source and destination, until the list is Reset, fails to Close or is destroyed. A list
that is closed and never submitted is thus discarded without any submission.
Each submission holds the same references until its queue passes a fence
signalled behind it. A release with no registered queue, and a hold, marker or
submission fence that cannot be recorded, keep the resource retained. A use that
`retire_hold` cannot record (no desktop object left, or no room for the entry)
leaves the class and marks the resource so the drain keeps its desktop object for
the process, as an unfenced submission is kept, instead of dropping an object a
list still names. Nothing waits: `ExecuteCommandLists` and `Present` check and
release without blocking.
The wrapper stays allocated, and a guest that uses a resource after its final
release is not supported with the switch. Slot 18 (`CopyTiles`) is not
forwarded at all and is outside this change.

The source-built fixture reads the process-wide counters through
`x2w_d3d12_budget_snapshot` and checks allocation sizes, final guest release,
and slot 148 page retirement after queue completion. See
[D3D12 budget accounting](experiments.md#d3d12-budget-accounting) for the test
command.
`startup_resource_retire_{warp,hardware}_debug{0,1}` use the retained column as
the oracle that a desktop object is destroyed, and read results back through
their own desktop queue. They cover two closed lists on one texture that are
submitted separately after the release, a Reset or destroyed list with no
submission (the destination stays unchanged), a second submission held behind
its own fence, a destination released after recording (slots 17, 16 and 19), a
list destroyed while its submission is held, a copy queue, an open list that
D3D12 refuses to Reset and one that is destroyed, a release that no list
refers to behind a held queue, and, in one run with the class, a buffer,
render target, viewed texture (while its slot-18 SRV stays live), resident texture and
presented texture that must stay retained, alongside slot 133 title-memory textures.
`startup_resource_retire_title_{warp,hardware}_debug{0,1}` additionally test slot 133
`CopyTextureRegion`/`CopyResource` upload holds, same-address slot 70 (`existing_memory_texture`)
reuse with `X2W_RESOURCE_INDEX_CHECK=1`, stale `written_textures` `bc7_width` cleanup,
slot-18 SRV `CopyDescriptorsSimple` / overwrite / heap-release lifetime, and, with the
switch on, a newer `written_textures` generation kept when the older generation retires
and a translated draw that samples such a texture through a descriptor table (the view
token's resolution and the draw's `retire_hold`: the readback holds the sampled texels and
a clear sentinel corner, and the texture leaves the budget only after the closed draw
recording is dropped); the `_off_` case checks that an ordinary committed texture (no
title memory) still records its upload. Since Slice 3a the title fixture also drives the
19-21 descriptor references: case 8 creates an RTV, a DSV and a UAV (and a buffer UAV with
a counter resource) in tracked heap slots and checks through the retire probe that each
materialized view adds one reference per slot - slot 19 one for its resource and one for
its counter - that the same view written again adds nothing, and that overwrite, a
`CopyDescriptorsSimple` copy and heap destruction each release exactly what was added
(including a view in an untracked desktop heap, which adds none); case 9 checks the
`view_slot` fallback in both switch modes: a refused RTV on a class texture in a tracked
slot stays in the class and retires, while the same call into a descriptor slot the shim
does not shadow keeps `retire_leave_class` and the texture stays retained. The fixture-only
exports `x2w_d3d12_retire_probe` (which reports the resource's tracked slot references as
`view_refs`), `x2w_d3d12_resource_index_probe`, `x2w_d3d12_texture_upload_probe` and
`x2w_d3d12_written_probe` read the bookkeeping the shim keeps; the retire probe returns
FALSE without the switch, like `x2w_d3d12_budget_snapshot`. `startup_resource_retire_off_*`
runs the same steps without the switch. The hardware cases report 77 without a
physical adapter, and the debug-layer cases report 77 where the debug layer is not installed.

### Lazy slot 133 textures (opt-in)

With `X2W_D3D12_LAZY_TEXTURES=1`, read-only textures placed at a title address through device slot 133 (`CreatePlacedResourceX`, without `D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET`, `D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL`, or `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`) defer their desktop `CreateCommittedResource` and `x2w_budget` ticket until first materialization. Until then:
- `Resource::GetDesc` (slot 10), `view_slot` validation, and capture metadata read the sanitized desktop `D3D12_RESOURCE_DESC` from the side record under `lazy_texture_mutex`, and `ID3D12Object::SetName` returns `S_OK` without materializing.
- `view_slot<18>` (SRV) creates a null `R8G8B8A8_UNORM` placeholder descriptor of the requested dimension and records the view normally in `HeapInfo::shadow` and `title_memory` (`X2WV` token) under `lazy_texture_mutex -> note_lock`.
- First real use (`CopyTextureRegion` subresource side, `CopyResource`, `ResolveSubresource`, `sampled_address_texture` in translated graphics/compute draws, `Resource::Map` without `title_memory`, or `WriteToSubresource`/`ReadFromSubresource`/`GetHeapProperties`) materializes the desktop resource once under `lazy_texture_mutex` (lock order `draw_lock -> lazy_texture_mutex -> note_lock -> retire_lock`), attaches its budget ticket, and refreshes every live tracked `slot 18` descriptor in `HeapInfo::shadow` (including `CopyDescriptors*` copies), revalidating heap membership and `HeapInfo::shadow` slot identity under `note_lock` so concurrent slot overwrites or heap retirement are not clobbered; `dispatch_translated` also records `retire_hold` so in-flight compute uses after final guest `Release()` stay live behind their fence.
- Untouched textures remain at 0 desktop bytes and clean up their side records and `resource_address_index` entries on final guest `Release()` (immediately when no live view remains, or via `drain_retired_resources()` under `X2W_D3D12_RELEASE_RETIRED=1` once tracked `slot 18` descriptor references and release-time fences clear).

Fixture: `startup_slot133_lazy_textures_{warp,hardware}_{debug0,debug1}` plus `noretire` and `off` controls (`tests/startup_slot133_lazy_textures_check.cpp`).

## D3D12 view allocation failure

A new view record goes into three containers: the record vector, the vector of
materialization flags and the map from content to id. If one of them cannot
allocate, the record goes into none of them, the process continues and the first
eight failures are logged:

`x2w-d3d12 note_view: out of memory in <view_records|view_materialized|view_ids> -> view of slot <n> not recorded, destination descriptor untracked`

The device's view slots return nothing, so the log line is the only report. The
destination descriptor is left untracked: its shadow entry and its token in
title memory are cleared, and the view it named before loses its reference. A
draw then sees a descriptor without a view, not the earlier one. If the desktop
view was created, its resources leave the retire class of
`X2W_D3D12_RELEASE_RETIRED` and are not released early. Earlier ids stay valid,
and a later view is recorded if its allocations succeed. Records already copied
elsewhere are not touched.

This handles the failure; it does not reduce the memory the records use, and it
does not cover other allocations, such as the map of render and depth targets.

`startup_view_alloc_<records|materialized|ids>_<warp|hardware>` fails the
allocation at one site three times through the fixture-only export
`x2w_d3d12_fail_view_allocations`. The failure is injected, not produced by
exhausting memory. For `view_ids` it is raised before the insertion, so the
fixture relies on the standard guarantee that a failed single insertion leaves
an `unordered_map` unchanged. The `control` variants never inject it and must
fail. The failure is exercised on slot 18 and on a slot 19 view with a counter
resource. For slot 19 only the reference counts are checked: resources that
allow unordered access are never in the retire class.

```powershell
ctest --preset dev --no-tests=error -R '^startup_view_alloc_' --output-on-failure
```

## D3D12 view census

The stand-in keeps one record per distinct view: slot, resource, counter,
description bytes and whether the desktop view was created. Records are not
removed. `X2W_D3D12_VIEW_CENSUS=1` shows which slot adds them. It changes no
record, id or descriptor binding.

The heartbeat adds a line:

`x2w-view-census distinct by slot: cbv= srv= uav= rtv= dsv= sampler= records= capacity= ids= buckets= grows-records= grows-ids= missed=`

- `cbv` to `sampler`: new records from device slots 17 to 22.
- `records`, `capacity`: size and capacity of the record vector.
- `ids`, `buckets`: size and bucket count of the hash map from content to id.

Before an insertion that is expected to grow either container, the same fields
are logged as `x2w-view-census before growth of <records|ids|records and ids>`,
with the slot of the new record. The values are those before the growth. The line
is written before the allocation, so it is in the log when that allocation fails,
also when no further heartbeat follows. The growth is predicted: from the
vector's capacity, and for the map from the MSVC rehash rule (the new size over
the bucket count exceeds the maximum load factor). `missed` counts capacity or
bucket changes that no line announced.

The census counts distinct byte contents, not distinct views. It does not say
why two records differ (for example a changing address, or description bytes
that the view type does not use), and it does not measure the size of the
containers in bytes or reduce it.

`startup_view_census_<warp|hardware>` calls slots 17, 18 and 22 of the device.
A repeated descriptor must add no record and grow nothing. A stream of 200,000
distinct constant buffer views, 3,000 buffer SRVs and 100 samplers must be
counted by slot, with one log line per announced growth. Both streams run under
a job commit limit 256 MiB above the commit held after setup. `off` checks that
nothing is reported without the switch; in `vary` the repeated stream changes
every call, and the fixture must fail. Slots 19 to 21 are not exercised.

```powershell
cmake --build --preset dev --target startup_view_census
ctest --preset dev --no-tests=error -R "^startup_view_census" -V
```

## Wrapper allocation failure

Every object the D3D12 stand-in returns is a wrapper allocated in `wrap()`. When that allocation (or its
registration) fails, the creating call releases the desktop object it made and returns `E_OUTOFMEMORY` with
a null output ([#2653](https://github.com/R0Y00/Xbox2Windows/issues/2653)). Before, the failure was a write
through a null pointer inside the stand-in. This was seen at the job commit limit in a startup run, at the
slot 55 derived-pipeline placeholder. Whether the title handles the error is not established.

`startup_wrap_allocation_warp` makes the next wrapper allocation fail through the fixture-only export
`x2w_d3d12_fail_wraps`. Device creation and slots 8, 9, 14, 16, 55, 73 and 127 must return
`E_OUTOFMEMORY` with a null output, then succeed. After a failed attempt to create a second device,
`GetDevice` on a live queue must still return the original device. The
`startup_wrap_allocation_control_warp` test leaves the failure injection disabled and checks that
the fixture detects the unexpected success.

```powershell
cmake --build --preset dev --target startup_wrap_allocation
ctest --preset dev --no-tests=error -R "^startup_wrap_allocation" -V
```

## Input script

`X2W_INPUT_SCRIPT` (`run.ps1 -InputScript`) drives the virtual gamepad of the `GameInput.dll`
stand-in. It is a comma-separated list; empty items and blanks around an item are ignored. Each
item is one of:

| Item | Effect |
|---|---|
| `<mask>@<s>[+<d>]` | sets the GameInput button bits `<mask>` (hexadecimal, `0x` optional; 4 = A, 8 = B, 1 = Menu) from second `<s>` for `<d>` seconds, default 0.3 |
| `<axis>=<value>@<s>[+<d>]` | sets one axis from second `<s>` for `<d>` seconds, default 0.3. `lx`, `ly`, `rx`, `ry` are the thumbsticks in [-1, 1] (+1 is right or up); `lt`, `rt` are the triggers in [0, 1]. These are the units of the GameInput gamepad state. |
| `base=poll` or `base=start` | time base of every entry; first item only. `poll` (the default) counts seconds from the first input poll of the process, `start` from process start. |

Entries that are active at the same time combine: button masks are OR-ed; for one axis the later
item wins. When an axis entry ends the axis returns to 0. Keyboard and XInput values still merge
in (buttons OR-ed, for an axis the larger magnitude wins). Examples:

```text
4@60,1@70,20@80                  A, Menu, then Y for 0.3 s each, counted from the first poll
4@60+2,8@61                      A held 2 s; B pressed at 61 s while A is still held (mask 0xC)
base=start,lx=-1@40+1.5,rt=1@41  left stick fully left for 1.5 s and the right trigger at 41 s after process start
```

A script that was valid before holds and axes (`<mask>@<s>` items only) keeps its meaning: the
same buttons at the same times on the first-poll base.

Up to 256 timed entries and 4,095 characters are accepted, as before. A longer value is ignored
as a whole and logged as `x2w-input X2W_INPUT_SCRIPT ignored: <n> characters, at most 4095`. When
any item is rejected, the stand-in logs
`x2w-input X2W_INPUT_SCRIPT: <used> of <items> entries used (<malformed> malformed, at most 256)`.
When it is created it also logs
`x2w-input script: <n> entries accepted, <m> rejected, time base first-poll|process-start` and, for
each of the first 32 rejected items,
`x2w-input script item <index> rejected (<reason>): "<item>"` (index 1 is the first non-empty item).
Reasons: `missing @<seconds>`, `bad time`, `bad duration`, `trailing text`, `bad mask`, `bad value`,
`value out of range [-1, 1]` or `[0, 1]`, `unknown token`, `base= must be the first item`,
`unknown base (use poll or start)` and `over 256 entries`. At the first poll it logs
`x2w-input script time base <base>: first poll at <s> s, first entry at <s> s since process start`,
and once the script time passes the end of the last entry
`x2w-input script time passed its last entry (<n> entries, ended at <s> s on the script time base)`.

The parser and playback are in `src/startup_stubs/input_script.h`. The check reads the stand-in
through `IGameInput::GetCurrentReading` with the script clock set by `X2W_INPUT_CLOCK_MS`
(milliseconds since process start; a test seam, not for runs):

```powershell
ctest --preset dev --no-tests=error -R '^startup_gameinput_script_(long|entries|syntax)$' -V
```

It exits 77 (skip) after the parser checks when XInput pad 0 is connected, because that pad's
values merge into every reading.

`IGameInput::GetTemporalReading` uses the v0 ABI and returns `E_NOTIMPL` as
[documented by Microsoft](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/deprecated/interfaces/igameinput/methods/igameinput_gettemporalreading). The
`startup_gameinput_contracts_check` fixture covers the refusal.

### Waiting for the presented picture

Fixed times drift when a run gets slower or faster. A script can instead wait until the presented
picture changes, and time the items after the wait from the moment it completes:

| Item | Effect |
|---|---|
| `wait[=<settle>[/<timeout>[/stop\|/go]]][@<s>]` | takes the presented picture at the moment the wait begins, waits until it changes and then until it has not changed for `<settle>` seconds (default 1). After `<timeout>` seconds (default 120) it logs a timeout and the script goes on (`go`, the default) or stops (`stop`). Empty fields keep their default: `wait=/30`. |
| `mark[=<label>][@<s>]` | logs the script time and the present number, for picking frames afterwards. The label is up to 32 letters, digits, `_`, `-` or `.`; the default is `mark<item index>`. |
| `@+<s>` | in place of `@<s>` on any timed item: `<s>` seconds after the previous timed item started (after a wait: after it completed). Without an earlier timed item it counts from 0. |

`wait` and `mark` without `@` mean `@+0`, so a wait begins when the previous item starts, before
the picture can react to that press. Items after a wait are shifted by the time the wait blocked and
never start before it completes. An absolute `@<s>` after a wait must not be earlier than the wait;
`@+` is the form to use there. A hold that started before a wait runs its length in real time.
A wait cannot take a `+<duration>`.

Wait for a new screen, pause 2 s, press down, pause 0.5 s, press accept, and repeat (about 5 s
per screen when each screen takes about 2.5 s to appear and settle):

```text
wait,80@+2,4@+0.5,mark=menu1@+0,wait,80@+2,4@+0.5,mark=menu2@+0
```

The picture signal comes from the D3D12 stand-in in render mode (`X2W_D3D12_RENDER=1`,
`run.ps1 -Render`). Sampling is opt-in: it starts only after a script with at least one `wait` has
asked for it (the GameInput stand-in sets a request field in the signature block and logs
`x2w-input script: picture signature requested`), so ordinary render runs record no extra GPU work.
`X2W_PRESENT_SIGNATURE_EVERY=<n>` samples every `n` presents from the start without a request, and
`0` turns the signature off. While sampling, every fourth present (or every `n`) it
copies 18 evenly spaced single rows of the presented surface into a small readback buffer on the
presenting queue, never the whole surface and never waiting for the copy, and on a later present
reduces them to 32 x 18 cells of mean luma. Command lists do not hold the resources they use, so the
sampler holds its own reference to the presented surface until the copy's fence has passed; a title
may release the surface right after the present. If `Signal` fails after the copy was submitted,
sampling stops and the surface, command list and readback buffer stay referenced for the rest of the
process. It publishes them in a named block,
`Local\x2w_present_signature_<process id>`, so other tools can read it too
(`src/startup_stubs/present_signature.h` describes the layout). The 8-bit and 10:10:10:2 colour
formats are read; for others it logs
`x2w-present picture signature: surface format <n> is not sampled` once. Two pictures differ when
at least 16 cells differ by more than 12 levels, so a blinking cursor does not count as a new screen.
Without a presenter every wait times out and says so.

Log lines, each once per item:

```text
x2w-input script: <n> waits, <m> marks; picture signal from Local\x2w_present_signature_<pid>
x2w-input script: picture signature requested
x2w-present picture signature: Local\x2w_present_signature_<pid>, every <k> presents once a script wait asks for it|from the start (X2W_PRESENT_SIGNATURE_EVERY)
x2w-input script wait item <i> begun at <s> s[ (no presenter signal yet)]
x2w-input script wait item <i>: picture changed at <s> s, <e> s after the wait began
x2w-input script wait item <i> done at <s> s after <e> s
x2w-input script wait item <i> timed out at <s> s after <e> s[ (no presenter signal)]; the script goes on|stops
x2w-input script mark <label> at <s> s (item <i>, present <n>|no presenter signal)
x2w-present picture signature: first sample published at present <n> in Local\x2w_present_signature_<pid> (every <k> presents)
x2w-present picture signature: Signal failed after a copy was submitted; sampling stopped, its surface stays referenced
```

Times are seconds on the script's time base. The rejection reasons of waits and marks are `bad wait option`,
`bad label`, `duration not allowed on wait or mark` and `time before the preceding wait`. A script
without `wait`, `mark` or `@+` plays exactly as before. One exception: `@+<s>` used to be read as
the absolute time `<s>`.

#### Change sensitivity, regions and the fine grid

Options after `;` on a wait (`wait=<settle>/<timeout>;<option>;...`, or `wait;<option>` with no
settle or timeout) set what counts as a change for that wait only:

| Option | Effect |
|---|---|
| `cells=<n>` | at least `n` cells must differ (default 16) |
| `level=<n>` | a cell differs when its mean luma moves by more than `n` levels of 255 (default 12) |
| `region=<x0>:<y0>:<x1>:<y1>` | watch only the cells whose centre lies in this rectangle, in fractions of the picture width and height (0:0 is the top left; default the whole picture) |
| `fine` | compare the fine 64 x 72 grid instead of the 32 x 18 one |

The default grid samples one row in the middle of each eighteenth of the picture, so anything less
than about 5.6 % of the picture height can fall between two sampled rows. A list highlight is often
that thin. The WARP check moves a 60 x 4 texel bar (30 % x 3.3 %, 1 % of a 200 x 120 picture) between
two sampled rows: the default grid sees 0 changed cells, the fine grid 40. The fine grid samples 72
rows (one every 1.4 % of the height) and 64 columns. The presenter copies its rows only once a script
with a `fine` wait has asked for them: the GameInput stand-in sets a request field in the signature
block and logs `x2w-input script: fine picture signature (64 x 72) requested`. Measured CPU-side cost
of one sampled present at 3840 x 2160 on WARP (release build, every fourth present by default):
0.16 ms with the default grid (18 rows, 270 KiB copied) and 0.56 ms with the fine grid (90 rows,
1,350 KiB), against 32,400 KiB for the whole frame; the WARP check prints both.

Rejected options log `bad wait option`, `bad region`, `empty region` (no cell centre inside) or
`cells larger than the region`.

#### Groups

`group[=<label>]` and `end` bracket a run of items. When a wait inside the group times out, the
rest of the group is skipped, and the script goes on after `end`, timed from the timeout: the script
time jumps to that of `end`, so an item `@+<s>` after `end` starts `<s>` seconds after the timeout.
Later groups run normally. A `/stop` wait inside a group still stops the whole script. Groups do not
nest. The default label is `group<item index>`. The timeout line then ends in
`; group <label> is skipped from here`, followed by

```text
x2w-input script group <label> (items <group>-<end>) failed at step item <i>: rest of the group skipped, the script goes on after item <end> from <s> s
```

Rejected group items log `nested group` (and its `end` logs `end of a rejected nested group`),
`end without group`, `group not closed` (its items then run ungrouped), `group and end take no time`,
`bad label`, and `time before the end of the preceding group` for an absolute time after `end` that
is earlier than the group's last timed item.

A two-entry list on the left of the screen, where each press must visibly move the highlight: wait
for the screen; press down and check that the highlight moved; press up and check that it moved back;
then accept. If a press shows no change within 2 s, the rest of the list steps is skipped and the
script waits for the next screen:

```text
wait,group=list,80@+1,wait=0.5/2;fine;cells=4;level=8;region=0.05:0.3:0.45:0.6,40@+0.5,wait=0.5/2;fine;cells=4;level=8;region=0.05:0.3:0.45:0.6,4@+0.5,end,wait=1/30
```

The timeline is in `src/startup_stubs/input_script.h` and the sampler in
`src/startup_stubs/present_signature_sampler.h`. The timeline cases run with a controllable clock
and picture signal; two cases run through the stand-in with the check process publishing the
pictures. The WARP check presents two different pictures and repeated identical ones, and compares
the published cells with a CPU reduction of the same rows. The held-source check runs under the
D3D12 debug layer: it stalls the queue on a fence, lets the sampler record its copy, releases the
caller's only reference to the surface, then releases the queue, and requires the correct signature
and no debug-layer error; it also makes `Signal` fail after submission and requires the sampler to
keep the surface. Its control repeats the stall with a copy that does not hold its source and passes
only when the debug layer reports the released surface:

```powershell
ctest --preset dev --no-tests=error -R '^startup_(gameinput_script_syntax|present_signature_(warp|held_source_warp|unheld_source_control_warp))$' -V
```

## Game saves

With `X2W_FAKE_USER=1`, the game-save stand-in keeps containers and blobs below
`X2W_STORAGE_PATH`, in `x2w-gamesave`: one directory per container, one file per blob
([#1200](https://github.com/R0Y00/Xbox2Windows/issues/1200)). Opening a container whose directory
exists succeeds. Names must be one plain path component of at most 128 bytes (no separators, `:`,
`..`, control characters, device names or the reserved suffixes `.x2w-part` and `.x2w-old`, in any
case) and a blob is at most 64 MiB. The storage root itself is trusted; a container directory or blob
that is a symbolic link or, on Windows, a junction is refused by every operation.

`X2W_STORAGE_PATH` is the root of the console drive, not the save folder. The title appends
`x2w-gamesave` to it, so `S:\` stores the files under `S:\x2w-gamesave` while
`S:\x2w-gamesave` stores them under `S:\x2w-gamesave\x2w-gamesave`. Point it at the drive root
(or a folder mounted as a drive) and keep one path per profile set.

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
Asynchronous read (21, with its result in 22) returns committed blobs.

For provider `704c3f58-e629-4cc2-b197-30511b996fe2` and interface `1bfff3af-f14a-40a3-8e35-9ada906593f9`,
container lookup (13), container enumeration (14), blob enumeration (18) and blob enumeration by
name prefix (19) call back with what the store holds. Each container's blob count and size come from
one listing under its shared publication lock, and no store lock is held while a callback runs. A
container created through slot 16 but never submitted exists and is reported with 0 blobs. Names
match case-insensitively, as the store's directories and files do. The display name is the
container name. Synchronous read (20) reads the requested blobs under one shared lock (`read_blobs`);
for a count of 0 it lists and reads every blob under one lock and checks the needed size against the
capacity before it reads any payload. The capacity is checked against that one snapshot, and only
then is the layout of `save_blob_pack.h` (records, then names, NUL terminators and payloads) written
into the caller's buffer. The count is in records, the capacity in bytes. A closed container handle
is refused (E_INVALIDARG). A lookup of a name with no container directory keeps the stand-in's
empty result (S_OK, zero callbacks). Slots 12 and 15 still report nothing, and other interface identities keep
the empty results.
[PC GDK thunk mapping](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/xgameruntime/gdk_table.rs#L261-L270) ·
[enumeration and sizing sample](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xgamesave/functions/xgamesaveenumerateblobinfo) ·
[synchronous read contract](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xgamesave/functions/xgamesavereadblobdata)
`ctest --preset dev --no-tests=error -R '^startup_gamesave_read(_after_async_submit)?$'` runs two source-built processes
through the provider vtable: a writer commits two blobs under a generated name (through slot 27, or
through slots 28 and 29 on a Windows gaming runtime queue in `startup_gamesave_read_after_async_submit`,
which returns 77 when that runtime is absent), and a restarted reader discovers the container, lists
the blobs (all, and by a prefix given in another case) and reads them. It also checks early enumeration stop, an unknown interface identity, a
short buffer (also for a count of 0) and a missing blob, both leaving the destination untouched, a
lookup of an absent name that does not call back, a slot 18 callback that runs with no store lock
held, and slots 18, 19 and 20 refusing a closed container handle.
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
`ctest --preset dev --no-tests=error -R '^startup_gamesave_read_publish_'` runs the synchronous read
(slot 20, by name and for every blob) through the provider vtable against the same paused writer.
During the pause the read waits for the store's lock timeout and fails with its destination
untouched. A read still waiting when the writer goes on returns the whole new generation. Racing
cases stop the reader instead, through a hook the generated copy adds to `read_blobs` and
`list_blobs` (null unless the test sets it; production builds keep only the anchor comments):
between two blob reads of slot 20, and at a second store read inside one slot 20 or slot 14 call,
another process commits a whole new generation (a third blob for slot 14). The reader must return
the old generation whole: a reader that locks per blob or lists twice mixes the two. The `_unlocked`
copy must be caught with a mixed or incomplete generation.

Slot 28 first asks the Windows gaming runtime to accept the caller's async block. On rejection it returns before taking staged actions. Once accepted it commits the files and calls `XAsyncComplete` with the status for that block. Slot 29 retrieves the status with the matching operation identity. File publication is synchronous within slot 28. The caller's completion routine can run inline or through its selected completion queue. [XAsyncBegin](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasyncbegin) · [XAsyncComplete](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasynccomplete) · [PC GDK async interface slots](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/xgameruntime/gdk_table.rs#L57-L64)

`ctest --preset dev --no-tests=error -R '^startup_gamesave_submit_'` checks denied starts and retries, two overlapping blocks and completion that reads the file and slot 29 before slot 28 returns. The system variant uses the installed Windows gaming runtime with an immediate work/completion queue. It reports a capability skip when the runtime or queue is unavailable.

Slot 21 reserves the caller's async block on that same XAsync service, with its own operation identity (`local XGameSaveReadBlobData`). If the service rejects the block, slot 21 returns before it reads. Once the block is accepted, it snapshots the named blobs through `read_blobs` (one shared file lock for the set), packs them, and completes with the status and the number of bytes slot 22 needs. It does not use the user service, and no save lock is held across `XAsyncComplete`. The packed bytes are an array of 24-byte records, payload pointer at offset 16, then each name, a NUL and the payload. Slot 22 copies that into the caller's buffer and reports the count. A short buffer returns `0x80830007` and a missing blob returns `0x80830008`; neither writes the caller's count or bytes. A busy snapshot, and any other storage failure, uses the same status as submit (`0x80004005`); a bad name is `0x80070057`. [XGameSaveReadBlobDataResult](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xgamesave/functions/xgamesavereadblobdataresult)

`ctest --preset dev --no-tests=error -R '^startup_gamesave_read_async_'` commits two blobs in one process and reads them in a second through slots 21 and 22. It checks the completion callback, the required size, the count and the exact bytes, a missing blob, a short buffer that leaves the output unchanged, and two overlapping reads. The controlled fixture always runs. The system variant uses an immediate Windows gaming runtime queue and skips only when that runtime is unavailable. `ctest --preset dev --no-tests=error -R '^startup_save_blob_pack'` checks `src/startup_stubs/save_blob_pack.h` directly without a runtime or disk store, including layout, `uint32_t` payload-size and `size_t` overflow refusals, and a negative control for the earlier size check.

### Completion on a manual queue

A completed XAsync operation can sit unconsumed when its block names a queue whose completion port is manual and nobody dispatches that port. `tests/startup_async_manual_completion.cpp` separates this from an operation that is still pending. It starts one synthetic operation on a queue whose work and completion ports are both manual. The provider completes it with an 8-byte payload on another thread. The test then prints four observations for each phase: the Begin return, the non-waiting `XAsyncGetStatus`, `XAsyncGetResultSize` and the callback count with what the callback retrieved. The phases are before completion (`pending`), after the provider thread completes the block, with no work-port callback and no completion-port dispatch (`unpumped`), and after one completion-port dispatch (`dispatched`). The callback count must be 0 until the completion port is dispatched. One dispatch must then run exactly one callback that retrieves the payload, and a second dispatch must find no work. A second block on an immediate queue is the contrast: its callback runs at completion with no dispatch. The `system` variant uses the installed Windows gaming runtime's XAsync service (`073b7dcb`) and prints its status and result-size answers. It accepts `E_PENDING` or `S_OK` before completion-port dispatch and requires `S_OK` after it. It reports a capability skip (77) when the runtime or its XAsync service is unavailable. Queue creation errors after obtaining the service fail the test. One measurement used Gaming Services 39.118.30001.0 (`xgameruntime.dll` 10.0.26100.9656) on Windows 10 build 19045. Before completion, both answers were `E_PENDING` (`0x8000000a`). Once the operation was complete but undispatched, the status was `S_OK`, the result size was 8 bytes and no callback had run. After the dispatched callback retrieved the result, the status stayed `S_OK` and the result size was 0. So a non-waiting status call separates an operation that is still pending from one whose completion has not been dispatched. The `controlled` variant uses a stand-in service in the test. In `inline_control` that stand-in runs the callback inside `XAsyncComplete`, as an immediate port would, and the control must print `FAIL: callback ran before the completion port was dispatched`. [XAsyncGetStatus](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasync/functions/xasyncgetstatus) · [XTaskQueueDispatch](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xtaskqueue/functions/xtaskqueuedispatch)

```powershell
ctest --preset dev --no-tests=error -R '^startup_async_manual_completion_(controlled|system|inline_control)$' -V
```

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
  already lowers the same forms. The VOP2 select `v_cndmask_b32_sdwa` lowers the same way, with VCC as
  its mask (`startup_gfx2hlsl_sdwa_cndmask`, drawn on WARP and hardware). Most float SDWA forms and a
  `dst_unused:UNUSED_SEXT` destination still
  stop translation rather than translating to something inexact. [#1141](https://github.com/R0Y00/Xbox2Windows/issues/1141)
  The one float exception is the binary16 arithmetic set `v_add_f16`, `v_sub_f16`, `v_subrev_f16` and
  `v_mul_f16` (`_sdwa`). `src0_sel`/`src1_sel` accept `WORD_0`, `WORD_1` or `DWORD` (`DWORD` reads the
  low half, as the 16-bit operation does); sources are single VGPRs, with `abs` and `neg`. The result
  is rounded once to binary16 and written to `dst_sel:WORD_0` or `WORD_1`, with `UNUSED_PAD` zeroing the
  other half (the default) and `UNUSED_PRESERVE` keeping the old one. Refused: `dst_sel:DWORD` and byte
  selects, `sext()`, `clamp`, output modifiers, `UNUSED_SEXT`, and non-VGPR sources. Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_f16_alu"`: the check covers the accepted and
  refused forms, and the draws on WARP and hardware read back every selector and unused mode against a
  CPU binary16 reference.
  The VOPC SDWA compares (`v_cmp_*_sdwa`, `v_cmpx_*_sdwa`) of the existing 32-bit compare table lower
  too: i32 and u32 sources take their `src0_sel`/`src1_sel` field, sign-extended under `sext()`, and
  the compare reads the extended value as its type. f32 SDWA compares stay refused (#1920). The
  result goes to `vcc_lo`, the printed SGPR, or EXEC for `v_cmpx`, as in the non-SDWA forms. Class
  compares, f32 compares and any other SDWA modifier are refused. Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_vopc_sdwa"`: the controls evaluate every
  integer form against an integer reference, and the draws on WARP and hardware check 16 x 16 pixels
  that a `v_cmpx_le_i32_sdwa` on a sign-extended byte keeps or discards, with translations that drop
  the select or `sext()` as failing controls. [#1906](https://github.com/R0Y00/Xbox2Windows/issues/1906)
- **DPP quad pair broadcasts.** The four pair permutations of a pixel stage (`quad_perm:[1,1,3,3]`, `[0,0,2,2]`, `[2,3,2,3]`, `[0,1,0,1]`, with
  `row_mask:0xf bank_mask:0xf`, and `bound_ctrl` since no source lane is disabled) lower on any VOP1 or VOP2 whose
  first source is a VGPR. The four lanes of the quad are read with the shader model 6.0 `QuadReadLaneAt` and the lane `quad_perm` names is selected by the parity of `SV_Position`; a stage that uses it compiles with DXC `ps_6_0` (`X2W_DXC`), and the result goes through a scratch register so a second source naming the
  same register keeps the pixel's own word. The form needs the four lanes of a quad to run together, as in a pixel
  shader with helper lanes, proved by `s_wqm_b64 exec, exec` on the entry mask (a restored exact mask loses the proof): vertex stages and a changed EXEC are refused (branches and loops: see below), as are `row_*`, `wave_*`, partial masks and `fi`. Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_dpp_pair"`: the controls model a quad on the CPU, and the
  draws on WARP and hardware compare 16 x 16 pixels of every pattern with a reference written from the lane table,
  with wrong-lane, own-lane, transposed-lane-index and permuted-second-source translations as failing controls.
- **DPP quad broadcasts.** `quad_perm:[k,k,k,k]` for k = 0, 1, 2 or 3, with full `row_mask` and
  `bank_mask` and `bound_ctrl:1` or none, lowers on `v_mov_b32` and the VOP1/VOP2 float, integer and
  integer-to-float tables as one read of lane k of the quad, `QuadReadLaneAt`. The four lanes of a pixel
  quad are one 2 x 2 pixel block (lane = x & 1 | (y & 1) << 1). The lowering applies to straight-line pixel
  stages that ran `s_wqm exec, exec` with EXEC unchanged since, so no lane of the quad is off; without that proof the broadcast is refused. `QuadReadLaneAt` is shader model
  6.0, so such a pixel stage compiles as `ps_6_0` with DXC (set `X2W_DXC`, or put `dxc.exe` in the Windows
  Kits `bin` folder or on `PATH`), and its vertex stage as `vs_6_0` too, because D3D12 rejects a pipeline
  state that mixes DXBC and DXIL; every other pixel stage stays `ps_5_1` with FXC. For single-lane broadcasts, row shifts, mirrors, DPP8, partial masks, `fi`, `bound_ctrl:0`, non-VGPR sources and vertex stages stay
  refused. Run `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_dpp_quad_broadcast"`: the controls
  check the translation and every refusal, and the draws on WARP and hardware compare all 256 pixels of two
  fixtures with a CPU reference, with wrong-lane and own-lane translations as failing controls.
  [#2125](https://github.com/R0Y00/Xbox2Windows/issues/2125)
- **DPP quad reads in branches and loops.** The quad swaps, broadcasts and pair broadcasts above also lower in a pixel
  program with branches or loops when a whole-program check proves that the four threads of the quad reach the read
  together: the `s_wqm exec, exec` runs before every branch and target, no other EXEC write can reach the read (an
  exact-mask restore after the last read is fine), and every conditional branch that can reach it is wave-uniform
  (SCC proven uniform; `s_cbranch_execz`/`execnz` under the whole-quad mask). A `vccz`/`vccnz` loop, SCC taken from a
  lane mask, an EXEC write inside the loop and a late `s_wqm` are refused with the offset that breaks the proof. No
  wave size is assumed. Run `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_dpp_quad_flow"`: the draws on
  WARP and hardware run swap, broadcast and pair reads inside a five-pass scalar loop with an `if`, compare 16 x 16
  pixels with a CPU reference, and require wrong-lane translations to fail.
- **wave32 `vcc_hi`.** In wave32 only `vcc_lo` is the condition mask and compilers allocate `vcc_hi` like any other
  scalar register. The decoder prints wave32 syntax, so the translator renames `vcc_hi` to an SGPR the program never
  names only when that is exact in both wave sizes: no instruction names the 64-bit pair `vcc`, and either a VALU
  compare or carry-out writes `vcc_hi` (or an odd SGPR), a VOP3 mask source reads one (a misaligned pair in wave64),
  or no VALU instruction and no `vccz`/`vccnz` branch uses `vcc_lo` as a lane mask (in wave64 those see `vcc_hi` for lanes 32..63). Otherwise `vcc_hi` stays refused; scalar writes into `vcc_hi` or
  `exec_hi` are now refusals instead of a Python TypeError. Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_vcc_hi"`: the draws on WARP and hardware compare 16 x 16
  pixels of a compare, a carry and a scalar use of `vcc_hi` with a CPU reference, with translations that drop the
  compare bit, the carry or read `vcc_hi` as zero as failing controls.
- **Wave reductions.** A whole wave32 reduction chain in a straight-line pixel stage (`s_or_saveexec_b32 sS, -1`,
  the identity fill `v_cndmask_b32_e64 vT, <identity>, vX, sS`, four `_dpp` steps `quad_perm:[1,0,3,2]`,
  `quad_perm:[2,3,0,1]`, `row_half_mirror`, `row_mirror`, `v_permlanex16_b32 vP, vT, 0, 0`, the combine,
  `s_mov_b32 exec_lo, sS` and `v_readlane_b32 sD, vT, 31`) of `v_or_b32`, `v_min_u32` or `v_max_u32` is recognised
  as one `WaveActiveBitOr`, `WaveActiveMin` or `WaveActiveMax` over the active lanes of the desktop wave (a correct
  program cannot depend on how pixels are packed into waves). The min chain may read its lane later, as the #2125
  censuses show: after its restore, one `v_cndmask_b32_sdwa` (lowered as usual), the saveexec of a max chain, then the
  readlane, carried by that max chain before its fill. The entry EXEC must still hold, without `s_wqm`, so helper
  lanes are left out on both sides. Other shapes that start like a chain are refused with their reason
  ([the note](experiments/startup-gfx2hlsl-wave-reduce.md)). Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_wave_reduce"`.
  [#2125](https://github.com/R0Y00/Xbox2Windows/issues/2125)
- **First active lane.** In a straight-line pixel stage whose EXEC is still the entry mask, without `s_wqm`,
  `s_ff1_i32_b32 sD, exec_lo` followed by `v_readlane_b32 sX, vY, sD` becomes `WaveReadLaneFirst(vY)` over the active
  lanes of the desktop wave. This follows the packing contract of the wave reductions. `sD` has no run-time value,
  and any other read of it is refused ([the note](experiments/startup-gfx2hlsl-pixel-first-lane.md)). Run
  `ctest --preset dev --no-tests=error -R "^startup_gfx2hlsl_pixel_first_lane"`.
  [#2618](https://github.com/R0Y00/Xbox2Windows/issues/2618)
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
- **Upload arena lifetime.** A queue waits on a fence while the fixture submits a copy of
  payload A (640 bytes, 16-byte alignment), resets the arena, writes a distinct payload B into
  a new page and submits a second copy. It opens the gate only after both submissions and
  compares the exact A and B bytes in readbacks. The reset releases the arena's own page
  reference, so the fixture holds A's page until the queue drains; this stands in for the
  retirement a production caller owes. An overwrite control writes B into A's live range
  before the gate opens and must read B back from A's copy. The fixture does not cover
  refusal after a normal Close or any command-buffer interaction. Run
  `ctest --preset release --no-tests=error -R '^startup_upload_lifetime_' -V`.
- **Slot 148 acceptance client.** This client tests the upload allocation that the title's
  acceleration-structure builder requests (#1758). It drives the stand-in's actual list and
  queue, uses whichever slot 148 the build installs, and keeps no upload page of its own.
  - **Request:** the traced shape, `(list, 0x280, 0x10)`, returning a pointer.
  - **Resolver:** the module that installs the slot advertises it by exporting
    `x2w_d3d12_upload_gpu_address(list, cpu, bytes)`. This is a fixture-only adapter to the
    production address resolver, outside the guest ABI, and it holds no page reference for the
    client. It returns the desktop GPU address only for a complete, non-empty range inside one
    allocation of that list's open recording. A closed list, a range from before the list's last
    Reset, or a range of another list resolves to 0.
  - **Reads:** each submitted list records its own GPU read of its allocation through the guest
    consumer. Slots 25 and 33 bind a guest compute pipeline and root signature, and slot 43 binds
    the upload pointer as a root SRV. Before slot 43 runs, a sentinel buffer is bound, so a
    refused binding reads the sentinel. Separately, an observer queue reads the resolved GPU
    ranges while the submissions are held.
  - **Held submissions:** A is submitted behind a gate fence. The list is Reset with another
    allocator, and B is written and submitted. Each range is filled with a stale pattern before
    its read is recorded and with its payload after, so only a read at execution sees the
    payload. After the gate opens, both reads of A and B must match byte for byte. The
    payloads are non-uniform: they mix the byte index with the lifetime fixture's A and B
    values. A read off by 4 or 16 bytes therefore differs at every byte.
  - **Other cases:**
    - a Reset on A's own allocator after that allocator's Reset;
    - the overwrite control;
    - an early-reuse control on an unsubmitted list;
    - wrong-list and stale-generation controls for slot 43 and the resolver;
    - refusal after a successful Close;
    - invalid arguments, including zero alignment.
  - **Failures:** a missing slot (no resolver export and a null result) prints
    `slot 148 not implemented (missing slot)` and exits 10. A null for the valid request from an
    advertised slot exits 21, on the first call too. Both are failures. Only a missing physical
    adapter skips (77).
  - **Serialized queues:** the held-phase observer read needs a second queue to progress while
    the console queue waits on an unsignalled fence.
    - **Preflight:** before the steps, the client checks this with a 1-second read of the
      sentinel. If the read times out but later drains and matches after the gate opens, it prints
      `queues serialized: held-phase checks not applicable` and skips the three held-phase
      observer reads. A failed setup, wait or readback keeps the test failing.
    - **Still checked:** the in-list reads after the gate opens, the readback staying zero while
      held, the controls, refusal after Close and the arguments.
    - **Result:** the run passes with a final line that starts
      `slot 148 acceptance passed with skips`. An adapter that passes the preflight and then stalls
      still fails with exit 18.
    - **Cleanup:** every exit opens the gate and drains both queues, with each wait bounded.
    - **Forced variants:** `startup_slot148_acceptance_serialized_*` set
      `X2W_TEST_FORCE_SERIALIZED_QUEUES=1` to take this path on adapters whose queues do progress.
  - **Run:** `cmake --build --preset release --target startup_slot148_acceptance` and
    `ctest --preset release --no-tests=error -R '^startup_slot148_acceptance_' -V`.
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
- **Pixel-stage image stores.** In a pixel stage, `image_store` with `dim:SQ_RSRC_IMG_2D` and
  `dmask:0xf` becomes a typed store to a `RWTexture2D<float4>` at its own `u` register, under the
  lane's exec bit, and the manifest becomes version 2 with a
  `uav <index> tex2d table <table> <offset>` line. 2D arrays use `tex2da`. The cache-policy
  bits are accepted. A partial channel mask reads the original texel to preserve the other channels.
  A pixel stage can also load mip zero with plain `image_load` and store through the same fixed eight-word
  table descriptor. The translator retries with one read-write binding for that image. Both operations
  use the lane's exec bit. Writes by other pixels in the draw are unordered.
  Shared views require matching dimensions and full-width load addresses and data. Explicit mip,
  integer-view, filtered, gather and LOD-query loads cannot share a stored view. A
  hand-assembled program stores four distinct values at texel (1, 2). The desktop draw reads all 16
  texels of a 4x4 image back; deleting the store or exchanging x and y must fail. Run
  `ctest --preset release --no-tests=error -R '^startup_pixel_image_store_' -V`.
- **Binding pixel-stage image stores.** A translated draw binds each `uav N tex2d table P O` line
  as a UAV range `u0..` placed after the SRVs in the translated root signature's view table. The view is made from the unordered-access
  view the title created at that table entry (`resolve_pixel_uavs` in `d3d12_draw.inl`), on
  mip 0 of a 2D texture or the slices named by a 2D-array view. The texture is in `UNORDERED_ACCESS` for the draw. Refusals are logged
  with their reason and nothing is bound in place of the view. When the manifest is loaded, the
  translation is not used for another image kind, a buffer line (`uav N raw table P O`,
  the vertex-stage store of #2059), a view qualifier (`float`, `uint`, `sint`, as in the compute line), another source, extra fields such as
  `channels`, an index out of order or more than eight views. At draw time,
  the draw is not translated for no table at the parameter, an entry outside the bound table, an
  entry without a view, a view that is not a UAV, a resource without `ALLOW_UNORDERED_ACCESS`, an
  array for a 2D view, 3D, 1D or multisampled resource, a view of another dimension, mip or plane, an integer,
  sRGB or non-colour format, a format without typed UAV store (or without typed UAV load when the manifest
  has a `uavload N` line for that view; `tests/startup_pixel_uav_support_check.cpp`), and a texture that is also an SRV,
  a render target or the depth texture of the draw; these checks run before any view of the draw is
  written or any texture changes state. Indirect draws refuse a translation with a `uav` line. A resource-specific
  UAV barrier before a draw with `uavload N` orders earlier writes to that resource. `tests/startup_pixel_uav_check.py`
  draws the program above through the runtime with four 4x4 two-mip textures in which every word is
  a distinct sentinel. The texel lands only in the texture the table names, and every other word of
  the four textures stays unchanged; each refusal leaves every word unchanged. The exec-off variant
  leaves the texel and fails once recompiled without its guard. The WARP suite also draws twice into a patterned
  8×8 2D image and an array slice. It checks every texture word against a CPU reference, requires a stale
  SRV copy to miss the second addition, and masks typed-load support to verify the runtime refusal. Run
  `ctest --preset release --no-tests=error -R '^startup_pixel_uav_' -V`.
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
- **Packed 2D sample coordinates.** Plain 2D `image_sample` and `image_sample_lz` with `a16` unpack two
  IEEE binary16 coordinates from one VGPR (`float2(f16tof32(p), f16tof32(p >> 16))`) without an NSA
  extension, and both the offline and runtime cache keys advance to `x2w-startup-shader-6` so pre-fix
  two-VGPR translations are not reused. Non-1D `image_sample` and `image_sample_lz` accept cache-policy
  flags (`glc`, `slc`, `dlc`) and refuse `unorm`, `r128`, `tfe`, `lwe`, `d16`, reserved bits, an empty
  `dmask`, wrong address-register counts, fake NSA extensions, and `a16` on 3D or 2D-array images
  (`ctest --preset dev --no-tests=error -R "^startup_image_sample_a16_" -V`). `image_sample_l` and 3D /
  2D-array `a16` stay refused.
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
  words still go through `float4` and have the same loss. Only the `p0` selector is lowered;
  `p10` and `p20` are refused in both encodings
  ([#1890](https://github.com/R0Y00/Xbox2Windows/issues/1890)). Reproduce with
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
- **DirectStorage request totals (#2816).** The heartbeat's totals line and the first `Queue::EnqueueStatus` lines print
  `requests raw=<n> inflated=<n> failed=<n> bcpack-blank=<n> completed=<n>`. `raw` counts the uncompressed copies only
  (it was printed as `done=`, which read as zero for a title whose requests are all compressed), `inflated` every
  compressed stream that was decoded (zlib, standalone or embedded BCPack), `bcpack-blank` the BCPack requests that
  completed without decoded data, and `completed` is `raw + inflated + bcpack-blank`: every request that finished and
  wrote its destination (a failed request is not in it). The text `inflated=<n> failed=<n>` is unchanged;
  `tools/startup/live_view.py` parses it, takes `completed` from new lines and rebuilds it from `done`, `inflated` and
  `bcpack-blank` on older ones. `startup_storage_failure_counts` checks the totals of a mixed run and of a run with only zlib
  requests (`raw=0`, `completed` equal to the finished requests); `startup_live_telemetry_check.py` checks both line formats.
  [Microsoft DirectStorage overview](https://learn.microsoft.com/en-us/xbox/gdk/docs/features/console/storage/directstorage/directstorage-overview)
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
- **DirectStorage loading progress and disk lines.** Every heartbeat (10 s), changed or not, the stand-in prints
  `x2w-dstorage progress <n> s: pending <p> (oldest <ms> ms), read <x> MB/s, inflated <y> MB/s, enqueued <e> retired <r> (written <w>, failed <f>), total read <KiB> inflated <KiB>`
  (pending = enqueued minus retired; retired = written + failed, "written" = requests that really wrote their destination; the oldest age comes from a ring of 4096 enqueue ticks and is approximate with several queues)
  and `x2w-dstorage disk <n> s: <r> reads (+<d>), <x> MB/s while reading, reading <p>% of the window; read sizes <=4K <=16K <=64K <=256K <=1M >1M;
  offsets sequential/forward/backward/first; read failures`. A high share of the window in reads with low MB/s is a seek-bound disk (a
  hard disk); a low share means the time goes to inflating or to waiting for the title. `startup_storage_failure_counts` checks the
  counters, the read classification and the exact lines.
- **DirectStorage HDD mode (`X2W_STORAGE_HDD=1`, off by default).** The queue's worker plans the file reads of its next 1024 entries
  (`dstorage_read_plan.h`: sorted by file and offset, reads within 64 KiB merged up to 4 MiB); one prefetch thread reads those groups into a
  cache of at most 128 MiB (`dstorage_read_cache.h`, unused entries dropped after 20 s) and the worker's own file read takes the bytes from
  the cache first. Entries still run in order, so a cache miss only costs speed. A third line, `x2w-dstorage hdd <n> s: cache hits ... misses ...;
  prefetch groups ..., disk reads saved by merging ...`, shows what it did. `startup_dstorage_read_plan` tests the plan;
  `startup_dstorage_hdd_queue_on` / `_off` run 48 shuffled requests through the real queue and compare every byte (on: 47 cache hits, a few
  merged reads; off: 48 reads).
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
  stand-in must be loaded for render-mode copy-in. `CopyFromSubresource` with the Scarlett
  option set to zero reads visible blocks back into pitched rows, preserving row padding.
  This CPU readback also works with rendering enabled and the XG stand-in alone.
  The reported pitch describes the padded block-row extent of the surface.
  Synthetic copy and draw checks run with
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
  Table entries take their LOD range from word 1, as `smpl` static samplers do: `MIN_LOD` (bits 11:0) and
  `MAX_LOD` (bits 23:12), unsigned 4.8 fixed point. An entry with word 1 = 0 samples mip 0 only; the border
  fallback is not a title descriptor and keeps the full mip range. A `MIN_LOD` above `MAX_LOD` is passed
  through unchanged; D3D12 documents `MaxLOD >= MinLOD`, and the console behaviour for that case is not
  established here. `startup_sampler_lod_{warp,hardware}` decode two table entries that differ only in
  word 1, both with `MinLOD <= MaxLOD`: entry 0 is `MIN_LOD` 2 with `MAX_LOD` 4095/256, entry 1 is
  `MIN_LOD` 0 with `MAX_LOD` 1. They read a four-mip texture with one value per mip at explicit LODs, and a
  static `smpc` sampler must keep `LESS`. The `_dropped_` control resets the decoded range and must read
  mip 0 and mip 3 instead: `ctest --preset dev --no-tests=error -R '^startup_sampler_lod_' -V`.
- The startup shader translator lowers `image_sample_l` (explicit level, 2D and 3D, any `dmask`) to
  `SampleLevel(sampler, uv, level)`, taking the level from the last address register. Offset, compare,
  gather, clamp, cube, 1D and `unorm` forms are refused. Only the translation is checked
  (`python -B tests/startup_image_sample_l_check.py`); a WARP readback of distinct mip levels is still open.
- `image_sample_b` in a pixel stage lowers to `SampleBias(sampler, uv, bias)` for 2D, 2D-array and 3D textures; the bias is the
  first address register. Offset, clamp, compare and gather forms, other stages, code after a branch, a sampler with a
  LOD bias, mip filter 0 or MIN_LOD above MAX_LOD, and any sampler from the heap or a register are refused. The descriptor's
  MIN_LOD and MAX_LOD reach the static sampler through an `smpl` manifest line (MinLOD, MaxLOD). `ctest --preset dev --no-tests=error -R '^startup_image_sample_b'` draws bias -1, 0 and +1 on a
  three-mip array (WARP) and requires mip 0, 1 and 2 (`docs/experiments.md`, "biased image sampling").
- `image_sample`, `image_sample_l` and `image_sample_lz` through a constant sampler carry the descriptor's MIN_LOD and MAX_LOD
  through the same `smpl` line (#2177). A constant sampler with a LOD bias or MIN_LOD above MAX_LOD is now refused for these forms;
  a mip filter of 0 is not (the sampler stays an ordinary `smp` line). A sampler from the heap, including the default-sampler
  shortcut, still translates as before: its MIN_LOD, MAX_LOD and bias are not validated, and `gfx2hlsl.py` logs that once per
  translated module (#2221). With the sampler-heap opt-in, a table entry applies its MIN_LOD and MAX_LOD at run time (see the
  sampler-heap item above); its LOD bias is not carried, and a MIN_LOD above MAX_LOD is passed through unchanged. The
  default-sampler shortcut keeps a static sampler with an unbounded LOD range. `image_sample_c` is unchanged.
  `ctest --preset dev --no-tests=error -R '^startup_image_sample_lod'` draws each form on its own against a plain-`smp` control
  (`docs/experiments.md`, "constant-sampler LOD limits").
- Slot 86 graphics streams can submit VS/PS requests to the offline cache. The fixture sends
  complete shader containers and a wrapped root through WARP, runs `gfx2hlsl.py --pending`, and
  reloads the compiled stages. A second fixture draws a clipped triangle through the translated
  pipeline, submits the wrapped list, waits for its fence and checks all 32 RGBA pixels on WARP and
  hardware. Swapping the red/green shader changes the 16 covered pixels. Reproduce with
  `ctest --preset release --no-tests=error -R "^startup_stream_(request|draw)_"`.
  A stream `RASTERIZER` or `RASTERIZER1` subobject sets the translated pipeline's fill, cull and depth
  clip. `RASTERIZER1` stores depth bias as a float; a value that is not an integer in the desktop range
  refuses the pipeline. Run `ctest --preset release --no-tests=error -R "^startup_stream_rasterizer_"`
  to check all 32 pixels for front and back culling, with each front-face winding and both stream
  subobject types. On a multi-GPU system, set `XBOX2WINDOWS_ADAPTER` to a case-insensitive substring of
  the adapter description to select hardware for the startup stream, rasterizer and depth fixtures.
  An unmatched hardware selector fails the test. The WARP fixtures always select the software adapter.
  The ignore control expects the no-cull readback and fails. Stream depth-stencil state is already
  applied on this path. Geometry, hull, domain and compute stages remain open.
- Translated draws use the bound typed RTV formats to request colour-export variants. The
  manifest's `targets` line lists exported slots. A separate `rtv1` line identifies the base
  shader and the ordered RTV formats, and the draw rejects a variant with a mismatched line.
  Uncompressed `R32G32B32A32`, `R32G32` and `R32` UINT and SINT outputs use integer shader
  signatures (`startup_mrt_*` reads back every stored word on WARP and hardware). `R32G32_FLOAT` and
  `R32_FLOAT` use the same bit-preserving path as `R32G32B32A32_FLOAT`, covering a NaN payload and a
  negative denormal. Uncompressed exports to `R8G8B8A8_UINT` and `R8G8B8A8_SINT` use unsigned
  and signed integer shader outputs. The output merger clamps each channel to 0–255 or −128–127
  respectively. With export words `ffffffff ffffff7f 00000080 00000100`, the stored bytes are
  `ff ff 80 ff` for UINT and `ff 80 7f 7f` for SINT. The same conversion applies to one or two
  8-bit channels and one, two or four 16-bit channels. An uncompressed UINT export to
  `R10G10B10A2_UINT` clamps RGB to 0–1023 and alpha to 0–3 before storage. Three-channel
  `R32G32B32_UINT` and `R32G32B32_SINT` views still require readback verification.
  Uncompressed binary32 FLOAT outputs bitcast the export words after XOR with a private
  zero-valued entry constant so optimized FXC keeps denormal bit patterns. UNORM outputs still
  undergo float-to-UNORM conversion. Compressed gfx1013 integer exports remain unsupported,
  including when the destination is `R10G10B10A2_UINT`.
- A colour export that leaves channels `off` does not write them, so the target keeps what it held
  there (#1674). `gfx2hlsl.py` records the written channels per target and writes
  `wmask <target> 0x<mask>` to the manifest when an export leaves any channel unwritten. The draw
  path removes the other channels from that target's `RenderTargetWriteMask`
  (`src/startup_stubs/export_write_mask.h`). It expands a shared blend state when active targets
  need different masks. A shared logic operation with different masks is refused because D3D12
  requires logic operations to use a shared blend state. Startup cache revision 7 invalidates older
  zero-filled exports. A compressed export
  lowers when it writes whole binary16 pairs (channels 0–1, 2–3 or both). A compressed mask that is not
  a whole pair, an export of no channel, two exports of one target with different masks, a packed
  integer compressed export and a partial compressed export to a vertex output stay refused
  (`ctest --preset release --no-tests=error -R "^startup_(gfx2hlsl_)?export_(write_)?mask"`).
- Colour-blend registers reach the draw path for pipelines created through device slot 57 only
  (#1216): state block A holds the eight `CB_BLENDn_CONTROL` words at +0xD0, and
  `src/startup_stubs/state_block_blend.h` decodes them into the translated pipeline's blend
  state. `CB_TARGET_MASK` is not located, so all four channels are written unless the translated
  export leaves some unwritten (manifest `wmask`, above). A word the decoder refuses for any of the eight targets leaves the
  registers unused (a blending target other than 0 with a `SRC1_*` dual-source factor, or any
  blending target other than 0 alongside a dual-source target 0, is refused too, since only
  target 0 has a second shader output): the pipeline keeps the path it had before and the refusal
  is logged. No target
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
- Depth-stencil registers are decoded, and applied only with `X2W_DEPTH_STATE=1` (#1999, item A;
  see the next bullet). State block A of a
  slot 57 pipeline holds `DB_DEPTH_CONTROL` at +0xB8, `DB_STENCIL_CONTROL` at +0xBC and
  `DB_STENCILREFMASK` / `_BF` at +0xC0 / +0xC4, measured locally, when the word at +0x004 is 4
  (vertex pipelines). Mesh pipelines carry 6 there and a second layout, also measured locally: every
  field from +0x058 on is 0x54 bytes later (`DB_DEPTH_CONTROL` at +0x10C); the reader adds that
  shift. A block with another value at +0x004 is read at the vertex offsets. The blend reader
  (`state_block_blend.h`, +0xD0) does not take the shift yet. `src/startup_stubs/pipeline_state_decode.h`
  reads the four words and decodes them into a `D3D12_DEPTH_STENCIL_DESC` and a stencil reference
  with the field layouts of Mesa's `src/amd/registers/gfx10.json`. It refuses stencil ops D3D12
  can't express (`ONES`, `REPLACE_OP`, the logic ops, add or subtract by other than one) and colour
  writes tied to the depth result. It flags, without applying, `DEPTH_BOUNDS_ENABLE` and back-face
  masks or references that differ from the front. A device-free check decodes hand-built words for
  each compare function and each stencil op with a D3D12 form, the refusals and a synthetic block.
  Its `_offset_control` run (words one word later) is the negative control. A vertex block and a mesh
  block each hold a different decodable state at the other layout's offsets; `_mesh_shift_control`
  (the mesh words at the vertex offsets) is that case's negative control
  (`ctest --preset dev --no-tests=error -R "^startup_pipeline_state" -V`).
- `X2W_DEPTH_STATE=1` (off by default) takes that decoded state for translated draws of slot 57
  pipelines (`src/startup_stubs/depth_state_apply.h`, applied in `translation_for` in
  `d3d12_draw.inl`). A slot 57 pipeline has no desktop `DEPTH_STENCIL` subobject, so there is no
  desktop state to outrank; a state found first is kept regardless. When the decoder refuses the
  words, or the block is too short, the pipeline keeps the desktop default (depth and stencil off)
  and one line says why (`depth-stencil registers not used (...)`). The decoded state needs a depth
  view: a draw with none bound keeps the desktop default. The stencil reference is list state in
  D3D12, so each draw of such a pipeline with an enabled stencil test calls `OMSetStencilRef` with
  the decoded value for that draw and restores the guest's list reference immediately after it.
  A successful list reset restores the default reference of zero. The mixed-pipeline test sets a
  distinct guest reference before the slot 57 draw, then checks a desktop-state stencil draw on
  the same list without another reference call. The pipeline key includes the
  decoded description. With the switch off nothing reads the words: no description, key or
  reference changes, and no log line appears. That is checked device-free
  (`startup_depth_state`, with `startup_depth_state_switch_control` as the negative control) and by
  the draw fixture's switch-off run, and the existing `startup_stream_depth_*` and
  `startup_slot57_blend_draw_*` tests, which run without the switch, still pass.
  `tests/startup_slot57_depth.cpp` and `tests/startup_slot57_depth_check.py` draw two overlapping
  triangles of unequal depth (z 0.25 and 0.75, same synthetic shaders and vertex program) into an
  8x4 target with a D24S8 view, with the same renderer on WARP and on a hardware adapter. With
  reversed-Z words (GREATER, depth cleared to 0) the nearer triangle wins (red); with the switch off
  the last-drawn triangle wins (green); with LESS, the wrong direction for that clear, nothing
  passes (blue); a forward-Z LESS case reverses the draw order and shows green with the switch on,
  red with it off. Stencil is checked separately: EQUAL against a stored 0x5A passes with reference
  0x5A and fails with 0x5B (the list's own reference is 0, so only the decoded one explains it), and
  REPLACE 7 followed by EQUAL 7 or EQUAL 8 leaves green or red. Words with a colour-write tie or a
  stencil `ONES` op fall back to the desktop result. Measured on WARP (10.0.26100.9818) and an
  NVIDIA GeForce RTX 3080 (driver 32.0.16.2124), Windows 11 10.0.26200
  (`ctest --preset dev --no-tests=error -R "^startup_(depth_state|slot57_depth)" -V`). Not covered:
  raster bias, depth bounds, back-face-only state, the title's own `OMSetStencilRef` order, clears
  (#1806), and any pipeline layout other than the measured one.
- Rasterizer state (#1999, item B) has a decoder for the register value only:
  `src/startup_stubs/rasterizer_state_decode.h` turns a `PA_SU_SC_MODE_CNTL` value into the cull mode,
  front-face winding and fill mode of a `D3D12_RASTERIZER_DESC`, with the field positions of the amdgpu
  `gc_10_3_0_sh_mask.h` header and Mesa's `gfx10.json`. Where a slot 57 pipeline keeps that word is not
  part of it and is left open until independently established. It refuses both faces culled, point or
  mixed polygon modes, polygon offset, last-vertex provoking and disabled perspective correction. The
  caller's `flip_winding` inverts the FACE convention, which is not established against the D3D12
  viewport. A device-free check covers each cull mode, winding and fill form and the refusals; its
  `_face_control`, `_cull_control` and `_fill_control` runs read one field wrongly and must fail
  (`ctest --preset dev --no-tests=error -R "^startup_rasterizer_decode" -V`). Not used by draws yet.
- Audio produces no output from the console audio path. A separate desktop render sink exists
  (see "Audio output"); nothing connects the console path to it yet.

## Audio output

While a permissive XApu connection is open, the desktop shim requests 1 ms Windows
timer resolution for the mix worker's short refill waits. The last disconnect
releases the request. This affects scheduling and preserves the existing audio
routing and sample format.

With `X2W_AUDIO_PERMISSIVE=1`, `xapu.dll` tracks active handles across `XApuConnect`
and `XApuDisconnect`. `XApuConnect` allocates a zeroed shared buffer of the size given by the
32-bit value at input `+0x0C` and writes its address to output `+0x00`. In one title's calls the caller read only that
pointer, and its later decode commands pointed into the buffer. A size of 0, or over a 64 MiB sanity cap, is refused
(`E_INVALIDARG`) without a handle, and the buffer is released on `XApuDisconnect`. It refuses null
or unreadable command headers (`E_POINTER`), null or unreadable inputs and null or unwritable outputs
(`E_POINTER`), and unknown or disconnected handles (`E_HANDLE`), and records one `(type, slot)`
entry per `XApuEnqueueCommand` call (reading the 32-bit command type at `+0x00` and 32-bit slot
index at `+0x04`) in a per-connection FIFO ring buffer capped at 16,384 entries (192 KiB per
connection; additional enqueues return `E_OUTOFMEMORY` without dropping queued entries).
`XApuGetResultCount` reports the queued entry count, and `XApuDequeueResult` retires one entry
and zeroes the `0x58`-byte result buffer. The experimental `X2W_XAPU_TYPED_COMPLETIONS=1`
also writes the command type at `+0x00` and slot index at `+0x04`. Decoded bytes at `+0x34`
remain zero in both modes unless the optional Opus decode below is built in. An empty
queue returns `XAPU_E_PENDING_RESULTS` (`0x8AC81001`) and preserves the result buffer. [Microsoft XAPU error codes](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/errorcodes?view=gdk-2604#xapu)

Command/result types `1` (open), `2` (close), `3` (packet), and `5` (stop) are established by
observed callers. The experimental mode echoes any 32-bit type value at `+0x00` verbatim
(including unobserved types such as `4`) rather than rejecting it, while keeping payload bytes
outside `+0x00` and `+0x04` zero (`+0x34` carries decoded bytes only in the optional Opus build below). Handles are never reused, and each connection's queue buffer is
released on `XApuDisconnect`. Default non-permissive mode still traps on every export
(`ctest --preset dev --no-tests=error -R '^startup_xapu_queue$' -V`).

The default build does not depend on libopus. Configuring with `-DX2W_WITH_OPUS=ON` (and `Opus_DIR` pointing at an
installed libopus CMake package) links libopus into `xapu.dll`. The stand-ins link the static MSVC runtime, so libopus
must be built with `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`. Tested with libopus 1.5.2;
its notice is in [`src/startup_stubs/THIRD_PARTY_NOTICES.md`](../src/startup_stubs/THIRD_PARTY_NOTICES.md). With typed
completions on, a type `3` command then decodes one Opus packet. The command fields as one title fills them:
`+0x1C` sample rate (32-bit float: 8000, 12000, 16000, 24000 or 48000), `+0x24` frames, `+0x28` packet bytes,
`+0x2C` packet pointer, `+0x34` output bytes (frames × channels × 4, 1 or 2 channels) and `+0x38` output pointer
(interleaved 32-bit float). The result carries the decoded byte count at `+0x34`, which is smaller than the output
size when the packet holds fewer frames. Each slot keeps its own decoder; type `5` resets it and type `2` frees it,
as does `XApuDisconnect`. A command with no packet (null pointer and 0 bytes) completes with 0 bytes, as without
libopus. Any other invalid field (rate, channel count, frames over 5760, packet over 64 KiB, unreadable packet,
unwritable output, slot 256 or above) returns `E_INVALIDARG`. A packet that libopus rejects, or one holding more
frames than `+0x24`, returns `E_FAIL`, and a command shorter than `0x40` readable bytes returns `E_POINTER`. A full
queue returns `E_OUTOFMEMORY` before any decode, reset or close. A refused command queues no result, leaves the
output untouched and keeps the existing decoder object. Invalid arguments and a full queue fail before calling
libopus. If libopus fails while using an existing decoder, its internal state follows libopus. Decoded samples
are copied from scratch only after the result is queued.

```powershell
cmake --preset dev -DX2W_WITH_OPUS=ON -DOpus_DIR=<libopus install>/lib/cmake/Opus
cmake --build --preset dev --target startup_stub_xapu startup_xapu_queue_check startup_xapu_opus_check
ctest --preset dev --no-tests=error -R '^startup_xapu_(queue|opus)$' -V
```

`startup_xapu_opus` encodes generated 1 kHz tones with the same libopus and checks exact byte counts (3840 for
48 kHz mono, 7680 for stereo, 1920 for a 10 ms packet in a 20 ms buffer, 1280 for 16 kHz), bit-exact PCM against
a reference decoder, untouched guard floats past the decoded frames, a malformed packet, each refusal above with
an untouched output and an empty queue, a full queue that leaves the slot's stream intact, reset and close each
starting a fresh decoder, a rate change on a live slot, and that type `3` decodes nothing with typed completions
off. It is registered only in the optional build.

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

Not established: `XApuEnqueueCommand`/`XApuDequeueResult` payload fields beyond `(+0x00, +0x04, +0x34)`,
the shared buffer's layout and output bytes past `+0x08` of `XApuConnect`, what is audible (no capture-path check), and any
hardware or driver glitch count.

## APU heap stand-in

With `X2W_AUDIO_PERMISSIVE=1`, `AcpHal.dll` serves `ApuAlloc` from the heap that `ApuCreateHeap` commits. The
heap is a first-fit free list in a static table of 4096 extents, guarded by one lock. `ApuFree` gives a
block back and merges it with free neighbours, so streaming code that allocates and frees repeatedly no
longer exhausts the heap. While nothing has been freed the offsets are the ones a bump allocator hands out.

- A block is aligned in address (a power of two, 16 at least; 0, anything below 16 and a value that is not a power of two
  all mean 16). Size and alignment are 32 bits. `ApuCreateHeap` creates and publishes the heap under one lock, so
  competing first calls cannot reset a heap that already has live blocks. A heap whose 64 KiB-rounded size
  exceeds 4 GiB is refused with `E_INVALIDARG` without committing memory, so every block fits the APU range.
- Null or non-writable output (`virtual_address` or optional `apu_address`) is `E_POINTER` without
  allocating; size 0 is `E_INVALIDARG`; no room, an offset past 32 bits, or a full extent table is
  `E_OUTOFMEMORY`. A refused call leaves the heap and the optional APU-address output as they were,
  and writes null to the virtual-address output when writable.
- A block that is reused is zeroed, as fresh heap memory is.
- `ApuFree(NULL)` is `S_OK`. An address that does not start a live block (never allocated, inside a block,
  already freed) is `E_INVALIDARG` and changes nothing.
- `ApuIsVirtualAddressValid` and the two map calls read the published heap bounds under the heap lock and
  test the allocatable 32-bit APU range `[0x10000, min(heap_bytes, 4 GiB))` (excluding the unused guard
  range below `0x10000`), not live blocks.

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
| `INSERTQ xmm, xmm, len, idx`, `INSERTQ xmm, xmm`, `EXTRQ xmm, len, idx`, `EXTRQ xmm, xmm` | a jump to a stub that does the field move in integer registers (xmm0-15) |
| `MOVNTSS m32, xmm`, `MOVNTSD m64, xmm` | `MOVSS`, `MOVSD` stores (one opcode byte; the non-temporal hint is dropped) |
| `RDPRU` | `RDTSC`; `NOP` (the time-stamp counter stands in for MPERF and APERF, so their ratio reads 1) |
| `RDPID r64` | a jump to a stub that runs `RDTSCP` and moves its `ecx` to the register |
| `SHA1RNDS4`, `SHA1NEXTE`, `SHA1MSG1`, `SHA1MSG2`, `SHA256RNDS2`, `SHA256MSG1`, `SHA256MSG2` | not rewritten: performed by the debugger on the stopped thread at every hit, counted as `sha (per hit)` |

The stubs keep the other registers and the flags. Instructions that the 5-byte jump overwrites
are re-encoded behind the stub with iced-x86. `events.txt` names the first 40 rewritten sites and
the summary counts them by instruction. `tools/startup/host_cpu.py` lists the forms that are
refused (memory operands, non-zero `ModRM.reg` on immediate `EXTRQ`, malformed prefixes, fields past bit 63).

`python -B tests/startup_host_cpu_check.py` assembles source-built functions, patches them the same
way and calls them on the host: 1162 `INSERTQ`/`EXTRQ` results against a model, `CLZERO` with a
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
experimental `x2w_xmem_page_stats` export. A page-token write (`x2w_xmem_write_physical`) to a page
that has no grant is refused with `ERROR_INVALID_PARAMETER`, like a mapping of it.

The slots are 64 KiB pieces of shared 256 MiB slabs (4096 slots each, `src/xmem_slab_slots.h`). A slab reserves
its capacity with `SEC_RESERVE`. Each slot commits on the first mapping or write. A freed slot remains
committed, is zeroed on reuse, and is chosen before a fresh slot; once all slots of a slab are freed and
unpinned, the empty slab is closed and its commit is returned to the OS. The experimental `x2w_xmem_slab_stats`
export reports open slabs and used and free slots. `xmem_slab_slots` checks the bookkeeping without Windows
calls. `xmem_mapped_pages_commit` measures commit for a small map, and `xmem_mapped_pages_commit_failure`
injects a commitment failure and checks that the reservation and slab counts are unchanged and the pages
can be mapped on retry. With the experimental `X2W_XMEM_LAZY_COMMIT=1`, a fresh slot waits for its first
CPU access instead of its mapping (see [Mapped pages committed on first touch](experiments.md#mapped-pages-committed-on-first-touch)).
Caller-owned placeholders must already match the requested view size. The
heartbeat also prints while the request counters are unchanged
when either memory value changed, so the cost of a stalled load is visible. Each report is followed
by a line of failed requests by cause and a line of page-destination requests by end (see
DirectStorage failure causes and DirectStorage page-request ends under Findings).

The live log window also shows DirectStorage heartbeat lines and `x2w-render totals` as they
arrive. It reports frame counts and FPS from the graphics heartbeat without inferring whether
draws were executed. Network refusals are shown when observed; the window does not establish
that all network activity is blocked. To check this presentation with generated logs, run
`python -B tests/startup_live_telemetry_check.py`. See [#1199](https://github.com/R0Y00/Xbox2Windows/issues/1199)
for memory accounting and [#1281](https://github.com/R0Y00/Xbox2Windows/issues/1281) for
the launcher's concurrent-hook limitation.

```powershell
cmake --build --preset release --target xmem_shim startup_stub_xcrdapi xmem_slab_slots_check
ctest --preset release --no-tests=error -R '^(xmem_slab_slots|xmem_page_stats|xmem_mapped_pages_commit(_failure)?|startup_heartbeat_commit)$' -V
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

## 4 KiB allocation answers for small textures

A console executable asks the allocation-info entries (slots 25 and 108) with `Alignment` 0x1000 for
a texture that is not a render or depth target and whose top level, padded to 4 KiB blocks, holds at
most 0x80007 bits. It keeps the answer only when the returned alignment is 0x1000, and otherwise asks
again with alignment 0. The stand-in used to drop the 4 KiB request (`sanitized()`) and forward the
desktop answer, 64 KiB at 64 KiB alignment. The executable then recorded 16 pages for a texture the
console sizes at one, and its streaming read for that texture ran 64 KiB through the following
resources' compressed chunks. It ended inside one of them, which the executable's own check rejects
before the request is submitted. The storage stand-in then refused the request as destination
capacity exceeded.

For a description in scope the stand-in now answers the console size at 4 KiB alignment
(`src/startup_stubs/small_texture_4k.h`): the GFX10 4 KiB thin-block layout as AMD addrlib computes it.
Each level is padded to whole blocks (128 x 64 texels at 4 bits a texel, 64 x 64 at 8, 64 x 32 at 16,
32 x 32 at 32, 32 x 16 at 64, 16 x 16 at 128; block-compressed formats count 4 x 4 blocks) until the
first level no wider than half a block and no taller than a block. That level and every smaller one
share one block. (addrlib also caps a tail at 8 levels; a 4 KiB thin tail never holds more than 7.) The first eight uses are logged as
`x2w-d3d12 allocation info: 4 KiB answer ...`. Desktop D3D12 asked with a 4 KiB alignment is not a
substitute: on the adapter measured it answered 0x2000 for a 128 x 4 R16G16B16A16 texture, where the
4 KiB layout takes 0x4000.

Scope: a 2D texture with layout UNKNOWN, `Alignment` 0x1000, one sample, one array slice, power-of-two
width and height up to 16384, `MipLevels` 0 (full chain) up to the full chain, no render-target or
depth flag, a single-plane format of 4, 8, 16, 32, 64 or 128 bits a texel (`small_texture_4k_bits`;
depth with stencil is not answered), the executable's 0x80007-bit bound, and a desktop answer that is
not a failure. Any other description, arrays, other layouts and 1D or 3D textures included, keeps the
desktop answer.

The executable's memory for such a texture is then only the answered size. The desktop linear footprint
(256-byte rows, 512-byte level placement) can be larger: a 32 x 32 R8G8B8A8 texture takes 0x1f80 bytes.
So for a description that got the 4 KiB answer (`Alignment` 0x1000 and in scope) the stand-in uses one
compact linear plan (`small_texture_4k_plan`) in every call that touches that memory:

- XG slot 2 reports the answered size at 4 KiB alignment and, per level, the plan's offset, row bytes and
  size;
- XG slot 8 writes every row of a level there, or nothing: a row pitch shorter than a row or a level past
  the last one fails without writing or announcing a write;
- texture upload (`upload_texture`) reads the same rows from there.

In the plan, every row has its exact byte width (a row of texels, or of 4 x 4 blocks), and the levels
follow each other from the top level with no padding. Its size never exceeds the answer:
`startup_small_texture_4k` checks this for every accepted shape. The texels are linear, not the console's
4 KiB swizzle. Data the executable streams in its own 4 KiB layout is therefore read in the wrong order,
as it already was with the desktop layout. A GFX10 4 KiB tiling plan, such as `Gfx10Texture` in #1469,
can replace the compact plan in all three calls at once. At alignment 0 nothing changes: the desktop
answer and the desktop footprint.

Measured on one console executable under the startup debugger: 18 such reads failed during
first-run setup in every run, each with the executable's check firing once. With this answer both
counts were 0 to the same screen. Over the executable's archives, every resource of the family it
places in 4 KiB blocks starts exactly the size above after the previous one.

- `ctest --preset release --no-tests=error -R '^startup_small_texture_4k(_whole_width_tail_control)?$' -V`
  checks literal sizes and the refused descriptions without a device; the control expects a tail one
  block wide and must fail.
- `ctest --preset release --no-tests=error -R '^startup_small_texture_4k_addrlib$' -V` compares every
  accepted power-of-two shape and mip count, for one format of each bit depth, with addrlib's
  `ADDR_SW_4KB_S` size (Mesa 25.1.0, `tools/fetch_addrlib.py`); it skips (77) without addrlib.
- `ctest --preset release --no-tests=error -R '^startup_small_texture_4k_device_' -V` goes through the
  wrapper: slots 25 and 108 must give the 4 KiB answer, and controls the desktop answer.
- `ctest --preset release --no-tests=error -R '^startup_small_texture_4k_xg' -V` covers a two-mip 32 x 32
  R8G8B8A8 texture at `Alignment` 0x1000:
  - slot 2's size, alignment and level records;
  - slot 8 into exactly the answered size in front of a no-access guard page, with distinct rows and a
    different sentinel per level, checked byte for byte;
  - refused copies that leave memory unchanged;
  - alignment 0 unchanged: slot 2 reports 0x10000 and slot 8 writes all 32 rows.

  The `_no_limit_control` run asks at an explicit 64 KiB alignment, which gets no compact plan, and must
  fail.
- `ctest --preset release --no-tests=error -R '^startup_small_texture_4k_draw_' -V` writes the same
  texture through slot 8 in front of a guard page, places it (slot 133), draws both levels through the
  translated path on WARP and on the hardware adapter, and compares all 48 x 32 read-back pixels.

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

## Committed resources (device slot 76)

Device slot 76 (Xbox order, +0x260) takes the seven arguments of desktop `CreateCommittedResource`:
heap properties, heap flags, desc, initial state, clear value, riid and ppv. Measured: a console
package module (an upscaler) creates its resources through it. The stand-in shares the
committed-resource path of slot 127 (the `CreateCommittedResource3` order, with protected session and
castable formats, which slot 76 does not pass): the same initial-state choice per heap, the same
clear-value retry and title memory, and a log line naming the slot. Run
`ctest --preset dev --no-tests=error -R '^startup_committed_slot76_(warp|control)$'`
(`tests/startup_committed_slot76_check.cpp`): a render target with a clear value and buffers on the
default, upload and readback heaps, passed with initial state 0xffffffff as a title passes them, must
be created and leave the device alive. The control asks for a 2D texture on an upload heap and must
return `E_INVALIDARG` with no resource and leave the device alive. The control also requires all four valid creations to pass. `startup_committed_slot76_hardware` runs the creation checks on a physical GPU.

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

## Placed buffer allocation walk (device slot 133)

Slot 133 aliases a buffer only when every byte from the base of the allocation holding the address
to the end of the buffer is committed in that allocation, and the memory is private. It used to
measure the whole committed run from the base with one `VirtualQuery` per region and compare the
buffer with that length. `VirtualQuery` scans page-table entries from the queried address to the end
of the run of identical attributes, so on a large private region with touched pages each call costs
milliseconds (about 5 ms per GiB with one touched page per 64 KiB, measured on Windows 10 19045).

`src/startup_stubs/placed_buffer_range.h` first asks `QueryVirtualMemoryInformation`
(`MemoryRegionInfo`, Windows 10 1607 and later, resolved at run time) for the allocation's base, size
and committed bytes, which come from the allocation's bookkeeping without that scan. When the
allocation is private and its committed bytes equal its size, every page is committed, so the full
walk's run is the whole allocation and the decision follows from base and size. Otherwise (a partly
committed or mapped allocation, a free address, the call missing) it walks the regions from the base
and stops at the end of the buffer, because regions after it cannot change the answer. The alias or
refusal is the same as before; nothing is remembered between placements, so a page decommitted,
released or replaced directly through the Windows API is seen by the next placement. The alignment,
memory-type and state rules are unchanged. The five-second `x2w-d3d12 views noted` line in
`stderr.txt` reports `placed buffers <n> (region-information proofs <n>, region queries <n>)`.

- `ctest --preset release --no-tests=error -R '^startup_placed_buffer_range$' -V` compares every
  decision with the full walk on the same memory: 256 buffers spread over a 64 MiB private block
  whose alternating 64 KiB pages have different protections (it prints the region queries per
  placement of both walks); a middle page decommitted under a buffer, before a buffer and five pages
  before a buffer, then committed again; a block released and replaced by a 1 MiB block at the same
  base; a mapped view; a reservation committed in two steps; and a placeholder piece replaced by
  committed private memory. Each case places a buffer before changing the memory, so state kept
  between placements would show up as a mismatch. A 512 MiB committed block with one touched page per
  64 KiB must be proven without any `VirtualQuery` call; the check prints the time per placement of
  the full walk, the bounded walk and the proof. `_region_only_control` trusts the allocation size
  without the committed-bytes check and passes only when its decisions differ in the partly committed
  cases and nowhere else; `_walk_only_control` never uses the region information and passes only when
  that call count is its sole failure. Neither is `WILL_FAIL`, so a crash cannot pass them. A
  refused allocation or mapping is a capability skip (77) in the controls.

## Command-list upload allocation (slot 148)

In render mode, slot 148 takes a byte count and a power-of-two alignment and returns CPU-writable D3D12 upload memory for the current recording. An invalid request or a closed list returns null. Compute and graphics root SRV slots 43 and 44 translate an aligned address within the current recording into its desktop GPU address. The desktop command allocator retains the pages until every recorded reference and tracked queue submission has retired and the allocator can be reset. A failed submission-fence signal prevents reuse.

The fixture export `x2w_d3d12_upload_gpu_address(list, cpu, bytes)` reports the GPU address only for a nonempty range contained in one allocation of the open recording. The independent client and its four WARP and hardware configurations are described in [#2210](https://github.com/R0Y00/Xbox2Windows/pull/2210).

## Textures published by address (device slot 70)

Device slot 70 gets a resource description, a view and the address of the texels. When a live
texture with that description exists at the address, its view is created as before. The shapes slot
70 already read from an address keep their own paths unchanged: the single texel, the `0x105`
BC3/BC7 surface (partial last tiles included) and the `0x109` BC chains. For any other description
with no live texture whose `Layout` is `0x101`, `0x102`, `0x105`, `0x106`, `0x109` or `0x10a` (256 B,
4 KiB and 64 KiB blocks in the standard and display orders, no XOR), the texels are read from the
address in that console layout (`src/startup_stubs/address_texture.h`, device-free). A translated
draw that samples the view copies each mip of each array slice from that layout into desktop
footprint rows (`src/startup_stubs/gfx10_layout.h`: address equations, block sizes and mip-tail rules
after AMD addrlib, Mesa 25.1.0, `src/amd/addrlib`, MIT). A 256 B layout is taken with one mip only.

The description must be two-dimensional, at most 16384 texels a side, with 1 to 15 mips, one sample,
no target, depth or unordered-access flag, and a format with a known element size (8- to 128-bit
texels and BC1 to BC7, whose largest mip must then be whole 4x4 blocks). The view must use a supported typed format that matches the storage format or belongs to its typeless parent. The 2D or 2D-array view must fit the mip and slice ranges. By default, a mismatched or interior view overlapping a live resource preserves the destination descriptor.
Anything else is refused as before and leaves the destination descriptor unchanged: layout `0` and
the linear mode `0x100` without a live texture, three-dimensional and cube views, multisampled
textures, the XOR swizzle modes and every other layout value. The texels must be readable and span
at most 64 MiB.

A console-layout description is held (not published, descriptor unchanged) in the two cases a
partial-tile BC view is held for. `console_layout_xg_rows`: the latest XG stand-in copy announced
for the address wrote desktop-footprint rows, so the bytes are not in the layout the description
names. As for a partial-tile view, a texture already published there samples the null view while
that is so. `console_layout_beyond_xg_size`: the layout runs past the size the XG stand-in reports
for the description (the desktop copyable footprints of a slice rounded up to 64 KiB, times the
slices), which the title may not own.

Upload lifetime: each upload of a console-layout texture is written into an upload buffer of its own
(`x2w_upload::make_page`), not into the shared upload ring. The recording keeps a reference to the
buffer and to the texture until the list is reset, and each `ExecuteCommandLists` of the recording
takes its own references and signals a fence behind the submit; a later submit drops the references
of the submits whose fence has completed. A submit whose fence cannot be created or signalled keeps
its references. The earlier by-address shapes keep using the shared ring as before. The desktop
texture of a by-address texture is made on the first draw that samples it and is not released. The
texels are compared with the uploaded copy on each frame that binds them, as for the earlier
by-address shapes, and at once after a repeated slot 70 call for the same texture, which announces
a write: a description and address that were published before are published again rather than
taken as a live resource, so changed texels are uploaded again.

Limits and choices that are not measured against a console:

- The texels are taken in the layout the description names. The XG stand-in's default copy writes
  rows whatever the swizzle mode; a description filled through it is held as above. A write by the
  title itself is not told apart from one in the named layout.
- One upload buffer is created per upload and there is no bound on the bytes uploaded in a frame.
  A policy for releasing textures that are no longer sampled and for spreading large uploads over
  frames is left to a follow-up.
- Second sources for the address equations: only the 16-byte standard rows of `0x105` and `0x109`
  are compared with another implementation in the tree (`bc7_4k_s.h`, `bc64_chain.h`);
  `tile64_plan.h` checks the `0x10a` mip starts and tail, not the bit order. The display (`_D`)
  rows in `gfx10_layout.h` that differ from the standard ones (1-, 8- and 16-byte elements; the 2-
  and 4-byte display rows equal the standard rows) and the standard rows for 1-, 2-, 4- and 8-byte
  elements are checked only against the addrlib pattern rows restated from the addrlib source text
  in `startup_gfx10_layout_check.cpp` (all five element sizes) and in the Python oracle of
  `startup_address_console_check.py` (4- and 8-byte elements), both written from the same source.

Log lines (stderr and `X2W_TRAP_LOG`):

- `x2w-d3d12 slot70 texture kind <n>: <outcome>; first seen res{...} view{...} address=...`, once
  for each new combination of format, layout, one or several mips, one or several slices, flags,
  dimension, view format, view dimension and outcome (at most 200). The outcome is
  `existing resource`, `published BC shape` (the earlier `0x105` and `0x109` BC paths),
  `published console layout`, one of the two hold reasons, `texture_no_resource` (no live resource at that base and description), `texture_unreadable` (the texels are not readable), `texture_over_64mib` (more than the 64 MiB single-copy limit) or `texture_unavailable` (publication failed otherwise).
- `x2w-draw by-address uploads: <n> submits completed and released (...)`, for the first twelve
  submits that drop retained uploads.

Checks (reproduction in `docs/experiments.md`, "Sample by-address textures of the non-XOR console
layouts"):

- `ctest --preset release --no-tests=error -R '^startup_gfx10_layout' -V`
- `ctest --preset release --no-tests=error -j 1 -R '^startup_address_console_' -V`

### Overlapping texture views

`X2W_SLOT70_ALIAS_TEXTURE=1` lets slot 70 publish a multi-mip 2D or 2D-array
SRV over a live slot 133 texture on the same device. The request must fit a
single owner and map to a contiguous range of whole mips in one or more
array slices. The layout check matches the storage format, swizzle, mip
offsets, block row widths and mip-tail origins. Multi-slice views also need
the same slice stride. The desktop descriptor names the matching mip and
slice range of the existing backing, so GPU writes through other views
are visible with normal resource transitions. A second, differently
described placed texture cannot generally inherit these bytes under
[D3D12's texture data inheritance rules](https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-aliasing-and-data-inheritance).

`overlapping_resource` still applies when the request crosses a resource
boundary, meets multiple live resources, or overlaps a buffer, a lazy
texture, or an incompatible texture. This includes offsets that miss whole
subresources, different formats or swizzles, backing with substituted
format or dimensions, depth resources and resources that deny SRVs.
A 2D view requires a single-slice owner. The existing console-layout
holds still apply. The switch defaults off.

The synthetic WARP fixture publishes the alias, clears the owner's mips
through RTVs and compares the draw's 32 pixels with a CPU reference.
It exercises both the first slice's mip tail and the second slice's
interior address. The disabled and misaligned cases preserve the null
descriptor. The wrong-slice control must differ from the interior reference
at all 32 pixels.

```powershell
ctest --preset dev --no-tests=error -R '^startup_slot70_texture_alias_warp$' -V
```

## Linear buffer backing (device slot 133, opt-in)

By default slot 133 aliases a buffer in private memory allocations of any size, through an existing heap cached by
allocation base and opened without a `D3D12_FEATURE_EXISTING_HEAPS` query. The opt-in
`X2W_D3D12_CAP_EXISTING_HEAP=1` limits that to allocations up to 256 MiB (`unplaced_existing_heap_allowed`):
the default keeps the formerly aliased path for larger allocations because a capped copied placement is
refused by `CopyBufferRegion` (list slot 15) as `title_backed_source`, losing a late guest write made after
recording and before submit, while the cap skips multi-GiB existing heaps to avoid pinning them. With
`X2W_XMEM_LAZY_STARTUP=1`, `commit_lazy_block` runs only when the buffer is eligible to open an existing heap
(checking `MEM_PRIVATE`, 64 KiB offset alignment, nonzero width, containment inside `allocation_bytes`, and the
256 MiB cap when `X2W_D3D12_CAP_EXISTING_HEAP=1` via `can_commit_lazy_placed_buffer` before re-measuring an
uncovered prefix, and right before
`OpenExistingHeapFromAddress` opens the heap). Anything else gets a
committed default buffer that does not alias the title memory, holds zeros and is reported as
success (`not aliased` in the log). `reserve_upload_ring` grows in demand
chunks from 16 MiB up to the 256 MiB total cap. A reservation owns its mapped
bytes for the process; the allocator may use an older unused tail but
refuses exhaustion or allocation failure rather than resetting live storage.
`upload_reservation` exercises the shared cursor and returned-pointer contract
without D3D12; on non-Windows hosts run
`c++ -std=c++17 -Isrc tests/upload_reservation_check.cpp -o out/upload_reservation_check && out/upload_reservation_check`.
`X2W_D3D12_LINEAR_BUFFERS` changes placement for one buffer kind: a
buffer description with row-major or unknown layout and initial state `COMMON`
(`src/startup_stubs/linear_buffer_backing.h`, [#1914](https://github.com/R0Y00/Xbox2Windows/issues/1914)).
Buffers in `UNORDERED_ACCESS` or acceleration-structure state, and textures, keep the default backing.

| Value | Backing |
|---|---|
| unset or `0` | the default above; none of the linear-buffer code runs |
| `1` | shared when the adapter reports `D3D12_FEATURE_EXISTING_HEAPS` (queried once, before any open) and the buffer lies in one committed private allocation at a 64 KiB-aligned offset; otherwise copied; otherwise the default backing, with the refusal recorded |
| `copy` | copied; otherwise the default backing, with the refusal recorded |

- Shared: a cross-adapter buffer placed on the private allocation's heap, cached by base. The GPU reads and writes title memory; nothing is copied.
- An XMem view is never shared. A heap opened from a view keeps it mapped, and XMem's unmap of those pages then fails with error 87. Sharing reports `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)` (`0x80070032`) and the buffer is copied. A buffer across several allocations, or at an unaligned offset, reports `E_INVALIDARG`. Both are decided before any D3D12 call.
- Copied: a committed default buffer that is refreshed from title memory at each submission through a command-queue wrapper, just before the title's lists run, not at bind or record. Only bytes that changed since the last upload are copied, through an upload staging buffer.
  - **Internal queue.** Every refresh runs on one internal COPY queue per device, which never waits on a title fence. It signals an internal fence value, and each buffer remembers the value of its last refresh.
  - **Consumer wait.** Every submission through a queue wrapper, of any type (direct, compute or copy), first makes its queue `Wait` for the highest such value among the live copied buffers that has not completed. A consumer on any queue therefore runs after every refresh enqueued before it, wherever that refresh came from. So when one CPU write is followed by submissions on two queues, both read the new bytes, even if the first is still held behind a title fence.
  - **Producer wait.** A refresh's copy is also ordered behind earlier GPU access to the backing. Simultaneous access allows concurrent operations on one buffer only on disjoint ranges, so the copy must not overlap an earlier submission that still reads or writes it. Each queue wrapper signals a user fence of its own after the title's lists. Before a refresh's copy executes, the internal queue `Wait`s for every queue's latest user fence value that has not completed (`producer_waits`). The wrappers cannot tell which buffers a submission reads, because titles bind them by GPU virtual address, so every earlier submission counts as a user of every copied backing. This conservative choice can idle the GPU between a frame's submissions and the next refresh.
  - **Unsafe refresh refused.** A submission whose refresh has no safe completion order is refused (`unsafe_refresh_refusals`, logged). That is the case when the refresh would have to wait for an earlier submission, on any queue including its own, that sits behind a title `Wait` (queue slot 17) on a fence that has not been reached, or for one whose user fence `Signal` failed. Waiting could deadlock: the fence may be signalled only by work that itself waits for the refresh, such as queue 2 signalling a fence that queue 1 waits on. Not waiting would let the copy overlap that read, because the blocked queue is not known to stay blocked until the copy finishes; the CPU or another queue may reach the fence meanwhile. The decision is made right before the refresh would be enqueued, and nothing of the submission reaches the GPU: no refresh, no consumer `Wait`, none of the title's lists and no user fence `Signal`. The staging buffers and refresh list are free again, no buffer's upload record or refresh state changes, and a recorded copy's GPU write stays pending on its recording. The next submission copies whatever still differs. The title's own `Signal` and `Wait` on its queues do not pass through this path, so a later `Signal` still releases the earlier queue. A submission with no changed bytes is never refused. A title `Wait` issued on the desktop queue past the wrapper is not seen, and that queue is waited for.
  - **Queue order lock.** A title can call slot 17 and `ExecuteCommandLists` on one queue from two threads at once. Each queue's linear-buffer record has an order lock, so the record of which submissions sit behind which title `Wait` matches the desktop queue's order. Slot 17 holds it from the desktop `Wait` through recording that `Wait`. A submission holds it from the title's `ExecuteCommandLists` through its user fence `Signal`, before `draw_lock` and `allocator_lock` in render mode. Without it, a submission could reach the desktop queue behind a `Wait` but take its user value before that `Wait` was recorded. Another queue's refresh would then wait for that blocked submission instead of being refused. The lock is not held across the refresh, so staging waits and fixture pauses run without it. Nothing under it waits on the CPU for the GPU. The title's `Signal` does not take it, so a queue whose submission was refused can still signal the fence its blocked peer waits on.
  - **Own refresh.** A submission also waits for the refresh it enqueued itself, even if the title released that buffer meanwhile.
  - **Staging reuse.** A staging buffer is reused only after the internal fence value of the copy that read it has completed.
  - **No internal queue.** If the internal queue or its fence cannot be created, copied placements keep the default backing with that HRESULT recorded.
  - In render mode the refresh and the consumer `Wait` run before the wrapper takes its draw and allocator locks, so they precede every list of the submission, including the producer fences `red_producers` adds and each split submission of mesh XMem recordings.
  - Budgets (`X2W_D3D12_BUDGET=1`) count the linear backings as the default backing counts its own. A shared buffer counts as placed memory on its heap, whose backing-heap ticket is in use while a buffer on it is held. A copied buffer counts as a committed buffer. Staging buffers count as upload memory and live for the process.
  - A copied buffer that a recorded copy writes (list `CopyBufferRegion` or `CopyResource`, `list_copy.inl`) keeps the GPU's bytes once that recording is submitted and is no longer refreshed, as the default backing behaves for it. The title memory never receives those bytes, and a refresh spanning them would copy older title bytes over them. Info and totals report it (`gpu_written`). Until the submission the write is pending on the recording, with the texture writes of `list_copy.inl`: a `Reset` or a failed `Close` drops it, and the buffer stays refreshed. The recording's own submission still refreshes the buffer first, and its queue waits for that refresh before the copy runs.
  - **Publish order.** Two submissions on different threads can stage one buffer's bytes at different times and execute their copies in one order but publish the upload record in the other. A buffer remembers the internal fence value of its newest applied publish. An older publish that arrives after a newer one drops the record (`reordered_publishes`), and the next submission copies the whole buffer.
  - **Title memory reads.** Every live copied buffer is snapshotted, not only those the submitted lists read. Its title memory is probed again right before it is staged, and staged under a fault guard. A buffer whose memory the title freed after the snapshot is skipped and counted (`unreadable`); a read that faults anyway is counted (`copy_faults`) and fails that buffer's refresh. A buffer released after the snapshot whose pages are still mapped is staged for that submission, which may read it.
  - The lock that guards the backing is held only to choose and publish. Waiting for a staging fence (a full pool), creating staging or a refresh list, recording it and `ExecuteCommandLists` all happen without it. A `Release`, the fixture exports and another queue's submission therefore never wait behind them.
- Kept default: where neither path keeps its guarantees, slot 133 keeps the default backing above and returns what it returns without the switch. On the console `CreatePlacedResourceX` does not fail, and a title that received a failure faulted on the null object. The refusal HRESULT is recorded instead:
  - a GPU-written use (`ALLOW_UNORDERED_ACCESS`) that cannot be shared records the sharing HRESULT, or `DXGI_ERROR_UNSUPPORTED` when sharing was not attempted. The copy has no point at which GPU writes reach title memory;
  - without the internal refresh queue: `E_NOT_VALID_STATE` (`0x8007139f`) under `X2W_TEST_SUBMIT_FENCE_FAIL=1`, else the HRESULT of creating the queue or its fence;
  - when the copy's default buffer cannot be created, that HRESULT;
  - when the mesh XMem lease path is on (`X2W_MESH86_XMEM_SRV` or `X2W_MESH_INDIRECT_XMEM`) and the placement is one it takes: XMem pages (`MEM_MAPPED`) with console flags 0. `HRESULT_FROM_WIN32(ERROR_BUSY)` (`0x800700aa`) means the placement was deferred to the lease path, which mirrors and publishes that default copied placement. The two never back one placement.
- A submission whose refresh cannot be staged still runs. This happens when the staging pool stays full of in-flight buffers past a 5 s wait (`DXGI_ERROR_WAS_STILL_DRAWING`) or when staging cannot be created. That buffer keeps its last upload and is reported stale until a later refresh succeeds. The first failures are logged (`x2w-d3d12 linear buffers: refresh of <resource> failed <hr>; the submission runs with its last upload (stale)`) and every one is counted.

A shared or copied placement logs `x2w-d3d12 Device::slot133(CreatePlacedResourceX?) linear buffer ... existing-heaps=<hr> share=<hr> -> <hr> shared|copied at submit`. A kept default logs the usual slot 133 line with `(kept default backing: linear refusal <hr>, use=... type=... existing-heaps=<hr> share=<hr>)` appended. `x2w_d3d12_linear_buffer_info` reports path 3 with `refusal_hr`, and `stale`, `refresh_failures` and `last_refresh_hr` for a copied buffer. A refused submission logs `x2w-d3d12 linear buffers: submission on queue <queue> refused: its refresh would have to wait for queue <queue> (<reason>), so it has no safe completion order; none of its lists ran` (the first eight). `x2w_d3d12_linear_buffer_totals` counts `kept_default`, `refresh_failures` and `unsafe_refresh_refusals`. Fixtures read the same facts through `x2w_d3d12_linear_buffer_info` and `x2w_d3d12_linear_buffer_totals` (`src/startup_stubs/linear_buffer_report.h`). Fixture-only switches: `X2W_TEST_EXISTING_HEAPS_UNSUPPORTED=1` makes the feature query report no support, as an adapter with `EXISTING_HEAPS = 0` does. `X2W_TEST_LINEAR_REFRESH_FAIL=<n>` makes staging requests 2, 2 + n, 2 + 2n and so on fail with `E_OUTOFMEMORY`. `X2W_TEST_LINEAR_STAGING_LIMIT=<n>` shrinks the staging pool from 256 buffers to `n`. `x2w_d3d12_linear_buffer_internal_queue` hands fixtures the internal queue, so they can hold refreshes in flight behind a fence of their own. `x2w_d3d12_linear_buffer_test_pause` parks the next refresh at one point (before staging, after staging, or after its copy executed and before it is published), so a fixture can order two threads' refreshes or release a buffer while one is parked. Point 3 instead parks the next title `Wait` (queue slot 17) after the desktop `Wait` and before it is recorded, while it holds that queue's order lock. `X2W_TEST_LINEAR_BUFFER_MUTANT=<name>` removes one guarantee for a negative control. Results and commands: [Production slice: slot 133 linear buffers](experiments.md#production-slice-slot-133-linear-buffers-1914).

Not covered: table SRVs of copied buffers in translated draws still take `frame_copy` at record, and root SRVs (`list_root_srv`) still take `upload_from_title` at bind. Neither reads the slot 133 backing. Shared buffers and their heaps live for the process, like every wrapper, so a private range under one cannot be released.

## Buffer memory SRVs (device slot 70)

Device slot 70 (`src/startup_stubs/d3d12_memory_srv.inl`, validation in
`src/startup_stubs/memory_srv_buffer.h`) also accepts `D3D12_RESOURCE_DIMENSION_BUFFER`
descriptions with:

- `D3D12_SRV_DIMENSION_BUFFER` (raw `R32_TYPELESS`, structured `Format == UNKNOWN` with a stride that is a multiple of 4 up to
  2048, or typed buffers). It resolves an existing live placed (slot 133) or committed (slot 127)
  buffer covering `[address, address + (FirstElement + NumElements) * element_bytes)`. When no live
  buffer exists at the address, `unplaced_memory_buffer` backs the unplaced range: it calculates a
  64 KiB page-aligned extent via `unplaced_buffer_extent`, verifies committed pages through
  `measure_placed_buffer`, and avoids placed alias conflicts (`placed_buffer_range`). It reuses
  cached unplaced buffers per device matching the base allocation when `unplaced_buffer_covers` is true,
  re-checking committed status to handle title memory deallocation/reallocation. On adapters supporting
  `D3D12_FEATURE_EXISTING_HEAPS` for `MEM_PRIVATE` memory, it opens the existing heap and creates a
  zero-copy placed resource alias; otherwise (or on WARP), it falls back to a committed upload buffer and
  synchronizes memory from the host. When the device wrapper is released (`refs == 0`),
  `clear_unplaced_buffers` safely cleans up cached entries. Once resolved, it checks alignment and bounds,
  adjusts `FirstElement` by `(address - base) / element_bytes`, and forwards to `CreateShaderResourceView` (slot 18).

With `X2W_SLOT70_MEMORY_BUFFER=1`, a structured view with first element 0, count 1 and
stride equal to the buffer width can recover from `buffer_unavailable` if its exact byte range
is readable and outside live resources. Slot 70 copies the requested bytes to a D3D12 upload
buffer and publishes a desktop SRV. The per-view guard is 1 MiB, subject to the existing
2048-byte structured-stride limit. The process-wide budget is 128 MiB, charged for the
desktop allocation and the retained CPU snapshot. A repeated publication reuses an identical
snapshot by device, address, size and bytes. A changed publication uses a separate backing
so older descriptors retain their original bytes until device cleanup. To observe a title
write through a desktop descriptor, publish the SRV again. The switch defaults to off.

The `memory SRV switches` census follows the slot 70 refusal census. A snapshot
attempt counts either a new backing or a failed attempt. Reusing unchanged bytes
increments the bind count only. The byte count includes bytes copied into new
backings. The regular buffer refusal cache can skip the regular lookup while the
snapshot path still checks the requested bytes. The tester archive summarizes the
latest cumulative buffer, texture alias and 1D counts in `null-inputs.txt`.

On Windows, run
`ctest --preset dev --no-tests=error -R '^startup_slot70_memory_buffer_(enabled|off|stale|recover|cache_first)_warp$'`.
The fixture publishes a synthetic 256-byte SRV inside a partially committed reservation
and compares WARP draw readback with CPU data. The recovery case first caches a refusal
and then makes the requested bytes readable while the cache witness stays reserved.
The cache-first mutant must sample null. The off control checks the refusal and null
readback. The stale mutant must fail the refreshed readback with the first generation.

With `X2W_SLOT70_DIAGNOSTICS=1`, the heartbeat also reports three snapshot conflicts
by requested resource shape. `snapshot_already_owned` means an indexed placement
covers the whole span. `snapshot_placement_rejected` means the placement lookup
found an ambiguous or crossing range. `snapshot_resource_overlap` means the
general resource index found an overlap after the placement lookup found none.
The legacy `view_does_not_fit` total still includes these conflicts and malformed
snapshot views.

The same switch reports why texture requests touching a live alias registration
missed. Examples include multiple overlapping resources, an uncovered range,
incompatible backing or mip layout, and a hold before alias lookup. A registration
whose range receives no request contributes only to the registration count.
The switch also reports which desktop format check failed. For a format-29
(`R8G8B8A8_UNORM_SRGB`) 1D view of width 7, the desktop-row fixture exercises
publication. A console-layout source is held as `texture_1d_layout_unmapped`
because there is no verified 1D source footprint mapping. The diagnostics switch
defaults to off and leaves descriptor publication rules unchanged.

Run the WARP fixtures with
`ctest --preset dev --no-tests=error -R '^startup_(address_srv|slot70_texture_alias|slot70_snapshot_diagnostics(_off|_no_snapshot)?)_warp$' -V`.
The snapshot fixture checks one request per conflict, exact structured-buffer
readback for the fitting request, the same readback with diagnostics disabled,
and null readback with snapshot publication disabled.

Acceleration-structure views (`D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`,
`src/startup_stubs/memory_srv_as.h`) take their address from the view's `Location`, or from the call's
address when `Location` is zero, do not read the resource description, and require `DXGI_FORMAT_UNKNOWN`
with `D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING` (`unsupported_description`). The view is published through
slot 18 with no resource and the desktop GPU address of the same byte when the address is 256-byte aligned
(`as_unaligned`), starts inside one live placed (slot 133) buffer (`as_not_placed`) and that buffer was
created in `D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE` (`as_wrong_state`). Any other case
leaves the destination descriptor unchanged and logs the reason. The pure checks run on any host
(`ctest -R '^startup_memory_srv_as'`); publishing a view on a device has not been run yet.

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
element, widths, addresses and first elements above 4 GiB, page extent calculation, and cache range
coverage. Its `_control` runs (`height`, `exact-fit`, `first-32`, `unplaced-extent`, `unplaced-covers`)
expect wrong answers and must fail. `startup_address_buffer_srv_*` exercises placed and unplaced buffer
views with valid addresses on a device (WARP and hardware), and verifies the refusal lines.

`startup_memory_srv_volume` is device-free and validates volume (Texture3D) memory SRVs in
device slot 70 (`supports_volume_memory_srv`, `address_view_fits`, `volume_texture_bytes`).
It pins the supported layout to `D3D12_TEXTURE_LAYOUT_UNKNOWN`, enforces `D3D12_SRV_DIMENSION_TEXTURE3D`,
validates block-compressed dimensions and sizing (BC3, BC7), and validates mip bounds. Its `_control`
runs (`layout`, `2d-view`, `3d-view`, `mips-overflow`) expect wrong answers and must fail.
In device runtime, `address_texture_bytes` uses `SliceFootprints::size` to account for desktop copyable
footprint layout (256-byte row pitch alignment and 512-byte subresource placement alignment), while
`volume_texture_bytes` provides device-free packed texel byte sizing.
`startup_address_volume_srv_*` exercises volume SRVs on a device (WARP and hardware)
with multiple depth slices and mip levels, where footprint row pitch (256 bytes) exceeds packed row bytes (16 bytes),
and verifies refusal lines for mismatched view dimensions, short backings, out-of-bounds mips, explicit clamps,
and console layouts. On Windows 11 Build 26100 with an AMD Radeon RX 6600M (driver 32.0.21045.11001),
`startup_address_volume_srv_hardware` passed with all 32 RGBA pixels exact.

- `ctest --preset release --no-tests=error -R '^startup_memory_srv_(report|buffer|volume)$' -V`
- `ctest --preset release --no-tests=error -R '^startup_memory_srv_(buffer|volume)_.*_control$' -V`
- `ctest --preset release --no-tests=error -R '^startup_address_(srv|buffer_srv|volume_srv)_' -V`

## Command list CopyTextureRegion (list slot 16)

Command list slot 16, `CopyTextureRegion`, maps each location's resource to its desktop resource, checks the
call, and forwards it only when the desktop would accept it. A refused call is counted and logged
(`x2w-d3d12 CopyTextureRegion refused: <reason>`), not forwarded. Capability, measured on the portable
check only:

- Argument checks (`src/startup_stubs/list_copy_texture.h`): footprint on both sides, empty or inverted box,
  source and destination bounds, row pitch (multiple of 256), offset (multiple of 512), footprint end inside
  the buffer (includes the destination origin, the source box origin and the footprint-height slice stride),
  block alignment, every multiply and add in the footprint arithmetic checked for wrapping. Block-compressed
  bounds are whole 4x4 blocks, as in D3D12: a 2x2 or 1x1 mip takes the 4x4 footprint GetCopyableFootprints gives it.
  An explicit box must also end on a block (`{0,0,2,2}` on a 2x2 mip is `bad_alignment`): forwarded, it makes the
  desktop list's Close fail with E_INVALIDARG.
- Subresource index: `mip + slice * mips + plane * mips * array_size` must lie inside the resource's
  `mips * array_size * planes`; a mip-only modulo would read an out-of-range slice as mip 0
  (`bad_subresource`). A planar format (more than one plane) is refused (`planar_unsupported`).
- The two sides must name the same format (`format_mismatch`) and the same sample count
  (`sample_mismatch`). Multisampled and single-plane depth-stencil copies require equally sized whole
  subresources, destination origin 0 and a null source box (`partial_multisample`, `partial_depth_stencil`).
  Buffer footprints are single-sample, so a multisampled texture and a buffer are refused.
- Upload and readback footprint buffers keep their fixed heap states and get no barrier; an upload buffer as
  destination or a readback buffer as source is refused (`bad_heap`).
- A source or destination texture is brought up to date from its guest backing (`upload_texture`) before the
  copy, so a later first sample does not replace the copied texels. A copied (non-aliased) source buffer is
  refused (`stale_source`).
- Two subresources of one resource are refused (`same_resource`): resource states are tracked per resource.

Wrapped-list readback (`tests/startup_list_copy_texture_warp.cpp`): two images round-trip through array
slices 1 and 2 of a 2-mip, 3-slice texture (upload buffer, texture, readback buffer, byte-for-byte), and a
list with an out-of-range subresource, an index of `0xffffffff` and a footprint format mismatch is refused
before any desktop copy, leaving a sentinel-filled readback buffer unchanged. On a BC1 2x2 mip, the same
list refuses boxes with an unaligned right or bottom edge, copies the aligned 4x4 block, and reads back
all eight block bytes with the next byte unchanged. Reproduce the portable and wrapped-list checks with
`ctest --preset dev --no-tests=error -R '^startup_list_copy_texture' -V`.
