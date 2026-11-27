# Fixed mesh pipelines through device slot 57

The varying fixture sends two distinct values through PARAM12.xy to the pixel shader. With the six-attachment configuration the covered texel reads back as `0xffff00ff`. A mismatched fixed descriptor and a missing color attachment each preserve the clear color. Run `ctest --preset dev --no-tests=error -R '^native_fixed_mesh57_varying_(warp|hardware)$' -V` to reproduce.

For an independently supplied raw pixel stage, `compile_dxil SDK CACHE gfx1013-raw-mesh-ps INPUT OUTPUT` reads a sibling `.pslayout` containing `meshps1 ENABLE ADDRESS WAVE_SIZE` as decimal integers. ENABLE and ADDRESS are the pixel input masks. WAVE_SIZE is 32 or 64. Append `guest_root` to load the corresponding serialized 1.1 root signature from the sibling `.root` file. A raw mesh producer using the same interface specifies `packed_parameters` in its `.mslayout`. Each PARAM occupies one signature register, following position at register zero. Channels with conflicting interpolation modes in one PARAM require a different layout.

The source-built fixture creates a fixed pipeline with guest MS and PS instructions. The stand-in translates both to DXIL using its C++ shader library, creates a desktop mesh PSO and submits a guest command-list slot 61 indirect command. A GPU compute pass supplies the dispatch arguments. The tested interface uses an empty guest root, one to eight single-sample color attachments, explicit rasterizer, blend, depth and stencil state, and an MS with no resource bindings. The PS exports a constant to color target zero.

From the x64 Windows build environment:

```powershell
cmake --build --preset dev --target startup_indirect_draw_indexed
ctest --preset dev --no-tests=error --output-on-failure -R '^(native_fixed_mesh57(_mask|_mrt)?|startup_indirect_mesh_(request|cache))_(warp|hardware)$'
```

The three fixture variants use one color target, six color targets with depth/stencil, and the same six with a red-only target-zero write mask. The six-target formats are 29, 28, 28, 28, 34 and 28, with depth/stencil format 20. Back-face culling, counterclockwise front faces, greater-or-equal depth, depth writes and stencil replacement are enabled. The four-by-two GPU readback has a single covered texel at (2,0). It is white with all channels enabled or red with the narrower write mask. Depth/stencil at that texel changes from 0.25/3 to 0.5/7. The other color attachments retain their clear values. Changing a fixed descriptor word or removing a color attachment produces no color or depth/stencil change.

The ten tests above passed on the working tree based on `671b9ac`, Windows build 26200, with the debug layer enabled on WARP and an NVIDIA GeForce RTX 4060 Laptop GPU (driver 32.0.16.1656).

## Per-pixel texture selection

The `native_fixed_mesh57_waterfall_{warp,hardware}` fixture draws a triangle covering eight pixels. The guest PS uses READFIRSTLANE, CMPX, an indexed descriptor load, IMAGE_SAMPLE and a WREXEC retry. Neighboring pixels select different Texture2D views. With a unit UV gradient the two views return red and green from mip 1. The `waterfall_fine` variant holds UV constant and both views return blue from mip 0. The shader computes quad derivatives before sampling the per-pixel texture. Both gradient cases passed on WARP and the RTX 4060 Laptop with the debug layer enabled. Run `ctest --preset dev --no-tests=error -R '^native_fixed_mesh57_(pixel_heap|waterfall(_fine)?)_(warp|hardware)$'` to reproduce.

## Exact state records

Set `X2W_SHADER_CACHE` to the request cache and `X2W_MESH57_CAPTURE_STATE=1` to retain the fixed creation state when a matching root consumes a mesh pipeline. A `req_<variant>.s57` file contains the little-endian 64-bit stage/root request key, three little-endian 32-bit lengths, and the corresponding byte spans for the normalized descriptor, register block A and register block B. The stand-in clears only the captured root and span pointer fields at DWORD-pair indices 0, 1, 3, 5 and 13. It preserves the lengths and other fields. The variant key extends the existing stage/root key with those three length-prefixed spans. Distinct fixed states can coexist for one stage pair. The fixture overwrites both blocks after creation and checks their archived bytes for accepted and refused variants.

Native preparation reads a separately reviewed `<variant>.mesh57-state` profile from the cache. Its versioned Windows x64 header is defined by `Mesh57Profile` in `src/startup_stubs/mesh57_profile.h`. The header supplies the MS entry layout and desktop fixed-function state. The file ends with the exact descriptor, register blocks, serialized root and both guest programs. The stand-in compares every byte before building a PSO. The fixture authors this profile from its own known inputs. A captured register block alone does not establish the desktop rasterizer state or MS entry layout.

The source-built `native_fixed_mesh57_constant_{warp,hardware}` tests exercise
one reviewed MS root-constant block with a deferred guest root. The version 3
sidecar stores the physical SGPR, guest root index and DWORD count after the
pixel entry. Slot 61 packs all written words into desktop MS b0 after checking
the creation root and the compiler's binding metadata. Changing guest root 4
(b5, space 2) changes the rendered green channel from 64 to 191 on the same
PSO. A missing word preserves the clear color. This profile supports one block
of up to eight words.

Run `ctest --preset dev --no-tests=error -R '^native_fixed_mesh57_buffers_'`
for the version 4 fixed mesh raw SRV fixture. One slot 61 command reads root
table 0 at byte 2848, root table 1 at bytes 32 and 96, a root 3 byte buffer
at offset 4, and the root 4 constant. Each contributes distinct bits to an
exported value. The expected RGBA at (2,0) is (255,62,255,255). A four-byte
view for an eight-byte span, a readable CPU address outside a GPU allocation,
and a missing root 4 word each preserve the clear color. Static tables share
immutable descriptor pages with other recorded commands on the same desktop
allocator; its queue fences govern page reuse.

The version 5 mesh profile enables one directly indexed R32_UINT buffer read
alongside the static bindings. The compiler supplies the root word and mask.
Run `ctest --preset dev --no-tests=error -R '^native_fixed_mesh57_uint_heap_'`.
The fixture uses two different first elements in one GPU buffer. Selectors
147 and 151 yield RGBA (64,62,64,64) and (191,62,191,191). Replacing view
147 with the second view yields the latter color; shortening the view to one
element yields (0,62,0,0) when reading element 1. A float SRV, absent slot
and out-of-range selector preserve the clear color. The guest source heap
and resource are released after recording and before submission.

The version 6 mesh profile also binds pixel texture reads from the directly
indexed guest resource heap. An eight-DWORD descriptor load must select a
32-byte-aligned slot within a proven interval. At recording time, each slot
in that interval must contain a registered float or normalized texture SRV
of the expected dimension, or an explicit typed null. The private desktop
heap retains the selected views for the recording and reserves separate
slots for the mesh-stage buffer selector and static bindings.

Run `ctest --preset dev --no-tests=error -R '^native_fixed_mesh57_pixel_heap_'`.
With the same mesh and pixel programs, changing root 4 selects guest slot 32
or 36. The source-built test reads back red or green respectively. Both SRVs
expose mip 1 and both textures have a blue mip 0. A typed null reads as
transparent black. An unregistered slot or an array view in the Texture2D
interval preserves the opaque clear color.

For offline fixed mesh pixel translation, `tools/startup/mesh57_pixel_entry.py`
takes a `req_<key>.ps57`, its creation archive `req_<variant>.s57`, and a
reviewed `<variant>.mesh57-state`. It checks the two content keys and compares
all six input spans before printing the `meshps1` line for the offline
compiler. The enable mask, address mask and wave size come from the reviewed
pixel entry. A creation archive alone preserves the two register blocks but
does not assign hardware register meanings to their offsets. Reproduce the
source-built positive and altered-stage controls with
`ctest --preset dev --no-tests=error -R '^native_direct_mesh$'`.
