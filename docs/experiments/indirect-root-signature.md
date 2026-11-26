# Translated indirect draw root signatures

The translated DRAW and DRAW_INDEXED paths compare the graphics root
signature recorded on the executing list with the root used to translate the
pipeline. A separately created copy of the pipeline root has a different
wrapper and a different metadata pointer. The original check reports
`missing_guest_root` for that pair.

Set `X2W_INDIRECT_ROOT_FIX=1` alongside `X2W_TRANSLATED_INDIRECT_DRAW=1`
to accept separately created roots with identical captured serialized bytes.
The translator still reads the arguments of the executing list. The switch
defaults to off.

A graphics command signature containing a root constant or root descriptor
argument falls outside the supported translated graphics signature shapes.
In D3D12 those arguments update individual slots of the current root
signature. They do not select the root signature.

The fixture creates two guest roots from the same blob, creates the pipeline
with the first, and binds the second and its descriptor table to the draw
list. The GPU writes the indirect arguments. With the switch enabled, the
CPU reference expects all pixels to be red. With the switch unset, it
expects the black clear color and one `missing_guest_root` census entry.
Run on Windows with WARP and a compatible physical adapter.

```powershell
ctest --preset dev --no-tests=error -R '^startup_indirect_translated_root(_control)?_(warp|hardware)$'
```
