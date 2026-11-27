# Missing buffer inputs in translated draws

The default direct draw path binds a null raw SRV with one DWORD in its
descriptor. Enable `X2W_DRAW_NULL_ZERO=1` to give a missing address real,
zero-filled backing. The `x2w-draw zero input` line identifies the translation
key, translated input index, source type, bind slot, table byte offset, declared
size and bound size. The null-input census still counts the missing source.
An oversized declaration or allocation failure refuses the draw.

On x64 Windows, build the `startup_indirect_draw_indexed` fixture with the
project's Windows preset and run:

```powershell
ctest --preset dev --no-tests=error -R '^startup_null_views_(census(_omit_index_control)?|empty_(zero|legacy_control))_warp$'
```

The synthetic WARP fixture compares each pixel against the CPU result for seven
zero DWORDs. It first binds a real address, then draws twice with only the
address removed. It checks that the two missing-address draws produce one
diagnostic for input 2. The legacy control disables the switch and must fail
the CPU comparison on the shader-visible byte count in both draws. Windows
execution is pending. The display writer's input 21 still needs a capture with
the switch enabled to identify its source slot and observe the render result.
