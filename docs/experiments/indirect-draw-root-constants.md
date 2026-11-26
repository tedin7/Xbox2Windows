# Indirect draws with root constants

The device shim handles CreateCommandSignature at slot 41. It returns a
placeholder for the console argument type `0x69` before it calls desktop D3D12.
ExecuteIndirect at list slot 61 counts these calls under
`placeholder-signature` and logs `missing or placeholder signature`. The
record layout for `0x69` is unverified.

Desktop D3D12 already creates standard command signatures with a matching
root signature. Before this change, the shim classified a sequence of root
constants followed by Dispatch, but refused the same sequence followed by
Draw at execution with `requires a supported signature from this device`.

Set `X2W_INDIRECT_FULL_SIGNATURE=1` to enable root constants followed by Draw.
Each constant argument supplies all words for one parameter. The complete
signature supplies every root parameter. The draw uses a desktop graphics
pipeline, the matching desktop root signature and triangle-list topology.
The existing single-command limit and no-count-buffer rule apply. Vertex
buffer views, index buffer views and root descriptors contain GPU addresses
and require a separate address translation design. Type `0x69` still uses
the placeholder path.

The synthetic WARP fixture runs two draws with the same command signature
and different GPU-written root constants. It checks argument words, guard
words, every pixel and the indirect-call census. The switch-off control
checks that the same GPU-produced records fail the draw's pixel reference,
leave the target clear and generate two refusals.

```powershell
ctest --preset dev --no-tests=error -R '^startup_indirect_full_signature(_off)?_warp$' -V
```

The WARP runs require verification on Windows.

[D3D12 command signature and argument layout](https://learn.microsoft.com/en-us/windows/win32/direct3d12/indirect-drawing)
