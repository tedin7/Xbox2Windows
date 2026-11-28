# BC4 typed format number

`texture_blocks.format_info("BC4_UNORM")` returns DXGI format 80, the public
`DXGI_FORMAT_BC4_UNORM` value. The old table returned 79, which names the typeless
parent. Both formats have eight-byte blocks, so the byte-row and volume tests
alone did not detect this metadata error.

The two existing format-table checks now expect 80 independently of the module's
table. With the old implementation each check failed with `79 != 80`; after the
correction all 17 tests pass on Windows with Python 3.14:

```powershell
python -B tests/texture_blocks_check.py
python -B tests/texture_bc_volume_check.py
```

This checks the format metadata and synthetic block rows. It does not establish
any change in rendered console textures or in the BCPack decoder.
