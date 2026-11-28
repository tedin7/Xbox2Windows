# Binary16 add, sub, subrev and mul keep the destination's high half

In the direct compute path (`src/gfx1013_direct_compute.inl`), the non-SDWA forms of `v_add_f16`, `v_sub_f16`,
`v_subrev_f16` and `v_mul_f16` wrote the rounded binary16 result to D[15:0] and zeroed D[31:16]. They now keep
D[31:16].

- **LLVM:** `GCNSubtarget::zeroesHigh16BitsOfDest` lists exactly these four operations, in both the VOP2 and VOP3 encodings. It states "On gfx10, all 16-bit instructions preserve the high bits" and returns false from GFX10 on.
- **The same path:** `v_cvt_f16_f32` already keeps the high half.
- **#669:** for the non-packed 16-bit integer operations, an inline-assembly measurement on RDNA 2 silicon found the high half kept.

No silicon measurement of these four f16 operations is known to me.

SDWA forms still place the result through their `dst_sel` and `dst_unused` fields, and are unchanged.

## Fixture

`tests/native_f16_alu_check.py model|ir|gpu` (`native_f16_alu_ir`, `native_f16_alu_hardware`). Every destination
already held `0xabcd` in its high half before the operation. The model, the IR interpreter and the readback now
require each stored word to be `0xabcd` in the high half and the binary16 result in the low half. A zeroed high half
is a rejected mutant, in place of the earlier mutant that rejected a kept one.

## Measured

On `cpp-migration` `999e17cd` plus this change, Windows 10 build 19045, NVIDIA GeForce RTX 3080, driver
32.0.15.9174:
- **Model:** 82 results match.
- **IR:** 82 results and 16 refusals match.
- **Readback:** all 164 words match with the high half kept.
- **Old lowering:** with the previous zeroing restored, both the IR check and the readback fail at the first stored word.

The readback used a local copy of `native_store_replay` linked with the app-local Agility SDK exports
(`D3D12SDKVersion` 620). The registered test reports a skip (77) on Windows 10, whose inbox runtime does not report
shader model 6.6.
