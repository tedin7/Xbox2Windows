# v_cvt_pk_u16_u32 and v_cvt_pk_i16_i32 in the direct C++ compute path (#2995)

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers the two saturating 16-bit packs that it
refused with `direct DXIL lowering for <name>`. The semantics are the `gfx2hlsl.py` VOP3-table expressions. Each
saturates S0 and S1 to 16 bits and packs S0 into the low half and S1 into the high half:

- `v_cvt_pk_u16_u32`: min(S, 0xffff) of the unsigned values.
- `v_cvt_pk_i16_i32`: the signed values clamped to [-32768, 32767].

A nonzero unused SRC2 field is refused (`<name> with unused src2 field set`), as in `gfx2hlsl.py`, and so are the
op_sel bits (`<name> with op_sel`). Source and output modifiers stay refused like the other integer VOP3 forms.

As `docs/experiments.md` notes for the Python path, the RDNA 2 ISA text does not define the `u32_to_u16` and
`i32_to_i16` conversions. The saturation follows Mesa and the existing translators, and it is not measured on the
console.

## Fixture

`tests/native_cvt_pk16_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_cvt_pk16_' -V`). Eight operand pairs are set by `v_mov` literals and
converted by both instructions; each of 64 lanes stores the sixteen results. The pairs cross both bounds of both forms:
0, 0xffff, 0x10000, 0xffffffff, -32768, -32769, 32767, 32768, 0x7fffffff and 0x80000000 all occur. The VOP3 words
were assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).
The other words were checked with `tools/disasm_rdna2.py --hex`.

- The controls compile the program, refuse the same program with SRC2 = 3 or an op_sel bit, and check that references that truncate
  instead of saturating, or that apply the unsigned bound to the signed form, differ from the expected words. The
  reference agrees with the `gfx2hlsl.py` table expressions evaluated in Python for all eight pairs. This was checked
  once while writing the fixture.

Measured on 27.11.2026, `cpp-migration` `4ff5e36d` plus this change, Windows 10.0.26100.9711. The controls printed
`compiled; a nonzero src2 field and op_sel refused; the truncating and unsigned-bound references differ`. The readback printed
`requires shader model 6.6 with wave32` and reported a skip (77) on the NVIDIA Quadro M1000M (driver 32.0.15.7371,
debug layer unavailable). The readback is not measured.
