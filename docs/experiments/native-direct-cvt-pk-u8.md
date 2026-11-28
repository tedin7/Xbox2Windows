# v_cvt_pk_u8_f32 in the direct C++ compute path (#2997)

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers `v_cvt_pk_u8_f32`, which it refused with
`direct DXIL lowering for v_cvt_pk_u8_f32`. It computes
D = (S2 & ~(0xff << 8 * S1[1:0])) | (flt32_to_uint8(S0) << 8 * S1[1:0]).

The conversion is the one `gfx2hlsl.py` (`op_v_cvt_pk_u8_f32`) uses, measured on RDNA 2 hardware
(`docs/experiments.md`, `tools/rdna_cvt_pk_u8.py`): round to nearest even, saturate to [0, 255]. NaN, both zeros and
every negative value give 0; +inf gives 255.

The lowering rounds the binary32 word with integer operations. The original DXIL
`FMin` / `Round_ne` / `fptoui` sequence returned 1 for literal 0.5 on multiple
NVIDIA adapters ([#3052](https://github.com/R0Y00/Xbox2Windows/issues/3052)).
The same issue reports 0 when 0.5 is computed at runtime. The cause inside the
driver remains unverified.

The unsigned test `(bits - 1) < 0x7f800000` accepts positive subnormals through
+Inf. Values above 255 are bounded to 255, and inputs that must produce zero
are bounded to 0.5, which rounds to zero. With the input in [0.5, 255], the
number of discarded significand bits is in [16, 24]. The lowering adds half
the discarded range minus one, plus the least significant retained bit,
before shifting. An exact tie rounds up only when the retained integer is
odd. The resulting byte is merged into S2.

Source and output modifiers stay refused (`direct ALU operand modifiers`), as `gfx2hlsl.py` refuses them.

## Fixture

`tests/native_cvt_pk_u8_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_cvt_pk_u8_' -V`).

Sixteen cases set S0 and S1 by `v_mov` literals over S2 = 0xa5a5a5a5, and every lane of two groups stores the
results. The cases cover:
- the ties 0.5, 1.5, 2.5 and 254.5;
- the bound 255.5, 1e30 and +inf;
- -0.0, -1.0 and -inf;
- a quiet NaN;
- the smallest subnormal;
- 127.4999, 0.0 and 255.0;
- all four byte selects, plus a select of 6 that must use only its low two bits.

The VOP3 word (opcode 0x15e) was assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013
-filetype asm -show-encoding`).

The second dispatch uses the lane and group to generate every integer from
0 to 255, its upper half tie and both adjacent binary32 inputs. Each dispatch
compares 1,024 packed results and all guard and untouched words.

The controls:
- compile both programs and check with `dxc -dumpbin` that neither module emits
  `Round_ne` (DXIL opcode 26) or `fptoui`;
- refuse it with a negated S0;
- check that three references differ from the expected words: rounding half up, truncating, and letting negatives
  through.

The reference agrees with `model_rne_sat` of `tools/rdna_cvt_pk_u8.py` for all sixteen inputs (checked once while
writing the fixture). Fourteen of the sixteen inputs are also rows of the measured table
`tests/data/rdna_cvt_pk_u8_measured.tsv`; 1e30 and 127.4999 rely on the model only.

The original compile-only check passed on `cpp-migration` `4ff5e36d` plus #2998
on Windows build 26100.9711. Its readback skipped on the NVIDIA Quadro M1000M
(32.0.15.7371) because the replay requires shader model 6.6 with wave32.
