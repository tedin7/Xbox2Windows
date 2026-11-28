# Console BVH node model and witness set

Refs [#1759](https://github.com/R0Y00/Xbox2Windows/issues/1759),
[#2413](https://github.com/R0Y00/Xbox2Windows/issues/2413) and
[#2603](https://github.com/R0Y00/Xbox2Windows/pull/2603). This note covers the
pure-Python node model in `tests/gfx1030_bvh_box/bvh_model.py` and the
model-only witness set in `tests/gfx1030_bvh_box/check.py --model-only`. The
HIP kernel, the gfx1030 code object and the shader translator are unchanged.

## Node formats

A node pointer has its kind in the low three bits. The rest addresses a
64-byte node as `(pointer >> 3) << 6`. The 32-bit form adds that offset to a
40-bit base held in descriptor DWORD 0 and the low byte of DWORD 1, in
256-byte units. The 64-bit form uses the absolute pointer.

| Kind | Node | Bytes | Layout |
|---|---|---|---|
| 0-3 | triangle 0-3 | 64 | five vertices, three f32 each, bytes 0-59; ID DWORD at byte 60 |
| 4 | box, f16 bounds | 64 | four child pointers, then four boxes of six f16 from byte 16 |
| 5 | box, f32 bounds | 128 | four child pointers, then four boxes of six f32 from byte 16; bytes 112-127 unused |
| 6, 7 | instance, procedural | - | not read by this instruction |

`TRIANGLE_VERTICES` is `((0, 1, 2), (1, 3, 2), (2, 3, 4), (2, 4, 0))`.
Triangle `k` uses those three node vertices. The ID byte for triangle `k`
selects the vertices weighted by I and J in bits 1:0 and 3:2; the remaining
vertex is `3 - I - J`. Bits 7:4 of the ID byte are ignored. A malformed ID
that names the fourth map nibble reads node vertex 0, as the arithmetic
lands.

Descriptor DWORD 1 bit 31 enables nearest-first box sorting. DWORD 3 bit 24
selects triangle return mode 1. DWORD 3 bits 31:28 hold the resource type,
`0x8`.

## Results

A box node returns four child pointers. A hit is included at its entry
distance; a miss or an absent child is `0xffffffff`. Sorting uses a strict
comparison, so equal entries keep their child order; misses sort last when
sorting is enabled.

Triangle return mode 1 is `{t_num, t_denom, I_num, J_num}`, all as f32. On a
miss it is `{+inf, 1.0, 0.0, 0.0}`. The instruction does not test the ray
extent for triangles. Return mode 0 is not modeled in this slice.

## Assumptions

- Arithmetic is binary32 at each step, rounding to nearest even. Python
  floats only carry the values; the result DWORDs are the f32 bit patterns.
- The inverse direction is an instruction input. The box test clamps it to
  `+-FLT_MAX` before multiplying, so an origin exactly on a face does not
  create `0 * infinity`.
- `fmin`/`fmax` return the non-NaN operand when one is NaN. A NaN minimum x
  marks an absent box. Other NaN box components are outside the defined
  input.
- A box is hit when `leave >= max(0, enter)` and `enter < extent`. An entry
  exactly at the extent is not a hit. A box containing the origin can be hit
  at a negative entry.
- The triangle test is the watertight Woop, Benthin and Wald formulation used
  by RADV. A ray exactly through an edge or vertex hits both triangles that
  share it. A NaN vertex, a zero determinant or a triangle behind the ray
  misses.
- A zero direction gives an infinite inverse; the triangle shear then
  produces NaN and the triangle misses.

These choices follow Mesa RADV's MIT-licensed software emulation and the
RDNA 2 ISA descriptor tables. They are a CPU model, not a native measurement:
hardware rounding, which of two equally near boxes is returned first, and the
exact edge case remain assumptions. No code is copied from the GPL-3.0
VIctory reference.

## Witness set

The model-only run has 315 cases: 231 node cases and 84 pointer-decoding
cases. The node cases are 66 f32 box, 46 f16 box and 119 triangle cases. The
expected four-DWORD results are literals in `check.py`; the runner does not
recompute them with the model. The existing five-ray FP32 box assertions in
`model_check()` still run before the witness set.

Coverage includes sorted and unsorted hits, misses, ties, extent clipping,
disabled and zero child pointers, inverted and infinite bounds, negative
zero, NaN bounds and vertices, zero directions, f16 rounding, subnormals and
65504, all six triangle vertex permutations for each kind, ID high bits,
malformed IDs, shared edges and vertices, t = 0, triangles behind the ray,
and degenerate triangles. Pointer cases cover every kind, nonzero and
high bases, 48-bit wrap and the absolute 64-bit form.

## Reproduce

```sh
python3 -B tests/gfx1030_bvh_box/check.py --model-only
```

The command needs no GPU, HIP library or code object. The native five-ray
case still needs LLVM/LLD 15 and a gfx1030 HIP device:

```sh
sh tests/gfx1030_bvh_box/build.sh
python3 -B tests/gfx1030_bvh_box/check.py --device 0
```

No game files, dumps or third-party source are part of this fixture.
