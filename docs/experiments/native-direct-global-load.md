# Root-pointer global loads in the direct C++ compute path (#2962)

The direct compute frontend (`src/gfx1013_compute_native.cpp`, `src/gfx1013_direct_compute.inl`) now lowers
`global_load_dword` in the `saddr` form, the form `tools/startup/gfx2hlsl.py` (`op_global_load_dword`) lowers for
the Python path. Before, the frontend refused it with `unsupported instruction global_load_dword` (measured with
`0xdc308000`).

## The form

ADDR = SADDR + zext(VADDR) + sext(OFFSET[11:0]) (RDNA 2 ISA, FLAT/GLOBAL addressing). SADDR must be the unadvanced
root address of a root SRV, with its matching high half: the same symbol and binding (`root_buffer_srv`) the
scalar root loads of this path already use. The load becomes a raw buffer load from that view.

- The 32-bit byte offset zext(VADDR) + OFFSET is guarded to [0, 2^32): a positive offset reads 0 when VADDR is
  above 2^32 - 1 - OFFSET, a negative one reads 0 when VADDR is below -OFFSET. Above the range, the unguarded
  32-bit sum would wrap into an in-bounds word of the view; the fixture shows this case. Below it, the wrapped value
  lies near 2^32, outside any view under 4 GiB, so that guard does not change a result here.
- The immediate offset must be a multiple of 4. The low two bits of the address are dropped, as for
  `buffer_load_dword` in this path, which assumes the DWORD alignment mode. `gfx2hlsl.py` instead refuses a VADDR it
  cannot prove 4-byte aligned; the console's `SH_MEM_CONFIG.ALIGNMENT_MODE` is not measured.
- `glc`, `slc` and `dlc` are accepted (compiled, not measured).
- The binding's minimum size grows to the immediate offset plus 4 bytes for a positive offset; other loads keep
  the size the scalar loads need.

Refused by name: the `off` form (no SADDR), the FLAT and scratch segments, other opcodes (stores,
atomics), the `lds` form, an SADDR that is not a root address, a replaced high half, and an unaligned immediate
offset. An SADDR advanced by scalar arithmetic is refused earlier, as an unbound numeric SGPR; `gfx2hlsl.py` accepts
a pointer advanced by a known literal. Graphics stages never decode the FLAT encoding and keep
`unsupported instruction` (not covered by the fixture).

## Limits

- Addresses outside the bound view read 0. On the console the same address reads whatever memory lies there:
  before the buffer for a negative byte offset, or past the end of the view. The translation then differs from
  the guest unless the runtime binds a view that covers every address the program reads. `gfx2hlsl.py` has the
  same limit. How the runtime sizes the view on `cpp-migration` was not checked here.

## Fixture

`tests/native_global_load_check.py <compile_dxil> <native_store_replay> <sdk bin dir>`
(`ctest --preset dev --no-tests=error -R '^native_global_load_hardware$' -V`). Every word was assembled with
LLVM 15.0.7: `clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`.

- Eleven refusals are checked by message, and a load with `glc slc dlc` compiles.
- `dxc -dumpbin` of the dispatch program's module must show six raw buffer loads (five global and one scalar root
  load), two upper-bound guards and one lower-bound guard.
- The dispatch (two groups of 32 lanes) reads the root SRV that `native_store_replay` fills with
  0x51aa0000 + dword index, sized to 64 dwords by a scalar load at byte 252. Each lane stores five words: offsets 0,
  16 and -8, a VADDR of 0xfffffff0 or above with offset 16 (guarded to 0), and an unaligned VADDR. References with
  the offset dropped or the upper guard dropped must differ from the readback.

The original single-DWORD change was measured on 27.11.2026, `cpp-migration` `e3670100`, Windows 10.0.26100.9711 (24H2): the refusals
and the module checks passed. The dispatch reported a skip (77) on the NVIDIA Quadro M1000M (driver 32.0.15.7371),
because `native_store_replay` needs shader model 6.6 with wave32. The readback is not measured. With the frontend
change reverted, the first refusal check fails on `unsupported instruction global_load_dword`.

## Multi-DWORD loads

`global_load_dwordx2`, `global_load_dwordx3`, and `global_load_dwordx4` use
the same root-address proof and write consecutive VGPRs under EXEC. The
lowering computes the address once before writing any destination, including
when the destination aliases VADDR. The last DWORD must also fit the 32-bit
view offset. The minimum view size accounts for the full transfer width.
Reads across the end of the bound view return zero for the out-of-range
components. The view must cover every guest address whose value is needed.

The same fixture hand-assembles the three wider forms. On `88cddb09` the
compiler refused all three at guest offset `+0x38`. On RTX 4070, driver
`32.0.16.2124` and Windows 11 build `26300.10075`, the updated fixture
checks the address/destination alias, the end of a 64-word view, negative
offsets with sparse EXEC, and positive overflow. Reversing the returned
registers changes 128 words for x2 and x3 and 256 for x4 across two groups.
The direct module requires shader model 6.6 and wave32.

The address and width follow the [AMD RDNA 2 ISA, §9.2](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
and [LLVM 21.1.0 GFX1030 FLAT syntax](https://releases.llvm.org/21.1.0/docs/AMDGPU/AMDGPUAsmGFX1030.html#flat),
release commit `3623fe661ae35c6c80ac221f14d85be76aa870f1` under Apache-2.0 WITH LLVM-exception.
