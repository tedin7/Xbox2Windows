# Permlane16 and permlanex16 in direct C++ compute

Issue #3021 lowers `v_permlane16_b32` (VOP3A opcode 0x377) and
`v_permlanex16_b32` (0x378). S0 and the destination are VGPRs. The uniform
selectors are SGPRs, inline constants, or a trailing 32-bit literal. S1 and
S2 can each reference the same literal word. Selectors refuse `vcc_lo` and,
on the wave64 path, `vcc_hi`. In wave32, `vcc_hi` is ordinary scalar data.

S1 packs the selectors for row lanes 0 through 7 and S2 for lanes 8 through
15. Each lane extracts its four-bit selector at runtime. Permlane16 reads the
selected lane in its own row and permlanex16 reads from the paired row in the
same physical wave32. Guest wave64 uses two physical wave32s, with s126 and
s127 supplying guest EXEC for the lower and upper halves respectively.

VOP3A op_sel bits 0 and 1 encode FI and BOUND_CTRL. By analogy with DPP, FI
allows reading a source whose guest EXEC bit is clear. Without FI, an
inactive source supplies zero with BOUND_CTRL or preserves the destination
otherwise. Destination writes obey guest EXEC. Other modifier bits and
nonuniform selectors are refused. Thread counts must be divisible by 16 for
permlane16 and by 32 for permlanex16.

`native_permlane_controls` covers the literal form, six refusals, and
compilation of a split-wave64 container with distinct EXEC halves.
`native_permlane_hardware` uses sparse guest EXEC in 64-thread wave32 groups.
It compares lane-dependent selectors, including a shared literal in S1 and
S2, and inline zero against a CPU reference, including every sentinel word
outside the two groups' stores.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_permlane_(controls|hardware)$'
```

On Windows 11 with an NVIDIA GeForce RTX 5080 and DXC from the Windows SDK 10.0.26100, on `main` (`6057071b`)
plus this branch, `native_permlane_controls` and `native_permlane_hardware` passed. The six original readbacks matched a
CPU reference in all 2,064 words. Three manual mutants each fail the hardware check: reading the same row for
`x16`, swapping S1 and S2, and ignoring BOUND_CTRL. The inactive-source rule is assumed from DPP; the readback
checks the lowering against that model, not RDNA2 itself. On `main`, the wave64 branch of `lower_permlane`
is compiled by the container control. Slot 67 requests use the wave32 model until #3006 lands, so readback
exercises wave32 only.

On `cpp-migration` (`999e17cd`) plus #3010, a local capture of 2,689 compute requests (counts and instruction
shape only) had 47 requests stop first at one of the two ops. With this change, 34 of them compile and none that compiled before stops: 393 to 427
compiled. The other 13 stop later, 7 of them at "direct scalar move destination". All 47 have the wave32 bit of
#2999 clear; without #3006 they still compile on the wave32 model.
