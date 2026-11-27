# DPP16 row shifts and rotates in direct C++ compute

Issue #3008 adds `row_shl:N`, `row_shr:N` and `row_ror:N` for N from 1 to 15
with full row and bank masks. A left shift reads from row lane r + N, a right
shift reads from r - N, and a rotate reads from (r - N) modulo 16. Each row
fits inside one physical wave32, including in a split guest wave64.

The source is usable when it is in the row and either FI is set or its guest
EXEC bit is set. For an unusable source, BOUND_CTRL supplies a zero source
word. With BOUND_CTRL clear, the destination keeps its old value. Destination
writes continue to obey guest EXEC. Row shifts and rotates require a thread
count divisible by 16. `row_share`, `row_xmask`, the pre-GFX10 controls
(`0x130..0x13c`, `0x142`, `0x143`) and partial row or bank masks stay refused.

`native_dpp_row_controls` checks a valid shift and four refusals covering
row_share:0, control 0x142, a partial row mask, and a 24-thread group.
`native_dpp_row_hardware` exercises sparse EXEC in two physical wave32s per
group. The reference checks all output words, including sentinel guards.
The fixtures cover moves with boundary shift and rotation amounts, integer
add with FI, and floating-point add and min with BOUND_CTRL.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_dpp_row_(controls|hardware)$'
```

On Windows 11 with an NVIDIA GeForce RTX 5080 and DXC from the Windows SDK
10.0.26100, on `cpp-migration` `999e17cd` plus this change, both checks passed:
16 GPU programs matched the CPU reference with all 2,064 output words, and the
four controls were refused. Three manual mutants each fail the hardware check:
rotating left instead of right (the shift 1 and 15 cases; `row_ror:8` is its
own inverse), treating an out-of-row source as usable, and ignoring BOUND_CTRL.

On a local capture of 2,689 compute requests (counts and instruction shape
only), 80 requests stopped first at "unsupported DPP lane selection": 62 with
`row_ror:8` and BOUND_CTRL, 18 with `row_shr:1` and FI. With this change, the
compiled count goes from 392 to 393 on `999e17cd`, and none that compiled before
stops. Of the other 79, 32 now stop first at `v_permlane16_b32`/`v_permlanex16_b32`
and 25 at "SDWA or DPP vector operands".
