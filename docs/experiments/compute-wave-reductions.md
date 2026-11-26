# Compute wave32 reductions

The translator recognizes a straight-line wave32 chain that saves EXEC, enables
all lanes, selects the reduction identity for inactive lanes with
`v_cndmask_b32`, reduces both 16-lane rows, restores EXEC and reads lane 31.
The fill can write a separate VGPR or overwrite its input. The identity can
use an inline immediate or a literal word.

The lowering supports unsigned sum, minimum, maximum, and bitwise OR, AND and
XOR. It uses a wave intrinsic when a host wave contains all 32 threads of one
guest wave. Otherwise it reads the 32 identity-filled words from group memory
using the guest's flattened thread index. The group barriers require
straight-line control flow. Groups must contain whole guest waves. The
permlane scratch register must be dead after the chain.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_wave_reduce_cs_(controls|warp_(dispatch|mutant))$' -V
```

The fixture uses two guest waves with different inputs and an inactive lane at
each end. The CPU reference checks the scalar result, the final vector result
in every lane, and a write predicated by the restored EXEC. The GPU controls
include inactive lanes in the reduction and must produce the first readback
mismatch predicted by the CPU. The WARP tests also force the group-memory path.

On Linux, `python -B tests/startup_gfx2hlsl_wave_reduce_cs_check.py` passed
18 synthetic forms and six refusal controls. Against parent revision
`e8015da7a`, the same forms produced 12 identity-fill refusals and six
pixel-stage refusals. Generated HLSL matched byte for byte across 42 existing
compute and pixel fixtures. The WARP readback tests require Windows.
