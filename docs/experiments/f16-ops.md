# Binary16 FMA, packed FMAC and LDEXP

The startup translator supports VOP3 `v_fma_f16` source and destination half selection,
VOP2 `v_pk_fmac_f16`, and VOP2/VOP3 `v_ldexp_f16`. The controls construct source-built
instruction streams and compare their expected results using exact rational binary16 arithmetic.
They also check a fused-rounding counterexample and half-selection counterexamples.

On Linux x86-64 (kernel 6.12.111), the worktree based on `a08c038e2` passed
`nice -n 15 python3 -B tests/startup_gfx2hlsl_f16_ops_check.py`. This generates the
compute shader but does not compile or dispatch it. On a configured Windows build, run
`ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_f16_ops_(controls|warp|hardware)$'`.
The WARP and hardware dispatches compare every result word against the independent CPU model;
return code 77 is a capability skip. Neither dispatch has been measured in this Linux environment.
