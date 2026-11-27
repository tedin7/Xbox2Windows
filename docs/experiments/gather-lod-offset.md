# Explicit-LOD gather with an offset

The startup translator supports 2D `image_gather4_l_o` with either contiguous
or NSA address registers and a validated constant sampler. Coordinates and the
signed six-bit texel offset may be computed at runtime. The float footprint can
select a neighbouring texel within about 2^-8 of a texel centre, as with the
other Load-based gather lowerings. The LOD and mip selection follow the
existing explicit-LOD gather.

The runtime can forward application resources into shader resource views and
exposes reserved-resource creation. Consequently this experiment does not assume
that every texture access is fully resident, and TFE remains refused. LWE and
R128 remain refused. No buffer TFE or optional sample forms were added.

Linux, Python 3, revision based on `a08c038e2d42` with this change:
`nice -n 15 python3 -B tests/startup_compute_gather_l_o_check.py` passes the
translation, negative and CPU reference controls. WARP readback is registered
but was not run on this host. It exercises the selected mip and offset; the
reference controls distinguish both ignoring the offset and forcing mip zero.
This fixture does not establish full-program startup.
