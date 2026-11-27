# VCC words in an inline pixel sampler

The pixel translator can use constant `s106`/`s107` as the last two words of
`s[104:107]` when every path to the sample supplies the same four constants.
It refuses a whole-word constant used as a lane mask, an unproved path, and an
ambiguous 64-bit literal. A saved VCC spill is elided only when its vector
register is unused. Other spill and sampler restrictions still apply.

The controls cover a saved mask, a dead two-word spill, a comparison sampler,
branch joins, and the manifest. On Linux x86-64, the Python controls passed
with `nice -n 15 python3 -B tests/startup_image_sample_lod_check.py` on the
change based on `a08c038e2`. They include a filter mutation that changes the
CPU reference result. The sibling VCC and scalar-load controls also passed.

The existing Windows tests `startup_image_sample_lod_warp` and
`startup_image_sample_lod_hardware` exercise filter and address modes against
a CPU reference. They were not run for this change on the Linux host; no GPU
or full-game execution is established by the Python controls.
