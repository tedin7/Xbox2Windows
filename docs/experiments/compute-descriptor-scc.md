# Compute scalar condition codes from descriptor words

The compute translator retries a targeted SCC refusal with descriptor-word and
carry-in lowering enabled. The original translation path still handles
successful shaders. The retry can use proven words from a fixed descriptor
table, with a zero or unchanged pointer high word. The dispatch binder snapshots
each requested dword into the entry constants. It refuses unavailable table
ranges and synthetic view tokens. Arbitrary pointer values, unverified SCC, and
unproved descriptor rewrites remain unsupported.

The synthetic compute fixture covers four shifted adds, four complement forms,
AND, absolute difference, signed and unsigned add and subtract, carry-in and
borrow-in, both source alias orders, and the special destinations `m0`,
`null`, `vcc_lo` and `exec_lo`. The WARP test compares the scalar result and copied SCC against a CPU
reference for 12 operand sets. The mutated shader inverts every copied SCC and
must mismatch the original reference, then match the inverted reference on all
outputs. The runner supplies synthetic descriptor dwords through the translated
shader's entry constants. The manifest is checked in Python. The WARP run
exercises the translated shader and UAV readback, with entry binding supplied by
the standalone runner.

```sh
python -B tests/startup_gfx2hlsl_descriptor_scc_check.py
```

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_gfx2hlsl_descriptor_scc_(controls|warp)$'
```
