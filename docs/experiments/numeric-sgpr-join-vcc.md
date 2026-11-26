# Numeric SGPR joins and unaligned VCC sources

A forward branch join can materialize an unchanged buffer descriptor word 3
through the existing `descword3` entry constant. A numeric consumer uses the
scalar variable populated on the incoming edges. Ordinary runtime definitions
keep their existing assignments.

For straight-line compute with a known wave size, an unaligned VCC source
with a readable low word supplies the logical lane bit. Wave32 uses the low
word. Wave64 selects the low or high word for the corresponding half of the
wave. A descriptor-only copy can also preserve both symbolic words for a
later buffer access. Numeric reads still require exact values from the
translator or runtime bindings.

```powershell
ctest --preset dev --no-tests=error -R '^(startup_compute_numeric_join_vcc_(controls|warp)|startup_gfx2hlsl_desc_vcc(_warp)?)$'
```

The compute fixtures exercise both branch outcomes and wave widths. The
graphics fixture uses a buffer descriptor with its upper words copied through
an odd SGPR source. Each WARP test compares readback against a CPU reference
and requires a targeted HLSL mutation to produce a mismatch. Python exercises
translation and builds the readback inputs. Windows readback is pending.
