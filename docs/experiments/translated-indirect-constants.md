# Translated indirect compute constants

The synthetic WARP fixture copies a title record to a GPU buffer at a nonzero
offset. A translated compute shader writes eight triples of A, B and thread ID
across four groups. A readback buffer captures the full output, including the
unwritten sentinel region.

The fixture checks two record constants, one record constant with a separately
bound B, and a padded record with an unused word between A and B. The padded
case executes the same signature twice in one recording. The default-off,
missing-B and resource-selection cases require a refusal and untouched output.
The title-order mutant keeps the shader and record fixed and swaps the manifest
entry destinations. The CPU reference must flag both constant words in all
eight triples.

```powershell
ctest --preset dev --no-tests=error -R '^startup_(slot67_indirect_(constants|dispatch)|command_signature|indirect_root_constants)_warp$' -V
```

The device-free oracle also checks the expected mismatch positions.

```bash
python3 -B tests/startup_slot67_indirect_dispatch_check.py mutation
```
