# TTPF manifest and DDS paths

The batch exporter rejects a manifest path that names a DDS requested by the
same batch or a supplied history or calibration input. Previously, choosing
`--indices 0 --manifest out/tex_00000.dds -o out`
could report a successful export and then replace that DDS with the JSON
manifest. A failed or skipped export could also lose a DDS preserved from a
previous run when the manifest was published.
Likewise, naming the history or calibration file as the manifest replaced the
input profile with an export-results array, preventing its reuse on the next run.

The path check runs after reading the pool count, before creating the output
directory or starting any export. It uses the existing resolved-path and file
identity checks. A manifest with the same filename in a different directory is
still accepted.
Profile collisions are rejected before loading the decoder tools.

The synthetic regression exercises successful, failed and skipped exports with
and without an existing DDS. All six cases fail with the old exporter because
it accepts the collision. The separate-directory control succeeds and verifies
the DDS bytes against the manifest checksum.
Two further cases verify that history and calibration inputs remain unchanged;
both fail with the old exporter because it accepts and overwrites the profile.

```powershell
python -B tests/ttpf_batch_export_check.py
python -B tests/ttpf_batch_export_cli_check.py
```

These checks use the mocked exporter and generated inputs; the optional local
Leonida integration test can skip when its tools are not installed. They make
no claim about image decoding correctness.
