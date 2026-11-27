# Patch manifests: schema, loader and hash gate

A patch manifest describes adjustments to a source-translated build without changing the input
file on disk. This first step defines the format and the loader, `tools/patch_manifest.py`.
Nothing applies a patch yet. Memory patches at load time, data patches in the file layer and
translator hooks will each come in their own change ([#3037](https://github.com/R0Y00/Xbox2Windows/issues/3037)).

## Format

```toml
[target]
input_sha256 = "<sha256 of the exact input file>"

[[patch]]
id = "p001"
kind = "memory"
why = "<issue or root cause>"
remove_when = "<condition under which the patch can go>"
rva = 0x1010                      # first byte to replace
expect_sha256 = "<sha256 of the original bytes at rva, as many as the replacement>"
bytes = "90c3"                    # the replacement, lowercase hex
```

- **Hash gate:** the manifest applies only to the file whose SHA-256 equals `input_sha256`. Any other input is refused with `manifest is for another input`.
- **No partial apply:** the whole manifest is validated, and the input hash compared, before the loader returns anything. One bad patch refuses the manifest.
- **Every patch** needs a unique `id`, a `kind`, and non-empty `why` and `remove_when`.
- **Strict keys:** unknown keys and missing keys are refused at every level, so a misspelled key fails instead of being ignored.
- **Kinds:** `kind` must be a string, and only `memory` is defined. `hook` and `data` are refused as `not supported` until their stages land.
- **Memory patches:** `bytes` must be non-empty, and every byte it replaces needs an RVA below 2^32: a 1-byte patch at `0xffffffff` is accepted, a 2-byte one is not. Two memory patches may not overlap. Whether the span lies in a mapped section is checked later, by the runtime stage.
- **Original bytes are never stored:** `expect_sha256` is a hash of the bytes the patch replaces. The loader only checks its shape; the memory stage will compare it with the mapped image.
- **Reporting:** the loader returns `manifest_sha256`, the SHA-256 of the manifest file's bytes, with the patch ids, for run reports to record.

Input-specific manifests stay local under `game-files/patches/`, which is gitignored. Do not
publish them or the hashes in them.

## Check

```bash
./out/pe-env/bin/python -B tests/patch_manifest_check.py
python -B tools/patch_manifest.py MANIFEST INPUT
```

On Windows: `ctest --preset dev --no-tests=error -R '^patch_manifest_controls$' -V`.

The check builds a synthetic PE32+ at run time with `build_pe32_plus` from
`tests/call_evidence_check.py`. Its manifest replaces two bytes at the start of the `.rdata` section.

- **Bound input:** the manifest loads, and the CLI prints the manifest hash and `p001`.
- **Wrong input:** the same image with its last byte flipped makes the CLI exit with status 1 and `manifest is for another input`, with nothing on stdout.
- **Malformed manifests:** each of these is refused with its own message: a misspelled key, kind `hook`, a list or inline-table `kind`, a missing `why`, a blank `remove_when`, a duplicate id, overlapping patches, odd-length `bytes`, a boolean `rva`, a 2-byte patch at `0xffffffff`, an upper-case `input_sha256`, no patches, and a TOML syntax error.
- **Edge of the RVA range:** a 1-byte patch at `0xffffffff` loads.
- **Wrong-type `kind` through the CLI:** exit status 1 with a `patch manifest:` message, and no traceback.
- **Mutations:**

  | Change | Result |
  |---|---|
  | Input hash comparison removed | The wrong-input test fails. |
  | Overlap check removed | The overlap case fails. |
  | String check on `kind` removed | Both wrong-type cases raise `TypeError`, and the CLI prints a traceback. |
  | Span check removed | The 2-byte patch at `0xffffffff` is accepted. |
  | Span check off by one (`>=`) | The 1-byte case at `0xffffffff` is refused. |

Measured on 27.11.2026 on Windows 11 Pro with Python 3.14: 3 tests, all passed. The `python-linux`
CI job ran the first version of the check with Python 3.12 and passed.
