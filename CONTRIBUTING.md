# Contributing to Xbox2Windows

You can contribute a hardware result, a source-built failing case, a documentation fix or a patch from an x64 Windows PC. Our Xbox Series X case study supplies research questions. The shared regression suite runs with public tools and fixtures built from this repository.

Visit [GitHub Discussions](https://github.com/R0Y00/Xbox2Windows/discussions) and follow the [pinned newcomer guide](https://github.com/R0Y00/Xbox2Windows/discussions/75) to get started. For build or debugging help, post the commit, command and observed result in [Development Q&A](https://github.com/R0Y00/Xbox2Windows/discussions/categories/development-q-a).

Use [Xbox2Windows Dev on Discord](https://discord.gg/DMcmNtpd6R) to ask quick questions, find a reviewer or coordinate an independent test. Record claims, reproduction evidence, review conclusions and technical decisions in the corresponding GitHub issue or PR.

Keep public project wording title-neutral. Refer to the platform, component or fixture when describing a result.

## Find and claim a task

1. Search the [issue tracker](https://github.com/R0Y00/Xbox2Windows/issues) for the test name or observed behavior. Issues labeled **good first issue** have a small first step. **help wanted** marks work that can proceed independently.
2. For an existing task, comment with the part you plan to test and your environment. If you expect to change a shared interface, name it in your claim and link any issue or PR you depend on or overlap with. For example, two changes to the same call translator can share a fixture and agree which lands first. Multiple adapters can be tested in parallel. For a new investigation, open a research task with a question, reproducible inputs and a completion condition. Small fixes can start with a PR.
3. If another contributor takes over, leave the commit, last command, observed result and next unanswered question on the issue. A maintainer can use the assignee field to record ownership of a patch.

## Build and test

Hardware reporters can use the [standalone package](tools/gpu-check/README.md). Paste the generated Markdown into the [ongoing hardware thread](https://github.com/R0Y00/Xbox2Windows/issues/1). Maintainers verify each incoming result, link the source report in the README coverage table, and record repeated runs on the same adapter as follow-ups. To prepare a batch for the table, export the thread with `gh api --paginate repos/R0Y00/Xbox2Windows/issues/1/comments > out/issue-1-comments.json` and run `python -B tools/hardware_reports.py out/issue-1-comments.json`. It lists the reports that the README does not link yet with suggested cells, and notes where a result differs from the existing row.

The [first independent clean build](https://github.com/R0Y00/Xbox2Windows/issues/2#issuecomment-7063351821) records the toolchain and three smoke-test outcomes. If a clean build fails on your machine, open a [build bug report](https://github.com/R0Y00/Xbox2Windows/issues/new?template=bug_report.yml) with the commit, toolchain, first failing command and diagnostic. Link a feedback-driven fix to its report and ask the reporter to verify the change.

Start with the [first build](README.md#first-build). Create the Python environment before configuring to include the script-path host and Python-driven script, texture, shader and PE checks. List the tests registered in your environment with `ctest --preset dev -N`, then run the closest group from the [experiment table](README.md#choose-an-experiment). Use `--no-tests=error` when checking a new filter so an empty selection fails. Run GPU tests serially when verifying results on a local adapter or WARP. Parallel runs compete for the same adapter or host resources and can time out; rerun such a test by itself. Pure-Python checks can run in parallel.

| Location | What to work on |
|---|---|
| `src/` | Windows runtime implementations and their observable behavior. |
| `tests/` | C++, HLSL and generated-input fixtures. Tests ending in `_warp` use the software adapter. Those ending in `_hardware` use a physical GPU. |
| `tools/` | Build setup and offline metadata tools. The PE fixtures live in `tests/intake/`. |
| `docs/experiments.md` | Older measured results, assumptions and reproduction commands. New notes go in `docs/experiments/<topic>.md`. |
| `tests/cmake/` | One `<family>.cmake` file per test family or pull request; [CMakeLists.txt](CMakeLists.txt) includes each. See [its README](tests/cmake/README.md). |
| `tests/checks/` | Text files that add Python checks to the Linux CI jobs without editing the workflow. See [its README](tests/checks/README.md). |

On pull requests, `Build and test on Windows` builds the smoke-test targets with `cmake --build --preset pr` and runs `ctest --preset pr --no-tests=error`. The preset selects tests for CPU translation, PE startup, memory, archive decoding and WARP rendering, including the SPIR-V bridge checks. The bridge checks can skip locally if their tools are missing; hosted CI installs the pinned tools and requires the WARP bridge checks to pass. For a strict local run, run [the pinned tool setup](tools/setup-spirv-tools.ps1) (`.\tools\setup-spirv-tools.ps1` in PowerShell) before configuring or reconfiguring, then set `$env:X2W_REQUIRE_SPIRV_TOOLS = '1'` before running CTest. Run the tests for your specific change locally and include their results in the PR.

Pull requests into `cpp-migration` also build `compile_dxil` and run `native_dxil_llvm_ident_controls` on Windows to check direct DXIL generation.

On `main` and manual runs, CI runs the full non-hardware suites in Debug and Release (/O2), generates the coverage chart and smoke-tests the standalone GPU package. To check a branch in both configurations before merge, select it under **Actions → Windows checks → Run workflow**. The Ubuntu jobs assemble gfx1013 fixtures with LLVM 15 and run the platform-independent Python checks. [CI workflow](.github/workflows/windows.yml)

Contributors supply physical GPU coverage. Record the commit, adapter and driver, Windows version, test names and debug-layer status. Include the feature tier or wave width when relevant. Report capability skips separately from passes and failures. [Non-Windows host guide](docs/non-windows-hosts.md)

## Share a reproducible finding

To register a new translated CPU fixture for the coverage dashboard, put
`; coverage: source cpu_your_test` on the first line of its MASM file. Name the
CTest that actually compares the translation. For macro-generated instructions,
use `; coverage: record cpu_your_test` and have the test emit the decoded forms
through `FORM_COVERAGE_RECORD`. Existing registrations still work.

To find instruction forms without a passing fixture in a measured run, use its
CTest JUnit file and matching records directory:

```powershell
python tools/form_coverage.py results.xml --records out/build/dev/form_coverage --gaps gfx1013
```

The TSV lists one row per uncovered reference form, sorted by group and mnemonic.
Choose `gfx1013`, `gfx1030`, or `x64_zen2`. A `skipped` row has evidence from a
skipped fixture. An `observed_without_pass` row appears in fixture evidence but
has no passing or skipped fixture. An `unobserved` row has no matching evidence
in the supplied measurement. A partial run can therefore list forms exercised
elsewhere. Check current fixtures and open PRs before choosing a form to work on.
Use the records directory of the preset that produced the JUnit file, for example
`out/build/release/form_coverage` for a `release` run. The tool exits with status
2 if that directory does not exist, and warns on stderr if none of its records
was written during the JUnit run, since no record-mode or shader form then counts.

Use original source, synthetic inputs or a public sample with its revision and license. Store local inputs in `game-files/` and generated output in `out/`. Keep game files, distribution links, SDK redistributables and dumps out of issues, PRs and Git. Remove personal paths and credentials from logs.

State what the test measured and which behavior remains a hypothesis. For example, a BVH report can identify the first differing ray and give the CPU, shader and DXR hit distances. If several primitives coincide, describe which IDs are valid. Claims about the inspected build should identify their source. A passing fixture supports the behavior under test; full-game startup requires its own execution evidence.

For findings from a local installation, share a limited observation such as a count or a comparison of two verified values. Describe how the values were checked. For a replay, identify who supplied the input event. For a decoder, say whether it verified a prefix, a full block or rendered pixels. Provide a source-built fixture where possible so other contributors can reproduce the check.

## Submit and review a patch

Create a topic branch from current `main` in your fork. In the PR description, say which task or milestone the change advances, or which independent problem it solves. Link related issues with `Refs #123` and name any PR that must land first. Use `Fixes #123` when the patch meets that issue's completion condition. Keep the PR focused on one reviewable behavior and include the exact test command and result. Submit original contributions under the [MIT License](LICENSE) and preserve applicable third-party notices.

Use `<area>: <summary>` for new pull request titles. Keep the area lowercase and write the summary in the imperative. The areas are `cpu`, `gpu`, `shader`, `ffx`, `textures`, `script`, `pe`, `runtime`, `tooling`, `docs`, `ci`, `build` and `meta`. For example:

```text
cpu: reject branches into instruction interiors
bcpack: add a stride-6 lane diagnostic
docs: add a contributor guide for non-Windows hosts
```

Do not rename existing pull requests or issues. Issue titles remain free-form; the issue forms and labels carry the classification. A title check is not enforced by a bot in this change.

When a PR changes a public capability or build requirement, update the relevant README row or first-build step in that PR. Put reproduction details in `docs/experiments.md`. Link an existing issue for unresolved behavior, or open one if another contributor can test it. The release tag identifies the exact standalone package a hardware reporter used.

### Who reviews and merges a PR?

R0Y00 owns the repository and currently serves as Maintainer. Route each PR by the behavior it changes.

| Change | Who reviews it | What to check |
|---|---|---|
| CPU translation or PE loading | CPU Reviewer | Instruction semantics, calling conventions, loader behavior, and comparisons with the original fixture. |
| Shader translation or GPU compatibility | GPU Reviewer | Generated DXIL, D3D12 behavior, and relevant WARP or hardware results. Check that capability skips are reported separately. |
| A shared CPU/GPU interface | Both reviewers, each on their side of the interface | Verify that the producer and consumer agree. Bring changes to the interface contract, compatibility guarantees, or overall architecture to the owner for a decision. |
| Documentation, hardware reports, or independent build fixes | A Maintainer familiar with the change | Check the source, original log, or reproduction command as appropriate. |

To accept a review invitation, reply on the PR. Reproduce the smallest relevant case, check the origin of new fixtures, and compare behavior before and after the change. Give your verdict and the commit you reviewed. If you request a change, follow up on the author's next revision and say whether it addresses the finding. While an invitation is pending, the Maintainer can review or find another reviewer. Independent PRs continue in parallel.

The Maintainer uses the reviewer's verdict once it covers the current head and all review conversations are resolved. Merge through the PR when the required `Build and test on Windows` check passes. If review is complete and the check is still running, enable GitHub auto-merge. Record remaining follow-up work in a linked issue.

The [code of conduct](CODE_OF_CONDUCT.md) applies to issues and reviews. Use the [security reporting instructions](SECURITY.md) for a vulnerability in project code or release tooling.
