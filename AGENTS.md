# AGENTS.md

This file is for coding agents working in Xbox2Windows. Read `CONTRIBUTING.md` and
`docs/experiments.md` before changing behavior. If this file conflicts with those
documents, they take precedence.

## Project

Xbox2Windows is a static-recompilation research project for Xbox Series X|S
(Scarlett) software on x64 Windows. The CPU experiments translate source-built
x64 functions from PE machine code through C++ to Windows DLLs. The shader
experiments translate selected gfx1013 programs to desktop DXIL. The repository
also contains Windows runtime shims, PE tooling and reproducible fixtures.

This is not a full emulator and not a playable port. Execution of the title's
CPU code, full-game startup and frame rate remain unverified. Do not claim
otherwise.

## Hard rules

- Never add game files, dumps, DLLs, EXEs, torrents, distribution links or
  credentials to the repository, issues or pull requests.
- Keep public project wording title-neutral, including the repository
  description and README. Describe results in terms of the platform, component
  or fixture tested. Do not claim that a game runs.
- Use only source-built fixtures, synthetic inputs or public samples with a
  known revision and license.
- Keep generated output under `out/` and local inputs under `game-files/`.
  Both directories are gitignored.
- Do not weaken or delete tests to make a change pass.
- One pull request should cover one reviewable behavior. Do not mix unrelated
  refactors, formatting changes or dependency updates.
- Original contributions are submitted under the MIT License. Preserve
  applicable third-party notices.

## Platforms

The full build, Windows runtime tests, D3D12 tests, MASM fixtures and GPU tests
require x64 Windows with Visual Studio's Desktop development with C++ workload,
the Windows SDK, CMake/Ninja, DXC and Python 3.11 or newer.

`dxc_compile_time_gvn` measures pass attribution through `-ftime-report`, which
arrived in DXC 1.7.2212 (December 2022). The 1.6 build that ships with Windows
SDK 10.0.22621 has no such flag, so that test reports a capability skip (77)
instead of failing.

The Ubuntu CI job assembles and translates gfx1013 fixtures with LLVM 15. The
pure-Python checks listed below also run on Linux and macOS. Do not stub D3D12,
MASM or `cl` to make the CMake configure succeed on another platform.

## Build on Windows

Install the Desktop development with C++ workload, including x64 MSVC, the C++
CMake tools with Ninja, the Windows SDK and DXC. From a PowerShell prompt at the
repository root:

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --no-tests=error -R 'desktop_root_signature|era_heap_allocation|offscreen_warp'
```

For the script, shader, texture and PE fixture tests, create the Python
environment before configuring:

```powershell
python -m venv out/pe-env
.\out\pe-env\Scripts\python.exe -m pip install -r requirements-dev.txt
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --no-tests=error
```

`CMakeLists.txt` uses `INTAKE_PYTHON` for the Python-driven tests. Its default is
`out/pe-env/Scripts/python.exe`.

## Tests

List the registered tests with `ctest --preset dev -N`. Use
`--no-tests=error` when checking a new filter so an empty selection fails.

| Area | Command |
|---|---|
| CPU translation | `ctest --preset dev --no-tests=error -R '^cpu_' -V` |
| Script states | `ctest --preset dev --no-tests=error -R '^script_(state_literals\|path_links)$' -V` |
| Textures | `ctest --preset dev --no-tests=error -R '^(bcpack_prefix_and_mip_tail\|bc7_tile_swizzle\|bc7_draw_warp\|texture_bc_volume)$' -V` |
| Shader translation | `ctest --preset dev -R '^pixel_aot_' -V` |
| D3D12 and ray tracing | `ctest --preset dev -R '^bvh_(differential\|wave64)_hardware$' -V` |
| Windows runtime | `ctest --preset dev -R 'era_heap\|desktop_memory_query'` |
| PE tooling | `ctest --preset dev -R 'static_exe_intake\|conditional_import_paths'` |

Tests ending in `_warp` use Microsoft WARP. Tests ending in `_hardware` use a
physical GPU and may return 77 (capability skip). A skip is not a pass and not a
failure; report it separately. GPU reports must identify the adapter, driver,
Windows build, commit and debug-layer status.

## Pure-Python checks

These checks do not need Windows, D3D12 or game files. From the repository
root, run the first two commands below to set up the environment, then choose
individual checks as needed.

The full Linux CI list is in the `Run Python checks` step of the `python-linux`
job in the [CI workflow](.github/workflows/windows.yml). To run it locally,
copy the step's `tools/run_checks_parallel.py` command with all its arguments
and replace the leading `python` with `./out/pe-env/bin/python`.
See the [Linux quick start](docs/non-windows-hosts.md#quick-start-on-linux)
for how skips affect the runner's exit status.

```bash
python3 -m venv out/pe-env
./out/pe-env/bin/python -m pip install -r requirements-dev.txt
./out/pe-env/bin/python -B tests/script_state_check.py
./out/pe-env/bin/python -B tests/resource_root_check.py
./out/pe-env/bin/python -B tests/pe_authentihash_check.py
./out/pe-env/bin/python -B tests/bcpack_check.py
./out/pe-env/bin/python -B tests/bcpack_symbols_check.py
./out/pe-env/bin/python -B tests/bcpack_tables_check.py
./out/pe-env/bin/python -B tests/bc7_blocks_check.py
./out/pe-env/bin/python -B tests/bc7_tiles_check.py
./out/pe-env/bin/python -B tests/gfx10_texture_roundtrip_check.py
./out/pe-env/bin/python -B tests/texture_bc_volume_check.py
./out/pe-env/bin/python -B tests/form_coverage_check.py
./out/pe-env/bin/python -B tests/resource_ranges_check.py
./out/pe-env/bin/python -B tests/ttpf_batch_export_check.py
./out/pe-env/bin/python -B tests/awc_metadata_check.py
./out/pe-env/bin/python -B tests/startup_audio_timeline_check.py
./out/pe-env/bin/python -B tests/sgd5_effects_check.py
./out/pe-env/bin/python -B tests/dxcontainer_part_offsets_check.py
./out/pe-env/bin/python -B tests/layout_chunks_check.py
./out/pe-env/bin/python -B tests/dependency_paths_check.py
./out/pe-env/bin/python -B tests/pe_package_preflight_check.py
./out/pe-env/bin/python -B tests/host_commit_check.py
./out/pe-env/bin/python -B tests/pe_package_ordinal_plan_check.py
./out/pe-env/bin/python -B tests/compat_report_check.py
./out/pe-env/bin/python -B tests/intake/paths_check.py
./out/pe-env/bin/python -B tests/intake/routing_check.py
./out/pe-env/bin/python -B tests/intake/import_forms_check.py
./out/pe-env/bin/python -B tests/shader_probe_check.py
./out/pe-env/bin/python -B tests/shader_census_check.py
./out/pe-env/bin/python -B tests/shader_corpus_report_check.py
./out/pe-env/bin/python -B tests/dxil_retarget_check.py
./out/pe-env/bin/python -B tests/rdna2_tools_check.py
./out/pe-env/bin/python -B tests/mesh_dynamic_geometry_check.py
./out/pe-env/bin/python -B tests/pso_readiness_check.py
./out/pe-env/bin/python -B tests/xmem_attribute_scan_check.py
./out/pe-env/bin/python -B tests/xgameruntime_table_check.py
./out/pe-env/bin/python -B tests/docs_links_check.py
./out/pe-env/bin/python -B tests/bidi_controls_check.py
# With no arguments, this tests result classification without Vulkan or DXC.
./out/pe-env/bin/python -B tests/pixel_denorm_vulkan_check.py
# With no arguments, this tests the Vulkan offscreen adapter without a device or DXC.
./out/pe-env/bin/python -B tests/vk_offscreen.py
./out/pe-env/bin/python -B tests/gpu_report_format_check.py
./out/pe-env/bin/python -B tests/hardware_reports_check.py
./out/pe-env/bin/python -B tests/rectlist_topology_check.py
./out/pe-env/bin/python -B tests/startup_sampler_heap_check.py
./out/pe-env/bin/python -B tests/startup_image_sample_l_check.py
./out/pe-env/bin/python -B tests/startup_image_sample_b_check.py
./out/pe-env/bin/python -B tests/startup_image_sample_lod_check.py
./out/pe-env/bin/python -B tests/startup_image_3d_check.py
./out/pe-env/bin/python -B tests/startup_image_load_mip_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_operand_count_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_scalar_arity_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_scalar_table_arity_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_mad_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_neg_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_interp_p2_hi_check.py
./out/pe-env/bin/python -B tests/shader_signed_zero_minmax_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_compute_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_compute_spill_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_compute_mask_spill_check.py
./out/pe-env/bin/python -B tests/startup_compute_image_store_check.py
./out/pe-env/bin/python -B tests/startup_compute_image_store_d16_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_heap_record_check.py
./out/pe-env/bin/python -B tests/startup_compute_store_format_xyzw_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_heap_word_check.py
./out/pe-env/bin/python -B tests/startup_compute_translate_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_swap_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_abs_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_quadmask_b64_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_ff0_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_bitset_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_flbit_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_ff1_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_s_quadmask_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_fma_mix_check.py
./out/pe-env/bin/python -B tests/startup_image_2d_array_check.py
./out/pe-env/bin/python -B tests/startup_image_sample_a16_check.py
./out/pe-env/bin/python -B tests/startup_image_load_d16_check.py
./out/pe-env/bin/python -B tests/startup_host_cpu_check.py
./out/pe-env/bin/python -B tests/startup_live_telemetry_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_capture_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_watch_check.py
./out/pe-env/bin/python -B tests/startup_translator_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_smem_offset_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sload_const_offset_check.py
./out/pe-env/bin/python -B tests/startup_archive_requests_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_refusals_check.py
./out/pe-env/bin/python -B tests/startup_compute_request_offline_check.py
./out/pe-env/bin/python -B tests/startup_shader_cache_rekey_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_scalar_m0_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_waitcnt_depctr_check.py
./out/pe-env/bin/python -B tests/startup_mesh57_skeleton_check.py
./out/pe-env/bin/python -B tests/startup_mesh57_buffer_read_check.py
./out/pe-env/bin/python -B tests/startup_mesh57_depth_only_check.py
./out/pe-env/bin/python -B tests/startup_mesh57_loop_refusal_check.py
./out/pe-env/bin/python -B tests/startup_mesh57_loop_check.py
./out/pe-env/bin/python -B tests/startup_mesh57_lane_spill_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_readfirstlane_uniform_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_loop_constants_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_join_symbols_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_readfirstlane_recompute_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_readfirstlane_load_recompute_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sop2_complement_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_pack_absdiff_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sdwa_cvt_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sdwa_cndmask_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sopk_accum_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_int_med3_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_cselect_mask_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_mask_overwrite_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_bfm_b64_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_mask_data_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_image_get_lod_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_add_sub_co_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_add_sub_co_ci_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_saveexec_same_reg_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_compare_dst_check.py
./out/pe-env/bin/python -B tests/startup_flat_param_check.py
./out/pe-env/bin/python -B tests/startup_build_rev_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_root_check.py
./out/pe-env/bin/python -B tests/startup_inline_sampler_check.py
./out/pe-env/bin/python -B tests/startup_buffer_address_check.py
./out/pe-env/bin/python -B tests/startup_buffer_format_word3_table_check.py
./out/pe-env/bin/python -B tests/startup_smem_address_check.py
./out/pe-env/bin/python -B tests/startup_stream_fetch_check.py
./out/pe-env/bin/python -B tests/startup_stream_interface_check.py
./out/pe-env/bin/python -B tests/compute_alu_cmp_ne_check.py
./out/pe-env/bin/python -B tests/compute_entry_layout_check.py
./out/pe-env/bin/python -B tests/compute_buffer_atomics_check.py
./out/pe-env/bin/python -B tests/compute_alu_bcnt64_check.py
./out/pe-env/bin/python -B tests/compute_load_format_check.py
./out/pe-env/bin/python -B tests/compute_heap_view_check.py
./out/pe-env/bin/python -B tests/compute_heap_store_check.py
./out/pe-env/bin/python -B tests/compute_desc_numeric_check.py controls
./out/pe-env/bin/python -B tests/compute_alu_lane_gaps_check.py
./out/pe-env/bin/python -B tests/startup_ngg_slice_check.py
./out/pe-env/bin/python -B tests/mesh_clip_cull_check.py
./out/pe-env/bin/python -B tests/mesh_dynamic_aot_check.py
./out/pe-env/bin/python -B tests/mesh_entry_check.py flat-scratch
./out/pe-env/bin/python -B tests/mesh_heap_sampler_check.py
./out/pe-env/bin/python -B tests/mesh_lane_spill_check.py
./out/pe-env/bin/python -B tests/mesh_join_descriptor_check.py
./out/pe-env/bin/python -B tests/mesh_loop_census_check.py
./out/pe-env/bin/python -B tests/startup_image_gather_lz_check.py
./out/pe-env/bin/python -B tests/startup_image_sample_c_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_cvt_alu_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_cvt_off_f32_i4_check.py
./out/pe-env/bin/python -B tests/startup_triage_run_check.py
./out/pe-env/bin/python -B tests/startup_run_ps1_check.py
./out/pe-env/bin/python -B tests/startup_run_sh_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_bit_alu_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sgpr_spill_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_spill_branches_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_shift64_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_vopc_sdwa_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_cmp_class_f16_sdwa_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_cmp16_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_f32_lg_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_f32_nan_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sopk_signext_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_scalar_brev_fold_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_const_copy_join_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_pack_b32_f16_check.py
./out/pe-env/bin/python -B tests/run_checks_parallel_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_handler_arity_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_getpc_refusal_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_image_arity_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sad_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_sad_hi_msad_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_cvt_pk_int16_check.py
./out/pe-env/bin/python -B tests/startup_gfx2hlsl_scalar_null_check.py
./out/pe-env/bin/python -B tests/startup_compute_global_load_check.py
```

`tests/pixel_aot_check.py binary` additionally needs LLVM with gfx1013 support
and the local `llvm-mc`, `llvm-objcopy` and `libLLVM` paths. CI uses LLVM 15;
newer versions may print a different `exp` spelling.

## Repository map

| Path | Purpose |
|---|---|
| `src/` | Windows runtime shims and shared shader helpers. |
| `tools/` | Offline translators, PE/image probes and inventory tools. |
| `tests/` | C++, HLSL, MASM and Python fixtures. `tests/cpu/` holds the PE fixtures. |
| `docs/experiments.md` | Older measured results, assumptions and reproduction commands. New notes go in `docs/experiments/<topic>.md`. |
| `tests/cmake/` | One `<family>.cmake` file per test family or pull request, included by `CMakeLists.txt` (see its README). |
| `tests/checks/` | Text files that add Python checks to the Linux CI jobs; see its README. |
| `game-files/` | Local inputs; gitignored. Never commit or upload. |
| `out/` | Build, generated sources and reports; gitignored. |

## Evidence

A change is ready when another contributor can reproduce it. Include:

- the commit or build under test;
- the exact command;
- the observed result and the expected result;
- the environment for hardware or driver-dependent results;
- the first mismatch or diagnostic for a failure;
- any capability skip, separate from passes and failures;
- a negative control or failure case for a new fixture.

State what was measured and what remains a hypothesis. A passing fixture supports
the behavior under test; it does not establish full-game startup.

## Pull requests

- Create a topic branch from current `main` in your fork.
- Use a focused branch name, for example `docs/agents-guide` or
  `bcpack/stride-6-diagnostic`.
- Link the issue with `Refs #123` or `Fixes #123` when the completion condition
  is met.
- Update the relevant README row or `docs/experiments.md` when a public
  capability, build requirement or reproduction command changes.
- Include the exact test command and result in the PR body.
- Do not include game files, dumps, distribution links or secrets.

## First places to look

- `README.md#choose-an-experiment`
- `CONTRIBUTING.md`
- `docs/experiments.md`
- Open issues labeled `help wanted` and `good first issue`.
