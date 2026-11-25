"""Synthetic PSO references and archived shader aliases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_entry_check import PROGRAM, enumerate_pcs
from pixel_aot_check import assembly
from pso_readiness import report


def with_rounding(instructions):
    """Insert the rounding forms and a multiply that pixel_aot lowers before the pack."""
    index = next(i for i, line in enumerate(instructions) if line.startswith("v_cvt_pkrtz"))
    extra = ["v_floor_f32_e32 v12, v12", "v_fract_f32_e32 v13, v13", "v_trunc_f32_e32 v15, v15",
             "v_add_f32_e32 v12, v12, v13", "v_mul_f32_e32 v15, v15, v12"]
    return instructions[:index] + extra + instructions[index:]


def with_lowered_ops(instructions):
    """Insert lowered SDWA, VOP3P, integer compare and conversion ops before the pack."""
    index = next(i for i, line in enumerate(instructions) if line.startswith("v_cvt_pkrtz"))
    extra = [
        "s_clause 0x1",
        "s_setprio 1",
        "s_sleep 2",
        "v_mov_b32_sdwa v12, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
        "v_cvt_f16_f32_e32 v13, v13",
        "v_cvt_f32_f16_sdwa v13, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
        "v_pk_add_f16 v14, v14, 1.0",
        "v_cvt_f32_ubyte0_e32 v14, v14",
        "v_cmp_lt_i32_e32 vcc_lo, v12, v13",
        "v_cndmask_b32_e32 v15, v12, v15, vcc_lo",
    ]
    return instructions[:index] + extra + instructions[index:]


def check_scanner_tracks_pixel_aot():
    import pixel_aot
    from scan_unsupported_shaders import is_instruction_supported, is_opcode_recognized
    for name in pixel_aot.ALU:
        opcode = f"v_{name}_f32"
        assert is_opcode_recognized(opcode) and is_opcode_recognized(opcode + "_e32"), opcode
        assert is_instruction_supported(f"{opcode}_e32 v1, v2, v3"), opcode
    for name, (arity, _) in pixel_aot.INTEGER.items():
        opcode = f"v_{name}"
        assert is_opcode_recognized(opcode), opcode
        operands = ", ".join(f"v{k}" for k in range(1, arity + 2))
        assert is_instruction_supported(f"{opcode} {operands}"), opcode
        if name in pixel_aot.INTEGER_VOP3_ONLY:
            # Two sources, but no VOP2 encoding on gfx10: the e32 spelling stays unsupported.
            assert not is_instruction_supported(f"{opcode}_e32 {operands}"), opcode
            assert is_instruction_supported(f"{opcode}_e64 {operands}"), opcode
        elif arity <= 2:
            assert is_opcode_recognized(opcode + "_e32"), opcode
            assert is_instruction_supported(f"{opcode}_e32 {operands}"), opcode
    for name, (arity, _, _) in pixel_aot.SCALAR.items():
        opcode = f"s_{name}"
        assert is_opcode_recognized(opcode), opcode
        operands = ", ".join(f"s{k}" for k in range(1, arity + 2))
        assert is_instruction_supported(f"{opcode} {operands}"), opcode
    for name in pixel_aot.SALU_BIT64:
        opcode = f"s_{name}_b64"
        assert is_opcode_recognized(opcode), opcode
        assert is_instruction_supported(f"{opcode} s[0:1], s[2:3], s[4:5]"), opcode
    for name in pixel_aot.VOP3P_F16:
        opcode = f"v_pk_{name}_f16"
        assert is_opcode_recognized(opcode), opcode
        assert is_instruction_supported(f"{opcode} v0, v1, v2"), opcode
    for name in pixel_aot.VOP3P_INT16:
        opcode = f"v_pk_{name}"
        assert is_opcode_recognized(opcode), opcode
        assert is_instruction_supported(f"{opcode} v0, v1, v2"), opcode
    assert len(pixel_aot.VOP3_INT16) == 12
    for name in pixel_aot.VOP3_INT16:
        opcode = f"v_{name}"
        assert is_opcode_recognized(opcode), opcode
        assert is_instruction_supported(f"{opcode} v1, v2, v3"), opcode
    assert len(pixel_aot.VOP3_INT16_TERNARY) == 8
    for name in pixel_aot.VOP3_INT16_TERNARY:
        opcode = f"v_{name}"
        assert is_opcode_recognized(opcode), opcode
        assert is_instruction_supported(f"{opcode} v1, v2, v3, v4"), opcode
    for op in ("v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16"):
        assert is_opcode_recognized(op), op
        assert is_instruction_supported(f"{op} v1, v2, v3, v4"), op
    for op in (
        "v_dot2_f32_f16", "v_dot2_i32_i16", "v_dot2_u32_u16",
        "v_dot4_i32_i8", "v_dot4_u32_u8", "v_dot8_i32_i4", "v_dot8_u32_u4",
    ):
        assert is_opcode_recognized(op), op
        assert is_instruction_supported(f"{op} v1, v2, v3, v4"), op
    for line in (
        "s_clause 63",
        "s_setprio 3",
        "s_sleep 127",
        "s_waitcnt vmcnt(15)",
        "s_bfm_b64 s[86:87], s58, s59",
        "s_bfe_u32 s72, s60, s61",
        "s_cmp_eq_i32 s0, s1",
        "v_mov_b32_sdwa v12, v12 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0",
        "v_cvt_f32_f16_sdwa v12, -|v13| clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
        "v_cvt_f16_f32_e64 v12, -|v13|",
        "v_cvt_off_f32_i4_e32 v12, v13",
        "v_ldexp_f32_e64 v12, v13, s44",
        "v_cubeid_f32_e64 v12, v13, v14, v15",
        "v_fma_mix_f32 v12, v13, v14, v15 op_sel_hi:[1,1,0]",
        "v_dot4_i32_i8 v12, v13, v14, v15 clamp",
        "v_med3_i32_e64 v12, v13, v14, v15",
        "v_cmp_lt_u32_e64 s[4:5], v12, v13",
        "image_sample v[12:14], v[8:9], s[28:35], s[40:43] dmask:0x7 dim:SQ_RSRC_IMG_2D",
    ):
        assert is_opcode_recognized(line.split()[0]), line
        assert is_instruction_supported(line), line
    assert not is_opcode_recognized("v_imaginary_opcode_sdwa")
    assert not is_instruction_supported("s_clause 64")
    assert not is_instruction_supported("s_setprio 4")
    assert not is_instruction_supported("s_sleep 128")
    assert not is_instruction_supported("s_waitcnt vmcnt(64)")
    assert not is_instruction_supported(
        "v_mov_b32_sdwa v12, v12 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:BYTE_0"
    )
    assert not is_instruction_supported("v_bfi_b32_e32 v0, v1, v2, v3")
    assert not is_instruction_supported("v_xor_b32 v0, v1")
    assert not is_instruction_supported("v_not_b32 v0, -v1")


def check_mesh():
    code = b"synthetic mesh"
    mesh = {"stage": "mesh", "sha256": hashlib.sha256(code).hexdigest()}
    manifest = {"shaders": [dict(mesh, native_file="mesh.bin")]}
    pipelines = {"pipelines": [
        {"id": "mesh", "effective_root_signature": "present", "shaders": [mesh]},
        {"id": "mesh-unknown-root", "effective_root_signature": "unknown", "shaders": [mesh]},
    ]}
    invalid_alloc = ["s_mov_b32 m0, 0" if op == "s_mov_b32 m0, s7" else op for op in PROGRAM]
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mesh.bin").write_bytes(code)
        for decoded, stage_status, pipeline_status, detail in (
                (enumerate_pcs(PROGRAM), "mesh_lowering_passed", "needs_pso_validation", None),
                # An empty allocation (0, 0) is refused by the count check.
                (enumerate_pcs(invalid_alloc), "mesh_lowering_rejected", "blocked",
                 "Conflicting mesh counts, vertices or primitive indices"),
                ([(0, "s_mov_b32 m0, s99")], "mesh_lowering_rejected", "blocked",
                 "Scalar register s99 used before it is written"),
                ([], "disassembly_failed", "blocked", "empty instruction stream"),
                (ValueError("Unknown instruction at byte 4"), "disassembly_failed", "blocked",
                 "Unknown instruction at byte 4")):
            patch_args = {"side_effect": decoded} if isinstance(decoded, Exception) else {"return_value": decoded}
            with patch("pixel_aot.disassemble", **patch_args) as scan:
                row, unknown = report(manifest, pipelines, root, "test-llvm")["pipelines"]
                assert row["shaders"]["mesh"] == {
                    "sha256": mesh["sha256"], "archive_records": 1,
                    "status": stage_status, "detail": detail,
                }, row
                assert row["status"] == pipeline_status, row
                assert unknown["status"] == ("blocked" if detail else "needs_evidence"), unknown
                expected_blocker = {"reason": stage_status, "stage": "mesh", "detail": detail} if detail else None
                assert row["first_blocker"] == expected_blocker, row
                scan.assert_called_once_with(
                    code, "test-llvm", offsets=True, features=b"+wavefrontsize64,-wavefrontsize32"
                )
        with patch("pixel_aot.disassemble", return_value=[(0, "s_getpc_b64 s[0:1]")]):
            row, unknown = report(manifest, pipelines, root, "test-llvm")["pipelines"]
        assert row["shaders"]["mesh"]["status"] == "mesh_lowering_rejected", row
        assert row["shaders"]["mesh"]["detail"] == "Unsupported scalar instruction s_getpc_b64", row
        assert row["status"] == "blocked" and unknown["status"] == "blocked", (row, unknown)
        # An error the lowering did not anticipate must not abort the whole report.
        with patch("pixel_aot.disassemble", return_value=[(0, "s_getpc_b64 s[0:1]")]), \
                patch("mesh_entry_aot.lower", side_effect=IndexError("operand")):
            row, unknown = report(manifest, pipelines, root, "test-llvm")["pipelines"]
        assert row["shaders"]["mesh"]["detail"] == "lowering failed unexpectedly: IndexError", row
        assert row["status"] == "blocked" and unknown["status"] == "blocked", (row, unknown)
        untested = report(manifest, pipelines, root)["pipelines"][0]
        assert untested["shaders"]["mesh"]["status"] == "mesh_scan_untested", untested
        assert untested["status"] == "needs_evidence", untested
    print("PSO mesh probe passed (1 lowered, 2 lowering rejections, 2 disassembly failures)")


def check_compute():
    from compute_store_check import READ_COPY
    code = b"synthetic compute"
    digest = hashlib.sha256(code).hexdigest()
    compute = {"stage": "compute", "sha256": digest}
    manifest = {"shaders": [dict(compute, native_file="compute.bin")]}
    pipelines = {"pipelines": [
        {"id": "compute", "effective_root_signature": "present", "shaders": [compute]},
        {"id": "compute-missing-root", "effective_root_signature": "missing", "shaders": [compute]},
    ]}
    program = [line for line in READ_COPY if not line.endswith(":")]
    mutated = [line.replace("buffer_load_dword", "buffer_load_imaginary") for line in program]
    assert mutated != program
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "compute.bin").write_bytes(code)
        for decoded, status, pipeline_status, detail in (
                (program, "compute_isa_recognized", "needs_evidence", "ISA recognized; bindings not assessed"),
                (mutated, "unsupported_opcodes", "blocked", "buffer_load_imaginary"),
                ([], "disassembly_failed", "blocked", "empty instruction stream"),
                (ValueError("Unknown instruction at byte 8"), "disassembly_failed", "blocked",
                 "Unknown instruction at byte 8"),
                ([None], "compute_scan_failed", "needs_evidence", "scan failed unexpectedly: AttributeError")):
            patch_args = {"side_effect": decoded} if isinstance(decoded, Exception) else {"return_value": decoded}
            with patch("pixel_aot.disassemble", **patch_args) as scan:
                row, missing = report(manifest, pipelines, root, "test-llvm")["pipelines"]
            actual = row["shaders"]["compute"]
            assert (actual["sha256"], actual["archive_records"], actual["status"]) == (digest, 1, status), row
            assert (actual["detail"] or "").startswith(detail), row
            assert row["status"] == pipeline_status, row
            assert missing["first_blocker"] == {"reason": "missing_root_signature"}, missing
            scan.assert_called_once_with(code, "test-llvm")
        untested = report(manifest, pipelines, root)["pipelines"][0]
        assert untested["shaders"]["compute"]["status"] == "compute_scan_untested", untested
        assert untested["status"] == "needs_evidence", untested
    print("PSO compute probe passed (1 recognized, 1 unsupported opcode, 2 disassembly failures, 1 scan failure)")


def check_invalid_input():
    pixel = {"stage": "pixel", "sha256": "a" * 64}
    manifest = {"shaders": [pixel]}

    def entry(**changes):
        return dict({"id": "p", "effective_root_signature": "present", "shaders": [pixel]}, **changes)

    valid = report(manifest, {"pipelines": [entry()]}, Path("."))["pipelines"]
    assert [row["status"] for row in valid] == ["needs_evidence"], valid
    bad_inputs = (
        ("duplicate ID", [entry(), entry()], "empty or duplicate pipeline ID 'p'"),
        ("empty ID", [entry(id="")], "empty or duplicate pipeline ID ''"),
        ("non-string ID", [entry(id=1)], "empty or duplicate pipeline ID 1"),
        ("root status", [entry(effective_root_signature="yes")], "invalid root signature status for p"),
        ("repeated stage", [entry(shaders=[pixel, pixel])], "pipeline p repeats stage pixel"),
        ("no shaders", [entry(shaders=[])], "pipeline p has no shader references"),
        ("short digest", [entry(shaders=[dict(pixel, sha256="ab")])],
         "shader references require a stage and a full native SHA-256"),
        ("empty stage", [entry(shaders=[dict(pixel, stage="")])],
         "shader references require a stage and a full native SHA-256"),
        ("shaders is a dict", [entry(shaders={"pixel": pixel})], "pipeline p shaders must be a list"),
        ("shaders is an int", [entry(shaders=5)], "pipeline p shaders must be a list"),
        ("shaders is a string", [entry(shaders="pixel")], "pipeline p shaders must be a list"),
        ("shader reference is a string", [entry(shaders=["pixel"])],
         "shader references require a stage and a full native SHA-256"),
        ("shader reference is an int", [entry(shaders=[5])],
         "shader references require a stage and a full native SHA-256"),
        ("pipeline is a string", ["p"], "pipeline entries must be JSON objects"),
        ("pipeline is a list", [[entry()]], "pipeline entries must be JSON objects"),
    )
    for name, pipelines, message in bad_inputs:
        try:
            report(manifest, {"pipelines": pipelines}, Path("."))
        except ValueError as exc:
            assert str(exc) == message, (name, str(exc))
        else:
            raise AssertionError(f"{name}: accepted invalid pipeline input")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        tool = str(Path(__file__).resolve().parents[1] / "tools" / "pso_readiness.py")
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name, pipelines, message in bad_inputs[:1] + bad_inputs[8:9]:
            (root / "pipelines.json").write_text(json.dumps({"pipelines": pipelines}), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-B", tool, "manifest.json", "pipelines.json"],
                                       cwd=root, capture_output=True, text=True)
            assert completed.returncode == 1 and completed.stdout == "", (name, completed)
            assert completed.stderr == f"PSO report: {message}\n", (name, completed.stderr)
    print(f"PSO input validation passed ({len(bad_inputs)} rejected inputs, 2 CLI exits, 1 valid control)")


def check():
    check_scanner_tracks_pixel_aot()
    code = b"synthetic native bytes"
    digest = hashlib.sha256(code).hexdigest()
    pixel = {"stage": "pixel", "sha256": digest}
    vertex = {"stage": "vertex", "sha256": digest}
    manifest = {"shaders": [
        dict(pixel, native_file="pixel.bin"), dict(pixel, native_file="alias.bin"),
        vertex, {"stage": "vertex"},
    ]}
    pipelines = {"pipelines": [
        {"id": "missing-root", "effective_root_signature": "missing", "shaders": [pixel]},
        {"id": "vertex-evidence", "effective_root_signature": "present", "shaders": [vertex]},
        {"id": "absent-record", "effective_root_signature": "unknown",
         "shaders": [dict(pixel, sha256="0" * 64)]},
        {"id": "pixel", "effective_root_signature": "present", "shaders": [pixel]},
    ]}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pixel.bin").write_bytes(code)
        for path, data in (("manifest.json", manifest), ("pipelines.json", pipelines)):
            (root / path).write_text(json.dumps(data), encoding="utf-8")
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parents[1] /
                                   "tools" / "pso_readiness.py"), "manifest.json", "pipelines.json"],
                                   cwd=root, capture_output=True, text=True, check=True)
        actual = json.loads(completed.stdout)
        assert (actual["archive_records"], actual["records_with_native_sha256"],
                actual["unique_native_stage_hashes"]) == (4, 3, 2)
        missing, vertex_row, absent, untested = actual["pipelines"]
        assert missing["first_blocker"] == {"reason": "missing_root_signature"}
        assert vertex_row["shaders"]["vertex"]["status"] == "stage_not_assessed"
        assert absent["shaders"]["pixel"]["status"] == "shader_not_in_manifest"
        assert untested["status"] == "needs_evidence"
        assert untested["shaders"]["pixel"]["archive_records"] == 2
        # Temporary paths can contain a directory alias on Windows CI.
        unnormalized_root = root / ".." / root.name
        for instructions, status, blocker, detail in (
                (assembly(False), "needs_pso_validation", None, None),
                (with_rounding(assembly(False)), "needs_pso_validation", None, None),
                (with_lowered_ops(assembly(False)), "needs_pso_validation", None, None),
                ([], "blocked", "disassembly_failed", "empty instruction stream"),
                (ValueError("Unknown instruction at byte 0"), "blocked", "disassembly_failed",
                 "Unknown instruction at byte 0"),
                (["s_endpgm"], "blocked", "pixel_lowering_rejected", None),
                (["unknown_instruction"], "blocked", "unsupported_opcodes", "unknown_instruction")):
            patch_args = {"side_effect": instructions} if isinstance(instructions, Exception) else {"return_value": instructions}
            with patch("pixel_aot.disassemble", **patch_args) as scan:
                rows = report(manifest, pipelines, unnormalized_root, "test-llvm")["pipelines"]
                assert rows[0]["first_blocker"] == {"reason": "missing_root_signature"}
                assert rows[2]["status"] == "needs_evidence" and rows[2]["first_blocker"] is None
                assert rows[-1]["status"] == status, rows[-1]
                if blocker:
                    assert rows[-1]["first_blocker"]["reason"] == blocker
                    assert rows[-1]["first_blocker"]["stage"] == "pixel"
                    if detail is not None:
                        assert rows[-1]["first_blocker"]["detail"] == detail
                else:
                    assert rows[-1]["first_blocker"] is None
                scan.assert_called_once_with(code, "test-llvm")
        (root / "pixel.bin").write_bytes(b"mismatched bytes")
        try:
            report(manifest, pipelines, root, "test-llvm")
        except ValueError as exc:
            assert "does not match SHA-256" in str(exc)
        else:
            raise AssertionError("accepted native bytes under the wrong hash")

        outside = {"shaders": [dict(pixel, native_file="../pixel.bin")]}
        try:
            report(outside, pipelines, unnormalized_root, "test-llvm")
        except ValueError as exc:
            assert "outside the manifest directory" in str(exc)
        else:
            raise AssertionError("accepted a native path outside the manifest directory")
    print("PSO census, evidence states, and native hash control passed")


if __name__ == "__main__":
    check_mesh()
    check_compute()
    check_invalid_input()
    check()
