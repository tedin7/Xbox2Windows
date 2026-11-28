"""Mocked CLI and temp-copy mutation unit suite for tools/record_gpu_cases.py."""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
RECORDER_PATH = TOOLS_DIR / "record_gpu_cases.py"


def _load_module(path, name="record_gpu_cases_sut"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_dxc(root):
    dxc = Path(root) / "fake_dxc.exe"
    dxc.write_bytes(b"DXC")
    return dxc


def _fake_run_case(name, dxc, runner, adapter, folder, mutation=None):
    assert Path(dxc).is_file(), f"run_case received non-file dxc: {dxc}"
    tag = (mutation or "base").encode("ascii")
    (folder / f"{name}.cso").write_bytes(b"DXBC" + tag)
    (folder / f"{name}.case").write_bytes(b"CASE:" + name.encode("ascii") + b":" + tag)


def _tmp_files(directory):
    return [p for p in Path(directory).iterdir() if p.name.endswith(".tmp") or p.name.startswith(".manifest.tsv.")]


def _make_symlink(target, link_path):
    try:
        os.symlink(target, link_path)
        return True
    except (OSError, NotImplementedError):
        return False


def _test_atomic_manifest_and_zero_temps(mod, root):
    dxc = _fake_dxc(root)
    out = Path(root) / "out"
    with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = mod.main([str(dxc), str(out)])
    assert rc == 0, f"expected rc=0, got {rc} stderr={stderr.getvalue()!r}"
    assert stderr.getvalue() == ""
    manifest = out / "manifest.tsv"
    assert manifest.is_file()
    lines = manifest.read_text(encoding="ascii").splitlines()
    assert len(lines) == len(mod.check.CASES) + len(mod.check.CONTROLS)
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 4
        ident, compiler, expect, _ = fields
        assert compiler == "dxc" and expect in ("pass", "fail")
        assert (out / f"{ident}.cso").is_file()
        assert (out / f"{ident}.case").is_file()
    assert not _tmp_files(out), f"leftover temp files: {_tmp_files(out)}"

    # Resolving dxc by name via shutil.which.
    out_which = Path(root) / "out_which"
    with mock.patch.object(mod.shutil, "which", return_value=str(dxc)):
        with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
            rows = mod.record("fake_dxc.exe", out_which, cases=(next(iter(mod.check.CASES)),), controls=())
    assert len(rows) == 1
    assert (out_which / "manifest.tsv").is_file()
    assert not _tmp_files(out_which)


def _test_dxc_validation(mod, root):
    out = Path(root) / "out"
    missing = Path(root) / "missing_dxc.exe"
    dxc_dir = Path(root) / "dxc_dir"
    dxc_dir.mkdir()
    for bad_dxc in (missing, dxc_dir):
        stderr = io.StringIO()
        with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
            with contextlib.redirect_stderr(stderr):
                rc = mod.main([str(bad_dxc), str(out)])
        assert rc == 1, f"expected rc=1 for bad dxc {bad_dxc}, got {rc}"
        assert "Traceback" not in stderr.getvalue() and stderr.getvalue().strip()
        assert not (out / "manifest.tsv").exists()


def _test_output_validation(mod, root):
    dxc = _fake_dxc(root)

    # 1. Existing regular file output.
    file_out = Path(root) / "out_file"
    file_out.write_bytes(b"KEEP_FILE")
    stderr = io.StringIO()
    with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
        try:
            mod.record(str(dxc), str(file_out))
        except ValueError:
            pass
        else:
            raise AssertionError("record() did not raise ValueError for file output")
        with contextlib.redirect_stderr(stderr):
            rc = mod.main([str(dxc), str(file_out)])
    assert rc == 1 and "Traceback" not in stderr.getvalue()
    assert file_out.read_bytes() == b"KEEP_FILE"

    # 2. File-backed parent.
    sub_out = file_out / "child"
    stderr = io.StringIO()
    with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
        try:
            mod.record(str(dxc), str(sub_out))
        except ValueError:
            pass
        else:
            raise AssertionError("record() did not raise ValueError for file-backed parent")
        with contextlib.redirect_stderr(stderr):
            rc = mod.main([str(dxc), str(sub_out)])
    assert rc == 1 and "Traceback" not in stderr.getvalue()
    assert file_out.read_bytes() == b"KEEP_FILE"

    # 3. samefile output == dxc (direct and mocked os.path.samefile for a directory/alias).
    stderr = io.StringIO()
    with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
        with contextlib.redirect_stderr(stderr):
            rc = mod.main([str(dxc), str(dxc)])
    assert rc == 1 and "Traceback" not in stderr.getvalue()
    assert dxc.read_bytes() == b"DXC"
    alias_out = Path(root) / "alias_out"
    alias_out.mkdir()
    with mock.patch.object(mod.os.path, "samefile", return_value=True):
        with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
            try:
                mod.record(str(dxc), str(alias_out))
            except ValueError:
                pass
            else:
                raise AssertionError("record() did not raise ValueError on samefile match")

    # 4. Symlink output.
    real_dir = Path(root) / "real_dir"
    real_dir.mkdir()
    link_out = Path(root) / "link_out"
    if _make_symlink(str(real_dir), str(link_out)):
        stderr = io.StringIO()
        with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
            with contextlib.redirect_stderr(stderr):
                rc = mod.main([str(dxc), str(link_out)])
        assert rc == 1 and "Traceback" not in stderr.getvalue()
        assert list(real_dir.iterdir()) == []
    else:
        with mock.patch.object(mod.Path, "is_symlink", autospec=True, side_effect=lambda p: p == link_out):
            stderr = io.StringIO()
            with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
                with contextlib.redirect_stderr(stderr):
                    rc = mod.main([str(dxc), str(link_out)])
            assert rc == 1 and "Traceback" not in stderr.getvalue()


def _test_replace_fail_preserves_manifest_and_cleans_temps(mod, root):
    dxc = _fake_dxc(root)
    out = Path(root) / "out_replace"
    out.mkdir()
    manifest = out / "manifest.tsv"
    manifest.write_text("OLD\tdxc\tpass\t\n", encoding="ascii")

    stderr = io.StringIO()
    with mock.patch.object(mod.check, "run_case", side_effect=_fake_run_case):
        with mock.patch.object(mod.os, "replace", side_effect=OSError("replace failed")):
            with contextlib.redirect_stderr(stderr):
                rc = mod.main([str(dxc), str(out)])
    assert rc == 1, f"expected rc=1 on os.replace failure, got {rc}"
    assert "Traceback" not in stderr.getvalue() and "replace failed" in stderr.getvalue()
    assert manifest.read_text(encoding="ascii") == "OLD\tdxc\tpass\t\n"
    assert not _tmp_files(out), f"leftover temp files after os.replace failure: {_tmp_files(out)}"


def _test_runtime_error_and_argparse(mod, root):
    dxc = _fake_dxc(root)
    out = Path(root) / "out_err"
    stderr = io.StringIO()
    with mock.patch.object(mod.check, "run_case", side_effect=RuntimeError("boom")):
        with contextlib.redirect_stderr(stderr):
            rc = mod.main([str(dxc), str(out)])
    assert rc == 1 and "Traceback" not in stderr.getvalue() and "boom" in stderr.getvalue()
    assert not (out / "manifest.tsv").exists()

    help_out = io.StringIO()
    try:
        with contextlib.redirect_stdout(help_out):
            mod.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help did not exit")
    assert "dxc" in help_out.getvalue() and "output" in help_out.getvalue()


def run_suite(mod):
    tests = [
        _test_atomic_manifest_and_zero_temps,
        _test_dxc_validation,
        _test_output_validation,
        _test_replace_fail_preserves_manifest_and_cleans_temps,
        _test_runtime_error_and_argparse,
    ]
    for fn in tests:
        with tempfile.TemporaryDirectory() as root:
            fn(mod, root)
    return len(tests)


def _run_mutations():
    source = RECORDER_PATH.read_text(encoding="utf-8")
    mutants = [
        (
            "drop_dxc_is_file_guard",
            "    if candidate.is_file():\n        return candidate",
            "    if True:\n        return candidate",
            _test_dxc_validation,
        ),
        (
            "drop_output_is_file_guard",
            "    if out.is_file() or (out.exists() and not out.is_dir()):\n"
            "        raise ValueError(f\"output path is not a directory: {out}\")",
            "    if False:\n        pass",
            _test_output_validation,
        ),
        (
            "drop_output_is_symlink_guard",
            "    if out.is_symlink():\n        raise ValueError(f\"output path cannot be a symlink: {out}\")",
            "    if False:\n        pass",
            _test_output_validation,
        ),
        (
            "drop_samefile_guard",
            "    if _samefile(out, dxc_path):\n        raise ValueError(f\"output path matches dxc executable: {out}\")",
            "    if False:\n        pass",
            _test_output_validation,
        ),
        (
            "non_atomic_manifest_write",
            "        os.replace(tmp_path, path)",
            "        path.write_text(text, encoding='ascii')\n        tmp_path.unlink(missing_ok=True)",
            _test_replace_fail_preserves_manifest_and_cleans_temps,
        ),
        (
            "drop_tmp_unlink_cleanup",
            "        if tmp_path is not None:\n            tmp_path.unlink(missing_ok=True)",
            "        pass",
            _test_replace_fail_preserves_manifest_and_cleans_temps,
        ),
    ]
    with tempfile.TemporaryDirectory() as mut_dir:
        for idx, (name, needle, repl, probe) in enumerate(mutants):
            assert source.count(needle) == 1, f"mutant needle not unique: {name}"
            mut_path = Path(mut_dir) / f"record_gpu_cases_mut_{idx}.py"
            mut_path.write_text(source.replace(needle, repl, 1), encoding="utf-8")
            mod = _load_module(mut_path, f"record_gpu_cases_mut_{idx}")
            with tempfile.TemporaryDirectory() as root:
                try:
                    probe(mod, root)
                except (AssertionError, OSError, SystemExit):
                    continue
                raise SystemExit(f"record_gpu_cases_cli_check: mutant {name} survived!")
    return len(mutants)


def main():
    mod = _load_module(RECORDER_PATH, "record_gpu_cases")
    passed = run_suite(mod)
    killed = _run_mutations()
    print(f"record_gpu_cases_cli_check: {passed}/{passed} unit checks passed, {killed}/{killed} mutants killed")


if __name__ == "__main__":
    main()
