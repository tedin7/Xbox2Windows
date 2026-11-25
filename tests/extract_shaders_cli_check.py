#!/usr/bin/env python3
"""Check tools/extract_shaders.py: sanitize_filename(), extract_native_rdna2_code(),
main(argv) across formats/stages/limits/prefixed names, atomic manifest writes,
and clean CLI error exits without raw tracebacks or temp-file leaks."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "extract_shaders.py"
sys.path.insert(0, str(ROOT / "tools"))
import extract_shaders as es


def _build_fssm_image(code: bytes, *, magic: bytes = b"FSSM", start: int = 256) -> bytes:
    image = bytearray(max(start, 196))
    image[:4] = magic
    struct.pack_into("<I", image, 192, start - 40)
    return bytes(image[:start]) + code if start >= 196 else bytes(image)


def _build_dxil_program(
    code: bytes | None,
    *,
    stage_kind: int = 0,
    xdat: bool = True,
    xbch_magic: bytes = b"XBCH",
    tag: int = 13,
    corrupt_zlib: bool = False,
    fssm_magic: bytes = b"FSSM",
    declared_size: int | None = None,
    bad_offset: int | None = None,
) -> bytes:
    header = bytearray(24)
    struct.pack_into("<H", header, 2, stage_kind)
    if code is None and not xdat:
        return bytes(header) + b"NOP_XDAT"
    if not xdat:
        return bytes(header) + b"NO_TABLE"
    image = _build_fssm_image(code or b"", magic=fssm_magic)
    payload = b"bad_zlib_payload" if corrupt_zlib else zlib.compress(image)
    size_field = len(image) if declared_size is None else declared_size
    xbch = xbch_magic + bytes(12) + struct.pack("<II", size_field, 0) + payload
    table_len = 12 + 12
    offset = 24 + table_len if bad_offset is None else bad_offset
    return (
        bytes(header)
        + b"XDAT"
        + struct.pack("<II", 1, 0)
        + struct.pack("<III", tag, offset, len(xbch))
        + xbch
    )


def _build_dxbc(program: bytes) -> bytes:
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<4I", 1, 44 + len(program), 1, 36)
        + b"DXIL"
        + struct.pack("<I", len(program))
        + program
    )


def _build_rpf_record(name: bytes, dxbc: bytes, *, length_prefix: bool = True) -> bytes:
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(dxbc) + encoder.flush()
    prefix = bytes([len(name) + 1]) if length_prefix else b""
    return (
        prefix
        + name
        + b"\0\0"
        + struct.pack("<4I", len(payload), 0, len(payload), len(dxbc))
        + payload
    )


def _expect_exit(argv, expected_code=1, fragment=""):
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            es.main(argv)
    except SystemExit as exc:
        assert exc.code == expected_code, (exc.code, stderr.getvalue())
    else:
        raise AssertionError(f"main({argv!r}) did not raise SystemExit")
    err = stderr.getvalue()
    assert err.strip(), f"expected error message on stderr for {argv!r}"
    assert "Traceback" not in err, err
    if fragment:
        assert fragment in err, err
    return err


def check_unit_helpers():
    """Positive and negative controls for sanitize_filename() and extract_native_rdna2_code()."""
    # sanitize_filename()
    assert es.sanitize_filename('..  bad<name>:"/\\|?*  ..') == "bad_name________"
    assert es.sanitize_filename("...   ...") == "unnamed_shader"
    assert es.sanitize_filename("") == "unnamed_shader"
    long_utf8 = "a€" * 100
    clean_utf8 = es.sanitize_filename(long_utf8)
    assert long_utf8.startswith(clean_utf8) and 0 < len(clean_utf8.encode("utf-8")) <= es.MAX_STEM_BYTES

    # extract_native_rdna2_code() positive controls (including multiple s_endpgm words)
    code_single = bytes.fromhex("000080bf 000081bf")
    prog_single = _build_dxil_program(code_single, stage_kind=0)
    assert es.extract_native_rdna2_code(prog_single) == code_single

    code_multi = bytes.fromhex("000080bf 000081bf 000382be 000081bf 11223344")
    prog_multi = _build_dxil_program(code_multi, stage_kind=5)
    assert es.extract_native_rdna2_code(prog_multi) == code_multi[:-4]

    # extract_native_rdna2_code() negative controls
    assert es.extract_native_rdna2_code(_build_dxil_program(None, xdat=False)) is None
    assert es.extract_native_rdna2_code(bytes(24) + b"XDAT\x00") is None
    assert es.extract_native_rdna2_code(_build_dxil_program(code_single, tag=99)) is None
    assert es.extract_native_rdna2_code(_build_dxil_program(code_single, xbch_magic=b"NOPE")) is None
    assert es.extract_native_rdna2_code(_build_dxil_program(code_single, bad_offset=8)) is None
    assert es.extract_native_rdna2_code(_build_dxil_program(code_single, corrupt_zlib=True)) is None
    assert es.extract_native_rdna2_code(_build_dxil_program(code_single, fssm_magic=b"NOPE")) is None
    assert es.extract_native_rdna2_code(_build_dxil_program(code_single, declared_size=300)) is None
    assert es.extract_native_rdna2_code(_build_dxil_program(bytes.fromhex("000080bf 000382be"))) is None


def check_cli_positive():
    """Verify main(argv) across --format native/dxbc/both, --stage, --limit, and --length-prefixed-names."""
    ps_code = bytes.fromhex("000080bf 000081bf")
    cs_code = bytes.fromhex("7f008cbf 000081bf")
    no_native_prog = _build_dxil_program(None, stage_kind=1, xdat=False)  # vertex fallback

    ps_dxbc = _build_dxbc(_build_dxil_program(ps_code, stage_kind=0))
    cs_dxbc = _build_dxbc(_build_dxil_program(cs_code, stage_kind=5))
    vs_dxbc = _build_dxbc(no_native_prog)

    # Create a printable length prefix (byte 0x41 = 'A' = 65 => 64-byte record name + NUL)
    prefixed_name = b"PS_" + b"P" * 61
    assert len(prefixed_name) + 1 == 0x41

    archive_bytes = (
        b"RPF9\0"
        + _build_rpf_record(prefixed_name, ps_dxbc)
        + _build_rpf_record(b"CS_Compute", cs_dxbc)
        + _build_rpf_record(b"VS_Fallback", vs_dxbc)
    )

    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        archive = work / "shaders.rpf"
        archive.write_bytes(archive_bytes)

        # 1. --format native + --length-prefixed-names (pixel/compute native .bin, vertex fallback .dxbc)
        out_native = work / "out_native"
        with redirect_stdout(io.StringIO()):
            rc = es.main([
                str(archive), "--output-dir", str(out_native),
                "--format", "native", "--length-prefixed-names",
            ])
        assert rc == 0, rc
        manifest = json.loads((out_native / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["archive"] == "shaders.rpf"
        assert manifest["format"] == "native"
        assert manifest["total_extracted"] == 3
        assert manifest["stage_distribution"] == {"pixel": 1, "compute": 1, "vertex": 1}
        rec_ps, rec_cs, rec_vs = manifest["shaders"]
        assert rec_ps["name"] == prefixed_name.decode("ascii")
        assert rec_ps["native_file"] == f"pixel/{prefixed_name.decode('ascii')}.bin"
        assert rec_ps["sha256"] == hashlib.sha256(ps_code).hexdigest()
        assert (out_native / rec_ps["native_file"]).read_bytes() == ps_code
        assert rec_cs["native_file"] == "compute/CS_Compute.bin"
        assert (out_native / rec_cs["native_file"]).read_bytes() == cs_code
        assert rec_vs["dxbc_file"] == "vertex/VS_Fallback.dxbc"
        assert (out_native / rec_vs["dxbc_file"]).read_bytes() == vs_dxbc
        assert not any(p.name.endswith(".tmp") for p in out_native.iterdir())

        # 2. --format dxbc + --stage compute
        out_dxbc = work / "out_dxbc"
        with redirect_stdout(io.StringIO()):
            rc = es.main([
                str(archive), "--output-dir", str(out_dxbc),
                "--format", "dxbc", "--stage", "compute",
            ])
        assert rc == 0, rc
        m_dxbc = json.loads((out_dxbc / "manifest.json").read_text(encoding="utf-8"))
        assert m_dxbc["total_extracted"] == 1
        assert m_dxbc["shaders"][0]["dxbc_file"] == "compute/CS_Compute.dxbc"
        assert "native_file" not in m_dxbc["shaders"][0]
        assert (out_dxbc / "compute" / "CS_Compute.dxbc").read_bytes() == cs_dxbc

        # 3. --format both + --limit 1
        out_both = work / "out_both"
        with redirect_stdout(io.StringIO()):
            rc = es.main([
                str(archive), "--output-dir", str(out_both),
                "--format", "both", "--limit", "1", "--length-prefixed-names",
            ])
        assert rc == 0, rc
        m_both = json.loads((out_both / "manifest.json").read_text(encoding="utf-8"))
        assert m_both["total_extracted"] == 1
        rec0 = m_both["shaders"][0]
        assert (out_both / rec0["native_file"]).read_bytes() == ps_code
        assert (out_both / rec0["dxbc_file"]).read_bytes() == ps_dxbc

        # 4. End-to-end subprocess check.
        out_sub = work / "out_sub"
        done = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(archive), "--output-dir", str(out_sub), "--stage", "compute"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stderr == "", done
        assert (out_sub / "compute" / "CS_Compute.bin").read_bytes() == cs_code
        assert (out_sub / "manifest.json").is_file()


def check_cli_negative():
    """Pin clean non-zero exits without raw tracebacks or temp leaks."""
    ps_dxbc = _build_dxbc(_build_dxil_program(bytes.fromhex("000080bf 000081bf"), stage_kind=0))
    archive_bytes = b"RPF9\0" + _build_rpf_record(b"PS_One", ps_dxbc)

    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        archive = work / "shaders.rpf"
        archive.write_bytes(archive_bytes)

        # Missing archive -> SystemExit(1), output dir not created.
        missing_archive = work / "missing.rpf"
        missing_out = work / "missing_out"
        _expect_exit([str(missing_archive), "--output-dir", str(missing_out)], fragment="File not found")
        assert not missing_out.exists()

        # Directory as archive -> SystemExit(1), output dir not created.
        dir_archive = work / "dir_archive"
        dir_archive.mkdir()
        _expect_exit([str(dir_archive), "--output-dir", str(missing_out)], fragment="not a file")
        assert not missing_out.exists()

        # Output == archive collision (exact and non-normalized alias) preserves archive bytes.
        orig_archive_bytes = archive.read_bytes()
        _expect_exit([str(archive), "--output-dir", str(archive)], fragment="must not match")
        assert archive.read_bytes() == orig_archive_bytes
        alias = work / "sub" / ".." / "shaders.rpf"
        _expect_exit([str(archive), "--output-dir", str(alias)], fragment="must not match")
        assert archive.read_bytes() == orig_archive_bytes

        # An archive may have the same name as a generated output.
        manifest_archive = work / "manifest.json"
        manifest_archive.write_bytes(archive_bytes)
        _expect_exit([str(manifest_archive), "--output-dir", str(work)], fragment="overwrite the input archive")
        assert manifest_archive.read_bytes() == archive_bytes
        assert not (work / "pixel").exists()

        shader_archive = work / "pixel" / "PS_One.dxbc"
        shader_archive.parent.mkdir()
        shader_archive.write_bytes(archive_bytes)
        with redirect_stdout(io.StringIO()):
            _expect_exit([str(shader_archive), "--output-dir", str(work), "--format", "dxbc"],
                         fragment="overwrite the input archive")
        assert shader_archive.read_bytes() == archive_bytes

        # Regular file as --output-dir -> SystemExit(1), preserves regular file contents.
        regular_out = work / "existing_file.txt"
        regular_out.write_text("keep_me", encoding="utf-8")
        _expect_exit([str(archive), "--output-dir", str(regular_out)], fragment="not a directory")
        assert regular_out.read_text(encoding="utf-8") == "keep_me"

        # --limit < 1 -> argparse usage exit(2), output dir not created.
        for bad_limit in ("0", "-1"):
            _expect_exit(
                [str(archive), "--output-dir", str(missing_out), "--limit", bad_limit],
                expected_code=2,
                fragment="--limit",
            )
            assert not missing_out.exists()

        # struct.error / zlib.error caught cleanly in main() without raw traceback.
        orig_extract = es.extract_shaders
        for err in (struct.error("bad struct"), zlib.error("bad zlib")):
            def fail_extract(*args, _err=err, **kwargs):
                raise _err

            es.extract_shaders = fail_extract
            try:
                _expect_exit([str(archive), "--output-dir", str(work / "err_out")], fragment=str(err))
            finally:
                es.extract_shaders = orig_extract

        # Write-failure temp cleanup: simulated replace failure preserves existing manifest and leaves zero .tmp files.
        sentinel_dir = work / "sentinel_out"
        sentinel_dir.mkdir()
        sentinel_manifest = sentinel_dir / "manifest.json"
        sentinel_manifest.write_text("sentinel", encoding="utf-8")
        orig_replace = Path.replace

        def fail_replace(self, target):
            raise OSError("simulated replace failure")

        Path.replace = fail_replace
        try:
            with redirect_stdout(io.StringIO()):
                _expect_exit([str(archive), "--output-dir", str(sentinel_dir)], fragment="simulated replace failure")
        finally:
            Path.replace = orig_replace
        assert sentinel_manifest.read_text(encoding="utf-8") == "sentinel"
        assert not any(p.name.endswith(".tmp") for p in sentinel_dir.iterdir())

        # Subprocess negative check (no traceback on stderr).
        sub_err = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(missing_archive), "--output-dir", str(missing_out)],
            capture_output=True,
            text=True,
        )
        assert sub_err.returncode == 1
        assert "File not found" in sub_err.stderr and "Traceback" not in sub_err.stderr, sub_err.stderr


def main():
    check_unit_helpers()
    check_cli_positive()
    check_cli_negative()
    print("extract_shaders_cli: helpers, CLI formats/stages/limits, atomic manifest, and error checks passed")


if __name__ == "__main__":
    main()
