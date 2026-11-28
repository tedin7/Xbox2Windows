"""Exercise safe startup staging using synthetic files; no game package is needed."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).resolve().parents[1] / "tools/startup/stage_package.ps1"


def call(powershell, package, exe, run):
    return subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-PackageDir", str(package), "-Exe", str(exe), "-RunDir", str(run)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=30,
    )


def stage_from_changed_location(powershell, root, package, location):
    return subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
         r"$ErrorActionPreference = 'Stop'; Set-Location -LiteralPath $env:X2W_TEST_LOCATION; "
         r"& $env:X2W_TEST_SCRIPT -PackageDir $env:X2W_TEST_PACKAGE -Exe 'bin\child.exe' -RunDir 'package-run'"],
        cwd=root, env=dict(os.environ, X2W_TEST_LOCATION=str(location), X2W_TEST_SCRIPT=str(SCRIPT),
                           X2W_TEST_PACKAGE=str(package)),
        capture_output=True, encoding="utf-8", errors="replace", timeout=30,
    )


def main():
    powershell = sys.argv[1]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        package = root / "package"
        package.mkdir()
        exe = package / "title.exe"
        dll = package / "runtime.dll"
        asset = package / "large.rpf"
        exe.write_bytes(b"synthetic executable")
        dll.write_bytes(b"synthetic runtime")
        asset.write_bytes(b"package data must not be staged")
        (package / "backup.dll_").write_bytes(b"not a DLL")
        (package / "backup.dllold").write_bytes(b"not a DLL either")
        original = {path.name: path.read_bytes() for path in (exe, dll, asset)}

        run = root / "scratch" / "run"
        result = call(powershell, package, exe.name, run)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (run / exe.name).read_bytes() == original[exe.name]
        assert (run / dll.name).read_bytes() == original[dll.name]
        assert {entry.name for entry in run.iterdir()} == {exe.name, dll.name}
        assert {path.name: path.read_bytes() for path in (exe, dll, asset)} == original

        # A reused run must never overwrite the previous copy or its stand-ins.
        (run / dll.name).write_bytes(b"stand-in installed for this run")
        result = call(powershell, package, exe.name, run)
        assert result.returncode != 0 and "absent or empty" in result.stderr
        assert (run / dll.name).read_bytes() == b"stand-in installed for this run"

        for destination in (package, package / "nested-run", root):
            result = call(powershell, package, exe.name, destination)
            assert result.returncode != 0 and "separate from PackageDir" in result.stderr
        result = call(powershell, package, asset.name, root / "bad")
        assert result.returncode != 0 and "executable file" in result.stderr
        outside = root / "outside.exe"
        outside.write_bytes(b"outside")
        result = call(powershell, package, outside, root / "bad")
        assert result.returncode != 0 and "inside PackageDir" in result.stderr
        result = call(powershell, package, exe.name, outside)
        assert result.returncode != 0 and "absent or empty" in result.stderr
        assert outside.read_bytes() == b"outside"

        bin_dir = package / "bin"
        bin_dir.mkdir()
        (bin_dir / "child.exe").write_bytes(b"child")
        (bin_dir / "side.DLL").write_bytes(b"beside child")
        location = root / "changed-location"
        location.mkdir()
        result = stage_from_changed_location(powershell, run.parent, package, root)
        assert result.returncode == 0, result.stdout + result.stderr
        sibling = root / "package-run"
        assert {entry.name for entry in sibling.iterdir()} == {"child.exe", "side.DLL"}
        assert (sibling / "side.DLL").read_bytes() == b"beside child"
        assert not (run.parent / "package-run").exists()

        junction = root / "link"
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(junction), str(location)],
                                capture_output=True, errors="replace", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        try:
            result = call(powershell, package, exe.name, junction / "new-run")
            assert result.returncode != 0 and "reparse point" in result.stderr
            assert not (location / "new-run").exists()
        finally:
            junction.rmdir()

        # The package can be named through an alias while the destination uses its real path.
        # Both spellings must refer to the same tree for the separation check.
        alias = root / "package-alias"
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(alias), str(package)],
                                capture_output=True, errors="replace", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        try:
            nested = package / "new" / "run"
            result = call(powershell, alias, exe.name, nested)
            assert result.returncode != 0 and "separate from PackageDir" in result.stderr, result.stdout + result.stderr
            assert not nested.exists()
            safe = root / "alias-safe-run"
            result = call(powershell, alias, exe.name, safe)
            assert result.returncode == 0, result.stdout + result.stderr
            assert (safe / exe.name).read_bytes() == original[exe.name]
        finally:
            alias.rmdir()

        # A path spelled inside the package can also lead outside through a junction.
        outside_dir = root / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "outside.exe").write_bytes(b"outside executable")
        escape = package / "escape"
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(escape), str(outside_dir)],
                                capture_output=True, errors="replace", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        try:
            refused = root / "escape-run"
            result = call(powershell, package, "escape/outside.exe", refused)
            assert result.returncode != 0 and "inside PackageDir" in result.stderr, result.stdout + result.stderr
            assert not refused.exists()
        finally:
            escape.rmdir()

        # A junction whose target stays inside the package remains usable.
        inside = package / "inside"
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(inside), str(bin_dir)],
                                capture_output=True, errors="replace", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        try:
            safe = root / "inside-run"
            result = call(powershell, package, "inside/child.exe", safe)
            assert result.returncode == 0, result.stdout + result.stderr
            assert (safe / "child.exe").read_bytes() == b"child"
            assert (safe / "side.DLL").read_bytes() == b"beside child"
        finally:
            inside.rmdir()

        # Volume GUID paths are returned as provider-qualified paths by PowerShell.
        # Staging to a sibling must work, while the underlying package is still protected.
        volume = ctypes.create_unicode_buffer(1024)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetVolumeNameForVolumeMountPointW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel.GetVolumeNameForVolumeMountPointW.restype = ctypes.c_int
        if kernel.GetVolumeNameForVolumeMountPointW(package.anchor, volume, len(volume)):
            volume_package = volume.value + str(package)[len(package.anchor):]
            nested = package / "volume-nested"
            result = call(powershell, volume_package, exe.name, nested)
            assert result.returncode != 0 and "separate from PackageDir" in result.stderr, result.stdout + result.stderr
            assert not nested.exists()
            safe = root / "volume-safe"
            result = call(powershell, volume_package, exe.name, safe)
            assert result.returncode == 0, result.stdout + result.stderr
            assert (safe / exe.name).read_bytes() == original[exe.name]
        else:
            print("volume GUID control skipped: temporary volume has no GUID path")
    print("safe startup staging: isolated executable/DLL copy, package preservation and overwrite/path guards passed")


if __name__ == "__main__":
    main()
