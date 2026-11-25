"""Prove strict package preflight refuses a real XGameRuntime import fixture."""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from pe_package_preflight import inspect  # noqa: E402
from pe_run_package import plan  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", type=Path, help="Source-built EXE importing XPackageMountWithUiAsync")
    parser.add_argument("xmem", type=Path, help="Normal xmem.dll shim")
    parser.add_argument("d3d12_xs", type=Path, help="Normal d3d12_xs.dll shim")
    parser.add_argument("fixture_shim_dir", type=Path, help="Directory containing the fixture DLL")
    parser.add_argument("normal_shim_dir", type=Path, help="Normal shim output directory")
    return parser.parse_args(argv)


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main(argv=None):
    args = parse_args(argv)
    entry = args.entry.resolve(strict=True)
    xmem = args.xmem.resolve(strict=True)
    d3d12_xs = args.d3d12_xs.resolve(strict=True)
    fixture_dir = args.fixture_shim_dir.resolve(strict=True)
    normal_dir = args.normal_shim_dir.resolve(strict=True)

    fixture_dll = fixture_dir / "xgameruntime.dll"
    if not fixture_dll.is_file():
        return fail(f"fixture DLL is missing: {fixture_dll}")
    if (normal_dir / "xgameruntime.dll").exists():
        return fail(f"normal shim directory contains xgameruntime.dll: {normal_dir}")

    with tempfile.TemporaryDirectory(prefix="xgameruntime-preflight-") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copy2(entry, package / entry.name)
        shutil.copy2(xmem, package / xmem.name)
        shutil.copy2(d3d12_xs, package / d3d12_xs.name)
        if (package / "xgameruntime.dll").exists():
            return fail("synthetic package unexpectedly contains xgameruntime.dll")

        report = inspect(package, entry.name)
        expected_images = {entry.name, xmem.name, d3d12_xs.name}
        if set(report["images"]) != expected_images:
            return fail(f"package images are {sorted(report['images'])}, expected "
                        f"{sorted(expected_images)}")
        row = next((item for item in report["imports"]
                    if item["image"] == entry.name
                    and item["dll"].casefold() == "xgameruntime.dll"
                    and item["symbol"] == "XPackageMountWithUiAsync"), None)
        if row is None:
            return fail("the source-built EXE has no xgameruntime.dll!XPackageMountWithUiAsync import")
        if row["status"] != "external_unverified":
            return fail(f"preflight status is {row['status']}, expected external_unverified")
        if row["kind"] != "normal":
            return fail(f"import kind is {row['kind']}, expected normal")
        if row["chain"][0].get("package"):
            return fail("preflight treated the fixture DLL as a package provider")

        order, _ = plan(report)
        if order != [entry.name]:
            return fail(f"permissive plan order is {order}, expected only the entry")
        try:
            plan(report, allow_external=False)
        except ValueError as error:
            expected = "xgameruntime.dll!XPackageMountWithUiAsync: external_unverified"
            if expected not in str(error):
                return fail(f"strict plan refused a different import: {error}")
        else:
            return fail("strict plan accepted the unimplemented XGameRuntime import")

    print("xgameruntime preflight: package=real-source-exe "
          "import=xgameruntime.dll!XPackageMountWithUiAsync status=external_unverified "
          "strict_plan=refused mapped=0 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
