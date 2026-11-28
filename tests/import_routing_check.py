"""Unit, boundary-validation, and temp-copy mutation suite for tools/import_routing.py."""

from collections import UserDict
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MappingProxyType
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "import_routing.py"
sys.path.insert(0, str(ROOT / "tools"))

import import_routing  # noqa: E402


MUTATIONS = (
    (
        "drop_non_str_guard",
        '    if not isinstance(module, str):\n'
        '        raise ValueError(f"module name must be a str, got {type(module).__name__}")\n',
        '    if False:\n'
        '        raise ValueError(f"module name must be a str, got {type(module).__name__}")\n',
    ),
    (
        "drop_empty_whitespace_control_guard",
        '    if not module or any(ch.isspace() or ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in module):\n'
        '        raise ValueError(f"invalid module name: {module!r}")\n',
        '    if False:\n'
        '        raise ValueError(f"invalid module name: {module!r}")\n',
    ),
    (
        "drop_mutable_mapping_guard",
        '    if not isinstance(row, MutableMapping):\n'
        '        raise TypeError(f"row must be a MutableMapping, got {type(row).__name__}")\n',
        '    if False:\n'
        '        raise TypeError(f"row must be a MutableMapping, got {type(row).__name__}")\n',
    ),
    (
        "drop_lower_normalization",
        "    name = module.lower()\n",
        "    name = module\n",
    ),
    (
        "substring_api_set_prefix",
        "        if name.startswith(prefix):\n",
        "        if prefix in name:\n",
    ),
    (
        "lowercase_candidate_provider",
        '        "candidate_provider": module if route == "hle" else None,\n',
        '        "candidate_provider": module.lower() if route == "hle" else None,\n',
    ),
)


def _load_temp_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not create module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


class ImportRoutingTests(unittest.TestCase):
    def _verify_module_contract(self, mod) -> None:
        self.assertEqual(mod.POLICY_VERSION, 2)
        self.assertEqual(mod.ROUTE_KIND, "candidate")
        self.assertEqual(mod.RESOLUTION, "not_checked")
        self.assertEqual(mod.ROUTES, ("hle", "windows", "stub", "unsupported"))

        # All 4 policy routes and mixed-case normalization.
        for name, expected_route in (
            ("xmem.dll", "hle"),
            ("XMem.DLL", "hle"),
            ("d3d12_xs.dll", "hle"),
            ("D3D12_Xs.Dll", "hle"),
            ("xgameruntime.dll", "stub"),
            ("XGameRuntime.DLL", "stub"),
            ("xg_xs.dll", "stub"),
            ("XG_XS.dll", "stub"),
            ("kernel32.dll", "windows"),
            ("Kernel32.DLL", "windows"),
            ("ntdll.dll", "windows"),
            ("xaudio2_8.dll", "windows"),
            ("XAudio2_9.DLL", "windows"),
            ("gameinput.dll", "windows"),
            ("mystery.dll", "unsupported"),
            ("Mystery.DLL", "unsupported"),
            ("winhttp2.dll", "unsupported"),
            ("my-api-ms-win-core.dll", "unsupported"),
            ("not-ext-ms-core.dll", "unsupported"),
        ):
            route, reason = mod.classify(name)
            self.assertEqual(route, expected_route, name)
            self.assertIsInstance(reason, str)
            self.assertTrue(reason)

        # Prefix routes preserve contract reason across mixed case.
        for prefix_name in (
            "api-ms-win-core-file-l1-1-0.dll",
            "API-MS-WIN-CORE-SYNCH-L1-2-0.DLL",
            "ext-ms-win-ntuser-window-l1-1-0.dll",
            "EXT-MS-ONECORE-TEST-L1-1-0.DLL",
        ):
            self.assertEqual(
                mod.classify(prefix_name),
                ("windows", "Windows API-set contract"),
                prefix_name,
            )

        # Invalid module names must raise ValueError (never AttributeError or 'unsupported').
        for bad_module in (
            None,
            123,
            True,
            b"xmem.dll",
            bytearray(b"xmem.dll"),
            ["xmem.dll"],
            {"module": "xmem.dll"},
            "",
            " ",
            "\t\n\r",
            "\u00a0",
            " xmem.dll",
            "xmem.dll ",
            "x mem.dll",
            "kernel32\t.dll",
            "\x00",
            "xmem\x00.dll",
            "xmem\x1f.dll",
            "xmem\x7f.dll",
            "xmem\x85.dll",
            "xmem\x9f.dll",
        ):
            with self.assertRaises(ValueError, msg=f"module={bad_module!r}"):
                mod.classify(bad_module)

        # annotate() requires a MutableMapping containing module_key before mutating.
        for bad_row, key in (
            (None, "module"),
            (123, "module"),
            ("xmem.dll", "module"),
            (["xmem.dll"], 0),
            (("xmem.dll",), 0),
            (MappingProxyType({"module": "xmem.dll"}), "module"),
        ):
            with self.assertRaises(TypeError, msg=f"row={bad_row!r}"):
                mod.annotate(bad_row, module_key=key)

        missing_row = {"dll": "xmem.dll", "caller": "app.exe"}
        with self.assertRaises(KeyError):
            mod.annotate(missing_row)
        self.assertEqual(missing_row, {"dll": "xmem.dll", "caller": "app.exe"})

        for invalid_val in ("", None, " bad.dll", "bad\x00.dll"):
            untouched = {"module": invalid_val, "caller": "app.exe", "symbol": "Sym"}
            snapshot = dict(untouched)
            with self.assertRaises(ValueError):
                mod.annotate(untouched)
            self.assertEqual(untouched, snapshot)

        # annotate() preserves caller keys, return schema, and original casing for HLE candidate_provider.
        hle_row = {"image": "app.exe", "kind": "normal", "dll": "XMem.DLL", "symbol": "XMemAlloc", "extra": 42}
        ret = mod.annotate(hle_row, module_key="dll")
        self.assertIs(ret, hle_row)
        self.assertEqual(
            hle_row,
            {
                "image": "app.exe",
                "kind": "normal",
                "dll": "XMem.DLL",
                "symbol": "XMemAlloc",
                "extra": 42,
                "route": "hle",
                "reason": mod.POLICY["hle"]["xmem.dll"],
                "route_kind": mod.ROUTE_KIND,
                "resolution": mod.RESOLUTION,
                "provider": None,
                "stub": False,
                "candidate_provider": "XMem.DLL",
            },
        )

        stub_row = UserDict({"module": "XGameRuntime.DLL", "symbol": "XUserGetId"})
        ret_stub = mod.annotate(stub_row)
        self.assertIs(ret_stub, stub_row)
        self.assertEqual(stub_row["route"], "stub")
        self.assertTrue(stub_row["stub"])
        self.assertIsNone(stub_row["candidate_provider"])
        self.assertIsNone(stub_row["provider"])
        self.assertEqual(stub_row["symbol"], "XUserGetId")

        win_row = {"module": "API-MS-WIN-CORE-FILE-L1-1-0.DLL", "caller": "app.exe"}
        mod.annotate(win_row)
        self.assertEqual(win_row["route"], "windows")
        self.assertFalse(win_row["stub"])
        self.assertIsNone(win_row["candidate_provider"])

        unsup_row = {"module": "Mystery.DLL", "caller": "app.exe"}
        mod.annotate(unsup_row)
        self.assertEqual(unsup_row["route"], "unsupported")
        self.assertFalse(unsup_row["stub"])
        self.assertIsNone(unsup_row["candidate_provider"])

    def test_contract_and_policy_coverage(self) -> None:
        self._verify_module_contract(import_routing)

        for mod_name, reason in import_routing.POLICY["hle"].items():
            self.assertEqual(import_routing.classify(mod_name), ("hle", reason))
        for mod_name, reason in import_routing.POLICY["stub"].items():
            self.assertEqual(import_routing.classify(mod_name), ("stub", reason))
        for mod_name in import_routing.POLICY["windows"]:
            self.assertEqual(import_routing.classify(mod_name), ("windows", "Windows system DLL"))

    def test_fast_temp_copy_mutations(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for label, target, replacement in MUTATIONS:
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mut_path = tmp_dir / f"import_routing_{label}.py"
                    mut_path.write_text(source.replace(target, replacement, 1), encoding="utf-8")
                    mod_name = f"import_routing_mut_{label}"
                    mod = _load_temp_module(mut_path, mod_name)
                    try:
                        with self.assertRaises(
                            (AssertionError, AttributeError, KeyError, TypeError),
                            msg=f"mutation {label} escaped detection",
                        ):
                            self._verify_module_contract(mod)
                    finally:
                        sys.modules.pop(mod_name, None)


if __name__ == "__main__":
    unittest.main()
