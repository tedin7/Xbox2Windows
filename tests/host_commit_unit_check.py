"""Pure-Python unit, boundary, mocked WinAPI, and temp-copy mutation suite for tools/host_commit.py."""

import ctypes
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MappingProxyType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "host_commit.py"
sys.path.insert(0, str(ROOT / "tools"))

import host_commit  # noqa: E402


MUTATIONS = (
    (
        "drop_is_int_bool_guard",
        "    return isinstance(value, int) and not isinstance(value, bool)\n",
        "    return isinstance(value, int)\n",
    ),
    (
        "drop_used_pages_nonneg_guard",
        "    if not _is_int(used_pages) or used_pages < 0:\n",
        "    if False:\n",
    ),
    (
        "drop_snapshot_mapping_guard",
        "    if not isinstance(snapshot, Mapping):\n",
        "    if False:\n",
    ),
    (
        "drop_snapshot_field_validation",
        "    for key in _SNAPSHOT_KEYS:\n",
        "    for key in ():\n",
    ),
    (
        "drop_free_bytes_clamp",
        "    free_bytes = max(limit_bytes - used_bytes, 0)\n",
        "    free_bytes = limit_bytes - used_bytes\n",
    ),
    (
        "drop_winmode_800",
        '    psapi = C.WinDLL("psapi.dll", winmode=0x800, use_last_error=True)\n',
        '    psapi = C.WinDLL("psapi.dll", use_last_error=True)\n',
    ),
)


class _FakeGetPerformanceInfo:
    def __init__(self, *, limit=1024, used=256, page_size=65536, success=True):
        self.limit = limit
        self.used = used
        self.page_size = page_size
        self.success = success
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, ptr, cb_size, struct_cls):
        info = ctypes.cast(ptr, ctypes.POINTER(struct_cls)).contents
        self.calls.append((info.cb, cb_size))
        if self.success:
            info.CommitLimit = self.limit
            info.CommitTotal = self.used
            info.PageSize = self.page_size
        return int(self.success)


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


class HostCommitUnitTests(unittest.TestCase):
    def _verify_module_contract(self, mod) -> None:
        mib = mod.MIB
        self.assertEqual(mib, 1 << 20)

        # Exact MiB math and 0-used floor.
        self.assertEqual(
            mod.snapshot_from_pages(64, 0, mib),
            {"limit_mib": 64, "used_mib": 0, "available_mib": 64, "available_bytes": 64 * mib},
        )
        self.assertEqual(
            mod.snapshot_from_pages(256, 128, 4096),
            {"limit_mib": 1, "used_mib": 0, "available_mib": 0, "available_bytes": 128 * 4096},
        )
        self.assertEqual(
            mod.snapshot_from_pages(100, 100, 65536),
            {"limit_mib": (100 * 65536) // mib, "used_mib": (100 * 65536) // mib,
             "available_mib": 0, "available_bytes": 0},
        )

        # Clamping when used_pages > limit_pages.
        clamped = mod.snapshot_from_pages(2, 5, mib)
        self.assertEqual(
            clamped,
            {"limit_mib": 2, "used_mib": 5, "available_mib": 0, "available_bytes": 0},
        )

        # Bool / non-int / negative / zero refusals in snapshot_from_pages.
        for bad_args in (
            (True, 0, 4096),
            (False, 0, 4096),
            (10, True, 4096),
            (10, False, 4096),
            (10, 0, True),
            (10, 0, False),
            (0, 0, 4096),
            (-1, 0, 4096),
            (10, 0, 0),
            (10, 0, -4096),
            (10, -1, 4096),
            ("10", 0, 4096),
            (10.0, 0, 4096),
            (10, 0.5, 4096),
            (10, 0, 4096.0),
            (None, 0, 4096),
        ):
            with self.assertRaises(ValueError, msg=f"args={bad_args!r}"):
                mod.snapshot_from_pages(*bad_args)

        # Budget boundary checks (exact pass, 1-byte-short fail, max cap).
        snap = mod.snapshot_from_pages(10, 4, mib)
        self.assertEqual(
            mod.require_commit_budget(6, snap),
            {"limit_mib": 10, "used_mib": 4, "available_mib": 6,
             "minimum_mib": 6, "meets_minimum": True},
        )
        self.assertEqual(mod.require_commit_budget(0, MappingProxyType(snap))["meets_minimum"], True)

        one_short = dict(snap, available_bytes=6 * mib - 1)
        with self.assertRaises(ValueError):
            mod.require_commit_budget(6, one_short)

        huge = mod.snapshot_from_pages((1 << 20) + 1, 0, mib)
        self.assertEqual(mod.require_commit_budget(1 << 20, huge)["minimum_mib"], 1 << 20)
        with self.assertRaises(ValueError):
            mod.require_commit_budget((1 << 20) + 1, huge)

        # Reject bool/non-int/out-of-range minimum_mib.
        for bad_min in (True, False, -1, (1 << 20) + 1, "6", 6.0, None):
            with self.assertRaises(ValueError, msg=f"minimum_mib={bad_min!r}"):
                mod.require_commit_budget(bad_min, snap)

        # Reject non-Mapping snapshot, missing keys, and bool/non-int/negative fields.
        for bad_snap in (None, [], (), "snapshot", 42, [0, 0, 0, 0]):
            with self.assertRaises(ValueError, msg=f"snapshot={bad_snap!r}"):
                mod.require_commit_budget(0, bad_snap)

        required = ("available_bytes", "limit_mib", "used_mib", "available_mib")
        for missing_key in required:
            partial = {k: v for k, v in snap.items() if k != missing_key}
            with self.assertRaises(ValueError, msg=f"missing={missing_key}"):
                mod.require_commit_budget(0, partial)

        for key in required:
            for bad_val in (True, False, -1, "1", 1.5, None):
                bad_row = dict(snap)
                bad_row[key] = bad_val
                with self.assertRaises(ValueError, msg=f"{key}={bad_val!r}"):
                    mod.require_commit_budget(0, bad_row)

        # Mocked read_host_commit success, GetPerformanceInfo==0, and non-win32 OSError.
        fake = _FakeGetPerformanceInfo(limit=512, used=128, page_size=65536, success=True)

        def call_success(ptr, cb_size):
            return fake(ptr, cb_size, mod.PerformanceInformation)

        with patch.object(mod.sys, "platform", "win32"), \
                patch.object(mod.C, "WinDLL", create=True) as win_dll:
            win_dll.return_value.GetPerformanceInfo = call_success
            got = mod.read_host_commit()
        win_dll.assert_called_once_with("psapi.dll", winmode=0x800, use_last_error=True)
        expected_cb = ctypes.sizeof(mod.PerformanceInformation)
        self.assertEqual(fake.calls, [(expected_cb, expected_cb)])
        self.assertEqual(
            got,
            {"limit_mib": (512 * 65536) // mib, "used_mib": (128 * 65536) // mib,
             "available_mib": (384 * 65536) // mib, "available_bytes": 384 * 65536},
        )

        fake_fail = _FakeGetPerformanceInfo(success=False)

        def call_fail(ptr, cb_size):
            return fake_fail(ptr, cb_size, mod.PerformanceInformation)

        with patch.object(mod.sys, "platform", "win32"), \
                patch.object(mod.C, "WinDLL", create=True) as win_dll, \
                patch.object(mod.C, "get_last_error", create=True, return_value=87):
            win_dll.return_value.GetPerformanceInfo = call_fail
            with self.assertRaises(OSError) as raised:
                mod.read_host_commit()
        self.assertEqual(raised.exception.errno, 87)

        for non_win in ("linux", "darwin"):
            with patch.object(mod.sys, "platform", non_win):
                with self.assertRaises(OSError):
                    mod.read_host_commit()

    def test_contract_and_boundaries(self) -> None:
        self._verify_module_contract(host_commit)

    def test_fast_temp_copy_mutations(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for label, target, replacement in MUTATIONS:
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mut_path = tmp_dir / f"host_commit_{label}.py"
                    mut_path.write_text(source.replace(target, replacement, 1), encoding="utf-8")
                    mod_name = f"host_commit_mut_{label}"
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
