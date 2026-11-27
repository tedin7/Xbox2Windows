"""Schema, loader and hash-gate controls for tools/patch_manifest.py on a synthetic PE32+."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

import pefile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "patch_manifest.py"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import patch_manifest  # noqa: E402
from call_evidence_check import SECTION_RVA, build_pe32_plus  # noqa: E402

REPLACEMENT = "90c3"


def manifest_text(input_sha256, original, patches=None):
    if patches is None:
        patches = f"""
[[patch]]
id = "p001"
kind = "memory"
why = "synthetic fixture"
remove_when = "never; this is a test"
rva = {SECTION_RVA:#x}
expect_sha256 = "{hashlib.sha256(original).hexdigest()}"
bytes = "{REPLACEMENT}"
"""
    return f'[target]\ninput_sha256 = "{input_sha256}"\n{patches}'


class PatchManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        image, _, _ = build_pe32_plus()
        self.original = pefile.PE(data=image).get_data(SECTION_RVA, len(REPLACEMENT) // 2)
        self.input = tmp / "input.bin"
        self.input.write_bytes(image)
        self.other = tmp / "other.bin"
        self.other.write_bytes(image[:-1] + bytes([image[-1] ^ 1]))
        self.digest = hashlib.sha256(image).hexdigest()
        self.manifest = tmp / "manifest.toml"
        self.manifest.write_text(manifest_text(self.digest, self.original), encoding="utf-8", newline="\n")

    def cli(self, input_path):
        return subprocess.run([sys.executable, "-B", str(TOOL), str(self.manifest), str(input_path)],
                              capture_output=True, text=True, check=False)

    def test_bound_input_loads(self):
        plan = patch_manifest.load(self.manifest, self.input)
        self.assertEqual(plan["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(plan["input_sha256"], self.digest)
        [patch] = plan["patches"]
        self.assertEqual((patch["id"], patch["rva"], patch["bytes"]), ("p001", SECTION_RVA, bytes.fromhex(REPLACEMENT)))
        result = self.cli(self.input)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"p001"', result.stdout)

    def test_wrong_input_is_refused(self):
        result = self.cli(self.other)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("manifest is for another input", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_malformed_manifests_are_refused(self):
        sha = self.digest
        ok = manifest_text(sha, self.original)
        second = ok.split("[target]", 1)[1].split("\n", 2)[2].replace('"p001"', '"p002"')
        cases = {
            "unknown keys: expect_sha265": ok.replace("expect_sha256 =", "expect_sha265 ="),
            "kind 'hook' is not supported": ok.replace('"memory"', '"hook"'),
            "kind ['memory'] is not supported": ok.replace('"memory"', '["memory"]'),
            "kind {'a': 1} is not supported": ok.replace('"memory"', "{a = 1}"),
            "span [0xffffffff, 0x100000001) does not fit": ok.replace(f"rva = {SECTION_RVA:#x}", "rva = 0xffffffff"),
            "missing: why": ok.replace('why = "synthetic fixture"\n', ""),
            "remove_when must be a non-empty": ok.replace('"never; this is a test"', '" "'),
            "duplicate patch id: p001": ok + ok.split("[target]", 1)[1].split("\n", 2)[2],
            "patches p001 and p002 overlap": ok + second,
            "bytes must be": ok.replace(REPLACEMENT, "90c"),
            "rva must be": ok.replace(f"rva = {SECTION_RVA:#x}", "rva = true"),
            "input_sha256 must be": ok.replace(sha, sha.upper()),
            "at least one [[patch]]": "patch = []\n" + manifest_text(sha, self.original, patches=""),
        }
        for message, text in cases.items():
            with self.subTest(message):
                with self.assertRaises(ValueError) as caught:
                    patch_manifest.parse(text)
                self.assertIn(message, str(caught.exception))
        with self.assertRaises(tomllib.TOMLDecodeError):  # the wording differs between Python versions
            patch_manifest.parse(ok.replace("[target]", "[target"))
        # Shape control for the span check: one byte at the last RVA ends exactly at 2^32.
        last = ok.replace(f"rva = {SECTION_RVA:#x}", "rva = 0xffffffff").replace(REPLACEMENT, "90")
        [patch] = patch_manifest.parse(last)[1]
        self.assertEqual((patch["rva"], patch["bytes"]), (0xFFFFFFFF, b"\x90"))
        # A wrong-type kind must reach the CLI as a diagnostic, not a traceback.
        self.manifest.write_text(ok.replace('"memory"', '["memory"]'), encoding="utf-8", newline="\n")
        result = self.cli(self.input)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("patch manifest: [[patch]] 0: kind ['memory'] is not supported", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
