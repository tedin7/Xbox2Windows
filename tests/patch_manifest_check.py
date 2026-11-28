"""Schema, loader and hash-gate controls for tools/patch_manifest.py on a synthetic PE32+."""

import hashlib
import json
from pathlib import Path
import re
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
# Revisions of each pipeline's form, not the runtime's current ones: the loader checks the form only.
REVISION = {"graphics": "x2w-startup-shader-12", "compute": "x2w-startup-shader-12cs",
            "mesh86": "x2w-startup-mesh-86-4"}


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


def memory_patch(pid, rva, replacement=REPLACEMENT):
    """One [[patch]] of kind memory; parse checks the shape of expect_sha256, not the bytes it hashes."""
    return f"""
[[patch]]
id = "{pid}"
kind = "memory"
why = "synthetic fixture"
remove_when = "never; this is a test"
rva = {rva:#x}
expect_sha256 = "{hashlib.sha256(pid.encode()).hexdigest()}"
bytes = "{replacement}"
"""


def shader_patch(pid="s001", key="00c0ffee0badf00d", pipeline="compute", revision=None, exts=None):
    """One [[patch]] of kind shader; its files are named after the patch and hashed from their names."""
    revision = revision or REVISION[pipeline]
    exts = patch_manifest.PIPELINES[pipeline][2] if exts is None else exts
    files = ", ".join(f'{{ ext = "{ext}", file = "{pid}{ext}", '
                      f'sha256 = "{hashlib.sha256((pid + ext).encode()).hexdigest()}" }}' for ext in exts)
    return f"""
[[patch]]
id = "{pid}"
kind = "shader"
why = "synthetic fixture"
remove_when = "never; this is a test"
key = "{key}"
pipeline = "{pipeline}"
revision = "{revision}"
files = [{files}]
"""


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
        self.assertEqual(patch, {"id": "p001", "kind": "memory", "why": "synthetic fixture",
                                 "remove_when": "never; this is a test", "rva": SECTION_RVA,
                                 "expect_sha256": hashlib.sha256(self.original).hexdigest(),
                                 "bytes": bytes.fromhex(REPLACEMENT)})
        result = self.cli(self.input)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"p001"', result.stdout)

    def test_wrong_input_is_refused(self):
        result = self.cli(self.other)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("manifest is for another input", result.stderr)
        self.assertEqual(result.stdout, "")
        # An input that cannot be opened is a diagnostic too, not a traceback.
        result = self.cli(self.input.with_name("absent.bin"))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("patch manifest: [Errno", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_memory_patches_side_by_side(self):
        # Adjacent spans do not overlap, the order of the [[patch]] tables does not matter, and RVA 0 is in range.
        # The ids follow neither the addresses nor their reverse, so spans sorted by id either way refuse this case.
        text = manifest_text(self.digest, self.original, patches="".join(
            memory_patch(pid, rva) for pid, rva in (("p001", SECTION_RVA + 4), ("p003", SECTION_RVA + 2),
                                                    ("p000", SECTION_RVA), ("p002", 0))))
        _, patches = patch_manifest.parse(text)
        self.assertEqual([(p["id"], p["rva"]) for p in patches],
                         [("p001", SECTION_RVA + 4), ("p003", SECTION_RVA + 2), ("p000", SECTION_RVA), ("p002", 0)])
        # One shared byte is an overlap, in either order.
        for first, second in ((SECTION_RVA, SECTION_RVA + 1), (SECTION_RVA + 1, SECTION_RVA)):
            with self.subTest(first=first, second=second):
                text = manifest_text(self.digest, self.original,
                                     patches=memory_patch("p001", first) + memory_patch("p002", second))
                with self.assertRaises(ValueError) as caught:
                    patch_manifest.parse(text)
                self.assertRegex(str(caught.exception), r"patches p00[12] and p00[12] overlap")

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
            "p001 bytes must be a non-empty": ok.replace(f'bytes = "{REPLACEMENT}"', 'bytes = ""'),
            "rva must be": ok.replace(f"rva = {SECTION_RVA:#x}", "rva = true"),
            "input_sha256 must be": ok.replace(sha, sha.upper()),
            # A hash or key with trailing text is malformed, not a hash of another input.
            "input_sha256 must be 64 lowercase hex digits": ok.replace(sha, sha + "0"),
            "p001 expect_sha256 must be 64": ok.replace(hashlib.sha256(self.original).hexdigest(),
                                                       hashlib.sha256(self.original).hexdigest() + "x"),
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

    def test_load_data_gates_the_bytes_given(self):
        image, other = self.input.read_bytes(), self.other.read_bytes()
        self.assertEqual(patch_manifest.load_data(self.manifest, image), patch_manifest.load(self.manifest, self.input))
        with self.assertRaises(ValueError) as caught:
            patch_manifest.load_data(self.manifest, other)
        self.assertIn("manifest is for another input", str(caught.exception))
        # The bytes in hand decide, not the file they came from: the input file now holds the other image.
        self.input.write_bytes(other)
        self.assertEqual(patch_manifest.load_data(self.manifest, bytearray(image))["input_sha256"], self.digest)
        with self.assertRaises(ValueError):
            patch_manifest.load(self.manifest, self.input)

    def test_shader_patches(self):
        sha = self.digest
        ok = manifest_text(sha, self.original)
        # Accepted beside a memory patch: each pipeline with its required files, mesh86 also with its pixel shader
        # and compute also with its Python translator's HLSL.
        text = (ok + shader_patch("s001") + shader_patch("s002", "0123456789abcdef", "graphics")
                + shader_patch("s003", "fedcba9876543210", "mesh86")
                + shader_patch("s004", "0000000000000001", "mesh86", exts=(".mesh86.txt", ".ms.cso", ".ps.cso"))
                + shader_patch("s005", "0000000000000002", "compute", exts=(".txt", ".cs.cso", ".cs.hlsl")))
        target, patches = patch_manifest.parse(text)
        self.assertEqual([(p["id"], p["kind"]) for p in patches], [("p001", "memory")] + [
            (pid, "shader") for pid in ("s001", "s002", "s003", "s004", "s005")])
        s001 = patches[1]
        self.assertEqual((s001["key"], s001["pipeline"], s001["revision"]),
                         ("00c0ffee0badf00d", "compute", "x2w-startup-shader-12cs"))
        self.assertEqual(s001["files"], [
            {"ext": ".txt", "file": "s001.txt", "sha256": hashlib.sha256(b"s001.txt").hexdigest()},
            {"ext": ".cs.cso", "file": "s001.cs.cso", "sha256": hashlib.sha256(b"s001.cs.cso").hexdigest()}])
        self.assertEqual([f["ext"] for f in patches[4]["files"]], [".mesh86.txt", ".ms.cso", ".ps.cso"])
        self.assertEqual([f["ext"] for f in patches[5]["files"]], [".txt", ".cs.cso", ".cs.hlsl"])
        # The CLI lists shader ids with the memory ones.
        self.manifest.write_text(text, encoding="utf-8", newline="\n")
        result = self.cli(self.input)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["patch_ids"], ["p001", "s001", "s002", "s003", "s004", "s005"])
        good = shader_patch()
        cases = {
            "key must be 16 lowercase hex digits": ok + shader_patch(key="00c0ffee0badf00"),
            "s001 key must be 16 lowercase hex": ok + shader_patch(key="00C0FFEE0BADF00D"),
            "patch s001 key must be 16 lowercase hex digits": ok + shader_patch(key="00c0ffee0badf00d0"),
            "pipeline must be one of: graphics, compute, mesh86": ok + shader_patch().replace('"compute"', '"vertex"'),
            "revision 'x2w-startup-shader-12' is not a compute key revision (x2w-startup-shader-<n>cs)":
                ok + shader_patch(revision="x2w-startup-shader-12"),
            "s002 revision 'x2w-startup-shader-12cs' is not a graphics key revision (x2w-startup-shader-<n>)":
                ok + shader_patch("s002", pipeline="graphics", revision="x2w-startup-shader-12cs"),
            "revision 'x2w-startup-mesh-57-3' is not a mesh86 key revision (x2w-startup-mesh-86-<n>)":
                ok + shader_patch(pipeline="mesh86", revision="x2w-startup-mesh-57-3"),
            "revision 'x2w-startup-shader-cs' is not a compute": ok + shader_patch(revision="x2w-startup-shader-cs"),
            "revision 'x2w-startup-shader-012cs' is not a compute":
                ok + shader_patch(revision="x2w-startup-shader-012cs"),
            "revision 29 is not a compute": ok + shader_patch().replace('"x2w-startup-shader-12cs"', "29"),
            "files must be a non-empty array": ok + good.replace(good[good.index("files = "):], "files = []\n"),
            "files[0] has unknown keys: size": ok + good.replace('{ ext = ".txt",', '{ size = 1, ext = ".txt",'),
            "files[1] is missing: sha256": ok + good.replace(
                f', sha256 = "{hashlib.sha256(b"s001.cs.cso").hexdigest()}"', ""),
            "files[1] ext '.vs.cso' is not a compute translation file (.txt, .cs.cso, .cs.hlsl)":
                ok + shader_patch(exts=(".txt", ".vs.cso")),
            "s001 lists .txt twice": ok + shader_patch(exts=(".txt", ".cs.cso", ".txt")),
            "s001 compute override is missing: .cs.cso": ok + shader_patch(exts=(".txt",)),
            "s002 graphics override is missing: .vs.cso, .ps.cso":
                ok + shader_patch("s002", pipeline="graphics", exts=(".txt",)),
            "files[0] file must be a non-empty string": ok + good.replace('file = "s001.txt"', 'file = ""'),
            **{f"files[0] file {name!r} must be a bare file name":
               ok + good.replace('file = "s001.txt"', f"file = '{name}'")
               for name in ("sub/s001.txt", "sub\\s001.txt", "/s001.txt", "C:s001.txt", "..", ".")},
            # The runtime that reads these files runs on Windows: names it can't hold, shortens or opens as a device.
            **{f"files[0] file {name!r} has a character a Windows file name cannot hold":
               ok + good.replace('file = "s001.txt"', f"file = '{name}'")
               for name in ("a*b", "a?b", "a<b", "a>b", "a|b", 'a"b', "a\tb")},
            "files[0] file 'a\\x01b' has a character": ok + good.replace('file = "s001.txt"', 'file = "a\\u0001b"'),
            **{f"files[0] file {name!r} ends in a dot or a space, which Windows drops":
               ok + good.replace('file = "s001.txt"', f"file = '{name}'") for name in ("s001.txt.", "s001.txt ")},
            **{f"files[0] file {name!r} is a Windows device name":
               ok + good.replace('file = "s001.txt"', f"file = '{name}'")
               for name in ("NUL", "CON.txt", "Com1", "nul .txt", "CONIN$", "conout$.txt",
                            "COM\u00b9", "lpt\u00b3.cso")},
            "files[0] sha256 must be 64 lowercase hex digits": ok + good.replace(
                hashlib.sha256(b"s001.txt").hexdigest(), hashlib.sha256(b"s001.txt").hexdigest().upper()),
            "patches s001 and s002 override the same key 00c0ffee0badf00d": ok + good + shader_patch("s002"),
            "has unknown keys: rva": ok + good.replace('key = "', 'rva = 0x1000\nkey = "'),
            "[[patch]] 1 is missing: files": ok + good[:good.index("files = ")],
            "kind 'shaders' is not supported": ok + good.replace('"shader"', '"shaders"'),
        }
        for message, case in cases.items():
            with self.subTest(message):
                with self.assertRaises(ValueError) as caught:
                    patch_manifest.parse(case)
                self.assertIn(message, str(caught.exception))
        # A device name only as a prefix, a space inside a name and a leading dot are still file names.
        for name in ("console.txt", "nulls.cso", "a b.txt", "com10.txt", "conin.txt", ".s001.txt"):
            with self.subTest(name):
                _, patches = patch_manifest.parse(ok + good.replace('file = "s001.txt"', f"file = '{name}'"))
                self.assertEqual(patches[-1]["files"][0]["file"], name)

    def test_shader_revision_forms_match_the_runtime(self):
        # The runtime's current revision of each pipeline has the loader's form, so a bump (a new <n>) still loads.
        # Whether a manifest names the current revision is for a later consumer to check against the runtime.
        draw = (ROOT / "src/startup_stubs/d3d12_draw.inl").read_text(encoding="utf-8")
        mesh86 = (ROOT / "src/startup_stubs/mesh86_cache.inl").read_text(encoding="utf-8")
        found = {
            "graphics": re.findall(r'constexpr char revision\[\] = "(x2w-startup-shader-[^"]*)";', draw),
            "compute": re.findall(r'constexpr char compute_revision\[\] = "([^"]*)";', draw),
            "mesh86": re.findall(r'constexpr char revision\[\] = "([^"]*)";', mesh86),
        }
        self.assertEqual(set(found), set(patch_manifest.PIPELINES))
        for pipeline, revisions in found.items():
            with self.subTest(pipeline):
                self.assertEqual(len(revisions), 1, revisions)
                self.assertTrue(patch_manifest.REVISIONS[pipeline].fullmatch(revisions[0]), revisions[0])

    def test_shader_files_match_the_runtime(self):
        # The translation files each pipeline's loader in the runtime reads for one key: cache_path(..., L"", key, ext)
        # and mesh86's load_stage(ext, ...). mesh86 reads its .ps.cso only for a request with a pixel shader, and
        # compute looks for its .cs.hlsl only when the .txt has no native line: neither is required.
        reads = re.compile(r'(?:cache_path\(\w+, L"", [\w.]+, |load_stage\()L"(\.[a-z0-9.]+)"')
        sources = {"graphics": "d3d12_draw.inl", "compute": "d3d12_compute_dispatch.inl", "mesh86": "mesh86_cache.inl"}
        self.assertEqual(set(sources), set(patch_manifest.PIPELINES))
        for pipeline, name in sources.items():
            text = (ROOT / "src/startup_stubs" / name).read_text(encoding="utf-8")
            with self.subTest(pipeline):
                self.assertEqual(set(reads.findall(text)), set(patch_manifest.PIPELINES[pipeline][1]))
        mesh86 = (ROOT / "src/startup_stubs/mesh86_cache.inl").read_text(encoding="utf-8")
        self.assertIn('(!request.ps.empty() && !load_stage(L".ps.cso", ps))', mesh86)
        self.assertEqual(set(patch_manifest.PIPELINES["mesh86"][1]) - set(patch_manifest.PIPELINES["mesh86"][2]),
                         {".ps.cso"})
        compute = (ROOT / "src/startup_stubs/d3d12_compute_dispatch.inl").read_text(encoding="utf-8")
        self.assertIn("if (output == x2w::NativeComputeOutput::unmarked) {", compute)
        self.assertEqual(set(patch_manifest.PIPELINES["compute"][1]) - set(patch_manifest.PIPELINES["compute"][2]),
                         {".cs.hlsl"})

    def test_deep_nesting_is_refused(self):
        def nested(depth):
            return "x = " + "[" * depth + "]" * depth + "\n" + manifest_text(self.digest, self.original)

        # Shape control: a shallow array gets past tomllib and is refused only for its unknown key.
        with self.assertRaises(ValueError) as caught:
            patch_manifest.parse(nested(50))
        self.assertIn("manifest has unknown keys: x", str(caught.exception))
        # Twice the recursion limit, set here so a runner that raised it cannot parse the array: refused, not a
        # RecursionError. The CLI runs in a new interpreter, whose limit is the default 1000.
        limit = sys.getrecursionlimit()
        self.addCleanup(sys.setrecursionlimit, limit)
        sys.setrecursionlimit(1000)
        with self.assertRaises(ValueError) as caught:
            patch_manifest.parse(nested(2000))
        self.assertIn("manifest is nested too deeply to parse", str(caught.exception))
        self.manifest.write_text(nested(2000), encoding="utf-8", newline="\n")
        result = self.cli(self.input)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertIn("patch manifest: manifest is nested too deeply to parse", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
