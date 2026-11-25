"""Source-built paged record for state hierarchy, typed values and lexical hashes."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import ctypes
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import script_path_replay as replay_mod  # noqa: E402
from script_path_replay import replay, select_tree, state_index  # noqa: E402
from script_state_probe import ATTRIBUTE, NODE, inspect, inspect_tree, joaat  # noqa: E402


def fixture(*, damage_value=False, self_parent=False, priorities=(9,), duplicate_attr=False,
            id_kind=5, parent_tag="state", drop=None, root_parent=65535, root_schema_size: int = 80):
    example = "synthetic_orbit"
    def attr(name, spelling, raw, kind):
        return struct.pack("<IIIBBH", joaat(name), joaat(spelling), raw, kind, 1, 0)

    attrs = bytearray()
    nodes = bytearray()

    def node(name, parent, attributes):
        index = len(nodes) // 16
        nodes.extend(struct.pack("<I5H2B", joaat(name), parent, 0, 0, 0,
                                 len(attrs) // 16, len(attributes), 1))
        attrs.extend(b"".join(attributes))
        return index

    if len(priorities) > 1:
        root_parent = node("root", root_parent, [])
    for priority in priorities:
        ids = [attr("id", example, 0, id_kind)]
        if duplicate_attr:
            ids.append(attr("id", "other", 0, 5))
        state = node(parent_tag, root_parent, [] if drop == 0 else ids)
        camera = [
            attr("priority", str(priority), priority, 3),
            attr("blendduration", "37", 37, 3),
            attr("orbitdistance", "1.500000", 0x40000000 if damage_value else 0x3fc00000, 2),
            attr("pitchlimits", "-0.250000,0.500000,0.000000", 0, 1),
            attr("headinglimits", "-1.000000,1.000000,0.000000", 1, 1),
        ]
        node("fixture_child", state + int(self_parent),
             [row for i, row in enumerate(camera, 1) if i != drop])
    vecs = struct.pack("<6f", -0.25, 0.5, 0, -1, 1, 0)
    arrays = ((len(attrs)//16, 2), (0, 0), (2, 4), (0, 0), (len(nodes)//16, 3))
    root = b"".join(struct.pack("<IIQ", n, n, ref) for n, ref in arrays)
    blocks = (root, attrs, nodes, vecs)
    instances = bytearray()
    for kind, blob in enumerate(blocks, 1):
        instances.extend(struct.pack("<IIQ", kind, len(blob), 0xaaaaaaaaaaaaaaaa))
    for blob in blocks:
        instances.extend(bytes(-(len(instances) + 8) % 16))
        instances.extend(blob)
    definitions = ((root_schema_size, (("attributes", 0, 0x5c),)),
                   (16, (("name", 0, 0x4a), ("valueHash", 4, 0x4a))),
                   (16, (("name", 0, 0x4a), ("parentIndex", 4, 0x13), ("attributeCount", 14, 0x11))),
                   (12, tuple((s, i * 4, 0x14) for i, s in enumerate("xyz"))))
    schema = b"".join(struct.pack("<IIIIQIHH", kind, 0, 0, 0, 0, size, 0, len(fields))
                      for kind, (size, fields) in enumerate(definitions, 1))
    schema += b"".join(struct.pack("<IIBBHI", joaat(s), off, code, 0, 0, 0)
                       for _, fields in definitions for s, off, code in fields)
    header = bytearray(80)
    header[:5] = b"0DRPz"
    struct.pack_into("<I", header, 16, 1)
    struct.pack_into("<3H", header, 56, 4, 0, 4)
    body = bytearray()
    for tag, part in ((b"PSIN", instances), (b"LSCH", schema), (b"LENM", b""), (b"LSNS", bytes(65536)),
                      (b"LGID", b""), (b"LRSC", header)):
        body += struct.pack(">4sI", tag, len(part) + 8) + part
    body += struct.pack(">4sII8x", b"CHKS", 20, len(body) + 20)
    archive = bytearray(b"9FPR")
    for i in range(0, len(body), 65536):
        archive += bytes(-len(archive) % 16)
        archive += zlib.compress(body[i:i + 65536])
    return archive


class DuplicateStateIdTests(unittest.TestCase):
    def test_duplicate_state_id_is_rejected(self):
        # Two label-kind id attributes on one state: the replay raises, so the probe must not let the last win.
        nodes = [(joaat("state"), 65535, 0, 0, 0, 0, 2, 1)]
        attrs = [(joaat("id"), joaat(label), 0, 5, 1, 0) for label in ("first", "second")]
        with self.assertRaisesRegex(ValueError, "Duplicate state ID"):
            inspect_tree(nodes, attrs, b"", joaat("second"))


class ScriptStateTests(unittest.TestCase):
    def run_probe(self, state="synthetic_orbit", **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.rpf"
            path.write_bytes(fixture(**options))
            return inspect(path, state)

    def test_linked_values(self):
        result = self.run_probe()
        self.assertEqual(result["counts"]["all_five_literals_verified"], 1)
        self.assertEqual(result["first_verified_example"]["orbitdistance"], "1.500000")

    def test_different_binary_value(self):
        result = self.run_probe(damage_value=True)
        self.assertEqual(result["counts"]["camera_parameter_sets"], 1)
        self.assertIsNone(result["first_verified_example"])

    def test_ambiguous_states_in_either_order(self):
        for priorities in ((9, 17), (17, 9)):
            result = self.run_probe(priorities=priorities)
            self.assertIsNone(result["first_verified_example"])
            matches = result["ambiguous_state_candidates"]
            self.assertEqual([m["node_path"] for m in matches], [[0, 1], [0, 3]])
            self.assertEqual([m["record_index"] for m in matches], [0, 0])
            self.assertEqual([m["first_verified_example"]["priority"] for m in matches],
                             list(map(str, priorities)))

    def test_matches_across_records(self):
        first = fixture()
        second = fixture(priorities=(17,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two.rpf"
            path.write_bytes(first + bytes(-len(first) % 16) + second[16:])
            result = inspect(path, "synthetic_orbit")
            self.assertNotIn("ambiguous_state_candidates", inspect(path, "missing"))
        self.assertIsNone(result["first_verified_example"])
        self.assertEqual([(m["record_index"], m["node_path"])
                          for m in result["ambiguous_state_candidates"]], [(0, [0]), (1, [0])])

    def test_mismatched_root_schema_does_not_select_state(self) -> None:
        mismatched = fixture(root_schema_size=64)
        valid = fixture(priorities=(17,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two.rpf"
            path.write_bytes(mismatched + bytes(-len(mismatched) % 16) + valid[16:])
            with self.subTest(consumer="probe"):
                result = inspect(path, "synthetic_orbit")
                self.assertEqual(result["counts"]["section_records"], 2)
                self.assertEqual(result["counts"]["node_trees"], 1)
                self.assertEqual(result["first_verified_example"]["priority"], "17")
            with self.subTest(consumer="replay"):
                tree, _, _ = select_tree(path, joaat("synthetic_orbit"))
                priorities = [value for name, _, value, _, _, _ in tree[1] if name == joaat("priority")]
                self.assertEqual(priorities, [17])

    def test_duplicate_id_attributes(self):
        with self.assertRaisesRegex(ValueError, "Duplicate state ID attribute"):
            self.run_probe(duplicate_attr=True)

    def test_parent_cycle(self):
        with self.assertRaisesRegex(ValueError, "parent index"):
            self.run_probe(self_parent=True)

    def test_non_label_state_id_is_malformed(self):
        # An integer or boolean id on a state is malformed, as resolve_script_path reports -3 for it.
        for kind in (3, 4):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Non-label state ID"):
                self.run_probe(id_kind=kind)

    def test_other_label_selects_nothing(self):
        result = self.run_probe(state="synthetic_other")
        self.assertEqual(result["counts"]["all_five_literals_verified"], 1)
        self.assertIsNone(result["first_verified_example"])

    def test_parent_must_be_a_state(self):
        counts = self.run_probe(parent_tag="zone")["counts"]
        self.assertNotIn("camera_parameter_sets", counts)
        self.assertNotIn("state_nodes", counts)

    def test_missing_camera_attribute(self):
        result = self.run_probe(drop=3)
        self.assertNotIn("camera_parameter_sets", result["counts"])
        self.assertIsNone(result["first_verified_example"])

    def test_root_with_parent(self):
        with self.assertRaisesRegex(ValueError, "parent index"):
            self.run_probe(root_parent=0)


class CommandLineTests(unittest.TestCase):
    def run_tool(self, script, *args):
        return subprocess.run([sys.executable, "-B", str(TOOLS / script), *map(str, args)],
                              capture_output=True, text=True)

    def test_damaged_input_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty, text = root / "empty.bin", root / "text.bin"
            empty.write_bytes(b"")
            text.write_bytes(b"not an archive")
            for archive in (empty, text, root / "missing.bin"):
                for script, args in (("script_state_probe.py", ()),
                                     ("script_path_replay.py", ("--host", root / "host.dll",
                                                                "--record-with", "anchor", "path"))):
                    with self.subTest(archive=archive.name, script=script):
                        result = self.run_tool(script, archive, *args)
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertIn(archive.name, result.stderr)
                        self.assertNotIn("Traceback", result.stderr)

    def test_replay_failure_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "replay.bin"
            archive.write_bytes(fixture(priorities=(9, 8)))
            result = self.run_tool("script_path_replay.py", archive, "--host", root / "host.dll",
                                   "--record-with", "synthetic_orbit", "a//b")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(archive.name, result.stderr)
            self.assertIn("Expected one root states container", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_non_ascii_state_label_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "text.bin"
            archive.write_bytes(b"9FPR")
            for script, args in (("script_state_probe.py", ("--state", "café")),
                                 ("script_path_replay.py", ("--host", "host.dll", "--record-with", "café", "path"))):
                with self.subTest(script=script):
                    result = self.run_tool(script, archive, *args)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


def _replay_archive_fixture(*, root_tag="root", container_count=1, extra_records=1):
    """Build a synthetic RPF9 archive with linked-sibling state hierarchy and camera children."""
    NO_NODE = 0xFFFF

    def attr(name, spelling, raw, kind):
        return struct.pack("<IIIBBH", joaat(name), joaat(spelling), raw, kind, 1, 0)

    def camera_attrs(priority):
        return [
            attr("priority", str(priority), priority, 3),
            attr("blendduration", "37", 37, 3),
            attr("orbitdistance", "1.500000", 0x3FC00000, 2),
            attr("pitchlimits", "-0.250000,0.500000,0.000000", 0, 1),
            attr("headinglimits", "-1.000000,1.000000,0.000000", 1, 1),
        ]

    raw_nodes = []
    attrs = bytearray()

    def add_node(name, parent, attributes):
        idx = len(raw_nodes)
        first = len(attrs) // 16
        attrs.extend(b"".join(attributes))
        raw_nodes.append([joaat(name), parent, NO_NODE, NO_NODE, NO_NODE, first, len(attributes), 1])
        return idx

    def link_child(parent, child):
        if parent == NO_NODE:
            return
        head = raw_nodes[parent][2]
        if head == NO_NODE:
            raw_nodes[parent][2] = child
            return
        cur = head
        while raw_nodes[cur][3] != NO_NODE:
            cur = raw_nodes[cur][3]
        raw_nodes[cur][3] = child
        raw_nodes[child][4] = cur

    root_idx = add_node(root_tag, NO_NODE, [])
    containers = [add_node("states", root_idx, []) for _ in range(container_count)]
    for c in containers:
        link_child(root_idx, c)

    if containers:
        c0 = containers[0]
        alpha = add_node("state", c0, [attr("id", "alpha", 0, 5)])
        link_child(c0, alpha)
        beta = add_node("state", alpha, [attr("id", "beta", 0, 5)])
        link_child(alpha, beta)
        beta_cam = add_node("fixture_child", beta, camera_attrs(9))
        link_child(beta, beta_cam)

        gamma = add_node("state", c0, [attr("id", "gamma", 0, 5)])
        link_child(c0, gamma)
        gamma_cam = add_node("fixture_child", gamma, camera_attrs(17))
        link_child(gamma, gamma_cam)
        dup1 = add_node("state", gamma, [attr("id", "dup", 0, 5)])
        dup2 = add_node("state", gamma, [attr("id", "dup", 0, 5)])
        link_child(gamma, dup1)
        link_child(gamma, dup2)

    nodes = b"".join(NODE.pack(*row) for row in raw_nodes)
    vecs = struct.pack("<6f", -0.25, 0.5, 0, -1, 1, 0)
    arrays = ((len(attrs) // 16, 2), (0, 0), (2, 4), (0, 0), (len(nodes) // 16, 3))
    root = b"".join(struct.pack("<IIQ", n, n, ref) for n, ref in arrays)
    blocks = (root, attrs, nodes, vecs)
    instances = bytearray()
    for kind, blob in enumerate(blocks, 1):
        instances.extend(struct.pack("<IIQ", kind, len(blob), 0xAAAAAAAAAAAAAAAA))
    for blob in blocks:
        instances.extend(bytes(-(len(instances) + 8) % 16))
        instances.extend(blob)
    definitions = (
        (80, (("attributes", 0, 0x5C),)),
        (16, (("name", 0, 0x4A), ("valueHash", 4, 0x4A))),
        (16, (("name", 0, 0x4A), ("parentIndex", 4, 0x13), ("attributeCount", 14, 0x11))),
        (12, tuple((s, i * 4, 0x14) for i, s in enumerate("xyz"))),
    )
    schema = b"".join(
        struct.pack("<IIIIQIHH", kind, 0, 0, 0, 0, size, 0, len(fields))
        for kind, (size, fields) in enumerate(definitions, 1)
    )
    schema += b"".join(
        struct.pack("<IIBBHI", joaat(s), off, code, 0, 0, 0)
        for _, fields in definitions
        for s, off, code in fields
    )
    header = bytearray(80)
    header[:5] = b"0DRPz"
    struct.pack_into("<I", header, 16, 1)
    struct.pack_into("<3H", header, 56, 4, 0, 4)
    body = bytearray()
    for tag, part in (
        (b"PSIN", instances),
        (b"LSCH", schema),
        (b"LENM", b""),
        (b"LSNS", bytes(65536)),
        (b"LGID", b""),
        (b"LRSC", header),
    ):
        body += struct.pack(">4sI", tag, len(part) + 8) + part
    body += struct.pack(">4sII8x", b"CHKS", 20, len(body) + 20)

    def compress_record():
        chunk = bytearray()
        for i in range(0, len(body), 65536):
            chunk += bytes(-len(chunk) % 16)
            chunk += zlib.compress(body[i:i + 65536])
        return chunk

    archive = bytearray(b"9FPR")
    for _ in range(extra_records):
        archive += bytes(-len(archive) % 16)
        archive += compress_record()
    return bytes(archive)


def _mock_cdll_factory(*, override_return=None, skip_active_update=False):
    """Return a fake CDLL modeling the native resolve_script_path contract.

    This pure-Python mock models tools/script_path_host.cpp so script_state_check
    stays platform-independent; tests/script_path_check.py exercises replay()
    against the real compiled script_path_host library on Windows.
    """
    STATE_HASH = joaat("state")
    ID_HASH = joaat("id")
    NO_NODE = 0xFFFF

    def fake_resolve(node_rows, node_count, attr_rows, attr_count, container, active_ptr, relative, up, keys_arr, depth):
        if override_return is not None:
            return override_return
        nodes = [NODE.unpack_from(bytes(node_rows), i * NODE.size) for i in range(node_count)]
        attrs = [ATTRIBUTE.unpack_from(bytes(attr_rows), i * ATTRIBUTE.size) for i in range(attr_count)]
        active_val = ctypes.cast(active_ptr, ctypes.POINTER(ctypes.c_uint32))
        parent = active_val.contents.value if relative else container
        for _ in range(up):
            if parent == container:
                return -1
            nxt = nodes[parent][1]
            if nxt >= parent:
                return -3
            parent = nxt
        for key in list(keys_arr)[:depth]:
            previous = NO_NODE
            match = NO_NODE
            ambiguous = False
            cur = nodes[parent][2]
            while cur != NO_NODE:
                node = nodes[cur]
                if node[1] != parent or node[4] != previous:
                    return -3
                previous = cur
                if node[0] == STATE_HASH:
                    first, count = node[5], node[6]
                    seen_id = False
                    for a in attrs[first:first + count]:
                        if a[0] != ID_HASH:
                            continue
                        if seen_id or a[3] != 5:
                            return -3
                        seen_id = True
                        if a[1] == key:
                            if match != NO_NODE:
                                ambiguous = True
                            else:
                                match = cur
                cur = node[3]
            if ambiguous:
                return -2
            if match == NO_NODE:
                return -1
            parent = match
        if not skip_active_update:
            active_val.contents.value = parent
        return int(parent)

    class _FakeLib:
        def __init__(self, _path):
            self.resolve_script_path = fake_resolve

    return _FakeLib


class ScriptPathReplayPositiveTests(unittest.TestCase):
    def test_select_tree_state_index_and_replay_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "tree.rpf"
            archive.write_bytes(_replay_archive_fixture())
            host = root / "host.dll"

            tree, states, children = select_tree(archive, joaat("alpha"))
            self.assertEqual(len(tree[0]), 9)
            self.assertEqual(states[2], joaat("alpha"))
            self.assertEqual(states[3], joaat("beta"))
            self.assertEqual(states[5], joaat("gamma"))
            self.assertEqual(children[0], [1])
            self.assertEqual(children[1], [2, 5])

            s_map, c_map = state_index(tree[0], tree[1])
            self.assertEqual(s_map, states)
            self.assertEqual(c_map, children)

            paths = ["alpha", "./beta", "../../../gamma", "../../gamma", "gamma/dup", "missing", "alpha/beta"]
            with mock.patch.object(replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory()):
                results = replay(tree, states, children, paths, host)

            self.assertEqual([r["path"] for r in results], paths)
            self.assertEqual([r["status"] for r in results], [
                "matched",
                "matched",
                "missing",
                "matched",
                "ambiguous",
                "missing",
                "matched",
            ])
            self.assertEqual(results[0]["after"], 2)
            self.assertEqual(results[1]["before"], 2)
            self.assertEqual(results[1]["after"], 3)
            self.assertEqual(results[1]["verified_cameras"][0]["priority"], "9")
            self.assertEqual(results[1]["verified_cameras"][0]["orbitdistance"], "1.500000")
            # Failed lookups must leave active unchanged.
            self.assertEqual((results[2]["before"], results[2]["after"]), (3, 3))
            self.assertEqual(results[3]["after"], 5)
            self.assertEqual(results[3]["verified_cameras"][0]["priority"], "17")
            self.assertEqual((results[4]["before"], results[4]["after"]), (5, 5))
            self.assertEqual((results[5]["before"], results[5]["after"]), (5, 5))
            self.assertEqual(results[6]["after"], 3)

    def test_main_cli_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "tree.rpf"
            archive.write_bytes(_replay_archive_fixture())
            host = root / "host.dll"

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory()),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                rc = replay_mod.main([
                    str(archive),
                    "--host", str(host),
                    "--record-with", "alpha",
                    "alpha",
                    "./beta",
                    "../../gamma",
                ])

            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload), 3)
            self.assertEqual([row["status"] for row in payload], ["matched", "matched", "matched"])
            self.assertEqual(payload[1]["verified_cameras"][0]["priority"], "9")
            self.assertEqual(payload[2]["verified_cameras"][0]["priority"], "17")


class ScriptPathReplayNegativeTests(unittest.TestCase):
    def test_select_tree_rejects_bad_magic_and_anchor_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_magic = root / "bad.rpf"
            bad_magic.write_bytes(b"NOPE" + b"\x00" * 32)
            with self.assertRaisesRegex(ValueError, "Expected an RPF9 archive"):
                select_tree(bad_magic, joaat("alpha"))

            single = root / "single.rpf"
            single.write_bytes(_replay_archive_fixture(extra_records=1))
            with self.assertRaisesRegex(ValueError, "No record contains this anchor"):
                select_tree(single, joaat("absent_anchor"))

            duplicate = root / "dup.rpf"
            duplicate.write_bytes(_replay_archive_fixture(extra_records=2))
            with self.assertRaisesRegex(ValueError, "Anchor appears in multiple records"):
                select_tree(duplicate, joaat("alpha"))

    def test_state_index_rejects_invalid_parent_attrs_and_duplicate_id(self):
        with self.assertRaisesRegex(ValueError, "Invalid parent index"):
            state_index([(joaat("root"), 0, 0, 0, 0, 0, 0, 1)], [])
        with self.assertRaisesRegex(ValueError, "Invalid parent index"):
            state_index([
                (joaat("root"), 65535, 0, 0, 0, 0, 0, 1),
                (joaat("state"), 1, 0, 0, 0, 0, 0, 1),
            ], [])
        with self.assertRaisesRegex(ValueError, "Attribute range outside pool"):
            state_index([(joaat("root"), 65535, 0, 0, 0, 0, 1, 1)], [])
        with self.assertRaisesRegex(ValueError, "Duplicate state ID attribute"):
            attrs = [
                (joaat("id"), joaat("a"), 0, 5, 1, 0),
                (joaat("id"), joaat("b"), 0, 5, 1, 0),
            ]
            state_index([(joaat("state"), 65535, 0, 0, 0, 0, 2, 1)], attrs)

    def test_replay_rejects_bad_containers_root_paths_and_native_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host.dll"

            for opts, desc in (
                ({"container_count": 0}, "missing states container"),
                ({"container_count": 2}, "multiple states containers"),
                ({"root_tag": "not_root"}, "non-root node 0"),
            ):
                with self.subTest(case=desc):
                    archive = root / f"{desc.replace(' ', '_')}.rpf"
                    archive.write_bytes(_replay_archive_fixture(**opts))
                    with mock.patch.object(replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory()):
                        if opts.get("container_count", 1):
                            tree, states, children = select_tree(archive, joaat("alpha"))
                        else:
                            tree = ([(joaat("root"), 65535, 0xFFFF, 0xFFFF, 0xFFFF, 0, 0, 1)], [], b"")
                            states = {}
                            children = {65535: [0]}
                        if opts.get("container_count") == 2:
                            self.assertEqual(len(children[0]), 2)
                        if opts.get("root_tag") == "not_root":
                            self.assertEqual(len(children[0]), 1)
                            self.assertNotEqual(tree[0][0][0], joaat("root"))
                        with self.assertRaisesRegex(ValueError, "Expected one root states container"):
                            replay(tree, states, children, ["alpha"], host)

            valid = root / "valid.rpf"
            valid.write_bytes(_replay_archive_fixture())
            tree, states, children = select_tree(valid, joaat("alpha"))

            for bad_path in ("", "./", "a//b", "a/.", "a/..", ".", ".."):
                with self.subTest(bad_path=bad_path), mock.patch.object(
                    replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory()
                ):
                    with self.assertRaisesRegex(ValueError, "Expected state labels"):
                        replay(tree, states, children, [bad_path], host)

            with mock.patch.object(
                replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory(override_return=-1)
            ):
                with self.assertRaisesRegex(ValueError, "Linked traversal returned -1"):
                    replay(tree, states, children, ["alpha"], host)

            with mock.patch.object(
                replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory(skip_active_update=True)
            ):
                with self.assertRaisesRegex(ValueError, "Linked traversal returned 2"):
                    replay(tree, states, children, ["alpha"], host)

            # Relative '..' escaping past current depth must report missing even when
            # negative slicing (active_keys[:len - up]) would wrap to a valid prefix.
            with mock.patch.object(replay_mod.ctypes, "CDLL", side_effect=_mock_cdll_factory()):
                over_up = replay(tree, states, children, ["../alpha", "alpha", "../../gamma"], host)
            self.assertEqual([r["status"] for r in over_up], ["missing", "matched", "missing"])
            self.assertEqual((over_up[0]["before"], over_up[0]["after"]), (1, 1))
            self.assertEqual((over_up[2]["before"], over_up[2]["after"]), (2, 2))

    def test_main_cli_rejects_missing_and_invalid_inputs_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host.dll"
            missing = root / "missing.rpf"
            invalid = root / "invalid.rpf"
            invalid.write_bytes(b"BAD!" + b"\x00" * 16)
            valid = root / "valid.rpf"
            valid.write_bytes(_replay_archive_fixture())

            cases = (
                ([str(missing), "--host", str(host), "--record-with", "alpha", "alpha"], "missing.rpf", None),
                ([str(invalid), "--host", str(host), "--record-with", "alpha", "alpha"], "Expected an RPF9 archive", None),
                ([str(valid), "--host", str(host), "--record-with", "absent", "alpha"], "No record contains this anchor", None),
                ([str(valid), "--host", str(root / "missing_host.dll"), "--record-with", "alpha", "alpha"], "valid.rpf", None),
                ([str(valid), "--host", str(host), "--record-with", "alpha", "alpha"], "cannot load host DLL", OSError("cannot load host DLL")),
                ([str(valid), "--host", str(host), "--record-with", "alpha", "a//b"], "Expected state labels", _mock_cdll_factory()),
                ([str(valid), "--host", str(host), "--record-with", "alpha", "alpha"], "Linked traversal returned -1", _mock_cdll_factory(override_return=-1)),
            )
            for argv, expected_msg, cdll_side_effect in cases:
                with self.subTest(expected=expected_msg):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    cm = (
                        mock.patch.object(replay_mod.ctypes, "CDLL", side_effect=cdll_side_effect)
                        if cdll_side_effect is not None
                        else nullcontext()
                    )
                    with cm, redirect_stdout(stdout), redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as exc:
                            replay_mod.main(argv)
                    self.assertEqual(exc.exception.code, 1)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertIn(expected_msg, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
