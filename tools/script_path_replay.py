"""Replay explicit state paths from one local script tree with a native Windows walker."""
import argparse
import ctypes
import json
import mmap
from pathlib import Path

from script_state_probe import ATTRIBUTE, NODE, joaat, records, state_ids, tree_arrays, verified_camera


def state_index(nodes, attrs):
    states = {}
    children = {}
    for i, (name, parent, _, _, _, first, count, _) in enumerate(nodes):
        if (i == 0 and parent != 65535) or (i and parent >= i):
            raise ValueError("Invalid parent index")
        if count and first + count > len(attrs):
            raise ValueError("Attribute range outside pool")
        children.setdefault(parent, []).append(i)
        if name == joaat("state"):
            ids = state_ids(attrs[first:first + count])
            if len(ids) > 1:
                raise ValueError("Duplicate state ID attribute")
            if ids:
                states[i] = ids[0]
    return states, children


def select_tree(path, anchor):
    selected = None
    with path.open("rb") as stream:
        if stream.read(4) != b"9FPR":
            raise ValueError("Expected an RPF9 archive")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for fields in records(data):
                tree = tree_arrays(fields)
                if tree is None:
                    continue
                states, children = state_index(*tree[:2])
                if anchor not in states.values():
                    continue
                if selected is not None:
                    raise ValueError("Anchor appears in multiple records")
                selected = tree, states, children
    if selected is None:
        raise ValueError("No record contains this anchor")
    return selected


def replay(tree, states, children, paths, host):
    nodes, attrs, vectors = tree
    containers = [i for i in children.get(0, []) if nodes[i][0] == joaat("states")]
    if len(containers) != 1 or nodes[0][0] != joaat("root"):
        raise ValueError("Expected one root states container")
    library = ctypes.CDLL(str(host.resolve()))
    resolve = library.resolve_script_path
    uint = ctypes.c_uint32
    resolve.argtypes = [ctypes.c_void_p, uint, ctypes.c_void_p, uint, uint, ctypes.POINTER(uint),
                        ctypes.c_bool, uint, ctypes.POINTER(uint), uint]
    resolve.restype = ctypes.c_int
    node_rows = ctypes.create_string_buffer(b"".join(NODE.pack(*row) for row in nodes))
    attr_rows = ctypes.create_string_buffer(b"".join(ATTRIBUTE.pack(*row) for row in attrs))
    results = []
    active = uint(containers[0])
    active_keys = []
    for path in paths:
        parts = path.split("/")
        relative = parts[0] in (".", "..")
        up = 0
        if parts[0] == ".":
            parts = parts[1:]
        else:
            while up < len(parts) and parts[up] == "..":
                up += 1
            parts = parts[up:]
        if not parts or any(part in ("", ".", "..") for part in parts):
            raise ValueError("Expected state labels after an optional relative path prefix")
        keys = [joaat(part) for part in parts]
        # The independent reference groups nodes by parent, ignoring their sibling links.
        prefix = active_keys[:len(active_keys) - up] if relative else []
        expected = -1 if up > len(active_keys) else containers[0]
        for key in prefix + keys:
            if expected < 0:
                break
            matches = [i for i in children.get(expected, []) if states.get(i) == key]
            if len(matches) != 1:
                expected = -2 if matches else -1
                break
            expected = matches[0]
        before = active.value
        actual = resolve(node_rows, len(nodes), attr_rows, len(attrs), containers[0], ctypes.byref(active),
                         relative, up, (uint * len(keys))(*keys), len(keys))
        if actual != expected or active.value != (before if expected < 0 else expected):
            raise ValueError(f"Linked traversal returned {actual}, parent index returned {expected} for {path}")
        row = {"path": path, "before": before, "after": active.value}
        if actual < 0:
            row["status"] = "missing" if actual == -1 else "ambiguous"
        else:
            active_keys = prefix + keys
            row.update(status="matched")
            cameras = []
            for i in children.get(actual, []):
                first, count = nodes[i][5:7]
                values = {key: (digest, value, kind) for key, digest, value, kind, _, _ in attrs[first:first + count]}
                camera = verified_camera(values, vectors)
                if camera is not None:
                    cameras.append(camera)
            row["verified_cameras"] = cameras
        results.append(row)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--record-with", required=True, help="State ID unique to one record")
    parser.add_argument("paths", nargs="+", help="State paths relative to the states container")
    args = parser.parse_args(argv)
    try:
        tree, states, children = select_tree(args.archive, joaat(args.record_with))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{args.archive}: {exc}\n")
    try:
        results = replay(tree, states, children, args.paths, args.host)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{args.archive}: {exc}\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
