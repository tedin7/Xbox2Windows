"""Verify state-linked camera settings in paged PSIN records of a local script archive."""
import argparse
from collections import Counter
import json
import math
import mmap
import os
from pathlib import Path
import re
import struct
import zlib

from section_probe import sections


def joaat(text):
    value = 0
    for byte in text.encode("ascii"):
        value = (value + byte) & 0xffffffff
        value = (value + (value << 10)) & 0xffffffff
        value ^= value >> 6
    value = (value + (value << 3)) & 0xffffffff
    value ^= value >> 11
    return (value + (value << 15)) & 0xffffffff


CAMERA = {joaat(s): s for s in ("blendduration", "orbitdistance", "priority", "pitchlimits", "headinglimits")}
NODE = struct.Struct("<I5H2B")
ATTRIBUTE = struct.Struct("<IIIBBH")
PAGE = 65536


def page(data, offset):
    decoder = zlib.decompressobj()
    sample = data[offset:offset + PAGE + 1024]
    try:
        plain = decoder.decompress(sample, PAGE + 1)
    except zlib.error:
        return None
    if decoder.eof and len(plain) <= PAGE:
        return plain, offset + len(sample) - len(decoder.unused_data)
    return None


def records(data):
    for marker in re.finditer(rb"\x78[\x01\x5e\x9c\xda]", data):
        start = marker.start()
        if start % 16:
            continue
        current = page(data, start)
        if not current or not current[0].startswith(b"PSIN"):
            continue
        body = bytearray()
        while current and len(body) < 4 << 20:
            part, end = current
            body.extend(part)
            layout = sections(body)
            if layout == ("PSIN", "LSCH", "LENM", "LSNS", "LGID", "LRSC", "CHKS"):
                fields = {}
                offset = 0
                for tag in layout:
                    size, = struct.unpack_from(">I", body, offset + 4)
                    fields[tag] = memoryview(body)[offset + 8:offset + size]
                    offset += size
                if len(fields["CHKS"]) == 12 and struct.unpack_from(">I", fields["CHKS"])[0] == len(body):
                    yield fields
                break
            if len(part) != PAGE:
                break
            current = page(data, (end + 15) & ~15)


def tree_arrays(fields):
    """Accept the five-array root only when the schema and instance directories agree."""
    header = fields["LRSC"]
    if len(header) != 80 or bytes(header[:5]) != b"0DRPz":
        return None
    root_id, = struct.unpack_from("<I", header, 16)
    struct_count, _, block_count = struct.unpack_from("<3H", header, 56)
    data = fields["PSIN"]
    schema = fields["LSCH"]
    if not 0 < root_id <= block_count or 16 * block_count > len(data) or 32 * struct_count > len(schema):
        raise ValueError("Invalid instance or schema directory")
    blocks = []
    pos = 16 * block_count
    for kind, size, marker in struct.iter_unpack("<IIQ", data[:pos]):
        pos = ((pos + 8 + 15) & ~15) - 8
        if marker != 0xaaaaaaaaaaaaaaaa or pos + size > len(data):
            raise ValueError("Invalid instance block")
        blocks.append((kind, data[pos:pos + size]))
        pos += size
    root_kind, root = blocks[root_id - 1]
    if len(root) != 80:
        return None
    schemas = {}
    pos = 32 * struct_count
    for kind, _, _, _, _, size, _, count in struct.iter_unpack("<IIIIQIHH", schema[:pos]):
        end = pos + count * 16
        if end > len(schema):
            raise ValueError("Truncated schema members")
        members = {(name, offset, code) for name, offset, code, _, _, _ in
                   struct.iter_unpack("<IIBBHI", schema[pos:end])}
        schemas[kind] = size, members
        pos = end
    root_size, root_members = schemas.get(root_kind, (0, set()))
    if root_size != len(root) or (joaat("attributes"), 0, 0x5c) not in root_members:
        return None

    def array(slot, stride, required):
        count, capacity, ref = struct.unpack_from("<IIQ", root, 16 * slot)
        if (count, capacity, ref) == (0, 0, 0):
            return b""
        if not 0 < ref <= block_count or count != capacity:
            raise ValueError("Invalid root array reference")
        kind, block = blocks[ref - 1]
        size, members = schemas.get(kind, (0, set()))
        if size != stride or len(block) != count * stride or not required <= members:
            raise ValueError("Array type, extent or member mismatch")
        return block

    nodes = array(4, NODE.size, {(joaat("parentIndex"), 4, 0x13), (joaat("attributeCount"), 14, 0x11),
                                 (joaat("name"), 0, 0x4a)})
    attrs = array(0, ATTRIBUTE.size, {(joaat("name"), 0, 0x4a), (joaat("valueHash"), 4, 0x4a)})
    # These scalar slots hold the bit patterns of the original float literals.
    vectors = array(2, 12, {(joaat(s), 4 * i, 0x14) for i, s in enumerate("xyz")})
    return list(NODE.iter_unpack(nodes)), list(ATTRIBUTE.iter_unpack(attrs)), vectors


def literal(value, kind, vectors):
    if kind == 3:
        return str(struct.unpack("<i", struct.pack("<I", value))[0])
    if kind == 2:
        values = struct.unpack("<f", struct.pack("<I", value))
    elif kind == 1 and value < len(vectors) // 12:
        values = struct.unpack_from("<3f", vectors, value * 12)
    else:
        return None
    if all(math.isfinite(x) for x in values):
        return ",".join(f"{x:.6f}" for x in values)
    return None


def verified_camera(values, vectors):
    if not CAMERA.keys() <= values.keys():
        return None
    verified = {}
    for key, label in CAMERA.items():
        digest, value, kind = values[key]
        spelling = literal(value, kind, vectors)
        if spelling is None or joaat(spelling) != digest:
            return None
        verified[label] = spelling
    return verified


def state_ids(rows):
    """Return a state's id digests, rejecting any id that is not label-kind (5) as resolve_script_path does."""
    ids = []
    for key, digest, _, kind, _, _ in rows:
        if key == joaat("id"):
            if kind != 5:
                raise ValueError("Non-label state ID attribute")
            ids.append(digest)
    return ids


def inspect_tree(nodes, attrs, vectors, state_id, record_index=0):
    counts = Counter()
    candidates = {}
    states = set()
    for index, (name, parent, _, _, _, first, count, _) in enumerate(nodes):
        if (index == 0 and parent != 65535) or (index and parent >= index):
            raise ValueError("Invalid parent index")
        if count and first + count > len(attrs):
            raise ValueError("Attribute range outside pool")
        values = {key: (digest, value, kind) for key, digest, value, kind, _, _ in attrs[first:first + count]}
        if name == joaat("state"):
            counts["state_nodes"] += 1
            states.add(index)
            ids = state_ids(attrs[first:first + count])
            if len(ids) > 1:
                raise ValueError("Duplicate state ID attribute")
            if ids and ids[0] == state_id:
                path = [index]
                while path[-1] != 0:
                    path.append(nodes[path[-1]][1])
                candidates[index] = {"record_index": record_index, "node_path": path[::-1],
                                     "first_verified_example": None}
        if parent not in states or not CAMERA.keys() <= values.keys():
            continue
        counts["camera_parameter_sets"] += 1
        verified = verified_camera(values, vectors)
        if verified is not None:
            counts["all_five_literals_verified"] += 1
            if parent in candidates and candidates[parent]["first_verified_example"] is None:
                candidates[parent]["first_verified_example"] = verified
    return counts, list(candidates.values())


def inspect(path, state=None):
    totals = Counter()
    candidates = []
    state_id = joaat(state) if state is not None else None
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            raise ValueError("Empty archive; expected an RPF9 archive")
        if stream.read(4) != b"9FPR":
            raise ValueError("Expected an RPF9 archive")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for record_index, fields in enumerate(records(data)):
                totals["section_records"] += 1
                tree = tree_arrays(fields)
                if tree is None:
                    continue
                found, matches = inspect_tree(*tree, state_id, record_index)
                totals.update(found)
                totals["node_trees"] += 1
                candidates.extend(matches)
    witness = candidates[0]["first_verified_example"] if len(candidates) == 1 else None
    result = {"counts": dict(totals), "requested_state": state, "first_verified_example": witness}
    if len(candidates) > 1:
        result["ambiguous_state_candidates"] = candidates
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--state", help="Report at most one hash-verified example for this state label")
    args = parser.parse_args(argv)
    try:
        result = inspect(args.archive, args.state)
    except (OSError, ValueError, struct.error, zlib.error) as exc:
        parser.exit(1, f"{args.archive}: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
