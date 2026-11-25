"""Synthetic (id, type) state dispatch fixture exercised against script_path_host.dll.

Models the structural pattern observed in serialized script state libraries and
shop-component carriers without embedding any game data:
  - A `states` container holds a `default` state at node 2 (depth 1), a sibling
    pair sharing one `id` digest with distinct `type` attributes (`item` vs
    `weaponitem`) leading to distinct leaf shapes (a vector preview-offset leaf
    vs a deeper state chain with camera parameters), and a 3-level branch pair
    (`location` -> `vehiclesize` -> `angle`) selecting distinct camera settings.
  - A separate `shopcomponents`-style carrier tree holds an `item`-named node
    with a kind-5 `weapon` attribute whose `textHash` is the state `id`: the
    carrier node's NAME supplies `type`, while the attribute's `textHash`
    supplies `id`. A nearby synthetic `weaponitem`-named carrier supplies the
    same `id` with the alternate `type` to select the sibling state.

LIMITS (preserved explicitly):
  1. `(id, type)` dispatch is inferred from serialized PSIN structure, not from
     runtime executable code.
  2. The reference native resolver (`resolve_script_path` in
     `tools/script_path_host.cpp`) is intentionally `id`-only, so returning -2
     on sibling `id` collisions is conservative; `filter_siblings_by_type`
     exercises that path via a caller-side pre-filter, while
     `resolve_script_typed_path` accepts caller-provided `type` hash arrays to
     disambiguate `(id, type)` siblings natively in-place.
  3. The native runtime event or caller that supplies `(id, type)` at runtime
     is still unidentified.
  4. A synthetic tree is a reproducible semantic model, not evidence of runtime
     behavior. In particular, the nearby `weaponitem`-named carrier node is a
     synthetic control: in local shop-component records, `weapon` attributes
     sit on `item`-named nodes while `weaponitem` appears as a `state` `type`.
"""
import ctypes
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from script_state_probe import ATTRIBUTE, NODE, joaat, literal, verified_camera

NONE = 0xffff
DEFAULT_HOST = Path(__file__).resolve().parents[1] / "out" / "build" / "dev" / "script_path_host.dll"
HOST = DEFAULT_HOST


def _load_symbol(host, symbol, extra_argtypes=()):
    library = ctypes.CDLL(str(Path(host).resolve()))
    resolve = getattr(library, symbol)
    u32 = ctypes.c_uint32
    resolve.argtypes = [
        ctypes.c_void_p, u32, ctypes.c_void_p, u32, u32, ctypes.POINTER(u32),
        ctypes.c_bool, u32, *extra_argtypes, u32,
    ]
    resolve.restype = ctypes.c_int
    return resolve


def load_resolver(host):
    return _load_symbol(host, "resolve_script_path", (ctypes.POINTER(ctypes.c_uint32),))


def load_typed_resolver(host):
    u32_ptr = ctypes.POINTER(ctypes.c_uint32)
    return _load_symbol(host, "resolve_script_typed_path", (u32_ptr, u32_ptr))


def _pack_tree(nodes, attrs):
    node_buf = ctypes.create_string_buffer(b"".join(NODE.pack(*row) for row in nodes))
    attr_buf = ctypes.create_string_buffer(b"".join(ATTRIBUTE.pack(*row) for row in attrs))
    return node_buf, attr_buf


def resolve_path(resolver, nodes, attrs, container, active, path_hashes, *, relative=False, up=0):
    u32 = ctypes.c_uint32
    node_buf, attr_buf = _pack_tree(nodes, attrs)
    active_var = u32(active)
    path_arr = (u32 * len(path_hashes))(*path_hashes)
    rc = resolver(
        node_buf, len(nodes), attr_buf, len(attrs), container, ctypes.byref(active_var),
        relative, up, path_arr, len(path_hashes),
    )
    return rc, active_var.value


def resolve_typed_path(
    resolver, nodes, attrs, container, active, path_hashes, type_hashes, *, relative=False, up=0
):
    u32 = ctypes.c_uint32
    node_buf, attr_buf = _pack_tree(nodes, attrs)
    active_var = u32(active)
    path_arr = (u32 * len(path_hashes))(*path_hashes) if path_hashes is not None else None
    type_arr = (u32 * len(type_hashes))(*type_hashes) if type_hashes is not None else None
    depth = len(path_hashes) if path_hashes is not None else 0
    rc = resolver(
        node_buf, len(nodes), attr_buf, len(attrs), container, ctypes.byref(active_var),
        relative, up, path_arr, type_arr, depth,
    )
    return rc, active_var.value


def filter_siblings_by_type(nodes, attrs, parent, type_hash):
    """Return a copy of `nodes` keeping only `state` children of `parent` whose
    single kind-5 `type` attribute matches `type_hash`.

    Original node indices and descendant subtrees are preserved untouched, and
    the remaining children of `parent` keep valid doubly-linked sibling pointers
    so the unmodified native `resolve_script_path` validates every link.
    """
    state_hash = joaat("state")
    type_attr_hash = joaat("type")
    out = [list(row) for row in nodes]
    kept = []
    cur = out[parent][2]  # firstChild
    seen = set()
    while cur != NONE:
        if cur in seen or cur >= len(out):
            raise ValueError("Invalid sibling chain during type pre-filter")
        seen.add(cur)
        nxt = out[cur][3]  # nextSibling
        row = out[cur]
        if row[0] == state_hash:
            first, count = row[5], row[6]
            matches = [
                a[1] for a in attrs[first:first + count]
                if a[0] == type_attr_hash and a[3] == 5
            ]
            keep = (len(matches) == 1 and matches[0] == type_hash)
        else:
            keep = True
        if keep:
            kept.append(cur)
        else:
            row[3] = NONE
            row[4] = NONE
        cur = nxt

    out[parent][2] = kept[0] if kept else NONE
    for pos, idx in enumerate(kept):
        out[idx][4] = kept[pos - 1] if pos > 0 else NONE
        out[idx][3] = kept[pos + 1] if pos + 1 < len(kept) else NONE
    return [tuple(row) for row in out]


def extract_carrier_pair(nodes, attrs, node_index, attr_name_hash):
    """Extract a candidate `(id_hash, type_hash)` pair from a carrier node.

    Carrier direction modeled from serialized shop-component records:
      - `type_hash` comes from the carrier node's `name` (`nodes[node_index][0]`)
      - `id_hash` comes from the kind-5 attribute named `attr_name_hash`
    """
    name, _, _, _, _, first, count, _ = nodes[node_index]
    ids = [
        text_hash for attr_name, text_hash, _, kind, _, _ in attrs[first:first + count]
        if attr_name == attr_name_hash and kind == 5
    ]
    if len(ids) != 1:
        raise ValueError(f"Expected 1 kind-5 attribute 0x{attr_name_hash:08x}, found {len(ids)}")
    return ids[0], name


def _attr_hash(name, value_label, *, kind=5, raw=0):
    return (joaat(name), joaat(value_label), raw, kind, 1, 0)


def _attr_int(name, value):
    spelling = str(value)
    raw = struct.unpack("<I", struct.pack("<i", value))[0]
    return (joaat(name), joaat(spelling), raw, 3, 1, 0)


def _attr_float(name, value):
    spelling = f"{value:.6f}"
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    return (joaat(name), joaat(spelling), raw, 2, 1, 0)


def _attr_vec3(name, vec3, vectors):
    index = len(vectors) // 12
    vectors.extend(struct.pack("<3f", *vec3))
    spelling = ",".join(f"{x:.6f}" for x in vec3)
    return (joaat(name), joaat(spelling), index, 1, 1, 0)


def _camera_attrs(blend, orbit, priority, pitch, heading, vectors):
    return [
        _attr_int("blendduration", blend),
        _attr_float("orbitdistance", orbit),
        _attr_int("priority", priority),
        _attr_vec3("pitchlimits", pitch, vectors),
        _attr_vec3("headinglimits", heading, vectors),
    ]


def build_states_library(*, malformed_bool_id=False):
    """Build a 16-node synthetic `states` library in 16-byte wire format.

    Layout:
      0: root
      1: states (container)
      2: state(id='default', type='default') -> 3: camera_params (baseline)
      4: state(id='trigger_a', type='item') -> 5: preview_offset (vector leaf)
      6: state(id='trigger_a', type='weaponitem')
           -> 7: state(id='detail_view', type='subview') -> 8: camera_params
      9: state(id='site_north', type='location')
           -> 10: state(id='compact', type='vehiclesize')
                -> 11: state(id='inspect', type='angle') -> 12: camera_params
           -> 13: state(id='heavy', type='vehiclesize')
                -> 14: state(id='inspect', type='angle') -> 15: camera_params
    """
    nodes = []
    attrs = []
    vectors = bytearray()

    def add_node(tag, parent, child, sibling, previous, node_attrs=()):
        first = len(attrs) if node_attrs else 0
        attrs.extend(node_attrs)
        nodes.append((joaat(tag), parent, child, sibling, previous, first, len(node_attrs), 1))

    # 0..1: root -> states
    add_node("root", NONE, 1, NONE, NONE)
    add_node("states", 0, 2, NONE, NONE)

    # 2..3: default state at node 2 depth 1
    add_node("state", 1, 3, 4, NONE, [_attr_hash("id", "default"), _attr_hash("type", "default")])
    add_node("camera_params", 2, NONE, NONE, NONE,
             _camera_attrs(500, 3.0, 1, (-0.3, 0.4, 0.0), (-1.0, 1.0, 0.0), vectors))

    # 4..5: sibling #1 with id='trigger_a', type='item' -> preview_offset vector leaf
    id_kind = 4 if malformed_bool_id else 5
    id_raw = 1 if malformed_bool_id else 0
    add_node("state", 1, 5, 6, 2, [
        _attr_hash("id", "trigger_a", kind=id_kind, raw=id_raw),
        _attr_hash("type", "item"),
    ])
    add_node("preview_offset", 4, NONE, NONE, NONE, [
        _attr_vec3("position", (0.10, 0.0, -0.25), vectors),
        _attr_vec3("rotation", (0.0, 0.50, 0.0), vectors),
    ])

    # 6..8: sibling #2 with id='trigger_a', type='weaponitem' -> deeper state -> camera leaf
    add_node("state", 1, 7, 9, 4, [
        _attr_hash("id", "trigger_a"),
        _attr_hash("type", "weaponitem"),
    ])
    add_node("state", 6, 8, NONE, NONE, [
        _attr_hash("id", "detail_view"),
        _attr_hash("type", "subview"),
    ])
    add_node("camera_params", 7, NONE, NONE, NONE,
             _camera_attrs(150, 1.75, 3, (-0.2, 0.2, 0.0), (-0.5, 0.5, 0.0), vectors))

    # 9..15: 3-level branch pair (location -> vehiclesize -> angle -> camera_params)
    add_node("state", 1, 10, NONE, 6, [
        _attr_hash("id", "site_north"),
        _attr_hash("type", "location"),
    ])
    add_node("state", 9, 11, 13, NONE, [
        _attr_hash("id", "compact"),
        _attr_hash("type", "vehiclesize"),
    ])
    add_node("state", 10, 12, NONE, NONE, [
        _attr_hash("id", "inspect"),
        _attr_hash("type", "angle"),
    ])
    add_node("camera_params", 11, NONE, NONE, NONE,
             _camera_attrs(0, 3.5, 2, (-0.4, -0.4, 0.0), (0.0, 0.0, 0.0), vectors))
    add_node("state", 9, 14, NONE, 10, [
        _attr_hash("id", "heavy"),
        _attr_hash("type", "vehiclesize"),
    ])
    add_node("state", 13, 15, NONE, NONE, [
        _attr_hash("id", "inspect"),
        _attr_hash("type", "angle"),
    ])
    add_node("camera_params", 14, NONE, NONE, NONE,
             _camera_attrs(0, 5.5, 2, (-0.4, -0.4, 0.0), (0.0, 0.0, 0.0), vectors))

    return nodes, attrs, bytes(vectors)


def build_shop_carrier():
    """Build a 7-node synthetic `shopcomponents`-style carrier tree.

    Layout:
      0: root -> 1: shopcomponents -> 2: shop_group -> 3: data -> 4: entries
      5: item(standflags=0, weapon='trigger_a')       [positive carrier]
      6: weaponitem(standflags=0, weapon='trigger_a') [synthetic nearby control]
    """
    nodes = []
    attrs = []

    def add_node(tag, parent, child, sibling, previous, node_attrs=()):
        first = len(attrs) if node_attrs else 0
        attrs.extend(node_attrs)
        nodes.append((joaat(tag), parent, child, sibling, previous, first, len(node_attrs), 1))

    add_node("root", NONE, 1, NONE, NONE)
    add_node("shopcomponents", 0, 2, NONE, NONE)
    add_node("shop_group", 1, 3, NONE, NONE)
    add_node("data", 2, 4, NONE, NONE)
    add_node("entries", 3, 5, NONE, NONE)
    add_node("item", 4, NONE, 6, NONE, [
        _attr_int("standflags", 0),
        _attr_hash("weapon", "trigger_a"),
    ])
    add_node("weaponitem", 4, NONE, NONE, 5, [
        _attr_int("standflags", 0),
        _attr_hash("weapon", "trigger_a"),
    ])
    return nodes, attrs


def direct_children(nodes, parent):
    result = []
    cur = nodes[parent][2]
    while cur != NONE:
        result.append(cur)
        cur = nodes[cur][3]
    return result


def read_camera_leaves(nodes, attrs, vectors, state_index):
    cameras = []
    for child in direct_children(nodes, state_index):
        first, count = nodes[child][5:7]
        values = {k: (h, v, kind) for k, h, v, kind, _, _ in attrs[first:first + count]}
        cam = verified_camera(values, vectors)
        if cam is not None:
            cameras.append((child, cam))
    return cameras


def read_vector_leaf(nodes, attrs, vectors, state_index, leaf_name):
    target_hash = joaat(leaf_name)
    for child in direct_children(nodes, state_index):
        if nodes[child][0] != target_hash:
            continue
        first, count = nodes[child][5:7]
        decoded = {}
        for k, h, v, kind, _, _ in attrs[first:first + count]:
            spelling = literal(v, kind, vectors)
            if spelling is not None and joaat(spelling) == h:
                decoded[k] = spelling
        return child, decoded
    return None, {}


class ScriptDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve = load_resolver(HOST)

    def test_default_and_id_only_collision_returns_ambiguous(self):
        nodes, attrs, vectors = build_states_library()
        # Activate 'default' at node 2 first so we can verify atomic preservation on error.
        rc, active = resolve_path(self.resolve, nodes, attrs, 1, 1, [joaat("default")])
        self.assertEqual((rc, active), (2, 2))
        cams = read_camera_leaves(nodes, attrs, vectors, 2)
        self.assertEqual(len(cams), 1)
        self.assertEqual((cams[0][0], cams[0][1]["orbitdistance"]), (3, "3.000000"))

        # ID-only lookup of 'trigger_a' collides across sibling nodes 4 and 6 -> -2, active preserved.
        rc, active = resolve_path(self.resolve, nodes, attrs, 1, active, [joaat("trigger_a")])
        self.assertEqual((rc, active), (-2, 2))

    def test_type_aware_dispatch_selects_positive_and_nearby_state(self):
        nodes, attrs, vectors = build_states_library()
        trigger_id = joaat("trigger_a")

        # Positive case: (id='trigger_a', type='item') -> node 4 -> preview_offset leaf 5.
        item_view = filter_siblings_by_type(nodes, attrs, 1, joaat("item"))
        # Dropped siblings must be unreachable: only node 4 survives under container 1.
        self.assertEqual(direct_children(item_view, 1), [4])
        rc, active = resolve_path(self.resolve, item_view, attrs, 1, 1, [joaat("default")])
        self.assertEqual((rc, active), (-1, 1))
        rc, active = resolve_path(self.resolve, item_view, attrs, 1, 1, [trigger_id])
        self.assertEqual((rc, active), (4, 4))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 4), [])
        leaf_idx, vecs = read_vector_leaf(nodes, attrs, vectors, 4, "preview_offset")
        self.assertEqual(leaf_idx, 5)
        self.assertEqual(vecs[joaat("position")], "0.100000,0.000000,-0.250000")
        self.assertEqual(vecs[joaat("rotation")], "0.000000,0.500000,0.000000")

        # Nearby input: same id='trigger_a', different type='weaponitem' -> node 6 -> state 7 -> cam 8.
        weapon_view = filter_siblings_by_type(nodes, attrs, 1, joaat("weaponitem"))
        self.assertEqual(direct_children(weapon_view, 1), [6])
        rc, active = resolve_path(self.resolve, weapon_view, attrs, 1, active, [trigger_id])
        self.assertEqual((rc, active), (6, 6))
        # Descend through the deeper chain both in one filtered call and via unmodified relative step.
        rc_deep, active_deep = resolve_path(
            self.resolve, weapon_view, attrs, 1, 1, [trigger_id, joaat("detail_view")]
        )
        rc_rel, active_rel = resolve_path(
            self.resolve, nodes, attrs, 1, active, [joaat("detail_view")], relative=True, up=0
        )
        self.assertEqual((rc_deep, active_deep), (7, 7))
        self.assertEqual((rc_rel, active_rel), (7, 7))
        cams = read_camera_leaves(nodes, attrs, vectors, 7)
        self.assertEqual(len(cams), 1)
        self.assertEqual((cams[0][0], cams[0][1]["orbitdistance"], cams[0][1]["priority"]),
                         (8, "1.750000", "3"))

        # Negative control: same id with unmatched type -> -1 (missing), active preserved.
        miss_view = filter_siblings_by_type(nodes, attrs, 1, joaat("unmatched_type"))
        rc_miss, active_miss = resolve_path(self.resolve, miss_view, attrs, 1, active_rel, [trigger_id])
        self.assertEqual((rc_miss, active_miss), (-1, 7))

    def test_shop_carrier_supplies_id_from_attr_and_type_from_node_name(self):
        lib_nodes, lib_attrs, lib_vecs = build_states_library()
        shop_nodes, shop_attrs = build_shop_carrier()

        # Positive carrier (node 5): node name='item', attr weapon='trigger_a' -> state 4 (leaf 5).
        pos_id, pos_type = extract_carrier_pair(shop_nodes, shop_attrs, 5, joaat("weapon"))
        self.assertEqual((pos_id, pos_type), (joaat("trigger_a"), joaat("item")))
        pos_view = filter_siblings_by_type(lib_nodes, lib_attrs, 1, pos_type)
        rc_pos, active = resolve_path(self.resolve, pos_view, lib_attrs, 1, 1, [pos_id])
        self.assertEqual((rc_pos, active), (4, 4))
        self.assertEqual(read_vector_leaf(lib_nodes, lib_attrs, lib_vecs, 4, "preview_offset")[0], 5)

        # Nearby carrier (node 6): node name='weaponitem', same attr weapon='trigger_a' -> state 6 -> 7 (cam 8).
        alt_id, alt_type = extract_carrier_pair(shop_nodes, shop_attrs, 6, joaat("weapon"))
        self.assertEqual((alt_id, alt_type), (joaat("trigger_a"), joaat("weaponitem")))
        alt_view = filter_siblings_by_type(lib_nodes, lib_attrs, 1, alt_type)
        rc_alt, active = resolve_path(self.resolve, alt_view, lib_attrs, 1, active, [alt_id])
        self.assertEqual((rc_alt, active), (6, 6))
        rc_sub, active = resolve_path(
            self.resolve, lib_nodes, lib_attrs, 1, active, [joaat("detail_view")], relative=True, up=0
        )
        self.assertEqual((rc_sub, active), (7, 7))
        self.assertEqual(read_camera_leaves(lib_nodes, lib_attrs, lib_vecs, 7)[0][0], 8)

    def test_deeper_chain_positive_and_divergent_branches(self):
        nodes, attrs, vectors = build_states_library()
        p_compact = [joaat("site_north"), joaat("compact"), joaat("inspect")]
        p_heavy = [joaat("site_north"), joaat("heavy"), joaat("inspect")]

        rc1, active = resolve_path(self.resolve, nodes, attrs, 1, 1, p_compact)
        self.assertEqual((rc1, active), (11, 11))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 11)[0], (
            12,
            {
                "blendduration": "0",
                "orbitdistance": "3.500000",
                "priority": "2",
                "pitchlimits": "-0.400000,-0.400000,0.000000",
                "headinglimits": "0.000000,0.000000,0.000000",
            },
        ))

        rc2, active = resolve_path(self.resolve, nodes, attrs, 1, active, p_heavy)
        self.assertEqual((rc2, active), (14, 14))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 14)[0][1]["orbitdistance"], "5.500000")

        # Relative branch switch: ../../compact/inspect restores node 11 and orbitdistance 3.500000.
        rc3, active = resolve_path(
            self.resolve, nodes, attrs, 1, active, [joaat("compact"), joaat("inspect")], relative=True, up=2
        )
        self.assertEqual((rc3, active), (11, 11))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 11)[0][1]["orbitdistance"], "3.500000")

    def test_malformed_non_hash_id_returns_error_and_preserves_active(self):
        nodes, attrs, _ = build_states_library(malformed_bool_id=True)
        # Start from active=2 ('default'); scanning siblings under container=1 hits kind=4 id on node 4 -> -3.
        rc, active = resolve_path(self.resolve, nodes, attrs, 1, 2, [joaat("trigger_a")])
        self.assertEqual((rc, active), (-3, 2))


def _build_typed_collision_tree(*, missing_type=False, duplicate_pair=False,
                                bad_type_kind=False, duplicate_type_attr=False):
    """Build a minimal `states` tree for resolve_script_typed_path edge cases."""
    nodes = []
    attrs = []

    def add_node(tag, parent, child, sibling, previous, node_attrs=()):
        first = len(attrs) if node_attrs else 0
        attrs.extend(node_attrs)
        nodes.append((joaat(tag), parent, child, sibling, previous, first, len(node_attrs), 1))

    add_node("root", NONE, 1, NONE, NONE)
    add_node("states", 0, 2, NONE, NONE)
    add_node("state", 1, NONE, 3, NONE, [_attr_hash("id", "default"), _attr_hash("type", "default")])

    node3_attrs = [_attr_hash("id", "trigger_a")]
    if bad_type_kind:
        node3_attrs.append(_attr_int("type", 42))
    elif not missing_type:
        node3_attrs.append(_attr_hash("type", "item"))
    if duplicate_type_attr:
        node3_attrs.append(_attr_hash("type", "item"))

    if duplicate_pair:
        add_node("state", 1, NONE, 4, 2, node3_attrs)
        add_node("state", 1, NONE, 5, 3, [_attr_hash("id", "trigger_a"), _attr_hash("type", "item")])
        add_node("state", 1, NONE, NONE, 4, [_attr_hash("id", "trigger_a"), _attr_hash("type", "weaponitem")])
    else:
        add_node("state", 1, NONE, NONE, 2, node3_attrs)
    return nodes, attrs


class TypedScriptPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve_typed = load_typed_resolver(HOST)

    def test_sibling_collision_resolved_by_type_hash_and_shop_carrier(self):
        nodes, attrs, vectors = build_states_library()
        shop_nodes, shop_attrs = build_shop_carrier()
        trigger_id = joaat("trigger_a")

        # Direct typed lookup resolves colliding sibling 4 ('item') without pre-filtering.
        rc, active = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, 1, [trigger_id], [joaat("item")]
        )
        self.assertEqual((rc, active), (4, 4))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 4), [])
        leaf_idx, vecs = read_vector_leaf(nodes, attrs, vectors, 4, "preview_offset")
        self.assertEqual(leaf_idx, 5)
        self.assertEqual(vecs[joaat("position")], "0.100000,0.000000,-0.250000")

        # Alternate type selects colliding sibling 6 ('weaponitem') -> relative subview 7 -> cam 8.
        rc, active = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, active, [trigger_id], [joaat("weaponitem")]
        )
        self.assertEqual((rc, active), (6, 6))
        rc_rel, active_rel = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, active,
            [joaat("detail_view")], [joaat("subview")], relative=True, up=0,
        )
        self.assertEqual((rc_rel, active_rel), (7, 7))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 7)[0][0], 8)

        # Shop-component carriers feed (id, type) directly into resolve_script_typed_path.
        pos_id, pos_type = extract_carrier_pair(shop_nodes, shop_attrs, 5, joaat("weapon"))
        alt_id, alt_type = extract_carrier_pair(shop_nodes, shop_attrs, 6, joaat("weapon"))
        rc_pos, active = resolve_typed_path(self.resolve_typed, nodes, attrs, 1, 1, [pos_id], [pos_type])
        self.assertEqual((rc_pos, active), (4, 4))
        rc_alt, active = resolve_typed_path(self.resolve_typed, nodes, attrs, 1, active, [alt_id], [alt_type])
        self.assertEqual((rc_alt, active), (6, 6))

    def test_multilevel_hierarchy_paths_and_relative_switch(self):
        nodes, attrs, vectors = build_states_library()

        rc_deep, active = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, 1,
            [joaat("trigger_a"), joaat("detail_view")],
            [joaat("weaponitem"), joaat("subview")],
        )
        self.assertEqual((rc_deep, active), (7, 7))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 7)[0][0], 8)

        p_compact = [joaat("site_north"), joaat("compact"), joaat("inspect")]
        t_hierarchy = [joaat("location"), joaat("vehiclesize"), joaat("angle")]
        p_heavy = [joaat("site_north"), joaat("heavy"), joaat("inspect")]

        rc1, active = resolve_typed_path(self.resolve_typed, nodes, attrs, 1, active, p_compact, t_hierarchy)
        self.assertEqual((rc1, active), (11, 11))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 11)[0][1]["orbitdistance"], "3.500000")

        rc2, active = resolve_typed_path(self.resolve_typed, nodes, attrs, 1, active, p_heavy, t_hierarchy)
        self.assertEqual((rc2, active), (14, 14))
        self.assertEqual(read_camera_leaves(nodes, attrs, vectors, 14)[0][1]["orbitdistance"], "5.500000")

        rc3, active = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, active,
            [joaat("compact"), joaat("inspect")], [joaat("vehiclesize"), joaat("angle")],
            relative=True, up=2,
        )
        self.assertEqual((rc3, active), (11, 11))

    def test_missing_type_attribute_and_mismatched_type_return_missing(self):
        # Missing type attribute on candidate matching id -> -1, active preserved.
        nodes_no_type, attrs_no_type = _build_typed_collision_tree(missing_type=True)
        rc, active = resolve_typed_path(
            self.resolve_typed, nodes_no_type, attrs_no_type, 1, 2,
            [joaat("trigger_a")], [joaat("item")],
        )
        self.assertEqual((rc, active), (-1, 2))

        # Mismatched type hash at single and deeper hierarchy levels -> -1, active preserved.
        nodes, attrs, _ = build_states_library()
        rc_miss, active_miss = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, 2, [joaat("trigger_a")], [joaat("unmatched_type")]
        )
        self.assertEqual((rc_miss, active_miss), (-1, 2))
        rc_deep_miss, active_deep_miss = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, 2,
            [joaat("site_north"), joaat("compact")], [joaat("location"), joaat("angle")],
        )
        self.assertEqual((rc_deep_miss, active_deep_miss), (-1, 2))

    def test_duplicate_identical_id_and_type_siblings_return_ambiguous(self):
        nodes, attrs = _build_typed_collision_tree(duplicate_pair=True)
        rc, active = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, 2, [joaat("trigger_a")], [joaat("item")]
        )
        self.assertEqual((rc, active), (-2, 2))

        # Distinct (id, type) sibling under the same parent still resolves uniquely.
        rc_ok, active_ok = resolve_typed_path(
            self.resolve_typed, nodes, attrs, 1, 2, [joaat("trigger_a")], [joaat("weaponitem")]
        )
        self.assertEqual((rc_ok, active_ok), (5, 5))

    def test_out_of_bounds_and_malformed_inputs_return_malformed(self):
        base_nodes, base_attrs, _ = build_states_library()

        # Out-of-bounds sibling index -> -3.
        oob_node = [list(row) for row in base_nodes]
        oob_node[4][3] = len(oob_node)
        rc, active = resolve_typed_path(
            self.resolve_typed, [tuple(r) for r in oob_node], base_attrs, 1, 2,
            [joaat("trigger_a")], [joaat("item")],
        )
        self.assertEqual((rc, active), (-3, 2))

        # Out-of-bounds attribute span -> -3.
        oob_attr = [list(row) for row in base_nodes]
        oob_attr[4][5] = len(base_attrs) - 1
        oob_attr[4][6] = 2
        rc, active = resolve_typed_path(
            self.resolve_typed, [tuple(r) for r in oob_attr], base_attrs, 1, 2,
            [joaat("trigger_a")], [joaat("item")],
        )
        self.assertEqual((rc, active), (-3, 2))

        # Out-of-bounds container / active, null types, and zero depth -> -3.
        self.assertEqual(
            resolve_typed_path(
                self.resolve_typed, base_nodes, base_attrs, len(base_nodes), 2,
                [joaat("default")], [joaat("default")],
            ),
            (-3, 2),
        )
        self.assertEqual(
            resolve_typed_path(
                self.resolve_typed, base_nodes, base_attrs, 1, len(base_nodes),
                [joaat("default")], [joaat("default")],
            ),
            (-3, len(base_nodes)),
        )
        self.assertEqual(
            resolve_typed_path(
                self.resolve_typed, base_nodes, base_attrs, 1, 2,
                [joaat("default")], None,
            ),
            (-3, 2),
        )
        self.assertEqual(
            resolve_typed_path(
                self.resolve_typed, base_nodes, base_attrs, 1, 2, [], []
            ),
            (-3, 2),
        )

        # Malformed non-label `type`, duplicate `type`, and non-label `id` -> -3.
        for kwargs in ({"bad_type_kind": True}, {"duplicate_type_attr": True}):
            with self.subTest(**kwargs):
                bad_nodes, bad_attrs = _build_typed_collision_tree(**kwargs)
                self.assertEqual(
                    resolve_typed_path(
                        self.resolve_typed, bad_nodes, bad_attrs, 1, 2,
                        [joaat("trigger_a")], [joaat("item")],
                    ),
                    (-3, 2),
                )

        bool_nodes, bool_attrs, _ = build_states_library(malformed_bool_id=True)
        self.assertEqual(
            resolve_typed_path(
                self.resolve_typed, bool_nodes, bool_attrs, 1, 2,
                [joaat("trigger_a")], [joaat("item")],
            ),
            (-3, 2),
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        HOST = Path(sys.argv.pop(1))
    unittest.main()
