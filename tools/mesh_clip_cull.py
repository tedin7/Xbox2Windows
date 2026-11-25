"""Clip and cull distance exports of a bounded mesh program.

The export alone does not say which position export carries distances: the
PA_CL_VS_OUT_CNTL context register does. References, pinned to Mesa commit
03ecd8b0a517e528d790a3d252194f7b898d0e8f (tag mesa-24.0.0, MIT):

- Field layout: the gfx10 register table,
  https://gitlab.freedesktop.org/mesa/mesa/-/blob/03ecd8b0a517e528d790a3d252194f7b898d0e8f/src/amd/registers/gfx10.json#L13456-13487
  CLIP_DIST_ENA_0..7 in bits 0-7, CULL_DIST_ENA_0..7 in bits 8-15, the USE_VTX_*
  bits, VS_OUT_MISC_VEC_ENA in bit 21 and VS_OUT_CCDIST0/1_VEC_ENA in bits 22
  and 23. radeonsi writes clipdist_mask | culldist_mask << 8 with the same
  component index n, which is component n % 4 of vector CCDIST(n // 4):
  https://gitlab.freedesktop.org/mesa/mesa/-/blob/03ecd8b0a517e528d790a3d252194f7b898d0e8f/src/gallium/drivers/radeonsi/si_state.c#L897-902
- Export order: ac_nir_export_position,
  https://gitlab.freedesktop.org/mesa/mesa/-/blob/03ecd8b0a517e528d790a3d252194f7b898d0e8f/src/amd/common/ac_nir.c#L254-396
  pos0 first, then the misc vector when enabled, then each enabled CCDIST
  vector, numbered without gaps, with done on the last position export.

The misc vector is refused here, so the first enabled CCDIST vector is pos1 and
the second is pos2. That numbering is Mesa's; it has not been checked on
console hardware.

The caller passes the word explicitly. Reading it from pipeline state is not
part of this module.
"""

from dataclasses import dataclass

U32 = (1 << 32) - 1
CLIP_DIST_ENA = 0
CULL_DIST_ENA = 8
VS_OUT_MISC_VEC_ENA = 21
VS_OUT_CCDIST_VEC_ENA = (22, 23)
# Bits this slice refuses by name. Each one changes what the position exports hold.
REFUSED_BITS = {
    16: "USE_VTX_POINT_SIZE",
    17: "USE_VTX_EDGE_FLAG",
    18: "USE_VTX_RENDER_TARGET_INDX",
    19: "USE_VTX_VIEWPORT_INDX",
    20: "USE_VTX_KILL_FLAG",
    21: "VS_OUT_MISC_VEC_ENA",
    24: "VS_OUT_MISC_SIDE_BUS_ENA",
    25: "USE_VTX_GS_CUT_FLAG",
    26: "USE_VTX_SHD_OBJPRIM_ID",
    27: "USE_VTX_LINE_WIDTH",
}
KNOWN_BITS = 0xFFFF | (1 << 22) | (1 << 23) | sum(1 << bit for bit in REFUSED_BITS)
POSITION_TARGETS = ("pos0", "pos1", "pos2", "pos3")
# D3D12 allows at most eight clip and cull distance components in all.
MAX_COMPONENTS = 8


@dataclass(frozen=True)
class DistanceLayout:
    word: int
    # (export target, CCDIST vector) in export order.
    vectors: tuple
    clip: frozenset
    cull: frozenset

    def fields(self):
        """(kind, vector, components): one HLSL field per kind and CCDIST vector."""
        out = []
        for kind, enabled in (("clip", self.clip), ("cull", self.cull)):
            for _, vector in self.vectors:
                components = tuple(c for c in range(4 * vector, 4 * vector + 4) if c in enabled)
                if components:
                    out.append((kind, vector, components))
        return out


def decode_vs_out_cntl(word):
    """The distance layout PA_CL_VS_OUT_CNTL describes, or a refusal that names the field."""
    if type(word) is not int or not 0 <= word <= U32:
        raise ValueError("PA_CL_VS_OUT_CNTL must be a 32-bit unsigned word")
    named = [name for bit, name in sorted(REFUSED_BITS.items()) if word >> bit & 1]
    if named:
        raise ValueError(f"PA_CL_VS_OUT_CNTL {word:#010x} sets {', '.join(named)}; "
                         "only clip and cull distance vectors are translated")
    if word & ~KNOWN_BITS:
        raise ValueError(f"PA_CL_VS_OUT_CNTL {word:#010x} sets unknown bits {word & ~KNOWN_BITS:#010x}")
    clip = frozenset(c for c in range(8) if word >> (CLIP_DIST_ENA + c) & 1)
    cull = frozenset(c for c in range(8) if word >> (CULL_DIST_ENA + c) & 1)
    vectors = []
    for vector, bit in enumerate(VS_OUT_CCDIST_VEC_ENA):
        components = sorted(c for c in clip | cull if c // 4 == vector)
        enabled = bool(word >> bit & 1)
        if enabled and not components:
            raise ValueError(f"PA_CL_VS_OUT_CNTL {word:#010x} sets VS_OUT_CCDIST{vector}_VEC_ENA "
                             f"with no clip or cull distance in components {4 * vector}-{4 * vector + 3}")
        if components and not enabled:
            raise ValueError(f"PA_CL_VS_OUT_CNTL {word:#010x} enables distance components "
                             f"{', '.join(map(str, components))} without VS_OUT_CCDIST{vector}_VEC_ENA")
        if enabled:
            vectors.append((POSITION_TARGETS[1 + len(vectors)], vector))
    if len(clip) + len(cull) > MAX_COMPONENTS:
        raise ValueError(f"PA_CL_VS_OUT_CNTL {word:#010x} enables {len(clip) + len(cull)} clip and cull "
                         f"components; D3D12 allows {MAX_COMPONENTS}")
    return DistanceLayout(word, tuple(vectors), clip, cull)


def vertex_distances(position_exports, layout):
    """Check every vertex's position exports against the layout.

    position_exports maps a vertex to its (target, values, done) exports in
    program order; values is None for pos0 and holds "off" as None. Returns
    {vertex: {(kind, vector): [values]}} in the order of layout.fields().
    """
    expected = ["pos0", *(target for target, _ in layout.vectors)] if layout else ["pos0"]
    out = {}
    for vertex, exports in sorted(position_exports.items()):
        targets = [target for target, _, _ in exports]
        for target in targets:
            if target in expected:
                continue
            if layout is None:
                raise ValueError(f"Mesh export {target} needs the PA_CL_VS_OUT_CNTL word")
            raise ValueError(f"Mesh export {target} is not enabled by PA_CL_VS_OUT_CNTL {layout.word:#010x}")
        if len(set(targets)) != len(targets):
            duplicate = next(t for t in targets if targets.count(t) > 1)
            raise ValueError(f"Duplicate position export {duplicate} for vertex {vertex}")
        for target, vector in layout.vectors if layout else ():
            if target not in targets:
                raise ValueError(f"PA_CL_VS_OUT_CNTL {layout.word:#010x} enables {target} (CCDIST{vector}), "
                                 f"which vertex {vertex} does not export")
        if "pos0" not in targets:
            raise ValueError(f"Distance export for vertex {vertex} without a position")
        if targets != expected:
            raise ValueError(f"Position exports of vertex {vertex} are out of order: {', '.join(targets)}")
        if [done for _, _, done in exports] != [False] * (len(exports) - 1) + [True]:
            if len(exports) == 1:
                raise ValueError("Expected completed primitive or position export")
            raise ValueError(f"Expected done on the last position export {targets[-1]} of vertex {vertex} only")
        if layout is None or not layout.vectors:
            continue
        values = {}
        for target, operands, _ in exports[1:]:
            vector = dict(layout.vectors)[target]
            for c, value in enumerate(operands):
                component = 4 * vector + c
                if component in layout.clip | layout.cull and value is None:
                    kind = "clip" if component in layout.clip else "cull"
                    raise ValueError(f"Enabled {kind} distance {component} is not exported in {target} "
                                     f"of vertex {vertex}")
                values[component] = value
        out[vertex] = {(kind, vector): [values[c] for c in components]
                       for kind, vector, components in layout.fields()}
    return out


SEMANTICS = {"clip": "SV_ClipDistance", "cull": "SV_CullDistance"}


def hlsl_fields(layout):
    """Struct fields for the enabled components, one per kind and CCDIST vector."""
    if layout is None:
        return ""
    return "".join(f"{vector_type(len(components))} {kind}{vector}:{SEMANTICS[kind]}{vector};"
                   for kind, vector, components in layout.fields())


def vector_type(count):
    return "float" if count == 1 else f"float{count}"


def hlsl_value(values):
    parts = [v if isinstance(v, str) else f"asfloat(0x{v:08x}u)" for v in values]
    return parts[0] if len(parts) == 1 else f"{vector_type(len(parts))}({', '.join(parts)})"


def hlsl_writes(target, distances):
    """Assignments of one vertex's distances, e.g. v[3].clip0=float2(...);"""
    return "".join(f"{target}.{kind}{vector}={hlsl_value(values)};"
                   for (kind, vector), values in distances.items())
