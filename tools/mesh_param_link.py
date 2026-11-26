"""Parameter export and pixel-input linkage (param0..param31) for mesh shaders (#2413 T4).

AMD RDNA 2 ISA section 10.2 (EXP targets): target 32 + N encodes paramN (0 <= N <= 31).
Desktop D3D12 SM 6.5 allows at most 32 float4 vectors in the vertex output / pixel input
signature (SV_Position, optional clip/cull distance vectors, PARAM0..PARAMk, plus any
flat FPARAM* and binary16 HPARAML*/HPARAMH* shadow semantics required by gfx2hlsl).
"""

from dataclasses import dataclass

MAX_PARAM = 31
MAX_VERTEX_VECTORS = 32


@dataclass(frozen=True)
class ParamLinkage:
    params: tuple
    flat: tuple = ()
    f16_plain: tuple = ()
    pixel_attrs: tuple = ()
    semantic: str = "param"


def _normalize_slots(slots, name):
    out = set()
    for s in slots:
        if type(s) is not int or not 0 <= s <= MAX_PARAM:
            raise ValueError(f"{name} index {s!r} must be an integer in 0..{MAX_PARAM}")
        out.add(s)
    return tuple(sorted(out))


def validate_linkage(
    exported_params,
    *,
    pixel_attrs=None,
    flat=(),
    f16_plain=(),
    distance_vectors=0,
    semantic="param",
):
    """Validate mesh parameter exports against pixel reads and D3D12's 32-vector signature limit."""
    params = _normalize_slots(exported_params, "Mesh parameter")
    flat_slots = _normalize_slots(flat, "Flat attribute")
    f16_slots = _normalize_slots(f16_plain, "Binary16 attribute")
    if semantic not in ("param", "texcoord"):
        raise ValueError(f"Unsupported mesh parameter semantic {semantic!r}")
    if semantic == "texcoord":
        if set(params) - {0} or flat_slots or f16_slots:
            raise ValueError("TEXCOORD0 linkage supports only interpolated param0")
    param_set = set(params)
    if not set(flat_slots) <= param_set:
        raise ValueError(f"Flat attributes {sorted(set(flat_slots) - param_set)} not exported by mesh stage")
    if not set(f16_slots) <= param_set:
        raise ValueError(f"Binary16 attributes {sorted(set(f16_slots) - param_set)} not exported by mesh stage")
    if pixel_attrs is not None:
        attrs = _normalize_slots(pixel_attrs, "Pixel attribute")
        missing = sorted(set(attrs) - param_set)
        if missing:
            raise ValueError(f"Pixel stage reads attributes {missing} the mesh stage does not export")
    else:
        attrs = ()
    param_count = (max(params) + 1) if params else 0
    total_vectors = 1 + distance_vectors + param_count + len(flat_slots) + 2 * len(f16_slots)
    if total_vectors > MAX_VERTEX_VECTORS:
        raise ValueError(
            f"Mesh vertex signature needs {total_vectors} vectors, exceeding D3D12 limit {MAX_VERTEX_VECTORS}"
        )
    return ParamLinkage(params=params, flat=flat_slots, f16_plain=f16_slots, pixel_attrs=attrs, semantic=semantic)


def merge_wave_params(waves, positions):
    """Merge per-wave parameter exports and require every slot to cover every exported vertex."""
    merged = {}
    for wave in waves:
        for slot, vertex_map in wave.params.items():
            dest = merged.setdefault(slot, {})
            for vertex, values in vertex_map.items():
                if vertex in dest:
                    raise ValueError("Duplicate vertex attribute" if slot == 0 else f"Duplicate vertex parameter param{slot}")
                dest[vertex] = values
    expected = set(positions)
    for slot, vertex_map in merged.items():
        if set(vertex_map) != expected:
            raise ValueError("Incomplete vertex attribute" if slot == 0 else f"Incomplete vertex parameter param{slot}")
    return merged


def _use_texcoord(exported_params, linkage):
    if linkage is not None:
        if set(exported_params) != set(linkage.params):
            raise ValueError("Mesh exported parameters do not match declared pixel linkage")
        return linkage.semantic == "texcoord"
    return not any(s > 0 for s in exported_params)


def hlsl_param_fields(exported_params, linkage=None):
    """HLSL Vertex struct fields for the mesh shader's exported parameters."""
    if _use_texcoord(exported_params, linkage):
        return "float4 attr : TEXCOORD0; " if exported_params else ""
    fields = []
    if exported_params:
        for p in range(max(exported_params) + 1):
            fields.append(f"float4 param{p} : PARAM{p};")
    if linkage is not None:
        for p in linkage.flat:
            fields.append(f"nointerpolation uint4 fparam{p} : FPARAM{p};")
        for p in linkage.f16_plain:
            fields.append(f"float4 hparam{p}_lo : HPARAML{p};")
            fields.append(f"float4 hparam{p}_hi : HPARAMH{p};")
    return (" ".join(fields) + " ") if fields else ""


def hlsl_param_writes(vertex_expr, vertex_params, exported_params, float4_fn, linkage=None):
    """HLSL statements assigning exported parameter fields for one vertex."""
    if not exported_params:
        return []
    if _use_texcoord(exported_params, linkage):
        return [f"{vertex_expr}.attr = {float4_fn(vertex_params[0])};"]
    writes = []
    flat = set(linkage.flat) if linkage is not None else set()
    f16_plain = set(linkage.f16_plain) if linkage is not None else set()
    for p in range(max(exported_params) + 1):
        if p not in vertex_params:
            writes.append(f"{vertex_expr}.param{p} = float4(0.0f, 0.0f, 0.0f, 0.0f);")
            continue
        expr = float4_fn(vertex_params[p])
        need_flat = p in flat
        need_f16 = p in f16_plain
        if need_flat or need_f16:
            writes.append(f"float4 p{p} = {expr}; {vertex_expr}.param{p} = p{p};")
            if need_flat:
                writes.append(f"{vertex_expr}.fparam{p} = asuint(p{p});")
            if need_f16:
                writes.append(
                    f"uint4 h{p} = asuint(p{p}); "
                    f"{vertex_expr}.hparam{p}_lo = f16tof32(h{p}); "
                    f"{vertex_expr}.hparam{p}_hi = f16tof32(h{p} >> 16);"
                )
        else:
            writes.append(f"{vertex_expr}.param{p} = {expr};")
    return writes
