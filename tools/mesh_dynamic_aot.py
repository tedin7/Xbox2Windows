"""Lower a bounded mesh with runtime geometry words to desktop mesh HLSL.

This reuses the mesh entry model and partitions inputs only at comparisons
between a geometry DWORD and a concrete value. Every reachable partition must
have concrete allocation, masks and indices. The caller binds the declared
number of DWORDs as a raw SRV at t0 and keeps the contents stable for the draw.
The entry profile is two wave64s, triangle output, and no task payload.

With a sampler heap SGPR (opt-in), image_sample_lz can read one texture through a
sampler loaded from the sampler heap. The texture descriptor is the eight words at
byte 0x40 of the s17 table and becomes t1. The sampler is s_load_dwordx4 from the
heap base plus a constant offset, or plus a register offset that the input cases
make concrete. It reads SamplerState qh[2048] : register(s0, space1), the table of
tools/startup/gfx2hlsl.py: entry n is the heap entry at byte 16 n, and entry 2047
is the host's fallback, which a proven offset never reaches.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from mesh_clip_cull import decode_vs_out_cntl, hlsl_fields, hlsl_writes, vertex_distances
from mesh_entry_aot import U32, UNDEF, Wave, decode_mesh, float4, live_barriers_agree, run_group


MAX_VARIANTS = 32
# The sampler heap table shared with gfx2hlsl's X2W_SAMPLER_HEAP=1 path (#1217, #1237).
SAMPLER_HEAP_SIZE = 2048
SAMPLER_HEAP_SPACE = 1
SAMPLER_HEAP_FALLBACK = SAMPLER_HEAP_SIZE - 1
SAMPLER_STRIDE = 16
TEXTURE_TABLE_OFFSET = 0x40
TEXTURE_NAME = "meshTexture"
ENTRY_SGPRS = (3, 17, 20, 21)
IMAGE_SAMPLE_LZ = re.compile(
    r"(v\d+|v\[\d+:\d+\]), (v\[\d+:\d+\]|\[v\d+, v\d+\]), (s\[\d+:\d+\]), (s\[\d+:\d+\])"
    r" dmask:(0x[0-9a-f]+) dim:SQ_RSRC_IMG_2D"
)


@dataclass(frozen=True)
class RuntimeWord:
    index: int


@dataclass(frozen=True)
class SamplerHeapBase:
    """The 32-bit sampler heap address from the entry SGPR."""


@dataclass(frozen=True)
class SamplerWord:
    entry: int
    word: int


@dataclass(frozen=True)
class TextureWord:
    word: int


DESCRIPTOR_VALUES = (SamplerHeapBase, SamplerWord, TextureWord)


def heap_entry(offset):
    """The table entry for a proven byte offset into the sampler heap."""
    if offset % SAMPLER_STRIDE or not 0 <= offset < SAMPLER_STRIDE * SAMPLER_HEAP_FALLBACK:
        raise ValueError(f"Sampler heap offset {offset:#x} is not one of the "
                         f"{SAMPLER_HEAP_FALLBACK} title entries")
    return offset // SAMPLER_STRIDE


def scalar_registers(token):
    match = re.fullmatch(r"s(\d+)|s\[(\d+):(\d+)\]", token)
    if not match:
        return []
    if match[1]:
        return [int(match[1])]
    return list(range(int(match[2]), int(match[3]) + 1))


class SplitInput(Exception):
    def __init__(self, index, value):
        self.index = index
        self.value = value


class Geometry:
    def __init__(self, word_count, conditions, sampler_heap_sgpr=None, distance_layout=None):
        self.word_count = word_count
        self.sampler_heap_sgpr = sampler_heap_sgpr
        # The decoded PA_CL_VS_OUT_CNTL word, or None when the caller gives none.
        self.distance_layout = distance_layout
        self.equal = {}
        self.excluded = {}
        for index, value, equal in conditions:
            if equal:
                self.equal[index] = value
            else:
                self.excluded.setdefault(index, set()).add(value)

    def __len__(self):
        return self.word_count

    def __getitem__(self, key):
        if isinstance(key, slice):
            if not 0 <= key.start <= key.stop <= self.word_count:
                raise ValueError("Geometry read outside binding")
            return [self[index] for index in range(key.start, key.stop)]
        if not 0 <= key < self.word_count:
            raise ValueError("Geometry read outside binding")
        return self.equal.get(key, RuntimeWord(key))


class RuntimeWave(Wave):
    def __init__(self, wave, geometry):
        super().__init__(wave, geometry)
        if geometry.sampler_heap_sgpr is not None:
            self.s[geometry.sampler_heap_sgpr] = SamplerHeapBase()

    def raw(self, register):
        """A scalar register as the model holds it, still subject to the load wait."""
        return Wave.scalar(self, f"s{register}")

    def descriptor_operand(self, tokens):
        for token in tokens:
            for register in scalar_registers(token):
                value = self.s.get(register)
                if isinstance(value, DESCRIPTOR_VALUES):
                    return token, value
        return None

    def refuse_descriptor_use(self, op, found, dest=None):
        token, value = found
        if isinstance(value, SamplerWord):
            if op.startswith("s_cselect_"):
                raise ValueError(f"Sampler heap word in {token} mixed with another value by {op}")
            if dest is not None and set(scalar_registers(dest)) & set(scalar_registers(token)):
                raise ValueError(f"Sampler heap word in {token} changed after the load by {op}")
            raise ValueError(f"Sampler heap word in {token} used by {op}")
        if isinstance(value, SamplerHeapBase):
            raise ValueError(
                f"Sampler heap base in {token} used by {op}; only an s_load_dwordx4 offset is proven"
            )
        raise ValueError(f"Mesh texture descriptor in {token} used by {op}")

    def descriptor_scalar_op(self, op, args):
        """Handle sampler heap and texture descriptor words; False for any other scalar op."""
        if self.geometry.sampler_heap_sgpr is None:
            return False
        dest, *sources = args
        if op == "s_load_dwordx8":
            self.load_texture(args)
            return True
        found = self.descriptor_operand(sources)
        if op == "s_load_dwordx4" and found and isinstance(found[1], SamplerHeapBase):
            self.load_sampler(args)
            return True
        if not found:
            return False
        if op in ("s_mov_b32", "s_mov_b64") and len(sources) == 1:
            source, target = scalar_registers(sources[0]), scalar_registers(dest)
            if len(source) == len(target) == (2 if op == "s_mov_b64" else 1):
                values = [self.raw(register) for register in source]
                if self.pending.intersection(target):
                    raise ValueError("Pending scalar destination")
                self.s.update(zip(target, values))
                return True
        self.refuse_descriptor_use(op, found, dest)

    def load_texture(self, args):
        dest, base, offset = args
        registers = scalar_registers(dest)
        if len(registers) != 8 or self.pending.intersection(registers):
            raise ValueError("Invalid texture descriptor load")
        if (self.scalar(base) != 0x10000
                or (0 if offset == "null" else self.scalar(offset)) != TEXTURE_TABLE_OFFSET):
            raise ValueError("Unknown texture descriptor table or offset")
        self.s.update((register, TextureWord(i)) for i, register in enumerate(registers))
        self.pending.update(registers)

    def load_sampler(self, args):
        dest, base, offset = args
        registers, pair = scalar_registers(dest), scalar_registers(base)
        if len(registers) != 4 or len(pair) != 2 or self.pending.intersection(registers):
            raise ValueError("Invalid sampler heap load")
        if self.raw(pair[0]) != SamplerHeapBase() or self.raw(pair[1]) != 0:
            raise ValueError("Sampler heap address is not the entry SGPR with a zero high word")
        if offset == "null":
            value = 0
        elif re.fullmatch(r"s\d+", offset):
            value = self.raw(int(offset[1:]))
        elif re.fullmatch(r"0x[0-9a-fA-F]+|\d+", offset):
            value = int(offset, 0)
        else:
            raise ValueError(f"Unsupported sampler heap offset operand {offset}")
        entry = heap_entry(self.proven_offset(value, offset) & U32)
        self.s.update((register, SamplerWord(entry, i)) for i, register in enumerate(registers))
        self.pending.update(registers)

    def proven_offset(self, value, token):
        """A concrete offset in this input case; a word no comparison has fixed is unbounded."""
        if isinstance(value, RuntimeWord):
            raise ValueError(f"Unbounded sampler heap offset: runtime geometry word {value.index} in {token}")
        if type(value) is not int:
            raise ValueError(f"Unbounded sampler heap offset in {token}")
        return value

    def sampler_words(self, values):
        """The entry when the operand holds the four words of one heap load, in order."""
        if (all(isinstance(value, SamplerWord) for value in values)
                and len({value.entry for value in values}) == 1
                and [value.word for value in values] == [0, 1, 2, 3]):
            return values[0].entry
        return None

    def sampler_entry(self, token):
        values = [self.raw(register) for register in scalar_registers(token)]
        entry = self.sampler_words(values) if len(values) == 4 else None
        if entry is not None:
            return entry
        if any(isinstance(value, SamplerWord) for value in values):
            raise ValueError(f"Sampler operand {token} no longer holds the four words of one heap load")
        if len(values) == 4 and all(type(value) is int for value in values):
            raise ValueError(f"Static sampler words in {token} are not translated in mesh programs")
        raise ValueError(f"Unknown sampler operand {token}")

    def image_op(self, op, tail, expressions):
        if self.geometry.sampler_heap_sgpr is None:
            return super().image_op(op, tail, expressions)
        if op != "image_sample_lz":
            raise ValueError(f"Unsupported mesh image instruction {op}")
        match = IMAGE_SAMPLE_LZ.fullmatch(tail)
        if not match:
            raise ValueError(f"Unsupported image_sample_lz operands: {tail}")
        dest, address, texture, sampler, dmask = match.groups()
        components = [c for c in range(4) if int(dmask, 16) >> c & 1]
        destinations = [int(n) for n in re.findall(r"\d+", dest)]
        if len(destinations) == 2:
            destinations = list(range(destinations[0], destinations[1] + 1))
        if not components or int(dmask, 16) > 15 or len(destinations) != len(components):
            raise ValueError(f"image_sample_lz destination {dest} does not match dmask {dmask}")
        coordinates = [int(n) for n in re.findall(r"\d+", address)]
        if address.startswith("v[") and coordinates[1] - coordinates[0] != 1:
            raise ValueError(f"Expected two 2D coordinates in {address}")
        if address.startswith("v["):
            coordinates = list(range(coordinates[0], coordinates[1] + 1))
        texture_words = [self.raw(register) for register in scalar_registers(texture)]
        if texture_words != [TextureWord(i) for i in range(8)]:
            raise ValueError(f"Unknown mesh texture descriptor {texture}")
        entry = self.sampler_entry(sampler)
        for lane in range(64):
            if not self.exec & (1 << lane):
                continue
            uv = ", ".join(value if isinstance(value, str) else f"asfloat(0x{value & U32:08x}u)"
                           for value in (self.lane_value(f"v{r}", lane) for r in coordinates))
            sample = f"{TEXTURE_NAME}.SampleLevel(qh[{entry}], float2({uv}), 0.0f)"
            for register, component in zip(destinations, components):
                name = expressions.setdefault(f"{sample}.{'xyzw'[component]}", f"f{len(expressions)}")
                if (lane, f"v{register}") in self.pending_v:
                    raise ValueError("Pending vector destination")
                self.v[lane][register] = name
                self.pending_v.add((lane, f"v{register}"))

    def lane_spill(self, op, args):
        raise ValueError(f"Unsupported runtime mesh instruction {op}")

    def scalar(self, token, allow_undef=False):
        value = super().scalar(token, allow_undef)
        if isinstance(value, RuntimeWord):
            raise ValueError(f"Unproven runtime geometry word {value.index} in {token}")
        return value

    def lane_value(self, token, lane):
        value = super().lane_value(token, lane)
        if isinstance(value, RuntimeWord):
            raise ValueError(f"Unproven runtime geometry word {value.index} in {token}")
        return value

    def scalar_op(self, op, args):
        if op == "s_cmp_eq_u32":
            left, right = (super(RuntimeWave, self).scalar(t, True) for t in args)
            if left is UNDEF or right is UNDEF:
                return super().scalar_op(op, args)
            if isinstance(right, RuntimeWord):
                left, right = right, left
            if isinstance(left, RuntimeWord) and type(right) is int:
                value = right & U32
                if value not in self.geometry.excluded.get(left.index, ()):
                    raise SplitInput(left.index, value)
                self.scc = False
                return
        if self.descriptor_scalar_op(op, args):
            return
        super().scalar_op(op, args)

    def vector_op(self, op, args, expressions):
        found = self.descriptor_operand(args if op.startswith("v_cmpx_") else args[1:])
        if found:
            self.refuse_descriptor_use(op, found)
        if op not in ("v_cndmask_b32_e32", "v_cndmask_b32_e64"):
            return super().vector_op(op, args, expressions)
        dest, low, high, predicate = args
        # No live lane reads the mask, so an undefined one is not an error.
        mask = self.scalar(predicate) if self.exec else 0
        concrete_model = super()
        for lane in range(64):
            if not self.exec & (1 << lane):
                continue
            # The unused value may be a defined runtime word. Both sources
            # still go through the existing checks for undefined/pending data.
            values = [concrete_model.lane_value(token, lane) if token.startswith("v")
                      else concrete_model.scalar(token) for token in (low, high)]
            value = values[bool(mask & (1 << lane))]
            if isinstance(value, RuntimeWord):
                raise ValueError(f"Unproven runtime geometry word {value.index} in {op}")
            if self.vector_pending(lane, dest):
                raise ValueError("Pending vector destination")
            self.v[lane][int(dest[1:])] = value & U32 if isinstance(value, int) else value


@dataclass
class Variant:
    conditions: tuple
    counts: tuple
    expressions: dict
    positions: dict
    attributes: dict
    indices: dict
    # {vertex: {(kind, vector): values}} for the enabled clip and cull distances.
    distances: dict = None


def run_variant(decoded, geometry, limits, conditions):
    expressions = {}
    first, second = RuntimeWave(0, geometry), RuntimeWave(1, geometry)
    run_group([first, second], decoded, expressions)
    if not live_barriers_agree(first, second) or len(first.counts) != 1 or second.counts:
        raise ValueError("Inconsistent mesh synchronization or allocation")
    nv, np = first.counts[0]
    pos = first.positions | second.positions
    attrs = first.attributes | second.attributes
    indices = first.indices | second.indices
    if (not 0 <= nv <= limits[0] or not 0 <= np <= limits[1]
            or set(pos) != set(range(nv)) or set(indices) != set(range(np))
            or any(max(triangle) >= nv for triangle in indices.values())):
        raise ValueError("Conflicting mesh counts, vertices or primitive indices")
    if attrs and attrs.keys() != pos.keys():
        raise ValueError("Incomplete vertex attribute")
    distances = vertex_distances(first.position_exports | second.position_exports, geometry.distance_layout)
    return Variant(conditions, (nv, np), expressions, pos, attrs, indices, distances)


def enumerate_variants(decoded, geometry_word_count, limits, sampler_heap_sgpr=None, vs_out_cntl=None):
    layout = None if vs_out_cntl is None else decode_vs_out_cntl(vs_out_cntl)
    # ByteAddressBuffer.Load takes a 32-bit byte offset.
    if type(geometry_word_count) is not int or not 0 < geometry_word_count <= 1 << 30:
        raise ValueError("Invalid runtime geometry word count")
    if len(limits) != 2 or any(type(n) is not int or not 0 < n <= 256 for n in limits):
        raise ValueError("Invalid mesh declaration")
    if sampler_heap_sgpr is not None and (
            type(sampler_heap_sgpr) is not int or not 0 <= sampler_heap_sgpr <= 105
            or sampler_heap_sgpr in ENTRY_SGPRS):
        raise ValueError(f"Invalid sampler heap SGPR {sampler_heap_sgpr}")
    pending, variants = [()], []
    while pending:
        conditions = pending.pop()
        geometry = Geometry(geometry_word_count, conditions, sampler_heap_sgpr, layout)
        try:
            variants.append(run_variant(decoded, geometry, limits, conditions))
        except SplitInput as split:
            # Replay only after a comparison splits the possible inputs into
            # disjoint cases. Neither branch substitutes a sample for an
            # unknown word. A case may discard an unknown word or compare it
            # again, but a relevant arithmetic/export use must be proven.
            if len(variants) + len(pending) + 2 > MAX_VARIANTS:
                raise ValueError(f"Runtime mesh exceeds {MAX_VARIANTS} input cases") from None
            pending.append((*conditions, (split.index, split.value, False)))
            pending.append((*conditions, (split.index, split.value, True)))
    if not any(variant.conditions for variant in variants):
        raise ValueError("No runtime geometry comparisons")
    if len({bool(variant.attributes) for variant in variants if variant.positions}) > 1:
        raise ValueError("Runtime mesh branches disagree on the vertex signature")
    return variants


def emit_hlsl(variants, limits, distance_layout=None):
    has_attributes = any(variant.attributes for variant in variants)
    input_words = sorted({index for variant in variants for index, _, _ in variant.conditions})
    samples = any(expr.startswith(TEXTURE_NAME + ".SampleLevel(")
                  for variant in variants for expr in variant.expressions)
    lines = [
        "ByteAddressBuffer geometryData : register(t0);",
        "cbuffer Points : register(b0) { float4 pointData[32]; };",
        "cbuffer Matrices : register(b1) { float4 matrixData[32]; };",
        *([f"Texture2D<float4> {TEXTURE_NAME} : register(t1);",
           f"SamplerState qh[{SAMPLER_HEAP_SIZE}] : register(s0, space{SAMPLER_HEAP_SPACE});"]
          if samples else []),
        "struct Vertex { float4 position : SV_Position; "
        + ("float4 attr : TEXCOORD0; " if has_attributes else "") + hlsl_fields(distance_layout) + "};",
        '[outputtopology("triangle")]',
        "[numthreads(128,1,1)]",
        "void main(uint id : SV_GroupIndex, "
        f"out vertices Vertex vertices[{limits[0]}], out indices uint3 primitives[{limits[1]}]) {{",
        *(f"  uint word{index} = geometryData.Load({index * 4});" for index in input_words),
        "  uint variant, vertexCount, primitiveCount;",
    ]
    for i, variant in enumerate(variants):
        if i + 1 == len(variants):
            condition = "else"
        else:
            predicate = " && ".join(
                f"word{index} {'==' if equal else '!='} 0x{value:08x}u"
                for index, value, equal in variant.conditions
            )
            condition = f"{'else ' if i else ''}if ({predicate})"
        vertices, primitives = variant.counts
        lines.append(
            f"  {condition} {{ variant = {i}; vertexCount = {vertices}; primitiveCount = {primitives}; }}"
        )
    lines.append("  SetMeshOutputCounts(vertexCount, primitiveCount);")
    for i, variant in enumerate(variants):
        if not variant.positions and not variant.indices:
            continue
        lines.append(f"  if (variant == {i}) {{")
        lines.extend(f"    precise float {name} = {expr};" for expr, name in variant.expressions.items())
        for index, position in sorted(variant.positions.items()):
            lines.append(f"    if (id == {index}) {{")
            lines.append(f"      vertices[{index}].position = {float4(position)};")
            if has_attributes:
                lines.append(f"      vertices[{index}].attr = {float4(variant.attributes[index])};")
            if variant.distances:
                lines.append(f"      {hlsl_writes(f'vertices[{index}]', variant.distances[index])}")
            lines.append("    }")
        for index, triangle in sorted(variant.indices.items()):
            indices = ", ".join(map(str, triangle))
            lines.append(f"    if (id == {index}) primitives[{index}] = uint3({indices});")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def lower(decoded, geometry_word_count, limits, sampler_heap_sgpr=None, vs_out_cntl=None):
    variants = enumerate_variants(decoded, geometry_word_count, limits, sampler_heap_sgpr, vs_out_cntl)
    layout = None if vs_out_cntl is None else decode_vs_out_cntl(vs_out_cntl)
    return emit_hlsl(variants, limits, layout)


def translate(blob, geometry_word_count, library, sampler_heap_sgpr=None, vs_out_cntl=None):
    decoded, limits = decode_mesh(blob, library)
    return lower(decoded, geometry_word_count, limits, sampler_heap_sgpr, vs_out_cntl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path)
    parser.add_argument("--geometry-word-count", type=int, required=True)
    parser.add_argument("--llvm", required=True)
    parser.add_argument("--sampler-heap-sgpr", type=int,
                        help="Opt in to sampler heap loads; the entry SGPR that holds the heap base")
    parser.add_argument("--pa-cl-vs-out-cntl", type=lambda s: int(s, 0),
                        help="PA_CL_VS_OUT_CNTL word that places clip and cull distances in pos1/pos2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".hlsl" or args.output.resolve() == args.container.resolve():
        parser.error("Expected a separate HLSL output")
    if args.sampler_heap_sgpr is not None and (
            not 0 <= args.sampler_heap_sgpr <= 105 or args.sampler_heap_sgpr in ENTRY_SGPRS):
        parser.error("--sampler-heap-sgpr must name a free SGPR (s0-s105, not s3, s17, s20 or s21)")
    try:
        hlsl = translate(args.container.read_bytes(), args.geometry_word_count, args.llvm,
                         args.sampler_heap_sgpr, args.pa_cl_vs_out_cntl)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"mesh_dynamic_aot: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(hlsl, encoding="ascii")
