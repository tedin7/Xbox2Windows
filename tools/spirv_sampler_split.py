"""Split a sampler shared by comparison and ordinary sampling in the bridge's named SPIR-V profile.

SPIRV-Cross declares one HLSL sampler per SPIR-V sampler variable. When one variable feeds both
OpImageSampleDrefExplicitLod and OpImageSampleExplicitLod, it emits a SamplerComparisonState and
passes it to an ordinary SampleLevel call, which DXC rejects. This pass keeps the comparison
sampler at its original binding and moves only the ordinary uses to a new sampler variable at a
binding the caller supplies. The caller (the host) owns that binding and must bind an ordinary
sampler there. A module may hold several such samplers: the caller then supplies one distinct free
binding per sampler, in declaration order, and each gets its own new sampler variable.

The pass first looks for a standalone UniformConstant sampler, loaded directly, whose sampled
images feed both kinds of sampling. Modules without one, including indexed sampler tables and
unrelated sampling, are returned byte-for-byte unchanged. For the sampler being split, the profile
is limited to direct float32 Texture1D, Texture2D, TextureCube and Texture3D sampling, including the
array forms of the first three, in a single compute entry point. The ordinary kind takes any LOD
operand whose result type is the sampled float type (a constant of any value or a computed value); the
depth-comparison kind stays at constant LOD zero, the only form SM 6.0 has for it (SampleCmpLevelZero). Texture3D is
only sampled by the ordinary kind: depth comparison on it, and an arrayed 3D image, are refused.
OpImageGather counts as an ordinary use and OpImageDrefGather as a comparison use, so a gather keeps its
sampler kind. Both are accepted without image operands only, on 2D, 2D array, cube and cube array images
(the forms D3D12 gathers), and OpImageGather needs an integer OpConstant component from 0 to 3. The sampler must appear in the
entry-point interface. Any other use of that sampler is rejected with a specific diagnostic.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import shlex
import sys
import tempfile

PREFIX = "%x2w_plain_sampler"
ZERO = {"0", "0.0", "0x0p+0", "-0", "-0.0", "-0x0p+0"}
GATHERS = {"OpImageGather": "ordinary", "OpImageDrefGather": "compare"}
ANNOTATIONS = {"OpName", "OpMemberName", "OpDecorate", "OpMemberDecorate", "OpEntryPoint", "OpExecutionMode"}


def _tokens(line):
    lexer = shlex.shlex(line, posix=False)
    lexer.commenters = ";"
    lexer.whitespace_split = True
    return list(lexer)


def _standalone_sampler(definitions, loaded):
    """Return the variable behind a direct OpLoad of a standalone UniformConstant sampler, else None."""
    load = definitions.get(loaded, [])
    if len(load) != 3 or load[0] != "OpLoad":
        return None
    variable = definitions.get(load[2], [])
    if len(variable) != 3 or variable[0] != "OpVariable" or variable[2] != "UniformConstant":
        return None
    pointer = definitions.get(variable[1], [])
    if (len(pointer) != 3 or pointer[:2] != ["OpTypePointer", "UniformConstant"] or pointer[2] != load[1]
            or definitions.get(pointer[2]) != ["OpTypeSampler"]):
        return None
    return load[2]


def _check_gather(image, use, texture, definitions):
    """Accept OpImageGather/OpImageDrefGather %type %image %coord %component|%dref with no image operands."""
    if len(use) != 5:
        raise ValueError(f"{image}: {use[0]} with image operands is not supported")
    if use[2] != image:
        raise ValueError(f"{image}: {use[0]} must gather from the sampled image itself")
    if texture[2] not in ("2D", "Cube"):
        raise ValueError(f"{image}: {use[0]} is only supported on Texture2D, Texture2DArray, TextureCube "
                         "and TextureCubeArray")
    if use[0] == "OpImageGather":
        component = definitions.get(use[4], [])
        try:
            value = int(component[2], 0)
        except (IndexError, ValueError):
            value = -1
        if (len(component) != 3 or component[0] != "OpConstant"
                or definitions.get(component[1], [])[:1] != ["OpTypeInt"] or not 0 <= value <= 3):
            raise ValueError(f"{image}: OpImageGather requires an integer constant component from 0 to 3")


def _ordinary_images(sampler, uses, definitions):
    """Apply the narrow profile to one sampler being split; return its ordinary-only sampled images."""
    images = []
    for _, use in uses[sampler]:
        if use[0] != "OpLoad":
            raise ValueError(f"{sampler}: {use[0]} uses the sampler; only direct loads are supported")
    for load in (result for result, use in uses[sampler]):
        for image, use in uses[load]:
            if use[0] != "OpSampledImage" or use[-1] != load:
                raise ValueError(f"{sampler}: loaded sampler feeds {use[0]}; only OpSampledImage is supported")
            images.append(image)
    ordinary_images = []
    for image in images:
        sampled = definitions.get(definitions[image][1], [])
        texture = definitions.get(sampled[1], []) if sampled[:1] == ["OpTypeSampledImage"] else []
        # OpTypeImage <sampled type> <dim> <depth> <arrayed 0|1> <ms> <sampled> <format>
        if (len(texture) != 8 or texture[0] != "OpTypeImage"
                or definitions.get(texture[1]) != ["OpTypeFloat", "32"]
                or texture[2] not in ("1D", "2D", "Cube", "3D") or texture[4] not in ("0", "1")
                or (texture[2] == "3D" and texture[4] != "0")
                or texture[5:] != ["0", "1", "Unknown"]):
            raise ValueError(f"{image}: only float32 Texture1D, Texture2D, TextureCube and non-arrayed Texture3D "
                             "sampling (including the array forms of the first three) is supported")
        kinds = set()
        for _, use in uses[image]:
            if use[0] in GATHERS:
                _check_gather(image, use, texture, definitions)
                kinds.add(GATHERS[use[0]])
                continue
            ordinary = use[0] == "OpImageSampleExplicitLod" and len(use) == 6
            compare = use[0] == "OpImageSampleDrefExplicitLod" and len(use) == 7
            if not (ordinary or compare) or use[2] != image or use[-2] != "Lod":
                raise ValueError(f"{image}: only explicit-LOD ordinary and depth-comparison sampling is supported")
            if compare and texture[2] == "3D":
                raise ValueError(f"{image}: depth-comparison sampling is not supported on Texture3D")
            lod = definitions.get(use[-1], [])
            if compare:
                if len(lod) != 3 or lod[:2] != ["OpConstant", texture[1]] or lod[2] not in ZERO:
                    raise ValueError(f"{image}: {use[0]} requires a constant LOD of zero")
            elif len(lod) < 2 or lod[1] != texture[1]:
                # An ordinary sample is a plain SampleLevel at any LOD: a constant or a computed value
                # whose result type is the sampled float type (spirv-val also requires a float scalar).
                raise ValueError(f"{image}: {use[0]} requires a LOD operand of the sampled float type")
            kinds.add("ordinary" if ordinary else "compare")
        if len(kinds) > 1:
            raise ValueError(f"{image}: one sampled image is used for both sampling kinds")
        if kinds == {"ordinary"}:
            ordinary_images.append(image)
    return ordinary_images


def split_comparison_sampler(assembly, ordinary_binding):
    """Return (assembly, mapping). mapping is None when no sampler has mixed uses.

    ordinary_binding is one int for a module with exactly one mixed sampler; mapping is then one dict
    and the new IDs are %x2w_plain_sampler and %x2w_plain_sampler_<k> for its loads. For several mixed
    samplers pass a sequence with one distinct binding per sampler, in the order the sampler variables
    are declared; mapping is then a list of dicts, the new variables are %x2w_plain_sampler_<n> (n counts
    from 0) and their loads %x2w_plain_sampler_<n>_load_<k>.
    """
    sequence = not isinstance(ordinary_binding, int)
    requested = list(ordinary_binding) if sequence else [ordinary_binding]
    if any(binding < 0 for binding in requested):
        raise ValueError("Ordinary sampler binding must be nonnegative")
    if len(set(requested)) != len(requested):
        raise ValueError("Ordinary sampler bindings must be distinct")
    lines = assembly.splitlines()
    parsed = [_tokens(line) for line in lines]
    definitions = {words[0]: words[2:] for words in parsed if len(words) >= 3 and words[1] == "="}
    defined_at = {words[0]: i for i, words in enumerate(parsed) if len(words) >= 3 and words[1] == "="}

    # Every non-annotation instruction that references an ID, keyed by that ID.
    uses = defaultdict(list)
    for words in parsed:
        args = words[2:] if len(words) >= 3 and words[1] == "=" else words
        if args and args[0] not in ANNOTATIONS:
            for operand in args[1:]:
                if operand.startswith("%"):
                    uses[operand].append((words[0] if args is not words else None, args))

    # 1. Find candidates without judging anything outside the standalone profile: a sampled image
    # whose sampler is a direct OpLoad of a standalone UniformConstant OpTypeSampler variable.
    kinds_by_sampler = defaultdict(set)
    for image, definition in definitions.items():
        if definition[:1] != ["OpSampledImage"] or len(definition) != 4:
            continue
        sampler = _standalone_sampler(definitions, definition[3])
        if sampler is None:
            continue
        for _, use in uses[image]:
            if use[0].startswith("OpImage") and use[0] != "OpImage":
                kinds_by_sampler[sampler].add("compare" if "Dref" in use[0] else "ordinary")
    mixed = sorted((sampler for sampler, kinds in kinds_by_sampler.items() if len(kinds) == 2),
                   key=defined_at.__getitem__)
    if not mixed:
        return assembly, None
    if len(mixed) != len(requested) or (len(mixed) > 1 and not sequence):
        raise ValueError(f"Sampler split found {len(mixed)} samplers with mixed uses and was given "
                         f"{len(requested)} ordinary bindings; supply one distinct binding for each")

    # 2. Apply the narrow profile to the samplers being split, and only to them.
    entries = [i for i, words in enumerate(parsed) if words[:1] == ["OpEntryPoint"]]
    if len(entries) != 1 or parsed[entries[0]][1] != "GLCompute":
        raise ValueError("Sampler split requires a single GLCompute entry point")
    entry = entries[0]
    ordinary_images = {sampler: _ordinary_images(sampler, uses, definitions) for sampler in mixed}

    bindings = {}
    descriptor_sets = {}
    for i, words in enumerate(parsed):
        if len(words) == 4 and words[0] == "OpDecorate" and words[2] in ("Binding", "DescriptorSet"):
            try:
                value = int(words[3], 0)
            except ValueError:
                raise ValueError(f"{words[1]}: unreadable {words[2]} decoration") from None
            if words[2] == "Binding":
                bindings[words[1]] = (i, value)
            else:
                descriptor_sets[words[1]] = value
    for sampler, binding in zip(mixed, requested):
        if sampler not in parsed[entry][4:]:
            raise ValueError(f"{sampler}: sampler must be listed in the entry-point interface")
        if sampler not in bindings or sampler not in descriptor_sets:
            raise ValueError(f"{sampler}: sampler needs a DescriptorSet and Binding decoration")
        for variable, (_, slot) in bindings.items():
            # A Binding without a DescriptorSet decoration is counted as set 0, so it cannot hide.
            if descriptor_sets.get(variable, 0) == descriptor_sets[sampler] and slot == binding:
                raise ValueError(f"Ordinary sampler binding {binding} is occupied in set "
                                 f"{descriptor_sets[sampler]} by {variable}")
    if any(word.startswith(PREFIX) for words in parsed for word in words):
        raise ValueError("Reserved sampler split ID is already in use")

    mappings = []
    new_ids = []
    for n, (sampler, binding) in enumerate(zip(mixed, requested)):
        plain = f"{PREFIX}_{n}" if sequence else PREFIX
        new_ids.append(plain)
        binding_line, comparison_binding = bindings[sampler]
        descriptor_set = descriptor_sets[sampler]
        lines[binding_line] += (f"\nOpDecorate {plain} DescriptorSet {descriptor_set}"
                                f"\nOpDecorate {plain} Binding {binding}")
        lines[defined_at[sampler]] += f"\n{plain} = OpVariable {definitions[sampler][1]} UniformConstant"
        sampler_type = definitions[definitions[sampler][1]][2]
        for index, image in enumerate(ordinary_images[sampler]):
            i = defined_at[image]
            load = f"{plain}_load_{index}" if sequence else f"{PREFIX}_{index}"
            lines[i] = f"{load} = OpLoad {sampler_type} {plain}\n" + " ".join([*parsed[i][:-1], load])
        mapping = {"descriptor_set": descriptor_set, "comparison_binding": comparison_binding,
                   "ordinary_binding": binding}
        if sequence:
            mapping = {"comparison_sampler": sampler, "plain_sampler": plain, **mapping}
        mappings.append(mapping)
    lines[entry] = " ".join([*parsed[entry], *new_ids])
    return "\n".join(lines) + "\n", mappings if sequence else mappings[0]


def _write_atomic(path: Path, content: str) -> None:
    """Write content to a sibling temp file and atomically replace path, leaving no temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="SPIR-V assembly")
    parser.add_argument("output", type=Path, help="lowered SPIR-V assembly")
    parser.add_argument("--ordinary-binding", type=int, required=True, action="append",
                        help="free binding, in the comparison sampler's descriptor set, for the ordinary sampler; "
                             "repeat once per mixed sampler, in declaration order, with distinct values")
    args = parser.parse_args(argv)
    try:
        if any(binding < 0 for binding in args.ordinary_binding):
            raise ValueError("Ordinary sampler binding must be nonnegative")
        if len(set(args.ordinary_binding)) != len(args.ordinary_binding):
            raise ValueError("Ordinary sampler bindings must be distinct")
        if args.output.resolve() == args.input.resolve():
            raise ValueError(f"output path must not match input path: {args.output}")
        bindings = args.ordinary_binding[0] if len(args.ordinary_binding) == 1 else args.ordinary_binding
        assembly, mapping = split_comparison_sampler(args.input.read_text(encoding="utf-8"), bindings)
        _write_atomic(args.output, assembly)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    if mapping:
        for entry in mapping if isinstance(mapping, list) else [mapping]:
            print(" ".join(f"{key}={value}" for key, value in entry.items()))
    else:
        print("unchanged: no sampler with mixed uses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
