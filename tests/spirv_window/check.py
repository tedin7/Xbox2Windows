"""Validate window and sampler lowering, composed ballot lowering, and rejection paths."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import spirv_address_window as window
from spirv_sampler_split import split_comparison_sampler


def validate_sampler(directory, name, source):
    assembly = directory / f"{name}.spvasm"
    binary = directory / f"{name}.spv"
    assembly.write_text(source, encoding="utf-8")
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", assembly, "-o", binary], check=True)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", binary], check=True)


def check_sampler_dimensions(directory):
    shared = Path(__file__).with_name("shared_sampler.spvasm").read_text(encoding="utf-8")
    for dimension, arrayed, width, capability, texture in (
        ("1D", 0, 1, "Sampled1D", "Texture1D"),
        ("1D", 1, 2, "Sampled1D", "Texture1DArray"),
        ("Cube", 0, 3, None, "TextureCube"),
        ("Cube", 1, 4, "SampledCubeArray", "TextureCubeArray"),
    ):
        image_type = f"OpTypeImage %float {dimension} 2 {arrayed} 0 1 Unknown"
        source = shared.replace("OpTypeImage %float 2D 2 0 0 1 Unknown", image_type)
        if capability:
            source = source.replace("OpCapability Shader", f"OpCapability Shader\nOpCapability {capability}")
        if width == 1:
            source = source.replace(" %15 %float_", " %float_0_25 %float_")
            source = source.replace(" %15 Lod", " %float_0_25 Lod")
        else:
            coordinate = " ".join(["%float_0_25"] * width)
            source = source.replace("OpTypeVector %float 2", f"OpTypeVector %float {width}")
            source = source.replace("OpConstantComposite %v2float %float_0_25 %float_0_25",
                                    f"OpConstantComposite %v2float {coordinate}")
            if width == 4:
                source = source.replace("%v4float = OpTypeVector %float 4", "")
                source = source.replace("%v2float", "%v4float")

        validate_sampler(directory, texture, source)
        lowered, mapping = split_comparison_sampler(source, 5)
        assert mapping == {"descriptor_set": 0, "comparison_binding": 2, "ordinary_binding": 5}, texture
        lines = {line.strip() for line in lowered.splitlines()}
        assert f"%type_2d_image = {image_type}" in lines, texture
        assert "%33 = OpLoad %type_sampler %cmpSampler" in lines, texture
        assert "%34 = OpSampledImage %type_sampled_image %32 %33" in lines, texture
        assert "%37 = OpSampledImage %type_sampled_image %32 %33" in lines, texture
        assert "%x2w_plain_sampler_0 = OpLoad %type_sampler %x2w_plain_sampler" in lines, texture
        assert "%42 = OpSampledImage %type_sampled_image %40 %x2w_plain_sampler_0" in lines, texture
        assert "OpDecorate %x2w_plain_sampler Binding 5" in lines, texture
        validate_sampler(directory, f"{texture}_split", lowered)
        print(f"{texture}: shared and split SPIR-V validated")

def check_phi_parent(directory, tool, direct):
    """An atomic splits its block, so a phi in a successor must name the block that now ends in the branch."""
    source = (direct
        .replace("%main = OpFunction",
                 "%atomic_rmw = OpConstant %uint 72\n"
                 "%main = OpFunction", 1)
        .replace("OpStore %p2 %d Aligned 4",
                 "%old_add = OpAtomicIAdd %uint %p1 %c1 %atomic_rmw %d\n"
                 "OpBranch %next\n"
                 "%next = OpLabel\n"
                 "%carried = OpPhi %uint %old_add %entry\n"
                 "OpStore %p2 %carried Aligned 4", 1))
    assert "%carried = OpPhi" in source
    path = Path(directory) / "phi_parent.spvasm"
    path.write_text(source, encoding="utf-8")
    target = Path(directory) / "phi_parent.spv"
    # The tool assembles and validates the lowered module, so a stale phi parent fails here.
    subprocess.run([*tool, path, target], check=True)
    assert target.read_bytes()[:4] == bytes.fromhex("03022307")
    assert "%carried = OpPhi %uint %old_add %x2w_atomic_old_add_merge" in window.lower(source).splitlines()


fixture = Path(__file__).with_name("subtract.spvasm")
original = fixture.read_text(encoding="utf-8")
adapted = window.lower(original)
assert "OpMemoryModel Logical GLSL450" in adapted
assert "OpConvertUToPtr" not in adapted and "OpUGreaterThanEqual" in adapted
# The range check applies to the indexed address, with a strict bound.
assert "%x2w_address_p0_range = OpULessThan %bool %x2w_address_p0_address %x2w_end" in adapted.splitlines()

# A direct word cast is checked from its own address.
direct = original.replace(
    "%v1 = OpConvertUToPtr %_ptr_PSB_arr %a1\n"
    "%p1 = OpAccessChain %_ptr_PSB_uint %v1 %c0",
    "%p1 = OpConvertUToPtr %_ptr_PSB_uint %a1",
)
assert direct != original
assert "%x2w_address_p1_low = OpBitwiseAnd %ulong %a1 %x2w_mask" in window.lower(direct).splitlines()

# Exercise atomics on both a direct word cast and an indexed physical address.
atomic_instructions = (
    "%old_add = OpAtomicIAdd %uint %p1 %c1 %atomic_rmw %d",
    "%old_exchange = OpAtomicExchange %uint %p2 %c1 %atomic_rmw %old_add",
    "%old_min = OpAtomicUMin %uint %p2 %c1 %atomic_rmw %old_exchange",
    "%old_max = OpAtomicUMax %uint %p2 %c1 %atomic_rmw %old_min",
    "%old_and = OpAtomicAnd %uint %p2 %c1 %atomic_rmw %old_max",
    "%old_or = OpAtomicOr %uint %p2 %c1 %atomic_rmw %old_and",
    "%old_xor = OpAtomicXor %uint %p2 %c1 %atomic_rmw %old_or",
    "%old_cmp = OpAtomicCompareExchange %uint %p2 %c1 %atomic_rmw %atomic_fail %d %old_xor",
)
atomics = (direct
    .replace("%main = OpFunction",
             "%atomic_rmw = OpConstant %uint 72\n"  # AcquireRelease | UniformMemory
             "%atomic_fail = OpConstant %uint 66\n"  # Acquire | UniformMemory
             "%main = OpFunction", 1)
    .replace("OpStore %p2 %d Aligned 4", "\n".join(atomic_instructions) + "\nOpStore %p0 %old_cmp Aligned 4", 1))

# Add a ballot read whose bit controls the value written through the physical address.
combined = (original
    .replace("OpCapability Shader\n", "OpCapability Shader\nOpCapability GroupNonUniformBallot\n", 1)
    .replace("%main = OpFunction",
             "%ballot_type = OpTypeVector %uint 4\n"
             "%ballot_mask = OpConstantComposite %ballot_type %c1 %c2 %c3 %c4\n"
             "%main = OpFunction", 1)
    .replace("OpStore %p2 %d Aligned 4",
             "%ballot_bit = OpGroupNonUniformBallotBitExtract %bool %c3 %ballot_mask %c0\n"
             "%selected = OpSelect %uint %ballot_bit %d %c0\n"
             "OpStore %p2 %selected Aligned 4", 1))
assert "OpGroupNonUniformBallotBitExtract" in combined

for source, reason in (
    (original.replace("%d ", "%x2w_d "), "identifier prefix"),
    (combined.replace("%ballot_mask ", "%x2w_ballot_c0 "), "ballot lowering ID"),
    (original.replace("OpDecorate %UserData Block", "OpDecorate %user_data DescriptorSet 2\nOpDecorate %UserData Block"),
     "descriptor sets"),
    (original.replace("OpStore %p2 %d", "%leak = OpConvertPtrToU %ulong %p2\nOpStore %p2 %d"),
     "physical word pointer"),
    (atomics.replace("OpAtomicIAdd", "OpAtomicISub"), "physical word pointer"),
    (original.replace("OpTypePointer PhysicalStorageBuffer %uint", "OpTypePointer PhysicalStorageBuffer %ulong"),
     "scalar layout"),
    (original.replace("OpDecorate %UserData Block",
                      "OpDecorate %user_data DescriptorSet 1\nOpDecorate %UserData Block"),
     "descriptor sets"),
    (original.replace('OpEntryPoint GLCompute %main "main" %user_data',
                      'OpEntryPoint GLCompute %main "main" %user_data\n'
                      'OpEntryPoint GLCompute %main "other" %user_data'),
     "expected one shader entry point"),
    (original.replace("OpDecorate %rt_uint ArrayStride 4", "OpDecorate %rt_uint ArrayStride 8"),
     "scalar layout"),
    (original.replace("%main = OpFunction",
                      "%unexpected_pc = OpTypePointer PushConstant %uint\n%main = OpFunction"),
     "unrecognized push-constant declaration"),
):
    try:
        window.lower(source)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f"accepted invalid profile: {reason}")

# A bounded 64-bit access is two DWORD accesses inside one branch taken only for an aligned, in-window, non-wrapping
# address; the plain-access substitute index is never used for it.
qword = fixture.with_name("qword.spvasm").read_text(encoding="utf-8")


def check_qword_lowering(text):
    lines = text.splitlines()
    assert "OpConvertUToPtr" not in text and "PhysicalStorageBuffer" not in text
    assert lines.count("%x2w_mask_wide = OpConstant %ulong 7") == 1
    assert lines.count("%x2w_end_wide = OpConstant %ulong 4294967288") == 1
    assert lines.count("%x2w_shift_wide = OpConstant %uint 3") == 1
    assert not any("_selected" in line and "x2w_address_source" in line or "x2w_address_target_selected" in line
                   for line in lines)

    def block(stem, condition):
        # The branch that guards a QWORD access, and the instructions that run only inside it.
        at = lines.index(f"{stem}_issue = OpLabel")
        assert lines[at - 2] == f"OpSelectionMerge {stem}_merge None"
        assert lines[at - 1] == f"OpBranchConditional {condition} {stem}_issue {stem}_merge"
        end = lines.index(f"OpBranch {stem}_merge")
        assert lines[end + 1] == f"{stem}_merge = OpLabel"
        return lines[at + 1:end]

    # The indexed source is checked for wrap-around; the direct target has no index addition to wrap.
    assert "%x2w_address_source_nowrap = OpUGreaterThanEqual %bool %x2w_address_source_address %base" in lines
    assert "%x2w_address_target_checked" not in text
    load = block("%x2w_load_value", "%x2w_address_source_checked")
    assert load[:3] == ["%x2w_load_value_next = OpIAdd %uint %x2w_address_source_word %x2w_one",
                        "%x2w_load_value_low_ptr = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %x2w_address_source_word",
                        "%x2w_load_value_high_ptr = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %x2w_load_value_next"]
    assert "%x2w_load_value_low = OpLoad %uint %x2w_load_value_low_ptr" in load
    assert "%x2w_load_value_high = OpLoad %uint %x2w_load_value_high_ptr" in load
    assert "%x2w_load_value_high_shifted = OpShiftLeftLogical %ulong %x2w_load_value_high64 %x2w_shift_half" in load
    assert "%value = OpPhi %ulong %x2w_load_value_value %x2w_load_value_issue %x2w_zero64 %entry" in lines
    stores = [line for line in lines if line.startswith("OpStore %x2w_store_")]
    store = block(next(line.split()[1].rsplit("_low_ptr", 1)[0] for line in stores if "_low_ptr" in line),
                  "%x2w_address_target_valid")
    assert len(stores) == 2 and all(line in store for line in stores)
    assert stores[0].split()[1].endswith("_low_ptr") and stores[1].split()[1].endswith("_high_ptr")
    assert [line for line in lines if line.startswith("OpStore ")] == stores


qword_lowered = window.lower(qword)
check_qword_lowering(qword_lowered)
# Each mutation of the lowering breaks a different part of that contract and must be caught.
for mutation, (old, new) in {
    "DWORD alignment": ("%x2w_mask_wide = OpConstant %ulong 7", "%x2w_mask_wide = OpConstant %ulong 3"),
    "window end": ("%x2w_end_wide = OpConstant %ulong 4294967288", "%x2w_end_wide = OpConstant %ulong 4294967292"),
    "wrap guard": ("OpBranchConditional %x2w_address_source_checked", "OpBranchConditional %x2w_address_source_valid"),
    "store guard": ("OpBranchConditional %x2w_address_target_valid", "OpBranchConditional %x2w_address_target_range"),
    "halves swapped": ("%x2w_load_value_low = OpLoad %uint %x2w_load_value_low_ptr",
                       "%x2w_load_value_low = OpLoad %uint %x2w_load_value_high_ptr"),
    "load result outside the branch": ("%x2w_zero64 %entry", "%x2w_load_value_value %entry"),
}.items():
    assert qword_lowered.count(old) == 1, mutation
    try:
        check_qword_lowering(qword_lowered.replace(old, new))
    except AssertionError:
        pass
    else:
        raise AssertionError(f"QWORD lowering mutation not caught: {mutation}")

# A successor phi that names the block holding a QWORD access must follow the block's terminator into the merge block.
phi_qword = qword.replace("OpReturn\nOpFunctionEnd", "OpBranch %exit\n%exit = OpLabel\n"
                          "%copied = OpPhi %ulong %result %entry\nOpStore %target %copied Aligned 8\nOpReturn\nOpFunctionEnd")
assert phi_qword != qword
with tempfile.TemporaryDirectory() as folder:
    validate_sampler(Path(folder), "phi_qword_original", phi_qword)
    phi_lowered = window.lower(phi_qword)
    assert "%copied = OpPhi %ulong %result %x2w_store_2_merge" in phi_lowered.splitlines()
    validate_sampler(Path(folder), "phi_qword_lowered", phi_lowered)

    mixed = phi_qword.replace("OpBranch %exit\n",
                              "%atomic_ptr = OpConvertUToPtr %_ptr_PSB_uint %base\n"
                              "%old_word = OpAtomicIAdd %uint %atomic_ptr %c1 %c0 %c1\nOpBranch %exit\n")
    assert mixed != phi_qword
    validate_sampler(Path(folder), "phi_mixed_original", mixed)
    mixed_lowered = window.lower(mixed)
    assert "%copied = OpPhi %ulong %result %x2w_atomic_old_word_merge" in mixed_lowered.splitlines()
    validate_sampler(Path(folder), "phi_mixed_lowered", mixed_lowered)

for source, reason in (
    (qword.replace("%value = OpLoad %ulong %source Aligned 8", "%value = OpAtomicIAdd %ulong %source %c1 %c0 %increment"),
     "physical word pointer"),
    (qword.replace("%value = OpLoad %ulong %source Aligned 8", "%value = OpLoad %ulong %source Aligned 4"),
     "physical word pointer"),
    (qword.replace("OpStore %target %result Aligned 8", "OpStore %target %result Volatile"), "physical word pointer"),
    (qword.replace("OpStore %target", "%leak = OpConvertPtrToU %ulong %target\nOpStore %target"), "physical word pointer"),
    (qword.replace("%array = OpConvertUToPtr %_ptr_PSB_arr64", "%array = OpConvertUToPtr %_ptr_PSB_arr"),
     "unknown physical array pointer"),
    (qword.replace("%_ptr_PSB_ulong = OpTypePointer PhysicalStorageBuffer %ulong", ""), "scalar layout"),
    (qword.replace("OpDecorate %rt_ulong ArrayStride 8", "OpDecorate %rt_ulong ArrayStride 4"), "scalar layout"),
):
    try:
        window.lower(source)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f"accepted invalid QWORD profile: {reason}")

output = ROOT / "out"
output.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=output) as directory:
    check_sampler_dimensions(Path(directory))
    target = Path(directory) / "window.spv"
    tool = [sys.executable, "-B", str(ROOT / "tools" / "spirv_address_window.py")]
    subprocess.run([*tool, fixture, target], check=True)
    valid = target.read_bytes()
    assert valid[:4] == bytes.fromhex("03022307")
    broken = Path(directory) / "broken.spvasm"
    broken.write_text(original.replace("%b = OpLoad %uint %p1", "%b = OpLoad %uint %undefined"), encoding="utf-8")
    failed = subprocess.run([*tool, broken, target], capture_output=True)
    assert failed.returncode and target.read_bytes() == valid

    # This input name previously overlapped the temporaries for %ballot_bit.
    renamed = combined.replace("%p2 ", "%ballot_ballot_bit ")
    assert renamed != combined
    for label, source in (("combined", combined), ("renamed_pointer", renamed), ("direct", direct),
                          ("atomics", atomics), ("qword", qword)):
        mixed = Path(directory) / f"{label}.spvasm"
        mixed.write_text(source, encoding="utf-8")
        mixed_target = Path(directory) / f"{label}.spv"
        subprocess.run([*tool, mixed, mixed_target], check=True)
        assert mixed_target.read_bytes()[:4] == bytes.fromhex("03022307")
    lowered = window.lower(combined)
    assert "OpMemoryModel Logical GLSL450" in lowered
    assert "OpGroupNonUniformBallotBitExtract" not in lowered
    assert "OpConvertUToPtr" not in lowered

    check_phi_parent(directory, tool, direct)

    lowered_atomics = window.lower(atomics).splitlines()
    for pointer in ("p1", "p2"):
        assert any(line.startswith(f"%{pointer} = OpAccessChain ")
                   and line.endswith(f"%x2w_heap %c0 %x2w_address_{pointer}_selected")
                   for line in lowered_atomics)
    # Each atomic keeps its operands, scope and semantics, but issues only in a branch taken for a valid address; the
    # result is a phi of that atomic's old value and zero, so an invalid address issues nothing.
    issued = [line for line in lowered_atomics if " = OpAtomic" in line]
    assert len(issued) == len(atomic_instructions)
    for line, original_line in zip(issued, atomic_instructions):
        result, rest = original_line.split(" = ", 1)
        stem = "%x2w_atomic_" + result[1:]
        assert line == f"{stem}_old = {rest}", line
        at = lowered_atomics.index(line)
        pointer = original_line.split()[4]
        assert lowered_atomics[at - 1] == f"{stem}_issue = OpLabel"
        assert lowered_atomics[at - 2].startswith("OpBranchConditional %x2w_address_" + pointer[1:])
        assert lowered_atomics[at - 3] == f"OpSelectionMerge {stem}_merge None"
        assert lowered_atomics[at + 1] == f"OpBranch {stem}_merge"
        assert lowered_atomics[at + 3].startswith(f"{result} = OpPhi %uint {stem}_old {stem}_issue %c0 ")

# Comments and 64-bit indices in the source assembly: the lowering must either keep the module valid or refuse it.
atomic_source = Path(__file__).with_name("atomic.spvasm").read_text(encoding="utf-8")
access = "%target = OpAccessChain %_ptr_PSB_uint %array %c1"
entry = 'OpEntryPoint GLCompute %main "main" %user_data'
assert access in atomic_source and entry in atomic_source
with tempfile.TemporaryDirectory() as folder:
    directory = Path(folder)
    # A trailing comment on an instruction the lowering rewrites, on the entry point, and one that
    # mentions a word the lowering looks for, must not change the result.
    commented = atomic_source.replace(access, access + " ; element one of the array")
    commented = commented.replace(entry, entry + " ; the interface lists the push constants")
    commented = commented.replace("OpReturn", "; a comment that says PushConstant\nOpReturn")
    validate_sampler(directory, "commented_original", commented)
    assert window.lower(commented) == window.lower(atomic_source)
    validate_sampler(directory, "commented_lowered", window.lower(commented))

    # A semicolon inside a quoted string literal must survive comment stripping.
    quoted_semi = atomic_source.replace("OpExecutionMode %main LocalSize 1 1 1",
                                        'OpExecutionMode %main LocalSize 1 1 1\nOpName %main "main;tag" ; trailing comment', 1)
    validate_sampler(directory, "quoted_semi_original", quoted_semi)
    quoted_lowered = window.lower(quoted_semi)
    validate_sampler(directory, "quoted_semi_lowered", quoted_lowered)
    assert 'OpName %main "main;tag"' in quoted_lowered.splitlines()

    # A string literal may span lines; a semicolon on a continuation line is text, not a comment, and the
    # literal keeps its inner newline and spacing.
    multiline = atomic_source.replace("OpExecutionMode %main LocalSize 1 1 1",
                                      'OpExecutionMode %main LocalSize 1 1 1\n%text = OpString "first line\n  int tmp = 0; }" ; note', 1)
    validate_sampler(directory, "multiline_original", multiline)
    multiline_lowered = window.lower(multiline)
    validate_sampler(directory, "multiline_lowered", multiline_lowered)
    assert '%text = OpString "first line\n  int tmp = 0; }"' in multiline_lowered

    # An index that is already 64 bits wide is used as it is (not converted again) and guarded against
    # shift wrap-around so a huge 64-bit index cannot drop high bits and alias an in-window address.
    wide_index = atomic_source.replace(access, "%index64 = OpUConvert %ulong %c1\n"
                                       "%target = OpAccessChain %_ptr_PSB_uint %array %index64")
    assert wide_index != atomic_source
    validate_sampler(directory, "wide_index_original", wide_index)
    wide_lowered = window.lower(wide_index)
    validate_sampler(directory, "wide_index_lowered", wide_lowered)
    wide_lines = wide_lowered.splitlines()
    assert "OpUConvert %ulong %index64" not in wide_lowered
    assert "%x2w_address_target_unshifted = OpShiftRightLogical %ulong %x2w_address_target_delta %x2w_shift" in wide_lines
    assert "%x2w_address_target_noshift = OpIEqual %bool %x2w_address_target_unshifted %index64" in wide_lines
    assert "%x2w_address_target_checked64 = OpLogicalAnd %bool %x2w_address_target_checked %x2w_address_target_noshift" in wide_lines
    assert "%x2w_address_target_selected = OpSelect %uint %x2w_address_target_checked64 %x2w_address_target_word %x2w_invalid" in wide_lines

    # A signed 64-bit index is bit-cast to %ulong before the shift and shift-wrap check so OpIEqual matches types.
    signed_wide = (atomic_source
                   .replace("%ulong = OpTypeInt 64 0",
                            "%ulong = OpTypeInt 64 0\n%long = OpTypeInt 64 1\n%index_s64 = OpConstant %long 1", 1)
                   .replace(access, "%target = OpAccessChain %_ptr_PSB_uint %array %index_s64"))
    validate_sampler(directory, "signed_wide_original", signed_wide)
    signed_wide_lowered = window.lower(signed_wide)
    validate_sampler(directory, "signed_wide_lowered", signed_wide_lowered)
    signed_wide_lines = signed_wide_lowered.splitlines()
    assert "%x2w_address_target_index = OpBitcast %ulong %index_s64" in signed_wide_lines
    assert "%x2w_address_target_noshift = OpIEqual %bool %x2w_address_target_unshifted %x2w_address_target_index" in signed_wide_lines

    narrow_lowered = window.lower(atomic_source)
    assert "OpUConvert %ulong %c1" in narrow_lowered, "a 32-bit index must still be widened"
print("physical, logical and combined SPIR-V validated; 8 atomics, bounded QWORD accesses, comments, 64-bit indices and rejection cases passed")
