"""Lower the named, DWORD-addressed SPIR-V profile to two logical storage buffers."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


from spirv_ballot_extract import lower_ballot_extract
from spirv_sampler_views import split_sampler_views


ID = r"%[A-Za-z0-9_]+"
CAST = re.compile(rf"({ID}) = OpConvertUToPtr %_ptr_PSB_(arr64|arri|arrf|arrv4f|arr|ulong|uint|int|float|v4float) ({ID})")
ACCESS = re.compile(rf"({ID}) = OpAccessChain %_ptr_PSB_uint ({ID}) ({ID})")
SIGNED_ACCESS = re.compile(rf"({ID}) = OpAccessChain %_ptr_PSB_int ({ID}) ({ID})")
SIGNED_POINTER_TYPES = ("%_ptr_PSB_int = OpTypePointer PhysicalStorageBuffer %int",
                        "%_ptr_PSB_arri = OpTypePointer PhysicalStorageBuffer %rt_int")
SIGNED_ARRAY_DECLARATIONS = ("%int = OpTypeInt 32 1", "%rt_int = OpTypeRuntimeArray %int", "OpDecorate %rt_int ArrayStride 4")
# A signed DWORD access is a plain load or store of one naturally aligned 32-bit value.
SIGNED_MEMORY_OPERANDS = ([], ["Aligned", "4"])
FLOAT_ACCESS = re.compile(rf"({ID}) = OpAccessChain %_ptr_PSB_float ({ID}) ({ID})")
FLOAT_POINTER_TYPES = ("%_ptr_PSB_float = OpTypePointer PhysicalStorageBuffer %float",
                       "%_ptr_PSB_arrf = OpTypePointer PhysicalStorageBuffer %rt_float")
FLOAT_ARRAY_DECLARATIONS = ("%float = OpTypeFloat 32", "%rt_float = OpTypeRuntimeArray %float",
                            "OpDecorate %rt_float ArrayStride 4")
# A float access is a plain load or store of one naturally aligned 32-bit value; atomics are not offered on floats.
FLOAT_MEMORY_OPERANDS = ([], ["Aligned", "4"])
VEC4_ACCESS = re.compile(rf"({ID}) = OpAccessChain %_ptr_PSB_v4float ({ID}) ({ID})")
VEC4_POINTER_TYPES = ("%_ptr_PSB_v4float = OpTypePointer PhysicalStorageBuffer %v4float",
                      "%_ptr_PSB_arrv4f = OpTypePointer PhysicalStorageBuffer %rt_v4float")
VEC4_ARRAY_DECLARATIONS = ("%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4",
                           "%rt_v4float = OpTypeRuntimeArray %v4float", "OpDecorate %rt_v4float ArrayStride 16")
# A four-float access is a plain load or store of one naturally aligned 16-byte value. Physical accesses must carry
# an Aligned operand, and only the natural alignment is accepted, as for a QWORD; other vector types are not lowered.
VEC4_MEMORY_OPERANDS = (["Aligned", "16"],)
CONSTANT = re.compile(rf"({ID}) = OpConstant %uint (\d+)")
SIGNED_ATOMICS = {"OpAtomicSMin": "OpSLessThan", "OpAtomicSMax": "OpSGreaterThan"}
# Atomics with no signedness, by operand count after the result type; their operands are bit-cast to unsigned.
SIGNED_WORD_ATOMICS = {"OpAtomicIAdd": 4, "OpAtomicExchange": 4, "OpAtomicAnd": 4, "OpAtomicOr": 4, "OpAtomicXor": 4,
                       "OpAtomicCompareExchange": 6}
WIDE_ACCESS = re.compile(rf"({ID}) = OpAccessChain %_ptr_PSB_ulong ({ID}) ({ID})")
WIDE_POINTER_TYPE = "%_ptr_PSB_ulong = OpTypePointer PhysicalStorageBuffer %ulong"
WIDE_ARRAY_POINTER_TYPE = "%_ptr_PSB_arr64 = OpTypePointer PhysicalStorageBuffer %rt_ulong"
WIDE_ARRAY_DECLARATIONS = ("%rt_ulong = OpTypeRuntimeArray %ulong", "OpDecorate %rt_ulong ArrayStride 8")
# A QWORD access is the plain load or store of one 64-bit value, naturally aligned.
WIDE_MEMORY_OPERANDS = ([], ["Aligned", "8"])
POINTER = re.compile(rf"({ID}) = OpTypePointer (\w+) ({ID})")
WORD_ATOMIC_OPERANDS = {
    "OpAtomicIAdd": 4,
    "OpAtomicExchange": 4,
    "OpAtomicUMin": 4,
    "OpAtomicUMax": 4,
    "OpAtomicAnd": 4,
    "OpAtomicOr": 4,
    "OpAtomicXor": 4,
    "OpAtomicCompareExchange": 6,
}


def address_name(dst):
    # Keep address temporaries separate from the generated x2w_ballot_ IDs.
    return "%x2w_address_" + dst[1:]


def word_pointer(dst, base, index, wide=False, defer=False, index_kind=None):
    """Select the last DWORD offset for an unaligned or unrepresentable address.

    Returns the lines and the ID of the boolean that is true only for an aligned, in-window address.
    A wide (QWORD, or "vec4" for sixteen bytes) or deferred pointer selects no substitute: it defines only the DWORD
    index {address}_word, and the access issues in a branch taken for a valid address. A sixteen-byte access ends
    twelve bytes before the window end, so that all four of its DWORDs are inside the window."""
    name = address_name(dst)
    scale, mask, end = ("%x2w_shift_vec4", "%x2w_mask_vec4", "%x2w_end_vec4") if wide == "vec4" else (
        "%x2w_shift_wide", "%x2w_mask_wide", "%x2w_end_wide") if wide else ("%x2w_shift", "%x2w_mask", "%x2w_end")
    lines = []
    addr = base
    if index:
        addr = name + "_address"
        if index_kind is None:
            lines.append(f"{name}_index = OpUConvert %ulong {index}")
            index = f"{name}_index"
        elif index_kind == "s64":
            lines.append(f"{name}_index = OpBitcast %ulong {index}")
            index = f"{name}_index"
        lines.extend([
            f"{name}_delta = OpShiftLeftLogical %ulong {index} {scale}",
            f"{addr} = OpIAdd %ulong {base} {name}_delta",
        ])
    lines.extend([
        f"{name}_low = OpBitwiseAnd %ulong {addr} {mask}",
        f"{name}_aligned = OpIEqual %bool {name}_low %x2w_zero64",
        f"{name}_range = OpULessThan %bool {addr} {end}",
        f"{name}_valid = OpLogicalAnd %bool {name}_aligned {name}_range",
    ])
    valid = name + "_valid"
    if index:
        valid = name + "_checked"
        lines.extend([
            f"{name}_nowrap = OpUGreaterThanEqual %bool {addr} {base}",
            f"{valid} = OpLogicalAnd %bool {name}_valid {name}_nowrap",
        ])
        if index_kind is not None:
            # A 64-bit index can drop high bits when shifted left; reject any shift that wraps.
            lines.extend([
                f"{name}_unshifted = OpShiftRightLogical %ulong {name}_delta {scale}",
                f"{name}_noshift = OpIEqual %bool {name}_unshifted {index}",
                f"{name}_checked64 = OpLogicalAnd %bool {valid} {name}_noshift",
            ])
            valid = name + "_checked64"
    lines.extend([
        f"{name}_word64 = OpShiftRightLogical %ulong {addr} %x2w_shift",
        f"{name}_word = OpUConvert %uint {name}_word64",
    ])
    if wide or defer:
        return lines, valid
    lines.extend([
        f"{name}_selected = OpSelect %uint {valid} {name}_word %x2w_invalid",
        f"{dst} = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 {name}_selected",
    ])
    return lines, valid


def wide_access(kind, stem, valid, label, word, fields):
    """Lower a QWORD load or store into two DWORD accesses inside a branch taken only for a valid address.

    A load yields zero and a store changes nothing otherwise. Returns the lines and the label that follows."""
    heap = "%_ptr_PSB_uint %x2w_heap %c0"
    lines = [
        f"OpSelectionMerge {stem}_merge None",
        f"OpBranchConditional {valid} {stem}_issue {stem}_merge",
        f"{stem}_issue = OpLabel",
        f"{stem}_next = OpIAdd %uint {word} %x2w_one",
        f"{stem}_low_ptr = OpAccessChain {heap} {word}",
        f"{stem}_high_ptr = OpAccessChain {heap} {stem}_next",
    ]
    if kind == "load":
        lines.extend([
            f"{stem}_low = OpLoad %uint {stem}_low_ptr",
            f"{stem}_high = OpLoad %uint {stem}_high_ptr",
            f"{stem}_low64 = OpUConvert %ulong {stem}_low",
            f"{stem}_high64 = OpUConvert %ulong {stem}_high",
            f"{stem}_high_shifted = OpShiftLeftLogical %ulong {stem}_high64 %x2w_shift_half",
            f"{stem}_value = OpBitwiseOr %ulong {stem}_low64 {stem}_high_shifted",
            f"OpBranch {stem}_merge",
            f"{stem}_merge = OpLabel",
            f"{fields[0]} = OpPhi %ulong {stem}_value {stem}_issue %x2w_zero64 {label}",
        ])
    else:
        lines.extend([
            f"{stem}_low = OpUConvert %uint {fields[2]}",
            f"{stem}_high64 = OpShiftRightLogical %ulong {fields[2]} %x2w_shift_half",
            f"{stem}_high = OpUConvert %uint {stem}_high64",
            f"OpStore {stem}_low_ptr {stem}_low Aligned 4",
            f"OpStore {stem}_high_ptr {stem}_high Aligned 4",
            f"OpBranch {stem}_merge",
            f"{stem}_merge = OpLabel",
        ])
    return lines, stem + "_merge"


def failure_semantics(name, value):
    """Return the ID and any new declaration of the compare-exchange failure semantics for an atomic's semantics.

    The failure path may not release, so Release is dropped and AcquireRelease becomes Acquire. The memory-class
    bits stay."""
    failure = value & ~0xc | (0x2 if value & 0x8 else 0)
    if failure == value:
        return name, []
    return f"%x2w_failure_{failure}", [f"%x2w_failure_{failure} = OpConstant %uint {failure}"]


def signed_word_atomic(fields, ptr, stem):
    """Lower a signed add, exchange, and, or, xor or compare-exchange on an in-window word.

    The atomic runs on the unsigned word with its scope and semantics unchanged; value and comparator are bit-cast
    and the old value is cast back by the caller."""
    values = fields[-2:] if fields[2] == "OpAtomicCompareExchange" else fields[-1:]
    names = [f"{stem}_value", f"{stem}_cmp"][:len(values)]
    lines = [f"{name} = OpBitcast %uint {value}" for name, value in zip(names, values)]
    lines.append(" ".join([f"{stem}_old = {fields[2]} %uint {ptr}", *fields[5:len(fields) - len(values)], *names]))
    return lines


def signed_access(fields, pointers, valid_of, words, label, stem, failures):
    """Lower a signed DWORD load, store, SMin or SMax into a branch taken only for a valid address.

    The heap holds unsigned words, so the value is bit-cast. A signed atomic min or max is a compare-exchange
    loop with a signed compare: SPIRV-Cross emits the unsigned InterlockedMin or InterlockedMax for it on a
    byte-address buffer. A load yields zero, a store changes nothing and an atomic returns zero otherwise.
    Returns the lines and the label that follows."""
    store = fields[0] == "OpStore"
    pointer = fields[1] if store else fields[4]
    atomic = not store and fields[2] in SIGNED_ATOMICS
    word_atomic = not store and fields[2] in SIGNED_WORD_ATOMICS
    operands = fields[3:] if store else fields[5:]
    if store:
        shape = len(fields) >= 3 and operands in SIGNED_MEMORY_OPERANDS
    elif atomic:
        shape = len(fields) == 8 and fields[1] == "=" and fields[3] == "%int"
    elif word_atomic:
        shape = len(fields) == 4 + SIGNED_WORD_ATOMICS[fields[2]] and fields[1] == "=" and fields[3] == "%int"
    else:
        shape = fields[1:4] == ["=", "OpLoad", "%int"] and fields[4:5] and operands in SIGNED_MEMORY_OPERANDS
    if not shape or pointer not in pointers or len(pointers) != 1:
        raise ValueError("unhandled physical word pointer use")
    ptr = stem + "_ptr"
    lines = [f"OpSelectionMerge {stem}_merge None", f"OpBranchConditional {valid_of[pointer]} {stem}_issue {stem}_merge",
             f"{stem}_issue = OpLabel", f"{ptr} = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 {words[pointer]}"]
    if store:
        lines.extend([f"{stem}_value = OpBitcast %uint {fields[2]}", " ".join(["OpStore", ptr, f"{stem}_value", *operands]),
                      f"OpBranch {stem}_merge", f"{stem}_merge = OpLabel"])
        return lines, stem + "_merge"
    if word_atomic:
        lines.extend(signed_word_atomic(fields, ptr, stem))
        lines.extend([f"OpBranch {stem}_merge", f"{stem}_merge = OpLabel",
                      f"{stem}_result = OpPhi %uint {stem}_old {stem}_issue %c0 {label}",
                      f"{fields[0]} = OpBitcast %int {stem}_result"])
        return lines, stem + "_merge"
    if atomic:
        scope, semantics, value = fields[5:8]
        lines.extend([
            f"{stem}_value = OpBitcast %uint {value}", f"OpBranch {stem}_head", f"{stem}_head = OpLabel",
            # The first guess is zero, not a read: a plain load would race with another invocation's atomic, and a
            # wrong guess only costs one more trip round the loop.
            f"{stem}_cur = OpPhi %uint %c0 {stem}_issue {stem}_seen {stem}_cont",
            f"OpLoopMerge {stem}_exit {stem}_cont None", f"OpBranch {stem}_body", f"{stem}_body = OpLabel",
            f"{stem}_cur_int = OpBitcast %int {stem}_cur",
            f"{stem}_pick = {SIGNED_ATOMICS[fields[2]]} %bool {stem}_cur_int {value}",
            f"{stem}_want = OpSelect %uint {stem}_pick {stem}_cur {stem}_value",
            f"{stem}_seen = OpAtomicCompareExchange %uint {ptr} {scope} {semantics} {failures[semantics]} {stem}_want {stem}_cur",
            f"{stem}_done = OpIEqual %bool {stem}_seen {stem}_cur",
            f"OpBranchConditional {stem}_done {stem}_exit {stem}_cont", f"{stem}_cont = OpLabel", f"OpBranch {stem}_head",
            f"{stem}_exit = OpLabel", f"OpBranch {stem}_merge", f"{stem}_merge = OpLabel",
            f"{stem}_old = OpPhi %uint {stem}_cur {stem}_exit %c0 {label}", f"{fields[0]} = OpBitcast %int {stem}_old"])
        return lines, stem + "_merge"
    lines.extend([f"{stem}_loaded = OpLoad %uint {ptr} {' '.join(operands)}".rstrip(), f"OpBranch {stem}_merge",
                  f"{stem}_merge = OpLabel", f"{stem}_old = OpPhi %uint {stem}_loaded {stem}_issue %c0 {label}",
                  f"{fields[0]} = OpBitcast %int {stem}_old"])
    return lines, stem + "_merge"


def float_access(fields, pointers, valid_of, words, label, stem):
    """Lower a float load or store into a branch taken only for a valid address.

    The heap holds unsigned words, so OpBitcast moves the value between %uint and %float. A bit-cast keeps every
    bit, including a NaN payload, the sign of a zero and a signalling-NaN bit. A load yields the bit pattern zero
    (0.0) and a store changes nothing otherwise. Returns the lines and the label that follows."""
    store = fields[0] == "OpStore"
    pointer = fields[1] if store else fields[4]
    operands = fields[3:] if store else fields[5:]
    if store:
        shape = len(fields) >= 3 and operands in FLOAT_MEMORY_OPERANDS and fields[2] not in pointers
    else:
        shape = fields[1:4] == ["=", "OpLoad", "%float"] and fields[4:5] and operands in FLOAT_MEMORY_OPERANDS
    if not shape or pointer not in pointers or len(pointers) != 1:
        raise ValueError("unhandled physical word pointer use")
    ptr = stem + "_ptr"
    lines = [f"OpSelectionMerge {stem}_merge None", f"OpBranchConditional {valid_of[pointer]} {stem}_issue {stem}_merge",
             f"{stem}_issue = OpLabel", f"{ptr} = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 {words[pointer]}"]
    if store:
        lines.extend([f"{stem}_value = OpBitcast %uint {fields[2]}", " ".join(["OpStore", ptr, f"{stem}_value", *operands]),
                      f"OpBranch {stem}_merge", f"{stem}_merge = OpLabel"])
        return lines, stem + "_merge"
    lines.extend([f"{stem}_loaded = OpLoad %uint {ptr} {' '.join(operands)}".rstrip(), f"OpBranch {stem}_merge",
                  f"{stem}_merge = OpLabel", f"{stem}_old = OpPhi %uint {stem}_loaded {stem}_issue %c0 {label}",
                  f"{fields[0]} = OpBitcast %float {stem}_old"])
    return lines, stem + "_merge"


def vec4_access(fields, pointers, valid_of, words, label, stem):
    """Lower a four-float load or store into four DWORD accesses inside one branch taken only for a valid address.

    The branch condition covers all sixteen bytes, so the four DWORDs are all accessed or none is. A load bit-casts
    each word to a float and builds the vector; a store extracts each lane and bit-casts it to a word. Outside the
    branch a load yields the zero vector (the bit pattern zero in every lane) and a store changes nothing.
    Returns the lines and the label that follows."""
    store = fields[0] == "OpStore"
    pointer = fields[1] if store else fields[4]
    operands = fields[3:] if store else fields[5:]
    if store:
        shape = len(fields) >= 3 and operands in VEC4_MEMORY_OPERANDS and fields[2] not in pointers
    else:
        shape = fields[1:4] == ["=", "OpLoad", "%v4float"] and fields[4:5] and operands in VEC4_MEMORY_OPERANDS
    if not shape or pointer not in pointers or len(pointers) != 1:
        raise ValueError("unhandled physical word pointer use")
    word = words[pointer]
    lines = [f"OpSelectionMerge {stem}_merge None", f"OpBranchConditional {valid_of[pointer]} {stem}_issue {stem}_merge",
             f"{stem}_issue = OpLabel", f"{stem}_word_1 = OpIAdd %uint {word} %x2w_one",
             f"{stem}_word_2 = OpIAdd %uint {word} %x2w_two", f"{stem}_word_3 = OpIAdd %uint {word} %x2w_three"]
    lines.extend(f"{stem}_ptr_{lane} = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 {word if lane == 0 else f'{stem}_word_{lane}'}"
                 for lane in range(4))
    if store:
        for lane in range(4):
            lines.extend([f"{stem}_lane_{lane} = OpCompositeExtract %float {fields[2]} {lane}",
                          f"{stem}_bits_{lane} = OpBitcast %uint {stem}_lane_{lane}",
                          f"OpStore {stem}_ptr_{lane} {stem}_bits_{lane} Aligned 4"])
        lines.extend([f"OpBranch {stem}_merge", f"{stem}_merge = OpLabel"])
        return lines, stem + "_merge"
    for lane in range(4):
        lines.extend([f"{stem}_bits_{lane} = OpLoad %uint {stem}_ptr_{lane}",
                      f"{stem}_lane_{lane} = OpBitcast %float {stem}_bits_{lane}"])
    lines.extend([f"{stem}_value = OpCompositeConstruct %v4float {' '.join(f'{stem}_lane_{lane}' for lane in range(4))}",
                  f"OpBranch {stem}_merge", f"{stem}_merge = OpLabel",
                  f"{fields[0]} = OpPhi %v4float {stem}_value {stem}_issue %x2w_vec4_zero {label}"])
    return lines, stem + "_merge"


def merge_pointer_types(lines):
    # The push-constant and physical pointer types can become the same logical
    # pointer type. SPIR-V validation requires a single such declaration.
    seen = {}
    aliases = {}
    kept = []
    for line in lines:
        match = POINTER.fullmatch(line)
        if match:
            name, storage, pointee = match.groups()
            key = storage, aliases.get(pointee, pointee)
            if key in seen:
                aliases[name] = seen[key]
                continue
            seen[key] = name
        kept.append(line)
    return "\n".join(re.sub(ID, lambda m: aliases.get(m[0], m[0]), line) for line in kept) + "\n"


def logical_lines(assembly):
    """Instruction lines without `;` comments; a quoted string may span lines and may hold a `;`.

    Quote and escape state carries across the whole module, and a literal that spans lines stays one
    entry with its inner newlines and spaces untouched.
    """
    lines, current = [], []
    quoted = escaped = False
    comment = False
    for char in assembly:
        if char == "\n" and not quoted:
            lines.append("".join(current).strip())
            current, comment = [], False
            continue
        if comment:
            continue
        if quoted and escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ";" and not quoted:
            comment = True
            continue
        current.append(char)
    lines.append("".join(current).strip())
    return [line for line in lines if line]


def lower(assembly):
    lines = logical_lines(split_sampler_views(lower_ballot_extract(assembly)))
    model = "OpMemoryModel PhysicalStorageBuffer64 GLSL450"
    if model not in lines or lines.count("%user_data = OpVariable %_ptr_PC_UserData PushConstant") != 1:
        raise ValueError("expected a physical-address module with the named user-data block")
    # Ballot lowering also emits x2w_ IDs, so check the namespace on the original input.
    if "%x2w_" in assembly or any(re.fullmatch(rf"OpDecorate {ID} DescriptorSet [12]", line) for line in lines):
        raise ValueError("address window needs unused identifier prefix x2w_ and descriptor sets 1 and 2")
    if sum(line.startswith("OpEntryPoint ") for line in lines) != 1:
        raise ValueError("expected one shader entry point")
    required = {"%uint = OpTypeInt 32 0", "%ulong = OpTypeInt 64 0", "%bool = OpTypeBool",
                "%c0 = OpConstant %uint 0", "%rt_uint = OpTypeRuntimeArray %uint",
                "OpDecorate %rt_uint ArrayStride 4", "OpDecorate %UserData Block",
                "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint",
                "%_ptr_PSB_arr = OpTypePointer PhysicalStorageBuffer %rt_uint"}
    if required.difference(lines):
        raise ValueError("unsupported physical pointer or scalar layout")
    annotations = ["OpDecorate %user_data DescriptorSet 1", "OpDecorate %user_data Binding 0",
                   "OpDecorate %user_data NonWritable", "OpDecorate %x2w_heap DescriptorSet 2",
                   "OpDecorate %x2w_heap Binding 0", "OpDecorate %x2w_block Block",
                   "OpMemberDecorate %x2w_block 0 Offset 0"]
    declarations = ["%x2w_block = OpTypeStruct %rt_uint",
                    "%x2w_heap_ptr = OpTypePointer StorageBuffer %x2w_block",
                    "%x2w_heap = OpVariable %x2w_heap_ptr StorageBuffer",
                    "%x2w_shift = OpConstant %uint 2", "%x2w_mask = OpConstant %ulong 3",
                    "%x2w_zero64 = OpConstant %ulong 0", "%x2w_end = OpConstant %ulong 4294967292",
                    "%x2w_invalid = OpConstant %uint 1073741823"]
    output = []
    arrays = {}
    wide_arrays = {}
    wide_words = {}
    valid_of = {}
    label = None
    origin = None
    # A guarded access splits its block, so a successor phi must name the block that now ends in the terminator.
    final_label = {}
    source_phis = []
    wide_accesses = 0
    physical_words = set()
    physical_qwords = set()
    physical_ints = set()
    physical_floats = set()
    physical_vec4s = set()
    for line in lines:
        cast = CAST.fullmatch(line)
        access = ACCESS.fullmatch(line)
        wide = WIDE_ACCESS.fullmatch(line)
        if cast and cast[2] == "int" or SIGNED_ACCESS.fullmatch(line):
            physical_ints.add((cast or SIGNED_ACCESS.fullmatch(line))[1])
        if cast and cast[2] == "float" or FLOAT_ACCESS.fullmatch(line):
            physical_floats.add((cast or FLOAT_ACCESS.fullmatch(line))[1])
        if cast and cast[2] == "v4float" or VEC4_ACCESS.fullmatch(line):
            physical_vec4s.add((cast or VEC4_ACCESS.fullmatch(line))[1])
        if access or cast and cast[2] == "uint":
            physical_words.add((access or cast)[1])
        if wide or cast and cast[2] == "ulong":
            physical_qwords.add((wide or cast)[1])
    uses_arrays64 = any(cast and cast[2] == "arr64" for cast in map(CAST.fullmatch, lines))
    if physical_qwords and WIDE_POINTER_TYPE not in lines or uses_arrays64 and (
            WIDE_ARRAY_POINTER_TYPE not in lines or not set(WIDE_ARRAY_DECLARATIONS) <= set(lines)):
        raise ValueError("unsupported physical pointer or scalar layout")
    signed_arrays = {}
    signed_words = {}
    signed_stores = 0
    if physical_ints or any(cast and cast[2] == "arri" for cast in map(CAST.fullmatch, lines)):
        if not set(SIGNED_POINTER_TYPES + SIGNED_ARRAY_DECLARATIONS) <= set(lines):
            raise ValueError("unsupported physical pointer or scalar layout")
    float_arrays = {}
    float_words = {}
    float_stores = 0
    if physical_floats or any(cast and cast[2] == "arrf" for cast in map(CAST.fullmatch, lines)):
        if not set(FLOAT_POINTER_TYPES + FLOAT_ARRAY_DECLARATIONS) <= set(lines):
            raise ValueError("unsupported physical pointer or scalar layout")
    vec4_arrays = {}
    vec4_words = {}
    vec4_stores = 0
    uses_vec4 = bool(physical_vec4s) or any(cast and cast[2] == "arrv4f" for cast in map(CAST.fullmatch, lines))
    if uses_vec4 and not set(VEC4_POINTER_TYPES + VEC4_ARRAY_DECLARATIONS) <= set(lines):
        raise ValueError("unsupported physical pointer or scalar layout")
    if physical_qwords:
        declarations += ["%x2w_one = OpConstant %uint 1", "%x2w_shift_wide = OpConstant %uint 3",
                         "%x2w_shift_half = OpConstant %uint 32", "%x2w_mask_wide = OpConstant %ulong 7",
                         # The last valid QWORD ends at the same DWORD as the last valid DWORD access.
                         "%x2w_end_wide = OpConstant %ulong 4294967288"]
    if uses_vec4:
        # A 16-byte access is valid up to the last aligned address whose final DWORD is below the window end, so the
        # end is the window end less twelve; a misaligned or straddling access is refused as a whole.
        declarations += ["%x2w_shift_vec4 = OpConstant %uint 4", "%x2w_mask_vec4 = OpConstant %ulong 15",
                         "%x2w_end_vec4 = OpConstant %ulong 4294967280", "%x2w_two = OpConstant %uint 2",
                         "%x2w_three = OpConstant %uint 3", "%x2w_vec4_zero = OpConstantNull %v4float"]
        if not physical_qwords:
            declarations.append("%x2w_one = OpConstant %uint 1")
    # A guarded access splits its block, which would strand an OpLoopMerge in the generated merge block.
    loop_headers, start = set(), 0
    for number, line in enumerate(lines):
        if line.endswith(" = OpLabel"):
            start = number
        elif line.startswith("OpLoopMerge "):
            loop_headers.update(range(start, number))
    wide_types = {m[1]: "s64" if m[2] == "1" else "u64"
                  for m in map(re.compile(rf"({ID}) = OpTypeInt 64 ([01])").fullmatch, lines) if m}
    wide_ids = {m[1]: wide_types[m[2]]
                for m in map(re.compile(rf"({ID}) = Op\w+ ({ID})").match, lines) if m and m[2] in wide_types}
    constants = {m[1]: int(m[2]) for m in map(CONSTANT.fullmatch, lines) if m}
    failures = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 8 and fields[2] in SIGNED_ATOMICS and fields[4] in physical_ints:
            if fields[6] not in constants:
                raise ValueError("signed atomic semantics must be a constant")
            failures[fields[6]], new = failure_semantics(fields[6], constants[fields[6]])
            declarations.extend(line for line in new if line not in declarations)
    for number, line in enumerate(lines):
        if line in ('OpExtension "SPV_KHR_physical_storage_buffer"', "OpCapability PhysicalStorageBufferAddresses",
                    "%_ptr_PSB_arr = OpTypePointer PhysicalStorageBuffer %rt_uint",
                    WIDE_POINTER_TYPE, WIDE_ARRAY_POINTER_TYPE, *SIGNED_POINTER_TYPES, *FLOAT_POINTER_TYPES,
                    *VEC4_POINTER_TYPES):
            continue
        if line.startswith("OpDecorate ") and annotations:
            output.extend(annotations)
            annotations = []
        if " = OpFunction " in line and declarations:
            output.extend(declarations)
            declarations = []
        if cast := CAST.fullmatch(line):
            dst, kind, addr = cast.groups()
            if kind == "arr":
                arrays[dst] = addr
            elif kind == "arr64":
                wide_arrays[dst] = addr
            elif kind == "arri":
                signed_arrays[dst] = addr
            elif kind == "arrf":
                float_arrays[dst] = addr
            elif kind == "arrv4f":
                vec4_arrays[dst] = addr
            elif kind == "v4float":
                new, valid_of[dst] = word_pointer(dst, addr, None, "vec4")
                vec4_words[dst] = address_name(dst) + "_word"
                output.extend(new)
            elif kind == "float":
                new, valid_of[dst] = word_pointer(dst, addr, None, defer=True)
                float_words[dst] = address_name(dst) + "_word"
                output.extend(new)
            elif kind == "int":
                new, valid_of[dst] = word_pointer(dst, addr, None, defer=True)
                signed_words[dst] = address_name(dst) + "_word"
                output.extend(new)
            elif kind == "ulong":
                new, valid_of[dst] = word_pointer(dst, addr, None, True)
                wide_words[dst] = address_name(dst) + "_word"
                output.extend(new)
            else:
                new, valid_of[dst] = word_pointer(dst, addr, None)
                output.extend(new)
            continue
        if access := ACCESS.fullmatch(line):
            dst, pointer, index = access.groups()
            if pointer not in arrays:
                raise ValueError("unknown physical array pointer")
            new, valid_of[dst] = word_pointer(dst, arrays[pointer], index, index_kind=wide_ids.get(index))
            output.extend(new)
            continue
        if signed := SIGNED_ACCESS.fullmatch(line):
            dst, pointer, index = signed.groups()
            if pointer not in signed_arrays:
                raise ValueError("unknown physical array pointer")
            new, valid_of[dst] = word_pointer(dst, signed_arrays[pointer], index, defer=True, index_kind=wide_ids.get(index))
            signed_words[dst] = address_name(dst) + "_word"
            output.extend(new)
            continue
        if floating := FLOAT_ACCESS.fullmatch(line):
            dst, pointer, index = floating.groups()
            if pointer not in float_arrays:
                raise ValueError("unknown physical array pointer")
            new, valid_of[dst] = word_pointer(dst, float_arrays[pointer], index, defer=True, index_kind=wide_ids.get(index))
            float_words[dst] = address_name(dst) + "_word"
            output.extend(new)
            continue
        if quad := VEC4_ACCESS.fullmatch(line):
            dst, pointer, index = quad.groups()
            if pointer not in vec4_arrays:
                raise ValueError("unknown physical array pointer")
            new, valid_of[dst] = word_pointer(dst, vec4_arrays[pointer], index, "vec4", index_kind=wide_ids.get(index))
            vec4_words[dst] = address_name(dst) + "_word"
            output.extend(new)
            continue
        if wide := WIDE_ACCESS.fullmatch(line):
            dst, pointer, index = wide.groups()
            if pointer not in wide_arrays:
                raise ValueError("unknown physical array pointer")
            new, valid_of[dst] = word_pointer(dst, wide_arrays[pointer], index, True, index_kind=wide_ids.get(index))
            wide_words[dst] = address_name(dst) + "_word"
            output.extend(new)
            continue
        if opened := re.fullmatch(rf"({ID}) = OpLabel", line):
            label = origin = opened[1]
        ids = set(re.findall(ID, line))
        wide_used = ids & physical_qwords
        if (wide_used or ids & (physical_ints | physical_floats | physical_vec4s)) and number in loop_headers:
            raise ValueError("a guarded physical access cannot share a block with OpLoopMerge")
        if wide_used:
            fields = line.split()
            wide_load = fields[1:4] == ["=", "OpLoad", "%ulong"] and fields[4:5] and fields[4] in wide_used
            wide_store = fields[0] == "OpStore" and fields[1] in wide_used and fields[2] not in wide_used
            operands = fields[5:] if wide_load else fields[3:]
            if ids & physical_words or len(wide_used) != 1 or not (wide_load or wide_store) \
                    or operands not in WIDE_MEMORY_OPERANDS:
                raise ValueError("unhandled physical word pointer use")
            wide_accesses += 1
            kind = "load" if wide_load else "store"
            pointer = fields[4] if wide_load else fields[1]
            stem = "%x2w_" + (f"load_{fields[0][1:]}" if wide_load else f"store_{wide_accesses}")
            new, label = wide_access(kind, stem, valid_of[pointer], label, wide_words[pointer], fields)
            final_label[origin] = label
            output.extend(new)
            continue
        if ids & physical_vec4s:
            if ids & (physical_words | physical_qwords | physical_ints | physical_floats):
                raise ValueError("unhandled physical word pointer use")
            fields = line.split()
            vec4_stores += fields[0] == "OpStore"
            stem = "%x2w_vec4_" + (f"result_{fields[0][1:]}" if fields[0] != "OpStore" else f"store_{vec4_stores}")
            new, label = vec4_access(fields, ids & physical_vec4s, valid_of, vec4_words, label, stem)
            final_label[origin] = label
            output.extend(new)
            continue
        if ids & physical_floats:
            if ids & (physical_words | physical_qwords | physical_ints):
                raise ValueError("unhandled physical word pointer use")
            fields = line.split()
            float_stores += fields[0] == "OpStore"
            stem = "%x2w_float_" + (f"result_{fields[0][1:]}" if fields[0] != "OpStore" else f"store_{float_stores}")
            new, label = float_access(fields, ids & physical_floats, valid_of, float_words, label, stem)
            final_label[origin] = label
            output.extend(new)
            continue
        if ids & physical_ints:
            if ids & (physical_words | physical_qwords):
                raise ValueError("unhandled physical word pointer use")
            fields = line.split()
            signed_stores += fields[0] == "OpStore"
            stem = "%x2w_signed_" + (f"result_{fields[0][1:]}" if fields[0] != "OpStore" else f"store_{signed_stores}")
            new, label = signed_access(fields, ids & physical_ints, valid_of, signed_words, label, stem, failures)
            final_label[origin] = label
            output.extend(new)
            continue
        used = ids & physical_words
        atomic_word = False
        if used:
            fields = line.split()
            reads_word = fields[1:4] == ["=", "OpLoad", "%uint"] and fields[4] in used
            writes_word = fields[0] == "OpStore" and fields[1] in used
            # The checked access chain retains the pointer ID. Keep the atomic's
            # result and operands, including scope and memory semantics, intact.
            atomic_word = (len(fields) >= 5 and fields[1] == "=" and fields[3] == "%uint"
                           and fields[4] in used and WORD_ATOMIC_OPERANDS.get(fields[2]) == len(fields) - 4)
            if not (reads_word or writes_word or atomic_word) or len(used) != 1:
                raise ValueError("unhandled physical word pointer use")
        if atomic_word:
            if number in loop_headers:
                raise ValueError("a guarded physical access cannot share a block with OpLoopMerge")
            # An address outside the window must not issue an atomic at the substitute index, so the atomic runs
            # only in a branch taken for a valid address; otherwise the result is zero and nothing is written.
            result = fields[0]
            stem = "%x2w_atomic_" + result[1:]
            output.extend([
                f"OpSelectionMerge {stem}_merge None",
                f"OpBranchConditional {valid_of[fields[4]]} {stem}_issue {stem}_merge",
                f"{stem}_issue = OpLabel",
                " ".join([f"{stem}_old", *fields[1:]]),
                f"OpBranch {stem}_merge",
                f"{stem}_merge = OpLabel",
                f"{result} = OpPhi %uint {stem}_old {stem}_issue %c0 {label}",
            ])
            label = stem + "_merge"
            final_label[origin] = label
            continue
        if line == model:
            line = "OpMemoryModel Logical GLSL450"
        elif line.startswith("OpEntryPoint "):
            line += " %x2w_heap"
        elif line == "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint":
            line = "%_ptr_PSB_uint = OpTypePointer StorageBuffer %uint"
        elif "PushConstant" in line:
            if line.split(" = ")[0] not in ("%_ptr_PC_UserData", "%_ptr_PC_uint", "%user_data"):
                raise ValueError("unrecognized push-constant declaration")
            line = line.replace("PushConstant", "StorageBuffer")
        elif any(word in line for word in ("PhysicalStorageBuffer", "OpConvertUToPtr", "OpConvertPtrToU", "%_ptr_PSB_arr",
                                                  "%_ptr_PSB_float", "%_ptr_PSB_v4float")):
            raise ValueError("unhandled physical pointer operation")
        if " = OpPhi " in line:
            source_phis.append(len(output))
        output.append(line)
    for at in source_phis:
        fields = output[at].split()
        fields[5::2] = [final_label.get(parent, parent) for parent in fields[5::2]]
        output[at] = " ".join(fields)
    return merge_pointer_types(output)


def _same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    if left.exists() and right.exists():
        return left.samefile(right)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assembler", default="spirv-as")
    parser.add_argument("--validator", default="spirv-val")
    args = parser.parse_args(argv)
    try:
        if _same_path(args.output, args.source):
            raise ValueError("use separate input assembly and output binary paths")
        if args.output.is_dir():
            raise ValueError(f"output path must not be a directory: {args.output}")
        original = args.source.read_text(encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
            path = Path(directory)
            binary = path / "module.spv"

            def validate(source):
                subprocess.run([args.assembler, "--target-env", "vulkan1.2", source, "-o", binary], check=True)
                subprocess.run([args.validator, "--target-env", "vulkan1.2", binary], check=True)

            snapshot = path / "original.spvasm"
            snapshot.write_text(original, encoding="utf-8")
            validate(snapshot)
            adapted = path / "logical.spvasm"
            adapted.write_text(lower(original), encoding="utf-8")
            validate(adapted)
            binary.replace(args.output)
    except (OSError, UnicodeDecodeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
