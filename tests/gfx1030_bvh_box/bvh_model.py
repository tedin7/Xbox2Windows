# SPDX-License-Identifier: MIT
"""CPU model of the console BVH node test.

This models one ``image_bvh_intersect_ray`` / ``image_bvh64_intersect_ray``
call: a ray and a node pointer in, four result DWORDs out. Node traversal,
the stack and instance transforms belong to the calling shader and are not
part of this module.

The node formats and the box test follow the public software emulation in
Mesa's RADV driver (MIT). The triangle test is the watertight formulation
from Woop, Benthin and Wald, JCGT 2013, as RADV uses it. The result layout
is the 40-bit ray-tracing descriptor's triangle return mode 1:
``{t_num, t_denom, I_num, J_num}``. Arithmetic is binary32 at every step,
rounding to nearest even; scalar Python floats only carry the values.
"""

import math
import struct


MASK32 = (1 << 32) - 1
MASK48 = (1 << 48) - 1
MASK64 = (1 << 64) - 1
INVALID = 0xffffffff

NODE_BOX16 = 4
NODE_BOX32 = 5
SORT_ENABLE = 1 << 31
RETURN_BARYCENTRICS = 1 << 24

TRIANGLE_VERTICES = ((0, 1, 2), (1, 3, 2), (2, 3, 4), (2, 4, 0))
MISS = (math.inf, 1.0, 0.0, 0.0)
FLT_MAX = struct.unpack("<f", struct.pack("<I", 0x7f7fffff))[0]


def f32(value):
    """Round a Python float to binary32, nearest even."""
    if math.isnan(value) or math.isinf(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.inf if value > 0.0 else -math.inf


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def bits_f32(word):
    return struct.unpack("<f", struct.pack("<I", word & MASK32))[0]


def fadd(a, b):
    return f32(a + b)


def fsub(a, b):
    return f32(a - b)


def fmul(a, b):
    return f32(a * b)


def fmin(a, b):
    # IEEE minNum: the non-NaN operand when one is NaN.
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a <= b else b


def fmax(a, b):
    # IEEE maxNum: the non-NaN operand when one is NaN.
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a >= b else b


def node_address(descriptor, pointer, wide=False):
    """Return ``(kind, byte address)`` for a node pointer.

    The low three bits are the node kind. The rest addresses a 64-byte
    node: ``(pointer >> 3) << 6``. The 32-bit form adds that offset to a
    40-bit base held in descriptor DWORD 0 and the low byte of DWORD 1, in
    256-byte units. The 64-bit form uses the absolute pointer.
    """
    kind = pointer & 7
    if wide:
        return kind, ((pointer >> 3) << 6) & MASK64
    base = (((descriptor[1] & 0xff) << 32) | (descriptor[0] & MASK32)) << 8
    return kind, (base + ((pointer >> 3) << 6)) & MASK48


def descriptor(base=0, *, sort=True, return_barycentrics=True, size=0x3ffffffffff):
    """Build the four descriptor DWORDs a node test reads.

    The base argument is a 256-byte aligned byte address. The DWORDs hold
    it in 256-byte units, as the instruction's resource does.
    """
    assert base & 0xff == 0, "the descriptor base is 256-byte aligned"
    units = base >> 8
    return (
        units & MASK32,
        ((units >> 32) & 0xff) | (SORT_ENABLE if sort else 0),
        size & MASK32,
        ((size >> 32) & 0x3fffff)
        | (RETURN_BARYCENTRICS if return_barycentrics else 0)
        | (8 << 28),
    )


def _load_box(data, offset, kind, index):
    if kind == NODE_BOX32:
        start = offset + 16 + 24 * index
        return struct.unpack_from("<6f", data, start)
    start = offset + 16 + 12 * index
    return struct.unpack_from("<6e", data, start)


def _box_entry(lo, hi, origin, inverse, extent):
    # RADV removes infinities so an origin exactly on a face does not turn
    # zero times infinity into NaN. max then min is the order its NIR uses.
    inv = [fmin(fmax(inverse[i], -FLT_MAX), FLT_MAX) for i in range(3)]
    low = [fmul(fsub(lo[i], origin[i]), inv[i]) for i in range(3)]
    high = [fmul(fsub(hi[i], origin[i]), inv[i]) for i in range(3)]
    enter = fmax(fmax(fmin(low[0], high[0]), fmin(low[1], high[1])), fmin(low[2], high[2]))
    leave = fmin(fmin(fmax(low[0], high[0]), fmax(low[1], high[1])), fmax(low[2], high[2]))
    # min.x is the inactive-slot marker. Other NaN components are outside
    # the defined input; fmin/fmax above use the non-NaN operand.
    if math.isnan(lo[0]):
        return None
    if leave >= fmax(0.0, enter) and enter < extent:
        return enter
    return None


def _boxes(data, offset, kind, origin, inverse, extent, sort):
    children = struct.unpack_from("<4I", data, offset)
    indices = [INVALID] * 4
    distances = [math.inf] * 4
    for i in range(4):
        if children[i] == INVALID:
            continue
        coords = _load_box(data, offset, kind, i)
        enter = _box_entry(coords[:3], coords[3:], origin, inverse, extent)
        if enter is not None:
            indices[i] = children[i]
            distances[i] = enter
    if sort:
        # RADV's five-comparator network. A strict comparison leaves equal
        # entries in child order.
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            if distances[b] < distances[a]:
                distances[a], distances[b] = distances[b], distances[a]
                indices[a], indices[b] = indices[b], indices[a]
    return indices


def _node_vertex(data, offset, index):
    return struct.unpack_from("<3f", data, offset + 12 * index)


def _triangle(data, offset, kind, origin, direction, inverse):
    id_dword = struct.unpack_from("<I", data, offset + 60)[0]
    id_byte = (id_dword >> (8 * (kind & 3))) & 0xff
    i = id_byte & 3
    j = (id_byte >> 2) & 3
    first = (3 - i - j) & 3
    # The fourth map nibble is zero in the node format; a malformed ID that
    # names it reads the node's first vertex, as the arithmetic lands.
    mapped = TRIANGLE_VERTICES[kind & 3] + (0,)
    a, b, c = (_node_vertex(data, offset, mapped[n]) for n in (first, i, j))

    size = (abs(direction[0]), abs(direction[1]), abs(direction[2]))
    if size[0] >= size[1]:
        kz = 0 if size[0] >= size[2] else 2
    elif size[1] >= size[2]:
        kz = 1
    else:
        kz = 2
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    if direction[kz] < 0.0:
        kx, ky = ky, kx

    sz = inverse[kz]
    sx = fmul(direction[kx], sz)
    sy = fmul(direction[ky], sz)

    def shear(vertex):
        rel = [fsub(vertex[n], origin[n]) for n in range(3)]
        return (
            fsub(rel[kx], fmul(sx, rel[kz])),
            fsub(rel[ky], fmul(sy, rel[kz])),
            fmul(sz, rel[kz]),
        )

    ax, ay, az = shear(a)
    bx, by, bz = shear(b)
    cx, cy, cz = shear(c)
    u = fsub(fmul(cx, by), fmul(cy, bx))
    v = fsub(fmul(ax, cy), fmul(ay, cx))
    w = fsub(fmul(bx, ay), fmul(by, ax))

    behind = u < 0.0 or v < 0.0 or w < 0.0
    before = u > 0.0 or v > 0.0 or w > 0.0
    det = fadd(u, fadd(v, w))
    t = fadd(fadd(fmul(u, az), fmul(v, bz)), fmul(w, cz))
    ahead = (det > 0.0 and t >= 0.0) or (det < 0.0 and t <= 0.0)
    if (behind and before) or not ahead:
        return MISS
    # The instruction does not test the ray extent for triangles.
    return f32(t), f32(det), f32(v), f32(w)


def intersect(data, descriptor_words, pointer, *, origin, direction, inverse, extent,
              wide=False):
    """Return the four result DWORDs for one node and one ray."""
    kind, address = node_address(descriptor_words, pointer, wide)
    sort = bool(descriptor_words[1] & SORT_ENABLE)
    if kind == NODE_BOX32 or kind == NODE_BOX16:
        return _boxes(data, address, kind, origin, inverse, extent, sort)
    if kind <= 3:
        if not descriptor_words[3] & RETURN_BARYCENTRICS:
            raise ValueError("triangle return mode 0 is not modeled")
        # The instruction does not test the ray extent for triangles.
        values = _triangle(data, address, kind, origin, direction, inverse)
        return [f32_bits(value) for value in values]
    # Instance and procedural nodes are not read by this instruction.
    return [f32_bits(value) for value in MISS]
