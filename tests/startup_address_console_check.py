"""Draw textures published by address (device slot 70) whose texels are in a console layout.

  <consumer> <backend>               fourteen textures, three cubes and nine volume mips, each drawn once; every
                                     pixel of the 512x128 target is compared with the texel the view names;
                                     then three held descriptions and one texture whose address is announced
                                     as XG rows after it was published
  <consumer> <backend> gate          the upload lifetime control described at check_gate
  <consumer> <backend> gate-ring     the same draws with X2W_TEST_ADDRESS_UPLOAD_RING=1: the 256 MiB ring
                                     refuses the uploads past its cap, the refused draws read the null view
                                     and the left half keeps payload 0
  <consumer> <backend> gate-ring-reuse  after completion, a new ring upload draws payload 18 over the right half
  <consumer> <backend> gate-ring-keep   list A uploads 32 payloads (512 MiB) from the ring, is submitted, completes
                                     and is never Reset; list B then draws payload 32 over the right half. Without
                                     X2W_D3D12_RING_RELEASE_AT_SUBMIT, A keeps its chunks: the ring refuses past 256 MiB
                                     and the right half reads the null view
  <consumer> <backend> gate-ring-keep-release  the same with X2W_D3D12_RING_RELEASE_AT_SUBMIT=1: A releases its
                                     chunks at its fenced submit, so B draws payload 32; A submitted again afterwards
                                     would read reused chunks, so it is refused before the execute: the target, read
                                     back after that submit, still holds payload 32 on the right (not A's payload 31)
  <consumer> <backend> linear-unpadded  the control described at check_linear_unpadded
  <consumer> <backend> cube-order    the control described at check_cube_order

The backing bytes are built here, independently of src/startup_stubs/gfx10_layout.h, from the address
equations of AMD addrlib's GFX10 pattern tables, written out below, and for the linear mode (0x100) from
HwlComputeSurfaceInfoLinear: rows whose pitch is the mip's rounded-up width aligned to 256 bytes, the mips
stored from the smallest to the largest. Reference: addrlib as Mesa carries it
under src/amd/addrlib at the revision tools/fetch_addrlib.py pins (Mesa 25.1.0, commit
5c142e46f3f6e752ed745fd48912ebb8fad67145), files src/gfx10/gfx10SwizzlePattern.h, src/gfx10/gfx10addrlib.cpp
and src/core/addrlib2.cpp, each with the notice "Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All
rights reserved. SPDX-License-Identifier: MIT". No addrlib code is included.
"""
import os
from pathlib import Path
import struct
import sys
import tempfile

from startup_address_srv_check import draw_texture

TARGET_W, TARGET_H = 512, 128
RGBA8, BC1 = 28, 71

# Address bits 0-7 of the 256-byte micro block by element size; above it y takes the even bits.
STANDARD = {4: "0 0 X0 X1 Y0 Y1 Y2 X2", 8: "0 0 0 X0 Y0 Y1 X1 X2"}
DISPLAY = {4: "0 0 X0 X1 Y0 Y1 Y2 X2", 8: "0 0 0 X0 Y0 X1 X2 Y1"}
MICRO = {4: (8, 8), 8: (8, 4)}


class Console:
    def __init__(self, width, height, mips, element, texels, layout, unpadded=False):
        self.width, self.height, self.mips, self.element, self.texels = width, height, mips, element, texels
        mode = layout & 0xff
        columns, rows = -(-width // texels), -(-height // texels)
        size = lambda v, mip: max(1, (v + (1 << mip) - 1) >> mip)
        self.linear = mode == 0
        if self.linear:
            # unpadded: rows exactly one mip wide, for the control at check_linear_unpadded.
            align = 1 if unpadded else 256 // element
            self.pitch, self.start, offset = {}, {}, 0
            for mip in range(mips - 1, -1, -1):
                self.pitch[mip] = -(-size(columns, mip) // align) * align
                self.start[mip] = offset
                offset += self.pitch[mip] * size(rows, mip) * element
            self.bytes = offset
            return
        self.log2 = {1: 8, 2: 8, 5: 12, 6: 12, 9: 16, 10: 16}[mode]
        bits = (DISPLAY if mode in (2, 6, 10) else STANDARD)[element].split()
        x_next, y_next = (sum(b[0] == axis for b in bits) for axis in "XY")
        for bit in range(8, self.log2):
            if bit & 1:
                bits.append(f"X{x_next}")
                x_next += 1
            else:
                bits.append(f"Y{y_next}")
                y_next += 1
        self.bits = bits
        self.block_w, self.block_h = 1 << x_next, 1 << y_next
        tail_mips = self.log2 - 4
        self.first_tail = mips
        if mips > 1:
            for mip in range(mips):
                if size(columns, mip) <= self.block_w // 2 and size(rows, mip) <= self.block_h and mips - mip <= tail_mips:
                    self.first_tail = mip
                    break
        block = 1 << self.log2
        self.start, self.blocks_wide, self.origin = {}, {}, {}
        offset = block if self.first_tail < mips else 0
        for mip in range(self.first_tail - 1, -1, -1):
            self.blocks_wide[mip] = -(-size(columns, mip) // self.block_w)
            self.start[mip] = offset
            offset += self.blocks_wide[mip] * -(-size(rows, mip) // self.block_h) * block
        self.bytes = offset
        for mip in range(self.first_tail, mips):
            m = tail_mips - 1 - (mip - self.first_tail)
            tail = 16 << m if m > 6 else m << 8
            x = sum(((tail >> (9 + 2 * b)) & 1) << b for b in range(6))
            y = sum(((tail >> (8 + 2 * b)) & 1) << b for b in range(6))
            self.origin[mip] = (x * MICRO[element][0], y * MICRO[element][1])

    def in_block(self, x, y):
        value = 0
        for bit, name in enumerate(self.bits):
            if name != "0" and ((x if name[0] == "X" else y) >> int(name[1:])) & 1:
                value |= 1 << bit
        return value

    def columns(self, mip):
        return -(-max(1, self.width >> mip) // self.texels)

    def rows(self, mip):
        return -(-max(1, self.height >> mip) // self.texels)

    def address(self, mip, x, y):
        if self.linear:
            return self.start[mip] + (y * self.pitch[mip] + x) * self.element
        if mip >= self.first_tail:
            ox, oy = self.origin[mip]
            return self.in_block(ox + x, oy + y)
        block = (y // self.block_h) * self.blocks_wide[mip] + x // self.block_w
        return self.start[mip] + (block << self.log2) + self.in_block(x % self.block_w, y % self.block_h)


# A volume in the 4 KiB standard mode (0x105): addrlib lays a 3D resource in a standard mode out in thick
# blocks (Gfx10Lib::HwlIsThick). The 12 address bits of an element follow the one-pipe row of
# GFX10_SW_4K_S3_PATINFO for its size; the block is 8x16x8 elements of 4 bytes, or 8x8x8 of 8. The volume is
# stored a layer of blocks (8 deep) at a time, and each layer holds that layer of every mip: the shared tail
# block first when the chain has one, then the mips from the smallest to the largest, blocks in x, then y.
# Mip sizes round up. A mip is in the tail when it is at most a block wide and half a block high and at most
# five mips remain; tail mip k from the end sits at byte (k << 8) of the tail block, which, read as 256-byte
# micro blocks (4x4x4 elements of 4 bytes, 4x2x4 of 8), is its origin in x and y. A BC1 volume is the volume
# of its 4x4 blocks, 8-byte elements one depth layer each: the sides are first rounded up to whole blocks,
# and every mip size after that is in blocks.
THICK_4K_S3 = {4: "0 0 X0 Y0 Z0 Y1 X1 Z1 Y2 X2 Z2 Y3", 8: "0 0 0 X0 Z0 Y0 X1 Z1 Y1 X2 Z2 Y2"}
MICRO_3D = {4: (4, 4), 8: (4, 2)}


class ConsoleVolume:
    def __init__(self, width, height, depth, element, mips=1, texels=1):
        width, height = -(-width // texels), -(-height // texels)
        self.width, self.height, self.depth = width, height, depth
        self.bits = THICK_4K_S3[element].split()
        self.block = [1 << sum(b[0] == axis for b in self.bits) for axis in "XYZ"]
        bw, bh, bd = self.block
        mw, mh = MICRO_3D[element]
        size = [(max(1, -(-width // (1 << m))), max(1, -(-height // (1 << m)))) for m in range(mips)]
        tail = next((m for m, (w, h) in enumerate(size) if mips > 1 and w <= bw and h <= bh // 2 and mips - m <= 5),
                    mips)
        self.wide, self.start, self.origin = {}, {}, {}
        offset = 4096 if tail < mips else 0
        for m in reversed(range(tail)):
            self.wide[m] = -(-size[m][0] // bw)
            self.start[m] = offset
            offset += self.wide[m] * -(-size[m][1] // bh) * 4096
        for m in range(tail, mips):
            at = (4 - (m - tail)) << 8
            self.start[m] = 0
            self.origin[m] = (mw * ((at >> 9 & 1) | (at >> 10 & 2)), mh * ((at >> 8 & 1) | (at >> 9 & 2)))
        self.layer = offset
        self.bytes = self.layer * -(-depth // bd)

    def address(self, x, y, z, mip=0):
        bw, bh, bd = self.block
        value = (z // bd) * self.layer + self.start[mip]
        if mip in self.origin:
            x, y = x + self.origin[mip][0], y + self.origin[mip][1]
        else:
            value += ((y // bh) * self.wide[mip] + x // bw) * 4096
        x, y = x % bw, y % bh
        for bit, name in enumerate(self.bits):
            coordinate = {"X": x % bw, "Y": y % bh, "Z": z % bd}.get(name[0])
            if coordinate is not None and (coordinate >> int(name[1:])) & 1:
                value |= 1 << bit
        return value


def rgba_texel(mip, x, y):
    return bytes(((7 * x + 40 * mip + 1) & 255, (5 * y + 3) & 255, (x ^ (3 * y)) & 255, 255))


def bc1_block(mip, x, y):
    # Both endpoints equal and every index 0: sixteen texels of one colour with full or empty channels.
    v = (5 * x + 3 * y + mip) % 8
    red, green, blue = v & 1, (v >> 1) & 1, (v >> 2) & 1
    colour = (0xf800 if red else 0) | (0x07e0 if green else 0) | (0x001f if blue else 0)
    return struct.pack("<HHI", colour, colour, 0), bytes((255 * red, 255 * green, 255 * blue, 255))


# name, width, height, mips, format, layout, viewed mip
CASES = [
    ("RGBA8 96x40 4 KiB standard", 96, 40, 1, RGBA8, 0x105, 0),
    ("BC1 192x96 4 KiB standard", 192, 96, 1, BC1, 0x105, 0),
    ("BC1 256x128 64 KiB display", 256, 128, 1, BC1, 0x10a, 0),
    ("RGBA8 256x256 chain 64 KiB display, mip 1", 256, 256, 4, RGBA8, 0x10a, 1),
    ("RGBA8 256x256 chain 64 KiB display, tail mip 2", 256, 256, 4, RGBA8, 0x10a, 2),
    ("RGBA8 256x256 chain 64 KiB display, tail mip 3", 256, 256, 4, RGBA8, 0x10a, 3),
    ("RGBA8 40x24 256 B standard", 40, 24, 1, RGBA8, 0x101, 0),
    # Linear: 100 texels are 400 bytes, so each row is padded to 512; 200 BC1 texels are 50 blocks, padded to 64.
    ("RGBA8 100x40 linear", 100, 40, 1, RGBA8, 0x100, 0),
    ("RGBA8 100x40 linear chain, mip 0", 100, 40, 4, RGBA8, 0x100, 0),
    ("RGBA8 100x40 linear chain, mip 2", 100, 40, 4, RGBA8, 0x100, 2),
    ("BC1 200x96 linear", 200, 96, 1, BC1, 0x100, 0),
    # Linear one-dimensional textures (a trailing "1d"): one row per mip, each mip padded to 256 bytes and
    # stored from the smallest up, so mip 0 of the 300-texel chain starts at 1536, not at 0.
    ("RGBA8 300 1D linear", 300, 1, 1, RGBA8, 0x100, 0, "1d"),
    ("RGBA8 300 1D linear chain, mip 0", 300, 1, 4, RGBA8, 0x100, 0, "1d"),
    ("RGBA8 300 1D linear chain, mip 2", 300, 1, 4, RGBA8, 0x100, 2, "1d"),
]


def volume_texel(x, y, z, mip=0):
    return bytes(((7 * x + 1) & 255, (5 * y + 3) & 255, (29 * z + 11) & 255, 255 - 16 * mip))


def volume_bc1_block(x, y, z, mip=0):
    """The block of BC1 volume block (x, y) of layer z: one colour, which differs between neighbours."""
    return bc1_block(mip, x, y + 3 * z)


# name, width, height, depth, format, layout, mips, viewed mip: RGBA8 or BC1 in the 4 KiB standard mode, one
# mip viewed as a Texture3D. Each target tile of the viewed mip's width and height draws one depth layer. The
# chains view a mip outside the tail and one inside it.
VOLUMES = [
    ("RGBA8 volume 17x17x17 4 KiB standard", 17, 17, 17, RGBA8, 0x105, 1, 0),
    ("RGBA8 volume 33x9x18 4 KiB standard", 33, 9, 18, RGBA8, 0x105, 1, 0),
    ("RGBA8 volume 32x32x32 6 mips, mip 1", 32, 32, 32, RGBA8, 0x105, 6, 1),
    ("RGBA8 volume 32x32x32 6 mips, mip 3 in the tail", 32, 32, 32, RGBA8, 0x105, 6, 3),
    ("RGBA8 volume 20x24x16 5 mips, mip 1", 20, 24, 16, RGBA8, 0x105, 5, 1),
    ("RGBA8 volume 20x24x16 5 mips, mip 3 in the tail", 20, 24, 16, RGBA8, 0x105, 5, 3),
    ("BC1 volume 48x40x12 4 KiB standard", 48, 40, 12, BC1, 0x105, 1, 0),
    ("BC1 volume 64x64x8 5 mips, mip 1", 64, 64, 8, BC1, 0x105, 5, 1),
    ("BC1 volume 64x64x8 5 mips, mip 3 in the tail", 64, 64, 8, BC1, 0x105, 5, 3),
]


def mip_sides(width, height, depth, mip):
    return max(1, width >> mip), max(1, height >> mip), max(1, depth >> mip)


def volume_colour(fmt, x, y, z, mip):
    """The RGBA8 colour a correct detile shows at texel (x, y, z) of a mip."""
    return volume_bc1_block(x // 4, y // 4, z, mip)[1] if fmt == BC1 else volume_texel(x, y, z, mip)


def build_volume(width, height, depth, mips=1, fmt=RGBA8):
    """Backing bytes of a volume and its mips, built from the S3 rows above."""
    texels, element = (4, 8) if fmt == BC1 else (1, 4)
    console = ConsoleVolume(width, height, depth, element, mips, texels)
    backing = bytearray(console.bytes)
    seen = set()
    for mip in range(mips):
        w, h, d = mip_sides(width, height, depth, mip)
        for z in range(d):
            for y in range(-(-h // texels)):
                for x in range(-(-w // texels)):
                    at = console.address(x, y, z, mip)
                    if at in seen or at + element > len(backing):
                        raise AssertionError(f"oracle address {at:#x} of mip {mip} ({x},{y},{z}) is shared or outside "
                                             "the volume")
                    seen.add(at)
                    if fmt == BC1:
                        backing[at:at + 8] = volume_bc1_block(x, y, z, mip)[0]
                    else:
                        backing[at:at + 4] = volume_texel(x, y, z, mip)
    return backing


def check_volume(directory, consumer, backend, case, held=None):
    name, width, height, depth, fmt, layout, mips, viewed = case
    source = directory / "texels.bin"
    source.write_bytes(build_volume(width, height, depth, mips, fmt))
    width, height, depth = mip_sides(width, height, depth, viewed)
    across = -(-TARGET_W // width)
    hlsl = ("Texture3D<float4> t : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            "  uint x = uint(p.x), y = uint(p.y);\n"
            f"  uint z = (x / {width} + (y / {height}) * {across}) % {depth};\n"
            f"  return t.Load(int4(x % {width}, y % {height}, z, 0));\n}}\n")
    trap_log = directory / "trap.log"
    modes = {"X2W_TEST_CONSOLE_VIEW": str(source), "X2W_TRAP_LOG": str(trap_log),
             "X2W_TEST_CONSOLE_DESC": f"{case[1]},{case[2]},{mips},{fmt},{layout},{viewed},{case[3]}"}
    actual = draw_texture(directory, consumer, backend, hlsl, modes, "tex3d")
    if actual is None:
        return 77
    if len(actual) != TARGET_W * TARGET_H * 4:
        raise AssertionError(f"{name}: readback length {len(actual)}")
    log = trap_log.read_text(errors="replace")
    if held:
        if any(actual):
            raise AssertionError(f"{name}: a held description drew texels")
        if f"slot70 texture kind 1: {held};" not in log or "published console layout" in log:
            raise AssertionError(f"{name}: no '{held}' line in the log, or a publication")
        print(f"{backend} {name}: held ({held}), {TARGET_W * TARGET_H} pixels read the null view")
        return 0
    layers = set()
    for y in range(TARGET_H):
        for x in range(TARGET_W):
            z = (x // width + (y // height) * across) % depth
            layers.add(z)
            pixel = actual[4 * (y * TARGET_W + x):4 * (y * TARGET_W + x) + 4]
            expect = volume_colour(fmt, x % width, y % height, z, viewed)
            if pixel != expect:
                raise AssertionError(f"{name}: pixel ({x},{y}) layer {z} actual {pixel.hex()} expected {expect.hex()}")
    if len(layers) != depth:
        raise AssertionError(f"{name}: the draw read {len(layers)} of {depth} layers")
    if "slot70 texture kind 1: published console layout" not in log:
        raise AssertionError(f"{name}: no 'published console layout' line in the log")
    print(f"{backend} {name}: {TARGET_W * TARGET_H} RGBA pixels exact over {depth} layers")
    return 0


# A volume past the size the XG stand-in reports: 64x8x40 RGBA8 is eight blocks across, one down and five
# layers of blocks (163,840 bytes); the desktop footprint (40 layers of 8 rows of 256 bytes) rounded to
# 64 KiB is 131,072. A volume has one slice, so the footprint is not counted once per layer.
HELD_VOLUME = (("RGBA8 volume 64x8x40 4 KiB standard past the XG size", 64, 8, 40, RGBA8, 0x105, 1, 0),
               "console_layout_beyond_xg_size")


def build(width, height, mips, fmt, layout, unpadded=False):
    """Backing bytes and, per mip, the RGBA texels a correct detile shows."""
    texels = 4 if fmt == BC1 else 1
    element = 8 if fmt == BC1 else 4
    images = []
    console = Console(width, height, mips, element, texels, layout, unpadded)
    backing = bytearray(console.bytes)
    seen = set()
    for mip in range(mips):
        w, h = max(1, width >> mip), max(1, height >> mip)
        image = bytearray(w * h * 4)
        for y in range(console.rows(mip)):
            for x in range(console.columns(mip)):
                at = console.address(mip, x, y)
                if at in seen or at + element > len(backing):
                    raise AssertionError(f"oracle address {at:#x} of mip {mip} ({x},{y}) is shared or outside the texture")
                seen.add(at)
                if fmt == BC1:
                    block, colour = bc1_block(mip, x, y)
                    backing[at:at + 8] = block
                    for row in range(4 * y, min(4 * y + 4, h)):
                        for column in range(4 * x, min(4 * x + 4, w)):
                            image[4 * (row * w + column):4 * (row * w + column) + 4] = colour
                else:
                    backing[at:at + 4] = rgba_texel(mip, x, y)
                    image[4 * (y * w + x):4 * (y * w + x) + 4] = rgba_texel(mip, x, y)
        images.append(image)
    return backing, images


# Held descriptions: nothing is published and the draw reads a null view (zeros). The first is the first
# case above over an address announced as an XG copy of rows. The second runs past the size the XG
# stand-in reports: 257x257 RGBA8 in 64 KiB blocks is nine blocks (589,824 bytes), and the desktop
# footprint (257 rows of 1,280 bytes) rounded to 64 KiB is 393,216.
HELD = [
    (("RGBA8 96x40 4 KiB standard over XG rows", 96, 40, 1, RGBA8, 0x105, 0), {"X2W_TEST_CONSOLE_XG_ROWS": "1"},
     "console_layout_xg_rows"),
    (("RGBA8 257x257 64 KiB standard past the XG size", 257, 257, 1, RGBA8, 0x109, 0), {}, "console_layout_beyond_xg_size"),
]

# Published, then the address is announced as an XG copy of rows before the draw: the texture published
# there must sample the null view, as a partial-tile BC view does.
LATE = (("RGBA8 96x40 4 KiB standard, XG rows after publication", 96, 40, 1, RGBA8, 0x105, 0),
        {"X2W_TEST_CONSOLE_XG_ROWS": "after"})


def check(directory, consumer, backend, case, extra=None, held=None, late=False, unpadded=False):
    name, width, height, mips, fmt, layout, viewed, *dimension = case
    one_d = dimension == ["1d"]
    backing, images = build(width, height, mips, fmt, layout, unpadded)
    source = directory / "texels.bin"
    source.write_bytes(backing)
    w, h = max(1, width >> viewed), max(1, height >> viewed)
    if one_d:
        hlsl = ("Texture1D<float4> t : register(t0);\n"
                "float4 main(float4 p : SV_Position) : SV_Target {\n"
                f"  return t.Load(int2(uint(p.x) % {w}, 0));\n}}\n")
    else:
        hlsl = ("Texture2D<float4> t : register(t0);\n"
                "float4 main(float4 p : SV_Position) : SV_Target {\n"
                f"  return t.Load(int3(uint(p.x) % {w}, uint(p.y) % {h}, 0));\n}}\n")
    trap_log = directory / "trap.log"
    desc = f"{width},{height},{mips},{fmt},{layout},{viewed}" + (",1d" if one_d else "")
    modes = {"X2W_TEST_CONSOLE_VIEW": str(source), "X2W_TRAP_LOG": str(trap_log),
             "X2W_TEST_CONSOLE_DESC": desc, **(extra or {})}
    actual = draw_texture(directory, consumer, backend, hlsl, modes, "tex1d" if one_d else "tex2d")
    if actual is None:
        return 77
    if len(actual) != TARGET_W * TARGET_H * 4:
        raise AssertionError(f"{name}: readback length {len(actual)}")
    log = trap_log.read_text(errors="replace")
    if held:
        if any(actual):
            raise AssertionError(f"{name}: a held description drew texels")
        if f"slot70 texture kind 1: {held};" not in log or "published console layout" in log:
            raise AssertionError(f"{name}: no '{held}' line in the log, or a publication")
        print(f"{backend} {name}: held ({held}), {TARGET_W * TARGET_H} pixels read the null view")
        return 0
    if late:
        if any(actual):
            raise AssertionError(f"{name}: the published texture drew texels after an XG copy of rows was announced")
        if "slot70 texture kind 1: published console layout" not in log:
            raise AssertionError(f"{name}: no 'published console layout' line in the log")
        print(f"{backend} {name}: published, then {TARGET_W * TARGET_H} pixels read the null view")
        return 0
    image = images[viewed]
    wrong = 0
    for y in range(TARGET_H):
        for x in range(TARGET_W):
            pixel = actual[4 * (y * TARGET_W + x):4 * (y * TARGET_W + x) + 4]
            expect = image[4 * ((y % h) * w + x % w):4 * ((y % h) * w + x % w) + 4]
            if pixel != expect and not unpadded:
                raise AssertionError(f"{name}: pixel ({x},{y}) actual {pixel.hex()} expected {expect.hex()}" + chr(10) + log[-6000:])
            wrong += pixel != expect
    if "slot70 texture kind 1: published console layout" not in log:
        raise AssertionError(f"{name}: no 'published console layout' line in the log")
    if unpadded:
        if not wrong:
            raise AssertionError(f"{name}: texels written without the 256-byte padding drew exactly; the padding is not checked")
        print(f"{backend} {name}, without the 256-byte padding: published, {wrong} of {TARGET_W * TARGET_H} pixels differ")
        return 0
    print(f"{backend} {name}: {TARGET_W * TARGET_H} RGBA pixels exact")
    return 0


def check_linear_unpadded(directory, consumer, backend):
    """The linear cases' negative control. The first 2D linear texture has its rows written one mip wide
    (400 bytes) instead of at the 256-byte pitch (512), and the 1D chain viewed at mip 2 has its mips packed
    without padding, so mip 2 starts at 152 instead of 256. The runtime reads the padded layout, so each draw
    must differ from the texels; an exact draw means the linear cases do not test the padding."""
    for name in ("RGBA8 100x40 linear", "RGBA8 300 1D linear chain, mip 2"):
        case = next(c for c in CASES if c[0] == name)
        with tempfile.TemporaryDirectory() as temporary:
            result = check(Path(temporary), consumer, backend, case, unpadded=True)
        if result:
            return result
    return 0


GATE_PAYLOADS = 18
KEEP_PAYLOADS = 32  # tests/startup_stream_draw.cpp, X2W_TEST_CONSOLE_KEEP


# Cube views: a two-dimensional array of six in a console layout, each slice one face (+X, -X, +Y, -Y, +Z,
# -Z) and the slices one slice size apart (the whole mip chain of a slice). TextureCube has no Load, so the
# shader point-samples the centre of every texel of every face of the viewed mip: target column x picks face
# (x mod 6w) div w and column x mod w, row y picks row y mod w. The directions are the D3D cube face rules.
# name, side, mips, format, layout, viewed mip
CUBES = [
    ("RGBA8 64x64 cube 4 KiB standard", 64, 1, RGBA8, 0x105, 0),
    ("RGBA8 64x64 cube 4 KiB standard chain, mip 1", 64, 4, RGBA8, 0x105, 1),
    ("BC1 64x64 cube 4 KiB standard", 64, 1, BC1, 0x105, 0),
]


def cube_texel(face, mip, x, y):
    texel = bytearray(rgba_texel(mip, x, y))
    texel[2] = (texel[2] + 37 * face) & 255
    return bytes(texel)


def build_cube(side, mips, fmt, layout, reversed_faces=False):
    """Backing bytes and images[face][mip]. reversed_faces writes face f into slice 5 - f (the control)."""
    texels = 4 if fmt == BC1 else 1
    element = 8 if fmt == BC1 else 4
    console = Console(side, side, mips, element, texels, layout)
    backing = bytearray(6 * console.bytes)
    images = [[None] * mips for _ in range(6)]
    for face in range(6):
        base = (5 - face if reversed_faces else face) * console.bytes
        for mip in range(mips):
            w = max(1, side >> mip)
            image = bytearray(w * w * 4)
            for y in range(console.rows(mip)):
                for x in range(console.columns(mip)):
                    at = base + console.address(mip, x, y)
                    if fmt == BC1:
                        block, colour = bc1_block(mip + 3 * face, x, y)
                        backing[at:at + 8] = block
                        for row in range(4 * y, min(4 * y + 4, w)):
                            for column in range(4 * x, min(4 * x + 4, w)):
                                image[4 * (row * w + column):4 * (row * w + column) + 4] = colour
                    else:
                        texel = cube_texel(face, mip, x, y)
                        backing[at:at + 4] = texel
                        image[4 * (y * w + x):4 * (y * w + x) + 4] = texel
            images[face][mip] = image
    return backing, images


def check_cube(directory, consumer, backend, case, reversed_faces=False):
    name, side, mips, fmt, layout, viewed = case
    backing, images = build_cube(side, mips, fmt, layout, reversed_faces)
    source = directory / "texels.bin"
    source.write_bytes(backing)
    w = max(1, side >> viewed)
    hlsl = ("TextureCube<float4> t : register(t0);\n"
            "SamplerState s : register(s0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            f"  uint x = uint(p.x) % {6 * w}, y = uint(p.y) % {w}, face = x / {w};\n"
            f"  float a = (float(x % {w}) + 0.5) / {w} * 2 - 1, b = (float(y) + 0.5) / {w} * 2 - 1;\n"
            "  float3 d = face == 0 ? float3(1, -b, -a) : face == 1 ? float3(-1, -b, a) :\n"
            "             face == 2 ? float3(a, 1, b) : face == 3 ? float3(a, -1, -b) :\n"
            "             face == 4 ? float3(a, -b, 1) : float3(-a, -b, -1);\n"
            "  return t.SampleLevel(s, d, 0);\n}\n")
    trap_log = directory / "trap.log"
    modes = {"X2W_TEST_CONSOLE_VIEW": str(source), "X2W_TRAP_LOG": str(trap_log),
             "X2W_TEST_CONSOLE_DESC": f"{side},{side},{mips},{fmt},{layout},{viewed},cube"}
    actual = draw_texture(directory, consumer, backend, hlsl, modes,
                          binding="srv 0 texcube table 0 0\nsmp 0 0 0 0 0")
    if actual is None:
        return 77
    if len(actual) != TARGET_W * TARGET_H * 4:
        raise AssertionError(f"{name}: readback length {len(actual)}")
    wrong = 0
    for y in range(TARGET_H):
        for x in range(TARGET_W):
            pixel = actual[4 * (y * TARGET_W + x):4 * (y * TARGET_W + x) + 4]
            column = x % (6 * w)
            image = images[column // w][viewed]
            expect = image[4 * ((y % w) * w + column % w):4 * ((y % w) * w + column % w) + 4]
            if pixel != expect and not reversed_faces:
                raise AssertionError(f"{name}: pixel ({x},{y}) actual {pixel.hex()} expected {expect.hex()}")
            wrong += pixel != expect
    if "slot70 texture kind 1: published console layout" not in trap_log.read_text(errors="replace"):
        raise AssertionError(f"{name}: no 'published console layout' line in the log")
    if reversed_faces:
        if not wrong:
            raise AssertionError(f"{name}: faces written in reverse slice order drew exactly; the face order is not checked")
        print(f"{backend} {name}, faces in reverse slice order: published, {wrong} of {TARGET_W * TARGET_H} pixels differ")
        return 0
    print(f"{backend} {name}: {TARGET_W * TARGET_H} RGBA pixels exact over 6 faces")
    return 0


def check_cube_order(consumer, backend):
    """The cube cases' negative control: the first cube with face f written into slice 5 - f. The runtime
    reads slice f as face f, so the draw must differ from the texels."""
    with tempfile.TemporaryDirectory() as temporary:
        return check_cube(Path(temporary), consumer, backend, CUBES[0], reversed_faces=True)


def gate_colour(payload):
    """The colour of every texel of a payload; tests/startup_address_srv.cpp writes the same bytes."""
    return bytes(((17 * payload + 3) & 255, (255 - 9 * payload) & 255, payload ^ 0x5a, 255))


def check_gate(directory, consumer, backend, mode):
    """Uploads recorded while the queue cannot run must still hold their bytes when it does.

    The normal fixture records and discards a draw using payload 0 before the first submit retries that
    payload. The fixture makes the queue wait on a fence it controls, then records and submits eighteen lists. Each
    republishes one 2048x2048 RGBA8 texture (64 KiB standard layout, 16 MiB) from the same address with a new
    payload and draws it: payload 0 over the left half of the target, the others over the right half. The list
    object and its allocator are reset between submits. Nothing may complete while the fence is unsignalled.
    The fixture then signals the fence, waits for the last submit and reads the target back: the left half
    must hold payload 0 and the right half payload 17.

    The shared ring refuses later reservations once its capacity is exhausted.
    The left half must retain payload 0 and the final right-half draw must
    read the null view.
    """
    hlsl = ("Texture2D<float4> t : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            "  return t.Load(int3(uint(p.x) % 2048, uint(p.y) % 2048, 0));\n}\n")
    trap_log = directory / "trap.log"
    modes = {"X2W_TEST_CONSOLE_GATE": "1", "X2W_TRAP_LOG": str(trap_log)}
    ring = mode != "gate"
    reuse = mode == "gate-ring-reuse"
    keep = mode in ("gate-ring-keep", "gate-ring-keep-release")
    release = mode == "gate-ring-keep-release"
    if keep:
        modes["X2W_TEST_CONSOLE_KEEP"] = "1"
    else:
        os.environ.pop("X2W_TEST_CONSOLE_KEEP", None)
    if release:
        modes["X2W_D3D12_RING_RELEASE_AT_SUBMIT"] = "1"
    else:
        os.environ.pop("X2W_D3D12_RING_RELEASE_AT_SUBMIT", None)
    if reuse:
        modes["X2W_TEST_ADDRESS_UPLOAD_RING_REUSE"] = "1"
    else:
        os.environ.pop("X2W_TEST_ADDRESS_UPLOAD_RING_REUSE", None)
    if ring:
        modes["X2W_TEST_ADDRESS_UPLOAD_RING"] = "1"
    else:
        # The Windows switch checks for presence, including an empty value.
        os.environ.pop("X2W_TEST_ADDRESS_UPLOAD_RING", None)
    actual = draw_texture(directory, consumer, backend, hlsl, modes)
    if actual is None:
        return 77
    if len(actual) != TARGET_W * TARGET_H * 4:
        raise AssertionError(f"gate: readback length {len(actual)}")
    first = gate_colour(0)
    last = bytes(4) if ring else gate_colour(GATE_PAYLOADS - 1)
    if reuse:
        last = gate_colour(GATE_PAYLOADS)
    if release:
        last = gate_colour(KEEP_PAYLOADS)
    wrong = {"left": 0, "right": 0}
    seen = set()
    for y in range(TARGET_H):
        for x in range(TARGET_W):
            pixel = actual[4 * (y * TARGET_W + x):4 * (y * TARGET_W + x) + 4]
            side = "left" if x < TARGET_W // 2 else "right"
            if pixel != (first if side == "left" else last):
                wrong[side] += 1
                seen.add(pixel.hex())
    log = trap_log.read_text(errors="replace")
    released = f"x2w-draw by-address uploads: {GATE_PAYLOADS} submits completed and released"
    half = TARGET_W * TARGET_H // 2
    if keep:
        if wrong["left"] or wrong["right"]:
            ring_lines = "\n".join(line for line in log.splitlines() if "upload ring" in line)
            raise AssertionError(f"{mode}: {wrong} wrong pixels, found {sorted(seen)[:4]}; expected {first.hex()} left, "
                                 f"{last.hex()} right\n{ring_lines}")
        refused = "x2w-draw upload ring reservation refused: no unused capacity" in log
        reported = "were reused; refused, it is not executed" in log
        if release and not reported:
            raise AssertionError(f"{mode}: list A submitted again after B reused its chunk was not refused")
        if not release and (not refused or reported):
            raise AssertionError(f"{mode}: expected the 256 MiB refusal and no resubmit refusal (refused={refused}, reported={reported})")
        print(f"{backend} {mode}: {half} left pixels hold payload 0 and {half} right pixels hold {last.hex()} after A's "
              "resubmit; "
              + ("A's chunks were released at its fenced submit and its stale resubmit was refused, not executed" if release
                 else "the never-Reset list kept its chunks and the ring refused past its cap"))
        return 0
    if ring:
        if wrong["left"] or wrong["right"]:
            raise AssertionError(f"{mode}: {wrong} wrong pixels, found {sorted(seen)[:4]}; expected {first.hex()} left, {last.hex()} right")
        if "x2w-draw upload ring reservation refused: no unused capacity" not in log:
            raise AssertionError("gate-ring: no upload reservation refusal")
        if "x2w-draw by-address uploads:" in log:
            raise AssertionError("gate-ring: the ring control retained upload buffers")
        print(f"{backend} gate, uploads from the shared ring: exhaustion refused; "
              f"{half} left pixels hold payload 0 and {half} right pixels hold {last.hex()}")
        return 0
    if wrong["left"] or wrong["right"]:
        raise AssertionError(f"gate: {wrong} wrong pixels, found {sorted(seen)[:4]}; expected {first.hex()} left, {last.hex()} right")
    if released not in log or f"at most {GATE_PAYLOADS} pending" not in log:
        raise AssertionError("gate: no line reporting the retained uploads released after completion:\n"
                             + "\n".join(line for line in log.splitlines() if "by-address" in line))
    print(f"{backend} gate: {GATE_PAYLOADS} submits held behind the fence; after release the left half holds payload 0 and the "
          f"right half payload {GATE_PAYLOADS - 1}, {TARGET_W * TARGET_H} RGBA pixels exact; {GATE_PAYLOADS} retained uploads released")
    return 0


GATES = ("gate", "gate-ring", "gate-ring-reuse", "gate-ring-keep", "gate-ring-keep-release")

if __name__ == "__main__":
    modes = GATES + ("linear-unpadded", "cube-order")
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] not in modes):
        usage = "usage: startup_address_console_check.py <consumer> <backend> [" + " | ".join(modes) + "]\n"
        sys.exit(usage + "\n\n".join(__doc__.split("\n\n")[:2]))
    if len(sys.argv) == 4 and sys.argv[3] == "linear-unpadded":
        sys.exit(check_linear_unpadded(None, sys.argv[1], sys.argv[2]))
    if len(sys.argv) == 4 and sys.argv[3] == "cube-order":
        sys.exit(check_cube_order(sys.argv[1], sys.argv[2]))
    if len(sys.argv) == 4:
        with tempfile.TemporaryDirectory() as temporary:
            sys.exit(check_gate(Path(temporary), sys.argv[1], sys.argv[2], sys.argv[3]))
    for case in CASES:
        with tempfile.TemporaryDirectory() as temporary:
            result = check(Path(temporary), *sys.argv[1:3], case)
            if result:
                sys.exit(result)
    for case in CUBES:
        with tempfile.TemporaryDirectory() as temporary:
            result = check_cube(Path(temporary), *sys.argv[1:3], case)
            if result:
                sys.exit(result)
    for case in VOLUMES:
        with tempfile.TemporaryDirectory() as temporary:
            result = check_volume(Path(temporary), *sys.argv[1:3], case)
            if result:
                sys.exit(result)
    for case, extra, held in HELD:
        with tempfile.TemporaryDirectory() as temporary:
            result = check(Path(temporary), *sys.argv[1:3], case, extra, held)
            if result:
                sys.exit(result)
    with tempfile.TemporaryDirectory() as temporary:
        result = check_volume(Path(temporary), *sys.argv[1:3], *HELD_VOLUME)
        if result:
            sys.exit(result)
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:3], LATE[0], LATE[1], late=True))
