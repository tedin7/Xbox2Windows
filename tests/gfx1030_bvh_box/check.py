# SPDX-License-Identifier: MIT
"""Capture all four result DWORDs from gfx1030 FP32 BVH box instructions."""
import argparse
import ctypes as ct
import hashlib
import os
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
OBJECT = ROOT / "out" / "native-gfx1030-bvh-box" / "box.hsaco"
INVALID = 0xffffffff
NODE_BYTES = 512
SORT_ENABLE = 1 << 31
RESOURCE_TYPE = 1 << 31
TRANSFER_TO_DEVICE = 1
TRANSFER_TO_HOST = 2
MASK32 = (1 << 32) - 1


def box(child, depth, x=-8):
    return child, (x, -8, depth), (x + 16, 8, depth + 0.5)


NODES = {
    0: (box(0x25, 4), box(0x1d, 1), box(0x2d, 2, x=32), box(INVALID, 3)),
    256: (box(0x45, 7), box(0x3d, 5), box(0x35, 3), box(0x0d, 1)),
}
ORIGIN = (0.5, -0.5, -1.0)
FORWARD = (0.25, -0.125, 0.5)
RAYS = (
    ("sorted", 0, 19.0, ORIGIN, FORWARD, (0x1d, 0x25, INVALID, INVALID)),
    ("clipped", 0, 6.0, ORIGIN, FORWARD, (0x1d, INVALID, INVALID, INVALID)),
    ("all_miss", 0, 2.0, ORIGIN, FORWARD, (INVALID,) * 4),
    ("all_four", 256, 19.0, ORIGIN, FORWARD, (0x0d, 0x35, 0x3d, 0x45)),
    ("reverse", 0, 21.0, (0.0, 0.0, 10.0), (0.25, -0.125, -0.5), (0x25, 0x1d, INVALID, INVALID)),
)


def node_image():
    data = bytearray(NODE_BYTES)
    for offset, children in NODES.items():
        struct.pack_into("<4I24f16x", data, offset, *(child for child, _, _ in children),
                         *(component for _, lo, hi in children for component in (*lo, *hi)))
    return data


def reference(nodes, extent, origin, direction, *, reverse=False):
    hits = []
    for child, lo, hi in nodes:
        if child == INVALID:
            continue
        near, far = 0.0, extent
        for lower, upper, start, step in zip(lo, hi, origin, direction, strict=True):
            first, second = (lower - start) / step, (upper - start) / step
            near, far = max(near, min(first, second)), min(far, max(first, second))
        if near <= far and near < extent:
            hits.append((near, child))
    result = tuple(child for _, child in sorted(hits, reverse=reverse))
    return result + (INVALID,) * (4 - len(result))


def first_difference(expected, actual):
    return next(((i, a, b) for i, (a, b) in enumerate(zip(expected, actual, strict=True)) if a != b), None)


def model_check():
    data = node_image()
    assert struct.unpack_from("<I", data, 256)[0] == 0x45
    for name, offset, extent, origin, direction, expected in RAYS:
        actual = reference(NODES[offset], extent, origin, direction)
        assert actual == expected, (name, actual, expected)
    name, offset, extent, origin, direction, expected = RAYS[0]
    difference = first_difference(expected, reference(NODES[offset], extent, origin, direction, reverse=True))
    assert difference == (0, 0x1d, 0x25), difference
    print(f"{name} reversed-sort control word=0 expected=0000001d actual=00000025")
    # The proposed miss fill is checked separately from the hit order.
    assert first_difference(RAYS[2][-1], (0,) * 4) == (0, INVALID, 0)
    print("all_miss zero-fill control word=0 expected=ffffffff actual=00000000")


def descriptor_for(base, wide):
    # Absolute typed pointers use base zero and the full 42-bit node-count field.
    # Relative pointers are constructed only for the two nodes in this allocation.
    address = 0 if wide else base >> 8
    bounds = (1 << 42) - 1 if wide else NODE_BYTES // 64 - 1
    return (address & MASK32, (address >> 32) | SORT_ENABLE, bounds & MASK32, (bounds >> 32) | RESOURCE_TYPE)


def kernarg(descriptor, output, node, extent, origin, direction):
    return struct.pack("<4IQ8xQ10f", *descriptor, output, node, extent, *origin, *direction,
                       *(1.0 / component for component in direction))


class Hip:
    def __init__(self, path):
        self.library = ct.CDLL(path)
        self.library.hipGetErrorString.argtypes = [ct.c_int]
        self.library.hipGetErrorString.restype = ct.c_char_p
        integer, unsigned, pointer = ct.c_int, ct.c_uint, ct.c_void_p
        signatures = {
            "hipInit": [unsigned], "hipSetDevice": [integer],
            "hipDeviceGetName": [pointer, integer, integer],
            "hipRuntimeGetVersion": [ct.POINTER(integer)], "hipDriverGetVersion": [ct.POINTER(integer)],
            "hipModuleLoad": [ct.POINTER(pointer), ct.c_char_p],
            "hipModuleGetFunction": [ct.POINTER(pointer), pointer, ct.c_char_p],
            "hipMalloc": [ct.POINTER(pointer), ct.c_size_t],
            "hipMemcpy": [pointer, pointer, ct.c_size_t, integer],
            "hipModuleLaunchKernel": [pointer] + [unsigned] * 7 + [pointer] * 3,
            "hipDeviceSynchronize": [], "hipFree": [pointer], "hipModuleUnload": [pointer],
        }
        for name, args in signatures.items():
            fn = getattr(self.library, name)
            fn.argtypes = args
            fn.restype = integer

    def call(self, name, *args):
        status = getattr(self.library, name)(*args)
        if status:
            detail = self.library.hipGetErrorString(status)
            raise RuntimeError(f"{name}: {status} {detail.decode() if detail else ''}")


def dwords(words):
    return " ".join(f"{word:08x}" for word in words)


def readback(hip, module, nodes, output, wide):
    kernel = "bvh64" if wide else "bvh32"
    function = ct.c_void_p()
    hip.call("hipModuleGetFunction", ct.byref(function), module, kernel.encode())
    descriptor = descriptor_for(nodes, wide)
    print(f"{kernel} node_base={nodes:016x} descriptor={dwords(descriptor)}", flush=True)
    failures = 0
    for name, offset, extent, origin, direction, expected in RAYS:
        node = ((nodes + offset if wide else offset) >> 3) | 5
        payload = kernarg(descriptor, output, node, extent, origin, direction)
        assert len(payload) == 80
        storage = ct.create_string_buffer(payload)
        size = ct.c_size_t(len(payload))
        # HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE and HIP_LAUNCH_PARAM_END.
        launch = (ct.c_void_p * 5)(1, ct.addressof(storage), 2, ct.addressof(size), 3)
        words = (ct.c_uint32 * 4)(0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef)
        hip.call("hipMemcpy", output, words, ct.sizeof(words), TRANSFER_TO_DEVICE)
        hip.call("hipModuleLaunchKernel", function, 1, 1, 1, 1, 1, 1, 0, None, None, launch)
        hip.call("hipDeviceSynchronize")
        hip.call("hipMemcpy", words, output, ct.sizeof(words), TRANSFER_TO_HOST)
        difference = first_difference(expected, words)
        print(f"{kernel}/{name} pointer={node:016x} raw={dwords(words)} expected={dwords(expected)}", flush=True)
        if difference:
            index, wanted, actual = difference
            print(f"first_mismatch word={index} expected={wanted:08x} actual={actual:08x}", flush=True)
            failures += 1
    return failures


def run(library, device):
    hip = Hip(library)
    hip.call("hipInit", 0)
    hip.call("hipSetDevice", device)
    name = ct.create_string_buffer(256)
    driver, runtime = ct.c_int(), ct.c_int()
    hip.call("hipDeviceGetName", name, len(name), device)
    hip.call("hipDriverGetVersion", ct.byref(driver))
    hip.call("hipRuntimeGetVersion", ct.byref(runtime))
    print(f"device={name.value.decode()} code_target=gfx1030 hip_driver={driver.value} hip_runtime={runtime.value}", flush=True)
    print(f"code_object_sha256={hashlib.sha256(OBJECT.read_bytes()).hexdigest()}", flush=True)
    module, nodes, output = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    data = (ct.c_ubyte * NODE_BYTES).from_buffer(node_image())
    hip.call("hipModuleLoad", ct.byref(module), os.fsencode(OBJECT))
    try:
        hip.call("hipMalloc", ct.byref(nodes), NODE_BYTES)
        hip.call("hipMalloc", ct.byref(output), 16)
        if nodes.value % 256 or (nodes.value + NODE_BYTES - 1) >> 45:
            raise RuntimeError(f"BVH allocation outside aligned gfx1030 address range: {nodes.value:#x}")
        hip.call("hipMemcpy", nodes, data, NODE_BYTES, TRANSFER_TO_DEVICE)
        failures = sum(readback(hip, module, nodes.value, output.value, wide) for wide in (False, True))
        print(f"native_cases=10 mismatches={failures}")
        return int(failures != 0)
    finally:
        if output:
            hip.call("hipFree", output)
        if nodes:
            hip.call("hipFree", nodes)
        hip.call("hipModuleUnload", module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", default="libamdhip64.so")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    model_check()
    if args.model_only:
        return 0
    if any(key.startswith("HSA_OVERRIDE_GFX_VERSION") for key in os.environ):
        parser.error("Unset HSA_OVERRIDE_GFX_VERSION before measuring native behavior")
    if not OBJECT.is_file():
        parser.error("Build the gfx1030 code object with sh tests/gfx1030_bvh_box/build.sh")
    return run(args.library, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
