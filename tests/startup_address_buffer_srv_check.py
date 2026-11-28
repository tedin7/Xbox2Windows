"""Publish placed buffer SRVs by address, sample them in translated draws, and check refusals."""
from pathlib import Path
import re
import sys
import tempfile

from startup_address_srv_check import draw_texture

HEX = r"0x[0-9a-f]+"


def check_refusal_lines(log):
    prefix = r"x2w-d3d12 Device::slot70 "
    buffer_unavailable = (
        prefix
        + r"unavailable buffer at the supplied address: reason=buffer_unavailable "
        + r"res\{dim=1 fmt=0 size=256x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0\} "
        + r"view\{dim=1 fmt=39 map=0x1688 first=FIRST count=4 stride=0 flags=0x1 \(buffer views are not supported\)\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1"
    )
    expected = [
        buffer_unavailable.replace("FIRST", "4"),  # injected wrapper allocation failure
        buffer_unavailable.replace("FIRST", "0"),  # unaligned interior address
        prefix
        + r"unavailable acceleration structure at the supplied address: reason=as_wrong_state "
        + r"res\{dim=1 fmt=0 size=256x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0\} "
        + r"view\{dim=11 fmt=0 map=0x1688 raw=(00000000,){5}00000000 "
        + r"\(acceleration structure view\)\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        buffer_unavailable.replace("FIRST", "0"),  # unplaced raw view at page + 1
        buffer_unavailable.replace("FIRST", "0"),  # unplaced raw view before stride-16 publication
    ]
    lines = [line.rstrip() for line in log.splitlines() if "Device::slot70" in line]
    if len(lines) != len(expected):
        raise AssertionError(f"{len(lines)} slot 70 lines, expected {len(expected)}:\n" + "\n".join(lines))
    for line, pattern in zip(lines, expected):
        if not re.fullmatch(pattern, line):
            raise AssertionError(f"slot 70 line does not match\n  line     {line}\n  expected {pattern}")


def check(directory, consumer, backend):
    hlsl = (
        "ByteAddressBuffer b : register(t0);\n"
        "float4 main(float4 p : SV_Position) : SV_Target {\n"
        "  return float4(float(b.Load(0)) / 255.0, 0, 0, 1);\n"
        "}\n"
    )
    trap_log = directory / "trap.log"
    actual = draw_texture(
        directory,
        consumer,
        backend,
        hlsl,
        {"X2W_TEST_ADDRESS_BUFFER_SRV": "1", "X2W_TRAP_LOG": str(trap_log)},
        binding="srv 0 buf table 0 0",
    )
    if actual is None:
        return 77
    values = (17, 63, 141, 141, 99, 17, 201, 201, 55, 77, 55, 201, 201, 173, 173)
    row = b"".join(bytes((v, 0, 0, 255)) for v in values)
    expected = row * 4
    if actual != expected:
        raise AssertionError(f"actual {actual.hex()} expected {expected.hex()}")
    check_refusal_lines(trap_log.read_text(errors="replace"))
    print(f"{backend}: buffer memory SRV (raw base, raw interior, unplaced buffer, structured, refused unaligned, refused acceleration structure view, unplaced structured view at a page offset the stride does not divide, unplaced buffer in a large allocation copied) exact")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
