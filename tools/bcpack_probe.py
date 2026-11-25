"""Test the published fixed-symbol rANS model against an eight-lane BCPack prefix.

The default header is interleaved at an eight-byte stride. `lane_stride=6` reads
eight contiguous six-byte records for diagnostics.
"""
import argparse
import json
from pathlib import Path
import sys


LOWER = 1 << 24
SCALE = 1 << 16
EXCEPTIONS = {0xdc: 4, 0xdd: 8, 0xde: 32}
LANE_STRIDES = (6, 8)


def lane_record(prefix, lane, lane_stride):
    """Return the six bytes for one lane in the selected header layout."""
    if lane_stride == 8:
        return prefix[lane::8]
    return prefix[lane * lane_stride:(lane + 1) * lane_stride]


def scan_lanes(prefix, *, lane_stride=8):
    """Report each lane's mode and budget without attempting a decode."""
    if len(prefix) != 48:
        raise ValueError("Expected a 48-byte prefix")
    if lane_stride not in LANE_STRIDES:
        raise ValueError("Expected lane stride 6 or 8")
    lanes = []
    for lane in range(8):
        mode, budget, *state_bytes = lane_record(prefix, lane, lane_stride)
        lanes.append({
            "lane": lane,
            "mode": mode,
            "budget": budget,
            "state": bytes(state_bytes).hex(),
            "known_constant_mode": mode in EXCEPTIONS,
        })
    return lanes


def _decode_constant_lane(state: int, freq: int, blocks: int, supplied: bytes) -> list[int] | None:
    """Return the zero-based symbols that consumed bytes, if the whole lane verifies."""
    refill_symbols: list[int] = []
    for symbol in range(blocks):
        quotient, slot = divmod(state, SCALE)
        if not 1 <= slot <= freq:
            return None
        state = freq * quotient + slot - 1
        if state < LOWER:
            if len(refill_symbols) == len(supplied):
                return None
            state = (state << 8) | supplied[len(refill_symbols)]
            refill_symbols.append(symbol)
    if state != LOWER or len(refill_symbols) != len(supplied):
        return None
    return refill_symbols


def probe(prefix, blocks, *, prefetch_mask=0, prefetch=b"", next_mask=0, next_bytes=b"",
          lane_stride=8):
    if len(prefix) != 48 or not 1 <= blocks <= 1 << 20:
        raise ValueError("Expected a 48-byte prefix and a positive bounded block count")
    if lane_stride not in LANE_STRIDES:
        raise ValueError("Expected lane stride 6 or 8")
    if not 0 <= prefetch_mask <= 255 or len(prefetch) != prefetch_mask.bit_count():
        raise ValueError("Expected one first-round byte per participating lane")
    if not 0 <= next_mask <= 255 or len(next_bytes) != next_mask.bit_count():
        raise ValueError("Expected one second-round byte per participating lane")
    rounds = ((prefetch_mask, prefetch, 48), (next_mask, next_bytes, 48 + len(prefetch)))
    matches = []
    for lane in range(8):
        mode, budget, *state_bytes = lane_record(prefix, lane, lane_stride)
        if mode not in EXCEPTIONS:
            continue
        freq = SCALE - EXCEPTIONS[mode]
        state = LOWER
        emitted = []
        for _ in range(blocks):
            if state >= (LOWER // SCALE) * 256 * freq:
                emitted.append(state & 255)
                state >>= 8
            quotient, remainder = divmod(state, freq)
            state = quotient * SCALE + remainder + 1
        # Encoding runs backwards. The last emitted byte is consumed first.
        refill_bytes = emitted[::-1]
        refills = len(refill_bytes)
        if budget != refills + 3 or state != int.from_bytes(state_bytes, "big"):
            continue
        match = {"lane": lane, "mode": mode, "refill_bytes": refills, "status": "header_match"}
        # Each round in which the lane reads takes its next refill byte, whichever round that is.
        supplied = bytearray()
        for mask, data, start in rounds:
            if not mask & (1 << lane):
                continue
            index = (mask & ((1 << lane) - 1)).bit_count()
            read = len(supplied)
            supplied.append(data[index])
            if read < refills:
                name = ("first", "second")[read]
                match[f"{name}_refill_offset"] = start + index
                match[f"{name}_refill_matches"] = data[index] == refill_bytes[read]
            else:
                # The header budget leaves this lane no byte for the read.
                match.setdefault("unbudgeted_read_offsets", []).append(start + index)
        if refills == 0:
            if _decode_constant_lane(state, freq, blocks, b"") is None:
                raise ValueError("Predicted constant lane did not reach its terminal state")
            match["status"] = "verified_constant_lane"
        elif len(supplied) == refills:
            symbols = _decode_constant_lane(state, freq, blocks, bytes(supplied))
            if symbols is not None:
                match["status"] = "verified_constant_lane_with_supplied_refills"
                match["refill_symbol_indices"] = symbols
        matches.append(match)
    return matches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--blocks", type=int,
                        help="Independent block count for the stream; required unless --scan-lanes is used")
    parser.add_argument("--prefetch-mask", type=lambda s: int(s, 0), default=0, choices=range(256),
                        metavar="MASK", help="Bit i means lane i reads a byte in the first round after the header")
    parser.add_argument("--next-mask", type=lambda s: int(s, 0), default=0, choices=range(256),
                        metavar="MASK", help="Bit i means lane i reads a byte in the immediately following round")
    parser.add_argument("--lane-stride", type=int, choices=LANE_STRIDES, default=8,
                        help="Header layout: 8 for the interleaved default, 6 for contiguous six-byte records")
    parser.add_argument("--scan-lanes", action="store_true",
                        help="Print each lane's mode and budget without attempting a decode")
    args = parser.parse_args(argv)
    if not args.scan_lanes and args.blocks is None:
        parser.error("--blocks is required unless --scan-lanes is used")
    try:
        if not args.scan_lanes and not 1 <= args.blocks <= 1 << 20:
            raise ValueError(f"Expected --blocks in 1..{1 << 20}, got {args.blocks}")
        with args.packet.open("rb") as stream:
            prefix = stream.read(48)
            if len(prefix) != 48:
                raise ValueError(f"Expected a 48-byte prefix, got {len(prefix)} bytes")
            expected_prefetch = args.prefetch_mask.bit_count()
            prefetch = stream.read(expected_prefetch)
            if len(prefetch) != expected_prefetch:
                raise ValueError(f"Expected {expected_prefetch} prefetch byte(s), got {len(prefetch)}")
            expected_next = args.next_mask.bit_count()
            next_bytes = stream.read(expected_next)
            if len(next_bytes) != expected_next:
                raise ValueError(f"Expected {expected_next} next-round byte(s), got {len(next_bytes)}")
        if args.scan_lanes:
            result = {"lane_stride": args.lane_stride,
                      "lanes": scan_lanes(prefix, lane_stride=args.lane_stride)}
        else:
            result = probe(prefix, args.blocks, prefetch_mask=args.prefetch_mask, prefetch=prefetch,
                           next_mask=args.next_mask, next_bytes=next_bytes,
                           lane_stride=args.lane_stride)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{args.packet}: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
