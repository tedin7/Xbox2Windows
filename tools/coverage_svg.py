"""Draw the result of tools/form_coverage.py as a static SVG (one square per reference form).

Usage:
    python tools/form_coverage.py out/test-results.xml --json out/coverage.json
    python tools/coverage_svg.py out/coverage.json docs/coverage/dashboard.svg --source "<commit, OS, adapter>"

Green squares are covered forms, amber squares are forms held back only by skipped tests, and grey
squares are forms no passing fixture translates. Counts only; the graphic never shows a percentage,
since instruction-form coverage is not a measure of port completion. Python standard library only.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from xml.sax.saxutils import escape

AREAS = (("x64_zen2", "x64 (Zen 2 user-mode mnemonics)", 220), ("gfx1013", "gfx1013 forms", 300),
         ("gfx1030", "gfx1030 (RDNA2) forms", 300))
WIDTH, BG, PANEL, BORDER, TEXT, MUTED = 1000, "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
COVERED, HELD, OPEN = "#2ea043", "#d29922", "#484f58"
LABEL_CHAR = 7.0  # px per character at 11 px, generous so a label never runs past its box
NOTE = "Instruction-form coverage of source-built fixtures; not a measure of port completion."


def text(x, y, s, size=12, fill=TEXT, bold=False):
    weight = ' font-weight="600"' if bold else ""
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{fill}"{weight}>{escape(s)}</text>'


def squarify(items, x, y, w, h):
    """Squarified treemap (Bruls et al.): [(key, size)] largest first -> {key: (x, y, w, h)}."""
    items = list(items)
    if not items or w <= 0 or h <= 0:
        return {}
    if sum(size for _, size in items) <= 0:
        return {}
    positive = [(key, size) for key, size in items if size > 0]
    if not positive:
        return {}
    scale = w * h / sum(size for _, size in positive)
    areas, out = [(key, size * scale) for key, size in positive], {}

    def worst(row, side):
        s = sum(a for _, a in row)
        return max(max(side * side * a / (s * s), s * s / (side * side * a)) for _, a in row)

    while areas:
        if w <= 0 or h <= 0:
            break
        side, n = min(w, h), 1
        while n < len(areas) and worst(areas[:n + 1], side) <= worst(areas[:n], side):
            n += 1
        row, areas = areas[:n], areas[n:]
        s = sum(a for _, a in row)
        if w >= h:
            cw, yy = s / h, y
            if cw <= 0:
                break
            for key, a in row:
                out[key] = (x, yy, cw, a / cw)
                yy += a / cw
            x, w = x + cw, w - cw
        else:
            rh, xx = s / w, x
            if rh <= 0:
                break
            for key, a in row:
                out[key] = (xx, y, a / rh, rh)
                xx += a / rh
            y, h = y + rh, h - rh
    return out


def fit(total, w, h):
    """Largest square size (2 px or more, 1 px gaps) and column count that fit `total` squares in w x h."""
    if total <= 0 or w <= 0 or h <= 0:
        return None
    for size in range(int(min(w, h)), 1, -1):
        cols = int(w + 1) // (size + 1)
        if cols and -(-total // cols) * (size + 1) - 1 <= h:
            return size, cols
    return None


def group(name, counts, x, y, w, h):
    """A box of squares (covered, then held back, then open), clipped to the box. The label goes inside
    only when it and the squares both fit; returns (svg, labelled)."""
    covered, held, total = counts
    label = f"{name} {covered}/{total}"
    layout = fit(total, w - 4, h - 20) if len(label) * LABEL_CHAR <= w - 8 else None
    labelled = layout is not None
    layout = layout or fit(total, w - 4, h - 4)
    size, step, cols = (layout[0], layout[0] + 1, layout[1]) if layout else (1, 1, max(int(w - 4), 1))
    top = 18 if labelled else 2
    parts = [f'<svg x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"><rect width="{w:.1f}" '
             f'height="{h:.1f}" fill="{PANEL}" stroke="{BG}" stroke-width="2"/>']
    if labelled:
        parts.append(text(4, 13, label, 11))
    for color, first, last in ((COVERED, 0, covered), (HELD, covered, covered + held), (OPEN, covered + held, total)):
        cells = "".join(f'<rect x="{2 + i % cols * step}" y="{top + i // cols * step}" '
                        f'width="{size}" height="{size}"/>' for i in range(first, last))
        parts.append(f'<g fill="{color}">{cells}</g>')
    return "".join(parts) + "</svg>", labelled


def render(m, source):
    if not isinstance(m, dict):
        raise TypeError("coverage root must be a JSON object")
    rows = [text(20, 34, "Instruction-form coverage", 20, bold=True), text(20, 56, NOTE, 12, MUTED),
            text(20, 74, f"Measured on {source}. Green: covered. Amber: held back by skipped tests. "
                         "Grey: not covered.", 12, MUTED)]
    y = 92
    for key, label, height in AREAS:
        r = m[key]
        if not isinstance(r, dict):
            raise TypeError(f"{key} entry must be a JSON object")
        blocked = r["blocked_by_skip"]
        groups = r["groups"]
        if not isinstance(blocked, list) or not isinstance(groups, dict):
            raise TypeError(f"{key} entry has invalid blocked_by_skip or groups")
        held = {}
        for form in blocked:
            if not isinstance(form, str):
                raise TypeError(f"{key} blocked_by_skip entries must be strings")
            parts = form.split()
            if not parts:
                raise ValueError(f"{key} blocked_by_skip entry must not be empty")
            held[parts[0]] = held.get(parts[0], 0) + 1
        title = (f"{label}: {r['covered']} of {r['total']} covered, "
                 f"{len(blocked)} held back by skips")
        rows.append(f'<rect x="20" y="{y}" width="{WIDTH - 40}" height="{height + 58}" rx="6" '
                    f'fill="{PANEL}" stroke="{BORDER}"/>' + text(32, y + 21, title, 13, bold=True))
        items = []
        for g, v in groups.items():
            if not isinstance(v, dict):
                raise TypeError(f"{key}.{g} must be a JSON object")
            total = v["total"]
            covered = v["covered"]
            if total < 0 or covered < 0:
                raise ValueError(f"{key}.{g} counts must be non-negative")
            if total > 0:
                items.append((g, total))
        items.sort(key=lambda kv: -kv[1])
        unlabelled = []
        for name, (gx, gy, gw, gh) in squarify(items, 30, y + 30, WIDTH - 60, height).items():
            g = groups[name]
            box, labelled = group(name, (g["covered"], held.get(name, 0), g["total"]), gx, gy, gw, gh)
            rows.append(box)
            if not labelled:
                unlabelled.append(f"{name} {g['covered']}/{g['total']}")
        if unlabelled:
            rows.append(text(32, y + height + 47, "Too small to label: " + ", ".join(unlabelled), 11, MUTED))
        y += height + 74
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{y - 6}" '
            f'viewBox="0 0 {WIDTH} {y - 6}" font-family="Segoe UI,Helvetica,Arial,sans-serif">'
            f'<rect width="{WIDTH}" height="{y - 6}" fill="{BG}"/>' + "".join(rows) + "</svg>\n")


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coverage", type=Path, help="JSON written by form_coverage.py --json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True, help="commit, OS build and adapter of the measured run")
    args = parser.parse_args(argv)
    try:
        coverage_path = args.coverage.resolve()
        output_path = args.output.resolve()
        if output_path == coverage_path:
            raise ValueError(f"output path must not match input path: {args.output}")
        svg = render(json.loads(args.coverage.read_text(encoding="utf-8")), args.source)
        _write_atomic(args.output, svg)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"coverage_svg: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
