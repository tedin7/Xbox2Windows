#!/usr/bin/env python3
"""Withdraw the cached translations that predate the packed-position derivation, and only those.

usage: shader_cache_repair_packed.py <shader cache dir> [--dry-run]

A graphics request without a stream description (req_<key>.psin) was translated with the packed pixel position left at 0
when its pixel stage reads one (Stage.derive_packed_position now takes the register from the program's own 16-bit split of a
live-in VGPR, tools/startup/ps_input_scan.py). The shader revision is not bumped for that, because a bump discards the whole
warm cache. Instead this tool finds the results that are wrong in exactly that way and removes them, so the next
`gfx2hlsl.py --pending` / `--watch` pass translates them again:

  - the request has no description (its own, or its RTV variant's source pipeline's), and
  - the pixel stage's contract has exactly one split live-in register, and
  - the published <key>.ps.hlsl has no `vN = (uint(i.pos.x) & 0xffffu) ...` line for it.

The manifest <key>.txt is removed first (it is the ready marker, written last by the translator), then the generated
files <key>.ps.hlsl, .ps.cso, .vs.hlsl, .vs.cso and .psdep. Request files, .fail files, results of other keys, results that
already carry the line, and results with no .ps.hlsl (nothing to judge) are not touched, so a second run removes nothing.
Run it before the translator starts (tester_run, shader_cache_rekey style); a running watcher is told nothing.
With --dry-run nothing is removed. The output is one line of counts. Exit 0, 2 on a usage error.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as g  # noqa: E402
import ps_input_scan as scan  # noqa: E402

PACKED = re.compile(r'\bv(\d+) = \(uint\(i\.pos\.x\) & 0xffffu\)')
GENERATED = ('.ps.hlsl', '.ps.cso', '.vs.hlsl', '.vs.cso', '.psdep')


def derived_register(out_dir, key):
    """The packed-position register the translator would now derive for this request, or None (not a graphics request,
    a description exists, or no single split live-in register)."""
    try:
        _, base, _, description = g.request_source(out_dir, key)
        if os.path.isfile(description):
            return None
        with open(os.path.join(out_dir, 'req_%016x.b' % base), 'rb') as file:
            halves = sorted(scan.contract(g.strip_header(file.read())).split16)
    except (g.Unsupported, OSError, UnicodeError, ValueError):
        return None
    return halves[0] if len(halves) == 1 else None


def repair(out_dir, dry_run=False):
    """(graphics requests, derivable, stale) and removes the stale results unless dry_run."""
    requests = derivable = stale = 0
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r'req_([0-9a-f]{16})\.(a|rtv)', name)
        if not m:
            continue
        requests += 1
        key = int(m.group(1), 16)
        register = derived_register(out_dir, key)
        if register is None:
            continue
        derivable += 1
        result = os.path.join(out_dir, '%016x' % key)
        try:
            with open(result + '.ps.hlsl', encoding='ascii', errors='replace') as file:
                text = file.read()
        except OSError:
            continue  # no published pixel source: nothing to judge
        if register in {int(n) for n in PACKED.findall(text)}:
            continue
        stale += 1
        if not dry_run:
            for suffix in ('.txt',) + GENERATED:
                try:
                    os.remove(result + suffix)
                except FileNotFoundError:
                    pass
    return requests, derivable, stale


def main(argv):
    args = [a for a in argv[1:] if a != '--dry-run']
    if len(args) != 1 or not os.path.isdir(args[0]) or len(argv) - len(args) > 2:
        print(__doc__)
        return 2
    dry_run = '--dry-run' in argv
    requests, derivable, stale = repair(args[0], dry_run)
    print('shader cache repair (packed position): %d graphics requests, %d without a description whose pixel stage reads the '
          'packed position, %d stale results %s' % (requests, derivable, stale, 'found' if dry_run else 'withdrawn'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
