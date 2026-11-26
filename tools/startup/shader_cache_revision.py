#!/usr/bin/env python3
"""Warn when a shader cache and the translator about to read it come from different revisions (#2815).

usage: shader_cache_revision.py <cache dir> [--translator <gfx2hlsl.py>]
       shader_cache_revision.py <cache dir> --fresh

Request keys start with a revision string on both sides: the runtime's (src/startup_stubs/d3d12_draw.inl) and
the translator's (pipeline_key, compute_key and mesh57_key in gfx2hlsl.py). Before its first request a process
writes its revisions to <cache dir>/revision.txt, one `<kind> <revision>` line each (graphics, compute, mesh57).
This tool compares that file with the revision literals of the translator (default: the gfx2hlsl.py next to
it). When they differ, every cached request is refused as a key mismatch and nothing says why, so it prints
one warning line on stdout with both revisions and the two fixes, and exits with 1. A cache that holds
requests (req_*) but no revision.txt comes from a runtime older than this check and gets the same warning.
A cache with no requests, or whose revisions match, prints nothing and exits with 0.

--fresh moves a cache folder that is not empty aside to <cache dir>.old-<date>-<time> and prints where it went
(exit 0), so the run starts with an empty cache. Usage errors exit with 2. The launchers (run.ps1, run.sh)
print the line as their own warning: only stdout is written, so Windows PowerShell does not take the output
for an error.
"""
import datetime
import os
import re
import sys

REVISION_FILE = 'revision.txt'
KINDS = ('graphics', 'compute', 'mesh57')
# The literals the translator hashes first: b'x2w-startup-shader-<n>' (graphics), the same with 'cs' (compute)
# and b'x2w-startup-mesh-57-<n>' (slot 57 mesh).
LITERALS = {
    'graphics': re.compile(rb"b'(x2w-startup-shader-[0-9]+)'"),
    'compute': re.compile(rb"b'(x2w-startup-shader-[0-9]+cs)'"),
    'mesh57': re.compile(rb"b'(x2w-startup-mesh-57-[0-9]+)'"),
}
FIXES = ('Fix: rebuild the stand-ins from this checkout, or start with an empty shader cache '
         '(run.ps1 -FreshShaderCache, run.sh --fresh-shader-cache, or another cache folder).')


def translator_revisions(path):
    """{kind: revision} from the translator's source; a kind whose literal is missing or not unique is left out."""
    with open(path, 'rb') as f:
        text = f.read()
    found = {}
    for kind, pattern in LITERALS.items():
        values = set(pattern.findall(text))
        if len(values) == 1:
            found[kind] = values.pop().decode('ascii')
    return found


def cache_revisions(cache):
    """{kind: revision} from <cache>/revision.txt, or None when the file does not exist."""
    try:
        with open(os.path.join(cache, REVISION_FILE), 'rb') as f:
            text = f.read().decode('ascii', 'replace')
    except FileNotFoundError:
        return None
    found = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in KINDS:
            found[parts[0]] = parts[1]
    return found


def has_requests(cache):
    try:
        return any(name.startswith('req_') for name in os.listdir(cache))
    except FileNotFoundError:
        return False


def spell(revisions):
    return ', '.join('%s %s' % (kind, revisions[kind]) for kind in KINDS if kind in revisions) or 'none readable'


def warning(cache, translator):
    """The warning line, or None when the cache and the translator agree (or the cache holds nothing to refuse)."""
    cached = cache_revisions(cache)
    if cached is None and not has_requests(cache):
        return None
    ours = translator_revisions(translator)
    if cached is None:
        return ('shader cache %s holds requests but no %s, so their revision is unknown: requests of a stand-in '
                'build older than this check (or written by archive_requests.py or shader_cache_rekey.py) may be '
                'keyed for another revision than the translator %s (%s). %s'
                % (cache, REVISION_FILE, translator, spell(ours), FIXES))
    shared = [kind for kind in KINDS if kind in cached and kind in ours]
    if shared and all(cached[kind] == ours[kind] for kind in shared):
        return None
    return ('shader cache %s was written by stand-ins of revision %s, but the translator %s is revision %s: '
            'every cached request will be refused as a key mismatch. %s'
            % (cache, spell(cached), translator, spell(ours), FIXES))


def move_aside(cache, now=None):
    """Rename a cache folder that is not empty to <cache>.old-<date>-<time>; return the new path or None."""
    try:
        if not os.listdir(cache):
            return None
    except FileNotFoundError:
        return None
    stamp = (now or datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
    target = '%s.old-%s' % (os.path.normpath(cache), stamp)
    os.rename(cache, target)
    return target


def main(argv):
    args = list(argv)
    fresh = '--fresh' in args
    if fresh:
        args.remove('--fresh')
    translator = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gfx2hlsl.py')
    if '--translator' in args:
        at = args.index('--translator')
        if at + 1 >= len(args):
            print(__doc__.split('\n\n')[1])
            return 2
        translator = args.pop(at + 1)
        args.pop(at)
    if len(args) != 1 or args[0].startswith('-') or (fresh and '--translator' in argv):
        print(__doc__.split('\n\n')[1])
        return 2
    cache = args[0]
    if fresh:
        moved = move_aside(cache)
        if moved:
            print('moved the old shader cache %s to %s' % (cache, moved))
        return 0
    try:
        line = warning(cache, translator)
    except OSError as error:
        print('cannot compare the shader cache %s with the translator %s: %s' % (cache, translator, error))
        return 1
    if line:
        print(line)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
