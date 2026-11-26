#!/usr/bin/env python3
"""Translate a list of cached requests again now, whatever an older translator build wrote for them.

usage: shader_cache_retry_keys.py <shader cache dir> [--report REPORT.txt] [--top N] [KEY ...]

The running translator (`gfx2hlsl.py --watch`) leaves every old <key>.fail alone, and the pass after a run retries them all in
key order under a time cap, so a refusal that a later translator build no longer gives can outlive several runs for the very
key a title draws most. This tool retries chosen keys first, one by one, before the title starts:

  - KEY arguments: 16-digit hex request keys;
  - --report: the previous run's REPORT.txt; its "<key>  <count> not drawn" lines name the keys the title asked for and did
    not get, most draws first (--top N keeps the first N, default 20).

For each key that has a request and no finished result (a <key>.fail, or nothing), the stale .fail is removed and the request is
translated with this build: the manifest is written last on success, a new .fail on a refusal. A key that is already translated,
or has no request here, is left alone, so a second run does nothing new. The output is one line of counts. Exit 0, 2 on a usage
error.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as g  # noqa: E402

NOT_DRAWN = re.compile(r'^\s+([0-9a-f]{16})\s+\d+ not drawn\b')
KEY = re.compile(r'[0-9a-f]{16}')


def report_keys(path, top):
    """The keys of the report's "not drawn" lines, in the order the report lists them (most draws first)."""
    keys = []
    with open(path, encoding='utf-8', errors='replace') as file:
        for line in file:
            m = NOT_DRAWN.match(line)
            if m and m.group(1) not in keys:
                keys.append(m.group(1))
    return keys[:top]


def retry(out_dir, keys):
    """(asked, retried, translated) for the hex `keys`; a key is retried when it has a request and no finished result."""
    pending = {'%016x' % k for k in g.pending_keys(out_dir, False)}
    asked = retried = translated = 0
    for text in keys:
        asked += 1
        if text not in pending:
            continue  # no request here, or already translated
        retried += 1
        key = int(text, 16)
        fail = os.path.join(out_dir, text + '.fail')
        if os.path.exists(fail):
            os.remove(fail)
        g.translate_one(out_dir, key)
        if not os.path.exists(fail):
            translated += 1
    return asked, retried, translated


def main(argv):
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    cache, keys, top, report = args.pop(0), [], 20, None
    while args:
        arg = args.pop(0)
        if arg == '--report' and args:
            report = args.pop(0)
        elif arg == '--top' and args and args[0].isdigit():
            top = int(args.pop(0))
        elif KEY.fullmatch(arg):
            keys.append(arg)
        else:
            print(__doc__)
            return 2
    return run(cache, keys, report, top) if keys or report else 2


def run(cache, keys, report, top):
    if not os.path.isdir(cache) or (report and not os.path.isfile(report)):
        print(__doc__)
        return 2
    wanted = list(keys) + [k for k in (report_keys(report, top) if report else []) if k not in keys]
    asked, retried, translated = retry(cache, wanted)
    print('shader cache retry: %d keys asked, %d retried (a request without a finished result), %d translated now'
          % (asked, retried, translated))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
