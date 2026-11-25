"""Generic entry for big CTest scripts: pyc_main.py <module> [args...] runs tests/<module>.py as __main__.

Python never caches the bytecode of the script it is started with, so launching a 9k-25k line script
by path recompiles it on every start. runpy loads it as a module instead, which reuses a valid .pyc
(the Win2025 runner precompiles tests/ into PYTHONPYCACHEPREFIX; -B still reads it).
Arguments, exit status and exit-time side effects stay those of <module>.__main__.
"""
import runpy
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: pyc_main.py <module> [args...]")
    module = sys.argv[1]
    del sys.argv[0]  # the module sees itself as argv[0]'s successor: [module, args...]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
