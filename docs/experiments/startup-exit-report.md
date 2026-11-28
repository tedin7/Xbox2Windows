# Startup exit report

Refs [#870](https://github.com/R0Y00/Xbox2Windows/issues/870). This reader answers a narrower question than
`triage_run.py`: how did one startup run end, and what was the last presented frame?

## Command

```powershell
python -B tools/startup/exit_report.py <logdir> [--json <out.json>]
```

The directory is the one written by `tools/startup/debug_launch.py` or `tools/startup/run.ps1`. The tool reads
`events.txt`, `finished.txt`, `run.out`, `stderr.txt`, `stdout.txt` and `trap.log` when present.

## Output

The Markdown table reports:

| Item | Source |
|---|---|
| Exit code | `EXIT_PROCESS` in `events.txt`, or `process exited with ...` in `finished.txt` |
| Classification | one of `clean exit`, `nonzero exit`, `exception stop`, `timeout`, `stopped by request`, `missing exit event` |
| End time | the timestamp of the exit, stop, timeout or stop-file record |
| Last present | the highest `x2w-d3d12 heartbeat: N frames presented in T s` sample |
| Last presented surface | the last `x2w-render presented surface` sample |
| First / last exception | only the 32-bit code and first- or second-chance |
| Observed exceptions | count after the loader, single-step, thread-name and debug-output first-chance events are excluded |

The JSON form contains the same allowlisted fields. It never contains an input path, a module name, a raw
address, an exception parameter or free-form log text.

Each log is read once, capped at 16 MiB, and only the capped text is parsed. `finished.txt` is parsed after
the other logs because it is authoritative for the final exit code and end time; a duplicate exit event in
`events.txt` cannot overwrite it. The highest present sample wins regardless of file order.

## What it proves

It distinguishes a process that returned 0 from one that returned 1, an exception stop, a timeout and a
stop-file request. It also names the last present sample, so an exit at 307.6 s can be compared with the last
presented frame at 301.8 s without opening the raw logs.

It does not prove that the exit code came from the title rather than a runtime stub, and it does not identify
the code path that called `ExitProcess`. Those need a debugger breakpoint or a runtime-side hook. The tool is
the bounded receipt; it is not the fix.

## Tests

```powershell
python -B tests/startup_exit_report_check.py
```

The synthetic check covers a clean exit, a nonzero exit, an exception stop, a timeout, a stop-file request, a
missing exit event, filtered first-chance events, hostile paths and addresses, the highest present sample,
the 16 MiB cap, the `finished.txt` authority, a second-chance filtered code, and the JSON form. It is listed
in `tests/checks/python-linux/startup_exit_report.txt` for the Linux CI job.
