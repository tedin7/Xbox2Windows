# Startup Opus completion counters

Refs [#1145](https://github.com/R0Y00/Xbox2Windows/issues/1145) and
[#870](https://github.com/R0Y00/Xbox2Windows/issues/870). This note covers the optional Opus
decode path built with `-DX2W_WITH_OPUS=ON` and `X2W_XAPU_TYPED_COMPLETIONS=1`.

## What is counted

`xapu.dll` keeps process-wide saturating counters for:

| Counter | Meaning |
|---|---|
| `type3` | readable type 3 commands seen while typed completions are on |
| `packets` | commands libopus decoded successfully |
| `completions` | type 3 results dequeued by the caller |
| `refusals` | sum of the refusal counters below |
| `invalid_argument` | a field or output range failed before libopus |
| `opus_error` | libopus rejected the packet |
| `capacity` | libopus reported `OPUS_BUFFER_TOO_SMALL` |
| `queue_full` | the per-connection queue was full |
| `short_command` | the decode fields crossed the readable header |
| `unknown_handle` | the command named a handle that was not connected |
| `out_of_memory` | a new decoder could not be created |
| `header` | the command header was null or unreadable |

The counters start on the first readable type 3 command. A run with typed completions off reads
and writes nothing here.

The audio sink also reports `accepted` and `not_open` submissions, so a run can distinguish an
endpoint that never opened from one that accepted decoded blocks.

## Summary line

At the last `XApuDisconnect`, and from `atexit` when a run never disconnects, the stand-in writes
one stable line when the counters moved since the previous summary:

```text
x2w-audio XApu opus summary: type3=10 packets=2 completions=2 refusals=9 invalid_argument=3 opus_error=1 capacity=1 queue_full=1 short_command=1 unknown_handle=1 out_of_memory=0 header=1
```

The line is written to stderr and to `X2W_TRAP_LOG`, like the other audio lines. It is a snapshot;
two threads can move different fields between their loads. Each field is exact on its own.

## Test

```powershell
cmake --preset dev -DX2W_WITH_OPUS=ON -DOpus_DIR=<libopus install>/lib/cmake/Opus
cmake --build --preset dev --target startup_stub_xapu startup_xapu_opus_count_check
ctest --preset dev --no-tests=error -R '^startup_xapu_opus_count$' -V
```

The fixture encodes synthetic 1 kHz tones with libopus and drives the real stand-in exports. It
checks two successful decodes and dequeues, a malformed packet, insufficient capacity, an invalid
rate, an out-of-range slot in both the legacy and typed-stream paths, a short command, an unknown
handle, a full queue, a null header, saturation at `UINT64_MAX`, one summary after the last
disconnect, and that typed completions off neither decode nor add a summary line. It passed in
Debug with libopus 1.5.2 on Windows 11 build 26300.

## Limits

The counters prove that the optional decode path ran and that the caller consumed the results.
They do not prove that the output was audible, that the desktop sink accepted the same block, or
that XMA/AcpHal works. The `startup_audio_output` fixture covers the sink's accepted/not-open
counters separately. The title run that produced the 27/11 audio observation was not available
locally, so this change adds the measurement, not the run receipt.
