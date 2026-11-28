# Startup input trace

`X2W_INPUT_TRACE=1` makes the `GameInput.dll` stand-in log the gamepad state it resolves for each
`IGameInput::GetCurrentReading` call. The state is read after the keyboard, XInput pad 0, the
`Local\x2w_input_pad` shared block, `X2W_INPUT_PRESS` and `X2W_INPUT_SCRIPT` have been merged, so the
trace shows the values the runtime returns to the title. It is a diagnostic for checking that a
scripted stick, trigger or button reaches the runtime when the corresponding picture is presented.
The switch is off by default and adds no line without it.

## Checking it

Build the source-built fixture and run its CTest:

```powershell
cmake --build --preset dev --target startup_gameinput_trace
ctest --preset dev --no-tests=error -R '^startup_gameinput_trace$' -V
```

For a normal startup run, set the switch next to the script:

```powershell
$env:X2W_INPUT_SCRIPT = 'wait,mark=screen@+0,lx=1@+0.5+2,4@+2.5'
$env:X2W_INPUT_TRACE = '1'
.\tools\startup\run.ps1 ...
```

The trace goes to stderr and to the file named by `X2W_TRAP_LOG`, like the stand-in's other lines.

## Line format

Every line starts with `x2w-input-trace`. A state line is written when the resolved gamepad state
changes and, while it is unchanged, at most once every 250 ms:

```text
x2w-input-trace state buttons=0x00000000 lt=0.000 rt=0.000 lx=1.000 ly=0.500 rx=0.000 ry=0.000 script=2.500 present=11
```

`buttons` is the GameInput button mask in hexadecimal. `lt`, `rt`, `lx`, `ly`, `rx` and `ry` use
the units of `GamepadState` (triggers in [0, 1], sticks in [-1, 1]). `script` is the current time on
the script's time base in seconds, or `off` when no timed entry or wait/mark is active. `present`
is the latest present number published by the picture signature, or `none` when no presenter
sample is available. The state line does not depend on the title having called
`GetGamepadState`; it is sampled when the stand-in captures a reading.

Script marks and waits add one event line each:

```text
x2w-input-trace wait event=begun item=2 at=0.000 elapsed=0.000 signal=1
x2w-input-trace wait event=changed item=2 at=1.000 elapsed=1.000 signal=1
x2w-input-trace wait event=done item=2 at=1.500 elapsed=1.500 signal=1
x2w-input-trace wait event=timeout item=4 at=2.000 elapsed=2.000 signal=0 stopped=0 group=-
x2w-input-trace mark label=turn item=5 at=2.500 present=11
x2w-input-trace group label=list items=2-6 failed_item=4 at=3.000
```

`item` is the 1-based script item, and `at` is on the same shifted script time base as the state
lines. A wait that times out names its group, or `-` when it has none. A mark without a presenter
prints `present=none`.

## Volume

The trace logs one state line per change and one periodic line per 250 ms of unchanged state, so a
steady run produces at most four state lines per second. Wait, mark and group lines occur once per
script event, and the parser limits a script to 256 timed items, so the event volume follows the
script. Without `X2W_INPUT_TRACE=1`, the stand-in reads no clock for the trace and writes no
`x2w-input-trace` line.

## What it verifies for movement

The `script` and `present` fields on one line tie an input value to the picture being presented. For
a movement question, this distinguishes an input that never reached the runtime from one that was
present but produced no visible movement. It also exposes axis signs and trigger ranges before they
are consumed: `lx=1.000` is right, `ly=1.000` is up, and a trigger outside [0, 1] would be a runtime
mapping problem rather than a rendering problem. Marks name the frame where a movement starts, and
waits show how long the script blocked before the movement was allowed to run.

The trace covers the GameInput stand-in only. It does not prove that the title read the reading,
that the reading was used by a particular title system, or that a draw or present changed. Pair it
with the stand-in's `x2w-input gamepad buttons ... read by the title` line, a present/capture log,
or a title-side movement observation for that evidence.

## Fixture

`startup_gameinput_trace` runs one synthetic script in a child process with the switch on and again
with it off. The child sets `X2W_INPUT_CLOCK_MS` before every poll and publishes a new picture at
1.0 s. The script is:

```text
mark=start@0,wait=0.5/2,4@+0.5+0.5,lx=1@+0.5+1,mark=turn@+0,ly=0.5@+0+0.5,8@+1.1+0.1
```

The wait completes at 1.5 s, so A is down at 2.0-2.5 s and the left stick is right with Y at 0.5 at
2.5-3.0 s. A 0.1 s B press runs at 3.6-3.7 s, off the 250 ms heartbeat grid. The trace child must
contain the exact state lines at 2.000 s, 2.500 s and 3.600 s, the begun, changed and done wait
lines, both marks, no timeout, and between 12 and 21 state lines over the four second run. The
control must observe the same gamepad states, keep the ordinary `x2w-input script`
mark line, and contain no `x2w-input-trace` line. The expected states and trace lines are built from
source; the fixture uses no game files, dump or captured input.

On the local Windows x64 Debug build (MSVC 19.44, CMake/Ninja), `startup_gameinput_trace` passed
with both modes. The same build ran `startup_gameinput_script_long`,
`startup_gameinput_script_entries` and `startup_gameinput_script_syntax` successfully, and the
stand-in's keys, history and contract checks still passed without the switch.
