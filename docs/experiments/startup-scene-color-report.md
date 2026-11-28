# Startup scene-colour report from per-draw texel history

Refs [#870](https://github.com/R0Y00/Xbox2Windows/issues/870). This note documents an
offline reader for the draw diagnostics that already exist. No title run was made for
this change, and no dump, game file or GPU result is checked in.

## Command

```powershell
python -B tools/startup/scene_color_report.py <X2W_DRAW_DUMP_DIR> `
    [--frame <n>] [--pixel <x>,<y>] [--baseline-raw <0xRRGGBBAA>] [--max-rows <n>]
```

The directory is the one passed to `X2W_DRAW_DUMP_DIR`. `--frame` and `--pixel`
choose the sample group; without either option the tool selects the largest group
and says so. `--baseline-raw` supplies a known pre-draw word, for the case where
the first retained sample is already the first colour write.

The tool prints Markdown to standard output. The input path, file names, raw log
lines and arbitrary text never appear in the report.

## Input

`pixel_history.txt` starts with `x2w-pixel-history 1 samples=<n>`. Each following
line is written by [`pixel_history_write`](../../src/startup_stubs/pixel_history.h):

```text
draw=<copy ordinal> frame=<frame> x=<x> y=<y> key=<16 hex> count=<n> format=<DXGI number> raw=<8 hex> [rgba=r,g,b,a]
```

The runtime records one texel of render target 0 after each translated draw of
`X2W_DRAW_DUMP_FRAME` (and the following frame, when the copy completes there). It
decodes `rgba=` for `R8G8B8A8_UNORM`, `R8G8B8A8_UNORM_SRGB`, `B8G8R8A8_UNORM` and
`B8G8R8A8_UNORM_SRGB`. The copy budget is 48, so a history that reaches 48 samples
may be missing later draws. See [Per-draw texel history](../experiments.md#per-draw-texel-history)
and the [startup runtime options](../startup-runtime.md#switches).

The `draw=` value is the runtime copy ordinal, not a title draw-call number. It is
monotonic in one process and is used here only to order the retained samples.

When full-dump BMPs from `X2W_DRAW_DUMP_FRAME` are present, the tool reads their
headers and the selected pixel. The runtime writes either the full BMP set or the
texel history in one run, so the two are never merged: only a BMP with the same
`draw`, `key` and `count` is compared. BMP files carry no frame field. The reader
accepts BI_RGB 24-bit or 32-bit bottom-up or top-down BMPs and ignores the alpha
byte, which the runtime forces to 255.

## What the report shows

For the selected frame and pixel it lists the retained samples with frame, copy
ordinal, opaque pipeline key, draw `count`, DXGI format, `raw` word and decoded
`rgba`. It names both the first retained sample whose value differs from the
baseline and the last retained writer, rather than only the final value.

| Classification | Meaning |
| --- | --- |
| `no_draw` (aucun draw) | No retained sample matches the selected frame and pixel. It does not prove the title issued no draw. |
| `ui_only` (draw UI seulement) | Samples exist, but no retained sample changes the selected pixel. This is an observation, not proof that the draws are UI. |
| `target_write` (écriture de cible) | At least one sample changed the value and the final value differs from the baseline. |
| `value_unchanged` (valeur inchangée) | At least one change was observed, but the final value equals the baseline; also used for one sample with no baseline, where no transition is observable. |
| `history_truncated` (historique tronqué) | The 48-copy cap, the input bound or a header/record-count mismatch was reached. The first and last writers are still reported from the retained samples. |
| `target_non_rgba8` (cible non-RGBA8) | The format is outside the four decodable 8-bit four-channel formats. The raw word is shown, but no colour change is evaluated. |

The report is bounded: at most 1 MiB of history, 4,096 records, 64 BMPs, eight
sample groups and 64 timeline rows are considered. Timeline rows omitted by the
default 24-row bound are counted. Invalid lines are counted and ignored, never
echoed. When `rgba=` disagrees with `raw`, the decoded value is used and the
mismatch is counted.

## What it proves and what it does not

It proves only that, among the translated draws retained in `pixel_history.txt`
for the selected frame and pixel, a particular value appeared after a particular
copy ordinal and changed on a particular later draw. A BMP with the same
`draw`/`key`/`count` confirms the RGB at that pixel.

It does not replace a run of the title. It does not show that a draw is a scene
draw rather than a UI draw, that a scene was visible, that a clear happened, or
that an untranslated draw was absent. It does not cover render targets other than
target 0, formats other than the four RGBA8/BGRA8 formats, draws after the 48-copy
cap, or the presented image.

The history has no sample before the first draw. If the first retained draw is
itself the first colour write, the tool cannot see the transition unless the
caller supplies `--baseline-raw`. The report states that limitation next to the
baseline. A frame whose first sample is the known clear colour is the simplest
way to see the first write without an extra baseline.

## Tests

```powershell
python -B tests/startup_scene_color_report_check.py
```

The synthetic check passes 14 cases: first change and last writer, a first change
that is also the last writer, a write that restores the baseline, no observed
change, an empty history, a malformed line, a non-RGBA8 target, BGRA channel
order, the 48-sample cap with a bounded table, frame and pixel selection, a
supplied baseline, BMP match and mismatch, a header/record mismatch, and a
missing history file. The check is listed in
[`tests/checks/python-linux/scene_color_report.txt`](../../tests/checks/python-linux/scene_color_report.txt)
so the Linux CI job runs it without Windows, a GPU or a dump.

No local dump was available under `out/` during this change, so the checked
results are synthetic. The tool was not run against the title.

## Limits

The parser accepts only the version-1 header and the exact field order above.
The BMP path accepts BI_RGB 24-bit or 32-bit images up to 16,384 pixels per side
and 64 MiB. A 48-sample history is reported as truncated by design, even when the
title happened to stop after exactly 48 copies.

