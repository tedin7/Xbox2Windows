"""Compare present colors against ordered producer readbacks on a source-built fixture."""
import os
import re
import subprocess
import sys

NEAR_RED = "254,0,0,255"
RED, GREEN, BLUE, MAGENTA = "255,0,0,255", "0,255,0,255", "0,0,255,255", "255,0,255,255"
BACKEND, EXECUTABLE, *OPTIONS = sys.argv[1:]


def run(mode, rgb="", *, onset="", default_cadence=False, floor="4,2", rearm=""):
    # onset: "" (off), True or "1" (flat color onset), "black" (flat black onset).
    onset = "1" if onset is True else (onset or "")
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_PRESENT_SIGNATURE_EVERY="1", X2W_RED_PRODUCER_FRAMES="", X2W_RED_PRODUCER_AUTO="",
               X2W_RED_PRODUCER_RGB=rgb, X2W_RED_PRODUCER_REARM=rearm, X2W_RED_PRODUCER_ONSET=onset)
    if default_cadence:
        env.pop("X2W_PRESENT_SIGNATURE_EVERY")
    if mode == "auto-black":
        env["X2W_RED_PRODUCER_AUTO"] = "1,2"
    elif mode.startswith("auto"):
        env["X2W_RED_PRODUCER_AUTO"] = floor
    else:
        env["X2W_RED_PRODUCER_FRAMES"] = "1,12"
    args = [EXECUTABLE, mode or "manual", BACKEND]
    done = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    if done.returncode == 77:
        print(f"{BACKEND}: capability skip")
        raise SystemExit(77)
    if done.returncode:
        raise RuntimeError(done.stdout + done.stderr)
    records = re.findall(r"producer order=(\d+) ordinal=(\d+) record=(\d+) submit=\d+ queue=\w+ target=(\w+) "
                         r"op=clear-rtv key=\w+ rt=0 rgba=([\d,]+) before=([\d,]+)", done.stderr)
    return done.stderr, records, done.stdout.strip()


def check(reverse):
    logs, records, adapter = run("reverse" if reverse else "")
    first, last = (RED, GREEN) if reverse else (GREEN, RED)
    target = records[0][3]
    assert records == [("1", "1", "2", target, first, BLUE), ("2", "2", "2", target, last, first),
                       ("3", "1", "2", target, first, last), ("4", "2", "2", target, last, first)], logs
    for frame in (2, 4):
        assert re.search(fr"sample present={frame} published_at=\d+ target={target} rgba={last} red={int(not reverse)} ",
                         logs), logs
    print(f"{'reversed control' if reverse else 'positive control'}: two executions and same-surface presents match ({adapter})")


def check_auto(magenta=False):
    color, rgb = (MAGENTA, "255,0,255") if magenta else (RED, "255,0,0")
    logs, records, adapter = run("auto-magenta" if magenta else "auto", rgb if magenta else "")
    assert len(records) == 2, logs
    target = records[0][3]
    assert (f"auto armed sampled_present=5 published_at=6 target={target} capture=7..8 rgb={rgb}" in logs and
            "x2w-red history record=5" in logs and "x2w-red history record=1" not in logs), logs
    assert records == [("1", "1", "7", target, GREEN, color), ("2", "2", "7", target, color, GREEN)], logs
    assert f"sample present=1 published_at=2 target={target} rgba={color} red={int(not magenta)} match=1 " in logs, logs
    print(f"auto {rgb} control: excludes early and near matches and compares paired samples within the armed window ({adapter})")


def check_default_cadence():
    logs, records, adapter = run("auto-default", "255,0,0", default_cadence=True)
    assert len(records) == 2, logs
    target = records[0][3]
    assert f"auto armed sampled_present=5 published_at=6 target={target} capture=7..8" in logs, logs
    assert records == [("1", "1", "7", target, GREEN, RED), ("2", "2", "7", target, RED, GREEN)], logs
    print(f"automatic producer readback with no explicit signature cadence ({adapter})")


def check_onset():
    logs, records, adapter = run("auto-onset", onset="1")
    assert len(records) == 2, logs
    target = records[0][3]
    assert f"x2w-red flat onset previous_present=4 sampled_present=5 previous_rgba={NEAR_RED}" in logs, logs
    assert f"sample present=5 published_at=6 target={target} rgba={NEAR_RED} red=0 match=1 " in logs, logs
    assert "x2w-red history record=3 " in logs, logs
    assert records == [("1", "1", "7", target, GREEN, NEAR_RED),
                       ("2", "2", "7", target, NEAR_RED, GREEN)], logs
    logs, records, _ = run("auto-onset")
    assert "auto armed" not in logs and not records, logs
    print(f"flat near-red edge arms after the nonuniform frame, with an onset-off control ({adapter})")


def check_levels(selection, floor, levels):
    logs, _, adapter = run("auto-levels", selection, onset=True, floor=f"{floor},1", rearm="3")
    armed = re.findall(
        r"auto armed sampled_present=(\d+) published_at=(\d+) target=(\w+) capture=(\d+)\.\.(\d+) "
        r"rgb=([\d,]+) ordinal=(\d+)", logs)
    writes = re.findall(r"producer order=\d+ ordinal=\d+ record=(\d+) submit=\d+ queue=\w+ target=(\w+) "
                        r"op=indirect-draw key=\w+ rt=0 rgba=([\d,]+) before=([\d,]+)", logs)
    assert len(armed) == len(levels) == len(writes), logs
    for ordinal, ((onset, level), capture, write) in enumerate(zip(levels, armed, writes), 1):
        target = capture[2]
        rgb = f"{level},{level},{level}"
        record = str(onset + 2)
        assert capture == (str(onset), str(onset + 1), target, record, record, rgb, str(ordinal)), logs
        assert write == (record, target, rgb + ",255", GREEN), logs
        # History must name the actual draw immediately before the onset, on the same target.
        end = logs.index(f" ordinal={ordinal}\n", logs.index(f"auto armed sampled_present={onset} "))
        section = logs[end:].split("auto armed", 1)[0]
        assert re.search(fr"x2w-red history record={onset - 1} list=\w+ target={target} op=indirect-draw ", section), logs
    if not levels:
        assert "x2w-red auto waiting present=25" in logs and "x2w-red auto floor reached" not in logs, logs
    print(f"producer onset at {levels}, floor {floor} ({adapter})")
    return armed


def black_capture(logs, records):
    arms = re.findall(r"auto armed sampled_present=(\d+) published_at=(\d+) target=(\w+) capture=(\d+)\.\.(\d+) rgb=0,0,0", logs)
    assert len(arms) == 1, logs
    sampled, published, target, first, last = arms[0]
    assert all(row[3] == target for row in records), logs
    return (sampled, published, first, last), [row[:3] + row[4:] for row in records]


def check_black_onset():
    # CPU reference: present 8 replaces (9,0,0) with (8,0,0) everywhere.
    # The next present publishes the sample. The two captured frames clear
    # 8 -> 0 -> 8 and 8 -> 0 respectively, with alpha always 255.
    black, dark = "0,0,0,255", "8,0,0,255"
    expected = ("8", "9", "10", "11"), [
        ("1", "1", "10", black, dark),
        ("2", "2", "10", dark, black),
        ("3", "3", "11", black, dark),
    ]
    logs, records, adapter = run("auto-black", onset="black", default_cadence=True)
    assert black_capture(logs, records) == expected, logs
    assert "black onset previous_present=7 sampled_present=8 previous_rgba=9,0,0,255" in logs, logs
    assert "sample present=8 published_at=9" in logs and "x2w-red history record=7 " in logs, logs

    # Exact RGB auto mode has no predecessor requirement. With an initial
    # exact black clear it exercises the premature-arm control on the same GPU.
    logs, records, _ = run("auto-black", "0,0,0")
    control = black_capture(logs, records)
    assert control[0] == ("1", "2", "3", "4") and control != expected, logs
    assert control[1] == [
        ("1", "1", "3", "255,255,255,255", black),
        ("2", "2", "4", "9,0,0,255", "255,255,255,255"),
    ], logs
    print(f"flat black onset arms on present 8. Initial-black control arms early on present 1 ({adapter})")


def check_barriers():
    logs, _, adapter = run("barriers")
    records = re.findall(r"CommandList::slot26\(ResourceBarrier\): forwarded=(\d+) refused=(\d+)", logs)
    assert records == [("2", "5"), ("0", "1"), ("0", "1")], logs
    print(f"slot 26: forwarded global and resource UAV barriers, refused five unsupported records, an unreadable array and a closed list ({adapter})")


if __name__ == "__main__" and OPTIONS == ["barriers"]:
    check_barriers()
elif __name__ == "__main__":
    check(False)
    check(True)
    check_auto()
    check_auto(magenta=True)
    check_default_cadence()
    check_onset()
    all_levels = [(6, 255), (16, 187), (21, 0)]
    any_captures = check_levels("any", "0s", all_levels)
    check_levels("any", "11", all_levels[1:])
    check_levels("any", "3600s", [])
    # Numeric black is also a GPU control for a matcher that incorrectly compares any-mode samples to a target.
    target_color_control = check_levels("0,0,0", "0s", all_levels[2:])
    assert [capture[0] for capture in target_color_control] != [capture[0] for capture in any_captures]
    check_black_onset()
    logs, records, adapter = run("auto-magenta", "255,0,256")
    assert "auto disabled: X2W_RED_PRODUCER_RGB" in logs and "auto armed" not in logs and not records, logs
    print(f"invalid RGB disables automatic arming ({adapter})")
