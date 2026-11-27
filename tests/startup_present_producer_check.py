"""Compare present colors against ordered producer readbacks on a source-built fixture."""
import math
import os
import re
import subprocess
import sys

NEAR_RED = "254,0,0,255"
RED, GREEN, BLUE, MAGENTA = "255,0,0,255", "0,255,0,255", "0,0,255,255", "255,0,255,255"
BACKEND, EXECUTABLE, *OPTIONS = sys.argv[1:]


def run(mode, rgb="", *, onset=False, default_cadence=False):
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_PRESENT_SIGNATURE_EVERY="1", X2W_RED_PRODUCER_FRAMES="", X2W_RED_PRODUCER_AUTO="",
               X2W_RED_PRODUCER_RGB=rgb, X2W_RED_PRODUCER_ONSET="1" if onset else "")
    if default_cadence:
        env.pop("X2W_PRESENT_SIGNATURE_EVERY")
    if mode.startswith("auto"):
        env["X2W_RED_PRODUCER_AUTO"] = "4,2"
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
    expected = [float(int(c) != 0) for c in last.split(",")]
    tree = re.search(fr"x2w-red tree record=2 level=0 target={target} writer=clear-rtv \w+ output=0 order=2 value=(.*)", logs)
    assert tree and [float(v) for v in re.findall(r"mean=([^)]+)", tree[1])] == expected, logs
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
    logs, records, adapter = run("auto-onset", onset=True)
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


def check_barriers():
    logs, _, adapter = run("barriers")
    records = re.findall(r"CommandList::slot26\(ResourceBarrier\): forwarded=(\d+) refused=(\d+)", logs)
    assert records == [("2", "5"), ("0", "1"), ("0", "1")], logs
    print(f"slot 26: forwarded global and resource UAV barriers, refused five unsupported records, an unreadable array and a closed list ({adapter})")


def run_tree(mode, enabled):
    env = dict(os.environ)
    for name in ("FRAMES", "AUTO", "ONSET", "RGB"):
        env.pop("X2W_RED_PRODUCER_" + name, None)
    if enabled:
        env["X2W_RED_PRODUCER_FRAMES"] = "2,4"
    result = subprocess.run([EXECUTABLE, mode, BACKEND], env=env, capture_output=True, text=True, timeout=25)
    if result.returncode == 77:
        raise SystemExit(77)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert not re.search(r"D3D12 (?:ERROR|CORRUPTION)", result.stderr), result.stderr
    pixels = re.findall(r"frame=(\d+) rgba=([0-9a-f]+)", result.stdout)
    expected = bytes(value for _ in range(8) for x in range(8) for value in (x + 1, 65, 128, 255)).hex()
    assert pixels == [(str(frame), expected) for frame in (2, 3, 4)], result.stdout
    return result.stderr, pixels


def verify_middle(rows, op):
    assert len(rows) >= 2, rows
    assert "level=1 input slot=0" in rows[1] and f"writer={op} 0000000000000022 output=0 order=3" in rows[1], rows


def verify_tree(logs):
    for frame, middle in ((2, "translated-draw"), (3, "translated-dispatch")):
        rows = re.findall(fr"^x2w-red tree record={frame} (level=.*)$", logs, flags=re.MULTILINE)
        verify_middle(rows, middle)
        assert len(rows) == 5, rows
        root, z, x, depth, y = rows
        assert "level=0 target=" in root and "writer=translated-draw 0000000000000033 output=0 order=4" in root, root
        assert "level=2 input slot=0" in x and "writer=translated-draw 0000000000000011 output=0 order=2" in x, x
        assert "level=2 input slot=1" in y and "last written in record 1" in y and " ZERO UNWRITTEN" in y, y
        assert "level=3 input slot=32" in depth and "writer=clear-dsv" in depth, depth
        assert all("UNWRITTEN" not in row and " ZERO" not in row for row in (root, z, x, depth)), rows
        expected_r = (5/255, 2/255, 8/255, 5/255)
        for row, expected in ((root, (expected_r, (65/255,) * 4, (128/255,) * 4, (1,) * 4)),
                              (z, (expected_r, (65/255,) * 4, (128/255,) * 4, (1,) * 4)),
                              (x, ((5/256, 2/256, 8/256, 5/256), (0.25,) * 4, (257/512,) * 4, (1,) * 4)),
                              (y, ((0,) * 4,) * 4),
                              (depth, ((0.375,) * 4,))):
            found = re.findall(r"c(\d+)\(center=([^,]+),min=([^,]+),max=([^,]+),mean=([^)]+)\)", row)
            assert [int(channel) for channel, *_ in found] == list(range(len(expected))), row
            for (_, *values), reference in zip(found, expected):
                assert all(math.isclose(float(v), e, abs_tol=1e-7) for v, e in zip(values, reference)), row
    assert "x2w-red tree record=4 same as record=3" in logs, logs
    assert "x2w-red tree record=4 level=" not in logs, logs


def check_tree():
    logs, captured = run_tree("tree-scene", True)
    verify_tree(logs)
    off_logs, off = run_tree("tree-scene", False)
    base_logs, baseline = run_tree("tree-baseline", False)
    assert "x2w-red tree" not in off_logs + base_logs and off == baseline == captured
    mutant, pixels = run_tree("tree-mutant", True)
    assert pixels == baseline
    rows = re.findall(r"^x2w-red tree record=3 (level=.*)$", mutant, flags=re.MULTILINE)
    assert len(rows) == 2 and "writer=last written in record 2" in rows[1] and "UNWRITTEN" in rows[1], mutant
    try:
        verify_middle(rows, "translated-dispatch")
    except AssertionError:
        print("compute writer omission rejected while the WARP pixels still match the CPU reference")
    else:
        raise AssertionError("compute writer omission passed the positive oracle")


if __name__ == "__main__" and OPTIONS == ["barriers"]:
    check_barriers()
elif __name__ == "__main__" and OPTIONS == ["tree"]:
    check_tree()
elif __name__ == "__main__":
    check(False)
    check(True)
    check_auto()
    check_auto(magenta=True)
    check_default_cadence()
    check_onset()
    logs, records, adapter = run("auto-magenta", "255,0,256")
    assert "auto disabled: X2W_RED_PRODUCER_RGB" in logs and "auto armed" not in logs and not records, logs
    print(f"invalid RGB disables automatic arming ({adapter})")
