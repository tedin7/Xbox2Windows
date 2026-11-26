"""Compare present colors against ordered producer readbacks on a source-built fixture."""
import os
import re
import subprocess
import sys

RED, GREEN, BLUE, MAGENTA = "255,0,0,255", "0,255,0,255", "0,0,255,255", "255,0,255,255"
BACKEND, EXECUTABLE, *OPTIONS = sys.argv[1:]


def run(mode, rgb=""):
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_PRESENT_SIGNATURE_EVERY="1", X2W_RED_PRODUCER_FRAMES="", X2W_RED_PRODUCER_AUTO="",
               X2W_RED_PRODUCER_RGB=rgb)
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
    # The writer is found from the rolling ring recorded BEFORE the colour was detected: the first operation on the target whose
    # centre pixel turns into the target colour while it was not that colour before.
    first_writer = re.search(r"x2w-red first writer: order=(\d+) record=(\d+) op=clear-rtv key=0+ rt=0 before=(\d+),(\d+),(\d+) after=" +
                             rgb + fr" target={target}", logs)
    assert "x2w-red rolling history:" in logs and first_writer, logs
    before = tuple(int(v) for v in first_writer.groups()[2:5])
    assert before != tuple(int(v) for v in rgb.split(",")) and int(first_writer.group(2)) < 7, logs  # recorded before the capture frames
    print(f"auto {rgb} control: excludes early and near matches and compares paired samples within the armed window ({adapter})")


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
    logs, records, adapter = run("auto-magenta", "255,0,256")
    assert "auto disabled: X2W_RED_PRODUCER_RGB" in logs and "auto armed" not in logs and not records, logs
    print(f"invalid RGB disables automatic arming ({adapter})")
