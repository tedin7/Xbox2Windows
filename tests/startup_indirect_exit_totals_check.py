"""Check the cumulative ExecuteIndirect totals written during DLL detach."""
import subprocess
import sys


def parse(line):
    fields = line.split(": ", 1)[1].split("; ")
    result = {}
    for field in fields:
        outcome, values = field.split(":", 1)
        tokens = values.split()
        if tokens != ["0"]:
            if len(tokens) % 2:
                raise ValueError(f"invalid totals: {line}")
            for kind, count in zip(tokens[::2], tokens[1::2]):
                result[outcome, kind] = int(count)
    return result


def main():
    run = subprocess.run([sys.argv[1], "warp"], capture_output=True, text=True, errors="replace")
    if run.returncode == 77:
        return 77
    if run.returncode:
        print(run.stderr, file=sys.stderr)
        return run.returncode
    lines = [line for line in run.stderr.splitlines() if line.startswith("x2w-render indirect")]
    if not lines or not lines[-1].startswith("x2w-render indirect at exit: "):
        raise AssertionError("missing final indirect exit totals")
    previous = [line for line in lines if line.startswith("x2w-render indirect: ")]
    if not previous:
        raise AssertionError("missing 300-frame totals")
    actual, earlier = parse(lines[-1]), parse(previous[-1])
    expected = {("forwarded", "drawindexed"): 3, ("refused graphics-pipeline", "drawindexed"): 1,
                ("compute-pipeline", "dispatch"): 1, ("placeholder-signature", "other"): 1,
                ("other", "drawindexed"): 3}
    if earlier != expected or any(actual.get(key, 0) < value for key, value in earlier.items()):
        raise AssertionError(f"unexpected indirect totals: earlier={earlier}, exit={actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
