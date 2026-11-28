"""Descriptor tables are captured even when the draw page time budget is spent.

The scene binds root addresses before every frame's table, and the budget is a few microseconds, so the vertex/constant
captures use it up first. The table is rewritten by the CPU every frame: the replay must still match the live pictures.
"""

from pathlib import Path
import sys
import tempfile

from startup_replay_draw_tables_check import PRESENTS, play, prepare, record


def check(folder, player, args, env):
    tiny = dict(env, X2W_REPLAY_DRAW_PAGES_MS="0.001", X2W_TEST_REPLAY_TABLE_PADDING="8")
    recording, live = record(folder, args, tiny, "1")
    played, _, _ = play(player, recording, folder / "played", env)
    assert played == live, "a table past the draw page time cap replays as stale memory"
    print(f"table cap: all {PRESENTS} replayed pictures match with a draw page budget of 1 microsecond")
    return 0


def main(scene, player):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepared = prepare(folder, scene)
        if prepared is None:
            return 77
        return check(folder, player, *prepared)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
