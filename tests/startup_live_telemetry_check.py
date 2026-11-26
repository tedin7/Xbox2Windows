"""Check the live viewer with generated startup log lines."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


class LiveViewCheck(unittest.TestCase):
    def view_log(self, first_chunk, second_chunk=None):
        viewer = Path(__file__).resolve().parents[1] / "tools" / "startup" / "live_view.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "trap.log"
            log.write_text(first_chunk, encoding="utf-8")
            finished = root / "finished.txt"

            def finish_run(_seconds):
                if not finished.exists():
                    if second_chunk is not None:
                        with log.open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(second_chunk)
                    finished.write_text("finished\n", encoding="utf-8")

            if second_chunk is None:
                finish_run(0)
            output = io.StringIO()
            with mock.patch.object(sys, "argv", [str(viewer), directory]), \
                    mock.patch("os.system"), mock.patch("time.sleep", side_effect=finish_run), \
                    contextlib.redirect_stdout(output):
                runpy.run_path(str(viewer), run_name="__main__")
        return output.getvalue()

    def test_frame_heartbeat_does_not_claim_draws_were_dropped(self):
        output = self.view_log("x2w-d3d12 heartbeat: 60 frames presented in 2.0 s (30.0 fps overall)\n")
        self.assertIn("60 frames presented, 30.0 fps", output)
        self.assertNotIn("every draw dropped", output)
        self.assertNotIn("game runs offline", output)

    def test_storage_and_render_totals_are_preserved(self):
        storage = ("x2w-dstorage heartbeat 10 s: requests done=12 inflated=7 failed=2 bcpack-blank=1; "
                   "page tokens 0 at enqueue=3, still 0 at run=1; commit peak=900 MiB current=800 MiB, "
                   "live physical-page sections=42")
        render = ("x2w-render totals: resets 2 (failed 0), closes 2, clears 1, executes 2, "
                  "draws 4 (skipped 1), translated draws 3 (not drawn 2), pipelines translated 4 / missing 1, "
                  "null bindings 5 (entry 1, buffer 1, token 1, wrapper 1, dimension 1), "
                  "textures uploaded 2 (16 KiB, unreadable 1)")
        output = self.view_log(storage + "\n" + render + "\n")
        self.assertIn(storage, output)
        self.assertIn(render, output)
        self.assertIn("7 chunks decompressed", output)

    def test_status_lines_old_and_new_format(self):
        # #2816: the counters of the status lines are raw= (uncompressed copies) and completed= (raw + inflated +
        # bcpack-blank); "inflated=N failed=N" is unchanged. An old line (done=) still parses, and a run with only
        # compressed requests reads as progressing in both.
        new = ("x2w-dstorage Queue::EnqueueStatus array=0x1 index=3 (requests raw=0 inflated=5 failed=1 "
               "bcpack-blank=2 completed=7)")
        old = "x2w-dstorage Queue::EnqueueStatus array=0x1 index=3 (requests done=0 inflated=5 failed=1)"
        beat = ("x2w-dstorage heartbeat 10 s: requests raw=0 inflated=5 failed=1 bcpack-blank=2 completed=7; "
                "page tokens 0 at enqueue=0, still 0 at run=0; commit unknown, live physical-page sections=unknown")
        old_beat = ("x2w-dstorage heartbeat 10 s: requests done=0 inflated=5 failed=1 bcpack-blank=2; "
                    "page tokens 0 at enqueue=0, still 0 at run=0; commit unknown, live physical-page sections=unknown")
        for line, completed in ((new, 7), (old, 5), (beat, 7), (old_beat, 7)):
            output = self.view_log(line + "\n")
            self.assertIn("5 chunks decompressed", output, line)
            self.assertIn(f"{completed} storage requests completed", output, line)
        self.assertIn("7 requests completed so far, 5 compressed chunks decompressed, 1 failed", self.view_log(new + "\n"))
        self.assertIn("5 requests completed so far, 5 compressed chunks decompressed, 1 failed", self.view_log(old + "\n"))

    def test_unknown_and_split_heartbeat(self):
        first = "x2w-dstorage heartbeat 20 s: requests done=0 inflated=0 failed=0 bcpack-blank=0; "
        second = "page tokens 0 at enqueue=0, still 0 at run=0; commit unknown, live physical-page sections=unknown\n"
        output = self.view_log(first, second)
        self.assertEqual(output.count(first + second.strip()), 1)


if __name__ == "__main__":
    unittest.main()
