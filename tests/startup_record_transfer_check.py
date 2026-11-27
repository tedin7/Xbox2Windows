"""Check tools/startup/record_transfer.py: a recording cut into verified chunks, copied into the shared Drive folder resumably with the
.done marker last, and fetched back byte for byte; damaged, incomplete and hostile recordings are refused."""
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
import record_transfer as R  # noqa: E402


def make_record(folder):
    rng = __import__("random").Random(5)
    files = {"capture.bin": bytes(rng.randrange(256) for _ in range(3 * 1024 * 1024 + 123)), "notes/frames.txt": b"frame 1\n" * 1000,
             "empty.dat": b"", "sub/deep/blob.bin": os.urandom(200 * 1024)}
    for name, data in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return files


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        record, staging, drive, out = tmp / "record", tmp / "staging", tmp / "drive" / R.FOLDER_NAME, tmp / "out"
        record.mkdir()
        files = make_record(record)
        manifest = R.pack(str(record), str(staging), "T1", chunk_bytes=1024 * 1024)
        assert len(manifest["chunks"]) >= 3 and manifest["source_bytes"] == sum(len(d) for d in files.values()), manifest["chunks"]
        assert all(c["bytes"] <= 1024 * 1024 for c in manifest["chunks"]) and manifest["chunks"][0]["bytes"] == 1024 * 1024
        for chunk in manifest["chunks"]:
            assert hashlib.sha256((staging / chunk["name"]).read_bytes()).hexdigest() == chunk["sha256"]
        assert sorted(f["name"] for f in manifest["files"]) == sorted(files), manifest["files"]

        # upload: chunks first, manifest next, .done last; a rerun copies nothing; a lost chunk is copied again alone
        lines = []
        assert R.upload(str(staging), str(drive), lines.append) == ["T1"]
        assert (drive / "record_T1.done").is_file() and (drive / "record_T1.json").is_file()
        assert not [n for n in os.listdir(drive) if n.endswith(".tmp")]
        lines.clear()
        R.upload(str(staging), str(drive), lines.append)
        assert all("already there" in l for l in lines) and len(lines) == len(manifest["chunks"]), lines
        (drive / manifest["chunks"][1]["name"]).unlink()
        lines.clear()
        R.upload(str(staging), str(drive), lines.append)
        assert sum("copied" in l for l in lines) == 1, lines

        # no synced folder: the browser fallback lists every chunk with its size, in order, the manifest and the .done marker last
        plan = R.browser_plan(str(staging))
        text = "\n".join(plan)
        numbered = [l for l in plan if l.startswith("  ") and "record_T1" in l]
        assert len(numbered) == len(manifest["chunks"]) + 2 and "record_T1.done" in numbered[-1] and "LAST" in numbered[-1], numbered
        assert "record_T1.json" in numbered[-2] and all(c["name"] in text for c in manifest["chunks"]), text
        assert "drive.google.com" in text and "Carica questi file" in text and "files for this recording" in text
        assert (staging / "UPLOAD_THESE_FILES.txt").read_text() .startswith("Recording T1:") and (staging / "record_T1.done").is_file()
        import subprocess
        no_drive = subprocess.run([sys.executable, "-B", str(Path(R.__file__)), "upload", str(staging)], capture_output=True, text=True,
                                  env=dict(os.environ, USERPROFILE=str(tmp / "nohome"), HOME=str(tmp / "nohome")))
        assert no_drive.returncode in (0, 3), no_drive.stderr   # 3 = fallback printed (no synced folder here); 0 only on a PC that has one
        if no_drive.returncode == 3:
            assert "Upload by hand in the browser" in no_drive.stdout and "record_T1.done" in no_drive.stdout

        # fetch: byte for byte
        got = R.fetch(str(drive), str(out))
        assert got["id"] == "T1"
        for name, data in files.items():
            assert (out / name).read_bytes() == data, name

        # a recording without .done (half synced) is refused; so is a damaged or missing chunk
        (drive / "record_T1.done").unlink()
        try:
            R.fetch(str(drive), str(tmp / "out2"))
            raise AssertionError("fetched a recording without .done")
        except ValueError as error:
            assert "no complete recording" in str(error), error
        (drive / "record_T1.done").write_text("complete\n")
        victim = drive / manifest["chunks"][0]["name"]
        data = bytearray(victim.read_bytes())
        data[100] ^= 0xFF
        victim.write_bytes(bytes(data))
        try:
            R.fetch(str(drive), str(tmp / "out3"))
            raise AssertionError("fetched a damaged chunk")
        except ValueError as error:
            assert "missing or damaged" in str(error), error

        # a hostile archive (member outside the output folder) is refused even with valid hashes
        evil = tmp / "evil"
        evil.mkdir()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w|gz") as tar:
            info = tarfile.TarInfo("../escaped.txt")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"bad"))
        blob = buffer.getvalue()
        (evil / "record_E.part").write_bytes(blob)
        (evil / "record_E.json").write_text(json.dumps({"id": "E", "chunks": [{"name": "record_E.part", "bytes": len(blob),
                                                         "sha256": hashlib.sha256(blob).hexdigest()}], "files": [], "source_bytes": 3}))
        (evil / "record_E.done").write_text("x")
        try:
            R.fetch(str(evil), str(tmp / "out4"))
            raise AssertionError("unpacked a hostile member")
        except ValueError as error:
            assert "refused archive member" in str(error), error
        assert not (tmp / "escaped.txt").exists()

        # find_drive: the first existing folder of the candidate list
        assert R.find_drive([str(tmp / "nowhere"), str(drive)]) == str(drive) and R.find_drive([str(tmp / "nowhere")]) is None
    print("record_transfer: chunks verified, resumable upload with .done last, fetch byte for byte, damaged/incomplete/hostile refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
