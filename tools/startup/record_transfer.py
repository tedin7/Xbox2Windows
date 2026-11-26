#!/usr/bin/env python3
"""Moves a recording (a folder of capture files, GBs) from the tester's PC to us through a shared Google Drive folder, with no
credentials in any tool: the tester's PC has Drive for desktop (or a shortcut to the shared folder 'x2w-registrazioni' in My Drive)
and this script only copies files into that synced folder.

  record_transfer.py pack   <record dir> <staging dir> [--id ID] [--chunk-mb 100]   tar+gzip stream cut into chunks + a manifest
  record_transfer.py upload <staging dir> <drive folder>                            resumable copy, chunk by chunk, DONE marker last
  record_transfer.py find-drive                                                     print the synced x2w-registrazioni folder, if any
  record_transfer.py fetch  <drive folder or local copy> <out dir> [--id ID]       (our side) verify every chunk, then unpack

Files in the staging/drive folder for one recording <id>:
  record_<id>_0001.part ...   chunks of one tar.gz stream
  record_<id>.json            manifest: chunk names, sizes, sha256, total bytes, file list
  record_<id>.done            written LAST, so a half-synced recording is never pulled (rclone copy gdrive:x2w-registrazioni <dir>,
                              then `fetch` refuses an id without its .done)
Recordings are PRIVATE game data: local and Drive only, never GitHub.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time

FOLDER_NAME = "x2w-registrazioni"
CHUNK_BYTES = 100 * 1024 * 1024
DRIVE_NAMES = ("My Drive", "Il mio Drive", "")   # Drive for desktop's root folder name depends on the Windows language


class ChunkWriter:
    """File-like object for tarfile's streaming writer: bytes go to numbered chunk files of at most `chunk_bytes`."""

    def __init__(self, directory, record_id, chunk_bytes):
        self.directory, self.record_id, self.chunk_bytes = directory, record_id, chunk_bytes
        self.chunks, self._file, self._hash, self._size = [], None, None, 0

    def _open(self):
        name = "record_%s_%04d.part" % (self.record_id, len(self.chunks) + 1)
        self._file = open(os.path.join(self.directory, name), "wb")
        self._hash, self._size, self._name = hashlib.sha256(), 0, name

    def _close(self):
        if self._file:
            self._file.close()
            self.chunks.append({"name": self._name, "bytes": self._size, "sha256": self._hash.hexdigest()})
            self._file = None

    def write(self, data):
        view = memoryview(data)
        while len(view):
            if self._file is None:
                self._open()
            room = self.chunk_bytes - self._size
            piece = view[:room]
            self._file.write(piece)
            self._hash.update(piece)
            self._size += len(piece)
            view = view[len(piece):]
            if self._size >= self.chunk_bytes:
                self._close()
        return len(data)

    def flush(self):
        pass

    def finish(self):
        self._close()


def pack(record_dir, staging_dir, record_id=None, chunk_bytes=CHUNK_BYTES):
    """Cuts the folder into chunks of one compressed tar stream; returns the manifest (also written as record_<id>.json)."""
    record_dir = os.path.abspath(record_dir)
    if not os.path.isdir(record_dir):
        raise ValueError("not a folder: " + record_dir)
    record_id = record_id or time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(staging_dir, exist_ok=True)
    files = []
    for root, _, names in os.walk(record_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                files.append((os.path.relpath(path, record_dir).replace(os.sep, "/"), os.path.getsize(path)))
    files.sort()
    writer = ChunkWriter(staging_dir, record_id, chunk_bytes)
    with gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=3, mtime=0) as zipped:   # level 3: GBs of capture data, speed over size
        with tarfile.open(fileobj=zipped, mode="w|") as tar:
            for relative, _ in files:
                tar.add(os.path.join(record_dir, relative), arcname=relative, recursive=False)
    writer.finish()
    manifest = {"id": record_id, "chunk_bytes": chunk_bytes, "chunks": writer.chunks, "packed_bytes": sum(c["bytes"] for c in writer.chunks),
                "source_bytes": sum(size for _, size in files), "files": [{"name": n, "bytes": b} for n, b in files]}
    with open(os.path.join(staging_dir, "record_%s.json" % record_id), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    return manifest


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(staging_dir, drive_dir, log=print):
    """Copies the chunks and the manifest of every recording in `staging_dir` into `drive_dir`, resumable: a chunk already there with
    the right size and hash is skipped, a copy goes to a temporary name first. The .done marker is written last. Returns the ids."""
    os.makedirs(drive_dir, exist_ok=True)
    ids = sorted(n[len("record_"):-len(".json")] for n in os.listdir(staging_dir) if n.startswith("record_") and n.endswith(".json"))
    for record_id in ids:
        with open(os.path.join(staging_dir, "record_%s.json" % record_id), encoding="utf-8") as handle:
            manifest = json.load(handle)
        for number, chunk in enumerate(manifest["chunks"], 1):
            source, target = os.path.join(staging_dir, chunk["name"]), os.path.join(drive_dir, chunk["name"])
            if os.path.isfile(target) and os.path.getsize(target) == chunk["bytes"] and sha256_of(target) == chunk["sha256"]:
                log("upload %s: chunk %d/%d already there" % (record_id, number, len(manifest["chunks"])))
                continue
            shutil.copyfile(source, target + ".tmp")
            os.replace(target + ".tmp", target)
            if sha256_of(target) != chunk["sha256"]:
                raise IOError("chunk %s differs after the copy" % chunk["name"])
            log("upload %s: chunk %d/%d copied (%.1f MB)" % (record_id, number, len(manifest["chunks"]), chunk["bytes"] / 1048576.0))
        shutil.copyfile(os.path.join(staging_dir, "record_%s.json" % record_id), os.path.join(drive_dir, "record_%s.json" % record_id))
        with open(os.path.join(drive_dir, "record_%s.done" % record_id), "w", encoding="utf-8") as handle:
            handle.write("complete: %d chunks, %d bytes\n" % (len(manifest["chunks"]), manifest["packed_bytes"]))
    return ids


def find_drive(roots=None):
    """The synced x2w-registrazioni folder on this PC: <letter>:\\My Drive\\..., <letter>:\\Il mio Drive\\..., <letter>:\\..., the
    user's 'Google Drive' folder. Returns the first that exists, else None."""
    candidates = []
    for letter in "GHIJKLMNOPQRSTUVWXYZDEF":
        for drive in DRIVE_NAMES:
            candidates.append(os.path.join("%s:\\" % letter, drive, FOLDER_NAME) if drive else os.path.join("%s:\\" % letter, FOLDER_NAME))
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    for base in ("Google Drive", "My Drive", "Il mio Drive", os.path.join("Google Drive", "My Drive")):
        candidates.append(os.path.join(home, base, FOLDER_NAME))
    for path in (roots if roots is not None else candidates):
        if os.path.isdir(path):
            return path
    return None


def fetch(source_dir, out_dir, record_id=None):
    """Our side: refuses a recording without its .done marker, verifies every chunk's size and sha256, then unpacks the stream into
    `out_dir` (member names are checked: no absolute path, no '..'). Returns the manifest."""
    done = sorted(n[len("record_"):-len(".done")] for n in os.listdir(source_dir) if n.startswith("record_") and n.endswith(".done"))
    record_id = record_id or (done[-1] if done else None)
    if not record_id or record_id not in done:
        raise ValueError("no complete recording (no record_<id>.done) in %s%s" % (source_dir, " for " + record_id if record_id else ""))
    with open(os.path.join(source_dir, "record_%s.json" % record_id), encoding="utf-8") as handle:
        manifest = json.load(handle)
    for chunk in manifest["chunks"]:
        path = os.path.join(source_dir, chunk["name"])
        if not os.path.isfile(path) or os.path.getsize(path) != chunk["bytes"] or sha256_of(path) != chunk["sha256"]:
            raise ValueError("chunk %s is missing or damaged" % chunk["name"])

    class Joiner:
        def __init__(self):
            self.iter = iter(manifest["chunks"])
            self.file = None

        def read(self, size=-1):
            out = b""
            while size < 0 or len(out) < size:
                if self.file is None:
                    chunk = next(self.iter, None)
                    if chunk is None:
                        break
                    self.file = open(os.path.join(source_dir, chunk["name"]), "rb")
                data = self.file.read(size - len(out) if size >= 0 else 1 << 20)
                if not data:
                    self.file.close()
                    self.file = None
                    continue
                out += data
            return out

    os.makedirs(out_dir, exist_ok=True)
    root = os.path.abspath(out_dir)
    with tarfile.open(fileobj=Joiner(), mode="r|gz") as tar:
        for member in tar:
            target = os.path.abspath(os.path.join(root, member.name))
            if not member.isfile() or os.path.commonpath([root, target]) != root:
                raise ValueError("refused archive member: " + member.name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tar.extractfile(member) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 20)
    return manifest


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("record_dir")
    p.add_argument("staging_dir")
    p.add_argument("--id")
    p.add_argument("--chunk-mb", type=float, default=CHUNK_BYTES / 1048576.0)
    u = sub.add_parser("upload")
    u.add_argument("staging_dir")
    u.add_argument("drive_dir")
    sub.add_parser("find-drive")
    f = sub.add_parser("fetch")
    f.add_argument("source_dir")
    f.add_argument("out_dir")
    f.add_argument("--id")
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        manifest = pack(args.record_dir, args.staging_dir, args.id, int(args.chunk_mb * 1048576))
        print("record %s: %d files, %.1f MB -> %d chunks, %.1f MB packed" % (manifest["id"], len(manifest["files"]),
              manifest["source_bytes"] / 1048576.0, len(manifest["chunks"]), manifest["packed_bytes"] / 1048576.0))
    elif args.cmd == "upload":
        for record_id in upload(args.staging_dir, args.drive_dir):
            print("record %s uploaded: complete (.done written)" % record_id)
    elif args.cmd == "find-drive":
        found = find_drive()
        print(found or "")
        return 0 if found else 1
    else:
        manifest = fetch(args.source_dir, args.out_dir, args.id)
        print("record %s unpacked: %d files, %.1f MB" % (manifest["id"], len(manifest["files"]), manifest["source_bytes"] / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
