"""Batch exporter for TTPF (RDS5) texture pools to DDS with manifest generation.

Accepts an RDS5/ETPF texture pool archive and a range of texture indices. Tries
strict export by default and optionally retries with provisional history
or an explicit local calibration profile. Writes each new export to a
temporary path and publishes it only on successful completion. Records index,
dimensions, codec, export mode, failure reason, and output DDS
SHA-256 in a structured JSON manifest.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

FORMAT_NAMES = {
    70: "BC1",
    71: "BC1",
    72: "BC1_SRGB",
    73: "BC2",
    74: "BC2",
    75: "BC2_SRGB",
    76: "BC3",
    77: "BC3",
    78: "BC3_SRGB",
    79: "BC4",
    80: "BC4",
    81: "BC4_SNORM",
    82: "BC5",
    83: "BC5",
    84: "BC5_SNORM",
    94: "BC6H",
    95: "BC6H",
    96: "BC6H_SF16",
    97: "BC7",
    98: "BC7",
    99: "BC7_SRGB",
}


def _index_int(token: str, part: str) -> int:
    try:
        return int(token.strip())
    except ValueError:
        raise ValueError(f"Invalid index '{token.strip()}' in '{part}': expected an integer") from None


def parse_indices(spec: str, total_count: int | None = None) -> list[int]:
    """Parse index specifications like '0,8253', '0-10', or '0-5,10,20-25'."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = _index_int(start_str, part), _index_int(end_str, part)
            if start > end:
                raise ValueError(f"Invalid range '{part}': start ({start}) > end ({end})")
            if total_count is not None:
                end = min(end, total_count - 1)
            result.update(range(start, end + 1))
        else:
            idx = _index_int(part, part)
            if total_count is None or idx < total_count:
                result.add(idx)
    return sorted(result)


def load_tools(tools_path: Path | None = None):
    """Import ttpf_mips and bcpack from tools_path, environment or sys.path."""
    candidate_paths = []
    if tools_path is not None:
        candidate_paths.append(Path(tools_path).resolve())
    for env_key in ("TTPF_TOOLS_DIR", "BCPACK_REFERENCE_TOOLS"):
        val = os.environ.get(env_key)
        if val:
            candidate_paths.append(Path(val).resolve())
    script_dir = Path(__file__).resolve().parent
    candidate_paths.append(script_dir)

    for path in candidate_paths:
        if path.is_dir() and (path / "ttpf_mips.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            break

    try:
        import ttpf_mips
        return ttpf_mips
    except ImportError:
        return None


def require_rust_backend(ttpf):
    """Check the native library used by the selected decoder before starting a batch."""
    native = getattr(getattr(ttpf, "bcpack", None), "_native", None)
    if native is None:
        raise RuntimeError("The selected Leonida tools do not include the optional Rust BCPack backend.")
    try:
        library = native.library()
        for kernel in ("unpack_lanes", "decode_strict_bc1", "decode_strict_mode6", "rebuild_calibrated_bc1"):
            getattr(library, kernel)
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "Could not load the Rust BCPack kernels. Build the selected Leonida "
            "tools/bcpack_calibration crate in release mode."
        ) from exc


def export_texture_entry(pool, index: int, out_dir: Path,
                         allow_provisional: bool = False,
                         history: dict | None = None,
                         ttpf_module=None,
                         fitted_decoder=None) -> dict:
    """Export one texture entry from the pool to a DDS file and return its manifest entry.

    Writes to a temporary file first and publishes to dds_path only upon success.
    A failed export leaves existing DDS files intact. Only success entries in the
    current run's manifest describe DDS files produced by that run.
    """
    dds_name = f"tex_{index:05d}.dds"
    dds_path = out_dir / dds_name
    temp_path = out_dir / f".tmp_{index}_{os.getpid()}_{time.time_ns()}.dds"

    if ttpf_module is None:
        ttpf_module = load_tools()
    if ttpf_module is None:
        raise RuntimeError("ttpf_mips module is required to export entries")

    analyze_fn = getattr(ttpf_module, "analyze")
    export_fn = getattr(ttpf_module, "export")
    bc_formats = getattr(ttpf_module, "BC_FORMATS", set(FORMAT_NAMES.keys()))

    try:
        a = analyze_fn(pool, index)
    except Exception as exc:
        return {
            "index": index,
            "status": "failed",
            "reason": str(exc),
        }

    if "layout" not in a or "available" not in a:
        return {
            "index": index,
            "status": "no_layout",
            "reason": a.get("blocked", "layout unavailable"),
        }

    d = a.get("descriptor", {})
    shape = a.get("shape", "2d")
    fmt = d.get("format")
    w, h = d.get("width", 0), d.get("height", 0)
    mips = d.get("mips", 0)

    if shape != "volume" and fmt not in bc_formats:
        return {
            "index": index,
            "status": "uncompressed",
            "format": fmt,
            "format_name": FORMAT_NAMES.get(fmt, str(fmt)),
            "shape": shape,
            "width": w,
            "height": h,
        }

    if not a.get("full_chain", False):
        return {
            "index": index,
            "status": "unsupported_codec",
            "format": fmt,
            "format_name": FORMAT_NAMES.get(fmt, str(fmt)),
            "width": w,
            "height": h,
        }

    details = {
        "index": index,
        "format": fmt,
        "format_name": FORMAT_NAMES.get(fmt, f"Format {fmt}"),
        "width": w,
        "height": h,
    }
    try:
        mode_used = "strict"
        ret = export_fn(pool, index, str(temp_path), allow_provisional=False)

        if ret == 2 and fitted_decoder is not None:
            mode_used = "fitted"
            decoder = fitted_decoder(pool)
            try:
                data = decoder.dds(index)
            finally:
                decoder.read_chunk.cache_clear()
            temp_path.write_bytes(data)
            ret = 0
        elif ret != 0 and allow_provisional:
            mode_used = "provisional"
            temp_path.unlink(missing_ok=True)
            ret = export_fn(pool, index, str(temp_path), allow_provisional=True, history=history)

        if ret != 0 or not temp_path.exists():
            return {
                **details,
                "status": "failed",
                "reason": f"export returned exit code {ret}",
            }

        # Atomic publish: move temporary file to final published path
        data = temp_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        os.replace(temp_path, dds_path)
        return {
            **details,
            "status": "success",
            "mode": mode_used,
            "mips": mips,
            "dds_size": len(data),
            "sha256": sha256,
            "file": dds_name,
        }
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        return {**details, "status": "failed", "reason": str(exc)}
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as staging_dir:
        staged = Path(staging_dir) / path.name
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, path)


def batch_export(pool_path: Path, out_dir: Path,
                 indices: list[int] | None = None,
                 start: int = 0, count: int | None = None,
                 allow_provisional: bool = False,
                 history_path: Path | None = None,
                 threads: int = 4,
                 manifest_path: Path | None = None,
                 tools_path: Path | None = None,
                 require_rust: bool = False,
                 calibration_path: Path | None = None) -> dict:
    """Run batch export over a range of texture pool indices and write manifest.json."""
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    if count is not None and count < 0:
        raise ValueError(f"--count must be >= 0, got {count}")

    if pool_path.exists() and (pool_path.is_dir() or not pool_path.is_file()):
        raise ValueError(f"Pool path must be a regular file: {pool_path}")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out path is not a directory: {out_dir}")
    if _same_path(out_dir, pool_path):
        raise ValueError(f"--out path collides with pool: {out_dir}")

    resolved_manifest = manifest_path if manifest_path is not None else out_dir / "manifest.json"
    if resolved_manifest.exists() and resolved_manifest.is_dir():
        raise ValueError(f"--manifest path is an existing directory: {resolved_manifest}")
    if _same_path(resolved_manifest, pool_path):
        raise ValueError(f"--manifest path collides with pool: {resolved_manifest}")
    if _same_path(resolved_manifest, out_dir):
        raise ValueError(f"--manifest path collides with --out: {resolved_manifest}")
    if resolved_manifest.resolve() in out_dir.resolve().parents:
        raise ValueError(f"--manifest path cannot be an ancestor of --out: {resolved_manifest}")
    for option, source_path in (("history", history_path), ("calibration", calibration_path)):
        if source_path is not None and _same_path(resolved_manifest, source_path):
            raise ValueError(f"--manifest path collides with --{option}: {resolved_manifest}")

    if calibration_path is not None and (allow_provisional or history_path is not None):
        raise ValueError("--calibration cannot be combined with --allow-provisional or --history")

    history_dict = None
    if history_path is not None:
        if history_path.exists() and history_path.is_dir():
            raise ValueError(f"Cannot read --history {history_path}: path is a directory")
        try:
            history_dict = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Cannot read --history {history_path}: {exc}") from exc
        if not isinstance(history_dict, dict):
            raise ValueError(f"Expected a JSON object in --history {history_path}")
        if not allow_provisional:
            history_dict = None

    calibration = None
    if calibration_path is not None:
        if calibration_path.exists() and calibration_path.is_dir():
            raise ValueError(f"Cannot read --calibration {calibration_path}: path is a directory")
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Cannot read --calibration {calibration_path}: {exc}") from exc
        if not isinstance(calibration, dict) or calibration.get("version") != 1:
            raise ValueError("Unsupported calibration version (expected version 1)")

    if tools_path is not None:
        if not tools_path.exists():
            raise FileNotFoundError(f"--tools-path not found: {tools_path}")
        if not tools_path.is_dir():
            raise ValueError(f"--tools-path must be a directory: {tools_path}")

    ttpf = load_tools(tools_path)
    if ttpf is None:
        raise RuntimeError("Could not find ttpf_mips module. Provide --tools-path or set TTPF_TOOLS_DIR.")
    if require_rust:
        require_rust_backend(ttpf)

    fitted_decoder = None
    if calibration is not None:
        try:
            calibrated = importlib.import_module("calibrated_texture")
            if (calibrated.ttpf is not ttpf
                    or Path(calibrated.__file__).resolve().parent != Path(ttpf.__file__).resolve().parent):
                raise ImportError("calibrated_texture and ttpf_mips must come from the same tools directory")
            calibrated.native()
        except (ImportError, AttributeError, OSError) as exc:
            raise RuntimeError(
                f"Cannot load the fitted decoder from the selected Leonida tools: {exc}. "
                "Install NumPy and build tools/bcpack_calibration in release mode."
            ) from exc

        def fitted_decoder(pool):
            return calibrated.LocalDecoder(pool, calibration)

    start_time = time.perf_counter()
    pool = ttpf.Pool(str(pool_path))
    try:
        total_textures = pool.textures
    finally:
        pool.close()

    manifest_path = resolved_manifest

    if indices is None:
        start_idx = max(0, start)
        end_idx = total_textures if count is None else min(total_textures, start_idx + count)
        target_indices = list(range(start_idx, end_idx))
    else:
        target_indices = [i for i in indices if 0 <= i < total_textures]

    # The manifest is published after the workers finish. Its path must not name
    # a requested DDS, including one preserved after a failed or skipped export.
    for index in target_indices:
        if _same_path(manifest_path, out_dir / f"tex_{index:05d}.dds"):
            raise ValueError(f"--manifest path collides with DDS output: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []
    stats = {"strict": 0, "provisional": 0, "failed": 0, "skipped": 0}
    if fitted_decoder is not None:
        stats["fitted"] = 0

    def _worker(idx: int) -> dict:
        local_pool = ttpf.Pool(str(pool_path))
        try:
            return export_texture_entry(local_pool, idx, out_dir,
                                        allow_provisional=allow_provisional,
                                        history=history_dict,
                                        ttpf_module=ttpf,
                                        fitted_decoder=fitted_decoder)
        finally:
            local_pool.close()

    max_workers = max(1, min(threads, len(target_indices) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, idx): idx for idx in target_indices}
        for future in as_completed(futures):
            res = future.result()
            manifest_entries.append(res)
            st = res.get("status")
            if st == "success":
                stats[res["mode"]] += 1
            elif st in ("no_layout", "uncompressed", "unsupported_codec"):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

    manifest_entries.sort(key=lambda x: x.get("index", 0))
    _atomic_write_text(manifest_path, json.dumps(manifest_entries, indent=2))
    elapsed = time.perf_counter() - start_time

    return {
        "pool": str(pool_path),
        "total_textures": total_textures,
        "processed": len(target_indices),
        "manifest": str(manifest_path),
        "entries": len(manifest_entries),
        "elapsed_seconds": round(elapsed, 3),
        "stats": stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path, help="Path to .ttpf texture pool archive")
    parser.add_argument("-o", "--out", type=Path, default=Path("extracted_dds"),
                        help="Output directory for exported DDS files")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices or ranges, e.g. '0,8253' or '0-10'")
    parser.add_argument("--start", type=int, default=0, help="Starting index (default: 0)")
    parser.add_argument("--count", type=int, default=None, help="Number of textures to export")
    parser.add_argument("--allow-provisional", action="store_true", default=False,
                        help="Opt in to fitted provisional history if strict export is rejected")
    parser.add_argument("--history", type=Path, default=None,
                        help="Path to history.json calibration file")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 4),
                        help="Number of worker threads")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Output path for JSON manifest (default: <out>/manifest.json)")
    parser.add_argument("--tools-path", type=Path, default=None,
                        help="Path to directory containing ttpf_mips.py and bcpack.py")
    parser.add_argument("--require-rust", action="store_true",
                        help="Require the optional Rust BCPack library to load before exporting")
    parser.add_argument("--calibration", type=Path,
                        help="Retry strict failures with a local version-1 Leonida calibration profile")
    args = parser.parse_args(argv)

    if not args.pool.exists():
        parser.error(f"Pool not found: {args.pool}")

    try:
        indices = parse_indices(args.indices) if args.indices else None
    except ValueError as exc:
        parser.error(f"--indices: {exc}")

    try:
        summary = batch_export(
            pool_path=args.pool,
            out_dir=args.out,
            indices=indices,
            start=args.start,
            count=args.count,
            allow_provisional=args.allow_provisional,
            history_path=args.history,
            threads=args.threads,
            manifest_path=args.manifest,
            tools_path=args.tools_path,
            require_rust=args.require_rust,
            calibration_path=args.calibration,
        )
    except (RuntimeError, OSError, ValueError, struct.error, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
