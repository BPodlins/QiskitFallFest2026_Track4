"""Save/load candidate subspaces under results/ (never overwrite)."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(os.environ.get("N2SQD_RESULTS", Path(__file__).resolve().parent.parent / "results"))

_PACKAGES = ("numpy", "scipy", "pyscf", "ffsim", "qiskit", "qiskit-ibm-runtime",
             "qiskit-addon-sqd", "matplotlib", "qfest-grader")


def _package_versions() -> dict:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "na"


def _jsonable(obj):
    """Convert numpy scalars/arrays to plain JSON-serialisable Python objects."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_candidate(strings_a, strings_b, hcore, eri, ecore, energy, *, method: str,
                   seed: int, rotated: bool = False, results_dir: Path | str = RESULTS_DIR,
                   amplitudes=None, orbital_rotation=None, details: dict | None = None,
                   canonical_hcore=None, canonical_eri=None) -> Path:
    """Save a candidate .npz + JSON sidecar; filename holds energy + UTC timestamp. Refuses to overwrite.

    Optional keyword-only payload: `amplitudes` (the (na, nb) coefficient matrix),
    `orbital_rotation` (cumulative 26x26 orthogonal U; identity for track A),
    `canonical_hcore` / `canonical_eri` (the original A-parent integrals for a
    self-contained B restart), and `details` (track, parent, verified energies,
    flags and timings). The saved hcore/eri/ecore always define the current basis.
    Compress to a temporary file, atomically replace the NPZ, then publish JSON
    last. An exclusive per-filename lock prevents concurrent writers clobbering
    an immutable checkpoint; failed compression leaves no discoverable candidate.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    etag = f"{float(energy):+.6f}".replace("-", "m").replace("+", "p").replace(".", "p")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"cand_E{etag}_{ts}_{_slug(method)}.npz"
    path = results_dir / name
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {path}")
    a = np.asarray(strings_a, dtype=np.int64)
    b = np.asarray(strings_b, dtype=np.int64)
    payload: dict = {"strings_a": a, "strings_b": b,
                     "hcore": np.asarray(hcore), "eri": np.asarray(eri),
                     "ecore": float(ecore), "energy": float(energy)}
    if amplitudes is not None:
        payload["amplitudes"] = np.asarray(amplitudes, dtype=float)
    if orbital_rotation is not None:
        payload["orbital_rotation"] = np.asarray(orbital_rotation, dtype=float)
    if canonical_hcore is not None:
        payload["canonical_hcore"] = np.asarray(canonical_hcore)
    if canonical_eri is not None:
        payload["canonical_eri"] = np.asarray(canonical_eri)
    meta = {
        "method": method, "seed": seed, "timestamp_utc": ts, "rotated_integrals": rotated,
        "dims": [len(a), len(b)], "energy": float(energy), "ecore": float(ecore),
        "python": platform.python_version(), "packages": _package_versions(),
        "details": _jsonable(dict(details)) if details is not None else {},
    }
    lock = path.with_suffix(".lock")
    with lock.open("x", encoding="utf-8"):
        pass
    temporary: Path | None = None
    try:
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"refusing to overwrite existing candidate: {path}")
        with tempfile.NamedTemporaryFile(dir=results_dir, prefix=f".{path.stem}_",
                                         suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
        with path.with_suffix(".json").open("x", encoding="utf-8") as sidecar:
            json.dump(meta, sidecar, indent=2)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()
    return path


def load_candidate(path: Path | str) -> tuple[dict, dict]:
    """Return (arrays dict, metadata dict)."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with np.load(path) as z:
        return dict(z), meta
