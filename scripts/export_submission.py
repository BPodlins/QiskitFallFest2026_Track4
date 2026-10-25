"""Export a track-B candidate archive into a self-contained submission directory.

Usage:
    python scripts/export_submission.py <candidate.npz> [--out submission] [--force]

Reads the candidate with ``n2sqd.candidates.load_candidate`` and writes rotation.npy (U),
hcore.npy / eri.npy (the rotated integrals actually submitted), ci_strs_a/b.npy
(int64), canonical_hcore.npy / canonical_eri.npy (the starting integrals U was
applied to), plus manifest.json with provenance and sha256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from n2sqd.candidates import load_candidate

CONVENTION = (
    "hcore = U.T @ canonical_hcore @ U; "
    "eri[p,q,r,s] = sum U[a,p]U[b,q]U[c,r]U[d,s] canonical_eri[a,b,c,d]; "
    "same U for both spins; core orbitals and ecore unchanged"
)

PAYLOAD = (
    ("orbital_rotation", "rotation.npy"),
    ("hcore", "hcore.npy"),
    ("eri", "eri.npy"),
    ("strings_a", "ci_strs_a.npy"),
    ("strings_b", "ci_strs_b.npy"),
    ("canonical_hcore", "canonical_hcore.npy"),
    ("canonical_eri", "canonical_eri.npy"),
)

FILE_NAMES = [name for _, name in PAYLOAD] + ["manifest.json"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source_name: str, energy: float, ecore: float,
                   dims: tuple[int, int], digests: dict, provenance: dict) -> dict:
    return {
        "source": source_name,
        "energy": float(energy),
        "ecore": repr(float(ecore)),
        "dims": [int(dims[0]), int(dims[1])],
        "convention": CONVENTION,
        "sha256": dict(digests),
        "provenance": dict(provenance),
    }


def export_candidate(candidate_path: Path | str, out_dir: Path | str,
                     *, force: bool = False) -> tuple[Path, dict]:
    """Stage and install the submission. Returns (out_dir, manifest).

    Raises ValueError for non-rotated candidates and FileExistsError when out_dir
    already holds a different submission and force is False.
    """
    candidate_path = Path(candidate_path)
    arrays, meta = load_candidate(candidate_path)
    if not meta.get("rotated_integrals", False) or any(
            key not in arrays for key, _ in PAYLOAD):
        raise ValueError(f"refusing non-rotated candidate: {candidate_path}")
    data = {
        "orbital_rotation": np.asarray(arrays["orbital_rotation"], dtype=float),
        "hcore": np.asarray(arrays["hcore"], dtype=float),
        "eri": np.asarray(arrays["eri"], dtype=float),
        "strings_a": np.asarray(arrays["strings_a"], dtype=np.int64),
        "strings_b": np.asarray(arrays["strings_b"], dtype=np.int64),
        "canonical_hcore": np.asarray(arrays["canonical_hcore"], dtype=float),
        "canonical_eri": np.asarray(arrays["canonical_eri"], dtype=float),
    }
    norb = data["canonical_hcore"].shape[0]
    if (data["orbital_rotation"].shape != (norb, norb)
            or data["hcore"].shape != (norb, norb)
            or data["eri"].shape != (norb,) * 4
            or data["canonical_hcore"].shape != (norb, norb)
            or data["canonical_eri"].shape != (norb,) * 4):
        raise ValueError(
            f"inconsistent integral shapes in {candidate_path}: "
            f"U{data['orbital_rotation'].shape} hcore{data['hcore'].shape} "
            f"eri{data['eri'].shape}")
    details = meta.get("details", {}) or {}
    provenance = {key: details.get(key)
                  for key in ("run_id", "parent", "canonical_parent")}
    for key in ("script", "args"):
        if key in details:
            provenance[key] = details[key]
        elif key in meta:
            provenance[key] = meta[key]

    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        digests = {}
        for key, name in PAYLOAD:
            np.save(staged / name, data[key])
            digests[name] = _sha256(staged / name)
        manifest = build_manifest(
            candidate_path.name, float(arrays["energy"]), float(arrays["ecore"]),
            (len(data["strings_a"]), len(data["strings_b"])), digests, provenance)
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists() and any(out_dir.iterdir()) and not force:
            existing = out_dir / "manifest.json"
            identical = False
            if existing.exists():
                try:
                    identical = json.loads(existing.read_text(encoding="utf-8")) == manifest
                except ValueError:
                    identical = False
            if not identical:
                raise FileExistsError(
                    f"refusing to overwrite existing different submission in {out_dir} "
                    "(use --force)")
            print(f"{out_dir}: identical submission already present, left in place.")
            return out_dir, manifest
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in FILE_NAMES:
            shutil.move(str(staged / name), str(out_dir / name))
    print(f"wrote {out_dir} from {candidate_path.name} "
          f"(dims {manifest['dims'][0]} x {manifest['dims'][1]}, "
          f"energy {manifest['energy']:.6f} Ha).")
    return out_dir, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", help="results/*.npz track-B candidate")
    parser.add_argument("--out", default="submission", help="output directory")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing different submission")
    args = parser.parse_args(argv)
    try:
        export_candidate(args.candidate, args.out, force=args.force)
    except (ValueError, FileExistsError, OSError) as err:
        print(f"export_submission: error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
