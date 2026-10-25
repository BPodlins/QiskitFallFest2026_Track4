"""Thin CLI for track A/B search. No algorithm here; see n2sqd/optimise.py."""

from __future__ import annotations

import argparse
import os
import sys


def _pre_parse_threads(argv: list[str]) -> str:
    for i, tok in enumerate(argv):
        if tok == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--threads="):
            return tok.split("=", 1)[1]
    return "8"


# OMP/OPENBLAS/MKL thread counts are fixed at import time: set before numpy import.
# Assigned, not setdefault: the image's ENV sets 16, which must not win over --threads.
_threads = _pre_parse_threads(sys.argv[1:])
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = _threads

from n2sqd.optimise import run_search  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run track A/B capped-space search.")
    p.add_argument("--track", required=True, choices=["A", "B"],
                   help="A = canonical integrals (never rotated); B = rotated orbitals.")
    p.add_argument("--hours", required=True, type=float,
                   help="Wall-clock budget; no solve starts after it, in-flight solves finish.")
    p.add_argument("--seed", default=2026, type=int)
    p.add_argument("--start", default=None,
                   help="Candidate .npz from any script; resume its saved Hamiltonian "
                        "and verify on load if verification/amplitudes are missing.")
    p.add_argument("--independent", action="store_true",
                   help="Independent alpha/beta sets (default: symmetric a=b runs).")
    p.add_argument("--threads", default=8, type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = run_search(args.track, seed=args.seed, hours=args.hours,
                        start=args.start, symmetric=not args.independent,
                        threads=args.threads)
    print(f"run record: {record}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
