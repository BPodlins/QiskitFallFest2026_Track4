"""Small read-only helpers for notebooks/track4.ipynb.

Only file readers and tiny numpy derivations used by the notebook live here.
Nothing here runs an eigensolver or checks energies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path | str) -> list[dict]:
    """Read a .jsonl run log, skipping blank lines."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def accepted_proposals(path: Path | str) -> list[tuple[float, int, float]]:
    """(wall seconds, proposal number, energy) of accepted proposals.

    Prefers the grader energy when stored, else the Ritz energy.
    """
    out = []
    for event in read_jsonl(path):
        if event.get("type") != "proposal" or not event.get("accepted"):
            continue
        energy = event.get("grader", event.get("ritz"))
        if energy is None:
            continue
        out.append((float(event.get("t_wall_s", 0.0) or 0.0),
                    int(event.get("n", 0) or 0), float(energy)))
    return out


def marginals(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised row (alpha) and column (beta) weights of an amplitude matrix."""
    c = np.asarray(coeffs, dtype=float)
    wa = np.sum(c * c, axis=1)
    wb = np.sum(c * c, axis=0)
    s = float(wa.sum())
    if s > 0:
        wa, wb = wa / s, wb / s
    return wa, wb


def orbital_occupations(strings: np.ndarray, weights: np.ndarray,
                        norb: int = 26) -> np.ndarray:
    """Mean occupation per orbital: weight each string's set bits. Pure numpy."""
    s = np.asarray(strings, dtype=np.int64).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    bits = ((s[:, None] >> np.arange(norb, dtype=np.int64)) & 1).astype(float)
    return bits.T @ w
