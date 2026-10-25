"""Local replica of the grader's exbonus checks (no network, never submits).

Mirrors `qfest_grader.challenges.warsaw_2026.lab4b._exbonus_answer`:
CI-string validity (same rules/messages), integral spectra check vs the
reference Hamiltonian (atol 1e-6, core 1e-6), recomputed energy via
`qiskit_addon_sqd.fermion.solve_sci` (no spin constraint) + core, and
`matches` (claimed >= recomputed - 1e-4).
"""

from __future__ import annotations

import time

import numpy as np
from qfest_grader.answers import AnswerError
from qiskit_addon_sqd.fermion import solve_sci

from n2sqd.hamiltonian import build_n2

MAX_SUBSPACE = 300
ENERGY_TOLERANCE = 1e-4
INTEGRAL_TOLERANCE = 1e-6


def _ci_strings(values, what: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise AnswerError(f"{what} must be a 1-D integer array of CI strings.")
    return array.astype(np.int64)


def spectra(one_body: np.ndarray, two_body: np.ndarray, n: int) -> list[np.ndarray]:
    """Same three spectra the grader compares: hcore, eri as n^2 x n^2, and their sum combo."""
    eye, pairs = np.eye(n), two_body.reshape(n * n, n * n)
    matrices = (one_body, pairs, pairs + np.kron(one_body, eye) + np.kron(eye, one_body))
    return [np.linalg.eigvalsh((m + m.T) / 2) for m in matrices]

def rotate_integrals(hcore: np.ndarray, eri: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate active orbitals by a real orthogonal u: h' = u^T h u, eri' on all four indices.

    Sequential one-index contractions (each O(n^5)); a single 4-index einsum is O(n^8) and hangs at n=26.
    """
    u = np.asarray(u, dtype=float)
    g = np.asarray(eri, dtype=float)
    h2 = u.T @ np.asarray(hcore, dtype=float) @ u
    g2 = np.einsum("ip,ijkl->pjkl", u, g)
    g2 = np.einsum("jq,pjkl->pqkl", u, g2)
    g2 = np.einsum("kr,pqkl->pqrl", u, g2)
    g2 = np.einsum("ls,pqrl->pqrs", u, g2)
    return h2, g2



def random_rotation(n: int, seed: int) -> np.ndarray:
    """Deterministic real orthogonal matrix (QR of a seeded Gaussian)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def check_candidate(strings_a, strings_b, hcore, eri, ecore, claimed_energy,
                    *, deadline: float | None = None) -> dict:
    """Validate like the grader; do not start a solve after an optional epoch deadline."""
    t_total = time.perf_counter()
    try:
        claimed = float(claimed_energy)
    except (TypeError, ValueError):
        raise AnswerError("Submit best_energy as a number and best_subspace as (ci_strs_a, ci_strs_b).") from None
    a = _ci_strings(strings_a, "best_subspace[0]")
    b = _ci_strings(strings_b, "best_subspace[1]")
    dims = [len(a), len(b)]
    out: dict = {"dims": dims, "claimed": claimed, "energy": None, "matches": False}
    if min(dims) == 0 or max(dims) > MAX_SUBSPACE:
        out["timings"] = {"total_s": time.perf_counter() - t_total}
        return out

    ref_h, ref_g, ref_core, n, nelec = build_n2()
    try:
        h = np.asarray(hcore, dtype=float)
        g = np.asarray(eri, dtype=float)
        core = float(ecore)
    except (TypeError, ValueError):
        h = g = None  # type: ignore[assignment]
    if (h is None or h.shape != (n, n) or g.shape != (n,) * 4 or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(g))):
        raise AnswerError(f"Pass hcore ({n} x {n}), eri ({n} x {n} x {n} x {n}) and nuclear_repulsion_energy from the "
                          "cell that builds the N2 Hamiltonian.")
    t_spec = time.perf_counter()
    spec_ok = (
        abs(core - ref_core) <= INTEGRAL_TOLERANCE
        and all(
            np.allclose(mine, ref, rtol=0, atol=INTEGRAL_TOLERANCE)
            for mine, ref in zip(spectra(h, g, n), spectra(ref_h, ref_g, n))
        )
    )
    spectra_s = time.perf_counter() - t_spec
    if not spec_ok:
        raise AnswerError("hcore, eri and nuclear_repulsion_energy are not the N2 integrals of this lab. Pass the ones "
                          "from the cell that builds the N2 Hamiltonian.")
    for i, strings in enumerate((a, b)):
        if np.any(np.diff(strings) <= 0):
            raise AnswerError(f"best_subspace[{i}] must hold distinct CI strings in ascending order (np.unique).")
        if np.any(strings < 0) or np.any(strings >= 1 << n) or any(
                bin(int(s)).count("1") != nelec[i] for s in strings):
            raise AnswerError(f"Every CI string in best_subspace[{i}] must have {nelec[i]} of the {n} "
                              "orbitals occupied.")
    t_sci = time.perf_counter()
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("verification deadline reached before solve_sci")
    energy = solve_sci((a, b), h, g, n, nelec).energy + core
    solve_s = time.perf_counter() - t_sci
    out["energy"] = float(energy)
    out["matches"] = claimed >= float(energy) - ENERGY_TOLERANCE
    out["timings"] = {"spectra_s": spectra_s, "solve_sci_s": solve_s,
                      "total_s": time.perf_counter() - t_total}
    return out


def official_facts(strings_a, strings_b, hcore, eri, ecore, claimed_energy,
                   *, deadline: float | None = None) -> dict:
    """Run exactly one local check and one grader reduce (no network).

    Return the official rounded facts plus ``check``, the precise local energy
    and timings. Callers must reuse this result, not cold-solve a third time.
    """
    from qfest_grader.challenges.warsaw_2026 import lab4b as _lab4b

    mine = check_candidate(strings_a, strings_b, hcore, eri, ecore, claimed_energy,
                           deadline=deadline)
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("verification deadline reached before grader reduce")
    facts = _lab4b._exbonus_answer(
        (float(claimed_energy), (np.asarray(strings_a), np.asarray(strings_b)),
         np.asarray(hcore, dtype=float), np.asarray(eri, dtype=float), float(ecore)))
    assert facts["dims"] == mine["dims"], (facts["dims"], mine["dims"])
    if mine["energy"] is None:
        assert facts["energy"] is None and facts["matches"] is False, facts
    else:
        assert facts["energy"] == round(mine["energy"], 6), (facts["energy"], mine["energy"])
        assert facts["matches"] == mine["matches"], (facts, mine)
    return {**facts, "check": mine}
