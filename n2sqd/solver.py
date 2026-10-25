"""Fixed product-space eigensolver used inside the optimisation loop.

Wraps PySCF's `selected_ci.kernel_fixed_space` (the engine behind
`qiskit_addon_sqd.fermion.solve_sci`) with the same cold default guess as the grader, but
returns the Ritz energy instead of solve_sci's RDM-based energy, plus convergence and
optional residual/RDMs. Submitted candidates are checked with `n2sqd.check`
(unmodified `solve_sci`).

No warm starts: measured on an enlarged 81x70 -> 120x70 space, a warm start from the previous
ground state converged to an excited state (-25.48144 vs cold -25.50261) unless kicked with
10 % noise, and gave no speed-up. Every solve is cold, exactly like the grader's.

Energies here are electronic (exclude ecore). Strings: int64, ascending, bit p = orbital p.
Pinned private dependencies: `selected_ci._as_SCIvector`, `_all_linkstr_index` (PySCF 2.14.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.fci import direct_spin1, selected_ci

NELEC = (5, 5)


@dataclass(frozen=True)
class Solution:
    strings_a: np.ndarray  # (na,) int64 ascending
    strings_b: np.ndarray  # (nb,) int64 ascending
    coeffs: np.ndarray  # (na, nb) float64, unit norm
    energy: float  # electronic Ritz energy, excludes ecore
    converged: bool
    residual: float | None  # ||H c - E c|| if computed
    rdm1: np.ndarray | None = None  # spin-summed (norb, norb)
    rdm2: np.ndarray | None = None  # spin-summed, PySCF convention: E = sum h*dm1 + 0.5 sum g*dm2


def _strings(x) -> np.ndarray:
    s = np.asarray(x, dtype=np.int64)
    if s.ndim != 1 or s.size == 0 or np.any(np.diff(s) <= 0):
        raise ValueError("strings must be a non-empty, strictly ascending 1-D int array")
    return s


def remap(sol: Solution, strings_a, strings_b) -> np.ndarray:
    """Coefficients of `sol` on the (strings_a x strings_b) grid, matched by determinant identity.

    New determinants get 0; dropped ones are discarded. Not renormalised.
    """
    a, b = _strings(strings_a), _strings(strings_b)
    out = np.zeros((a.size, b.size))
    ia = np.searchsorted(sol.strings_a, a)
    ib = np.searchsorted(sol.strings_b, b)
    ia_ok = (ia < sol.strings_a.size) & (sol.strings_a[np.minimum(ia, sol.strings_a.size - 1)] == a)
    ib_ok = (ib < sol.strings_b.size) & (sol.strings_b[np.minimum(ib, sol.strings_b.size - 1)] == b)
    out[np.ix_(ia_ok, ib_ok)] = sol.coeffs[np.ix_(ia[ia_ok], ib[ib_ok])]
    return out


def solve_space(strings_a, strings_b, h, g, *, tol: float = 1e-9, max_cycle: int = 100,
                residual: bool = False, rdms: bool = False) -> Solution:
    """Lowest eigenpair of H projected on strings_a x strings_b (no spin constraint).

    Defaults equal the grader's solve_sci call (tol 1e-9, max_cycle 100, cold guess).
    """
    a, b = _strings(strings_a), _strings(strings_b)
    h = np.asarray(h, dtype=float)
    g = np.asarray(g, dtype=float)
    norb = h.shape[0]
    strs = (a, b)
    myci = selected_ci.SelectedCI()
    myci.verbose = 0
    e, vec = selected_ci.kernel_fixed_space(myci, h, g, norb, NELEC, ci_strs=strs,
                                            tol=tol, max_cycle=max_cycle)
    c = np.asarray(vec).reshape(a.size, b.size)
    c = c / np.linalg.norm(c)
    res = None
    if residual:
        h2e = direct_spin1.absorb_h1e(h, g, norb, NELEC, .5)
        link = selected_ci._all_linkstr_index(strs, norb, NELEC)
        hc = np.asarray(myci.contract_2e(h2e, selected_ci._as_SCIvector(c, strs), norb, NELEC, link))
        rq = float(np.vdot(c, hc))
        res = float(np.linalg.norm(hc.reshape(c.shape) - rq * c))
        e = rq
    dm1 = dm2 = None
    if rdms:
        sv = selected_ci._as_SCIvector(c, strs)
        dm1 = np.asarray(myci.make_rdm1(sv, norb, NELEC))
        dm2 = np.asarray(myci.make_rdm2(sv, norb, NELEC))
    return Solution(a, b, c, float(e), bool(getattr(myci, "converged", False)), res, dm1, dm2)
