"""Real orbital optimisation for the capped product-space search (track B).

Two proposal branches:

- ``method='ffsim'``: fixed-RDM gradient optimisation over the 325 real generators
  via ``ffsim.optimize_orbitals`` (L-BFGS-B from identity).
- ``method='natural'``: eigenvectors of the symmetrised spin-summed 1-RDM.

``reseed_rotated`` re-expresses a capped state in the rotated basis with determinant
overlap minors. For old occupied set ``I`` and new occupied set ``J`` (both ascending,
creation operators in ascending orbital order), the rotated determinant expands as
``|I> = sum_J T[J, I] |J>`` with ``T[J, I] = det(u[np.ix_(I, J)])`` (rows = old
occupied orbitals, columns = new occupied orbitals).
This is a passive basis change, not a forward orbital rotation of the state:
the complete-space norm and energy with the rotated integrals must be invariant.

ffsim convention (verified numerically on a 60x60 N2 solution: ``|E_ours(u) - fun|`` is
1.4e-14 for ``u = R`` versus 0.13 Ha for ``u = R.T``): ``ffsim.optimize_orbitals``
returns ``R`` and our local rotation is ``u = R`` (not ``R.T``), i.e. new integrals
``check.rotate_integrals(h, g, u)`` give ``sum h'*dm1 + 0.5*sum g'*dm2 == res.fun``.
The fixed-RDM loss rotates the RDMs by ``R``; rotating RDMs by ``R`` equals rotating
the Hamiltonian by ``R.T``, which is exactly our ``u`` convention (``h' = u.T h u``).
``propose_rotation`` re-checks this identity at runtime and refuses to return ``u``
if it ever drifts (tolerance 1e-6).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import ffsim
import numpy as np

from n2sqd.check import rotate_integrals
from n2sqd.solver import Solution

_RDM_TOL = 1e-8  # RDM energy must reproduce Solution.energy (controller skips otherwise)
_ORTH_TOL = 1e-10  # required orthogonality of a proposal u
_RESEED_ORTH_TOL = 1e-8  # required orthogonality of the u handed to reseed_rotated
_CONVENTION_TOL = 1e-6  # u=R convention lock: |E_ours(u) - res.fun|
_DEG_TOL = 1e-8  # occupation gap below which natural orbitals are "degenerate"
_BLOCK = 256  # new-string block size for overlap minors
_NELEC = 5  # electrons per spin


@dataclass(frozen=True)
class RotationProposal:
    u: np.ndarray  # (norb, norb) real orthogonal; new integrals = rotate_integrals(h, g, u)
    method: str  # 'ffsim' | 'natural'
    objective_before: float  # fixed-RDM energy (electronic) at identity
    objective_after: float  # fixed-RDM energy after rotation
    info: dict


def _fixed_rdm_energy(dm1: np.ndarray, dm2: np.ndarray, h: np.ndarray, g: np.ndarray) -> float:
    """Electronic fixed-RDM energy: sum h*dm1 + 0.5*sum g*dm2 (spin-summed convention)."""
    return float(
        np.einsum("pq,pq->", h, dm1) + 0.5 * np.einsum("pqrs,pqrs->", g, dm2)
    )


def _check_u(u: np.ndarray, norb: int, *, tol: float) -> np.ndarray:
    u = np.asarray(u)
    if not np.isrealobj(u):
        raise ValueError("orbital rotation must be real")
    if u.shape != (norb, norb) or not np.all(np.isfinite(u)):
        raise ValueError(f"orbital rotation must be a finite ({norb}, {norb}) array")
    err = float(np.max(np.abs(u.T @ u - np.eye(norb))))
    if not np.isfinite(err) or err > tol:
        raise ValueError(f"orbital rotation not orthogonal: max|u.T@u - I| = {err:.3e} (> {tol:.0e})")
    return np.asarray(u, dtype=float)


def _natural_orbitals(dm1: np.ndarray) -> tuple[np.ndarray, list[float], list[list[int]]]:
    """Natural orbitals: descending occupations, deterministic signs, Procrustes-aligned blocks."""
    d = 0.5 * (np.asarray(dm1, dtype=float) + np.asarray(dm1, dtype=float).T)
    norb = d.shape[0]
    vals, vecs = np.linalg.eigh(d)
    order = np.argsort(-vals, kind="stable")
    occ = vals[order]
    u = np.ascontiguousarray(vecs[:, order])

    def fix_signs(m: np.ndarray) -> None:
        for k in range(m.shape[1]):
            col = m[:, k]
            j = int(np.argmax(np.abs(col)))
            if col[j] < 0:
                m[:, k] = -col

    fix_signs(u)
    blocks: list[list[int]] = []
    start = 0
    for k in range(1, norb + 1):
        if k == norb or occ[start] - occ[k] > _DEG_TOL:
            blocks.append(list(range(start, k)))
            start = k
    for block in blocks:
        if len(block) < 2:
            continue
        target = np.eye(norb)[:, block]  # align degenerate block to the identity columns
        q, _, qt = np.linalg.svd(u[:, block].T @ target)
        u[:, block] = u[:, block] @ (q @ qt)
    fix_signs(u)
    return u, [float(v) for v in occ], [b for b in blocks if len(b) > 1]


def propose_rotation(sol: Solution, h, g, *, method: str = "ffsim", maxiter: int = 30) -> RotationProposal:
    """Propose a real orthogonal orbital rotation lowering the fixed-RDM energy.

    Raises a clear ``ValueError`` when the solution carries no RDMs or when the RDM
    energy does not reproduce ``sol.energy`` within 1e-8: the controller skips the
    rotation step in both cases.
    """
    if method not in ("ffsim", "natural"):
        raise ValueError(f"method must be 'ffsim' or 'natural', got {method!r}")
    h = np.asarray(h, dtype=float)
    g = np.asarray(g, dtype=float)
    norb = h.shape[0]
    if h.shape != (norb, norb) or g.shape != (norb,) * 4:
        raise ValueError("h/g shapes inconsistent")
    if sol.rdm1 is None or sol.rdm2 is None:
        raise ValueError(
            "propose_rotation needs RDMs on the Solution (re-solve with rdms=True); "
            "controller: skip this rotation step"
        )
    dm1 = np.asarray(sol.rdm1, dtype=float)
    dm2 = np.asarray(sol.rdm2, dtype=float)
    if dm1.shape != (norb, norb) or dm2.shape != (norb,) * 4:
        raise ValueError("RDM shapes inconsistent with h/g")
    before = _fixed_rdm_energy(dm1, dm2, h, g)
    if abs(before - sol.energy) > _RDM_TOL:
        raise ValueError(
            f"RDM energy {before:.9f} differs from Solution.energy {sol.energy:.9f} by "
            f"{abs(before - sol.energy):.2e} Ha (> {_RDM_TOL:.0e}); refusing rotation "
            "(controller: skip this rotation step)"
        )

    if method == "ffsim":
        rdm = ffsim.ReducedDensityMatrix(dm1, dm2)
        ham = ffsim.MolecularHamiltonian(h, g, constant=0.0)
        rot, res = ffsim.optimize_orbitals(
            rdm,
            ham,
            real=True,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter)},
            return_optimize_result=True,
        )
        if np.iscomplexobj(rot):
            raise ValueError("ffsim returned a complex rotation")
        u = np.asarray(rot, dtype=float)
        after = float(res.fun)
        info = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "fun": float(res.fun),
            "maxiter": int(maxiter),
        }
    else:
        u, occ, blocks = _natural_orbitals(dm1)
        hr, gr = rotate_integrals(h, g, u)
        after = _fixed_rdm_energy(dm1, dm2, hr, gr)
        info = {"occupations": occ, "degenerate_blocks": blocks, "deg_tol": _DEG_TOL}

    u = _check_u(u, norb, tol=_ORTH_TOL)
    hr, gr = rotate_integrals(h, g, u)
    check = _fixed_rdm_energy(dm1, dm2, hr, gr)
    info["convention_diff"] = abs(check - after)
    if abs(check - after) > _CONVENTION_TOL:
        raise RuntimeError(
            f"rotation convention broken: our E(u)={check:.9f} vs reported {after:.9f}; "
            "expected u=R, refusing proposal"
        )
    return RotationProposal(
        u=u,
        method=method,
        objective_before=float(before),
        objective_after=float(after),
        info=info,
    )


def _occ_matrix(strings, norb: int) -> np.ndarray:
    """Occupied-orbital lists (ascending) for each string: (n, 5) int array."""
    s = np.asarray(strings, dtype=np.int64).reshape(-1)
    if np.any(np.bitwise_count(s) != _NELEC):
        raise ValueError("every string must have exactly 5 orbitals occupied")
    bits = ((s[:, None] >> np.arange(norb, dtype=np.int64)) & np.int64(1)).astype(np.int64)
    return np.argsort(-bits, axis=1, kind="stable")[:, :_NELEC]


def _all_strings(norb: int) -> np.ndarray:
    """All C(norb, 5) string integers, ascending."""
    out = np.fromiter(
        (sum(1 << p for p in c) for c in itertools.combinations(range(norb), _NELEC)),
        dtype=np.int64,
        count=math.comb(norb, _NELEC),
    )
    out.sort()
    return out


def _overlap_minors(u: np.ndarray, old_occ: np.ndarray, new_occ: np.ndarray) -> np.ndarray:
    """Determinant overlap minors ``T[J, i] = det(u[np.ix_(I_i, J)])``: (n_new, n_old).

    ``old_occ``/``new_occ`` are ascending occupied-orbital lists as built by
    ``_occ_matrix``; ascending order encodes the fermionic sign convention
    (creation operators in ascending orbital order).
    """
    old = np.asarray(old_occ, dtype=np.int64)
    new = np.asarray(new_occ, dtype=np.int64)
    sub = u[old[None, :, :, None], new[:, None, None, :]]
    return np.linalg.det(sub)


def reseed_rotated(
    sol: Solution, u: np.ndarray, *, cap: int = 300, symmetric: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-express a capped state in the rotated basis and keep the top ``cap`` strings.

    ``u`` is the single-step rotation (current basis -> new basis), e.g. from
    ``propose_rotation``. New strings run over all C(norb, 5); minors are computed in
    blocks of 256 new strings so no 65780^2 matrix is ever formed. Exact transformed
    marginals ``wa[J] = sum_k (T_a C)[J, k]^2`` (beta with ``C.T``) drive selection:
    combined ``(wa + wb) / 2`` with a common ``a == b`` set for ``symmetric=True``,
    independent top-``cap`` sets otherwise. Returns ascending int64 strings plus the
    retained norm ``||P (T_a C T_b.T)||^2`` restricted to the chosen pairs.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("u must be a square 2-D array")
    norb = u.shape[0]
    u = _check_u(u, norb, tol=_RESEED_ORTH_TOL)
    c = np.asarray(sol.coeffs, dtype=float)
    a_old = np.asarray(sol.strings_a, dtype=np.int64)
    b_old = np.asarray(sol.strings_b, dtype=np.int64)
    if c.shape != (a_old.size, b_old.size):
        raise ValueError("coeff shape inconsistent with strings")
    n_new = math.comb(norb, _NELEC)
    cap = int(cap)
    if cap < 1:
        raise ValueError("cap must be >= 1")
    cap = min(cap, n_new)
    new_strings = _all_strings(norb)
    new_occ = _occ_matrix(new_strings, norb)
    occ_a = _occ_matrix(a_old, norb)
    occ_b = _occ_matrix(b_old, norb)
    same_old = a_old.size == b_old.size and bool(np.array_equal(a_old, b_old))

    wa = np.zeros(n_new)
    wb = np.zeros(n_new)
    for start in range(0, n_new, _BLOCK):
        sl = slice(start, min(start + _BLOCK, n_new))
        ta = _overlap_minors(u, occ_a, new_occ[sl])
        ma = ta @ c
        wa[sl] = np.einsum("Jk,Jk->J", ma, ma)
        tb = ta if same_old else _overlap_minors(u, occ_b, new_occ[sl])
        mb = tb @ c.T
        wb[sl] = np.einsum("Jk,Jk->J", mb, mb)

    if symmetric:
        take = np.argsort(-(wa + wb) / 2, kind="stable")[:cap]
        take.sort()
        a = b = new_strings[take]
        ta_c = _overlap_minors(u, occ_a, new_occ[take])
        tb_c = ta_c if same_old else _overlap_minors(u, occ_b, new_occ[take])
    else:
        ia = np.argsort(-wa, kind="stable")[:cap]
        ia.sort()
        ib = np.argsort(-wb, kind="stable")[:cap]
        ib.sort()
        a, b = new_strings[ia], new_strings[ib]
        ta_c = _overlap_minors(u, occ_a, new_occ[ia])
        tb_c = _overlap_minors(u, occ_b, new_occ[ib])
    proj = (ta_c @ c) @ tb_c.T
    retained = float(np.einsum("Jj,Jj->", proj, proj))
    return np.ascontiguousarray(a), np.ascontiguousarray(b), retained
