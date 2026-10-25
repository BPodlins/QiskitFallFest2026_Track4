"""Capped product-space string selection.

Enumerate singles/doubles, shortlist at most 4096 outsiders per spin, then rank
their exact coherent external residuals with a 0.05 Ha denominator floor.
Growth mixes 80% residual and 20% screening picks. Cold enlarged/pruned solves
and marginal (default) or energy-aware pruning retain only converged,
lower-energy capped states.
Swaps reuse the residual ranking, with at most two bounded passes.

Bit p means orbital p; nelec=(5,5); energies exclude ecore. All output string
arrays are ascending unique int64. Symmetric searches keep identical spin sets,
sum their residual scores and use averaged marginals or exact joint-removal loss.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from n2sqd.solver import Solution, solve_space

NELEC = (5, 5)
_DENOM_FLOOR = 0.05
_PLATEAU_TOL = 2e-5
_PLATEAU_PATIENCE = 3
_SCREEN_MAIN = 3840
_DIAG_EXTRA = 256


# ---------------------------------------------------------------------------
# String <-> occupation helpers (vectorised, numpy only)
# ---------------------------------------------------------------------------

def _as_sorted(x) -> np.ndarray:
    s = np.asarray(x, dtype=np.int64)
    if s.ndim != 1 or s.size == 0 or np.any(np.diff(s) <= 0):
        raise ValueError("strings must be a non-empty, strictly ascending 1-D int array")
    return s


def _check_popcount(s: np.ndarray) -> None:
    if bool((np.bitwise_count(np.asarray(s, dtype=np.int64)) != NELEC[0]).any()):
        raise ValueError("all strings must hold exactly 5 electrons")


def _occ_list(s: np.ndarray, norb: int) -> np.ndarray:
    """Occupation lists, shape (len(s), 5), ascending orbital indices."""
    s = np.asarray(s, dtype=np.int64)
    _check_popcount(s)
    bits = ((s[:, None] >> np.arange(norb, dtype=np.int64)) & 1).astype(bool)
    if bool((bits.sum(axis=1) != NELEC[0]).any()):
        raise ValueError("string has bits outside the orbital range")
    cs = np.cumsum(bits, axis=1)
    return np.stack([np.argmax(cs == k + 1, axis=1) for k in range(NELEC[0])], axis=1)


# ---------------------------------------------------------------------------
# Diagonal H_xx (matches SelectedCI.make_hdiag, a-major order)
# ---------------------------------------------------------------------------

def diag_block(a: np.ndarray, b: np.ndarray, h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Full diagonal of the product-space Hamiltonian, shape (len(a), len(b)).

    ``diag_block(a, b).ravel()`` equals ``SelectedCI.make_hdiag`` (a-major):
    same-spin one-body, same-spin two-body (J-K)/2, opposite-spin direct J.
    """
    a = _as_sorted(a)
    b = _as_sorted(b)
    h = np.asarray(h, dtype=float)
    g = np.asarray(g, dtype=float)
    norb = h.shape[0]
    oa = _occ_list(a, norb)
    ob = _occ_list(b, norb)
    ii = np.arange(norb)
    jmat = g[ii[:, None], ii[:, None], ii[None, :], ii[None, :]]  # (mm|nn)
    kmat = g[ii[:, None], ii[None, :], ii[None, :], ii[:, None]]  # (mn|nm)
    ia = np.zeros((a.size, norb))
    ia[np.arange(a.size)[:, None], oa] = 1.0
    ib = np.zeros((b.size, norb))
    ib[np.arange(b.size)[:, None], ob] = 1.0
    ea = ia @ np.diag(h) + 0.5 * (ia * (ia @ (jmat - kmat))).sum(axis=1)
    eb = ib @ np.diag(h) + 0.5 * (ib * (ib @ (jmat - kmat))).sum(axis=1)
    return ea[:, None] + eb[None, :] + ia @ jmat @ ib.T


# ---------------------------------------------------------------------------
# Neighbour tables: singles + doubles with exact same-spin couplings
# ---------------------------------------------------------------------------

class _Neighbours:
    """All single/double excitations of `cur` with exact same-spin <x|H|i>.

    Singles (nsrc*105 entries): xs, src_s, vs, ps (created), qs (annihilated).
    Doubles (nsrc*2100 entries): xd, sd, vd. Couplings are exact Slater-Condon
    (Fock-dressed singles, antisymmetrised ERIs for doubles) incl. fermionic
    signs ``(-1)^(#occ strictly between)`` (verified against PySCF links).
    Flat order is (source, replaced-occ..., virtual...): entry arrays align.
    """

    def __init__(self, cur: np.ndarray, h: np.ndarray, g: np.ndarray, norb: int):
        cur = _as_sorted(cur)
        occ = _occ_list(cur, norb)                       # (nsrc, 5)
        nsrc = cur.size
        is_occ = np.zeros((nsrc, norb), dtype=bool)
        is_occ[np.arange(nsrc)[:, None], occ] = True
        vir = np.argsort(is_occ.astype(np.int64), axis=1, kind="stable")[:, :norb - 5]
        nvir = vir.shape[1]
        bits = np.int64(1) << np.arange(norb, dtype=np.int64)
        self.cur = cur

        # -- singles: replace occupied position k with virtual iv --
        mask = bits[occ[:, :, None]] | bits[vir[:, None, :]]       # (nsrc,5,nvir)
        self.xs = (cur[:, None, None] ^ mask).reshape(-1)
        between = ((occ[:, None, None, :] > np.minimum(occ[:, :, None], vir[:, None, :])[:, :, :, None])
                   & (occ[:, None, None, :] < np.maximum(occ[:, :, None], vir[:, None, :])[:, :, :, None])
                   ).sum(axis=-1)                                   # (nsrc,5,nvir)
        sign = np.where(between % 2 == 0, 1.0, -1.0)
        self.sign_s = sign.reshape(-1)
        # F[p,q] = h[p,q] + sum_r ((pq|rr) - (pr|rq)); r == q cancels.
        a4 = vir[:, :, None, None]          # (nsrc,nvir,1,1)
        n4 = occ[:, None, None, :]          # (nsrc,1,1,5)
        q4 = occ[:, None, :, None]          # (nsrc,1,5,1)
        fmat = g[a4, q4, n4, n4] - g[a4, n4, n4, q4]
        fock = (h[vir[:, :, None], occ[:, None, :]]
                + fmat.sum(axis=-1)).transpose(0, 2, 1)  # (nsrc,5,nvir)
        self.vs = (sign * fock).reshape(-1)
        self.src_s = np.repeat(np.arange(nsrc), 5 * nvir)
        self.ps = np.tile(vir, (1, 5)).reshape(-1)          # created orbital
        self.qs = np.repeat(occ, nvir, axis=1).reshape(-1)  # annihilated orbital

        # -- doubles: replace occupied pair q with virtual pair v --
        o1, o2 = np.triu_indices(5, 1)
        k1, k2 = np.triu_indices(nvir, 1)
        bits_occ = bits[occ]                             # (nsrc,5)
        bits_vir = bits[vir]                             # (nsrc,nvir)
        npp, nvp = o1.size, k1.size
        xd = np.zeros((nsrc, npp, nvp), dtype=np.int64)
        vd = np.zeros((nsrc, npp, nvp))
        for s_idx in range(nsrc):
            oi, vv = occ[s_idx], vir[s_idx]
            a1v, a2v = vv[k1], vv[k2]                    # (nvp,)
            i1v, i2v = oi[o1], oi[o2]                    # (npp,)
            c0 = (g[a1v[None, :], i1v[:, None], a2v[None, :], i2v[:, None]]
                  - g[a1v[None, :], i2v[:, None], a2v[None, :], i1v[:, None]])
            cnt = between[s_idx]                        # (5,nvir)
            c1 = cnt[o1[:, None], k1[None, :]]           # parity of E[a1,i1] on occ
            base = cnt[o2[:, None], k2[None, :]]
            # Parity of E[a2,i2] on the mid set (occ - {i1} + {a1}).
            lo = np.minimum(i2v[:, None], a2v[None, :])
            hi = np.maximum(i2v[:, None], a2v[None, :])
            c2 = (base - ((i1v[:, None] > lo) & (i1v[:, None] < hi)).astype(float)
                  + ((a1v[None, :] > lo) & (a1v[None, :] < hi)).astype(float))
            vd[s_idx] = np.where((c1 + c2) % 2 == 0, 1.0, -1.0) * c0
            xd[s_idx] = (cur[s_idx] ^ (bits_occ[s_idx][o1][:, None]
                                       | bits_occ[s_idx][o2][:, None]
                                       | bits_vir[s_idx][k1][None, :]
                                       | bits_vir[s_idx][k2][None, :]))
        self.xd = xd.reshape(-1)
        self.sd = np.repeat(np.arange(nsrc), npp * nvp)
        self.vd = vd.reshape(-1)


# ---------------------------------------------------------------------------
# Screening + residual + scoring
# ---------------------------------------------------------------------------

def _screen(tab: _Neighbours, w: np.ndarray, fro: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Cheap bounds per outsider: sum_i bound(x,i)^2 w_i (design 2.3).

    Singles add ``5*||g[p,q]||_fro`` to ``|Ha|``; doubles use ``|Ha|``.
    Returns (candidates, scores), sorted, with current strings removed.
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    x_all = np.concatenate([tab.xs, tab.xd])
    v_all = np.concatenate([np.abs(tab.vs) + 5.0 * fro[tab.ps, tab.qs],
                            np.abs(tab.vd)])
    s_all = np.concatenate([tab.src_s, tab.sd])
    xu, inv = np.unique(x_all, return_inverse=True)
    scr = np.bincount(inv, weights=(v_all ** 2) * w[s_all], minlength=xu.size)
    keep = ~np.isin(xu, tab.cur)
    return xu[keep], scr[keep]


def _single_links_of_set(cur: np.ndarray, norb: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """Number-operator and single-excitation links internal to a string set.

    Returns (J src col, B dest row, R created, S annihilated, G sign) with
    ``E[R,S] |J> = G |B>``.
    """
    cur = _as_sorted(cur)
    occ = _occ_list(cur, norb)
    is_occ = np.zeros((cur.size, norb), dtype=bool)
    is_occ[np.arange(cur.size)[:, None], occ] = True
    vir = np.argsort(is_occ.astype(np.int64), axis=1, kind="stable")[:, :norb - 5]
    nvir = vir.shape[1]
    bits = np.int64(1) << np.arange(norb, dtype=np.int64)
    dest = (cur[:, None, None]
            ^ (bits[occ[:, :, None]] | bits[vir[:, None, :]]))     # (nsrc,5,nvir)
    between = ((occ[:, None, None, :] > np.minimum(occ[:, :, None], vir[:, None, :])[:, :, :, None])
               & (occ[:, None, None, :] < np.maximum(occ[:, :, None], vir[:, None, :])[:, :, :, None])
               ).sum(axis=-1)
    sgn = np.where(between % 2 == 0, 1.0, -1.0)
    pos = {int(s): r for r, s in enumerate(cur.tolist())}
    hit = np.array([pos.get(int(v), -1) for v in dest.reshape(-1).tolist()])
    ok = hit >= 0
    n_each = 5 * nvir
    # Number-operator links E[r,r] are essential when the other spin stays put.
    diagonal_src = np.repeat(np.arange(cur.size), 5)
    diagonal_orb = occ.reshape(-1)
    return (np.concatenate([np.repeat(np.arange(cur.size), n_each)[ok], diagonal_src]),
            np.concatenate([hit[ok], diagonal_src]),
            np.concatenate([np.tile(vir, (1, 5)).reshape(-1)[ok], diagonal_orb]),
            np.concatenate([np.repeat(occ, nvir, axis=1).reshape(-1)[ok], diagonal_orb]),
            np.concatenate([sgn.reshape(-1)[ok], np.ones(diagonal_src.size)]))


def _external_residual(short: np.ndarray, cmat: np.ndarray, tab: _Neighbours,
                       blink: tuple[np.ndarray, ...], g: np.ndarray) -> np.ndarray:
    """Coherent residual rows for outsiders, including all other-spin links.

    R[x,b] = Ha_ext[x,:] C + sum_{pq} Ea_ext[x,:;pq] (C Vb[pq].T).
    Number-operator and signed off-diagonal links both contribute to Vb.
    """
    short = _as_sorted(short)
    norb = g.shape[0]
    nshort, noth = short.size, cmat.shape[1]
    ha = np.zeros((nshort, tab.cur.size))
    # Same-spin part: table entries landing in short.
    for xa, sa, va in ((tab.xs, tab.src_s, tab.vs), (tab.xd, tab.sd, tab.vd)):
        idx = np.searchsorted(short, xa)
        idx = np.minimum(idx, nshort - 1)
        hit = short[idx] == xa
        if bool(hit.any()):
            np.add.at(ha, (idx[hit], sa[hit]), va[hit])
    # Form only the small external one-spin block, never a
    # (number_of_excitations, number_of_other_strings) scatter temporary.
    rmat = ha @ cmat
    # Cross-spin part: outsiders x = E[p,q] i pick up sign * V[q->p][i, b].
    jb, bb, rb, sb, gb = blink
    by_b: list[np.ndarray] = [np.nonzero(bb == b)[0] for b in range(noth)]
    idx = np.searchsorted(short, tab.xs)
    idx = np.minimum(idx, nshort - 1)
    touches = short[idx] == tab.xs
    qr = tab.qs[touches]
    rr, ii_, pp = idx[touches], tab.src_s[touches], tab.ps[touches]
    sgn = tab.sign_s[touches]
    for q in range(norb):
        sel = qr == q
        if not bool(sel.any()):
            continue
        yq = np.zeros((norb, cmat.shape[0], noth))
        for b, el in enumerate(by_b):
            if el.size == 0:
                continue
            gmat = g[:, q][:, rb[el], sb[el]]            # g[p,q,r_l,s_l]
            mmat = cmat[:, jb[el]] * gb[el]              # C[i,J_l] G_l
            yq[:, :, b] = gmat @ mmat.T
        np.add.at(rmat, rr[sel], sgn[sel][:, None] * yq[pp[sel], ii_[sel]])
    return rmat


def _cipsi_scores(rmat: np.ndarray, dmat: np.ndarray, energy: float) -> np.ndarray:
    denom = np.maximum(np.abs(dmat - energy), _DENOM_FLOOR)
    return ((np.abs(rmat) ** 2) / denom ** 2).sum(axis=1)


def _shortlist(cand: np.ndarray, scr: np.ndarray, h: np.ndarray,
               g: np.ndarray, norb: int) -> np.ndarray:
    """Top-3840 screening outsiders plus up to 256 lowest-diagonal extras."""
    order = np.argsort(-scr, kind="stable")
    n_main = min(_SCREEN_MAIN, cand.size)
    main = cand[order[:n_main]]
    if cand.size > n_main:
        rest = cand[order[n_main:]]
        oa = _occ_list(rest, norb)
        ii = np.arange(norb)
        jk = (g[ii[:, None], ii[:, None], ii[None, :], ii[None, :]]
              - g[ii[:, None], ii[None, :], ii[None, :], ii[:, None]])
        ind = np.zeros((rest.size, norb))
        ind[np.arange(rest.size)[:, None], oa] = 1.0
        est = ind @ np.diag(h) + 0.5 * (ind * (ind @ jk)).sum(axis=1)
        extra = rest[np.lexsort((rest, est))[:min(_DIAG_EXTRA, rest.size)]]
        return np.unique(np.concatenate([main, extra]))
    return np.sort(main)


def _residual_on_cand(cand: np.ndarray, short: np.ndarray, cmat: np.ndarray,
                      other: np.ndarray, tab: _Neighbours,
                      h: np.ndarray, g: np.ndarray, norb: int,
                      energy: float) -> np.ndarray:
    """Map evaluated CIPSI scores back; unscored strings cannot win exploitation."""
    full = np.full(cand.size, -np.inf)
    if short.size == 0:
        return full
    blink = _single_links_of_set(other, norb)
    rmat = _external_residual(short, cmat, tab, blink, g)
    dmat = diag_block(short, other, h, g)
    full[np.searchsorted(cand, short)] = _cipsi_scores(rmat, dmat, energy)
    return full


def _rank_outsiders(sol: Solution, h: np.ndarray, g: np.ndarray, symmetric: bool):
    """Shared screening/residual rankings for growth and swaps."""
    norb = h.shape[0]
    a, b = sol.strings_a, sol.strings_b
    wa, wb = _marginals(sol)
    fro = np.linalg.norm(g.reshape(norb * norb, norb * norb), axis=1).reshape(norb, norb)
    if symmetric:
        tab = _Neighbours(a, h, g, norb)
        cand, scr = _screen(tab, (wa + wb) / 2.0, fro)
        score = np.zeros(cand.size)
        if cand.size:
            short = _shortlist(cand, scr, h, g, norb)
            score = (_residual_on_cand(cand, short, sol.coeffs, b, tab,
                                       h, g, norb, sol.energy)
                     + _residual_on_cand(cand, short, sol.coeffs.T, a, tab,
                                         h, g, norb, sol.energy))
        ranked = (cand, scr, score)
        return ranked, ranked
    return (_side_scores(a, wa, sol.coeffs, b, h, g, norb, fro, sol.energy),
            _side_scores(b, wb, sol.coeffs.T, a, h, g, norb, fro, sol.energy))


def grow(sol: Solution, h, g, *, n_add: int = 100, symmetric: bool = True
         ) -> tuple[np.ndarray, np.ndarray]:
    """Add up to n_add outsiders per side; return ascending int64 supersets."""
    a, b = _as_sorted(sol.strings_a), _as_sorted(sol.strings_b)
    if symmetric and not np.array_equal(a, b):
        raise ValueError("symmetric grow requires sol.strings_a == sol.strings_b")
    if n_add <= 0:
        return a.copy(), b.copy()
    scores_a, scores_b = _rank_outsiders(
        sol, np.asarray(h, dtype=float), np.asarray(g, dtype=float), symmetric
    )
    a = np.union1d(a, _pick(*scores_a, n_add))
    if symmetric:
        return a, a.copy()
    return a, np.union1d(b, _pick(*scores_b, n_add))


def _pick(cand: np.ndarray, scr: np.ndarray, res: np.ndarray, n_add: int
          ) -> np.ndarray:
    """80% residual / 20% screening picks from the candidate pool."""
    n_add = min(n_add, cand.size)
    if n_add <= 0:
        return np.zeros(0, dtype=np.int64)
    n_res = min(int(round(0.8 * n_add)), cand.size)
    take_res = np.argsort(-res, kind="stable")[:n_res]
    scr2 = scr.copy()
    scr2[take_res] = -np.inf
    take_scr = np.argsort(-scr2, kind="stable")[:n_add - n_res]
    return cand[np.unique(np.concatenate([take_res, take_scr]))]


def _side_scores(cur: np.ndarray, w: np.ndarray, cmat: np.ndarray,
                 other: np.ndarray, h: np.ndarray, g: np.ndarray, norb: int,
                 fro: np.ndarray, energy: float):
    tab = _Neighbours(cur, h, g, norb)
    cand, scr = _screen(tab, w, fro)
    full = np.zeros(cand.size)
    if cand.size:
        short = _shortlist(cand, scr, h, g, norb)
        full = _residual_on_cand(cand, short, cmat, other, tab,
                                 h, g, norb, energy)
    return cand, scr, full


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _marginals(sol: Solution) -> tuple[np.ndarray, np.ndarray]:
    weights = np.square(sol.coeffs)
    return weights.sum(axis=1), weights.sum(axis=0)


def _check_strings(a: np.ndarray, b: np.ndarray, cap: int) -> None:
    for name, s in (("a", a), ("b", b)):
        if (s.ndim != 1 or s.size == 0 or s.size > cap
                or np.any(np.diff(s) <= 0)):
            raise ValueError(f"strings_{name} must be ascending unique, 1..{cap}")
        _check_popcount(s)


def _internal_spin_hamiltonian(strings: np.ndarray, h: np.ndarray, g: np.ndarray):
    """Off-diagonal same-spin H and internal single-excitation links."""
    tab = _Neighbours(strings, h, g, h.shape[0])
    block = np.zeros((strings.size, strings.size))
    for dest, src, value in ((tab.xs, tab.src_s, tab.vs),
                             (tab.xd, tab.sd, tab.vd)):
        pos = np.minimum(np.searchsorted(strings, dest), strings.size - 1)
        inside = strings[pos] == dest
        block[pos[inside], src[inside]] = value[inside]
    links = _single_links_of_set(strings, h.shape[0])
    offdiag = links[0] != links[1]
    return block, tuple(part[offdiag] for part in links)


def _row_block_quadratics(c: np.ndarray, fixed: np.ndarray, g: np.ndarray,
                          diagonal: np.ndarray, moving_h: np.ndarray, links):
    """All c_x.T H_xx c_x, without constructing any product-space H."""
    result = np.einsum("ij,ij,ij->i", c, c, diagonal)
    result += np.einsum("ij,ij->i", c, c @ moving_h.T)
    src, dest, p, q, sign = links
    # A fixed spin contributes only its occupied-orbital Coulomb potential
    # to off-diagonal singles of the moving spin. Doubles are in moving_h.
    occ = _occ_list(fixed, g.shape[0])
    potential = np.zeros((fixed.size, g.shape[0], g.shape[0]))
    for k in range(NELEC[0]):
        potential += g[occ[:, k], occ[:, k]]
    result += np.einsum("ij,ij,ij,j->i", c[:, src], c[:, dest],
                        potential[:, p, q], sign)
    return result


def removal_energies(sol: Solution, h, g, *, symmetric: bool = False
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Rayleigh energies after removing each single string, without solving.

    Requires a normalised, converged eigenvector. Independent mode returns
    alpha-row and beta-column removal energies. Symmetric mode requires a == b
    and returns the exact joint row-and-column removal energies on both sides,
    counting the intersection once and including mixed-spin single couplings.
    A removal leaving zero norm has energy +inf.

    For removed block r of weight w, E' = E + (r.T H r - E*w)/(1-w).
    Slater-Condon blocks and internal links make all estimates cheap compared
    with a cold solve; no full determinant Hamiltonian is allocated.
    """
    a, b = _as_sorted(sol.strings_a), _as_sorted(sol.strings_b)
    if symmetric and not np.array_equal(a, b):
        raise ValueError("symmetric removal requires strings_a == strings_b")
    h, g = np.asarray(h, dtype=float), np.asarray(g, dtype=float)
    c = sol.coeffs
    diagonal = diag_block(a, b, h, g)
    ha, alinks = _internal_spin_hamiltonian(a, h, g)
    hb, blinks = ((ha, alinks) if np.array_equal(a, b)
                  else _internal_spin_hamiltonian(b, h, g))
    qa = _row_block_quadratics(c, a, g, diagonal, hb, blinks)
    qb = _row_block_quadratics(c.T, b, g, diagonal.T, ha, alinks)
    wa, wb = _marginals(sol)

    def after_removal(quadratic, weight):
        loss = np.full(weight.shape, np.inf)
        np.divide(quadratic - sol.energy * weight, 1.0 - weight,
                  out=loss, where=weight < 1.0)
        return sol.energy + loss

    if not symmetric:
        return after_removal(qa, wa), after_removal(qb, wb)
    intersection = np.square(np.diag(c))
    joint = qa + qb - intersection * np.diag(diagonal)
    src, dest, p, q, sign = alinks
    # The only couplings between the disjoint row and column arms excite
    # one electron of each spin. All cell/arm couplings are already in qa+qb.
    for x in range(a.size):
        el = src == x
        kernel = g[p[el, None], q[el, None], p[None, el], q[None, el]]
        joint[x] += 2.0 * ((c[dest[el], x] * sign[el])
                           @ kernel @ (c[x, dest[el]] * sign[el]))
    energies = after_removal(joint, wa + wb - intersection)
    return energies, energies.copy()


def prune(sol: Solution, h=None, g=None, *, cap: int = 300, symmetric: bool = True,
          prune_mode: str = "marginal") -> tuple[np.ndarray, np.ndarray]:
    """Keep highest marginal weights or single-removal energies, ascending.

    ``energy`` needs h/g and ranks exact removal Rayleigh quotients from the
    converged enlarged eigenvector (joint row/column removal when symmetric).
    All removals are ranked together; the capped space is then cold-solved.
    ``marginal`` preserves the original averaged-weight ranking and tie order.
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")
    if prune_mode not in ("marginal", "energy"):
        raise ValueError("prune_mode must be 'marginal' or 'energy'")
    if prune_mode == "energy":
        if h is None or g is None:
            raise ValueError("energy pruning requires h and g")
        wa, wb = removal_energies(sol, h, g, symmetric=symmetric)
    else:
        wa, wb = _marginals(sol)
    if symmetric:
        if not np.array_equal(sol.strings_a, sol.strings_b):
            raise ValueError("symmetric prune requires strings_a == strings_b")
        w = (wa + wb) / 2.0
        k = min(cap, w.size)
        keep = np.sort(sol.strings_a[np.argsort(-w, kind="stable")[:k]])
        return keep, keep.copy()
    ka = min(cap, wa.size)
    kb = min(cap, wb.size)
    a = np.sort(sol.strings_a[np.argsort(-wa, kind="stable")[:ka]])
    b = np.sort(sol.strings_b[np.argsort(-wb, kind="stable")[:kb]])
    return a, b


# ---------------------------------------------------------------------------
# Grow -> solve -> prune -> solve loop
# ---------------------------------------------------------------------------

def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def select_strings(sol: Solution, h, g, *, cap: int = 300, n_add: int = 100,
                   symmetric: bool = True, prune_mode: str = "marginal",
                   max_iters: int = 20,
                   deadline: float | None = None,
                   on_proposal: Callable[[Solution, dict], None] | None = None
                   ) -> Solution:
    """Grow/cold-solve/prune/cold-solve from a converged, already capped state.

    Report every capped proposal, including failed solves, through
    ``on_proposal(solution, info)``. Only a lower converged Ritz energy
    replaces the incumbent. Three sub-2e-5 gains permit one ``n_add=300``
    plateau escape before stopping; a successful escape resumes normal growth.
    ``deadline`` is a time.time() epoch timestamp; an in-flight cold solve cannot
    be interrupted.
    ``prune_mode`` is ``marginal`` (original weights) or ``energy`` (exact
    single-removal Rayleigh loss, joint on both sides for symmetric searches).
    """
    h = np.asarray(h, dtype=float)
    g = np.asarray(g, dtype=float)
    if prune_mode not in ("marginal", "energy"):
        raise ValueError("prune_mode must be 'marginal' or 'energy'")
    if not sol.converged:
        raise ValueError("select_strings needs a converged starting Solution")
    if symmetric and not np.array_equal(sol.strings_a, sol.strings_b):
        raise ValueError("symmetric select_strings requires strings_a == strings_b")
    _check_strings(sol.strings_a, sol.strings_b, cap)
    best = sol
    flat = 0
    escaped = False
    for it in range(1, max_iters + 1):
        if _expired(deadline):
            break
        escape = flat >= _PLATEAU_PATIENCE
        if escape and escaped:
            break
        if escape:
            escaped = True
        batch = 300 if escape else n_add
        ea, eb = grow(best, h, g, n_add=batch, symmetric=symmetric)
        if (np.array_equal(ea, best.strings_a) and np.array_equal(eb, best.strings_b)):
            break
        if _expired(deadline):
            break
        big = solve_space(ea, eb, h, g)
        # An enlargement already under cap is itself a capped proposal;
        # solving exactly the same space a second time adds nothing.
        capped = big if ea.size <= cap and eb.size <= cap else None
        if big.converged and capped is None:
            pa, pb = prune(big, h, g, cap=cap, symmetric=symmetric, prune_mode=prune_mode)
            if _expired(deadline):
                break
            capped = solve_space(pa, pb, h, g)
        gain = 0.0
        if capped is not None:
            if on_proposal is not None:
                on_proposal(capped, {"kind": "grow-prune", "iter": it,
                                     "n_add": batch, "plateau_escape": escape,
                                     "prune_mode": prune_mode,
                                     "converged": capped.converged,
                                     "energy": capped.energy,
                                     "dims": list(capped.coeffs.shape),
                                     "parent_energy": best.energy,
                                     "enlarged_energy": big.energy})
            if capped.converged:
                gain = best.energy - capped.energy
                if gain > 0:
                    best = capped
        flat = 0 if gain >= _PLATEAU_TOL else flat + 1
        if escape and gain < _PLATEAU_TOL:
            break
    return best


# ---------------------------------------------------------------------------
# Bounded one-string swaps
# ---------------------------------------------------------------------------

def swap_search(sol: Solution, h, g, *, symmetric: bool = True, n_swaps: int = 16,
                deadline: float | None = None,
                on_proposal: Callable[[Solution, dict], None] | None = None
                ) -> Solution:
    """Evaluate up to n_swaps one-string swaps; at most two passes.

    Candidates: 8 weakest incumbent strings x strongest outsiders, ordered by
    external importance divided by removed marginal weight. For symmetric runs
    swap in both sets; for independent runs change one spin side at a time.
    Rankings are recomputed after an accepted swap.
    """
    h = np.asarray(h, dtype=float)
    g = np.asarray(g, dtype=float)
    if not sol.converged:
        raise ValueError("swap_search needs a converged starting Solution")
    if symmetric and not np.array_equal(sol.strings_a, sol.strings_b):
        raise ValueError("symmetric swap_search requires strings_a == strings_b")
    best = sol
    if n_swaps <= 0:
        return best
    for _pass in range(2):
        if _expired(deadline):
            break
        rankings = _rank_outsiders(best, h, g, symmetric)
        wa, wb = _marginals(best)
        if symmetric:
            wa = (wa + wb) / 2.0
        proposals: list[tuple[str, int, int, float]] = []
        for side, w, (cand, _scr, importance) in zip(("a", "b"), (wa, wb), rankings):
            if symmetric and side == "b":
                break
            weak_pos = np.argsort(w, kind="stable")[:8]
            strong = np.argsort(-importance, kind="stable")[:8]
            for op in weak_pos:
                for sp in strong:
                    key = float(importance[sp]) / max(float(w[op]), 1e-30)
                    proposals.append((side, int(op), int(cand[sp]), key))
        proposals.sort(key=lambda t: -t[3])
        accepted = False
        for rank, (side, op, newcomer, _key) in enumerate(proposals[:max(n_swaps, 0)]):
            if _expired(deadline):
                break
            base = best.strings_a if side == "a" else best.strings_b
            trial = base.copy()
            trial[op] = newcomer  # Outsiders are disjoint, so uniqueness is preserved.
            trial.sort()
            if symmetric:
                ta, tb = trial, trial.copy()
            elif side == "a":
                ta, tb = trial, best.strings_b
            else:
                ta, tb = best.strings_a, trial
            trial_sol = solve_space(ta, tb, h, g)
            if on_proposal is not None:
                on_proposal(trial_sol, {"kind": "swap", "pass": _pass,
                                        "rank": rank, "side": side,
                                        "converged": trial_sol.converged,
                                        "energy": trial_sol.energy,
                                        "dims": [len(ta), len(tb)]})
            if not trial_sol.converged:
                continue
            if trial_sol.energy < best.energy:
                best = trial_sol
                accepted = True
                break
        if not accepted:
            break
    return best
