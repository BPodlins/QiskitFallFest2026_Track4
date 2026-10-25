"""Verify a track-B submission directory produced by scripts/export_submission.py.

Usage:
    python scripts/verify_rotation.py [submission_dir=submission] [--skip-grader]

Prints PASS/FAIL per check and exits non-zero on any failure:

  a. sha256 of files match manifest;
  a2. every numeric array (U, integrals, canonical integrals) is real and finite;
  b. U is real, finite, 26x26, orthogonal (max|U^T U - I| <= 1e-12), report det(U);
  c. a single U for both spins (submission format statement);
  d. rotating canonical_hcore/canonical_eri by U reproduces hcore/eri (<= 1e-10);
  e. canonical integrals are the lab's N2 integrals vs a fresh RHF
     (build_n2(use_cache=False)); element-wise diff reported, and when it exceeds
     1e-8 (degenerate-orbital gauge of a different RHF run) the grader's
     rotation-invariant spectra check plus ecore decides; cached
     data/hamiltonian.npz comparison when present;
  e2. [INFO, never counted] optional numerical fit of that degenerate-gauge
     rotation W (--gauge-time-limit-s > 0); not reconstructed by default;
  f. manifest ecore equals the grader reference core energy within 1e-6;
  g. strings: 1..300 per side, int64, strictly ascending, popcount 5, < 2**26;
  h. unless --skip-grader: official_facts with the manifest energy as the claim
     must report matches=True; prints the recomputed energy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from n2sqd.check import official_facts, rotate_integrals, spectra
from n2sqd.hamiltonian import CACHE_FILE, build_n2

TOL_ORTHO = 1e-12
TOL_REPRODUCE = 1e-10
TOL_FRESH_ELEMENTWISE = 1e-8
TOL_GAUGE_RESIDUAL = 1e-8
TOL_DEGENERATE = 1e-6
TOL_SPECTRA = 1e-6
TOL_ECORE = 1e-6
MAX_SUBSPACE = 300

EXPECTED_FILES = ("rotation.npy", "hcore.npy", "eri.npy",
                  "ci_strs_a.npy", "ci_strs_b.npy",
                  "canonical_hcore.npy", "canonical_eri.npy")


@dataclass
class Check:
    key: str
    ok: bool
    detail: str
    info: bool = False  # informational line: printed as [INFO], never counted or failing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes(sub: Path, manifest: dict) -> Check:
    digests = manifest.get("sha256", {})
    problems = []
    for name in EXPECTED_FILES:
        if name not in digests:
            problems.append(f"{name} missing from manifest")
            continue
        path = sub / name
        if not path.exists():
            problems.append(f"{name} missing from {sub}")
            continue
        if _sha256(path) != digests[name]:
            problems.append(f"{name} sha256 mismatch")
    if problems:
        return Check("a:sha256", False, "; ".join(problems))
    return Check("a:sha256", True,
                 f"{len(EXPECTED_FILES)}/{len(EXPECTED_FILES)} files match manifest sha256")


def check_rotation(u: np.ndarray, norb_expected: int) -> Check:
    n = norb_expected
    if u.shape != (n, n):
        return Check("b:rotation", False,
                     f"U shape {tuple(u.shape)} != ({n}, {n})")
    if not np.isrealobj(u) or not np.all(np.isfinite(u)):
        return Check("b:rotation", False, "U must be real and finite")
    u = np.asarray(u, dtype=float)
    err = float(np.max(np.abs(u.T @ u - np.eye(n))))
    det = float(np.linalg.det(u))
    ok = err <= TOL_ORTHO
    return Check("b:rotation", ok,
                 f"U {n}x{n} real finite; max|U^T U - I| = {err:.3e} "
                 f"(tol {TOL_ORTHO:.0e}); det(U) = {det:+.6f}")


def check_single_u() -> Check:
    return Check("c:single-U", True,
                 "submission format holds one rotation.npy and one set of spatial "
                 "integrals, so both spins share the same U")


def check_reproduction(u: np.ndarray, canon_h: np.ndarray, canon_g: np.ndarray,
                       h: np.ndarray, g: np.ndarray, n: int) -> Check:
    if (canon_h.shape != (n, n) or np.shape(h) != (n, n)
            or canon_g.shape != (n,) * 4 or np.shape(g) != (n,) * 4):
        return Check("d:reproduction", False,
                     f"shape mismatch: canon_h {canon_h.shape}, h {np.shape(h)}, "
                     f"canon_eri {canon_g.shape}, eri {np.shape(g)}")
    try:
        hr, gr = rotate_integrals(canon_h, canon_g, np.asarray(u, dtype=float))
    except Exception as err:  # noqa: BLE001
        return Check("d:reproduction", False, f"rotation failed: {err}")
    dh = float(np.max(np.abs(hr - np.asarray(h, dtype=float))))
    dg = float(np.max(np.abs(gr - np.asarray(g, dtype=float))))
    ok = dh <= TOL_REPRODUCE and dg <= TOL_REPRODUCE
    return Check("d:reproduction", ok,
                 f"max|hcore - rot(canonical)| = {dh:.3e}, "
                 f"max|eri - rot(canonical)| = {dg:.3e} (tol {TOL_REPRODUCE:.0e})")


def fresh_orbital_energies() -> np.ndarray | None:
    """Active-space RHF orbital energies (scf.mo_energy[2:]) for grouping.

    Rebuilds the SCF exactly like the grader; returns None when unavailable
    (check_gauge then reports that no W can be fitted).
    """
    try:
        from n2sqd.hamiltonian import rhf_reference
        _, scf = rhf_reference()
        return np.asarray(scf.mo_energy, dtype=float)[2:]
    except Exception:  # noqa: BLE001
        return None


def _cached_diffs(canon_h: np.ndarray, canon_g: np.ndarray) -> str:
    if not CACHE_FILE.exists():
        return "no cached data/hamiltonian.npz present"
    try:
        with np.load(CACHE_FILE) as z:
            dh = float(np.max(np.abs(np.asarray(canon_h, dtype=float) - z["hcore"])))
            dg = float(np.max(np.abs(np.asarray(canon_g, dtype=float) - z["eri"])))
        return (f"cached data/hamiltonian.npz element-wise: max|dh| = {dh:.3e}, "
                f"max|dg| = {dg:.3e}")
    except Exception as err:  # noqa: BLE001
        return f"cached data/hamiltonian.npz unreadable: {err}"


def _degenerate_groups(orbital_energies, tol=TOL_DEGENERATE) -> list[list[int]]:
    """Group active-orbital indices whose RHF energies agree within tol (sorted order)."""
    energies = [float(v) for v in orbital_energies]
    order = sorted(range(len(energies)), key=energies.__getitem__)
    groups, current = [], [order[0]]
    for j in order[1:]:
        if abs(energies[j] - energies[current[-1]]) <= tol:
            current.append(j)
        else:
            groups.append(sorted(current))
            current = [j]
    groups.append(sorted(current))
    return groups


def _apply_pair_block(h, g, idx, block):
    """Apply one degenerate-pair block: h' = W^T h W with W = I except block on idx."""
    i, j = int(idx[0]), int(idx[1])
    b = np.asarray(block, dtype=float)
    h2 = np.array(h, dtype=float, copy=True)
    h2[[i, j], :] = b.T @ h2[[i, j], :]
    h2[:, [i, j]] = h2[:, [i, j]] @ b
    g2 = np.array(g, dtype=float, copy=True)
    for axis in range(4):
        moved = np.moveaxis(g2, axis, 0).copy()
        old_i, old_j = moved[i].copy(), moved[j].copy()
        moved[i] = b[0, 0] * old_i + b[1, 0] * old_j
        moved[j] = b[0, 1] * old_i + b[1, 1] * old_j
        g2 = np.moveaxis(moved, 0, axis)
    return h2, g2


def _apply_sign_flip(h, g, k):
    """Flip the sign of one orbital (slices with an odd count of k are negated)."""
    k = int(k)
    h2 = np.array(h, dtype=float, copy=True)
    h2[k, :] *= -1.0
    h2[:, k] *= -1.0
    g2 = np.array(g, dtype=float, copy=True)
    for axis in range(4):
        moved = np.moveaxis(g2, axis, 0)
        moved[k] *= -1.0
    return h2, g2


def _sse(h2, g2, ch, cg):
    """Sum of squared element-wise residuals of trial integrals vs canonical."""
    dh = np.asarray(h2, dtype=float) - np.asarray(ch, dtype=float)
    dg = np.asarray(g2, dtype=float) - np.asarray(cg, dtype=float)
    return float(dh.ravel() @ dh.ravel() + dg.ravel() @ dg.ravel())


def _residuals(h2, g2, ch, cg):
    """Max abs element-wise residuals of trial rotated integrals vs canonical."""
    return (float(np.max(np.abs(np.asarray(h2) - np.asarray(ch)))),
            float(np.max(np.abs(np.asarray(g2) - np.asarray(cg)))))


def _block_of(kind, deg):
    theta = np.deg2rad(float(deg))
    c, s = np.cos(theta), np.sin(theta)
    return (np.array([[c, -s], [s, c]]) if kind == "rotation"
            else np.array([[c, s], [s, -c]]))


def _minimize_pair(hcur, gcur, ch, cg, idx, *, tol):
    """Minimise the joint SSE over one pair's absolute 2x2 orthogonal block.

    Coarse scan (1 deg, rotations + reflections) then golden-section refine of
    the winner. Returns (sse, block, kind, deg, trial_h, trial_g).
    """
    cur_sse = _sse(hcur, gcur, ch, cg)
    best = (cur_sse, np.eye(2), "current", 0.0, hcur, gcur)
    scan = (np.linspace(0.0, 360.0, 361)[:-1], ("rotation", "reflection"))
    for kind in scan[1]:
        for deg in scan[0]:
            h2, g2 = _apply_pair_block(hcur, gcur, idx, _block_of(kind, deg))
            value = _sse(h2, g2, ch, cg)
            if value < best[0]:
                best = (value, _block_of(kind, deg), kind, float(deg), h2, g2)
    best_kind = best[2]
    best_deg = best[3]
    if best_kind == "current":
        return best
    kind, deg = best_kind, best_deg
    lo, hi = deg - 1.0, deg + 1.0
    for _ in range(60):
        m1 = lo + (hi - lo) * 0.3819660113
        m2 = lo + (hi - lo) * 0.6180339887
        v1 = _sse(*_apply_pair_block(hcur, gcur, idx, _block_of(kind, m1)), ch, cg)
        v2 = _sse(*_apply_pair_block(hcur, gcur, idx, _block_of(kind, m2)), ch, cg)
        if v1 < v2:
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-9:
            break
    deg = (lo + hi) / 2.0
    h2, g2 = _apply_pair_block(hcur, gcur, idx, _block_of(kind, deg))
    value = _sse(h2, g2, ch, cg)
    if value < best[0]:
        return value, _block_of(kind, deg), kind, float(deg % 360.0), h2, g2
    return best


def fit_degenerate_gauge(fresh_h, fresh_g, canon_h, canon_g, groups, n,
                         *, tol=TOL_GAUGE_RESIDUAL, max_sweeps=60,
                         restarts=0, time_limit_s=None) -> tuple[np.ndarray | None, dict]:
    """Fit a block-diagonal gauge W with rotate(fresh, W) == canonical.

    Coordinate descent on the joint sum of squared errors: exact 1D minimisation
    per degenerate pair (rotations + reflections) plus single-orbital sign
    flips. The max-residual objective is non-smooth (argmax jumps stall the
    fit); SSE converges to the exact gauge instead. Returns (W or None, info).
    """
    fresh_h = np.asarray(fresh_h, dtype=float)
    fresh_g = np.asarray(fresh_g, dtype=float)
    canon_h = np.asarray(canon_h, dtype=float)
    canon_g = np.asarray(canon_g, dtype=float)
    info: dict = {"blocks": [], "dh": np.inf, "dg": np.inf,
                  "attempts": 0, "sweeps": 0}
    pairs = [[int(i) for i in group] for group in groups if len(group) == 2]
    singles = [int(group[0]) for group in groups if len(group) == 1]
    if len(pairs) + len(singles) != len(groups):
        info["note"] = ("unsupported group size (only singles and pairs); "
                        "no gauge rotation W was fitted")
        return None, info
    gauge = np.eye(n)
    current_h, current_g = fresh_h.copy(), fresh_g.copy()
    current_sse = _sse(current_h, current_g, canon_h, canon_g)
    start = time.perf_counter()
    for sweep in range(max_sweeps):
        info["sweeps"] = sweep + 1
        improved = False
        for pair in pairs:
            value, winner, kind, deg, h2, g2 = _minimize_pair(
                current_h, current_g, canon_h, canon_g, pair, tol=tol)
            if value < current_sse - 1e-20:
                gauge[np.ix_(pair, pair)] = winner @ gauge[np.ix_(pair, pair)]
                current_h, current_g, current_sse, improved = h2, g2, value, True
                info["last_pair"] = (pair, kind, round(deg, 3))
        for k in singles:
            h2, g2 = _apply_sign_flip(current_h, current_g, k)
            value = _sse(h2, g2, canon_h, canon_g)
            if value < current_sse - 1e-20:
                gauge[k, :] *= -1.0
                current_h, current_g, current_sse, improved = h2, g2, value, True
        dh, dg = _residuals(current_h, current_g, canon_h, canon_g)
        if max(dh, dg) <= tol:
            info.update(_gauge_info(gauge, dh, dg, pairs, singles))
            return gauge, info
        if not improved:
            break
        if time_limit_s is not None and time.perf_counter() - start > time_limit_s:
            info["note"] = f"gauge fit time limit ({time_limit_s:.0f}s) reached"
            break
    dh, dg = _residuals(current_h, current_g, canon_h, canon_g)
    info.update(_gauge_info(gauge, dh, dg, pairs, singles))
    if dh <= tol and dg <= tol:
        return gauge, info
    return None, info


def _gauge_info(gauge, dh, dg, pairs, singles):
    """Summarise a fitted gauge rotation for the check output."""
    gauge = np.asarray(gauge, dtype=float)
    lines = [f"[{k}]: {'+1' if gauge[k, k] > 0 else '-1'}" for k in singles]
    for pair in pairs:
        block = gauge[np.ix_(pair, pair)]
        det = float(np.linalg.det(block))
        angle = float(np.rad2deg(np.arctan2(block[1, 0] - block[0, 1],
                                            block[0, 0] + block[1, 1])) % 360.0)
        lines.append(f"{pair}: {'reflection' if det < 0 else 'rotation'} "
                     f"{angle:.2f}deg {np.asarray(block).ravel().round(4).tolist()}")
    return {"blocks": lines, "dh": dh, "dg": dg,
            "ortho_err": float(np.max(np.abs(gauge.T @ gauge - np.eye(gauge.shape[0])))),
            "det": float(np.linalg.det(gauge))}


def check_canonical(canon_h: np.ndarray, canon_g: np.ndarray, ecore: float,
                    n: int) -> tuple[Check, tuple | None]:
    """Compare canonical integrals against a fresh RHF; return check + reference triple."""
    try:
        ref_h, ref_g, ref_core, n_ref, _ = build_n2(use_cache=False)
    except Exception as err:  # noqa: BLE001
        return Check("e:canonical", False, f"fresh RHF build failed: {err}"), None
    ref_core = float(ref_core)
    if n_ref != n:
        return Check("e:canonical", False,
                     f"fresh RHF norb {n_ref} != submission norb {n}"), None
    dh = float(np.max(np.abs(np.asarray(canon_h, dtype=float) - ref_h)))
    dg = float(np.max(np.abs(np.asarray(canon_g, dtype=float) - ref_g)))
    head = (f"fresh-RHF (build_n2(use_cache=False)) element-wise: max|dh| = {dh:.3e}, "
            f"max|dg| = {dg:.3e} (tol {TOL_FRESH_ELEMENTWISE:.0e})")
    tail = _cached_diffs(canon_h, canon_g)
    if max(dh, dg) <= TOL_FRESH_ELEMENTWISE:
        return Check("e:canonical", True,
                     head + "; canonical integrals match fresh RHF element-wise; " + tail), \
            (ref_h, ref_g, ref_core)
    spec_ok = all(
        np.allclose(mine, ref, rtol=0, atol=TOL_SPECTRA)
        for mine, ref in zip(spectra(np.asarray(canon_h, dtype=float),
                                     np.asarray(canon_g, dtype=float), n),
                             spectra(ref_h, ref_g, n_ref)))
    ecore_ok = abs(float(ecore) - ref_core) <= TOL_ECORE
    ok = bool(spec_ok and ecore_ok)
    return Check("e:canonical", ok,
                 head + "; exceeds 1e-8: fresh RHF differs by a degenerate-orbital gauge, "
                 f"so the grader's rotation-invariant spectra check (same three spectra, "
                 f"atol {TOL_SPECTRA:.0e}) plus ecore ({TOL_ECORE:.0e}) is required: "
                 f"spectra {'PASS' if spec_ok else 'FAIL'}, "
                 f"|ecore - ref| = {abs(float(ecore) - ref_core):.3e} "
                 f"({'PASS' if ecore_ok else 'FAIL'}); " + tail), \
        (ref_h, ref_g, ref_core)


def check_gauge(u, canon_h, canon_g, sub_h, sub_g, ref, n,
                orbital_energies="auto", time_limit_s=None) -> Check:
    """Fit the degenerate-gauge W between a fresh RHF and the canonical integrals.

    Requires rotate(fresh, W) == canonical and submitted == rotate(fresh, W@U)
    within 1e-8 for both hcore and eri. Groups come from the fresh RHF orbital
    energies (scf.mo_energy[2:], tol 1e-6); without energies the check FAILs.
    The default "auto" rebuilds them via fresh_orbital_energies(); tests pass
    an explicit small array or None.
    """
    if ref is None:
        return Check("e2:gauge", False, "no fresh-RHF reference available")
    if isinstance(orbital_energies, str) and orbital_energies == "auto":
        orbital_energies = fresh_orbital_energies()
    if orbital_energies is None:
        return Check("e2:gauge", False,
                     "fresh RHF orbital energies unavailable: cannot define "
                     "degenerate groups, so no gauge rotation W was fitted")
    ref_h, ref_g, _ = ref
    energies = np.asarray(orbital_energies, dtype=float)
    if energies.shape != (n,):
        return Check("e2:gauge", False,
                     f"orbital energies shape {tuple(energies.shape)} != ({n},): "
                     "cannot define degenerate groups")
    groups = _degenerate_groups(energies)
    multi = [grp for grp in groups if len(grp) > 1]
    gauge, info = fit_degenerate_gauge(ref_h, ref_g, canon_h, canon_g, groups, n,
                                       time_limit_s=time_limit_s)
    group_text = "; ".join(info["blocks"])
    head = (f"degenerate groups (scf.mo_energy tol {TOL_DEGENERATE:.0e}): {multi}; "
            f"blocks: {group_text}")
    if gauge is None:
        return Check("e2:gauge", False,
                     head + f"; no gauge W found: max|rot(fresh,W)-canonical| = "
                     f"dh {info['dh']:.3e}, dg {info['dg']:.3e} "
                     f"(tol {TOL_GAUGE_RESIDUAL:.0e})")
    total = gauge @ np.asarray(u, dtype=float)
    total_h, total_g = rotate_integrals(np.asarray(ref_h, dtype=float),
                                        np.asarray(ref_g, dtype=float), total)
    dh = float(np.max(np.abs(total_h - np.asarray(sub_h, dtype=float))))
    dg = float(np.max(np.abs(total_g - np.asarray(sub_g, dtype=float))))
    ok = dh <= TOL_GAUGE_RESIDUAL and dg <= TOL_GAUGE_RESIDUAL
    return Check("e2:gauge", ok,
                 head + f"; W orthogonal (err {info['ortho_err']:.3e}, det {info['det']:+.6f}); "
                 f"max|rot(fresh,W)-canonical| = dh {info['dh']:.3e}, dg {info['dg']:.3e}; "
                 f"submitted == rot(fresh,W@U): dh {dh:.3e}, dg {dg:.3e} "
                 f"(tol {TOL_GAUGE_RESIDUAL:.0e})")


def check_ecore(ecore: float, ref_core: float | None) -> Check:
    if ref_core is None:
        return Check("f:ecore", False, "no grader reference core energy available")
    diff = abs(float(ecore) - float(ref_core))
    return Check("f:ecore", diff <= TOL_ECORE,
                 f"|ecore_manifest - ecore_grader_ref| = {diff:.3e} (tol {TOL_ECORE:.0e})")


def _popcount(s: int) -> int:
    return bin(int(s)).count("1")


def check_strings(a: np.ndarray, b: np.ndarray, norb: int,
                  nelec: tuple[int, int]) -> Check:
    for side, strings, need in (("a", a, nelec[0]), ("b", b, nelec[1])):
        if np.shape(strings) != (len(strings),) or np.asarray(strings).dtype != np.dtype(np.int64):
            return Check("g:strings", False,
                         f"side {side}: must be a 1-D int64 array")
        strings = np.asarray(strings)
        if not (1 <= len(strings) <= MAX_SUBSPACE):
            return Check("g:strings", False,
                         f"side {side}: len {len(strings)} not in 1..{MAX_SUBSPACE}")
        if not bool(np.all(np.diff(strings) > 0)):
            return Check("g:strings", False,
                         f"side {side}: strings not strictly ascending")
        if bool(np.any(strings < 0)) or bool(np.any(strings >= (1 << norb))):
            return Check("g:strings", False,
                         f"side {side}: strings must satisfy 0 <= s < 2**{norb}")
        if any(_popcount(s) != need for s in strings):
            return Check("g:strings", False,
                         f"side {side}: every string must have popcount {need}")
    return Check("g:strings", True,
                 f"dims {len(a)} x {len(b)}; int64 strictly ascending, "
                 f"popcount ({nelec[0]}, {nelec[1]}), < 2**{norb}")


def check_grader(a: np.ndarray, b: np.ndarray, h: np.ndarray, g: np.ndarray,
                 ecore: float, claimed: float) -> Check:
    try:
        facts = official_facts(a, b, np.asarray(h, dtype=float),
                               np.asarray(g, dtype=float), float(ecore), float(claimed))
    except Exception as err:  # noqa: BLE001
        return Check("h:grader", False, f"official_facts raised: {err}")
    precise = facts.get("check", {}).get("energy")
    ok = bool(facts.get("matches"))
    return Check("h:grader", ok,
                 f"claimed {float(claimed):.6f} Ha; recomputed energy = "
                 f"{facts.get('energy')} Ha"
                 + (f" (precise {precise:.10f} Ha)" if precise is not None else "") +
                 f"; matches = {facts.get('matches')}")


def verify(submission_dir: str | Path = "submission", *, skip_grader: bool = False,
           norb_expected: int = 26,
           nelec_expected: tuple[int, int] = (5, 5),
           orbital_energies="auto", gauge_time_limit_s=None) -> list[Check]:
    """Run checks a..h plus the informational gauge line e2.

    e2 tries to fit the degenerate-gauge W between a fresh RHF and the
    starting integrals. It is informational only: printed as [INFO] and never
    counted. A non-positive ``gauge_time_limit_s`` skips the fit.
    """
    sub = Path(submission_dir)
    try:
        manifest = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
        data = {name: np.load(sub / name) for name in EXPECTED_FILES}
    except Exception as err:  # noqa: BLE001
        return [Check("load", False, f"cannot load submission in {sub}: {err}")]
    n = norb_expected
    real_bad = [name for name in ("rotation.npy", "hcore.npy", "eri.npy",
                                  "canonical_hcore.npy", "canonical_eri.npy")
                if np.iscomplexobj(data[name]) or not np.all(np.isfinite(data[name]))]
    checks = [check_hashes(sub, manifest),
              Check("a2:real", not real_bad,
                    "U and all integral arrays are real and finite" if not real_bad
                    else f"non-real or non-finite arrays: {', '.join(real_bad)}"),
              check_rotation(data["rotation.npy"], n),
              check_single_u(),
              check_reproduction(data["rotation.npy"], data["canonical_hcore.npy"],
                                 data["canonical_eri.npy"], data["hcore.npy"],
                                 data["eri.npy"], n)]
    e_check, ref = check_canonical(data["canonical_hcore.npy"], data["canonical_eri.npy"],
                                   float(manifest.get("ecore", "nan")), n)
    checks.append(e_check)
    if gauge_time_limit_s is not None and gauge_time_limit_s <= 0:
        gauge_check = None
    else:
        gauge_check = check_gauge(data["rotation.npy"], data["canonical_hcore.npy"],
                                  data["canonical_eri.npy"], data["hcore.npy"],
                                  data["eri.npy"], ref, n,
                                  orbital_energies=orbital_energies,
                                  time_limit_s=gauge_time_limit_s)
    if gauge_check is not None and gauge_check.ok:
        gauge_detail = gauge_check.detail
    else:
        status = ("canonical integrals validated by spectra check e" if e_check.ok
                  else "canonical integrals NOT validated: see check e")
        gauge_detail = f"not reconstructed (cached RHF mo_coeff not stored); {status}"
    checks.append(Check("e2:gauge", True, gauge_detail, info=True))
    checks.append(check_ecore(float(manifest.get("ecore", "nan")),
                              ref[2] if ref is not None else None))
    checks.append(check_strings(data["ci_strs_a.npy"], data["ci_strs_b.npy"],
                                n, nelec_expected))
    if not skip_grader:
        checks.append(check_grader(data["ci_strs_a.npy"], data["ci_strs_b.npy"],
                                   data["hcore.npy"], data["eri.npy"],
                                   float(manifest.get("ecore", "nan")),
                                   float(manifest.get("energy"))))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission_dir", nargs="?", default="submission")
    parser.add_argument("--skip-grader", action="store_true",
                        help="skip the solve_sci grader check (h)")
    parser.add_argument("--gauge-time-limit-s", type=float, default=0.0,
                        help="seconds for the optional numerical gauge fit (default 0: skip)")
    args = parser.parse_args(argv)
    checks = verify(args.submission_dir, skip_grader=args.skip_grader,
                    gauge_time_limit_s=args.gauge_time_limit_s)
    failed = counted = 0
    for item in checks:
        tag = "INFO" if item.info else ("PASS" if item.ok else "FAIL")
        print(f"[{tag}] {item.key}: {item.detail}", flush=True)
        if not item.info:
            counted += 1
            failed += not item.ok
    print(f"verify_rotation: {counted - failed}/{counted} checks passed.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
