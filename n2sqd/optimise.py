"""Track A/B search controller for step-3 optimisation.

Schedules string selection and orbital rotation, verifies every improving
capped proposal with the grader replica, checkpoints winners immutably, and
keeps an append-only JSONL run record. No solver algebra here.

Track A uses the canonical cached integrals from ``build_n2()`` and is never
rotated. Track B starts from the best verified A candidate (or ``--start``)
and alternates ffsim/natural rotations (integrals always re-rotated from the
original h/g with the cumulative U) with ``select_strings`` refinement.

``n2sqd.selection`` / ``n2sqd.orbitals`` are imported lazily so this module
loads (and runs seed-only searches) before those siblings land; a missing
sibling degrades gracefully to the phases that do not need it.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from n2sqd.candidates import RESULTS_DIR as CANDIDATE_RESULTS_DIR
from n2sqd.candidates import load_candidate, save_candidate
from n2sqd.check import official_facts, rotate_integrals
from n2sqd.hamiltonian import build_n2
from n2sqd.solver import Solution, solve_space

__all__ = ["run_search", "best_track_candidate",
           "seed_hf_singles_doubles", "seed_lex14_first300"]

RESULTS_DIR = CANDIDATE_RESULTS_DIR
RUNS_DIR = Path(os.environ.get("N2SQD_RUNS", Path(__file__).resolve().parent.parent / "runs"))

CAP = 300
RITZ_MATCH_TOL = 1e-6
ORTHO_TOL = 1e-10
MACRO_MAX = 20
MACRO_MIN_GAIN = 5e-5
MACRO_POOR_STOP = 3


def _as_int(combo) -> int:
    s = 0
    for p in combo:
        s |= 1 << int(p)
    return s


def seed_hf_singles_doubles(norb: int = 26) -> np.ndarray:
    """300 strings: HF + all singles + lowest (sum, lex) doubles."""
    hf = (0, 1, 2, 3, 4)
    singles = sorted({tuple(sorted((set(hf) - {i}) | {a}))
                      for i in hf for a in range(5, norb)})
    doubles = sorted(
        {tuple(sorted((set(hf) - {i, j}) | {a, b}))
         for k, i in enumerate(hf) for j in hf[k + 1:]
         for ia, a in enumerate(range(5, norb)) for b in range(5, norb) if b > a},
        key=lambda c: (sum(c), c))
    take = [hf] + singles + doubles[: 300 - 1 - len(singles)]
    return np.array(sorted(_as_int(c) for c in take), dtype=np.int64)


def seed_lex14_first300() -> np.ndarray:
    """First 300 lexicographic 5-orbital combos of orbitals 0..13 (sorted ints)."""
    return np.array(
        sorted(_as_int(c) for c in itertools.islice(itertools.combinations(range(14), 5), 300)),
        dtype=np.int64)


def _try_import(dotted: str, attr: str | None = None):
    try:
        mod = importlib.import_module(dotted)
    except ImportError:
        return None
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError:
        return None


class _Run:
    """Mutable per-invocation search state: incumbent, record, counters."""

    def __init__(self, *, track: str, run_id: str, seed: int, symmetric: bool,
                 h0, g0, ecore: float, start, results_dir, record_path,
                 t0: float, deadline: float):
        self.track = track
        self.run_id = run_id
        self.seed = seed
        self.symmetric = symmetric
        self.h0 = np.asarray(h0, dtype=float)
        self.g0 = np.asarray(g0, dtype=float)
        self.ecore = float(ecore)
        self.canonical_parent: str | None = None
        self.norb = self.h0.shape[0]
        self.start = start
        self.results_dir = Path(results_dir)
        self.t0 = t0
        self.deadline = deadline
        self.n = 0
        self.inc: dict[str, Any] | None = None
        self.record_path = Path(record_path)
        self._fh = open(self.record_path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def log(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("t_wall_s", time.time() - self.t0)
        event.setdefault("t_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._fh.write(json.dumps(event, default=str) + "\n")
        self._fh.flush()

    def say(self, msg: str) -> None:
        print(f"[run {self.run_id} {self.track}] {msg}", flush=True)

    def bound(self, h, g, u, kind: str):
        """Basis-bound on_proposal callback for select_strings."""
        def cb(sol: Solution, info: dict | None = None) -> None:
            self.consider(sol, h, g, u, kind=kind, info=info)
        return cb

    def _save(self, sol: Solution, h, g, u, *, method: str, ritz: float,
              grader: float, match: bool, submittable: bool, verify_s: float) -> Path:
        na, nb = int(len(sol.strings_a)), int(len(sol.strings_b))
        details = {
            "track": self.track, "run_id": self.run_id,
            "parent": self.inc["path"].name if self.inc else None,
            "canonical_parent": self.canonical_parent,
            "iteration": self.n, "kind": method.split("_it")[-1],
            "ritz_energy": ritz, "grader_energy": grader,
            "grader_matches_ritz": match, "submittable": submittable,
            "verified": True, "converged": True, "dims": [na, nb],
            "symmetric": self.symmetric, "seed": self.seed,
            "flags": {"rotated": self.track == "B", "symmetric": self.symmetric,
                      "submittable": submittable,
                      "grader_matches_ritz": match},
            "timings": {"verify_s": verify_s},
        }
        try:
            return save_candidate(
                sol.strings_a, sol.strings_b, h, g, self.ecore, grader,
                method=method, seed=self.seed, rotated=self.track == "B",
                results_dir=self.results_dir, amplitudes=sol.coeffs,
                orbital_rotation=np.asarray(u, dtype=float), details=details,
                canonical_hcore=self.h0 if self.track == "B" else None,
                canonical_eri=self.g0 if self.track == "B" else None)
        except FileExistsError:
            return save_candidate(
                sol.strings_a, sol.strings_b, h, g, self.ecore, grader,
                method=method + "_dup", seed=self.seed, rotated=self.track == "B",
                results_dir=self.results_dir, amplitudes=sol.coeffs,
                orbital_rotation=np.asarray(u, dtype=float), details=details,
                canonical_hcore=self.h0 if self.track == "B" else None,
                canonical_eri=self.g0 if self.track == "B" else None)

    def consider(self, sol: Solution, h, g, u, *, kind: str,
                 info: dict | None = None) -> bool:
        """Verify and maybe promote a capped proposal. True iff new incumbent.

        Promotion requires convergence, both verification matches, grader/Ritz
        agreement within 1e-6, and a strictly lower verified grader energy.
        ``official_facts`` supplies both checks in exactly two cold solves.
        Verification/save failures keep the incumbent intact.
        """
        self.n += 1
        ritz = float(sol.energy) + self.ecore
        na, nb = int(len(sol.strings_a)), int(len(sol.strings_b))
        evt: dict[str, Any] = {
            "type": "proposal", "n": self.n, "kind": kind,
            "na": na, "nb": nb, "ritz": ritz,
            "converged": bool(sol.converged),
            "incumbent_ritz": self.inc["ritz"] if self.inc else None,
            "info": info,
        }
        if not sol.converged:
            self.log({**evt, "accepted": False, "reason": "unconverged"})
            return False
        if min(na, nb) == 0 or max(na, nb) > CAP:
            self.log({**evt, "accepted": False, "reason": "over_cap"})
            return False
        if not np.isfinite(ritz):
            self.log({**evt, "accepted": False, "reason": "nonfinite_energy"})
            return False
        if self.inc is not None and ritz >= self.inc["ritz"]:
            self.log({**evt, "accepted": False, "reason": "not_lower"})
            return False
        if time.time() >= self.deadline:
            self.log({**evt, "accepted": False, "reason": "deadline"})
            return False
        t = time.perf_counter()
        try:
            facts = official_facts(sol.strings_a, sol.strings_b, h, g,
                                   self.ecore, ritz, deadline=self.deadline)
            chk = facts["check"]
            grader = float(chk["energy"])
        except Exception as err:  # noqa: BLE001
            self.log({**evt, "accepted": False,
                      "reason": f"verify_failed: {type(err).__name__}: {err}"})
            return False
        verify_s = time.perf_counter() - t
        match = bool(np.isfinite(grader) and abs(grader - ritz) <= RITZ_MATCH_TOL)
        submittable = bool(facts["matches"] and chk["matches"] and match)
        if not submittable:
            self.log({**evt, "accepted": False, "reason": "verification_mismatch",
                      "grader": grader, "matches": bool(facts["matches"]),
                      "grader_matches_ritz": match, "verify_s": verify_s})
            return False
        if self.inc is not None and grader >= self.inc["grader"]:
            self.log({**evt, "accepted": False, "reason": "grader_not_lower",
                      "grader": grader, "verify_s": verify_s})
            return False
        method = f"{self.track}_run{self.run_id}_it{self.n:04d}_{kind}"
        try:
            path = self._save(sol, h, g, u, method=method, ritz=ritz,
                              grader=grader, match=match,
                              submittable=submittable, verify_s=verify_s)
        except Exception as err:  # noqa: BLE001
            self.log({**evt, "accepted": False,
                      "reason": f"save_failed: {type(err).__name__}: {err}"})
            return False
        self.inc = {"sol": sol, "h": np.asarray(h, dtype=float),
                    "g": np.asarray(g, dtype=float),
                    "u": np.asarray(u, dtype=float),
                    "ritz": ritz, "grader": grader,
                    "submittable": submittable, "path": path}
        self.log({**evt, "accepted": True, "grader": grader,
                  "submittable": submittable, "file": path.name,
                  "verify_s": verify_s})
        self.say(f"promoted it={self.n} {kind} {na}x{nb} "
                 f"ritz={ritz:.6f} grader={grader:.6f} sub={submittable} {path.name}")
        return True


def _read_meta(js_path: Path) -> dict | None:
    try:
        return json.loads(js_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _verified_metadata(meta: dict) -> bool:
    """Require positive convergence/submission evidence and precise energy agreement."""
    d = meta.get("details") or {}
    if (d.get("converged") is not True or d.get("submittable") is not True
            or d.get("grader_matches_ritz") is not True
            or d.get("verified") is False or d.get("matches") is False):
        return False
    try:
        ritz, grader = float(d["ritz_energy"]), float(d["grader_energy"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(ritz) and np.isfinite(grader)
                and abs(grader - ritz) <= RITZ_MATCH_TOL)


def best_track_candidate(track: str, results_dir=RESULTS_DIR):
    """Lowest verified grader energy of ``track``; exact ties choose the latest.

    Ignore unconverged, failed-verification, over-cap or incomplete checkpoints.
    Explicit ``--start`` files need not use the controller's metadata schema.
    """
    cands = []
    try:
        paths = sorted(Path(results_dir).glob("cand_*.json"))
    except OSError:
        return None
    for js in paths:
        meta = _read_meta(js)
        if not meta:
            continue
        d = meta.get("details") or {}
        if d.get("track") != track or not _verified_metadata(meta):
            continue
        if bool(meta.get("rotated_integrals", False)) != (track == "B"):
            continue
        npz = js.with_suffix(".npz")
        if not npz.exists():
            continue
        try:
            with np.load(npz) as z:
                required = {"strings_a", "strings_b", "amplitudes", "hcore", "eri", "ecore"}
                if not required.issubset(z.files):
                    continue
                a, b = z["strings_a"], z["strings_b"]
                if any(s.ndim != 1 or not 0 < s.size <= CAP
                       or np.any(np.diff(s) <= 0) for s in (a, b)):
                    continue
                if z["amplitudes"].shape != (a.size, b.size):
                    continue
        except (OSError, ValueError, KeyError):
            continue
        cands.append((float(d["grader_energy"]), str(meta.get("timestamp_utc", "")),
                      npz, meta))
    if not cands:
        return None
    rmin = min(c[0] for c in cands)
    tied = [c for c in cands if c[0] == rmin]
    _, _, npz, meta = max(tied, key=lambda c: c[1])
    return npz, meta


def _canonical_origin(npz: Path, arrays: dict, meta: dict):
    """Use archived/A-parent originals, or invert an external file's saved cumulative U."""
    d = meta.get("details") or {}
    if "canonical_hcore" in arrays and "canonical_eri" in arrays:
        return (np.asarray(arrays["canonical_hcore"], dtype=float),
                np.asarray(arrays["canonical_eri"], dtype=float),
                d.get("canonical_parent"))
    seen = {npz.resolve()}
    current, current_meta = npz, meta
    while True:
        details = current_meta.get("details") or {}
        parent = details.get("canonical_parent") or details.get("parent")
        if not parent:
            # Arbitrary scripts may save U and rotated integrals without our
            # provenance schema. Invert that saved basis, never today's cache.
            h0, g0 = rotate_integrals(arrays["hcore"], arrays["eri"],
                                      np.asarray(arrays["orbital_rotation"]).T)
            return h0, g0, None
        path = current.parent / Path(parent).with_suffix(".npz")
        if path.resolve() in seen:
            raise ValueError(f"cycle in canonical parent chain for {npz}")
        seen.add(path.resolve())
        parent_arrays, current_meta = load_candidate(path)
        if not current_meta.get("rotated_integrals", False):
            if float(parent_arrays["ecore"]) != float(arrays["ecore"]):
                raise ValueError(f"canonical parent ecore differs from {npz}")
            return (np.asarray(parent_arrays["hcore"], dtype=float),
                    np.asarray(parent_arrays["eri"], dtype=float), path.name)
        current = path


def _adopt_file(run: _Run, npz: Path, *, source: str) -> None:
    """Resume the exact saved basis; verify external files without sufficient evidence."""
    arrays, meta = load_candidate(npz)
    a = np.asarray(arrays["strings_a"], dtype=np.int64)
    b = np.asarray(arrays["strings_b"], dtype=np.int64)
    if any(s.ndim != 1 or not 0 < s.size <= CAP or np.any(np.diff(s) <= 0)
           for s in (a, b)):
        raise ValueError(f"start candidate {npz} needs ascending unique capped strings")
    h, g = np.asarray(arrays["hcore"], dtype=float), np.asarray(arrays["eri"], dtype=float)
    core, norb = float(arrays["ecore"]), h.shape[0]
    u = np.asarray(arrays.get("orbital_rotation", np.eye(norb)), dtype=float)
    if u.shape != (norb, norb) or not np.all(np.isfinite(u)):
        raise ValueError(f"invalid saved orbital rotation in {npz}")
    if np.max(np.abs(u.T @ u - np.eye(norb))) > ORTHO_TOL:
        raise ValueError(f"nonorthogonal saved orbital rotation in {npz}")
    d = meta.get("details") or {}
    rotated = bool(meta.get("rotated_integrals", False) or d.get("track") == "B"
                   or not np.array_equal(u, np.eye(norb)))
    if run.track == "A":
        if (rotated or not np.array_equal(h, run.h0) or not np.array_equal(g, run.g0)
                or core != run.ecore):
            raise ValueError(f"track A resume {npz}: canonical integrals differ from build_n2() cache")
    elif rotated:
        if "orbital_rotation" not in arrays:
            raise ValueError(f"track B resume {npz} needs its saved cumulative orbital rotation")
        run.h0, run.g0, run.canonical_parent = _canonical_origin(npz, arrays, meta)
    else:
        run.h0, run.g0, run.canonical_parent = h, g, npz.name
    run.ecore, run.norb = core, norb
    if _verified_metadata(meta) and "amplitudes" in arrays:
        ritz, grader = float(d["ritz_energy"]), float(d["grader_energy"])
        c = np.asarray(arrays["amplitudes"], dtype=float)
        if c.shape != (a.size, b.size):
            raise ValueError(f"saved amplitudes do not match strings in {npz}")
        sol = Solution(a, b, c, ritz - core, True, None, None, None)
    else:
        if time.time() >= run.deadline:
            raise TimeoutError("deadline reached before verifying explicit start")
        sol = solve_space(a, b, h, g)
        if not sol.converged:
            raise ValueError(f"explicit start {npz} did not converge")
        ritz = float(sol.energy) + core
        facts = official_facts(a, b, h, g, core, ritz, deadline=run.deadline)
        grader = float(facts["check"]["energy"])
        if (not facts["matches"] or not facts["check"]["matches"]
                or not np.isfinite(grader) or abs(grader - ritz) > RITZ_MATCH_TOL):
            raise ValueError(f"explicit start {npz} failed verification")
    run.inc = {"sol": sol, "h": h, "g": g, "u": u, "ritz": ritz,
               "grader": grader, "submittable": True, "path": Path(npz)}
    run.log({"type": source, "file": Path(npz).name, "ritz": ritz,
             "grader": grader, "submittable": True, "dims": [len(a), len(b)]})
    run.say(f"{source}: {Path(npz).name} ritz={ritz:.6f} grader={grader:.6f}")


def _resolve_start(run: _Run) -> None:
    """Adopt an explicit --start file, or auto-discover the latest best."""
    if run.start:
        p = Path(run.start)
        if not p.exists():
            raise FileNotFoundError(f"--start not found: {p}")
        _adopt_file(run, p, source="explicit_start")
        return
    if run.track == "A":
        hit = best_track_candidate("A", run.results_dir)
        if hit is not None:
            _adopt_file(run, hit[0], source="resume")
    else:
        hit = best_track_candidate("B", run.results_dir)
        if hit is not None:
            _adopt_file(run, hit[0], source="resume")
            return
        hit = best_track_candidate("A", run.results_dir)
        if hit is not None:
            _adopt_file(run, hit[0], source="start_from_A")


def _run_seeds(run: _Run) -> None:
    """Solve the two cheap canonical seeds; each is a capped proposal."""
    h, g, u = run.h0, run.g0, np.eye(run.norb)
    seeds = (("seed_hfsd300", seed_hf_singles_doubles(run.norb)),
             ("seed_lex14_first300", seed_lex14_first300()))
    for kind, ab in seeds:
        if time.time() >= run.deadline:
            run.log({"type": "deadline", "phase": kind})
            break
        run.say(f"{kind}: solving {len(ab)}x{len(ab)}")
        t = time.perf_counter()
        try:
            sol = solve_space(np.asarray(ab, dtype=np.int64),
                              np.asarray(ab, dtype=np.int64), h, g)
        except Exception as err:  # noqa: BLE001
            run.log({"type": "seed_failed", "kind": kind,
                     "error": f"{type(err).__name__}: {err}"})
            continue
        run.consider(sol, h, g, u, kind=kind,
                     info={"solve_s": time.perf_counter() - t})


def _swap_tail(run: _Run, swap_search) -> None:
    """Bounded swap pass over the incumbent; missing sibling -> logged skip."""
    if swap_search is None:
        run.log({"type": "skip", "phase": "swap",
                 "reason": "n2sqd.selection.swap_search unavailable"})
        return
    if time.time() >= run.deadline:
        run.log({"type": "deadline", "phase": "swap"})
        return
    cur = run.inc
    try:
        swap_search(cur["sol"], cur["h"], cur["g"], symmetric=run.symmetric,
                    deadline=run.deadline,
                    on_proposal=run.bound(cur["h"], cur["g"], cur["u"], "swap"))
    except Exception as err:  # noqa: BLE001
        run.log({"type": "selection_error", "phase": "swap",
                 "error": f"{type(err).__name__}: {err}"})


def _selection_loop(run: _Run, select_strings, *, max_iters: int) -> None:
    """One deadline-bounded select_strings call from the current incumbent."""
    cur = run.inc
    cb = run.bound(cur["h"], cur["g"], cur["u"], "select")
    select_strings(cur["sol"], cur["h"], cur["g"], cap=CAP,
                   symmetric=run.symmetric, max_iters=max_iters,
                   deadline=run.deadline, on_proposal=cb)


def _run_track_A(run: _Run, select_strings, swap_search=None) -> None:
    _resolve_start(run)
    if run.inc is None:
        _run_seeds(run)
    if run.inc is None:
        run.log({"type": "abort", "reason": "no_incumbent_no_time"})
        return
    if select_strings is None:
        run.log({"type": "skip", "phase": "selection",
                 "reason": "n2sqd.selection unavailable"})
        return
    improved = True
    while improved and time.time() < run.deadline:
        before = run.inc["ritz"]
        try:
            _selection_loop(run, select_strings, max_iters=20)
        except Exception as err:  # noqa: BLE001
            run.log({"type": "selection_error", "error": f"{type(err).__name__}: {err}"})
            break
        improved = run.inc["ritz"] < before
    _swap_tail(run, swap_search)
    if time.time() >= run.deadline:
        run.log({"type": "deadline", "phase": "selection"})


def _rdm_ok(sol: Solution, h, g, ecore: float) -> bool:
    """RDM energy must reproduce the Ritz energy to 1e-8 (else skip rotation)."""
    if sol.rdm1 is None or sol.rdm2 is None:
        return False
    e_rdm = float(np.sum(h * sol.rdm1) + 0.5 * np.sum(g * sol.rdm2)) + float(ecore)
    return abs(e_rdm - (float(sol.energy) + float(ecore))) <= 1e-8


def _refresh(run: _Run):
    """Re-solve the incumbent with RDMs; routed through consider. None if no time."""
    if time.time() >= run.deadline:
        return None
    inc = run.inc
    t = time.perf_counter()
    try:
        sol = solve_space(inc["sol"].strings_a, inc["sol"].strings_b,
                          inc["h"], inc["g"], rdms=True)
    except Exception as err:  # noqa: BLE001
        run.log({"type": "refresh_failed",
                 "error": f"{type(err).__name__}: {err}"})
        return None
    run.consider(sol, inc["h"], inc["g"], inc["u"], kind="rdm_refresh",
                 info={"solve_s": time.perf_counter() - t})
    return sol


def _try_rotation(run: _Run, select_strings, reseed_rotated, prop,
                  *, base_sol: Solution, full_reseed: bool = True) -> None:
    """Contain each rotation proposal's failures without losing a verified incumbent."""
    if time.time() >= run.deadline:
        return
    try:
        _evaluate_rotation(run, select_strings, reseed_rotated, prop,
                           base_sol=base_sol, full_reseed=full_reseed)
    except Exception as err:  # noqa: BLE001
        run.log({"type": "rotation_failed", "method": str(getattr(prop, "method", "unknown")),
                 "error": f"{type(err).__name__}: {err}"})


def _evaluate_rotation(run: _Run, select_strings, reseed_rotated, prop,
                       *, base_sol: Solution, full_reseed: bool) -> None:
    """Evaluate fixed-label and scheduled full-overlap seeds, then two select iterations."""
    u_step = np.asarray(prop.u)
    if not np.isrealobj(u_step):
        raise ValueError("orbital rotation must be real")
    if u_step.shape != (run.norb, run.norb) or not np.all(np.isfinite(u_step)):
        run.log({"type": "rotation_rejected", "method": str(prop.method),
                 "reason": "u not finite real orthogonal-shaped"})
        return
    if float(np.max(np.abs(u_step.T @ u_step - np.eye(run.norb)))) > ORTHO_TOL:
        run.log({"type": "rotation_rejected", "method": str(prop.method),
                 "reason": "u not orthogonal"})
        return
    u_new = run.inc["u"] @ u_step
    h_new, g_new = rotate_integrals(run.h0, run.g0, u_new)
    run.log({"type": "rotation", "method": str(prop.method),
             "objective_before": float(prop.objective_before),
             "objective_after": float(prop.objective_after),
             "info": getattr(prop, "info", None)})
    run.say(f"rotation {prop.method}: fixed-RDM {float(prop.objective_before):.6f} "
            f"-> {float(prop.objective_after):.6f}")
    if time.time() >= run.deadline:
        return
    t = time.perf_counter()
    s1 = solve_space(base_sol.strings_a, base_sol.strings_b, h_new, g_new)
    run.say(f"evaluated rot_{prop.method}_sameset: "
            f"ritz={s1.energy + run.ecore:.9f} converged={s1.converged}")
    run.consider(s1, h_new, g_new, u_new, kind=f"rot_{prop.method}_sameset",
                 info={"solve_s": time.perf_counter() - t})
    if time.time() >= run.deadline:
        return
    if full_reseed and reseed_rotated is not None:
        try:
            res = reseed_rotated(base_sol, u_step, cap=CAP,
                                 symmetric=run.symmetric)
            a2, b2, retained = res[0], res[1], res[-1]
        except Exception as err:  # noqa: BLE001
            run.log({"type": "reseed_failed", "method": str(prop.method),
                     "error": f"{type(err).__name__}: {err}"})
            a2 = None
        if time.time() >= run.deadline:
            return
        if a2 is not None:
            t = time.perf_counter()
            s2 = solve_space(np.asarray(a2, dtype=np.int64),
                             np.asarray(b2, dtype=np.int64), h_new, g_new)
            run.say(f"evaluated rot_{prop.method}_reseed: "
                    f"ritz={s2.energy + run.ecore:.9f} converged={s2.converged}")
            run.consider(s2, h_new, g_new, u_new,
                         kind=f"rot_{prop.method}_reseed",
                         info={"solve_s": time.perf_counter() - t,
                               "retained_norm": float(retained)})
    if select_strings is not None and time.time() < run.deadline:
        cur = run.inc
        cb = run.bound(cur["h"], cur["g"], cur["u"], f"select_{prop.method}")
        try:
            select_strings(cur["sol"], cur["h"], cur["g"], cap=CAP,
                           symmetric=run.symmetric, max_iters=2,
                           deadline=run.deadline, on_proposal=cb)
        except Exception as err:  # noqa: BLE001
            run.log({"type": "selection_error",
                     "error": f"{type(err).__name__}: {err}"})


def _run_track_B(run: _Run, select_strings, orbitals, swap_search=None) -> None:
    _resolve_start(run)
    if run.inc is None:
        _run_seeds(run)
    if run.inc is None:
        run.log({"type": "abort", "reason": "no_incumbent_no_time"})
        return
    propose = _try_import("n2sqd.orbitals", "propose_rotation") \
        if orbitals is not None else None
    reseed = _try_import("n2sqd.orbitals", "reseed_rotated") \
        if orbitals is not None else None
    if propose is None or reseed is None:
        run.log({"type": "skip", "phase": "rotation",
                 "reason": "n2sqd.orbitals unavailable"})
        if select_strings is not None and time.time() < run.deadline:
            try:
                while time.time() < run.deadline:
                    before = run.inc["ritz"]
                    _selection_loop(run, select_strings, max_iters=20)
                    if not run.inc["ritz"] < before:
                        break
            except Exception as err:  # noqa: BLE001
                run.log({"type": "selection_error",
                         "error": f"{type(err).__name__}: {err}"})
        return
    # Natural-orbital branch, once.
    if time.time() < run.deadline:
        sol_r = _refresh(run)
        if time.time() >= run.deadline:
            return
        if sol_r is not None and sol_r.converged and _rdm_ok(
                sol_r, run.inc["h"], run.inc["g"], run.ecore):
            try:
                prop = propose(sol_r, run.inc["h"], run.inc["g"],
                               method="natural")
            except Exception as err:  # noqa: BLE001
                run.log({"type": "rotation_rejected", "method": "natural",
                         "reason": f"propose failed: {type(err).__name__}: {err}"})
                prop = None
            if prop is not None:
                _try_rotation(run, select_strings, reseed, prop,
                              base_sol=sol_r)
        else:
            run.log({"type": "skip", "phase": "natural_branch",
                     "reason": "rdm unavailable or inconsistent"})
    # ffsim macrocycles with plateau stop.
    poor, macro = 0, 0
    while time.time() < run.deadline and macro < MACRO_MAX:
        before = run.inc["ritz"]
        sol_r = _refresh(run)
        if sol_r is None or time.time() >= run.deadline:
            break
        if not sol_r.converged:
            run.log({"type": "skip", "phase": "rotation",
                     "reason": "refresh unconverged"})
            break
        if not _rdm_ok(sol_r, run.inc["h"], run.inc["g"], run.ecore):
            run.log({"type": "skip", "phase": "rotation",
                     "reason": "rdm does not reproduce ritz"})
            break
        try:
            prop = propose(sol_r, run.inc["h"], run.inc["g"], method="ffsim")
        except Exception as err:  # noqa: BLE001
            run.log({"type": "rotation_rejected", "method": "ffsim",
                     "reason": f"propose failed: {type(err).__name__}: {err}"})
            break
        _try_rotation(run, select_strings, reseed, prop, base_sol=sol_r,
                      full_reseed=(macro + 1) % 3 == 0)
        macro += 1
        gain = before - run.inc["ritz"]
        poor = poor + 1 if gain < MACRO_MIN_GAIN else 0
        if poor >= MACRO_POOR_STOP:
            run.log({"type": "plateau", "macros": macro, "poor": poor})
            break
    # Final selection with whatever time remains.
    if select_strings is not None and time.time() < run.deadline:
        try:
            while time.time() < run.deadline:
                before = run.inc["ritz"]
                _selection_loop(run, select_strings, max_iters=20)
                if not run.inc["ritz"] < before:
                    break
        except Exception as err:  # noqa: BLE001
            run.log({"type": "selection_error",
                     "error": f"{type(err).__name__}: {err}"})
    _swap_tail(run, swap_search)
    if time.time() >= run.deadline:
        run.log({"type": "deadline", "phase": "track_B"})


def run_search(track: str, *, seed: int = 2026, hours: float,
               start: str | None = None, symmetric: bool = True,
               threads: int = 8) -> Path:
    """Run a track search for `hours` hours; return the run record.

    Track ``"A"`` = canonical integrals, never rotated. Track ``"B"`` starts
    from the best verified A candidate (or ``start``) and rotates. Promotion
    verifies improving converged proposals with one local check and one grader
    reduce, then saves only matching, submittable improvements in grader energy.
    Without ``start``, resumes from the latest best verified candidate of the
    track, using its saved Hamiltonian rather than rebuilding its orbital basis.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(int(threads)))
    name = str(track).upper()
    if name not in ("A", "B"):
        raise ValueError(f"track must be 'A' or 'B', got {track!r}")
    run_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    deadline = t0 + float(hours) * 3600.0
    h0, g0, ecore, norb, nelec = build_n2()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = RUNS_DIR / f"{name}_{run_id}.jsonl"
    run = _Run(track=name, run_id=run_id, seed=int(seed),
               symmetric=bool(symmetric), h0=h0, g0=g0, ecore=ecore,
               start=start, results_dir=RESULTS_DIR, record_path=record,
               t0=t0, deadline=deadline)
    select_strings = _try_import("n2sqd.selection", "select_strings")
    orbitals = _try_import("n2sqd.orbitals")
    swap_search = _try_import("n2sqd.selection", "swap_search")
    run.log({"type": "run_start", "track": name, "run_id": run_id,
             "seed": int(seed), "hours": float(hours),
             "start": str(start) if start else None,
             "symmetric": bool(symmetric), "threads": int(threads),
             "norb": int(norb), "nelec": [int(nelec[0]), int(nelec[1])],
             "selection": select_strings is not None,
             "swap_search": swap_search is not None,
             "orbitals": orbitals is not None})
    run.say(f"start hours={hours} seed={seed} symmetric={symmetric} "
            f"threads={threads} record={record.name}")
    try:
        if name == "A":
            _run_track_A(run, select_strings, swap_search)
        else:
            _run_track_B(run, select_strings, orbitals, swap_search)
    finally:
        w = run.inc
        run.log({"type": "run_end", "wall_s": time.time() - t0,
                 "n_proposals": run.n,
                 "winner": {"file": w["path"].name, "ritz": w["ritz"],
                            "grader": w["grader"],
                            "submittable": w["submittable"],
                            "dims": [int(len(w["sol"].strings_a)),
                                     int(len(w["sol"].strings_b))]}
                 if w else None})
        if w is not None:
            run.say(f"end winner {w['path'].name} ritz={w['ritz']:.6f} "
                    f"grader={w['grader']:.6f} sub={w['submittable']}")
        else:
            run.say("end: no incumbent")
        run.close()
    return record
