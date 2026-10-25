"""Canonical-orbital (track A) string selection from a chosen seed set.

First used on 2026-10-23 (seed hfsd); its best candidate (-108.914953 Ha) is the
parent of the production runs. Later used as a multi-start
probe from other seeds. Runs `select_strings` only; every capped proposal that improves
this run's best is verified with the grader's own local reduce and saved immutably as
a track-A candidate. Canonical integrals come from the verified HF+SD candidate file
(same gauge as all track-A results).

    python scripts/interim_select_A.py --seed hfsd|lex14|bysum [--n-add 100] [--hours 6]
                                      [--prune marginal|energy]
"""
import argparse
import glob
import itertools
import json
import time
from pathlib import Path

import numpy as np
from qfest_grader.challenges.warsaw_2026 import lab4b

from n2sqd.candidates import load_candidate, save_candidate
from n2sqd.optimise import seed_hf_singles_doubles, seed_lex14_first300
from n2sqd.selection import select_strings
from n2sqd.solver import solve_space


def seed_bysum300(norb: int = 26) -> np.ndarray:
    combos = sorted(itertools.combinations(range(norb), 5), key=lambda c: (sum(c), c))[:300]
    return np.array(sorted(sum(1 << p for p in c) for c in combos), dtype=np.int64)


SEEDS = {"hfsd": seed_hf_singles_doubles, "lex14": seed_lex14_first300, "bysum": seed_bysum300}

ap = argparse.ArgumentParser()
ap.add_argument("--seed", choices=sorted(SEEDS), default="hfsd")
ap.add_argument("--n-add", type=int, default=100)
ap.add_argument("--prune", choices=("marginal", "energy"), default="marginal",
                help="string removal ranking (default: marginal weight)")
ap.add_argument("--hours", type=float, default=6.0)
ap.add_argument("--start", help="continue from a canonical (track A) candidate .npz instead of a seed set")
ap.add_argument("--independent", action="store_true", help="independent alpha/beta string sets")
ap.add_argument("--kick", type=int, default=0,
                help="basin hop: drop this many random strings (never HF) from --start before selecting")
ap.add_argument("--kick-seed", type=int, default=0)
ap.add_argument("--start-strings", help=".npz with strings_a/strings_b only, read in the canonical gauge")
args = ap.parse_args()

path = sorted(glob.glob("results/*hfsd300*.npz"))[0]
arr, meta = load_candidate(path)
h, g, ecore = arr["hcore"], arr["eri"], float(arr["ecore"])
best = {}
if args.start:
    s_arr, s_meta = load_candidate(args.start)
    if s_meta.get("rotated_integrals") or not np.array_equal(s_arr["hcore"], h):
        raise SystemExit(f"{args.start} is not in the canonical track-A gauge")
    a, b = s_arr["strings_a"], s_arr["strings_b"]
    best["e"] = float(s_arr["energy"]) - ecore  # only improvements over the start are saved
    if args.kick:
        rng = np.random.default_rng(args.kick_seed)

        def _drop(s):
            pool = np.flatnonzero(s != 31)
            return np.delete(s, rng.choice(pool, size=args.kick, replace=False))

        a = _drop(a)
        b = a.copy() if not args.independent else _drop(b)
    sol = solve_space(a, b, h, g)
    label = "start"
elif args.start_strings:
    z = np.load(args.start_strings)
    a, b = np.asarray(z["strings_a"], dtype=np.int64), np.asarray(z["strings_b"], dtype=np.int64)
    if not args.independent and not np.array_equal(a, b):
        raise SystemExit("symmetric run needs strings_a == strings_b (use --independent)")
    sol = solve_space(a, b, h, g)
    # save only what beats the best track-A candidate on disk
    best_a = sorted(glob.glob("results/cand_Em108p9*_A_*.npz"), reverse=True)[0]
    best["e"] = float(load_candidate(best_a)[0]["energy"]) - ecore
    label = "strings"
else:
    ab = SEEDS[args.seed]()
    sol = solve_space(ab, ab, h, g)
    label = args.seed
best["e"] = min(best.get("e", sol.energy), sol.energy)
t0 = time.time()
tag = "a_quick" if label == "hfsd" and args.n_add == 100 else f"a_seed_{label}_n{args.n_add}"
if args.prune != "marginal":
    tag += f"_{args.prune}"
if args.independent:
    tag += "_ind"
if args.kick:
    tag += f"_k{args.kick}s{args.kick_seed}"
log = open(Path(__file__).resolve().parent.parent / "runs" / f"{tag}.jsonl", "a")
print(f"{label}: {sol.energy + ecore:.6f} converged={sol.converged} dims={len(sol.strings_a)}x{len(sol.strings_b)}",
      flush=True)


def on_proposal(s, info):
    rec = {"t": time.time() - t0, "ritz": s.energy + ecore, "converged": s.converged,
           "dims": [len(s.strings_a), len(s.strings_b)], "info": {k: str(v) for k, v in info.items()}}
    if s.converged and len(s.strings_a) <= 300 and len(s.strings_b) <= 300 and s.energy < best["e"] - 1e-7:
        claim = s.energy + ecore
        facts = lab4b._exbonus_answer((claim, (s.strings_a, s.strings_b), h, g, ecore))
        rec["grader"] = facts
        if facts["matches"] and abs(facts["energy"] - claim) <= 1e-6:
            best["e"] = s.energy
            p = save_candidate(s.strings_a, s.strings_b, h, g, ecore, facts["energy"],
                               method=f"A_{tag}", seed=2026, rotated=False,
                               amplitudes=s.coeffs, orbital_rotation=np.eye(h.shape[0]),
                               details={"track": "A", "parent": path, "seed_set": args.seed,
                                        "ritz": claim, "grader": facts, "verified": True,
                                        "submittable": True})
            rec["saved"] = str(p)
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print(json.dumps(rec), flush=True)


out = select_strings(sol, h, g, cap=300, n_add=args.n_add, symmetric=not args.independent, max_iters=60,
                     prune_mode=args.prune, deadline=t0 + args.hours * 3600,
                     on_proposal=on_proposal)
print(f"done: best ritz {out.energy + ecore:.6f}", flush=True)
