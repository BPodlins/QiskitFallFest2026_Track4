"""Basin hopping for track B (rotated orbitals) from a verified B candidate.

Keeps the start file's orbital basis: rotated hcore/eri, the cumulative rotation U and the
canonical integrals it was applied to (checked: U applied to the canonical integrals must
reproduce the stored hcore/eri to 1e-10). Drops `--kick` random strings per side, then runs
`select_strings` (residual-ranked growth + pruning). Every capped proposal that beats the
start energy is re-checked with the grader's own local reduce and saved immutably as a
track-B candidate with the same U and canonical integrals.

    python scripts/basin_hop_B.py --start results/<B>.npz --kick 15 --kick-seed 1 \
        [--n-add 30] [--prune energy] [--symmetric] [--hours 1.2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from qfest_grader.challenges.warsaw_2026 import lab4b

from n2sqd.candidates import load_candidate, save_candidate
from n2sqd.check import rotate_integrals
from n2sqd.selection import select_strings
from n2sqd.solver import solve_space

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True)
ap.add_argument("--kick", type=int, default=15)
ap.add_argument("--kick-seed", type=int, default=0)
ap.add_argument("--n-add", type=int, default=30)
ap.add_argument("--prune", choices=("marginal", "energy"), default="energy")
ap.add_argument("--symmetric", action="store_true", help="keep alpha == beta (default: independent)")
ap.add_argument("--hours", type=float, default=1.2)
args = ap.parse_args()

arr, meta = load_candidate(args.start)
if not meta.get("rotated_integrals"):
    raise SystemExit(f"{args.start} is not a rotated (track B) candidate")
h, g, ecore = arr["hcore"], arr["eri"], float(arr["ecore"])
U, h0, g0 = arr["orbital_rotation"], arr["canonical_hcore"], arr["canonical_eri"]
hr, gr = rotate_integrals(h0, g0, U)
if max(np.abs(hr - h).max(), np.abs(gr - g).max()) > 1e-10:
    raise SystemExit("stored hcore/eri are not U applied to the stored canonical integrals")

rng = np.random.default_rng(args.kick_seed)


def drop(s):
    return np.delete(s, rng.choice(s.size, size=args.kick, replace=False)) if args.kick else s


a = drop(arr["strings_a"])
b = a.copy() if args.symmetric else drop(arr["strings_b"])
sol = solve_space(a, b, h, g)
# the stored "energy" is the grader's 6-decimal value; compare against the precise Ritz energy
best = {"e": float(meta.get("details", {}).get("ritz_energy", arr["energy"])) - ecore}
t0 = time.time()
tag = f"hopB_k{args.kick}s{args.kick_seed}_n{args.n_add}_{args.prune}" + ("_sym" if args.symmetric else "_ind")
run_out = Path(__file__).resolve().parent.parent / "runs"
run_out.mkdir(exist_ok=True)
log = open(run_out / f"{tag}.jsonl", "a")
print(f"start {args.start}: {float(arr['energy']):.6f}; kicked {sol.energy + ecore:.6f} "
      f"dims={len(a)}x{len(b)}", flush=True)


def on_proposal(s, info):
    rec = {"t": time.time() - t0, "ritz": s.energy + ecore, "converged": s.converged,
           "dims": [len(s.strings_a), len(s.strings_b)], "info": {k: str(v) for k, v in info.items()}}
    if s.converged and len(s.strings_a) <= 300 and len(s.strings_b) <= 300 and s.energy < best["e"] - 1e-7:
        claim = s.energy + ecore
        facts = lab4b._exbonus_answer((claim, (s.strings_a, s.strings_b), h, g, ecore))
        rec["grader"] = facts
        if facts["matches"] and abs(facts["energy"] - claim) <= 1e-6:
            best["e"] = s.energy
            p = save_candidate(
                s.strings_a, s.strings_b, h, g, ecore, facts["energy"], method=f"B_{tag}",
                seed=args.kick_seed, rotated=True, amplitudes=s.coeffs, orbital_rotation=U,
                canonical_hcore=h0, canonical_eri=g0,
                details={"track": "B", "parent": Path(args.start).name,
                         "canonical_parent": meta.get("details", {}).get("canonical_parent"),
                         "script": "scripts/basin_hop_B.py", "args": vars(args),
                         "ritz_energy": claim, "grader_energy": facts["energy"],
                         "grader_matches_ritz": True, "verified": True, "submittable": True,
                         "converged": True, "symmetric": args.symmetric})
            rec["saved"] = str(p)
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print(json.dumps(rec), flush=True)


out = select_strings(sol, h, g, cap=300, n_add=args.n_add, symmetric=args.symmetric, max_iters=60,
                     prune_mode=args.prune, deadline=t0 + args.hours * 3600, on_proposal=on_proposal)
print(f"done: best ritz {min(out.energy, best['e']) + ecore:.6f}", flush=True)
