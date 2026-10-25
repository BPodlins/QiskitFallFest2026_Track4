"""Verify a saved candidate and (only with --send) submit it to the Lab 4b bonus grader.

Default is a local dry run: the local grader replica plus the grader's own local reduce
recompute the energy from the saved strings and integrals; nothing is sent.

    python scripts/submit_bonus.py results/<candidate>.npz                       # dry run
    python scripts/submit_bonus.py results/<candidate>.npz --send --email <GRADER_EMAIL>

--send registers the email in this (ephemeral) container with `save_account` and calls
`grade_lab4b_exbonus`, which recomputes the energy locally and sends the facts to the
grading server. Run it inside the n2-sqd image (the grader recomputes on this machine).
"""

from __future__ import annotations

import argparse
import sys

from n2sqd.candidates import load_candidate
from n2sqd.check import official_facts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("candidate", help="results/*.npz written by n2sqd.candidates")
    p.add_argument("--send", action="store_true", help="really submit (default: dry run)")
    p.add_argument("--email", help="grader account email (required with --send)")
    args = p.parse_args(argv)
    if args.send and not args.email:
        p.error("--send requires --email")

    arr, meta = load_candidate(args.candidate)
    a, b = arr["strings_a"], arr["strings_b"]
    h, g, ecore = arr["hcore"], arr["eri"], float(arr["ecore"])
    claim = float(arr["energy"])
    rotated = bool(meta.get("rotated_integrals", False))
    print(f"candidate: {args.candidate}")
    print(f"dims: {len(a)} x {len(b)}  claimed: {claim:.6f} Ha  rotated integrals: {rotated}", flush=True)

    facts = official_facts(a, b, h, g, ecore, claim)
    print(f"grader reduce (local): energy={facts['energy']} matches={facts['matches']} dims={facts['dims']}",
          flush=True)
    if not facts["matches"]:
        print("NOT submittable: claimed energy is below the recomputed one.", flush=True)
        return 1
    if not args.send:
        print("dry run: nothing sent (use --send --email <GRADER_EMAIL> to submit).")
        return 0

    from qfest_grader import save_account
    from qfest_grader.challenges.warsaw_2026 import grade_lab4b_exbonus

    save_account(args.email)
    grade_lab4b_exbonus(claim, (a, b), h, g, ecore)
    return 0


if __name__ == "__main__":
    sys.exit(main())
