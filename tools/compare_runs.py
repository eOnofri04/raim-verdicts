#!/usr/bin/env python3
"""Compare two verdict trees measured on different boxes.

The question this answers: a re-run on new hardware produced new verdicts — are
they the same measurement, or a different one?

They will not be identical, and that is expected rather than a fault. Inference is
not bit-reproducible across GPU model or vLLM version, which is precisely why the
repositories were split on reproducibility in the first place. So "identical" is
the wrong test, and applying it would either fail always or tempt someone into
ignoring a real regression.

What must hold exactly, and is checked as such:

  the instance set   same identifiers, same count, same gold labels. A difference
                     here is a dataset problem, not an inference one, and it
                     invalidates every comparison below.

What should hold closely, and is reported with numbers rather than a verdict:

  agreement          per member, the fraction of shared instances on which the two
                     runs return the same verdict. Greedy decoding on the same
                     prompt should agree overwhelmingly; anything below ~0.95 says
                     the model, the template or the truncation behaviour moved.
  competence         per member, Cohen's kappa against gold in each run. The
                     agreement rate can look healthy while both runs drift the same
                     way, so the quantity the paper actually reports is compared
                     too.
  coverage           per member, the fraction with a parseable verdict. A parser
                     that stopped matching a newer model's phrasing shows up here
                     and nowhere else.

Usage:
    ./tools/compare_runs.py --a runs --b /path/to/other/runs
    ./tools/compare_runs.py --a runs --b other/runs --dataset aggrefact_cnn
    ./tools/compare_runs.py --a runs --b other/runs --min-agreement 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _kappa(yt, yp) -> float:
    yt, yp = np.asarray(yt), np.asarray(yp)
    if len(yt) == 0 or len(set(yt.tolist())) < 2:
        return float("nan")
    labels = np.unique(np.concatenate([yt, yp]))
    idx = {l: i for i, l in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)))
    for a, b in zip(yt, yp):
        C[idx[a], idx[b]] += 1
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(0) * C.sum(1)).sum() / (n * n)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def load(path: Path, condition: str | None = None):
    """votes[model][uid], gold[uid] for one verdict file, under its own frame."""
    rows = [json.loads(l) for l in path.open() if l.strip()]
    if condition is None:
        conds = {r.get("condition") for r in rows}
        condition = "def_on" if "def_on" in conds else sorted(conds)[0]
    votes, gold = defaultdict(dict), {}
    for r in rows:
        if r.get("condition") != condition:
            continue
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
    return votes, gold, condition


def compare(a: Path, b: Path, ds: str, min_agreement: float) -> bool:
    pa, pb = a / ds / "experiment.jsonl", b / ds / "experiment.jsonl"
    if not pa.exists() or not pb.exists():
        side = "a" if not pa.exists() else "b"
        print(f"\n=== {ds} ===\n  !! MISSING on side {side} -- the comparison is incomplete")
        return False
    va, ga, ca = load(pa)
    vb, gb, cb = load(pb)

    print(f"\n=== {ds} ===")
    ok = True

    # --- what must hold exactly ---
    if ca != cb:
        print(f"  !! FRAME DIFFERS: {ca} vs {cb} -- not comparable"); return False
    ua, ub = set(ga), set(gb)
    if ua != ub:
        print(f"  !! INSTANCE SETS DIFFER: {len(ua)} vs {len(ub)}, "
              f"{len(ua ^ ub)} not shared -- a dataset problem, not an inference one")
        ok = False
    shared = sorted(ua & ub)
    if any(ga[u] != gb[u] for u in shared):
        n = sum(ga[u] != gb[u] for u in shared)
        print(f"  !! GOLD LABELS DIFFER on {n} shared instances"); ok = False
    print(f"  frame {ca}, {len(shared)} shared instances, gold identical"
          if ok else f"  frame {ca}, {len(shared)} shared instances")

    # --- what should hold closely ---
    print(f"  {'member':45s} {'agree':>7s} {'kA':>7s} {'kB':>7s} {'dk':>7s} "
          f"{'covA':>6s} {'covB':>6s}")
    for m in sorted(set(va) & set(vb)):
        both = [u for u in shared
                if va[m].get(u) is not None and vb[m].get(u) is not None]
        agree = (sum(va[m][u] == vb[m][u] for u in both) / len(both)) if both else float("nan")
        ka = _kappa([ga[u] for u in both], [va[m][u] for u in both])
        kb = _kappa([gb[u] for u in both], [vb[m][u] for u in both])
        cova = sum(va[m].get(u) is not None for u in shared) / max(len(shared), 1)
        covb = sum(vb[m].get(u) is not None for u in shared) / max(len(shared), 1)
        flag = "" if not (agree == agree) or agree >= min_agreement else "  <-- LOW"
        print(f"  {m:45s} {agree:7.3f} {ka:+7.3f} {kb:+7.3f} {kb-ka:+7.3f} "
              f"{cova:6.3f} {covb:6.3f}{flag}")
        if agree == agree and agree < min_agreement:
            ok = False

    only = (set(va) ^ set(vb))
    if only:
        print(f"  !! members present on one side only: {sorted(only)}"); ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="reference verdict tree")
    ap.add_argument("--b", required=True, type=Path, help="the new one")
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--min-agreement", type=float, default=0.95,
                    help="flag a member below this per-item agreement (default 0.95)")
    args = ap.parse_args()

    names = args.dataset or sorted(
        p.parent.name for p in args.a.glob("*/experiment.jsonl"))
    if not names:
        sys.exit(f"!! no <dataset>/experiment.jsonl under {args.a}")

    allok = all([compare(args.a, args.b, ds, args.min_agreement) for ds in names])
    print("\n" + ("### comparable: instance sets identical, every member agrees "
                  f"at or above {args.min_agreement:.2f}."
                  if allok else
                  "### DIVERGENT -- see the flags above. Verdicts differing across "
                  "hardware is expected; instance sets differing is not, and a "
                  "member below the agreement floor means something moved."))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
