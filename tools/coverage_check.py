#!/usr/bin/env python3
"""Coverage gate over already-written panel or judge JSONL.

Reports, per model (and per arm with --by-condition), the fraction of rows
carrying a non-null `pred` -- the same definition scripts/run_judge.py uses --
and exits non-zero if any falls below the threshold, so a pipeline stops on a
contaminated panel rather than scoring it.  A path that matches no file, and
input holding no rows, fail the gate as well.

The gate exists because a scoring pathology can compute cleanly: a model whose
answers the scorer cannot read scores 0% coverage, and every number downstream
is still produced.  It must therefore announce itself BEFORE anything reads the
file.

Imports nothing from the raim package, so it stays usable against a tree whose
library is mid-edit. No GPU, no re-inference.

Usage:
    ./tools/coverage_check.py runs/medhallu/experiment.jsonl
    ./tools/coverage_check.py runs/medhallu/experiment_shards/*.jsonl
    ./tools/coverage_check.py --by-condition runs/<ds>/experiment_shards/*.jsonl
    ./tools/coverage_check.py --min 0.95 --also prob_yes runs/lp_probe/<ds>_lpbase_v3_panel.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def report(rows, *, by_condition=False, minimum=0.999, also=None):
    """Print coverage per key; return (key label, field, fraction) below `minimum`.

    The key is the model, or (model, condition) with `by_condition`; every
    count -- coverage, the `also` field, distinct instances -- is kept per key.
    """
    tot = defaultdict(lambda: [0, 0])   # key -> [covered, total]
    extra = defaultdict(int)            # key -> rows carrying `also`
    uids = defaultdict(set)             # key -> distinct instances
    for r in rows:
        key = (r["model"], r["condition"]) if by_condition else (r["model"],)
        tot[key][0] += r.get("pred") is not None
        tot[key][1] += 1
        uids[key].add(r["uid"])
        if also:
            extra[key] += r.get(also) is not None

    w = max(len(k[0]) for k in tot)
    print(f"{'model':<{w}}  {'condition':<16}  cover"
          f"{'  ' + also.rjust(6) if also else ''}      n   uids")

    flagged = []
    for key in sorted(tot):
        c, n = tot[key]
        cond = key[1] if by_condition else "(all arms)"
        label = f"{key[0]} [{cond}]" if by_condition else key[0]
        cov = c / n
        line = f"{key[0]:<{w}}  {cond:<16}  {cov:6.3f}"
        below = cov < minimum
        if below:
            flagged.append((label, "pred", cov))
        if also:
            pe = extra[key] / n
            line += f"  {pe:6.3f}"
            if pe < minimum:
                flagged.append((label, also, pe))
                below = True
        mark = "   <-- BELOW THRESHOLD" if below else ""
        print(f"{line}  {n:5d}   {len(uids[key])}{mark}")
    return flagged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="JSONL files or globs")
    ap.add_argument("--by-condition", action="store_true",
                    help="break the report down per arm as well as per model")
    ap.add_argument("--min", type=float, default=0.999, dest="minimum",
                    help="fail below this coverage (default 0.999, i.e. full)")
    ap.add_argument("--also", metavar="FIELD",
                    help="gate a second field's presence too, e.g. prob_yes")
    args = ap.parse_args()

    paths, unmatched = [], []
    for a in args.paths:
        hits = sorted(glob.glob(a))
        paths += hits
        if not hits:
            unmatched.append(a)
    if unmatched:
        sys.exit(f"!! coverage gate FAILED -- no file matches: {' '.join(unmatched)}")
    rows = load(paths)
    if not rows:
        sys.exit("!! coverage gate FAILED -- the input holds no rows")

    flagged = report(rows, by_condition=args.by_condition,
                     minimum=args.minimum, also=args.also)
    if flagged:
        print(f"\nBELOW {args.minimum:.3f}:")
        for label, field, frac in sorted(flagged):
            print(f"  {label} ({field}): {frac:.3f}")
        sys.exit(f"!! coverage gate FAILED -- stop and report; nothing downstream "
                 f"should read these files.")
    print(f"\nAll models at or above {args.minimum:.3f} coverage.")


if __name__ == "__main__":
    main()
