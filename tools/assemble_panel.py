#!/usr/bin/env python3
"""Assemble per-member scorer files into one gated panel JSONL.

The constrained scorer writes one file per (member, dataset). A panel is their
concatenation -- but only once every member has been checked, because a panel
silently missing a member, or carrying one scored under a stale prompt frame,
produces numbers that compute perfectly and mean nothing.

The gate, per member file:

  - the recorded model id matches the roster entry it is supposed to be;
  - the recorded condition matches the dataset's frozen frame (claimcheck on the
    LLM-AggreFact sets, def_on elsewhere), so no member scored under the other
    frame can enter the panel;
  - no duplicate instance identifiers;
  - coverage and, where required, continuous-confidence presence at or above the
    threshold;

and across members: identical instance-identifier sets, since a divergence means
the members were scored over different evaluation sets (seed or limit drift) and
the panel would be comparing judges on different items.

Only if every check passes is the panel written, atomically (a temporary file
renamed into place), so nothing partial is ever emitted and a failed run leaves
any previous panel untouched.

Its caller is scripts/run_scorer.sh, which assembles the base and the instruct
panel of the base-versus-instruct probe per dataset.

Usage:
    ./tools/assemble_panel.py --dataset aggrefact_xsum \
        --out runs/lp_probe/aggrefact_xsum_lpbase_v3_panel.jsonl \
        --members "meta-llama/Llama-3.1-8B|llama_base_v3" ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root; this file lives in tools/
from raim.suite import condition_of  # noqa: E402


def member_path(runs: Path, dataset: str, tag: str) -> Path:
    return runs / "lp_probe" / f"{dataset}_lp_{tag}.jsonl"


def check_member(path: Path, model: str, dataset: str, minimum: float,
                 require_prob: bool) -> tuple[list[dict], set[str]]:
    """Return (rows, uid set) for one member, or exit with the reason it failed."""
    try:
        rows = [json.loads(l) for l in path.open() if l.strip()]
    except FileNotFoundError:
        sys.exit(f"!! [{dataset}] missing member file {path} -- scoring incomplete")
    if not rows:
        sys.exit(f"!! [{dataset}] {path} is empty")

    ids = {r["model"] for r in rows}
    if ids != {model}:
        sys.exit(f"!! [{dataset}] {path} carries model={sorted(ids)}, expected {model}")

    want = condition_of(dataset)
    conds = {r.get("condition") for r in rows}
    if conds != {want}:
        sys.exit(f"!! [{dataset}] {path} carries condition={sorted(map(str, conds))}, "
                 f"expected {want} -- stale-frame member file?")

    uids = {r["uid"] for r in rows}
    if len(uids) != len(rows):
        sys.exit(f"!! [{dataset}] {path} has duplicate instance identifiers")

    cov = sum(r["pred"] is not None for r in rows) / len(rows)
    prob = sum(r.get("prob_yes") is not None for r in rows) / len(rows)
    print(f"    {model:<45} n={len(rows)}  coverage={cov:.3f}  prob_yes={prob:.3f}")
    if cov < minimum or (require_prob and prob < minimum):
        sys.exit(f"!! [{dataset}] coverage gate FAILED for {path} "
                 f"(threshold {minimum:.2f}) -- panel NOT written")
    return rows, uids


def assemble(dataset: str, out: Path, members: list[str], runs: Path,
             minimum: float = 0.95, require_prob: bool = True) -> None:
    collected, uidsets = [], []
    for spec in members:
        model, tag = spec.split("|")
        rows, uids = check_member(member_path(runs, dataset, tag), model, dataset,
                                  minimum, require_prob)
        collected.append(rows)
        uidsets.append(uids)

    if any(u != uidsets[0] for u in uidsets[1:]):
        sys.exit(f"!! [{dataset}] instance sets differ across members -- evaluation "
                 f"sets misaligned (seed or limit drift); panel NOT written")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with tmp.open("w") as fh:
            for rows in collected:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    print(f"    wrote {out}  ({len(collected)} members x {len(uidsets[0])} instances)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--members", required=True, nargs="+", metavar="HF_ID|TAG",
                    help="roster entries, in order; the tag names the member file")
    ap.add_argument("--runs", default=Path("runs"), type=Path,
                    help="root of the verdict tree (default: runs)")
    ap.add_argument("--min", type=float, default=0.95, dest="minimum",
                    help="coverage threshold below which the panel is not written")
    ap.add_argument("--no-prob-yes", action="store_true",
                    help="do not require continuous confidences (hard-vote panels)")
    args = ap.parse_args()

    assemble(args.dataset, args.out, args.members, args.runs,
             minimum=args.minimum, require_prob=not args.no_prob_yes)


if __name__ == "__main__":
    main()
