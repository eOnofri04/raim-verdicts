"""Join-key check (see the README, "The datasets and the join").

Rebuilds each dataset and compares the reconstructed keys against ground truth,
one line per dataset.

  .venv/bin/python3 tools/joincheck.py                    # offline, vs the committed index
  .venv/bin/python3 tools/joincheck.py --online           # fetch, but at the pinned revisions
  .venv/bin/python3 tools/joincheck.py --unpinned         # vs CURRENT upstream: the drift test
  .venv/bin/python3 tools/joincheck.py --vs-verdicts      # also compare against runs/ (laptop only)

Two ground truths, and which ones are available depends on the machine:

  * `dataset_index/` is COMMITTED, so it travels to any checkout. It is the
    authoritative mapping for the released verdicts (see the README, "The
    datasets and the join") and is therefore the reference this script uses
    by default.
  * `runs/` is gitignored and exists only where the runs were produced, so the
    verdict comparison is opt-in and silently skipped when the tree is absent.

Three strengths of check, and it is worth knowing which one you are running.

  (default)   Rebuild from the local cache and compare. Tests only that the
              builder is deterministic -- the weakest claim, but the honest one
              on a machine that already holds the snapshot.
  --online    Fetch, at the revisions raim/tasks.py pins. Tests that those pins
              still RESOLVE and still yield the committed mapping.
  --unpinned  Fetch at each source's current default branch, ignoring the pins.
              Tests whether upstream has MOVED since the reported runs. Implies
              --online, and is the interesting one on a COLD cache.

The distinction matters because the revisions have been pinned since
2026-08-04: from then on `--online` resolves the pinned commit and cannot see
past it, so whether upstream has moved is asked by `--unpinned` alone
(raim.tasks.revision_of).

`src_sha1` is the decisive column throughout: it hashes the item's own text, so
an upstream edit that leaves the uid in place still shows up here.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dataset_index import DATASETS, HERE, content_key, set_offline

# Every path resolves from the repository root, not the working directory.
INDEX = HERE / "dataset_index"
RUNS = HERE / "runs"


def load_index(ds: str) -> dict | None:
    p = INDEX / f"{ds}.jsonl"
    if not p.exists():
        return None
    out = {}
    with p.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["uid"]] = (r["row_id"], r["gold"], r.get("stratum"),
                             r.get("category"), r["src_sha1"], r.get("kind"))
    return out


def load_verdicts(ds: str) -> dict | None:
    p = RUNS / ds / "experiment.jsonl"
    if not p.exists():
        return None
    out = {}
    with p.open() as fh:
        for line in fh:
            r = json.loads(line)
            out.setdefault(r["uid"], (r["row_id"], r["gold"], r.get("stratum"),
                                      r.get("category")))
    return out


def compare(built: dict, ref: dict, fields: tuple[str, ...]) -> str:
    """Return an empty string when the two agree, else a compact diagnosis."""
    only_built, only_ref = set(built) - set(ref), set(ref) - set(built)
    shared = set(built) & set(ref)
    per_field = defaultdict(int)
    for u in shared:
        for i, name in enumerate(fields):
            if built[u][i] != ref[u][i]:
                per_field[name] += 1
    bits = []
    if only_built or only_ref:
        bits.append(f"uid +{len(only_built)}/-{len(only_ref)}")
    if per_field:
        bits.append(" ".join(f"{k} {v}" for k, v in sorted(per_field.items())))
    return ", ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true",
                    help="allow Hugging Face to fetch, at the pinned revisions; "
                         "without this the local cache is used and no network "
                         "call is made")
    ap.add_argument("--unpinned", action="store_true",
                    help="ignore DATASET_REVISIONS and resolve each source at its "
                         "current default branch: the upstream-drift test. "
                         "Implies --online")
    ap.add_argument("--vs-verdicts", action="store_true",
                    help="additionally compare against runs/<ds>/experiment.jsonl "
                         "(needs the run tree, which is gitignored)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    # Set before any builder runs: revision_of reads it per call, and a
    # half-pinned rebuild would be meaningless.
    if args.unpinned:
        args.online = True
        os.environ["RAIM_IGNORE_DATASET_PINS"] = "1"

    set_offline(not args.online)

    from raim.tasks import build_instances  # repo root already on sys.path (dataset_index)

    mode = ("UNPINNED (current default branch -- the drift test)" if args.unpinned
            else "ONLINE (at the pinned revisions)" if args.online
            else "offline (local cache)")
    print(f"\njoincheck: {mode}; reference = committed dataset_index/"
          + (" + runs/" if args.vs_verdicts else ""))

    rows, failures = [], 0
    for ds in args.datasets:
        idx = load_index(ds)
        if idx is None:
            rows.append((ds, "NO INDEX", "run dataset_index.py first", ""))
            failures += 1
            continue
        try:
            _task, inst = build_instances(ds)
        except Exception as e:
            rows.append((ds, "BUILD FAILED", f"{type(e).__name__}: {e}"[:56], ""))
            failures += 1
            continue

        built = {x.uid: (x.row_id, x.gold, x.stratum, x.category, content_key(x),
                         x.kind)
                 for x in inst}
        diag = compare(built, idx,
                       ("row_id", "gold", "stratum", "category", "src_sha1", "kind"))

        vdiag = ""
        if args.vs_verdicts:
            ver = load_verdicts(ds)
            if ver is None:
                vdiag = "no runs/"
            else:
                vb = {u: v[:4] for u, v in built.items()}
                vdiag = compare(vb, ver, ("row_id", "gold", "stratum",
                                          "category")) or "match"

        ok = not diag
        failures += 0 if ok else 1
        rows.append((ds, "MATCH" if ok else "DRIFT",
                     diag or f"n={len(built)}", vdiag))

    w = max(len(r[0]) for r in rows)
    print(f"\n{'dataset':<{w}}  {'vs index':<9}  {'detail':<46}"
          + ("  vs verdicts" if args.vs_verdicts else ""))
    print("-" * (w + 60))
    for ds, verdict, detail, vdiag in rows:
        print(f"{ds:<{w}}  {verdict:<9}  {detail:<46}"
              + (f"  {vdiag}" if args.vs_verdicts else ""))

    if failures:
        print(f"\n{failures} dataset(s) did not reproduce the committed index.")
        if args.unpinned:
            print("Under --unpinned this means UPSTREAM HAS MOVED since the reported "
                  "runs.\nThe committed index remains authoritative for the released "
                  "verdicts;\nwhat needs recording is which datasets drifted, so the "
                  "release documents it.")
        elif args.online:
            print("Under --online this means a PINNED revision no longer yields what "
                  "it did,\nwhich is more serious than drift: the pin itself is not "
                  "holding. Re-run\nwith --unpinned to see what upstream now serves.")
    else:
        print("\nAll datasets reproduce the committed index exactly"
              + (", at the current default branch." if args.unpinned
                 else ", at the pinned revisions." if args.online
                 else " from the local cache."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
