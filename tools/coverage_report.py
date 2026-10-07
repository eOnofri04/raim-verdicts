#!/usr/bin/env python3
"""Consolidated post-hoc coverage report across ALL datasets under runs/.

Reads only already-written JSONL — panel experiment shards (or a concatenated
experiment.jsonl) plus any judge_<tag>.jsonl — and recomputes coverage as
    coverage = #{pred is not None} / #rows
the same definition as raim/scoring.py (cluster_bootstrap). Writes one Markdown
report, which records the host it ran on and is therefore gitignored. Imports
nothing from the raim package, so it is safe against an un-reconciled tree. No
GPU, no re-inference.

Usage:
    python tools/coverage_report.py                      # scan ./runs -> coverage_report.md
    python tools/coverage_report.py --runs /path/to/runs --out coverage_report.md
    python tools/coverage_report.py --expect 10          # expected panel size (flags missing members)
    python tools/coverage_report.py --no-judges          # panel experiment only
"""
import argparse, glob, json, os, socket
from collections import defaultdict
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="coverage_report.md")
ap.add_argument("--expect", type=int, default=10, help="expected panel size")
ap.add_argument("--no-judges", action="store_true")
args = ap.parse_args()


def aggregate(paths):
    """Group rows by the 'model' field -> [covered, total, set(uids)]."""
    agg = defaultdict(lambda: [0, 0, set()])
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                a = agg[r.get("model", "?")]
                a[0] += r.get("pred") is not None
                a[1] += 1
                a[2].add(r.get("uid"))
    return agg


def panel_paths(ds_dir):
    shards = sorted(glob.glob(os.path.join(ds_dir, "experiment_shards", "*.jsonl")))
    if shards:
        return shards, "shards"
    concat = os.path.join(ds_dir, "experiment.jsonl")
    if os.path.isfile(concat):
        return [concat], "experiment.jsonl"
    return [], None


runs = args.runs
datasets = sorted(
    d for d in (os.listdir(runs) if os.path.isdir(runs) else [])
    if os.path.isdir(os.path.join(runs, d)) and not d.startswith("_")
)

lines = []
incomplete = []          # (dataset, source, model, cov)
missing = []             # (dataset, present, expected)
n_sources = 0

for ds in datasets:
    ds_dir = os.path.join(runs, ds)
    ppaths, psrc = panel_paths(ds_dir)
    jpaths = [] if args.no_judges else sorted(glob.glob(os.path.join(ds_dir, "judge_*.jsonl")))
    if not ppaths and not jpaths:
        continue

    rows = []  # (source, model, cov, n, uids)
    panel_models = 0
    max_uids = 0

    if ppaths:
        for m, (c, n, u) in sorted(aggregate(ppaths).items()):
            cov = c / n if n else float("nan")
            rows.append(("panel", m, cov, n, len(u)))
            panel_models += 1
            max_uids = max(max_uids, len(u))
            if cov < 0.999:
                incomplete.append((ds, "panel", m, cov))
        n_sources += 1

    for jp in jpaths:
        tag = os.path.basename(jp)[len("judge_"):-len(".jsonl")]
        for m, (c, n, u) in sorted(aggregate([jp]).items()):
            cov = c / n if n else float("nan")
            rows.append((f"judge:{tag}", m, cov, n, len(u)))
            if cov < 0.999:
                incomplete.append((ds, f"judge:{tag}", m, cov))
        n_sources += 1

    if ppaths:
        missing.append((ds, panel_models, args.expect))

    # per-dataset section
    note = f"panel members present: **{panel_models}/{args.expect}**" if ppaths else "_no panel experiment found_"
    src_note = f" · panel source: {psrc}" if psrc else ""
    lines.append(f"## {ds}\n")
    lines.append(f"{note} · max distinct uids: {max_uids}{src_note}\n")
    lines.append("| source | model | coverage | n | uids |")
    lines.append("|---|---|---:|---:|---:|")
    for src, m, cov, n, u in rows:
        flag = "" if cov >= 0.999 else " ⚠"
        lines.append(f"| {src} | `{m}` | {cov:.3f}{flag} | {n} | {u} |")
    lines.append("")

# header + summary
host = socket.gethostname()
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
head = []
head.append("# RAIM — coverage report (all datasets)\n")
head.append(
    f"_Generated {ts} on `{host}`._\n"
    f"_Coverage is `#{{pred is not None}} / #rows`, the raim/scoring.py definition, "
    f"computed directly from written JSONL ({'panel shards + judges' if not args.no_judges else 'panel shards only'}); "
    f"no re-inference, no `raim` import._\n"
    f"_Source root: `{os.path.abspath(runs)}` · datasets scanned: {len(datasets)} · sources: {n_sources}._\n"
)
head.append("## Summary\n")
if incomplete:
    head.append(f"**Incomplete coverage ({len(incomplete)}):**\n")
    for ds, src, m, cov in incomplete:
        head.append(f"- {ds} · {src} · `{m}` — {cov:.3f}")
    head.append("")
else:
    head.append("**Incomplete coverage:** none — every source at 1.000.\n")
short = [(ds, p, e) for ds, p, e in missing if p < e]
if short:
    head.append("**Panel members below expected count (e.g. pending/failed members):**\n")
    for ds, p, e in short:
        head.append(f"- {ds}: {p}/{e} present")
    head.append("")
else:
    head.append(f"**Panel completeness:** every scanned dataset has all {args.expect} members present.\n")
head.append("---\n")

with open(args.out, "w") as fh:
    fh.write("\n".join(head + lines).rstrip() + "\n")
print(f"wrote {args.out}: {len(datasets)} datasets, {n_sources} sources, "
      f"{len(incomplete)} incomplete, {len(short)} datasets short of {args.expect} members")
