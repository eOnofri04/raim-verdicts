#!/usr/bin/env python
"""The panel over one dataset.

Runs the ten-member panel (`raim.panel.PANEL`) over one dataset, writes one
verdict record per member, instance and arm, and analyses the result.

WHAT A RUN MEASURES is an ARM-SET (`raim.suite.ARMSETS`), which fixes both the
conditions swept and the file they land in, so the two cannot drift apart and
every verdict file is the output of exactly one command:

  experiment   one arm, the dataset's own frame       -> experiment.jsonl
  defon        the frame contrast, aggrefact only     -> experiment_defon.jsonl

FRAME-AWARE.  Each dataset is measured under its own frozen frame, resolved
from the dataset name (`raim.suite`): claim-support on the LLM-AggreFact
family, whose builders set question == candidate == claim so the question/answer
scaffolding would show the same string twice, and the definition frame
elsewhere.  An arm-set that has no meaning under a frame refuses rather than
measuring something vacuous.

GUARDED.  A path already holding a different arm-set is never overwritten --
different frame, subset or superset alike (`raim.guard`).  An interrupted run
resumes: the panel is shard-cached per member.  A mock or synthetic run defaults
to `.smoke/` rather than `runs/`, since it records the same arm-set as the real
measurement and the guard alone would let it replace the canonical file.

Examples
--------
python scripts/run.py --dataset medhallu                      # the default arm-set
python scripts/run.py --dataset aggrefact_cnn                 # resolves to claim-support
python scripts/run.py --dataset aggrefact_cnn --armset defon  # the frame contrast

python scripts/run.py --dataset medhallu --backend mock --synthetic   # offline smoke, into .smoke/
python scripts/run.py --dataset medhallu --analyze-only --raw runs/medhallu/experiment.jsonl

The reference judges are a separate measurement, run over the whole suite by
`scripts/run_judge.sh` (registry: `raim.legs`).
"""
from __future__ import annotations
import argparse
from pathlib import Path

from raim import build_instances, synthetic_instances, run_panel, Arm, PANEL
from raim import guard
from raim.analysis import load_rows, analyze_experiment
from raim.suite import ARMSETS, SUITE, armset as get_armset, conditions_for, frame_of
from raim.tasks import TASKS


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The registry holds three further LLM-AggreFact subsets that are buildable
    # but were never part of the reported suite; they stay selectable here, and
    # only here, so a diagnostic run can reach them.
    ap.add_argument("--dataset", required=True, choices=sorted(TASKS),
                    help=f"the suite is: {' '.join(SUITE)}")
    ap.add_argument("--armset", default="experiment", choices=sorted(ARMSETS),
                    help="what to measure and where it lands (default: experiment)")
    ap.add_argument("--conditions", default=None,
                    help="explicit comma-separated condition list, overriding the "
                         "arm-set's own. No file in the verdict tree needs it: "
                         "it is the escape hatch for a sweep no arm-set names")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap source rows")
    ap.add_argument("--synthetic", action="store_true",
                    help="use network-free synthetic instances (testing)")
    ap.add_argument("--synth-seed", type=int, default=None,
                    help="fix the synthetic stratum draw, so two smoke runs are "
                         "comparable; without it the draw varies per process")
    ap.add_argument("--raw", default=None,
                    help="JSONL path (default: runs/<ds>/<the arm-set's filename>; "
                         ".smoke/ in place of runs/ under --backend mock or --synthetic)")
    ap.add_argument("--outdir", default=None,
                    help="analysis dir (default: runs/<ds>/<armset>, or .smoke/ as for --raw)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip inference; analyse an existing --raw file")
    ap.add_argument("--gpu-util", type=float, default=0.40,
                    help="vLLM gpu_memory_utilization; raise to ~0.85 on a "
                         "dedicated card, raise further for large judges")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM max_model_len (prompt+generation cap)")
    ap.add_argument("--force", action="store_true",
                    help="recompute a cached shard whose recorded model "
                         "disagrees with the current PANEL order, instead of "
                         "stopping and naming the shard for manual deletion")
    args = ap.parse_args()

    aset = get_armset(args.armset)
    conditions = conditions_for(args.dataset, args.armset, args.conditions)
    arms = [Arm(condition=c, fmt=f, paraphrase=p)
            for c, f, p in aset.arms(conditions)]

    # A mock or synthetic run tests the machinery and measures nothing, yet it
    # records the same arm-set as the real run, so `guard.check` would pass it
    # and the changed manifest would clear the real shard cache.  Its default
    # therefore lands in the smoke tree; only an explicit --raw reaches runs/.
    root = Path(".smoke" if args.synthetic or args.backend == "mock" else "runs")
    raw = Path(args.raw or root / args.dataset / aset.filename)
    outdir = Path(args.outdir or root / args.dataset / args.armset)

    if not args.analyze_only:
        # skip_on_match=False: a matching file is not "done", it is a run to be
        # resumed -- the shard cache decides what still needs computing.
        guard.check(raw, conditions, "panel", skip_on_match=False,
                    label=f"{args.armset}/{args.dataset}")
        print(f"[{args.dataset}] arm-set {args.armset} "
              f"(frame={frame_of(args.dataset)}): {', '.join(conditions)}")
        if args.synthetic:
            task, instances = synthetic_instances(args.dataset, n=args.limit or 400,
                                                  seed=args.synth_seed)
        else:
            task, instances = build_instances(args.dataset, args.limit)
        run_panel(task, instances, arms, raw,
                  backend_kind=args.backend, panel=PANEL, force=args.force,
                  gpu_util=args.gpu_util, max_model_len=args.max_model_len)

    rows = load_rows(raw)
    analyze_experiment(rows, outdir)


if __name__ == "__main__":
    main()
