#!/usr/bin/env python
"""Run a PURPOSE-TRAINED cheap judge over a dataset and score it.

The competing route to a cheap judge fine-tunes one small model expressly for
evaluation (Prometheus 2 / JudgeLM / Auto-J) rather than aggregating a panel of
off-the-shelf generalists. This runner pits one such judge, off the shelf,
against the same balanced binary faithfulness instances as the panel, through the
per-judge adapter in `raim/pt_judges.py`: the adapter renders each instance in the
judge's NATIVE graded interface and parses its native output back to our labels
(1 = hallucinated, 0 = supported), thresholding the graded score at the scale
midpoint (override with --threshold).

Output mirrors run_judge.py (both live in scripts/) so the result flows into the cost table and the
frontier comparison unchanged: runs/<ds>/judge_pt_<key>.{jsonl,json}, with κ and
balanced accuracy under the SAME paired cluster-bootstrap (B=2000, unit row_id,
non-answer counted as an error via coverage).

Frame: --frame auto (default) scores each dataset under its own frozen frame —
claim-support (claimcheck) on the aggrefact sets, qa elsewhere — via the
frame-aware adapters in raim/pt_judges.py; the resolved frame is recorded in
both output files ("frame": ...), which raim/guard.py and full_tables.py read.
A file carrying no "frame" key is accepted as qa on the non-aggrefact sets,
where qa is the only valid frame, and refused on the aggrefact sets.

Backends:
  vllm : the judge's own weights on local GPU(s); adapters own the conversation
         template, so generation runs with apply_template=False.
  mock : offline (no GPU) — emits each adapter's native output format so the
         parsers and the scoring path can be smoke-tested without weights.

Examples:
  # offline smoke test of the adapter + parser (no GPU, no network):
  .venv/bin/python3 scripts/run_ptjudge.py --dataset medhallu --judge prometheus2 \
      --backend mock --synthetic --limit 200 --B 200

  # real run (GPU box; autoj-13b wants a 48 GB card or --tensor-parallel 2):
  .venv/bin/python3 scripts/run_ptjudge.py --dataset medhallu --judge prometheus2 \
      --backend vllm --gpu-util 0.85
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from raim import build_instances, synthetic_instances, make_backend, resolve_frame
from raim import guard
from raim.pt_judges import get_judge, PT_JUDGES
from raim.scoring import cluster_bootstrap, report
from raim.tasks import TASKS


def run_ptjudge(dataset, judge_key, backend, *, tag=None, threshold=None,
                frame="auto", limit=None, synthetic=False, B=2000, seed=0,
                out_dir=None, gpu_util=0.85, tensor_parallel=1,
                max_model_len=None, force=False):
    """Run one purpose-trained judge over `dataset`; write JSONL + JSON, return dict.

    Guarded the same way `judge_one` (`run_judge.py`) is: an existing
    `judge_<tag>.json` recording a different frame refuses rather than being
    overwritten, whether this is called through `raim/legs.py`'s sweep or
    standalone.
    """
    judge = get_judge(judge_key)
    thr = judge.threshold if threshold is None else float(threshold)
    tag = tag or f"pt_{judge.key}"
    mml = judge.max_model_len if max_model_len is None else int(max_model_len)
    frame = resolve_frame(frame, dataset)
    # A mock or synthetic run tests the machinery and measures nothing, so it
    # defaults to the smoke tree and cannot occupy a canonical path.
    out_dir = out_dir or (".smoke" if synthetic or backend == "mock" else "runs")
    raw = Path(out_dir) / dataset / f"judge_{tag}.jsonl"
    if not guard.check(raw.with_suffix(".json"), [frame], "ptjudge",
                       skip_on_match=True, force=force, label=f"{tag}/{dataset}"):
        return None

    if synthetic:
        task, instances = synthetic_instances(dataset, n=limit or 400)
    else:
        task, instances = build_instances(dataset, limit)

    prompts = [judge.build(task, x, frame=frame) for x in instances]

    if backend == "mock":
        texts = [judge.mock(x.gold) for x in instances]
    else:
        be = make_backend(backend, judge.model_id, gpu_util=gpu_util,
                          tensor_parallel=tensor_parallel, max_model_len=mml,
                          max_new_tokens=judge.max_new_tokens, dtype=judge.dtype)
        print(f"[{task.name}] purpose-trained judge={judge.model_id} "
              f"({backend}) over {len(instances)} instances "
              f"(frame={frame}, threshold={thr}, scale={judge.score_range}) ...")
        texts = be.generate(prompts, meta=None, apply_template=False)

    raw.parent.mkdir(parents=True, exist_ok=True)
    gold, rowid_of, pred, score_of = {}, {}, {}, {}
    with raw.open("w") as fh:
        for x, txt in zip(instances, texts):
            s = judge.parse(txt)
            p = judge.label(s)
            gold[x.uid] = x.gold; rowid_of[x.uid] = x.row_id
            pred[x.uid] = p; score_of[x.uid] = s
            fh.write(json.dumps(dict(
                uid=x.uid, row_id=x.row_id, kind=x.kind, gold=x.gold,
                stratum=x.stratum, category=x.category, model=judge.model_id,
                judge=judge.key, frame=frame, score=s, threshold=thr, pred=p,
                raw=txt)) + "\n")

    uids = sorted(gold)
    scores = [score_of[u] for u in uids if score_of[u] is not None]
    k, ba, coverage = cluster_bootstrap(gold, pred, rowid_of, B=B, seed=seed)
    out = dict(dataset=task.name, judge=judge.key, model=judge.model_id,
               backend=backend, tag=tag, family="purpose_trained", frame=frame,
               score_range=list(judge.score_range), threshold=thr,
               n=len(uids), coverage=coverage,
               score_mean=(float(np.mean(scores)) if scores else None),
               parse_failures=len(uids) - len(scores),
               max_model_len=mml,
               kappa=k, balanced_accuracy=ba)
    json.dump(out, (raw.with_suffix(".json")).open("w"), indent=2)
    print(f"  wrote {raw} (+ .json)")
    report(k, ba, coverage,
           extra=f"  score_mean={('%.2f' % out['score_mean']) if scores else 'n/a'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TASKS))
    ap.add_argument("--judge", required=True, choices=sorted(PT_JUDGES))
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the midpoint score->label cut (score >= "
                         "threshold is SUPPORTED)")
    ap.add_argument("--frame", default="auto",
                    choices=["auto", "qa", "claimcheck"],
                    help="prompt scaffolding: auto (default) resolves to "
                         "claimcheck on the LLM-AggreFact sets and qa elsewhere, "
                         "matching the frozen per-dataset frames; an explicit "
                         "qa on an aggrefact set is a mis-framed measurement, "
                         "which scripts/run_judge.sh refuses to treat as done")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="override the judge's default window")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="output tree (default runs, or .smoke under --backend "
                         "mock or --synthetic; mirrors run_judge.py)")
    ap.add_argument("--force", action="store_true",
                    help="redo a measurement that already exists UNDER THE RIGHT "
                         "frame; it can never override a frame mismatch "
                         "(move the stale file aside by hand instead)")
    args = ap.parse_args()
    run_ptjudge(args.dataset, args.judge, args.backend, tag=args.tag,
                threshold=args.threshold, frame=args.frame, limit=args.limit,
                synthetic=args.synthetic, B=args.B, out_dir=args.out_dir,
                gpu_util=args.gpu_util, tensor_parallel=args.tensor_parallel,
                max_model_len=args.max_model_len, force=args.force)


if __name__ == "__main__":
    main()
