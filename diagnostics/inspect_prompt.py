#!/usr/bin/env python
"""One instance through the panel, printed in full. A DIAGNOSTIC, not a runner.

Useful for eyeballing prompts and individual model behaviour before launching a
full run. Shows the built prompt, each model's raw output, its parsed verdict,
the aggregated panel decision, and the gold label.

It writes NOTHING -- no verdict file, no shard, no analysis -- which is why it
lives in diagnostics/ rather than among the run_* scripts, all of which
produce measurements.

`--condition` defaults to the dataset's own frame, like every other entry
point; an explicit value is honoured, since seeing what a different frame
renders is the point of the tool, and the condition it used is echoed above the
prompt either way.

Examples
--------
python diagnostics/inspect_prompt.py --dataset truthfulqa --index 5 --backend mock --synthetic
python diagnostics/inspect_prompt.py --dataset aggrefact_cnn --index 0 --condition def_on \
    --backend vllm      # the frame contrast: def_on scaffolding on a claim-support set
"""
from __future__ import annotations
import argparse

import numpy as np

from raim import (build_instances, synthetic_instances, build_prompt,
                  parse_verdict, make_backend, PANEL)
from raim.suite import condition_of
from raim.tasks import TASKS


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(TASKS))
    ap.add_argument("--index", type=int, default=0, help="instance index")
    ap.add_argument("--condition", default=None,
                    help="default: the dataset's own frame")
    ap.add_argument("--fmt", default="brief_reason")
    ap.add_argument("--paraphrase", default="p0")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    condition = args.condition or condition_of(args.dataset)

    if args.synthetic:
        task, instances = synthetic_instances(args.dataset, n=args.index + 1)
    else:
        task, instances = build_instances(args.dataset, limit=args.index + 1)
    x = instances[args.index]

    prompt = build_prompt(task, x, condition, args.fmt, args.paraphrase)
    print("=" * 70)
    print(f"{task.name}  uid={x.uid}  gold={x.gold} "
          f"({'HALLUCINATED' if x.gold else 'SUPPORTED'})  stratum={x.stratum}")
    print(f"condition={condition}  fmt={args.fmt}  paraphrase={args.paraphrase}")
    print("=" * 70)
    print("PROMPT:\n" + prompt)
    print("=" * 70)

    meta = [dict(uid=x.uid, gold=x.gold, stratum=x.stratum,
                 condition=condition, fmt=args.fmt,
                 paraphrase=args.paraphrase)]
    votes = []
    for model in PANEL:
        backend = make_backend(args.backend, model)
        txt = backend.generate([prompt], meta)[0]
        pred = parse_verdict(txt)
        votes.append(pred)
        label = {1: "HALLUCINATED", 0: "SUPPORTED", None: "ABSTAIN"}[pred]
        snippet = txt.strip().replace("\n", " ")[:80]
        print(f"{model:<40} -> {label:<13} | {snippet}")
        del backend

    valid = [v for v in votes if v is not None]
    if valid:
        p = float(np.mean(valid)); panel = int(p >= 0.5)
        print("=" * 70)
        print(f"PANEL: {'HALLUCINATED' if panel else 'SUPPORTED'}  "
              f"(agreement={max(p, 1 - p):.2f}, {len(valid)}/{len(PANEL)} voted)  "
              f"| GOLD: {'HALLUCINATED' if x.gold else 'SUPPORTED'}  "
              f"| {'CORRECT' if panel == x.gold else 'WRONG'}")


if __name__ == "__main__":
    main()