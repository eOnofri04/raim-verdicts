#!/usr/bin/env python
"""Forced-choice log-prob scoring of one model, for the base-vs-instruct probe.

Scores ONE model on ONE dataset by reading P(Yes) vs P(No) as the first
continuation token of a few-shot completion prompt — no instruction-following,
no generation, no parsing, 100% coverage. Works identically for base and
instruct models, so base-vs-instruct is the only variable. Like run_judge.py,
one model per invocation (VRAM isolated per subprocess by scripts/run_scorer.sh),
writing the SAME experiment-JSONL schema so the analysis consumes it unchanged.

Convention: the few-shot question asks "is the candidate SUPPORTED / truthful?"
so Yes = good = gold 0 and No = gold 1. We store pred in gold-space.

Frame: the few-shot block is claim-support in wording ("is the claim fully
supported"), and it prints a "Question:" line, which on the aggrefact sets
duplicates the claim verbatim (their builders set question == candidate == claim).
--frame auto (default) resolves to claimcheck on the aggrefact sets and drops
that duplicated line (a GENUINE question, where present, is kept — mirroring
prompts._build_claimcheck_prompt); qa keeps the line. The resolved frame also
sets the recorded `condition` (claimcheck / def_on) unless --condition is given
explicitly, and is stored per row for the downstream frame gates. Changing the
frame changes the prompt, so member files scored under different frames carry
different --tag values.

Usage: scripts/run_scorer.sh drives this over the matched base and instruct
twins; a single member is
  python scripts/run_lpscore.py --dataset medhallu --model meta-llama/Llama-3.1-8B \
      --tag llama_base --backend vllm --k 6 --gpu-util 0.55 --max-model-len 4096
and tools/assemble_panel.py joins the member files into one panel file behind
the completeness gate.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from raim import guard
from raim.tasks import build_instances, synthetic_instances
from raim.backends import make_backend, SCORER_REV
from raim.prompts import resolve_frame


def pick_exemplars(instances, k, seed=0):
    """k balanced exemplars (k/2 gold0 + k/2 gold1), removed from the eval set.

    Exclusion is by ROW, not uid: pos/neg twins share a row_id (same question +
    context), so if an exemplar showed the correct answer for row 17, the twin
    `17:neg` must not be evaluated — the prompt would leak its answer.
    """
    rng = random.Random(seed)
    pos = [x for x in instances if x.gold == 0]
    neg = [x for x in instances if x.gold == 1]
    rng.shuffle(pos); rng.shuffle(neg)
    half = k // 2
    ex = pos[:half] + neg[:half]
    ex_rows = {x.row_id for x in ex}
    rng.shuffle(ex)
    evalset = [x for x in instances if x.row_id not in ex_rows]
    return ex, evalset


EXEMPLAR_CTX_CHARS = 600    # exemplars only demonstrate the format -> trim hard
QUERY_CTX_CHARS = 2400      # the judged item's evidence -> preserve as much as fits
# Budget check (Yi-1.5-9B = 4096 ctx, k=6): 6*(150+100) + (600+100) + head ~= 2.2k tok.


def fmt_block(task, x, answer=None, is_query=False, frame="qa"):
    """One completion block. Ends at 'Answer:' when answer is None (the query).

    Context may be a str (MedHallu) or a list (RAGTruth/AggreFact); normalise to a
    list first, exactly like prompts._context_block, then trim. The query keeps
    far more context than exemplars, so the verdict still sees its evidence.
    Under frame="claimcheck" a Question line that merely copies the claim is
    dropped (the aggrefact duplication artefact); a genuine question is kept.
    """
    lines = []
    if task.has_context and x.context:
        ctx = x.context
        if isinstance(ctx, str):                 # MedHallu Knowledge is a bare str
            ctx = [ctx]
        ctx_str = "\n".join(c.strip() for c in ctx if c and c.strip())
        cap = QUERY_CTX_CHARS if is_query else EXEMPLAR_CTX_CHARS
        if len(ctx_str) > cap:
            ctx_str = ctx_str[:cap] + " ..."
        lines.append(f"{task.context_label}: {ctx_str}")
    if x.question and not (frame == "claimcheck"
                           and x.question.strip() == x.candidate.strip()):
        lines.append(f"Question: {x.question}")
    lines.append(f"Claim: {x.candidate}")
    tail = "Answer:" if answer is None else f"Answer: {answer}"
    lines.append(tail)
    return "\n".join(lines)


def build_fewshot(task, exemplars, x, frame="qa"):
    head = ("Decide whether each claim is fully supported by the information "
            "given. Answer Yes or No.")
    shots = []
    for e in exemplars:
        shots.append(fmt_block(task, e, answer="Yes" if e.gold == 0 else "No",
                               frame=frame))
    query = fmt_block(task, x, answer=None, is_query=True, frame=frame)
    return head + "\n\n" + "\n\n".join(shots) + "\n\n" + query + " "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--k", type=int, default=8, help="few-shot exemplars (balanced)")
    ap.add_argument("--frame", default="auto",
                    choices=["auto", "qa", "claimcheck"],
                    help="few-shot block scaffolding: auto (default) resolves "
                         "to claimcheck on the aggrefact sets (drops the "
                         "claim-duplicating Question line) and qa elsewhere; "
                         "qa reproduces the original v3 block byte-for-byte")
    ap.add_argument("--condition", default=None,
                    help="recorded condition label; default: claimcheck under "
                         "the claimcheck frame, def_on otherwise")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--out-dir", default=None,
                    help="default runs/lp_probe, or .smoke/lp_probe under "
                         "--backend mock or --synthetic")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file (default: skip)")
    args = ap.parse_args()

    # A mock or synthetic run tests the machinery and measures nothing, so it
    # defaults to the smoke tree and cannot occupy a canonical path.
    root = ".smoke" if args.synthetic or args.backend == "mock" else "runs"
    out_dir = Path(args.out_dir or f"{root}/lp_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{args.dataset}_lp_{args.tag}.jsonl"

    frame = resolve_frame(args.frame, args.dataset)
    condition = args.condition or ("claimcheck" if frame == "claimcheck"
                                   else "def_on")

    # The shared guard reads the condition each file records, so a mis-framed
    # member cannot be skipped as done, and --force cannot overwrite one.
    if not guard.check(raw, [condition], "lp", skip_on_match=True,
                       force=args.force, label=f"{args.tag}/{args.dataset}"):
        return

    if args.synthetic:
        task, instances = synthetic_instances(args.dataset, n=args.limit or 400)
    else:
        task, instances = build_instances(args.dataset, args.limit)

    exemplars, evalset = pick_exemplars(instances, args.k, seed=args.seed)
    prompts = [build_fewshot(task, exemplars, x, frame=frame) for x in evalset]
    meta = [dict(uid=x.uid, gold=x.gold) for x in evalset]

    kw = dict(gpu_util=args.gpu_util, max_model_len=args.max_model_len,
              dtype=args.dtype, tensor_parallel=args.tensor_parallel)
    be = make_backend(args.backend, args.model, **kw)
    print(f"[{task.name}] lp-score model={args.model} ({args.backend}) "
          f"k={args.k} over {len(evalset)} instances "
          f"(frame={frame}, cond={condition}) ...")
    scored = be.score_choice(prompts, meta=meta)

    raw.parent.mkdir(parents=True, exist_ok=True)
    gold, rowid_of, pred = {}, {}, {}
    with raw.open("w") as fh:
        for x, (choice, p_yes) in zip(evalset, scored):
            # Yes (choice=1) -> supported -> gold 0; No -> gold 1; None -> abstain
            g_pred = None if choice is None else (0 if choice == 1 else 1)
            gold[x.uid] = x.gold; rowid_of[x.uid] = x.row_id; pred[x.uid] = g_pred
            fh.write(json.dumps(dict(
                uid=x.uid, row_id=x.row_id, kind=x.kind, gold=x.gold,
                stratum=x.stratum, category=x.category, model=args.model,
                condition=condition, frame=frame, fmt="lp_fewshot",
                paraphrase="p0", scorer=SCORER_REV, pred=g_pred,
                prob_yes=p_yes)) + "\n")

    uids = sorted(gold)
    cov = sum(pred[u] is not None for u in uids) / len(uids)
    yt = [gold[u] for u in uids if pred[u] is not None]
    yp = [pred[u] for u in uids if pred[u] is not None]
    k = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 else float("nan")
    print(f"  wrote {raw}  coverage={cov:.3f}  kappa={k:.3f}  (n={len(uids)})")


if __name__ == "__main__":
    main()