#!/usr/bin/env python
"""Grounded-factuality SPECIALIST verifiers (MiniCheck) over a dataset.

Why. The LLM-AggreFact leaderboard publishes balanced accuracies on the very test
rows four of our grounded sets are drawn from, and its strongest compact entries
(MiniCheck-Flan-T5-Large at 0.8B, Bespoke-MiniCheck-7B) sit in the panel members'
weight class — a reader's natural "why not one trained specialist instead of ten
generalists?" baseline, which the paper answers with data rather than argument.
This runner scores those two checkpoints on OUR exact balanced instances, so the
comparison lands in the purpose-trained-judge table beside Prometheus, JudgeLM
and Auto-J.

How. Inference goes through the OFFICIAL `minicheck` package, not our judge-prompt
harness: their prompt, their document chunking, their calibrated operating point.
That is deliberate — a specialist mis-prompted by us would understate the very
baseline we are pre-empting. Consequently this script does NOT use raim.backends,
and it must run in its own environment:

    python -m venv ~/minicheck-env && source ~/minicheck-env/bin/activate
    pip install "minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main"
    pip install accelerate
    pip install datasets scikit-learn        # for raim.tasks + the scoring below
    python -c "import nltk; nltk.download('punkt_tab')"

    DO NOT pip-install minicheck into the main .venv: the package pins its own
    vllm/torch and can break the panel/pt-judge environment mid-campaign.

Applicability. MiniCheck verifies a claim AGAINST A DOCUMENT, so only the grounded
sets apply (medhallu, ragtruth, aggrefact_*); the runner refuses truthfulqa and
factscore. For the aggrefact_* sets the instance already carries the native
LLM-AggreFact (doc, claim) pair, so the specialist sees exactly what the
leaderboard scored (modulo our class-balancing subsample). For the QA-form sets
(medhallu, ragtruth) the question is folded into the document — the same
convention LLM-AggreFact itself uses for its RAGTruth rows.

Known risk (bespoke leg). Bespoke-MiniCheck-7B is fine-tuned from
internlm2_5-7b-chat, whose chat template can emit special tokens above the
embedding table on size-respecting runtimes. The MiniCheck package drives it
through its own vLLM path; should that leg crash with an out-of-range token id,
the flan-t5 leg (transformers only) is unaffected, and the model must not be
re-prompted by hand.

Output mirrors run_ptjudge.py (both live in scripts/) so full_tables.tab_ptjudge picks it up unchanged:
runs/<ds>/judge_pt_<key>.{jsonl,json} with κ and balanced accuracy under the SAME
paired cluster-bootstrap (B=2000, unit row_id). `score` is the package's support
probability in [0,1]; `pred` uses the package's own label (1 = supported), mapped
to our gold convention (1 = hallucinated).

Examples:
  # offline smoke test of the mapping + scoring path (no GPU, no package):
  .venv/bin/python3 scripts/run_minicheck.py --dataset medhallu --model minicheck_ft5 \
      --backend mock --synthetic --limit 200 --B 200

  # real runs (minicheck venv; ft5 fits anywhere, bespoke wants one 24GB+ card):
  ~/minicheck-env/bin/python3 scripts/run_minicheck.py --dataset aggrefact_xsum --model minicheck_ft5
  ~/minicheck-env/bin/python3 scripts/run_minicheck.py --dataset aggrefact_xsum --model bespoke_minicheck
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from raim import build_instances, synthetic_instances
from raim import guard
from raim.scoring import cluster_bootstrap, report
from raim.suite import GROUNDED

# key -> (package model_name, jsonl model id for provenance)
MC_MODELS = {
    "minicheck_ft5":     ("flan-t5-large",        "lytang/MiniCheck-Flan-T5-Large"),
    "bespoke_minicheck": ("Bespoke-MiniCheck-7B", "bespokelabs/Bespoke-MiniCheck-7B"),
}

# The frame recorded in every output, so the shared guard applies here as it
# does to every other runner. It is a constant because MiniCheck has ONE frame
# by construction: the package verifies a claim against a document natively.
# A file carrying no frame key is therefore accepted on every dataset.
FRAME = "claimcheck"

# QA-form sets: the claim's support depends on the question, which is not part of
# the candidate text; LLM-AggreFact folds the question into `doc` for its RAGTruth
# rows, and we follow the same convention. The aggrefact_* builders already carry
# the native (doc, claim) pair (question == candidate there), so context alone is
# the document.
QA_STYLE = {"medhallu", "ragtruth"}


def _doc_and_claim(task, x):
    ctx = x.context
    if isinstance(ctx, str):
        ctx = [ctx]
    doc = "\n".join(c.strip() for c in (ctx or []) if c and c.strip())
    if task.name in QA_STYLE:
        doc = f"{x.question.strip()}\n\n{doc}"
    return doc, x.candidate.strip()


def run_minicheck(dataset, model_key, backend, *, limit=None, synthetic=False,
                  B=2000, seed=0, out_dir=None, cache_dir=None, force=False):
    """Score one MiniCheck checkpoint over `dataset`; write JSONL + JSON, return dict.

    Guarded like every other worker: FRAME is a constant here (MiniCheck has
    one frame by construction), so a mismatch can only mean a stray file from
    a different runner at this path, not a stale scoring frame -- but the
    check still runs.
    """
    if dataset not in GROUNDED:
        raise SystemExit(f"MiniCheck needs a grounding document; {dataset!r} is "
                         f"ungrounded (claim-only) and not applicable.")
    mc_name, model_id = MC_MODELS[model_key]
    # A mock or synthetic run tests the machinery and measures nothing, so it
    # defaults to the smoke tree and cannot occupy a canonical path.
    out_dir = out_dir or (".smoke" if synthetic or backend == "mock" else "runs")
    raw = Path(out_dir) / dataset / f"judge_pt_{model_key}.jsonl"
    if not guard.check(raw.with_suffix(".json"), [FRAME], "minicheck",
                       skip_on_match=True, force=force,
                       label=f"pt_{model_key}/{dataset}"):
        return None

    if synthetic:
        task, instances = synthetic_instances(dataset, n=limit or 400)
    else:
        task, instances = build_instances(dataset, limit)

    pairs = [_doc_and_claim(task, x) for x in instances]
    docs = [d for d, _ in pairs]
    claims = [c for _, c in pairs]

    if backend == "mock":
        # emulate the package's (pred_label, prob) outputs, correlated with gold,
        # so the mapping/bootstrap/writing path is testable offline.
        rng = np.random.default_rng(seed)
        prob = np.clip([(0.8 if x.gold == 0 else 0.2) + rng.normal(0, 0.15)
                        for x in instances], 0.0, 1.0)
        pred_label = (prob >= 0.5).astype(int).tolist()  # 1 = supported
        prob = prob.tolist()
    else:
        try:
            from minicheck.minicheck import MiniCheck
        except ImportError as e:
            raise SystemExit(
                "the `minicheck` package is not importable here — run this script "
                "from its own venv (see the module docstring); do NOT install "
                f"minicheck into the main .venv. ({e})")
        kwargs = dict(model_name=mc_name)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if model_key == "bespoke_minicheck":
            kwargs["enable_prefix_caching"] = False  # per the package README
        scorer = MiniCheck(**kwargs)
        print(f"[{task.name}] specialist verifier={model_id} "
              f"over {len(instances)} instances ...")
        # Bespoke-MiniCheck-7B (InternLM2, 32768-token ctx) chunks long docs at
        # max_model_len-300 by default; that 300-token headroom is one token too
        # tight for the longest WiCE/ExpertQA docs (chunk + chat template + claim
        # hit 32769, over vLLM 0.24's hard limit) and aborts the whole batch.
        # Shrink the chunk to leave ample room; chunks are max-pooled, so the
        # short-doc sets (single-chunk under either size) are unaffected. FT5 has
        # a 2048-token ctx and must keep its own small default — never pass this.
        score_kwargs = {}
        if model_key == "bespoke_minicheck":
            score_kwargs["chunk_size"] = 30000
        pred_label, prob, _, _ = scorer.score(docs=docs, claims=claims, **score_kwargs)

    raw.parent.mkdir(parents=True, exist_ok=True)
    gold, rowid_of, pred = {}, {}, {}
    with raw.open("w") as fh:
        for x, lab, p in zip(instances, pred_label, prob):
            g = 1 - int(lab)                    # package: 1 = supported -> our 1 = hallucinated
            gold[x.uid] = x.gold; rowid_of[x.uid] = x.row_id; pred[x.uid] = g
            fh.write(json.dumps(dict(
                uid=x.uid, row_id=x.row_id, kind=x.kind, gold=x.gold,
                stratum=x.stratum, category=x.category, model=model_id,
                judge=model_key, frame=FRAME, score=float(p), threshold=0.5,
                pred=g, raw=None)) + "\n")

    # The classifier always answers, so coverage comes out at 1.0 rather than
    # being asserted -- the shared estimator computes it, and a future MiniCheck
    # that could abstain would be scored correctly without a change here.
    k, ba, coverage = cluster_bootstrap(gold, pred, rowid_of, B=B, seed=seed)
    out = dict(dataset=task.name, judge=model_key, model=model_id,
               backend=backend, tag=f"pt_{model_key}", family="purpose_trained",
               subfamily="grounded_specialist", frame=FRAME,
               score_range=[0.0, 1.0], threshold=0.5,
               n=len(gold), coverage=coverage,
               score_mean=float(np.mean(prob)),
               parse_failures=0,
               max_model_len=None,             # the package chunks documents internally
               kappa=k, balanced_accuracy=ba)
    json.dump(out, (raw.with_suffix(".json")).open("w"), indent=2)
    print(f"  wrote {raw} (+ .json)")
    report(k, ba, coverage)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=GROUNDED)
    ap.add_argument("--model", required=True, choices=sorted(MC_MODELS))
    ap.add_argument("--backend", default="package", choices=["package", "mock"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--cache-dir", default=None,
                    help="checkpoint cache for the minicheck package")
    ap.add_argument("--out-dir", default=None,
                    help="root for judge_pt_<model>.{jsonl,json} (default runs, "
                         "or .smoke under --backend mock or --synthetic; mirrors "
                         "scripts/run_judge.py/run_ptjudge.py)")
    ap.add_argument("--force", action="store_true",
                    help="redo a measurement that already exists at this path "
                         "(FRAME is a constant here, so a mismatch means a "
                         "stray file from a different runner, not a stale "
                         "frame; move it aside by hand instead)")
    args = ap.parse_args()
    run_minicheck(args.dataset, args.model, args.backend, limit=args.limit,
                  synthetic=args.synthetic, B=args.B, cache_dir=args.cache_dir,
                  out_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
