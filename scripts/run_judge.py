#!/usr/bin/env python
"""One reference JUDGE over one dataset: the worker behind `run_judge.sh`.

The judge is the expensive reference the panel approaches.  It uses the SAME
prompt as the panel, under the same per-dataset frame, and writes the SAME
JSONL schema.

ONE MODEL, ONE DATASET, one invocation.  The sweep over judges and datasets --
which judge exists, what it costs, which card it wants, which key it needs --
lives in `raim/legs.py` and is driven by `run_judge.sh`; this file knows none
of that, and gains nothing by knowing it.

FRAME.  `--condition` defaults to the dataset's own (`raim.suite.condition_of`):
`def_on` on the question-form sets, `claimcheck` on the LLM-AggreFact family.
`judge_one` calls `raim.guard.check` itself, so this holds whether invoked
through `run_judge.sh`/`raim.legs` or standalone.

Backends:
  api    : Claude Sonnet -- no GPU, needs ANTHROPIC_API_KEY. PAID: standalone
           use requires --paid as an explicit opt-in so you can't spend by accident.
  vllm   : large open-weight judge (Qwen2.5-32B/72B); --tensor-parallel N.
  mock   : offline testing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from raim import (build_instances, synthetic_instances, build_prompt,
                  parse_verdict, make_backend)
from raim import guard
from raim.scoring import cluster_bootstrap, report
from raim.suite import condition_of
from raim.tasks import TASKS

# API list prices, USD per Mtok (input, output), for the cost block written
# into judge_<tag>.json. Snapshot at run time -- the JSON records the rates it
# used, so later price changes cannot silently corrupt old records. Longest
# matching prefix wins; unknown models get tokens but cost_usd=None.
API_USD_PER_MTOK = {
    "claude-sonnet-4-6": (3.0, 15.0),
}

# Backends that bill per token, hence need --paid.
BACKENDS = ["api", "vllm", "mock"]
PAID = {"api"}


def _api_cost(model, usage):
    """Cost block from measured token usage, or None if usage is missing."""
    if not usage:
        return None
    rate = next((API_USD_PER_MTOK[k] for k in
                 sorted(API_USD_PER_MTOK, key=len, reverse=True)
                 if model.startswith(k)), None)
    cost = dict(usd_per_mtok_in=rate[0] if rate else None,
                usd_per_mtok_out=rate[1] if rate else None,
                cost_usd=None, cost_usd_per_1k_items=None)
    if rate:
        usd = (usage["input_tokens"] * rate[0]
               + usage["output_tokens"] * rate[1]) / 1e6
        cost["cost_usd"] = round(usd, 4)
        if usage["calls"]:
            cost["cost_usd_per_1k_items"] = round(1000 * usd / usage["calls"], 4)
    return cost


def judge_one(dataset, backend, model, *, tag=None, condition=None,
              limit=None, synthetic=False, B=2000, seed=0, out_dir=None,
              force=False, **backend_kw):
    """Run one judge over `dataset`, write JSONL + JSON, return result dict.

    Guarded the same way every other entry point is: an existing
    `judge_<tag>.json` recording a different condition refuses rather than
    being overwritten, `--force` redoes a match only, and this holds whether
    the call comes from `raim/legs.py`'s sweep or a standalone invocation.
    An existing file recorded for a DIFFERENT judge model stops the run
    outright, `--force` included: the tag is what names the file, and two
    models sharing one tag would otherwise have the second skipped as done.
    """
    tag = tag or backend
    condition = condition or condition_of(dataset)
    # A mock or synthetic run tests the machinery and measures nothing, so it
    # defaults to the smoke tree and cannot occupy a canonical path.
    out_dir = out_dir or (".smoke" if synthetic or backend == "mock" else "runs")
    raw = Path(out_dir) / dataset / f"judge_{tag}.jsonl"
    summary = raw.with_suffix(".json")
    if summary.exists():
        recorded = json.loads(summary.read_text()).get("judge")
        if recorded != model:
            raise SystemExit(
                f"{summary} was recorded for judge {recorded!r}, not {model!r}; "
                f"pass a distinct --tag, or move the file aside by hand "
                f"(--force cannot override this).")
    if not guard.check(raw.with_suffix(".json"), [condition], "judge",
                       skip_on_match=True, force=force, label=f"{tag}/{dataset}"):
        return None

    if synthetic:
        task, instances = synthetic_instances(dataset, n=limit or 400)
    else:
        task, instances = build_instances(dataset, limit)

    prompts = [build_prompt(task, x, condition=condition) for x in instances]
    meta = [dict(uid=x.uid, gold=x.gold, stratum=x.stratum, condition=condition,
                 fmt="brief_reason", paraphrase="p0") for x in instances]

    be = make_backend(backend, model, **backend_kw)
    print(f"[{task.name}] judge={model} ({backend}) over {len(instances)} "
          f"instances (condition={condition}) ...")
    texts = be.generate(prompts, meta)

    raw.parent.mkdir(parents=True, exist_ok=True)
    gold, rowid_of, pred = {}, {}, {}
    with raw.open("w") as fh:
        for x, txt in zip(instances, texts):
            p = parse_verdict(txt)
            gold[x.uid] = x.gold; rowid_of[x.uid] = x.row_id; pred[x.uid] = p
            fh.write(json.dumps(dict(
                uid=x.uid, row_id=x.row_id, kind=x.kind, gold=x.gold,
                stratum=x.stratum, category=x.category, model=model,
                condition=condition, fmt="brief_reason", paraphrase="p0",
                pred=p, raw=txt)) + "\n")

    k, ba, coverage = cluster_bootstrap(gold, pred, rowid_of, B=B, seed=seed)
    out = dict(dataset=task.name, judge=model, backend=backend, tag=tag,
               condition=condition, n=len(gold), coverage=coverage,
               max_model_len=backend_kw.get("max_model_len"),
               kappa=k, balanced_accuracy=ba)
    usage = getattr(be, "last_usage", None)   # API backend only
    if usage:
        out["usage"] = usage
        out["cost"] = _api_cost(model, usage)
    json.dump(out, (raw.with_suffix(".json")).open("w"), indent=2)
    print(f"  wrote {raw} (+ .json)")
    report(k, ba, coverage)
    if usage:
        c = out["cost"] or {}
        cost_s = (f"  cost=${c['cost_usd']}" if c.get("cost_usd") is not None
                  else "  cost=n/a (model not in API_USD_PER_MTOK)")
        print(f"  tokens: in={usage['input_tokens']:,} "
              f"out={usage['output_tokens']:,} "
              f"({usage['input_tokens_per_item']}/{usage['output_tokens_per_item']} per item)"
              f"{cost_s}  errors={usage['api_errors']}  "
              f"wall={usage['wall_seconds']}s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(TASKS))
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--paid", action="store_true",
                    help="REQUIRED with a paid API backend (guards against spend)")
    ap.add_argument("--condition", default=None,
                    help="recorded condition; default: the dataset's own frame "
                         "(def_on on the question-form sets, claimcheck on the "
                         "LLM-AggreFact family)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "auto"],
                    help="vLLM compute dtype. Use float16 for AWQ MoE "
                         "checkpoints (the fused Marlin MoE kernel asserts fp16).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="root for judge_<tag>.{jsonl,json} (default runs, or "
                         ".smoke under --backend mock or --synthetic; use a "
                         "staging tree to keep runs/ pristine)")
    ap.add_argument("--force", action="store_true",
                    help="redo a measurement that already exists UNDER THE RIGHT "
                         "condition; it can never override a condition mismatch "
                         "(move the stale file aside by hand instead)")
    args = ap.parse_args()
    if args.backend in PAID and not args.paid:
        ap.error("refusing to call a PAID API without --paid")
    backend_kw = dict(tensor_parallel=args.tensor_parallel,
                      gpu_util=args.gpu_util, max_model_len=args.max_model_len,
                      dtype=args.dtype, workers=args.workers)
    judge_one(args.dataset, args.backend, args.model, tag=args.tag,
              condition=args.condition, limit=args.limit, synthetic=args.synthetic,
              B=args.B, out_dir=args.out_dir, force=args.force, **backend_kw)


if __name__ == "__main__":
    main()