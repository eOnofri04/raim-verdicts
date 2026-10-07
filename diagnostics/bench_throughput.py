#!/usr/bin/env python
"""Throughput benchmark for the analysis repository's cost table.

`cost_table.py` (raim-analysis/scripts) prices inference from two measured
rates, both in ITEMS per second (full prefill + decode wall time per item), on
the hardware the panel was run on:

  ITEMS_PER_S_SMALL_MEMBER   the HARMONIC mean of the ten members' items/s, so
                             that panel size x this rate reproduces the true sum
                             of per-member times
  ITEMS_PER_S_OSS_FRONTIER   items/s of the larger open judge (32B AWQ)

Those are a property of the GPU, not of the run artefacts, so they must be
measured. This script measures them through the repository's own vLLM backend
(`raim.backends.make_backend`), so the decoding settings (dtype, context
window, MAX_NEW_TOKENS) match the real judge runs.

The reported figures were taken with exactly these commands (batch 128,
~500 input tokens, every model forced to decode the same 68-token verdict so
that items/s compares hardware rather than output length):

  python diagnostics/bench_throughput.py --all --gpu-util 0.90 --fixed-output 68
  python diagnostics/bench_throughput.py --model Qwen/Qwen2.5-32B-Instruct-AWQ \\
      --gpu-util 0.90 --fixed-output 68

"""
from __future__ import annotations
import argparse
import time

from raim.backends import make_backend, MAX_NEW_TOKENS
from raim.panel import PANEL

# A representative judge prompt: a short rubric framing plus a filler "evidence"
# block padded to roughly the input length the real datasets present, so the
# prefill cost is realistic rather than trivially short. --in-tokens tunes it.
_FRAMING = (
    "You are assessing a candidate answer for faithfulness to the evidence.\n"
    "A HALLUCINATED answer contains information that is incorrect, fabricated, "
    "or not supported by the evidence; a SUPPORTED answer is fully consistent "
    "with it. Think briefly, then end with exactly one line: "
    "VERDICT: <HALLUCINATED|SUPPORTED>.\n\n"
)


def _make_prompt(approx_in_tokens: int) -> str:
    # ~0.75 words/token; pad the evidence block to hit the target input length.
    filler_words = max(0, int(approx_in_tokens * 0.75) - 60)
    evidence = " ".join(["lorem", "ipsum", "dolor", "sit", "amet"] *
                        (filler_words // 5 + 1))[: filler_words * 6]
    return (_FRAMING +
            f"Question: What does the study report?\n"
            f"Evidence: {evidence}\n"
            f"Candidate answer: The study reports a significant effect.\n")


def bench(model: str, args) -> dict:
    kw = dict(gpu_util=args.gpu_util, max_model_len=args.max_model_len,
              dtype=args.dtype, tensor_parallel=args.tensor_parallel)
    print(f"\n[loading] {model}  (backend={args.backend}, dtype={args.dtype}, "
          f"max_model_len={args.max_model_len})")
    t0 = time.time()
    be = make_backend(args.backend, model, **kw)
    load_s = time.time() - t0

    # Force a uniform output length so throughput is comparable across models.
    # Without this, members that stop early (emitting only the verdict) post
    # inflated items/s versus members that generate a full reasoning, and the
    # cross-model figures conflate speed with output length. ignore_eos + a fixed
    # min/max makes every model decode exactly N tokens (~68 = the real judge
    # output), isolating the hardware term. vLLM only.
    if args.fixed_output and hasattr(be, "sp"):
        from vllm import SamplingParams
        be.sp = SamplingParams(temperature=0.0, min_tokens=args.fixed_output,
                               max_tokens=args.fixed_output, ignore_eos=True)
        print(f"[fixed-output] forcing exactly {args.fixed_output} output tokens/item")
    elif args.fixed_output:
        print(f"[fixed-output] WARNING: this backend exposes no sampling params; "
              f"--fixed-output {args.fixed_output} NOT applied (natural stopping)")

    prompts = [_make_prompt(args.in_tokens) for _ in range(args.batch)]
    # vLLM ignores meta; the mock backend reads 'gold' to synthesise a verdict,
    # so supply a dummy one to keep the script runnable under --backend mock too.
    meta = [{"gold": i % 2} for i in range(args.batch)]

    # warm-up (kernel autotune / cache) -- excluded from timing
    _ = be.generate(prompts[: min(4, len(prompts))], meta[: min(4, len(prompts))])

    t0 = time.time()
    outs = be.generate(prompts, meta)
    gen_s = time.time() - t0
    # Output length in TOKENS, re-encoded with the model's own tokenizer; the
    # mock backend has none, so there it falls back to a words/0.75 estimate.
    # items/s is timed and does not depend on this.
    tok = getattr(be, "tok", None)
    if tok is not None:
        out_tokens = sum(len(tok.encode(o, add_special_tokens=False)) for o in outs)
        method = "tokenizer"
    else:
        out_tokens = sum(len(o.split()) for o in outs) / 0.75
        method = "approx (words/0.75)"
    res = {
        "model": model, "load_s": round(load_s, 1), "batch": args.batch,
        "gen_s": round(gen_s, 2),
        "items_per_s": round(args.batch / gen_s, 2),
        "out_tok_per_item": round(out_tokens / args.batch, 1),
        "out_tok_per_s": round(out_tokens / gen_s, 1),
        "out_tok_method": method,
        "max_new_tokens_cap": MAX_NEW_TOKENS,
    }


    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PANEL[0],
                    help=f"HF id to benchmark (default: {PANEL[0]})")
    ap.add_argument("--all", action="store_true",
                    help="benchmark every member of the panel roster in sequence")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--batch", type=int, default=128,
                    help="prompts per timed generate() call (sustained throughput)")
    ap.add_argument("--in-tokens", type=int, default=500,
                    help="approx input tokens per prompt (match your datasets)")
    ap.add_argument("--fixed-output", type=int, default=0,
                    help="force exactly N output tokens/item (ignore_eos) so items/s "
                         "is comparable across models that otherwise stop early; the "
                         "reported figures use 68 (the real judge output length). "
                         "0 = natural stopping.")
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.set_defaults(score=False)
    args = ap.parse_args()

    models = PANEL if args.all else [args.model]
    rows, failed = [], []
    for m in models:
        try:
            rows.append(bench(m, args))
        except Exception as e:                       # keep going across the roster
            print(f"[skip] {m}: {e}")
            failed.append(m)

    print("\n==== throughput summary "
          f"(batch={args.batch}, in≈{args.in_tokens} tok) ====")
    hdr = f"{'model':<40} {'items/s':>8} {'out tok/s':>10} {'out tok/item':>13}"
    print(hdr)
    for r in rows:
        line = (f"{r['model'].split('/')[-1]:<40} {r['items_per_s']:>8} "
                f"{r['out_tok_per_s']:>10} {r['out_tok_per_item']:>13}")
        print(line)
    if rows:
        print(f"\n(output tokens counted by: {rows[0]['out_tok_method']})")
    print("\n-> in cost_table.py, ITEMS_PER_S_SMALL_MEMBER is the HARMONIC mean of the")
    print("   ten members' items/s above, and ITEMS_PER_S_OSS_FRONTIER the 32B judge's.")
    if failed:
        raise SystemExit(f"\n!! {len(failed)} of {len(models)} model(s) failed and are "
                         f"missing from the table (a harmonic mean over the rest would "
                         f"be wrong): {', '.join(failed)}")


if __name__ == "__main__":
    main()
