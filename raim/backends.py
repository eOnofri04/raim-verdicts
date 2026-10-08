"""Inference backends.

Interface: backend.generate(prompts: List[str], meta: List[dict]) -> List[str]
where meta[i] carries {gold, stratum, condition, fmt, paraphrase} so the mock
can synthesize realistic verdicts; the vLLM backend ignores meta.
"""
from __future__ import annotations
import hashlib
import math
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List
from .experimental import experimental, deprecated

MAX_NEW_TOKENS = 384
TEMPERATURE = 0.0  # greedy, single sample

SCORER_REV = "v3_constrained"  # constrained-decoding forced choice; recorded
# verbatim into every scorer output record (scripts/run_lpscore.py's "scorer" field),
# so this string is frozen.
#
# Scorer revisions (v1 and v2 are kept as the targets of tests/test_matcher.py):
#   v1 (_score_v1_decoded_string)          decoded-string match, ignoring
#                                           token id.
#   v2 (_score_from_logprobs)              token-id match, with the v1 string
#                                           match kept as its fallback; both
#                                           capped at the top-20 logprobs
#                                           vLLM returned, so flat base
#                                           models scored 0% coverage.
#   v3 (_score_from_constrained_logprobs,  constrained decoding removes the
#       used by VLLMBackend.score_choice)   top-k cap entirely: every token id
#                                           vLLM can return is already in the
#                                           pos/neg set, by construction.

# Word-boundary markers that appear in RAW vocab pieces (and, for some tokenizer
# families, in vLLM's Logprob.decoded_token):
#   "\u2581" (METASPACE)  SentencePiece word-start  (gemma, Yi, Mistral, Llama-2)
#   "\u0120" (G-dot)      byte-level-BPE space      (Llama-3, Qwen2, GPT-2 line)
#   "\u010a" (C-dot)      byte-level-BPE newline
_MARKERS = str.maketrans({"\u2581": " ", "\u0120": " ", "\u010a": " "})


def _choice_token_ids(tokenizer, words):
    """All vocab ids whose piece reads as one of `words` once leading
    word-boundary markers and whitespace are ignored (case-insensitive).

    Matching on TOKEN IDS rather than decoded strings is representation-
    independent: for some tokenizer families vLLM surfaces top-k pieces as
    '\u2581Yes'-style strings, and str.strip() does NOT remove U+2581 (it is
    not whitespace), so a string match silently zeroes coverage for those
    families (gemma, Yi)."""
    targets = {w.strip().lower() for w in words}
    return {tid for piece, tid in tokenizer.get_vocab().items()
            if piece.translate(_MARKERS).strip().lower() in targets}

def _allowed_token_ids(pos_ids, neg_ids):
    """The constrained-decoding candidate set: pos ∪ neg, deduplicated, sorted
    (vLLM's `allowed_token_ids` takes a list)."""
    return sorted(set(pos_ids) | set(neg_ids))


@deprecated("This was the v1 decoded-string matcher. "
            "Use _score_from_constrained_logprobs() instead.")
def _score_v1_decoded_string(lp, posset, negset):
    """The retired v1 algorithm: match on the DECODED STRING of each
    candidate (stripped, lower-cased), ignoring token id entirely.

    Superseded because vLLM's `decoded_token` is not uniform across tokenizer
    families: some surface a raw piece carrying a leading word-boundary
    marker (e.g. '▁Yes') that str.strip() does not remove -- U+2581 is
    not whitespace -- so this saw nothing for those models. `_score_from_logprobs`
    (v2) keeps this exact rule as its fallback for ids outside the token-id
    sets, so no v1 match is ever missed by v2; see test_matcher.py's superset
    test, which calls this function directly rather than re-deriving it.
    """
    pos_lp = max((info.logprob for info in lp.values()
                  if (info.decoded_token or "").strip().lower() in posset),
                 default=float("-inf"))
    neg_lp = max((info.logprob for info in lp.values()
                  if (info.decoded_token or "").strip().lower() in negset),
                 default=float("-inf"))
    if pos_lp == float("-inf") and neg_lp == float("-inf"):
        return (None, None)
    if neg_lp == float("-inf"):
        return (1, 1.0)
    if pos_lp == float("-inf"):
        return (0, 0.0)
    m = max(pos_lp, neg_lp)
    p = math.exp(pos_lp - m) / (math.exp(pos_lp - m) + math.exp(neg_lp - m))
    return (1 if p >= 0.5 else 0, p)


@deprecated("This was the v2 token-id matcher (string-match fallback). "
            "Use _score_from_constrained_logprobs() instead.")
def _score_from_logprobs(lp, pos_ids, neg_ids, posset, negset):
    """One item's first-token logprobs {token_id: Logprob} -> (choice, p_pos).

    Token-id sets are primary; the v1 decoded-string match is kept as a
    fallback for ids in neither set, so v2 matches are a strict superset of
    v1's. choice=1 iff a pos word is the more likely continuation;
    (None, None) if neither class appears at all."""
    pos_lp = neg_lp = float("-inf")
    for tid, info in lp.items():
        if tid in pos_ids:
            pos_lp = max(pos_lp, info.logprob)
        elif tid in neg_ids:
            neg_lp = max(neg_lp, info.logprob)
        else:
            tok = (info.decoded_token or "").strip().lower()
            if tok in posset:
                pos_lp = max(pos_lp, info.logprob)
            elif tok in negset:
                neg_lp = max(neg_lp, info.logprob)
    if pos_lp == float("-inf") and neg_lp == float("-inf"):
        return (None, None)
    if neg_lp == float("-inf"):
        return (1, 1.0)
    if pos_lp == float("-inf"):
        return (0, 0.0)
    m = max(pos_lp, neg_lp)
    p = math.exp(pos_lp - m) / (math.exp(pos_lp - m) + math.exp(neg_lp - m))
    return (1 if p >= 0.5 else 0, p)


def _score_from_constrained_logprobs(lp, pos_ids, neg_ids):
    """One item's first-token logprobs, already masked to pos_ids|neg_ids by
    constrained decoding, -> (choice, p_pos): the live v3_constrained
    algorithm used by VLLMBackend.score_choice.

    Unlike v2 (max logprob per class), this sums the PROBABILITY MASS across
    every token id in a class -- a material difference whenever more than one
    surface variant of "Yes"/"No" carries real weight, which is exactly what
    _choice_token_ids exists to collect. It is a top-level function so that
    tests/test_constrained.py exercises the production code directly.

    (None, None) should be unreachable given a genuinely constrained `lp`
    (every returned id is in pos_ids or neg_ids by construction); kept as a
    safety net rather than trusted blindly.
    """
    pos_mass = neg_mass = 0.0
    for tid, info in lp.items():
        p = math.exp(info.logprob)
        if tid in pos_ids:
            pos_mass += p
        elif tid in neg_ids:
            neg_mass += p
    tot = pos_mass + neg_mass
    if tot <= 0.0:
        return (None, None)
    p_pos = pos_mass / tot
    return (1 if p_pos >= 0.5 else 0, p_pos)


def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class Backend(ABC):
    """The interface every backend implements.

    `generate` is the REQUIRED contract: raw text completions, one per
    prompt. `meta[i]` carries {gold, stratum, condition, fmt, paraphrase} so
    MockBackend can synthesise plausible verdicts; every other backend
    ignores it.

    `score_choice` is an OPTIONAL capability, not part of the required
    contract: only MockBackend and VLLMBackend implement it, for the
    constrained forced-choice scorer (scripts/run_lpscore.py, diagnostics/bench_throughput.py --
    both restrict --backend to {vllm, mock} for exactly this reason). The
    API backends inherit the default below, which raises with a clear
    message rather than the AttributeError a caller would otherwise hit.
    Every implementation keeps `pos_words`/`neg_words`/`topk` in its
    signature even where unused (e.g. MockBackend's random draw ignores
    them) so the shape of the optional contract stays uniform and a reader
    can see what a caller is entitled to pass, not just what happens to be
    read.
    """
    name: str

    @abstractmethod
    def generate(self, prompts: List[str], meta: List[Dict]) -> List[str]:
        ...

    def score_choice(self, prompts: List[str], meta=None,
                     pos_words=("Yes",), neg_words=("No",), topk: int = 20):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the constrained "
            "forced-choice scorer (score_choice); only MockBackend and "
            "VLLMBackend do.")


class MockBackend(Backend):
    """Deterministic fake judge with tunable skill, for testing the pipeline.

    Error rate depends on model and stratum, so a pipeline test shows sensible,
    model-dependent kappa without a GPU.
    """
    def __init__(self, model_name: str):
        self.name = model_name
        h = _seed(model_name)
        self.base_err = 0.10 + (h % 7) * 0.015      # 0.10 .. 0.19
        self.frag = 0.05 + (h % 5) * 0.02            # paraphrase instability
        self.seed = h

    def score_choice(self, prompts: List[str], meta=None,
                     pos_words=("Yes",), neg_words=("No",), topk: int = 20):
        """Mock log-prob scorer. If meta with gold is supplied, returns a
        gold-correlated choice (so a pipeline test shows sensible κ); otherwise
        deterministic from the prompt hash. choice=1 means a 'pos' (Yes) win."""
        out = []
        for i, p in enumerate(prompts):
            key = meta[i].get("uid", p[:200]) if (meta and i < len(meta)) else p[:200]
            rng = random.Random(self.seed ^ _seed(str(key)))
            if meta is not None and i < len(meta) and meta[i].get("gold") is not None:
                gold = meta[i]["gold"]            # 0 supported -> Yes, 1 -> No
                yes = (gold == 0) if rng.random() > self.base_err else (gold == 1)
                prob = 0.8 if yes else 0.2
                out.append((1 if yes else 0, prob))
            else:
                yes = rng.random() > 0.5
                out.append((1 if yes else 0, 0.7 if yes else 0.3))
        return out

    def generate(self, prompts: List[str], meta: List[Dict],
                 apply_template: bool = True) -> List[str]:
        # apply_template is accepted for interface parity with VLLMBackend and
        # ignored here (the mock emits its own verdict lines).
        # per-stratum difficulty bump (mock only): harder strata -> more error.
        # Unknown strata (e.g. dataset-specific labels) fall back to 0.05.
        strat_bump = {"easy": 0.0, "medium": 0.05, "hard": 0.10,
                      "Adversarial": 0.08, "Non-Adversarial": 0.02}
        cond_bump = {"def_on": 0.0}
        fmt_bump = {"brief_reason": 0.0}
        outs = []
        for p, m in zip(prompts, meta):
            gold = m["gold"]
            p_err = (self.base_err + strat_bump.get(m.get("stratum"), 0.05)
                     + cond_bump.get(m.get("condition"), 0.0)
                     + fmt_bump.get(m.get("fmt"), 0.0))
            # stable draw keyed by item+model+condition+fmt
            base_rng = random.Random(self.seed ^ _seed(
                f"{m.get('uid')}|{m.get('condition')}|{m.get('fmt')}"))
            pred = gold if base_rng.random() > p_err else 1 - gold
            verdict = "HALLUCINATED" if pred == 1 else "SUPPORTED"
            outs.append(f"(mock reasoning)\nVERDICT: {verdict}")
        return outs


class VLLMBackend(Backend):
    """Offline batched inference for one model on local GPUs.

    For a LARGE judge model (e.g. 27B-72B) that won't fit on one GPU, pass
    tensor_parallel=N to shard across N GPUs, and raise gpu_util/max_model_len
    as the card(s) allow.
    """
    def __init__(self, model_name: str, **kw):
        # Decoding is greedy (TEMPERATURE = 0.0, no top_p or top_k), so vLLM's
        # FlashInfer top-p/top-k sampler filters nothing; left on, it JIT-compiles a
        # kernel at the first sampling step, which fails wherever the system CUDA
        # toolkit is older than 12 or the venv's `ninja` is not on PATH. Off unless
        # the environment says otherwise; the engine's subprocesses inherit it.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 trust_remote_code=True)
        # dtype defaults to bfloat16 (as every panel run used). Some quantised
        # MoE models (e.g. AWQ Mixtral via the fused Marlin kernel) hard-assert
        # float16 and crash if cast to bf16 — pass dtype="float16" for those.
        self.llm = LLM(model=model_name, dtype=kw.get("dtype", "bfloat16"),
                       trust_remote_code=True,
                       tensor_parallel_size=int(kw.get("tensor_parallel", 1)),
                       gpu_memory_utilization=kw.get("gpu_util", 0.40),
                       max_model_len=kw.get("max_model_len", 8192))
        self.max_model_len = int(kw.get("max_model_len", 8192))
        # purpose-trained judges (Auto-J) emit a long critique before the rating,
        # so the output budget is configurable; defaults to the panel's MAX_NEW_TOKENS.
        self.max_new_tokens = int(kw.get("max_new_tokens", MAX_NEW_TOKENS))
        self.sp = SamplingParams(temperature=TEMPERATURE,
                                 max_tokens=self.max_new_tokens)

    def _fit_prompt(self, p: str, reserve: int = MAX_NEW_TOKENS):
        """Middle-out truncation so the chat-templated input always leaves room for
        generation. vLLM hard-errors on prompts longer than max_model_len (some
        older releases truncated silently); without this guard a single
        over-length document kills the whole run. build_prompt's layout is
        [instructions][question][CONTEXT — the bulky middle][candidate][verdict cue],
        so we keep the HEAD (instructions + question) and the TAIL (candidate +
        verdict cue) and trim the context in the middle — neither left- nor
        right-truncation would preserve both. Runs on the RAW prompt, before the
        chat template, so template/special tokens are never cut. `reserve` is the
        number of output tokens to keep free (generation reserves MAX_NEW_TOKENS;
        constrained forced-choice scoring needs only a handful). Returns
        (prompt, was_truncated)."""
        budget = self.max_model_len - reserve - 128  # reserve output + template/marker
        ids = self.tok.encode(p, add_special_tokens=False)
        if budget <= 0 or len(ids) <= budget:
            return p, False
        head = budget // 2
        tail = budget - head
        kept = (self.tok.decode(ids[:head])
                + "\n…[context truncated to fit the model window]…\n"
                + self.tok.decode(ids[-tail:]))
        return kept, True

    def generate(self, prompts: List[str], meta: List[Dict],
                 apply_template: bool = True) -> List[str]:
        """Batched greedy generation. With apply_template=True (default, the panel
        path) each raw prompt is wrapped by the tokenizer's chat template. With
        apply_template=False the prompt is sent verbatim: purpose-trained judges
        (Prometheus/JudgeLM/Auto-J) carry their own conversation markers ([INST],
        the JudgeLM system+role block) inside the adapter string, so re-wrapping
        them would double the template — see raim/pt_judges.py."""
        fitted = [self._fit_prompt(p, reserve=self.max_new_tokens)
                  for p in prompts]
        n_trunc = sum(t for _, t in fitted)
        if n_trunc:
            print(f"  [generate] {self.name}: middle-out truncated {n_trunc}/"
                  f"{len(prompts)} over-length prompt(s) to fit "
                  f"max_model_len={self.max_model_len} (reserved "
                  f"{self.max_new_tokens} for output) — NOT silent; check if this "
                  f"dataset needs a bigger --max-model-len.")
        if apply_template:
            chats = [self.tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p, _ in fitted]
        else:
            chats = [p for p, _ in fitted]
        res = self.llm.generate(chats, self.sp)
        return [r.outputs[0].text for r in res]

    def score_choice(self, prompts: List[str], meta=None,
                     pos_words=("Yes", "yes"), neg_words=("No", "no"),
                     topk: int = 20):
        """Forced-choice first-token scoring via CONSTRAINED DECODING (base OR
        instruct). Delivers the design's 100% coverage guarantee.

        Sends RAW prompts (the few-shot prompt already ends at 'Answer:') and
        constrains the next token to the pos∪neg id set with allowed_token_ids,
        so every other logit is masked to -inf BEFORE the softmax. We then read
        the renormalised mass and compare P(pos) vs P(neg). choice=1 iff a pos
        word is the more likely continuation. Identical for base and instruct, so
        scoring is not a confound.

        Constraining is what guarantees coverage. Read from the unconstrained
        first-token top-k instead (vLLM caps it at 20), flat base models
        (gemma-2-9b, Yi-1.5-9B) place Yes/No OUTSIDE the top-20 on every item,
        giving 0% coverage, and gemma-2-9b-it surfaces only one side, giving
        saturated, one-sided votes. Constraining the candidate set removes both
        pathologies and is robust to which surface variant ('Yes' vs '▁Yes')
        carries the mass, since the id set contains all of them (see
        _choice_token_ids). `topk` is kept for signature uniformity.
        """
        from vllm import SamplingParams
        key = (tuple(pos_words), tuple(neg_words))
        if getattr(self, "_choice_ids_key", None) != key:
            self._pos_ids = _choice_token_ids(self.tok, pos_words)
            self._neg_ids = _choice_token_ids(self.tok, neg_words)
            self._allowed = _allowed_token_ids(self._pos_ids, self._neg_ids)
            self._choice_ids_key = key
            print(f"  [score_choice] {self.name}: |pos_ids|={len(self._pos_ids)} "
                  f"|neg_ids|={len(self._neg_ids)} allowed={len(self._allowed)} "
                  f"(constrained decoding, {SCORER_REV})")
        if not self._allowed:
            raise RuntimeError(f"{self.name}: empty candidate id set for "
                               f"pos={pos_words} neg={neg_words} — cannot score")
        sp = SamplingParams(temperature=TEMPERATURE, max_tokens=1,
                            logprobs=min(len(self._allowed), 20),
                            allowed_token_ids=self._allowed)
        # guard against an over-length few-shot prompt (newer vLLM hard-errors);
        # forced choice emits one token, so reserve only a handful, not MAX_NEW_TOKENS.
        fitted = [self._fit_prompt(p, reserve=8) for p in prompts]
        n_trunc = sum(t for _, t in fitted)
        if n_trunc:
            print(f"  [score_choice] {self.name}: middle-out truncated {n_trunc}/"
                  f"{len(prompts)} over-length prompt(s) to fit "
                  f"max_model_len={self.max_model_len} — NOT silent; raise "
                  f"--max-model-len if this fires.")
        res = self.llm.generate([p for p, _ in fitted], sp)
        return [_score_from_constrained_logprobs(
                    r.outputs[0].logprobs[0],  # {token_id: Logprob}, masked to allowed
                    self._pos_ids, self._neg_ids)
                for r in res]


def _api_usage(results, n, t0):
    """The shared last_usage block of the concurrent API backends."""
    import time
    tok_in = sum(r[1] for r in results)
    tok_out = sum(r[2] for r in results)
    return dict(
        calls=n,
        api_errors=sum(r[3] for r in results),
        input_tokens=tok_in,
        output_tokens=tok_out,
        input_tokens_per_item=round(tok_in / n, 2) if n else None,
        output_tokens_per_item=round(tok_out / n, 2) if n else None,
        wall_seconds=round(time.time() - t0, 1),
    )


class APIBackend(Backend):
    """Frontier judge via the Anthropic Messages API (no GPU needed).

    Reads the key from ANTHROPIC_API_KEY. Model string is configurable and
    evolves over time (e.g. a current Claude Sonnet); check the docs. Calls run
    concurrently with retry/backoff on rate-limit/overload errors.
    """
    def __init__(self, model_name: str, **kw):
        import os
        from anthropic import Anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY in the environment")
        self.name = model_name
        self.client = Anthropic()
        self.workers = int(kw.get("workers", 8))
        self.max_tokens = int(kw.get("max_tokens", MAX_NEW_TOKENS))
        self.max_retries = int(kw.get("max_retries", 6))
        self.last_usage: Dict | None = None  # set by generate(); read by run_judge

    def _one(self, prompt: str):
        """Return (text, input_tokens, output_tokens, failed)."""
        import time
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                m = self.client.messages.create(
                    model=self.name, max_tokens=self.max_tokens,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in m.content if b.type == "text")
                return text, m.usage.input_tokens, m.usage.output_tokens, False
            except Exception as e:  # rate limit / overload / transient
                if attempt == self.max_retries - 1:
                    return f"(api error: {e})\nVERDICT: ABSTAIN", 0, 0, True
                time.sleep(delay + random.random())
                delay = min(delay * 2, 30)

    def generate(self, prompts: List[str], meta: List[Dict]) -> List[str]:
        import time
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(self._one, prompts))
        self.last_usage = _api_usage(results, len(prompts), t0)
        return [r[0] for r in results]




def make_backend(kind: str, model_name: str, **kw):
    if kind == "mock":
        return MockBackend(model_name)
    if kind == "vllm":
        return VLLMBackend(model_name, **kw)
    if kind == "api":
        return APIBackend(model_name, **kw)
    raise ValueError(f"unknown backend {kind!r}")