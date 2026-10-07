"""Task abstraction and dataset registry.

Eight benchmarks, spanning two axes of hallucination:
  - grounded / faithfulness   : medhallu, ragtruth, and four LLM-AggreFact
                 subsets (aggrefact_xsum, aggrefact_wice, aggrefact_cnn,
                 aggrefact_expertqa). A reference passage or document is
                 supplied, so the judge does entailment, not world-knowledge
                 recall.
  - ungrounded / world-factuality : truthfulqa and factscore. No context is
                 supplied, so the judge must itself know the truth. Both are
                 adversarial by design -- truthfulqa's imitative falsehoods,
                 factscore's atomic biographical claims -- which is what
                 maximally stresses correlated error across the panel.

All are reduced to the SAME binary detection format: each source row yields
one SUPPORTED instance (gold 0) and one HALLUCINATED instance (gold 1), giving
a balanced, stratified set. New datasets only need a builder that returns
(TaskSpec, List[Instance]); nothing downstream changes.

Three further LLM-AggreFact subsets are registered below (aggrefact_lfqa,
aggrefact_tofueval, aggrefact_claimverify) but are not part of the reported
study; they are available, runnable extensions, not additional results.
"""
from __future__ import annotations
import hashlib as _hashlib_key
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# NUL cannot occur in any source field, so concatenating under it is unambiguous
# and the join key cannot be forged by a candidate that happens to contain the
# separator. Changing this constant invalidates every committed key.
_KEY_SEP = b"\x00"


# Pinned upstream dataset revisions (in place since 2026-08-04). Every builder
# below passes its entry to load_dataset, so a rebuild resolves the SAME
# snapshot the reported verdicts were produced against, on any machine and at
# any later date.
#
# Why this is necessary: for the four LLM-AggreFact sets both uid and row_id are
# positional artefacts of a seeded shuffle over the loaded rows, so an upstream
# revision would re-point every uid silently -- nothing raises, the numbers still
# compute, and they are wrong. See the README, "The datasets and the join".
#
# These SHAs were verified on 2026-08-04 by a cold-cache online rebuild on a
# second machine, which reproduced the committed dataset_index/ exactly for
# all eight datasets, and each repository's last_modified (2024-11 to 2025-02)
# predates the reported runs. `dataset_index/` remains the valid mapping
# regardless, since it verifies content rather than trusting a pointer, and it
# survives a dataset being revised, gated or withdrawn.
DATASET_REVISIONS: Dict[str, str] = {
    "UTAustin-AIHealth/MedHallu":  "515060458a945c633debc6fd5baac7764416b724",
    "wandb/RAGTruth-processed":    "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b",
    "truthfulqa/truthful_qa":      "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
    "lytang/LLM-AggreFact":        "981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4",
}


def revision_of(name: str) -> Optional[str]:
    """The pinned commit for `name`, or None when the pins are off on purpose.

    Pinning closes a real hazard and introduces a smaller one: a pinned build
    cannot notice that upstream has MOVED, because it resolves the commit we
    named rather than whatever the default branch now serves. Those are
    different questions -- "does the pin still resolve?" and "has upstream
    drifted?" -- and a pinned build (`tools/joincheck.py --online`) answers only
    the first.

    Setting RAIM_IGNORE_DATASET_PINS=1 resolves every Hub source at its current
    default branch, which asks the second. It is a diagnostic, not a
    convenience: nothing in the measurement path should ever set it, since a run
    produced against an unpinned build cannot be joined back with any confidence.
    `tools/joincheck.py --unpinned` is the supported way in.
    """
    import os as _os
    if _os.environ.get("RAIM_IGNORE_DATASET_PINS"):
        return None
    return DATASET_REVISIONS[name]


# The eighth dataset, factscore, has no Hub revision: it is a hand-fetched file,
# reached via FACTSCORE_PATH, so it is pinned by checksum instead. This is the
# sha256 of the datasets/InstructGPT.jsonl (183 lines) against which every
# reported factscore verdict was produced. The file is NOT redistributed here --
# datasets/ is gitignored and tools/fetch_factscore.py gives the download route -- so
# this checksum is what identifies it without need of redistribution. Verified in
# _build_factscore below.
FACTSCORE_SHA256 = "4ec1fa538a9f99606130c76db63d4af7f203ca460954a3edd7c4d0e385cd2d58"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    subject_phrase: str   # how the question is described in the prompt
    has_context: bool     # is reference knowledge supplied?
    context_label: str    # heading for the context block, if any


@dataclass
class Instance:
    uid: str                      # f"{row_id}:{kind}"
    row_id: int
    kind: str                     # "pos" (clean) or "neg" (hallucinated)
    gold: int                     # 0 supported, 1 hallucinated
    stratum: str                  # primary reporting stratum (see per-task)
    category: str                 # secondary group (free-form)
    question: str
    context: List[str]            # [] when the task is ungrounded
    candidate: str

    @property
    def src_key(self) -> str:
        """Content-addressed join key: sha1 over question, context and candidate.

        The single definition of the released join key. `uid` and `row_id` are
        positional over a seeded shuffle of the loaded rows, so they identify an
        instance only relative to a particular build; this identifies it by what
        it actually contains, which is what lets a third party verify the join
        without our redistributing any source text.

        It lives on Instance, rather than in the indexing script, so that the key
        travels with the schema it keys: tools/dataset_index.py writes it into the
        index and panel.py writes it into every emitted verdict, from one
        definition that cannot drift between them.
        """
        h = _hashlib_key.sha1()
        h.update(self.question.encode())
        h.update(_KEY_SEP)
        for c in self.context:
            h.update(c.encode())
            h.update(_KEY_SEP)
        h.update(_KEY_SEP)
        h.update(self.candidate.encode())
        return h.hexdigest()


# ------------------------------ MedHallu ------------------------------
# GROUNDED / medical faithfulness: the reference abstract is supplied, so the
# judge does entailment, not world-knowledge recall. Source: Pandit et al.
# 2025, EMNLP (arXiv:2502.14302), derived from PubMedQA. HF:
# UTAustin-AIHealth/MedHallu, config pqa_labeled; already paired one supported
# and one hallucinated answer per question, so no pairing logic is needed here.
MEDHALLU = TaskSpec(
    name="medhallu",
    subject_phrase="a medical research question",
    has_context=True,
    context_label="Reference knowledge",
)


def _build_medhallu(limit: Optional[int]) -> List[Instance]:
    from datasets import load_dataset
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled",
                       revision=revision_of("UTAustin-AIHealth/MedHallu"))["train"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[Instance] = []
    for i, r in enumerate(ds):
        q, k = r["Question"], r["Knowledge"]
        strat = r.get("Difficulty Level", "unknown")
        cat = r.get("Category of Hallucination", "n/a")
        out.append(Instance(f"{i}:pos", i, "pos", 0, strat, cat, q, k,
                            r["Ground Truth"]))
        out.append(Instance(f"{i}:neg", i, "neg", 1, strat, cat, q, k,
                            r["Hallucinated Answer"]))
    return out


# ----------------------------- TruthfulQA -----------------------------
# UNGROUNDED / world-factuality: no context is supplied, so the judge must
# itself know the truth. Adversarial by design (imitative falsehoods), which
# is what maximally stresses correlated error across the panel. Source: Lin,
# Hilton and Evans 2022, ACL (arXiv:2109.07958). HF: truthfulqa/truthful_qa,
# config generation.
TRUTHFULQA = TaskSpec(
    name="truthfulqa",
    subject_phrase="a general-knowledge question",
    has_context=False,
    context_label="",
)


def _build_truthfulqa(limit: Optional[int]) -> List[Instance]:
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation",
                       revision=revision_of("truthfulqa/truthful_qa"))["validation"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[Instance] = []
    for i, r in enumerate(ds):
        q = r["question"]
        correct = r.get("best_answer") or (r["correct_answers"][0]
                                           if r["correct_answers"] else None)
        incorrect = r["incorrect_answers"][0] if r["incorrect_answers"] else None
        if not correct or not incorrect:
            continue
        strat = r.get("type", "unknown")          # Adversarial / Non-Adversarial
        cat = r.get("category", "n/a")
        out.append(Instance(f"{i}:pos", i, "pos", 0, strat, cat, q, [], correct))
        out.append(Instance(f"{i}:neg", i, "neg", 1, strat, cat, q, [], incorrect))
    return out


TASKS: Dict[str, TaskSpec] = {MEDHALLU.name: MEDHALLU, TRUTHFULQA.name: TRUTHFULQA}
_BUILDERS: Dict[str, Callable[[Optional[int]], List[Instance]]] = {
    "medhallu": _build_medhallu,
    "truthfulqa": _build_truthfulqa,
}


# ------------------------------ RAGTruth ------------------------------
# GROUNDED, non-medical, QA subtask: generalises MedHallu's faithfulness
# story to web/news/general-knowledge sources. Source: Niu et al. 2024.
# HF mirror: wandb/RAGTruth-processed (the ParticleMedia release re-cast as a
# single HF dataset). Each record carries one model RESPONSE plus the
# original SOURCE_INFO (question + passages) and a span-level hallucination
# annotation. We pair one faithful response (no spans) with one hallucinated
# response (>=1 span) per question; questions missing either class are
# dropped (typical retention on QA: ~60-70%).
RAGTRUTH = TaskSpec(
    name="ragtruth",
    subject_phrase="a retrieval-augmented question",
    has_context=True,
    context_label="Retrieved passages",
)


def _build_ragtruth(limit: Optional[int]) -> List[Instance]:
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split="train",
                       revision=revision_of("wandb/RAGTruth-processed"))

    # Real schema (verified on the wandb/RAGTruth-processed mirror):
    #   id, query, context, output, task_type, quality, model, temperature,
    #   hallucination_labels (string-encoded JSON list of spans),
    #   hallucination_labels_processed (dict with evident_conflict / baseless_info counts),
    #   input_str.
    # Group key: `query` (no source_id). ~995 unique queries across 17.8k rows.
    def is_hallucinated(r):
        proc = r.get("hallucination_labels_processed") or {}
        if isinstance(proc, dict):
            return ((proc.get("evident_conflict") or 0) > 0 or
                    (proc.get("baseless_info") or 0) > 0)
        raw = r.get("hallucination_labels")
        if raw in (None, "", "[]", "null"): return False
        return True

    qa_rows = [r for r in ds if str(r.get("task_type", "")).lower() == "qa"]
    if not qa_rows:
        sample = next(iter(ds))
        raise RuntimeError(
            "RAGTruth: no rows with task_type=QA. "
            f"First row columns: {list(sample.keys())}. "
            "Verify the mirror still uses these field names."
        )

    # Pair one faithful + one hallucinated response per question (query string)
    pairs: Dict[str, Dict] = {}
    for r in qa_rows:
        q = (r.get("query") or "").strip()
        if not q: continue
        slot = "neg" if is_hallucinated(r) else "pos"
        bucket = pairs.setdefault(q, {"pos": None, "neg": None, "context": None})
        if bucket[slot] is None:
            ctx = r.get("context") or ""
            ctx = [ctx] if isinstance(ctx, str) and ctx else (
                    list(ctx) if isinstance(ctx, list) else [])
            if bucket["context"] is None:
                bucket["context"] = ctx
                bucket[slot] = r.get("output") or ""
            elif bucket["context"] == ctx:
                bucket[slot] = r.get("output") or ""
            else:
                print(f"WARNING: the context mismatched for {q!r} query. Skipping the entry.")

    paired = [(q, p) for q, p in pairs.items()
              if p["pos"] and p["neg"] and p["context"]]
    if limit: paired = paired[:limit]
    out: List[Instance] = []
    for i, (q, p) in enumerate(paired):
        out.append(Instance(f"{i}:pos", i, "pos", 0, "ragtruth_qa", "faithful",
                            q, p["context"], p["pos"]))
        out.append(Instance(f"{i}:neg", i, "neg", 1, "ragtruth_qa", "hallucinated",
                            q, p["context"], p["neg"]))
    return out


TASKS[RAGTRUTH.name] = RAGTRUTH
_BUILDERS["ragtruth"] = _build_ragtruth


# ------------------------------ FActScore -----------------------------
# UNGROUNDED biographical fact-verification: complements TruthfulQA on the
# adversarial/world-factuality axis. Each instance is one ATOMIC claim
# about a person (e.g. "Marvin Minsky founded the MIT AI Lab"); the panel
# decides truthful vs not, with NO Wikipedia evidence in the prompt.
# Source: Min et al. 2023, EMNLP. The labeled data is shipped via Google
# Drive in the FActScore GitHub repo (`data/labeled/{InstructGPT,ChatGPT,
# PerplexityAI}.jsonl`). Download once, then point this builder at the
# local file via FACTSCORE_PATH env var.
FACTSCORE = TaskSpec(
    name="factscore",
    subject_phrase="a biographical claim",
    has_context=False,
    context_label="",
)


def _build_factscore(limit: Optional[int]) -> List[Instance]:
    import hashlib as _hashlib
    import json as _json
    import os as _os

    # FActScore is not redistributed here: it is the one third-party file we
    # would otherwise ship, and obtaining it from its authors
    # costs a reader one download. The default location is datasets/, which is
    # gitignored; FACTSCORE_PATH overrides it. `tools/fetch_factscore.py` explains the
    # download and verifies the result.
    _local = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "datasets", "InstructGPT.jsonl")
    path = _os.environ.get("FACTSCORE_PATH") or _local
    if not _os.path.exists(path):
        raise RuntimeError(
            f"FActScore requires a local JSONL and none was found at {path}.\n"
            "  It is not redistributed with this repository. Obtain it from the\n"
            "  authors: https://github.com/shmsw25/FActScore -> the Google Drive\n"
            "  link in their README -> data/labeled/InstructGPT.jsonl, and place\n"
            "  it at datasets/InstructGPT.jsonl (or point FACTSCORE_PATH at it).\n"
            "  `python tools/fetch_factscore.py` walks through this and verifies the\n"
            "  checksum, which is what guarantees the line order the released\n"
            "  verdicts were keyed against."
        )

    # FActScore is the one dataset with no upstream revision to pin: it is a file
    # fetched by hand from a Google Drive link. Its uid is the LINE NUMBER in
    # whatever file this path names, so substituting ChatGPT.jsonl or a
    # differently-ordered re-download would silently re-point every instance --
    # the same hazard DATASET_REVISIONS closes for the four Hub datasets, and
    # sharper here because the alternatives share a schema. Hence the checksum.
    digest = _hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != FACTSCORE_SHA256 and not _os.environ.get("FACTSCORE_ALLOW_UNVERIFIED"):
        raise RuntimeError(
            f"FActScore file at {path} does not match the copy the reported runs "
            f"used.\n  expected sha256 {FACTSCORE_SHA256}\n  found    sha256 {digest}\n"
            "This must be data/labeled/InstructGPT.jsonl, 183 lines, NOT its "
            "ChatGPT.jsonl or PerplexityAI.jsonl siblings -- they share the schema, "
            "so substituting one yields plausible instances joined to entirely "
            "different biographies. Set FACTSCORE_ALLOW_UNVERIFIED=1 only if you "
            "intend a different file, in which case the instance identifiers will "
            "NOT correspond to dataset_index/factscore.jsonl."
        )

    rows = [_json.loads(l) for l in open(path) if l.strip()]
    # Real schema (verified on data/labeled/InstructGPT.jsonl):
    #   {"topic": str, "input": str, "output": str, "cat": [str, ...],
    #    "annotations": [
    #       {"text": "<sentence from output>", "is-relevant": bool,
    #        "model-atomic-facts": [{"text": str}, ...]   (no labels),
    #        "human-atomic-facts": [{"text": str, "label": "S"|"NS"|"IR"}, ...]
    #       }, ...]}
    # Note hyphenated keys: "is-relevant", "human-atomic-facts".
    # Per-fact labels live only under "human-atomic-facts".
    out: List[Instance] = []
    n_kept = 0
    for i, rec in enumerate(rows):
        if limit and n_kept >= limit: break
        topic = rec.get("topic") or f"person_{i}"
        supported, refuted = None, None
        for ann in (rec.get("annotations") or []):
            if ann.get("is-relevant") is False: continue
            for af in (ann.get("human-atomic-facts") or []):
                if not isinstance(af, dict): continue
                lbl = str(af.get("label", "")).upper()
                text = af.get("text")
                if not text: continue
                if lbl == "S" and supported is None: supported = text
                elif lbl == "NS" and refuted is None: refuted = text
                if supported and refuted: break
            if supported and refuted: break
        if not (supported and refuted): continue
        q = f"Is the following biographical claim about {topic} factually correct?"
        out.append(Instance(f"{i}:pos", i, "pos", 0, "factscore_bio",
                            "supported", q, [], supported))
        out.append(Instance(f"{i}:neg", i, "neg", 1, "factscore_bio",
                            "not_supported", q, [], refuted))
        n_kept += 1
    if not out:
        raise RuntimeError(
            f"FActScore: parsed {len(rows)} records from {path} but found no "
            "topics with both a supported AND a not-supported relevant fact. "
            "Inspect the file schema and adjust _build_factscore if needed."
        )
    return out


TASKS[FACTSCORE.name] = FACTSCORE
_BUILDERS["factscore"] = _build_factscore


# --------------------------- LLM-AggreFact ----------------------------
# GROUNDED claim-verification. Source: Tang et al. 2024 (MiniCheck, EMNLP),
# HF: lytang/LLM-AggreFact. Unified schema (doc, claim, label, dataset) where
# label=1 means the claim IS supported by doc. We map gold = 1 - label
# (gold=1 = hallucinated/unsupported), filter to one sub-dataset, balance the
# two classes by subsampling the majority (to match the 50/50 design of the
# other tasks), and cluster by `doc` so claims sharing a document form one
# row_id (keeps the cluster-bootstrap honest). Each registered name selects a
# sub-dataset; do NOT use the RAGTruth sub-dataset here (already in the suite).
# The suite uses XSum, CNN, WiCE and ExpertQA; LFQA, TofuEval-MediaS and
# ClaimVerify are registered as unexplored diagnostic options.
_AGGREFACT_SUBSETS = {
    "aggrefact_expertqa": "ExpertQA",
    "aggrefact_wice":     "Wice",
    "aggrefact_lfqa":     "LFQA",
    "aggrefact_tofueval": "TofuEval-MediaS",
    "aggrefact_claimverify": "ClaimVerify",
    "aggrefact_cnn":      "AggreFact-CNN",
    "aggrefact_xsum":     "AggreFact-XSum",
}


def _make_aggrefact_builder(subset_name: str):
    def _build(limit: Optional[int]) -> List[Instance]:
        import random
        from datasets import load_dataset
        ds = load_dataset("lytang/LLM-AggreFact", split="test",
                            revision=revision_of("lytang/LLM-AggreFact"))
        rows = [r for r in ds if r.get("dataset") == subset_name]
        if not rows:
            avail = sorted({r.get("dataset") for r in ds})
            raise RuntimeError(
                f"LLM-AggreFact: no rows for subset {subset_name!r}. "
                f"Available: {avail}")
        # cluster by document; assign a row_id per unique doc
        doc_to_rid: Dict[str, int] = {}
        buckets = {0: [], 1: []}  # gold -> list of (doc, claim)
        for r in rows:
            doc = r.get("doc") or ""
            claim = r.get("claim") or ""
            label = r.get("label")
            if label is None or not claim:
                continue
            gold = 1 - int(label)             # label 1 supported -> gold 0
            buckets[gold].append((doc, claim))
        # balance the two classes by subsampling the majority
        rng = random.Random(0)
        n = min(len(buckets[0]), len(buckets[1]))
        rng.shuffle(buckets[0]); rng.shuffle(buckets[1])
        chosen = [(d, c, 0) for d, c in buckets[0][:n]] + \
                 [(d, c, 1) for d, c in buckets[1][:n]]
        rng.shuffle(chosen)
        if limit:
            chosen = chosen[:limit]
        out: List[Instance] = []
        for i, (doc, claim, gold) in enumerate(chosen):
            rid = doc_to_rid.setdefault(doc, len(doc_to_rid))
            kind = "neg" if gold == 1 else "pos"
            out.append(Instance(f"{i}:{kind}", rid, kind, gold,
                                subset_name.lower(),
                                "supported" if gold == 0 else "not_supported",
                                claim, [doc], claim))
        return out
    return _build


for _name, _subset in _AGGREFACT_SUBSETS.items():
    TASKS[_name] = TaskSpec(name=_name, subject_phrase="a grounded claim",
                            has_context=True, context_label="Grounding document")
    _BUILDERS[_name] = _make_aggrefact_builder(_subset)


def build_instances(dataset: str, limit: Optional[int] = None):
    if dataset not in _BUILDERS:
        raise ValueError(f"unknown dataset {dataset!r}; have {list(_BUILDERS)}")
    return TASKS[dataset], _BUILDERS[dataset](limit)


# --------------------- synthetic (offline / testing) ------------------
def synthetic_instances(dataset: str, n: int = 400, seed: Optional[int] = None):
    """Network-free stand-ins that respect each task's shape (context or not).

    `seed` fixes the stratum draw. Left as None the generator seeds from
    `hash(dataset)`, which Python randomises per process, so two smoke runs draw
    different strata and report slightly different mock accuracies. That is
    harmless -- the data is synthetic and no released number touches it -- and it
    is arguably the more honest default for a smoke test, since it varies the
    input rather than replaying one fixed draw.

    Pass a seed when the run must be comparable to another: `scripts/run.py --synth-seed 0`
    makes the mock panel reproducible, which is what turns `make check` from "it
    ran" into "it produced the same thing it produced last time".
    """
    import random
    task = TASKS[dataset]
    if seed is None:
        draw = hash(dataset) & 0xFFFF          # per-process, varying
    else:
        # Stable across processes and platforms: str.__hash__ is randomised by
        # PYTHONHASHSEED, so a reproducible draw cannot be built from it.
        draw = int(_hashlib_key.sha1(f"{seed}:{dataset}".encode()).hexdigest()[:8], 16)
    rng = random.Random(draw)
    if dataset == "medhallu":
        strata = ["easy", "medium", "hard"]
    elif dataset == "truthfulqa":
        strata = ["Adversarial", "Non-Adversarial"]
    elif dataset == "ragtruth":
        strata = ["ragtruth_qa"]
    elif dataset == "factscore":
        strata = ["factscore_bio"]
    else:
        strata = ["synthetic"]
    out: List[Instance] = []
    for i in range(n):
        s = rng.choice(strata)
        ctx = ["synthetic reference sentence."] if task.has_context else []
        out.append(Instance(f"{i}:pos", i, "pos", 0, s, "synthetic", "q?",
                            ctx, "clean answer"))
        out.append(Instance(f"{i}:neg", i, "neg", 1, s, "synthetic", "q?",
                            ctx, "fabricated answer"))
    return task, out