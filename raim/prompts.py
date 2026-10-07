"""Prompt construction and verdict parsing.

Prompt factors:

  condition  what the prompt tells the judge a hallucination is:
    def_on          : an explicit hallucination rubric is supplied (the study's
                      condition on the question-form sets).
    claimcheck      : the claim-support frame (see FRAME_CONDITIONS below), the
                      study's condition on the LLM-AggreFact sets.

  fmt        the reasoning format; the study uses brief_reason (think briefly,
             then the verdict line).
  paraphrase the instruction wording; the study uses p0.

All combinations request the same parseable final line:
    VERDICT: HALLUCINATED | SUPPORTED
"""
from __future__ import annotations
import re
from typing import List, Optional

FORMATS = ("brief_reason",)
PARAPHRASES = ("p0",)

# Frame conditions are a SEPARATE axis from the definition: they change the
# document/claim SCAFFOLDING of the prompt, not the hallucination definition.
# They are kept out of ALL_CONDITIONS/DEF_ON_VARIANTS on purpose, so a
# def_on-family sweep never picks one up as an extra condition to try.
# `claimcheck` reproduces the MiniCheck-style claim-support framing for the
# grounded claim-verification sets (LLM-AggreFact), where the balanced binary
# builder sets question == candidate == claim, so the standard
# question/candidate scaffolding would show the same string twice and misframe
# a summary-consistency item. Output contract and parser are unchanged.
# `resolve_frame` below is how a caller picks the frame FOR A GIVEN DATASET.
FRAME_CONDITIONS = ("claimcheck",)


def resolve_frame(frame: str, dataset: str) -> str:
    """`auto` -> the dataset's own scoring frame, the frozen methodology:
    claim-support (claimcheck) on the LLM-AggreFact family (whose builders set
    question == candidate == claim, misframed by the qa scaffolding), qa
    elsewhere. An unrecognised dataset name (typo, or a new one not yet in the
    aggrefact_* family) falls back to qa -- the more common frame, and never a
    silent guess at claimcheck -- rather than a hard failure.
    """
    if frame != "auto":
        return frame
    return "claimcheck" if dataset.startswith("aggrefact_") else "qa"

ALLOW_NOT_SURE = False  # flip True to offer NOT SURE -> abstain (abstention study)

# The explicit-definition rubric(s), keyed by condition.
DEFINITIONS = {
    # the study's rubric (grounding + examples)
    "def_on": (
        "A HALLUCINATED answer contains information that is incorrect, "
        "fabricated, or not supported by the evidence: e.g. it misreads the "
        "question, reverses a finding, invents a mechanism, or adds specifics "
        "absent from the source. A SUPPORTED answer is fully consistent with "
        "the truth/evidence."),
}
DEF_ON_VARIANTS = tuple(DEFINITIONS)          # ("def_on", ...)
ALL_CONDITIONS = DEF_ON_VARIANTS

# instruction phrasings, indexed by paraphrase
_ASK = {
    "def_on": {
        "p0": "Decide whether the candidate answer is a HALLUCINATION.",
    },
}

_FMT_INSTR = {
    "brief_reason": "Think briefly (1-2 sentences), then end with exactly one line:",
}


def _opts() -> str:
    return "HALLUCINATED, SUPPORTED, or NOT SURE" if ALLOW_NOT_SURE \
        else "HALLUCINATED or SUPPORTED"


def _context_block(task, instance) -> str:
    if not task.has_context or not instance.context:
        return ""
    ctx = instance.context
    if isinstance(ctx, str):
        ctx = [ctx]
    body = "\n".join(f"- {c.strip()}" for c in ctx if c and c.strip())
    return f"\n# {task.context_label}\n{body}\n"


def build_prompt(task, instance, condition="def_on", fmt="brief_reason",
                 paraphrase="p0") -> str:
    if condition not in ALL_CONDITIONS and condition not in FRAME_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; "
                         f"have {ALL_CONDITIONS + FRAME_CONDITIONS}")
    if fmt not in FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}")
    if paraphrase not in PARAPHRASES:
        raise ValueError(f"unknown paraphrase {paraphrase!r}")

    if condition == "claimcheck":
        return _build_claimcheck_prompt(task, instance, fmt)

    header = (f"You are assessing a candidate answer to {task.subject_phrase}. "
              f"You are given the question"
              + (", reference evidence," if task.has_context else "")
              + " and one candidate answer.")

    # any def_on* condition supplies its rubric from the registry; the off
    # conditions supply none. All def_on* reuse the def_on question phrasing.
    defn = f"\nDefinition. {DEFINITIONS[condition]}\n" if condition in DEFINITIONS \
        else "\n"
    ask_key = "def_on" if condition in DEFINITIONS else condition
    ask = _ASK[ask_key][paraphrase]
    fmt_line = f"{_FMT_INSTR[fmt]}\nVERDICT: <{_opts()}>"

    return (
        f"{header}\n{defn}\n{ask}\n\n{fmt_line}\n\n"
        f"# Question\n{instance.question.strip()}\n"
        f"{_context_block(task, instance)}"
        f"\n# Candidate answer\n{instance.candidate.strip()}\n"
    )


def _build_claimcheck_prompt(task, instance, fmt="brief_reason") -> str:
    """MiniCheck-style claim-support framing: Document / Claim / is-it-consistent.

    Drops the question/candidate-answer scaffolding of build_prompt (which shows
    the same string twice on the LLM-AggreFact sets, where question==candidate==
    claim) and presents the document and the claim directly, as a fact-checker
    would. The reasoning budget (`fmt`) and the VERDICT output contract are held
    identical to the standard prompt, so the ONLY thing that varies against the
    def_on prompt is the framing. On a QA-form set (question != candidate) the
    question is folded into the document, so the claim's support is judged in
    context; on an ungrounded set (no context) it degrades to a bare factuality
    check, but this frame is intended for the grounded claim-verification sets.
    """
    doc_parts = []
    if task.has_context and instance.context:
        ctx = instance.context
        if isinstance(ctx, str):
            ctx = [ctx]
        doc_parts += [c.strip() for c in ctx if c and c.strip()]
    # fold a genuine question into the document (QA-form sets); skip it when the
    # question is just a copy of the claim (the aggrefact sets).
    q = instance.question.strip()
    if q and q != instance.candidate.strip():
        doc_parts.insert(0, f"Question under discussion: {q}")
    document = "\n".join(doc_parts) if doc_parts else "(no document provided)"

    intro = ("Determine whether the claim below is fully supported by the "
             "document. A claim is SUPPORTED if every piece of information in it "
             "is stated in or directly entailed by the document; it is "
             "HALLUCINATED if it adds, alters, or implies anything the document "
             "does not support.")
    fmt_line = f"{_FMT_INSTR[fmt]}\nVERDICT: <{_opts()}>"
    return (
        f"{intro}\n\n{fmt_line}\n\n"
        f"# Document\n{document}\n"
        f"\n# Claim\n{instance.candidate.strip()}\n"
    )


_VERDICT_RE = re.compile(
    r"verdict\s*[:\-]?\s*\**\s*(hallucinat\w*|support\w*|not\s*sure|unsure)",
    re.IGNORECASE,
)

# label space: 1 = hallucinated, 0 = supported, None = abstain/unparseable
def parse_verdict(text: str) -> Optional[int]:
    if not text:
        return None
    m = _VERDICT_RE.findall(text)
    token = m[-1].lower() if m else None
    if token is None:
        tail = text[-200:].lower()
        if "hallucinat" in tail:
            token = "hallucinat"
        elif "support" in tail:
            token = "support"
    if token is None:
        return None
    if token.startswith("hallucinat"):
        return 1
    if token.startswith("support"):
        return 0
    return None
