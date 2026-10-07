"""Purpose-trained cheap judges — per-judge prompt adapters.

The competing route to a cheap judge trains a single small model expressly for
evaluation (Prometheus / JudgeLM / Auto-J), rather than aggregating a panel of
off-the-shelf generalists after the fact. This module supplies
the ADAPTERS that let us run each such judge, off the shelf, on the same balanced
binary faithfulness instances as the panel: each adapter maps one of our
``Instance`` objects onto the judge's NATIVE graded interface, and parses the
judge's native output back to our label space (1 = hallucinated, 0 = supported).

Each judge has its own conversation format and its own graded output, so the
mapping is genuinely per-judge:

  * Prometheus 2 (``prometheus-eval/prometheus-7b-v2.0``) — absolute grading:
    a task description + instruction + response + reference answer + a 1–5 score
    rubric, wrapped in the Mistral ``[INST]`` template, emitting
    ``Feedback: ... [RESULT] N`` with N in 1–5.
  * JudgeLM (``BAAI/JudgeLM-7B-v1.0``) — a reference-guided quality score, run
    through its single-answer grading path (score the lone Assistant answer
    1–10). NB (verified on the checkpoint, 2026-07-07/08): it needs (a) a CLOSED
    ``[The Start/End of Reference Answer]`` block after the answer — a bare
    ``[Reference Answer]`` header made it echo the reference (coverage ~0.24) — and
    (b) an assistant PREFILL of ``Score:`` at the end of the prompt, because even
    in native form it often defers the number and trails off into a qualitative
    verdict ("... a low score", coverage ~0.51). The prefill forces a numeric
    score as the first generated token; see ``_JUDGELM_PREFILL``/``_judgelm_parse``.
  * Auto-J (``GAIR/autoj-13b``) — single-response critique-then-rate, wrapped in
    the Llama-2 ``[INST]`` template, emitting ``Rating: [[N]]`` with N in 1–10.

Because each judge is graded rather than binary, the adapter thresholds the score
at the scale midpoint: score >= ``threshold`` -> supported (0), else hallucinated
(1). The midpoint is a documented default, not a tuned operating point — sweep it
on a dev split if a judge's score distribution is skewed (see ``scripts/run_ptjudge.py
--threshold``).

Frame axis (mirrors raim/prompts.py). Each ``build`` takes a ``frame``
argument: ``qa`` (default) is the question/candidate scaffolding;
``claimcheck`` is the MiniCheck-style claim-support frame the four
LLM-AggreFact sets are scored under. Those sets set
question == candidate == claim, so the qa scaffolding shows the claim twice and
misframes a claim-verification item — under ``claimcheck`` the adapter instead
presents the document (via ``_claim_document``, which folds a GENUINE question in
and skips a copied one, exactly as ``prompts._build_claimcheck_prompt`` does) and
the claim once, in the judge's native slots: the claim rides in the
response/answer position, the document in the evidence/reference position, and
only the rated dimension's wording changes. ``scripts/run_ptjudge.py --frame auto``
resolves the frame per dataset and records it in the output JSON.

Provenance of the literal templates (fetched from upstream, 2026-07-04):
  * Prometheus 2 — the ABS_SYSTEM_PROMPT + absolute-grading template are verbatim
    from the model card (prometheus-eval/prometheus-7b-v2.0); Mistral conv wrap.
  * Auto-J — the single-response template is verbatim from
    GAIR-NLP/auto-j codes/usage/constants_prompt.py (``single``); Llama-2 ``[INST]``
    wrap with no system message; rating parsed as ``Rating: [[N]]``.
  * JudgeLM — the system prompt and the ``[Question]``/``[The Start/End of
    Assistant's Answer]``/``[The Start/End of Reference Answer]``/``[System]``
    block structure are from baaivision/JudgeLM judgelm/llm_judge/common.py, in
    its NATIVE score-first single-answer form (only the rated dimension is swapped
    to faithfulness). VERIFIED on the checkpoint (2026-07-07/08): this form yields
    a leading score line, which ``_judgelm_parse`` reads first (falling back to
    trailing score-context language for the minority explanation-first outputs);
    a score-last instruction with a bare ``[Reference Answer]`` header instead
    makes the model echo the reference and omit the score.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .experimental import experimental
from .tasks import TaskSpec, Instance


def _context_text(task: TaskSpec, instance: Instance) -> str:
    """The reference evidence as a single block, or '' when ungrounded."""
    if not task.has_context or not instance.context:
        return ""
    ctx = instance.context
    if isinstance(ctx, str):
        ctx = [ctx]
    return "\n".join(c.strip() for c in ctx if c and c.strip())


FRAMES = ("qa", "claimcheck")


def _check_frame(frame: str) -> None:
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; have {FRAMES}")


def _claim_document(task: TaskSpec, instance: Instance) -> str:
    """The claim-support frame's document, mirroring prompts._build_claimcheck_prompt:
    a genuine question (QA-form sets) is folded in front of the context; a question
    that merely copies the claim (the LLM-AggreFact sets, question == candidate) is
    skipped, so the claim appears in the prompt exactly once."""
    parts = []
    q = instance.question.strip()
    if q and q != instance.candidate.strip():
        parts.append(f"Question under discussion: {q}")
    ctx = _context_text(task, instance)
    if ctx:
        parts.append(ctx)
    return "\n".join(parts) if parts else "(no document provided)"


# --------------------------------------------------------------------------- #
#  Prometheus 2 — absolute grading (1–5)                                       #
# --------------------------------------------------------------------------- #

_PROM_SYSTEM = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects the "
    "absolute standards set for performance.")

# verbatim absolute-grading template (model card). {reference} is filled with a
# rubric-style description of the ideal faithful answer rather than the gold
# response, so no label leaks; the source evidence rides in the instruction.
_PROM_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[{criteria}]
Score 1: {s1}
Score 2: {s2}
Score 3: {s3}
Score 4: {s4}
Score 5: {s5}

###Feedback:"""

_PROM_CRITERIA = ("Is the response faithful to and fully supported by the "
                  "reference evidence, without introducing unsupported or "
                  "fabricated claims?")
_PROM_RUBRIC = {
    "s1": ("The response is contradicted by the reference evidence, or its "
           "central claim is fabricated and unsupported."),
    "s2": ("The response contains a clear unsupported or incorrect claim that "
           "the reference evidence does not license."),
    "s3": ("The response is broadly on topic but adds specifics or a claim not "
           "grounded in the reference evidence."),
    "s4": ("The response is faithful to the reference evidence with only a "
           "negligible unsupported detail."),
    "s5": ("The response is entirely faithful to and supported by the reference "
           "evidence, introducing no unsupported or fabricated claim."),
}
_PROM_REFERENCE = ("A response that is entirely faithful to and supported by the "
                   "reference evidence, introducing no unsupported or fabricated "
                   "claim.")

# claim-support (claimcheck) variants: the same rubric semantics with the rated
# object renamed response->claim and evidence->document, matching the panel's
# claim-support definition ("stated in or directly entailed by the document").
_PROM_CRITERIA_CLAIM = ("Is the claim fully supported by the provided document, "
                        "without adding, altering, or implying anything the "
                        "document does not state or directly entail?")
_PROM_RUBRIC_CLAIM = {
    "s1": ("The claim is contradicted by the document, or is entirely "
           "fabricated and unsupported."),
    "s2": ("The claim contains a clear assertion that the document does not "
           "state or entail."),
    "s3": ("The claim is broadly on topic but adds specifics or an assertion "
           "not grounded in the document."),
    "s4": ("The claim is supported by the document with only a negligible "
           "unsupported detail."),
    "s5": ("The claim is fully supported: every piece of information in it is "
           "stated in or directly entailed by the document."),
}
_PROM_REFERENCE_CLAIM = ("A claim in which every piece of information is stated "
                         "in or directly entailed by the document, adding, "
                         "altering, and implying nothing beyond it.")


def _prometheus_build(task: TaskSpec, instance: Instance,
                      frame: str = "qa") -> str:
    _check_frame(frame)
    if frame == "claimcheck":
        instruction = ("Determine whether the following claim is fully supported "
                       "by the document, i.e. whether every piece of information "
                       "in the claim is stated in or directly entailed by it.\n\n"
                       "(Input — the document the claim must be supported by:)\n"
                       f"{_claim_document(task, instance)}")
        body = _PROM_TEMPLATE.format(
            instruction=instruction, response=instance.candidate.strip(),
            reference=_PROM_REFERENCE_CLAIM, criteria=_PROM_CRITERIA_CLAIM,
            **_PROM_RUBRIC_CLAIM)
        return f"[INST] {_PROM_SYSTEM}\n\n{body} [/INST]"
    ctx = _context_text(task, instance)
    if ctx:
        instruction = (f"{instance.question.strip()}\n\n"
                       f"(Input — reference evidence the answer must be faithful "
                       f"to:)\n{ctx}")
    else:
        instruction = (f"{instance.question.strip()}\n\n"
                       f"(Judge faithfulness against established world facts; no "
                       f"reference evidence is provided.)")
    body = _PROM_TEMPLATE.format(
        instruction=instruction, response=instance.candidate.strip(),
        reference=_PROM_REFERENCE, criteria=_PROM_CRITERIA, **_PROM_RUBRIC)
    # Mistral conversation wrap (get_conv_template("mistral")); vLLM prepends BOS.
    return f"[INST] {_PROM_SYSTEM}\n\n{body} [/INST]"


_PROM_RESULT_RE = re.compile(r"\[RESULT\]\s*\**\s*([1-5])")
_PROM_FALLBACK_RE = re.compile(r"\b([1-5])\s*(?:/\s*5|out of 5)")


def _prometheus_parse(text: str) -> Optional[float]:
    if not text:
        return None
    m = _PROM_RESULT_RE.findall(text)
    if m:
        return float(m[-1])
    m = _PROM_FALLBACK_RE.findall(text)  # "Score: 4/5" style stragglers
    return float(m[-1]) if m else None


def _prometheus_mock(gold: int) -> str:
    n = 2 if gold == 1 else 5
    return f"Feedback: (mock) the response is judged. [RESULT] {n}"


# --------------------------------------------------------------------------- #
#  JudgeLM — reference-guided single-answer grading (1–10)                     #
# --------------------------------------------------------------------------- #

_JUDGELM_SYSTEM = ("You are a helpful and precise assistant for checking the "
                   "quality of the answer.")

# Native single-answer format, retargeted to faithfulness (verified on the
# checkpoint 2026-07-07/08). Two lessons from the smoke run drive this wording:
#   * JudgeLM is trained to output the SCORE FIRST ("Please first output a single
#     line containing only one value ..."); asking for score-LAST made it write a
#     critique and often stop WITHOUT ever emitting a score (coverage ~0.24). We
#     therefore keep its native score-first phrasing verbatim in spirit.
#   * A bare "[Reference Answer]" header is out-of-distribution and the model
#     echoes/continues it instead of judging; its native reference form is a CLOSED
#     "[The Start of Reference Answer] ... [The End of Reference Answer]" block
#     placed AFTER the answer (see _judgelm_build), which removes the echo.
# Only the rated dimension is swapped (helpfulness/relevance -> faithfulness); the
# score-first control phrasing is left as JudgeLM was trained on.
_JUDGELM_INSTR = (
    "We would like to request your feedback on the faithfulness of the AI "
    "assistant's answer to the user question displayed above{ref_clause}. An "
    "answer is faithful when every claim it makes is supported by "
    "{support_target}; it is unfaithful when it contradicts, fabricates, or adds "
    "claims not supported there. The assistant receives an overall score on a "
    "scale of 1 to 10, where 10 means the answer is entirely faithful and fully "
    "supported and 1 means it is unsupported or fabricated.\n"
    "On the first line output only 'Score: N', where N is a single integer from "
    "1 to 10; then, on the following lines, give a brief explanation.")

# Assistant PREFILL. Even in its native score-first form the model often defers the
# number and trails off into a qualitative verdict ("... receives a low score",
# "... not faithful"), leaving no parseable score (coverage ~0.5). Because the
# runner sends raw prompts (apply_template=False), we prime the response with
# "Score:" so the model's first generated token MUST be the score — forcing it to
# commit a number before it explains. The generated text is the continuation only
# (" 4\n<explanation>"), which _judgelm_parse reads from the leading line.
_JUDGELM_PREFILL = "Score:"

# claim-support (claimcheck) variants: the claim rides in the answer slot (rated
# object), the document in the native CLOSED reference block, and a fixed
# claim-verification question fills [Question] -- so the claim appears once and
# the in-distribution question/answer/reference structure is preserved.
_JUDGELM_QUESTION_CLAIM = ("Is the following claim fully supported by the "
                           "provided reference document?")
_JUDGELM_INSTR_CLAIM = (
    "We would like to request your feedback on whether the claim given as the "
    "assistant's answer above is supported by the provided reference document. "
    "A claim is supported when every piece of information it contains is stated "
    "in or directly entailed by the reference document; it is unsupported when "
    "it adds, alters, or implies anything the document does not support. The "
    "assistant receives an overall score on a scale of 1 to 10, where 10 means "
    "the claim is fully supported by the reference document and 1 means it is "
    "unsupported or fabricated.\n"
    "On the first line output only 'Score: N', where N is a single integer from "
    "1 to 10; then, on the following lines, give a brief explanation.")


def _judgelm_build(task: TaskSpec, instance: Instance,
                   frame: str = "qa") -> str:
    _check_frame(frame)
    if frame == "claimcheck":
        parts = [_JUDGELM_SYSTEM, "",
                 "[Question]", _JUDGELM_QUESTION_CLAIM, "",
                 "[The Start of Assistant's Answer]", instance.candidate.strip(),
                 "[The End of Assistant's Answer]", "",
                 "[The Start of Reference Answer]",
                 _claim_document(task, instance),
                 "[The End of Reference Answer]", "",
                 "[System]", _JUDGELM_INSTR_CLAIM, "", _JUDGELM_PREFILL]
        return "\n".join(parts)
    ctx = _context_text(task, instance)
    parts = [_JUDGELM_SYSTEM, "",
             "[Question]", instance.question.strip(), "",
             "[The Start of Assistant's Answer]", instance.candidate.strip(),
             "[The End of Assistant's Answer]", ""]
    if ctx:
        # native reference form: a CLOSED block AFTER the answer, before [System].
        parts += ["[The Start of Reference Answer]", ctx,
                  "[The End of Reference Answer]", ""]
        instr = _JUDGELM_INSTR.format(
            ref_clause=" and the provided reference answer",
            support_target="the reference")
    else:
        instr = _JUDGELM_INSTR.format(
            ref_clause="", support_target="established world facts")
    # end the prompt on the prefill so generation continues from " N".
    parts += ["[System]", instr, "", _JUDGELM_PREFILL]
    return "\n".join(parts)


# With the "Score:" prefill the generated continuation begins with the score
# (" 4", "4/10", "Score: 4" if the model repeats the cue, or "10"). We read that
# leading number first. The regex matches 10 or 1–9 (optional decimal) at the very
# start, requiring the next char to be a boundary (end / space / slash / dot-space),
# so a run-on like "45" does not misfire.
# CAVEAT (see @experimental below): the regex ALSO matches a genuine list marker
# such as "1. The response is faithful..." -- verified directly, not assumed --
# and cannot be told apart syntactically from a score written as "8. The answer
# ...". It has never fired on real data: checked against every released JudgeLM raw output (7,670
# rows, all eight datasets), 7,610 took this leading-regex path and none had that
# shape, so the "Score:" prefill discipline has held in practice even though the
# regex alone does not structurally guarantee it. As a fallback (should the
# prefill ever fail) we anchor on trailing score-context language and take the
# LAST match — never a bare first prose number (e.g. "30-day", "L3").
_JUDGELM_LEADING_RE = re.compile(
    r"^(?:score\s*[:=]?\s*)?(10(?:\.0+)?|[1-9](?:\.\d+)?)(?=$|[^\d.]|\.\D|\.$)", re.I)
_JUDGELM_FINAL_RE = re.compile(r"\bscores?\s*[:=]\s*(\d+(?:\.\d+)?)", re.I)
_JUDGELM_CTX_RES = (
    re.compile(r"\bscores?\s+(?:of|is|was|would\s+be|:|=)\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"\brat(?:e|ed|ing)\s+(?:it\s+)?(?:a\s+|as\s+|of\s+)?(\d+(?:\.\d+)?)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10\b", re.I),
)


def _judgelm_parse(text: str) -> Optional[float]:
    if not text:
        return None
    # 1) native score-first: the first non-empty line is (or begins with) the score.
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _JUDGELM_LEADING_RE.match(line)
        if m:
            v = float(m.group(1))
            if 1.0 <= v <= 10.0:
                return v
        break  # first non-empty line is not a bare score -> try the fallback
    # 2) fallback: explanation-then-verdict. Terminal "Score: N" first, else any
    #    score-context phrasing; take the LAST valid match (the final judgment).
    fin = _JUDGELM_FINAL_RE.findall(text)
    for v in reversed(fin):
        if 1.0 <= float(v) <= 10.0:
            return float(v)
    cands = []
    for rx in _JUDGELM_CTX_RES:
        for m in rx.finditer(text):
            v = float(m.group(1))
            if 1.0 <= v <= 10.0:
                cands.append((m.end(), v))
    if cands:
        cands.sort()
        return cands[-1][1]
    return None


def _judgelm_mock(gold: int) -> str:
    n = 3 if gold == 1 else 8
    return (f"{n}\nThe assistant's answer is assessed against the reference.")


# --------------------------------------------------------------------------- #
#  Auto-J — single-response critique-then-rate (1–10)                          #
# --------------------------------------------------------------------------- #

# verbatim single-response template (constants_prompt.py, `single`).
_AUTOJ_TEMPLATE = """Write critiques for a submitted response on a given user's query, and grade the response:

[BEGIN DATA]
***
[Query]: {query}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""


def _autoj_build(task: TaskSpec, instance: Instance,
                 frame: str = "qa") -> str:
    _check_frame(frame)
    if frame == "claimcheck":
        # the claim rides in [Response] (the rated object); the document, in the
        # [Query] slot, so the claim appears once and the native single-response
        # critique-then-rate structure is preserved.
        query = ("Determine whether the response, read as a standalone claim, is "
                 "fully supported by the following document, and let the rating "
                 "reflect that support (10 = every piece of information in the "
                 "claim is stated in or directly entailed by the document, 1 = "
                 "the claim adds, alters, or implies things the document does "
                 f"not support):\n{_claim_document(task, instance)}")
        body = _AUTOJ_TEMPLATE.format(query=query,
                                      response=instance.candidate.strip())
        return f"[INST] {body} [/INST]"
    ctx = _context_text(task, instance)
    if ctx:
        query = (f"{instance.question.strip()}\n\n"
                 f"Judge whether the response is faithful to and fully supported "
                 f"by the following reference evidence, and let the rating reflect "
                 f"that faithfulness (10 = entirely supported, 1 = unsupported or "
                 f"fabricated):\n{ctx}")
    else:
        query = (f"{instance.question.strip()}\n\n"
                 f"Judge whether the response is factually correct, and let the "
                 f"rating reflect that (10 = entirely correct, 1 = false or "
                 f"fabricated).")
    body = _AUTOJ_TEMPLATE.format(query=query, response=instance.candidate.strip())
    # Llama-2 [INST] wrap, no system message (build_autoj_input); vLLM prepends BOS.
    return f"[INST] {body} [/INST]"


_AUTOJ_RATING_RE = re.compile(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]")


def _autoj_parse(text: str) -> Optional[float]:
    if not text:
        return None
    m = _AUTOJ_RATING_RE.findall(text)
    if not m:
        return None
    v = float(m[-1])
    return v if 1.0 <= v <= 10.0 else None


def _autoj_mock(gold: int) -> str:
    n = 3 if gold == 1 else 8
    return f"(mock) critique of the response.\nRating: [[{n}]]"


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PTJudge:
    """One purpose-trained judge and its native adapter.

    build(task, instance, frame="qa") -> the fully-templated raw prompt
      (conversation markers included; the runner calls the backend with
      apply_template=False). frame="claimcheck" is the claim-support scaffolding
      for the re-framed LLM-AggreFact sets (module docstring).
    parse(text) -> the native graded score, or None if unparseable/abstain.
    score_range, threshold -> map score to label: score >= threshold is SUPPORTED
      (0), below is HALLUCINATED (1). Higher score always means more faithful.
    mock(gold) -> a native-format string correlated with gold, for offline
      (no-GPU) validation of the adapter's parser via `--backend mock`.
    """
    key: str
    model_id: str
    max_new_tokens: int
    max_model_len: int
    dtype: str
    score_range: Tuple[float, float]
    threshold: float
    build: Callable[..., str]
    parse: Callable[[str], Optional[float]]
    mock: Callable[[int], str]

    def label(self, score: Optional[float]) -> Optional[int]:
        """Native score -> {1 hallucinated, 0 supported, None abstain}."""
        if score is None:
            return None
        return 0 if score >= self.threshold else 1


PT_JUDGES = {
    "prometheus2": PTJudge(
        key="prometheus2",
        model_id="prometheus-eval/prometheus-7b-v2.0",
        # Fine-tuned from Mistral-7B-Instruct-v0.2 (32k context), so it comfortably
        # runs at the study's 8192 window — matching the frontier ladder and
        # Sonnet's full-context API view, removing a truncation confound. The
        # middle-out guard only fires above 8192.
        max_new_tokens=512, max_model_len=8192, dtype="bfloat16",
        score_range=(1.0, 5.0), threshold=3.0,
        build=_prometheus_build, parse=_prometheus_parse, mock=_prometheus_mock),
    "judgelm": PTJudge(
        key="judgelm",
        model_id="BAAI/JudgeLM-7B-v1.0",
        # JudgeLM-7B-v1.0 is fine-tuned from Vicuna-v1.3 (LLaMA-1), whose
        # max_position_embeddings is 2048 (RoPE) — vLLM hard-errors on any larger
        # max_model_len, and forcing it past 2048 yields nan. So the window is
        # fixed at the checkpoint's true 2048; on long-document sets the middle-out
        # guard truncates the reference evidence (it PRINTS when it fires). The
        # verdict comes at the END of the critique (see _judgelm_parse), so the
        # output budget must be large enough to REACH it — too small and the score
        # is cut off (coverage collapses). 200 leaves ~1720 prompt tokens at 2048.
        max_new_tokens=200, max_model_len=2048, dtype="bfloat16",
        score_range=(1.0, 10.0), threshold=5.5,
        build=_judgelm_build, parse=_judgelm_parse, mock=_judgelm_mock),
    "autoj": PTJudge(
        key="autoj",
        model_id="GAIR/autoj-13b",
        # Built on LLaMA-2-13B-chat, whose max_position_embeddings is 4096 — that
        # is the architectural ceiling (past it → RoPE nan), so unlike Prometheus~2
        # this judge cannot reach the study's 8192 window. On long-document sets
        # the middle-out guard truncates the evidence (it PRINTS when it fires) —
        # an inherent handicap, recorded in the appendix.
        max_new_tokens=1024, max_model_len=4096, dtype="bfloat16",
        score_range=(1.0, 10.0), threshold=5.5,
        build=_autoj_build, parse=_autoj_parse, mock=_autoj_mock),
}


@experimental(
    "JudgeLM's leading-score regex (_judgelm_parse) also matches a list marker "
    "such as '1. ...', which no parser can tell apart from a score written as "
    "'8. ...'. It never fired on any released row (see the note above "
    "_JUDGELM_LEADING_RE), but that rests on the 'Score:' prefill holding, not "
    "on the regex itself; check parse coverage and leading-line shapes on any "
    "new run.")
def get_judge(key: str) -> PTJudge:
    if key not in PT_JUDGES:
        raise ValueError(f"unknown purpose-trained judge {key!r}; "
                         f"have {sorted(PT_JUDGES)}")
    return PT_JUDGES[key]
