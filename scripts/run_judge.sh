#!/usr/bin/env bash
# EVERY REFERENCE JUDGE, selected by LEG. The panel is compared against these.
#
#   bash scripts/run_judge.sh                       # list the legs and their state
#   LEG=sonnet bash scripts/run_judge.sh            # one judge, all eight datasets
#   LEG=oss    bash scripts/run_judge.sh            # a family (api, oss, pt, specialist)
#   LEG=all    bash scripts/run_judge.sh            # everything whose prerequisites exist
#   LEG=qwen32b,qwen72b bash scripts/run_judge.sh   # a comma-separated mix
#   LEG=all PLAN=1 bash scripts/run_judge.sh        # print what would run, run nothing
#   LEG=sonnet FORCE=1 bash scripts/run_judge.sh    # redo a measurement of the RIGHT frame
#
# There is NO DEFAULT leg on purpose: `all` is real money and real GPU hours,
# so it is a sentence someone has to type. Bare invocation prints the registry.
#
# THE LEGS live in raim/legs.py, one row each: model id, tag, tensor-parallel
# size, which card to pin, which API key must be present, which interpreter can
# import the package. Everything else about them is identical -- one model over
# the same eight datasets under the dataset's own frame, writing
# runs/<ds>/judge_<tag>.{jsonl,json}, scored by the same estimator -- so the
# loop is written once, in Python, where the frames and the guard already are.
#
# FAMILIES
#   api          Sonnet, the frontier reference. No GPU; needs ANTHROPIC_API_KEY.
#                Ballpark spend at snapshot rates over ~7.7k items: ~$20.
#                VERIFY the model id and the $/Mtok rate in scripts/run_judge.py's
#                API_USD_PER_MTOK before a paid pass -- both move; an unknown
#                model still records token usage, with cost_usd=None.
#   oss          Qwen2.5 32B + 72B AWQ at full context (8192), the scale ladder.
#                72B runs TP=2 across both cards; 32B TP=1 pinned to GPU 1.
#   pt           Prometheus-2, JudgeLM, Auto-J: the competing route to a cheap
#                judge, one model fine-tuned expressly for evaluation. Auto-J
#                wants 48 GB in one piece or TP=2 (AUTOJ_TP=1 AUTOJ_GPUS=0 on a
#                single 48 GB card).
#   specialist   MiniCheck-Flan-T5 and Bespoke-MiniCheck, through the OFFICIAL
#                minicheck package in ITS OWN venv -- never the main .venv, which it
#                would break by pinning its own vllm/torch. Set MC_PY to that
#                interpreter (default ~/minicheck-env/bin/python3); grounded
#                sets only, since a specialist verifies a claim against a
#                document.
#
# RESUMABLE. An existing measurement under the RIGHT frame is skipped; FORCE=1
# redoes it. A frame MISMATCH always refuses, FORCE or not, because a mismatch
# means the file is not the thing being redone and its one copy may be
# irreplaceable -- moving it aside is a human's decision.
#
# The four aggrefact_* sets come from lytang/LLM-AggreFact, gated on the HF Hub:
# set HF_TOKEN (or `huggingface-cli login`) or they will fail to download.
set -uo pipefail
cd "$(dirname "$0")/.."   # this script lives in scripts/; land at the repo root
# PY defaults, as in the Makefile, to the .venv the README creates when it
# exists, and to the python3 on PATH otherwise.
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python3 ]; then PY="$PWD/.venv/bin/python3"; else PY=python3; fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set -- the four gated aggrefact_* sets will fail" >&2
  echo "         unless the HF datasets are already cached or you are logged in." >&2
fi
# factscore needs the local InstructGPT.jsonl, which is obtained rather than
# redistributed (tools/fetch_factscore.py gives the route and verifies it).
export FACTSCORE_PATH="${FACTSCORE_PATH:-$PWD/datasets/InstructGPT.jsonl}"

ARGS=()
[[ -n "${LEG:-}" ]]            && ARGS+=(--leg "$LEG")
[[ "${PLAN:-0}" == "1" ]]      && ARGS+=(--plan)
[[ "${FORCE:-0}" == "1" ]]     && ARGS+=(--force)
[[ "${CONTRAST:-0}" == "1" ]]  && ARGS+=(--contrast)

# ${ARGS[@]+"${ARGS[@]}"}, not "${ARGS[@]}": macOS ships bash 3.2, where an
# EMPTY array expanded under `set -u` is an unbound variable and aborts the
# script -- which is exactly the bare `bash scripts/run_judge.sh` that prints the
# registry, the one invocation a first-time reader makes.
exec "$PY" -m raim.legs ${ARGS[@]+"${ARGS[@]}"} "$@"
