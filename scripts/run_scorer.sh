#!/usr/bin/env bash
# THE CONSTRAINED SCORER — forced-choice log-prob scoring, in two stages.
#
#   bash scripts/run_scorer.sh                  # every stage, in order
#   STAGE=probe  bash scripts/run_scorer.sh     # the base-versus-instruct probe alone
#   FORCE=1 bash scripts/run_scorer.sh          # redo scoring runs and reassemble
#
# WHAT THE SCORER IS. Instead of asking a model to answer and parsing what it
# says, this reads P(Yes) against P(No) as the first continuation token of a
# few-shot completion prompt, with every other logit masked before the softmax
# (constrained decoding). No instruction-following, no generation, no parsing,
# 100% coverage by construction -- which is what makes it work identically on a
# base model and on its instruct twin, so base-versus-instruct is the only
# variable. Constraining the candidate set is what guarantees coverage: read
# from the unconstrained first-token top-k instead, some models (gemma-2-9b,
# Yi-1.5-9B) offer no Yes/No candidate at all, and others only one side.
#
#   probe   Eight matched base/instruct twin pairs, sixteen models, all eight
#           datasets, assembled into a base panel and an instruct panel per
#           dataset. Answers whether the panel's error-correlation is inherited
#           from pretraining or induced by alignment (app:base, fig_baseprobe).
#           Two of the ten panel families are absent because they have no
#           public base checkpoint at all -- Phi-4-mini (Microsoft released only
#           instruct/reasoning/multimodal) and Command-R7B (Cohere ships only the
#           aligned weights) -- so a matched comparison is impossible for them,
#           not merely omitted.
#
# FRAMES. Every run goes through scripts/run_lpscore.py at --frame auto, which
# resolves to claim-support on the four LLM-AggreFact sets and the question form
# elsewhere. The probe's def_on measurement on those four sets is kept as
# *_defon siblings under runs/lp_probe/, and scripts/run_lpscore.py checks the
# condition each file records before skipping it.
#
# WHERE THIS SCRIPT ENDS. At the gated panel files, which are the last
# MEASUREMENT artefacts in the chain: assembly concatenates scorer outputs and
# the gate decides whether they are fit to be scored at all, so both belong with
# the instrument. The analysis that consumes them lives in raim-analysis, driven
# by `make derive`, and a dataset that fails the gate never gets a panel file
# for it to read.
set -uo pipefail
cd "$(dirname "$0")/.."   # this script lives in scripts/; land at the repo root
# PY defaults, as in the Makefile, to the .venv the README creates when it
# exists, and to the python3 on PATH otherwise.
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python3 ]; then PY="$PWD/.venv/bin/python3"; else PY=python3; fi
fi
export FACTSCORE_PATH="${FACTSCORE_PATH:-$PWD/datasets/InstructGPT.jsonl}"

STAGE="${STAGE:-all}"
QA_DATASETS="$("$PY" -m raim.suite --print qa)"
CC_DATASETS="$("$PY" -m raim.suite --print cc)"
DATASETS="${DATASETS:-$("$PY" -m raim.suite --print all)}"

K=6            # few-shot exemplars; six keep the prompt within Yi-1.5-9B's 4k context
REV="v3"       # the scorer revision, carried in every member tag
# Context stays at 4096: 01-ai/Yi-1.5-9B base is a 4k-context model and is the
# binding constraint; every other base twin has >= 8k.
LP_FLAGS=(--backend vllm --k "$K" --gpu-util 0.55 --max-model-len 4096)
[[ "${FORCE:-0}" == "1" ]] && LP_FLAGS+=(--force)

declare -a FAILED=()
declare -a PENDING=()

score () {  # $1=dataset  $2="HF_id|tag"
  local ds="$1" model="${2%%|*}" tag="${2##*|}"
  echo ">>> lp-score [$ds] $model  (tag=$tag)"
  if ! "$PY" scripts/run_lpscore.py --dataset "$ds" --model "$model" --tag "$tag" \
        "${LP_FLAGS[@]}"; then
    echo "    !! FAILED: ${tag}/${ds} — continuing"
    FAILED+=("${tag}/${ds}")
  fi
}

# ============================================================================
# STAGE 1 — the base-versus-instruct probe
# ============================================================================
BASE=(
  "meta-llama/Llama-3.1-8B|llama_base_${REV}"
  "Qwen/Qwen2.5-7B|qwen_base_${REV}"
  "mistralai/Mistral-7B-v0.3|mistral_base_${REV}"
  "google/gemma-2-9b|gemma_base_${REV}"
  "01-ai/Yi-1.5-9B|yi_base_${REV}"
  "tiiuae/Falcon3-7B-Base|falcon_base_${REV}"
  "ibm-granite/granite-3.3-8b-base|granite_base_${REV}"
  "zai-org/glm-4-9b-hf|glm_base_${REV}"
)
INST=(
  "meta-llama/Llama-3.1-8B-Instruct|llama_inst_${REV}"
  "Qwen/Qwen2.5-7B-Instruct|qwen_inst_${REV}"
  "mistralai/Mistral-7B-Instruct-v0.3|mistral_inst_${REV}"
  "google/gemma-2-9b-it|gemma_inst_${REV}"
  "01-ai/Yi-1.5-9B-Chat|yi_inst_${REV}"
  "tiiuae/Falcon3-7B-Instruct|falcon_inst_${REV}"
  "ibm-granite/granite-3.3-8b-instruct|granite_inst_${REV}"
  "zai-org/glm-4-9b-chat-hf|glm_inst_${REV}"
)

assemble_probe () {  # $1=dataset  $2=arm (base|inst)  $3...=member specs
  local ds="$1" arm="$2"; shift 2
  local out="runs/lp_probe/${ds}_lp${arm}_${REV}_panel.jsonl"
  # The assembly gate (tools/assemble_panel.py): every member present, model id
  # as rostered, condition matching the dataset's frame, identical instance sets,
  # coverage and prob_yes at or above 0.95. The panel is written only if every
  # member passes, so a failing panel never reaches the analysis.
  echo ">>> assemble [$ds] ${arm} probe panel"
  if ! "$PY" tools/assemble_panel.py --dataset "$ds" --out "$out" --members "$@"; then
    echo "    !! PENDING: ${ds} ${arm} panel (member files incomplete or gate failed)"
    PENDING+=("${ds}/${arm}")
    return 1
  fi
}

if [[ "$STAGE" == "probe" || "$STAGE" == "all" ]]; then
  for ds in $DATASETS; do
    for mt in "${BASE[@]}"; do score "$ds" "$mt"; done
    for mt in "${INST[@]}"; do score "$ds" "$mt"; done
    assemble_probe "$ds" base "${BASE[@]}"
    assemble_probe "$ds" inst "${INST[@]}"
  done
fi


echo
if ((${#FAILED[@]} + ${#PENDING[@]})); then
  ((${#FAILED[@]}))  && echo "### SCORING FAILURES: ${FAILED[*]}"
  # FAILURES were attempted and did not complete; PENDING means a dataset's
  # members were not all available to assemble, or the gate refused them.
  ((${#PENDING[@]})) && echo "### DATASETS PENDING: ${PENDING[*]}"
  echo "  (fix, then re-run — completed files are skipped automatically)"
  exit 1
fi
echo "### SCORER DONE. Gated panels per dataset, under runs/lp_probe/:"
echo "    <ds>_lpbase_${REV}_panel.jsonl  <ds>_lpinst_${REV}_panel.jsonl"
echo
echo "  A log of nothing but '[skip] ... already holds' is a COMPLETE pass, not a"
echo "  stalled one. Next, off the GPU box, in raim-analysis: 'make derive'."
