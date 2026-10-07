#!/usr/bin/env bash
# THE PANEL — the study's spine, over all eight datasets.
#
# One arm per dataset, under the dataset's own frame: the definition condition
# on the four question-form sets, claim-support on the four LLM-AggreFact ones,
# resolved from the dataset name (raim/suite.py) rather than passed in. The
# suite is mixed-frame by design, matched to input structure: the AggreFact
# builders set question == candidate == claim, so the question/answer
# scaffolding would present the same string twice and misframe a claim-
# verification item (the frame contrast, scripts/run_contrast.sh, measures it).
#
# WHAT ELSE EXISTS, and why it is not here. This script measures the panel and
# nothing else. Each of the other measurements is one script, named for itself,
# alongside this one in scripts/:
#
#   run_judge.sh     every reference judge (Sonnet, the Qwen ladder, the
#                    purpose-trained judges, MiniCheck)
#   run_contrast.sh  the def_on-on-AggreFact frame contrast, panel and judges
#   run_scorer.sh    the constrained forced-choice scorer (base-vs-instruct probe)
#
# ONE ARM PER DATASET. Questions about the prompt itself are separate arm-sets
# in files of their own, so that each verdict file is the output of exactly one
# command and can be regenerated alone.
#
# GUARDS AND RESUMPTION. The panel is shard-cached per member, so an interrupted
# run resumes rather than restarting, and a box that already holds some members'
# full-context shards recomputes only the rest. A path already holding a
# DIFFERENT arm-set is never overwritten (raim/guard.py) -- including one that
# holds MORE arms than this run would write, which is the expensive mistake:
# the shard manifest hashes the arm list, so narrowing a file first clears the
# cache it was built from.
#
# DETERMINISM. The analysis layer is seed=0 exact; inference is NOT
# bit-reproducible across GPU or vLLM version, so record both with the run. A
# panel re-run CHANGES the votes, so everything panel-dependent must be
# regenerated from it (`make derive` in raim-analysis).
set -euo pipefail
cd "$(dirname "$0")/.."   # this script lives in scripts/; land at the repo root
# PY defaults, as in the Makefile, to the .venv the README creates when it
# exists, and to the python3 on PATH otherwise.
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python3 ]; then PY="$PWD/.venv/bin/python3"; else PY=python3; fi
fi
export FACTSCORE_PATH="${FACTSCORE_PATH:-$PWD/datasets/InstructGPT.jsonl}"   # factscore only

DATASETS="${DATASETS:-$("$PY" -m raim.suite --print all)}"

for ds in $DATASETS; do
  echo "==== PANEL: $ds ===="
  "$PY" scripts/run.py --dataset "$ds" --backend vllm --gpu-util 0.90
done

echo
echo "### PANEL DONE -> runs/<ds>/experiment.jsonl, each set under its own frame."
echo
echo "    Check coverage:  \"$PY\" tools/coverage_report.py --expect 10"
echo "    Then, off the GPU box:  make lock && make export"
echo "    and in raim-analysis:   make verify && make derive"
echo
echo "    The judges are a separate measurement and are NOT run here:"
echo "      bash scripts/run_judge.sh  # see LEG= in that file, or --list"
echo "    The scoring layer reads them and never regenerates them."
