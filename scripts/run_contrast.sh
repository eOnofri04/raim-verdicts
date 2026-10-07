#!/usr/bin/env bash
# THE FRAME CONTRAST — def_on on the four LLM-AggreFact sets.
#
# These four are scored under a claim-support frame, because their balanced
# binary builder makes the question and the candidate coincide, so the
# question/answer scaffolding of def_on presents the same claim twice. The paper
# reports what happens when they ARE scored that way (tab_frame, app:extbase: the
# unweighted panel and Sonnet under both frames), and those are claims about a
# measurement, so the measurement has to stay reproducible.
#
# This is the WRONG frame for these sets, on purpose. It is therefore the one
# script in the tree that measures something the methodology forbids elsewhere,
# and everything about how it is kept apart is deliberate:
#
#   * the panel arm lands in experiment_defon.jsonl, never experiment.jsonl.
#     Every generator addresses its input by name and none of them globs, so an
#     arm under a different name in the same directory is invisible to all of
#     them -- whilst still sitting inside the tree that tools/verdict_lock.py and
#     tools/export_votes.py walk, so it is pinned by the release digest and mirrored
#     into raim-analysis for free, at about half a megabyte.
#   * the judge arms land in judge_<tag>_defon.json, likewise.
#   * shards go to experiment_defon_shards/, which ends in the marker both the
#     lock and the export exclude, so the cache is not double-counted.
#
# What must never happen is this arm landing in the canonical file, because the
# scoring layer selects def_on wherever it finds it and would silently prefer it
# over the claim-support run. The arm-set's filename is fixed in raim/suite.py
# rather than passed in, and raim/guard.py refuses the collision independently.
#
# PANEL AND JUDGES TOGETHER. It is one experiment, so it is one script.
#
#   bash scripts/run_contrast.sh              # panel + Sonnet (all tab_frame needs)
#   LEG=all bash scripts/run_contrast.sh      # panel + every judge that has a def_on arm
#   PANEL_ONLY=1 bash scripts/run_contrast.sh # just the panel
#
# The judge step defaults to Sonnet, not `all`: every other contrast leg is
# further spend that no table reads, so it is something to ask for by name.
#
# COST. The panel arm is 23,440 generations across the four sets, the single
# def_on condition analyze_frame_contrast.py reads.
set -euo pipefail
cd "$(dirname "$0")/.."   # this script lives in scripts/; land at the repo root
# PY defaults, as in the Makefile, to the .venv the README creates when it
# exists, and to the python3 on PATH otherwise.
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python3 ]; then PY="$PWD/.venv/bin/python3"; else PY=python3; fi
fi
export FACTSCORE_PATH="${FACTSCORE_PATH:-$PWD/datasets/InstructGPT.jsonl}"

CC_DATASETS="$("$PY" -m raim.suite --print cc)"

echo "==== CONTRAST: def_on panel on the LLM-AggreFact sets ===="
echo "     Measures these sets under the def_on frame, as"
echo "     runs/<ds>/experiment_defon.jsonl -- its own name and its own shard"
echo "     cache, so nothing that reads experiment.jsonl can reach it."
for ds in $CC_DATASETS; do
  "$PY" scripts/run.py --dataset "$ds" --armset defon --backend vllm \
      --gpu-util 0.90 --max-model-len 8192
done


if [[ "${PANEL_ONLY:-0}" != "1" ]]; then
  echo
  echo "==== CONTRAST: def_on judges on the LLM-AggreFact sets ===="
  LEG="${LEG:-sonnet}" CONTRAST=1 bash scripts/run_judge.sh
fi

echo
echo "### CONTRAST DONE. The canonical experiment.jsonl in each set is untouched."
echo "    These sit inside the verdict tree, so \`make lock\` pins them and"
echo "    \`make export\` mirrors them. Then, in raim-analysis, \`make derive\`"
echo "    scores the contrast (analyze_frame_contrast.py) and writes tab_frame."
