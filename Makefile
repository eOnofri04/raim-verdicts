# raim-verdicts — verification targets.
#
# There is no target that runs the panel or a judge: those cost GPU
# hours or money, they are not idempotent, and they must be launched knowingly
# from the orchestration scripts. What `make` covers is everything that answers
# "is this checkout sound?", which is cheap and should be run often.
#
# PY defaults to the virtual environment the README's install block creates,
# .venv, when it exists, and to the python3 on your PATH otherwise;
#   make check PY=/path/to/python3
# picks another.

PY ?= $(if $(wildcard .venv/bin/python3),.venv/bin/python3,python3)
# A PY given as a path is made absolute and free of `..`: newer Pythons warn when
# a venv's interpreter is reached through `..`.
ifneq ($(findstring /,$(PY)),)
override PY := $(abspath $(PY))
endif

# Recipes keep their output short by showing only a tail. A pipeline's status is
# its LAST stage's, so `cmd | tail` reports success over any failure -- which is
# the one thing a verification harness must never do. It did, until 2026-08-05,
# when a cold-cache run on a fresh box printed "7 dataset(s) did not reproduce the
# committed index" and then "checkout sound: ... all pass".
#
# `SHELL := /bin/bash` with `.SHELLFLAGS := -o pipefail -c` would fix it on GNU
# make >= 3.82 and do NOTHING on macOS, which still ships 3.81 and ignores
# .SHELLFLAGS silently -- a fix that works on one of the two machines is worse
# than none. So the status is captured explicitly instead, which needs no
# particular make or shell.
#
#   $(call run_tail,<command>,<lines to show>)
run_tail = out=$$(mktemp); { $(1) ; } >$$out 2>&1; st=$$?; tail -$(2) $$out; rm -f $$out; exit $$st

# joincheck pins Hugging Face to the local cache by default, because on a machine
# holding the snapshot that is the honest check -- it must not silently re-fetch.
# On a cold cache there is nothing to read, so pass ONLINE=1 (or run
# `tools/joincheck.py --online` directly) the first time on a fresh box.
ONLINE ?= 0
ifeq ($(ONLINE),1)
  JOINCHECK_ENV  =
  JOINCHECK_ARGS = --online
  JOINCHECK_NOTE = live upstream
else
  JOINCHECK_ENV  = HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
  JOINCHECK_ARGS =
  JOINCHECK_NOTE = offline, local cache
endif

.PHONY: help runs check smoke tests joincheck indexcheck export lock lockcheck clean

help:
	@echo "make runs       — unpack the committed runs.tar.xz into runs/ and verify it"
	@echo "make check      — smoke + tests + joincheck (the full standalone check)"
	@echo "make smoke      — mock/synthetic panel end to end, no GPU, no network"
	@echo "make tests      — the offline matcher and constrained-decoding suites"
	@echo "make joincheck  — rebuild vs the committed dataset_index/ (ONLINE=1 for a cold cache)"
	@echo "make indexcheck — regenerate the index into a temp tree and diff it"
	@echo "make export     — project the verdicts into raim-analysis (FORCE=1 to override the guard)"
	@echo "make lock       — write verdicts.lock.json over the verdict tree"
	@echo "make lockcheck  — verify the verdict tree against that lockfile"
	@echo "make clean      — remove __pycache__ and the smoke-test output"

check: tests smoke joincheck
	@printf '\n### checkout sound: imports, tests, mock panel and join all pass.\n'

# The smoke panel writes into SMOKE, never into runs/. Zone 1 is the
# irreproducible tree, and a verification target has no business leaving anything
# in it: until 2026-08-06 this wrote runs/medhallu/, so a `make check` followed by
# a `make lock` would have pinned the release to a forty-item mock panel. The
# lockfile now refuses that too, but the better fix is for the check never to put
# a file there in the first place.
SMOKE ?= .smoke

smoke:
	@echo "==> mock/synthetic panel (no GPU, no network)"
	@$(PY) -m compileall -q . && echo "    every module parses"
	@for f in scripts/*.sh; do bash -n $$f || exit 1; done && echo "    every script parses"
	@rm -rf $(SMOKE)
	@$(call run_tail,$(PY) scripts/run.py --dataset medhallu \
	    --backend mock --synthetic --synth-seed 0 --limit 40 \
	    --raw $(SMOKE)/medhallu/experiment.jsonl \
	    --outdir $(SMOKE)/medhallu/experiment,2)
	@$(call run_tail,$(PY) tools/coverage_check.py $(SMOKE)/medhallu/experiment.jsonl,1)
	@rm -rf $(SMOKE)

tests:
	@echo "==> offline test suites"
	@$(call run_tail,$(PY) tests/test_matcher.py,1)
	@$(call run_tail,$(PY) tests/test_constrained.py,1)
	@$(call run_tail,$(PY) tests/test_panel_atomic.py,1)

joincheck:
	@echo "==> join vs the committed index ($(JOINCHECK_NOTE))"
	@$(call run_tail,$(JOINCHECK_ENV) $(PY) tools/joincheck.py $(JOINCHECK_ARGS),13)

# The frozen-artefact check: regenerate the index somewhere disposable and diff.
# Run this after ANY change to raim/tasks.py — it is what proves the builders
# still produce the instances the released verdicts were taken on. It never
# writes over the committed index, which is the authoritative mapping.
indexcheck:
	@echo "==> regenerating dataset_index/ into a temp tree and diffing"
	@tmp=$$(mktemp -d) && \
	  rsync -a --exclude '.git' --exclude '__pycache__' --exclude 'runs' ./ $$tmp/ && \
	  rm -rf $$tmp/dataset_index && \
	  ( cd $$tmp && HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 $(abspath $(PY)) tools/dataset_index.py >/dev/null 2>&1 ) && \
	  for f in dataset_index/*.jsonl; do \
	    if diff -q $$f $$tmp/$$f >/dev/null; then echo "    identical  $$(basename $$f)"; \
	    else echo "    DIFFERS    $$(basename $$f)"; fi; \
	  done; \
	  rm -rf $$tmp

# The zone-1/zone-2 pin. `make lock` after a measurement run, so the analysis
# repository has a release to cite; `make lockcheck` before trusting a synced
# tree, since a half-finished rsync is otherwise indistinguishable from a
# complete one. RUNS overrides the tree, e.g. RUNS=/mnt/box/runs.
RUNS ?= runs

# THE HANDOFF. This is how zone 1 becomes zone 2. Run it after `make lock`, so
# the manifest carries the release digest of the tree it was projected from.
#
#   raim-verdicts:  run the panel/judges  ->  make lock  ->  make export
#   raim-analysis:  make verify  ->  make derive  ->  make tables figures
LOCK ?= verdicts.lock.json

# FORCE=1, not --force: make parses a leading -- as its own option and fails
# with "unrecognized option" before the target ever runs.
export:
	@$(call run_tail,$(PY) tools/export_votes.py --runs $(RUNS) --lock $(LOCK) --out $(ANALYSIS)/verdicts $(if $(FORCE),--force),4)
	@echo "    now, in raim-analysis: make verify && make derive"

ANALYSIS ?= ../raim-analysis

lock:
	@$(call run_tail,$(PY) tools/verdict_lock.py --runs $(RUNS) --lock $(LOCK),4)

lockcheck:
	@$(PY) tools/verdict_lock.py --check --runs $(RUNS) --lock $(LOCK)

# Zone 1 is committed compressed, as runs.tar.xz (about 8 MB against 160 MB
# unpacked), and unpacked on demand; the lockfile then confirms the tree is the
# released one, byte for byte.
runs:
	@if [ -d runs ] && [ -n "$$(ls -A runs | grep -v '^\.gitkeep$$')" ]; then \
	  echo "runs/ is not empty; remove it first to unpack afresh"; \
	else tar -xJf runs.tar.xz && echo "unpacked runs.tar.xz into runs/"; fi
	@$(PY) tools/verdict_lock.py --check --runs runs --lock $(LOCK)

clean:
	@rm -rf __pycache__ raim/__pycache__ scripts/__pycache__ tools/__pycache__ \
	        tests/__pycache__ diagnostics/__pycache__ $(SMOKE)
	@echo "cleaned"
