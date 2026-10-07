# Bringing the instrument up on a fresh box

A from-scratch procedure for re-running the measurement on your own hardware: build an environment, rebuild the datasets against upstream and check them against the committed join index, run real inference, and compare what comes out with the released verdicts.

Run it in a directory of its own, with its own environment and its own model cache, so that a pass means the repository is self-sufficient rather than that some local leftover happened to fill a gap.

## What a pass does and does not establish

**Establishes.**
The repository is self-sufficient.
The dataset builders reproduce the committed join index exactly from a cold cache.
Real inference runs through `raim/panel.py`, including the sharded worker path.
Emitted verdicts carry a join key that matches the index.
The coverage gates fire against real output rather than fixtures.

**Does not establish, and cannot.**
That the new verdicts equal the released ones.
Inference is not bit-reproducible across GPU model or vLLM version — that asymmetry is the whole reason the repositories are split — so expect the votes to differ slightly and do not treat that as a failure.
What must match exactly is the **join**: same instances, same identifiers, same content keys.

> **A pass on the dataset check is the strong result.** If `joincheck` reports MATCH on all eight from a cold cache on a machine that has never seen this project, then the released verdicts are anchored to source rows by something better than a local cache directory.

## 0. What you need

- an **`HF_TOKEN`** — `lytang/LLM-AggreFact` is gated, and the Llama-3.1 and gemma-2 weights require accepting their licences;
- the **FActScore annotation file**, which this repository does not redistribute (see `NOTICE` §3, and step 3 below);
- for inference, a GPU box; the 32B and 72B reference judges of `scripts/run_judge.sh` potentially need two cards at full context, whereas the panel runs on one.

## 1. An isolated tree

```bash
export RAIMTEST=$HOME/raim-freshcheck
mkdir -p "$RAIMTEST"
cd "$RAIMTEST"
git clone https://github.com/eOnofri04/raim-verdicts.git
git clone https://github.com/eOnofri04/raim-analysis.git     # for step 7

# a cache of its own, so nothing is served from a previous install
export HF_HOME="$RAIMTEST/hf"
export HF_TOKEN=...            # your token
mkdir -p "$HF_HOME"
```

Keeping `HF_HOME` inside the test tree is the point of the exercise: with a warm shared cache the dataset check silently degrades into the offline one.

## 2. The measurement environment

The `Makefile` and the scripts default to the `.venv` created below; exporting `PY` as well makes every command name its interpreter explicitly, whichever directory it runs from.

```bash
cd "$RAIMTEST/raim-verdicts"
python3 -m venv .venv
.venv/bin/pip install -r requirements-core.txt      # numpy, scikit-learn, datasets==5.0.0
.venv/bin/pip install -e .                          # raim itself, importable from any directory
.venv/bin/pip install vllm transformers             # the GPU half
export PY="$RAIMTEST/raim-verdicts/.venv/bin/python3"
```

Pin `datasets==5.0.0` as the requirements file does.
A newer release can change row ordering or split handling by itself, and since every instance identifier is positional over the loaded rows, a `DRIFT` verdict in step 4 would then not distinguish upstream having moved from the library having moved.

**Record the versions now**, since inference is not reproducible without them:

```bash
.venv/bin/python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

### Where a new stack can bite

The offline suites use a fake tokeniser and the mock backend imports neither vLLM nor `transformers`, so nothing before step 5 exercises them, and the first real run is a compatibility test as much as a correctness one.
Two places in `raim/backends.py` are worth watching.

- `_choice_token_ids` builds its allowed-token set from `AutoTokenizer.get_vocab()`, so a tokeniser change in `transformers` surfaces as changed `|pos_ids|`/`|neg_ids|` counts, which `score_choice` prints on first use.
- The constrained scorer passes `allowed_token_ids` to `SamplingParams` and reads first-token log-probabilities, an API that has moved considerably across vLLM versions.

Neither affects the generative panel that writes `experiment.jsonl`.
Both affect `scripts/run_lpscore.py`, so run that separately and read the printed identifier counts before trusting scorer output: a vocabulary mismatch there silently zeroes a member's coverage rather than raising.

### If the run dies before generating anything

**`RuntimeError: Ninja build failed` / `error: "CUDA versions below 12 are not supported."`**

The model loads (`Loading safetensors checkpoint shards: 100%`) and the failure comes at the first sampling step, inside `flashinfer_sample`.
vLLM logged `Using FlashInfer for top-p & top-k sampling` and then tried to JIT-compile that kernel with `/usr/bin/nvcc` — the *system* toolkit, which is older than CUDA 12.
`nvidia-smi` reporting a CUDA version of 12 or above does not contradict this: that is the driver's ceiling, not the toolkit's version.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

**This cannot change any verdict.**
`raim/backends.py` sets `TEMPERATURE = 0.0` and never sets `top_p` or `top_k`, so decoding is greedy and the top-p/top-k sampler does no filtering at all — it is pure overhead on this workload.
Disabling it removes a code path we do not use.

The alternative is to install a CUDA toolkit of 12 or later and point `CUDA_HOME` at it.
A venv built with `--system-site-packages` off also avoids mixing another Python distribution's headers into the compile.

## 3. The one file that is not in the repository

```bash
.venv/bin/python3 tools/fetch_factscore.py
```

It will report `NOT FOUND` and print the route: the Google Drive link in the README of `github.com/shmsw25/FActScore`, file `data/labeled/InstructGPT.jsonl`, placed at `datasets/InstructGPT.jsonl`.
Re-run it until it reports `OK`.

That check is not ceremony.
The builder keys FActScore instances by **line number**, and the sibling releases `ChatGPT.jsonl` and `PerplexityAI.jsonl` share the schema exactly — so the wrong file yields 330 entirely plausible instances joined to the wrong biographies, and only the checksum catches it.

## 4. Rebuild the datasets against live upstream

This is the strongest check available and it costs no GPU.

**Order matters here, and the obvious order is wrong.**
`make joincheck` pins Hugging Face to the local cache by default, because on a machine that holds the snapshot a silent re-fetch would defeat the check.
On a *cold* cache there is nothing to read, so running it first gives seven `BUILD FAILED  ConnectionError` lines that mean only "no cache yet".
Go online first, which both verifies against upstream and populates the cache:

```bash
.venv/bin/python3 tools/joincheck.py --online     # fetch at the pinned revisions, and fill the cache
.venv/bin/python3 tools/joincheck.py --unpinned   # then ask whether upstream itself has moved
make check PY="$PY"                                # now the offline leg has something to read
```

Equivalently, `make check ONLINE=1 PY="$PY"` does the online form in one step.

**Run both online forms, because they answer different questions.**
`--online` resolves the commit SHAs `DATASET_REVISIONS` pins, so it asks whether those pins still resolve and still yield the committed mapping.
`--unpinned` ignores the pins and resolves each source at its current default branch, so it asks whether upstream has moved since the reported runs.
Only the second is a drift test; a pinned build cannot see past its own pin.

**Reading it.**
`MATCH` on all eight under `--online` means the builders still produce exactly the instances the released verdicts were taken on, on hardware that has never seen this project.
`DRIFT` under `--unpinned` means upstream has moved; the committed `dataset_index/` remains authoritative and no reported number changes.
`DRIFT` under `--online` is more serious, since it means a pinned revision no longer yields what it did.

> **Do not run `tools/dataset_index.py --online` here.** That script *writes* the index. Running it against a possibly-moved upstream would replace the authoritative mapping with one built from whatever upstream now says — the exact failure the index exists to prevent. `tools/joincheck.py` only reads.

## 5. Real inference, smallest first

`aggrefact_cnn` is the cheapest set at n=114.
Work in a copy of the tree, or move the released `runs/` aside first, so that new verdicts do not land beside the released ones.

```bash
.venv/bin/python3 scripts/run.py --dataset aggrefact_cnn \
    --backend vllm --gpu-util 0.90 --limit 20
```

`--backend vllm` matters: `--backend mock` runs in process and never writes a shard, so it exercises none of the code this test is for.

Then check what came out:

```bash
.venv/bin/python3 tools/coverage_check.py runs/aggrefact_cnn/experiment.jsonl

.venv/bin/python3 - <<'PY'
import json, pathlib
rec = json.loads(open("runs/aggrefact_cnn/experiment.jsonl").readline())
idx = {json.loads(l)["uid"]: json.loads(l)["src_sha1"]
       for l in open("dataset_index/aggrefact_cnn.jsonl")}
print("uid      ", rec["uid"])
print("emitted  ", rec["src_key"])
print("index    ", idx[rec["uid"]])
print("JOIN OK" if rec["src_key"] == idx[rec["uid"]] else "JOIN MISMATCH")
PY
```

The join must be `JOIN OK`.
The verdicts themselves need not match anything.

Confirm the sharded path actually ran:

```bash
ls runs/aggrefact_cnn/experiment_shards/
```

Ten shards, one per panel member.
If that directory is empty, the run took the in-process path and this test proved nothing about sharding.

## 6. Then the full set, if step 5 is clean

```bash
bash scripts/run_panel.sh               # all eight; hours of GPU
.venv/bin/python3 tools/coverage_report.py --expect 10
.venv/bin/python3 tools/verdict_lock.py --runs runs --lock this-box.lock.json
```

`scripts/run_panel.sh` runs **each set under its own frame**: it loops `scripts/run.py` over all eight datasets, and `run.py` resolves each dataset's own frame automatically (`raim.suite`) — `def_on` on the four QA-form sets, a claim-support arm on the four LLM-AggreFact ones — one arm each, written under its own name and its own shard cache.
Re-running it resumes rather than recomputing or overwriting anything.

> **Write this box's lockfile somewhere of its own**, as above, rather than over `verdicts.lock.json`.
> That file describes the released tree; a fresh box holds only what it has just measured, so `make lock` will refuse rather than replace it, which is the guard working.

### The frame contrast, optionally

The released tree also carries the `def_on` arm on the AggreFact sets, as `experiment_defon.jsonl` beside each canonical file, because the paper reports what that frame did to those sets.
It is not needed to reproduce any headline number, so it has a script of its own rather than running by default:

```bash
PANEL_ONLY=1 bash scripts/run_contrast.sh   # the panel arm, 23,440 generations
bash scripts/run_contrast.sh                # and every judge that has a def_on arm
```

## 6a. Is the new measurement the same measurement?

New verdicts will not equal the released ones and should not be expected to.
What must hold is the join and the behaviour; `tools/compare_runs.py` separates the two.

```bash
$PY tools/compare_runs.py --a /path/to/the/released/runs --b runs
```

It checks exactly — instance identifiers, count and gold labels, a difference in which is a dataset problem rather than an inference one — and reports closely, per member: agreement on shared instances, Cohen's kappa against gold in each run, and coverage.
It exits non-zero if any member falls below `--min-agreement` (0.95 by default).

Reading it: greedy decoding on identical prompts should agree overwhelmingly, so a member at 0.93 means the model, the chat template or the truncation behaviour moved, not that sampling was noisy.
Coverage is reported separately because a parser that stopped matching a newer model's phrasing shows up there and in no other column.

All eight compare like for like, since `scripts/run_panel.sh` produces each set under the frame the released tree carries.
The contrast arm, if it was run, compares separately — `tools/compare_runs.py` matches on `experiment.jsonl` and ignores the sibling `experiment_defon.jsonl`.

> **Judges need not be re-measured.**
> They are independent measurements over the same instances, so where the join checks out they carry over unchanged; copying them from the released tree is sound rather than a shortcut.
> A tree assembled that way is a hybrid — new panel, released judges — and its lockfile should carry a name that says so.

## 6b. Exporting a new tree for the analysis half

`make export` writes the votes-only mirror that `raim-analysis` reads, and refuses to overwrite a mirror with a tree its lockfile does not describe.
To exercise it safely, pin this box's tree to its own lockfile and send the export somewhere disposable:

```bash
.venv/bin/python3 tools/verdict_lock.py --runs runs --lock this-box.lock.json
make export LOCK=this-box.lock.json ANALYSIS=/tmp/export-test PY="$PY"
python -m json.tool /tmp/export-test/verdicts/MANIFEST.json | head -20
```

Without `LOCK=`, the export is refused and says so: the committed lockfile describes the release, not this tree.
The guard permits re-exporting the *same* release over itself, since that is a no-op by construction, and `--force` overrides it.

## 7. The analysis half

`raim-analysis/QUICKSTART.md` rebuilds every table and figure from the released mirror.
It does not depend on steps 5 and 6 at all: it reads the committed verdicts, so a difference there means the analysis code changed rather than the measurement.
To run the analysis over a new tree instead, replace its `verdicts/` with the export of step 6b and run `make derive` before `make tables`, as its README explains.
