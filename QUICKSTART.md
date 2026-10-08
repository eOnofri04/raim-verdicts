# Quickstart — raim-verdicts

The short version. `README.md` has the detail; this page is for deciding whether you need this repository at all, and getting it running if you do.

## What this is, in a paragraph

RAIM asks whether a panel of ten cheap open-weight language models, used as hallucination judges and combined by a small learned aggregator, can stand in for an expensive frontier judge like Claude Sonnet — and, more usefully, *when* it can.
This repository is the instrument that produces the evidence: it runs the panel and the reference judges over eight benchmarks and records what each judge said about each item.
Its companion, [`raim-analysis`](https://github.com/eOnofri04/raim-analysis) (cloned beside this one, at `../raim-analysis`), turns those recordings into the paper's numbers.

The split is on cost, not on subject matter.
Everything here costs GPU hours or paid API calls and cannot be reproduced exactly on other hardware.
Everything there runs on a laptop in minutes and reproduces the paper's tables exactly, and its intermediate results to within float drift.

## Do you actually need this repository?

| you want to | go to |
|---|---|
| re-derive the paper's tables and figures | **`raim-analysis`** — its input is its own `verdicts/`; you do not need a GPU |
| check how a number was computed | **`raim-analysis`** |
| read what each judge actually answered | here — `runs/` (`make runs` unpacks it), generated text included |
| check that the analysis input is the projection of these judgements | here — `$PY tools/export_votes.py --check` |
| understand how items map to source rows | here — `dataset_index/` and `tools/joincheck.py` |
| re-run the panel on your own hardware, or judge a new dataset | here |

## What it contains

For every item of every benchmark, one line per judge, recording which judge was asked, under which prompt framing, and what it answered: 284 files under `runs/`, committed compressed as `runs.tar.xz` and pinned by `verdicts.lock.json`.  
**No source text**: no question, document or candidate answer is stored, so this repository redistributes none of the benchmarks it measures — you obtain those from their own publishers, under their own terms (`NOTICE`).

That creates one obligation, met by `dataset_index/`: a content-addressed map from every released verdict back to the source row it was taken on, so a third party can verify the join without our shipping any benchmark content.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-core.txt        # numpy, scikit-learn, datasets
.venv/bin/pip install -e .                             # raim itself, importable from any directory
export PY=$PWD/.venv/bin/python3

make runs PY=$PY             # seconds: unpack runs.tar.xz, and are the judgements exactly as pinned?
make check PY=$PY            # seconds to a couple of minutes, by hardware: does this checkout work at all?
$PY tools/export_votes.py --check     # is ../raim-analysis/verdicts their projection?
```

`make check` needs no GPU and no network beyond a warm dataset cache, plus the FActScore file (`$PY tools/fetch_factscore.py` explains where to get it).
It parses everything, runs the offline test suites, runs the whole panel end to end against a mock backend, and verifies that rebuilding the datasets reproduces the committed join index.

On a machine with a **cold** dataset cache, go online first — the offline check has nothing to read otherwise:

```bash
$PY tools/joincheck.py --online     # verifies against live upstream and fills the cache
make check PY=$PY
```

### Actually measuring something

Needs a GPU, Python 3.12 or later, `pip install vllm transformers`, an `HF_TOKEN` (`lytang/LLM-AggreFact` and the Llama-3.1 and gemma-2 weights are gated), and the FActScore file.
A re-run writes into `runs/`, so work on a copy if you want to keep the released files as they are.

```bash
$PY scripts/run.py --dataset aggrefact_cnn --backend vllm --gpu-util 0.90 --limit 20
$PY tools/coverage_check.py runs/aggrefact_cnn/experiment.jsonl
```

Start small. `aggrefact_cnn` is the cheapest set. `bash scripts/run_panel.sh` does all eight and takes hours.

`FRESH_BOX.md` is the full procedure for a machine that has never run this, including what to check and the traps worth knowing about.

> **Every file in the verdict tree is the output of exactly one command**, with no flags to remember.
>
> | file | command |
> |---|---|
> | `<ds>/experiment.jsonl` | `bash scripts/run_panel.sh` |
> | `<ds>/experiment_defon.jsonl` | `bash scripts/run_contrast.sh` |
> | `<ds>/judge_<tag>.*` | `LEG=<leg> bash scripts/run_judge.sh` |
> | `<ds>/judge_sonnet_defon.*` | `bash scripts/run_contrast.sh` |
> | `lp_probe/<ds>_*_v3*.jsonl` | `bash scripts/run_scorer.sh` |

### Handing results to the analysis repository

```bash
make lock PY=$PY      # a checksummed identity for what you measured
make export PY=$PY    # project it into ../raim-analysis/verdicts, ~5 MB
```

`make export` refuses to overwrite an existing mirror unless the lockfile it is given actually describes the tree being exported, file for file by content, *and* names the release that mirror already holds.

On a box that has measured something of its own, pin it to its own lockfile rather than overwriting the released one:

```bash
$PY tools/verdict_lock.py --runs runs --lock this-box.lock.json
make export LOCK=this-box.lock.json ANALYSIS=/tmp/export-test PY=$PY
```

`ANALYSIS` names an analysis checkout, so the mirror lands in its `verdicts/` subdirectory (`/tmp/export-test/verdicts/` above), never in `ANALYSIS` itself.
A directory with no mirror in it needs none of that — `ANALYSIS=/tmp/somewhere` just works, and the manifest records `release_digest: null` rather than claiming one that does not apply.
`FORCE=1` overrides the guard; it is spelled that way because `make` rejects a leading `--force` as an option of its own.

## The four things worth knowing before you change anything

1. **The panel's order is load-bearing.** Shards are cached by index, not by model name; a reordered `raim/panel.py` is refused at the first cached shard rather than silently pairing it with the wrong model.
2. **Instance identifiers are positional** over a seeded shuffle. That is why every loader pins a dataset revision and why `dataset_index/` exists; run `make indexcheck` after touching `raim/tasks.py`.
3. **The four LLM-AggreFact sets use a different prompt frame** (claim-support, not `def_on`), because their question and candidate are the same string. The pipeline detects this per dataset — do not override it.
4. **Verdict records carry no source text, and must not start to.** That property is what allows this data to be published at all.

## If something breaks

`FRESH_BOX.md` has a troubleshooting section.
The backend turns vLLM's FlashInfer sampler off by default (`VLLM_USE_FLASHINFER_SAMPLER=0`, unless the environment sets the variable): the panel decodes greedily, so that sampler does nothing for us, and left on it compiles CUDA kernels at startup, which fails on many machines.
A startup failure compiling kernels therefore means the variable is exported as `1`.

The one you are most likely to hit is `scripts/run_scorer.sh` reporting `### DATASETS PENDING`: a dataset's member files were not all available to assemble, or the gate refused them, so a scoring run above it failed.
A log of nothing but `[skip] … already holds` means the scoring was complete before you started.
