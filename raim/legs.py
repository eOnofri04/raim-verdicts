"""Every reference judge as a row in one table, and the sweep that runs them.

A "leg" is one judge measured over the suite; the registry is LEGS below:

  python -m raim.legs --list                 reports them

Every leg runs one model over the same eight datasets (the grounded six for a
document-bound specialist) under the dataset's own frame, writes
`runs/<ds>/judge_<tag>.{jsonl,json}`, and is scored by the same estimator.
What actually varies is a handful of VALUES -- a model id, a tensor-parallel
size, which card to pin, which API key must be present, which interpreter can
import the package -- so those are fields here, and the loop is written once.

  python -m raim.legs --leg sonnet --plan    what a run would do, running nothing
  python -m raim.legs --leg api              a family
  python -m raim.legs --leg all              everything whose prerequisites exist

`--contrast` runs the def_on-on-AggreFact arm instead, for the legs that have
one; it is the judge half of what `scripts/run_contrast.sh` drives, and it writes
`judge_<tag>_defon.json`, never the canonical file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import guard
from .suite import CC_DATASETS, condition_of, datasets, frame_of

ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Leg:
    """One judge, everything that distinguishes it from the others.

    `harness` picks the worker and, with it, the record vocabulary the guard
    reads: `judge` records a `condition` (def_on / claimcheck), `ptjudge` a
    `frame` (qa / claimcheck), `minicheck` a frame it always sets to
    claim-support.  Nothing else about a leg is behavioural.
    """
    key: str
    family: str                       # api | oss | pt | specialist
    harness: str                      # judge | ptjudge | minicheck
    tag: str                          # -> runs/<ds>/judge_<tag>.{jsonl,json}
    model: str = ""                   # harness=judge only; the others resolve it
    backend: str = "vllm"             # harness=judge only
    group: str = "all"                # which dataset group applies
    env: Tuple[str, ...] = ()         # any ONE of these must be set
    env_model: str = ""               # env var overriding `model` (ids move)
    tensor_parallel: int = 1
    gpus: str = ""                    # CUDA_VISIBLE_DEVICES for this leg
    interpreter_env: str = ""         # env var naming a foreign interpreter
    interpreter_default: str = ""
    args: Tuple[str, ...] = ()        # extra worker flags
    contrast: bool = False            # has a def_on-on-AggreFact arm
    note: str = ""

    def resolved_model(self) -> str:
        return os.environ.get(self.env_model, "") or self.model

    def interpreter(self) -> str:
        if not self.interpreter_env:
            return os.environ.get("PY") or sys.executable
        return os.environ.get(self.interpreter_env) or os.path.expanduser(
            self.interpreter_default)

    def missing_prerequisite(self) -> Optional[str]:
        """Why this leg cannot run here, or None if it can."""
        if self.env and not any(os.environ.get(v) for v in self.env):
            return f"{' / '.join(self.env)} unset -- export one first"
        if self.interpreter_env:
            py = self.interpreter()
            if not (Path(py).is_file() and os.access(py, os.X_OK)):
                return (f"{py} not found -- create the venv "
                        f"(see scripts/run_minicheck.py's header), then re-run")
        return None


# --- the table ---------------------------------------------------------------
# Model ids and prices move; both are overridable by environment variable, and
# scripts/run_judge.py records the rates it actually used in each output JSON, so a
# later price change cannot silently corrupt an old record.
LEGS: Tuple[Leg, ...] = (
    # The primary frontier judge, the one the panel is compared against.
    Leg("sonnet", "api", "judge", "sonnet", "claude-sonnet-4-6", "api",
        env=("ANTHROPIC_API_KEY",), args=("--paid", "--workers", "8", "--B", "2000"),
        contrast=True, note="the frontier reference the panel is measured against"),

    # The scale ladder.  Both at 8192, so 32B-vs-72B is a clean scale comparison
    # and matches the full context Sonnet sees through the API.
    Leg("qwen72b", "oss", "judge", "qwen72b_awq",
        "Qwen/Qwen2.5-72B-Instruct-AWQ", "vllm", tensor_parallel=2, gpus="0,1",
        args=("--gpu-util", "0.85", "--max-model-len", "8192"), contrast=True),
    Leg("qwen32b", "oss", "judge", "qwen32b_awq",
        "Qwen/Qwen2.5-32B-Instruct-AWQ", "vllm", tensor_parallel=1, gpus="1",
        args=("--gpu-util", "0.85", "--max-model-len", "8192"), contrast=True,
        note="TP=1, pinned to GPU 1"),

    # The competing route to a cheap judge: one small model fine-tuned expressly
    # for evaluation, rather than a panel of generalists aggregated afterwards.
    Leg("prometheus2", "pt", "ptjudge", "pt_prometheus2", gpus="0",
        args=("--backend", "vllm", "--gpu-util", "0.85")),
    Leg("judgelm", "pt", "ptjudge", "pt_judgelm", gpus="0",
        args=("--backend", "vllm", "--gpu-util", "0.85"),
        note="natively pairwise; the adapter drives its single-answer path"),
    # Auto-J-13B (bf16, ~26 GB) needs 48 GB in one piece or TP=2 across two
    # smaller cards.  On a single 48 GB card: AUTOJ_TP=1 AUTOJ_GPUS=0.
    Leg("autoj", "pt", "ptjudge", "pt_autoj", tensor_parallel=2, gpus="0,1",
        args=("--backend", "vllm", "--gpu-util", "0.85")),

    # Grounded specialists, through the OFFICIAL minicheck package in its own
    # venv -- their prompt, their chunking, their operating point.  A specialist
    # mis-prompted by us would understate the baseline we are pre-empting.
    Leg("minicheck_ft5", "specialist", "minicheck", "pt_minicheck_ft5",
        group="grounded", gpus="0", interpreter_env="MC_PY",
        interpreter_default="~/minicheck-env/bin/python3"),
    Leg("bespoke_minicheck", "specialist", "minicheck", "pt_bespoke_minicheck",
        group="grounded", gpus="0", interpreter_env="MC_PY",
        interpreter_default="~/minicheck-env/bin/python3",
        note="fine-tuned from the internlm2_5 checkpoint the panel dropped"),
)

BY_KEY: Dict[str, Leg] = {leg.key: leg for leg in LEGS}
FAMILIES: Tuple[str, ...] = ("api", "oss", "pt", "specialist")

# The guard's record vocabulary, per harness.
_KIND = {"judge": "judge", "ptjudge": "ptjudge", "minicheck": "minicheck"}


def select(spec: str) -> List[Leg]:
    """Resolve `LEG=` to legs: a key, a family, `all`, or a comma-separated mix.

    There is no default on purpose.  `all` is a fair amount of GPU time and
    real money, so asking for it is a sentence someone has to type.
    """
    out: List[Leg] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token == "all":
            out += list(LEGS)
        elif token in FAMILIES:
            out += [leg for leg in LEGS if leg.family == token]
        elif token in BY_KEY:
            out.append(BY_KEY[token])
        else:
            raise SystemExit(
                f"unknown leg {token!r}. Legs: {', '.join(BY_KEY)}. "
                f"Families: {', '.join(FAMILIES)}. Or `all`.")
    seen, uniq = set(), []
    for leg in out:                       # a key named twice runs once
        if leg.key not in seen:
            seen.add(leg.key)
            uniq.append(leg)
    return uniq


def expected_arms(leg: Leg, ds: str, contrast: bool) -> Tuple[str, ...]:
    """What this (leg, dataset) run records, in the leg's own vocabulary."""
    if contrast:
        return ("def_on",)
    if leg.harness == "judge":
        return (condition_of(ds),)
    if leg.harness == "ptjudge":
        return (frame_of(ds),)
    return ("claimcheck",)                # minicheck: claim-support by design


def command(leg: Leg, ds: str, contrast: bool, force: bool = False) -> Tuple[List[str], str]:
    """The worker invocation for one (leg, dataset), and the tag it writes.

    `force` is passed through to the worker's own `--force`, not just used
    here: each worker calls `guard.check` itself (so a standalone invocation
    is protected too), and its default is `force=False` -- without passing
    this along, `sweep`'s pre-check would agree to redo a match but the worker
    would then skip that very same match.
    """
    tag = f"{leg.tag}_defon" if contrast else leg.tag
    py = leg.interpreter()
    force_flag = ["--force"] if force else []
    if leg.harness == "judge":
        cmd = [py, str(ROOT / "scripts" / "run_judge.py"), "--dataset", ds,
               "--backend", leg.backend, "--model", leg.resolved_model(),
               "--tag", tag, "--condition", expected_arms(leg, ds, contrast)[0],
               "--tensor-parallel", str(leg.tensor_parallel), *leg.args, *force_flag]
    elif leg.harness == "ptjudge":
        cmd = [py, str(ROOT / "scripts" / "run_ptjudge.py"), "--dataset", ds,
               "--judge", leg.key, "--tag", tag,
               "--tensor-parallel", str(leg.tensor_parallel), *leg.args, *force_flag]
    else:
        cmd = [py, str(ROOT / "scripts" / "run_minicheck.py"), "--dataset", ds,
               "--model", leg.key, *leg.args, *force_flag]
    return cmd, tag


def sweep(legs: Sequence[Leg], *, runs: Path, contrast: bool = False,
          force: bool = False, plan: bool = False) -> List[str]:
    """Run each leg over its datasets.  Returns the list of failures.

    One dataset's failure never aborts the sweep: a paid pass that dies on item
    six of eight should not lose the other seven, and everything completed is
    skipped on the next run anyway.
    """
    failed: List[str] = []
    for leg in legs:
        why = leg.missing_prerequisite()
        if why:
            print(f"[{leg.key}] SKIPPED: {why}")
            continue
        pool = CC_DATASETS if contrast else datasets(leg.group)
        if contrast and not leg.contrast:
            print(f"[{leg.key}] no def_on contrast arm defined; skipped")
            continue
        print(f"\n==== {leg.key} ({leg.family}/{leg.harness})"
              f"{' [contrast]' if contrast else ''} ====")
        for ds in pool:
            cmd, tag = command(leg, ds, contrast, force)
            out = runs / ds / f"judge_{tag}.json"
            label = f"{tag}/{ds}"
            try:
                todo = guard.check(out, expected_arms(leg, ds, contrast),
                                   _KIND[leg.harness], skip_on_match=True,
                                   force=force, label=label)
            except guard.StaleMeasurement as e:
                print(str(e))
                failed.append(f"{label} (refused: stale frame)")
                continue
            if not todo:
                continue
            if plan:
                pin = f"CUDA_VISIBLE_DEVICES={leg.gpus} " if leg.gpus else ""
                print(f"  would run: {pin}{' '.join(cmd)}")
                continue
            env = dict(os.environ)
            if leg.gpus:
                env["CUDA_VISIBLE_DEVICES"] = leg.gpus
            print(f">>> {label}")
            if subprocess.run(cmd, env=env, cwd=ROOT).returncode == 0:
                print(f"    ok -> {out}")
            else:
                print(f"    !! FAILED: {label}")
                failed.append(label)
    return failed


def _list() -> None:
    print(f"{'leg':<20} {'family':<11} {'harness':<10} {'tag':<22} "
          f"{'datasets':<9} prerequisite")
    for leg in LEGS:
        why = leg.missing_prerequisite()
        state = "ready" if why is None else f"MISSING: {why[:44]}"
        print(f"{leg.key:<20} {leg.family:<11} {leg.harness:<10} "
              f"{leg.tag:<22} {leg.group:<9} {state}")
        if leg.note:
            print(f"{'':<20} {leg.note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", help="key, family (api/oss/pt/specialist), `all`, "
                                  "or a comma-separated mix")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--plan", action="store_true",
                    help="print what would run, run nothing (no GPU, no spend)")
    ap.add_argument("--contrast", action="store_true",
                    help="the def_on-on-AggreFact arm instead of the canonical one")
    ap.add_argument("--runs", default=str(ROOT / "runs"), type=Path,
                    help="verdict tree to write into (default: runs/)")
    ap.add_argument("--force", action="store_true",
                    help="redo a measurement that already exists UNDER THE RIGHT "
                         "frame; it can never override a frame mismatch")
    args = ap.parse_args()

    if args.list or not args.leg:
        _list()
        if not args.leg:
            print("\nPick one with LEG=<leg> bash scripts/run_judge.sh (--leg, "
                  "calling raim.legs directly). There is no default: `all` is real "
                  "money and real GPU hours.\n'ready' means the leg's API key or "
                  "interpreter is in place; it does not check for a GPU.")
        return

    failed = sweep(select(args.leg), runs=Path(args.runs), contrast=args.contrast,
                   force=args.force, plan=args.plan)
    print()
    if failed:
        print(f"### FINISHED WITH FAILURES: {', '.join(failed)}")
        print("  (fix, then re-run -- completed measurements skip automatically)")
        raise SystemExit(1)
    print("### judges done -> runs/<ds>/judge_<tag>.{jsonl,json}")


if __name__ == "__main__":
    main()
