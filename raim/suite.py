"""The study's suite, its frames, and its named arm-sets — in one place.

Three facts, kept here once rather than re-derived at each call site: WHICH
eight datasets the study runs (`raim.tasks.TASKS` registers eleven -- three
further LLM-AggreFact subsets are buildable but were never part of the
reported suite); WHICH FRAME each one takes (`qa` or `claimcheck`, decided by
`resolve_frame`); and WHAT A RUN MEASURES -- the named ARMSETS below, each
fixing an arm list and an output filename together, so the guard can ask
"what should be in this file?" of any runner without re-deriving it.

Nothing here performs a measurement; it is a table plus the lookups over it.
Shell scripts, which cannot import Python cheaply per line, read the same
table through this module's own CLI instead of re-implementing it:

    DATASETS=$(python -m raim.suite --print all)     # or qa / cc / grounded
    python -m raim.suite --frames                    # dataset, frame, condition per line
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .prompts import ALL_CONDITIONS, FRAME_CONDITIONS, resolve_frame

# --- the suite ---------------------------------------------------------------
# Order is the paper's, not the registry's: the four question-form sets, then
# the four LLM-AggreFact ones.  It is the order every sweep runs in, and the
# order the derived artefacts record, so it is pinned here rather than sorted.
QA_DATASETS: Tuple[str, ...] = ("medhallu", "ragtruth", "truthfulqa", "factscore")
CC_DATASETS: Tuple[str, ...] = ("aggrefact_xsum", "aggrefact_wice",
                                "aggrefact_cnn", "aggrefact_expertqa")
SUITE: Tuple[str, ...] = QA_DATASETS + CC_DATASETS

# The six sets carrying a grounding document.  TruthfulQA and FActScore are the
# ungrounded contrast, and a verifier that checks a claim AGAINST a document
# (MiniCheck) simply does not apply to them.
GROUNDED: Tuple[str, ...] = ("medhallu", "ragtruth") + CC_DATASETS

_GROUPS: Dict[str, Tuple[str, ...]] = {
    "all": SUITE, "qa": QA_DATASETS, "cc": CC_DATASETS, "grounded": GROUNDED,
}


def datasets(group: str = "all") -> List[str]:
    """The suite, or one of its named subsets, in the pinned order."""
    try:
        return list(_GROUPS[group])
    except KeyError:
        raise SystemExit(f"unknown dataset group {group!r}; "
                         f"have {', '.join(sorted(_GROUPS))}") from None


def frame_of(dataset: str) -> str:
    """`qa` or `claimcheck` — the dataset's own frozen scoring frame."""
    return resolve_frame("auto", dataset)


def condition_of(dataset: str) -> str:
    """The single condition label a default run records for `dataset`.

    The frame is the prompt SCAFFOLDING (`qa` / `claimcheck`); the condition is
    what a record's `condition` field holds (`def_on` / `claimcheck`).  Under
    the qa frame the study's condition is `def_on`; under claimcheck the two
    names coincide, which is why they were so easily confused.
    """
    return "claimcheck" if frame_of(dataset) == "claimcheck" else "def_on"


# --- what a run measures -----------------------------------------------------
@dataclass(frozen=True)
class ArmSet(ABC):
    """A named measurement: which conditions, where it lands, where it applies.

    `filename` is part of the identity, not a convenience.  The scoring layer
    addresses every input by name and never globs, so an arm that must not be
    read as the canonical measurement is kept apart by its filename alone —
    which is what lets the def_on contrast sit inside the verdict tree, pinned
    by the lockfile and mirrored by the export, whilst being invisible to
    everything that reads `experiment.jsonl`.

    `conditions` is the REQUIRED contract, mirroring `backends.Backend`:
    every arm-set below overrides it, and the abstract method makes that an
    enforced contract rather than a convention an unimplemented default would
    only break at call time.
    """
    name: str
    filename: str
    frames: Tuple[str, ...]          # frames this arm-set is defined for
    doc: str

    @abstractmethod
    def conditions(self, dataset: str) -> Tuple[str, ...]:
        ...

    def applies_to(self, dataset: str) -> bool:
        return frame_of(dataset) in self.frames

    def arms(self, conditions: Tuple[str, ...]) -> List[Tuple[str, str, str]]:
        """The (condition, fmt, paraphrase) triples this arm-set measures.

        The base implementation holds format and paraphrase at the study's
        defaults and varies only the condition; an arm-set that sweeps them
        overrides this.
        """
        return [(c, "brief_reason", "p0") for c in conditions]

    def vocabulary(self, dataset: str) -> Tuple[str, ...]:
        """Which conditions an explicit `--conditions` may name here.

        The dataset's frame is the right answer for the default arm-set and the
        WRONG one for the contrast, whose entire purpose is to apply a
        def_on-family condition to a claim-support dataset.  Keying the check on
        the arm-set rather than the dataset is what keeps both true at once:
        `--conditions def_on` on an aggrefact set is refused under `experiment`,
        as it should be, yet is permitted under `defon`, where it is the
        measurement being asked for and it lands under its own filename.
        """
        return ALL_CONDITIONS if frame_of(dataset) == "qa" else FRAME_CONDITIONS


class _Experiment(ArmSet):
    """The default: one arm, the dataset's own condition.

    This represents the default way of running the panel on all eight datasets.
    """

    def conditions(self, dataset: str) -> Tuple[str, ...]:
        return (condition_of(dataset),)




class _Defon(ArmSet):
    """The frame contrast: def_on applied to the claim-support sets.

    This is the WRONG frame for the four LLM-AggreFact sets, made reproducible
    to study its impact.  One arm, because `analyze_frame_contrast.py` reads
    `def_on` alone.
    """

    def conditions(self, dataset: str) -> Tuple[str, ...]:
        return ("def_on",)

    def vocabulary(self, dataset: str) -> Tuple[str, ...]:
        # The def_on family, on datasets whose own frame is claim-support --
        # which is the whole point of this arm-set, and why the check cannot be
        # the dataset's own frame the way it is for `experiment`.
        return ALL_CONDITIONS




ARMSETS: Dict[str, ArmSet] = {
    a.name: a for a in (
        _Experiment("experiment", "experiment.jsonl", ("qa", "claimcheck"),
                    "the panel under each dataset's own frame (the default)"),
        _Defon("defon", "experiment_defon.jsonl", ("claimcheck",),
               "the def_on frame contrast, LLM-AggreFact sets only"),
    )
}


def armset(name: str) -> ArmSet:
    try:
        return ARMSETS[name]
    except KeyError:
        raise SystemExit(f"unknown arm-set {name!r}; have "
                         f"{', '.join(ARMSETS)}") from None


def parse_conditions(spec: str, dataset: str, aset: ArmSet) -> Tuple[str, ...]:
    """An explicit `--conditions` override: a comma-separated condition list.

    The escape hatch for a sweep no arm-set names.  No file in the verdict tree
    requires it -- every one of them is the output of an arm-set's own command --
    so reaching for this is a sign of measuring something new, which is what it
    is for.  Validated against the ARM-SET's vocabulary rather than the
    dataset's frame, because those differ exactly where it matters: `def_on` on
    an aggrefact set is an example of misframing arriving at the canonical
    path under `experiment`, and the measurement being asked for under `defon`,
    where it lands under a filename the scoring layer cannot reach.
    """
    conds = tuple(c.strip() for c in spec.split(",") if c.strip())
    if not conds:
        raise SystemExit("--conditions given but empty")
    allowed = aset.vocabulary(dataset)
    for c in conds:
        if c in allowed:
            continue
        if frame_of(dataset) == "claimcheck" and c in ALL_CONDITIONS:
            raise SystemExit(
                f"--conditions {c!r} on {dataset!r} under arm-set "
                f"{aset.name!r}: that dataset is scored under the claim-support "
                f"frame, and a def_on-family condition here is a misframing."
                f"If that is what you mean to measure, name the contrast arm-set,"
                f"which records it under its own filename: --armset defon")
        raise SystemExit(
            f"--conditions {c!r} is not available under arm-set {aset.name!r} "
            f"on {dataset!r}; have {', '.join(allowed)}")
    return conds


def conditions_for(dataset: str, armset_name: str,
                   override: Optional[str] = None) -> Tuple[str, ...]:
    """The arms a run will write: the arm-set's, unless overridden explicitly."""
    a = armset(armset_name)
    if not a.applies_to(dataset):
        raise SystemExit(
            f"arm-set {a.name!r} ({a.doc}) is not defined for {dataset!r}, "
            f"which resolves to the {frame_of(dataset)!r} frame.")
    if override:
        return parse_conditions(override, dataset, a)
    return a.conditions(dataset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", dest="group", nargs="?", const="all",
                    choices=sorted(_GROUPS),
                    help="print a dataset group, space-separated, for the shell")
    ap.add_argument("--frames", action="store_true",
                    help="print 'dataset frame condition' per line")
    args = ap.parse_args()
    if args.frames:
        for ds in SUITE:
            print(f"{ds}\t{frame_of(ds)}\t{condition_of(ds)}")
    else:
        print(" ".join(datasets(args.group or "all")))


if __name__ == "__main__":
    main()
