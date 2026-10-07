"""One rule: never write over a measurement that is not the one you asked for.

THE RULE.  An existing measurement is honoured only if the arm-set it records
is EXACTLY the arm-set this run would write.  Anything else -- a different
frame, a subset, a superset -- refuses, names the file, and says what to do.
`force` redoes a measurement that matches; it can never blow through a
mismatch, because a mismatch means the file is not the thing being redone, and
the one copy of it may be irreplaceable.  Moving it aside is a human's
decision, not a flag's.

Narrowing is a refusal for a reason worth stating.  `raim.panel._manifest`
hashes the arm list, so a run that writes fewer arms than the file holds does
not merely shrink it: it first clears the shard cache the file was assembled
from, and then pays to recompute what it kept.  A file holding more than you
asked for is therefore the most expensive thing to overwrite by accident and
should instead be decomposed and reused.

The same rule, read-only, over a whole verdict tree (worth running on any tree
copied in from another machine):

    python -m raim.guard --scan runs
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional, Tuple

# Which field carries the arm label, per harness.  The vocabularies genuinely
# differ, so the mapping is data, not a convention to be remembered.
_FIELD = {
    "panel": "condition",     # scripts/run.py            JSONL rows
    "judge": "condition",     # scripts/run_judge.py      JSON sidecar
    "ptjudge": "frame",       # scripts/run_ptjudge.py    JSON sidecar, {qa, claimcheck}
    "minicheck": "frame",     # scripts/run_minicheck.py  JSON sidecar, claimcheck always
    "lp": "condition",        # scripts/run_lpscore.py    JSONL rows
}
_SIDECAR = {"judge", "ptjudge", "minicheck"}   # record is a single JSON object


class StaleMeasurement(SystemExit):
    """Raised instead of overwriting a measurement that is not the one asked for."""


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def recorded_armset(path: Path, kind: str) -> Optional[FrozenSet[str]]:
    """The arm-set an existing file records.

    Three outcomes:

      None              no file at all -- nothing to compare.
      frozenset()       a file that exists but carries no arm label, because it
                        is empty, truncated, or predates the field.
      a non-empty set   what it records.

    An unlabelled file means opposite things to the two shapes of runner, so it
    is reported as its own outcome rather than collapsed into "no evidence".
    A panel JSONL is shard-resumable, and an aborted run leaves exactly this --
    a re-run is meant to complete it.  A judge sidecar is written in one pass,
    so an unlabelled one is a schema vintage, and on an aggrefact set that is
    precisely the wrong-frame hazard the guard exists for.

    A malformed JSONL line -- typically the truncated last line of an
    interrupted write -- is skipped on its own rather than discarding every
    row read before it: an aborted write should not erase the evidence its
    earlier, complete rows already carry, which a single try/except around the
    whole loop would have done.
    """
    path = Path(path)
    if not path.exists():
        return None
    field = _FIELD[kind]
    if kind in _SIDECAR:
        try:
            value = json.loads(path.read_text()).get(field)
        except (OSError, ValueError):
            return frozenset()
        return frozenset() if value is None else frozenset({value})
    seen = set()
    try:
        with _open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line).get(field)
                except ValueError:
                    continue
                if value is not None:
                    seen.add(value)
    except OSError:
        return frozenset()
    return frozenset(seen)


def _vintage_ok(recorded: FrozenSet[str], expect: FrozenSet[str],
                kind: str) -> bool:
    """Whether an unlabelled sidecar is a schema vintage rather than a mismatch.

    Two harnesses need this, for different reasons, and the asymmetry between
    them is the whole content of the rule.

    `scripts/run_ptjudge.py` records a `frame` field only since 2026-07-16, so
    an earlier file has no `frame` key at all.  On the qa-form sets that is a
    vintage and nothing more -- `qa` is their only valid frame, the
    claim-support frame concerning the LLM-AggreFact family alone.  On an
    aggrefact set the same absence is exactly the hazard of a file scored under
    the wrong frame, so it refuses.

    For `scripts/run_minicheck.py` the absence is never a hazard on any
    dataset: MiniCheck verifies a claim against a document natively, through
    its own package, so it has one frame by construction and no other frame to
    be stale from.  New runs record the frame like everything else; older
    files without it are honoured, and `check` says so rather than printing
    the "[skip] already holds" line it prints for a file that carries the
    label.
    """
    if recorded:
        return False
    if kind == "minicheck":
        return True
    return kind == "ptjudge" and expect == frozenset({"qa"})


def check(path, expect: Iterable[str], kind: str, *, skip_on_match: bool,
          force: bool = False, label: str = "") -> bool:
    """Decide whether to run.  True = measure; False = already done, skip.

    Refuses (raises StaleMeasurement) rather than returning when the file holds
    a different arm-set.  `skip_on_match` distinguishes the two shapes of
    runner: a judge writes its output in one pass and an existing one is done,
    whereas the panel is shard-resumable and a matching file must be re-entered
    so the remaining members are computed.
    """
    path = Path(path)
    expect = frozenset(expect)
    recorded = recorded_armset(path, kind)

    if recorded is None:
        return True
    vintage = _vintage_ok(recorded, expect, kind)
    if vintage:
        recorded = expect
    if not recorded and kind not in _SIDECAR:
        # A row-based file that exists but carries no labels: an interrupted,
        # shard-resumable run.  Completing it is the intended behaviour, and
        # the shard cache -- not this guard -- is what makes that safe.
        return True

    if recorded != expect:
        raise StaleMeasurement(_refusal(path, recorded, expect, kind, label))
    if skip_on_match and not force:
        if vintage:
            print(f"[skip] {path} predates the frame field; honoured as "
                 f"{_fmt(expect)} by the {kind} vintage rule, not a recorded label")
        else:
            print(f"[skip] {path} already holds {_fmt(expect)}")
        return False
    if skip_on_match:
        print(f"[FORCE] redoing {path} ({_fmt(expect)} unchanged)")
    return True


def mismatch(path, expect: Iterable[str], kind: str) -> Optional[str]:
    """Why `path` does not hold `expect`, or None when it does.

    The read-only form of `check`, by the same rule: None for an absent file,
    for an honoured schema vintage, and for an unlabelled row-based file (an
    interrupted, resumable panel run); otherwise the recorded arm-set must
    equal `expect` exactly.  Nothing is printed, raised or touched.
    """
    path = Path(path)
    expect = frozenset(expect)
    recorded = recorded_armset(path, kind)
    if recorded is None or _vintage_ok(recorded, expect, kind):
        return None
    if not recorded and kind not in _SIDECAR:
        return None
    if recorded != expect:
        return f"holds {_fmt(recorded)}, expected {_fmt(expect)}"
    return None


def scan(runs) -> List[Tuple[Path, str]]:
    """Every file in a verdict tree that does not hold what its path implies.

    Covers, per dataset of the suite, the panel file (`experiment.jsonl`), the
    frame contrast (`experiment_defon.jsonl`, claim-support sets only) and every
    judge summary whose tag the registry (`raim.legs`) knows, canonical or
    `_defon`, each checked against the arm-set its runner would write.  Judge
    files the registry does not know (e.g. derived `judge_pair_*`) are skipped.
    """
    from .legs import LEGS, expected_arms       # legs imports guard; defer
    from .suite import SUITE, condition_of, frame_of
    by_tag = {leg.tag: leg for leg in LEGS}
    found: List[Tuple[Path, str]] = []
    for ds in SUITE:
        d = Path(runs) / ds
        checks = [(d / "experiment.jsonl", (condition_of(ds),), "panel")]
        if frame_of(ds) == "claimcheck":
            checks.append((d / "experiment_defon.jsonl", ("def_on",), "panel"))
        for f in sorted(d.glob("judge_*.json")):
            tag = f.stem[len("judge_"):]
            contrast = tag.endswith("_defon")
            leg = by_tag.get(tag[: -len("_defon")] if contrast else tag)
            if leg is None or (contrast and not leg.contrast):
                continue
            checks.append((f, expected_arms(leg, ds, contrast), leg.harness))
        for path, expect, kind in checks:
            why = mismatch(path, expect, kind)
            if why:
                found.append((path, why))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", metavar="RUNS", type=Path, required=True,
                    help="verdict tree to check, read-only (e.g. runs)")
    args = ap.parse_args()
    found = scan(args.scan)
    if not found:
        print(">>> guard scan: every panel and judge file holds its expected frame.")
        return
    print("\n!! guard scan: files that do not hold what their path implies:")
    for path, why in found:
        print(f"     {path}  ({why})")
    print("   Nothing was moved or deleted. The runners refuse to treat these as")
    print("   done; move each aside by hand once you have confirmed what it is.")


def _fmt(arms: FrozenSet[str]) -> str:
    return ",".join(sorted(arms)) if arms else "no arm label at all"


def _refusal(path: Path, recorded: FrozenSet[str], expect: FrozenSet[str],
             kind: str, label: str) -> str:
    who = f"{label}: " if label else ""
    lines = [f"[REFUSE] {who}{path} holds {_fmt(recorded)}, "
             f"but this run would write {_fmt(expect)}."]
    if not recorded:
        lines.append("         It predates the field that records the frame, so it "
                     "cannot be shown to be the current measurement.")
    if recorded and expect < recorded:
        lines += [
            "         That is a NARROWING: the file holds arms this run does not.",
            "         Overwriting it would drop those arms and, because the shard",
            "         manifest hashes the arm list, clear the cache they were",
            "         assembled from -- so the run would also pay to recompute what",
            "         it kept.",
            "         The usual cause is a tree measured before an experiment was",
            "         split out of this file into one of its own: move the file",
            "         aside rather than overwrite it.",
            "         To measure the wider sweep here instead, name it:",
            f"           --conditions {_fmt(recorded)}"]
    elif recorded and expect > recorded:
        lines += [
            "         The file holds a SUBSET of this run's arms. Re-running would",
            "         recompute the arms it already has as well as the new ones;",
            "         that is usually right, but it is not a resumption, so it is",
            "         a decision rather than a default. Move the file aside to take",
            "         it: mv {p} {p}.partial".format(p=path)]
    else:
        suffix = _fmt(recorded).replace(",", "_") if recorded else "preframe"
        lines += [
            "         Different frame, not a stale cache -- this looks like a",
            "         measurement taken under another scoring frame (e.g. a",
            "         def_on file at the claim-support path).",
            "         FORCE does not override this. Move it aside by hand:",
            f"           mv {path} {path.with_suffix('')}_{suffix}{path.suffix}"]
        if kind in _SIDECAR:
            raw = path.with_suffix(".jsonl")
            lines.append(f"           mv {raw} {raw.with_suffix('')}_{suffix}.jsonl")
        else:
            shards = path.parent / (path.stem + "_shards")
            lines.append(f"           mv {shards} {shards}_{suffix}   # if present")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
