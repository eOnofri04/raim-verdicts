"""Join index for the verdict release (see the README, "The datasets and the join").

  .venv/bin/python3 tools/dataset_index.py                  # -> dataset_index/<ds>.jsonl + manifest.json
  .venv/bin/python3 tools/dataset_index.py --online         # rebuild against the Hub, at the pins

It WRITES the committed, authoritative index, so it refuses to replace an
existing one unless given --force; to check the index rather than rewrite it,
use `make indexcheck` (a rebuild in a disposable copy) or joincheck.py.  A
forced rebuild is written to a temporary directory and swapped in only once
every requested dataset has built, so a failure leaves the committed index as
it was, and datasets not named in --datasets are carried over unchanged.

Emits, per released instance, the positional keys the verdicts carry (uid,
row_id, kind) alongside a CONTENT key, so a third party can verify the join
without our redistributing any source text: they recompute the hash from the
source datasets and check it against this index.

Why the content key is necessary. For the four LLM-AggreFact sets both uid and
row_id are positional artefacts of a seeded shuffle over the loaded rows, so
upstream revising a dataset re-points every uid silently -- nothing raises, the
numbers still compute, and they are wrong. The loaders pin a revision
(raim/tasks.py DATASET_REVISIONS, in place since 2026-08-04, i.e. after the
reported runs were produced), but a pinned revision is a POINTER: it can be
withdrawn, gated or force-moved, and it cannot be checked without the Hub. A
content key can, which is why this index -- not any upstream snapshot, and not
the pins either -- is the authoritative mapping for the released verdicts.

Pairs with joincheck.py, which verifies that a rebuild still reproduces it and
which imports content_key() from here so the two cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root; this file lives in tools/
sys.path.insert(0, str(HERE))
# Safe before set_offline(): raim loads the `datasets` library only inside the
# builders, and set_offline() refuses to proceed if that ever stops being true.
from raim.suite import SUITE  # noqa: E402

DATASETS = list(SUITE)
OUT = HERE / "dataset_index"


def content_key(inst) -> str:
    """The released join key for one instance.

    Delegates to raim.tasks.Instance.src_key, which is the single definition:
    the key now travels with the schema it keys, so the index written here, the
    per-record key panel.py emits, and joincheck.py's verification cannot drift
    apart. This wrapper is kept because joincheck.py imports it by this name.
    """
    return inst.src_key


def set_offline(offline: bool = True) -> None:
    """Pin Hugging Face to the local cache, and point FActScore at its local file."""
    if offline:
        # `datasets` reads these switches once, when it is imported; set after
        # that, they are silently ignored and the "offline" run goes online.
        if "datasets" in sys.modules:
            raise RuntimeError(
                "the `datasets` library was imported before set_offline(); it "
                "reads HF_DATASETS_OFFLINE/HF_HUB_OFFLINE only at import, so "
                "this run would not be offline")
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    # Resolved against this file, not the cwd: the index and the checksum guard in
    # raim.tasks are only meaningful against the one verified copy
    # (FACTSCORE_SHA256), so a run from another directory must not silently fall
    # through to a different file.
    os.environ.setdefault("FACTSCORE_PATH", str(HERE / "datasets" / "InstructGPT.jsonl"))


def build(datasets: list[str], out: Path) -> dict:
    from raim.tasks import build_instances

    out.mkdir(exist_ok=True)
    manifest = {}
    for ds in datasets:
        _task, inst = build_instances(ds)
        keys: Counter[str] = Counter()
        with (out / f"{ds}.jsonl").open("w") as fh:
            for x in inst:
                k = content_key(x)
                keys[k] += 1
                fh.write(json.dumps(dict(
                    uid=x.uid, row_id=x.row_id, kind=x.kind, gold=x.gold,
                    stratum=x.stratum, category=x.category, src_sha1=k)) + "\n")
        dup = sum(1 for v in keys.values() if v > 1)
        manifest[ds] = dict(
            n=len(inst),
            n_row_clusters=len({x.row_id for x in inst}),
            class_balance={str(g): sum(1 for x in inst if x.gold == g)
                           for g in (0, 1)},
            distinct_content_keys=len(keys),
            duplicate_content_keys=dup,
        )
        print(f"{ds:20s} n={len(inst):5d}  clusters={manifest[ds]['n_row_clusters']:5d}"
              f"  distinct keys={len(keys):5d}  dup={dup}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true",
                    help="allow Hugging Face to fetch instead of using the local cache")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--force", action="store_true",
                    help="replace an existing dataset_index/ (the committed, "
                         "authoritative mapping); without it the run refuses")
    args = ap.parse_args()

    if OUT.exists() and any(OUT.iterdir()) and not args.force:
        raise SystemExit(
            f"{OUT} already holds the committed join index, which this script "
            f"would overwrite. To CHECK it, run `make indexcheck` or "
            f"tools/joincheck.py; to replace it, pass --force.")

    set_offline(not args.online)
    # Built beside the target and swapped in only on success, so a failure
    # part-way leaves the committed index intact. Datasets not being rebuilt
    # are carried over, index files and manifest entries alike.
    tmp = OUT.with_name(OUT.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    old_manifest = {}
    if OUT.exists():
        shutil.copytree(OUT, tmp)
        mf = OUT / "manifest.json"
        if mf.exists():
            old_manifest = json.loads(mf.read_text()).get("datasets", {})
    try:
        manifest = {**old_manifest, **build(args.datasets, tmp)}
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    manifest = {ds: manifest[ds] for ds in DATASETS if ds in manifest} | {
        ds: m for ds, m in manifest.items() if ds not in DATASETS}

    (tmp / "manifest.json").write_text(json.dumps(dict(
        note="Join index for the RAIM verdict release. src_sha1 = sha1 over "
             "question, context items and candidate, NUL-separated. Upstream "
             "revisions were not pinned when the reported runs were produced, so "
             "this index -- not any upstream snapshot -- is the authoritative "
             "mapping for the released verdicts. Verify with joincheck.py.",
        builder="raim/tasks.py",
        generator="dataset_index.py",
        built_offline=not args.online,
        datasets=manifest), indent=2) + "\n")
    if OUT.exists():
        shutil.rmtree(OUT)
    tmp.rename(OUT)
    print(f"\nwrote {OUT}/manifest.json and {len(args.datasets)} index files")


if __name__ == "__main__":
    main()
