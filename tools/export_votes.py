#!/usr/bin/env python3
"""Export the verdict tree as a votes-only mirror — zone 1 out, zone 2 in.

(Zones as defined in the README, "Artefact zones": 1 = this repository's
verdicts, 2 = their read-only mirror in the analysis repository.)

Why this exists. The paper build reads raw verdict files, not merely the
derived JSONs: most table and figure generators do item-level work (member
kappa, the recovery regression, out-of-fold calibration, stacker coefficients,
error correlations) that no derived JSON carries, so the analysis repository
needs the verdicts themselves.

No consumer anywhere in the analysis layer reads the `raw` field -- the model's
generated text -- which is roughly half the bulk, and what remains is heavily
repeated model ids, conditions and instance identifiers, which compress well.
Dropping `raw` and gzipping brings the tree to a few megabytes, small enough to
commit alongside the analysis code as the input that regenerates every number.

The full tree, `raw` text included, remains the archival artefact, committed in
raim-verdicts as runs.tar.xz. This is a projection for analysis, not a
replacement: the verdicts a judge actually emitted are the measurement, and they
are not discarded.

Determinism. gzip stamps an mtime by default, which would make every re-export
differ; this pins it to zero, so an unchanged tree exports byte-identically and a
diff on the mirror always means a verdict genuinely moved.

Usage:
    ./tools/export_votes.py                       # runs/ -> ../raim-analysis/verdicts/
    ./tools/export_votes.py --out /tmp/mirror     # elsewhere
    ./tools/export_votes.py --keep-raw            # full records, still compressed
    ./tools/export_votes.py --check               # verify a mirror and tie it to the source
    ./tools/export_votes.py --runs R --lock L --out D   # a tree that is not the release
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root; this file lives in tools/
DEFAULT_RUNS = HERE / "runs"
DEFAULT_OUT = HERE.parent / "raim-analysis" / "verdicts"
SHARD_MARKER = "_shards"

# Dropped from every record unless --keep-raw. Checked on 2026-08-04 against every
# generator, the aggregator, the bootstrap machinery and the transfer family:
# nothing reads it. It is the judge's generated text, kept in the archival tree.
DROP_FIELDS = ("raw",)


def is_summary(name: str) -> bool:
    """A per-judge summary: measurement, not derived (raim_lib.verdicts.is_measurement).

    judge_pair_*.json is computed in the analysis repository from two of these, so
    it is derived and stays out.
    """
    return (name.startswith("judge_") and name.endswith(".json")
            and not name.startswith("judge_pair_"))


def sources(runs: Path) -> list[Path]:
    """Every assembled verdict file; shard caches concatenate into these."""
    return sorted(p for p in runs.rglob("*.jsonl")
                  if p.is_file()
                  and not any(part.endswith(SHARD_MARKER)
                              for part in p.relative_to(runs).parts))


def summaries(runs: Path) -> list[Path]:
    """The judge summaries, which travel verbatim rather than being projected.

    They carry each judge's kappa, coverage and, for the paid ones, measured usage
    and cost -- numbers that exist nowhere else and that no amount of re-analysis
    recovers, since they record what an API charged on a particular day. They are
    small and already structured, so there is nothing to drop and no point
    compressing them.
    """
    return sorted(p for p in runs.rglob("*.json")
                  if p.is_file() and is_summary(p.name)
                  and not any(part.endswith(SHARD_MARKER)
                              for part in p.relative_to(runs).parts))


def project(src: Path, dst: Path, keep_raw: bool) -> tuple[str, int, int]:
    """Write the votes-only gzip of `src`; return (sha256, bytes, records)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    # mtime=0 so the output is a pure function of the input
    with open(dst, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", compresslevel=6, mtime=0) as gz:
            with src.open() as inp:
                for line in inp:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if not keep_raw:
                        for f in DROP_FIELDS:
                            rec.pop(f, None)
                    gz.write((json.dumps(rec, separators=(",", ":")) + "\n").encode())
                    records += 1
    data = dst.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), records


def source_release(lock: Path) -> dict:
    """Carry the source tree's identity forward, so the mirror names its origin."""
    if not lock.exists():
        return {}
    d = json.loads(lock.read_text())
    return {k: d.get(k) for k in ("release_digest", "dataset_index_digest",
                                  "raim_verdicts_commit", "generated")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def lock_describes(runs: Path, lock: Path) -> tuple[bool, str]:
    """Does `lock` describe the tree about to be exported, file for file?

    The release-digest comparison in the guard is only meaningful if it does, and
    on a box that has just measured something new it usually does not: the
    lockfile arrived from git describing the RELEASED tree. Both digests would
    then read as the released one and the export would pass as a no-op whilst
    replacing the mirror with a projection of a different tree.

    Compared by CONTENT (sha256), not by name and size: a re-run whose votes flip
    0 <-> 1 keeps every byte count. Hashing the whole tree takes about a second.
    """
    if not lock.exists():
        return False, f"{lock.name} does not exist"
    try:
        recorded = json.loads(lock.read_text())["files"]
    except (OSError, ValueError, KeyError):
        return False, f"{lock.name} is unreadable"
    present = {p.relative_to(runs).as_posix(): p
               for p in sources(runs) + summaries(runs)}
    missing = set(recorded) - set(present)
    extra = set(present) - set(recorded)
    changed = [k for k in set(recorded) & set(present)
               if recorded[k].get("sha256") != _sha256(present[k])]
    if missing or extra or changed:
        return False, (f"{lock.name} records {len(recorded)} files; against {runs} "
                       f"that is {len(missing)} absent, {len(extra)} untracked, "
                       f"{len(changed)} with different content")
    return True, ""


def guard_existing(runs: Path, out: Path, lock: Path, force: bool,
                   described: tuple[bool, str]) -> None:
    """Refuse to overwrite a mirror that does not describe what is being exported.

    The failure this prevents: a box holds both clones, someone measures into
    runs/, then `make export` with its default target — and the committed,
    released mirror in raim-analysis is silently replaced. Every number downstream
    would then be derived from another tree's data, and nothing would say so.

    Two things have to hold before an overwrite is allowed to look like a no-op.
    The lockfile must actually describe the tree being exported (`described`,
    from lock_describes), or its digest says nothing about it; and that digest
    must match the one the mirror records. The second alone would compare two
    pieces of metadata to each other without looking at the measurements.
    """
    if force:
        return

    # Nothing to protect. A directory with no manifest is a fresh or disposable
    # target -- the escape route the refusal below recommends -- so it needs no
    # lockfile; export() then records the source as unpinned rather than claiming
    # a digest that does not apply.
    manifest = out / "MANIFEST.json"
    if not manifest.exists():
        return

    ok, why = described
    if not ok:
        sys.exit(
            f"!! {out} already holds a mirror, and the lockfile does not describe\n"
            f"   {runs}, so its release digest cannot say whether this is the same\n"
            f"   data or different data.\n"
            f"     {why}\n"
            f"   Pin this tree first, to its own lockfile if it is not the release:\n"
            f"       ./tools/verdict_lock.py --runs {runs} --lock this-box.lock.json\n"
            f"       ./tools/export_votes.py --runs {runs} --lock this-box.lock.json --out <dir>\n"
            f"   Or send it somewhere disposable, which needs no lockfile at all:\n"
            f"       make export ANALYSIS=/tmp/export-test\n"
            f"   make export FORCE=1 overrides this.\n")

    try:
        existing = json.loads(manifest.read_text()).get("source", {}).get("release_digest")
    except (OSError, ValueError):
        return
    incoming = source_release(lock).get("release_digest")
    if existing and incoming and existing == incoming:
        return          # same release, re-exporting is a no-op by construction
    sys.exit(
        f"!! {out} already holds a mirror of a DIFFERENT release.\n"
        f"     there now : {str(existing)[:16]}\n"
        f"     exporting : {str(incoming)[:16]}\n"
        f"   Overwriting would replace the measurements every committed number is\n"
        f"   derived from. If that is genuinely what you want, pass --force; if you\n"
        f"   are testing the export, send it somewhere else:\n"
        f"       make export ANALYSIS=/tmp/export-test\n")


def export(runs: Path, out: Path, keep_raw: bool, lock: Path,
           force: bool = False) -> dict:
    srcs = sources(runs)
    if not srcs:
        sys.exit(f"!! no .jsonl under {runs} -- is the verdict tree synced?")
    described = lock_describes(runs, lock)      # hashed once, used twice
    guard_existing(runs, out, lock, force, described)

    files, before, after = {}, 0, 0
    for src in srcs:
        rel = src.relative_to(runs).as_posix()
        dst = out / (rel + ".gz")
        sha, size, n = project(src, dst, keep_raw)
        files[rel + ".gz"] = dict(sha256=sha, bytes=size, records=n)
        before += src.stat().st_size
        after += size

    for src in summaries(runs):
        rel = src.relative_to(runs).as_posix()
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dst.write_bytes(data)
        files[rel] = dict(sha256=hashlib.sha256(data).hexdigest(),
                          bytes=len(data), records=None)
        before += len(data)
        after += len(data)

    manifest = dict(
        note="Votes-only projection of the RAIM verdict files released with the "
             "paper, for analysis. The judge's generated text ('raw') is dropped "
             "-- no consumer reads it -- and the result is gzipped with mtime=0 so "
             "an unchanged tree exports byte-identically.",
        generator="export_votes.py",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        keep_raw=keep_raw,
        dropped_fields=[] if keep_raw else list(DROP_FIELDS),
        # Honest provenance: the lockfile's digest is carried forward only when it
        # actually describes this tree. Otherwise the mirror would name a release
        # it is not a projection of, which is worse than naming none.
        source=(source_release(lock) if described[0]
                else {"release_digest": None,
                      "note": f"unpinned: {described[1]}"}),
        n_files=len(files),
        source_bytes=before,
        mirror_bytes=after,
        files=files,
    )
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def check(runs: Path, out: Path, lock: Path) -> int:
    """Verify the mirror, then tie it to the source tree.

    Intact: every file matches the manifest, and the file set matches runs/.
    Tied: the manifest names the release it was projected from, the lockfile
    pins that same release, and runs/ matches the lockfile by content -- so the
    mirror is a projection of the tree actually present, not merely of a tree
    with the same file names.
    """
    mpath = out / "MANIFEST.json"
    if not mpath.exists():
        sys.exit(f"!! no manifest at {mpath}; export first")
    manifest = json.loads(mpath.read_text())
    recorded = manifest["files"]

    bad = []
    for rel, meta in sorted(recorded.items()):
        p = out / rel
        if not p.exists():
            bad.append((rel, "missing")); continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != meta["sha256"]:
            bad.append((rel, "changed"))

    expected = ({p.relative_to(runs).as_posix() + ".gz" for p in sources(runs)}
                | {p.relative_to(runs).as_posix() for p in summaries(runs)})
    stale = sorted(set(recorded) - expected)
    fresh = sorted(expected - set(recorded))

    for rel, why in bad:
        print(f"  {why.upper():8s} {rel}")
    for rel in stale:
        print(f"  ORPHAN   {rel}  (no longer in the source tree)")
    for rel in fresh:
        print(f"  MISSING  {rel}  (in the source tree, not exported)")

    if bad or stale or fresh:
        print(f"\n!! mirror does not match: {len(bad)} bad, {len(stale)} orphaned, "
              f"{len(fresh)} unexported. Re-export.")
        return 1

    projected = (manifest.get("source") or {}).get("release_digest")
    pinned = source_release(lock).get("release_digest")
    ok, why = lock_describes(runs, lock)
    if not projected:
        print("!! NOT TIED: the manifest records no source release (an unpinned export).")
        return 1
    if not ok:
        print(f"!! NOT TIED: the lockfile does not describe {runs}: {why}.")
        return 1
    if projected != pinned:
        print(f"!! NOT TIED: the mirror projects release {projected[:16]}, "
              f"the lockfile pins {str(pinned)[:16]}.")
        return 1
    print(f"OK: {len(recorded)} files intact, projected from release {projected[:16]}, "
          f"which the lockfile pins and {runs} matches by content.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the generated text (much larger; nothing reads it)")
    ap.add_argument("--check", action="store_true",
                    help="verify an existing mirror instead of writing one")
    ap.add_argument("--lock", type=Path, default=HERE / "verdicts.lock.json",
                    help="the lockfile pinning --runs; it must describe that tree")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a mirror of a different release (see the guard)")
    args = ap.parse_args()

    if not args.runs.is_dir():
        sys.exit(f"!! no verdict tree at {args.runs}")
    if args.check:
        sys.exit(check(args.runs, args.out, args.lock))

    m = export(args.runs, args.out, args.keep_raw, args.lock, args.force)
    print(f"  {m['n_files']} files")
    print(f"  {m['source_bytes']/1e6:8.1f} MB source")
    print(f"  {m['mirror_bytes']/1e6:8.1f} MB mirror  "
          f"({100*m['mirror_bytes']/m['source_bytes']:.1f}%)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
