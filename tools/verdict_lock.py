#!/usr/bin/env python3
"""Checksum lockfile over the verdict tree — the zone-1/zone-2 pin.

(Zones as defined in the README, "Artefact zones": 1 = this repository's
verdicts, 2 = their read-only mirror in the analysis repository, 3 = what the
analysis derives from them.)

Why this exists. The verdicts are 300+ MB of irreproducible measurement that no
repository holds: they live on the computing device, on whichever GPU box last
received a sync, and eventually in an external archive. The analysis repository
consumes them and produces the paper's numbers. The lockfile answers the
question that links the two -- *which data release produced this table?* -- and
detects the matching hazard: a verdict file amended, truncated, half-synced or
overwritten between one analysis run and the next.

This writes `verdicts.lock.json`: one sha256 per verdict file, plus a single
`release_digest` over all of them that names the release in one string. The
lockfile is small and text, so it is COMMITTED here even though the verdicts it
describes are not -- making it the only in-git record of what the measurements
were. The analysis repository pins the `release_digest` it built against, and
`--check` re-verifies a tree against a lockfile at any later date.

What is covered. Every `.jsonl` under the verdict tree: the panel runs, every
judge, and the scorer probes -- the artefacts that cost GPU
hours or API spend -- together with the per-judge summaries beside them. Shard
directories are excluded by default: they concatenate into the assembled file
beside them, so hashing both would double-count a resumability cache. Pass
--include-shards to cover them too.

Derived `.json` are NOT covered. They are seeded and regenerate from the verdicts
to float drift, so they belong to zone 3, are committed in the analysis repository, and
are diffable there directly -- a checksum would add nothing.

Usage:
    ./tools/verdict_lock.py                       # write verdicts.lock.json
    ./tools/verdict_lock.py --check               # verify the tree against it
    ./tools/verdict_lock.py --check --lock other.json --runs /mnt/box/runs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root; this file lives in tools/
DEFAULT_RUNS = HERE / "runs"
DEFAULT_LOCK = HERE / "verdicts.lock.json"
SHARD_MARKER = "_shards"
CHUNK = 1 << 20


def digest_file(path: Path) -> tuple[str, int, int]:
    """Stream `path` once, returning (sha256, bytes, records)."""
    h = hashlib.sha256()
    size = records = 0
    tail_newline = True
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
            records += chunk.count(b"\n")
            tail_newline = chunk.endswith(b"\n")
    if size and not tail_newline:      # a final line without its terminator
        records += 1
    return h.hexdigest(), size, records


def is_measurement(name: str) -> bool:
    """Did this repository produce `name`, as opposed to the analysis layer?

    The same predicate raim_lib.verdicts.is_measurement applies on the other side,
    restated here rather than imported because the two repositories do not depend
    on each other. Keep them in step.

      *.jsonl            every raw verdict file
      judge_<tag>.json   the per-judge summary written beside it, carrying the
                         judge's kappa, coverage and (for the paid ones) measured
                         usage and cost -- numbers that exist nowhere else and
                         cannot be recomputed without re-running the judge

    judge_pair_*.json is the exception: judge_pair.py computes it in the analysis
    repository from two of those summaries, so it is derived.
    """
    if name.endswith(".jsonl"):
        return True
    return (name.startswith("judge_") and name.endswith(".json")
            and not name.startswith("judge_pair_"))


def collect(runs: Path, include_shards: bool) -> list[Path]:
    # The judge summaries are covered as well as the .jsonl: the cost and
    # purpose-trained-judge tables read them, and they exist nowhere else.
    paths = sorted(p for p in runs.rglob("*")
                   if p.is_file() and is_measurement(p.name))
    if not include_shards:
        paths = [p for p in paths if not any(part.endswith(SHARD_MARKER)
                                             for part in p.relative_to(runs).parts)]
    return paths


def release_digest(entries: dict) -> str:
    """One digest over the whole set: sha256 of "path sha256" lines, path-sorted.

    Order-independent of the filesystem walk, so the same tree always yields the
    same identifier no matter which machine produced the lockfile.
    """
    h = hashlib.sha256()
    for path in sorted(entries):
        h.update(f"{path} {entries[path]['sha256']}\n".encode())
    return h.hexdigest()


def index_digest() -> str | None:
    """Digest of the committed join index, tying the two halves of zone 1 together.

    A verdict release is only meaningful with the index that joins it to source
    rows, so the lockfile records which index it was taken against.
    """
    idx = HERE / "dataset_index"
    if not idx.is_dir():
        return None
    h = hashlib.sha256()
    for p in sorted(idx.glob("*.jsonl")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def guard_existing(lock_path: Path, paths: list[Path], runs: Path,
                   force: bool) -> None:
    """Refuse to overwrite a lockfile describing measurements this tree lacks.

    The failure this prevents. `build` aborts only when it finds NO verdict files
    at all, which is a weaker condition than it looks: a tree holding a smoke
    test or a partial sync would otherwise replace a lockfile naming every real
    measurement with one naming scratch work, and every "which data release
    produced this table?" answer downstream would point at it, silently.

    An overwrite is fine when the release genuinely has moved on and a disaster
    when it has not, and the two are indistinguishable without asking -- so
    compare the file sets and stop. This runs on names, before the tree is
    hashed; guard_changed() covers the same names with different content.
    """
    if not lock_path.exists() or force:
        return
    try:
        recorded = json.loads(lock_path.read_text()).get("files", {})
    except (OSError, ValueError):
        return
    have = {p.relative_to(runs).as_posix() for p in paths}
    missing = sorted(set(recorded) - have)
    if not missing:
        return
    shown = "\n".join(f"     {rel}" for rel in missing[:5])
    more = f"\n     ... and {len(missing) - 5} more" if len(missing) > 5 else ""
    sys.exit(
        f"!! {lock_path.name} records {len(recorded)} verdict files; {len(missing)} of "
        f"them\n   are absent from {runs}, which holds {len(have)}.\n"
        f"{shown}{more}\n"
        f"   Overwriting would replace the pin on the released measurements with a\n"
        f"   pin on this tree. If you are locking a smoke test or a partial sync,\n"
        f"   point --runs at the real tree; if the release genuinely shrank, pass\n"
        f"   --force.\n")


def guard_changed(lock_path: Path, entries: dict, force: bool) -> None:
    """Refuse to re-pin a recorded file whose CONTENT has changed.

    The names guard cannot see a file that is still there but no longer the
    same measurement -- an amended, re-measured or overwritten verdict file.
    Re-pinning it silently would move the release under every number derived
    from it, so it takes --force. The tree has just been hashed, so this is free.
    """
    if not lock_path.exists() or force:
        return
    try:
        recorded = json.loads(lock_path.read_text()).get("files", {})
    except (OSError, ValueError):
        return
    changed = sorted(rel for rel in set(recorded) & set(entries)
                     if recorded[rel].get("sha256") != entries[rel]["sha256"])
    if not changed:
        return
    shown = "\n".join(f"     {rel}" for rel in changed[:5])
    more = f"\n     ... and {len(changed) - 5} more" if len(changed) > 5 else ""
    sys.exit(
        f"!! {len(changed)} file(s) recorded in {lock_path.name} have different "
        f"content now:\n{shown}{more}\n"
        f"   Re-pinning would move the release under every number derived from it.\n"
        f"   If these are deliberate re-measurements, pass --force.\n")


def build(runs: Path, include_shards: bool, lock_path: Path | None = None,
          force: bool = False) -> dict:
    paths = collect(runs, include_shards)
    if not paths:
        sys.exit(f"!! no .jsonl found under {runs} -- is the verdict tree synced?")
    if lock_path is not None:
        guard_existing(lock_path, paths, runs, force)

    entries, total = {}, 0
    for p in paths:
        sha, size, records = digest_file(p)
        rel = p.relative_to(runs).as_posix()
        entries[rel] = dict(sha256=sha, bytes=size, records=records)
        total += size
        print(f"  {sha[:12]}  {size:>11,}  {records:>8,}  {rel}")
    if lock_path is not None:
        guard_changed(lock_path, entries, force)

    return dict(
        note="Checksum lockfile over the RAIM verdict tree. release_digest names "
             "this release in one string; the analysis repository pins it to make "
             "'which data release produced this table?' answerable. Derived JSONs "
             "are excluded: they are seeded and regenerate from these files to float drift.",
        generator="verdict_lock.py",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        raim_verdicts_commit=git_commit(),
        dataset_index_digest=index_digest(),
        includes_shards=include_shards,
        n_files=len(entries),
        total_bytes=total,
        release_digest=release_digest(entries),
        files=entries,
    )


def check(runs: Path, lock_path: Path, include_shards: bool) -> int:
    if not lock_path.exists():
        sys.exit(f"!! no lockfile at {lock_path}; write one first")
    lock = json.loads(lock_path.read_text())
    recorded = lock["files"]

    present = {p.relative_to(runs).as_posix(): p
               for p in collect(runs, lock.get("includes_shards", include_shards))}

    missing = sorted(set(recorded) - set(present))
    extra = sorted(set(present) - set(recorded))
    changed = []
    for rel in sorted(set(recorded) & set(present)):
        sha, size, _ = digest_file(present[rel])
        if sha != recorded[rel]["sha256"]:
            changed.append((rel, recorded[rel]["bytes"], size))

    print(f"\nlockfile   : {lock_path}")
    print(f"generated  : {lock.get('generated')}")
    print(f"release    : {lock['release_digest']}")
    print(f"recorded   : {len(recorded)} files, {lock['total_bytes']:,} bytes\n")

    for rel in missing:
        print(f"  MISSING   {rel}")
    for rel in extra:
        print(f"  UNTRACKED {rel}  (not in the lockfile -- a newer run?)")
    for rel, was, now in changed:
        print(f"  CHANGED   {rel}  ({was:,} -> {now:,} bytes)")

    if missing or changed:
        print(f"\n!! DRIFT: {len(missing)} missing, {len(changed)} changed. "
              f"Numbers derived from this tree do NOT correspond to the recorded "
              f"release.")
        return 1
    if extra:
        print(f"\nOK, with {len(extra)} untracked file(s): every recorded verdict "
              f"matches. Re-write the lockfile to adopt them.")
        return 0
    print("\nOK: the tree matches the lockfile exactly.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                    help="root of the verdict tree (default: runs/)")
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK,
                    help="lockfile path (default: verdicts.lock.json)")
    ap.add_argument("--check", action="store_true",
                    help="verify the tree against the lockfile instead of writing it")
    ap.add_argument("--include-shards", action="store_true",
                    help="also cover *_shards/ (they concatenate into the assembled files)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a lockfile recording files this tree lacks, "
                         "or files whose content has changed (see the guards)")
    args = ap.parse_args()

    if not args.runs.is_dir():
        sys.exit(f"!! no verdict tree at {args.runs}; copy it here, or unpack the release with `make runs`")

    if args.check:
        sys.exit(check(args.runs, args.lock, args.include_shards))

    lock = build(args.runs, args.include_shards, args.lock, args.force)
    args.lock.write_text(json.dumps(lock, indent=1) + "\n")
    print(f"\n  {lock['n_files']} files, {lock['total_bytes']:,} bytes")
    print(f"  release_digest {lock['release_digest']}")
    print(f"  wrote {args.lock}")


if __name__ == "__main__":
    main()
