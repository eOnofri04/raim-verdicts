#!/usr/bin/env python
"""run_panel must never destroy an existing verdict file it fails to replace.

The failure this pins: a run whose first member dies must leave an existing,
complete verdict file exactly as it was. Opening out_path with "w" before the
worker failure is raised would truncate it and overwrite a complete file from
an empty shard list, reported merely as partial output.

The exposure is structural, not unlucky. `guard.check` is called for the panel
with skip_on_match=False, so a matching file is RE-ENTERED and
resumed; the shard cache is what makes re-entry safe, yet a verdict file can
hold complete contents with no shard directory behind it, and for such a file
re-entry has nothing to rebuild from.

Offline and GPU-free: the failures are induced with an unknown backend kind,
which make_backend rejects, so the test behaves identically on a laptop and on a
box with vLLM installed. The body is under a __main__ guard because run_panel's
sharded path spawns, which re-imports this module in the child.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raim import synthetic_instances, run_panel, Arm  # noqa: E402

# Stands in for a complete measurement: what a resumed run finds on disk, and
# what every one of these cases must leave exactly as it was.
COMPLETE = "".join(f'{{"uid":"u{i}","model":"pre-existing","pred":1}}\n'
                   for i in range(2_000))
fails = 0


def case(name, ok, detail=""):
    global fails
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}"
          + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        fails += 1


def fresh(root):
    for p in root.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    out = root / "experiment.jsonl"
    out.write_text(COMPLETE)
    return out


def main():
    root = Path(tempfile.mkdtemp(prefix="raim-panel-atomic-"))
    task, instances = synthetic_instances("medhallu", n=4, seed=0)
    arms = [Arm(condition="def_on")]
    tmp = root / "experiment.jsonl.tmp"

    # 1. The incident: sharded path, no shard cache, the first member fails.
    out = fresh(root)
    before = out.read_bytes()
    try:
        run_panel(task, instances, arms, out, backend_kind="no-such-backend",
                  panel=["m0", "m1"], isolate=True)
        case("1. sharded run with a failing member raises", False, "no exception")
    except RuntimeError as e:
        case("1. sharded run with a failing member raises", True)
        case("   and says the file was not written", "NOT written" in str(e),
             str(e)[:100])
    case("   existing complete file is byte-identical",
         out.exists() and out.read_bytes() == before,
         f"{len(before)} bytes before, "
         f"{out.stat().st_size if out.exists() else 'GONE'} after")
    case("   no .tmp left behind", not tmp.exists())

    # 2. In-process path: it opened "w" before generating anything, so a failure
    #    during generation destroyed the file outright, with no shards to resume.
    out = fresh(root)
    before = out.read_bytes()
    try:
        run_panel(task, instances, arms, out, backend_kind="no-such-backend",
                  panel=["m0"], isolate=False)
        case("2. in-process run with a bad backend raises", False, "no exception")
    except Exception:
        case("2. in-process run with a bad backend raises", True)
    case("   file is byte-identical",
         out.exists() and out.read_bytes() == before,
         f"{len(before)} bytes before, "
         f"{out.stat().st_size if out.exists() else 'GONE'} after")
    case("   no .tmp left behind", not tmp.exists())

    # 3. Success must still replace the file, in full.
    out = fresh(root)
    before = out.read_bytes()
    run_panel(task, instances, arms, out, backend_kind="mock",
              panel=["m1", "m2"], isolate=False)
    rows = out.read_text().splitlines()
    expected = len(instances) * 2 * len(arms)
    case("3. a successful run replaces the file", out.read_bytes() != before)
    case("   with every (instance, model, arm) row", len(rows) == expected,
         f"{len(rows)} rows, expected {expected}")
    case("   no .tmp left behind", not tmp.exists())

    shutil.rmtree(root)
    if fails:
        print(f"\n{fails} panel-atomicity test(s) FAILED")
        return 1
    print("\nall panel-atomicity tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
