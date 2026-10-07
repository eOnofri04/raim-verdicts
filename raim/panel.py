"""The panel (the array) and the inference run loop.

An 'arm' is one (condition, fmt, paraphrase) prompt configuration, and the
arm-set a run measures is fixed by `raim.suite`. run_panel is agnostic to which:
it runs every model over every arm and writes one JSONL row per
(instance, model, arm).

GPU-memory note: vLLM only reliably releases a model's GPU memory when the
*process* that owns it exits, so each vLLM model runs in its own spawned
subprocess and we concatenate per-model JSONL shards. The mock backend touches
no GPU, so it runs in-process. The shard layout makes the run resumable.
"""
from __future__ import annotations
import json
import time
import contextlib
import multiprocessing as mp
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .prompts import build_prompt, parse_verdict
from .backends import make_backend

# The panel. ORDER IS LOAD-BEARING: shards are cached by panel INDEX
# (<i:02d>.jsonl), not by model name, and the manifest does not hash the model
# list, so a member may only ever be appended, never reordered or inserted
# mid-list. run_panel's cache-hit check (below) refuses a shard whose recorded
# `model` field disagrees with panel[i], so a reorder cannot silently pair a
# shard with the wrong model: by default it STOPS and names the offending
# shard, so a human deletes it (or undoes the reorder) rather than the run
# quietly paying for a recompute nobody asked for; pass force=True (--force on
# the CLI) to have it recompute the mismatched member instead. The members are
# chosen for decorrelated lineage (distinct families, all with >= 8192 native
# context, so none truncates more than the 8192 panel cap).
PANEL = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "microsoft/Phi-4-mini-instruct",
    "01-ai/Yi-1.5-9B-Chat-16K",            # 01.AI; 16K ctx (NOT the 4K base)
    "CohereLabs/c4ai-command-r7b-12-2024", # Cohere; 8192 native ctx, RAG-tuned
    "zai-org/glm-4-9b-chat-hf",            # Zhipu/Z.ai; 128K ctx (HF-native; org renamed THUDM→zai-org)
    "ibm-granite/granite-3.3-8b-instruct", # IBM; 128K ctx (3.3 upgrade over pre-reg 3.1)
    "tiiuae/Falcon3-7B-Instruct",          # TII; 32K ctx (reasoning/maths axis)
]

# Bench / clean substitute. Index 06 replaces internlm/internlm2_5-7b-chat,
# which was dropped because its tokeniser emits <|im_start|>=92549 and <|im_end|>=92548
# — both above the model's 92544-row embedding — so its own chat template OOMs the
# embedding gather on any size-respecting runtime (vLLM and llama.cpp break alike).
# It produced no data (died at generation), hence the swap is not selection-on-outcome.
# EXAONE-3.5 is held in reserve: if another member fails mid-flight, drop this id into
# that index (it is a distinct lineage, 32K ctx, 7.8B, trust_remote_code already wired).
STANDBY = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # LG AI Research; 32K ctx

# --- startup integrity guard --------------------------------------------------
# A dropped comma between two adjacent string literals does NOT raise a syntax
# error: Python silently concatenates them into one fused repo id (e.g.
# "...Instructibm-granite/...") AND the list loses an entry. Both are invisible
# until a model load fails partway through a run. Fail loudly at
# import instead: every id must be a single clean 'namespace/name' (exactly one
# slash, no whitespace), the list must be the expected length, and no duplicates.
_EXPECTED_PANEL_SIZE = 10
_ID_RE = re.compile(r"[^/\s]+/[^/\s]+")
assert len(PANEL) == _EXPECTED_PANEL_SIZE, (
    f"PANEL has {len(PANEL)} entries, expected {_EXPECTED_PANEL_SIZE} "
    "— a dropped comma fuses two entries into one and shortens the list"
)
for _i, _m in enumerate(PANEL):
    assert _ID_RE.fullmatch(_m), (
        f"PANEL[{_i:02d}] is not a clean 'namespace/name' id: {_m!r} "
        "— check for a dropped comma fusing two ids"
    )
assert len(set(PANEL)) == _EXPECTED_PANEL_SIZE, "PANEL contains a duplicate id"
assert _ID_RE.fullmatch(STANDBY), f"STANDBY malformed: {STANDBY!r}"


@dataclass(frozen=True)
class Arm:
    condition: str = "def_on"
    fmt: str = "brief_reason"
    paraphrase: str = "p0"


def _rows_for_model(task, instances, arms, model, backend_kind, backend_kw):
    backend = make_backend(backend_kind, model, **backend_kw)
    rows = []
    for arm in arms:
        prompts = [build_prompt(task, x, arm.condition, arm.fmt, arm.paraphrase)
                   for x in instances]
        meta = [dict(uid=x.uid, gold=x.gold, stratum=x.stratum,
                     condition=arm.condition, fmt=arm.fmt,
                     paraphrase=arm.paraphrase) for x in instances]
        texts = backend.generate(prompts, meta)
        for x, txt in zip(instances, texts):
            rows.append(dict(
                uid=x.uid, row_id=x.row_id, kind=x.kind, gold=x.gold,
                stratum=x.stratum, category=x.category,
                # Content-addressed join key (raim.tasks.Instance.src_key), so a
                # verdict file is self-joining rather than needing dataset_index/
                # alongside it. The field dates from 2026-08-04, after the
                # reported runs, whose records therefore lack it and join through
                # the index, which remains the authoritative mapping for them.
                src_key=x.src_key,
                model=model,
                condition=arm.condition, fmt=arm.fmt,
                paraphrase=arm.paraphrase,
                pred=parse_verdict(txt), raw=txt))
    return rows


def _worker(task, instances, arms, model, backend_kind, backend_kw, shard_path):
    t0 = time.time()
    rows = _rows_for_model(task, instances, arms, model, backend_kind, backend_kw)
    with open(shard_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  {model}: {time.time() - t0:.1f}s ({len(rows)} rows)", flush=True)


def _manifest(task, instances, arms, max_model_len=None) -> str:
    h = hashlib.sha256()
    h.update(task.name.encode())
    for x in instances:
        h.update(x.uid.encode())
    for a in arms:
        h.update(f"{a.condition}|{a.fmt}|{a.paraphrase}".encode())
    # max_model_len is part of the cache key: a re-run at a different context
    # length (e.g. 2048 -> 8192) must NOT silently reuse truncated shards.
    h.update(f"mml={max_model_len}".encode())
    return h.hexdigest()


def _line_count(path: Path) -> int:
    with path.open() as fh:
        return sum(1 for _ in fh)


def _shard_model(shard: Path) -> Optional[str]:
    """The `model` field recorded in a shard's first row, or None if empty."""
    with shard.open() as fh:
        first = fh.readline()
    return json.loads(first).get("model") if first.strip() else None


def _check_shard_schema(shards: List[Path]) -> None:
    """Refuse to assemble shards that disagree about the record schema.

    Why this is needed, and why it is a check rather than part of the cache key.
    A shard is reused when it exists and has the expected line count; `_manifest`
    hashes the task, the instances, the arms and the context length, but NOT the
    shape of the record. That is deliberate -- a manifest mismatch DELETES the
    cached shards (see run_panel), so folding the schema in would destroy GPU
    work on the first run after any record change.

    The exposure it leaves is real, though: emitted records carry `src_key`
    only since 2026-08-04, so a resumed run on a box holding older shards would
    reuse them silently and assemble a panel carrying the key for some members
    and not others -- a half-valid provenance artefact that no line count and no
    downstream analysis would notice.

    So: detect it here, say exactly which shards are old, and stop. Nothing is
    deleted; re-running only the named shards repairs the set.
    """
    if not shards:
        return
    with_key, without_key = [], []
    for shard in shards:
        with shard.open() as fh:
            first = fh.readline()
        if not first.strip():
            continue
        if "src_key" in json.loads(first):
            with_key.append(shard.name)
        else:
            without_key.append(shard.name)

    if with_key and without_key:
        raise RuntimeError(
            "cached shards disagree about the record schema, so the assembled "
            "panel would carry src_key for some members and not others.\n"
            f"  with src_key   : {', '.join(sorted(with_key))}\n"
            f"  without src_key: {', '.join(sorted(without_key))}\n"
            "These predate the 2026-08-04 addition of the content-addressed join "
            "key. Delete the shards listed as 'without' and re-run to recompute "
            "just those members; nothing else is affected, and the panel votes "
            "themselves are unchanged by the key.")
    if without_key:
        print("  NB: these shards predate the src_key field, so the assembled "
              "file joins to source rows through dataset_index/ rather than "
              "carrying its own key. That is the state of every reported run.")


@contextlib.contextmanager
def _atomic(out_path: Path):
    """Write through a temporary sibling, renamed over `out_path` on success only.

    Opening `out_path` directly would destroy a complete measurement the moment
    anything goes wrong: `open("w")` truncates before the first row is produced
    or the first shard is read, so a failure afterwards would leave zero bytes
    where a finished file had been.

    The exposure is structural rather than unlucky. `guard.check` is called for
    the panel with `skip_on_match=False`, so a matching file is
    RE-ENTERED and resumed rather than skipped; the shard cache is what makes
    re-entry safe, yet a verdict file can hold complete contents with no shard
    directory behind it (one synced from another box, say), and for such a
    file re-entry has nothing to rebuild from.

    `Path.replace` is atomic within a directory, so a reader sees either the old
    file or the new one, never a half-written one -- which also closes the
    narrower window where an interrupt during assembly left a short file
    indistinguishable from a finished measurement.

    The suffix keeps the temporary out of the release without needing its own
    rule: `verdict_lock.is_measurement`, and the copy of it in raim-analysis,
    both select on a `.jsonl` name, and `<name>.jsonl.tmp` is not one.
    """
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        with tmp.open("w") as fh:
            yield fh
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)


def run_panel(task, instances, arms: List[Arm], out_path: Path,
              backend_kind: str = "vllm", panel: List[str] = None,
              isolate: bool = None, force: bool = False, **backend_kw) -> Path:
    panel = panel or PANEL
    n = len(instances) * len(panel) * len(arms)
    print(f"[{task.name}] {len(instances)} instances x {len(panel)} models "
          f"x {len(arms)} arms = {n} generations")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isolate is None:
        isolate = (backend_kind == "vllm")

    if not isolate:
        # This path has no shard cache to fall back on, so a failure part-way
        # through would otherwise leave nothing at all -- hence the same atomic
        # write as the sharded branch, not merely for symmetry.
        with _atomic(out_path) as fh:
            for model in panel:
                t0 = time.time()
                for r in _rows_for_model(task, instances, arms, model,
                                         backend_kind, backend_kw):
                    fh.write(json.dumps(r) + "\n")
                print(f"  {model}: {time.time() - t0:.1f}s")
        print(f"wrote {out_path}")
        return out_path

    ctx = mp.get_context("spawn")
    shard_dir = out_path.parent / (out_path.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    expected = len(instances) * len(arms)
    sig = _manifest(task, instances, arms, backend_kw.get("max_model_len"))
    sig_file = shard_dir / "manifest.txt"
    if sig_file.exists() and sig_file.read_text().strip() != sig:
        for f in shard_dir.glob("*.jsonl"):
            f.unlink()
        print("  (run config changed -> cleared stale cached shards)")
    sig_file.write_text(sig)

    shards, failed = [], None
    for i, model in enumerate(panel):
        shard = shard_dir / f"{i:02d}.jsonl"
        if shard.exists() and _line_count(shard) == expected:
            recorded = _shard_model(shard)
            if recorded == model:
                print(f"  {model}: cached ({expected} rows, skipped)")
                shards.append(shard)
                continue
            if not force:
                raise RuntimeError(
                    f"shard {shard} was recorded for model {recorded!r}, but "
                    f"panel[{i:02d}] is now {model!r} -- PANEL was reordered "
                    f"since this shard was cached, exactly what 'NEVER "
                    f"reorder PANEL' warns against.\n"
                    f"  Delete {shard} and re-run to recompute just this "
                    f"member, or pass force=True (--force on the CLI) to "
                    f"have run_panel recompute it for you.")
            print(f"  {model}: shard {shard.name} was recorded for "
                  f"{recorded!r}, not {model!r} -- PANEL was reordered since "
                  f"this shard was cached; recomputing mismatched shard "
                  f"(force=True)")
        p = ctx.Process(target=_worker,
                        args=(task, instances, arms, model, backend_kind,
                              dict(backend_kw), str(shard)))
        p.start()
        p.join()
        if p.exitcode != 0:
            failed = model
            break
        shards.append(shard)

    _check_shard_schema(shards)

    # Assemble only a COMPLETE panel.
    if failed is not None:
        raise RuntimeError(
            f"model {failed!r} failed in its subprocess. Common causes: gated "
            f"model not approved, GPU OOM (lower --gpu-util/--max-model-len), or "
            f"vLLM too old.\n"
            f"  {out_path} was NOT written and is exactly as it was -- which "
            f"matters, because it may hold a complete measurement this run was "
            f"resuming.\n"
            f"  {len(shards)}/{len(panel)} completed shards are cached in "
            f"{shard_dir}/, so re-running the same command recomputes only the "
            f"unfinished models and assembles the file once they are all there.")

    with _atomic(out_path) as fh:
        for shard in shards:
            with shard.open() as s:
                for line in s:
                    fh.write(line)
    print(f"wrote {out_path}")
    return out_path