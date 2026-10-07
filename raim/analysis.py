"""Analysis harness.

An on-box diagnostic readout for run_panel's JSONL output -- not the paper's
analysis, which lives in raim-analysis and reads the run JSONs directly (see
the note above the plot helpers below). Computes, depending on mode:

  experiment:
    - frontier   : Cohen's kappa vs cost, for every committee of size K, so a
                   user can see where a bigger panel stops paying off
    - selective  : accuracy vs coverage as the commit threshold tau sweeps
    - probe      : inter-model Fleiss' kappa + panel-vs-gold kappa, per
                   condition recorded in the file

Decision rule (selective): with K binary votes let p = mean(votes==1) over
non-abstaining models; commit argmax iff max(p,1-p) >= tau, else abstain.
"""
from __future__ import annotations
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score, cohen_kappa_score)


# ------------------------------- loading ------------------------------
def load_rows(path: Path) -> List[dict]:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def pivot(rows, key_fields: Tuple[str, ...]):
    """Index rows by an arbitrary key, e.g. ("condition",) or ("fmt", "paraphrase").

    Returns (votes, gold, stratum, models, uids):
      votes[key][model][uid] -> pred, gold[uid] -> label, stratum[uid] -> stratum,
      models is the sorted list of models seen, uids is every uid seen across
      ALL keys (not scoped to one key), sorted.
    """
    votes = defaultdict(lambda: defaultdict(dict))
    gold, stratum = {}, {}
    models = set()
    for r in rows:
        key = tuple(r[f] for f in key_fields)
        votes[key][r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        stratum[r["uid"]] = r["stratum"]
        models.add(r["model"])
    return votes, gold, stratum, sorted(models), sorted(gold)


# ----------------------------- aggregation ----------------------------
def decide(vbm: Dict[str, Dict[str, Optional[int]]], members, uids, tau: float):
    """Binary consensus over `members`' votes for each uid, with abstention.

    Drop missing votes; let p = mean(remaining). Commit int(p >= 0.5) if
    max(p, 1-p) >= tau, else abstain (None). A tie (p == 0.5) always commits
    to 1, never abstains, at any tau <= 0.5.
    """
    preds, committed = [], []
    for uid in uids:
        vs = [vbm[m].get(uid) for m in members]
        vs = [v for v in vs if v is not None]
        if not vs:
            preds.append(None); committed.append(False); continue
        p = float(np.mean(vs)); conf = max(p, 1 - p)
        if conf >= tau:
            preds.append(int(p >= 0.5)); committed.append(True)
        else:
            preds.append(None); committed.append(False)
    return preds, committed


def _bal_acc_nonanswer_error(preds, committed, gold, uids):
    """Balanced accuracy with an abstention counted as a miss, full per-class
    denominator -- mirrors the actual bacc() used downstream in raim-analysis.
    Note. bal_acc_committed drops abstentions instead: one asks "how good is
    it when it answers", the other "how good is it overall".
    """
    tp = fn_p = tn = fn_n = 0
    for uid, pr, c in zip(uids, preds, committed):
        pred = pr if c else None
        if gold[uid] == 1:
            if pred == 1: tp += 1
            else:         fn_p += 1
        else:
            if pred == 0: tn += 1
            else:         fn_n += 1
    recs = []
    if tp + fn_p: recs.append(tp / (tp + fn_p))
    if tn + fn_n: recs.append(tn / (tn + fn_n))
    return sum(recs) / len(recs) if recs else float("nan")


def score(preds, committed, gold, uids):
    """Coverage plus two balanced-accuracy readings, kappa and macro-F1.

    bal_acc_committed drops abstentions (per-method denominator, same
    treatment as kappa/macro_f1 here and as raim-analysis's kap());
    bal_acc_nonanswer_error charges them as errors on the full per-class
    denominator.
    kappa/macro_f1 and bal_acc_committed return NaN, not an error, when
    nothing committed or the committed subset is single-class, so a sweep
    over many (K, tau) cells doesn't crash on one degenerate cell.
    """
    yt, yp = [], []
    for uid, pr, c in zip(uids, preds, committed):
        if c:
            yt.append(gold[uid]); yp.append(pr)
    cov = len(yt) / len(uids) if uids else 0.0
    nae = _bal_acc_nonanswer_error(preds, committed, gold, uids)
    if len(yt) == 0 or len(set(yt)) < 2:
        return dict(coverage=cov, n=len(yt), bal_acc_committed=float("nan"),
                    bal_acc_nonanswer_error=nae,
                    macro_f1=float("nan"), kappa=float("nan"))
    return dict(coverage=cov, n=len(yt),
                bal_acc_committed=balanced_accuracy_score(yt, yp),
                bal_acc_nonanswer_error=nae,
                macro_f1=f1_score(yt, yp, average="macro"),
                kappa=cohen_kappa_score(yt, yp))


def fleiss_kappa(vbm, members, uids):
    """Two-category Fleiss' kappa: inter-model agreement, not agreement vs gold.

    Only over uids where every model in `members` voted -- a stricter subset
    than `decide`'s per-uid abstention, since agreement is undefined for a
    partial slate. Returns (kappa, n_full), n_full being that subset's size.
    """
    rows = []
    for uid in uids:
        vs = [vbm[m].get(uid) for m in members]
        if any(v is None for v in vs):
            continue
        n1 = sum(vs); rows.append((len(vs) - n1, n1))
    if not rows:
        return float("nan"), 0
    M = np.array(rows, float); n = M.sum(1)[0]
    P_i = (np.square(M).sum(1) - n) / (n * (n - 1))
    p_j = M.sum(0) / M.sum()
    P_e = np.square(p_j).sum()
    k = (P_i.mean() - P_e) / (1 - P_e) if (1 - P_e) > 0 else float("nan")
    return k, len(rows)


# --------------------------- experiment mode --------------------------
def analyze_experiment(rows, outdir: Path, taus=(0.5, 0.6, 0.7, 0.8, 1.0)):
    """Frontier + selective + probe + divergence outputs; see the module docstring."""
    outdir.mkdir(parents=True, exist_ok=True)
    votes, gold, _, models, uids = pivot(rows, ("condition",))
    conds = sorted({k[0] for k in votes})
    # Committee sizes swept: 1, 3, 5, 7, 9 i.e. every odd size up to the panel size,
    # and the full panel itself -- a complete odd-K curve regardless of how large
    # the panel is, with the full panel always included (max(ks) == M).
    M = len(models)
    ks = sorted({k for k in (1, 3, 5) if k <= M}
                | {k for k in range(1, M + 1) if k % 2 == 1} | {M})
    main = ("def_on",) if ("def_on",) in votes else sorted(votes)[0]

    # frontier + selective (main condition)
    vbm = votes[main]
    frontier = []
    for k in ks:
        for members in combinations(models, k):
            for tau in taus:
                pr, cm = decide(vbm, members, uids, tau)
                frontier.append(dict(K=k, tau=tau, members=list(members),
                                     cost=k * len(uids),
                                     **score(pr, cm, gold, uids)))
    json.dump(frontier, (outdir / "frontier.json").open("w"), indent=2)

    # probe across conditions
    probe = []
    for c in conds:
        fk, nfull = fleiss_kappa(votes[(c,)], tuple(models), uids)
        pr, cm = decide(votes[(c,)], tuple(models), uids, 0.5)
        s = score(pr, cm, gold, uids)
        probe.append(dict(condition=c, fleiss_kappa=fk, n_full=nfull,
                          panel_gold_kappa=s["kappa"],
                          panel_bal_acc_committed=s["bal_acc_committed"]))
    json.dump(probe, (outdir / "probe.json").open("w"), indent=2)


    _plot_frontier(frontier, ks, outdir / "frontier.png")
    _plot_selective(frontier, max(ks), outdir / "selective.png")

    full = [r for r in frontier if r["K"] == max(ks) and r["tau"] == 0.5][0]
    print(f"[experiment] panel K={max(ks)} tau=0.5 {main[0]}: "
          f"bal_acc={full['bal_acc_committed']:.3f} kappa={full['kappa']:.3f} "
          f"coverage={full['coverage']:.2f}")
    print(f"  conditions={conds}  -> readouts in {outdir}/")
    return frontier, probe




# -------------------------------- plots -------------------------------
# These are on-box diagnostic readouts, not paper figures -- the paper's figures
# are generated in raim-analysis from the run JSONs. matplotlib is therefore an
# OPTIONAL dependency here: a GPU box that lacks it should still complete a run
# and write its verdicts rather than crash after the inference has been paid for.
def _skip_without_matplotlib(fn):
    """Run `fn`, or say why the diagnostic PNG was not written and carry on."""
    def wrapper(*a, **kw):
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            print(f"  [skip] {a[-1].name}: matplotlib not installed "
                  f"(diagnostic only; the JSONs and verdicts are unaffected)")
            return None
        return fn(*a, **kw)
    return wrapper


@_skip_without_matplotlib
def _plot_frontier(frontier, ks, png):
    import matplotlib.pyplot as plt
    by_k = defaultdict(list)
    for r in frontier:
        if r["tau"] == 0.5 and not np.isnan(r["kappa"]):
            by_k[r["K"]].append((r["cost"], r["kappa"]))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k in sorted(by_k):
        pts = np.array(by_k[k])
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=28, label=f"K={k}")
        ax.scatter([pts[:, 0].mean()], [pts[:, 1].mean()], marker="x",
                   s=120, color="black", zorder=5)
    ax.set_xlabel("cost (model calls)"); ax.set_ylabel("Cohen's kappa vs gold")
    ax.set_title("RAIM cost/reliability frontier (tau=0.5)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(png, dpi=150)
    plt.close(fig)


@_skip_without_matplotlib
def _plot_selective(frontier, k, png):
    import matplotlib.pyplot as plt
    pts = sorted((r["coverage"], r["bal_acc_committed"]) for r in frontier
                 if r["K"] == k and not np.isnan(r["bal_acc_committed"]))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if pts:
        cov, acc = zip(*pts); ax.plot(cov, acc, marker="o")
    ax.set_xlabel("coverage"); ax.set_ylabel("balanced accuracy (committed)")
    ax.set_title(f"Selective evaluation, K={k} (sweep tau)")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)




