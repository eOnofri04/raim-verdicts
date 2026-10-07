"""The frozen scoring block: paired cluster-bootstrap kappa and balanced accuracy.

Support file to compute Cohen's kappa primary, balanced accuracy
as companion with a non-answer counted as an error via coverage, resampling
CLUSTERS of `row_id` rather than items, because the balanced binary builders
pair a supported and a hallucinated candidate within one source row and those
two items are not independent.  B = 2000, seed 0, intervals reported as
[point, lo, hi] by `raim.ci.ci`.

Recomputed from the committed `.jsonl` for every judge on every dataset, it
reproduces the kappa and balanced-accuracy triples of the committed
`judge_<tag>.json` exactly.

A note on abstention, the one place `scripts/run_judge.py`/`scripts/run_ptjudge.py` and
`scripts/run_minicheck.py` genuinely differ.  A generative judge can decline to
answer, so its items are dropped pairwise INSIDE each bootstrap draw;
MiniCheck is a classifier and always answers, so nothing is ever dropped for
it and coverage is pinned at 1.0.  Those are the same code when nothing is
None -- the filter is a no-op and the RNG draw sequence is identical -- so one
implementation serves all three, and `coverage` is computed rather than
asserted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from .ci import ci

B_DEFAULT = 2000
SEED_DEFAULT = 0


def cluster_bootstrap(gold: Dict[str, int], pred: Dict[str, Optional[int]],
                      row_of: Dict[str, str], *, B: int = B_DEFAULT,
                      seed: int = SEED_DEFAULT
                      ) -> Tuple[List[float], List[float], float]:
    """(kappa, balanced accuracy, coverage) for one judge over one dataset.

    Each is a [point, lo, hi] triple bar coverage, which is a scalar.  Keys are
    instance uids; `row_of` maps each to its clustering unit.  A `pred` of None
    is an abstention: it lowers coverage and is dropped pairwise from each
    draw, which is how a non-answer is counted as an error.
    """
    uids = sorted(gold)
    if not uids:
        raise ValueError("no instances to score")
    coverage = sum(pred[u] is not None for u in uids) / len(uids)

    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[row_of[u]].append(u)
    rids = sorted(row_to_uids)

    rng = np.random.default_rng(seed)
    ks, bas = [], []
    for _ in range(B):
        pick = rng.integers(0, len(rids), size=len(rids))
        sel = [u for p in pick for u in row_to_uids[rids[p]]]
        yt = [gold[u] for u in sel if pred[u] is not None]
        yp = [pred[u] for u in sel if pred[u] is not None]
        # A draw that happens to hold one class only leaves kappa undefined;
        # skipping it is what every reported interval was computed under.
        if len(set(yt)) < 2:
            continue
        ks.append(cohen_kappa_score(yt, yp))
        bas.append(balanced_accuracy_score(yt, yp))
    return list(ci(ks)), list(ci(bas)), coverage


def report(kappa: Sequence[float], bacc: Sequence[float], coverage: float,
           extra: str = "") -> None:
    """The one-line summary every runner prints after scoring."""
    print(f"  coverage={coverage:.3f}{extra}  "
          f"kappa={kappa[0]:.3f} [{kappa[1]:.3f},{kappa[2]:.3f}]  "
          f"bal_acc={bacc[0]:.3f} [{bacc[1]:.3f},{bacc[2]:.3f}]")
